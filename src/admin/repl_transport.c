/**
 * @file repl_transport.c
 * @brief TCP WAL replication transport for GigaVector.
 */

#include "admin/repl_transport.h"
#include "admin/replication.h"
#include "storage/database.h"
#include "storage/wal.h"
#include "core/utils.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <pthread.h>
#include <time.h>

#ifndef _WIN32
#include <unistd.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <poll.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#endif

#define REPL_MAX_MSG_BYTES (16 * 1024 * 1024)
#define REPL_MAX_CONNECTIONS 32

typedef struct {
    int fd;
    char *node_id;
    int active;
    uint8_t *pending_wal;
    size_t pending_wal_len;
    uint32_t pending_wal_req_id;
    int pending_wal_ready;
    uint8_t pending_heartbeat[16];
    int pending_heartbeat_ready;
} ReplConnection;

struct GV_ReplTransport {
    GV_ReplicationManager *mgr;
    int running;
    int stop_requested;
    GV_ReplTransportHooks hooks;

#ifndef _WIN32
    int listen_fd;
    pthread_t accept_thread;
    int accept_thread_started;
    pthread_t follower_thread;
    int follower_thread_started;
    int leader_fd;
    ReplConnection connections[REPL_MAX_CONNECTIONS];
    pthread_mutex_t conn_mutex;
#endif
};

static void write_u32_be(uint8_t *buf, uint32_t val) {
    buf[0] = (uint8_t)(val >> 24);
    buf[1] = (uint8_t)(val >> 16);
    buf[2] = (uint8_t)(val >> 8);
    buf[3] = (uint8_t)(val);
}

static uint32_t read_u32_be(const uint8_t *buf) {
    return ((uint32_t)buf[0] << 24) | ((uint32_t)buf[1] << 16) |
           ((uint32_t)buf[2] << 8) | ((uint32_t)buf[3]);
}

static void write_u64_be(uint8_t *buf, uint64_t val) {
    write_u32_be(buf, (uint32_t)(val >> 32));
    write_u32_be(buf + 4, (uint32_t)(val & 0xFFFFFFFFu));
}

static uint64_t read_u64_be(const uint8_t *buf) {
    return ((uint64_t)read_u32_be(buf) << 32) | read_u32_be(buf + 4);
}

#ifndef _WIN32

static int recv_exact(int fd, uint8_t *buf, size_t len) {
    size_t total = 0;
    while (total < len) {
        ssize_t n = recv(fd, buf + total, len - total, 0);
        if (n <= 0) return -1;
        total += (size_t)n;
    }
    return 0;
}

static int send_exact(int fd, const uint8_t *buf, size_t len) {
    size_t total = 0;
    while (total < len) {
        ssize_t n = send(fd, buf + total, len - total, MSG_NOSIGNAL);
        if (n <= 0) return -1;
        total += (size_t)n;
    }
    return 0;
}

static int repl_send_message(int fd, uint8_t msg_type, uint32_t request_id,
                             const uint8_t *payload, size_t payload_len) {
    uint32_t length = (uint32_t)(5 + payload_len);
    uint8_t header[9];
    write_u32_be(header, length);
    header[4] = msg_type;
    write_u32_be(header + 5, request_id);
    if (send_exact(fd, header, 9) != 0) return -1;
    if (payload_len > 0 && payload) {
        if (send_exact(fd, payload, payload_len) != 0) return -1;
    }
    return 0;
}

static int repl_recv_message(int fd, uint8_t *msg_type, uint32_t *request_id,
                             uint8_t **payload, size_t *payload_len) {
    uint8_t header[4];
    if (recv_exact(fd, header, 4) != 0) return -1;
    uint32_t length = read_u32_be(header);
    if (length < 5 || length > REPL_MAX_MSG_BYTES) return -1;

    uint8_t meta[5];
    if (recv_exact(fd, meta, 5) != 0) return -1;
    *msg_type = meta[0];
    *request_id = read_u32_be(meta + 1);
    size_t plen = length - 5;
    *payload_len = plen;
    if (plen == 0) {
        *payload = NULL;
        return 0;
    }
    *payload = (uint8_t *)malloc(plen);
    if (!*payload) return -1;
    if (recv_exact(fd, *payload, plen) != 0) {
        free(*payload);
        *payload = NULL;
        return -1;
    }
    return 0;
}

static int repl_transport_send(GV_ReplTransport *transport, int fd, uint8_t msg_type,
                               uint32_t request_id, const uint8_t *payload, size_t payload_len) {
    if (transport && transport->hooks.filter_outbound) {
        if (transport->hooks.filter_outbound(transport->hooks.ctx, msg_type,
                                             payload, payload_len) != 0) {
            return 0;
        }
    }
    return repl_send_message(fd, msg_type, request_id, payload, payload_len);
}

static int repl_transport_recv(GV_ReplTransport *transport, int fd, uint8_t *msg_type,
                               uint32_t *request_id, uint8_t **payload, size_t *payload_len) {
    if (repl_recv_message(fd, msg_type, request_id, payload, payload_len) != 0) {
        return -1;
    }
    if (transport && transport->hooks.filter_inbound && *payload_len > 0) {
        if (transport->hooks.filter_inbound(transport->hooks.ctx, *msg_type,
                                            *payload, *payload_len) != 0) {
            free(*payload);
            *payload = NULL;
            *payload_len = 0;
            return -1;
        }
    }
    return 0;
}

static int parse_host_port(const char *address, char *host, size_t host_cap, uint16_t *port) {
    if (!address || !host || !port) return -1;
    const char *colon = strrchr(address, ':');
    if (!colon || colon == address) return -1;
    size_t hlen = (size_t)(colon - address);
    if (hlen >= host_cap) return -1;
    memcpy(host, address, hlen);
    host[hlen] = '\0';
    *port = (uint16_t)atoi(colon + 1);
    return (*port > 0) ? 0 : -1;
}

#define REPL_CONNECT_TIMEOUT_MS 3000

static int repl_connect_host(const char *host, uint16_t port) {
    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%u", (unsigned)port);

    struct addrinfo hints;
    struct addrinfo *res = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    if (getaddrinfo(host, port_str, &hints, &res) != 0 || !res) return -1;

    int fd = -1;
    for (struct addrinfo *ai = res; ai != NULL; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;

        int flags = fcntl(fd, F_GETFL, 0);
        if (flags >= 0) {
            (void)fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        }

        int rc = connect(fd, ai->ai_addr, ai->ai_addrlen);
        if (rc == 0) {
            if (flags >= 0) {
                (void)fcntl(fd, F_SETFL, flags);
            }
            break;
        }
        if (errno == EINPROGRESS) {
            struct pollfd pfd = {.fd = fd, .events = POLLOUT};
            if (poll(&pfd, 1, REPL_CONNECT_TIMEOUT_MS) > 0) {
                int err = 0;
                socklen_t errlen = sizeof(err);
                if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &errlen) == 0 && err == 0) {
                    if (flags >= 0) {
                        (void)fcntl(fd, F_SETFL, flags);
                    }
                    break;
                }
            }
        }
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    return fd;
}

static void repl_tune_socket(int fd) {
    if (fd < 0) return;
    int flag = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
    struct timeval tv = {.tv_sec = 10, .tv_usec = 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

static void repl_clear_connection_pending(ReplConnection *conn) {
    if (!conn) return;
    free(conn->pending_wal);
    conn->pending_wal = NULL;
    conn->pending_wal_len = 0;
    conn->pending_wal_req_id = 0;
    conn->pending_wal_ready = 0;
    conn->pending_heartbeat_ready = 0;
}

static void repl_flush_connection_pending(GV_ReplTransport *transport, ReplConnection *conn) {
    if (!conn || !conn->active || conn->fd < 0) return;
    if (conn->pending_wal_ready) {
        repl_transport_send(transport, conn->fd, REPL_MSG_WAL, conn->pending_wal_req_id,
                            conn->pending_wal, conn->pending_wal_len);
        conn->pending_wal_ready = 0;
    }
    if (conn->pending_heartbeat_ready) {
        repl_transport_send(transport, conn->fd, REPL_MSG_HEARTBEAT, conn->pending_wal_req_id + 1,
                          conn->pending_heartbeat, sizeof(conn->pending_heartbeat));
        conn->pending_heartbeat_ready = 0;
    }
}

static void repl_update_replica_ack(GV_ReplicationManager *mgr, const char *node_id,
                                    uint64_t entry_index) {
    replication_replica_ack(mgr, node_id, entry_index);
}

static void repl_handle_wal_on_follower(GV_ReplicationManager *mgr, uint64_t entry_index,
                                        const uint8_t *record, size_t record_len) {
    replication_follower_apply_entry(mgr, entry_index, record, record_len);
}

static int repl_send_catchup(GV_ReplTransport *transport, int fd, GV_Database *db,
                             uint64_t from_entry) {
    const char *path = db_wal_path(db);
    if (!path) return 0;

    uint64_t total = wal_count_entries(path);
    for (uint64_t i = from_entry; i < total; i++) {
        uint8_t type = 0;
        uint8_t *record = NULL;
        size_t record_len = 0;
        if (wal_read_entry_at(path, i, &type, &record, &record_len) != 0) {
            free(record);
            return -1;
        }

        size_t payload_len = 12 + record_len;
        uint8_t *payload = (uint8_t *)malloc(payload_len);
        if (!payload) {
            free(record);
            return -1;
        }
        write_u64_be(payload, i);
        write_u32_be(payload + 8, (uint32_t)record_len);
        memcpy(payload + 12, record, record_len);
        free(record);

        int rc = repl_transport_send(transport, fd, REPL_MSG_WAL, (uint32_t)(i + 1),
                                       payload, payload_len);
        free(payload);
        if (rc != 0) return -1;
    }
    return 0;
}

static void repl_handle_client(GV_ReplTransport *transport, int fd) {
    GV_ReplicationManager *mgr = transport->mgr;
    repl_tune_socket(fd);
    char node_id[256] = {0};
    uint8_t msg_type = 0;
    uint32_t req_id = 0;
    uint8_t *payload = NULL;
    size_t payload_len = 0;

    if (repl_recv_message(fd, &msg_type, &req_id, &payload, &payload_len) != 0 ||
        msg_type != REPL_MSG_HELLO || payload_len < 4) {
        free(payload);
        close(fd);
        return;
    }
    uint32_t nid_len = read_u32_be(payload);
    if (nid_len >= sizeof(node_id) || payload_len < 4 + nid_len) {
        free(payload);
        close(fd);
        return;
    }
    memcpy(node_id, payload + 4, nid_len);
    node_id[nid_len] = '\0';
    free(payload);

    uint64_t catchup_from = 0;
    replication_replica_handshake(mgr, node_id, &catchup_from);

    if (repl_recv_message(fd, &msg_type, &req_id, &payload, &payload_len) == 0 &&
        msg_type == REPL_MSG_CATCHUP && payload_len >= 8) {
        catchup_from = read_u64_be(payload);
        free(payload);
    } else {
        free(payload);
    }

    repl_send_catchup(transport, fd, replication_get_db(mgr), catchup_from);

    int slot = -1;
    pthread_mutex_lock(&transport->conn_mutex);
    for (int i = 0; i < REPL_MAX_CONNECTIONS; i++) {
        if (!transport->connections[i].active) {
            slot = i;
            break;
        }
    }
    if (slot >= 0) {
        transport->connections[slot].fd = fd;
        transport->connections[slot].node_id = gv_dup_cstr(node_id);
        transport->connections[slot].active = 1;
        repl_clear_connection_pending(&transport->connections[slot]);
    }
    pthread_mutex_unlock(&transport->conn_mutex);

    if (slot < 0) {
        close(fd);
        return;
    }

    while (!transport->stop_requested) {
        pthread_mutex_lock(&transport->conn_mutex);
        repl_flush_connection_pending(transport, &transport->connections[slot]);
        pthread_mutex_unlock(&transport->conn_mutex);

        struct pollfd pfd = {.fd = fd, .events = POLLIN};
        int prc = poll(&pfd, 1, 100);
        if (prc < 0) break;
        if (prc == 0) continue;
        if (!(pfd.revents & POLLIN)) continue;

        if (repl_transport_recv(transport, fd, &msg_type, &req_id, &payload, &payload_len) != 0) break;
        if (msg_type == REPL_MSG_ACK && payload_len >= 8) {
            repl_update_replica_ack(mgr, node_id, read_u64_be(payload));
        }
        free(payload);
    }

    pthread_mutex_lock(&transport->conn_mutex);
    if (slot >= 0 && slot < REPL_MAX_CONNECTIONS) {
        repl_clear_connection_pending(&transport->connections[slot]);
        transport->connections[slot].active = 0;
        transport->connections[slot].fd = -1;
        free(transport->connections[slot].node_id);
        transport->connections[slot].node_id = NULL;
    }
    pthread_mutex_unlock(&transport->conn_mutex);
    shutdown(fd, SHUT_RDWR);
    close(fd);
}

typedef struct {
    GV_ReplTransport *transport;
    int fd;
} ReplClientArg;

static void *repl_client_handler_thread(void *arg) {
    ReplClientArg *client = (ReplClientArg *)arg;
    repl_handle_client(client->transport, client->fd);
    free(client);
    return NULL;
}

static void *repl_accept_thread_func(void *arg) {
    GV_ReplTransport *transport = (GV_ReplTransport *)arg;
    while (!transport->stop_requested) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(transport->listen_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) {
            if (transport->stop_requested) break;
            usleep(100000);
            continue;
        }
        ReplClientArg *client = (ReplClientArg *)malloc(sizeof(*client));
        if (!client) {
            close(client_fd);
            continue;
        }
        client->transport = transport;
        client->fd = client_fd;
        pthread_t tid;
        if (pthread_create(&tid, NULL, repl_client_handler_thread, client) != 0) {
            free(client);
            close(client_fd);
            continue;
        }
        pthread_detach(tid);
    }
    return NULL;
}

static void *repl_follower_thread_func(void *arg) {
    GV_ReplTransport *transport = (GV_ReplTransport *)arg;
    GV_ReplicationManager *mgr = transport->mgr;
    const GV_ReplicationConfig *cfg = replication_get_config(mgr);
    if (!mgr || !cfg || !cfg->leader_address) return NULL;

    char host[256];
    uint16_t port = 0;
    if (parse_host_port(cfg->leader_address, host, sizeof(host), &port) != 0) {
        return NULL;
    }

    while (!transport->stop_requested) {
        int fd = repl_connect_host(host, port);
        if (fd < 0) {
            usleep(500000);
            continue;
        }
        repl_tune_socket(fd);
        transport->leader_fd = fd;

        const char *node_id = replication_get_node_id(mgr);
        if (!node_id) node_id = "follower";
        size_t nid_len = strlen(node_id);
        uint8_t *hello = (uint8_t *)malloc(4 + nid_len);
        if (!hello) {
            close(fd);
            continue;
        }
        write_u32_be(hello, (uint32_t)nid_len);
        memcpy(hello + 4, node_id, nid_len);
        repl_send_message(fd, REPL_MSG_HELLO, 1, hello, 4 + nid_len);
        free(hello);

        uint64_t catchup_from = 0;
        replication_get_positions(mgr, &catchup_from, NULL);

        uint8_t catchup_payload[8];
        write_u64_be(catchup_payload, catchup_from);
        repl_send_message(fd, REPL_MSG_CATCHUP, 2, catchup_payload, sizeof(catchup_payload));

        while (!transport->stop_requested) {
            uint8_t msg_type = 0;
            uint32_t req_id = 0;
            uint8_t *payload = NULL;
            size_t payload_len = 0;
            if (repl_transport_recv(transport, fd, &msg_type, &req_id, &payload, &payload_len) != 0) {
                break;
            }

            if (msg_type == REPL_MSG_WAL && payload_len >= 12) {
                uint64_t entry_index = read_u64_be(payload);
                uint32_t record_len = read_u32_be(payload + 8);
                if ((size_t)(12 + record_len) <= payload_len) {
                    repl_handle_wal_on_follower(mgr, entry_index, payload + 12, record_len);
                    uint8_t ack[8];
                    write_u64_be(ack, entry_index);
                    repl_transport_send(transport, fd, REPL_MSG_ACK, req_id, ack, sizeof(ack));
                }
            } else if (msg_type == REPL_MSG_HEARTBEAT) {
                replication_note_leader_heartbeat(mgr);
            }

            free(payload);
        }

        close(fd);
        transport->leader_fd = -1;
        usleep(500000);
    }
    return NULL;
}

static int repl_start_leader_listener(GV_ReplTransport *transport) {
    const GV_ReplicationConfig *cfg = replication_get_config(transport->mgr);
    if (!cfg || !cfg->listen_address) return 0;

    char host[256];
    uint16_t port = 0;
    if (parse_host_port(cfg->listen_address, host, sizeof(host), &port) != 0) {
        return -1;
    }

    transport->listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (transport->listen_fd < 0) return -1;

    int opt = 1;
    setsockopt(transport->listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (strcmp(host, "0.0.0.0") == 0) {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        inet_pton(AF_INET, host, &addr.sin_addr);
    }

    if (bind(transport->listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0 ||
        listen(transport->listen_fd, 16) < 0) {
        close(transport->listen_fd);
        transport->listen_fd = -1;
        return -1;
    }

    if (pthread_create(&transport->accept_thread, NULL, repl_accept_thread_func, transport) != 0) {
        close(transport->listen_fd);
        transport->listen_fd = -1;
        return -1;
    }
    transport->accept_thread_started = 1;
    return 0;
}

#endif /* !_WIN32 */

GV_ReplTransport *repl_transport_create(GV_ReplicationManager *mgr) {
    GV_ReplTransport *t = (GV_ReplTransport *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->mgr = mgr;
#ifndef _WIN32
    t->listen_fd = -1;
    t->leader_fd = -1;
    pthread_mutex_init(&t->conn_mutex, NULL);
    for (int i = 0; i < REPL_MAX_CONNECTIONS; i++) {
        t->connections[i].fd = -1;
    }
#endif
    return t;
}

void repl_transport_destroy(GV_ReplTransport *transport) {
    if (!transport) return;
    repl_transport_stop(transport);
#ifndef _WIN32
    pthread_mutex_destroy(&transport->conn_mutex);
#endif
    free(transport);
}

int repl_transport_start(GV_ReplTransport *transport) {
    if (!transport || !transport->mgr || transport->running) return -1;
    transport->stop_requested = 0;
    transport->running = 1;

#ifndef _WIN32
    GV_ReplicationManager *mgr = transport->mgr;
    const GV_ReplicationConfig *cfg = replication_get_config(mgr);
    GV_ReplicationRole role = replication_get_role_for_transport(mgr);
    if (role == GV_REPL_LEADER || (cfg && cfg->leader_address == NULL)) {
        if (repl_start_leader_listener(transport) != 0) {
            fprintf(stderr, "[GV_Repl] leader listener unavailable (continuing in-process)\n");
        }
    }
    if (cfg && cfg->leader_address != NULL) {
        if (pthread_create(&transport->follower_thread, NULL, repl_follower_thread_func, transport) != 0) {
            transport->running = 0;
            return -1;
        }
        transport->follower_thread_started = 1;
    }
#endif
    return 0;
}

int repl_transport_stop(GV_ReplTransport *transport) {
    if (!transport || !transport->running) return 0;
    transport->stop_requested = 1;

#ifndef _WIN32
    if (transport->listen_fd >= 0) {
        shutdown(transport->listen_fd, SHUT_RDWR);
        close(transport->listen_fd);
        transport->listen_fd = -1;
    }
    pthread_mutex_lock(&transport->conn_mutex);
    for (int i = 0; i < REPL_MAX_CONNECTIONS; i++) {
        if (transport->connections[i].active && transport->connections[i].fd >= 0) {
            shutdown(transport->connections[i].fd, SHUT_RDWR);
            close(transport->connections[i].fd);
            transport->connections[i].fd = -1;
            transport->connections[i].active = 0;
            free(transport->connections[i].node_id);
            transport->connections[i].node_id = NULL;
            repl_clear_connection_pending(&transport->connections[i]);
        }
    }
    pthread_mutex_unlock(&transport->conn_mutex);
    if (transport->accept_thread_started) {
        pthread_join(transport->accept_thread, NULL);
        transport->accept_thread_started = 0;
    }
    if (transport->follower_thread_started) {
        if (transport->leader_fd >= 0) {
            shutdown(transport->leader_fd, SHUT_RDWR);
            close(transport->leader_fd);
            transport->leader_fd = -1;
        }
        pthread_join(transport->follower_thread, NULL);
        transport->follower_thread_started = 0;
    }
#endif

    transport->running = 0;
    return 0;
}

void repl_transport_set_hooks(GV_ReplTransport *transport, const GV_ReplTransportHooks *hooks) {
    if (!transport) return;
    if (hooks) {
        transport->hooks = *hooks;
    } else {
        memset(&transport->hooks, 0, sizeof(transport->hooks));
    }
}

void repl_transport_clear_hooks(GV_ReplTransport *transport) {
    repl_transport_set_hooks(transport, NULL);
}

int repl_parse_frame_buffer(const uint8_t *data, size_t len, size_t max_bytes,
                            uint8_t *msg_type, uint32_t *request_id,
                            uint8_t **payload, size_t *payload_len) {
    if (!data || !msg_type || !request_id || !payload || !payload_len) return -1;
    *payload = NULL;
    *payload_len = 0;
    if (len < 9) return -1;

    uint32_t length = read_u32_be(data);
    if (length < 5 || length > max_bytes) return -1;
    if (len < 4u + length) return -1;

    *msg_type = data[4];
    *request_id = read_u32_be(data + 5);
    size_t plen = length - 5;
    *payload_len = plen;
    if (plen == 0) return 0;

    *payload = (uint8_t *)malloc(plen);
    if (!*payload) return -1;
    memcpy(*payload, data + 9, plen);
    return 0;
}

int repl_transport_broadcast_entry(GV_ReplTransport *transport, GV_Database *db,
                                   uint64_t entry_index) {
    if (!transport || !db) return -1;
    const char *path = db_wal_path(db);
    if (!path) return 0;

#ifndef _WIN32
    uint8_t type = 0;
    uint8_t *record = NULL;
    size_t record_len = 0;
    if (wal_read_entry_at(path, entry_index, &type, &record, &record_len) != 0) {
        free(record);
        return -1;
    }

    size_t payload_len = 12 + record_len;
    uint8_t *payload = (uint8_t *)malloc(payload_len);
    if (!payload) {
        free(record);
        return -1;
    }
    write_u64_be(payload, entry_index);
    write_u32_be(payload + 8, (uint32_t)record_len);
    memcpy(payload + 12, record, record_len);
    free(record);

    uint8_t heartbeat[16];
    {
        uint64_t wal_pos = 0, commit_pos = 0;
        replication_get_positions(transport->mgr, &wal_pos, &commit_pos);
        write_u64_be(heartbeat, wal_pos);
        write_u64_be(heartbeat + 8, commit_pos);
    }

    pthread_mutex_lock(&transport->conn_mutex);
    for (int i = 0; i < REPL_MAX_CONNECTIONS; i++) {
        if (!transport->connections[i].active) continue;
        ReplConnection *conn = &transport->connections[i];
        free(conn->pending_wal);
        conn->pending_wal = (uint8_t *)malloc(payload_len);
        if (!conn->pending_wal) continue;
        memcpy(conn->pending_wal, payload, payload_len);
        conn->pending_wal_len = payload_len;
        conn->pending_wal_req_id = (uint32_t)(entry_index + 1);
        conn->pending_wal_ready = 1;
        memcpy(conn->pending_heartbeat, heartbeat, sizeof(heartbeat));
        conn->pending_heartbeat_ready = 1;
    }
    pthread_mutex_unlock(&transport->conn_mutex);

    free(payload);
    return 0;
#else
    (void)entry_index;
    return -1;
#endif
}
