/**
 * @file repl_transport.c
 * @brief TCP WAL replication transport for GigaVector.
 */

#include "admin/repl_transport.h"
#include "core/memory.h"
#include "admin/replication.h"
#include "storage/database.h"
#include "storage/wal.h"
#include "core/utils.h"
#include "security/crypto.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
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
/* Generous headroom over REPL_MAX_CONNECTIONS: a connection is tracked here
 * from the instant accept() returns (before it has done any handshake I/O and
 * long before it might win a connections[] slot), so more of these can be
 * transiently in flight than there are registered connections at once. */
#define REPL_MAX_HANDLER_THREADS (REPL_MAX_CONNECTIONS * 2)

/* One queued WAL frame awaiting delivery to a follower. */
typedef struct PendingWal {
    uint8_t *data;
    size_t   len;
    uint32_t req_id;
    struct PendingWal *next;
} PendingWal;

/*
 * Tracks one repl_client_handler_thread from the moment it is created (see
 * repl_accept_thread_func) until it exits, independent of whether/when it
 * ever registers into transport->connections[]. Without this, a thread still
 * mid-handshake (blocked in recv() for HELLO/CATCHUP, or streaming a WAL
 * catchup) is invisible to repl_transport_stop() — it can then outlive
 * repl_transport_stop()/repl_transport_destroy() and touch conn_mutex/
 * transport after both have been destroyed/freed. All fields guarded by
 * conn_mutex.
 */
typedef struct {
    pthread_t thread;
    int fd;          /* the raw accepted fd this thread owns; -1 once closed */
    int in_use;      /* slot holds a tracked thread, from creation to reaping */
    int stop_owns;   /* repl_transport_stop() claimed this thread to join it;
                       * the thread must stay joinable (not self-detach) */
} ReplHandlerSlot;

typedef struct {
    /* Atomic so repl_transport_stop() can safely read it (to shutdown()) while
     * the owning handler thread may concurrently close it in its own cleanup;
     * only the handler thread ever clears it (via atomic_exchange), and only
     * while holding conn_mutex, so it can never race a slot being reused. */
    _Atomic int fd;
    char *node_id;
    int active;
    /* FIFO of pending WAL frames. A batch (or rapid appends) enqueues one frame
     * per entry; a single-slot buffer would drop all but the last and leave the
     * follower with a silent, permanent gap. */
    PendingWal *wal_head;
    PendingWal *wal_tail;
    uint32_t last_wal_req_id;   /* req_id of the most recently flushed WAL frame */
    uint8_t pending_heartbeat[16];
    int pending_heartbeat_ready;
} ReplConnection;

struct GV_ReplTransport {
    GV_ReplicationManager *mgr;
    int running;
    _Atomic int stop_requested;
    GV_ReplTransportHooks hooks;

#ifndef _WIN32
    int listen_fd;
    pthread_t accept_thread;
    int accept_thread_started;
    pthread_t follower_thread;
    int follower_thread_started;
    /* Written by the follower thread (on connect/reconnect) and read/closed by
     * the stop path. Atomic so the cross-thread int access is race-free; the fd
     * is only close()d after the follower thread is joined (a close() while the
     * follower is mid-recv() on it is a use-after-close / TSAN fd race). */
    _Atomic int leader_fd;
    ReplConnection connections[REPL_MAX_CONNECTIONS];
    ReplHandlerSlot handler_threads[REPL_MAX_HANDLER_THREADS];
    pthread_mutex_t conn_mutex;
    /* Separate from conn_mutex: repl_transport_send()/repl_transport_recv()
     * (which read `hooks`) are called from places that already hold
     * conn_mutex (e.g. repl_flush_connection_pending under repl_handle_client's
     * poll loop) — reusing conn_mutex here would self-deadlock on relock of a
     * non-recursive mutex. hooks_mutex guards only the `hooks` field. */
    pthread_mutex_t hooks_mutex;
#endif
};

static void write_u64_be(uint8_t *buf, uint64_t val) {
    gv_put_u32_be(buf, (uint32_t)(val >> 32));
    gv_put_u32_be(buf + 4, (uint32_t)(val & 0xFFFFFFFFu));
}

static uint64_t read_u64_be(const uint8_t *buf) {
    return ((uint64_t)gv_get_u32_be(buf) << 32) | gv_get_u32_be(buf + 4);
}

#ifndef _WIN32

/*
 * Optional replication shared secret.
 *
 * Backward-compatible auth: if the GV_REPL_SECRET environment variable is set
 * and non-empty, the leader requires every connecting replica to present the
 * same secret (appended to the HELLO frame after the node id) and rejects the
 * connection otherwise, using a constant-time comparison. If the variable is
 * unset/empty the transport behaves exactly as before (no auth), so existing
 * deployments and tests are unaffected.
 *
 * The env var is read once on first use and cached; the returned pointer is
 * either NULL (no secret) or a stable string owned by the environment.
 */
static const char *repl_shared_secret(void) {
    static const char *cached = NULL;
    static int initialized = 0;
    if (!initialized) {
        const char *s = getenv("GV_REPL_SECRET");
        if (s && s[0] != '\0') {
            cached = s;
        }
        initialized = 1;
    }
    return cached;
}

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
    gv_put_u32_be(header, length);
    header[4] = msg_type;
    gv_put_u32_be(header + 5, request_id);
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
    uint32_t length = gv_get_u32_be(header);
    if (length < 5 || length > REPL_MAX_MSG_BYTES) return -1;

    uint8_t meta[5];
    if (recv_exact(fd, meta, 5) != 0) return -1;
    *msg_type = meta[0];
    *request_id = gv_get_u32_be(meta + 1);
    size_t plen = length - 5;
    *payload_len = plen;
    if (plen == 0) {
        *payload = NULL;
        return 0;
    }
    *payload = (uint8_t *)gv_alloc(plen);
    if (!*payload) return -1;
    if (recv_exact(fd, *payload, plen) != 0) {
        gv_free(*payload);
        *payload = NULL;
        return -1;
    }
    return 0;
}

/* transport->hooks can be set/cleared (by test/DST code) from a thread other
 * than the one calling send/recv, concurrently with the accept/follower/
 * handler threads reading it here — copy it out under hooks_mutex rather than
 * reading the live struct field, and invoke the callback on the copy outside
 * the lock (so a hook that itself touches the transport can't deadlock).
 * hooks_mutex (not conn_mutex) because callers of repl_transport_send/recv can
 * already hold conn_mutex (see repl_flush_connection_pending). */
static GV_ReplTransportHooks repl_transport_hooks_snapshot(GV_ReplTransport *transport) {
    GV_ReplTransportHooks hooks;
    memset(&hooks, 0, sizeof(hooks));
    if (!transport) return hooks;
    pthread_mutex_lock(&transport->hooks_mutex);
    hooks = transport->hooks;
    pthread_mutex_unlock(&transport->hooks_mutex);
    return hooks;
}

static int repl_transport_send(GV_ReplTransport *transport, int fd, uint8_t msg_type,
                               uint32_t request_id, const uint8_t *payload, size_t payload_len) {
    GV_ReplTransportHooks hooks = repl_transport_hooks_snapshot(transport);
    if (hooks.filter_outbound) {
        if (hooks.filter_outbound(hooks.ctx, msg_type, payload, payload_len) != 0) {
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
    GV_ReplTransportHooks hooks = repl_transport_hooks_snapshot(transport);
    if (hooks.filter_inbound && *payload_len > 0) {
        if (hooks.filter_inbound(hooks.ctx, *msg_type, *payload, *payload_len) != 0) {
            gv_free(*payload);
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
    *port = (uint16_t)(int)strtol(colon + 1, NULL, 10);
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
    PendingWal *p = conn->wal_head;
    while (p) { PendingWal *n = p->next; gv_free(p->data); gv_free(p); p = n; }
    conn->wal_head = NULL;
    conn->wal_tail = NULL;
    conn->pending_heartbeat_ready = 0;
}

static void repl_flush_connection_pending(GV_ReplTransport *transport, ReplConnection *conn) {
    if (!conn || !conn->active || conn->fd < 0) return;
    /* Send every queued WAL frame in order (not just the latest). */
    while (conn->wal_head) {
        PendingWal *p = conn->wal_head;
        conn->wal_head = p->next;
        if (!conn->wal_head) conn->wal_tail = NULL;
        repl_transport_send(transport, conn->fd, REPL_MSG_WAL, p->req_id, p->data, p->len);
        conn->last_wal_req_id = p->req_id;
        gv_free(p->data);
        gv_free(p);
    }
    if (conn->pending_heartbeat_ready) {
        repl_transport_send(transport, conn->fd, REPL_MSG_HEARTBEAT, conn->last_wal_req_id + 1,
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
            gv_free(record);
            return -1;
        }

        size_t payload_len = 12 + record_len;
        uint8_t *payload = (uint8_t *)gv_alloc(payload_len);
        if (!payload) {
            gv_free(record);
            return -1;
        }
        write_u64_be(payload, i);
        gv_put_u32_be(payload + 8, (uint32_t)record_len);
        memcpy(payload + 12, record, record_len);
        gv_free(record);

        int rc = repl_transport_send(transport, fd, REPL_MSG_WAL, (uint32_t)(i + 1),
                                       payload, payload_len);
        gv_free(payload);
        if (rc != 0) return -1;
    }
    return 0;
}

/*
 * Releases a connection's fd from its owning handler thread. `tslot` (may be
 * -1 if the tracking table was full at accept time) is cleared under
 * conn_mutex *before* the fd is physically closed, so repl_transport_stop()
 * can never observe a stale fd number through handler_threads[tslot].fd and
 * shutdown() an unrelated descriptor that number was recycled into after
 * close(). shutdown() before close() is a no-op if the peer already dropped
 * the connection and otherwise unblocks any last poll()/recv() cleanly.
 */
static void repl_release_fd(GV_ReplTransport *transport, int tslot, int fd) {
    if (fd < 0) return;
    if (tslot >= 0) {
        pthread_mutex_lock(&transport->conn_mutex);
        transport->handler_threads[tslot].fd = -1;
        pthread_mutex_unlock(&transport->conn_mutex);
    }
    shutdown(fd, SHUT_RDWR);
    close(fd);
}

/* Handles one accepted connection to completion. `tslot` identifies this
 * thread's entry in transport->handler_threads[] (see repl_accept_thread_func
 * and ReplHandlerSlot) so every fd-release point can keep that tracking
 * current; the join-vs-detach decision itself is made by the caller
 * (repl_client_handler_thread) after this function returns. */
static void repl_handle_client(GV_ReplTransport *transport, int fd, int tslot) {
    GV_ReplicationManager *mgr = transport->mgr;
    repl_tune_socket(fd);
    char node_id[256] = {0};
    uint8_t msg_type = 0;
    uint32_t req_id = 0;
    uint8_t *payload = NULL;
    size_t payload_len = 0;

    if (repl_recv_message(fd, &msg_type, &req_id, &payload, &payload_len) != 0 ||
        msg_type != REPL_MSG_HELLO || payload_len < 4) {
        gv_free(payload);
        repl_release_fd(transport, tslot, fd);
        return;
    }
    uint32_t nid_len = gv_get_u32_be(payload);
    if (nid_len >= sizeof(node_id) || payload_len < 4 + nid_len) {
        gv_free(payload);
        repl_release_fd(transport, tslot, fd);
        return;
    }
    memcpy(node_id, payload + 4, nid_len);
    node_id[nid_len] = '\0';

    /*
     * Optional shared-secret authentication (backward compatible). When a
     * secret is configured, the replica must append exactly that secret to the
     * HELLO frame after the node id. Reject any client that omits it or
     * presents a wrong one, using a constant-time comparison. When no secret is
     * configured this check is skipped and any trailing bytes are ignored,
     * preserving the previous (unauthenticated) behavior.
     */
    const char *secret = repl_shared_secret();
    if (secret) {
        size_t secret_len = strlen(secret);
        size_t offered_off = 4 + (size_t)nid_len;
        size_t offered_len = payload_len - offered_off;
        if (offered_len != secret_len ||
            crypto_constant_time_compare((const unsigned char *)(payload + offered_off),
                                         (const unsigned char *)secret, secret_len) != 0) {
            gv_free(payload);
            repl_release_fd(transport, tslot, fd);
            return;
        }
    }
    gv_free(payload);

    uint64_t catchup_from = 0;
    replication_replica_handshake(mgr, node_id, &catchup_from);

    if (repl_recv_message(fd, &msg_type, &req_id, &payload, &payload_len) == 0 &&
        msg_type == REPL_MSG_CATCHUP && payload_len >= 8) {
        catchup_from = read_u64_be(payload);
        gv_free(payload);
    } else {
        gv_free(payload);
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
        repl_release_fd(transport, tslot, fd);
        return;
    }

    while (!atomic_load(&transport->stop_requested)) {
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
        gv_free(payload);
    }

    int own_fd = -1;
    pthread_mutex_lock(&transport->conn_mutex);
    if (slot >= 0 && slot < REPL_MAX_CONNECTIONS) {
        repl_clear_connection_pending(&transport->connections[slot]);
        /* Capture-and-clear the fd before dropping `active`: once active is 0
         * this slot is eligible for reuse by a new connection, and exchanging
         * the fd after that point could steal or clobber the new occupant's fd. */
        own_fd = atomic_exchange(&transport->connections[slot].fd, -1);
        transport->connections[slot].active = 0;
        gv_free(transport->connections[slot].node_id);
        transport->connections[slot].node_id = NULL;
    }
    /* Clear handler_threads[tslot].fd in this same critical section, before
     * own_fd is physically closed below — see repl_release_fd for why. */
    if (tslot >= 0) {
        transport->handler_threads[tslot].fd = -1;
    }
    pthread_mutex_unlock(&transport->conn_mutex);
    if (own_fd >= 0) {
        shutdown(own_fd, SHUT_RDWR);
        close(own_fd);
    }
}

typedef struct {
    GV_ReplTransport *transport;
    int fd;
    int tslot;   /* index into transport->handler_threads[], or -1 if the
                  * tracking table was full at accept time (see below) */
} ReplClientArg;

static void *repl_client_handler_thread(void *arg) {
    ReplClientArg *client = (ReplClientArg *)arg;
    GV_ReplTransport *transport = client->transport;
    int tslot = client->tslot;
    if (tslot < 0) {
        /* handler_threads[] was full at accept time: this thread is invisible
         * to repl_transport_stop(), so it must detach itself now rather than
         * leak (never joined, never detached) until process exit. */
        pthread_detach(pthread_self());
    }
    repl_handle_client(transport, client->fd, tslot);
    if (tslot >= 0) {
        int self_detach = 0;
        pthread_mutex_lock(&transport->conn_mutex);
        if (!transport->handler_threads[tslot].stop_owns) {
            /* No stop is in progress, so nobody will ever join this thread;
             * detach it now to release its resources. If stop already claimed
             * it (stop_owns), stay joinable — joining an already-detached
             * thread is undefined behavior; repl_transport_stop() clears
             * in_use once its pthread_join() returns. */
            transport->handler_threads[tslot].in_use = 0;
            self_detach = 1;
        }
        pthread_mutex_unlock(&transport->conn_mutex);
        if (self_detach) {
            pthread_detach(pthread_self());
        }
    }
    gv_free(client);
    return NULL;
}

static void *repl_accept_thread_func(void *arg) {
    GV_ReplTransport *transport = (GV_ReplTransport *)arg;
    while (!atomic_load(&transport->stop_requested)) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(transport->listen_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) {
            if (atomic_load(&transport->stop_requested)) break;
            usleep(100000);
            continue;
        }
        ReplClientArg *client = (ReplClientArg *)gv_alloc(sizeof(*client));
        if (!client) {
            close(client_fd);
            continue;
        }
        client->transport = transport;
        client->fd = client_fd;

        /*
         * Reserve a handler_threads[] slot (recording its fd) and spawn the
         * thread in one critical section, so there is no window where the
         * thread is running but not yet visible to repl_transport_stop() —
         * that window is what let a still-handshaking connection outlive
         * stop()/destroy() and touch a freed transport. Holding conn_mutex
         * across pthread_create() is safe: it does not call back into any of
         * our locking code.
         */
        pthread_mutex_lock(&transport->conn_mutex);
        int tslot = -1;
        for (int i = 0; i < REPL_MAX_HANDLER_THREADS; i++) {
            if (!transport->handler_threads[i].in_use) {
                tslot = i;
                break;
            }
        }
        if (tslot >= 0) {
            transport->handler_threads[tslot].in_use = 1;
            transport->handler_threads[tslot].stop_owns = 0;
            transport->handler_threads[tslot].fd = client_fd;
        }
        client->tslot = tslot;
        pthread_t tid;
        int rc = pthread_create(&tid, NULL, repl_client_handler_thread, client);
        if (rc == 0) {
            if (tslot >= 0) transport->handler_threads[tslot].thread = tid;
        } else if (tslot >= 0) {
            transport->handler_threads[tslot].in_use = 0;
        }
        pthread_mutex_unlock(&transport->conn_mutex);
        if (rc != 0) {
            gv_free(client);
            close(client_fd);
            continue;
        }
        /* Not detached here: repl_client_handler_thread decides for itself
         * whether to stay joinable or detach, based on handler_threads[tslot]. */
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

    while (!atomic_load(&transport->stop_requested)) {
        int fd = repl_connect_host(host, port);
        if (fd < 0) {
            usleep(500000);
            continue;
        }
        repl_tune_socket(fd);
        pthread_mutex_lock(&transport->conn_mutex);
        if (atomic_load(&transport->stop_requested)) {
            /*
             * A stop was already requested. repl_transport_stop() takes this
             * same lock to read+shutdown() leader_fd exactly once before
             * joining this thread; if it already ran that step (or runs it
             * concurrently and loses this lock race), publishing this
             * brand-new fd now would leave it un-shut-down, and this thread
             * would then block in the recv loop below until its 10s
             * SO_RCVTIMEO fires — stalling repl_transport_stop()'s join for
             * that long. Discard the fd here instead of entering the loop.
             */
            pthread_mutex_unlock(&transport->conn_mutex);
            close(fd);
            break;
        }
        transport->leader_fd = fd;
        pthread_mutex_unlock(&transport->conn_mutex);

        const char *node_id = replication_get_node_id(mgr);
        if (!node_id) node_id = "follower";
        size_t nid_len = strlen(node_id);
        /*
         * If a shared secret is configured, append it to the HELLO frame after
         * the node id so the leader can authenticate this replica. When no
         * secret is configured the frame is identical to before.
         */
        const char *secret = repl_shared_secret();
        size_t secret_len = secret ? strlen(secret) : 0;
        size_t hello_len = 4 + nid_len + secret_len;
        uint8_t *hello = (uint8_t *)gv_alloc(hello_len);
        if (!hello) {
            close(fd);
            continue;
        }
        gv_put_u32_be(hello, (uint32_t)nid_len);
        memcpy(hello + 4, node_id, nid_len);
        if (secret_len > 0) {
            memcpy(hello + 4 + nid_len, secret, secret_len);
        }
        repl_send_message(fd, REPL_MSG_HELLO, 1, hello, hello_len);
        gv_free(hello);

        uint64_t catchup_from = 0;
        replication_get_positions(mgr, &catchup_from, NULL);

        uint8_t catchup_payload[8];
        write_u64_be(catchup_payload, catchup_from);
        repl_send_message(fd, REPL_MSG_CATCHUP, 2, catchup_payload, sizeof(catchup_payload));

        while (!atomic_load(&transport->stop_requested)) {
            uint8_t msg_type = 0;
            uint32_t req_id = 0;
            uint8_t *payload = NULL;
            size_t payload_len = 0;
            if (repl_transport_recv(transport, fd, &msg_type, &req_id, &payload, &payload_len) != 0) {
                break;
            }

            if (msg_type == REPL_MSG_WAL && payload_len >= 12) {
                uint64_t entry_index = read_u64_be(payload);
                uint32_t record_len = gv_get_u32_be(payload + 8);
                /*
                 * Bound check written to avoid 32-bit unsigned overflow:
                 * `12 + record_len` would wrap for record_len near UINT32_MAX
                 * and spuriously pass. payload_len >= 12 is guaranteed by the
                 * outer condition, so `payload_len - 12` cannot underflow.
                 */
                if (record_len <= payload_len - 12) {
                    repl_handle_wal_on_follower(mgr, entry_index, payload + 12, record_len);
                    uint8_t ack[8];
                    write_u64_be(ack, entry_index);
                    repl_transport_send(transport, fd, REPL_MSG_ACK, req_id, ack, sizeof(ack));
                }
            } else if (msg_type == REPL_MSG_HEARTBEAT) {
                replication_note_leader_heartbeat(mgr);
            }

            gv_free(payload);
        }

        /*
         * Close the leader fd exactly once. The stop path may concurrently
         * capture-and-close leader_fd; whoever clears the sentinel first owns
         * the close. atomic_exchange alone (no mutex needed) makes this a
         * single-variable race-free handoff between the two possible closers.
         */
        int own_fd = atomic_exchange(&transport->leader_fd, -1);
        if (own_fd >= 0) {
            close(own_fd);
        }
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
    GV_ReplTransport *t = (GV_ReplTransport *)gv_calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->mgr = mgr;
#ifndef _WIN32
    t->listen_fd = -1;
    t->leader_fd = -1;
    pthread_mutex_init(&t->conn_mutex, NULL);
    pthread_mutex_init(&t->hooks_mutex, NULL);
    for (int i = 0; i < REPL_MAX_CONNECTIONS; i++) {
        t->connections[i].fd = -1;
    }
    for (int i = 0; i < REPL_MAX_HANDLER_THREADS; i++) {
        t->handler_threads[i].fd = -1;
    }
#endif
    return t;
}

void repl_transport_destroy(GV_ReplTransport *transport) {
    if (!transport) return;
    repl_transport_stop(transport);
#ifndef _WIN32
    pthread_mutex_destroy(&transport->conn_mutex);
    pthread_mutex_destroy(&transport->hooks_mutex);
#endif
    gv_free(transport);
}

int repl_transport_start(GV_ReplTransport *transport) {
    if (!transport || !transport->mgr || transport->running) return -1;
    atomic_store(&transport->stop_requested, 0);
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
    atomic_store(&transport->stop_requested, 1);

#ifndef _WIN32
    /* Only shut the listener down here to unblock a blocked accept(); do NOT
     * close/clear listen_fd yet — the accept thread still reads it each loop
     * (repl_accept_thread_func), so mutating it before the join is a data race.
     * The close+clear happens after the join below. */
    if (transport->listen_fd >= 0) {
        shutdown(transport->listen_fd, SHUT_RDWR);
    }
    /*
     * Only shutdown() handler fds here (never close()) to unblock each
     * handler thread's blocking recv()/send()/poll() without invalidating the
     * fd it may still be using — this covers a connection at any stage, from
     * still mid-handshake (not yet in connections[]) through fully registered,
     * since handler_threads[] tracks every accepted connection from the
     * instant its thread is created (see repl_accept_thread_func). Claim
     * (stop_owns) every still-tracked thread while holding the lock so that
     * when it wakes on the shutdown and reaches its own cleanup, it sees the
     * claim and stays joinable instead of self-detaching — then join each of
     * them below so repl_transport_stop() never returns (and
     * repl_transport_destroy() never frees/destroys transport/conn_mutex)
     * while a handler thread might still be touching either. The handler
     * thread itself performs the actual close() as part of that same cleanup
     * (repl_release_fd / repl_handle_client's tail), so there is nothing left
     * for stop() to close here.
     */
    int join_slots[REPL_MAX_HANDLER_THREADS];
    int join_count = 0;
    pthread_mutex_lock(&transport->conn_mutex);
    for (int i = 0; i < REPL_MAX_HANDLER_THREADS; i++) {
        if (transport->handler_threads[i].in_use) {
            int fd = transport->handler_threads[i].fd;
            if (fd >= 0) shutdown(fd, SHUT_RDWR);
            transport->handler_threads[i].stop_owns = 1;
            join_slots[join_count++] = i;
        }
    }
    pthread_mutex_unlock(&transport->conn_mutex);
    if (transport->accept_thread_started) {
        pthread_join(transport->accept_thread, NULL);
        transport->accept_thread_started = 0;
    }
    /* Accept thread has exited; now no one else touches listen_fd. */
    if (transport->listen_fd >= 0) {
        close(transport->listen_fd);
        transport->listen_fd = -1;
    }
    for (int i = 0; i < join_count; i++) {
        pthread_join(transport->handler_threads[join_slots[i]].thread, NULL);
    }
    if (join_count > 0) {
        /* Recycle the claimed slots so a later repl_transport_start() on this
         * same transport doesn't find handler_threads[] permanently exhausted. */
        pthread_mutex_lock(&transport->conn_mutex);
        for (int i = 0; i < join_count; i++) {
            transport->handler_threads[join_slots[i]].in_use = 0;
            transport->handler_threads[join_slots[i]].stop_owns = 0;
        }
        pthread_mutex_unlock(&transport->conn_mutex);
    }
    if (transport->follower_thread_started) {
        /*
         * shutdown() (not close()) to unblock the follower's recv() without
         * invalidating the fd it is still using — closing it here while the
         * follower reads it is a use-after-close / TSAN fd race. Read
         * leader_fd under conn_mutex — the same lock the follower takes
         * before publishing a freshly-connected fd (see repl_follower_thread_func)
         * — so the two can't race: either this runs first and shuts down
         * whatever the follower already published, or the follower's publish
         * step runs first, sees stop_requested (already set above) under this
         * same lock, and discards its new fd itself instead of publishing it.
         * Join first; only then capture-and-close, so exactly one of
         * {follower, stop} closes (atomic_exchange on the -1 sentinel) with no
         * thread still using the fd.
         */
        pthread_mutex_lock(&transport->conn_mutex);
        int lf = atomic_load(&transport->leader_fd);
        if (lf >= 0) shutdown(lf, SHUT_RDWR);
        pthread_mutex_unlock(&transport->conn_mutex);
        pthread_join(transport->follower_thread, NULL);
        transport->follower_thread_started = 0;
        int own = atomic_exchange(&transport->leader_fd, -1);
        if (own >= 0) close(own);
    }
#endif

    transport->running = 0;
    return 0;
}

void repl_transport_set_hooks(GV_ReplTransport *transport, const GV_ReplTransportHooks *hooks) {
    if (!transport) return;
    /* hooks_mutex (non-Windows only) also guards reads of transport->hooks in
     * repl_transport_send/recv, which run concurrently with this from the
     * accept/follower/handler threads — see repl_transport_hooks_snapshot(). */
#ifndef _WIN32
    pthread_mutex_lock(&transport->hooks_mutex);
#endif
    if (hooks) {
        transport->hooks = *hooks;
    } else {
        memset(&transport->hooks, 0, sizeof(transport->hooks));
    }
#ifndef _WIN32
    pthread_mutex_unlock(&transport->hooks_mutex);
#endif
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

    uint32_t length = gv_get_u32_be(data);
    if (length < 5 || length > max_bytes) return -1;
    if (len < 4u + length) return -1;

    *msg_type = data[4];
    *request_id = gv_get_u32_be(data + 5);
    size_t plen = length - 5;
    *payload_len = plen;
    if (plen == 0) return 0;

    *payload = (uint8_t *)gv_alloc(plen);
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
        gv_free(record);
        return -1;
    }

    size_t payload_len = 12 + record_len;
    uint8_t *payload = (uint8_t *)gv_alloc(payload_len);
    if (!payload) {
        gv_free(record);
        return -1;
    }
    write_u64_be(payload, entry_index);
    gv_put_u32_be(payload + 8, (uint32_t)record_len);
    memcpy(payload + 12, record, record_len);
    gv_free(record);

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
        PendingWal *node = (PendingWal *)gv_alloc(sizeof(PendingWal));
        if (!node) continue;
        node->data = (uint8_t *)gv_alloc(payload_len);
        if (!node->data) { gv_free(node); continue; }
        memcpy(node->data, payload, payload_len);
        node->len = payload_len;
        node->req_id = (uint32_t)(entry_index + 1);
        node->next = NULL;
        if (conn->wal_tail) conn->wal_tail->next = node; else conn->wal_head = node;
        conn->wal_tail = node;
        memcpy(conn->pending_heartbeat, heartbeat, sizeof(heartbeat));
        conn->pending_heartbeat_ready = 1;
    }
    pthread_mutex_unlock(&transport->conn_mutex);

    gv_free(payload);
    return 0;
#else
    (void)entry_index;
    return -1;
#endif
}
