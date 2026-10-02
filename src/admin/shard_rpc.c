/**
 * @file shard_rpc.c
 * @brief TCP transport for distributed k-NN search (see shard_rpc.h).
 */

#ifdef _WIN32
/* This TCP transport relies on POSIX socket headers that are not available in
 * the MinGW/Windows build environment.  The Python wheel uses the library
 * directly (in-process) and never starts this network server, so provide an
 * in-process stub that preserves the API contract so the library still links
 * on Windows (the networked RPC features are never exercised there). */
#include "admin/shard_rpc.h"
#include "core/memory.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

struct GV_ShardRpcServer { int _unused; };

GV_ShardRpcServer *shard_rpc_serve(GV_ShardManager *mgr,
                                   const char *bind_addr, uint16_t port) {
    (void)mgr; (void)bind_addr; (void)port;
    return NULL;
}

uint16_t shard_rpc_server_port(const GV_ShardRpcServer *server) {
    (void)server;
    return 0;
}

void shard_rpc_set_members_provider(GV_ShardRpcServer *server,
                                    GV_MembersProviderFn fn, void *ctx) {
    (void)server; (void)fn; (void)ctx;
}

int shard_rpc_fetch_members(const char *host, uint16_t port,
                            char *out, size_t out_sz) {
    (void)host; (void)port; (void)out; (void)out_sz;
    return -1;
}

void shard_rpc_set_raft_handler(GV_ShardRpcServer *server,
                                GV_RaftMsgHandler fn, void *ctx) {
    (void)server; (void)fn; (void)ctx;
}

int shard_rpc_send_raft(const char *host, uint16_t port,
                        const void *bytes, size_t len) {
    (void)host; (void)port; (void)bytes; (void)len;
    return -1;
}

void shard_rpc_server_stop(GV_ShardRpcServer *server) {
    (void)server;
}

int shard_rpc_search(const char *host, uint16_t port,
                     const float *query, size_t dim, size_t k,
                     GV_DistanceType distance_type, GV_SearchResult *results) {
    (void)host; (void)port; (void)query; (void)dim; (void)k;
    (void)distance_type; (void)results;
    return -1;
}

int shard_rpc_search_distributed(GV_ShardManager *mgr, const float *query,
                                 size_t dim, size_t k,
                                 GV_DistanceType distance_type,
                                 GV_SearchResult *results) {
    (void)mgr; (void)query; (void)dim; (void)k; (void)distance_type; (void)results;
    return -1;
}

#else  /* POSIX implementation below */

#include "admin/shard_rpc.h"
#include "admin/shard.h"
#include "schema/vector.h"
#include "storage/database.h"  /* gv_search_results_free */
#include "core/memory.h"
#include "core/net_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <unistd.h>
#include <pthread.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <arpa/inet.h>

#define io_write_all gv_net_write_all
#define io_read_all  gv_net_read_all

#define SHARD_RPC_REQ_MAGIC  0x47565351u  /* "GVSQ" search request  */
#define SHARD_RPC_RESP_MAGIC 0x47565352u  /* "GVSR" search response */
#define SHARD_RPC_MEM_MAGIC  0x47564D45u  /* "GVME" members request/response */
#define SHARD_RPC_RAFT_MAGIC 0x47565254u  /* "GVRT" raft message (fire-and-forget) */
#define SHARD_RPC_MAX_DIM    (1u << 20)
#define SHARD_RPC_MAX_K      100000u
#define SHARD_RPC_MAX_MEM    (1u << 16)   /* max serialized member-list bytes */
#define SHARD_RPC_MAX_RAFT   (1u << 20)   /* max serialized raft-message bytes */

struct GV_ShardRpcServer {
    GV_ShardManager       *mgr;
    int                    listen_fd;
    uint16_t               port;
    pthread_t              thread;
    _Atomic int            running;
    GV_MembersProviderFn   members_fn;    /* optional: serve membership list */
    void                  *members_ctx;
    GV_RaftMsgHandler      raft_fn;        /* optional: deliver raft messages */
    void                  *raft_ctx;
};

/* ── server ───────────────────────────────────────────────────────────── */

static void shard_rpc_handle_search(GV_ShardManager *mgr, int fd) {
    uint32_t k, dim, dist;
    if (io_read_all(fd, &k, sizeof(k)) != 0) return;
    if (io_read_all(fd, &dim, sizeof(dim)) != 0) return;
    if (io_read_all(fd, &dist, sizeof(dist)) != 0) return;
    if (dim == 0 || dim > SHARD_RPC_MAX_DIM || k == 0 || k > SHARD_RPC_MAX_K) return;

    float *query = (float *)gv_alloc((size_t)dim * sizeof(float));
    if (!query) return;
    if (io_read_all(fd, query, (size_t)dim * sizeof(float)) != 0) { gv_free(query); return; }

    GV_SearchResult *res = (GV_SearchResult *)gv_calloc(k, sizeof(GV_SearchResult));
    if (!res) { gv_free(query); return; }

    int n = shard_search(mgr, query, k, res, (GV_DistanceType)dist);
    gv_free(query);
    if (n < 0) n = 0;

    uint32_t resp[2] = { SHARD_RPC_RESP_MAGIC, (uint32_t)n };
    int ok = io_write_all(fd, resp, sizeof(resp)) == 0;
    for (int i = 0; ok && i < n; i++) {
        uint64_t id = res[i].id;
        float d = res[i].distance;
        ok = io_write_all(fd, &id, sizeof(id)) == 0
          && io_write_all(fd, &d, sizeof(d)) == 0
          && res[i].vector != NULL
          && io_write_all(fd, res[i].vector->data, (size_t)dim * sizeof(float)) == 0;
    }
    gv_search_results_free(res, (size_t)n);
    gv_free(res);
}

static void shard_rpc_handle_members(GV_ShardRpcServer *s, int fd) {
    char buf[SHARD_RPC_MAX_MEM];
    int len = s->members_fn ? s->members_fn(buf, sizeof(buf), s->members_ctx) : 0;
    if (len < 0) len = 0;
    uint32_t resp[2] = { SHARD_RPC_MEM_MAGIC, (uint32_t)len };
    if (io_write_all(fd, resp, sizeof(resp)) == 0 && len > 0) {
        io_write_all(fd, buf, (size_t)len);
    }
}

/* Deliver a fire-and-forget raft message to the registered handler. */
static void shard_rpc_handle_raft(GV_ShardRpcServer *s, int fd) {
    uint32_t len;
    if (io_read_all(fd, &len, sizeof(len)) != 0) return;
    if (len == 0 || len > SHARD_RPC_MAX_RAFT) return;
    void *buf = gv_alloc(len);
    if (!buf) return;
    if (io_read_all(fd, buf, len) == 0 && s->raft_fn) {
        s->raft_fn(buf, len, s->raft_ctx);
    }
    gv_free(buf);
}

/* Dispatch one connection by leading request magic. */
static void shard_rpc_handle(GV_ShardRpcServer *s, int fd) {
    uint32_t magic;
    if (io_read_all(fd, &magic, sizeof(magic)) != 0) return;
    if (magic == SHARD_RPC_REQ_MAGIC) shard_rpc_handle_search(s->mgr, fd);
    else if (magic == SHARD_RPC_MEM_MAGIC) shard_rpc_handle_members(s, fd);
    else if (magic == SHARD_RPC_RAFT_MAGIC) shard_rpc_handle_raft(s, fd);
}

static void *shard_rpc_accept_loop(void *arg) {
    GV_ShardRpcServer *s = (GV_ShardRpcServer *)arg;
    while (s->running) {
        int cfd = accept(s->listen_fd, NULL, NULL);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            break; /* listen_fd closed by stop() */
        }
        if (s->running) shard_rpc_handle(s, cfd);
        close(cfd);
    }
    return NULL;
}

GV_ShardRpcServer *shard_rpc_serve(GV_ShardManager *mgr,
                                   const char *bind_addr, uint16_t port) {
    if (!mgr) return NULL;

    GV_ShardRpcServer *s = (GV_ShardRpcServer *)gv_calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->mgr = mgr;

    s->listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (s->listen_fd < 0) { gv_free(s); return NULL; }
    int one = 1;
    setsockopt(s->listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = (bind_addr && *bind_addr)
        ? inet_addr(bind_addr) : htonl(INADDR_ANY);

    if (bind(s->listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0 ||
        listen(s->listen_fd, 16) < 0) {
        close(s->listen_fd); gv_free(s); return NULL;
    }

    struct sockaddr_in bound;
    socklen_t blen = sizeof(bound);
    if (getsockname(s->listen_fd, (struct sockaddr *)&bound, &blen) == 0) {
        s->port = ntohs(bound.sin_port);
    } else {
        s->port = port;
    }

    s->running = 1;
    if (pthread_create(&s->thread, NULL, shard_rpc_accept_loop, s) != 0) {
        close(s->listen_fd); gv_free(s); return NULL;
    }
    return s;
}

uint16_t shard_rpc_server_port(const GV_ShardRpcServer *server) {
    return server ? server->port : 0;
}

void shard_rpc_set_members_provider(GV_ShardRpcServer *server,
                                    GV_MembersProviderFn fn, void *ctx) {
    if (!server) return;
    server->members_fn = fn;
    server->members_ctx = ctx;
}

void shard_rpc_set_raft_handler(GV_ShardRpcServer *server,
                                GV_RaftMsgHandler fn, void *ctx) {
    if (!server) return;
    server->raft_fn = fn;
    server->raft_ctx = ctx;
}

int shard_rpc_send_raft(const char *host, uint16_t port,
                        const void *bytes, size_t len) {
    if (!host || !bytes || len == 0 || len > SHARD_RPC_MAX_RAFT) return -1;

    char portstr[16];
    snprintf(portstr, sizeof(portstr), "%u", (unsigned)port);
    struct addrinfo hints, *ai = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, portstr, &hints, &ai) != 0 || !ai) return -1;

    int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (fd < 0 || connect(fd, ai->ai_addr, ai->ai_addrlen) != 0) {
        if (fd >= 0) close(fd);
        freeaddrinfo(ai);
        return -1;
    }
    freeaddrinfo(ai);

    uint32_t magic = SHARD_RPC_RAFT_MAGIC;
    uint32_t l = (uint32_t)len;
    int rc = (io_write_all(fd, &magic, sizeof(magic)) == 0 &&
              io_write_all(fd, &l, sizeof(l)) == 0 &&
              io_write_all(fd, bytes, len) == 0) ? 0 : -1;
    close(fd);
    return rc;
}

int shard_rpc_fetch_members(const char *host, uint16_t port,
                            char *out, size_t out_sz) {
    if (!host || !out || out_sz == 0) return -1;

    char portstr[16];
    snprintf(portstr, sizeof(portstr), "%u", (unsigned)port);
    struct addrinfo hints, *ai = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, portstr, &hints, &ai) != 0 || !ai) return -1;

    int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (fd < 0 || connect(fd, ai->ai_addr, ai->ai_addrlen) != 0) {
        if (fd >= 0) close(fd);
        freeaddrinfo(ai);
        return -1;
    }
    freeaddrinfo(ai);

    uint32_t magic = SHARD_RPC_MEM_MAGIC;
    if (io_write_all(fd, &magic, sizeof(magic)) != 0) { close(fd); return -1; }

    uint32_t resp[2];
    if (io_read_all(fd, resp, sizeof(resp)) != 0 || resp[0] != SHARD_RPC_MEM_MAGIC) {
        close(fd);
        return -1;
    }
    uint32_t len = resp[1];
    if (len >= out_sz) len = (uint32_t)out_sz - 1;
    if (len > 0 && io_read_all(fd, out, len) != 0) { close(fd); return -1; }
    out[len] = '\0';
    close(fd);
    return (int)len;
}

void shard_rpc_server_stop(GV_ShardRpcServer *server) {
    if (!server) return;
    server->running = 0;
    shutdown(server->listen_fd, SHUT_RDWR); /* unblock accept() */
    close(server->listen_fd);
    pthread_join(server->thread, NULL);
    gv_free(server);
}

/* ── client ───────────────────────────────────────────────────────────── */

int shard_rpc_search(const char *host, uint16_t port,
                     const float *query, size_t dim, size_t k,
                     GV_DistanceType distance_type, GV_SearchResult *results) {
    if (!host || !query || !results || dim == 0 || k == 0) return -1;

    char portstr[16];
    snprintf(portstr, sizeof(portstr), "%u", (unsigned)port);
    struct addrinfo hints, *ai = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, portstr, &hints, &ai) != 0 || !ai) return -1;

    int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (fd < 0 || connect(fd, ai->ai_addr, ai->ai_addrlen) != 0) {
        if (fd >= 0) close(fd);
        freeaddrinfo(ai);
        return -1;
    }
    freeaddrinfo(ai);

    uint32_t hdr[4] = { SHARD_RPC_REQ_MAGIC, (uint32_t)k, (uint32_t)dim,
                        (uint32_t)distance_type };
    int rc = -1;
    if (io_write_all(fd, hdr, sizeof(hdr)) != 0 ||
        io_write_all(fd, query, dim * sizeof(float)) != 0) {
        close(fd);
        return -1;
    }

    uint32_t resp[2];
    if (io_read_all(fd, resp, sizeof(resp)) != 0 ||
        resp[0] != SHARD_RPC_RESP_MAGIC) {
        close(fd);
        return -1;
    }

    uint32_t count = resp[1];
    if (count > k) count = (uint32_t)k; /* defensive */

    uint32_t got = 0;
    for (; got < count; got++) {
        uint64_t id; float d;
        float *vec = (float *)gv_alloc(dim * sizeof(float));
        if (!vec) break;
        if (io_read_all(fd, &id, sizeof(id)) != 0 ||
            io_read_all(fd, &d, sizeof(d)) != 0 ||
            io_read_all(fd, vec, dim * sizeof(float)) != 0) {
            gv_free(vec);
            break;
        }
        results[got].vector = vector_create_from_data(dim, vec);
        gv_free(vec);
        results[got].sparse_vector = NULL;
        results[got].is_sparse = 0;
        results[got].distance = d;
        results[got].id = (size_t)id;
    }
    rc = (int)got;

    close(fd);
    return rc;
}

int shard_rpc_search_distributed(GV_ShardManager *mgr, const float *query,
                                 size_t dim, size_t k,
                                 GV_DistanceType distance_type,
                                 GV_SearchResult *results) {
    if (!mgr || !query || !results || dim == 0 || k == 0) return -1;

    GV_ShardInfo *shards = NULL;
    size_t nshards = 0;
    if (shard_list(mgr, &shards, &nshards) != 0) return -1;

    /* Upper bound: local top-k plus one top-k per shard entry. */
    GV_SearchResult *all =
        (GV_SearchResult *)gv_alloc((nshards + 1) * k * sizeof(GV_SearchResult));
    if (!all) { shard_free_list(shards, nshards); return -1; }

    size_t total = 0;

    int ln = shard_search(mgr, query, k, all + total, distance_type);
    if (ln > 0) total += (size_t)ln;

    for (size_t i = 0; i < nshards; i++) {
        if (shards[i].state == GV_SHARD_OFFLINE) continue;
        if (shard_get_local_db(mgr, shards[i].shard_id) != NULL) continue;
        const char *addr = shards[i].node_address;
        if (!addr) continue;

        /* Fan out once per distinct remote address. */
        int seen = 0;
        for (size_t j = 0; j < i; j++) {
            if (shards[j].node_address &&
                strcmp(shards[j].node_address, addr) == 0 &&
                shards[j].state != GV_SHARD_OFFLINE &&
                shard_get_local_db(mgr, shards[j].shard_id) == NULL) {
                seen = 1; break;
            }
        }
        if (seen) continue;

        const char *colon = strrchr(addr, ':');
        if (!colon || colon == addr) continue;
        size_t hlen = (size_t)(colon - addr);
        char host[256];
        if (hlen >= sizeof(host)) continue;
        memcpy(host, addr, hlen);
        host[hlen] = '\0';
        uint16_t port = (uint16_t)(int)strtol(colon + 1, NULL, 10);
        if (port == 0) continue;

        int rn = shard_rpc_search(host, port, query, dim, k, distance_type,
                                  all + total);
        if (rn > 0) total += (size_t)rn; /* unreachable node -> skipped */
    }

    shard_free_list(shards, nshards);

    size_t kept = shard_merge_topk(all, total, k, results);
    gv_free(all);
    return (int)kept;
}

#endif /* _WIN32 */
