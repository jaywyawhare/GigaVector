/**
 * @file graph_rpc.c
 * @brief TCP transport for cross-partition graph traversal (see graph_rpc.h).
 */

#ifdef _WIN32
/* This TCP transport relies on POSIX socket headers that are not available in
 * the MinGW/Windows build environment.  The Python wheel uses the library
 * directly (in-process) and never starts this network server, so provide an
 * in-process stub that preserves the API contract so the library still links
 * on Windows (the networked RPC features are never exercised there). */
#include "features/graph_rpc.h"
#include "core/memory.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

struct GV_GraphRpcServer { int _unused; };

GV_GraphRpcServer *graph_rpc_serve(GV_GraphDB *partition,
                                   const char *bind_addr, uint16_t port) {
    (void)partition; (void)bind_addr; (void)port;
    return NULL;
}

uint16_t graph_rpc_server_port(const GV_GraphRpcServer *server) {
    (void)server;
    return 0;
}

void graph_rpc_server_stop(GV_GraphRpcServer *server) {
    (void)server;
}

int graph_rpc_neighbors(const char *host, uint16_t port, uint64_t node_id,
                        const char *predicate, uint64_t *out, size_t max) {
    (void)host; (void)port; (void)node_id; (void)predicate; (void)out; (void)max;
    return -1;
}

int graph_dist_khop_net(const GV_GraphPartition *parts, size_t nparts,
                        GV_GraphPlacementFn place, void *place_ctx,
                        uint64_t start, size_t k, const char *predicate,
                        uint64_t *out_ids, size_t max_out, size_t max_nodes) {
    (void)parts; (void)nparts; (void)place; (void)place_ctx; (void)start;
    (void)k; (void)predicate; (void)out_ids; (void)max_out; (void)max_nodes;
    return -1;
}

int cypher_dist_match(const GV_GraphPartition *parts, size_t nparts,
                      GV_GraphPlacementFn place, void *place_ctx,
                      uint64_t start_id, const char *rel_type,
                      size_t min_hops, size_t max_hops,
                      uint64_t *out_ids, size_t max_out, size_t max_nodes) {
    (void)parts; (void)nparts; (void)place; (void)place_ctx; (void)start_id;
    (void)rel_type; (void)min_hops; (void)max_hops; (void)out_ids; (void)max_out;
    (void)max_nodes;
    return -1;
}

int cypher_dist_path(const GV_GraphPartition *parts, size_t nparts,
                     GV_GraphPlacementFn place, void *place_ctx,
                     uint64_t start_id, const GV_CypherSegment *segments,
                     size_t n_segments, uint64_t *out_ids, size_t max_out,
                     size_t max_nodes) {
    (void)parts; (void)nparts; (void)place; (void)place_ctx; (void)start_id;
    (void)segments; (void)n_segments; (void)out_ids; (void)max_out;
    (void)max_nodes;
    return -1;
}

int cypher_dist_query(const GV_GraphPartition *parts, size_t nparts,
                      GV_GraphPlacementFn place, void *place_ctx,
                      const char *query,
                      uint64_t *out_ids, size_t max_out, size_t max_nodes) {
    (void)parts; (void)nparts; (void)place; (void)place_ctx; (void)query;
    (void)out_ids; (void)max_out; (void)max_nodes;
    return -1;
}

#else  /* POSIX implementation below */

#include "features/graph_rpc.h"
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
#include <sys/socket.h>
#include <arpa/inet.h>

#define GRAPH_RPC_REQ_MAGIC  0x47475251u  /* "GGRQ" */
#define GRAPH_RPC_RESP_MAGIC 0x47475253u  /* "GGRS" */
#define GRAPH_RPC_MAX_NEIGH  (1u << 24)

struct GV_GraphRpcServer {
    GV_GraphDB *partition;
    int         listen_fd;
    uint16_t    port;
    pthread_t   thread;
    _Atomic int running;
};

/* ── server ───────────────────────────────────────────────────────────── */

#define GRAPH_RPC_MAX_PRED 4096u

static void graph_rpc_handle(GV_GraphDB *g, int fd) {
    uint32_t magic;
    uint64_t node_id;
    uint32_t max, plen;
    if (gv_net_read_all(fd, &magic, sizeof(magic)) != 0 || magic != GRAPH_RPC_REQ_MAGIC)
        return;
    if (gv_net_read_all(fd, &node_id, sizeof(node_id)) != 0) return;
    if (gv_net_read_all(fd, &max, sizeof(max)) != 0) return;
    if (gv_net_read_all(fd, &plen, sizeof(plen)) != 0) return;
    if (max > GRAPH_RPC_MAX_NEIGH) max = GRAPH_RPC_MAX_NEIGH;
    if (plen > GRAPH_RPC_MAX_PRED) return;

    char predbuf[GRAPH_RPC_MAX_PRED + 1];
    const char *predicate = NULL;
    if (plen > 0) {
        if (gv_net_read_all(fd, predbuf, plen) != 0) return;
        predbuf[plen] = '\0';
        predicate = predbuf;
    }

    uint64_t *neigh = max ? (uint64_t *)gv_alloc((size_t)max * sizeof(uint64_t)) : NULL;
    int n = 0;
    if (neigh) {
        n = graph_get_out_neighbors_typed(g, node_id, predicate, neigh, max);
        if (n < 0) n = 0;
    }

    uint32_t resp[2] = { GRAPH_RPC_RESP_MAGIC, (uint32_t)n };
    if (gv_net_write_all(fd, resp, sizeof(resp)) == 0 && n > 0) {
        gv_net_write_all(fd, neigh, (size_t)n * sizeof(uint64_t));
    }
    gv_free(neigh);
}

static void *graph_rpc_accept_loop(void *arg) {
    GV_GraphRpcServer *s = (GV_GraphRpcServer *)arg;
    while (s->running) {
        int cfd = accept(s->listen_fd, NULL, NULL);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (s->running) graph_rpc_handle(s->partition, cfd);
        close(cfd);
    }
    return NULL;
}

GV_GraphRpcServer *graph_rpc_serve(GV_GraphDB *partition,
                                   const char *bind_addr, uint16_t port) {
    if (!partition) return NULL;

    GV_GraphRpcServer *s = (GV_GraphRpcServer *)gv_calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->partition = partition;

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
    s->port = (getsockname(s->listen_fd, (struct sockaddr *)&bound, &blen) == 0)
        ? ntohs(bound.sin_port) : port;

    s->running = 1;
    if (pthread_create(&s->thread, NULL, graph_rpc_accept_loop, s) != 0) {
        close(s->listen_fd); gv_free(s); return NULL;
    }
    return s;
}

uint16_t graph_rpc_server_port(const GV_GraphRpcServer *server) {
    return server ? server->port : 0;
}

void graph_rpc_server_stop(GV_GraphRpcServer *server) {
    if (!server) return;
    server->running = 0;
    shutdown(server->listen_fd, SHUT_RDWR);
    close(server->listen_fd);
    pthread_join(server->thread, NULL);
    gv_free(server);
}

/* ── client ───────────────────────────────────────────────────────────── */

int graph_rpc_neighbors(const char *host, uint16_t port, uint64_t node_id,
                        const char *predicate, uint64_t *out, size_t max) {
    if (!host || !out || max == 0) return -1;
    size_t plen = predicate ? strlen(predicate) : 0;
    if (plen > GRAPH_RPC_MAX_PRED) return -1;

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

    uint32_t magic = GRAPH_RPC_REQ_MAGIC;
    uint32_t mx = (uint32_t)(max > GRAPH_RPC_MAX_NEIGH ? GRAPH_RPC_MAX_NEIGH : max);
    uint32_t pl = (uint32_t)plen;
    if (gv_net_write_all(fd, &magic, sizeof(magic)) != 0 ||
        gv_net_write_all(fd, &node_id, sizeof(node_id)) != 0 ||
        gv_net_write_all(fd, &mx, sizeof(mx)) != 0 ||
        gv_net_write_all(fd, &pl, sizeof(pl)) != 0 ||
        (plen > 0 && gv_net_write_all(fd, predicate, plen) != 0)) {
        close(fd);
        return -1;
    }

    uint32_t resp[2];
    if (gv_net_read_all(fd, resp, sizeof(resp)) != 0 || resp[0] != GRAPH_RPC_RESP_MAGIC) {
        close(fd);
        return -1;
    }
    uint32_t count = resp[1];
    if (count > max) count = (uint32_t)max;
    if (count > 0 && gv_net_read_all(fd, out, (size_t)count * sizeof(uint64_t)) != 0) {
        close(fd);
        return -1;
    }
    close(fd);
    return (int)count;
}

/* ── distributed traversal over local + remote partitions ─────────────── */

typedef struct {
    const GV_GraphPartition *parts;
    size_t                   nparts;
    GV_GraphPlacementFn      place;
    void                    *place_ctx;
    const char              *predicate;   /* NULL = any out-edge */
} NetFetchCtx;

static int net_fetch(uint64_t node_id, uint64_t *out, size_t max, void *vctx) {
    NetFetchCtx *c = (NetFetchCtx *)vctx;
    uint32_t p = c->place ? c->place(node_id, c->nparts, c->place_ctx)
                          : (uint32_t)(node_id % c->nparts);
    if (p >= c->nparts) return 0;

    const GV_GraphPartition *part = &c->parts[p];
    if (part->local)
        return graph_get_out_neighbors_typed(part->local, node_id, c->predicate, out, max);
    if (!part->remote_addr) return 0;

    const char *colon = strrchr(part->remote_addr, ':');
    if (!colon || colon == part->remote_addr) return 0;
    size_t hlen = (size_t)(colon - part->remote_addr);
    char host[256];
    if (hlen >= sizeof(host)) return 0;
    memcpy(host, part->remote_addr, hlen);
    host[hlen] = '\0';
    int pt = atoi(colon + 1);
    if (pt <= 0 || pt > 65535) return 0;

    int r = graph_rpc_neighbors(host, (uint16_t)pt, node_id, c->predicate, out, max);
    return r < 0 ? 0 : r; /* unreachable remote -> no neighbours (partial) */
}

int graph_dist_khop_net(const GV_GraphPartition *parts, size_t nparts,
                        GV_GraphPlacementFn place, void *place_ctx,
                        uint64_t start, size_t k, const char *predicate,
                        uint64_t *out_ids, size_t max_out, size_t max_nodes) {
    if (!parts || nparts == 0) return -1;
    NetFetchCtx c = { parts, nparts, place, place_ctx, predicate };
    return graph_dist_khop_fetch(net_fetch, &c, max_nodes ? max_nodes : 16,
                                 start, k, 0, out_ids, max_out);
}

/* ── distributed Cypher pattern match ─────────────────────────────────── */

int cypher_dist_match(const GV_GraphPartition *parts, size_t nparts,
                      GV_GraphPlacementFn place, void *place_ctx,
                      uint64_t start_id, const char *rel_type,
                      size_t min_hops, size_t max_hops,
                      uint64_t *out_ids, size_t max_out, size_t max_nodes) {
    if (!parts || nparts == 0 || !out_ids || max_out == 0) return -1;
    if (max_hops < min_hops) return -1;
    NetFetchCtx c = { parts, nparts, place, place_ctx, rel_type };
    return graph_dist_khop_fetch(net_fetch, &c, max_nodes ? max_nodes : 16,
                                 start_id, max_hops, min_hops, out_ids, max_out);
}

int cypher_dist_path(const GV_GraphPartition *parts, size_t nparts,
                     GV_GraphPlacementFn place, void *place_ctx,
                     uint64_t start_id, const GV_CypherSegment *segments,
                     size_t n_segments, uint64_t *out_ids, size_t max_out,
                     size_t max_nodes) {
    if (!parts || nparts == 0 || !out_ids || max_out == 0 || !segments || n_segments == 0)
        return -1;
    if (max_nodes < 16) max_nodes = 16;

    uint64_t *cur = (uint64_t *)gv_alloc(max_nodes * sizeof(uint64_t));
    uint64_t *nxt = (uint64_t *)gv_alloc(max_nodes * sizeof(uint64_t));
    if (!cur || !nxt) { gv_free(cur); gv_free(nxt); return -1; }

    NetFetchCtx c = { parts, nparts, place, place_ctx, NULL };
    cur[0] = start_id;
    size_t cn = 1;

    /* Expand each segment from the running frontier (multi-source). */
    for (size_t s = 0; s < n_segments; s++) {
        if (segments[s].max_hops < segments[s].min_hops) { gv_free(cur); gv_free(nxt); return -1; }
        c.predicate = segments[s].rel_type;
        int m = graph_dist_khop_multi(net_fetch, &c, max_nodes, cur, cn,
                                      segments[s].max_hops, segments[s].min_hops,
                                      nxt, max_nodes);
        if (m < 0) { gv_free(cur); gv_free(nxt); return -1; }
        uint64_t *tmp = cur; cur = nxt; nxt = tmp;
        cn = (size_t)m;
        if (cn == 0) break;
    }

    size_t kept = cn < max_out ? cn : max_out;
    memcpy(out_ids, cur, kept * sizeof(uint64_t));
    gv_free(cur);
    gv_free(nxt);
    return (int)kept;
}

/* Parse one or more chained `-[:REL*MIN..MAX]->` segments plus `id(a)=N`.
 * ":REL" and the min bound are optional. Fills @p segs (rel_type points into
 * @p reltbl). Returns 0 and sets @p n_segs and @p start, or -1 on failure. */
static int cypher_parse_segments(const char *q, GV_CypherSegment *segs,
                                 char reltbl[][128], size_t max_segs,
                                 size_t *n_segs, uint64_t *start) {
    if (!q) return -1;
    const char *idp = strstr(q, "id(");
    if (!idp) return -1;
    const char *eq = strchr(idp, '=');
    if (!eq) return -1;
    *start = strtoull(eq + 1, NULL, 10);

    size_t n = 0;
    const char *p = q;
    while (n < max_segs) {
        const char *star = strchr(p, '*');
        if (!star) break;

        /* Nearest '[' before the star delimits this relationship segment. */
        const char *lb = star;
        while (lb > q && *lb != '[') lb--;
        if (*lb != '[') { p = star + 1; continue; }

        const char *tp = lb + 1;
        if (*tp == ':') tp++;
        size_t tl = 0;
        char *rel = reltbl[n];
        while (tp < star && *tp && *tp != '*' && *tp != ']' && tl < 127) rel[tl++] = *tp++;
        rel[tl] = '\0';

        const char *rp = star + 1;
        char *end = NULL;
        unsigned long lo = strtoul(rp, &end, 10);
        unsigned long hi;
        if (end && end[0] == '.' && end[1] == '.') {
            hi = strtoul(end + 2, NULL, 10);
        } else {
            hi = lo;
            lo = (lo == 0) ? 1 : lo;
        }
        if (hi == 0) return -1;

        segs[n].rel_type = rel[0] ? rel : NULL;
        segs[n].min_hops = (size_t)lo;
        segs[n].max_hops = (size_t)hi;
        n++;

        const char *rb = strchr(star, ']');
        p = rb ? rb + 1 : star + 1;
    }
    if (n == 0) return -1;
    *n_segs = n;
    return 0;
}

#define CYPHER_MAX_SEGMENTS 16

int cypher_dist_query(const GV_GraphPartition *parts, size_t nparts,
                      GV_GraphPlacementFn place, void *place_ctx,
                      const char *query,
                      uint64_t *out_ids, size_t max_out, size_t max_nodes) {
    GV_CypherSegment segs[CYPHER_MAX_SEGMENTS];
    char reltbl[CYPHER_MAX_SEGMENTS][128];
    size_t n_segs = 0;
    uint64_t start;
    if (cypher_parse_segments(query, segs, reltbl, CYPHER_MAX_SEGMENTS,
                              &n_segs, &start) != 0)
        return -1;
    return cypher_dist_path(parts, nparts, place, place_ctx, start, segs, n_segs,
                            out_ids, max_out, max_nodes);
}

#endif /* _WIN32 */
