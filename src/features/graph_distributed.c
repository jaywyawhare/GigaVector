/**
 * @file graph_distributed.c
 * @brief Cross-partition graph traversal (see graph_distributed.h).
 */

#include "features/graph_distributed.h"
#include "core/memory.h"

#include <string.h>

/* Open-addressing set of node ids (ids are > 0, so 0 marks an empty slot). */
typedef struct {
    uint64_t *slots;
    size_t    cap;   /* power of two */
} IdSet;

static size_t next_pow2(size_t n) {
    size_t p = 16;
    while (p < n) p <<= 1;
    return p;
}

static int idset_init(IdSet *s, size_t hint) {
    s->cap = next_pow2(hint * 2 + 16);
    s->slots = (uint64_t *)gv_calloc(s->cap, sizeof(uint64_t));
    return s->slots ? 0 : -1;
}

static void idset_free(IdSet *s) { gv_free(s->slots); s->slots = NULL; }

/** @return 1 if @p id was newly inserted, 0 if already present. */
static int idset_add(IdSet *s, uint64_t id) {
    uint64_t h = (id * 1099511628211ULL) & (s->cap - 1);
    while (s->slots[h] != 0) {
        if (s->slots[h] == id) return 0;
        h = (h + 1) & (s->cap - 1);
    }
    s->slots[h] = id;
    return 1;
}

static uint32_t default_placement(uint64_t node_id, size_t nparts, void *ctx) {
    (void)ctx;
    return (uint32_t)(node_id % nparts);
}

int graph_dist_khop_multi(GV_GraphNeighborFn fetch, void *fetch_ctx,
                          size_t max_nodes, const uint64_t *starts, size_t n_starts,
                          size_t k, size_t min_hops,
                          uint64_t *out_ids, size_t max_out) {
    if (!fetch || !out_ids || max_out == 0 || !starts || n_starts == 0) return -1;
    if (max_nodes < 16) max_nodes = 16;
    if (max_nodes < n_starts) max_nodes = n_starts; /* frontier must hold all seeds */

    IdSet visited;
    if (idset_init(&visited, max_nodes) != 0) return -1;

    uint64_t *frontier = (uint64_t *)gv_alloc(max_nodes * sizeof(uint64_t));
    uint64_t *next = (uint64_t *)gv_alloc(max_nodes * sizeof(uint64_t));
    uint64_t *neigh = (uint64_t *)gv_alloc(max_nodes * sizeof(uint64_t));
    if (!frontier || !next || !neigh) {
        gv_free(frontier); gv_free(next); gv_free(neigh);
        idset_free(&visited);
        return -1;
    }

    size_t out_n = 0;
    size_t fn = 0;
    for (size_t i = 0; i < n_starts; i++) {
        if (!idset_add(&visited, starts[i])) continue; /* dedup seeds */
        if (min_hops == 0 && out_n < max_out) out_ids[out_n++] = starts[i];
        if (fn < max_nodes) frontier[fn++] = starts[i];
    }

    /* A node is emitted at its first (shortest) discovery hop; only nodes whose
     * shortest hop distance falls in [min_hops, k] are returned. */
    for (size_t hop = 0; hop < k && fn > 0; hop++) {
        size_t nn = 0;
        for (size_t i = 0; i < fn; i++) {
            int got = fetch(frontier[i], neigh, max_nodes, fetch_ctx);
            for (int j = 0; got > 0 && j < got; j++) {
                uint64_t m = neigh[j];
                if (!idset_add(&visited, m)) continue; /* already seen */
                if (hop + 1 >= min_hops && out_n < max_out) out_ids[out_n++] = m;
                if (nn < max_nodes) next[nn++] = m;
            }
        }
        memcpy(frontier, next, nn * sizeof(uint64_t));
        fn = nn;
    }

    gv_free(frontier);
    gv_free(next);
    gv_free(neigh);
    idset_free(&visited);
    return (int)out_n;
}

int graph_dist_khop_fetch(GV_GraphNeighborFn fetch, void *fetch_ctx,
                          size_t max_nodes, uint64_t start, size_t k,
                          size_t min_hops, uint64_t *out_ids, size_t max_out) {
    return graph_dist_khop_multi(fetch, fetch_ctx, max_nodes, &start, 1,
                                 k, min_hops, out_ids, max_out);
}

/* Neighbour fetch that routes to a local partition array by placement. */
typedef struct {
    GV_GraphDB *const  *parts;
    size_t              nparts;
    GV_GraphPlacementFn place;
    void               *place_ctx;
    const char         *predicate;   /* NULL = any out-edge */
} LocalFetchCtx;

static int local_fetch(uint64_t node_id, uint64_t *out, size_t max, void *ctx) {
    LocalFetchCtx *c = (LocalFetchCtx *)ctx;
    uint32_t p = c->place ? c->place(node_id, c->nparts, c->place_ctx)
                          : default_placement(node_id, c->nparts, NULL);
    if (p >= c->nparts || !c->parts[p]) return 0;
    return graph_get_out_neighbors_typed(c->parts[p], node_id, c->predicate, out, max);
}

int graph_dist_khop(GV_GraphDB *const *parts, size_t nparts,
                    GV_GraphPlacementFn place, void *place_ctx,
                    uint64_t start, size_t k, const char *predicate,
                    uint64_t *out_ids, size_t max_out) {
    if (!parts || nparts == 0) return -1;

    size_t total = 0;
    for (size_t i = 0; i < nparts; i++) {
        if (parts[i]) total += graph_node_count(parts[i]);
    }

    LocalFetchCtx c = { parts, nparts, place, place_ctx, predicate };
    return graph_dist_khop_fetch(local_fetch, &c, total ? total : 16,
                                 start, k, 0, out_ids, max_out);
}
