/**
 * graph_matrix.c — matrix-native graph traversal (GraphBLAS-lite): variable-
 * length reachability as repeated boolean SpMV, and spreading activation as
 * weighted SpMV, both over the CSR adjacency matrix (graph_csr.h). These are the
 * property-graph equivalents of FalkorDB's "traversal == matrix multiply".
 */
#include "features/graph_algos.h"
#include "features/graph_csr.h"
#include "core/memory.h"

#include <string.h>

/* Allocate node_ids[N] filled from ctx; caller owns. Returns NULL on failure. */
static uint64_t *fill_node_ids(const GV_GAContext *ctx, size_t N) {
    uint64_t *ids = (uint64_t *)gv_alloc(N * sizeof(uint64_t));
    if (!ids) return NULL;
    for (size_t i = 0; i < N; i++) ids[i] = gv_ga_id(ctx, i);
    return ids;
}

int graph_khop_reachable(const GV_GraphDB *g, const uint64_t *sources, size_t num_sources,
                         size_t k, int directed, GV_GraphNodeScores *out) {
    if (!g || !out || (num_sources > 0 && !sources)) return -1;
    memset(out, 0, sizeof(*out));

    GV_GAContext *ctx = gv_ga_build(g);
    if (!ctx) return -1;
    size_t N = gv_ga_count(ctx);
    if (N == 0) { gv_ga_free(ctx); return 0; }

    /* Forward reachability = successors: y = bool(In)·frontier lights up nodes
     * that have an in-neighbor in the frontier. Undirected uses the symmetric
     * matrix directly. */
    GV_CSR *M = gv_csr_build(g, ctx, directed ? GV_CSR_IN : GV_CSR_UNDIRECTED, 0);
    uint8_t *reached  = (uint8_t *)gv_calloc(N, sizeof(uint8_t));
    uint8_t *frontier = (uint8_t *)gv_calloc(N, sizeof(uint8_t));
    uint8_t *next     = (uint8_t *)gv_calloc(N, sizeof(uint8_t));
    uint64_t *ids     = fill_node_ids(ctx, N);
    double  *scores   = (double *)gv_calloc(N, sizeof(double));
    if (!M || !reached || !frontier || !next || !ids || !scores) {
        gv_csr_free(M); gv_free(reached); gv_free(frontier); gv_free(next);
        gv_free(ids); gv_free(scores); gv_ga_free(ctx);
        return -1;
    }

    for (size_t s = 0; s < num_sources; s++) {
        size_t idx = gv_ga_index(ctx, sources[s]);
        if (idx != (size_t)-1) { frontier[idx] = 1; reached[idx] = 1; }
    }

    for (size_t hop = 0; hop < k; hop++) {
        gv_csr_bool_spmv(M, frontier, next);
        int anynew = 0;
        for (size_t i = 0; i < N; i++) {
            if (next[i] && !reached[i]) { reached[i] = 1; next[i] = 1; anynew = 1; }
            else next[i] = 0;   /* only newly-reached nodes form the next frontier */
        }
        uint8_t *tmp = frontier; frontier = next; next = tmp;
        if (!anynew) break;
    }

    for (size_t i = 0; i < N; i++) scores[i] = reached[i] ? 1.0 : 0.0;

    out->node_ids = ids; out->scores = scores; out->count = N;
    gv_csr_free(M); gv_free(reached); gv_free(frontier); gv_free(next);
    gv_ga_free(ctx);
    return 0;
}

int graph_spread_activation(const GV_GraphDB *g, const uint64_t *seeds, size_t num_seeds,
                            size_t iters, double decay, int directed, int weighted,
                            GV_GraphNodeScores *out) {
    if (!g || !out || (num_seeds > 0 && !seeds)) return -1;
    memset(out, 0, sizeof(*out));
    if (iters == 0) iters = 3;
    if (decay <= 0.0) decay = 0.5;

    GV_GAContext *ctx = gv_ga_build(g);
    if (!ctx) return -1;
    size_t N = gv_ga_count(ctx);
    if (N == 0) { gv_ga_free(ctx); return 0; }

    /* Activation flows along edges toward their targets: a_new[j] = decay ·
     * sum over in-neighbors i of w·a[i] = decay·(In·a)[j]. */
    GV_CSR *M    = gv_csr_build(g, ctx, directed ? GV_CSR_IN : GV_CSR_UNDIRECTED, weighted);
    double *a    = (double *)gv_calloc(N, sizeof(double));
    double *tot  = (double *)gv_calloc(N, sizeof(double));
    double *delt = (double *)gv_calloc(N, sizeof(double));
    uint64_t *ids = fill_node_ids(ctx, N);
    if (!M || !a || !tot || !delt || !ids) {
        gv_csr_free(M); gv_free(a); gv_free(tot); gv_free(delt); gv_free(ids);
        gv_ga_free(ctx);
        return -1;
    }

    for (size_t s = 0; s < num_seeds; s++) {
        size_t idx = gv_ga_index(ctx, seeds[s]);
        if (idx != (size_t)-1) { a[idx] += 1.0; tot[idx] += 1.0; }
    }

    for (size_t it = 0; it < iters; it++) {
        gv_csr_spmv(M, a, delt);
        for (size_t i = 0; i < N; i++) delt[i] *= decay;
        for (size_t i = 0; i < N; i++) tot[i] += delt[i];
        double *tmp = a; a = delt; delt = tmp;   /* a <- new activation */
    }

    out->node_ids = ids; out->scores = tot; out->count = N;
    gv_csr_free(M); gv_free(a); gv_free(delt); gv_ga_free(ctx);
    return 0;
}
