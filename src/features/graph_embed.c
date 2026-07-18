/**
 * graph_embed.c — graph node-embedding algorithms for GigaVector.
 *
 * FastRP structural embeddings (sparse random projection + degree-normalized
 * neighborhood propagation) and node2vec-style biased 2nd-order random walks.
 * Both treat the graph as UNDIRECTED: a node's neighbors are the dedup union of
 * its out-edge and in-edge endpoints (self-loops excluded). Read-only on the
 * graph and fully deterministic given `seed`.
 */
#include "features/graph_algos.h"
#include "features/graph_csr.h"
#include "core/memory.h"

#include <math.h>
#include <string.h>
#include <stdint.h>

/* Deterministic xorshift64 PRNG. */
static inline uint64_t xs64(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    return x;
}

/* Uniform double in [0,1) from the top 53 bits of an xorshift64 draw. */
static inline double xs64_unit(uint64_t *s) {
    return (double)(xs64(s) >> 11) * (1.0 / 9007199254740992.0);
}

/* Seed mixing (splitmix64 finalizer) — avoids the all-zero xorshift fixed point
 * and decorrelates seeds derived from small integers. */
static inline uint64_t mix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return z ? z : 0x9E3779B97F4A7C15ULL; /* never hand xorshift a zero state */
}

/* ── Undirected adjacency (dense indices) ───────────────────────────────────
 * Build, once, per-node dedup neighbor lists expressed as dense [0,N) indices.
 * Backed by one flat buffer + per-node offsets (CSR-style) so there is a single
 * allocation to free. Neighbors outside the context (should not happen) and
 * self-loops are dropped. Returns 0 on success, -1 on alloc failure. */
typedef struct {
    size_t *off;   /* N+1 offsets into `adj` */
    size_t *adj;   /* concatenated neighbor indices */
    size_t  total; /* number of entries in `adj` */
} UndirAdj;

static void undir_free(UndirAdj *ua) {
    if (!ua) return;
    gv_free(ua->off);
    gv_free(ua->adj);
    ua->off = NULL;
    ua->adj = NULL;
    ua->total = 0;
}

static int undir_build(const GV_GraphDB *g, const GV_GAContext *ctx, UndirAdj *ua) {
    size_t N = gv_ga_count(ctx);
    ua->off = NULL;
    ua->adj = NULL;
    ua->total = 0;

    ua->off = (size_t *)gv_calloc(N + 1, sizeof(size_t));
    if (!ua->off) return -1;
    if (N == 0) return 0;

    /* Scratch mark buffer for dedup within a single node's neighbor union.
     * mark[k] holds the 1-based "generation" of the node last seen so we can
     * clear it in O(1) by bumping the generation instead of re-zeroing. */
    size_t *mark = (size_t *)gv_calloc(N, sizeof(size_t));
    if (!mark) { undir_free(ua); return -1; }

    /* Pass 1: count deduped neighbors per node. */
    size_t total = 0;
    for (size_t i = 0; i < N; i++) {
        uint64_t id = gv_ga_id(ctx, i);
        const GV_GraphNode *node = graph_get_node(g, id);
        size_t cnt = 0;
        if (node) {
            for (size_t e = 0; e < node->out_count; e++) {
                size_t ni = gv_ga_index(ctx, node->out_edges[e].neighbor_id);
                if (ni == (size_t)-1 || ni == i) continue;
                if (mark[ni] != i + 1) { mark[ni] = i + 1; cnt++; }
            }
            for (size_t e = 0; e < node->in_count; e++) {
                size_t ni = gv_ga_index(ctx, node->in_edges[e].neighbor_id);
                if (ni == (size_t)-1 || ni == i) continue;
                if (mark[ni] != i + 1) { mark[ni] = i + 1; cnt++; }
            }
        }
        ua->off[i + 1] = cnt;
        total += cnt;
    }
    /* Prefix-sum offsets. */
    for (size_t i = 0; i < N; i++) ua->off[i + 1] += ua->off[i];
    ua->total = total;

    if (total > 0) {
        ua->adj = (size_t *)gv_alloc(total * sizeof(size_t));
        if (!ua->adj) { gv_free(mark); undir_free(ua); return -1; }
    }

    /* Reset marks: generations used above were i+1 in [1,N]; use N+1+i now so
     * they never collide with the values written in pass 1. */
    for (size_t i = 0; i < N; i++) mark[i] = 0;

    /* Pass 2: fill neighbor indices (same dedup, same order). */
    for (size_t i = 0; i < N; i++) {
        uint64_t id = gv_ga_id(ctx, i);
        const GV_GraphNode *node = graph_get_node(g, id);
        size_t w = ua->off[i];
        if (node) {
            for (size_t e = 0; e < node->out_count; e++) {
                size_t ni = gv_ga_index(ctx, node->out_edges[e].neighbor_id);
                if (ni == (size_t)-1 || ni == i) continue;
                if (mark[ni] != i + 1) { mark[ni] = i + 1; ua->adj[w++] = ni; }
            }
            for (size_t e = 0; e < node->in_count; e++) {
                size_t ni = gv_ga_index(ctx, node->in_edges[e].neighbor_id);
                if (ni == (size_t)-1 || ni == i) continue;
                if (mark[ni] != i + 1) { mark[ni] = i + 1; ua->adj[w++] = ni; }
            }
        }
    }

    gv_free(mark);
    return 0;
}

/* L2-normalize a row of `dim` floats in place (no-op on zero/degenerate norm). */
static void l2_normalize_row(float *row, size_t dim) {
    double ss = 0.0;
    for (size_t d = 0; d < dim; d++) ss += (double)row[d] * (double)row[d];
    if (ss <= 0.0) return;
    double inv = 1.0 / sqrt(ss);
    for (size_t d = 0; d < dim; d++) row[d] = (float)((double)row[d] * inv);
}

/* ── FastRP ─────────────────────────────────────────────────────────────────*/

int graph_fastrp(const GV_GraphDB *g, size_t dim, size_t iters,
                 const double *weights, uint64_t seed, GV_GraphEmbeddings *out) {
    if (!g || !out) return -1;

    memset(out, 0, sizeof(*out));
    if (dim == 0) dim = 128;
    if (iters == 0) iters = 3;

    GV_GAContext *ctx = gv_ga_build(g);
    if (!ctx) return -1;
    size_t N = gv_ga_count(ctx);

    /* Empty graph: valid empty embeddings. */
    if (N == 0) {
        out->node_ids = NULL;
        out->vectors = NULL;
        out->count = 0;
        out->dim = dim;
        gv_ga_free(ctx);
        return 0;
    }

    /* Undirected weighted CSR: symmetric adjacency A used for propagation. */
    GV_CSR *A = gv_csr_build(g, ctx, GV_CSR_UNDIRECTED, 1);
    if (!A) { gv_ga_free(ctx); return -1; }

    /* Working matrices: cur = embed at hop t-1, nxt = embed at hop t (both dense
     * n×dim, double for the SpMM), result = accumulated weighted sum,
     * plus a normalized snapshot buffer. deg = weighted degree per node. */
    float  *result = (float *)gv_calloc(N * dim, sizeof(float));
    double *cur    = (double *)gv_alloc(N * dim * sizeof(double));
    double *nxt    = (double *)gv_alloc(N * dim * sizeof(double));
    double *scaled = (double *)gv_alloc(N * dim * sizeof(double));
    float  *norm   = (float *)gv_alloc(N * dim * sizeof(float));
    double *invsd  = (double *)gv_alloc(N * sizeof(double)); /* 1/sqrt(deg) per row */
    uint64_t *ids  = (uint64_t *)gv_alloc(N * sizeof(uint64_t));
    if (!result || !cur || !nxt || !scaled || !norm || !invsd || !ids) {
        gv_free(result); gv_free(cur); gv_free(nxt); gv_free(scaled);
        gv_free(norm); gv_free(invsd); gv_free(ids);
        gv_csr_free(A); gv_ga_free(ctx);
        return -1;
    }

    /* Weighted degree per node; invsd[i] = 1/sqrt(deg(i)), 0 for isolated nodes.
     * Folding this scaling into cur before the SpMM and into nxt after it yields
     * the symmetric normalization D^{-1/2} A D^{-1/2}. */
    gv_csr_row_sums(A, invsd);
    for (size_t i = 0; i < N; i++) {
        double deg = invsd[i];
        invsd[i] = (deg > 0.0) ? 1.0 / sqrt(deg) : 0.0;
    }

    /* Base embedding: very sparse random projection (s = 3).
     * Each entry is +sqrt(s) w.p. 1/(2s), -sqrt(s) w.p. 1/(2s), else 0. */
    const double s = 3.0;
    const double scale = sqrt(s);       /* magnitude of nonzero entries */
    const double p_pos = 1.0 / (2.0 * s);
    const double p_neg = 2.0 / (2.0 * s); /* cumulative threshold for negative */
    for (size_t i = 0; i < N; i++) {
        ids[i] = gv_ga_id(ctx, i);
        /* Per-node deterministic stream mixed from seed and the (stable) id. */
        uint64_t st = mix64(seed ^ (mix64(ids[i]) * 0x100000001B3ULL));
        double *row = &cur[i * dim];
        for (size_t d = 0; d < dim; d++) {
            double u = xs64_unit(&st);
            if (u < p_pos)      row[d] = scale;
            else if (u < p_neg) row[d] = -scale;
            else                row[d] = 0.0;
        }
    }

    for (size_t t = 0; t < iters; t++) {
        /* Symmetric normalization D^{-1/2} A D^{-1/2} · cur, folded as:
         *   scaled = D^{-1/2} · cur   (pre-scale rows)
         *   nxt    = A · scaled       (SpMM)
         *   nxt    = D^{-1/2} · nxt   (post-scale rows)
         * Isolated nodes (deg 0 => invsd 0) get a zero row. */
        for (size_t i = 0; i < N; i++) {
            double si = invsd[i];
            const double *crow = &cur[i * dim];
            double *srow = &scaled[i * dim];
            for (size_t d = 0; d < dim; d++) srow[d] = crow[d] * si;
        }
        gv_csr_spmm_dense(A, scaled, dim, nxt);
        for (size_t i = 0; i < N; i++) {
            double si = invsd[i];
            double *nrow = &nxt[i * dim];
            for (size_t d = 0; d < dim; d++) nrow[d] *= si;
        }

        /* L2-normalize this hop into `norm`, accumulate weighted into result. */
        double w = 1.0; /* default weight: uniform (mild, no decay). */
        if (weights) w = weights[t];
        for (size_t i = 0; i < N; i++) {
            double *nrow = &nxt[i * dim];
            float  *snap = &norm[i * dim];
            for (size_t d = 0; d < dim; d++) snap[d] = (float)nrow[d];
            l2_normalize_row(snap, dim);
            float *rrow = &result[i * dim];
            for (size_t d = 0; d < dim; d++)
                rrow[d] += (float)(w * (double)snap[d]);
        }

        /* Swap cur <- nxt for the next hop (propagate the un-normalized embed). */
        double *tmp = cur; cur = nxt; nxt = tmp;
    }

    /* Final L2-normalize each result row (guard zero norm). */
    for (size_t i = 0; i < N; i++)
        l2_normalize_row(&result[i * dim], dim);

    gv_free(cur);
    gv_free(nxt);
    gv_free(scaled);
    gv_free(norm);
    gv_free(invsd);
    gv_csr_free(A);
    gv_ga_free(ctx);

    out->node_ids = ids;
    out->vectors = result;
    out->count = N;
    out->dim = dim;
    return 0;
}

/* ── node2vec biased random walks ───────────────────────────────────────────*/

/* Is dense index `x` a neighbor of node `t`? Linear scan of t's neighbor list.
 * Adequate: node2vec transition scan already visits every neighbor of v, and
 * degrees in property graphs are typically small. */
static int is_neighbor(const UndirAdj *ua, size_t t, size_t x) {
    size_t begin = ua->off[t], end = ua->off[t + 1];
    for (size_t k = begin; k < end; k++)
        if (ua->adj[k] == x) return 1;
    return 0;
}

int graph_node2vec_walks(const GV_GraphDB *g, size_t num_walks, size_t walk_len,
                         double p, double q, uint64_t seed,
                         uint64_t **out_walks, size_t *out_num_walks, size_t *out_walk_len) {
    if (!g || !out_walks || !out_num_walks || !out_walk_len) return -1;
    if (num_walks == 0 || walk_len == 0) return -1;
    if (!(p > 0.0) || !(q > 0.0)) return -1; /* transition weights need finite 1/p,1/q */

    *out_walks = NULL;
    *out_num_walks = 0;
    *out_walk_len = 0;

    GV_GAContext *ctx = gv_ga_build(g);
    if (!ctx) return -1;
    size_t N = gv_ga_count(ctx);

    if (N == 0) {
        /* No nodes: empty walk set. */
        *out_walks = NULL;
        *out_num_walks = 0;
        *out_walk_len = walk_len;
        gv_ga_free(ctx);
        return 0;
    }

    UndirAdj ua;
    if (undir_build(g, ctx, &ua) != 0) { gv_ga_free(ctx); return -1; }

    /* Overflow-guarded row count and buffer size. */
    if (num_walks > SIZE_MAX / N) { undir_free(&ua); gv_ga_free(ctx); return -1; }
    size_t rows = N * num_walks;
    if (walk_len > SIZE_MAX / rows) { undir_free(&ua); gv_ga_free(ctx); return -1; }
    size_t cells = rows * walk_len;

    uint64_t *walks = (uint64_t *)gv_calloc(cells, sizeof(uint64_t));
    if (!walks) { undir_free(&ua); gv_ga_free(ctx); return -1; }

    /* Scratch weights buffer for weighted neighbor sampling (sized to max deg). */
    size_t max_deg = 0;
    for (size_t i = 0; i < N; i++) {
        size_t deg = ua.off[i + 1] - ua.off[i];
        if (deg > max_deg) max_deg = deg;
    }
    double *w = NULL;
    if (max_deg > 0) {
        w = (double *)gv_alloc(max_deg * sizeof(double));
        if (!w) { gv_free(walks); undir_free(&ua); gv_ga_free(ctx); return -1; }
    }

    const double inv_p = 1.0 / p;
    const double inv_q = 1.0 / q;

    for (size_t start = 0; start < N; start++) {
        for (size_t wk = 0; wk < num_walks; wk++) {
            uint64_t *row = &walks[(start * num_walks + wk) * walk_len];
            /* Walk already zero-filled (calloc); set the start node. */
            row[0] = gv_ga_id(ctx, start);

            size_t cur = start;
            size_t prev = (size_t)-1; /* no predecessor for the first step */

            for (size_t step = 1; step < walk_len; step++) {
                size_t begin = ua.off[cur], end = ua.off[cur + 1];
                size_t deg = end - begin;
                if (deg == 0) break; /* dead end: leave remaining slots as 0 */

                /* Deterministic per-(start,walk,step) PRNG stream. */
                uint64_t st = mix64(seed
                                    ^ (mix64(gv_ga_id(ctx, start)) * 0x100000001B3ULL)
                                    ^ (mix64(wk + 1) * 0x27D4EB2F165667C5ULL)
                                    ^ (mix64(step + 1) * 0x9E3779B97F4A7C15ULL));

                size_t chosen;
                if (prev == (size_t)-1) {
                    /* First step: uniform over neighbors. */
                    size_t r = (size_t)(xs64(&st) % (uint64_t)deg);
                    chosen = ua.adj[begin + r];
                } else {
                    /* Biased: weight each candidate x by
                     *   1/p  if x == prev            (return),
                     *   1    if x is a neighbor of prev (BFS dist 1),
                     *   1/q  otherwise                (BFS dist 2). */
                    double sum = 0.0;
                    for (size_t k = 0; k < deg; k++) {
                        size_t x = ua.adj[begin + k];
                        double wt;
                        if (x == prev)                    wt = inv_p;
                        else if (is_neighbor(&ua, prev, x)) wt = 1.0;
                        else                              wt = inv_q;
                        sum += wt;
                        w[k] = sum; /* store running CDF */
                    }
                    /* Sample by inverse-CDF; sum > 0 since every weight > 0. */
                    double target = xs64_unit(&st) * sum;
                    size_t k = 0;
                    while (k + 1 < deg && target >= w[k]) k++;
                    chosen = ua.adj[begin + k];
                }

                row[step] = gv_ga_id(ctx, chosen);
                prev = cur;
                cur = chosen;
            }
        }
    }

    gv_free(w);
    undir_free(&ua);
    gv_ga_free(ctx);

    *out_walks = walks;
    *out_num_walks = rows;
    *out_walk_len = walk_len;
    return 0;
}
