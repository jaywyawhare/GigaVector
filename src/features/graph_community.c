/**
 * graph_community.c — community detection / clustering algorithms for the
 * graph-algorithms layer (graph_algos.h):
 *
 *   - graph_label_propagation      Label Propagation (LPA), undirected
 *   - graph_louvain                Louvain modularity maximization, undirected/weighted
 *   - graph_triangle_count         per-node & total undirected triangle counts
 *   - graph_kcore                  k-core decomposition (core numbers), undirected
 *   - graph_strongly_connected_components  Tarjan SCC (directed, iterative)
 *
 * Conventions (see graph_algos.h / graph_algos_util.c):
 *   - Work over a dense node index [0, N) via GV_GAContext.
 *   - Undirected neighbors = deduplicated union of a node's out_edges and
 *     in_edges neighbor_ids, mapped to dense indices (skip absent / self-loops).
 *   - Results memset to 0, node_ids[i]=gv_ga_id(ctx,i), count=N, num_labels=K.
 *   - N==0 -> zeroed struct + return 0. Alloc failure -> free all + return -1.
 *   - Deterministic (dense-index order, no rand/time). All loops are capped.
 *   - Read-only w.r.t. the graph; no leaks on any path.
 */
#include "features/graph_algos.h"
#include "core/memory.h"

#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

/* ── Undirected CSR adjacency over the dense index ──────────────────────────
 * Deduplicated undirected neighbor lists in compressed-sparse-row form:
 *   neighbors of dense node u are adj[off[u] .. off[u+1]).
 * A parallel `wt` array holds the (summed) undirected edge weight to each
 * neighbor (missing edge -> 1.0). Self-loops and absent endpoints are dropped;
 * a u<->v pair appears once in u's list and once in v's list. */
typedef struct {
    size_t   *off;    /* N+1 offsets */
    size_t   *adj;    /* M dense neighbor indices */
    double   *wt;     /* M edge weights parallel to adj */
    size_t    N;
    size_t    M;      /* total directed adjacency entries = 2 * undirected edges */
} GV_UAdj;

static void uadj_free(GV_UAdj *a) {
    if (!a) return;
    gv_free(a->off);
    gv_free(a->adj);
    gv_free(a->wt);
    a->off = NULL; a->adj = NULL; a->wt = NULL; a->N = 0; a->M = 0;
}

/* Weight of an edge ref (missing edge -> 1.0). */
static double edge_weight(const GV_GraphDB *g, uint64_t edge_id) {
    const GV_GraphEdge *e = graph_get_edge(g, edge_id);
    return e ? (double)e->weight : 1.0;
}

/**
 * Build the deduplicated, weighted undirected CSR adjacency.
 * Returns 0 on success (a is filled, caller frees with uadj_free), -1 on error
 * (a is left zeroed / freed).
 *
 * Duplicate parallel edges between the same pair are collapsed and their
 * weights summed, so the CSR neighbor list of each node is unique.
 */
static int uadj_build(const GV_GraphDB *g, GV_GAContext *ctx, GV_UAdj *a) {
    memset(a, 0, sizeof(*a));
    size_t N = gv_ga_count(ctx);
    a->N = N;
    if (N == 0) return 0;

    a->off = (size_t *)gv_calloc(N + 1, sizeof(size_t));
    if (!a->off) { uadj_free(a); return -1; }

    /* Per-node scratch: dense neighbor index + accumulated weight, deduped via
     * a "seen at pass" stamp array (avoids clearing an N-sized map per node). */
    size_t *tmp_idx = (size_t *)gv_alloc(N * sizeof(size_t));
    double *tmp_w   = (double *)gv_alloc(N * sizeof(double));
    size_t *slot    = (size_t *)gv_alloc(N * sizeof(size_t)); /* neighbor -> tmp position */
    size_t *stamp   = (size_t *)gv_calloc(N, sizeof(size_t)); /* last pass touching a neighbor */
    if (!tmp_idx || !tmp_w || !slot || !stamp) {
        gv_free(tmp_idx); gv_free(tmp_w); gv_free(slot); gv_free(stamp);
        uadj_free(a);
        return -1;
    }

    /* Pass 1: count deduplicated undirected neighbors per node -> off[u+1]. */
    for (size_t u = 0; u < N; u++) {
        uint64_t uid = gv_ga_id(ctx, u);
        const GV_GraphNode *node = graph_get_node(g, uid);
        size_t cnt = 0;
        size_t pass = u + 1; /* nonzero, unique per node */
        if (node) {
            for (size_t e = 0; e < node->out_count; e++) {
                size_t v = gv_ga_index(ctx, node->out_edges[e].neighbor_id);
                if (v == (size_t)-1 || v == u) continue;
                if (stamp[v] != pass) { stamp[v] = pass; cnt++; }
            }
            for (size_t e = 0; e < node->in_count; e++) {
                size_t v = gv_ga_index(ctx, node->in_edges[e].neighbor_id);
                if (v == (size_t)-1 || v == u) continue;
                if (stamp[v] != pass) { stamp[v] = pass; cnt++; }
            }
        }
        a->off[u + 1] = cnt;
    }
    for (size_t u = 0; u < N; u++) a->off[u + 1] += a->off[u];
    a->M = a->off[N];

    a->adj = (size_t *)gv_alloc((a->M ? a->M : 1) * sizeof(size_t));
    a->wt  = (double *)gv_alloc((a->M ? a->M : 1) * sizeof(double));
    if (!a->adj || !a->wt) {
        gv_free(tmp_idx); gv_free(tmp_w); gv_free(slot); gv_free(stamp);
        uadj_free(a);
        return -1;
    }

    /* Reset stamps for pass 2 (reuse as "seen this node"). */
    memset(stamp, 0, N * sizeof(size_t));

    /* Pass 2: fill neighbor indices + summed weights. */
    for (size_t u = 0; u < N; u++) {
        uint64_t uid = gv_ga_id(ctx, u);
        const GV_GraphNode *node = graph_get_node(g, uid);
        size_t local = 0;
        size_t pass = u + 1;
        if (node) {
            for (size_t e = 0; e < node->out_count; e++) {
                size_t v = gv_ga_index(ctx, node->out_edges[e].neighbor_id);
                if (v == (size_t)-1 || v == u) continue;
                double w = edge_weight(g, node->out_edges[e].edge_id);
                if (stamp[v] == pass) {
                    tmp_w[slot[v]] += w;
                } else {
                    stamp[v] = pass;
                    slot[v] = local;
                    tmp_idx[local] = v;
                    tmp_w[local] = w;
                    local++;
                }
            }
            for (size_t e = 0; e < node->in_count; e++) {
                size_t v = gv_ga_index(ctx, node->in_edges[e].neighbor_id);
                if (v == (size_t)-1 || v == u) continue;
                double w = edge_weight(g, node->in_edges[e].edge_id);
                if (stamp[v] == pass) {
                    tmp_w[slot[v]] += w;
                } else {
                    stamp[v] = pass;
                    slot[v] = local;
                    tmp_idx[local] = v;
                    tmp_w[local] = w;
                    local++;
                }
            }
        }
        size_t base = a->off[u];
        for (size_t k = 0; k < local; k++) {
            a->adj[base + k] = tmp_idx[k];
            a->wt[base + k]  = tmp_w[k];
        }
    }

    gv_free(tmp_idx); gv_free(tmp_w); gv_free(slot); gv_free(stamp);
    return 0;
}

/* Compact arbitrary raw labels[0..N) into dense ids 0..K-1 in first-appearance
 * order (dense-index scan -> deterministic). Returns K. Uses an open-addressing
 * map allocated by the caller (map_key/map_val, cap must be > N, occ zeroed). */
static size_t compact_labels(int64_t *labels, size_t N,
                             int64_t *map_key, size_t *map_val, uint8_t *occ,
                             size_t cap) {
    size_t K = 0;
    for (size_t i = 0; i < N; i++) {
        int64_t raw = labels[i];
        uint64_t h = (uint64_t)raw * 11400714819323198485ULL;
        size_t start = (size_t)(h % cap);
        size_t found = (size_t)-1;
        for (size_t j = 0; j < cap; j++) {
            size_t p = (start + j) % cap;
            if (!occ[p]) {
                occ[p] = 1;
                map_key[p] = raw;
                map_val[p] = K;
                found = K;
                K++;
                break;
            }
            if (map_key[p] == raw) { found = map_val[p]; break; }
        }
        labels[i] = (int64_t)found;
    }
    return K;
}

/* ── Label Propagation (LPA), undirected ────────────────────────────────────
 * Each node adopts the most frequent label among its undirected neighbors;
 * ties broken toward the smallest label for determinism. Iterate (updating in
 * dense-index order, reading current labels) until stable or max_iters. Labels
 * are then compacted to 0..K-1. */
int graph_label_propagation(const GV_GraphDB *g, size_t max_iters,
                            GV_GraphNodeLabels *out) {
    if (!g || !out) return -1;
    memset(out, 0, sizeof(*out));
    if (max_iters == 0) max_iters = 100;

    GV_GAContext *ctx = gv_ga_build(g);
    if (!ctx) return -1;
    size_t N = gv_ga_count(ctx);
    if (N == 0) { gv_ga_free(ctx); return 0; }

    GV_UAdj adj;
    if (uadj_build(g, ctx, &adj) != 0) { gv_ga_free(ctx); return -1; }

    uint64_t *node_ids = (uint64_t *)gv_alloc(N * sizeof(uint64_t));
    int64_t  *labels   = (int64_t *)gv_alloc(N * sizeof(int64_t));
    /* Voting scratch: for each candidate label observed among a node's
     * neighbors, count votes. Deduped via a stamp array keyed by label value
     * (labels are dense-node indices, so within [0, N)). */
    int64_t  *cand     = (int64_t *)gv_alloc((N ? N : 1) * sizeof(int64_t));
    double   *votes    = (double *)gv_alloc((N ? N : 1) * sizeof(double));
    size_t   *vpos     = (size_t *)gv_alloc(N * sizeof(size_t));
    size_t   *vstamp   = (size_t *)gv_calloc(N, sizeof(size_t));
    if (!node_ids || !labels || !cand || !votes || !vpos || !vstamp) {
        gv_free(node_ids); gv_free(labels); gv_free(cand);
        gv_free(votes); gv_free(vpos); gv_free(vstamp);
        uadj_free(&adj); gv_ga_free(ctx);
        return -1;
    }

    for (size_t i = 0; i < N; i++) {
        node_ids[i] = gv_ga_id(ctx, i);
        labels[i]   = (int64_t)i; /* seed: unique label per node */
    }

    for (size_t iter = 0; iter < max_iters; iter++) {
        int changed = 0;
        for (size_t u = 0; u < N; u++) {
            size_t deg = adj.off[u + 1] - adj.off[u];
            if (deg == 0) continue;

            size_t nc = 0;      /* number of distinct candidate labels */
            size_t pass = u + 1;
            for (size_t k = adj.off[u]; k < adj.off[u + 1]; k++) {
                size_t v = adj.adj[k];
                int64_t lv = labels[v];
                /* key candidate list by label value using vstamp/vpos indexed
                 * by the (dense) label; labels are in [0, N). */
                size_t key = (size_t)lv;
                if (vstamp[key] == pass) {
                    votes[vpos[key]] += 1.0;
                } else {
                    vstamp[key] = pass;
                    vpos[key] = nc;
                    cand[nc] = lv;
                    votes[nc] = 1.0;
                    nc++;
                }
            }

            /* Pick max votes; tie -> smallest label. */
            int64_t best = labels[u];
            double  best_v = -1.0;
            for (size_t c = 0; c < nc; c++) {
                if (votes[c] > best_v ||
                    (votes[c] == best_v && cand[c] < best)) {
                    best_v = votes[c];
                    best = cand[c];
                }
            }
            if (best != labels[u]) { labels[u] = best; changed = 1; }
        }
        if (!changed) break;
    }

    gv_free(cand); gv_free(votes); gv_free(vpos); gv_free(vstamp);
    uadj_free(&adj);

    /* Compact labels to 0..K-1. */
    size_t cap = N * 2 + 1;
    int64_t *map_key = (int64_t *)gv_alloc(cap * sizeof(int64_t));
    size_t  *map_val = (size_t *)gv_alloc(cap * sizeof(size_t));
    uint8_t *occ     = (uint8_t *)gv_calloc(cap, sizeof(uint8_t));
    if (!map_key || !map_val || !occ) {
        gv_free(map_key); gv_free(map_val); gv_free(occ);
        gv_free(node_ids); gv_free(labels);
        gv_ga_free(ctx);
        return -1;
    }
    size_t K = compact_labels(labels, N, map_key, map_val, occ, cap);
    gv_free(map_key); gv_free(map_val); gv_free(occ);
    gv_ga_free(ctx);

    out->node_ids   = node_ids;
    out->labels     = labels;
    out->count      = N;
    out->num_labels = K;
    return 0;
}

/* ── Louvain (undirected, weighted) ─────────────────────────────────────────
 * First-level local-moving modularity optimization to convergence. Each node
 * starts in its own community; repeatedly move each node (dense-index order) to
 * the neighboring community giving the largest positive modularity gain, using
 * the standard gain
 *     dQ = k_i_in - (Sigma_tot * k_i) / (2m)
 * (constant terms dropped) where k_i_in is the summed weight from node i to the
 * community, Sigma_tot is the total incident weight of the community, and k_i is
 * node i's weighted degree. Multi-level aggregation is optional and omitted;
 * labels are compacted to 0..K-1. */
int graph_louvain(const GV_GraphDB *g, size_t max_passes, GV_GraphNodeLabels *out) {
    if (!g || !out) return -1;
    memset(out, 0, sizeof(*out));
    if (max_passes == 0) max_passes = 10;

    GV_GAContext *ctx = gv_ga_build(g);
    if (!ctx) return -1;
    size_t N = gv_ga_count(ctx);
    if (N == 0) { gv_ga_free(ctx); return 0; }

    GV_UAdj adj;
    if (uadj_build(g, ctx, &adj) != 0) { gv_ga_free(ctx); return -1; }

    uint64_t *node_ids = (uint64_t *)gv_alloc(N * sizeof(uint64_t));
    int64_t  *labels   = (int64_t *)gv_alloc(N * sizeof(int64_t)); /* community of node */
    double   *k        = (double *)gv_alloc(N * sizeof(double));   /* weighted degree */
    double   *sigma    = (double *)gv_alloc(N * sizeof(double));   /* Sigma_tot per community */
    /* Per-node scratch: accumulate k_i_in per neighboring community, deduped by
     * a stamp keyed by community id (community ids live in [0, N)). */
    int64_t  *comm_cand = (int64_t *)gv_alloc((N ? N : 1) * sizeof(int64_t));
    double   *comm_w    = (double *)gv_alloc((N ? N : 1) * sizeof(double));
    size_t   *cpos      = (size_t *)gv_alloc(N * sizeof(size_t));
    size_t   *cstamp    = (size_t *)gv_calloc(N, sizeof(size_t));
    if (!node_ids || !labels || !k || !sigma ||
        !comm_cand || !comm_w || !cpos || !cstamp) {
        gv_free(node_ids); gv_free(labels); gv_free(k); gv_free(sigma);
        gv_free(comm_cand); gv_free(comm_w); gv_free(cpos); gv_free(cstamp);
        uadj_free(&adj); gv_ga_free(ctx);
        return -1;
    }

    double two_m = 0.0; /* 2m = sum of weighted degrees */
    for (size_t u = 0; u < N; u++) {
        node_ids[u] = gv_ga_id(ctx, u);
        double deg = 0.0;
        for (size_t e = adj.off[u]; e < adj.off[u + 1]; e++) deg += adj.wt[e];
        k[u] = deg;
        sigma[u] = deg;        /* each node alone in its community */
        labels[u] = (int64_t)u;
        two_m += deg;
    }

    if (two_m > 0.0) {
        size_t pass_no = 0;
        for (; pass_no < max_passes; pass_no++) {
            int moved = 0;
            for (size_t u = 0; u < N; u++) {
                int64_t cur = labels[u];
                /* Remove u from its community. */
                sigma[cur] -= k[u];

                /* Accumulate weight from u to each neighboring community.
                 * self-community weight is included (loops already stripped). */
                size_t nc = 0;
                size_t pass = u + 1;
                for (size_t e = adj.off[u]; e < adj.off[u + 1]; e++) {
                    int64_t c = labels[adj.adj[e]];
                    size_t key = (size_t)c;
                    if (cstamp[key] == pass) {
                        comm_w[cpos[key]] += adj.wt[e];
                    } else {
                        cstamp[key] = pass;
                        cpos[key] = nc;
                        comm_cand[nc] = c;
                        comm_w[nc] = adj.wt[e];
                        nc++;
                    }
                }

                /* Baseline: staying in `cur`. Gain relative to isolated node is
                 *   w_to_c - sigma_c * k_u / (2m).
                 * Choose the community maximizing this; tie -> smaller community
                 * id, and prefer staying in `cur` for stability. */
                double w_cur = 0.0;
                if (cstamp[(size_t)cur] == pass) w_cur = comm_w[cpos[(size_t)cur]];
                double best_gain = w_cur - sigma[cur] * k[u] / two_m;
                int64_t best_c = cur;

                for (size_t c = 0; c < nc; c++) {
                    int64_t cid = comm_cand[c];
                    if (cid == cur) continue;
                    double gain = comm_w[c] - sigma[cid] * k[u] / two_m;
                    if (gain > best_gain ||
                        (gain == best_gain && cid < best_c)) {
                        best_gain = gain;
                        best_c = cid;
                    }
                }

                /* Insert u into chosen community. */
                sigma[best_c] += k[u];
                if (best_c != cur) { labels[u] = best_c; moved = 1; }
            }
            if (!moved) break;
        }
    }

    gv_free(k); gv_free(sigma);
    gv_free(comm_cand); gv_free(comm_w); gv_free(cpos); gv_free(cstamp);
    uadj_free(&adj);

    size_t cap = N * 2 + 1;
    int64_t *map_key = (int64_t *)gv_alloc(cap * sizeof(int64_t));
    size_t  *map_val = (size_t *)gv_alloc(cap * sizeof(size_t));
    uint8_t *occ     = (uint8_t *)gv_calloc(cap, sizeof(uint8_t));
    if (!map_key || !map_val || !occ) {
        gv_free(map_key); gv_free(map_val); gv_free(occ);
        gv_free(node_ids); gv_free(labels);
        gv_ga_free(ctx);
        return -1;
    }
    size_t K = compact_labels(labels, N, map_key, map_val, occ, cap);
    gv_free(map_key); gv_free(map_val); gv_free(occ);
    gv_ga_free(ctx);

    out->node_ids   = node_ids;
    out->labels     = labels;
    out->count      = N;
    out->num_labels = K;
    return 0;
}

/* ── Triangle counting (undirected) ─────────────────────────────────────────
 * For each node, count triangles it participates in via neighbor-set
 * intersection. To count each triangle exactly once (before dividing), iterate
 * unordered pairs of neighbors u's neighbor v (with v>u by dense index) and, for
 * that edge {u,v}, count common neighbors w with w>v. Distribute +1 to u, v, w
 * per triangle so per-node scores are the true participation counts, and total
 * = sum(scores)/3. */
int graph_triangle_count(const GV_GraphDB *g, GV_GraphNodeScores *out, uint64_t *total) {
    if (total) *total = 0;
    if (!g || !out) return -1;
    memset(out, 0, sizeof(*out));

    GV_GAContext *ctx = gv_ga_build(g);
    if (!ctx) return -1;
    size_t N = gv_ga_count(ctx);
    if (N == 0) { gv_ga_free(ctx); return 0; }

    GV_UAdj adj;
    if (uadj_build(g, ctx, &adj) != 0) { gv_ga_free(ctx); return -1; }

    uint64_t *node_ids = (uint64_t *)gv_alloc(N * sizeof(uint64_t));
    double   *scores   = (double *)gv_calloc(N, sizeof(double));
    /* Membership marker for "is x a neighbor of the current node u", keyed by
     * dense index and stamped per-u to avoid O(N) clears. */
    size_t   *mark     = (size_t *)gv_calloc(N, sizeof(size_t));
    if (!node_ids || !scores || !mark) {
        gv_free(node_ids); gv_free(scores); gv_free(mark);
        uadj_free(&adj); gv_ga_free(ctx);
        return -1;
    }
    for (size_t i = 0; i < N; i++) node_ids[i] = gv_ga_id(ctx, i);

    uint64_t tri_total = 0;
    for (size_t u = 0; u < N; u++) {
        size_t pass = u + 1;
        /* Mark u's neighbors. */
        for (size_t e = adj.off[u]; e < adj.off[u + 1]; e++) mark[adj.adj[e]] = pass;

        /* For each neighbor v>u, count common neighbors w>v. */
        for (size_t e = adj.off[u]; e < adj.off[u + 1]; e++) {
            size_t v = adj.adj[e];
            if (v <= u) continue;
            for (size_t f = adj.off[v]; f < adj.off[v + 1]; f++) {
                size_t w = adj.adj[f];
                if (w <= v) continue;
                if (mark[w] == pass) {
                    /* triangle {u, v, w} */
                    scores[u] += 1.0;
                    scores[v] += 1.0;
                    scores[w] += 1.0;
                    tri_total++;
                }
            }
        }
    }

    gv_free(mark);
    uadj_free(&adj);
    gv_ga_free(ctx);

    out->node_ids = node_ids;
    out->scores   = scores;
    out->count    = N;
    if (total) *total = tri_total;
    return 0;
}

/* ── k-core decomposition (undirected) ──────────────────────────────────────
 * Core number of each node via the standard peeling algorithm: repeatedly
 * remove the node of smallest current degree; its core number is the max
 * degree-threshold reached so far. Implemented with a bucket/bin sort
 * (Batagelj–Zaversnik) for O(V+E). labels[i] = core number; num_labels =
 * max core + 1. */
int graph_kcore(const GV_GraphDB *g, GV_GraphNodeLabels *out) {
    if (!g || !out) return -1;
    memset(out, 0, sizeof(*out));

    GV_GAContext *ctx = gv_ga_build(g);
    if (!ctx) return -1;
    size_t N = gv_ga_count(ctx);
    if (N == 0) { gv_ga_free(ctx); return 0; }

    GV_UAdj adj;
    if (uadj_build(g, ctx, &adj) != 0) { gv_ga_free(ctx); return -1; }

    uint64_t *node_ids = (uint64_t *)gv_alloc(N * sizeof(uint64_t));
    int64_t  *labels   = (int64_t *)gv_alloc(N * sizeof(int64_t));
    size_t   *deg      = (size_t *)gv_alloc(N * sizeof(size_t));
    /* Bucket-sort structures (Batagelj–Zaversnik). */
    size_t   *pos      = (size_t *)gv_alloc(N * sizeof(size_t)); /* vert -> position in `vert` */
    size_t   *vert     = (size_t *)gv_alloc(N * sizeof(size_t)); /* ordered by degree */
    if (!node_ids || !labels || !deg || !pos || !vert) {
        gv_free(node_ids); gv_free(labels); gv_free(deg);
        gv_free(pos); gv_free(vert);
        uadj_free(&adj); gv_ga_free(ctx);
        return -1;
    }

    size_t maxdeg = 0;
    for (size_t u = 0; u < N; u++) {
        node_ids[u] = gv_ga_id(ctx, u);
        deg[u] = adj.off[u + 1] - adj.off[u];
        if (deg[u] > maxdeg) maxdeg = deg[u];
    }

    /* bin[d] = start index in `vert` of vertices with degree d. */
    size_t *bin = (size_t *)gv_calloc(maxdeg + 1, sizeof(size_t));
    if (!bin) {
        gv_free(node_ids); gv_free(labels); gv_free(deg);
        gv_free(pos); gv_free(vert);
        uadj_free(&adj); gv_ga_free(ctx);
        return -1;
    }

    for (size_t u = 0; u < N; u++) bin[deg[u]]++;
    /* Prefix sum -> bin[d] = first slot for degree d. */
    size_t start = 0;
    for (size_t d = 0; d <= maxdeg; d++) {
        size_t cnt = bin[d];
        bin[d] = start;
        start += cnt;
    }
    /* Bucket-place vertices (stable in dense-index order for determinism). */
    for (size_t u = 0; u < N; u++) {
        pos[u] = bin[deg[u]];
        vert[pos[u]] = u;
        bin[deg[u]]++;
    }
    /* Restore bin[d] to point at the start of degree-d block. */
    for (size_t d = maxdeg; d > 0; d--) bin[d] = bin[d - 1];
    bin[0] = 0;

    /* Peel: process vertices in increasing current degree. */
    for (size_t i = 0; i < N; i++) {
        size_t u = vert[i];
        labels[u] = (int64_t)deg[u]; /* core number fixed at removal time */
        for (size_t e = adj.off[u]; e < adj.off[u + 1]; e++) {
            size_t v = adj.adj[e];
            if (deg[v] > deg[u]) {
                /* Decrease deg[v] by moving it to the front of its degree block:
                 * swap with the first vertex of the same degree, then advance
                 * that bin's boundary. */
                size_t dv = deg[v];
                size_t pv = pos[v];
                size_t pw = bin[dv];
                size_t w = vert[pw];
                if (u != w && pv != pw) {
                    vert[pv] = w; pos[w] = pv;
                    vert[pw] = v; pos[v] = pw;
                }
                bin[dv]++;
                deg[v]--;
            }
        }
    }

    size_t K = (size_t)0;
    for (size_t u = 0; u < N; u++) {
        if ((size_t)labels[u] + 1 > K) K = (size_t)labels[u] + 1;
    }

    gv_free(bin);
    gv_free(deg); gv_free(pos); gv_free(vert);
    uadj_free(&adj);
    gv_ga_free(ctx);

    out->node_ids   = node_ids;
    out->labels     = labels;
    out->count      = N;
    out->num_labels = K;
    return 0;
}

/* ── Tarjan strongly connected components (directed, iterative) ──────────────
 * Directed: follows out_edges only. Iterative DFS with an explicit work stack
 * (frame = (node, next-edge-cursor)) so deep graphs cannot overflow the C
 * stack. SCC ids are assigned in the order components are completed, then are
 * already dense 0..K-1. */
int graph_strongly_connected_components(const GV_GraphDB *g, GV_GraphNodeLabels *out) {
    if (!g || !out) return -1;
    memset(out, 0, sizeof(*out));

    GV_GAContext *ctx = gv_ga_build(g);
    if (!ctx) return -1;
    size_t N = gv_ga_count(ctx);
    if (N == 0) { gv_ga_free(ctx); return 0; }

    /* Directed out-adjacency in dense-index CSR (drops self-loops / absent). */
    size_t *doff = (size_t *)gv_calloc(N + 1, sizeof(size_t));
    if (!doff) { gv_ga_free(ctx); return -1; }
    for (size_t u = 0; u < N; u++) {
        const GV_GraphNode *node = graph_get_node(g, gv_ga_id(ctx, u));
        size_t cnt = 0;
        if (node) {
            for (size_t e = 0; e < node->out_count; e++) {
                size_t v = gv_ga_index(ctx, node->out_edges[e].neighbor_id);
                if (v == (size_t)-1 || v == u) continue;
                cnt++;
            }
        }
        doff[u + 1] = cnt;
    }
    for (size_t u = 0; u < N; u++) doff[u + 1] += doff[u];
    size_t M = doff[N];

    size_t *dadj = (size_t *)gv_alloc((M ? M : 1) * sizeof(size_t));
    if (!dadj) { gv_free(doff); gv_ga_free(ctx); return -1; }
    for (size_t u = 0; u < N; u++) {
        const GV_GraphNode *node = graph_get_node(g, gv_ga_id(ctx, u));
        size_t w = doff[u];
        if (node) {
            for (size_t e = 0; e < node->out_count; e++) {
                size_t v = gv_ga_index(ctx, node->out_edges[e].neighbor_id);
                if (v == (size_t)-1 || v == u) continue;
                dadj[w++] = v;
            }
        }
    }

    uint64_t *node_ids = (uint64_t *)gv_alloc(N * sizeof(uint64_t));
    int64_t  *labels   = (int64_t *)gv_alloc(N * sizeof(int64_t));
    size_t   *index    = (size_t *)gv_alloc(N * sizeof(size_t));  /* discovery index */
    size_t   *lowlink  = (size_t *)gv_alloc(N * sizeof(size_t));
    uint8_t  *onstack  = (uint8_t *)gv_calloc(N, sizeof(uint8_t));
    uint8_t  *visited  = (uint8_t *)gv_calloc(N, sizeof(uint8_t));
    size_t   *tstack   = (size_t *)gv_alloc(N * sizeof(size_t));  /* Tarjan stack */
    /* Explicit DFS work stack: parallel node + edge-cursor arrays. */
    size_t   *wnode    = (size_t *)gv_alloc(N * sizeof(size_t));
    size_t   *wedge    = (size_t *)gv_alloc(N * sizeof(size_t));
    if (!node_ids || !labels || !index || !lowlink || !onstack ||
        !visited || !tstack || !wnode || !wedge) {
        gv_free(node_ids); gv_free(labels); gv_free(index); gv_free(lowlink);
        gv_free(onstack); gv_free(visited); gv_free(tstack);
        gv_free(wnode); gv_free(wedge);
        gv_free(doff); gv_free(dadj); gv_ga_free(ctx);
        return -1;
    }
    for (size_t i = 0; i < N; i++) { node_ids[i] = gv_ga_id(ctx, i); labels[i] = -1; }

    size_t next_index = 0;   /* global discovery counter */
    size_t tsp = 0;          /* Tarjan stack pointer */
    size_t comp = 0;         /* next SCC id */

    for (size_t s = 0; s < N; s++) {
        if (visited[s]) continue;

        size_t wsp = 0;
        /* initialize root frame */
        wnode[wsp] = s;
        wedge[wsp] = 0;
        wsp++;
        /* discover s */
        index[s] = next_index; lowlink[s] = next_index; next_index++;
        visited[s] = 1;
        tstack[tsp++] = s; onstack[s] = 1;

        while (wsp > 0) {
            size_t u = wnode[wsp - 1];
            size_t ei = wedge[wsp - 1];
            size_t e_begin = doff[u];
            size_t e_end   = doff[u + 1];

            if (e_begin + ei < e_end) {
                size_t v = dadj[e_begin + ei];
                wedge[wsp - 1] = ei + 1;          /* advance cursor */
                if (!visited[v]) {
                    /* discover v, recurse (push frame) */
                    index[v] = next_index; lowlink[v] = next_index; next_index++;
                    visited[v] = 1;
                    tstack[tsp++] = v; onstack[v] = 1;
                    wnode[wsp] = v; wedge[wsp] = 0; wsp++;
                } else if (onstack[v]) {
                    /* back/cross edge to a node on stack */
                    if (index[v] < lowlink[u]) lowlink[u] = index[v];
                }
            } else {
                /* all edges of u explored: finalize */
                if (lowlink[u] == index[u]) {
                    /* pop an SCC */
                    for (;;) {
                        size_t w = tstack[--tsp];
                        onstack[w] = 0;
                        labels[w] = (int64_t)comp;
                        if (w == u) break;
                    }
                    comp++;
                }
                wsp--;
                if (wsp > 0) {
                    size_t parent = wnode[wsp - 1];
                    if (lowlink[u] < lowlink[parent]) lowlink[parent] = lowlink[u];
                }
            }
        }
    }

    size_t K = comp;

    gv_free(index); gv_free(lowlink); gv_free(onstack); gv_free(visited);
    gv_free(tstack); gv_free(wnode); gv_free(wedge);
    gv_free(doff); gv_free(dadj);
    gv_ga_free(ctx);

    out->node_ids   = node_ids;
    out->labels     = labels;
    out->count      = N;
    out->num_labels = K;
    return 0;
}
