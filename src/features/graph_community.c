/**
 * graph_community.c — community detection / clustering algorithms for the
 * graph-algorithms layer (graph_algos.h):
 *
 *   - graph_label_propagation      Label Propagation (LPA), undirected
 *   - graph_louvain                multi-level Louvain modularity maximization
 *   - graph_leiden                 Leiden (Louvain + connectivity refinement)
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
#include "core/compat.h"

#include <math.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
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
static double edge_weight(const GV_GAContext *ctx, uint64_t edge_id) {
    const GV_GraphEdge *e = gv_ga_edge(ctx, edge_id);
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
    (void)g;
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
        const GV_GraphNode *node = gv_ga_node(ctx, uid);
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
        const GV_GraphNode *node = gv_ga_node(ctx, uid);
        size_t local = 0;
        size_t pass = u + 1;
        if (node) {
            for (size_t e = 0; e < node->out_count; e++) {
                size_t v = gv_ga_index(ctx, node->out_edges[e].neighbor_id);
                if (v == (size_t)-1 || v == u) continue;
                double w = edge_weight(ctx, node->out_edges[e].edge_id);
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
                double w = edge_weight(ctx, node->in_edges[e].edge_id);
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

/* ── Multi-level modularity machinery (Louvain / Leiden) ────────────────────
 * Both algorithms share the same level loop:
 *   1. local moving: move nodes/supernodes to the neighboring community with
 *      the largest positive modularity gain dQ = w_ic - Sigma_tot_c*k_i/(2m)
 *      until stable,
 *   2. (Leiden only) refinement: within each community, re-partition the nodes
 *      into internally-connected sub-communities via restricted local moving
 *      (a node may not leave its sub-community if it is a cut vertex of it),
 *   3. aggregation: build the next-level graph over the refined partition and
 *      repeat.
 * Leiden's refinement guarantees aggregated communities stay connected —
 * unlike plain Louvain, which can produce arbitrarily badly-connected
 * communities. Everything is deterministic (dense-index order, ties broken
 * toward smaller community ids).
 *
 * Level graphs are weighted CSR plus a self-loop weight array (aggregated
 * internal edges). Self-loops contribute 2w to the weighted degree (two
 * stubs), matching the standard modularity convention where 2m = sum(k). */

typedef struct {
    size_t *off;     /* N+1 offsets */
    size_t *adj;     /* M neighbor ids */
    double *wt;      /* M weights parallel to adj */
    double *selfw;   /* per-node self-loop weight (0 at level 0) */
    size_t  N;
    size_t  M;
} GV_LGraph;

static void lgraph_free(GV_LGraph *lg) {
    if (!lg) return;
    gv_free(lg->off); gv_free(lg->adj); gv_free(lg->wt); gv_free(lg->selfw);
    memset(lg, 0, sizeof(*lg));
}

static GV_LGraph lgraph_from_uadj(const GV_UAdj *ua) {
    GV_LGraph lg;
    lg.N = ua->N; lg.M = ua->M;
    lg.off = NULL; lg.adj = NULL; lg.wt = NULL; lg.selfw = NULL;
    if (ua->N == 0) return lg;
    lg.off = (size_t *)gv_alloc((ua->N + 1) * sizeof(size_t));
    lg.adj = (size_t *)gv_alloc((ua->M ? ua->M : 1) * sizeof(size_t));
    lg.wt  = (double *)gv_alloc((ua->M ? ua->M : 1) * sizeof(double));
    lg.selfw = (double *)gv_calloc(ua->N, sizeof(double));
    if (!lg.off || !lg.adj || !lg.wt || !lg.selfw) {
        lgraph_free(&lg);
        return lg;
    }
    memcpy(lg.off, ua->off, (ua->N + 1) * sizeof(size_t));
    memcpy(lg.adj, ua->adj, ua->M * sizeof(size_t));
    memcpy(lg.wt,  ua->wt,  ua->M * sizeof(double));
    return lg;
}

static void lgraph_init_singletons(const GV_LGraph *lg, int64_t *labels,
                                   double *k, double *sigma, double *two_m) {
    double tm = 0.0;
    for (size_t i = 0; i < lg->N; i++) {
        double ki = 2.0 * lg->selfw[i]; /* self-loop counts as two stubs */
        for (size_t e = lg->off[i]; e < lg->off[i + 1]; e++) ki += lg->wt[e];
        k[i] = ki;
        sigma[i] = ki;
        labels[i] = (int64_t)i;
        tm += ki;
    }
    *two_m = tm;
}

/* One round of local moving over the whole graph. Returns 1 if any node moved. */
static int local_move_round(const GV_LGraph *lg, int64_t *labels,
                            double *k, double *sigma, double two_m,
                            int64_t *cand, double *cw, size_t *cpos,
                            size_t *cstamp) {
    int moved = 0;
    /* Stamp keys are unique per visitation (zeroed per round), so stale
     * values from previous rounds can never alias. */
    memset(cstamp, 0, lg->N * sizeof(size_t));
    for (size_t u = 0; u < lg->N; u++) {
        int64_t cur = labels[u];
        sigma[cur] -= k[u];

        size_t nc = 0;
        size_t pass = u + 1;
        for (size_t e = lg->off[u]; e < lg->off[u + 1]; e++) {
            int64_t c = labels[lg->adj[e]];
            size_t key = (size_t)c;
            if (cstamp[key] == pass) {
                cw[cpos[key]] += lg->wt[e];
            } else {
                cstamp[key] = pass;
                cpos[key] = nc;
                cand[nc] = c;
                cw[nc] = lg->wt[e];
                nc++;
            }
        }

        /* Own-community edge weight includes the self-loop: it stays with u
         * whichever community holds it, so it belongs in the baseline. */
        double w_cur = lg->selfw[u];
        if (cstamp[(size_t)cur] == pass) w_cur += cw[cpos[(size_t)cur]];
        double best_gain = w_cur - sigma[cur] * k[u] / two_m;
        int64_t best_c = cur;

        for (size_t c = 0; c < nc; c++) {
            int64_t cid = cand[c];
            if (cid == cur) continue;
            double gain = cw[c] - sigma[cid] * k[u] / two_m;
            if (gain > best_gain || (gain == best_gain && cid < best_c)) {
                best_gain = gain;
                best_c = cid;
            }
        }

        sigma[best_c] += k[u];
        if (best_c != cur) { labels[u] = best_c; moved = 1; }
    }
    return moved;
}

/* ── Leiden refinement ───────────────────────────────────────────────────────
 * Within each community of `comm`, re-partition nodes into sub-communities
 * that are internally connected. Nodes start as singletons; restricted local
 * moving merges sub-communities, but a node may only leave its current
 * sub-community if it is NOT a cut vertex of it (checked by BFS over the
 * sub-community minus the node), so every final sub-community is guaranteed
 * connected. Output `sub` uses dense ids in [0, K). Returns the number of
 * rounds run (moves stop early once a full pass makes no move). */
static void refine_partitions(const GV_LGraph *lg, const int64_t *comm,
                              int64_t *sub, size_t max_rounds,
                              double *k, double *sigma,
                              int64_t *cand, double *cw, size_t *cpos,
                              size_t *cstamp, size_t *bfs_queue,
                              size_t *bfs_stamp) {
    const size_t N = lg->N;
    double two_m = 0.0;
    for (size_t i = 0; i < N; i++) {
        double ki = 2.0 * lg->selfw[i];
        for (size_t e = lg->off[i]; e < lg->off[i + 1]; e++) ki += lg->wt[e];
        k[i] = ki;
        sigma[i] = ki; /* singleton sub-communities */
        sub[i] = (int64_t)i;
        two_m += ki;
    }
    if (two_m <= 0.0) return;
    memset(bfs_stamp, 0, N * sizeof(size_t));

    for (size_t round = 0; round < max_rounds; round++) {
        int moved = 0;
        /* Zero per round so repeating stamp keys can never alias stale
         * entries from earlier rounds. */
        memset(cstamp, 0, N * sizeof(size_t));
        memset(bfs_stamp, 0, N * sizeof(size_t));
        for (size_t u = 0; u < N; u++) {
            int64_t s = sub[u];
            int64_t cu = comm[u];

            /* Walk u's neighbors within u's own community: accumulate weight
             * to u's current sub-community (`same_w`), and per other
             * sub-community weights as merge candidates. */
            size_t nc = 0;
            size_t pass = N + u + 1;
            double same_w = 0.0;
            size_t same_deg = 0;
            for (size_t e = lg->off[u]; e < lg->off[u + 1]; e++) {
                size_t v = lg->adj[e];
                if (comm[v] != cu) continue;
                if (sub[v] == s) {
                    same_w += lg->wt[e];
                    same_deg++;
                    continue;
                }
                int64_t t = sub[v];
                size_t key = (size_t)t;
                if (cstamp[key] == pass) {
                    cw[cpos[key]] += lg->wt[e];
                } else {
                    cstamp[key] = pass;
                    cpos[key] = nc;
                    cand[nc] = t;
                    cw[nc] = lg->wt[e];
                    nc++;
                }
            }
            if (nc == 0) continue;

            /* Cut-vertex guard: if u has >= 2 neighbors inside its
             * sub-community and removing u disconnects them, u must stay put
             * (leaving would split s\{u}). */
            if (same_deg >= 2) {
                size_t first = (size_t)-1;
                for (size_t e = lg->off[u]; e < lg->off[u + 1]; e++) {
                    size_t v = lg->adj[e];
                    if (comm[v] == cu && sub[v] == s) { first = v; break; }
                }
                if (first != (size_t)-1) {
                    size_t bpass = N + u + 1;
                    size_t qh = 0, qt = 0;
                    bfs_stamp[first] = bpass;
                    bfs_queue[qt++] = first;
                    while (qh < qt) {
                        size_t x = bfs_queue[qh++];
                        for (size_t e = lg->off[x]; e < lg->off[x + 1]; e++) {
                            size_t y = lg->adj[e];
                            if (y == u || sub[y] != s || bfs_stamp[y] == bpass)
                                continue;
                            bfs_stamp[y] = bpass;
                            bfs_queue[qt++] = y;
                        }
                    }
                    int connected = 1;
                    for (size_t e = lg->off[u]; e < lg->off[u + 1]; e++) {
                        size_t v = lg->adj[e];
                        if (comm[v] == cu && sub[v] == s &&
                            bfs_stamp[v] != bpass) {
                            connected = 0;
                            break;
                        }
                    }
                    if (!connected) continue;
                }
            }

            /* Best strict-improvement target; baseline = staying in s. */
            double base = same_w + lg->selfw[u] - sigma[s] * k[u] / two_m;
            int64_t best_t = s;
            double best_gain = base;
            for (size_t c = 0; c < nc; c++) {
                int64_t t = cand[c];
                double gain = cw[c] - sigma[t] * k[u] / two_m;
                if (gain > best_gain || (gain == best_gain && t < best_t)) {
                    best_gain = gain;
                    best_t = t;
                }
            }
            if (best_t == s || best_gain <= base) continue;

            sigma[s] -= k[u];
            sigma[best_t] += k[u];
            sub[u] = best_t;
            moved = 1;
        }
        if (!moved) break;
    }

    /* Densify sub ids so downstream code can safely use them as indices. */
    size_t cap = N * 2 + 1;
    int64_t *map_key = (int64_t *)gv_alloc(cap * sizeof(int64_t));
    size_t  *map_val = (size_t *)gv_alloc(cap * sizeof(size_t));
    uint8_t *occ     = (uint8_t *)gv_calloc(cap, sizeof(uint8_t));
    if (map_key && map_val && occ)
        compact_labels(sub, N, map_key, map_val, occ, cap);
    gv_free(map_key); gv_free(map_val); gv_free(occ);
}

/* Aggregate `lg` over dense partition `part` ([0,K)). Returns a fresh graph
 * with K supernodes: parallel edges between supernode pairs are summed into
 * one entry; internal edges become self-loop weight. */
static int lgraph_aggregate(const GV_LGraph *lg, const int64_t *part,
                            GV_LGraph *out) {
    memset(out, 0, sizeof(*out));
    size_t K = 0;
    for (size_t i = 0; i < lg->N; i++)
        if ((size_t)part[i] + 1 > K) K = (size_t)part[i] + 1;
    if (K == 0) return 0;

    out->N = K;
    out->off = (size_t *)gv_calloc(K + 1, sizeof(size_t));
    out->selfw = (double *)gv_calloc(K, sizeof(double));
    if (!out->off || !out->selfw) { lgraph_free(out); return -1; }

    /* Pass 1: count deduped neighbor entries per supernode row (+ selfw). */
    size_t *slot  = (size_t *)gv_alloc(K * sizeof(size_t));
    size_t *stamp = (size_t *)gv_calloc(K, sizeof(size_t));
    if (!slot || !stamp) {
        gv_free(slot); gv_free(stamp); lgraph_free(out);
        return -1;
    }
    for (size_t u = 0; u < lg->N; u++) {
        size_t pu = (size_t)part[u];
        size_t pass = pu + 1;
        for (size_t e = lg->off[u]; e < lg->off[u + 1]; e++) {
            size_t pv = (size_t)part[lg->adj[e]];
            if (pv == pu) { out->selfw[pu] += lg->wt[e]; continue; }
            if (stamp[pv] != pass) { stamp[pv] = pass; out->off[pu + 1]++; }
        }
    }
    for (size_t c = 0; c < K; c++) out->off[c + 1] += out->off[c];
    out->M = out->off[K];

    out->adj = (size_t *)gv_alloc((out->M ? out->M : 1) * sizeof(size_t));
    out->wt  = (double *)gv_alloc((out->M ? out->M : 1) * sizeof(double));
    if (!out->adj || !out->wt) {
        gv_free(slot); gv_free(stamp); lgraph_free(out);
        return -1;
    }
    memset(stamp, 0, K * sizeof(size_t));

    /* Pass 2: fill rows (summing parallel edges). All nodes of one supernode
     * share its pass stamp, so dedup spans the whole supernode; each row has
     * its own fill cursor so nodes of the same supernode append instead of
     * overwrite. */
    size_t *fill = (size_t *)gv_calloc(K, sizeof(size_t));
    if (!fill) {
        gv_free(slot); gv_free(stamp); lgraph_free(out);
        return -1;
    }
    for (size_t u = 0; u < lg->N; u++) {
        size_t pu = (size_t)part[u];
        size_t pass = pu + 1;
        size_t base = out->off[pu];
        for (size_t e = lg->off[u]; e < lg->off[u + 1]; e++) {
            size_t pv = (size_t)part[lg->adj[e]];
            if (pv == pu) continue;
            if (stamp[pv] == pass) {
                out->wt[base + slot[pv]] += lg->wt[e];
            } else {
                stamp[pv] = pass;
                slot[pv] = fill[pu];
                fill[pu]++;
                out->adj[base + slot[pv]] = pv;
                out->wt[base + slot[pv]] = lg->wt[e];
            }
        }
    }
    gv_free(fill); gv_free(slot); gv_free(stamp);

    return 0;
}

/* Shared multi-level driver for Louvain (use_refine=0) and Leiden (=1).
 *
 * Level loop on graph G_l:
 *   1. local moving from singleton communities until stable -> partition P_l;
 *   2. Leiden: refine each community of P_l into connected sub-communities,
 *      and the refined partition replaces P_l;
 *   3. aggregate G_{l+1} over P_l; repeat unless P_l is all singletons or no
 *      move happened anywhere.
 *
 * Every level's partition is recorded; the final label of an original node is
 * obtained by composing partitions top-down: C(i) = P_L(...P_0(i)). */
static int modularity_levels(const GV_GraphDB *g, size_t max_levels,
                             int use_refine, GV_GraphNodeLabels *out) {
    if (!g || !out) return -1;
    memset(out, 0, sizeof(*out));
    if (max_levels == 0) max_levels = 10;

    GV_GAContext *ctx = gv_ga_build(g);
    if (!ctx) return -1;
    size_t N = gv_ga_count(ctx);
    if (N == 0) { gv_ga_free(ctx); return 0; }

    GV_UAdj ua;
    if (uadj_build(g, ctx, &ua) != 0) { gv_ga_free(ctx); return -1; }

    uint64_t *node_ids = (uint64_t *)gv_alloc(N * sizeof(uint64_t));
    if (!node_ids) { uadj_free(&ua); gv_ga_free(ctx); return -1; }
    for (size_t i = 0; i < N; i++) node_ids[i] = gv_ga_id(ctx, i);

    /* Scratch sized for the largest (level-0) graph. Community/sub ids are
     * always dense in [0, N), so stamps keyed by id fit. Refinement stamps
     * use the offset N+u+1 so they can never collide with stale values. */
    double   *k       = (double *)gv_alloc(N * sizeof(double));
    double   *sigma   = (double *)gv_alloc(N * sizeof(double));
    int64_t  *cand    = (int64_t *)gv_alloc(N * sizeof(int64_t));
    double   *cw      = (double *)gv_alloc(N * sizeof(double));
    size_t   *cpos    = (size_t *)gv_alloc(N * sizeof(size_t));
    size_t   *cstamp  = (size_t *)gv_calloc(N, sizeof(size_t));
    size_t   *bfs_q   = (size_t *)gv_alloc(N * sizeof(size_t));
    size_t   *bfs_st  = (size_t *)gv_calloc(N, sizeof(size_t));
    int64_t **parts   = (int64_t **)gv_calloc(max_levels + 1, sizeof(int64_t *));
    int ok = k && sigma && cand && cw && cpos && cstamp && bfs_q && bfs_st &&
             parts;

    GV_LGraph cur;
    memset(&cur, 0, sizeof(cur));
    size_t nparts = 0;
    int64_t *labels = NULL, *sub = NULL, *comm = NULL, *fold = NULL;

    if (ok) {
        cur = lgraph_from_uadj(&ua);
        ok = cur.off != NULL;
    }
    if (ok) {
        labels = (int64_t *)gv_alloc(cur.N * sizeof(int64_t));
        sub    = (int64_t *)gv_alloc(cur.N * sizeof(int64_t));
        comm   = (int64_t *)gv_alloc(cur.N * sizeof(int64_t));
        ok = labels && sub && comm;
    }

    if (ok) {
        for (size_t lvl = 0; lvl < max_levels; lvl++) {
            double two_m;
            lgraph_init_singletons(&cur, labels, k, sigma, &two_m);
            if (two_m <= 0.0) break; /* edgeless: singletons are optimal */

            int changed = 0;
            for (;;) {
                if (!local_move_round(&cur, labels, k, sigma, two_m,
                                      cand, cw, cpos, cstamp))
                    break;
                changed = 1;
            }

            const int64_t *level_part = labels;
            if (use_refine && changed) {
                memcpy(comm, labels, cur.N * sizeof(int64_t));
                refine_partitions(&cur, comm, sub, max_levels,
                                  k, sigma, cand, cw, cpos, cstamp,
                                  bfs_q, bfs_st);
                level_part = sub;
            } else if (!changed) {
                parts[nparts] = (int64_t *)gv_alloc(cur.N * sizeof(int64_t));
                if (!parts[nparts]) { ok = 0; break; }
                memcpy(parts[nparts], labels, cur.N * sizeof(int64_t));
                nparts++;
                break; /* converged: further levels cannot improve */
            }

            /* Record this level's partition. */
            parts[nparts] = (int64_t *)gv_alloc(cur.N * sizeof(int64_t));
            if (!parts[nparts]) { ok = 0; break; }
            memcpy(parts[nparts], level_part, cur.N * sizeof(int64_t));
            nparts++;

            /* All singletons -> aggregation cannot change anything. */
            size_t K = 0;
            for (size_t i = 0; i < cur.N; i++)
                if ((size_t)level_part[i] + 1 > K) K = (size_t)level_part[i] + 1;
            if (K == cur.N) break;

            GV_LGraph next;
            if (lgraph_aggregate(&cur, level_part, &next) != 0) { ok = 0; break; }
            lgraph_free(&cur);
            cur = next;

            gv_free(labels); gv_free(sub); gv_free(comm);
            labels = (int64_t *)gv_alloc(cur.N * sizeof(int64_t));
            sub    = (int64_t *)gv_alloc(cur.N * sizeof(int64_t));
            comm   = (int64_t *)gv_alloc(cur.N * sizeof(int64_t));
            if (!labels || !sub || !comm) { ok = 0; break; }
        }
    }

    lgraph_free(&cur);
    gv_free(k); gv_free(sigma); gv_free(cand); gv_free(cw);
    gv_free(cpos); gv_free(cstamp); gv_free(bfs_q); gv_free(bfs_st);
    gv_free(labels); gv_free(sub); gv_free(comm);
    uadj_free(&ua);
    gv_ga_free(ctx);

    /* Compose partitions down to original nodes. With no recorded levels
     * (edgeless graph) every node keeps its singleton label. */
    fold = (int64_t *)gv_alloc((N ? N : 1) * sizeof(int64_t));
    if (!ok || !fold) {
        for (size_t l = 0; l < max_levels + 1; l++) gv_free(parts[l]);
        gv_free(parts);
        gv_free(fold);
        gv_free(node_ids);
        return -1;
    }
    if (nparts == 0) {
        for (size_t i = 0; i < N; i++) fold[i] = (int64_t)i;
    } else {
        for (size_t i = 0; i < N; i++) {
            int64_t c = parts[0][i];
            for (size_t l = 1; l < nparts; l++)
                c = parts[l][(size_t)c];
            fold[i] = c;
        }
    }
    for (size_t l = 0; l < max_levels + 1; l++) gv_free(parts[l]);
    gv_free(parts);

    size_t cap = N * 2 + 1;
    int64_t *map_key = (int64_t *)gv_alloc(cap * sizeof(int64_t));
    size_t  *map_val = (size_t *)gv_alloc(cap * sizeof(size_t));
    uint8_t *occ     = (uint8_t *)gv_calloc(cap, sizeof(uint8_t));
    if (!map_key || !map_val || !occ) {
        gv_free(map_key); gv_free(map_val); gv_free(occ);
        gv_free(fold); gv_free(node_ids);
        return -1;
    }
    size_t K = compact_labels(fold, N, map_key, map_val, occ, cap);
    gv_free(map_key); gv_free(map_val); gv_free(occ);

    out->node_ids   = node_ids;
    out->labels     = fold;
    out->count      = N;
    out->num_labels = K;
    return 0;
}

/** Louvain modularity-maximizing community detection, multi-level: iterates
 *  local moves then aggregates the graph, repeating until modularity stops
 *  improving (at most max_levels aggregations). Treated as undirected. */
int graph_louvain(const GV_GraphDB *g, size_t max_levels, GV_GraphNodeLabels *out) {
    return modularity_levels(g, max_levels, 0, out);
}

/** Leiden community detection: like multi-level Louvain but each aggregation
 *  is preceded by a refinement phase that re-partitions communities into
 *  internally-connected sub-communities (restricted local moving with a
 *  cut-vertex guard), guaranteeing well-connected communities. Deterministic. */
int graph_leiden(const GV_GraphDB *g, size_t max_levels, GV_GraphNodeLabels *out) {
    return modularity_levels(g, max_levels, 1, out);
}
/* ── Triangle counting (undirected) ─────────────────────────────────────────
 * For each node, count triangles it participates in via neighbor-set
 * intersection. To count each triangle exactly once (before dividing), iterate
 * unordered pairs of neighbors u's neighbor v (with v>u by dense index) and, for
 * that edge {u,v}, count common neighbors w with w>v. Distribute +1 to u, v, w
 * per triangle so per-node scores are the true participation counts, and total
 * = sum(scores)/3. */
/* Per-worker state: private mark stamps + partial participation counts so
 * workers never share writable cache lines. */
typedef struct {
    size_t   *mark;
    double   *partial;
    uint64_t  triangles;
} TriWorker;

typedef struct {
    const GV_GraphDB *g;
    const GV_GAContext *ctx;
    size_t N;
    GV_UAdj *adj;
    TriWorker *workers;
    size_t next_node;            /* shared work counter */
    int    ok;
} TriJob;

static void triangle_run_node(TriJob *job, size_t u, size_t t) {
    GV_UAdj *adj = job->adj;
    TriWorker *wk = &job->workers[t];
    size_t pass = u + 1;
    for (size_t e = adj->off[u]; e < adj->off[u + 1]; e++) wk->mark[adj->adj[e]] = pass;

    for (size_t e = adj->off[u]; e < adj->off[u + 1]; e++) {
        size_t v = adj->adj[e];
        if (v <= u) continue;
        for (size_t f = adj->off[v]; f < adj->off[v + 1]; f++) {
            size_t x = adj->adj[f];
            if (x <= v) continue;
            if (wk->mark[x] == pass) {
                /* triangle {u, v, x}: credit all three participants */
                wk->partial[u]++;
                wk->partial[v]++;
                wk->partial[x]++;
                wk->triangles++;
            }
        }
    }
}

typedef struct {
    TriJob *job;
    size_t  slot;
} TriSlotArg;

static void *triangle_worker(void *arg) {
    TriSlotArg *sa = (TriSlotArg *)arg;
    TriJob *job = sa->job;
    size_t t = sa->slot;
    for (;;) {
        size_t u = __atomic_fetch_add(&job->next_node, 1, __ATOMIC_RELAXED);
        if (u >= job->N || !job->ok) break;
        triangle_run_node(job, u, t);
    }
    return NULL;
}

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

    long ncpu_long = gv_get_cpu_count();
    size_t nthreads = (size_t)ncpu_long;
    if (nthreads > N) nthreads = N;

    TriJob job;
    memset(&job, 0, sizeof(job));
    job.g = g; job.ctx = ctx; job.N = N; job.adj = &adj; job.ok = 1;

    int ok = 1;
    job.workers = (TriWorker *)gv_calloc(nthreads, sizeof(TriWorker));
    pthread_t *threads = (pthread_t *)gv_calloc(nthreads, sizeof(pthread_t));
    TriSlotArg *args = (TriSlotArg *)gv_calloc(nthreads, sizeof(TriSlotArg));
    if (!job.workers || !threads || !args) ok = 0;

    for (size_t t = 0; ok && t < nthreads; t++) {
        job.workers[t].mark = (size_t *)gv_calloc(N, sizeof(size_t));
        job.workers[t].partial = (double *)gv_calloc(N, sizeof(double));
        if (!job.workers[t].mark || !job.workers[t].partial) { ok = 0; break; }
    }

    if (ok) {
        for (size_t t = 0; t < nthreads; t++) {
            args[t].job = &job;
            args[t].slot = t;
            if (t == nthreads - 1) break;   /* main thread runs last slot */
            if (pthread_create(&threads[t], NULL, triangle_worker,
                               &args[t]) != 0) {
                nthreads = t + 1;
                break;
            }
        }
        triangle_worker(&args[nthreads - 1]);
        for (size_t t = 0; t < nthreads - 1; t++)
            pthread_join(threads[t], NULL);
    }

    uint64_t *node_ids = (uint64_t *)gv_alloc((N ? N : 1) * sizeof(uint64_t));
    double   *scores   = (double *)gv_calloc(N ? N : 1, sizeof(double));
    if (!node_ids || !scores) ok = 0;

    uint64_t tri_total = 0;
    if (ok && node_ids && scores) {
        for (size_t i = 0; i < N; i++) node_ids[i] = gv_ga_id(ctx, i);
        for (size_t t = 0; t < nthreads; t++) {
            tri_total += job.workers[t].triangles;
            for (size_t i = 0; i < N; i++) scores[i] += job.workers[t].partial[i];
        }
    }

    if (job.workers) {
        for (size_t t = 0; t < nthreads; t++) {
            gv_free(job.workers[t].mark);
            gv_free(job.workers[t].partial);
        }
    }
    gv_free(job.workers); gv_free(threads); gv_free(args);
    uadj_free(&adj);
    gv_ga_free(ctx);

    if (!ok) {
        gv_free(node_ids); gv_free(scores);
        return -1;
    }

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
        const GV_GraphNode *node = gv_ga_node(ctx, gv_ga_id(ctx, u));
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
        const GV_GraphNode *node = gv_ga_node(ctx, gv_ga_id(ctx, u));
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
