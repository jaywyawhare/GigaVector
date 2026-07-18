/**
 * graph_centrality.c — centrality algorithms for the GigaVector property graph.
 *
 * Implements the centrality portion of graph_algos.h:
 *   - graph_betweenness_centrality     (Brandes: BFS unweighted / Dijkstra weighted)
 *   - graph_closeness_centrality        (per-node SSSP, Wasserman-Faust normalized)
 *   - graph_harmonic_centrality         (sum of 1/dist over reachable nodes)
 *   - graph_eigenvector_centrality      (power iteration on out-adjacency)
 *   - graph_degree_centrality           (degree / (N-1), in/out/total)
 *   - graph_pagerank_all                (PageRank power iteration)
 *   - graph_personalized_pagerank       (PageRank with a source teleport set)
 *   - graph_article_rank                (PageRank variant, degree-dampened)
 *
 * All algorithms are read-only w.r.t. the graph, use the shared dense node
 * index context (gv_ga_*), and allocate their results with gv_alloc/gv_calloc
 * so callers can free them with graph_node_scores_free. No leaks on any path.
 */
#include "features/graph_algos.h"
#include "features/graph_csr.h"
#include "core/memory.h"

#include <math.h>
#include <string.h>
#include <float.h>
#include <stdint.h>
#include <stdlib.h>

/* ── Result allocation helper ───────────────────────────────────────────────
 * Allocate node_ids[N] + scores[N] into `out`, filling node_ids from ctx and
 * zeroing scores. Returns 0 on success, -1 on alloc failure (out left zeroed,
 * partial allocations freed). Assumes out already memset to 0 and N > 0. */
static int scores_alloc(const GV_GAContext *ctx, size_t N, GV_GraphNodeScores *out) {
    out->node_ids = (uint64_t *)gv_alloc(N * sizeof(uint64_t));
    out->scores   = (double *)gv_calloc(N, sizeof(double));
    if (!out->node_ids || !out->scores) {
        gv_free(out->node_ids);
        gv_free(out->scores);
        out->node_ids = NULL;
        out->scores = NULL;
        return -1;
    }
    for (size_t i = 0; i < N; i++) out->node_ids[i] = gv_ga_id(ctx, i);
    out->count = N;
    return 0;
}

/* Clamp a (possibly negative) edge weight to a non-negative double. */
static double clamp_w(float w) {
    double d = (double)w;
    return d < 0.0 ? 0.0 : d;
}

/* ── Binary min-heap keyed by dense index, ordered by a double key ──────────── */

typedef struct {
    size_t *idx;    /* heap array of dense node indices */
    double *key;    /* key[node] = current priority of that node */
    size_t *pos;    /* pos[node] = position in idx[], or (size_t)-1 if absent */
    size_t  size;
    size_t  cap;
} MinHeap;

static int mh_init(MinHeap *h, size_t n) {
    h->size = 0;
    h->cap = n;
    h->idx = NULL;
    h->key = NULL;
    h->pos = NULL;
    if (n == 0) return 0;
    h->idx = (size_t *)gv_alloc(n * sizeof(size_t));
    h->key = (double *)gv_alloc(n * sizeof(double));
    h->pos = (size_t *)gv_alloc(n * sizeof(size_t));
    if (!h->idx || !h->key || !h->pos) {
        gv_free(h->idx); gv_free(h->key); gv_free(h->pos);
        h->idx = NULL; h->key = NULL; h->pos = NULL;
        return -1;
    }
    for (size_t i = 0; i < n; i++) h->pos[i] = (size_t)-1;
    return 0;
}

static void mh_free(MinHeap *h) {
    if (!h) return;
    gv_free(h->idx); gv_free(h->key); gv_free(h->pos);
    h->idx = NULL; h->key = NULL; h->pos = NULL;
    h->size = 0; h->cap = 0;
}

static void mh_swap(MinHeap *h, size_t a, size_t b) {
    size_t ta = h->idx[a], tb = h->idx[b];
    h->idx[a] = tb; h->idx[b] = ta;
    h->pos[tb] = a; h->pos[ta] = b;
}

static void mh_sift_up(MinHeap *h, size_t i) {
    while (i > 0) {
        size_t parent = (i - 1) / 2;
        if (h->key[h->idx[parent]] <= h->key[h->idx[i]]) break;
        mh_swap(h, parent, i);
        i = parent;
    }
}

static void mh_sift_down(MinHeap *h, size_t i) {
    for (;;) {
        size_t l = 2 * i + 1, r = 2 * i + 2, smallest = i;
        if (l < h->size && h->key[h->idx[l]] < h->key[h->idx[smallest]]) smallest = l;
        if (r < h->size && h->key[h->idx[r]] < h->key[h->idx[smallest]]) smallest = r;
        if (smallest == i) break;
        mh_swap(h, i, smallest);
        i = smallest;
    }
}

/* Insert or decrease-key: set key[node]=k and (re)position node in the heap. */
static void mh_push(MinHeap *h, size_t node, double k) {
    h->key[node] = k;
    if (h->pos[node] == (size_t)-1) {
        h->idx[h->size] = node;
        h->pos[node] = h->size;
        h->size++;
        mh_sift_up(h, h->pos[node]);
    } else {
        mh_sift_up(h, h->pos[node]);
        mh_sift_down(h, h->pos[node]);
    }
}

static size_t mh_pop(MinHeap *h) {
    size_t top = h->idx[0];
    h->pos[top] = (size_t)-1;
    h->size--;
    if (h->size > 0) {
        h->idx[0] = h->idx[h->size];
        h->pos[h->idx[0]] = 0;
        mh_sift_down(h, 0);
    }
    return top;
}

/* ── Adjacency iteration ─────────────────────────────────────────────────────
 * Visitor abstraction over a node's neighbours. When `directed` is set we walk
 * out_edges only; otherwise we walk out_edges + in_edges (the undirected union,
 * counting parallel edges as separate transitions — fine for these algorithms).
 * The callback receives the neighbour's dense index and the clamped weight. */

typedef void (*neighbor_fn)(size_t nb_idx, double w, void *user);

static void for_each_neighbor(const GV_GraphDB *g, const GV_GAContext *ctx,
                              uint64_t node_id, int directed,
                              neighbor_fn cb, void *user) {
    const GV_GraphNode *n = graph_get_node(g, node_id);
    if (!n) return;
    for (size_t k = 0; k < n->out_count; k++) {
        size_t nb = gv_ga_index(ctx, n->out_edges[k].neighbor_id);
        if (nb == (size_t)-1) continue;
        const GV_GraphEdge *e = graph_get_edge(g, n->out_edges[k].edge_id);
        double w = e ? clamp_w(e->weight) : 1.0;
        cb(nb, w, user);
    }
    if (!directed) {
        for (size_t k = 0; k < n->in_count; k++) {
            size_t nb = gv_ga_index(ctx, n->in_edges[k].neighbor_id);
            if (nb == (size_t)-1) continue;
            const GV_GraphEdge *e = graph_get_edge(g, n->in_edges[k].edge_id);
            double w = e ? clamp_w(e->weight) : 1.0;
            cb(nb, w, user);
        }
    }
}

/* ── Single-source shortest path (BFS for unweighted, Dijkstra for weighted) ──
 * Fills dist[N] with distances from `src` (INFINITY if unreachable). Returns
 * 0 on success, -1 on allocation failure. Uses the heap only when weighted. */

typedef struct {
    const GV_GraphDB *g;
    const GV_GAContext *ctx;
    /* Dijkstra relaxation context */
    double *dist;
    MinHeap *heap;
    size_t   u;
} SSSPCtx;

static void sssp_relax_cb(size_t nb, double w, void *user) {
    SSSPCtx *s = (SSSPCtx *)user;
    double nd = s->dist[s->u] + w;
    if (nd < s->dist[nb]) {
        s->dist[nb] = nd;
        mh_push(s->heap, nb, nd);
    }
}

static int sssp(const GV_GraphDB *g, const GV_GAContext *ctx, size_t N,
                size_t src, int weighted, int directed, double *dist) {
    for (size_t i = 0; i < N; i++) dist[i] = INFINITY;
    dist[src] = 0.0;

    if (!weighted) {
        /* BFS: a simple ring queue over dense indices. */
        size_t *q = (size_t *)gv_alloc(N * sizeof(size_t));
        if (!q) return -1;
        size_t head = 0, tail = 0;
        q[tail++] = src;
        while (head < tail) {
            size_t u = q[head++];
            const GV_GraphNode *n = graph_get_node(g, gv_ga_id(ctx, u));
            if (!n) continue;
            double nd = dist[u] + 1.0;
            for (size_t k = 0; k < n->out_count; k++) {
                size_t nb = gv_ga_index(ctx, n->out_edges[k].neighbor_id);
                if (nb == (size_t)-1 || dist[nb] != INFINITY) continue;
                dist[nb] = nd;
                q[tail++] = nb;
            }
            if (!directed) {
                for (size_t k = 0; k < n->in_count; k++) {
                    size_t nb = gv_ga_index(ctx, n->in_edges[k].neighbor_id);
                    if (nb == (size_t)-1 || dist[nb] != INFINITY) continue;
                    dist[nb] = nd;
                    q[tail++] = nb;
                }
            }
        }
        gv_free(q);
        return 0;
    }

    /* Weighted: Dijkstra with the binary min-heap. */
    MinHeap heap;
    if (mh_init(&heap, N) != 0) return -1;
    SSSPCtx s = { g, ctx, dist, &heap, 0 };
    mh_push(&heap, src, 0.0);
    while (heap.size > 0) {
        size_t u = mh_pop(&heap);
        s.u = u;
        for_each_neighbor(g, ctx, gv_ga_id(ctx, u), directed, sssp_relax_cb, &s);
    }
    mh_free(&heap);
    return 0;
}

/* ═══════════════════════════ Closeness ═════════════════════════════════════ */

int graph_closeness_centrality(const GV_GraphDB *g, int weighted, int directed,
                               GV_GraphNodeScores *out) {
    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    if (!g) return -1;

    GV_GAContext *ctx = gv_ga_build(g);
    if (!ctx) return -1;
    size_t N = gv_ga_count(ctx);
    if (N == 0) { gv_ga_free(ctx); return 0; }

    if (scores_alloc(ctx, N, out) != 0) { gv_ga_free(ctx); memset(out, 0, sizeof(*out)); return -1; }

    double *dist = (double *)gv_alloc(N * sizeof(double));
    if (!dist) { graph_node_scores_free(out); memset(out, 0, sizeof(*out)); gv_ga_free(ctx); return -1; }

    for (size_t i = 0; i < N; i++) {
        if (sssp(g, ctx, N, i, weighted, directed, dist) != 0) {
            gv_free(dist); graph_node_scores_free(out); memset(out, 0, sizeof(*out));
            gv_ga_free(ctx); return -1;
        }
        double sum = 0.0;
        size_t reachable = 0;   /* includes self */
        for (size_t j = 0; j < N; j++) {
            if (dist[j] != INFINITY) { sum += dist[j]; reachable++; }
        }
        double score = 0.0;
        if (reachable > 1 && sum > 0.0) {
            score = (double)(reachable - 1) / sum;
            /* Wasserman-Faust: scale by the fraction of the graph reachable. */
            if (N > 1) score *= (double)(reachable - 1) / (double)(N - 1);
        }
        out->scores[i] = score;
    }

    gv_free(dist);
    gv_ga_free(ctx);
    return 0;
}

/* ═══════════════════════════ Harmonic ══════════════════════════════════════ */

int graph_harmonic_centrality(const GV_GraphDB *g, int weighted, int directed,
                              GV_GraphNodeScores *out) {
    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    if (!g) return -1;

    GV_GAContext *ctx = gv_ga_build(g);
    if (!ctx) return -1;
    size_t N = gv_ga_count(ctx);
    if (N == 0) { gv_ga_free(ctx); return 0; }

    if (scores_alloc(ctx, N, out) != 0) { gv_ga_free(ctx); memset(out, 0, sizeof(*out)); return -1; }

    double *dist = (double *)gv_alloc(N * sizeof(double));
    if (!dist) { graph_node_scores_free(out); memset(out, 0, sizeof(*out)); gv_ga_free(ctx); return -1; }

    for (size_t i = 0; i < N; i++) {
        if (sssp(g, ctx, N, i, weighted, directed, dist) != 0) {
            gv_free(dist); graph_node_scores_free(out); memset(out, 0, sizeof(*out));
            gv_ga_free(ctx); return -1;
        }
        double h = 0.0;
        for (size_t j = 0; j < N; j++) {
            if (j == i) continue;
            if (dist[j] != INFINITY && dist[j] > 0.0) h += 1.0 / dist[j];
        }
        out->scores[i] = h;
    }

    gv_free(dist);
    gv_ga_free(ctx);
    return 0;
}

/* ═══════════════════════════ Betweenness (Brandes) ═════════════════════════ */

/* Per-source accumulation state, reused across sources to avoid re-allocation. */
typedef struct {
    double *dist;    /* shortest distance from source */
    double *sigma;   /* number of shortest paths from source */
    double *delta;   /* dependency accumulation */
    size_t *order;   /* nodes in order of finalization (for reverse pass) */
    size_t  order_n;
    size_t **preds;  /* predecessor lists on shortest paths */
    size_t  *pred_n; /* predecessor count per node */
    size_t  *pred_cap;
} BrandesCtx;

static int bc_push_pred(BrandesCtx *b, size_t v, size_t u) {
    if (b->pred_n[v] == b->pred_cap[v]) {
        size_t nc = b->pred_cap[v] ? b->pred_cap[v] * 2 : 4;
        size_t *np = (size_t *)gv_realloc(b->preds[v], nc * sizeof(size_t));
        if (!np) return -1;
        b->preds[v] = np;
        b->pred_cap[v] = nc;
    }
    b->preds[v][b->pred_n[v]++] = u;
    return 0;
}

int graph_betweenness_centrality(const GV_GraphDB *g, int weighted, int directed,
                                 GV_GraphNodeScores *out) {
    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    if (!g) return -1;

    GV_GAContext *ctx = gv_ga_build(g);
    if (!ctx) return -1;
    size_t N = gv_ga_count(ctx);
    if (N == 0) { gv_ga_free(ctx); return 0; }

    if (scores_alloc(ctx, N, out) != 0) { gv_ga_free(ctx); memset(out, 0, sizeof(*out)); return -1; }

    /* Allocate all reusable per-source buffers up front. */
    BrandesCtx b;
    memset(&b, 0, sizeof(b));
    MinHeap heap;
    memset(&heap, 0, sizeof(heap));
    int ok = 1;

    b.dist     = (double *)gv_alloc(N * sizeof(double));
    b.sigma    = (double *)gv_alloc(N * sizeof(double));
    b.delta    = (double *)gv_alloc(N * sizeof(double));
    b.order    = (size_t *)gv_alloc(N * sizeof(size_t));
    b.preds    = (size_t **)gv_calloc(N, sizeof(size_t *));
    b.pred_n   = (size_t *)gv_calloc(N, sizeof(size_t));
    b.pred_cap = (size_t *)gv_calloc(N, sizeof(size_t));
    if (!b.dist || !b.sigma || !b.delta || !b.order ||
        !b.preds || !b.pred_n || !b.pred_cap) ok = 0;
    if (ok && weighted && mh_init(&heap, N) != 0) ok = 0;

    for (size_t s = 0; ok && s < N; s++) {
        for (size_t i = 0; i < N; i++) {
            b.dist[i] = INFINITY;
            b.sigma[i] = 0.0;
            b.delta[i] = 0.0;
            b.pred_n[i] = 0;
        }
        b.dist[s] = 0.0;
        b.sigma[s] = 1.0;
        b.order_n = 0;

        if (!weighted) {
            /* BFS accumulation. */
            size_t *q = (size_t *)gv_alloc(N * sizeof(size_t));
            if (!q) { ok = 0; break; }
            size_t head = 0, tail = 0;
            q[tail++] = s;
            while (head < tail) {
                size_t u = q[head++];
                b.order[b.order_n++] = u;
                const GV_GraphNode *n = graph_get_node(g, gv_ga_id(ctx, u));
                if (!n) continue;
                /* iterate out then (if undirected) in neighbours */
                for (int pass = 0; pass < (directed ? 1 : 2); pass++) {
                    size_t cnt = pass == 0 ? n->out_count : n->in_count;
                    const GV_GraphEdgeRef *refs = pass == 0 ? n->out_edges : n->in_edges;
                    for (size_t k = 0; k < cnt; k++) {
                        size_t w = gv_ga_index(ctx, refs[k].neighbor_id);
                        if (w == (size_t)-1) continue;
                        if (b.dist[w] == INFINITY) {
                            b.dist[w] = b.dist[u] + 1.0;
                            q[tail++] = w;
                        }
                        if (b.dist[w] == b.dist[u] + 1.0) {
                            b.sigma[w] += b.sigma[u];
                            if (bc_push_pred(&b, w, u) != 0) { ok = 0; }
                        }
                    }
                }
                if (!ok) break;
            }
            gv_free(q);
            if (!ok) break;
        } else {
            /* Dijkstra-based Brandes: finalize nodes in increasing distance. */
            uint8_t *done = (uint8_t *)gv_calloc(N, sizeof(uint8_t));
            if (!done) { ok = 0; break; }
            mh_push(&heap, s, 0.0);
            while (heap.size > 0) {
                size_t u = mh_pop(&heap);
                if (done[u]) continue;
                done[u] = 1;
                b.order[b.order_n++] = u;
                const GV_GraphNode *n = graph_get_node(g, gv_ga_id(ctx, u));
                if (!n) continue;
                for (int pass = 0; pass < (directed ? 1 : 2); pass++) {
                    size_t cnt = pass == 0 ? n->out_count : n->in_count;
                    const GV_GraphEdgeRef *refs = pass == 0 ? n->out_edges : n->in_edges;
                    for (size_t k = 0; k < cnt; k++) {
                        size_t w = gv_ga_index(ctx, refs[k].neighbor_id);
                        if (w == (size_t)-1 || done[w]) continue;
                        const GV_GraphEdge *e = graph_get_edge(g, refs[k].edge_id);
                        double weight = e ? clamp_w(e->weight) : 1.0;
                        double nd = b.dist[u] + weight;
                        if (nd < b.dist[w]) {
                            b.dist[w] = nd;
                            b.sigma[w] = b.sigma[u];
                            b.pred_n[w] = 0;
                            if (bc_push_pred(&b, w, u) != 0) { ok = 0; break; }
                            mh_push(&heap, w, nd);
                        } else if (nd == b.dist[w]) {
                            b.sigma[w] += b.sigma[u];
                            if (bc_push_pred(&b, w, u) != 0) { ok = 0; break; }
                        }
                    }
                    if (!ok) break;
                }
                if (!ok) break;
            }
            /* drain heap on failure so pos[] is clean for next source */
            while (heap.size > 0) mh_pop(&heap);
            gv_free(done);
            if (!ok) break;
        }

        /* Reverse accumulation of dependencies. */
        for (size_t oi = b.order_n; oi-- > 0; ) {
            size_t w = b.order[oi];
            for (size_t p = 0; p < b.pred_n[w]; p++) {
                size_t v = b.preds[w][p];
                if (b.sigma[w] > 0.0)
                    b.delta[v] += (b.sigma[v] / b.sigma[w]) * (1.0 + b.delta[w]);
            }
            if (w != s) out->scores[w] += b.delta[w];
        }
    }

    if (weighted) {
        /* heap may hold stale pos[] entries only if we broke early; clear fully */
        while (heap.size > 0) mh_pop(&heap);
        mh_free(&heap);
    }
    if (b.preds) for (size_t i = 0; i < N; i++) gv_free(b.preds[i]);
    gv_free(b.dist); gv_free(b.sigma); gv_free(b.delta); gv_free(b.order);
    gv_free(b.preds); gv_free(b.pred_n); gv_free(b.pred_cap);

    if (!ok) { graph_node_scores_free(out); memset(out, 0, sizeof(*out)); gv_ga_free(ctx); return -1; }
    gv_ga_free(ctx);
    return 0;
}

/* ═══════════════════════════ Eigenvector ═══════════════════════════════════ */

int graph_eigenvector_centrality(const GV_GraphDB *g, size_t max_iters, double tol,
                                 GV_GraphNodeScores *out) {
    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    if (!g) return -1;

    if (max_iters == 0) max_iters = 100;
    if (tol <= 0.0) tol = 1e-6;

    GV_GAContext *ctx = gv_ga_build(g);
    if (!ctx) return -1;
    size_t N = gv_ga_count(ctx);
    if (N == 0) { gv_ga_free(ctx); return 0; }

    if (scores_alloc(ctx, N, out) != 0) { gv_ga_free(ctx); memset(out, 0, sizeof(*out)); return -1; }

    /* Build the weighted OUT adjacency. The existing semantics scatter x[u] to
     * u's out-neighbors, i.e. x_new = Outᵀ·x (column push over out-edges). */
    GV_CSR *Out = gv_csr_build(g, ctx, GV_CSR_OUT, 1);
    if (!Out) { graph_node_scores_free(out); memset(out, 0, sizeof(*out)); gv_ga_free(ctx); return -1; }

    double *x    = (double *)gv_alloc(N * sizeof(double));
    double *xnew = (double *)gv_alloc(N * sizeof(double));
    if (!x || !xnew) {
        gv_free(x); gv_free(xnew); gv_csr_free(Out);
        graph_node_scores_free(out); memset(out, 0, sizeof(*out)); gv_ga_free(ctx); return -1;
    }

    double init = 1.0 / sqrt((double)N);
    for (size_t i = 0; i < N; i++) x[i] = init;

    for (size_t it = 0; it < max_iters; it++) {
        /* xnew = Outᵀ·x: for every edge u->w add weight*x[u] into xnew[w]. */
        for (size_t i = 0; i < N; i++) xnew[i] = 0.0;
        gv_csr_spmv_transpose(Out, x, xnew);
        double norm = 0.0;
        for (size_t i = 0; i < N; i++) norm += xnew[i] * xnew[i];
        norm = sqrt(norm);
        if (norm == 0.0) {
            /* No incoming influence anywhere: keep the uniform vector. */
            break;
        }
        double diff = 0.0;
        for (size_t i = 0; i < N; i++) {
            xnew[i] /= norm;
            double d = xnew[i] - x[i];
            diff += d * d;
        }
        double *tmp = x; x = xnew; xnew = tmp;
        if (sqrt(diff) < tol) break;
    }

    for (size_t i = 0; i < N; i++) out->scores[i] = x[i];

    gv_free(x); gv_free(xnew); gv_csr_free(Out);
    gv_ga_free(ctx);
    return 0;
}

/* ═══════════════════════════ Degree ════════════════════════════════════════ */

int graph_degree_centrality(const GV_GraphDB *g, int mode, GV_GraphNodeScores *out) {
    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    if (!g) return -1;

    GV_GAContext *ctx = gv_ga_build(g);
    if (!ctx) return -1;
    size_t N = gv_ga_count(ctx);
    if (N == 0) { gv_ga_free(ctx); return 0; }

    if (scores_alloc(ctx, N, out) != 0) { gv_ga_free(ctx); memset(out, 0, sizeof(*out)); return -1; }

    double denom = (N > 1) ? (double)(N - 1) : 1.0;
    for (size_t i = 0; i < N; i++) {
        const GV_GraphNode *n = graph_get_node(g, gv_ga_id(ctx, i));
        double deg = 0.0;
        if (n) {
            switch (mode) {
                case 1:  deg = (double)n->in_count; break;
                case 2:  deg = (double)n->out_count; break;
                default: deg = (double)(n->in_count + n->out_count); break;
            }
        }
        out->scores[i] = deg / denom;
    }

    gv_ga_free(ctx);
    return 0;
}

/* ═══════════════════ PageRank family (shared power iteration) ══════════════ */

/* Shared PageRank-style power iteration.
 *   teleport[i]      — per-node teleport (restart) probability distribution,
 *                      must sum to 1. NULL => uniform 1/N.
 *   article_rank     — when non-zero, divide each node's outgoing contribution
 *                      by (out_degree + average_out_degree) instead of out_degree.
 * Writes the stationary distribution into out->scores. */
static int pagerank_core(const GV_GAContext *ctx, const GV_GraphDB *g, size_t N,
                         size_t iters, double damping, const double *teleport,
                         int article_rank, GV_GraphNodeScores *out) {
    /* IN adjacency (boolean): row j holds j's in-neighbors, so In·contrib pulls
     * each node's incoming rank contributions in one SpMV. */
    GV_CSR *In = gv_csr_build(g, ctx, GV_CSR_IN, 0);
    /* OUT adjacency (boolean): row_ptr gives per-node out-degree (edge count). */
    GV_CSR *Out = gv_csr_build(g, ctx, GV_CSR_OUT, 0);
    if (!In || !Out) { gv_csr_free(In); gv_csr_free(Out); return -1; }

    double *rank    = (double *)gv_alloc(N * sizeof(double));
    double *next    = (double *)gv_alloc(N * sizeof(double));
    double *contrib = (double *)gv_alloc(N * sizeof(double));
    double *tmpvec  = (double *)gv_alloc(N * sizeof(double));
    double *divisor = (double *)gv_alloc(N * sizeof(double)); /* per-node contrib divisor */
    if (!rank || !next || !contrib || !tmpvec || !divisor) {
        gv_free(rank); gv_free(next); gv_free(contrib); gv_free(tmpvec); gv_free(divisor);
        gv_csr_free(In); gv_csr_free(Out);
        return -1;
    }

    /* Per-node out-degree from the OUT matrix's row_ptr, and the total. */
    double total_out = 0.0;
    for (size_t i = 0; i < N; i++) {
        double odeg = (double)(Out->row_ptr[i + 1] - Out->row_ptr[i]);
        divisor[i] = odeg;
        total_out += odeg;
    }

    /* Average out-degree over ALL nodes (ArticleRank definition). */
    double avg_out = total_out / (double)N;

    /* Final divisor: article-rank shifts by the average out-degree. A zero
     * out-degree still marks a dangling node (divisor stays 0). */
    for (size_t i = 0; i < N; i++) {
        if (Out->row_ptr[i + 1] - Out->row_ptr[i] == 0) {
            divisor[i] = 0.0; /* dangling */
        } else if (article_rank) {
            divisor[i] = divisor[i] + avg_out;
        }
    }

    for (size_t i = 0; i < N; i++) rank[i] = teleport ? teleport[i] : 1.0 / (double)N;

    for (size_t it = 0; it < iters; it++) {
        /* contrib[i] = rank[i]/divisor[i] (0 if dangling); dangling mass D. */
        double dangling = 0.0;
        for (size_t i = 0; i < N; i++) {
            if (divisor[i] > 0.0) {
                contrib[i] = rank[i] / divisor[i];
            } else {
                contrib[i] = 0.0;
                dangling += rank[i];
            }
        }

        /* tmpvec = In · contrib: each node's incoming rank contribution. */
        gv_csr_spmv(In, contrib, tmpvec);

        for (size_t j = 0; j < N; j++) {
            double tp = teleport ? teleport[j] : 1.0 / (double)N;
            next[j] = (1.0 - damping) * tp
                    + damping * (tmpvec[j] + dangling / (double)N);
        }

        double *tmp = rank; rank = next; next = tmp;
    }

    for (size_t i = 0; i < N; i++) out->scores[i] = rank[i];

    gv_free(rank); gv_free(next); gv_free(contrib); gv_free(tmpvec); gv_free(divisor);
    gv_csr_free(In); gv_csr_free(Out);
    return 0;
}

int graph_pagerank_all(const GV_GraphDB *g, size_t iters, double damping,
                       GV_GraphNodeScores *out) {
    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    if (!g) return -1;

    if (iters == 0) iters = 100;
    if (damping <= 0.0) damping = 0.85;

    GV_GAContext *ctx = gv_ga_build(g);
    if (!ctx) return -1;
    size_t N = gv_ga_count(ctx);
    if (N == 0) { gv_ga_free(ctx); return 0; }

    if (scores_alloc(ctx, N, out) != 0) { gv_ga_free(ctx); memset(out, 0, sizeof(*out)); return -1; }

    if (pagerank_core(ctx, g, N, iters, damping, NULL, 0, out) != 0) {
        graph_node_scores_free(out); memset(out, 0, sizeof(*out)); gv_ga_free(ctx); return -1;
    }
    gv_ga_free(ctx);
    return 0;
}

int graph_personalized_pagerank(const GV_GraphDB *g,
                                const uint64_t *sources, size_t num_sources,
                                size_t iters, double damping,
                                GV_GraphNodeScores *out) {
    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    if (!g) return -1;

    if (iters == 0) iters = 100;
    if (damping <= 0.0) damping = 0.85;

    GV_GAContext *ctx = gv_ga_build(g);
    if (!ctx) return -1;
    size_t N = gv_ga_count(ctx);
    if (N == 0) { gv_ga_free(ctx); return 0; }

    if (scores_alloc(ctx, N, out) != 0) { gv_ga_free(ctx); memset(out, 0, sizeof(*out)); return -1; }

    double *teleport = (double *)gv_calloc(N, sizeof(double));
    if (!teleport) { graph_node_scores_free(out); memset(out, 0, sizeof(*out)); gv_ga_free(ctx); return -1; }

    /* Build the teleport distribution: uniform over valid `sources`, else uniform-all. */
    size_t valid = 0;
    if (sources && num_sources > 0) {
        for (size_t i = 0; i < num_sources; i++) {
            size_t idx = gv_ga_index(ctx, sources[i]);
            if (idx == (size_t)-1) continue;
            if (teleport[idx] == 0.0) valid++;
            teleport[idx] = 1.0; /* mark; normalized below */
        }
    }
    if (valid == 0) {
        for (size_t i = 0; i < N; i++) teleport[i] = 1.0 / (double)N;
    } else {
        for (size_t i = 0; i < N; i++) teleport[i] = (teleport[i] > 0.0) ? 1.0 / (double)valid : 0.0;
    }

    int rc = pagerank_core(ctx, g, N, iters, damping, teleport, 0, out);
    gv_free(teleport);
    if (rc != 0) { graph_node_scores_free(out); memset(out, 0, sizeof(*out)); gv_ga_free(ctx); return -1; }
    gv_ga_free(ctx);
    return 0;
}

int graph_article_rank(const GV_GraphDB *g, size_t iters, double damping,
                       GV_GraphNodeScores *out) {
    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    if (!g) return -1;

    if (iters == 0) iters = 100;
    if (damping <= 0.0) damping = 0.85;

    GV_GAContext *ctx = gv_ga_build(g);
    if (!ctx) return -1;
    size_t N = gv_ga_count(ctx);
    if (N == 0) { gv_ga_free(ctx); return 0; }

    if (scores_alloc(ctx, N, out) != 0) { gv_ga_free(ctx); memset(out, 0, sizeof(*out)); return -1; }

    if (pagerank_core(ctx, g, N, iters, damping, NULL, 1, out) != 0) {
        graph_node_scores_free(out); memset(out, 0, sizeof(*out)); gv_ga_free(ctx); return -1;
    }
    gv_ga_free(ctx);
    return 0;
}

/* ═══════════════════════════ HITS ══════════════════════════════════════════ */

int graph_hits(const GV_GraphDB *g, size_t iters, double tol,
               GV_GraphNodeScores *hubs, GV_GraphNodeScores *authorities) {
    if (!hubs && !authorities) return -1;
    if (hubs) memset(hubs, 0, sizeof(*hubs));
    if (authorities) memset(authorities, 0, sizeof(*authorities));
    if (!g) return -1;

    if (iters == 0) iters = 100;
    if (tol <= 0.0) tol = 1e-8;

    GV_GAContext *ctx = gv_ga_build(g);
    if (!ctx) return -1;
    size_t N = gv_ga_count(ctx);
    if (N == 0) { gv_ga_free(ctx); return 0; }

    if (hubs && scores_alloc(ctx, N, hubs) != 0) {
        if (authorities) memset(authorities, 0, sizeof(*authorities));
        gv_ga_free(ctx); memset(hubs, 0, sizeof(*hubs)); return -1;
    }
    if (authorities && scores_alloc(ctx, N, authorities) != 0) {
        if (hubs) { graph_node_scores_free(hubs); memset(hubs, 0, sizeof(*hubs)); }
        gv_ga_free(ctx); memset(authorities, 0, sizeof(*authorities)); return -1;
    }

    /* OUT adjacency (boolean): edge u->v. Authorities are pulled from hubs that
     * point to them (auth = Outᵀ·hub); hubs collect their authorities (hub = Out·auth). */
    GV_CSR *Out  = gv_csr_build(g, ctx, GV_CSR_OUT, 0);
    double *hub  = (double *)gv_alloc(N * sizeof(double));
    double *auth = (double *)gv_alloc(N * sizeof(double));
    double *prev = (double *)gv_alloc(N * sizeof(double)); /* previous hub vector */
    if (!Out || !hub || !auth || !prev) {
        gv_csr_free(Out); gv_free(hub); gv_free(auth); gv_free(prev);
        if (hubs) { graph_node_scores_free(hubs); memset(hubs, 0, sizeof(*hubs)); }
        if (authorities) { graph_node_scores_free(authorities); memset(authorities, 0, sizeof(*authorities)); }
        gv_ga_free(ctx); return -1;
    }

    double init = 1.0 / sqrt((double)N);
    for (size_t i = 0; i < N; i++) { hub[i] = init; auth[i] = init; }

    for (size_t it = 0; it < iters; it++) {
        for (size_t i = 0; i < N; i++) prev[i] = hub[i];

        /* auth = Outᵀ·hub: a node's authority is the sum of hub scores pointing in. */
        for (size_t i = 0; i < N; i++) auth[i] = 0.0;
        gv_csr_spmv_transpose(Out, hub, auth);
        double anorm = 0.0;
        for (size_t i = 0; i < N; i++) anorm += auth[i] * auth[i];
        anorm = sqrt(anorm);
        if (anorm > 0.0) for (size_t i = 0; i < N; i++) auth[i] /= anorm;

        /* hub = Out·auth: a node's hub score is the sum of authorities it points to. */
        gv_csr_spmv(Out, auth, hub);
        double hnorm = 0.0;
        for (size_t i = 0; i < N; i++) hnorm += hub[i] * hub[i];
        hnorm = sqrt(hnorm);
        if (hnorm > 0.0) for (size_t i = 0; i < N; i++) hub[i] /= hnorm;

        /* Converged when neither vector carries any mass (empty edge set) or the
         * hub vector stops moving. */
        if (anorm == 0.0 && hnorm == 0.0) break;
        double diff = 0.0;
        for (size_t i = 0; i < N; i++) {
            double d = hub[i] - prev[i];
            diff += d * d;
        }
        if (sqrt(diff) < tol) break;
    }

    if (hubs) for (size_t i = 0; i < N; i++) hubs->scores[i] = hub[i];
    if (authorities) for (size_t i = 0; i < N; i++) authorities->scores[i] = auth[i];

    gv_csr_free(Out); gv_free(hub); gv_free(auth); gv_free(prev);
    gv_ga_free(ctx);
    return 0;
}
