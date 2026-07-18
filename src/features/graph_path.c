/**
 * graph_path.c — path / flow algorithms for the GigaVector property graph.
 *
 * Implements the path & traversal portion of graph_algos.h:
 *   - graph_single_source_shortest_path (Dijkstra / BFS)
 *   - graph_astar                       (A* with an admissible heuristic)
 *   - graph_detect_cycle                (iterative white/gray/black DFS)
 *   - graph_minimum_spanning_forest     (Kruskal + union-find)
 *   - graph_max_flow                    (Edmonds-Karp)
 *
 * All algorithms are read-only w.r.t. the graph, use the shared dense node
 * index context (gv_ga_*), and allocate their results with gv_alloc/gv_calloc
 * so callers can free them with the matching *_free helpers.
 */
#include "features/graph_algos.h"
#include "core/memory.h"

#include <math.h>
#include <float.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

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

/* Insert node `node` with priority `key`, or decrease its key if already in. */
static void mh_push(MinHeap *h, size_t node, double key) {
    if (h->pos[node] != (size_t)-1) {
        /* already present: update if this key is smaller */
        if (key < h->key[node]) {
            h->key[node] = key;
            mh_sift_up(h, h->pos[node]);
        }
        return;
    }
    h->key[node] = key;
    h->idx[h->size] = node;
    h->pos[node] = h->size;
    h->size++;
    mh_sift_up(h, h->size - 1);
}

/* Pop the min node; returns (size_t)-1 when empty. */
static size_t mh_pop(MinHeap *h) {
    if (h->size == 0) return (size_t)-1;
    size_t top = h->idx[0];
    h->size--;
    if (h->size > 0) {
        h->idx[0] = h->idx[h->size];
        h->pos[h->idx[0]] = 0;
        mh_sift_down(h, 0);
    }
    h->pos[top] = (size_t)-1;
    return top;
}

/* ── Single-source shortest path (Dijkstra weighted / BFS unweighted) ───────── */

int graph_single_source_shortest_path(const GV_GraphDB *g, uint64_t source,
                                      int weighted, int directed,
                                      GV_GraphNodeScores *out) {
    if (!g || !out) return -1;
    out->node_ids = NULL; out->scores = NULL; out->count = 0;

    GV_GAContext *ctx = gv_ga_build(g);
    if (!ctx) return -1;
    size_t N = gv_ga_count(ctx);
    if (N == 0) { gv_ga_free(ctx); return 0; }  /* empty graph: empty result */

    uint64_t *node_ids = (uint64_t *)gv_alloc(N * sizeof(uint64_t));
    double   *scores   = (double *)gv_alloc(N * sizeof(double));
    if (!node_ids || !scores) {
        gv_free(node_ids); gv_free(scores);
        gv_ga_free(ctx);
        return -1;
    }
    for (size_t i = 0; i < N; i++) {
        node_ids[i] = gv_ga_id(ctx, i);
        scores[i] = INFINITY;
    }

    size_t src = gv_ga_index(ctx, source);
    if (src == (size_t)-1) {
        /* source not present: everything unreachable (all INFINITY) */
        out->node_ids = node_ids; out->scores = scores; out->count = N;
        gv_ga_free(ctx);
        return 0;
    }
    scores[src] = 0.0;

    if (weighted) {
        MinHeap heap;
        if (mh_init(&heap, N) != 0) {
            gv_free(node_ids); gv_free(scores);
            gv_ga_free(ctx);
            return -1;
        }
        mh_push(&heap, src, 0.0);
        size_t u;
        while ((u = mh_pop(&heap)) != (size_t)-1) {
            double du = scores[u];
            const GV_GraphNode *nu = graph_get_node(g, gv_ga_id(ctx, u));
            if (!nu) continue;
            /* forward direction: out_edges */
            for (size_t e = 0; e < nu->out_count; e++) {
                const GV_GraphEdge *edge = graph_get_edge(g, nu->out_edges[e].edge_id);
                double w = edge ? (double)edge->weight : 1.0;
                if (w < 0.0) w = 0.0;
                size_t v = gv_ga_index(ctx, nu->out_edges[e].neighbor_id);
                if (v == (size_t)-1) continue;
                double nd = du + w;
                if (nd < scores[v]) { scores[v] = nd; mh_push(&heap, v, nd); }
            }
            if (!directed) {
                /* undirected: also relax in_edges */
                for (size_t e = 0; e < nu->in_count; e++) {
                    const GV_GraphEdge *edge = graph_get_edge(g, nu->in_edges[e].edge_id);
                    double w = edge ? (double)edge->weight : 1.0;
                    if (w < 0.0) w = 0.0;
                    size_t v = gv_ga_index(ctx, nu->in_edges[e].neighbor_id);
                    if (v == (size_t)-1) continue;
                    double nd = du + w;
                    if (nd < scores[v]) { scores[v] = nd; mh_push(&heap, v, nd); }
                }
            }
        }
        mh_free(&heap);
    } else {
        /* BFS: hop-count distances */
        size_t *queue = (size_t *)gv_alloc(N * sizeof(size_t));
        if (!queue) {
            gv_free(node_ids); gv_free(scores);
            gv_ga_free(ctx);
            return -1;
        }
        size_t head = 0, tail = 0;
        queue[tail++] = src;
        while (head < tail) {
            size_t u = queue[head++];
            double du = scores[u];
            const GV_GraphNode *nu = graph_get_node(g, gv_ga_id(ctx, u));
            if (!nu) continue;
            for (size_t e = 0; e < nu->out_count; e++) {
                size_t v = gv_ga_index(ctx, nu->out_edges[e].neighbor_id);
                if (v == (size_t)-1) continue;
                if (scores[v] == INFINITY) { scores[v] = du + 1.0; queue[tail++] = v; }
            }
            if (!directed) {
                for (size_t e = 0; e < nu->in_count; e++) {
                    size_t v = gv_ga_index(ctx, nu->in_edges[e].neighbor_id);
                    if (v == (size_t)-1) continue;
                    if (scores[v] == INFINITY) { scores[v] = du + 1.0; queue[tail++] = v; }
                }
            }
        }
        gv_free(queue);
    }

    out->node_ids = node_ids;
    out->scores = scores;
    out->count = N;
    gv_ga_free(ctx);
    return 0;
}

/* ── A* (Dijkstra when heuristic == NULL) over out_edges ────────────────────── */

int graph_astar(const GV_GraphDB *g, uint64_t from, uint64_t to,
                GV_GraphHeuristic heuristic, void *user, GV_GraphPath *path) {
    if (path) {
        path->node_ids = NULL; path->edge_ids = NULL;
        path->length = 0; path->total_weight = 0.0f;
    }
    if (!g || !path) return -1;

    GV_GAContext *ctx = gv_ga_build(g);
    if (!ctx) return -1;
    size_t N = gv_ga_count(ctx);
    if (N == 0) { gv_ga_free(ctx); return -1; }

    size_t sidx = gv_ga_index(ctx, from);
    size_t tidx = gv_ga_index(ctx, to);
    if (sidx == (size_t)-1 || tidx == (size_t)-1) { gv_ga_free(ctx); return -1; }

    double   *gscore  = (double *)gv_alloc(N * sizeof(double));
    size_t   *pred    = (size_t *)gv_alloc(N * sizeof(size_t));
    uint64_t *predE   = (uint64_t *)gv_alloc(N * sizeof(uint64_t));
    uint8_t  *closed  = (uint8_t *)gv_calloc(N, sizeof(uint8_t));
    if (!gscore || !pred || !predE || !closed) {
        gv_free(gscore); gv_free(pred); gv_free(predE); gv_free(closed);
        gv_ga_free(ctx);
        return -1;
    }
    for (size_t i = 0; i < N; i++) { gscore[i] = INFINITY; pred[i] = (size_t)-1; predE[i] = 0; }

    MinHeap heap;
    if (mh_init(&heap, N) != 0) {
        gv_free(gscore); gv_free(pred); gv_free(predE); gv_free(closed);
        gv_ga_free(ctx);
        return -1;
    }

    gscore[sidx] = 0.0;
    double h0 = heuristic ? heuristic(from, to, user) : 0.0;
    if (!(h0 >= 0.0)) h0 = 0.0;
    mh_push(&heap, sidx, gscore[sidx] + h0);

    int found = 0;
    size_t u;
    while ((u = mh_pop(&heap)) != (size_t)-1) {
        if (u == tidx) { found = 1; break; }
        if (closed[u]) continue;
        closed[u] = 1;
        double du = gscore[u];
        const GV_GraphNode *nu = graph_get_node(g, gv_ga_id(ctx, u));
        if (!nu) continue;
        for (size_t e = 0; e < nu->out_count; e++) {
            size_t v = gv_ga_index(ctx, nu->out_edges[e].neighbor_id);
            if (v == (size_t)-1 || closed[v]) continue;
            const GV_GraphEdge *edge = graph_get_edge(g, nu->out_edges[e].edge_id);
            double w = edge ? (double)edge->weight : 1.0;
            if (w < 0.0) w = 0.0;
            double tentative = du + w;
            if (tentative < gscore[v]) {
                gscore[v] = tentative;
                pred[v] = u;
                predE[v] = nu->out_edges[e].edge_id;
                double hv = heuristic ? heuristic(nu->out_edges[e].neighbor_id, to, user) : 0.0;
                if (!(hv >= 0.0)) hv = 0.0;
                mh_push(&heap, v, tentative + hv);
            }
        }
    }
    mh_free(&heap);

    if (!found) {
        gv_free(gscore); gv_free(pred); gv_free(predE); gv_free(closed);
        gv_ga_free(ctx);
        return -1;
    }

    /* Reconstruct path by walking predecessors backwards from target. */
    size_t hops = 0;
    for (size_t cur = tidx; cur != sidx; cur = pred[cur]) {
        hops++;
        if (pred[cur] == (size_t)-1) { /* shouldn't happen when found, guard anyway */
            gv_free(gscore); gv_free(pred); gv_free(predE); gv_free(closed);
            gv_ga_free(ctx);
            return -1;
        }
    }

    /* node_ids has hops+1 entries; edge_ids has hops entries. */
    uint64_t *node_ids = (uint64_t *)gv_alloc((hops + 1) * sizeof(uint64_t));
    uint64_t *edge_ids = hops ? (uint64_t *)gv_alloc(hops * sizeof(uint64_t)) : NULL;
    if (!node_ids || (hops && !edge_ids)) {
        gv_free(node_ids); gv_free(edge_ids);
        gv_free(gscore); gv_free(pred); gv_free(predE); gv_free(closed);
        gv_ga_free(ctx);
        return -1;
    }

    /* fill back-to-front */
    size_t cur = tidx;
    size_t ni = hops;         /* node write index */
    node_ids[ni] = gv_ga_id(ctx, cur);
    for (size_t k = hops; k > 0; k--) {
        edge_ids[k - 1] = predE[cur];
        cur = pred[cur];
        node_ids[k - 1] = gv_ga_id(ctx, cur);
    }

    path->node_ids = node_ids;
    path->edge_ids = edge_ids;
    path->length = hops;
    path->total_weight = (float)gscore[tidx];

    gv_free(gscore); gv_free(pred); gv_free(predE); gv_free(closed);
    gv_ga_free(ctx);
    return 0;
}

/* ── Directed cycle detection (iterative white/gray/black DFS) ──────────────── */

/* colors: 0 = white (unvisited), 1 = gray (on stack), 2 = black (done) */
int graph_detect_cycle(const GV_GraphDB *g, int *is_dag, GV_GraphPath *cycle_out) {
    if (cycle_out) {
        cycle_out->node_ids = NULL; cycle_out->edge_ids = NULL;
        cycle_out->length = 0; cycle_out->total_weight = 0.0f;
    }
    if (!g) return -1;

    GV_GAContext *ctx = gv_ga_build(g);
    if (!ctx) return -1;
    size_t N = gv_ga_count(ctx);
    if (N == 0) { if (is_dag) *is_dag = 1; gv_ga_free(ctx); return 0; }

    uint8_t *color = (uint8_t *)gv_calloc(N, sizeof(uint8_t));
    size_t  *parent = (size_t *)gv_alloc(N * sizeof(size_t));  /* dfs-tree parent */
    /* explicit DFS stack: node + iteration cursor over its out_edges */
    size_t  *st_node = (size_t *)gv_alloc(N * sizeof(size_t));
    size_t  *st_iter = (size_t *)gv_alloc(N * sizeof(size_t));
    if (!color || !parent || !st_node || !st_iter) {
        gv_free(color); gv_free(parent); gv_free(st_node); gv_free(st_iter);
        gv_ga_free(ctx);
        return -1;
    }
    for (size_t i = 0; i < N; i++) parent[i] = (size_t)-1;

    int acyclic = 1;
    size_t cyc_from = (size_t)-1, cyc_to = (size_t)-1;  /* back-edge from -> to */

    for (size_t start = 0; start < N && acyclic; start++) {
        if (color[start] != 0) continue;
        size_t sp = 0;                 /* stack pointer */
        st_node[sp] = start;
        st_iter[sp] = 0;
        color[start] = 1;              /* gray */
        sp++;

        while (sp > 0 && acyclic) {
            size_t u = st_node[sp - 1];
            const GV_GraphNode *nu = graph_get_node(g, gv_ga_id(ctx, u));
            size_t oc = nu ? nu->out_count : 0;

            if (st_iter[sp - 1] < oc) {
                size_t ei = st_iter[sp - 1]++;
                size_t v = gv_ga_index(ctx, nu->out_edges[ei].neighbor_id);
                if (v == (size_t)-1) continue;
                if (color[v] == 0) {
                    color[v] = 1;      /* gray */
                    parent[v] = u;
                    st_node[sp] = v;
                    st_iter[sp] = 0;
                    sp++;
                } else if (color[v] == 1) {
                    /* back edge to a gray node -> directed cycle u -> v */
                    acyclic = 0;
                    cyc_from = u;
                    cyc_to = v;
                }
                /* color[v]==2 (black): cross/forward edge, ignore */
            } else {
                color[u] = 2;          /* black: fully explored */
                sp--;
            }
        }
    }

    if (is_dag) *is_dag = acyclic;

    int rc = 0;
    if (!acyclic && cycle_out) {
        /* Reconstruct the cycle: v = cyc_to ... up parents ... to cyc_from, then close.
         * The cycle node list is: cyc_to, then the chain from cyc_from up to cyc_to
         * following parent pointers, reversed. We collect cyc_from's ancestor chain
         * until we reach cyc_to, giving the nodes of the cycle in path order. */
        size_t maxlen = N;
        size_t *chain = (size_t *)gv_alloc(maxlen * sizeof(size_t));
        if (!chain) { rc = -1; }
        else {
            size_t len = 0;
            size_t w = cyc_from;
            chain[len++] = w;                 /* start from the tail of the back edge */
            while (w != cyc_to && parent[w] != (size_t)-1 && len < maxlen) {
                w = parent[w];
                chain[len++] = w;
            }
            /* chain now holds cyc_from ... cyc_to (inclusive), from deepest up.
             * Reverse it so it reads cyc_to -> ... -> cyc_from (edge direction). */
            uint64_t *node_ids = (uint64_t *)gv_alloc(len * sizeof(uint64_t));
            if (!node_ids) { rc = -1; }
            else {
                for (size_t i = 0; i < len; i++)
                    node_ids[i] = gv_ga_id(ctx, chain[len - 1 - i]);
                cycle_out->node_ids = node_ids;
                cycle_out->edge_ids = NULL;
                cycle_out->length = len;      /* number of nodes in the cycle */
                cycle_out->total_weight = 0.0f;
            }
            gv_free(chain);
        }
    }

    gv_free(color); gv_free(parent); gv_free(st_node); gv_free(st_iter);
    gv_ga_free(ctx);
    return rc;
}

/* ── Minimum spanning forest (Kruskal + union-find over undirected edges) ───── */

typedef struct {
    uint64_t edge_id;
    double   weight;
    size_t   u, v;   /* dense endpoint indices */
} MSFEdge;

static int msf_cmp(const void *a, const void *b) {
    const MSFEdge *ea = (const MSFEdge *)a;
    const MSFEdge *eb = (const MSFEdge *)b;
    if (ea->weight < eb->weight) return -1;
    if (ea->weight > eb->weight) return 1;
    if (ea->edge_id < eb->edge_id) return -1;
    if (ea->edge_id > eb->edge_id) return 1;
    return 0;
}

static size_t uf_find(size_t *par, size_t x) {
    while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; }
    return x;
}

int graph_minimum_spanning_forest(const GV_GraphDB *g, GV_GraphEdgeSet *out) {
    if (!g || !out) return -1;
    out->edge_ids = NULL; out->count = 0; out->total_weight = 0.0;

    GV_GAContext *ctx = gv_ga_build(g);
    if (!ctx) return -1;
    size_t N = gv_ga_count(ctx);
    if (N == 0) { gv_ga_free(ctx); return 0; }

    /* Collect unique undirected edges. Each edge appears once in its source
     * node's out_edges; iterate all nodes' out_edges to enumerate every edge. */
    size_t total_edges = 0;
    for (size_t i = 0; i < N; i++) {
        const GV_GraphNode *ni = graph_get_node(g, gv_ga_id(ctx, i));
        if (ni) total_edges += ni->out_count;
    }

    MSFEdge *edges = NULL;
    if (total_edges > 0) {
        edges = (MSFEdge *)gv_alloc(total_edges * sizeof(MSFEdge));
        if (!edges) { gv_ga_free(ctx); return -1; }
    }

    size_t ecount = 0;
    for (size_t i = 0; i < N; i++) {
        const GV_GraphNode *ni = graph_get_node(g, gv_ga_id(ctx, i));
        if (!ni) continue;
        for (size_t e = 0; e < ni->out_count; e++) {
            size_t v = gv_ga_index(ctx, ni->out_edges[e].neighbor_id);
            if (v == (size_t)-1) continue;
            const GV_GraphEdge *edge = graph_get_edge(g, ni->out_edges[e].edge_id);
            double w = edge ? (double)edge->weight : 1.0;
            edges[ecount].edge_id = ni->out_edges[e].edge_id;
            edges[ecount].weight = w;
            edges[ecount].u = i;
            edges[ecount].v = v;
            ecount++;
        }
    }

    if (ecount > 1) qsort(edges, ecount, sizeof(MSFEdge), msf_cmp);

    size_t *par  = (size_t *)gv_alloc(N * sizeof(size_t));
    size_t *rank = (size_t *)gv_calloc(N, sizeof(size_t));
    uint64_t *chosen = (ecount > 0) ? (uint64_t *)gv_alloc(ecount * sizeof(uint64_t)) : NULL;
    if (!par || !rank || (ecount > 0 && !chosen)) {
        gv_free(par); gv_free(rank); gv_free(chosen); gv_free(edges);
        gv_ga_free(ctx);
        return -1;
    }
    for (size_t i = 0; i < N; i++) par[i] = i;

    size_t chosen_count = 0;
    double total_w = 0.0;
    for (size_t k = 0; k < ecount; k++) {
        size_t ru = uf_find(par, edges[k].u);
        size_t rv = uf_find(par, edges[k].v);
        if (ru == rv) continue;         /* would form a cycle */
        /* union by rank */
        if (rank[ru] < rank[rv]) { size_t t = ru; ru = rv; rv = t; }
        par[rv] = ru;
        if (rank[ru] == rank[rv]) rank[ru]++;
        chosen[chosen_count++] = edges[k].edge_id;
        total_w += edges[k].weight;
    }

    uint64_t *edge_ids = NULL;
    if (chosen_count > 0) {
        edge_ids = (uint64_t *)gv_alloc(chosen_count * sizeof(uint64_t));
        if (!edge_ids) {
            gv_free(par); gv_free(rank); gv_free(chosen); gv_free(edges);
            gv_ga_free(ctx);
            return -1;
        }
        memcpy(edge_ids, chosen, chosen_count * sizeof(uint64_t));
    }

    out->edge_ids = edge_ids;
    out->count = chosen_count;
    out->total_weight = total_w;

    gv_free(par); gv_free(rank); gv_free(chosen); gv_free(edges);
    gv_ga_free(ctx);
    return 0;
}

/* ── Maximum flow (Edmonds-Karp) over a dense residual matrix ───────────────── */

double graph_max_flow(const GV_GraphDB *g, uint64_t source, uint64_t sink) {
    if (!g) return -1.0;
    if (source == sink) return -1.0;

    GV_GAContext *ctx = gv_ga_build(g);
    if (!ctx) return -1.0;
    size_t N = gv_ga_count(ctx);
    if (N == 0) { gv_ga_free(ctx); return -1.0; }

    size_t s = gv_ga_index(ctx, source);
    size_t t = gv_ga_index(ctx, sink);
    if (s == (size_t)-1 || t == (size_t)-1) { gv_ga_free(ctx); return -1.0; }

    /* Dense residual capacity matrix cap[u*N + v]. Directed; parallel edges
     * between the same (u,v) sum their capacities. Back-edges start at 0 and
     * accumulate reverse residual capacity as flow is pushed. */
    double *cap = (double *)gv_calloc(N * N, sizeof(double));
    if (!cap) { gv_ga_free(ctx); return -1.0; }

    for (size_t u = 0; u < N; u++) {
        const GV_GraphNode *nu = graph_get_node(g, gv_ga_id(ctx, u));
        if (!nu) continue;
        for (size_t e = 0; e < nu->out_count; e++) {
            size_t v = gv_ga_index(ctx, nu->out_edges[e].neighbor_id);
            if (v == (size_t)-1 || v == u) continue;
            const GV_GraphEdge *edge = graph_get_edge(g, nu->out_edges[e].edge_id);
            double c = edge ? (double)edge->weight : 1.0;
            if (c < 0.0) c = 0.0;
            cap[u * N + v] += c;         /* sum parallel-edge capacities */
        }
    }

    size_t *parent = (size_t *)gv_alloc(N * sizeof(size_t));
    size_t *queue  = (size_t *)gv_alloc(N * sizeof(size_t));
    uint8_t *seen  = (uint8_t *)gv_alloc(N * sizeof(uint8_t));
    if (!parent || !queue || !seen) {
        gv_free(parent); gv_free(queue); gv_free(seen); gv_free(cap);
        gv_ga_free(ctx);
        return -1.0;
    }

    double max_flow = 0.0;
    for (;;) {
        /* BFS for a shortest augmenting path in the residual graph. */
        for (size_t i = 0; i < N; i++) { parent[i] = (size_t)-1; seen[i] = 0; }
        size_t head = 0, tail = 0;
        queue[tail++] = s;
        seen[s] = 1;
        int reached = 0;
        while (head < tail) {
            size_t u = queue[head++];
            if (u == t) { reached = 1; break; }
            for (size_t v = 0; v < N; v++) {
                if (!seen[v] && cap[u * N + v] > 0.0) {
                    seen[v] = 1;
                    parent[v] = u;
                    queue[tail++] = v;
                }
            }
        }
        if (!reached) break;

        /* Bottleneck along the found path. */
        double bottleneck = DBL_MAX;
        for (size_t v = t; v != s; v = parent[v]) {
            size_t u = parent[v];
            double residual = cap[u * N + v];
            if (residual < bottleneck) bottleneck = residual;
        }
        if (!(bottleneck > 0.0) || bottleneck == DBL_MAX) break;

        /* Augment: subtract on forward, add on reverse residual edges. */
        for (size_t v = t; v != s; v = parent[v]) {
            size_t u = parent[v];
            cap[u * N + v] -= bottleneck;
            cap[v * N + u] += bottleneck;
        }
        max_flow += bottleneck;
    }

    gv_free(parent); gv_free(queue); gv_free(seen); gv_free(cap);
    gv_ga_free(ctx);
    return max_flow;
}
