/**
 * graph_classify.c — graph classification algorithms for the graph-algorithms
 * layer: greedy graph coloring and greedy maximal independent set.
 *
 * Both treat the graph as UNDIRECTED (a node's neighbors are the union of its
 * out-edge and in-edge endpoints), process nodes in dense-index order for
 * determinism, are read-only w.r.t. the graph, and own their results (freed
 * with the matching graph_node_*_free helper).
 */
#include "features/graph_algos.h"
#include "core/memory.h"

#include <string.h>
#include <stdint.h>

/* ── Greedy graph coloring (undirected) ─────────────────────────────────────
 * Process nodes in dense-index order; assign each node the smallest color
 * index not used by any already-colored neighbor. labels[i] = color of node i;
 * num_labels = number of distinct colors used. */
int graph_coloring(const GV_GraphDB *g, GV_GraphNodeLabels *out) {
    if (!out) return -1;
    memset(out, 0, sizeof(*out));

    GV_GAContext *ctx = gv_ga_build(g);
    if (!ctx) return -1;

    size_t N = gv_ga_count(ctx);
    if (N == 0) {
        gv_ga_free(ctx);
        return 0;  /* out already zeroed */
    }

    uint64_t *node_ids = (uint64_t *)gv_alloc(N * sizeof(uint64_t));
    int64_t  *labels   = (int64_t *)gv_alloc(N * sizeof(int64_t));
    /* Scratch: "is color c forbidden by a neighbor of the current node".
     * A node can need at most (degree) distinct neighbor colors, and colors
     * are dense in [0, N), so N slots always suffice. */
    uint8_t  *used     = (uint8_t *)gv_calloc(N, sizeof(uint8_t));  /* must start all-zero */
    if (!node_ids || !labels || !used) {
        gv_free(node_ids);
        gv_free(labels);
        gv_free(used);
        gv_ga_free(ctx);
        return -1;
    }

    for (size_t i = 0; i < N; i++) {
        node_ids[i] = gv_ga_id(ctx, i);
        labels[i]   = -1;  /* uncolored sentinel */
    }

    size_t num_colors = 0;

    for (size_t i = 0; i < N; i++) {
        const GV_GraphNode *node = graph_get_node(g, node_ids[i]);

        /* Mark colors of already-colored neighbors as forbidden, then reset
         * only those slots afterward by rescanning the same neighbors (avoids
         * an O(N) clear of `used` per node). */
        if (node) {
            for (size_t e = 0; e < node->out_count; e++) {
                size_t ni = gv_ga_index(ctx, node->out_edges[e].neighbor_id);
                if (ni == (size_t)-1 || ni == i) continue;
                int64_t nc = labels[ni];
                if (nc >= 0 && (size_t)nc < N) used[nc] = 1;
            }
            for (size_t e = 0; e < node->in_count; e++) {
                size_t ni = gv_ga_index(ctx, node->in_edges[e].neighbor_id);
                if (ni == (size_t)-1 || ni == i) continue;
                int64_t nc = labels[ni];
                if (nc >= 0 && (size_t)nc < N) used[nc] = 1;
            }
        }

        /* Smallest color index not forbidden. */
        size_t c = 0;
        while (c < N && used[c]) c++;
        labels[i] = (int64_t)c;
        if (c + 1 > num_colors) num_colors = c + 1;

        /* Reset the forbidden markers we set (walk the same neighbors). */
        if (node) {
            for (size_t e = 0; e < node->out_count; e++) {
                size_t ni = gv_ga_index(ctx, node->out_edges[e].neighbor_id);
                if (ni == (size_t)-1 || ni == i) continue;
                int64_t nc = labels[ni];
                if (nc >= 0 && (size_t)nc < N) used[nc] = 0;
            }
            for (size_t e = 0; e < node->in_count; e++) {
                size_t ni = gv_ga_index(ctx, node->in_edges[e].neighbor_id);
                if (ni == (size_t)-1 || ni == i) continue;
                int64_t nc = labels[ni];
                if (nc >= 0 && (size_t)nc < N) used[nc] = 0;
            }
        }
    }

    gv_free(used);
    gv_ga_free(ctx);

    out->node_ids   = node_ids;
    out->labels     = labels;
    out->count      = N;
    out->num_labels = num_colors;
    return 0;
}

/* ── Greedy maximal independent set (undirected) ────────────────────────────
 * Process nodes in dense-index order; add a node to the set if none of its
 * neighbors are already in the set. scores[i] = 1.0 if node i is in the set,
 * else 0.0; node_ids[i] = gv_ga_id(ctx, i); count = N. */
int graph_maximal_independent_set(const GV_GraphDB *g, GV_GraphNodeScores *out) {
    if (!out) return -1;
    memset(out, 0, sizeof(*out));

    GV_GAContext *ctx = gv_ga_build(g);
    if (!ctx) return -1;

    size_t N = gv_ga_count(ctx);
    if (N == 0) {
        gv_ga_free(ctx);
        return 0;  /* out already zeroed */
    }

    uint64_t *node_ids = (uint64_t *)gv_alloc(N * sizeof(uint64_t));
    double   *scores   = (double *)gv_alloc(N * sizeof(double));
    uint8_t  *in_set   = (uint8_t *)gv_calloc(N, sizeof(uint8_t));
    if (!node_ids || !scores || !in_set) {
        gv_free(node_ids);
        gv_free(scores);
        gv_free(in_set);
        gv_ga_free(ctx);
        return -1;
    }

    for (size_t i = 0; i < N; i++) {
        node_ids[i] = gv_ga_id(ctx, i);
    }

    for (size_t i = 0; i < N; i++) {
        const GV_GraphNode *node = graph_get_node(g, node_ids[i]);
        int has_set_neighbor = 0;

        if (node) {
            for (size_t e = 0; e < node->out_count && !has_set_neighbor; e++) {
                size_t ni = gv_ga_index(ctx, node->out_edges[e].neighbor_id);
                if (ni == (size_t)-1 || ni == i) continue;
                if (in_set[ni]) has_set_neighbor = 1;
            }
            for (size_t e = 0; e < node->in_count && !has_set_neighbor; e++) {
                size_t ni = gv_ga_index(ctx, node->in_edges[e].neighbor_id);
                if (ni == (size_t)-1 || ni == i) continue;
                if (in_set[ni]) has_set_neighbor = 1;
            }
        }

        in_set[i] = has_set_neighbor ? 0 : 1;
        scores[i] = in_set[i] ? 1.0 : 0.0;
    }

    gv_free(in_set);
    gv_ga_free(ctx);

    out->node_ids = node_ids;
    out->scores   = scores;
    out->count    = N;
    return 0;
}
