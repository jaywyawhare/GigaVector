/*
 * CBMC harness: property-graph structural consistency and shortest paths
 * (features/graph_db.c, knowledge_graph.c, context_graph.c, graph_path.c,
 *  graph_linkpred.c, graph_classify.c, recommend.c).
 *
 * Two families of provable property here, neither of which is statistical:
 *
 *  1. ADJACENCY CONSISTENCY -- an edge added from a to b must appear in a's
 *     out-edges and b's in-edges, and must disappear from both when removed.
 *     A half-removed edge is a dangling reference that later traversals follow.
 *
 *  2. SHORTEST-PATH SOUNDNESS -- distances are non-negative, the source is at
 *     distance zero, and an unreachable node is INFINITY rather than a stale or
 *     uninitialised value. Every centrality and recommendation score is built on
 *     these, so a wrong sentinel propagates everywhere.
  *
 * CBMC-SOURCES: src/features/graph_db.c src/features/graph_path.c src/features/graph_algos_util.c src/core/memory.c
 * CBMC-UNWIND: 12
*/
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "features/graph_db.h"
#include "features/graph_algos.h"

#if defined(__CPROVER__) || defined(__CPROVER)
uint64_t nondet_u64(void);
double   nondet_double(void);
#  define ASSUME(c) __CPROVER_assume(c)
#else
static uint64_t g_seed = 0x9E3779B97F4A7C15ULL;
static uint64_t nondet_u64(void) {
    g_seed ^= g_seed << 13; g_seed ^= g_seed >> 7; g_seed ^= g_seed << 17;
    return g_seed;
}
static double nondet_double(void) { return (double)(nondet_u64() % 100); }
#  define ASSUME(c) do { if (!(c)) return 0; } while (0)
#endif

#define NODES 3
#define MAXE  4

int main(void) {
    GV_GraphDBConfig cfg;
    graph_config_init(&cfg);
    GV_GraphDB *g = graph_create(&cfg);
    if (!g) return 0;

    uint64_t n[NODES];
    for (unsigned i = 0; i < NODES; i++) {
        n[i] = graph_add_node(g, "n");
        if (n[i] == 0) { graph_destroy(g); return 0; }
    }

    /* A nondeterministic edge between two real nodes. */
    unsigned a = (unsigned)(nondet_u64() % NODES);
    unsigned b = (unsigned)(nondet_u64() % NODES);
    ASSUME(a != b);

    uint64_t e = graph_add_edge(g, n[a], n[b], "rel", 1.0f);
    if (e != 0) {
        uint64_t outs[MAXE], ins[MAXE];
        int no = graph_get_edges_out(g, n[a], outs, MAXE);
        int ni = graph_get_edges_in(g, n[b], ins, MAXE);

        /* ADJACENCY CONSISTENCY: the edge is visible from both endpoints. */
        int found_out = 0, found_in = 0;
        for (int i = 0; i < no; i++) if (outs[i] == e) found_out = 1;
        for (int i = 0; i < ni; i++) if (ins[i] == e) found_in = 1;
        assert(found_out);
        assert(found_in);

        /* ... and disappears from BOTH on removal. A half-removal leaves a
         * dangling edge id that later traversals dereference. */
        if (graph_remove_edge(g, e) == 0) {
            no = graph_get_edges_out(g, n[a], outs, MAXE);
            ni = graph_get_edges_in(g, n[b], ins, MAXE);
            for (int i = 0; i < no; i++) assert(outs[i] != e);
            for (int i = 0; i < ni; i++) assert(ins[i] != e);
        }
    }

    /* Rebuild a small connected shape and check shortest-path soundness. */
    (void)graph_add_edge(g, n[0], n[1], "rel", 1.0f);
    (void)graph_add_edge(g, n[1], n[2], "rel", 1.0f);
    /* n[3] is deliberately left unreachable from n[0]. */

    GV_GraphNodeScores sc;
    memset(&sc, 0, sizeof(sc));
    if (graph_single_source_shortest_path(g, n[0], 0, 1, &sc) == 0) {
        for (size_t i = 0; i < sc.count; i++) {
            double dv = sc.scores[i];
            /* Distances are non-negative or the explicit unreachable sentinel;
             * never negative, never NaN. */
            assert(!(dv < 0.0));
            assert(dv == dv || isinf(dv));
            if (sc.node_ids[i] == n[0]) assert(dv == 0.0);   /* source is zero */
            if (sc.node_ids[i] == n[3]) assert(isinf(dv));   /* unreachable */
        }
        graph_node_scores_free(&sc);
    }

    graph_destroy(g);
    return 0;
}
