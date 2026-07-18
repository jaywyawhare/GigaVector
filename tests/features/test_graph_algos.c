/**
 * test_graph_algos.c — exercises the graph data-science library (graph_algos.h)
 * on a small hand-computable graph, asserting known values where possible and
 * sanity-checking the rest. Every result is freed so the ASAN/valgrind CI runs
 * prove the algorithms are memory-safe.
 *
 * Test graph (directed, weight 1 unless noted):
 *   Triangle / 3-cycle:   1->2, 2->3, 3->1
 *   Bridge:               3->4
 *   2-cycle:              4->5, 5->4
 *   Separate 2-cycle:     6->7, 7->6   (disconnected from 1..5)
 */
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "features/graph_db.h"
#include "features/graph_algos.h"
#include "core/memory.h"

static int failures = 0;
#define ASSERT(c, m) do { \
    if (!(c)) { printf("FAIL: %s:%d: %s\n", __FILE__, __LINE__, (m)); failures++; } \
    else { printf("ok: %s\n", (m)); } } while (0)

/* find the score for a given node_id in a scores result */
static double score_of(const GV_GraphNodeScores *s, uint64_t id) {
    for (size_t i = 0; i < s->count; i++) if (s->node_ids[i] == id) return s->scores[i];
    return NAN;
}
static int64_t label_of(const GV_GraphNodeLabels *l, uint64_t id) {
    for (size_t i = 0; i < l->count; i++) if (l->node_ids[i] == id) return l->labels[i];
    return -999;
}

static GV_GraphDB *build_graph(uint64_t ids[8]) {
    GV_GraphDB *g = graph_create(NULL);
    for (int i = 1; i <= 7; i++) ids[i] = graph_add_node(g, "N");
    graph_add_edge(g, ids[1], ids[2], "E", 1.0f);
    graph_add_edge(g, ids[2], ids[3], "E", 1.0f);
    graph_add_edge(g, ids[3], ids[1], "E", 1.0f);
    graph_add_edge(g, ids[3], ids[4], "E", 1.0f);
    graph_add_edge(g, ids[4], ids[5], "E", 1.0f);
    graph_add_edge(g, ids[5], ids[4], "E", 1.0f);
    graph_add_edge(g, ids[6], ids[7], "E", 1.0f);
    graph_add_edge(g, ids[7], ids[6], "E", 1.0f);
    return g;
}

int main(void) {
    uint64_t id[8] = {0};
    GV_GraphDB *g = build_graph(id);
    ASSERT(g != NULL, "graph created");
    ASSERT(graph_node_count(g) == 7, "7 nodes");
    ASSERT(graph_edge_count(g) == 8, "8 edges");

    /* node enumeration */
    uint64_t ids[7];
    ASSERT(graph_get_all_node_ids(g, ids, 7) == 7, "enumerate all node ids");

    GV_GraphNodeScores sc; GV_GraphNodeLabels lb;

    /* ---- centrality ---- */
    ASSERT(graph_degree_centrality(g, 0, &sc) == 0 && sc.count == 7, "degree centrality runs");
    graph_node_scores_free(&sc);

    ASSERT(graph_betweenness_centrality(g, 0, 1, &sc) == 0, "betweenness runs");
    /* node 3 is the only bridge from the triangle into 4/5 → highest betweenness */
    double b3 = score_of(&sc, id[3]), b1 = score_of(&sc, id[1]);
    ASSERT(b3 >= b1, "node 3 (bridge) betweenness >= node 1");
    graph_node_scores_free(&sc);

    ASSERT(graph_closeness_centrality(g, 0, 1, &sc) == 0 && sc.count == 7, "closeness runs");
    graph_node_scores_free(&sc);
    ASSERT(graph_harmonic_centrality(g, 0, 1, &sc) == 0, "harmonic runs");
    graph_node_scores_free(&sc);
    ASSERT(graph_eigenvector_centrality(g, 100, 1e-6, &sc) == 0, "eigenvector runs");
    graph_node_scores_free(&sc);

    ASSERT(graph_pagerank_all(g, 100, 0.85, &sc) == 0 && sc.count == 7, "pagerank runs");
    double psum = 0; for (size_t i = 0; i < sc.count; i++) psum += sc.scores[i];
    ASSERT(fabs(psum - 1.0) < 1e-3, "pagerank sums to ~1");
    graph_node_scores_free(&sc);

    uint64_t src1[1] = { id[1] };
    ASSERT(graph_personalized_pagerank(g, src1, 1, 100, 0.85, &sc) == 0, "personalized PR runs");
    graph_node_scores_free(&sc);
    ASSERT(graph_article_rank(g, 100, 0.85, &sc) == 0, "article rank runs");
    graph_node_scores_free(&sc);

    /* ---- community ---- */
    uint64_t tris = 0;
    ASSERT(graph_triangle_count(g, &sc, &tris) == 0, "triangle count runs");
    ASSERT(tris == 1, "exactly 1 undirected triangle {1,2,3}");
    ASSERT((int)(score_of(&sc, id[1]) + 0.5) == 1, "node 1 in 1 triangle");
    ASSERT((int)(score_of(&sc, id[4]) + 0.5) == 0, "node 4 in 0 triangles");
    graph_node_scores_free(&sc);

    ASSERT(graph_strongly_connected_components(g, &lb) == 0, "SCC runs");
    ASSERT(label_of(&lb, id[1]) == label_of(&lb, id[2]) &&
           label_of(&lb, id[2]) == label_of(&lb, id[3]), "nodes 1,2,3 same SCC");
    ASSERT(label_of(&lb, id[4]) == label_of(&lb, id[5]), "nodes 4,5 same SCC");
    ASSERT(label_of(&lb, id[1]) != label_of(&lb, id[4]), "triangle SCC != {4,5} SCC");
    graph_node_labels_free(&lb);

    ASSERT(graph_label_propagation(g, 100, &lb) == 0 && lb.count == 7, "label propagation runs");
    graph_node_labels_free(&lb);
    ASSERT(graph_louvain(g, 10, &lb) == 0 && lb.count == 7, "louvain runs");
    graph_node_labels_free(&lb);
    ASSERT(graph_kcore(g, &lb) == 0 && lb.count == 7, "k-core runs");
    /* triangle nodes have core number >= 2 */
    ASSERT(label_of(&lb, id[1]) >= 2, "triangle node core >= 2");
    graph_node_labels_free(&lb);

    /* ---- paths ---- */
    ASSERT(graph_single_source_shortest_path(g, id[1], 0, 1, &sc) == 0, "SSSP runs");
    ASSERT((int)(score_of(&sc, id[2]) + 0.5) == 1, "dist 1->2 == 1");
    ASSERT((int)(score_of(&sc, id[3]) + 0.5) == 2, "dist 1->3 == 2");
    ASSERT(isinf(score_of(&sc, id[6])), "node 6 unreachable from 1 (directed)");
    graph_node_scores_free(&sc);

    GV_GraphPath path; memset(&path, 0, sizeof(path));
    ASSERT(graph_astar(g, id[1], id[4], NULL, NULL, &path) == 0, "A* finds path 1->4");
    ASSERT(path.length == 3, "path 1->2->3->4 has length 3");
    graph_free_path(&path);

    int is_dag = -1;
    ASSERT(graph_detect_cycle(g, &is_dag, NULL) == 0, "cycle detection runs");
    ASSERT(is_dag == 0, "graph has a directed cycle (not a DAG)");

    GV_GraphEdgeSet mst; memset(&mst, 0, sizeof(mst));
    ASSERT(graph_minimum_spanning_forest(g, &mst) == 0, "MST forest runs");
    graph_edge_set_free(&mst);

    /* max flow 1->4: single augmenting path through the 3->4 bridge, capacity 1 */
    double flow = graph_max_flow(g, id[1], id[4]);
    ASSERT(flow >= 0.99 && flow <= 1.01, "max flow 1->4 == 1");

    /* ---- link prediction ---- */
    /* nodes 1 and 3 both neighbor node 2 (undirected) → common neighbor >= 1 */
    ASSERT(graph_common_neighbors(g, id[1], id[3]) >= 1.0, "common neighbors 1,3 >= 1");
    ASSERT(graph_jaccard_similarity(g, id[1], id[3]) > 0.0, "jaccard 1,3 > 0");
    ASSERT(graph_cosine_neighborhood(g, id[1], id[3]) > 0.0, "cosine 1,3 > 0");
    ASSERT(graph_adamic_adar(g, id[1], id[3]) >= 0.0, "adamic-adar valid");
    ASSERT(graph_resource_allocation(g, id[1], id[3]) >= 0.0, "resource allocation valid");
    ASSERT(graph_preferential_attachment(g, id[1], id[3]) > 0.0, "pref attachment > 0");
    ASSERT(graph_common_neighbors(g, 99999, id[3]) < 0.0, "absent node -> -1");

    ASSERT(graph_topk_similar(g, id[1], GV_GSIM_JACCARD, 3, &sc) == 0, "top-k similar runs");
    graph_node_scores_free(&sc);

    /* ---- classification ---- */
    ASSERT(graph_coloring(g, &lb) == 0 && lb.count == 7, "coloring runs");
    /* proper coloring: adjacent nodes differ. check the triangle edge 1-2 */
    ASSERT(label_of(&lb, id[1]) != label_of(&lb, id[2]), "adjacent nodes differ in color");
    ASSERT(lb.num_labels >= 3, "triangle needs >= 3 colors");
    graph_node_labels_free(&lb);

    ASSERT(graph_maximal_independent_set(g, &sc) == 0 && sc.count == 7, "MIS runs");
    graph_node_scores_free(&sc);

    /* ---- embeddings ---- */
    GV_GraphEmbeddings emb; memset(&emb, 0, sizeof(emb));
    ASSERT(graph_fastrp(g, 32, 3, NULL, 42, &emb) == 0, "FastRP runs");
    ASSERT(emb.count == 7 && emb.dim == 32, "FastRP shape 7x32");
    graph_embeddings_free(&emb);

    uint64_t *walks = NULL; size_t nw = 0, wl = 0;
    ASSERT(graph_node2vec_walks(g, 2, 5, 1.0, 1.0, 7, &walks, &nw, &wl) == 0, "node2vec walks run");
    ASSERT(nw == 7 * 2 && wl == 5, "walks shape (7*2)x5");
    ASSERT(walks && walks[0] != 0, "first walk starts at a real node");
    gv_free(walks);

    graph_destroy(g);

    if (failures == 0) printf("All graph-algorithm tests PASSED.\n");
    else printf("%d graph-algorithm test(s) FAILED.\n", failures);
    return failures ? 1 : 0;
}
