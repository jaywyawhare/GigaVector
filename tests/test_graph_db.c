#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "gigavector/gv_graph_db.h"

#define ASSERT(cond, msg)         \
    do {                          \
        if (!(cond)) {            \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1;            \
        }                         \
    } while (0)

/* ---- Lifecycle ---- */

static int test_create_destroy(void) {
    GV_GraphDB *g = gv_graph_create(NULL);
    ASSERT(g != NULL, "create with NULL config");
    ASSERT(gv_graph_node_count(g) == 0, "empty graph node count");
    ASSERT(gv_graph_edge_count(g) == 0, "empty graph edge count");
    gv_graph_destroy(g);

    /* Custom config */
    GV_GraphDBConfig cfg;
    gv_graph_config_init(&cfg);
    ASSERT(cfg.node_bucket_count == 4096, "default node buckets");
    ASSERT(cfg.edge_bucket_count == 8192, "default edge buckets");
    ASSERT(cfg.enforce_referential_integrity == 1, "default ref integrity");

    cfg.node_bucket_count = 128;
    g = gv_graph_create(&cfg);
    ASSERT(g != NULL, "create with custom config");
    gv_graph_destroy(g);

    /* Destroy NULL is safe */
    gv_graph_destroy(NULL);
    return 0;
}

/* ---- Node Operations ---- */

static int test_add_get_nodes(void) {
    GV_GraphDB *g = gv_graph_create(NULL);

    uint64_t n1 = gv_graph_add_node(g, "Person");
    ASSERT(n1 > 0, "add node 1");
    uint64_t n2 = gv_graph_add_node(g, "Person");
    ASSERT(n2 > 0, "add node 2");
    uint64_t n3 = gv_graph_add_node(g, "Company");
    ASSERT(n3 > 0, "add node 3");
    ASSERT(n1 != n2 && n2 != n3, "unique node IDs");

    ASSERT(gv_graph_node_count(g) == 3, "node count");

    const GV_GraphNode *node = gv_graph_get_node(g, n1);
    ASSERT(node != NULL, "get node 1");
    ASSERT(strcmp(node->label, "Person") == 0, "node 1 label");

    ASSERT(gv_graph_get_node(g, 99999) == NULL, "get nonexistent node");

    gv_graph_destroy(g);
    return 0;
}

static int test_node_properties(void) {
    GV_GraphDB *g = gv_graph_create(NULL);
    uint64_t n = gv_graph_add_node(g, "Person");

    ASSERT(gv_graph_set_node_prop(g, n, "name", "Alice") == 0, "set prop name");
    ASSERT(gv_graph_set_node_prop(g, n, "age", "30") == 0, "set prop age");

    const char *name = gv_graph_get_node_prop(g, n, "name");
    ASSERT(name != NULL && strcmp(name, "Alice") == 0, "get prop name");

    const char *age = gv_graph_get_node_prop(g, n, "age");
    ASSERT(age != NULL && strcmp(age, "30") == 0, "get prop age");

    /* Overwrite */
    ASSERT(gv_graph_set_node_prop(g, n, "name", "Bob") == 0, "overwrite prop");
    name = gv_graph_get_node_prop(g, n, "name");
    ASSERT(name != NULL && strcmp(name, "Bob") == 0, "get overwritten prop");

    /* Missing prop */
    ASSERT(gv_graph_get_node_prop(g, n, "email") == NULL, "get missing prop");

    /* Invalid node */
    ASSERT(gv_graph_set_node_prop(g, 99999, "k", "v") != 0, "set prop on invalid node");

    gv_graph_destroy(g);
    return 0;
}

static int test_find_nodes_by_label(void) {
    GV_GraphDB *g = gv_graph_create(NULL);
    gv_graph_add_node(g, "Person");
    gv_graph_add_node(g, "Person");
    gv_graph_add_node(g, "Company");
    gv_graph_add_node(g, "Person");

    uint64_t ids[10];
    int n = gv_graph_find_nodes_by_label(g, "Person", ids, 10);
    ASSERT(n == 3, "find 3 Person nodes");

    n = gv_graph_find_nodes_by_label(g, "Company", ids, 10);
    ASSERT(n == 1, "find 1 Company node");

    n = gv_graph_find_nodes_by_label(g, "Unknown", ids, 10);
    ASSERT(n == 0, "find 0 Unknown nodes");

    gv_graph_destroy(g);
    return 0;
}

static int test_remove_node(void) {
    GV_GraphDB *g = gv_graph_create(NULL);
    uint64_t n1 = gv_graph_add_node(g, "A");
    uint64_t n2 = gv_graph_add_node(g, "B");
    gv_graph_add_edge(g, n1, n2, "LINK", 1.0f);

    ASSERT(gv_graph_node_count(g) == 2, "2 nodes before remove");
    ASSERT(gv_graph_edge_count(g) == 1, "1 edge before remove");

    /* Remove node should cascade-delete edges */
    ASSERT(gv_graph_remove_node(g, n1) == 0, "remove node");
    ASSERT(gv_graph_node_count(g) == 1, "1 node after remove");
    ASSERT(gv_graph_edge_count(g) == 0, "0 edges after cascade remove");
    ASSERT(gv_graph_get_node(g, n1) == NULL, "removed node gone");

    /* Remove nonexistent */
    ASSERT(gv_graph_remove_node(g, 99999) != 0, "remove nonexistent");

    gv_graph_destroy(g);
    return 0;
}

/* ---- Edge Operations ---- */

static int test_add_get_edges(void) {
    GV_GraphDB *g = gv_graph_create(NULL);
    uint64_t n1 = gv_graph_add_node(g, "A");
    uint64_t n2 = gv_graph_add_node(g, "B");
    uint64_t n3 = gv_graph_add_node(g, "C");

    uint64_t e1 = gv_graph_add_edge(g, n1, n2, "KNOWS", 1.0f);
    ASSERT(e1 > 0, "add edge 1");
    uint64_t e2 = gv_graph_add_edge(g, n2, n3, "LIKES", 2.5f);
    ASSERT(e2 > 0, "add edge 2");
    ASSERT(gv_graph_edge_count(g) == 2, "edge count");

    const GV_GraphEdge *edge = gv_graph_get_edge(g, e1);
    ASSERT(edge != NULL, "get edge 1");
    ASSERT(edge->source_id == n1, "edge source");
    ASSERT(edge->target_id == n2, "edge target");
    ASSERT(strcmp(edge->label, "KNOWS") == 0, "edge label");
    ASSERT(edge->weight >= 0.99f && edge->weight <= 1.01f, "edge weight");

    ASSERT(gv_graph_get_edge(g, 99999) == NULL, "get nonexistent edge");

    gv_graph_destroy(g);
    return 0;
}

static int test_edge_properties(void) {
    GV_GraphDB *g = gv_graph_create(NULL);
    uint64_t n1 = gv_graph_add_node(g, "A");
    uint64_t n2 = gv_graph_add_node(g, "B");
    uint64_t e = gv_graph_add_edge(g, n1, n2, "REL", 1.0f);

    ASSERT(gv_graph_set_edge_prop(g, e, "since", "2024") == 0, "set edge prop");
    const char *val = gv_graph_get_edge_prop(g, e, "since");
    ASSERT(val != NULL && strcmp(val, "2024") == 0, "get edge prop");
    ASSERT(gv_graph_get_edge_prop(g, e, "missing") == NULL, "missing edge prop");

    gv_graph_destroy(g);
    return 0;
}

static int test_adjacency_queries(void) {
    GV_GraphDB *g = gv_graph_create(NULL);
    uint64_t a = gv_graph_add_node(g, "A");
    uint64_t b = gv_graph_add_node(g, "B");
    uint64_t c = gv_graph_add_node(g, "C");

    uint64_t e1 = gv_graph_add_edge(g, a, b, "R1", 1.0f);
    uint64_t e2 = gv_graph_add_edge(g, a, c, "R2", 1.0f);
    uint64_t e3 = gv_graph_add_edge(g, b, a, "R3", 1.0f);

    uint64_t out[10];
    int n = gv_graph_get_edges_out(g, a, out, 10);
    ASSERT(n == 2, "a has 2 outgoing edges");

    n = gv_graph_get_edges_in(g, a, out, 10);
    ASSERT(n == 1, "a has 1 incoming edge");
    ASSERT(out[0] == e3, "incoming edge is e3");

    n = gv_graph_get_neighbors(g, a, out, 10);
    ASSERT(n == 2, "a has 2 unique neighbors");

    gv_graph_destroy(g);
    return 0;
}

static int test_remove_edge(void) {
    GV_GraphDB *g = gv_graph_create(NULL);
    uint64_t n1 = gv_graph_add_node(g, "A");
    uint64_t n2 = gv_graph_add_node(g, "B");
    uint64_t e = gv_graph_add_edge(g, n1, n2, "R", 1.0f);

    ASSERT(gv_graph_remove_edge(g, e) == 0, "remove edge");
    ASSERT(gv_graph_edge_count(g) == 0, "0 edges after remove");
    ASSERT(gv_graph_get_edge(g, e) == NULL, "removed edge gone");
    ASSERT(gv_graph_remove_edge(g, 99999) != 0, "remove nonexistent edge");

    gv_graph_destroy(g);
    return 0;
}

/* ---- Traversal ---- */

static int test_bfs_dfs(void) {
    /*  1 -> 2 -> 3 -> 4
     *  |         ^
     *  +-> 5 ----+
     */
    GV_GraphDB *g = gv_graph_create(NULL);
    uint64_t n1 = gv_graph_add_node(g, "N");
    uint64_t n2 = gv_graph_add_node(g, "N");
    uint64_t n3 = gv_graph_add_node(g, "N");
    uint64_t n4 = gv_graph_add_node(g, "N");
    uint64_t n5 = gv_graph_add_node(g, "N");

    gv_graph_add_edge(g, n1, n2, "E", 1.0f);
    gv_graph_add_edge(g, n2, n3, "E", 1.0f);
    gv_graph_add_edge(g, n3, n4, "E", 1.0f);
    gv_graph_add_edge(g, n1, n5, "E", 1.0f);
    gv_graph_add_edge(g, n5, n3, "E", 1.0f);

    /* BFS from n1 should reach all 5 nodes */
    uint64_t visited[10];
    int n = gv_graph_bfs(g, n1, 10, visited, 10);
    ASSERT(n == 5, "BFS reaches all 5 nodes");
    ASSERT(visited[0] == n1, "BFS starts at n1");

    /* BFS with depth 1 */
    n = gv_graph_bfs(g, n1, 1, visited, 10);
    ASSERT(n == 3, "BFS depth 1: n1, n2, n5");

    /* DFS from n1 */
    n = gv_graph_dfs(g, n1, 10, visited, 10);
    ASSERT(n == 5, "DFS reaches all 5 nodes");

    gv_graph_destroy(g);
    return 0;
}

static int test_shortest_path(void) {
    /*  1 --1.0--> 2 --1.0--> 4
     *  |                      ^
     *  +--5.0--> 3 --1.0-----+
     */
    GV_GraphDB *g = gv_graph_create(NULL);
    uint64_t n1 = gv_graph_add_node(g, "N");
    uint64_t n2 = gv_graph_add_node(g, "N");
    uint64_t n3 = gv_graph_add_node(g, "N");
    uint64_t n4 = gv_graph_add_node(g, "N");

    gv_graph_add_edge(g, n1, n2, "E", 1.0f);
    gv_graph_add_edge(g, n2, n4, "E", 1.0f);
    gv_graph_add_edge(g, n1, n3, "E", 5.0f);
    gv_graph_add_edge(g, n3, n4, "E", 1.0f);

    GV_GraphPath path;
    int rc = gv_graph_shortest_path(g, n1, n4, &path);
    ASSERT(rc == 0, "shortest path found");
    ASSERT(path.length == 2, "path length 2 (n1->n2->n4)");
    ASSERT(path.total_weight >= 1.99f && path.total_weight <= 2.01f, "path weight ~2.0");
    ASSERT(path.node_ids[0] == n1, "path starts at n1");
    ASSERT(path.node_ids[2] == n4, "path ends at n4");
    gv_graph_free_path(&path);

    /* No path */
    uint64_t isolated = gv_graph_add_node(g, "Isolated");
    rc = gv_graph_shortest_path(g, n1, isolated, &path);
    ASSERT(rc != 0, "no path to isolated node");

    gv_graph_destroy(g);
    return 0;
}

/* ---- Analytics ---- */

static int test_degree(void) {
    GV_GraphDB *g = gv_graph_create(NULL);
    uint64_t a = gv_graph_add_node(g, "A");
    uint64_t b = gv_graph_add_node(g, "B");
    uint64_t c = gv_graph_add_node(g, "C");

    gv_graph_add_edge(g, a, b, "R", 1.0f);
    gv_graph_add_edge(g, a, c, "R", 1.0f);
    gv_graph_add_edge(g, b, a, "R", 1.0f);

    ASSERT(gv_graph_out_degree(g, a) == 2, "a out_degree 2");
    ASSERT(gv_graph_in_degree(g, a) == 1, "a in_degree 1");
    ASSERT(gv_graph_degree(g, a) == 3, "a total degree 3");
    ASSERT(gv_graph_degree(g, c) == 1, "c total degree 1");

    gv_graph_destroy(g);
    return 0;
}

static int test_pagerank(void) {
    /* Simple 3-node graph: 1 -> 2 -> 3, 3 -> 1 */
    GV_GraphDB *g = gv_graph_create(NULL);
    uint64_t n1 = gv_graph_add_node(g, "N");
    uint64_t n2 = gv_graph_add_node(g, "N");
    uint64_t n3 = gv_graph_add_node(g, "N");

    gv_graph_add_edge(g, n1, n2, "E", 1.0f);
    gv_graph_add_edge(g, n2, n3, "E", 1.0f);
    gv_graph_add_edge(g, n3, n1, "E", 1.0f);

    float pr1 = gv_graph_pagerank(g, n1, 50, 0.85f);
    float pr2 = gv_graph_pagerank(g, n2, 50, 0.85f);
    float pr3 = gv_graph_pagerank(g, n3, 50, 0.85f);

    /* Symmetric cycle: all should have equal PageRank ~0.333 */
    ASSERT(pr1 > 0.3f && pr1 < 0.4f, "pr1 ~0.333");
    ASSERT(pr2 > 0.3f && pr2 < 0.4f, "pr2 ~0.333");
    float sum = pr1 + pr2 + pr3;
    ASSERT(sum > 0.95f && sum < 1.05f, "PageRank sums to ~1.0");

    gv_graph_destroy(g);
    return 0;
}

static int test_connected_components(void) {
    GV_GraphDB *g = gv_graph_create(NULL);
    uint64_t a = gv_graph_add_node(g, "A");
    uint64_t b = gv_graph_add_node(g, "B");
    uint64_t c = gv_graph_add_node(g, "C");
    uint64_t d = gv_graph_add_node(g, "D");

    gv_graph_add_edge(g, a, b, "R", 1.0f);
    gv_graph_add_edge(g, c, d, "R", 1.0f);
    /* Two disconnected components: {a,b} and {c,d} */

    uint64_t comps[4];
    int n = gv_graph_connected_components(g, comps, 4);
    ASSERT(n == 2, "2 connected components");

    gv_graph_destroy(g);
    return 0;
}

static int test_clustering_coefficient(void) {
    /* Triangle: a-b, b-c, a-c => cc = 1.0 */
    GV_GraphDB *g = gv_graph_create(NULL);
    uint64_t a = gv_graph_add_node(g, "A");
    uint64_t b = gv_graph_add_node(g, "B");
    uint64_t c = gv_graph_add_node(g, "C");

    gv_graph_add_edge(g, a, b, "R", 1.0f);
    gv_graph_add_edge(g, b, c, "R", 1.0f);
    gv_graph_add_edge(g, a, c, "R", 1.0f);

    float cc = gv_graph_clustering_coefficient(g, a);
    ASSERT(cc > 0.95f, "full triangle clustering = 1.0");

    /* Star: a->b, a->c, a->d (no edges among b,c,d) => cc = 0 */
    uint64_t d = gv_graph_add_node(g, "D");
    gv_graph_add_edge(g, a, d, "R", 1.0f);
    /* b,c,d have one edge among them (b->c), 3 possible => 1/3 */

    gv_graph_destroy(g);
    return 0;
}

/* ---- Persistence ---- */

static int test_save_load(void) {
    const char *path = "/tmp/test_gv_graph.gvgr";

    GV_GraphDB *g = gv_graph_create(NULL);
    uint64_t n1 = gv_graph_add_node(g, "Person");
    uint64_t n2 = gv_graph_add_node(g, "Company");
    gv_graph_set_node_prop(g, n1, "name", "Alice");
    uint64_t e = gv_graph_add_edge(g, n1, n2, "WORKS_AT", 1.0f);
    gv_graph_set_edge_prop(g, e, "role", "Engineer");

    ASSERT(gv_graph_save(g, path) == 0, "save graph");
    gv_graph_destroy(g);

    GV_GraphDB *g2 = gv_graph_load(path);
    ASSERT(g2 != NULL, "load graph");
    ASSERT(gv_graph_node_count(g2) == 2, "loaded node count");
    ASSERT(gv_graph_edge_count(g2) == 1, "loaded edge count");

    const char *name = gv_graph_get_node_prop(g2, n1, "name");
    ASSERT(name != NULL && strcmp(name, "Alice") == 0, "loaded node prop");

    const GV_GraphEdge *edge = gv_graph_get_edge(g2, e);
    ASSERT(edge != NULL, "loaded edge exists");
    ASSERT(strcmp(edge->label, "WORKS_AT") == 0, "loaded edge label");

    const char *role = gv_graph_get_edge_prop(g2, e, "role");
    ASSERT(role != NULL && strcmp(role, "Engineer") == 0, "loaded edge prop");

    gv_graph_destroy(g2);
    unlink(path);
    return 0;
}

/* ---- Main ---- */

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestEntry;

int main(void) {
    TestEntry tests[] = {
        {"create/destroy",          test_create_destroy},
        {"add/get nodes",           test_add_get_nodes},
        {"node properties",         test_node_properties},
        {"find nodes by label",     test_find_nodes_by_label},
        {"remove node (cascade)",   test_remove_node},
        {"add/get edges",           test_add_get_edges},
        {"edge properties",         test_edge_properties},
        {"adjacency queries",       test_adjacency_queries},
        {"remove edge",             test_remove_edge},
        {"BFS/DFS",                 test_bfs_dfs},
        {"shortest path",           test_shortest_path},
        {"degree",                  test_degree},
        {"PageRank",                test_pagerank},
        {"connected components",    test_connected_components},
        {"clustering coefficient",  test_clustering_coefficient},
        {"save/load",               test_save_load},
    };

    int total = (int)(sizeof(tests) / sizeof(tests[0]));
    int passed = 0;

    for (int i = 0; i < total; i++) {
        printf("Testing %s... ", tests[i].name);
        if (tests[i].fn() == 0) {
            printf("[OK]\n");
            passed++;
        }
    }

    printf("\n%d/%d tests passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}
