#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "features/graph_db.h"
#include "features/graph_distributed.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int contains(const uint64_t *a, int n, uint64_t v) {
    for (int i = 0; i < n; i++) if (a[i] == v) return 1;
    return 0;
}

/* A 5-node chain 1->2->3->4->5 partitioned across 3 shards by id % 3, so every
 * edge crosses a partition boundary. A k-hop traversal must route each node's
 * expansion to its owning partition; no single partition can reach the far end. */
static int test_cross_partition_khop(void) {
    GV_GraphDBConfig cfg;
    graph_config_init(&cfg);
    cfg.enforce_referential_integrity = 0; /* edges may target remote nodes */

    GV_GraphDB *parts[3];
    for (int i = 0; i < 3; i++) {
        parts[i] = graph_create(&cfg);
        ASSERT(parts[i] != NULL, "create partition");
    }

    /* Node id N lives on parts[N % 3]; its out-edge is stored there too. */
    #define OWNER(N) parts[(N) % 3]
    for (uint64_t n = 1; n <= 5; n++) {
        ASSERT(graph_add_node_with_id(OWNER(n), n, "N") == n, "add node with global id");
    }
    ASSERT(graph_add_edge(OWNER(1), 1, 2, "next", 1.0f) != 0, "edge 1->2 on part1");
    ASSERT(graph_add_edge(OWNER(2), 2, 3, "next", 1.0f) != 0, "edge 2->3 on part2");
    ASSERT(graph_add_edge(OWNER(3), 3, 4, "next", 1.0f) != 0, "edge 3->4 on part0");
    ASSERT(graph_add_edge(OWNER(4), 4, 5, "next", 1.0f) != 0, "edge 4->5 on part1");

    uint64_t out[16];

    /* 1 hop: only {1,2}; the far end is not yet reachable. */
    int n1 = graph_dist_khop((GV_GraphDB *const *)parts, 3, NULL, NULL, 1, 1, NULL, out, 16);
    ASSERT(n1 == 2 && contains(out, n1, 1) && contains(out, n1, 2), "1-hop = {1,2}");
    ASSERT(!contains(out, n1, 5), "node 5 not reachable in 1 hop");

    /* 2 hops crosses into a third partition: {1,2,3}. */
    int n2 = graph_dist_khop((GV_GraphDB *const *)parts, 3, NULL, NULL, 1, 2, NULL, out, 16);
    ASSERT(n2 == 3 && contains(out, n2, 3), "2-hop reaches node 3 (cross-partition)");

    /* 4 hops reaches the whole chain across all partitions. */
    int n4 = graph_dist_khop((GV_GraphDB *const *)parts, 3, NULL, NULL, 1, 4, NULL, out, 16);
    ASSERT(n4 == 5, "4-hop reaches all five nodes across partitions");
    ASSERT(contains(out, n4, 5), "node 5 (on a different partition) reached");

    /* Error handling. */
    ASSERT(graph_dist_khop(NULL, 3, NULL, NULL, 1, 4, NULL, out, 16) == -1, "NULL parts rejected");
    ASSERT(graph_dist_khop((GV_GraphDB *const *)parts, 0, NULL, NULL, 1, 4, NULL, out, 16) == -1,
           "0 partitions rejected");

    #undef OWNER
    for (int i = 0; i < 3; i++) graph_destroy(parts[i]);
    return 0;
}

/* Typed traversal follows only edges whose label matches the predicate. */
static int test_typed_traversal(void) {
    GV_GraphDBConfig cfg;
    graph_config_init(&cfg);
    GV_GraphDB *g = graph_create(&cfg);
    ASSERT(g != NULL, "create graph");
    for (uint64_t n = 1; n <= 4; n++)
        ASSERT(graph_add_node_with_id(g, n, "N") == n, "add node");
    ASSERT(graph_add_edge(g, 1, 2, "friend", 1.0f) != 0, "1-friend->2");
    ASSERT(graph_add_edge(g, 1, 3, "coworker", 1.0f) != 0, "1-coworker->3");
    ASSERT(graph_add_edge(g, 2, 4, "friend", 1.0f) != 0, "2-friend->4");

    GV_GraphDB *parts[1] = { g };
    uint64_t out[16];

    /* Follow only "friend": 1 -> 2 -> 4; the coworker edge to 3 is skipped. */
    int nf = graph_dist_khop((GV_GraphDB *const *)parts, 1, NULL, NULL, 1, 2, "friend", out, 16);
    ASSERT(nf == 3, "friend-only reaches {1,2,4}");
    ASSERT(contains(out, nf, 2) && contains(out, nf, 4), "friend chain followed");
    ASSERT(!contains(out, nf, 3), "coworker edge not followed under 'friend' predicate");

    /* No predicate: every out-edge is followed. */
    int na = graph_dist_khop((GV_GraphDB *const *)parts, 1, NULL, NULL, 1, 2, NULL, out, 16);
    ASSERT(na == 4, "untyped traversal reaches all four");

    graph_destroy(g);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing cross_partition_khop...", test_cross_partition_khop},
        {"Testing typed_traversal...", test_typed_traversal},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        printf("  %s ", tests[i].name);
        if (tests[i].fn() == 0) { printf("OK\n"); passed++; }
        else printf("FAILED\n");
    }
    printf("\n%d/%d tests passed\n", passed, n);
    return passed == n ? 0 : 1;
}
