#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "features/graph_db.h"
#include "features/graph_rpc.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int contains(const uint64_t *a, int n, uint64_t v) {
    for (int i = 0; i < n; i++) if (a[i] == v) return 1;
    return 0;
}

/* Chain 1->2->3->4 partitioned by id % 2, with partition 1 living on a remote
 * node reached over TCP. A k-hop traversal must hop local<->remote across the
 * network to reach the far end. */
static int test_networked_khop(void) {
    GV_GraphDBConfig cfg;
    graph_config_init(&cfg);
    cfg.enforce_referential_integrity = 0;

    /* Remote partition (odd ids): nodes 1,3; edges 1->2 and 3->4 stored here. */
    GV_GraphDB *p1 = graph_create(&cfg);
    ASSERT(p1 != NULL, "create remote partition");
    ASSERT(graph_add_node_with_id(p1, 1, "N") == 1, "p1 node 1");
    ASSERT(graph_add_node_with_id(p1, 3, "N") == 3, "p1 node 3");
    ASSERT(graph_add_edge(p1, 1, 2, "next", 1.0f) != 0, "edge 1->2");
    ASSERT(graph_add_edge(p1, 3, 4, "next", 1.0f) != 0, "edge 3->4");

    GV_GraphRpcServer *srv = graph_rpc_serve(p1, "127.0.0.1", 0);
    ASSERT(srv != NULL, "serve remote partition");
    char addr[64];
    snprintf(addr, sizeof(addr), "127.0.0.1:%u", graph_rpc_server_port(srv));

    /* Local partition (even ids): nodes 2,4; edge 2->3 stored here. */
    GV_GraphDB *p0 = graph_create(&cfg);
    ASSERT(p0 != NULL, "create local partition");
    ASSERT(graph_add_node_with_id(p0, 2, "N") == 2, "p0 node 2");
    ASSERT(graph_add_node_with_id(p0, 4, "N") == 4, "p0 node 4");
    ASSERT(graph_add_edge(p0, 2, 3, "next", 1.0f) != 0, "edge 2->3");

    GV_GraphPartition parts[2] = {
        { p0, NULL },     /* partition 0: local */
        { NULL, addr },   /* partition 1: remote over TCP */
    };

    uint64_t out[16];

    /* 1 hop from node 1 (which lives on the remote partition). */
    int n1 = graph_dist_khop_net(parts, 2, NULL, NULL, 1, 1, NULL, out, 16, 64);
    ASSERT(n1 == 2 && contains(out, n1, 1) && contains(out, n1, 2), "1-hop = {1,2}");
    ASSERT(!contains(out, n1, 4), "node 4 not reachable in 1 hop");

    /* 3 hops must cross local<->remote several times to reach node 4. */
    int n3 = graph_dist_khop_net(parts, 2, NULL, NULL, 1, 3, NULL, out, 16, 64);
    ASSERT(n3 == 4, "3-hop reaches all four nodes over the network");
    ASSERT(contains(out, n3, 3) && contains(out, n3, 4), "reached remote-then-local far end");

    /* Direct remote neighbour fetch. */
    uint64_t nb[8];
    int nn = graph_rpc_neighbors("127.0.0.1", graph_rpc_server_port(srv), 3, NULL, nb, 8);
    ASSERT(nn == 1 && nb[0] == 4, "remote neighbours(3) = {4}");
    ASSERT(graph_rpc_neighbors("127.0.0.1", 1, 1, NULL, nb, 8) == -1, "dead port fails");

    /* With the remote node down, traversal degrades to local-only. */
    graph_rpc_server_stop(srv);
    int nd = graph_dist_khop_net(parts, 2, NULL, NULL, 1, 3, NULL, out, 16, 64);
    ASSERT(nd >= 1, "remote down -> partial (best-effort) results");

    graph_destroy(p0);
    graph_destroy(p1);
    return 0;
}

/* Distributed Cypher: MATCH (a)-[:KNOWS*min..max]->(b) WHERE id(a)=1, over a
 * KNOWS chain that spans a local and a remote partition, with a LIKES distractor
 * edge that the typed match must ignore. */
static int test_dist_cypher(void) {
    GV_GraphDBConfig cfg;
    graph_config_init(&cfg);
    cfg.enforce_referential_integrity = 0;

    /* Remote partition (odd ids): 1,3,5; edges from odd-id sources. */
    GV_GraphDB *p1 = graph_create(&cfg);
    ASSERT(p1 != NULL, "create remote");
    graph_add_node_with_id(p1, 1, "N");
    graph_add_node_with_id(p1, 3, "N");
    graph_add_node_with_id(p1, 5, "N");
    ASSERT(graph_add_edge(p1, 1, 2, "KNOWS", 1.0f) != 0, "1-KNOWS->2");
    ASSERT(graph_add_edge(p1, 1, 5, "LIKES", 1.0f) != 0, "1-LIKES->5 (distractor)");
    ASSERT(graph_add_edge(p1, 3, 4, "KNOWS", 1.0f) != 0, "3-KNOWS->4");

    GV_GraphRpcServer *srv = graph_rpc_serve(p1, "127.0.0.1", 0);
    ASSERT(srv != NULL, "serve remote");
    char addr[64];
    snprintf(addr, sizeof(addr), "127.0.0.1:%u", graph_rpc_server_port(srv));

    /* Local partition (even ids): 2,4; edge from even-id source. */
    GV_GraphDB *p0 = graph_create(&cfg);
    ASSERT(p0 != NULL, "create local");
    graph_add_node_with_id(p0, 2, "N");
    graph_add_node_with_id(p0, 4, "N");
    ASSERT(graph_add_edge(p0, 2, 3, "KNOWS", 1.0f) != 0, "2-KNOWS->3");

    GV_GraphPartition parts[2] = { { p0, NULL }, { NULL, addr } };
    uint64_t out[16];

    /* MATCH (a)-[:KNOWS*1..3]->(b) WHERE id(a)=1 RETURN b  => {2,3,4}, not 5. */
    int n = cypher_dist_query(parts, 2, NULL, NULL,
                              "MATCH (a)-[:KNOWS*1..3]->(b) WHERE id(a) = 1 RETURN b",
                              out, 16, 64);
    ASSERT(n == 3, "KNOWS*1..3 matches three nodes");
    ASSERT(contains(out, n, 2) && contains(out, n, 3) && contains(out, n, 4),
           "KNOWS chain matched across partitions");
    ASSERT(!contains(out, n, 5), "LIKES distractor excluded by typed match");
    ASSERT(!contains(out, n, 1), "start node not returned (min hop 1)");

    /* min-hop bound: KNOWS*2..3 excludes the 1-hop neighbour (node 2). */
    int n2 = cypher_dist_match(parts, 2, NULL, NULL, 1, "KNOWS", 2, 3, out, 16, 64);
    ASSERT(n2 == 2 && contains(out, n2, 3) && contains(out, n2, 4) && !contains(out, n2, 2),
           "min-hop 2 excludes the direct neighbour");

    /* Untyped 1-hop from 1 includes both KNOWS(2) and LIKES(5). */
    int n3 = cypher_dist_match(parts, 2, NULL, NULL, 1, NULL, 1, 1, out, 16, 64);
    ASSERT(n3 == 2 && contains(out, n3, 2) && contains(out, n3, 5),
           "untyped 1-hop includes every out-edge");

    ASSERT(cypher_dist_query(parts, 2, NULL, NULL, "not a query", out, 16, 64) == -1,
           "unparseable query rejected");

    graph_rpc_server_stop(srv);
    graph_destroy(p0);
    graph_destroy(p1);
    return 0;
}

/* Multi-clause chained pattern across partitions:
 *   MATCH (a)-[:KNOWS*1]->(b)-[:WORKS_AT*1]->(c) WHERE id(a)=1 RETURN c
 * segment 1 (KNOWS from 1) -> {2,4}; segment 2 (WORKS_AT from {2,4}) -> {3,5}. */
static int test_dist_cypher_chain(void) {
    GV_GraphDBConfig cfg;
    graph_config_init(&cfg);
    cfg.enforce_referential_integrity = 0;

    /* Remote partition (odd ids): 1,3,5; KNOWS edges from node 1. */
    GV_GraphDB *p1 = graph_create(&cfg);
    ASSERT(p1 != NULL, "create remote");
    graph_add_node_with_id(p1, 1, "N");
    graph_add_node_with_id(p1, 3, "N");
    graph_add_node_with_id(p1, 5, "N");
    ASSERT(graph_add_edge(p1, 1, 2, "KNOWS", 1.0f) != 0, "1-KNOWS->2");
    ASSERT(graph_add_edge(p1, 1, 4, "KNOWS", 1.0f) != 0, "1-KNOWS->4");

    GV_GraphRpcServer *srv = graph_rpc_serve(p1, "127.0.0.1", 0);
    ASSERT(srv != NULL, "serve remote");
    char addr[64];
    snprintf(addr, sizeof(addr), "127.0.0.1:%u", graph_rpc_server_port(srv));

    /* Local partition (even ids): 2,4,6; WORKS_AT edges + a KNOWS distractor. */
    GV_GraphDB *p0 = graph_create(&cfg);
    ASSERT(p0 != NULL, "create local");
    graph_add_node_with_id(p0, 2, "N");
    graph_add_node_with_id(p0, 4, "N");
    graph_add_node_with_id(p0, 6, "N");
    ASSERT(graph_add_edge(p0, 2, 3, "WORKS_AT", 1.0f) != 0, "2-WORKS_AT->3");
    ASSERT(graph_add_edge(p0, 4, 5, "WORKS_AT", 1.0f) != 0, "4-WORKS_AT->5");
    ASSERT(graph_add_edge(p0, 2, 6, "KNOWS", 1.0f) != 0, "2-KNOWS->6 (distractor)");

    GV_GraphPartition parts[2] = { { p0, NULL }, { NULL, addr } };
    uint64_t out[16];

    /* Text query. */
    int n = cypher_dist_query(parts, 2, NULL, NULL,
        "MATCH (a)-[:KNOWS*1..1]->(b)-[:WORKS_AT*1..1]->(c) WHERE id(a) = 1 RETURN c",
        out, 16, 64);
    ASSERT(n == 2, "two-segment chain matches two nodes");
    ASSERT(contains(out, n, 3) && contains(out, n, 5), "reached c-nodes via both segments");
    ASSERT(!contains(out, n, 2) && !contains(out, n, 4), "intermediate b-nodes not returned");
    ASSERT(!contains(out, n, 6), "KNOWS distractor not followed by WORKS_AT segment");

    /* Programmatic segment API. */
    GV_CypherSegment segs[2] = {
        { "KNOWS", 1, 1 },
        { "WORKS_AT", 1, 1 },
    };
    int m = cypher_dist_path(parts, 2, NULL, NULL, 1, segs, 2, out, 16, 64);
    ASSERT(m == 2 && contains(out, m, 3) && contains(out, m, 5), "programmatic chain matches");

    graph_rpc_server_stop(srv);
    graph_destroy(p0);
    graph_destroy(p1);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing networked_khop...", test_networked_khop},
        {"Testing dist_cypher...", test_dist_cypher},
        {"Testing dist_cypher_chain...", test_dist_cypher_chain},
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
