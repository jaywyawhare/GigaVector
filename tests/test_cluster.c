#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_cluster.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_config_init(void) {
    GV_ClusterConfig cfg;
    memset(&cfg, 0xFF, sizeof(cfg));
    gv_cluster_config_init(&cfg);

    /* Verify defaults are set (non-garbage) */
    ASSERT(cfg.heartbeat_interval_ms > 0, "heartbeat_interval_ms should be positive");
    ASSERT(cfg.failure_timeout_ms > 0, "failure_timeout_ms should be positive");
    ASSERT(cfg.role == GV_NODE_DATA || cfg.role == GV_NODE_COORDINATOR || cfg.role == GV_NODE_QUERY,
           "role should be a valid GV_NodeRole");
    return 0;
}

static int test_create_destroy(void) {
    GV_ClusterConfig cfg;
    gv_cluster_config_init(&cfg);
    cfg.node_id = "test-node-1";
    cfg.listen_address = "127.0.0.1:7000";
    cfg.seed_nodes = NULL;
    cfg.role = GV_NODE_DATA;

    GV_Cluster *cluster = gv_cluster_create(&cfg);
    ASSERT(cluster != NULL, "gv_cluster_create should succeed");

    gv_cluster_destroy(cluster);
    return 0;
}

static int test_destroy_null(void) {
    /* Destroying NULL must not crash */
    gv_cluster_destroy(NULL);
    return 0;
}

static int test_create_coordinator(void) {
    GV_ClusterConfig cfg;
    gv_cluster_config_init(&cfg);
    cfg.node_id = "coordinator-1";
    cfg.listen_address = "127.0.0.1:7001";
    cfg.seed_nodes = NULL;
    cfg.role = GV_NODE_COORDINATOR;

    GV_Cluster *cluster = gv_cluster_create(&cfg);
    ASSERT(cluster != NULL, "create coordinator node should succeed");

    gv_cluster_destroy(cluster);
    return 0;
}

static int test_create_query_node(void) {
    GV_ClusterConfig cfg;
    gv_cluster_config_init(&cfg);
    cfg.node_id = "query-1";
    cfg.listen_address = "127.0.0.1:7002";
    cfg.seed_nodes = NULL;
    cfg.role = GV_NODE_QUERY;

    GV_Cluster *cluster = gv_cluster_create(&cfg);
    ASSERT(cluster != NULL, "create query node should succeed");

    gv_cluster_destroy(cluster);
    return 0;
}

static int test_get_local_node(void) {
    GV_ClusterConfig cfg;
    gv_cluster_config_init(&cfg);
    cfg.node_id = "local-node";
    cfg.listen_address = "127.0.0.1:7003";
    cfg.seed_nodes = NULL;
    cfg.role = GV_NODE_DATA;

    GV_Cluster *cluster = gv_cluster_create(&cfg);
    ASSERT(cluster != NULL, "create");

    GV_NodeInfo info;
    memset(&info, 0, sizeof(info));
    int rc = gv_cluster_get_local_node(cluster, &info);
    ASSERT(rc == 0, "gv_cluster_get_local_node should succeed");

    /* The local node info should have our node ID */
    if (info.node_id != NULL) {
        ASSERT(strcmp(info.node_id, "local-node") == 0, "local node_id should match config");
    }

    /* Role should match what we set */
    ASSERT(info.role == GV_NODE_DATA, "local node role should be DATA");

    gv_cluster_free_node_info(&info);
    gv_cluster_destroy(cluster);
    return 0;
}

static int test_get_stats(void) {
    GV_ClusterConfig cfg;
    gv_cluster_config_init(&cfg);
    cfg.node_id = "stats-node";
    cfg.listen_address = "127.0.0.1:7004";
    cfg.seed_nodes = NULL;
    cfg.role = GV_NODE_DATA;

    GV_Cluster *cluster = gv_cluster_create(&cfg);
    ASSERT(cluster != NULL, "create");

    GV_ClusterStats stats;
    memset(&stats, 0xFF, sizeof(stats));
    int rc = gv_cluster_get_stats(cluster, &stats);
    ASSERT(rc == 0, "gv_cluster_get_stats should succeed");

    /* With a single-node cluster, we expect at least 1 node */
    ASSERT(stats.total_nodes >= 1, "total_nodes should be at least 1");
    ASSERT(stats.active_nodes >= 0, "active_nodes should be non-negative");

    gv_cluster_destroy(cluster);
    return 0;
}

static int test_is_healthy(void) {
    GV_ClusterConfig cfg;
    gv_cluster_config_init(&cfg);
    cfg.node_id = "health-node";
    cfg.listen_address = "127.0.0.1:7005";
    cfg.seed_nodes = NULL;
    cfg.role = GV_NODE_DATA;

    GV_Cluster *cluster = gv_cluster_create(&cfg);
    ASSERT(cluster != NULL, "create");

    int healthy = gv_cluster_is_healthy(cluster);
    /* A freshly created single-node cluster should be healthy (1) or not yet ready (0) */
    ASSERT(healthy == 0 || healthy == 1, "is_healthy should return 0 or 1");

    gv_cluster_destroy(cluster);
    return 0;
}

static int test_list_nodes(void) {
    GV_ClusterConfig cfg;
    gv_cluster_config_init(&cfg);
    cfg.node_id = "list-node";
    cfg.listen_address = "127.0.0.1:7006";
    cfg.seed_nodes = NULL;
    cfg.role = GV_NODE_DATA;

    GV_Cluster *cluster = gv_cluster_create(&cfg);
    ASSERT(cluster != NULL, "create");

    GV_NodeInfo *nodes = NULL;
    size_t count = 0;
    int rc = gv_cluster_list_nodes(cluster, &nodes, &count);
    ASSERT(rc == 0, "gv_cluster_list_nodes should succeed");
    ASSERT(count >= 1, "should have at least 1 node (self)");
    ASSERT(nodes != NULL, "nodes pointer should not be NULL");

    gv_cluster_free_node_list(nodes, count);
    gv_cluster_destroy(cluster);
    return 0;
}

static int test_free_node_list_null(void) {
    /* Freeing NULL / 0 count should not crash */
    gv_cluster_free_node_list(NULL, 0);
    return 0;
}

static int test_get_shard_manager(void) {
    GV_ClusterConfig cfg;
    gv_cluster_config_init(&cfg);
    cfg.node_id = "shard-node";
    cfg.listen_address = "127.0.0.1:7007";
    cfg.seed_nodes = NULL;
    cfg.role = GV_NODE_DATA;

    GV_Cluster *cluster = gv_cluster_create(&cfg);
    ASSERT(cluster != NULL, "create");

    GV_ShardManager *sm = gv_cluster_get_shard_manager(cluster);
    ASSERT(sm != NULL, "gv_cluster_get_shard_manager should return non-NULL");

    gv_cluster_destroy(cluster);
    return 0;
}

static int test_start_stop(void) {
    GV_ClusterConfig cfg;
    gv_cluster_config_init(&cfg);
    cfg.node_id = "lifecycle-node";
    cfg.listen_address = "127.0.0.1:7008";
    cfg.seed_nodes = NULL;
    cfg.role = GV_NODE_DATA;

    GV_Cluster *cluster = gv_cluster_create(&cfg);
    ASSERT(cluster != NULL, "create");

    int rc = gv_cluster_start(cluster);
    ASSERT(rc == 0, "gv_cluster_start should succeed");

    rc = gv_cluster_stop(cluster);
    ASSERT(rc == 0, "gv_cluster_stop should succeed");

    gv_cluster_destroy(cluster);
    return 0;
}

static int test_get_node_by_id(void) {
    GV_ClusterConfig cfg;
    gv_cluster_config_init(&cfg);
    cfg.node_id = "lookup-node";
    cfg.listen_address = "127.0.0.1:7009";
    cfg.seed_nodes = NULL;
    cfg.role = GV_NODE_DATA;

    GV_Cluster *cluster = gv_cluster_create(&cfg);
    ASSERT(cluster != NULL, "create");

    GV_NodeInfo info;
    memset(&info, 0, sizeof(info));

    /* Lookup the local node by its ID */
    int rc = gv_cluster_get_node(cluster, "lookup-node", &info);
    ASSERT(rc == 0, "gv_cluster_get_node for local node should succeed");

    /* Lookup non-existent node should fail */
    GV_NodeInfo info2;
    memset(&info2, 0, sizeof(info2));
    rc = gv_cluster_get_node(cluster, "nonexistent-node-xyz", &info2);
    ASSERT(rc != 0, "gv_cluster_get_node for unknown node should fail");

    gv_cluster_free_node_info(&info);
    gv_cluster_destroy(cluster);
    return 0;
}

static int test_config_with_seeds(void) {
    GV_ClusterConfig cfg;
    gv_cluster_config_init(&cfg);
    cfg.node_id = "seeded-node";
    cfg.listen_address = "127.0.0.1:7010";
    cfg.seed_nodes = "127.0.0.1:7000,127.0.0.1:7001";
    cfg.role = GV_NODE_DATA;

    GV_Cluster *cluster = gv_cluster_create(&cfg);
    ASSERT(cluster != NULL, "create with seed nodes should succeed");

    gv_cluster_destroy(cluster);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing config_init...",         test_config_init},
        {"Testing create_destroy...",      test_create_destroy},
        {"Testing destroy_null...",        test_destroy_null},
        {"Testing create_coordinator...",  test_create_coordinator},
        {"Testing create_query_node...",   test_create_query_node},
        {"Testing get_local_node...",      test_get_local_node},
        {"Testing get_stats...",           test_get_stats},
        {"Testing is_healthy...",          test_is_healthy},
        {"Testing list_nodes...",          test_list_nodes},
        {"Testing free_node_list_null...", test_free_node_list_null},
        {"Testing get_shard_manager...",   test_get_shard_manager},
        {"Testing start_stop...",          test_start_stop},
        {"Testing get_node_by_id...",      test_get_node_by_id},
        {"Testing config_with_seeds...",   test_config_with_seeds},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
