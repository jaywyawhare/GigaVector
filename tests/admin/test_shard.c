#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "admin/shard.h"

#include "schema/metadata.h"
#include "storage/database.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_shard_config_init(void) {
    GV_ShardConfig config;
    memset(&config, 0xFF, sizeof(config));
    shard_config_init(&config);

    ASSERT(config.shard_count > 0, "default shard_count should be positive");
    ASSERT(config.strategy == GV_SHARD_CONSISTENT, "default strategy should be GV_SHARD_CONSISTENT");
    ASSERT(config.replication_factor >= 1, "default replication_factor should be >= 1");
    ASSERT(config.virtual_nodes > 0, "default virtual_nodes should be positive");
    return 0;
}

static int test_shard_config_init_idempotent(void) {
    GV_ShardConfig c1, c2;
    memset(&c1, 0xAA, sizeof(c1));
    memset(&c2, 0x55, sizeof(c2));
    shard_config_init(&c1);
    shard_config_init(&c2);
    ASSERT(memcmp(&c1, &c2, sizeof(GV_ShardConfig)) == 0,
           "config_init should produce identical results on repeated calls");
    return 0;
}

static int test_shard_create_destroy(void) {
    GV_ShardManager *mgr = shard_manager_create(NULL);
    ASSERT(mgr != NULL, "shard_manager_create with NULL config should succeed");
    shard_manager_destroy(mgr);

    GV_ShardConfig config;
    shard_config_init(&config);
    config.shard_count = 4;
    mgr = shard_manager_create(&config);
    ASSERT(mgr != NULL, "shard_manager_create with explicit config should succeed");
    shard_manager_destroy(mgr);

    shard_manager_destroy(NULL);
    return 0;
}

static int test_shard_add_and_list(void) {
    GV_ShardManager *mgr = shard_manager_create(NULL);
    ASSERT(mgr != NULL, "create shard manager");

    int rc = shard_add(mgr, 1, "node1:6000");
    ASSERT(rc == 0, "add shard 1 should succeed");

    rc = shard_add(mgr, 2, "node2:6000");
    ASSERT(rc == 0, "add shard 2 should succeed");

    rc = shard_add(mgr, 3, "node3:6000");
    ASSERT(rc == 0, "add shard 3 should succeed");

    GV_ShardInfo *shards = NULL;
    size_t count = 0;
    rc = shard_list(mgr, &shards, &count);
    ASSERT(rc == 0, "shard_list should succeed");
    ASSERT(count == 3, "should have 3 shards");

    shard_free_list(shards, count);
    shard_manager_destroy(mgr);
    return 0;
}

static int test_shard_add_duplicate(void) {
    GV_ShardManager *mgr = shard_manager_create(NULL);
    ASSERT(mgr != NULL, "create shard manager");

    int rc = shard_add(mgr, 1, "node1:6000");
    ASSERT(rc == 0, "first add should succeed");

    rc = shard_add(mgr, 1, "node1b:6000");
    ASSERT(rc == -1, "duplicate shard_id add should fail");

    shard_manager_destroy(mgr);
    return 0;
}

static int test_shard_for_vector_consistent(void) {
    GV_ShardManager *mgr = shard_manager_create(NULL);
    ASSERT(mgr != NULL, "create shard manager");

    shard_add(mgr, 0, "node0:6000");
    shard_add(mgr, 1, "node1:6000");
    shard_add(mgr, 2, "node2:6000");

    int s1 = shard_for_vector(mgr, 42);
    int s2 = shard_for_vector(mgr, 42);
    ASSERT(s1 >= 0, "shard_for_vector should return >= 0");
    ASSERT(s1 == s2, "shard_for_vector should be consistent for same vector_id");

    int s3 = shard_for_vector(mgr, 100);
    ASSERT(s3 >= 0, "shard_for_vector(100) should return >= 0");

    int s4 = shard_for_vector(mgr, 0);
    ASSERT(s4 >= 0, "shard_for_vector(0) should return >= 0");

    shard_manager_destroy(mgr);
    return 0;
}

static int test_shard_for_key(void) {
    GV_ShardManager *mgr = shard_manager_create(NULL);
    ASSERT(mgr != NULL, "create shard manager");

    shard_add(mgr, 0, "node0:6000");
    shard_add(mgr, 1, "node1:6000");

    const char *key1 = "document-abc";
    const char *key2 = "document-xyz";

    int s1 = shard_for_key(mgr, key1, strlen(key1));
    ASSERT(s1 >= 0, "shard_for_key should return >= 0");

    int s1b = shard_for_key(mgr, key1, strlen(key1));
    ASSERT(s1 == s1b, "shard_for_key should be consistent");

    int s2 = shard_for_key(mgr, key2, strlen(key2));
    ASSERT(s2 >= 0, "shard_for_key with different key should return >= 0");

    int s_null = shard_for_key(NULL, key1, strlen(key1));
    ASSERT(s_null == -1, "shard_for_key(NULL, ...) should return -1");

    shard_manager_destroy(mgr);
    return 0;
}

static int test_shard_get_info(void) {
    GV_ShardManager *mgr = shard_manager_create(NULL);
    ASSERT(mgr != NULL, "create shard manager");

    shard_add(mgr, 5, "node5:7000");

    GV_ShardInfo info;
    memset(&info, 0, sizeof(info));
    int rc = shard_get_info(mgr, 5, &info);
    ASSERT(rc == 0, "get_info for existing shard should succeed");
    ASSERT(info.shard_id == 5, "shard_id should match");
    ASSERT(info.state == GV_SHARD_ACTIVE, "new shard should be ACTIVE");
    gv_free(info.node_address);

    rc = shard_get_info(mgr, 999, &info);
    ASSERT(rc == -1, "get_info for non-existent shard should return -1");

    shard_manager_destroy(mgr);
    return 0;
}

static int test_shard_set_state(void) {
    GV_ShardManager *mgr = shard_manager_create(NULL);
    ASSERT(mgr != NULL, "create shard manager");

    shard_add(mgr, 1, "node1:6000");

    int rc = shard_set_state(mgr, 1, GV_SHARD_READONLY);
    ASSERT(rc == 0, "set_state to READONLY should succeed");

    GV_ShardInfo info;
    memset(&info, 0, sizeof(info));
    shard_get_info(mgr, 1, &info);
    ASSERT(info.state == GV_SHARD_READONLY, "state should be READONLY after set");
    gv_free(info.node_address);

    rc = shard_set_state(mgr, 1, GV_SHARD_MIGRATING);
    ASSERT(rc == 0, "set_state to MIGRATING should succeed");

    shard_get_info(mgr, 1, &info);
    ASSERT(info.state == GV_SHARD_MIGRATING, "state should be MIGRATING");
    gv_free(info.node_address);

    rc = shard_set_state(mgr, 1, GV_SHARD_OFFLINE);
    ASSERT(rc == 0, "set_state to OFFLINE should succeed");

    rc = shard_set_state(mgr, 999, GV_SHARD_ACTIVE);
    ASSERT(rc == -1, "set_state on non-existent shard should fail");

    shard_manager_destroy(mgr);
    return 0;
}

static int test_shard_remove(void) {
    GV_ShardManager *mgr = shard_manager_create(NULL);
    ASSERT(mgr != NULL, "create shard manager");

    shard_add(mgr, 1, "node1:6000");
    shard_add(mgr, 2, "node2:6000");

    int rc = shard_remove(mgr, 1);
    ASSERT(rc == 0, "remove shard 1 should succeed");

    GV_ShardInfo *shards = NULL;
    size_t count = 0;
    shard_list(mgr, &shards, &count);
    ASSERT(count == 1, "should have 1 shard after removal");
    shard_free_list(shards, count);

    rc = shard_remove(mgr, 1);
    ASSERT(rc == -1, "removing already-removed shard should fail");

    GV_ShardInfo info;
    rc = shard_get_info(mgr, 1, &info);
    ASSERT(rc == -1, "get_info on removed shard should fail");

    shard_manager_destroy(mgr);
    return 0;
}

static int test_shard_attach_local(void) {
    GV_ShardManager *mgr = shard_manager_create(NULL);
    ASSERT(mgr != NULL, "create shard manager");

    shard_add(mgr, 0, "local:6000");

    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "create test database");

    int rc = shard_attach_local(mgr, 0, db);
    ASSERT(rc == 0, "attach_local should succeed");

    rc = shard_attach_local(mgr, 999, db);
    ASSERT(rc == -1, "attach_local to non-existent shard should fail");

    shard_manager_destroy(mgr);
    db_close(db);
    return 0;
}

static int test_shard_get_local_db(void) {
    GV_ShardManager *mgr = shard_manager_create(NULL);
    ASSERT(mgr != NULL, "create shard manager");

    shard_add(mgr, 0, "local:6000");

    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "create test database");

    shard_attach_local(mgr, 0, db);

    GV_Database *retrieved = shard_get_local_db(mgr, 0);
    ASSERT(retrieved == db, "get_local_db should return the attached database");

    GV_Database *null_db = shard_get_local_db(mgr, 999);
    ASSERT(null_db == NULL, "get_local_db for non-existent shard should return NULL");

    shard_add(mgr, 1, "remote:6000");
    GV_Database *no_local = shard_get_local_db(mgr, 1);
    ASSERT(no_local == NULL, "get_local_db for shard without attached db should return NULL");

    shard_manager_destroy(mgr);
    db_close(db);
    return 0;
}

static int test_shard_migrate_preserves_metadata(void) {
    GV_ShardManager *mgr = shard_manager_create(NULL);
    ASSERT(mgr != NULL, "create shard manager");

    shard_add(mgr, 0, "node0:6000");
    shard_add(mgr, 1, "node1:6000");

    GV_Database *db0 = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    GV_Database *db1 = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db0 != NULL && db1 != NULL, "open databases");

    ASSERT(shard_attach_local(mgr, 0, db0) == 0, "attach shard 0");
    ASSERT(shard_attach_local(mgr, 1, db1) == 0, "attach shard 1");

    float vec[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    const char *keys[] = {"memory_id", "content"};
    const char *values[] = {"mem-1", "Alice likes dark mode"};
    ASSERT(db_add_vector_with_rich_metadata(db0, vec, 4, keys, values, 2) == 0,
           "add vector with metadata");

    int migrated = shard_migrate_vectors(mgr, 0, 1, 1);
    ASSERT(migrated == 1, "migrate one vector");
    ASSERT(database_count(db1) == 1, "destination should have one vector");

    float q[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    GV_SearchResult src_res;
    int src_n = db_search_filtered(db0, q, 1, &src_res, GV_DISTANCE_EUCLIDEAN, "memory_id", "mem-1");
    ASSERT(src_n == 0, "source should have no searchable vectors after migrate");

    GV_SearchResult res;
    int n = db_search_filtered(db1, q, 1, &res, GV_DISTANCE_EUCLIDEAN, "memory_id", "mem-1");
    ASSERT(n == 1, "memory_id metadata searchable on destination");
    ASSERT(res.vector != NULL, "search result has vector");
    const char *content = vector_get_metadata(res.vector, "content");
    ASSERT(content != NULL && strcmp(content, "Alice likes dark mode") == 0,
           "content metadata preserved");
    gv_search_results_free(&res, 1);

    shard_manager_destroy(mgr);
    db_close(db0);
    db_close(db1);
    return 0;
}

static int test_shard_migrate_vector_at(void) {
    GV_ShardManager *mgr = shard_manager_create(NULL);
    ASSERT(mgr != NULL, "create shard manager");

    shard_add(mgr, 0, "node0:6000");
    shard_add(mgr, 1, "node1:6000");

    GV_Database *db0 = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    GV_Database *db1 = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db0 != NULL && db1 != NULL, "open databases");

    ASSERT(shard_attach_local(mgr, 0, db0) == 0, "attach shard 0");
    ASSERT(shard_attach_local(mgr, 1, db1) == 0, "attach shard 1");

    float v0[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v1[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    ASSERT(db_add_vector_with_metadata(db0, v0, 4, "memory_id", "first") == 0, "add v0");
    ASSERT(db_add_vector_with_metadata(db0, v1, 4, "memory_id", "second") == 0, "add v1");
    ASSERT(database_count(db0) == 2, "source has two vectors");

    size_t new_idx = 999;
    ASSERT(shard_migrate_vector_at(mgr, 0, 1, 0, &new_idx) == 0, "migrate index 0");
    ASSERT(new_idx == 0, "destination index is 0");
    ASSERT(database_count(db1) == 1, "destination has one vector");

    float q[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    GV_SearchResult res;
    int n = db_search_filtered(db1, q, 1, &res, GV_DISTANCE_EUCLIDEAN, "memory_id", "first");
    ASSERT(n == 1, "migrated vector searchable on destination");
    gv_search_results_free(&res, 1);

    n = db_search_filtered(db0, q, 1, &res, GV_DISTANCE_EUCLIDEAN, "memory_id", "first");
    ASSERT(n == 0, "migrated vector removed from source");
    gv_search_results_free(&res, 1);

    n = db_search_filtered(db0, q, 1, &res, GV_DISTANCE_EUCLIDEAN, "memory_id", "second");
    ASSERT(n == 1, "remaining vector still on source");
    gv_search_results_free(&res, 1);

    shard_manager_destroy(mgr);
    db_close(db0);
    db_close(db1);
    return 0;
}

static int test_shard_rebalance(void) {
    GV_ShardManager *mgr = shard_manager_create(NULL);
    ASSERT(mgr != NULL, "create shard manager");

    shard_add(mgr, 0, "node0:6000");
    shard_add(mgr, 1, "node1:6000");

    int rc = shard_rebalance_start(mgr);
    ASSERT(rc == 0, "rebalance_start should succeed");

    double progress = -1.0;
    int status = shard_rebalance_status(mgr, &progress);
    ASSERT(status >= 0, "rebalance_status should return >= 0");
    if (status == 1) {
        ASSERT(progress >= 0.0 && progress <= 1.0,
               "progress should be between 0.0 and 1.0");
    }

    rc = shard_rebalance_cancel(mgr);
    ASSERT(rc == 0, "rebalance_cancel should succeed");

    shard_manager_destroy(mgr);
    return 0;
}

static int test_shard_rebalance_null(void) {
    int rc = shard_rebalance_start(NULL);
    ASSERT(rc == -1, "rebalance_start(NULL) should return -1");

    double progress;
    rc = shard_rebalance_status(NULL, &progress);
    ASSERT(rc == -1, "rebalance_status(NULL) should return -1");

    rc = shard_rebalance_cancel(NULL);
    ASSERT(rc == -1, "rebalance_cancel(NULL) should return -1");
    return 0;
}

static int test_shard_strategies(void) {
    GV_ShardStrategy strategies[] = {GV_SHARD_HASH, GV_SHARD_RANGE, GV_SHARD_CONSISTENT};
    const char *names[] = {"HASH", "RANGE", "CONSISTENT"};

    for (int i = 0; i < 3; i++) {
        GV_ShardConfig config;
        shard_config_init(&config);
        config.strategy = strategies[i];

        GV_ShardManager *mgr = shard_manager_create(&config);
        ASSERT(mgr != NULL, "create manager with strategy should succeed");

        shard_add(mgr, 0, "node0:6000");
        shard_add(mgr, 1, "node1:6000");

        int s = shard_for_vector(mgr, 42);
        ASSERT(s >= 0, "shard_for_vector with strategy should return valid shard");
        (void)names[i]; /* suppress unused warning */

        shard_manager_destroy(mgr);
    }
    return 0;
}

static int test_shard_list_empty(void) {
    GV_ShardManager *mgr = shard_manager_create(NULL);
    ASSERT(mgr != NULL, "create shard manager");

    GV_ShardInfo *shards = NULL;
    size_t count = 99;
    int rc = shard_list(mgr, &shards, &count);
    ASSERT(rc == 0, "listing empty shard manager should succeed");
    ASSERT(count == 0, "empty manager should have 0 shards");

    shard_free_list(shards, count);
    shard_manager_destroy(mgr);
    return 0;
}

static int test_shard_free_list_null(void) {
    shard_free_list(NULL, 0);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing shard_config_init...", test_shard_config_init},
        {"Testing shard_config_init_idempotent...", test_shard_config_init_idempotent},
        {"Testing shard_create_destroy...", test_shard_create_destroy},
        {"Testing shard_add_and_list...", test_shard_add_and_list},
        {"Testing shard_add_duplicate...", test_shard_add_duplicate},
        {"Testing shard_for_vector_consistent...", test_shard_for_vector_consistent},
        {"Testing shard_for_key...", test_shard_for_key},
        {"Testing shard_get_info...", test_shard_get_info},
        {"Testing shard_set_state...", test_shard_set_state},
        {"Testing shard_remove...", test_shard_remove},
        {"Testing shard_attach_local...", test_shard_attach_local},
        {"Testing shard_get_local_db...", test_shard_get_local_db},
        {"Testing shard_migrate_preserves_metadata...", test_shard_migrate_preserves_metadata},
        {"Testing shard_migrate_vector_at...", test_shard_migrate_vector_at},
        {"Testing shard_rebalance...", test_shard_rebalance},
        {"Testing shard_rebalance_null...", test_shard_rebalance_null},
        {"Testing shard_strategies...", test_shard_strategies},
        {"Testing shard_list_empty...", test_shard_list_empty},
        {"Testing shard_free_list_null...", test_shard_free_list_null},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        printf("  %s ", tests[i].name);
        if (tests[i].fn() == 0) {
            printf("OK\n");
            passed++;
        } else {
            printf("FAILED\n");
        }
    }
    printf("\n%d/%d tests passed\n", passed, n);
    return passed == n ? 0 : 1;
}
