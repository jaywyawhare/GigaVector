#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_shard.h"
#include "gigavector/gv_database.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_shard_config_init(void) {
    GV_ShardConfig config;
    memset(&config, 0xFF, sizeof(config));
    gv_shard_config_init(&config);

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
    gv_shard_config_init(&c1);
    gv_shard_config_init(&c2);
    ASSERT(memcmp(&c1, &c2, sizeof(GV_ShardConfig)) == 0,
           "config_init should produce identical results on repeated calls");
    return 0;
}

static int test_shard_create_destroy(void) {
    GV_ShardManager *mgr = gv_shard_manager_create(NULL);
    ASSERT(mgr != NULL, "shard_manager_create with NULL config should succeed");
    gv_shard_manager_destroy(mgr);

    /* Create with explicit config */
    GV_ShardConfig config;
    gv_shard_config_init(&config);
    config.shard_count = 4;
    mgr = gv_shard_manager_create(&config);
    ASSERT(mgr != NULL, "shard_manager_create with explicit config should succeed");
    gv_shard_manager_destroy(mgr);

    /* Destroy NULL should be safe */
    gv_shard_manager_destroy(NULL);
    return 0;
}

static int test_shard_add_and_list(void) {
    GV_ShardManager *mgr = gv_shard_manager_create(NULL);
    ASSERT(mgr != NULL, "create shard manager");

    int rc = gv_shard_add(mgr, 1, "node1:6000");
    ASSERT(rc == 0, "add shard 1 should succeed");

    rc = gv_shard_add(mgr, 2, "node2:6000");
    ASSERT(rc == 0, "add shard 2 should succeed");

    rc = gv_shard_add(mgr, 3, "node3:6000");
    ASSERT(rc == 0, "add shard 3 should succeed");

    /* List shards */
    GV_ShardInfo *shards = NULL;
    size_t count = 0;
    rc = gv_shard_list(mgr, &shards, &count);
    ASSERT(rc == 0, "shard_list should succeed");
    ASSERT(count == 3, "should have 3 shards");

    gv_shard_free_list(shards, count);
    gv_shard_manager_destroy(mgr);
    return 0;
}

static int test_shard_add_duplicate(void) {
    GV_ShardManager *mgr = gv_shard_manager_create(NULL);
    ASSERT(mgr != NULL, "create shard manager");

    int rc = gv_shard_add(mgr, 1, "node1:6000");
    ASSERT(rc == 0, "first add should succeed");

    /* Adding the same shard_id again should fail */
    rc = gv_shard_add(mgr, 1, "node1b:6000");
    ASSERT(rc == -1, "duplicate shard_id add should fail");

    gv_shard_manager_destroy(mgr);
    return 0;
}

static int test_shard_for_vector_consistent(void) {
    GV_ShardManager *mgr = gv_shard_manager_create(NULL);
    ASSERT(mgr != NULL, "create shard manager");

    gv_shard_add(mgr, 0, "node0:6000");
    gv_shard_add(mgr, 1, "node1:6000");
    gv_shard_add(mgr, 2, "node2:6000");

    /* Same vector_id should always map to the same shard */
    int s1 = gv_shard_for_vector(mgr, 42);
    int s2 = gv_shard_for_vector(mgr, 42);
    ASSERT(s1 >= 0, "shard_for_vector should return >= 0");
    ASSERT(s1 == s2, "shard_for_vector should be consistent for same vector_id");

    /* Different vector_ids should return valid shard IDs */
    int s3 = gv_shard_for_vector(mgr, 100);
    ASSERT(s3 >= 0, "shard_for_vector(100) should return >= 0");

    int s4 = gv_shard_for_vector(mgr, 0);
    ASSERT(s4 >= 0, "shard_for_vector(0) should return >= 0");

    gv_shard_manager_destroy(mgr);
    return 0;
}

static int test_shard_for_key(void) {
    GV_ShardManager *mgr = gv_shard_manager_create(NULL);
    ASSERT(mgr != NULL, "create shard manager");

    gv_shard_add(mgr, 0, "node0:6000");
    gv_shard_add(mgr, 1, "node1:6000");

    const char *key1 = "document-abc";
    const char *key2 = "document-xyz";

    int s1 = gv_shard_for_key(mgr, key1, strlen(key1));
    ASSERT(s1 >= 0, "shard_for_key should return >= 0");

    /* Same key should map to same shard */
    int s1b = gv_shard_for_key(mgr, key1, strlen(key1));
    ASSERT(s1 == s1b, "shard_for_key should be consistent");

    int s2 = gv_shard_for_key(mgr, key2, strlen(key2));
    ASSERT(s2 >= 0, "shard_for_key with different key should return >= 0");

    /* NULL manager should return -1 */
    int s_null = gv_shard_for_key(NULL, key1, strlen(key1));
    ASSERT(s_null == -1, "shard_for_key(NULL, ...) should return -1");

    gv_shard_manager_destroy(mgr);
    return 0;
}

static int test_shard_get_info(void) {
    GV_ShardManager *mgr = gv_shard_manager_create(NULL);
    ASSERT(mgr != NULL, "create shard manager");

    gv_shard_add(mgr, 5, "node5:7000");

    GV_ShardInfo info;
    memset(&info, 0, sizeof(info));
    int rc = gv_shard_get_info(mgr, 5, &info);
    ASSERT(rc == 0, "get_info for existing shard should succeed");
    ASSERT(info.shard_id == 5, "shard_id should match");
    ASSERT(info.state == GV_SHARD_ACTIVE, "new shard should be ACTIVE");

    /* Non-existent shard should fail */
    rc = gv_shard_get_info(mgr, 999, &info);
    ASSERT(rc == -1, "get_info for non-existent shard should return -1");

    gv_shard_manager_destroy(mgr);
    return 0;
}

static int test_shard_set_state(void) {
    GV_ShardManager *mgr = gv_shard_manager_create(NULL);
    ASSERT(mgr != NULL, "create shard manager");

    gv_shard_add(mgr, 1, "node1:6000");

    /* Transition to READONLY */
    int rc = gv_shard_set_state(mgr, 1, GV_SHARD_READONLY);
    ASSERT(rc == 0, "set_state to READONLY should succeed");

    GV_ShardInfo info;
    memset(&info, 0, sizeof(info));
    gv_shard_get_info(mgr, 1, &info);
    ASSERT(info.state == GV_SHARD_READONLY, "state should be READONLY after set");

    /* Transition to MIGRATING */
    rc = gv_shard_set_state(mgr, 1, GV_SHARD_MIGRATING);
    ASSERT(rc == 0, "set_state to MIGRATING should succeed");

    gv_shard_get_info(mgr, 1, &info);
    ASSERT(info.state == GV_SHARD_MIGRATING, "state should be MIGRATING");

    /* Transition to OFFLINE */
    rc = gv_shard_set_state(mgr, 1, GV_SHARD_OFFLINE);
    ASSERT(rc == 0, "set_state to OFFLINE should succeed");

    /* Non-existent shard */
    rc = gv_shard_set_state(mgr, 999, GV_SHARD_ACTIVE);
    ASSERT(rc == -1, "set_state on non-existent shard should fail");

    gv_shard_manager_destroy(mgr);
    return 0;
}

static int test_shard_remove(void) {
    GV_ShardManager *mgr = gv_shard_manager_create(NULL);
    ASSERT(mgr != NULL, "create shard manager");

    gv_shard_add(mgr, 1, "node1:6000");
    gv_shard_add(mgr, 2, "node2:6000");

    int rc = gv_shard_remove(mgr, 1);
    ASSERT(rc == 0, "remove shard 1 should succeed");

    /* Verify only shard 2 remains */
    GV_ShardInfo *shards = NULL;
    size_t count = 0;
    gv_shard_list(mgr, &shards, &count);
    ASSERT(count == 1, "should have 1 shard after removal");
    gv_shard_free_list(shards, count);

    /* Removing non-existent shard should fail */
    rc = gv_shard_remove(mgr, 1);
    ASSERT(rc == -1, "removing already-removed shard should fail");

    /* get_info on removed shard should fail */
    GV_ShardInfo info;
    rc = gv_shard_get_info(mgr, 1, &info);
    ASSERT(rc == -1, "get_info on removed shard should fail");

    gv_shard_manager_destroy(mgr);
    return 0;
}

static int test_shard_attach_local(void) {
    GV_ShardManager *mgr = gv_shard_manager_create(NULL);
    ASSERT(mgr != NULL, "create shard manager");

    gv_shard_add(mgr, 0, "local:6000");

    /* Create a real in-memory database */
    GV_Database *db = gv_db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "create test database");

    int rc = gv_shard_attach_local(mgr, 0, db);
    ASSERT(rc == 0, "attach_local should succeed");

    /* Attaching to non-existent shard should fail */
    rc = gv_shard_attach_local(mgr, 999, db);
    ASSERT(rc == -1, "attach_local to non-existent shard should fail");

    gv_shard_manager_destroy(mgr);
    gv_db_close(db);
    return 0;
}

static int test_shard_get_local_db(void) {
    GV_ShardManager *mgr = gv_shard_manager_create(NULL);
    ASSERT(mgr != NULL, "create shard manager");

    gv_shard_add(mgr, 0, "local:6000");

    GV_Database *db = gv_db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "create test database");

    gv_shard_attach_local(mgr, 0, db);

    GV_Database *retrieved = gv_shard_get_local_db(mgr, 0);
    ASSERT(retrieved == db, "get_local_db should return the attached database");

    /* Non-existent shard should return NULL */
    GV_Database *null_db = gv_shard_get_local_db(mgr, 999);
    ASSERT(null_db == NULL, "get_local_db for non-existent shard should return NULL");

    /* Shard without local db should return NULL */
    gv_shard_add(mgr, 1, "remote:6000");
    GV_Database *no_local = gv_shard_get_local_db(mgr, 1);
    ASSERT(no_local == NULL, "get_local_db for shard without attached db should return NULL");

    gv_shard_manager_destroy(mgr);
    gv_db_close(db);
    return 0;
}

static int test_shard_rebalance(void) {
    GV_ShardManager *mgr = gv_shard_manager_create(NULL);
    ASSERT(mgr != NULL, "create shard manager");

    gv_shard_add(mgr, 0, "node0:6000");
    gv_shard_add(mgr, 1, "node1:6000");

    /* Start rebalance */
    int rc = gv_shard_rebalance_start(mgr);
    ASSERT(rc == 0, "rebalance_start should succeed");

    /* Check status */
    double progress = -1.0;
    int status = gv_shard_rebalance_status(mgr, &progress);
    /* status: 1 if rebalancing, 0 if not, -1 error */
    ASSERT(status >= 0, "rebalance_status should return >= 0");
    if (status == 1) {
        ASSERT(progress >= 0.0 && progress <= 1.0,
               "progress should be between 0.0 and 1.0");
    }

    /* Cancel rebalance */
    rc = gv_shard_rebalance_cancel(mgr);
    ASSERT(rc == 0, "rebalance_cancel should succeed");

    gv_shard_manager_destroy(mgr);
    return 0;
}

static int test_shard_rebalance_null(void) {
    int rc = gv_shard_rebalance_start(NULL);
    ASSERT(rc == -1, "rebalance_start(NULL) should return -1");

    double progress;
    rc = gv_shard_rebalance_status(NULL, &progress);
    ASSERT(rc == -1, "rebalance_status(NULL) should return -1");

    rc = gv_shard_rebalance_cancel(NULL);
    ASSERT(rc == -1, "rebalance_cancel(NULL) should return -1");
    return 0;
}

static int test_shard_strategies(void) {
    /* Test with each strategy */
    GV_ShardStrategy strategies[] = {GV_SHARD_HASH, GV_SHARD_RANGE, GV_SHARD_CONSISTENT};
    const char *names[] = {"HASH", "RANGE", "CONSISTENT"};

    for (int i = 0; i < 3; i++) {
        GV_ShardConfig config;
        gv_shard_config_init(&config);
        config.strategy = strategies[i];

        GV_ShardManager *mgr = gv_shard_manager_create(&config);
        ASSERT(mgr != NULL, "create manager with strategy should succeed");

        gv_shard_add(mgr, 0, "node0:6000");
        gv_shard_add(mgr, 1, "node1:6000");

        int s = gv_shard_for_vector(mgr, 42);
        ASSERT(s >= 0, "shard_for_vector with strategy should return valid shard");
        (void)names[i]; /* suppress unused warning */

        gv_shard_manager_destroy(mgr);
    }
    return 0;
}

static int test_shard_list_empty(void) {
    GV_ShardManager *mgr = gv_shard_manager_create(NULL);
    ASSERT(mgr != NULL, "create shard manager");

    GV_ShardInfo *shards = NULL;
    size_t count = 99;
    int rc = gv_shard_list(mgr, &shards, &count);
    ASSERT(rc == 0, "listing empty shard manager should succeed");
    ASSERT(count == 0, "empty manager should have 0 shards");

    gv_shard_free_list(shards, count);
    gv_shard_manager_destroy(mgr);
    return 0;
}

static int test_shard_free_list_null(void) {
    /* Should not crash */
    gv_shard_free_list(NULL, 0);
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
