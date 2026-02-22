#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_vacuum.h"
#include "gigavector/gv_database.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

/* Helper: create a small in-memory database with some vectors */
static GV_Database *create_test_db(void) {
    GV_Database *db = gv_db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    if (!db) return NULL;
    float v1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v2[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    float v3[4] = {0.0f, 0.0f, 1.0f, 0.0f};
    float v4[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    gv_db_add_vector(db, v1, 4);
    gv_db_add_vector(db, v2, 4);
    gv_db_add_vector(db, v3, 4);
    gv_db_add_vector(db, v4, 4);
    return db;
}

/* ---------- test functions ---------- */

static int test_config_init(void) {
    GV_VacuumConfig config;
    memset(&config, 0xFF, sizeof(config));
    gv_vacuum_config_init(&config);

    /* After init, defaults should be set */
    ASSERT(config.min_deleted_count == 100, "default min_deleted_count == 100");
    ASSERT(config.batch_size == 1000, "default batch_size == 1000");
    ASSERT(config.priority == 0, "default priority == 0 (low)");
    ASSERT(config.interval_sec == 600, "default interval_sec == 600");
    ASSERT(config.min_fragmentation_ratio > 0.0, "default frag ratio > 0");
    return 0;
}

static int test_create_destroy(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create test db");

    GV_VacuumConfig config;
    gv_vacuum_config_init(&config);

    GV_VacuumManager *mgr = gv_vacuum_create(db, &config);
    ASSERT(mgr != NULL, "gv_vacuum_create returned NULL");
    ASSERT(gv_vacuum_get_state(mgr) == GV_VACUUM_IDLE, "initial state should be IDLE");

    gv_vacuum_destroy(mgr);
    gv_db_close(db);
    return 0;
}

static int test_manual_vacuum(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create test db");

    /* Delete some vectors to create fragmentation */
    gv_db_delete_vector_by_index(db, 1);
    gv_db_delete_vector_by_index(db, 3);

    GV_VacuumConfig config;
    gv_vacuum_config_init(&config);
    config.min_deleted_count = 1; /* lower threshold for test */

    GV_VacuumManager *mgr = gv_vacuum_create(db, &config);
    ASSERT(mgr != NULL, "create vacuum manager");

    int rc = gv_vacuum_run(mgr);
    ASSERT(rc == 0, "vacuum_run should succeed");

    gv_vacuum_destroy(mgr);
    gv_db_close(db);
    return 0;
}

static int test_get_fragmentation(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create test db");

    GV_VacuumConfig config;
    gv_vacuum_config_init(&config);
    GV_VacuumManager *mgr = gv_vacuum_create(db, &config);
    ASSERT(mgr != NULL, "create vacuum manager");

    /* No deletions, fragmentation should be 0 or near 0 */
    double frag = gv_vacuum_get_fragmentation(mgr);
    ASSERT(frag >= 0.0, "fragmentation should be >= 0");

    /* Delete a vector and check again */
    gv_db_delete_vector_by_index(db, 0);
    double frag2 = gv_vacuum_get_fragmentation(mgr);
    ASSERT(frag2 >= 0.0, "fragmentation after delete should be >= 0");

    gv_vacuum_destroy(mgr);
    gv_db_close(db);
    return 0;
}

static int test_get_stats(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create test db");

    GV_VacuumConfig config;
    gv_vacuum_config_init(&config);
    config.min_deleted_count = 1;

    GV_VacuumManager *mgr = gv_vacuum_create(db, &config);
    ASSERT(mgr != NULL, "create vacuum manager");

    GV_VacuumStats stats;
    memset(&stats, 0, sizeof(stats));
    int rc = gv_vacuum_get_stats(mgr, &stats);
    ASSERT(rc == 0, "get_stats should succeed");
    ASSERT(stats.state == GV_VACUUM_IDLE, "state should be IDLE before run");
    ASSERT(stats.total_runs == 0, "total_runs should be 0 initially");

    /* Delete and run vacuum, then check stats */
    gv_db_delete_vector_by_index(db, 0);
    gv_vacuum_run(mgr);

    rc = gv_vacuum_get_stats(mgr, &stats);
    ASSERT(rc == 0, "get_stats after run should succeed");

    gv_vacuum_destroy(mgr);
    gv_db_close(db);
    return 0;
}

static int test_auto_vacuum_start_stop(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create test db");

    GV_VacuumConfig config;
    gv_vacuum_config_init(&config);
    config.interval_sec = 3600; /* long interval so it does not run during test */

    GV_VacuumManager *mgr = gv_vacuum_create(db, &config);
    ASSERT(mgr != NULL, "create vacuum manager");

    int rc = gv_vacuum_start_auto(mgr);
    ASSERT(rc == 0, "start_auto should succeed");

    rc = gv_vacuum_stop_auto(mgr);
    ASSERT(rc == 0, "stop_auto should succeed");

    gv_vacuum_destroy(mgr);
    gv_db_close(db);
    return 0;
}

static int test_vacuum_with_null_config(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create test db");

    /* Passing NULL config should use defaults */
    GV_VacuumManager *mgr = gv_vacuum_create(db, NULL);
    ASSERT(mgr != NULL, "vacuum_create with NULL config should succeed");

    gv_vacuum_destroy(mgr);
    gv_db_close(db);
    return 0;
}

/* ---------- harness ---------- */

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing config init...", test_config_init},
        {"Testing create/destroy...", test_create_destroy},
        {"Testing manual vacuum...", test_manual_vacuum},
        {"Testing get fragmentation...", test_get_fragmentation},
        {"Testing get stats...", test_get_stats},
        {"Testing auto vacuum start/stop...", test_auto_vacuum_start_stop},
        {"Testing vacuum with NULL config...", test_vacuum_with_null_config},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
