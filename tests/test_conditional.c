#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_conditional.h"
#include "gigavector/gv_database.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

/* Helper: create a database with some vectors and metadata */
static GV_Database *create_test_db(void) {
    GV_Database *db = gv_db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    if (!db) return NULL;

    float v0[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v1[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    float v2[4] = {0.0f, 0.0f, 1.0f, 0.0f};
    gv_db_add_vector_with_metadata(db, v0, 4, "model", "v1");
    gv_db_add_vector_with_metadata(db, v1, 4, "model", "v1");
    gv_db_add_vector_with_metadata(db, v2, 4, "model", "v2");
    return db;
}

/* ---------- test functions ---------- */

static int test_create_destroy(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create test db");

    GV_CondManager *mgr = gv_cond_create((void *)db);
    ASSERT(mgr != NULL, "gv_cond_create returned NULL");

    gv_cond_destroy(mgr);
    gv_db_close(db);
    return 0;
}

static int test_unconditional_update(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create test db");
    GV_CondManager *mgr = gv_cond_create((void *)db);
    ASSERT(mgr != NULL, "create cond manager");

    /* Update vector 0 with no conditions */
    float new_data[4] = {9.0f, 8.0f, 7.0f, 6.0f};
    GV_ConditionalResult res = gv_cond_update_vector(mgr, 0, new_data, 4, NULL, 0);
    ASSERT(res == GV_COND_OK, "unconditional update should succeed");

    /* Version should now be 1 */
    uint64_t ver = gv_cond_get_version(mgr, 0);
    ASSERT(ver == 1, "version should be 1 after first update");

    gv_cond_destroy(mgr);
    gv_db_close(db);
    return 0;
}

static int test_version_eq_condition(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create test db");
    GV_CondManager *mgr = gv_cond_create((void *)db);
    ASSERT(mgr != NULL, "create cond manager");

    /* First unconditional update to set version to 1 */
    float d1[4] = {2.0f, 2.0f, 2.0f, 2.0f};
    gv_cond_update_vector(mgr, 0, d1, 4, NULL, 0);

    /* Conditional update requiring version == 1 (should pass) */
    GV_Condition cond;
    cond.type = GV_COND_VERSION_EQ;
    cond.version = 1;
    cond.field_name = NULL;
    cond.field_value = NULL;

    float d2[4] = {3.0f, 3.0f, 3.0f, 3.0f};
    GV_ConditionalResult res = gv_cond_update_vector(mgr, 0, d2, 4, &cond, 1);
    ASSERT(res == GV_COND_OK, "version_eq with correct version should succeed");
    ASSERT(gv_cond_get_version(mgr, 0) == 2, "version should be 2");

    /* Now try with stale version (should fail) */
    cond.version = 1; /* stale */
    float d3[4] = {4.0f, 4.0f, 4.0f, 4.0f};
    res = gv_cond_update_vector(mgr, 0, d3, 4, &cond, 1);
    ASSERT(res == GV_COND_FAILED || res == GV_COND_CONFLICT,
           "version_eq with stale version should fail");

    gv_cond_destroy(mgr);
    gv_db_close(db);
    return 0;
}

static int test_conditional_delete(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create test db");
    GV_CondManager *mgr = gv_cond_create((void *)db);
    ASSERT(mgr != NULL, "create cond manager");

    /* Set up: unconditional update to establish version */
    float d[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    gv_cond_update_vector(mgr, 1, d, 4, NULL, 0);

    /* Conditional delete with NOT_DELETED condition */
    GV_Condition cond;
    cond.type = GV_COND_NOT_DELETED;
    cond.field_name = NULL;
    cond.field_value = NULL;
    cond.version = 0;

    GV_ConditionalResult res = gv_cond_delete(mgr, 1, &cond, 1);
    ASSERT(res == GV_COND_OK, "conditional delete should succeed");

    gv_cond_destroy(mgr);
    gv_db_close(db);
    return 0;
}

static int test_conditional_metadata_update(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create test db");
    GV_CondManager *mgr = gv_cond_create((void *)db);
    ASSERT(mgr != NULL, "create cond manager");

    /* Update metadata with no conditions */
    GV_ConditionalResult res = gv_cond_update_metadata(mgr, 0, "status", "active", NULL, 0);
    ASSERT(res == GV_COND_OK, "unconditional metadata update should succeed");

    gv_cond_destroy(mgr);
    gv_db_close(db);
    return 0;
}

static int test_not_found(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create test db");
    GV_CondManager *mgr = gv_cond_create((void *)db);
    ASSERT(mgr != NULL, "create cond manager");

    /* Try to update a vector index that does not exist */
    float d[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    GV_ConditionalResult res = gv_cond_update_vector(mgr, 999, d, 4, NULL, 0);
    ASSERT(res == GV_COND_NOT_FOUND, "update on nonexistent index should return NOT_FOUND");

    gv_cond_destroy(mgr);
    gv_db_close(db);
    return 0;
}

static int test_batch_update(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create test db");
    GV_CondManager *mgr = gv_cond_create((void *)db);
    ASSERT(mgr != NULL, "create cond manager");

    size_t indices[2] = {0, 1};
    float new0[4] = {10.0f, 10.0f, 10.0f, 10.0f};
    float new1[4] = {20.0f, 20.0f, 20.0f, 20.0f};
    const float *vectors[2] = {new0, new1};
    const GV_Condition *conditions[2] = {NULL, NULL};
    size_t condition_counts[2] = {0, 0};
    GV_ConditionalResult results[2];

    int updated = gv_cond_batch_update(mgr, indices, vectors, conditions,
                                        condition_counts, 2, results);
    ASSERT(updated == 2, "batch_update should update 2 vectors");
    ASSERT(results[0] == GV_COND_OK, "batch result[0] should be OK");
    ASSERT(results[1] == GV_COND_OK, "batch result[1] should be OK");

    gv_cond_destroy(mgr);
    gv_db_close(db);
    return 0;
}

static int test_migrate_embedding(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create test db");
    GV_CondManager *mgr = gv_cond_create((void *)db);
    ASSERT(mgr != NULL, "create cond manager");

    /* Establish version by doing an update */
    float d1[4] = {5.0f, 5.0f, 5.0f, 5.0f};
    gv_cond_update_vector(mgr, 0, d1, 4, NULL, 0);
    ASSERT(gv_cond_get_version(mgr, 0) == 1, "version should be 1");

    /* Migrate embedding expecting version 1 */
    float new_emb[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    GV_ConditionalResult res = gv_cond_migrate_embedding(mgr, 0, new_emb, 4, 1);
    ASSERT(res == GV_COND_OK, "migrate_embedding with correct version should succeed");
    ASSERT(gv_cond_get_version(mgr, 0) == 2, "version should be 2 after migration");

    /* Retry with stale version should fail */
    float stale_emb[4] = {0.5f, 0.6f, 0.7f, 0.8f};
    res = gv_cond_migrate_embedding(mgr, 0, stale_emb, 4, 1);
    ASSERT(res == GV_COND_FAILED || res == GV_COND_CONFLICT,
           "migrate with stale version should fail");

    gv_cond_destroy(mgr);
    gv_db_close(db);
    return 0;
}

/* ---------- harness ---------- */

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing create/destroy...", test_create_destroy},
        {"Testing unconditional update...", test_unconditional_update},
        {"Testing version_eq condition...", test_version_eq_condition},
        {"Testing conditional delete...", test_conditional_delete},
        {"Testing conditional metadata update...", test_conditional_metadata_update},
        {"Testing not found...", test_not_found},
        {"Testing batch update...", test_batch_update},
        {"Testing migrate embedding...", test_migrate_embedding},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        printf("%s", tests[i].name);
        if (tests[i].fn() == 0) { printf(" [OK]\n"); passed++; }
        else { printf(" [FAIL]\n"); }
    }
    printf("\n%d/%d tests passed\n", passed, n);
    return passed == n ? 0 : 1;
}
