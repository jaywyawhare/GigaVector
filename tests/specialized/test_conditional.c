#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "specialized/conditional.h"
#include "storage/database.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static GV_Database *create_test_db(void) {
    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    if (!db) return NULL;

    float v0[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v1[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    float v2[4] = {0.0f, 0.0f, 1.0f, 0.0f};
    db_add_vector_with_metadata(db, v0, 4, "model", "v1");
    db_add_vector_with_metadata(db, v1, 4, "model", "v1");
    db_add_vector_with_metadata(db, v2, 4, "model", "v2");
    return db;
}

static int test_create_destroy(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create test db");

    GV_CondManager *mgr = cond_create((void *)db);
    ASSERT(mgr != NULL, "cond_create returned NULL");

    cond_destroy(mgr);
    db_close(db);
    return 0;
}

static int test_unconditional_update(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create test db");
    GV_CondManager *mgr = cond_create((void *)db);
    ASSERT(mgr != NULL, "create cond manager");

    float new_data[4] = {9.0f, 8.0f, 7.0f, 6.0f};
    GV_ConditionalResult res = cond_update_vector(mgr, 0, new_data, 4, NULL, 0);
    ASSERT(res == GV_COND_OK, "unconditional update should succeed");

    uint64_t ver = cond_get_version(mgr, 0);
    ASSERT(ver == 1, "version should be 1 after first update");

    cond_destroy(mgr);
    db_close(db);
    return 0;
}

static int test_version_eq_condition(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create test db");
    GV_CondManager *mgr = cond_create((void *)db);
    ASSERT(mgr != NULL, "create cond manager");

    float d1[4] = {2.0f, 2.0f, 2.0f, 2.0f};
    cond_update_vector(mgr, 0, d1, 4, NULL, 0);

    GV_Condition cond;
    cond.type = GV_COND_VERSION_EQ;
    cond.version = 1;
    cond.field_name = NULL;
    cond.field_value = NULL;

    float d2[4] = {3.0f, 3.0f, 3.0f, 3.0f};
    GV_ConditionalResult res = cond_update_vector(mgr, 0, d2, 4, &cond, 1);
    ASSERT(res == GV_COND_OK, "version_eq with correct version should succeed");
    ASSERT(cond_get_version(mgr, 0) == 2, "version should be 2");

    cond.version = 1;
    float d3[4] = {4.0f, 4.0f, 4.0f, 4.0f};
    res = cond_update_vector(mgr, 0, d3, 4, &cond, 1);
    ASSERT(res == GV_COND_FAILED || res == GV_COND_CONFLICT,
           "version_eq with stale version should fail");

    cond_destroy(mgr);
    db_close(db);
    return 0;
}

static int test_conditional_delete(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create test db");
    GV_CondManager *mgr = cond_create((void *)db);
    ASSERT(mgr != NULL, "create cond manager");

    float d[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    cond_update_vector(mgr, 1, d, 4, NULL, 0);

    GV_Condition cond;
    cond.type = GV_COND_NOT_DELETED;
    cond.field_name = NULL;
    cond.field_value = NULL;
    cond.version = 0;

    GV_ConditionalResult res = cond_delete(mgr, 1, &cond, 1);
    ASSERT(res == GV_COND_OK, "conditional delete should succeed");

    cond_destroy(mgr);
    db_close(db);
    return 0;
}

static int test_conditional_metadata_update(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create test db");
    GV_CondManager *mgr = cond_create((void *)db);
    ASSERT(mgr != NULL, "create cond manager");

    GV_ConditionalResult res = cond_update_metadata(mgr, 0, "status", "active", NULL, 0);
    ASSERT(res == GV_COND_OK, "unconditional metadata update should succeed");

    cond_destroy(mgr);
    db_close(db);
    return 0;
}

static int test_not_found(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create test db");
    GV_CondManager *mgr = cond_create((void *)db);
    ASSERT(mgr != NULL, "create cond manager");

    float d[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    GV_ConditionalResult res = cond_update_vector(mgr, 999, d, 4, NULL, 0);
    ASSERT(res == GV_COND_NOT_FOUND, "update on nonexistent index should return NOT_FOUND");

    cond_destroy(mgr);
    db_close(db);
    return 0;
}

static int test_batch_update(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create test db");
    GV_CondManager *mgr = cond_create((void *)db);
    ASSERT(mgr != NULL, "create cond manager");

    size_t indices[2] = {0, 1};
    float new0[4] = {10.0f, 10.0f, 10.0f, 10.0f};
    float new1[4] = {20.0f, 20.0f, 20.0f, 20.0f};
    const float *vectors[2] = {new0, new1};
    const GV_Condition *conditions[2] = {NULL, NULL};
    size_t condition_counts[2] = {0, 0};
    GV_ConditionalResult results[2];

    int updated = cond_batch_update(mgr, indices, vectors, conditions,
                                        condition_counts, 2, results);
    ASSERT(updated == 2, "batch_update should update 2 vectors");
    ASSERT(results[0] == GV_COND_OK, "batch result[0] should be OK");
    ASSERT(results[1] == GV_COND_OK, "batch result[1] should be OK");

    cond_destroy(mgr);
    db_close(db);
    return 0;
}

static int test_migrate_embedding(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create test db");
    GV_CondManager *mgr = cond_create((void *)db);
    ASSERT(mgr != NULL, "create cond manager");

    float d1[4] = {5.0f, 5.0f, 5.0f, 5.0f};
    cond_update_vector(mgr, 0, d1, 4, NULL, 0);
    ASSERT(cond_get_version(mgr, 0) == 1, "version should be 1");

    float new_emb[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    GV_ConditionalResult res = cond_migrate_embedding(mgr, 0, new_emb, 4, 1);
    ASSERT(res == GV_COND_OK, "migrate_embedding with correct version should succeed");
    ASSERT(cond_get_version(mgr, 0) == 2, "version should be 2 after migration");

    float stale_emb[4] = {0.5f, 0.6f, 0.7f, 0.8f};
    res = cond_migrate_embedding(mgr, 0, stale_emb, 4, 1);
    ASSERT(res == GV_COND_FAILED || res == GV_COND_CONFLICT,
           "migrate with stale version should fail");

    cond_destroy(mgr);
    db_close(db);
    return 0;
}

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
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
