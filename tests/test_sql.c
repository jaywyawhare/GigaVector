#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_sql.h"
#include "gigavector/gv_database.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

#define DIM 4

/* Helper: create and populate a test database */
static GV_Database *create_test_db(void) {
    GV_Database *db = gv_db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    if (!db) return NULL;

    float v0[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v1[] = {0.0f, 1.0f, 0.0f, 0.0f};
    float v2[] = {0.0f, 0.0f, 1.0f, 0.0f};
    float v3[] = {0.5f, 0.5f, 0.5f, 0.5f};

    gv_db_add_vector_with_metadata(db, v0, DIM, "category", "science");
    gv_db_add_vector_with_metadata(db, v1, DIM, "category", "tech");
    gv_db_add_vector_with_metadata(db, v2, DIM, "category", "science");
    gv_db_add_vector_with_metadata(db, v3, DIM, "category", "tech");

    return db;
}

/* --- Test: create and destroy engine --- */
static int test_create_destroy(void) {
    GV_Database *db = gv_db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db open should succeed");

    GV_SQLEngine *eng = gv_sql_create(db);
    ASSERT(eng != NULL, "gv_sql_create should return non-NULL");

    gv_sql_destroy(eng);
    /* NULL destroy should be safe */
    gv_sql_destroy(NULL);

    gv_db_close(db);
    return 0;
}

/* --- Test: execute simple SELECT --- */
static int test_select_all(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create_test_db should succeed");

    GV_SQLEngine *eng = gv_sql_create(db);
    ASSERT(eng != NULL, "sql engine create should succeed");

    GV_SQLResult result;
    memset(&result, 0, sizeof(result));
    int rc = gv_sql_execute(eng, "SELECT * FROM vectors LIMIT 10", &result);
    ASSERT(rc == 0, "SELECT * LIMIT 10 should succeed");
    ASSERT(result.row_count <= 10, "result should have at most 10 rows");

    gv_sql_free_result(&result);
    gv_sql_destroy(eng);
    gv_db_close(db);
    return 0;
}

/* --- Test: ANN query --- */
static int test_ann_query(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create_test_db should succeed");

    GV_SQLEngine *eng = gv_sql_create(db);
    ASSERT(eng != NULL, "sql engine create should succeed");

    GV_SQLResult result;
    memset(&result, 0, sizeof(result));
    int rc = gv_sql_execute(eng,
        "SELECT * FROM vectors ANN(query=[1.0,0.0,0.0,0.0], k=3, metric=cosine)",
        &result);
    ASSERT(rc == 0, "ANN query should succeed");
    ASSERT(result.row_count >= 1, "ANN should return at least 1 result");
    ASSERT(result.row_count <= 3, "ANN should return at most k=3 results");

    gv_sql_free_result(&result);
    gv_sql_destroy(eng);
    gv_db_close(db);
    return 0;
}

/* --- Test: explain query plan --- */
static int test_explain(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create_test_db should succeed");

    GV_SQLEngine *eng = gv_sql_create(db);
    ASSERT(eng != NULL, "sql engine create should succeed");

    char plan[1024];
    memset(plan, 0, sizeof(plan));
    int rc = gv_sql_explain(eng,
        "SELECT * FROM vectors ANN(query=[1.0,0.0,0.0,0.0], k=3)",
        plan, sizeof(plan));
    ASSERT(rc == 0, "explain should succeed");
    ASSERT(strlen(plan) > 0, "plan should be non-empty");

    gv_sql_destroy(eng);
    gv_db_close(db);
    return 0;
}

/* --- Test: last error on invalid query --- */
static int test_last_error(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create_test_db should succeed");

    GV_SQLEngine *eng = gv_sql_create(db);
    ASSERT(eng != NULL, "sql engine create should succeed");

    GV_SQLResult result;
    memset(&result, 0, sizeof(result));
    int rc = gv_sql_execute(eng, "THIS IS NOT VALID SQL AT ALL", &result);
    ASSERT(rc == -1, "invalid SQL should return -1");

    const char *err = gv_sql_last_error(eng);
    ASSERT(err != NULL, "last_error should return non-NULL");
    ASSERT(strlen(err) > 0, "error message should be non-empty");

    gv_sql_free_result(&result);
    gv_sql_destroy(eng);
    gv_db_close(db);
    return 0;
}

/* --- Test: free_result on zeroed struct --- */
static int test_free_result_empty(void) {
    GV_SQLResult result;
    memset(&result, 0, sizeof(result));
    /* Should be safe on an already-zeroed / empty result */
    gv_sql_free_result(&result);
    ASSERT(result.row_count == 0, "freed result should have row_count 0");
    return 0;
}

/* --- Test: SELECT with WHERE filter --- */
static int test_select_where(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create_test_db should succeed");

    GV_SQLEngine *eng = gv_sql_create(db);
    ASSERT(eng != NULL, "sql engine create should succeed");

    GV_SQLResult result;
    memset(&result, 0, sizeof(result));
    int rc = gv_sql_execute(eng,
        "SELECT * FROM vectors WHERE category = 'science' LIMIT 10",
        &result);
    ASSERT(rc == 0, "SELECT with WHERE should succeed");
    /* We inserted 2 science vectors */
    ASSERT(result.row_count <= 10, "should return at most LIMIT results");

    gv_sql_free_result(&result);
    gv_sql_destroy(eng);
    gv_db_close(db);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing sql create/destroy...", test_create_destroy},
        {"Testing sql SELECT all...", test_select_all},
        {"Testing sql ANN query...", test_ann_query},
        {"Testing sql explain...", test_explain},
        {"Testing sql last error...", test_last_error},
        {"Testing sql free_result empty...", test_free_result_empty},
        {"Testing sql SELECT WHERE...", test_select_where},
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
