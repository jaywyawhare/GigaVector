#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "features/sql.h"
#include "storage/database.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

#define DIM 4

static GV_Database *create_test_db(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    if (!db) return NULL;

    float v0[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v1[] = {0.0f, 1.0f, 0.0f, 0.0f};
    float v2[] = {0.0f, 0.0f, 1.0f, 0.0f};
    float v3[] = {0.5f, 0.5f, 0.5f, 0.5f};

    db_add_vector_with_metadata(db, v0, DIM, "category", "science");
    db_add_vector_with_metadata(db, v1, DIM, "category", "tech");
    db_add_vector_with_metadata(db, v2, DIM, "category", "science");
    db_add_vector_with_metadata(db, v3, DIM, "category", "tech");

    return db;
}

static int test_create_destroy(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db open should succeed");

    GV_SQLEngine *eng = sql_create(db);
    ASSERT(eng != NULL, "sql_create should return non-NULL");

    sql_destroy(eng);
    sql_destroy(NULL);

    db_close(db);
    return 0;
}

static int test_select_all(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create_test_db should succeed");

    GV_SQLEngine *eng = sql_create(db);
    ASSERT(eng != NULL, "sql engine create should succeed");

    GV_SQLResult result;
    memset(&result, 0, sizeof(result));
    int rc = sql_execute(eng, "SELECT * FROM vectors LIMIT 10", &result);
    ASSERT(rc == 0, "SELECT * LIMIT 10 should succeed");
    ASSERT(result.row_count <= 10, "result should have at most 10 rows");

    sql_free_result(&result);
    sql_destroy(eng);
    db_close(db);
    return 0;
}

static int test_ann_query(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create_test_db should succeed");

    GV_SQLEngine *eng = sql_create(db);
    ASSERT(eng != NULL, "sql engine create should succeed");

    GV_SQLResult result;
    memset(&result, 0, sizeof(result));
    int rc = sql_execute(eng,
        "SELECT * FROM vectors ANN(query=[1.0,0.0,0.0,0.0], k=3, metric=cosine)",
        &result);
    ASSERT(rc == 0, "ANN query should succeed");
    ASSERT(result.row_count >= 1, "ANN should return at least 1 result");
    ASSERT(result.row_count <= 3, "ANN should return at most k=3 results");

    sql_free_result(&result);
    sql_destroy(eng);
    db_close(db);
    return 0;
}

static int test_explain(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create_test_db should succeed");

    GV_SQLEngine *eng = sql_create(db);
    ASSERT(eng != NULL, "sql engine create should succeed");

    char plan[1024];
    memset(plan, 0, sizeof(plan));
    int rc = sql_explain(eng,
        "SELECT * FROM vectors ANN(query=[1.0,0.0,0.0,0.0], k=3)",
        plan, sizeof(plan));
    ASSERT(rc == 0, "explain should succeed");
    ASSERT(strlen(plan) > 0, "plan should be non-empty");

    sql_destroy(eng);
    db_close(db);
    return 0;
}

static int test_last_error(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create_test_db should succeed");

    GV_SQLEngine *eng = sql_create(db);
    ASSERT(eng != NULL, "sql engine create should succeed");

    GV_SQLResult result;
    memset(&result, 0, sizeof(result));
    int rc = sql_execute(eng, "THIS IS NOT VALID SQL AT ALL", &result);
    ASSERT(rc == -1, "invalid SQL should return -1");

    const char *err = sql_last_error(eng);
    ASSERT(err != NULL, "last_error should return non-NULL");
    ASSERT(strlen(err) > 0, "error message should be non-empty");

    sql_free_result(&result);
    sql_destroy(eng);
    db_close(db);
    return 0;
}

static int test_free_result_empty(void) {
    GV_SQLResult result;
    memset(&result, 0, sizeof(result));
    sql_free_result(&result);
    ASSERT(result.row_count == 0, "freed result should have row_count 0");
    return 0;
}

static int test_select_where(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create_test_db should succeed");

    GV_SQLEngine *eng = sql_create(db);
    ASSERT(eng != NULL, "sql engine create should succeed");

    GV_SQLResult result;
    memset(&result, 0, sizeof(result));
    int rc = sql_execute(eng,
        "SELECT * FROM vectors WHERE category = 'science' LIMIT 10",
        &result);
    ASSERT(rc == 0, "SELECT with WHERE should succeed");
    ASSERT(result.row_count <= 10, "should return at most LIMIT results");

    sql_free_result(&result);
    sql_destroy(eng);
    db_close(db);
    return 0;
}

static int test_select_projection_list(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create_test_db should succeed");

    GV_SQLEngine *eng = sql_create(db);
    ASSERT(eng != NULL, "sql engine create should succeed");

    GV_SQLResult result;
    memset(&result, 0, sizeof(result));
    int rc = sql_execute(eng,
        "SELECT category, score FROM vectors WHERE category = 'science' LIMIT 2",
        &result);
    ASSERT(rc == 0, "SELECT with projection list should parse and execute");
    ASSERT(result.row_count <= 2, "should respect LIMIT");

    sql_free_result(&result);
    sql_destroy(eng);
    db_close(db);
    return 0;
}

static int test_semicolon_and_not_equal_angle(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create_test_db should succeed");

    GV_SQLEngine *eng = sql_create(db);
    ASSERT(eng != NULL, "sql engine create should succeed");

    GV_SQLResult result;
    memset(&result, 0, sizeof(result));
    int rc = sql_execute(eng,
        "SELECT * FROM vectors WHERE category <> 'science' LIMIT 10;",
        &result);
    ASSERT(rc == 0, "SELECT with <> and trailing semicolon should succeed");
    ASSERT(result.row_count == 2, "two tech rows should match category <> science");

    sql_free_result(&result);
    sql_destroy(eng);
    db_close(db);
    return 0;
}

static int test_order_by_function_syntax(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create_test_db should succeed");

    GV_SQLEngine *eng = sql_create(db);
    ASSERT(eng != NULL, "sql engine create should succeed");

    GV_SQLResult result;
    memset(&result, 0, sizeof(result));
    int rc = sql_execute(eng,
        "SELECT * FROM vectors ANN(query=[1.0,0.0,0.0,0.0], k=3) "
        "ORDER BY vector_distance(query=[1.0,0.0,0.0,0.0]) DESC LIMIT 2",
        &result);
    ASSERT(rc == 0, "ORDER BY function-style expression should parse");
    ASSERT(result.row_count <= 2, "should respect LIMIT");

    sql_free_result(&result);
    sql_destroy(eng);
    db_close(db);
    return 0;
}

static int test_explain_trailing_semicolon(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create_test_db should succeed");

    GV_SQLEngine *eng = sql_create(db);
    ASSERT(eng != NULL, "sql engine create should succeed");

    char plan[1024];
    memset(plan, 0, sizeof(plan));
    int rc = sql_explain(eng,
        "SELECT * FROM vectors WHERE category = 'science';",
        plan, sizeof(plan));
    ASSERT(rc == 0, "EXPLAIN with trailing semicolon should succeed");
    ASSERT(strlen(plan) > 0, "plan should be non-empty");

    sql_destroy(eng);
    db_close(db);
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
        {"Testing sql SELECT projection list...", test_select_projection_list},
        {"Testing sql semicolon and <> ...", test_semicolon_and_not_equal_angle},
        {"Testing sql ORDER BY function syntax...", test_order_by_function_syntax},
        {"Testing sql EXPLAIN semicolon...", test_explain_trailing_semicolon},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
