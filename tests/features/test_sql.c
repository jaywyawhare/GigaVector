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

static int test_delete(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create_test_db should succeed");

    GV_SQLEngine *eng = sql_create(db);
    ASSERT(eng != NULL, "sql engine create should succeed");

    GV_SQLResult result;
    memset(&result, 0, sizeof(result));
    int rc = sql_execute(eng, "DELETE FROM vectors WHERE category = 'science'", &result);
    ASSERT(rc == 0, "DELETE should succeed");
    ASSERT(result.row_count == 1, "DELETE should return one summary row");
    ASSERT(result.indices[0] == 2, "should delete two science vectors");

    sql_free_result(&result);
    sql_destroy(eng);
    db_close(db);
    return 0;
}

static int test_update(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create_test_db should succeed");

    GV_SQLEngine *eng = sql_create(db);
    ASSERT(eng != NULL, "sql engine create should succeed");

    GV_SQLResult result;
    memset(&result, 0, sizeof(result));
    int rc = sql_execute(eng,
        "UPDATE vectors SET status = 'archived' WHERE category = 'tech'",
        &result);
    ASSERT(rc == 0, "UPDATE should succeed");
    ASSERT(result.row_count == 1, "UPDATE should return one summary row");
    ASSERT(result.indices[0] == 2, "should update two tech vectors");

    sql_free_result(&result);
    sql_destroy(eng);
    db_close(db);
    return 0;
}

static int test_projection_columns(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create_test_db should succeed");

    GV_SQLEngine *eng = sql_create(db);
    ASSERT(eng != NULL, "sql engine create should succeed");

    GV_SQLResult result;
    memset(&result, 0, sizeof(result));
    int rc = sql_execute(eng,
        "SELECT category FROM vectors WHERE category = 'science' LIMIT 10",
        &result);
    ASSERT(rc == 0, "projection SELECT should succeed");
    ASSERT(result.column_count == 1, "should have one projected column");
    ASSERT(result.column_names != NULL, "column names should be allocated");
    ASSERT(strcmp(result.column_names[0], "category") == 0, "column name should be category");
    ASSERT(result.row_count == 2, "two science rows should match");
    ASSERT(result.column_values != NULL, "column values should be allocated");
    ASSERT(strcmp(result.column_values[0], "science") == 0, "first projected value");

    sql_free_result(&result);
    sql_destroy(eng);
    db_close(db);
    return 0;
}

static int test_order_by_metadata(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create_test_db should succeed");

    GV_SQLEngine *eng = sql_create(db);
    ASSERT(eng != NULL, "sql engine create should succeed");

    GV_SQLResult result;
    memset(&result, 0, sizeof(result));
    int rc = sql_execute(eng,
        "SELECT category FROM vectors ORDER BY category DESC LIMIT 1",
        &result);
    ASSERT(rc == 0, "ORDER BY metadata should succeed");
    ASSERT(result.row_count == 1, "LIMIT 1 should return one row");
    ASSERT(result.column_values != NULL, "column values should be allocated");
    ASSERT(strcmp(result.column_values[0], "tech") == 0,
           "DESC on category should return tech first");

    sql_free_result(&result);
    sql_destroy(eng);
    db_close(db);
    return 0;
}

static int test_order_by_vector_distance_desc(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create_test_db should succeed");

    GV_SQLEngine *eng = sql_create(db);
    ASSERT(eng != NULL, "sql engine create should succeed");

    GV_SQLResult result;
    memset(&result, 0, sizeof(result));
    int rc = sql_execute(eng,
        "SELECT index, distance FROM vectors "
        "ANN(query=[1.0,0.0,0.0,0.0], k=4) "
        "ORDER BY vector_distance(query=[1.0,0.0,0.0,0.0]) DESC LIMIT 1",
        &result);
    ASSERT(rc == 0, "ORDER BY vector_distance DESC should succeed");
    ASSERT(result.row_count == 1, "LIMIT 1 should return one row");
    ASSERT(result.distances != NULL, "distances should be populated");
    ASSERT(result.column_count == 2, "should project index and distance");
    ASSERT(result.distances[0] >= 0.0f, "distance should be non-negative");

    sql_free_result(&result);
    sql_destroy(eng);
    db_close(db);
    return 0;
}

/* DB with a numeric "score" metadata field for aggregate/range tests. */
static GV_Database *create_scored_db(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    if (!db) return NULL;
    const char *scores[] = {"10", "20", "30", "40"};
    for (int i = 0; i < 4; i++) {
        float v[DIM] = {(float)i, 0.0f, 0.0f, 0.0f};
        db_add_vector_with_metadata(db, v, DIM, "score", scores[i]);
    }
    return db;
}

static int test_offset(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create_test_db should succeed");
    GV_SQLEngine *eng = sql_create(db);
    ASSERT(eng != NULL, "sql engine create should succeed");
    GV_SQLResult r;

    memset(&r, 0, sizeof(r));
    ASSERT(sql_execute(eng, "SELECT * FROM vectors LIMIT 2 OFFSET 1", &r) == 0, "LIMIT/OFFSET should succeed");
    ASSERT(r.row_count == 2, "LIMIT 2 OFFSET 1 should yield 2 rows");
    sql_free_result(&r);

    memset(&r, 0, sizeof(r));
    ASSERT(sql_execute(eng, "SELECT * FROM vectors OFFSET 3", &r) == 0, "OFFSET 3 should succeed");
    ASSERT(r.row_count == 1, "OFFSET 3 of 4 rows should yield 1 row");
    sql_free_result(&r);

    memset(&r, 0, sizeof(r));
    ASSERT(sql_execute(eng, "SELECT * FROM vectors OFFSET 10", &r) == 0, "OFFSET past end should succeed");
    ASSERT(r.row_count == 0, "OFFSET beyond count should yield 0 rows");
    sql_free_result(&r);

    sql_destroy(eng); db_close(db);
    return 0;
}

static int test_where_in(void) {
    GV_Database *db = create_test_db();
    GV_SQLEngine *eng = sql_create(db);
    ASSERT(db && eng, "setup");
    GV_SQLResult r;

    memset(&r, 0, sizeof(r));
    ASSERT(sql_execute(eng, "SELECT COUNT(*) FROM vectors WHERE category IN ('science','tech')", &r) == 0, "IN should succeed");
    ASSERT(r.indices && r.indices[0] == 4, "IN two categories matches all 4");
    sql_free_result(&r);

    memset(&r, 0, sizeof(r));
    ASSERT(sql_execute(eng, "SELECT COUNT(*) FROM vectors WHERE category IN ('science')", &r) == 0, "IN one should succeed");
    ASSERT(r.indices && r.indices[0] == 2, "IN ('science') matches 2");
    sql_free_result(&r);

    memset(&r, 0, sizeof(r));
    ASSERT(sql_execute(eng, "SELECT COUNT(*) FROM vectors WHERE category IN ('nope')", &r) == 0, "IN none should succeed");
    ASSERT(r.indices && r.indices[0] == 0, "IN ('nope') matches 0");
    sql_free_result(&r);

    sql_destroy(eng); db_close(db);
    return 0;
}

static int test_where_between(void) {
    GV_Database *db = create_scored_db();
    GV_SQLEngine *eng = sql_create(db);
    ASSERT(db && eng, "setup");
    GV_SQLResult r;

    memset(&r, 0, sizeof(r));
    ASSERT(sql_execute(eng, "SELECT COUNT(*) FROM vectors WHERE score BETWEEN 20 AND 30", &r) == 0, "BETWEEN should succeed");
    ASSERT(r.indices && r.indices[0] == 2, "score BETWEEN 20 AND 30 matches 2 (20,30)");
    sql_free_result(&r);

    memset(&r, 0, sizeof(r));
    ASSERT(sql_execute(eng, "SELECT COUNT(*) FROM vectors WHERE score BETWEEN 100 AND 200", &r) == 0, "BETWEEN empty should succeed");
    ASSERT(r.indices && r.indices[0] == 0, "BETWEEN out of range matches 0");
    sql_free_result(&r);

    sql_destroy(eng); db_close(db);
    return 0;
}

static int test_where_is_null(void) {
    GV_Database *db = create_test_db();
    /* add one vector with no category metadata */
    float vn[DIM] = {9.0f, 9.0f, 9.0f, 9.0f};
    db_add_vector(db, vn, DIM);
    GV_SQLEngine *eng = sql_create(db);
    ASSERT(db && eng, "setup");
    GV_SQLResult r;

    memset(&r, 0, sizeof(r));
    ASSERT(sql_execute(eng, "SELECT COUNT(*) FROM vectors WHERE category IS NULL", &r) == 0, "IS NULL should succeed");
    ASSERT(r.indices && r.indices[0] == 1, "one row lacks category");
    sql_free_result(&r);

    memset(&r, 0, sizeof(r));
    ASSERT(sql_execute(eng, "SELECT COUNT(*) FROM vectors WHERE category IS NOT NULL", &r) == 0, "IS NOT NULL should succeed");
    ASSERT(r.indices && r.indices[0] == 4, "four rows have category");
    sql_free_result(&r);

    sql_destroy(eng); db_close(db);
    return 0;
}

static int test_aggregates(void) {
    GV_Database *db = create_scored_db();
    GV_SQLEngine *eng = sql_create(db);
    ASSERT(db && eng, "setup");
    GV_SQLResult r;

    struct { const char *q; double want; } cases[] = {
        {"SELECT SUM(score) FROM vectors", 100.0},
        {"SELECT MIN(score) FROM vectors", 10.0},
        {"SELECT MAX(score) FROM vectors", 40.0},
        {"SELECT AVG(score) FROM vectors", 25.0},
    };
    for (int i = 0; i < 4; i++) {
        memset(&r, 0, sizeof(r));
        ASSERT(sql_execute(eng, cases[i].q, &r) == 0, "aggregate should succeed");
        ASSERT(r.row_count == 1 && r.column_count == 1, "aggregate is single-cell");
        ASSERT(r.column_values && r.column_values[0], "aggregate value present");
        double got = atof(r.column_values[0]);
        ASSERT(got == cases[i].want, "aggregate value matches expected");
        sql_free_result(&r);
    }

    /* aggregate with WHERE filter */
    memset(&r, 0, sizeof(r));
    ASSERT(sql_execute(eng, "SELECT SUM(score) FROM vectors WHERE score >= 30", &r) == 0, "filtered SUM should succeed");
    ASSERT(r.column_values && atof(r.column_values[0]) == 70.0, "SUM(score) where score>=30 == 70");
    sql_free_result(&r);

    sql_destroy(eng); db_close(db);
    return 0;
}

static int test_insert(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    GV_SQLEngine *eng = sql_create(db);
    ASSERT(db && eng, "setup");
    GV_SQLResult r;

    /* INSERT with metadata column */
    memset(&r, 0, sizeof(r));
    ASSERT(sql_execute(eng,
        "INSERT INTO vectors (vector, category) VALUES ([1.0,0.0,0.0,0.0], 'science')", &r) == 0,
        "INSERT with metadata should succeed");
    ASSERT(r.indices && r.indices[0] == 1, "one row inserted");
    sql_free_result(&r);

    /* bare vector INSERT (no metadata) */
    memset(&r, 0, sizeof(r));
    ASSERT(sql_execute(eng, "INSERT INTO vectors VALUES ([0.0,1.0,0.0,0.0])", &r) == 0,
        "bare INSERT should succeed");
    sql_free_result(&r);

    /* verify count and that the metadata was stored */
    memset(&r, 0, sizeof(r));
    ASSERT(sql_execute(eng, "SELECT COUNT(*) FROM vectors", &r) == 0, "count should succeed");
    ASSERT(r.indices && r.indices[0] == 2, "two rows after two inserts");
    sql_free_result(&r);

    memset(&r, 0, sizeof(r));
    ASSERT(sql_execute(eng, "SELECT COUNT(*) FROM vectors WHERE category = 'science'", &r) == 0, "filter should succeed");
    ASSERT(r.indices && r.indices[0] == 1, "inserted metadata is queryable");
    sql_free_result(&r);

    /* dimension mismatch is rejected */
    memset(&r, 0, sizeof(r));
    ASSERT(sql_execute(eng, "INSERT INTO vectors VALUES ([1.0,2.0])", &r) != 0,
        "wrong-dimension INSERT should fail");

    sql_destroy(eng); db_close(db);
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
        {"Testing sql DELETE...", test_delete},
        {"Testing sql UPDATE...", test_update},
        {"Testing sql projection columns...", test_projection_columns},
        {"Testing sql ORDER BY metadata...", test_order_by_metadata},
        {"Testing sql ORDER BY vector_distance DESC...", test_order_by_vector_distance_desc},
        {"Testing sql OFFSET...", test_offset},
        {"Testing sql WHERE IN...", test_where_in},
        {"Testing sql WHERE BETWEEN...", test_where_between},
        {"Testing sql WHERE IS NULL...", test_where_is_null},
        {"Testing sql aggregates SUM/MIN/MAX/AVG...", test_aggregates},
        {"Testing sql INSERT...", test_insert},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
