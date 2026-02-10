#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_payload_index.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_payload_index_create_destroy(void) {
    GV_PayloadIndex *idx = gv_payload_index_create();
    ASSERT(idx != NULL, "payload index creation should succeed");

    gv_payload_index_destroy(idx);

    /* Destroying NULL should be safe */
    gv_payload_index_destroy(NULL);
    return 0;
}

static int test_payload_index_add_remove_field(void) {
    GV_PayloadIndex *idx = gv_payload_index_create();
    ASSERT(idx != NULL, "payload index creation");

    int rc = gv_payload_index_add_field(idx, "age", GV_FIELD_INT);
    ASSERT(rc == 0, "adding int field should succeed");

    rc = gv_payload_index_add_field(idx, "name", GV_FIELD_STRING);
    ASSERT(rc == 0, "adding string field should succeed");

    ASSERT(gv_payload_index_field_count(idx) == 2, "should have 2 fields");

    rc = gv_payload_index_remove_field(idx, "age");
    ASSERT(rc == 0, "removing field should succeed");
    ASSERT(gv_payload_index_field_count(idx) == 1, "should have 1 field after removal");

    gv_payload_index_destroy(idx);
    return 0;
}

static int test_payload_index_insert_int(void) {
    GV_PayloadIndex *idx = gv_payload_index_create();
    ASSERT(idx != NULL, "payload index creation");

    gv_payload_index_add_field(idx, "score", GV_FIELD_INT);

    int rc = gv_payload_index_insert_int(idx, 0, "score", 100);
    ASSERT(rc == 0, "inserting int value for vector 0 should succeed");

    rc = gv_payload_index_insert_int(idx, 1, "score", 200);
    ASSERT(rc == 0, "inserting int value for vector 1 should succeed");

    rc = gv_payload_index_insert_int(idx, 2, "score", 50);
    ASSERT(rc == 0, "inserting int value for vector 2 should succeed");

    ASSERT(gv_payload_index_total_entries(idx) == 3, "should have 3 entries");

    gv_payload_index_destroy(idx);
    return 0;
}

static int test_payload_index_query_eq(void) {
    GV_PayloadIndex *idx = gv_payload_index_create();
    ASSERT(idx != NULL, "payload index creation");

    gv_payload_index_add_field(idx, "category", GV_FIELD_STRING);

    gv_payload_index_insert_string(idx, 0, "category", "sports");
    gv_payload_index_insert_string(idx, 1, "category", "tech");
    gv_payload_index_insert_string(idx, 2, "category", "sports");
    gv_payload_index_insert_string(idx, 3, "category", "music");

    GV_PayloadQuery query;
    memset(&query, 0, sizeof(query));
    query.field_name = "category";
    query.op = GV_PAYLOAD_OP_EQ;
    query.value.string_val = "sports";
    query.field_type = GV_FIELD_STRING;

    size_t results[10];
    int count = gv_payload_index_query(idx, &query, results, 10);
    ASSERT(count == 2, "EQ query for 'sports' should return 2 results");

    gv_payload_index_destroy(idx);
    return 0;
}

static int test_payload_index_query_range(void) {
    GV_PayloadIndex *idx = gv_payload_index_create();
    ASSERT(idx != NULL, "payload index creation");

    gv_payload_index_add_field(idx, "price", GV_FIELD_FLOAT);

    gv_payload_index_insert_float(idx, 0, "price", 9.99);
    gv_payload_index_insert_float(idx, 1, "price", 29.99);
    gv_payload_index_insert_float(idx, 2, "price", 49.99);
    gv_payload_index_insert_float(idx, 3, "price", 99.99);

    GV_PayloadQuery query;
    memset(&query, 0, sizeof(query));
    query.field_name = "price";
    query.op = GV_PAYLOAD_OP_LT;
    query.value.float_val = 50.0;
    query.field_type = GV_FIELD_FLOAT;

    size_t results[10];
    int count = gv_payload_index_query(idx, &query, results, 10);
    ASSERT(count == 3, "LT 50.0 query should return 3 results");

    gv_payload_index_destroy(idx);
    return 0;
}

static int test_payload_index_query_multi(void) {
    GV_PayloadIndex *idx = gv_payload_index_create();
    ASSERT(idx != NULL, "payload index creation");

    gv_payload_index_add_field(idx, "age", GV_FIELD_INT);
    gv_payload_index_add_field(idx, "active", GV_FIELD_BOOL);

    gv_payload_index_insert_int(idx, 0, "age", 25);
    gv_payload_index_insert_bool(idx, 0, "active", 1);

    gv_payload_index_insert_int(idx, 1, "age", 30);
    gv_payload_index_insert_bool(idx, 1, "active", 0);

    gv_payload_index_insert_int(idx, 2, "age", 22);
    gv_payload_index_insert_bool(idx, 2, "active", 1);

    GV_PayloadQuery queries[2];
    memset(queries, 0, sizeof(queries));

    queries[0].field_name = "age";
    queries[0].op = GV_PAYLOAD_OP_LT;
    queries[0].value.int_val = 30;
    queries[0].field_type = GV_FIELD_INT;

    queries[1].field_name = "active";
    queries[1].op = GV_PAYLOAD_OP_EQ;
    queries[1].value.bool_val = 1;
    queries[1].field_type = GV_FIELD_BOOL;

    size_t results[10];
    int count = gv_payload_index_query_multi(idx, queries, 2, results, 10);
    ASSERT(count == 2, "multi query (age<30 AND active=1) should return 2 results");

    gv_payload_index_destroy(idx);
    return 0;
}

static int test_payload_index_remove_entry(void) {
    GV_PayloadIndex *idx = gv_payload_index_create();
    ASSERT(idx != NULL, "payload index creation");

    gv_payload_index_add_field(idx, "tag", GV_FIELD_STRING);

    gv_payload_index_insert_string(idx, 0, "tag", "alpha");
    gv_payload_index_insert_string(idx, 1, "tag", "beta");
    ASSERT(gv_payload_index_total_entries(idx) == 2, "should have 2 entries");

    int rc = gv_payload_index_remove(idx, 0);
    ASSERT(rc == 0, "removing vector 0 should succeed");
    ASSERT(gv_payload_index_total_entries(idx) == 1, "should have 1 entry after removal");

    gv_payload_index_destroy(idx);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing payload index create/destroy...", test_payload_index_create_destroy},
        {"Testing payload index add/remove field...", test_payload_index_add_remove_field},
        {"Testing payload index insert int...", test_payload_index_insert_int},
        {"Testing payload index query EQ...", test_payload_index_query_eq},
        {"Testing payload index query range...", test_payload_index_query_range},
        {"Testing payload index query multi...", test_payload_index_query_multi},
        {"Testing payload index remove entry...", test_payload_index_remove_entry},
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
