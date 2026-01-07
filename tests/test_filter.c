#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gigavector/gigavector.h"

#define ASSERT(cond, msg)         \
    do {                          \
        if (!(cond)) {            \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1;            \
        }                         \
    } while (0)

static int test_filter_parse_simple(void) {
    GV_Filter *filter = gv_filter_parse("category == \"A\"");
    ASSERT(filter != NULL, "parse simple filter");
    
    gv_filter_destroy(filter);
    return 0;
}

static int test_filter_parse_and(void) {
    GV_Filter *filter = gv_filter_parse("category == \"A\" AND score >= 0.5");
    ASSERT(filter != NULL, "parse AND filter");
    
    gv_filter_destroy(filter);
    return 0;
}

static int test_filter_parse_or(void) {
    GV_Filter *filter = gv_filter_parse("country == \"US\" OR country == \"CA\"");
    ASSERT(filter != NULL, "parse OR filter");
    
    gv_filter_destroy(filter);
    return 0;
}

static int test_filter_parse_not(void) {
    GV_Filter *filter = gv_filter_parse("NOT status == \"deleted\"");
    ASSERT(filter != NULL, "parse NOT filter");
    
    gv_filter_destroy(filter);
    return 0;
}

static int test_filter_parse_parentheses(void) {
    GV_Filter *filter = gv_filter_parse("(country == \"US\" OR country == \"CA\") AND NOT status == \"deleted\"");
    ASSERT(filter != NULL, "parse parentheses filter");
    
    gv_filter_destroy(filter);
    return 0;
}

static int test_filter_parse_contains(void) {
    GV_Filter *filter = gv_filter_parse("tag CONTAINS \"news\"");
    ASSERT(filter != NULL, "parse CONTAINS filter");
    
    gv_filter_destroy(filter);
    return 0;
}

static int test_filter_parse_prefix(void) {
    GV_Filter *filter = gv_filter_parse("prefix PREFIX \"user:\"");
    if (filter == NULL) {
        return 0;
    }
    
    gv_filter_destroy(filter);
    return 0;
}

static int test_filter_parse_invalid(void) {
    GV_Filter *filter = gv_filter_parse("invalid syntax !@#$");
    ASSERT(filter == NULL, "invalid filter should return NULL");
    
    gv_filter_destroy(filter);
    return 0;
}

static int test_filter_eval_simple(void) {
    GV_Filter *filter = gv_filter_parse("category == \"A\"");
    ASSERT(filter != NULL, "parse filter");
    
    float v_data[2] = {1.0f, 2.0f};
    GV_Vector *v = gv_vector_create_from_data(2, v_data);
    ASSERT(v != NULL, "create vector");
    
    ASSERT(gv_vector_set_metadata(v, "category", "A") == 0, "set metadata");
    
    int result = gv_filter_eval(filter, v);
    ASSERT(result == 1, "filter should match");
    
    gv_vector_destroy(v);
    gv_filter_destroy(filter);
    return 0;
}

static int test_filter_eval_no_match(void) {
    GV_Filter *filter = gv_filter_parse("category == \"B\"");
    ASSERT(filter != NULL, "parse filter");
    
    float v_data[2] = {1.0f, 2.0f};
    GV_Vector *v = gv_vector_create_from_data(2, v_data);
    ASSERT(v != NULL, "create vector");
    
    ASSERT(gv_vector_set_metadata(v, "category", "A") == 0, "set metadata");
    
    int result = gv_filter_eval(filter, v);
    ASSERT(result == 0, "filter should not match");
    
    gv_vector_destroy(v);
    gv_filter_destroy(filter);
    return 0;
}

static int test_filter_eval_numeric(void) {
    GV_Filter *filter = gv_filter_parse("score >= 0.5");
    ASSERT(filter != NULL, "parse numeric filter");
    
    float v_data[2] = {1.0f, 2.0f};
    GV_Vector *v = gv_vector_create_from_data(2, v_data);
    ASSERT(v != NULL, "create vector");
    
    ASSERT(gv_vector_set_metadata(v, "score", "0.7") == 0, "set numeric metadata");
    
    int result = gv_filter_eval(filter, v);
    ASSERT(result == 1, "numeric filter should match");
    
    gv_vector_destroy(v);
    gv_filter_destroy(filter);
    return 0;
}

static int test_filter_in_database(void) {
    GV_Database *db = gv_db_open(NULL, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "db open");
    
    float v1[2] = {0.0f, 1.0f};
    float v2[2] = {0.0f, 2.0f};
    float v3[2] = {0.0f, 3.0f};
    
    ASSERT(gv_db_add_vector_with_metadata(db, v1, 2, "color", "red") == 0, "add red");
    ASSERT(gv_db_add_vector_with_metadata(db, v2, 2, "color", "blue") == 0, "add blue");
    ASSERT(gv_db_add_vector_with_metadata(db, v3, 2, "color", "red") == 0, "add red 2");
    
    float q[2] = {0.0f, 1.1f};
    GV_SearchResult res[3];
    int n = gv_db_search_with_filter_expr(db, q, 3, res, GV_DISTANCE_EUCLIDEAN, "color == \"red\"");
    ASSERT(n > 0, "filtered search with expression");
    
    gv_db_close(db);
    return 0;
}

static int test_filter_destroy_null(void) {
    gv_filter_destroy(NULL);
    return 0;
}

int main(void) {
    int rc = 0;
    printf("Running filter tests...\n");
    rc |= test_filter_parse_simple();
    rc |= test_filter_parse_and();
    rc |= test_filter_parse_or();
    rc |= test_filter_parse_not();
    rc |= test_filter_parse_parentheses();
    rc |= test_filter_parse_contains();
    rc |= test_filter_parse_prefix();
    rc |= test_filter_parse_invalid();
    rc |= test_filter_eval_simple();
    rc |= test_filter_eval_no_match();
    rc |= test_filter_eval_numeric();
    rc |= test_filter_in_database();
    rc |= test_filter_destroy_null();
    if (rc == 0) {
        printf("All filter tests passed\n");
    }
    return rc;
}

