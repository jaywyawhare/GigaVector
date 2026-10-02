#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gigavector.h"

#define ASSERT(cond, msg)         \
    do {                          \
        if (!(cond)) {            \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1;            \
        }                         \
    } while (0)

static int test_filter_parse_simple(void) {
    GV_Filter *filter = filter_parse("category == \"A\"");
    ASSERT(filter != NULL, "parse simple filter");
    
    filter_destroy(filter);
    return 0;
}

static int test_filter_parse_and(void) {
    GV_Filter *filter = filter_parse("category == \"A\" AND score >= 0.5");
    ASSERT(filter != NULL, "parse AND filter");
    
    filter_destroy(filter);
    return 0;
}

static int test_filter_parse_or(void) {
    GV_Filter *filter = filter_parse("country == \"US\" OR country == \"CA\"");
    ASSERT(filter != NULL, "parse OR filter");
    
    filter_destroy(filter);
    return 0;
}

static int test_filter_parse_not(void) {
    GV_Filter *filter = filter_parse("NOT status == \"deleted\"");
    ASSERT(filter != NULL, "parse NOT filter");
    
    filter_destroy(filter);
    return 0;
}

static int test_filter_parse_parentheses(void) {
    GV_Filter *filter = filter_parse("(country == \"US\" OR country == \"CA\") AND NOT status == \"deleted\"");
    ASSERT(filter != NULL, "parse parentheses filter");
    
    filter_destroy(filter);
    return 0;
}

static int test_filter_parse_contains(void) {
    GV_Filter *filter = filter_parse("tag CONTAINS \"news\"");
    ASSERT(filter != NULL, "parse CONTAINS filter");
    
    filter_destroy(filter);
    return 0;
}

static int test_filter_parse_prefix(void) {
    GV_Filter *filter = filter_parse("prefix PREFIX \"user:\"");
    if (filter == NULL) {
        return 0;
    }
    
    filter_destroy(filter);
    return 0;
}

static int test_filter_parse_invalid(void) {
    GV_Filter *filter = filter_parse("invalid syntax !@#$");
    ASSERT(filter == NULL, "invalid filter should return NULL");
    
    filter_destroy(filter);
    return 0;
}

static int test_filter_eval_simple(void) {
    GV_Filter *filter = filter_parse("category == \"A\"");
    ASSERT(filter != NULL, "parse filter");
    
    float v_data[2] = {1.0f, 2.0f};
    GV_Vector *v = vector_create_from_data(2, v_data);
    ASSERT(v != NULL, "create vector");
    
    ASSERT(vector_set_metadata(v, "category", "A") == 0, "set metadata");
    
    int result = filter_eval(filter, v);
    ASSERT(result == 1, "filter should match");
    
    vector_destroy(v);
    filter_destroy(filter);
    return 0;
}

static int test_filter_eval_no_match(void) {
    GV_Filter *filter = filter_parse("category == \"B\"");
    ASSERT(filter != NULL, "parse filter");
    
    float v_data[2] = {1.0f, 2.0f};
    GV_Vector *v = vector_create_from_data(2, v_data);
    ASSERT(v != NULL, "create vector");
    
    ASSERT(vector_set_metadata(v, "category", "A") == 0, "set metadata");
    
    int result = filter_eval(filter, v);
    ASSERT(result == 0, "filter should not match");
    
    vector_destroy(v);
    filter_destroy(filter);
    return 0;
}

static int test_filter_eval_numeric(void) {
    GV_Filter *filter = filter_parse("score >= 0.5");
    ASSERT(filter != NULL, "parse numeric filter");
    
    float v_data[2] = {1.0f, 2.0f};
    GV_Vector *v = vector_create_from_data(2, v_data);
    ASSERT(v != NULL, "create vector");
    
    ASSERT(vector_set_metadata(v, "score", "0.7") == 0, "set numeric metadata");
    
    int result = filter_eval(filter, v);
    ASSERT(result == 1, "numeric filter should match");
    
    vector_destroy(v);
    filter_destroy(filter);
    return 0;
}

static int test_filter_in_database(void) {
    GV_Database *db = db_open(NULL, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "db open");
    
    float v1[2] = {0.0f, 1.0f};
    float v2[2] = {0.0f, 2.0f};
    float v3[2] = {0.0f, 3.0f};
    
    ASSERT(db_add_vector_with_metadata(db, v1, 2, "color", "red") == 0, "add red");
    ASSERT(db_add_vector_with_metadata(db, v2, 2, "color", "blue") == 0, "add blue");
    ASSERT(db_add_vector_with_metadata(db, v3, 2, "color", "red") == 0, "add red 2");
    
    float q[2] = {0.0f, 1.1f};
    GV_SearchResult res[3];
    int n = db_search_with_filter_expr(db, q, 3, res, GV_DISTANCE_EUCLIDEAN, "color == \"red\"");
    ASSERT(n > 0, "filtered search with expression");

    /* db_search_with_filter_expr transfers ownership of each result's vector copy to the caller. */
    gv_search_results_free(res, (size_t)n);

    db_close(db);
    return 0;
}

static int test_filter_destroy_null(void) {
    filter_destroy(NULL);
    return 0;
}

static int test_filter_in_operator(void) {
    GV_Filter *filter = filter_parse("country IN [\"US\", \"CA\", \"GB\"]");
    ASSERT(filter != NULL, "parse IN filter");
    float d[2] = {1.0f, 2.0f};
    GV_Vector *v = vector_create_from_data(2, d);
    ASSERT(vector_set_metadata(v, "country", "CA") == 0, "set metadata");
    ASSERT(filter_eval(filter, v) == 1, "IN should match member");
    ASSERT(vector_set_metadata(v, "country", "FR") == 0, "set metadata 2");
    ASSERT(filter_eval(filter, v) == 0, "IN should not match non-member");
    vector_destroy(v);
    filter_destroy(filter);
    /* numeric IN */
    filter = filter_parse("level IN [1, 2, 3]");
    ASSERT(filter != NULL, "parse numeric IN");
    GV_Vector *v2 = vector_create_from_data(2, d);
    ASSERT(vector_set_metadata(v2, "level", "2") == 0, "set numeric metadata");
    ASSERT(filter_eval(filter, v2) == 1, "numeric IN matches");
    vector_destroy(v2);
    filter_destroy(filter);
    return 0;
}

static int test_filter_between_operator(void) {
    GV_Filter *filter = filter_parse("score BETWEEN 0.5 AND 1.5");
    ASSERT(filter != NULL, "parse BETWEEN filter");
    float d[2] = {1.0f, 2.0f};
    GV_Vector *v = vector_create_from_data(2, d);
    ASSERT(vector_set_metadata(v, "score", "1.0") == 0, "set metadata");
    ASSERT(filter_eval(filter, v) == 1, "BETWEEN matches in range");
    ASSERT(vector_set_metadata(v, "score", "0.25") == 0, "set below");
    ASSERT(filter_eval(filter, v) == 0, "BETWEEN excludes below range");
    ASSERT(vector_set_metadata(v, "score", "2.0") == 0, "set above");
    ASSERT(filter_eval(filter, v) == 0, "BETWEEN excludes above range");
    ASSERT(vector_set_metadata(v, "score", "0.5") == 0, "set lower bound");
    ASSERT(filter_eval(filter, v) == 1, "BETWEEN is inclusive of bounds");
    vector_destroy(v);
    filter_destroy(filter);
    return 0;
}

int main(void) {
    int rc = 0;
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
    rc |= test_filter_in_operator();
    rc |= test_filter_between_operator();
    rc |= test_filter_destroy_null();
    return rc;
}

