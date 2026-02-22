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

static int test_metadata_set_get(void) {
    float v_data[3] = {1.0f, 2.0f, 3.0f};
    GV_Vector *v = gv_vector_create_from_data(3, v_data);
    ASSERT(v != NULL, "vector creation");
    
    ASSERT(gv_vector_set_metadata(v, "key1", "value1") == 0, "set metadata");
    const char *val = gv_vector_get_metadata(v, "key1");
    ASSERT(val != NULL, "get metadata");
    ASSERT(strcmp(val, "value1") == 0, "metadata value match");
    
    gv_vector_destroy(v);
    return 0;
}

static int test_metadata_multiple_keys(void) {
    float v_data[3] = {1.0f, 2.0f, 3.0f};
    GV_Vector *v = gv_vector_create_from_data(3, v_data);
    ASSERT(v != NULL, "vector creation");
    
    ASSERT(gv_vector_set_metadata(v, "key1", "value1") == 0, "set metadata 1");
    ASSERT(gv_vector_set_metadata(v, "key2", "value2") == 0, "set metadata 2");
    ASSERT(gv_vector_set_metadata(v, "key3", "value3") == 0, "set metadata 3");
    
    ASSERT(strcmp(gv_vector_get_metadata(v, "key1"), "value1") == 0, "get key1");
    ASSERT(strcmp(gv_vector_get_metadata(v, "key2"), "value2") == 0, "get key2");
    ASSERT(strcmp(gv_vector_get_metadata(v, "key3"), "value3") == 0, "get key3");
    
    gv_vector_destroy(v);
    return 0;
}

static int test_metadata_update(void) {
    float v_data[3] = {1.0f, 2.0f, 3.0f};
    GV_Vector *v = gv_vector_create_from_data(3, v_data);
    ASSERT(v != NULL, "vector creation");
    
    ASSERT(gv_vector_set_metadata(v, "key1", "value1") == 0, "set metadata");
    ASSERT(gv_vector_set_metadata(v, "key1", "value2") == 0, "update metadata");
    
    const char *val = gv_vector_get_metadata(v, "key1");
    ASSERT(val != NULL, "get updated metadata");
    ASSERT(strcmp(val, "value2") == 0, "updated value match");
    
    gv_vector_destroy(v);
    return 0;
}

static int test_metadata_remove(void) {
    float v_data[3] = {1.0f, 2.0f, 3.0f};
    GV_Vector *v = gv_vector_create_from_data(3, v_data);
    ASSERT(v != NULL, "vector creation");
    
    ASSERT(gv_vector_set_metadata(v, "key1", "value1") == 0, "set metadata");
    ASSERT(gv_vector_remove_metadata(v, "key1") == 0, "remove metadata");
    
    const char *val = gv_vector_get_metadata(v, "key1");
    ASSERT(val == NULL, "removed metadata should be NULL");
    
    gv_vector_destroy(v);
    return 0;
}

static int test_metadata_clear(void) {
    float v_data[3] = {1.0f, 2.0f, 3.0f};
    GV_Vector *v = gv_vector_create_from_data(3, v_data);
    ASSERT(v != NULL, "vector creation");
    
    ASSERT(gv_vector_set_metadata(v, "key1", "value1") == 0, "set metadata 1");
    ASSERT(gv_vector_set_metadata(v, "key2", "value2") == 0, "set metadata 2");
    
    gv_vector_clear_metadata(v);
    
    ASSERT(gv_vector_get_metadata(v, "key1") == NULL, "cleared key1 should be NULL");
    ASSERT(gv_vector_get_metadata(v, "key2") == NULL, "cleared key2 should be NULL");
    
    gv_vector_destroy(v);
    return 0;
}

static int test_metadata_nonexistent_key(void) {
    float v_data[3] = {1.0f, 2.0f, 3.0f};
    GV_Vector *v = gv_vector_create_from_data(3, v_data);
    ASSERT(v != NULL, "vector creation");
    
    const char *val = gv_vector_get_metadata(v, "nonexistent");
    ASSERT(val == NULL, "nonexistent key should return NULL");
    
    gv_vector_destroy(v);
    return 0;
}

static int test_metadata_null_handling(void) {
    float v_data[3] = {1.0f, 2.0f, 3.0f};
    GV_Vector *v = gv_vector_create_from_data(3, v_data);
    ASSERT(v != NULL, "vector creation");
    
    ASSERT(gv_vector_set_metadata(NULL, "key", "value") < 0, "set metadata with NULL vector");
    ASSERT(gv_vector_get_metadata(NULL, "key") == NULL, "get metadata with NULL vector");
    ASSERT(gv_vector_remove_metadata(NULL, "key") < 0, "remove metadata with NULL vector");
    
    gv_vector_destroy(v);
    return 0;
}

static int test_metadata_in_database(void) {
    GV_Database *db = gv_db_open(NULL, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "db open");
    
    float v[2] = {1.0f, 2.0f};
    ASSERT(gv_db_add_vector_with_metadata(db, v, 2, "tag", "test") == 0, "add vector with metadata");
    
    GV_SearchResult res[1];
    float q[2] = {1.0f, 2.0f};
    int n = gv_db_search(db, q, 1, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n == 1, "search count");
    ASSERT(res[0].vector != NULL, "result vector");
    ASSERT(res[0].vector->metadata != NULL, "result metadata");
    
    const char *tag = gv_vector_get_metadata(res[0].vector, "tag");
    ASSERT(tag != NULL, "metadata tag exists");
    ASSERT(strcmp(tag, "test") == 0, "metadata tag value");
    
    gv_db_close(db);
    return 0;
}

int main(void) {
    int rc = 0;
    rc |= test_metadata_set_get();
    rc |= test_metadata_multiple_keys();
    rc |= test_metadata_update();
    rc |= test_metadata_remove();
    rc |= test_metadata_clear();
    rc |= test_metadata_nonexistent_key();
    rc |= test_metadata_null_handling();
    rc |= test_metadata_in_database();
    return rc;
}

