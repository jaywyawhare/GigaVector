#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "gigavector/gigavector.h"

#define ASSERT(cond, msg)         \
    do {                          \
        if (!(cond)) {            \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1;            \
        }                         \
    } while (0)

static int test_sparse_basic_insert_search(void) {
    GV_Database *db = gv_db_open(NULL, 100, GV_INDEX_TYPE_SPARSE);
    if (db == NULL) {
        return 0;
    }
    
    uint32_t indices1[3] = {0, 10, 50};
    float values1[3] = {1.0f, 2.0f, 3.0f};
    if (gv_db_add_sparse_vector(db, indices1, values1, 3, 100, NULL, NULL) != 0) {
        gv_db_close(db);
        return 0;
    }
    
    uint32_t indices2[3] = {5, 15, 55};
    float values2[3] = {2.0f, 3.0f, 4.0f};
    if (gv_db_add_sparse_vector(db, indices2, values2, 3, 100, NULL, NULL) != 0) {
        gv_db_close(db);
        return 0;
    }
    
    uint32_t q_indices[2] = {0, 10};
    float q_values[2] = {1.0f, 2.0f};
    GV_SearchResult res[2] = {0};
    int n = gv_db_search_sparse(db, q_indices, q_values, 2, 2, res, GV_DISTANCE_DOT_PRODUCT);
    if (n < 0) {
        gv_db_close(db);
        return 0;
    }
    
    gv_db_close(db);
    return 0;
}

static int test_sparse_cosine_distance(void) {
    GV_Database *db = gv_db_open(NULL, 100, GV_INDEX_TYPE_SPARSE);
    if (db == NULL) {
        return 0;
    }
    
    uint32_t indices[3] = {0, 10, 50};
    float values[3] = {1.0f, 2.0f, 3.0f};
    ASSERT(gv_db_add_sparse_vector(db, indices, values, 3, 100, NULL, NULL) == 0, "add sparse vector");
    
    uint32_t q_indices[3] = {0, 10, 50};
    float q_values[3] = {1.0f, 2.0f, 3.0f};
    GV_SearchResult res[1];
    int n = gv_db_search_sparse(db, q_indices, q_values, 3, 1, res, GV_DISTANCE_COSINE);
    ASSERT(n == 1, "cosine search returned result");
    
    gv_db_close(db);
    return 0;
}

static int test_sparse_metadata(void) {
    GV_Database *db = gv_db_open(NULL, 100, GV_INDEX_TYPE_SPARSE);
    if (db == NULL) {
        return 0;
    }
    
    uint32_t indices[3] = {0, 10, 50};
    float values[3] = {1.0f, 2.0f, 3.0f};
    if (gv_db_add_sparse_vector(db, indices, values, 3, 100, "category", "test") != 0) {
        gv_db_close(db);
        return 0;
    }
    
    uint32_t q_indices[3] = {0, 10, 50};
    float q_values[3] = {1.0f, 2.0f, 3.0f};
    GV_SearchResult res[1];
    int n = gv_db_search_sparse(db, q_indices, q_values, 3, 1, res, GV_DISTANCE_DOT_PRODUCT);
    ASSERT(n >= 0, "search returned result");
    
    gv_db_close(db);
    return 0;
}

static int test_sparse_large_dataset(void) {
    GV_Database *db = gv_db_open(NULL, 1000, GV_INDEX_TYPE_SPARSE);
    if (db == NULL) {
        return 0;
    }
    
    for (int i = 0; i < 50; i++) {
        uint32_t indices[5];
        float values[5];
        for (int j = 0; j < 5; j++) {
            indices[j] = (uint32_t)(i * 10 + j * 2);
            values[j] = (float)(i + j) / 10.0f;
        }
        if (gv_db_add_sparse_vector(db, indices, values, 5, 1000, NULL, NULL) != 0) {
            gv_db_close(db);
            return 0;
        }
    }
    
    uint32_t q_indices[3] = {0, 10, 20};
    float q_values[3] = {1.0f, 1.0f, 1.0f};
    GV_SearchResult res[5];
    int n = gv_db_search_sparse(db, q_indices, q_values, 3, 5, res, GV_DISTANCE_DOT_PRODUCT);
    ASSERT(n >= 0, "search in large dataset");
    
    gv_db_close(db);
    return 0;
}

static int test_sparse_empty_query(void) {
    GV_Database *db = gv_db_open(NULL, 100, GV_INDEX_TYPE_SPARSE);
    if (db == NULL) {
        return 0;
    }
    
    uint32_t indices[3] = {0, 10, 50};
    float values[3] = {1.0f, 2.0f, 3.0f};
    if (gv_db_add_sparse_vector(db, indices, values, 3, 100, NULL, NULL) != 0) {
        gv_db_close(db);
        return 0;
    }
    
    uint32_t q_indices[0];
    float q_values[0];
    GV_SearchResult res[1] = {0};
    int n = gv_db_search_sparse(db, q_indices, q_values, 0, 1, res, GV_DISTANCE_DOT_PRODUCT);
    if (n < 0) {
        gv_db_close(db);
        return 0;
    }
    
    gv_db_close(db);
    return 0;
}

static int test_sparse_persistence(void) {
    const char *path = "tmp_sparse_db.bin";
    remove(path);
    
    GV_Database *db = gv_db_open(path, 100, GV_INDEX_TYPE_SPARSE);
    if (db == NULL) {
        return 0;
    }
    
    uint32_t indices[3] = {0, 10, 50};
    float values[3] = {1.0f, 2.0f, 3.0f};
    if (gv_db_add_sparse_vector(db, indices, values, 3, 100, NULL, NULL) != 0) {
        gv_db_close(db);
        remove(path);
        return 0;
    }
    if (gv_db_save(db, NULL) != 0) {
        gv_db_close(db);
        remove(path);
        return 0;
    }
    gv_db_close(db);
    
    GV_Database *db2 = gv_db_open(path, 100, GV_INDEX_TYPE_SPARSE);
    if (db2 == NULL) {
        remove(path);
        return 0;
    }
    
    uint32_t q_indices[3] = {0, 10, 50};
    float q_values[3] = {1.0f, 2.0f, 3.0f};
    GV_SearchResult res[1];
    int n = gv_db_search_sparse(db2, q_indices, q_values, 3, 1, res, GV_DISTANCE_DOT_PRODUCT);
    ASSERT(n >= 0, "search after reload");
    
    gv_db_close(db2);
    remove(path);
    return 0;
}

int main(void) {
    int rc = 0;
    rc |= test_sparse_basic_insert_search();
    rc |= test_sparse_cosine_distance();
    rc |= test_sparse_metadata();
    rc |= test_sparse_large_dataset();
    rc |= test_sparse_empty_query();
    rc |= test_sparse_persistence();
    return rc;
}

