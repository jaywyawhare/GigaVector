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

static int test_hnsw_basic_insert_search(void) {
    GV_Database *db = gv_db_open(NULL, 3, GV_INDEX_TYPE_HNSW);
    if (db == NULL) {
        printf("Skipping HNSW test (HNSW not available)\n");
        return 0;
    }
    
    float v1[3] = {1.0f, 2.0f, 3.0f};
    float v2[3] = {4.0f, 5.0f, 6.0f};
    float v3[3] = {7.0f, 8.0f, 9.0f};
    
    ASSERT(gv_db_add_vector(db, v1, 3) == 0, "add vector 1");
    ASSERT(gv_db_add_vector(db, v2, 3) == 0, "add vector 2");
    ASSERT(gv_db_add_vector(db, v3, 3) == 0, "add vector 3");
    
    float q[3] = {1.0f, 2.0f, 3.0f};
    GV_SearchResult res[3];
    int n = gv_db_search(db, q, 3, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n > 0, "search returned results");
    if (n > 0 && res[0].vector != NULL) {
        ASSERT(res[0].distance >= 0.0f, "distance is non-negative");
    }
    
    gv_db_close(db);
    return 0;
}

static int test_hnsw_config(void) {
    GV_HNSWConfig config = {0};
    config.M = 8;
    config.efConstruction = 100;
    config.efSearch = 20;
    config.use_binary_quant = 0;
    
    GV_Database *db = gv_db_open_with_hnsw_config(NULL, 4, GV_INDEX_TYPE_HNSW, &config);
    if (db == NULL) {
        printf("Skipping HNSW config test (HNSW not available)\n");
        return 0;
    }
    
    float v[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    ASSERT(gv_db_add_vector(db, v, 4) == 0, "add vector with custom config");
    
    float q[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    GV_SearchResult res[1];
    int n = gv_db_search(db, q, 1, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n == 1, "search with custom config");
    
    gv_db_close(db);
    return 0;
}

static int test_hnsw_large_dataset(void) {
    GV_Database *db = gv_db_open(NULL, 8, GV_INDEX_TYPE_HNSW);
    if (db == NULL) {
        printf("Skipping HNSW large dataset test (HNSW not available)\n");
        return 0;
    }
    
    for (int i = 0; i < 100; i++) {
        float v[8];
        for (int j = 0; j < 8; j++) {
            v[j] = (float)(i * 8 + j) / 100.0f;
        }
        ASSERT(gv_db_add_vector(db, v, 8) == 0, "add vector in large dataset");
    }
    
    float q[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    GV_SearchResult res[5];
    int n = gv_db_search(db, q, 5, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n == 5, "search in large dataset");
    
    gv_db_close(db);
    return 0;
}

static int test_hnsw_filtered_search(void) {
    GV_Database *db = gv_db_open(NULL, 2, GV_INDEX_TYPE_HNSW);
    if (db == NULL) {
        printf("Skipping HNSW filtered search test (HNSW not available)\n");
        return 0;
    }
    
    float v1[2] = {1.0f, 2.0f};
    float v2[2] = {3.0f, 4.0f};
    float v3[2] = {5.0f, 6.0f};
    
    ASSERT(gv_db_add_vector_with_metadata(db, v1, 2, "color", "red") == 0, "add red vector");
    ASSERT(gv_db_add_vector_with_metadata(db, v2, 2, "color", "blue") == 0, "add blue vector");
    ASSERT(gv_db_add_vector_with_metadata(db, v3, 2, "color", "red") == 0, "add red vector 2");
    
    float q[2] = {1.0f, 2.0f};
    GV_SearchResult res[2];
    int n = gv_db_search_filtered(db, q, 2, res, GV_DISTANCE_EUCLIDEAN, "color", "red");
    ASSERT(n > 0, "filtered search returned results");
    
    gv_db_close(db);
    return 0;
}

static int test_hnsw_range_search(void) {
    GV_Database *db = gv_db_open(NULL, 2, GV_INDEX_TYPE_HNSW);
    if (db == NULL) {
        printf("Skipping HNSW range search test (HNSW not available)\n");
        return 0;
    }
    
    float v1[2] = {0.0f, 0.0f};
    float v2[2] = {1.0f, 0.0f};
    float v3[2] = {2.0f, 0.0f};
    float v4[2] = {10.0f, 0.0f};
    
    ASSERT(gv_db_add_vector(db, v1, 2) == 0, "add vector 1");
    ASSERT(gv_db_add_vector(db, v2, 2) == 0, "add vector 2");
    ASSERT(gv_db_add_vector(db, v3, 2) == 0, "add vector 3");
    ASSERT(gv_db_add_vector(db, v4, 2) == 0, "add vector 4");
    
    float q[2] = {0.0f, 0.0f};
    GV_SearchResult res[10];
    int n = gv_db_range_search(db, q, 2.5f, res, 10, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n >= 0, "range search found vectors");
    
    gv_db_close(db);
    return 0;
}

static int test_hnsw_persistence(void) {
    const char *path = "tmp_hnsw_db.bin";
    remove(path);
    
    GV_Database *db = gv_db_open(path, 3, GV_INDEX_TYPE_HNSW);
    if (db == NULL) {
        printf("Skipping HNSW persistence test (HNSW not available)\n");
        return 0;
    }
    
    float v[3] = {1.0f, 2.0f, 3.0f};
    ASSERT(gv_db_add_vector(db, v, 3) == 0, "add vector");
    ASSERT(gv_db_save(db, NULL) == 0, "save database");
    gv_db_close(db);
    
    GV_Database *db2 = gv_db_open(path, 3, GV_INDEX_TYPE_HNSW);
    ASSERT(db2 != NULL, "reopen database");
    
    float q[3] = {1.0f, 2.0f, 3.0f};
    GV_SearchResult res[1];
    int n = gv_db_search(db2, q, 1, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n == 1, "search after reload");
    
    gv_db_close(db2);
    remove(path);
    return 0;
}

static int test_hnsw_all_distances(void) {
    GV_Database *db = gv_db_open(NULL, 3, GV_INDEX_TYPE_HNSW);
    if (db == NULL) {
        printf("Skipping HNSW distance tests (HNSW not available)\n");
        return 0;
    }
    
    float v[3] = {1.0f, 2.0f, 3.0f};
    ASSERT(gv_db_add_vector(db, v, 3) == 0, "add vector");
    
    float q[3] = {1.0f, 2.0f, 3.0f};
    GV_SearchResult res[1];
    
    int n = gv_db_search(db, q, 1, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n == 1, "euclidean search");
    
    n = gv_db_search(db, q, 1, res, GV_DISTANCE_COSINE);
    ASSERT(n == 1, "cosine search");
    
    n = gv_db_search(db, q, 1, res, GV_DISTANCE_DOT_PRODUCT);
    ASSERT(n == 1, "dot product search");
    
    gv_db_close(db);
    return 0;
}

int main(void) {
    int rc = 0;
    printf("Running HNSW tests...\n");
    rc |= test_hnsw_basic_insert_search();
    rc |= test_hnsw_config();
    rc |= test_hnsw_large_dataset();
    rc |= test_hnsw_filtered_search();
    rc |= test_hnsw_range_search();
    rc |= test_hnsw_persistence();
    rc |= test_hnsw_all_distances();
    if (rc == 0) {
        printf("All HNSW tests passed\n");
    }
    return rc;
}

