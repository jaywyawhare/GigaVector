#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "gigavector.h"
#include "../test_tmp.h"

#define ASSERT(cond, msg)         \
    do {                          \
        if (!(cond)) {            \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1;            \
        }                         \
    } while (0)

static int test_hnsw_basic_insert_search(void) {
    GV_Database *db = db_open(NULL, 3, GV_INDEX_TYPE_HNSW);
    if (db == NULL) {
        return 0;
    }
    
    float v1[3] = {1.0f, 2.0f, 3.0f};
    float v2[3] = {4.0f, 5.0f, 6.0f};
    float v3[3] = {7.0f, 8.0f, 9.0f};
    
    ASSERT(db_add_vector(db, v1, 3) == 0, "add vector 1");
    ASSERT(db_add_vector(db, v2, 3) == 0, "add vector 2");
    ASSERT(db_add_vector(db, v3, 3) == 0, "add vector 3");
    
    float q[3] = {1.0f, 2.0f, 3.0f};
    GV_SearchResult res[3];
    int n = db_search(db, q, 3, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n > 0, "search returned results");
    if (n > 0 && res[0].vector != NULL) {
        ASSERT(res[0].distance >= 0.0f, "distance is non-negative");
    }

    gv_search_results_free(res, (size_t)n);
    db_close(db);
    return 0;
}

static int test_hnsw_config(void) {
    GV_HNSWConfig config = {0};
    config.M = 8;
    config.efConstruction = 100;
    config.efSearch = 20;
    config.use_binary_quant = 0;
    
    GV_Database *db = db_open_with_hnsw_config(NULL, 4, GV_INDEX_TYPE_HNSW, &config);
    if (db == NULL) {
        return 0;
    }
    
    float v[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    ASSERT(db_add_vector(db, v, 4) == 0, "add vector with custom config");
    
    float q[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    GV_SearchResult res[1];
    int n = db_search(db, q, 1, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n == 1, "search with custom config");

    gv_search_results_free(res, (size_t)n);
    db_close(db);
    return 0;
}

static int test_hnsw_large_dataset(void) {
    GV_Database *db = db_open(NULL, 8, GV_INDEX_TYPE_HNSW);
    if (db == NULL) {
        return 0;
    }

    for (int i = 0; i < 100; i++) {
        float v[8];
        for (int j = 0; j < 8; j++) {
            v[j] = (float)(i * 8 + j) / 100.0f;
        }
        ASSERT(db_add_vector(db, v, 8) == 0, "add vector in large dataset");
    }

    /* Query with the exact first vector; it must be the nearest neighbor */
    float q[8] = {0.0f, 0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f};
    GV_SearchResult res[5];
    int n = db_search(db, q, 5, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n == 5, "search in large dataset");
    /* Nearest neighbor must have distance 0 (exact match) */
    if (n >= 1) {
        ASSERT(res[0].distance < 1e-5f, "nearest neighbor is correct");
    }

    gv_search_results_free(res, (size_t)n);
    db_close(db);
    return 0;
}

static int test_hnsw_filtered_search(void) {
    GV_Database *db = db_open(NULL, 2, GV_INDEX_TYPE_HNSW);
    if (db == NULL) {
        return 0;
    }
    
    float v1[2] = {1.0f, 2.0f};
    float v2[2] = {3.0f, 4.0f};
    float v3[2] = {5.0f, 6.0f};
    
    ASSERT(db_add_vector_with_metadata(db, v1, 2, "color", "red") == 0, "add red vector");
    ASSERT(db_add_vector_with_metadata(db, v2, 2, "color", "blue") == 0, "add blue vector");
    ASSERT(db_add_vector_with_metadata(db, v3, 2, "color", "red") == 0, "add red vector 2");
    
    float q[2] = {1.0f, 2.0f};
    GV_SearchResult res[2];
    int n = db_search_filtered(db, q, 2, res, GV_DISTANCE_EUCLIDEAN, "color", "red");
    ASSERT(n > 0, "filtered search returned results");

    gv_search_results_free(res, (size_t)n);
    db_close(db);
    return 0;
}

static int test_hnsw_range_search(void) {
    GV_Database *db = db_open(NULL, 2, GV_INDEX_TYPE_HNSW);
    if (db == NULL) {
        return 0;
    }
    
    float v1[2] = {0.0f, 0.0f};
    float v2[2] = {1.0f, 0.0f};
    float v3[2] = {2.0f, 0.0f};
    float v4[2] = {10.0f, 0.0f};
    
    ASSERT(db_add_vector(db, v1, 2) == 0, "add vector 1");
    ASSERT(db_add_vector(db, v2, 2) == 0, "add vector 2");
    ASSERT(db_add_vector(db, v3, 2) == 0, "add vector 3");
    ASSERT(db_add_vector(db, v4, 2) == 0, "add vector 4");
    
    float q[2] = {0.0f, 0.0f};
    GV_SearchResult res[10];
    int n = db_range_search(db, q, 2.5f, res, 10, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n >= 0, "range search found vectors");

    if (n > 0) gv_search_results_free(res, (size_t)n);
    db_close(db);
    return 0;
}

static int test_hnsw_persistence(void) {
    char path[256];
    if (gv_test_make_temp_path(path, sizeof(path), "gv_hnsw_persist", ".bin") != 0) return 0;
    remove(path);

    GV_Database *db = db_open(path, 3, GV_INDEX_TYPE_HNSW);
    if (db == NULL) {
        return 0;
    }

    float v[3] = {1.0f, 2.0f, 3.0f};
    ASSERT(db_add_vector(db, v, 3) == 0, "add vector");
    ASSERT(db_save(db, NULL) == 0, "save database");
    db_close(db);

    GV_Database *db2 = db_open(path, 3, GV_INDEX_TYPE_HNSW);
    ASSERT(db2 != NULL, "reopen database");

    float q[3] = {1.0f, 2.0f, 3.0f};
    GV_SearchResult res[1];
    int n = db_search(db2, q, 1, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n == 1, "search after reload");
    /* The only inserted vector should be the nearest neighbor */
    if (n == 1 && res[0].vector != NULL) {
        ASSERT(res[0].distance < 1e-5f, "nearest neighbor is exact match after reload");
    }

    gv_search_results_free(res, (size_t)n);
    db_close(db2);
    remove(path);
    return 0;
}

static int test_hnsw_all_distances(void) {
    GV_Database *db = db_open(NULL, 3, GV_INDEX_TYPE_HNSW);
    if (db == NULL) {
        return 0;
    }
    
    float v[3] = {1.0f, 2.0f, 3.0f};
    ASSERT(db_add_vector(db, v, 3) == 0, "add vector");
    
    float q[3] = {1.0f, 2.0f, 3.0f};
    GV_SearchResult res[1];
    
    int n = db_search(db, q, 1, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n == 1, "euclidean search");
    gv_search_results_free(res, (size_t)n);

    n = db_search(db, q, 1, res, GV_DISTANCE_COSINE);
    ASSERT(n == 1, "cosine search");
    gv_search_results_free(res, (size_t)n);

    n = db_search(db, q, 1, res, GV_DISTANCE_DOT_PRODUCT);
    ASSERT(n == 1, "dot product search");
    gv_search_results_free(res, (size_t)n);

    db_close(db);
    return 0;
}

/* Regression: the construction/search distance batch once reset its 4-slot
 * accumulator only for EUCLIDEAN, so a COSINE/DOT index whose nodes have >4
 * neighbours at a level (level 0 has ~2*M) overflowed the stack buffer on the
 * 5th neighbour - in both hnsw_insert_impl (build) and gv_hnsw_search. This
 * builds at M=16 with 300 vectors so nodes exceed 4 neighbours, exercising both
 * paths for each metric; under ASan it caught the overflow every time. */
static int test_hnsw_non_euclidean_batch_metrics(void) {
    const GV_DistanceType metrics[2] = { GV_DISTANCE_COSINE, GV_DISTANCE_DOT_PRODUCT };
    const size_t dim = 32, n = 300;

    for (size_t mi = 0; mi < 2; ++mi) {
        GV_HNSWConfig config = {0};
        config.M = 16;             /* level 0 holds up to ~2*M = 32 neighbours */
        config.efConstruction = 100;
        config.efSearch = 50;
        config.distance_type = metrics[mi];

        GV_Database *db = db_open_with_hnsw_config(NULL, dim, GV_INDEX_TYPE_HNSW, &config);
        if (db == NULL) {
            return 0;
        }

        float v[32];
        for (size_t i = 0; i < n; ++i) {
            for (size_t d = 0; d < dim; ++d) v[d] = (float)rand() / (float)RAND_MAX - 0.5f;
            ASSERT(db_add_vector(db, v, dim) == 0, "non-euclidean batch: add vector");
        }

        float q[32];
        for (size_t d = 0; d < dim; ++d) q[d] = (float)rand() / (float)RAND_MAX - 0.5f;
        GV_SearchResult res[10];
        int got = db_search(db, q, 10, res, metrics[mi]);
        ASSERT(got > 0, "non-euclidean batch: search returned results");
        gv_search_results_free(res, (size_t)got);

        db_close(db);
    }
    return 0;
}

int main(void) {
    int rc = 0;
    rc |= test_hnsw_basic_insert_search();
    rc |= test_hnsw_config();
    rc |= test_hnsw_large_dataset();
    rc |= test_hnsw_filtered_search();
    rc |= test_hnsw_range_search();
    rc |= test_hnsw_persistence();
    rc |= test_hnsw_all_distances();
    rc |= test_hnsw_non_euclidean_batch_metrics();
    return rc;
}

