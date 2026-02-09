#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "gigavector/gigavector.h"

#define ASSERT(cond) do { if (!(cond)) { fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #cond); return -1; } } while (0)

/* ---------------------------------------------------------------------------
 * 1. test_ivfflat_create_destroy
 * --------------------------------------------------------------------------- */
static int test_ivfflat_create_destroy(void) {
    printf("  test_ivfflat_create_destroy...\n");

    GV_IVFFlatConfig config = {
        .nlist      = 4,
        .nprobe     = 2,
        .train_iters = 10,
        .use_cosine = 0
    };

    void *index = gv_ivfflat_create(8, &config);
    ASSERT(index != NULL);

    gv_ivfflat_destroy(index);

    /* NULL config should also work (uses defaults) */
    void *index2 = gv_ivfflat_create(16, NULL);
    ASSERT(index2 != NULL);
    gv_ivfflat_destroy(index2);

    printf("    PASSED\n");
    return 0;
}

/* ---------------------------------------------------------------------------
 * 2. test_ivfflat_train_insert_search
 * --------------------------------------------------------------------------- */
static int test_ivfflat_train_insert_search(void) {
    printf("  test_ivfflat_train_insert_search...\n");

    const size_t dim = 8;
    const size_t ntrain = 100;
    const size_t ninsert = 50;
    const size_t k = 5;

    GV_IVFFlatConfig config = {
        .nlist      = 4,
        .nprobe     = 2,
        .train_iters = 10,
        .use_cosine = 0
    };

    void *index = gv_ivfflat_create(dim, &config);
    ASSERT(index != NULL);

    /* Generate deterministic training data */
    float train_data[100 * 8]; /* ntrain * dim */
    for (size_t i = 0; i < ntrain; i++) {
        for (size_t j = 0; j < dim; j++) {
            train_data[i * dim + j] = (float)(i * dim + j) / 100.0f;
        }
    }

    int rc = gv_ivfflat_train(index, train_data, ntrain);
    ASSERT(rc == 0);

    /* Insert vectors */
    for (size_t i = 0; i < ninsert; i++) {
        float vec[8];
        for (size_t j = 0; j < dim; j++) {
            vec[j] = (float)(i * dim + j) / 100.0f;
        }
        GV_Vector *v = gv_vector_create_from_data(dim, vec);
        ASSERT(v != NULL);
        rc = gv_ivfflat_insert(index, v);
        ASSERT(rc == 0);
    }

    ASSERT(gv_ivfflat_count(index) == ninsert);

    /* Search */
    float query_data[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    GV_Vector *query = gv_vector_create_from_data(dim, query_data);
    ASSERT(query != NULL);

    GV_SearchResult results[5];
    memset(results, 0, sizeof(results));
    int found = gv_ivfflat_search(index, query, k, results, GV_DISTANCE_EUCLIDEAN, NULL, NULL);
    ASSERT(found > 0);
    ASSERT(found <= (int)k);

    /* Verify distances are non-negative and sorted ascending */
    for (int i = 0; i < found; i++) {
        ASSERT(results[i].distance >= 0.0f);
        if (i > 0) {
            ASSERT(results[i].distance >= results[i - 1].distance);
        }
    }

    gv_vector_destroy(query);
    gv_ivfflat_destroy(index);
    printf("    PASSED\n");
    return 0;
}

/* ---------------------------------------------------------------------------
 * 3. test_ivfflat_is_trained
 * --------------------------------------------------------------------------- */
static int test_ivfflat_is_trained(void) {
    printf("  test_ivfflat_is_trained...\n");

    const size_t dim = 8;
    GV_IVFFlatConfig config = {
        .nlist      = 4,
        .nprobe     = 2,
        .train_iters = 10,
        .use_cosine = 0
    };

    void *index = gv_ivfflat_create(dim, &config);
    ASSERT(index != NULL);

    /* Before training: should not be trained */
    ASSERT(gv_ivfflat_is_trained(index) == 0);

    /* Train */
    float train_data[100 * 8];
    for (size_t i = 0; i < 100; i++) {
        for (size_t j = 0; j < dim; j++) {
            train_data[i * dim + j] = (float)(i * dim + j) / 100.0f;
        }
    }
    int rc = gv_ivfflat_train(index, train_data, 100);
    ASSERT(rc == 0);

    /* After training: should be trained */
    ASSERT(gv_ivfflat_is_trained(index) == 1);

    gv_ivfflat_destroy(index);
    printf("    PASSED\n");
    return 0;
}

/* ---------------------------------------------------------------------------
 * 4. test_ivfflat_range_search
 * --------------------------------------------------------------------------- */
static int test_ivfflat_range_search(void) {
    printf("  test_ivfflat_range_search...\n");

    const size_t dim = 8;
    const size_t ntrain = 100;
    const size_t ninsert = 50;

    GV_IVFFlatConfig config = {
        .nlist      = 4,
        .nprobe     = 4, /* probe all lists for thorough range search */
        .train_iters = 10,
        .use_cosine = 0
    };

    void *index = gv_ivfflat_create(dim, &config);
    ASSERT(index != NULL);

    /* Train */
    float train_data[100 * 8];
    for (size_t i = 0; i < ntrain; i++) {
        for (size_t j = 0; j < dim; j++) {
            train_data[i * dim + j] = (float)(i * dim + j) / 100.0f;
        }
    }
    ASSERT(gv_ivfflat_train(index, train_data, ntrain) == 0);

    /* Insert vectors with values in a known range */
    for (size_t i = 0; i < ninsert; i++) {
        float vec[8];
        for (size_t j = 0; j < dim; j++) {
            vec[j] = (float)i / (float)ninsert;
        }
        GV_Vector *v = gv_vector_create_from_data(dim, vec);
        ASSERT(v != NULL);
        ASSERT(gv_ivfflat_insert(index, v) == 0);
    }

    /* Query near the origin with a generous radius */
    float query_data[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    GV_Vector *query = gv_vector_create_from_data(dim, query_data);
    ASSERT(query != NULL);

    GV_SearchResult results[50];
    memset(results, 0, sizeof(results));
    float radius = 5.0f;
    int found = gv_ivfflat_range_search(index, query, radius, results, 50,
                                         GV_DISTANCE_EUCLIDEAN, NULL, NULL);
    ASSERT(found >= 0);

    /* All returned results must be within the radius */
    for (int i = 0; i < found; i++) {
        ASSERT(results[i].distance <= radius);
    }

    gv_vector_destroy(query);
    gv_ivfflat_destroy(index);
    printf("    PASSED\n");
    return 0;
}

/* ---------------------------------------------------------------------------
 * 5. test_ivfflat_delete_update
 * --------------------------------------------------------------------------- */
static int test_ivfflat_delete_update(void) {
    printf("  test_ivfflat_delete_update...\n");

    const size_t dim = 8;
    const size_t ntrain = 100;
    const size_t ninsert = 20;

    GV_IVFFlatConfig config = {
        .nlist      = 4,
        .nprobe     = 2,
        .train_iters = 10,
        .use_cosine = 0
    };

    void *index = gv_ivfflat_create(dim, &config);
    ASSERT(index != NULL);

    /* Train */
    float train_data[100 * 8];
    for (size_t i = 0; i < ntrain; i++) {
        for (size_t j = 0; j < dim; j++) {
            train_data[i * dim + j] = (float)(i * dim + j) / 100.0f;
        }
    }
    ASSERT(gv_ivfflat_train(index, train_data, ntrain) == 0);

    /* Insert */
    for (size_t i = 0; i < ninsert; i++) {
        float vec[8];
        for (size_t j = 0; j < dim; j++) {
            vec[j] = (float)(i + 1) * 0.1f;
        }
        GV_Vector *v = gv_vector_create_from_data(dim, vec);
        ASSERT(v != NULL);
        ASSERT(gv_ivfflat_insert(index, v) == 0);
    }
    ASSERT(gv_ivfflat_count(index) == ninsert);

    /* Delete vector at entry index 0 */
    int rc = gv_ivfflat_delete(index, 0);
    ASSERT(rc == 0);
    ASSERT(gv_ivfflat_count(index) == ninsert - 1);

    /* Update vector at entry index 1 with new data */
    float new_data[8] = {9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f};
    rc = gv_ivfflat_update(index, 1, new_data, dim);
    ASSERT(rc == 0);
    /* Count should remain the same after update */
    ASSERT(gv_ivfflat_count(index) == ninsert - 1);

    gv_ivfflat_destroy(index);
    printf("    PASSED\n");
    return 0;
}

/* ---------------------------------------------------------------------------
 * 6. test_ivfflat_db_integration
 * --------------------------------------------------------------------------- */
static int test_ivfflat_db_integration(void) {
    printf("  test_ivfflat_db_integration...\n");

    const size_t dim = 8;
    const size_t ntrain = 100;
    const size_t ninsert = 30;

    /* Open an in-memory IVF-Flat database */
    GV_Database *db = gv_db_open(NULL, dim, GV_INDEX_TYPE_IVFFLAT);
    ASSERT(db != NULL);

    /* Prepare training data */
    float train_data[100 * 8];
    for (size_t i = 0; i < ntrain; i++) {
        for (size_t j = 0; j < dim; j++) {
            train_data[i * dim + j] = (float)(i * dim + j) / 100.0f;
        }
    }

    /* Train */
    int rc = gv_db_ivfflat_train(db, train_data, ntrain, dim);
    ASSERT(rc == 0);

    /* Insert vectors */
    for (size_t i = 0; i < ninsert; i++) {
        float vec[8];
        for (size_t j = 0; j < dim; j++) {
            vec[j] = (float)(i * dim + j) / 100.0f;
        }
        rc = gv_db_add_vector(db, vec, dim);
        ASSERT(rc == 0);
    }

    /* Search */
    float query[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    GV_SearchResult results[5];
    memset(results, 0, sizeof(results));
    int found = gv_db_search(db, query, 5, results, GV_DISTANCE_EUCLIDEAN);
    ASSERT(found > 0);
    ASSERT(found <= 5);

    /* Verify distances are non-negative */
    for (int i = 0; i < found; i++) {
        ASSERT(results[i].distance >= 0.0f);
    }

    gv_db_close(db);
    printf("    PASSED\n");
    return 0;
}

/* ---------------------------------------------------------------------------
 * 7. test_ivfflat_save_load
 * --------------------------------------------------------------------------- */
static int test_ivfflat_save_load(void) {
    printf("  test_ivfflat_save_load...\n");

    const char *filepath = "test_ivfflat_save.db";
    const size_t dim = 8;
    const size_t ntrain = 100;
    const size_t ninsert = 30;

    /* Remove any leftover file from a previous run */
    unlink(filepath);

    /* --- Phase 1: create, train, insert, save --- */
    GV_Database *db = gv_db_open(NULL, dim, GV_INDEX_TYPE_IVFFLAT);
    ASSERT(db != NULL);

    float train_data[100 * 8];
    for (size_t i = 0; i < ntrain; i++) {
        for (size_t j = 0; j < dim; j++) {
            train_data[i * dim + j] = (float)(i * dim + j) / 100.0f;
        }
    }
    ASSERT(gv_db_ivfflat_train(db, train_data, ntrain, dim) == 0);

    for (size_t i = 0; i < ninsert; i++) {
        float vec[8];
        for (size_t j = 0; j < dim; j++) {
            vec[j] = (float)(i * dim + j) / 100.0f;
        }
        ASSERT(gv_db_add_vector(db, vec, dim) == 0);
    }

    /* Search before save to get reference results */
    float query[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    GV_SearchResult results_before[5];
    memset(results_before, 0, sizeof(results_before));
    int found_before = gv_db_search(db, query, 5, results_before, GV_DISTANCE_EUCLIDEAN);
    ASSERT(found_before > 0);

    /* Save */
    ASSERT(gv_db_save(db, filepath) == 0);
    gv_db_close(db);

    /* --- Phase 2: reopen and search --- */
    GV_Database *db2 = gv_db_open(filepath, dim, GV_INDEX_TYPE_IVFFLAT);
    ASSERT(db2 != NULL);

    GV_SearchResult results_after[5];
    memset(results_after, 0, sizeof(results_after));
    int found_after = gv_db_search(db2, query, 5, results_after, GV_DISTANCE_EUCLIDEAN);
    ASSERT(found_after > 0);

    /* The number of results should match */
    ASSERT(found_before == found_after);

    /* Distances should be identical (or very close due to float precision) */
    for (int i = 0; i < found_before; i++) {
        float diff = fabsf(results_before[i].distance - results_after[i].distance);
        ASSERT(diff < 1e-5f);
    }

    gv_db_close(db2);

    /* Clean up */
    unlink(filepath);

    printf("    PASSED\n");
    return 0;
}

/* ---------------------------------------------------------------------------
 * main
 * --------------------------------------------------------------------------- */
int main(void) {
    int rc = 0;
    printf("Running IVF-Flat tests...\n");

    rc |= test_ivfflat_create_destroy();
    rc |= test_ivfflat_train_insert_search();
    rc |= test_ivfflat_is_trained();
    rc |= test_ivfflat_range_search();
    rc |= test_ivfflat_delete_update();
    rc |= test_ivfflat_db_integration();
    rc |= test_ivfflat_save_load();

    if (rc == 0) {
        printf("All IVF-Flat tests passed\n");
    } else {
        printf("Some IVF-Flat tests FAILED\n");
    }
    return rc;
}
