#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "gigavector.h"

#define ASSERT(cond) do { if (!(cond)) { fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #cond); return -1; } } while (0)

static int test_ivfsq8_create_destroy(void) {
    GV_IVFSQ8Config config = {
        .nlist = 4,
        .nprobe = 2,
        .train_iters = 10,
        .use_cosine = 0,
        .per_dimension = 0,
        .default_rerank = 50
    };

    void *index = ivfsq8_create(8, &config);
    ASSERT(index != NULL);
    ivfsq8_destroy(index);

    void *index2 = ivfsq8_create(16, NULL);
    ASSERT(index2 != NULL);
    ivfsq8_destroy(index2);

    return 0;
}

static int test_ivfsq8_train_insert_search(void) {
    const size_t dim = 8;
    const size_t ntrain = 100;
    const size_t ninsert = 50;
    const size_t k = 5;

    GV_IVFSQ8Config config = {
        .nlist = 4,
        .nprobe = 2,
        .train_iters = 10,
        .use_cosine = 0,
        .per_dimension = 0,
        .default_rerank = 50
    };

    void *index = ivfsq8_create(dim, &config);
    ASSERT(index != NULL);

    float train_data[100 * 8];
    for (size_t i = 0; i < ntrain; i++) {
        for (size_t j = 0; j < dim; j++) {
            train_data[i * dim + j] = (float)(i * dim + j) / 100.0f;
        }
    }

    ASSERT(ivfsq8_train(index, train_data, ntrain) == 0);

    for (size_t i = 0; i < ninsert; i++) {
        float vec[8];
        for (size_t j = 0; j < dim; j++) {
            vec[j] = (float)(i * dim + j) / 100.0f;
        }
        GV_Vector *v = vector_create_from_data(dim, vec);
        ASSERT(v != NULL);
        ASSERT(ivfsq8_insert(index, v) == 0);
    }

    ASSERT(ivfsq8_count(index) == ninsert);

    float query_data[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    GV_Vector *query = vector_create_from_data(dim, query_data);
    ASSERT(query != NULL);

    GV_SearchResult results[5];
    memset(results, 0, sizeof(results));
    int found = ivfsq8_search(index, query, k, results, GV_DISTANCE_EUCLIDEAN, NULL, NULL);
    ASSERT(found > 0);
    ASSERT(found <= (int)k);

    for (int i = 0; i < found; i++) {
        ASSERT(results[i].distance >= 0.0f);
        if (i > 0) {
            ASSERT(results[i].distance >= results[i - 1].distance);
        }
        if (results[i].vector) {
            vector_destroy((GV_Vector *)results[i].vector);
        }
    }

    vector_destroy(query);
    ivfsq8_destroy(index);
    return 0;
}

static int test_ivfsq8_db_integration(void) {
    const size_t dim = 8;
    const size_t ntrain = 100;
    const size_t ninsert = 30;

    GV_Database *db = db_open(NULL, dim, GV_INDEX_TYPE_IVFSQ8);
    ASSERT(db != NULL);

    float train_data[100 * 8];
    for (size_t i = 0; i < ntrain; i++) {
        for (size_t j = 0; j < dim; j++) {
            train_data[i * dim + j] = (float)(i * dim + j) / 100.0f;
        }
    }

    ASSERT(db_ivfsq8_train(db, train_data, ntrain, dim) == 0);

    for (size_t i = 0; i < ninsert; i++) {
        float vec[8];
        for (size_t j = 0; j < dim; j++) {
            vec[j] = (float)(i * dim + j) / 100.0f;
        }
        ASSERT(db_add_vector(db, vec, dim) == 0);
    }

    float query[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    GV_SearchResult results[5];
    memset(results, 0, sizeof(results));
    int found = db_search(db, query, 5, results, GV_DISTANCE_EUCLIDEAN);
    ASSERT(found > 0);
    ASSERT(found <= 5);

    for (int i = 0; i < found; i++) {
        ASSERT(results[i].distance >= 0.0f);
        if (results[i].vector) {
            vector_destroy((GV_Vector *)results[i].vector);
        }
    }

    db_close(db);
    return 0;
}

static int test_ivfsq8_save_load(void) {
    const char *filepath = "test_ivfsq8_save.db";
    const size_t dim = 8;
    const size_t ntrain = 100;
    const size_t ninsert = 30;

    unlink(filepath);

    GV_Database *db = db_open(NULL, dim, GV_INDEX_TYPE_IVFSQ8);
    ASSERT(db != NULL);

    float train_data[100 * 8];
    for (size_t i = 0; i < ntrain; i++) {
        for (size_t j = 0; j < dim; j++) {
            train_data[i * dim + j] = (float)(i * dim + j) / 100.0f;
        }
    }
    ASSERT(db_ivfsq8_train(db, train_data, ntrain, dim) == 0);

    for (size_t i = 0; i < ninsert; i++) {
        float vec[8];
        for (size_t j = 0; j < dim; j++) {
            vec[j] = (float)(i * dim + j) / 100.0f;
        }
        ASSERT(db_add_vector(db, vec, dim) == 0);
    }

    float query[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    GV_SearchResult results_before[5];
    memset(results_before, 0, sizeof(results_before));
    int found_before = db_search(db, query, 5, results_before, GV_DISTANCE_EUCLIDEAN);
    ASSERT(found_before > 0);

    float saved_dist[5];
    for (int i = 0; i < found_before; i++) {
        saved_dist[i] = results_before[i].distance;
        if (results_before[i].vector) {
            vector_destroy((GV_Vector *)results_before[i].vector);
        }
    }

    ASSERT(db_save(db, filepath) == 0);
    db_close(db);

    GV_Database *db2 = db_open(filepath, dim, GV_INDEX_TYPE_IVFSQ8);
    ASSERT(db2 != NULL);

    GV_SearchResult results_after[5];
    memset(results_after, 0, sizeof(results_after));
    int found_after = db_search(db2, query, 5, results_after, GV_DISTANCE_EUCLIDEAN);
    ASSERT(found_after > 0);
    ASSERT(found_before == found_after);

    for (int i = 0; i < found_before; i++) {
        float diff = fabsf(saved_dist[i] - results_after[i].distance);
        ASSERT(diff < 1e-4f);
        if (results_after[i].vector) {
            vector_destroy((GV_Vector *)results_after[i].vector);
        }
    }

    db_close(db2);
    unlink(filepath);
    return 0;
}

int main(void) {
    int rc = 0;
    rc |= test_ivfsq8_create_destroy();
    rc |= test_ivfsq8_train_insert_search();
    rc |= test_ivfsq8_db_integration();
    rc |= test_ivfsq8_save_load();
    return rc;
}
