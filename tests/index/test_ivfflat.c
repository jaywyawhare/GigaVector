#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "gigavector.h"

#define ASSERT(cond) do { if (!(cond)) { fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #cond); return -1; } } while (0)

static int test_ivfflat_create_destroy(void) {
    GV_IVFFlatConfig config = {
        .nlist      = 4,
        .nprobe     = 2,
        .train_iters = 10,
        .use_cosine = 0
    };

    void *index = ivfflat_create(8, &config);
    ASSERT(index != NULL);

    ivfflat_destroy(index);

    void *index2 = ivfflat_create(16, NULL);
    ASSERT(index2 != NULL);
    ivfflat_destroy(index2);

    return 0;
}

static int test_ivfflat_train_insert_search(void) {
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

    void *index = ivfflat_create(dim, &config);
    ASSERT(index != NULL);

    float train_data[100 * 8];
    for (size_t i = 0; i < ntrain; i++) {
        for (size_t j = 0; j < dim; j++) {
            train_data[i * dim + j] = (float)(i * dim + j) / 100.0f;
        }
    }

    int rc = ivfflat_train(index, train_data, ntrain);
    ASSERT(rc == 0);

    for (size_t i = 0; i < ninsert; i++) {
        float vec[8];
        for (size_t j = 0; j < dim; j++) {
            vec[j] = (float)(i * dim + j) / 100.0f;
        }
        GV_Vector *v = vector_create_from_data(dim, vec);
        ASSERT(v != NULL);
        rc = ivfflat_insert(index, v);
        ASSERT(rc == 0);
    }

    ASSERT(ivfflat_count(index) == ninsert);

    float query_data[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    GV_Vector *query = vector_create_from_data(dim, query_data);
    ASSERT(query != NULL);

    GV_SearchResult results[5];
    memset(results, 0, sizeof(results));
    int found = ivfflat_search(index, query, k, results, GV_DISTANCE_EUCLIDEAN, NULL, NULL);
    ASSERT(found > 0);
    ASSERT(found <= (int)k);

    for (int i = 0; i < found; i++) {
        ASSERT(results[i].distance >= 0.0f);
        if (i > 0) {
            ASSERT(results[i].distance >= results[i - 1].distance);
        }
    }

    vector_destroy(query);
    ivfflat_destroy(index);
    return 0;
}

static int test_ivfflat_is_trained(void) {
    const size_t dim = 8;
    GV_IVFFlatConfig config = {
        .nlist      = 4,
        .nprobe     = 2,
        .train_iters = 10,
        .use_cosine = 0
    };

    void *index = ivfflat_create(dim, &config);
    ASSERT(index != NULL);

    ASSERT(ivfflat_is_trained(index) == 0);

    float train_data[100 * 8];
    for (size_t i = 0; i < 100; i++) {
        for (size_t j = 0; j < dim; j++) {
            train_data[i * dim + j] = (float)(i * dim + j) / 100.0f;
        }
    }
    int rc = ivfflat_train(index, train_data, 100);
    ASSERT(rc == 0);

    ASSERT(ivfflat_is_trained(index) == 1);

    ivfflat_destroy(index);
    return 0;
}

static int test_ivfflat_range_search(void) {
    const size_t dim = 8;
    const size_t ntrain = 100;
    const size_t ninsert = 50;

    GV_IVFFlatConfig config = {
        .nlist      = 4,
        .nprobe     = 4, /* probe all lists for thorough range search */
        .train_iters = 10,
        .use_cosine = 0
    };

    void *index = ivfflat_create(dim, &config);
    ASSERT(index != NULL);

    float train_data[100 * 8];
    for (size_t i = 0; i < ntrain; i++) {
        for (size_t j = 0; j < dim; j++) {
            train_data[i * dim + j] = (float)(i * dim + j) / 100.0f;
        }
    }
    ASSERT(ivfflat_train(index, train_data, ntrain) == 0);

    for (size_t i = 0; i < ninsert; i++) {
        float vec[8];
        for (size_t j = 0; j < dim; j++) {
            vec[j] = (float)i / (float)ninsert;
        }
        GV_Vector *v = vector_create_from_data(dim, vec);
        ASSERT(v != NULL);
        ASSERT(ivfflat_insert(index, v) == 0);
    }

    float query_data[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    GV_Vector *query = vector_create_from_data(dim, query_data);
    ASSERT(query != NULL);

    GV_SearchResult results[50];
    memset(results, 0, sizeof(results));
    float radius = 5.0f;
    int found = ivfflat_range_search(index, query, radius, results, 50,
                                         GV_DISTANCE_EUCLIDEAN, NULL, NULL);
    ASSERT(found >= 0);

    for (int i = 0; i < found; i++) {
        ASSERT(results[i].distance <= radius);
    }

    vector_destroy(query);
    ivfflat_destroy(index);
    return 0;
}

static int test_ivfflat_delete_update(void) {
    const size_t dim = 8;
    const size_t ntrain = 100;
    const size_t ninsert = 20;

    GV_IVFFlatConfig config = {
        .nlist      = 4,
        .nprobe     = 2,
        .train_iters = 10,
        .use_cosine = 0
    };

    void *index = ivfflat_create(dim, &config);
    ASSERT(index != NULL);

    float train_data[100 * 8];
    for (size_t i = 0; i < ntrain; i++) {
        for (size_t j = 0; j < dim; j++) {
            train_data[i * dim + j] = (float)(i * dim + j) / 100.0f;
        }
    }
    ASSERT(ivfflat_train(index, train_data, ntrain) == 0);

    for (size_t i = 0; i < ninsert; i++) {
        float vec[8];
        for (size_t j = 0; j < dim; j++) {
            vec[j] = (float)(i + 1) * 0.1f;
        }
        GV_Vector *v = vector_create_from_data(dim, vec);
        ASSERT(v != NULL);
        ASSERT(ivfflat_insert(index, v) == 0);
    }
    ASSERT(ivfflat_count(index) == ninsert);

    int rc = ivfflat_delete(index, 0);
    ASSERT(rc == 0);
    ASSERT(ivfflat_count(index) == ninsert - 1);

    float new_data[8] = {9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f};
    rc = ivfflat_update(index, 1, new_data, dim);
    ASSERT(rc == 0);
    ASSERT(ivfflat_count(index) == ninsert - 1);

    ivfflat_destroy(index);
    return 0;
}

static int test_ivfflat_db_integration(void) {
    const size_t dim = 8;
    const size_t ntrain = 100;
    const size_t ninsert = 30;

    GV_Database *db = db_open(NULL, dim, GV_INDEX_TYPE_IVFFLAT);
    ASSERT(db != NULL);

    float train_data[100 * 8];
    for (size_t i = 0; i < ntrain; i++) {
        for (size_t j = 0; j < dim; j++) {
            train_data[i * dim + j] = (float)(i * dim + j) / 100.0f;
        }
    }

    int rc = db_ivfflat_train(db, train_data, ntrain, dim);
    ASSERT(rc == 0);

    for (size_t i = 0; i < ninsert; i++) {
        float vec[8];
        for (size_t j = 0; j < dim; j++) {
            vec[j] = (float)(i * dim + j) / 100.0f;
        }
        rc = db_add_vector(db, vec, dim);
        ASSERT(rc == 0);
    }

    float query[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    GV_SearchResult results[5];
    memset(results, 0, sizeof(results));
    int found = db_search(db, query, 5, results, GV_DISTANCE_EUCLIDEAN);
    ASSERT(found > 0);
    ASSERT(found <= 5);

    for (int i = 0; i < found; i++) {
        ASSERT(results[i].distance >= 0.0f);
    }

    db_close(db);
    return 0;
}

static int test_ivfflat_save_load(void) {
    const char *filepath = "test_ivfflat_save.db";
    const size_t dim = 8;
    const size_t ntrain = 100;
    const size_t ninsert = 30;

    unlink(filepath);

    GV_Database *db = db_open(NULL, dim, GV_INDEX_TYPE_IVFFLAT);
    ASSERT(db != NULL);

    float train_data[100 * 8];
    for (size_t i = 0; i < ntrain; i++) {
        for (size_t j = 0; j < dim; j++) {
            train_data[i * dim + j] = (float)(i * dim + j) / 100.0f;
        }
    }
    ASSERT(db_ivfflat_train(db, train_data, ntrain, dim) == 0);

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

    ASSERT(db_save(db, filepath) == 0);
    db_close(db);

    GV_Database *db2 = db_open(filepath, dim, GV_INDEX_TYPE_IVFFLAT);
    ASSERT(db2 != NULL);

    GV_SearchResult results_after[5];
    memset(results_after, 0, sizeof(results_after));
    int found_after = db_search(db2, query, 5, results_after, GV_DISTANCE_EUCLIDEAN);
    ASSERT(found_after > 0);

    ASSERT(found_before == found_after);

    for (int i = 0; i < found_before; i++) {
        float diff = fabsf(results_before[i].distance - results_after[i].distance);
        ASSERT(diff < 1e-5f);
    }

    db_close(db2);

    unlink(filepath);

    return 0;
}

int main(void) {
    int rc = 0;
    rc |= test_ivfflat_create_destroy();
    rc |= test_ivfflat_train_insert_search();
    rc |= test_ivfflat_is_trained();
    rc |= test_ivfflat_range_search();
    rc |= test_ivfflat_delete_update();
    rc |= test_ivfflat_db_integration();
    rc |= test_ivfflat_save_load();
    return rc;
}
