#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "gigavector.h"

#define ASSERT(cond) do { if (!(cond)) { fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #cond); return -1; } } while (0)

#define DIM 16
#define M   4
#define TRAIN_COUNT 200
#define INSERT_COUNT 50

static void generate_training_data(float *data, size_t count, size_t dim) {
    for (size_t i = 0; i < count; i++) {
        for (size_t j = 0; j < dim; j++) {
            data[i * dim + j] = sinf((float)(i * dim + j));
        }
    }
}

static int test_pq_create_destroy(void) {
    GV_PQConfig config;
    config.m           = M;
    config.nbits       = 8;
    config.train_iters = 10;

    void *index = pq_create(DIM, &config);
    ASSERT(index != NULL);

    pq_destroy(index);
    return 0;
}

static int test_pq_train_insert_search(void) {
    GV_PQConfig config;
    config.m           = M;
    config.nbits       = 8;
    config.train_iters = 10;

    void *index = pq_create(DIM, &config);
    ASSERT(index != NULL);

    float train_data[TRAIN_COUNT * DIM];
    generate_training_data(train_data, TRAIN_COUNT, DIM);
    ASSERT(pq_train(index, train_data, TRAIN_COUNT) == 0);

    for (size_t i = 0; i < INSERT_COUNT; i++) {
        GV_Vector *vec = vector_create_from_data(DIM, &train_data[i * DIM]);
        ASSERT(vec != NULL);
        ASSERT(pq_insert(index, vec) == 0);
        /* Note: pq_insert takes ownership of the vector */
    }
    ASSERT(pq_count(index) == INSERT_COUNT);

    float query[DIM];
    for (size_t j = 0; j < DIM; j++) {
        query[j] = sinf((float)j);
    }
    GV_Vector *qvec = vector_create_from_data(DIM, query);
    ASSERT(qvec != NULL);

    GV_SearchResult results[5];
    memset(results, 0, sizeof(results));
    int count = pq_search(index, qvec, 5, results, GV_DISTANCE_EUCLIDEAN, NULL, NULL);
    ASSERT(count > 0);

    vector_destroy(qvec);
    pq_destroy(index);
    return 0;
}

static int test_pq_is_trained(void) {
    GV_PQConfig config;
    config.m           = M;
    config.nbits       = 8;
    config.train_iters = 10;

    void *index = pq_create(DIM, &config);
    ASSERT(index != NULL);

    ASSERT(pq_is_trained(index) == 0);

    float train_data[TRAIN_COUNT * DIM];
    generate_training_data(train_data, TRAIN_COUNT, DIM);
    ASSERT(pq_train(index, train_data, TRAIN_COUNT) == 0);

    ASSERT(pq_is_trained(index) == 1);

    pq_destroy(index);
    return 0;
}

static int test_pq_range_search(void) {
    GV_PQConfig config;
    config.m           = M;
    config.nbits       = 8;
    config.train_iters = 10;

    void *index = pq_create(DIM, &config);
    ASSERT(index != NULL);

    float train_data[TRAIN_COUNT * DIM];
    generate_training_data(train_data, TRAIN_COUNT, DIM);
    ASSERT(pq_train(index, train_data, TRAIN_COUNT) == 0);

    for (size_t i = 0; i < INSERT_COUNT; i++) {
        GV_Vector *vec = vector_create_from_data(DIM, &train_data[i * DIM]);
        ASSERT(vec != NULL);
        ASSERT(pq_insert(index, vec) == 0);
        /* pq_insert takes ownership of vec */
    }

    float query[DIM];
    for (size_t j = 0; j < DIM; j++) {
        query[j] = sinf((float)j);
    }
    GV_Vector *qvec = vector_create_from_data(DIM, query);
    ASSERT(qvec != NULL);

    GV_SearchResult results[INSERT_COUNT];
    memset(results, 0, sizeof(results));
    int count = pq_range_search(index, qvec, 100.0f, results, INSERT_COUNT,
                                   GV_DISTANCE_EUCLIDEAN, NULL, NULL);
    ASSERT(count > 0);

    for (int i = 0; i < count; i++) {
        ASSERT(results[i].distance <= 100.0f);
    }

    vector_destroy(qvec);
    pq_destroy(index);
    return 0;
}

static int test_pq_delete_update(void) {
    GV_PQConfig config;
    config.m           = M;
    config.nbits       = 8;
    config.train_iters = 10;

    void *index = pq_create(DIM, &config);
    ASSERT(index != NULL);

    float train_data[TRAIN_COUNT * DIM];
    generate_training_data(train_data, TRAIN_COUNT, DIM);
    ASSERT(pq_train(index, train_data, TRAIN_COUNT) == 0);

    for (size_t i = 0; i < INSERT_COUNT; i++) {
        GV_Vector *vec = vector_create_from_data(DIM, &train_data[i * DIM]);
        ASSERT(vec != NULL);
        ASSERT(pq_insert(index, vec) == 0);
        /* pq_insert takes ownership of vec */
    }
    ASSERT(pq_count(index) == INSERT_COUNT);

    ASSERT(pq_delete(index, 0) == 0);

    float new_data[DIM];
    for (size_t j = 0; j < DIM; j++) {
        new_data[j] = 1.0f;
    }
    ASSERT(pq_update(index, 1, new_data, DIM) == 0);

    pq_destroy(index);
    return 0;
}

static int test_pq_db_integration(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_PQ);
    ASSERT(db != NULL);

    float train_data[TRAIN_COUNT * DIM];
    generate_training_data(train_data, TRAIN_COUNT, DIM);
    ASSERT(db_pq_train(db, train_data, TRAIN_COUNT, DIM) == 0);

    for (size_t i = 0; i < INSERT_COUNT; i++) {
        ASSERT(db_add_vector(db, &train_data[i * DIM], DIM) == 0);
    }

    float query[DIM];
    for (size_t j = 0; j < DIM; j++) {
        query[j] = sinf((float)j);
    }

    GV_SearchResult results[5];
    memset(results, 0, sizeof(results));
    int count = db_search(db, query, 5, results, GV_DISTANCE_EUCLIDEAN);
    ASSERT(count > 0);

    db_close(db);
    return 0;
}

static int test_pq_save_load(void) {
    const char *filepath = "test_pq_save.db";

    unlink(filepath);

    GV_Database *db = db_open(filepath, DIM, GV_INDEX_TYPE_PQ);
    ASSERT(db != NULL);

    float train_data[TRAIN_COUNT * DIM];
    generate_training_data(train_data, TRAIN_COUNT, DIM);
    ASSERT(db_pq_train(db, train_data, TRAIN_COUNT, DIM) == 0);

    for (size_t i = 0; i < INSERT_COUNT; i++) {
        ASSERT(db_add_vector(db, &train_data[i * DIM], DIM) == 0);
    }

    ASSERT(db_save(db, filepath) == 0);
    db_close(db);

    GV_Database *db2 = db_open(filepath, DIM, GV_INDEX_TYPE_PQ);
    ASSERT(db2 != NULL);

    float query[DIM];
    for (size_t j = 0; j < DIM; j++) {
        query[j] = sinf((float)j);
    }

    GV_SearchResult results[5];
    memset(results, 0, sizeof(results));
    int count = db_search(db2, query, 5, results, GV_DISTANCE_EUCLIDEAN);
    ASSERT(count > 0);

    db_close(db2);

    unlink(filepath);
    return 0;
}

int main(void) {
    int rc = 0;

    rc |= test_pq_create_destroy();
    rc |= test_pq_train_insert_search();
    rc |= test_pq_is_trained();
    rc |= test_pq_range_search();
    rc |= test_pq_delete_update();
    rc |= test_pq_db_integration();
    rc |= test_pq_save_load();

    return rc != 0;
}
