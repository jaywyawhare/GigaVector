#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "gigavector/gigavector.h"

#define ASSERT(cond) do { if (!(cond)) { fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #cond); return -1; } } while (0)

#define DIM 16
#define M   4
#define TRAIN_COUNT 200
#define INSERT_COUNT 50

/* Generate deterministic training data using sinf. */
static void generate_training_data(float *data, size_t count, size_t dim) {
    for (size_t i = 0; i < count; i++) {
        for (size_t j = 0; j < dim; j++) {
            data[i * dim + j] = sinf((float)(i * dim + j));
        }
    }
}

/* 1. test_pq_create_destroy                                          */
static int test_pq_create_destroy(void) {
    GV_PQConfig config;
    config.m           = M;
    config.nbits       = 8;
    config.train_iters = 10;

    void *index = gv_pq_create(DIM, &config);
    ASSERT(index != NULL);

    gv_pq_destroy(index);
    return 0;
}

/* 2. test_pq_train_insert_search                                     */
static int test_pq_train_insert_search(void) {
    GV_PQConfig config;
    config.m           = M;
    config.nbits       = 8;
    config.train_iters = 10;

    void *index = gv_pq_create(DIM, &config);
    ASSERT(index != NULL);

    /* Train */
    float train_data[TRAIN_COUNT * DIM];
    generate_training_data(train_data, TRAIN_COUNT, DIM);
    ASSERT(gv_pq_train(index, train_data, TRAIN_COUNT) == 0);

    /* Insert */
    for (size_t i = 0; i < INSERT_COUNT; i++) {
        GV_Vector *vec = gv_vector_create_from_data(DIM, &train_data[i * DIM]);
        ASSERT(vec != NULL);
        ASSERT(gv_pq_insert(index, vec) == 0);
        /* Note: gv_pq_insert takes ownership of the vector */
    }
    ASSERT(gv_pq_count(index) == INSERT_COUNT);

    /* Search */
    float query[DIM];
    for (size_t j = 0; j < DIM; j++) {
        query[j] = sinf((float)j);
    }
    GV_Vector *qvec = gv_vector_create_from_data(DIM, query);
    ASSERT(qvec != NULL);

    GV_SearchResult results[5];
    memset(results, 0, sizeof(results));
    int count = gv_pq_search(index, qvec, 5, results, GV_DISTANCE_EUCLIDEAN, NULL, NULL);
    ASSERT(count > 0);

    gv_vector_destroy(qvec);
    gv_pq_destroy(index);
    return 0;
}

/* 3. test_pq_is_trained                                              */
static int test_pq_is_trained(void) {
    GV_PQConfig config;
    config.m           = M;
    config.nbits       = 8;
    config.train_iters = 10;

    void *index = gv_pq_create(DIM, &config);
    ASSERT(index != NULL);

    /* Before training */
    ASSERT(gv_pq_is_trained(index) == 0);

    /* Train */
    float train_data[TRAIN_COUNT * DIM];
    generate_training_data(train_data, TRAIN_COUNT, DIM);
    ASSERT(gv_pq_train(index, train_data, TRAIN_COUNT) == 0);

    /* After training */
    ASSERT(gv_pq_is_trained(index) == 1);

    gv_pq_destroy(index);
    return 0;
}

/* 4. test_pq_range_search                                            */
static int test_pq_range_search(void) {
    GV_PQConfig config;
    config.m           = M;
    config.nbits       = 8;
    config.train_iters = 10;

    void *index = gv_pq_create(DIM, &config);
    ASSERT(index != NULL);

    /* Train */
    float train_data[TRAIN_COUNT * DIM];
    generate_training_data(train_data, TRAIN_COUNT, DIM);
    ASSERT(gv_pq_train(index, train_data, TRAIN_COUNT) == 0);

    /* Insert */
    for (size_t i = 0; i < INSERT_COUNT; i++) {
        GV_Vector *vec = gv_vector_create_from_data(DIM, &train_data[i * DIM]);
        ASSERT(vec != NULL);
        ASSERT(gv_pq_insert(index, vec) == 0);
        /* gv_pq_insert takes ownership of vec */
    }

    /* Range search with a generous radius */
    float query[DIM];
    for (size_t j = 0; j < DIM; j++) {
        query[j] = sinf((float)j);
    }
    GV_Vector *qvec = gv_vector_create_from_data(DIM, query);
    ASSERT(qvec != NULL);

    GV_SearchResult results[INSERT_COUNT];
    memset(results, 0, sizeof(results));
    int count = gv_pq_range_search(index, qvec, 100.0f, results, INSERT_COUNT,
                                   GV_DISTANCE_EUCLIDEAN, NULL, NULL);
    ASSERT(count > 0);

    /* Verify all returned results are within the radius */
    for (int i = 0; i < count; i++) {
        ASSERT(results[i].distance <= 100.0f);
    }

    gv_vector_destroy(qvec);
    gv_pq_destroy(index);
    return 0;
}

/* 5. test_pq_delete_update                                           */
static int test_pq_delete_update(void) {
    GV_PQConfig config;
    config.m           = M;
    config.nbits       = 8;
    config.train_iters = 10;

    void *index = gv_pq_create(DIM, &config);
    ASSERT(index != NULL);

    /* Train */
    float train_data[TRAIN_COUNT * DIM];
    generate_training_data(train_data, TRAIN_COUNT, DIM);
    ASSERT(gv_pq_train(index, train_data, TRAIN_COUNT) == 0);

    /* Insert */
    for (size_t i = 0; i < INSERT_COUNT; i++) {
        GV_Vector *vec = gv_vector_create_from_data(DIM, &train_data[i * DIM]);
        ASSERT(vec != NULL);
        ASSERT(gv_pq_insert(index, vec) == 0);
        /* gv_pq_insert takes ownership of vec */
    }
    ASSERT(gv_pq_count(index) == INSERT_COUNT);

    /* Delete entry 0 */
    ASSERT(gv_pq_delete(index, 0) == 0);

    /* Update entry 1 with new data */
    float new_data[DIM];
    for (size_t j = 0; j < DIM; j++) {
        new_data[j] = 1.0f;
    }
    ASSERT(gv_pq_update(index, 1, new_data, DIM) == 0);

    gv_pq_destroy(index);
    return 0;
}

/* 6. test_pq_db_integration                                          */
static int test_pq_db_integration(void) {
    GV_Database *db = gv_db_open(NULL, DIM, GV_INDEX_TYPE_PQ);
    ASSERT(db != NULL);

    /* Train */
    float train_data[TRAIN_COUNT * DIM];
    generate_training_data(train_data, TRAIN_COUNT, DIM);
    ASSERT(gv_db_pq_train(db, train_data, TRAIN_COUNT, DIM) == 0);

    /* Insert vectors */
    for (size_t i = 0; i < INSERT_COUNT; i++) {
        ASSERT(gv_db_add_vector(db, &train_data[i * DIM], DIM) == 0);
    }

    /* Search */
    float query[DIM];
    for (size_t j = 0; j < DIM; j++) {
        query[j] = sinf((float)j);
    }

    GV_SearchResult results[5];
    memset(results, 0, sizeof(results));
    int count = gv_db_search(db, query, 5, results, GV_DISTANCE_EUCLIDEAN);
    ASSERT(count > 0);

    gv_db_close(db);
    return 0;
}

/* 7. test_pq_save_load                                               */
static int test_pq_save_load(void) {
    const char *filepath = "test_pq_save.db";

    /* Remove any leftover file from a previous run */
    unlink(filepath);

    /* Open, train, insert, save */
    GV_Database *db = gv_db_open(filepath, DIM, GV_INDEX_TYPE_PQ);
    ASSERT(db != NULL);

    float train_data[TRAIN_COUNT * DIM];
    generate_training_data(train_data, TRAIN_COUNT, DIM);
    ASSERT(gv_db_pq_train(db, train_data, TRAIN_COUNT, DIM) == 0);

    for (size_t i = 0; i < INSERT_COUNT; i++) {
        ASSERT(gv_db_add_vector(db, &train_data[i * DIM], DIM) == 0);
    }

    ASSERT(gv_db_save(db, filepath) == 0);
    gv_db_close(db);

    /* Reopen and search */
    GV_Database *db2 = gv_db_open(filepath, DIM, GV_INDEX_TYPE_PQ);
    ASSERT(db2 != NULL);

    float query[DIM];
    for (size_t j = 0; j < DIM; j++) {
        query[j] = sinf((float)j);
    }

    GV_SearchResult results[5];
    memset(results, 0, sizeof(results));
    int count = gv_db_search(db2, query, 5, results, GV_DISTANCE_EUCLIDEAN);
    ASSERT(count > 0);

    gv_db_close(db2);

    /* Clean up */
    unlink(filepath);
    return 0;
}

/* main                                                               */
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
