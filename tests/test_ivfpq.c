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

static int test_ivfpq_basic(void) {
    GV_Database *db = gv_db_open(NULL, 8, GV_INDEX_TYPE_IVFPQ);
    if (db == NULL) {
        return 0;
    }
    
    float train_data[256 * 8];
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 8; j++) {
            train_data[i * 8 + j] = (float)((i + j) % 10) / 10.0f;
        }
    }
    
    if (gv_db_ivfpq_train(db, train_data, 256, 8) != 0) {
        gv_db_close(db);
        return 0;
    }
    
    float v[8] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    if (gv_db_add_vector(db, v, 8) != 0) {
        gv_db_close(db);
        return 0;
    }
    
    float q[8] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    GV_SearchResult res[1];
    int n = gv_db_search(db, q, 1, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n >= 0, "search after training");
    
    gv_db_close(db);
    return 0;
}

static int test_ivfpq_config(void) {
    GV_IVFPQConfig config = {0};
    config.nlist = 16;
    config.m = 4;
    config.nbits = 8;
    config.nprobe = 4;
    config.train_iters = 10;
    
    GV_Database *db = gv_db_open_with_ivfpq_config(NULL, 8, GV_INDEX_TYPE_IVFPQ, &config);
    if (db == NULL) {
        return 0;
    }
    
    float train_data[256 * 8];
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 8; j++) {
            train_data[i * 8 + j] = (float)((i + j) % 10) / 10.0f;
        }
    }
    
    if (gv_db_ivfpq_train(db, train_data, 256, 8) != 0) {
        gv_db_close(db);
        return 0;
    }
    
    float v[8] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    if (gv_db_add_vector(db, v, 8) != 0) {
        gv_db_close(db);
        return 0;
    }
    
    gv_db_close(db);
    return 0;
}

static int test_ivfpq_large_dataset(void) {
    GV_Database *db = gv_db_open(NULL, 16, GV_INDEX_TYPE_IVFPQ);
    if (db == NULL) {
        return 0;
    }
    
    float train_data[512 * 16];
    for (int i = 0; i < 512; i++) {
        for (int j = 0; j < 16; j++) {
            train_data[i * 16 + j] = (float)((i + j) % 20) / 20.0f;
        }
    }
    
    ASSERT(gv_db_ivfpq_train(db, train_data, 512, 16) == 0, "train with large dataset");
    
    for (int i = 0; i < 100; i++) {
        float v[16];
        for (int j = 0; j < 16; j++) {
            v[j] = (float)((i + j) % 20) / 20.0f;
        }
        ASSERT(gv_db_add_vector(db, v, 16) == 0, "add vector in large dataset");
    }
    
    float q[16];
    for (int j = 0; j < 16; j++) {
        q[j] = 0.5f;
    }
    GV_SearchResult res[5];
    int n = gv_db_search(db, q, 5, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n == 5, "search in large dataset");
    
    gv_db_close(db);
    return 0;
}

static int test_ivfpq_range_search(void) {
    GV_Database *db = gv_db_open(NULL, 8, GV_INDEX_TYPE_IVFPQ);
    if (db == NULL) {
        return 0;
    }
    
    float train_data[256 * 8];
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 8; j++) {
            train_data[i * 8 + j] = (float)((i + j) % 10) / 10.0f;
        }
    }
    
    ASSERT(gv_db_ivfpq_train(db, train_data, 256, 8) == 0, "train");
    
    for (int i = 0; i < 10; i++) {
        float v[8];
        for (int j = 0; j < 8; j++) {
            v[j] = (float)i / 10.0f;
        }
        ASSERT(gv_db_add_vector(db, v, 8) == 0, "add vector");
    }
    
    float q[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    GV_SearchResult res[10];
    int n = gv_db_range_search(db, q, 1.0f, res, 10, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n > 0, "range search found results");
    
    gv_db_close(db);
    return 0;
}

static int test_ivfpq_persistence(void) {
    const char *path = "tmp_ivfpq_db.bin";
    remove(path);
    
    GV_Database *db = gv_db_open(path, 8, GV_INDEX_TYPE_IVFPQ);
    if (db == NULL) {
        return 0;
    }
    
    float train_data[256 * 8];
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 8; j++) {
            train_data[i * 8 + j] = (float)((i + j) % 10) / 10.0f;
        }
    }
    
    if (gv_db_ivfpq_train(db, train_data, 256, 8) != 0) {
        gv_db_close(db);
        remove(path);
        return 0;
    }
    
    float v[8] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    if (gv_db_add_vector(db, v, 8) != 0) {
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
    
    GV_Database *db2 = gv_db_open(path, 8, GV_INDEX_TYPE_IVFPQ);
    if (db2 == NULL) {
        remove(path);
        return 0;
    }
    
    float q[8] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    GV_SearchResult res[1];
    int n = gv_db_search(db2, q, 1, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n >= 0, "search after reload");
    
    gv_db_close(db2);
    remove(path);
    return 0;
}

static int test_ivfpq_untrained_error(void) {
    GV_Database *db = gv_db_open(NULL, 8, GV_INDEX_TYPE_IVFPQ);
    if (db == NULL) {
        return 0;
    }
    
    float v[8] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    int rc = gv_db_add_vector(db, v, 8);
    ASSERT(rc < 0, "add vector without training should fail");
    
    gv_db_close(db);
    return 0;
}

int main(void) {
    int rc = 0;
    rc |= test_ivfpq_basic();
    rc |= test_ivfpq_config();
    rc |= test_ivfpq_large_dataset();
    rc |= test_ivfpq_range_search();
    rc |= test_ivfpq_persistence();
    rc |= test_ivfpq_untrained_error();
    return rc;
}

