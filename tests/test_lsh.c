#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "gigavector/gigavector.h"

#define ASSERT(cond) do { if (!(cond)) { fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #cond); return -1; } } while (0)

static int test_lsh_create_destroy(void) {
    const size_t dim = 8;
    GV_LSHConfig config;
    config.num_tables = 4;
    config.num_hash_bits = 8;
    config.seed = 42;

    GV_SoAStorage *storage = gv_soa_storage_create(dim, 0);
    ASSERT(storage != NULL);

    void *index = gv_lsh_create(dim, &config, storage);
    ASSERT(index != NULL);

    gv_lsh_destroy(index);
    gv_soa_storage_destroy(storage);

    return 0;
}

static int test_lsh_insert_search(void) {
    const size_t dim = 8;
    const int num_vectors = 20;
    GV_LSHConfig config;
    config.num_tables = 4;
    config.num_hash_bits = 8;
    config.seed = 42;

    GV_SoAStorage *storage = gv_soa_storage_create(dim, 0);
    ASSERT(storage != NULL);

    void *index = gv_lsh_create(dim, &config, storage);
    ASSERT(index != NULL);

    for (int i = 0; i < num_vectors; i++) {
        float data[8];
        for (int j = 0; j < (int)dim; j++) {
            data[j] = (float)(i * (int)dim + j) / 10.0f;
        }
        GV_Vector *vec = gv_vector_create_from_data(dim, data);
        ASSERT(vec != NULL);
        int rc = gv_lsh_insert(index, vec);
        ASSERT(rc == 0);
        /* gv_lsh_insert takes ownership — do NOT destroy vec */
    }

    ASSERT(gv_lsh_count(index) == (size_t)num_vectors);

    float query_data[8];
    for (int j = 0; j < (int)dim; j++) {
        query_data[j] = (float)j / 10.0f;
    }
    GV_Vector *query = gv_vector_create_from_data(dim, query_data);
    ASSERT(query != NULL);

    GV_SearchResult results[5];
    memset(results, 0, sizeof(results));
    int count = gv_lsh_search(index, query, 5, results, GV_DISTANCE_EUCLIDEAN, NULL, NULL);
    ASSERT(count > 0);

    gv_vector_destroy(query);
    gv_lsh_destroy(index);
    gv_soa_storage_destroy(storage);

    return 0;
}

static int test_lsh_range_search(void) {
    const size_t dim = 8;
    const int num_vectors = 20;
    GV_LSHConfig config;
    config.num_tables = 4;
    config.num_hash_bits = 8;
    config.seed = 42;

    GV_SoAStorage *storage = gv_soa_storage_create(dim, 0);
    ASSERT(storage != NULL);

    void *index = gv_lsh_create(dim, &config, storage);
    ASSERT(index != NULL);

    for (int i = 0; i < num_vectors; i++) {
        float data[8];
        for (int j = 0; j < (int)dim; j++) {
            data[j] = (float)(i * (int)dim + j) / 10.0f;
        }
        GV_Vector *vec = gv_vector_create_from_data(dim, data);
        ASSERT(vec != NULL);
        ASSERT(gv_lsh_insert(index, vec) == 0);
        /* gv_lsh_insert takes ownership — do NOT destroy vec */
    }

    float query_data[8] = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f};
    GV_Vector *query = gv_vector_create_from_data(dim, query_data);
    ASSERT(query != NULL);

    GV_SearchResult results[20];
    memset(results, 0, sizeof(results));
    int count = gv_lsh_range_search(index, query, 50.0f, results, 20,
                                    GV_DISTANCE_EUCLIDEAN, NULL, NULL);
    ASSERT(count >= 0);

    for (int i = 0; i < count; i++) {
        ASSERT(results[i].distance <= 50.0f);
    }

    gv_vector_destroy(query);
    gv_lsh_destroy(index);
    gv_soa_storage_destroy(storage);

    return 0;
}

static int test_lsh_delete(void) {
    const size_t dim = 8;
    GV_LSHConfig config;
    config.num_tables = 4;
    config.num_hash_bits = 8;
    config.seed = 42;

    GV_SoAStorage *storage = gv_soa_storage_create(dim, 0);
    ASSERT(storage != NULL);

    void *index = gv_lsh_create(dim, &config, storage);
    ASSERT(index != NULL);

    for (int i = 0; i < 5; i++) {
        float data[8];
        for (int j = 0; j < (int)dim; j++) {
            data[j] = (float)(i * (int)dim + j) / 10.0f;
        }
        GV_Vector *vec = gv_vector_create_from_data(dim, data);
        ASSERT(vec != NULL);
        ASSERT(gv_lsh_insert(index, vec) == 0);
        /* gv_lsh_insert takes ownership — do NOT destroy vec */
    }

    ASSERT(gv_lsh_count(index) == 5);

    int rc = gv_lsh_delete(index, 2);
    ASSERT(rc == 0);

    gv_lsh_destroy(index);
    gv_soa_storage_destroy(storage);

    return 0;
}

static int test_lsh_update(void) {
    const size_t dim = 8;
    GV_LSHConfig config;
    config.num_tables = 4;
    config.num_hash_bits = 8;
    config.seed = 42;

    GV_SoAStorage *storage = gv_soa_storage_create(dim, 0);
    ASSERT(storage != NULL);

    void *index = gv_lsh_create(dim, &config, storage);
    ASSERT(index != NULL);

    float data[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    GV_Vector *vec = gv_vector_create_from_data(dim, data);
    ASSERT(vec != NULL);
    ASSERT(gv_lsh_insert(index, vec) == 0);
    /* gv_lsh_insert takes ownership — do NOT destroy vec */

    ASSERT(gv_lsh_count(index) == 1);

    float new_data[8] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
    int rc = gv_lsh_update(index, 0, new_data, dim);
    ASSERT(rc == 0);

    gv_lsh_destroy(index);
    gv_soa_storage_destroy(storage);

    return 0;
}

static int test_lsh_db_integration(void) {
    const size_t dim = 8;

    GV_Database *db = gv_db_open(NULL, dim, GV_INDEX_TYPE_LSH);
    if (db == NULL) {
        return 0;
    }

    for (int i = 0; i < 20; i++) {
        float data[8];
        for (int j = 0; j < (int)dim; j++) {
            data[j] = (float)(i * (int)dim + j) / 10.0f;
        }
        ASSERT(gv_db_add_vector(db, data, dim) == 0);
    }

    float query[8] = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f};
    GV_SearchResult results[5];
    memset(results, 0, sizeof(results));
    int count = gv_db_search(db, query, 5, results, GV_DISTANCE_EUCLIDEAN);
    ASSERT(count > 0);

    for (int i = 0; i < count; i++) {
        ASSERT(results[i].distance >= 0.0f);
    }

    gv_db_close(db);

    return 0;
}

static int test_lsh_save_load(void) {
    const char *filepath = "test_lsh_save.db";
    const size_t dim = 8;

    unlink(filepath);

    GV_Database *db = gv_db_open(NULL, dim, GV_INDEX_TYPE_LSH);
    if (db == NULL) {
        return 0;
    }

    for (int i = 0; i < 10; i++) {
        float data[8];
        for (int j = 0; j < (int)dim; j++) {
            data[j] = (float)(i * (int)dim + j) / 10.0f;
        }
        ASSERT(gv_db_add_vector(db, data, dim) == 0);
    }

    ASSERT(gv_db_save(db, filepath) == 0);
    gv_db_close(db);

    GV_Database *db2 = gv_db_open(filepath, dim, GV_INDEX_TYPE_LSH);
    ASSERT(db2 != NULL);

    float query[8] = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f};
    GV_SearchResult results[5];
    memset(results, 0, sizeof(results));
    int count = gv_db_search(db2, query, 5, results, GV_DISTANCE_EUCLIDEAN);
    ASSERT(count > 0);

    gv_db_close(db2);

    unlink(filepath);

    return 0;
}

static int test_lsh_metadata_filter(void) {
    const size_t dim = 8;

    GV_Database *db = gv_db_open(NULL, dim, GV_INDEX_TYPE_LSH);
    if (db == NULL) {
        return 0;
    }

    for (int i = 0; i < 10; i++) {
        float data[8];
        for (int j = 0; j < (int)dim; j++) {
            data[j] = (float)(i * (int)dim + j) / 10.0f;
        }
        const char *value = (i % 2 == 0) ? "even" : "odd";
        ASSERT(gv_db_add_vector_with_metadata(db, data, dim, "category", value) == 0);
    }

    float query[8] = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f};
    GV_SearchResult results[10];
    memset(results, 0, sizeof(results));
    int count = gv_db_search_filtered(db, query, 5, results, GV_DISTANCE_EUCLIDEAN,
                                      "category", "even");
    ASSERT(count > 0);

    for (int i = 0; i < count; i++) {
        if (results[i].vector != NULL && results[i].vector->metadata != NULL) {
            GV_Metadata *md = results[i].vector->metadata;
            int found = 0;
            while (md != NULL) {
                if (strcmp(md->key, "category") == 0 && strcmp(md->value, "even") == 0) {
                    found = 1;
                    break;
                }
                md = md->next;
            }
            ASSERT(found);
        }
    }

    gv_db_close(db);

    return 0;
}

int main(void) {
    int rc = 0;

    rc |= test_lsh_create_destroy();
    rc |= test_lsh_insert_search();
    rc |= test_lsh_range_search();
    rc |= test_lsh_delete();
    rc |= test_lsh_update();
    rc |= test_lsh_db_integration();
    rc |= test_lsh_save_load();
    rc |= test_lsh_metadata_filter();

    return rc != 0;
}
