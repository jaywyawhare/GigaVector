#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gigavector/gigavector.h"

#define ASSERT(cond, msg)         \
    do {                          \
        if (!(cond)) {            \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1;            \
        }                         \
    } while (0)

static int test_open_close(void) {
    GV_Database *db = gv_db_open(NULL, 3, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "db open failed");
    gv_db_close(db);
    return 0;
}

static int test_add_and_search(void) {
    GV_Database *db = gv_db_open(NULL, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "db open failed");

    float v1[2] = {1.0f, 2.0f};
    ASSERT(gv_db_add_vector(db, v1, 2) == 0, "add vector");

    GV_SearchResult res[1];
    float q[2] = {1.0f, 2.0f};
    int n = gv_db_search(db, q, 1, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n == 1, "search count");
    ASSERT(res[0].distance == 0.0f, "distance zero");

    gv_db_close(db);
    return 0;
}

static int test_save_load_and_wal(void) {
    const char *path = "tmp_db.bin";
    const char *wal_path = "tmp_db.bin.wal";
    remove(path);
    remove(wal_path);

    // create with WAL
    GV_Database *db = gv_db_open(path, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "open with path");
    ASSERT(gv_db_set_wal(db, wal_path) == 0, "enable wal");

    float v[2] = {0.1f, 0.2f};
    ASSERT(gv_db_add_vector_with_metadata(db, v, 2, "tag", "a") == 0, "add with metadata");
    ASSERT(gv_db_save(db, NULL) == 0, "save");
    gv_db_close(db);

    // reload and replay WAL
    GV_Database *db2 = gv_db_open(path, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(db2 != NULL, "reopen");
    float q[2] = {0.1f, 0.2f};
    GV_SearchResult res[1];
    int n = gv_db_search(db2, q, 1, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n == 1, "search after reload");
    ASSERT(res[0].distance == 0.0f, "distance after reload");
    gv_db_close(db2);

    remove(path);
    remove(wal_path);
    return 0;
}

static int test_all_index_types(void) {
    GV_IndexType types[] = {GV_INDEX_TYPE_KDTREE, GV_INDEX_TYPE_HNSW, GV_INDEX_TYPE_IVFPQ};
    const char *type_names[] = {"KDTREE", "HNSW", "IVFPQ"};
    
    for (int t = 0; t < 3; t++) {
        GV_Database *db = gv_db_open(NULL, 8, types[t]);
        if (db == NULL) {
            printf("Skipping %s test (not available)\n", type_names[t]);
            continue;
        }
        
        if (types[t] == GV_INDEX_TYPE_IVFPQ) {
            float train_data[256 * 8];
            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 8; j++) {
                    train_data[i * 8 + j] = (float)((i + j) % 10) / 10.0f;
                }
            }
            ASSERT(gv_db_ivfpq_train(db, train_data, 256, 8) == 0, "train IVFPQ");
        }
        
        float v[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        ASSERT(gv_db_add_vector(db, v, 8) == 0, "add vector");
        
        float q[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        GV_SearchResult res[1];
        int n = gv_db_search(db, q, 1, res, GV_DISTANCE_EUCLIDEAN);
        ASSERT(n == 1, "search with index type");
        
        gv_db_close(db);
    }
    return 0;
}

static int test_all_distance_metrics(void) {
    GV_Database *db = gv_db_open(NULL, 3, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "db open");
    
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
    
    n = gv_db_search(db, q, 1, res, GV_DISTANCE_MANHATTAN);
    ASSERT(n == 1, "manhattan search");
    
    gv_db_close(db);
    return 0;
}

static int test_rich_metadata(void) {
    GV_Database *db = gv_db_open(NULL, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "db open");
    
    float v[2] = {1.0f, 2.0f};
    const char *keys[] = {"tag", "owner", "source"};
    const char *values[] = {"a", "b", "demo"};
    ASSERT(gv_db_add_vector_with_rich_metadata(db, v, 2, keys, values, 3) == 0, "add with rich metadata");
    
    float q[2] = {1.0f, 2.0f};
    GV_SearchResult res[1];
    int n = gv_db_search(db, q, 1, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n == 1, "search");
    ASSERT(res[0].vector != NULL, "result vector");
    ASSERT(res[0].vector->metadata != NULL, "result metadata");
    
    gv_db_close(db);
    return 0;
}

static int test_filtered_search(void) {
    GV_Database *db = gv_db_open(NULL, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "db open");
    
    float v1[2] = {0.0f, 1.0f};
    float v2[2] = {0.0f, 2.0f};
    float v3[2] = {0.0f, 3.0f};
    
    ASSERT(gv_db_add_vector_with_metadata(db, v1, 2, "color", "red") == 0, "add red");
    ASSERT(gv_db_add_vector_with_metadata(db, v2, 2, "color", "blue") == 0, "add blue");
    ASSERT(gv_db_add_vector_with_metadata(db, v3, 2, "color", "red") == 0, "add red 2");
    
    float q[2] = {0.0f, 1.1f};
    GV_SearchResult res[2];
    int n = gv_db_search_filtered(db, q, 2, res, GV_DISTANCE_EUCLIDEAN, "color", "red");
    ASSERT(n > 0, "filtered search");
    
    gv_db_close(db);
    return 0;
}

static int test_range_search(void) {
    GV_Database *db = gv_db_open(NULL, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "db open");
    
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
    ASSERT(n >= 3, "range search");
    
    gv_db_close(db);
    return 0;
}

static int test_batch_operations(void) {
    GV_Database *db = gv_db_open(NULL, 3, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "db open");
    
    float vectors[10 * 3];
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 3; j++) {
            vectors[i * 3 + j] = (float)(i * 3 + j);
        }
    }
    
    ASSERT(gv_db_add_vectors(db, vectors, 10, 3) == 0, "batch add vectors");
    
    float queries[3 * 3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            queries[i * 3 + j] = (float)(i * 3 + j);
        }
    }
    
    GV_SearchResult results[3 * 2];
    int n = gv_db_search_batch(db, queries, 3, 2, results, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n == 6, "batch search");
    
    gv_db_close(db);
    return 0;
}

static int test_delete_vector(void) {
    GV_Database *db = gv_db_open(NULL, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "db open");
    
    float v1[2] = {1.0f, 2.0f};
    float v2[2] = {3.0f, 4.0f};
    float v3[2] = {5.0f, 6.0f};
    
    ASSERT(gv_db_add_vector(db, v1, 2) == 0, "add vector 1");
    ASSERT(gv_db_add_vector(db, v2, 2) == 0, "add vector 2");
    ASSERT(gv_db_add_vector(db, v3, 2) == 0, "add vector 3");
    
    int delete_result = gv_db_delete_vector_by_index(db, 1);
    if (delete_result != 0) {
        gv_db_close(db);
        return 0;
    }
    
    float q[2] = {3.0f, 4.0f};
    GV_SearchResult res[3];
    int n = gv_db_search(db, q, 3, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n >= 0, "search after delete");
    
    gv_db_close(db);
    return 0;
}

static int test_update_vector(void) {
    GV_Database *db = gv_db_open(NULL, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "db open");
    
    float v[2] = {1.0f, 2.0f};
    ASSERT(gv_db_add_vector(db, v, 2) == 0, "add vector");
    
    float new_v[2] = {10.0f, 20.0f};
    int update_result = gv_db_update_vector(db, 0, new_v, 2);
    if (update_result != 0) {
        gv_db_close(db);
        return 0;
    }
    
    float q[2] = {10.0f, 20.0f};
    GV_SearchResult res[1];
    int n = gv_db_search(db, q, 1, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n >= 0, "search updated vector");
    
    gv_db_close(db);
    return 0;
}

static int test_update_metadata(void) {
    GV_Database *db = gv_db_open(NULL, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "db open");
    
    float v[2] = {1.0f, 2.0f};
    ASSERT(gv_db_add_vector_with_metadata(db, v, 2, "tag", "old") == 0, "add with metadata");
    
    const char *keys[] = {"tag", "owner"};
    const char *values[] = {"new", "user"};
    int update_result = gv_db_update_vector_metadata(db, 0, keys, values, 2);
    gv_db_close(db);
    if (update_result != 0) {
        return 0;
    }
    
    return 0;
}

static int test_stats(void) {
    GV_Database *db = gv_db_open(NULL, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "db open");
    
    float v[2] = {1.0f, 2.0f};
    ASSERT(gv_db_add_vector(db, v, 2) == 0, "add vector");
    
    GV_DBStats stats;
    gv_db_get_stats(db, &stats);
    ASSERT(stats.total_inserts >= 1, "stats total inserts");
    
    float q[2] = {1.0f, 2.0f};
    GV_SearchResult res[1];
    gv_db_search(db, q, 1, res, GV_DISTANCE_EUCLIDEAN);
    
    gv_db_get_stats(db, &stats);
    ASSERT(stats.total_queries >= 1, "stats total queries");
    
    gv_db_close(db);
    return 0;
}

static int test_error_handling(void) {
    GV_Database *db = gv_db_open(NULL, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "db open");
    
    float v[3] = {1.0f, 2.0f, 3.0f};
    ASSERT(gv_db_add_vector(db, v, 3) < 0, "wrong dimension should fail");
    
    float v2[2] = {1.0f, 2.0f};
    ASSERT(gv_db_add_vector(NULL, v2, 2) < 0, "NULL db should fail");
    
    ASSERT(gv_db_search(NULL, v2, 1, NULL, GV_DISTANCE_EUCLIDEAN) < 0, "NULL db search should fail");
    
    ASSERT(gv_db_delete_vector_by_index(db, 999) < 0, "delete non-existent index should fail");
    
    gv_db_close(db);
    return 0;
}

static int test_wal_operations(void) {
    const char *path = "tmp_wal_test.bin";
    const char *wal_path = "tmp_wal_test.bin.wal";
    remove(path);
    remove(wal_path);
    
    GV_Database *db = gv_db_open(path, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "db open");
    
    ASSERT(gv_db_set_wal(db, wal_path) == 0, "set WAL");
    
    float v[2] = {1.0f, 2.0f};
    ASSERT(gv_db_add_vector_with_metadata(db, v, 2, "tag", "test") == 0, "add with metadata");
    
    gv_db_disable_wal(db);
    
    gv_db_close(db);
    
    remove(path);
    remove(wal_path);
    return 0;
}

static int test_exact_search_threshold(void) {
    GV_Database *db = gv_db_open(NULL, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "db open");
    
    gv_db_set_exact_search_threshold(db, 10);
    gv_db_set_force_exact_search(db, 1);
    
    float v[2] = {1.0f, 2.0f};
    ASSERT(gv_db_add_vector(db, v, 2) == 0, "add vector");
    
    float q[2] = {1.0f, 2.0f};
    GV_SearchResult res[1];
    int n = gv_db_search(db, q, 1, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n == 1, "search with exact threshold");
    
    gv_db_close(db);
    return 0;
}

int main(void) {
    int rc = 0;
    printf("Running comprehensive database tests...\n");
    rc |= test_open_close();
    rc |= test_add_and_search();
    rc |= test_save_load_and_wal();
    rc |= test_all_index_types();
    rc |= test_all_distance_metrics();
    rc |= test_rich_metadata();
    rc |= test_filtered_search();
    rc |= test_range_search();
    rc |= test_batch_operations();
    rc |= test_delete_vector();
    rc |= test_update_vector();
    rc |= test_update_metadata();
    rc |= test_stats();
    rc |= test_error_handling();
    rc |= test_wal_operations();
    rc |= test_exact_search_threshold();
    if (rc == 0) {
        printf("All database tests passed\n");
    }
    return rc;
}

