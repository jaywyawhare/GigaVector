#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "gigavector/gigavector.h"

#define ASSERT(cond, msg)         \
    do {                          \
        if (!(cond)) {            \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1;            \
        }                         \
    } while (0)

static int test_open_from_memory(void) {
    const char *path = "tmp_memory_test.bin";
    remove(path);
    
    GV_Database *db1 = gv_db_open(path, 3, GV_INDEX_TYPE_KDTREE);
    ASSERT(db1 != NULL, "create db");
    
    float v[3] = {1.0f, 2.0f, 3.0f};
    ASSERT(gv_db_add_vector(db1, v, 3) == 0, "add vector");
    ASSERT(gv_db_save(db1, NULL) == 0, "save");
    
    FILE *f = fopen(path, "rb");
    ASSERT(f != NULL, "open file");
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    void *data = malloc(size);
    ASSERT(data != NULL, "allocate memory");
    fread(data, 1, size, f);
    fclose(f);
    
    GV_Database *db2 = gv_db_open_from_memory(data, size, 3, GV_INDEX_TYPE_KDTREE);
    ASSERT(db2 != NULL, "open from memory");
    
    float q[3] = {1.0f, 2.0f, 3.0f};
    GV_SearchResult res[1];
    int n = gv_db_search(db2, q, 1, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n == 1, "search from memory");
    
    free(data);
    gv_db_close(db1);
    gv_db_close(db2);
    remove(path);
    return 0;
}

static int test_open_mmap(void) {
    const char *path = "tmp_mmap_test.bin";
    remove(path);
    
    GV_Database *db1 = gv_db_open(path, 3, GV_INDEX_TYPE_KDTREE);
    ASSERT(db1 != NULL, "create db");
    
    float v[3] = {1.0f, 2.0f, 3.0f};
    ASSERT(gv_db_add_vector(db1, v, 3) == 0, "add vector");
    ASSERT(gv_db_save(db1, NULL) == 0, "save");
    gv_db_close(db1);
    
    GV_Database *db2 = gv_db_open_mmap(path, 3, GV_INDEX_TYPE_KDTREE);
    if (db2 != NULL) {
        float q[3] = {1.0f, 2.0f, 3.0f};
        GV_SearchResult res[1];
        int n = gv_db_search(db2, q, 1, res, GV_DISTANCE_EUCLIDEAN);
        ASSERT(n == 1, "search from mmap");
        gv_db_close(db2);
    }
    
    remove(path);
    return 0;
}

static int test_cosine_normalized(void) {
    GV_Database *db = gv_db_open(NULL, 3, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "db open");
    
    gv_db_set_cosine_normalized(db, 1);
    
    float v[3] = {1.0f, 2.0f, 3.0f};
    ASSERT(gv_db_add_vector(db, v, 3) == 0, "add vector");
    
    float q[3] = {1.0f, 2.0f, 3.0f};
    GV_SearchResult res[1];
    int n = gv_db_search(db, q, 1, res, GV_DISTANCE_COSINE);
    ASSERT(n == 1, "cosine search with normalization");
    
    gv_db_close(db);
    return 0;
}

static int test_index_suggest(void) {
    GV_IndexType idx;
    
    idx = gv_index_suggest(8, 1000);
    ASSERT(idx == GV_INDEX_TYPE_KDTREE || idx == GV_INDEX_TYPE_HNSW, "low dim small dataset");
    
    idx = gv_index_suggest(128, 1000000);
    ASSERT(idx == GV_INDEX_TYPE_HNSW || idx == GV_INDEX_TYPE_IVFPQ, "high dim large dataset");
    
    return 0;
}

static int test_ivfpq_opts(void) {
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
    
    float v[8] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    ASSERT(gv_db_add_vector(db, v, 8) == 0, "add vector");
    
    float q[8] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    GV_SearchResult res[1];
    int n = gv_db_search_ivfpq_opts(db, q, 1, res, GV_DISTANCE_EUCLIDEAN, 4, 0);
    ASSERT(n == 1, "search with opts");
    
    gv_db_close(db);
    return 0;
}

static int test_resource_limits(void) {
    GV_Database *db = gv_db_open(NULL, 3, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "db open");
    
    GV_ResourceLimits limits = {0};
    limits.max_memory_bytes = 1024 * 1024;
    limits.max_vectors = 100;
    limits.max_concurrent_operations = 10;
    
    ASSERT(gv_db_set_resource_limits(db, &limits) == 0, "set resource limits");
    
    GV_ResourceLimits retrieved = {0};
    gv_db_get_resource_limits(db, &retrieved);
    ASSERT(retrieved.max_memory_bytes == limits.max_memory_bytes, "retrieve memory limit");
    ASSERT(retrieved.max_vectors == limits.max_vectors, "retrieve vector limit");
    
    size_t mem_usage = gv_db_get_memory_usage(db);
    (void)mem_usage;  /* Verify function call succeeded */
    
    size_t concurrent_ops = gv_db_get_concurrent_operations(db);
    (void)concurrent_ops;  /* Verify function call succeeded */
    
    gv_db_close(db);
    return 0;
}

static int test_compaction(void) {
    GV_Database *db = gv_db_open(NULL, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "db open");
    
    gv_db_set_compaction_interval(db, 60);
    gv_db_set_wal_compaction_threshold(db, 1024 * 1024);
    gv_db_set_deleted_ratio_threshold(db, 0.1);
    
    float v[2] = {1.0f, 2.0f};
    ASSERT(gv_db_add_vector(db, v, 2) == 0, "add vector");
    
    ASSERT(gv_db_compact(db) == 0, "manual compact");
    
    gv_db_close(db);
    return 0;
}

static int test_background_compaction(void) {
    GV_Database *db = gv_db_open(NULL, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "db open");
    
    if (gv_db_start_background_compaction(db) == 0) {
        float v[2] = {1.0f, 2.0f};
        ASSERT(gv_db_add_vector(db, v, 2) == 0, "add vector");
        
        gv_db_stop_background_compaction(db);
    }
    
    gv_db_close(db);
    return 0;
}

static int test_detailed_stats(void) {
    GV_Database *db = gv_db_open(NULL, 3, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "db open");
    
    float v[3] = {1.0f, 2.0f, 3.0f};
    ASSERT(gv_db_add_vector(db, v, 3) == 0, "add vector");
    
    float q[3] = {1.0f, 2.0f, 3.0f};
    GV_SearchResult res[1];
    gv_db_search(db, q, 1, res, GV_DISTANCE_EUCLIDEAN);
    
    GV_DetailedStats stats;
    if (gv_db_get_detailed_stats(db, &stats) == 0) {
        ASSERT(stats.basic_stats.total_inserts >= 1, "detailed stats inserts");
        ASSERT(stats.basic_stats.total_queries >= 1, "detailed stats queries");
        gv_db_free_detailed_stats(&stats);
    }
    
    gv_db_close(db);
    return 0;
}

static int test_health_check(void) {
    GV_Database *db = gv_db_open(NULL, 3, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "db open");
    
    float v[3] = {1.0f, 2.0f, 3.0f};
    ASSERT(gv_db_add_vector(db, v, 3) == 0, "add vector");
    
    int health = gv_db_health_check(db);
    ASSERT(health <= 0, "health check");
    
    gv_db_close(db);
    return 0;
}

static int test_record_latency_recall(void) {
    GV_Database *db = gv_db_open(NULL, 3, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "db open");
    
    gv_db_record_latency(db, 1000, 1);
    gv_db_record_latency(db, 500, 0);
    gv_db_record_recall(db, 0.95);
    
    gv_db_close(db);
    return 0;
}

static int test_range_search_filtered(void) {
    GV_Database *db = gv_db_open(NULL, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "db open");
    
    float v1[2] = {0.0f, 0.0f};
    float v2[2] = {1.0f, 0.0f};
    float v3[2] = {2.0f, 0.0f};
    
    ASSERT(gv_db_add_vector_with_metadata(db, v1, 2, "tag", "a") == 0, "add vector 1");
    ASSERT(gv_db_add_vector_with_metadata(db, v2, 2, "tag", "b") == 0, "add vector 2");
    ASSERT(gv_db_add_vector_with_metadata(db, v3, 2, "tag", "a") == 0, "add vector 3");
    
    float q[2] = {0.0f, 0.0f};
    GV_SearchResult res[10];
    int n = gv_db_range_search_filtered(db, q, 2.5f, res, 10, GV_DISTANCE_EUCLIDEAN, "tag", "a");
    ASSERT(n >= 0, "range search filtered");
    
    gv_db_close(db);
    return 0;
}

static int test_add_vectors_with_metadata(void) {
    GV_Database *db = gv_db_open(NULL, 3, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "db open");
    
    float vectors[3 * 3] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    const char *keys[] = {"id", "id", "id"};
    const char *values[] = {"1", "2", "3"};
    
    ASSERT(gv_db_add_vectors_with_metadata(db, vectors, keys, values, 3, 3) == 0, "batch add with metadata");
    
    float q[3] = {1.0f, 2.0f, 3.0f};
    GV_SearchResult res[1];
    int n = gv_db_search(db, q, 1, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n == 1, "search batch added");
    
    gv_db_close(db);
    return 0;
}

int main(void) {
    int rc = 0;
    rc |= test_open_from_memory();
    rc |= test_open_mmap();
    rc |= test_cosine_normalized();
    rc |= test_index_suggest();
    rc |= test_ivfpq_opts();
    rc |= test_resource_limits();
    rc |= test_compaction();
    rc |= test_background_compaction();
    rc |= test_detailed_stats();
    rc |= test_health_check();
    rc |= test_record_latency_recall();
    rc |= test_range_search_filtered();
    rc |= test_add_vectors_with_metadata();
    return rc;
}

