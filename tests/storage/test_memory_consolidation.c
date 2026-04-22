/**
 * @file test_memory_consolidation.c
 * @brief Unit tests for memory consolidation (memory_consolidation.h).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "storage/database.h"
#include "storage/memory_layer.h"
#include "storage/memory_consolidation.h"

#define ASSERT(cond, msg)         \
    do {                          \
        if (!(cond)) {            \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1;            \
        }                         \
    } while (0)

#define TEST_DB "tmp_test_memcons.bin"
#define DIM 4

static GV_MemoryLayer *create_test_layer(GV_Database **out_db) {
    *out_db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    if (!*out_db) return NULL;

    GV_MemoryLayerConfig mlconfig = {0};
    mlconfig.consolidation_threshold = 0.8;
    return memory_layer_create(*out_db, &mlconfig);
}

static void cleanup(GV_MemoryLayer *layer, GV_Database *db) {
    memory_layer_destroy(layer);
    db_close(db);
}

static int test_find_similar_empty(void) {
    GV_Database *db;
    GV_MemoryLayer *layer = create_test_layer(&db);
    ASSERT(layer != NULL, "memory layer creation");

    GV_MemoryPair pairs[10];
    memset(pairs, 0, sizeof(pairs));
    size_t actual_count = 999;

    int ret = memory_find_similar(layer, 0.5, pairs, 10, &actual_count);
    ASSERT(ret == 0, "find_similar on empty layer should succeed");
    ASSERT(actual_count == 0, "empty layer should return 0 pairs");

    cleanup(layer, db);
    return 0;
}

static int test_find_similar_with_data(void) {
    GV_Database *db;
    GV_MemoryLayer *layer = create_test_layer(&db);
    ASSERT(layer != NULL, "memory layer creation");

    float emb1[DIM] = {1.0f, 0.0f, 0.0f, 0.0f};
    float emb2[DIM] = {0.99f, 0.01f, 0.0f, 0.0f};
    float emb3[DIM] = {0.0f, 0.0f, 0.0f, 1.0f};

    GV_MemoryMetadata meta1;
    memset(&meta1, 0, sizeof(meta1));
    meta1.memory_type = GV_MEMORY_TYPE_FACT;
    meta1.timestamp = time(NULL);
    meta1.importance_score = 0.9;

    char *id1 = memory_add(layer, "The sky is blue", emb1, &meta1);
    ASSERT(id1 != NULL, "add memory 1");

    char *id2 = memory_add(layer, "The sky appears blue", emb2, NULL);
    ASSERT(id2 != NULL, "add memory 2");

    char *id3 = memory_add(layer, "Dogs are mammals", emb3, NULL);
    ASSERT(id3 != NULL, "add memory 3");

    GV_MemoryPair pairs[10];
    memset(pairs, 0, sizeof(pairs));
    size_t actual_count = 0;

    int ret = memory_find_similar(layer, 0.1, pairs, 10, &actual_count);
    ASSERT(ret == 0, "find_similar should succeed");

    memory_pairs_free(pairs, actual_count);
    free(id1);
    free(id2);
    free(id3);
    cleanup(layer, db);
    return 0;
}

static int test_memory_merge(void) {
    GV_Database *db;
    GV_MemoryLayer *layer = create_test_layer(&db);
    ASSERT(layer != NULL, "memory layer creation");

    float emb1[DIM] = {1.0f, 0.0f, 0.0f, 0.0f};
    float emb2[DIM] = {0.9f, 0.1f, 0.0f, 0.0f};

    char *id1 = memory_add(layer, "User likes Python", emb1, NULL);
    ASSERT(id1 != NULL, "add memory 1");

    char *id2 = memory_add(layer, "User prefers Python over Java", emb2, NULL);
    ASSERT(id2 != NULL, "add memory 2");

    char *merged_id = memory_merge(layer, id1, id2);
    if (merged_id != NULL) {
        GV_MemoryResult result;
        int ret = memory_get(layer, merged_id, &result);
        if (ret == 0) {
            ASSERT(result.content != NULL, "merged memory should have content");
            memory_result_free(&result);
        }
        free(merged_id);
    }

    free(id1);
    free(id2);
    cleanup(layer, db);
    return 0;
}

static int test_memory_merge_invalid(void) {
    GV_Database *db;
    GV_MemoryLayer *layer = create_test_layer(&db);
    ASSERT(layer != NULL, "memory layer creation");

    char *result = memory_merge(layer, "nonexistent-1", "nonexistent-2");
    ASSERT(result == NULL, "merge with invalid IDs should return NULL");

    cleanup(layer, db);
    return 0;
}

static int test_memory_link(void) {
    GV_Database *db;
    GV_MemoryLayer *layer = create_test_layer(&db);
    ASSERT(layer != NULL, "memory layer creation");

    float emb1[DIM] = {1.0f, 0.0f, 0.0f, 0.0f};
    float emb2[DIM] = {0.0f, 1.0f, 0.0f, 0.0f};

    char *id1 = memory_add(layer, "Python is a programming language", emb1, NULL);
    ASSERT(id1 != NULL, "add memory 1");

    char *id2 = memory_add(layer, "Python is used for machine learning", emb2, NULL);
    ASSERT(id2 != NULL, "add memory 2");

    int ret = memory_link(layer, id1, id2);
    ASSERT(ret == 0, "linking memories should succeed");

    free(id1);
    free(id2);
    cleanup(layer, db);
    return 0;
}

static int test_memory_link_invalid(void) {
    GV_Database *db;
    GV_MemoryLayer *layer = create_test_layer(&db);
    ASSERT(layer != NULL, "memory layer creation");

    int ret = memory_link(layer, "fake-id-1", "fake-id-2");
    ASSERT(ret == -1, "linking invalid IDs should fail");

    cleanup(layer, db);
    return 0;
}

static int test_memory_archive(void) {
    GV_Database *db;
    GV_MemoryLayer *layer = create_test_layer(&db);
    ASSERT(layer != NULL, "memory layer creation");

    float emb[DIM] = {1.0f, 0.0f, 0.0f, 0.0f};
    char *id = memory_add(layer, "Old fact that is no longer relevant", emb, NULL);
    ASSERT(id != NULL, "add memory");

    int ret = memory_archive(layer, id);
    ASSERT(ret == 0, "archiving memory should succeed");

    free(id);
    cleanup(layer, db);
    return 0;
}

static int test_memory_archive_invalid(void) {
    GV_Database *db;
    GV_MemoryLayer *layer = create_test_layer(&db);
    ASSERT(layer != NULL, "memory layer creation");

    int ret = memory_archive(layer, "nonexistent-id");
    ASSERT(ret == -1, "archiving non-existent memory should fail");

    cleanup(layer, db);
    return 0;
}

static int test_memory_pairs_free_null(void) {
    memory_pairs_free(NULL, 0);
    memory_pairs_free(NULL, 5);
    memory_pair_free(NULL);
    return 0;
}

static int test_memory_pairs_free_empty(void) {
    GV_MemoryPair pairs[3];
    memset(pairs, 0, sizeof(pairs));
    memory_pairs_free(pairs, 0);
    return 0;
}

static int test_consolidate_pair(void) {
    GV_Database *db;
    GV_MemoryLayer *layer = create_test_layer(&db);
    ASSERT(layer != NULL, "memory layer creation");

    float emb1[DIM] = {1.0f, 0.0f, 0.0f, 0.0f};
    float emb2[DIM] = {0.95f, 0.05f, 0.0f, 0.0f};

    char *id1 = memory_add(layer, "User enjoys hiking", emb1, NULL);
    ASSERT(id1 != NULL, "add memory 1");

    char *id2 = memory_add(layer, "User likes outdoor activities like hiking", emb2, NULL);
    ASSERT(id2 != NULL, "add memory 2");

    char *consolidated = memory_consolidate_pair(layer, id1, id2, GV_CONSOLIDATION_MERGE);
    if (consolidated != NULL) {
        free(consolidated);
    }

    free(id1);
    free(id2);
    cleanup(layer, db);
    return 0;
}

static int test_memory_update_from_new(void) {
    GV_Database *db;
    GV_MemoryLayer *layer = create_test_layer(&db);
    ASSERT(layer != NULL, "memory layer creation");

    float emb1[DIM] = {1.0f, 0.0f, 0.0f, 0.0f};
    float emb2[DIM] = {0.8f, 0.2f, 0.0f, 0.0f};

    char *id1 = memory_add(layer, "User works at Company A", emb1, NULL);
    ASSERT(id1 != NULL, "add memory 1");

    char *id2 = memory_add(layer, "User now works at Company B", emb2, NULL);
    ASSERT(id2 != NULL, "add memory 2");

    int ret = memory_update_from_new(layer, id1, id2);
    (void)ret;

    free(id1);
    free(id2);
    cleanup(layer, db);
    return 0;
}

static int test_find_similar_high_threshold(void) {
    GV_Database *db;
    GV_MemoryLayer *layer = create_test_layer(&db);
    ASSERT(layer != NULL, "memory layer creation");

    float emb1[DIM] = {1.0f, 0.0f, 0.0f, 0.0f};
    float emb2[DIM] = {0.0f, 1.0f, 0.0f, 0.0f};

    char *id1 = memory_add(layer, "Cats are felines", emb1, NULL);
    ASSERT(id1 != NULL, "add memory 1");

    char *id2 = memory_add(layer, "Cars are vehicles", emb2, NULL);
    ASSERT(id2 != NULL, "add memory 2");

    GV_MemoryPair pairs[10];
    memset(pairs, 0, sizeof(pairs));
    size_t actual_count = 0;

    int ret = memory_find_similar(layer, 0.99, pairs, 10, &actual_count);
    ASSERT(ret == 0, "find_similar with high threshold should succeed");
    (void)actual_count;

    memory_pairs_free(pairs, actual_count);
    free(id1);
    free(id2);
    cleanup(layer, db);
    return 0;
}

int main(void) {
    int failed = 0;
    int passed = 0;

    remove(TEST_DB);

    struct { const char *name; int (*fn)(void); } tests[] = {
        {"test_find_similar_empty",           test_find_similar_empty},
        {"test_find_similar_with_data",       test_find_similar_with_data},
        {"test_memory_merge",                 test_memory_merge},
        {"test_memory_merge_invalid",         test_memory_merge_invalid},
        {"test_memory_link",                  test_memory_link},
        {"test_memory_link_invalid",          test_memory_link_invalid},
        {"test_memory_archive",              test_memory_archive},
        {"test_memory_archive_invalid",      test_memory_archive_invalid},
        {"test_memory_pairs_free_null",      test_memory_pairs_free_null},
        {"test_memory_pairs_free_empty",     test_memory_pairs_free_empty},
        {"test_consolidate_pair",            test_consolidate_pair},
        {"test_memory_update_from_new",      test_memory_update_from_new},
        {"test_find_similar_high_threshold", test_find_similar_high_threshold},
    };

    int num_tests = (int)(sizeof(tests) / sizeof(tests[0]));
    for (int i = 0; i < num_tests; i++) {
        int result = tests[i].fn();
        if (result == 0) {
            printf("  OK   %s\n", tests[i].name);
            passed++;
        } else {
            printf("  FAILED %s\n", tests[i].name);
            failed++;
        }
    }

    printf("\n%d/%d tests passed\n", passed, num_tests);
    remove(TEST_DB);
    return failed > 0 ? 1 : 0;
}
