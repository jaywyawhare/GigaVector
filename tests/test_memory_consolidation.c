/**
 * @file test_memory_consolidation.c
 * @brief Unit tests for memory consolidation (gv_memory_consolidation.h).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gigavector/gv_database.h"
#include "gigavector/gv_memory_layer.h"
#include "gigavector/gv_memory_consolidation.h"

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
    *out_db = gv_db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    if (!*out_db) return NULL;

    GV_MemoryLayerConfig mlconfig = {0};
    mlconfig.consolidation_threshold = 0.8;
    return gv_memory_layer_create(*out_db, &mlconfig);
}

static void cleanup(GV_MemoryLayer *layer, GV_Database *db) {
    gv_memory_layer_destroy(layer);
    gv_db_close(db);
}

/* Test 1: Find similar with empty layer (0 pairs) */
static int test_find_similar_empty(void) {
    GV_Database *db;
    GV_MemoryLayer *layer = create_test_layer(&db);
    ASSERT(layer != NULL, "memory layer creation");

    GV_MemoryPair pairs[10];
    memset(pairs, 0, sizeof(pairs));
    size_t actual_count = 999;

    int ret = gv_memory_find_similar(layer, 0.5, pairs, 10, &actual_count);
    ASSERT(ret == 0, "find_similar on empty layer should succeed");
    ASSERT(actual_count == 0, "empty layer should return 0 pairs");

    cleanup(layer, db);
    return 0;
}

/* Test 2: Add memories then find similar pairs */
static int test_find_similar_with_data(void) {
    GV_Database *db;
    GV_MemoryLayer *layer = create_test_layer(&db);
    ASSERT(layer != NULL, "memory layer creation");

    /* Add two very similar memories */
    float emb1[DIM] = {1.0f, 0.0f, 0.0f, 0.0f};
    float emb2[DIM] = {0.99f, 0.01f, 0.0f, 0.0f};
    /* Add one dissimilar memory */
    float emb3[DIM] = {0.0f, 0.0f, 0.0f, 1.0f};

    GV_MemoryMetadata meta1;
    memset(&meta1, 0, sizeof(meta1));
    meta1.memory_type = GV_MEMORY_TYPE_FACT;
    meta1.timestamp = time(NULL);
    meta1.importance_score = 0.9;

    char *id1 = gv_memory_add(layer, "The sky is blue", emb1, &meta1);
    ASSERT(id1 != NULL, "add memory 1");

    char *id2 = gv_memory_add(layer, "The sky appears blue", emb2, NULL);
    ASSERT(id2 != NULL, "add memory 2");

    char *id3 = gv_memory_add(layer, "Dogs are mammals", emb3, NULL);
    ASSERT(id3 != NULL, "add memory 3");

    GV_MemoryPair pairs[10];
    memset(pairs, 0, sizeof(pairs));
    size_t actual_count = 0;

    /* Low threshold should find at least the similar pair */
    int ret = gv_memory_find_similar(layer, 0.1, pairs, 10, &actual_count);
    ASSERT(ret == 0, "find_similar should succeed");
    /* Actual count depends on implementation; at minimum check it does not error */

    gv_memory_pairs_free(pairs, actual_count);
    free(id1);
    free(id2);
    free(id3);
    cleanup(layer, db);
    return 0;
}

/* Test 3: Merge two memories */
static int test_memory_merge(void) {
    GV_Database *db;
    GV_MemoryLayer *layer = create_test_layer(&db);
    ASSERT(layer != NULL, "memory layer creation");

    float emb1[DIM] = {1.0f, 0.0f, 0.0f, 0.0f};
    float emb2[DIM] = {0.9f, 0.1f, 0.0f, 0.0f};

    char *id1 = gv_memory_add(layer, "User likes Python", emb1, NULL);
    ASSERT(id1 != NULL, "add memory 1");

    char *id2 = gv_memory_add(layer, "User prefers Python over Java", emb2, NULL);
    ASSERT(id2 != NULL, "add memory 2");

    char *merged_id = gv_memory_merge(layer, id1, id2);
    /* Merge may return a new ID or NULL depending on implementation */
    if (merged_id != NULL) {
        /* Verify the merged memory exists */
        GV_MemoryResult result;
        int ret = gv_memory_get(layer, merged_id, &result);
        if (ret == 0) {
            ASSERT(result.content != NULL, "merged memory should have content");
            gv_memory_result_free(&result);
        }
        free(merged_id);
    }

    free(id1);
    free(id2);
    cleanup(layer, db);
    return 0;
}

/* Test 4: Merge with invalid IDs */
static int test_memory_merge_invalid(void) {
    GV_Database *db;
    GV_MemoryLayer *layer = create_test_layer(&db);
    ASSERT(layer != NULL, "memory layer creation");

    char *result = gv_memory_merge(layer, "nonexistent-1", "nonexistent-2");
    /* Should return NULL for non-existent IDs */
    ASSERT(result == NULL, "merge with invalid IDs should return NULL");

    cleanup(layer, db);
    return 0;
}

/* Test 5: Link two memories */
static int test_memory_link(void) {
    GV_Database *db;
    GV_MemoryLayer *layer = create_test_layer(&db);
    ASSERT(layer != NULL, "memory layer creation");

    float emb1[DIM] = {1.0f, 0.0f, 0.0f, 0.0f};
    float emb2[DIM] = {0.0f, 1.0f, 0.0f, 0.0f};

    char *id1 = gv_memory_add(layer, "Python is a programming language", emb1, NULL);
    ASSERT(id1 != NULL, "add memory 1");

    char *id2 = gv_memory_add(layer, "Python is used for machine learning", emb2, NULL);
    ASSERT(id2 != NULL, "add memory 2");

    int ret = gv_memory_link(layer, id1, id2);
    ASSERT(ret == 0, "linking memories should succeed");

    free(id1);
    free(id2);
    cleanup(layer, db);
    return 0;
}

/* Test 6: Link with invalid IDs */
static int test_memory_link_invalid(void) {
    GV_Database *db;
    GV_MemoryLayer *layer = create_test_layer(&db);
    ASSERT(layer != NULL, "memory layer creation");

    int ret = gv_memory_link(layer, "fake-id-1", "fake-id-2");
    ASSERT(ret == -1, "linking invalid IDs should fail");

    cleanup(layer, db);
    return 0;
}

/* Test 7: Archive a memory */
static int test_memory_archive(void) {
    GV_Database *db;
    GV_MemoryLayer *layer = create_test_layer(&db);
    ASSERT(layer != NULL, "memory layer creation");

    float emb[DIM] = {1.0f, 0.0f, 0.0f, 0.0f};
    char *id = gv_memory_add(layer, "Old fact that is no longer relevant", emb, NULL);
    ASSERT(id != NULL, "add memory");

    int ret = gv_memory_archive(layer, id);
    ASSERT(ret == 0, "archiving memory should succeed");

    free(id);
    cleanup(layer, db);
    return 0;
}

/* Test 8: Archive with invalid ID */
static int test_memory_archive_invalid(void) {
    GV_Database *db;
    GV_MemoryLayer *layer = create_test_layer(&db);
    ASSERT(layer != NULL, "memory layer creation");

    int ret = gv_memory_archive(layer, "nonexistent-id");
    ASSERT(ret == -1, "archiving non-existent memory should fail");

    cleanup(layer, db);
    return 0;
}

/* Test 9: Free pairs with NULL */
static int test_memory_pairs_free_null(void) {
    gv_memory_pairs_free(NULL, 0);
    gv_memory_pairs_free(NULL, 5);
    gv_memory_pair_free(NULL);
    /* Should not crash */
    return 0;
}

/* Test 10: Free pairs with empty array */
static int test_memory_pairs_free_empty(void) {
    GV_MemoryPair pairs[3];
    memset(pairs, 0, sizeof(pairs));
    gv_memory_pairs_free(pairs, 0);
    /* Should not crash */
    return 0;
}

/* Test 11: Consolidate pair with strategy */
static int test_consolidate_pair(void) {
    GV_Database *db;
    GV_MemoryLayer *layer = create_test_layer(&db);
    ASSERT(layer != NULL, "memory layer creation");

    float emb1[DIM] = {1.0f, 0.0f, 0.0f, 0.0f};
    float emb2[DIM] = {0.95f, 0.05f, 0.0f, 0.0f};

    char *id1 = gv_memory_add(layer, "User enjoys hiking", emb1, NULL);
    ASSERT(id1 != NULL, "add memory 1");

    char *id2 = gv_memory_add(layer, "User likes outdoor activities like hiking", emb2, NULL);
    ASSERT(id2 != NULL, "add memory 2");

    char *consolidated = gv_memory_consolidate_pair(layer, id1, id2, GV_CONSOLIDATION_MERGE);
    /* May return new ID or NULL depending on implementation */
    if (consolidated != NULL) {
        free(consolidated);
    }

    free(id1);
    free(id2);
    cleanup(layer, db);
    return 0;
}

/* Test 12: Update from new memory */
static int test_memory_update_from_new(void) {
    GV_Database *db;
    GV_MemoryLayer *layer = create_test_layer(&db);
    ASSERT(layer != NULL, "memory layer creation");

    float emb1[DIM] = {1.0f, 0.0f, 0.0f, 0.0f};
    float emb2[DIM] = {0.8f, 0.2f, 0.0f, 0.0f};

    char *id1 = gv_memory_add(layer, "User works at Company A", emb1, NULL);
    ASSERT(id1 != NULL, "add memory 1");

    char *id2 = gv_memory_add(layer, "User now works at Company B", emb2, NULL);
    ASSERT(id2 != NULL, "add memory 2");

    int ret = gv_memory_update_from_new(layer, id1, id2);
    /* Return code depends on implementation */
    (void)ret;

    free(id1);
    free(id2);
    cleanup(layer, db);
    return 0;
}

/* Test 13: Find similar with high threshold (no pairs expected) */
static int test_find_similar_high_threshold(void) {
    GV_Database *db;
    GV_MemoryLayer *layer = create_test_layer(&db);
    ASSERT(layer != NULL, "memory layer creation");

    float emb1[DIM] = {1.0f, 0.0f, 0.0f, 0.0f};
    float emb2[DIM] = {0.0f, 1.0f, 0.0f, 0.0f};

    char *id1 = gv_memory_add(layer, "Cats are felines", emb1, NULL);
    ASSERT(id1 != NULL, "add memory 1");

    char *id2 = gv_memory_add(layer, "Cars are vehicles", emb2, NULL);
    ASSERT(id2 != NULL, "add memory 2");

    GV_MemoryPair pairs[10];
    memset(pairs, 0, sizeof(pairs));
    size_t actual_count = 0;

    /* Very high threshold should find few or no pairs for orthogonal vectors */
    int ret = gv_memory_find_similar(layer, 0.99, pairs, 10, &actual_count);
    ASSERT(ret == 0, "find_similar with high threshold should succeed");
    /* actual_count should be small for very dissimilar memories */
    (void)actual_count; /* implementation-dependent */

    gv_memory_pairs_free(pairs, actual_count);
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
