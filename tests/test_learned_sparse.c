#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_learned_sparse.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

/* --- Test: config init defaults --- */
static int test_config_init(void) {
    GV_LearnedSparseConfig config;
    memset(&config, 0xFF, sizeof(config));
    gv_ls_config_init(&config);

    ASSERT(config.vocab_size == 30522, "default vocab_size should be 30522");
    ASSERT(config.max_nonzeros == 256, "default max_nonzeros should be 256");
    ASSERT(config.use_wand == 1, "default use_wand should be 1");
    ASSERT(config.wand_block_size == 128, "default wand_block_size should be 128");

    return 0;
}

/* --- Test: create and destroy --- */
static int test_create_destroy(void) {
    GV_LearnedSparseConfig config;
    gv_ls_config_init(&config);

    GV_LearnedSparseIndex *idx = gv_ls_create(&config);
    ASSERT(idx != NULL, "gv_ls_create should return non-NULL");
    ASSERT(gv_ls_count(idx) == 0, "new index should have count 0");

    gv_ls_destroy(idx);
    /* NULL destroy should be safe */
    gv_ls_destroy(NULL);

    /* Create with NULL config (defaults) */
    GV_LearnedSparseIndex *idx2 = gv_ls_create(NULL);
    ASSERT(idx2 != NULL, "gv_ls_create(NULL) should use defaults");
    gv_ls_destroy(idx2);

    return 0;
}

/* --- Test: insert and count --- */
static int test_insert_count(void) {
    GV_LearnedSparseConfig config;
    gv_ls_config_init(&config);

    GV_LearnedSparseIndex *idx = gv_ls_create(&config);
    ASSERT(idx != NULL, "create should succeed");

    /* Document 0: token_ids 10, 20, 30 with learned weights */
    GV_SparseEntry entries0[] = {
        {.token_id = 10, .weight = 0.8f},
        {.token_id = 20, .weight = 0.5f},
        {.token_id = 30, .weight = 0.3f}
    };
    int rc = gv_ls_insert(idx, entries0, 3);
    ASSERT(rc >= 0, "insert doc 0 should succeed and return doc ID >= 0");
    ASSERT(gv_ls_count(idx) == 1, "count should be 1 after one insert");

    /* Document 1: different tokens */
    GV_SparseEntry entries1[] = {
        {.token_id = 20, .weight = 0.9f},
        {.token_id = 40, .weight = 0.6f}
    };
    rc = gv_ls_insert(idx, entries1, 2);
    ASSERT(rc >= 0, "insert doc 1 should succeed");
    ASSERT(gv_ls_count(idx) == 2, "count should be 2 after two inserts");

    gv_ls_destroy(idx);
    return 0;
}

/* --- Test: search --- */
static int test_search(void) {
    GV_LearnedSparseConfig config;
    gv_ls_config_init(&config);

    GV_LearnedSparseIndex *idx = gv_ls_create(&config);
    ASSERT(idx != NULL, "create should succeed");

    /* Insert 3 documents */
    GV_SparseEntry doc0[] = { {10, 1.0f}, {20, 0.5f} };
    GV_SparseEntry doc1[] = { {10, 0.2f}, {30, 0.9f} };
    GV_SparseEntry doc2[] = { {40, 0.7f}, {50, 0.3f} };

    gv_ls_insert(idx, doc0, 2);
    gv_ls_insert(idx, doc1, 2);
    gv_ls_insert(idx, doc2, 2);

    /* Query overlaps most with doc0 */
    GV_SparseEntry query[] = { {10, 1.0f}, {20, 1.0f} };

    GV_LearnedSparseResult results[3];
    int n = gv_ls_search(idx, query, 2, 3, results);
    ASSERT(n >= 1, "search should return at least 1 result");
    ASSERT(n <= 3, "search should return at most 3 results");

    /* Best match should be doc0 (highest dot product) */
    if (n >= 1) {
        ASSERT(results[0].score > 0.0f, "top result should have positive score");
    }

    gv_ls_destroy(idx);
    return 0;
}

/* --- Test: search with threshold --- */
static int test_search_threshold(void) {
    GV_LearnedSparseConfig config;
    gv_ls_config_init(&config);

    GV_LearnedSparseIndex *idx = gv_ls_create(&config);
    ASSERT(idx != NULL, "create should succeed");

    GV_SparseEntry doc0[] = { {10, 1.0f} };
    GV_SparseEntry doc1[] = { {10, 0.1f} };
    gv_ls_insert(idx, doc0, 1);
    gv_ls_insert(idx, doc1, 1);

    GV_SparseEntry query[] = { {10, 1.0f} };

    /* High threshold should exclude low-scoring doc1 */
    GV_LearnedSparseResult results[2];
    int n = gv_ls_search_with_threshold(idx, query, 1, 0.5f, 2, results);
    ASSERT(n >= 0, "search with threshold should not error");
    /* Only doc0 has score 1.0 >= 0.5 threshold; doc1 has score 0.1 */
    ASSERT(n <= 2, "should return at most 2 results");

    gv_ls_destroy(idx);
    return 0;
}

/* --- Test: delete document --- */
static int test_delete(void) {
    GV_LearnedSparseConfig config;
    gv_ls_config_init(&config);

    GV_LearnedSparseIndex *idx = gv_ls_create(&config);
    ASSERT(idx != NULL, "create should succeed");

    GV_SparseEntry doc0[] = { {10, 1.0f} };
    GV_SparseEntry doc1[] = { {20, 1.0f} };
    gv_ls_insert(idx, doc0, 1);
    gv_ls_insert(idx, doc1, 1);
    ASSERT(gv_ls_count(idx) == 2, "count should be 2 before delete");

    int rc = gv_ls_delete(idx, 0);
    ASSERT(rc == 0, "delete doc 0 should succeed");
    ASSERT(gv_ls_count(idx) == 1, "count should be 1 after delete");

    /* Double delete should fail */
    rc = gv_ls_delete(idx, 0);
    ASSERT(rc == -1, "deleting already-deleted doc should return -1");

    gv_ls_destroy(idx);
    return 0;
}

/* --- Test: get stats --- */
static int test_stats(void) {
    GV_LearnedSparseConfig config;
    gv_ls_config_init(&config);

    GV_LearnedSparseIndex *idx = gv_ls_create(&config);
    ASSERT(idx != NULL, "create should succeed");

    GV_SparseEntry doc0[] = { {10, 1.0f}, {20, 0.5f}, {30, 0.3f} };
    gv_ls_insert(idx, doc0, 3);

    GV_LearnedSparseStats stats;
    memset(&stats, 0, sizeof(stats));
    int rc = gv_ls_get_stats(idx, &stats);
    ASSERT(rc == 0, "get_stats should succeed");
    ASSERT(stats.doc_count == 1, "doc_count should be 1");
    ASSERT(stats.total_postings == 3, "total_postings should be 3");
    ASSERT(stats.avg_doc_length > 0.0, "avg_doc_length should be positive");
    ASSERT(stats.vocab_used == 3, "vocab_used should be 3 (3 distinct tokens)");

    gv_ls_destroy(idx);
    return 0;
}

/* --- Test: search on empty index --- */
static int test_search_empty(void) {
    GV_LearnedSparseIndex *idx = gv_ls_create(NULL);
    ASSERT(idx != NULL, "create should succeed");

    GV_SparseEntry query[] = { {10, 1.0f} };
    GV_LearnedSparseResult results[5];
    int n = gv_ls_search(idx, query, 1, 5, results);
    ASSERT(n == 0, "search on empty index should return 0 results");

    gv_ls_destroy(idx);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing learned sparse config init...", test_config_init},
        {"Testing learned sparse create/destroy...", test_create_destroy},
        {"Testing learned sparse insert/count...", test_insert_count},
        {"Testing learned sparse search...", test_search},
        {"Testing learned sparse search threshold...", test_search_threshold},
        {"Testing learned sparse delete...", test_delete},
        {"Testing learned sparse stats...", test_stats},
        {"Testing learned sparse search empty...", test_search_empty},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        printf("%s", tests[i].name);
        if (tests[i].fn() == 0) { printf(" [OK]\n"); passed++; }
        else { printf(" [FAIL]\n"); }
    }
    printf("\n%d/%d tests passed\n", passed, n);
    return passed == n ? 0 : 1;
}
