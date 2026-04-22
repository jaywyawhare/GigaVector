#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "multimodal/learned_sparse.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_config_init(void) {
    GV_LearnedSparseConfig config;
    memset(&config, 0xFF, sizeof(config));
    ls_config_init(&config);

    ASSERT(config.vocab_size == 30522, "default vocab_size should be 30522");
    ASSERT(config.max_nonzeros == 256, "default max_nonzeros should be 256");
    ASSERT(config.use_wand == 1, "default use_wand should be 1");
    ASSERT(config.wand_block_size == 128, "default wand_block_size should be 128");

    return 0;
}

static int test_create_destroy(void) {
    GV_LearnedSparseConfig config;
    ls_config_init(&config);

    GV_LearnedSparseIndex *idx = ls_create(&config);
    ASSERT(idx != NULL, "ls_create should return non-NULL");
    ASSERT(ls_count(idx) == 0, "new index should have count 0");

    ls_destroy(idx);
    ls_destroy(NULL);

    GV_LearnedSparseIndex *idx2 = ls_create(NULL);
    ASSERT(idx2 != NULL, "ls_create(NULL) should use defaults");
    ls_destroy(idx2);

    return 0;
}

static int test_insert_count(void) {
    GV_LearnedSparseConfig config;
    ls_config_init(&config);

    GV_LearnedSparseIndex *idx = ls_create(&config);
    ASSERT(idx != NULL, "create should succeed");

    GV_LSSparseEntry entries0[] = {
        {.token_id = 10, .weight = 0.8f},
        {.token_id = 20, .weight = 0.5f},
        {.token_id = 30, .weight = 0.3f}
    };
    int rc = ls_insert(idx, entries0, 3);
    ASSERT(rc >= 0, "insert doc 0 should succeed and return doc ID >= 0");
    ASSERT(ls_count(idx) == 1, "count should be 1 after one insert");

    GV_LSSparseEntry entries1[] = {
        {.token_id = 20, .weight = 0.9f},
        {.token_id = 40, .weight = 0.6f}
    };
    rc = ls_insert(idx, entries1, 2);
    ASSERT(rc >= 0, "insert doc 1 should succeed");
    ASSERT(ls_count(idx) == 2, "count should be 2 after two inserts");

    ls_destroy(idx);
    return 0;
}

static int test_search(void) {
    GV_LearnedSparseConfig config;
    ls_config_init(&config);

    GV_LearnedSparseIndex *idx = ls_create(&config);
    ASSERT(idx != NULL, "create should succeed");

    GV_LSSparseEntry doc0[] = { {10, 1.0f}, {20, 0.5f} };
    GV_LSSparseEntry doc1[] = { {10, 0.2f}, {30, 0.9f} };
    GV_LSSparseEntry doc2[] = { {40, 0.7f}, {50, 0.3f} };

    ls_insert(idx, doc0, 2);
    ls_insert(idx, doc1, 2);
    ls_insert(idx, doc2, 2);

    GV_LSSparseEntry query[] = { {10, 1.0f}, {20, 1.0f} };

    GV_LearnedSparseResult results[3];
    int n = ls_search(idx, query, 2, 3, results);
    ASSERT(n >= 1, "search should return at least 1 result");
    ASSERT(n <= 3, "search should return at most 3 results");

    if (n >= 1) {
        ASSERT(results[0].score > 0.0f, "top result should have positive score");
    }

    ls_destroy(idx);
    return 0;
}

static int test_search_threshold(void) {
    GV_LearnedSparseConfig config;
    ls_config_init(&config);

    GV_LearnedSparseIndex *idx = ls_create(&config);
    ASSERT(idx != NULL, "create should succeed");

    GV_LSSparseEntry doc0[] = { {10, 1.0f} };
    GV_LSSparseEntry doc1[] = { {10, 0.1f} };
    ls_insert(idx, doc0, 1);
    ls_insert(idx, doc1, 1);

    GV_LSSparseEntry query[] = { {10, 1.0f} };

    GV_LearnedSparseResult results[2];
    int n = ls_search_with_threshold(idx, query, 1, 0.5f, 2, results);
    ASSERT(n >= 0, "search with threshold should not error");
    ASSERT(n <= 2, "should return at most 2 results");

    ls_destroy(idx);
    return 0;
}

static int test_delete(void) {
    GV_LearnedSparseConfig config;
    ls_config_init(&config);

    GV_LearnedSparseIndex *idx = ls_create(&config);
    ASSERT(idx != NULL, "create should succeed");

    GV_LSSparseEntry doc0[] = { {10, 1.0f} };
    GV_LSSparseEntry doc1[] = { {20, 1.0f} };
    ls_insert(idx, doc0, 1);
    ls_insert(idx, doc1, 1);
    ASSERT(ls_count(idx) == 2, "count should be 2 before delete");

    int rc = ls_delete(idx, 0);
    ASSERT(rc == 0, "delete doc 0 should succeed");
    ASSERT(ls_count(idx) == 1, "count should be 1 after delete");

    rc = ls_delete(idx, 0);
    ASSERT(rc == -1, "deleting already-deleted doc should return -1");

    ls_destroy(idx);
    return 0;
}

static int test_stats(void) {
    GV_LearnedSparseConfig config;
    ls_config_init(&config);

    GV_LearnedSparseIndex *idx = ls_create(&config);
    ASSERT(idx != NULL, "create should succeed");

    GV_LSSparseEntry doc0[] = { {10, 1.0f}, {20, 0.5f}, {30, 0.3f} };
    ls_insert(idx, doc0, 3);

    GV_LearnedSparseStats stats;
    memset(&stats, 0, sizeof(stats));
    int rc = ls_get_stats(idx, &stats);
    ASSERT(rc == 0, "get_stats should succeed");
    ASSERT(stats.doc_count == 1, "doc_count should be 1");
    ASSERT(stats.total_postings == 3, "total_postings should be 3");
    ASSERT(stats.avg_doc_length > 0.0, "avg_doc_length should be positive");
    ASSERT(stats.vocab_used == 3, "vocab_used should be 3 (3 distinct tokens)");

    ls_destroy(idx);
    return 0;
}

static int test_search_empty(void) {
    GV_LearnedSparseIndex *idx = ls_create(NULL);
    ASSERT(idx != NULL, "create should succeed");

    GV_LSSparseEntry query[] = { {10, 1.0f} };
    GV_LearnedSparseResult results[5];
    int n = ls_search(idx, query, 1, 5, results);
    ASSERT(n == 0, "search on empty index should return 0 results");

    ls_destroy(idx);
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
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
