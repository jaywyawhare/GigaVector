#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "multimodal/bm25.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_config_defaults(void) {
    GV_BM25Config config;
    bm25_config_init(&config);
    ASSERT(config.k1 > 1.19 && config.k1 < 1.21, "k1 should default to ~1.2");
    ASSERT(config.b > 0.74 && config.b < 0.76, "b should default to ~0.75");
    return 0;
}

static int test_create_destroy(void) {
    GV_BM25Index *idx = bm25_create(NULL);
    ASSERT(idx != NULL, "bm25_create(NULL) should succeed");
    bm25_destroy(idx);
    bm25_destroy(NULL);
    return 0;
}

static int test_add_and_search(void) {
    GV_BM25Index *idx = bm25_create(NULL);
    ASSERT(idx != NULL, "create should succeed");

    int rc = bm25_add_document(idx, 0, "the quick brown fox jumps over the lazy dog");
    ASSERT(rc == 0, "add doc 0 should succeed");
    rc = bm25_add_document(idx, 1, "a fast brown fox leaps over a sleepy hound");
    ASSERT(rc == 0, "add doc 1 should succeed");
    rc = bm25_add_document(idx, 2, "the cat sat on the mat");
    ASSERT(rc == 0, "add doc 2 should succeed");

    GV_BM25Result results[3];
    int n = bm25_search(idx, "brown fox", 3, results);
    ASSERT(n >= 1, "search for 'brown fox' should find at least 1 result");
    /* The top result should be doc 0 or 1 (both contain "brown fox") */
    ASSERT(results[0].doc_id == 0 || results[0].doc_id == 1,
           "top result should be doc 0 or 1");
    ASSERT(results[0].score > 0.0, "top result score should be positive");

    bm25_destroy(idx);
    return 0;
}

static int test_remove_document(void) {
    GV_BM25Index *idx = bm25_create(NULL);
    ASSERT(idx != NULL, "create should succeed");

    bm25_add_document(idx, 0, "alpha beta gamma");
    bm25_add_document(idx, 1, "delta epsilon zeta");

    ASSERT(bm25_has_document(idx, 0) == 1, "doc 0 should exist");
    int rc = bm25_remove_document(idx, 0);
    ASSERT(rc == 0, "remove doc 0 should succeed");
    ASSERT(bm25_has_document(idx, 0) == 0, "doc 0 should be gone");

    rc = bm25_remove_document(idx, 99);
    ASSERT(rc == -1, "removing non-existent doc should return -1");

    bm25_destroy(idx);
    return 0;
}

static int test_update_document(void) {
    GV_BM25Index *idx = bm25_create(NULL);
    ASSERT(idx != NULL, "create should succeed");

    bm25_add_document(idx, 0, "hello world");
    int rc = bm25_update_document(idx, 0, "goodbye universe");
    ASSERT(rc == 0, "update should succeed");

    GV_BM25Result results[1];
    int n = bm25_search(idx, "goodbye", 1, results);
    ASSERT(n >= 1, "search for 'goodbye' should find updated doc");
    ASSERT(results[0].doc_id == 0, "updated doc should be doc 0");

    bm25_destroy(idx);
    return 0;
}

static int test_stats(void) {
    GV_BM25Index *idx = bm25_create(NULL);
    ASSERT(idx != NULL, "create should succeed");

    bm25_add_document(idx, 0, "one two three");
    bm25_add_document(idx, 1, "four five six");

    GV_BM25Stats stats;
    int rc = bm25_get_stats(idx, &stats);
    ASSERT(rc == 0, "get_stats should succeed");
    ASSERT(stats.total_documents == 2, "should have 2 documents");
    ASSERT(stats.total_terms > 0, "should have some terms");
    ASSERT(stats.avg_document_length > 0.0, "avg doc length should be positive");

    bm25_destroy(idx);
    return 0;
}

static int test_doc_freq_and_has_document(void) {
    GV_BM25Index *idx = bm25_create(NULL);
    ASSERT(idx != NULL, "create should succeed");

    bm25_add_document(idx, 0, "apple banana cherry");
    bm25_add_document(idx, 1, "banana cherry date");
    bm25_add_document(idx, 2, "cherry date elderberry");

    size_t freq = bm25_get_doc_freq(idx, "cherry");
    ASSERT(freq == 3, "cherry should appear in 3 docs");

    freq = bm25_get_doc_freq(idx, "apple");
    ASSERT(freq == 1, "apple should appear in 1 doc");

    freq = bm25_get_doc_freq(idx, "zzzzz");
    ASSERT(freq == 0, "non-existent term should have freq 0");

    ASSERT(bm25_has_document(idx, 2) == 1, "doc 2 should exist");
    ASSERT(bm25_has_document(idx, 99) == 0, "doc 99 should not exist");

    bm25_destroy(idx);
    return 0;
}

static int test_score_document(void) {
    GV_BM25Index *idx = bm25_create(NULL);
    ASSERT(idx != NULL, "create should succeed");

    bm25_add_document(idx, 0, "machine learning deep neural network");
    bm25_add_document(idx, 1, "cooking recipes for dinner");

    double score0 = 0.0, score1 = 0.0;
    int rc = bm25_score_document(idx, 0, "machine learning", &score0);
    ASSERT(rc == 0, "score_document for doc 0 should succeed");
    rc = bm25_score_document(idx, 1, "machine learning", &score1);
    ASSERT(rc == 0, "score_document for doc 1 should succeed");

    ASSERT(score0 > score1, "doc 0 should score higher for 'machine learning'");

    bm25_destroy(idx);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing BM25 config defaults...",            test_config_defaults},
        {"Testing BM25 create/destroy...",             test_create_destroy},
        {"Testing BM25 add and search...",             test_add_and_search},
        {"Testing BM25 remove document...",            test_remove_document},
        {"Testing BM25 update document...",            test_update_document},
        {"Testing BM25 stats...",                      test_stats},
        {"Testing BM25 doc freq and has_document...",  test_doc_freq_and_has_document},
        {"Testing BM25 score_document...",             test_score_document},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
