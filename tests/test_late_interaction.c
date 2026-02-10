#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_late_interaction.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

/* Helper: token dimension used throughout tests */
#define TOKEN_DIM 4

/* --- Test: config init defaults --- */
static int test_config_init(void) {
    GV_LateInteractionConfig config;
    memset(&config, 0xFF, sizeof(config));  /* dirty memory */
    gv_late_interaction_config_init(&config);

    ASSERT(config.token_dimension > 0, "token_dimension should be set to a positive default");
    ASSERT(config.max_doc_tokens > 0, "max_doc_tokens should be set to a positive default");
    ASSERT(config.max_query_tokens > 0, "max_query_tokens should be set to a positive default");
    ASSERT(config.candidate_pool > 0, "candidate_pool should be set to a positive default");

    return 0;
}

/* --- Test: create and destroy --- */
static int test_create_destroy(void) {
    GV_LateInteractionConfig config;
    gv_late_interaction_config_init(&config);
    config.token_dimension = TOKEN_DIM;

    GV_LateInteractionIndex *idx = gv_late_interaction_create(&config);
    ASSERT(idx != NULL, "gv_late_interaction_create should return non-NULL");
    ASSERT(gv_late_interaction_count(idx) == 0, "new index should have count 0");

    gv_late_interaction_destroy(idx);
    /* destroying NULL should be safe */
    gv_late_interaction_destroy(NULL);
    return 0;
}

/* --- Test: add documents and count --- */
static int test_add_doc_count(void) {
    GV_LateInteractionConfig config;
    gv_late_interaction_config_init(&config);
    config.token_dimension = TOKEN_DIM;

    GV_LateInteractionIndex *idx = gv_late_interaction_create(&config);
    ASSERT(idx != NULL, "create should succeed");

    /* Document 0: 3 tokens, each of dimension TOKEN_DIM */
    float doc0_tokens[] = {
        1.0f, 0.0f, 0.0f, 0.0f,  /* token 0 */
        0.0f, 1.0f, 0.0f, 0.0f,  /* token 1 */
        0.0f, 0.0f, 1.0f, 0.0f   /* token 2 */
    };
    int rc = gv_late_interaction_add_doc(idx, doc0_tokens, 3);
    ASSERT(rc == 0, "add_doc for document 0 should succeed");
    ASSERT(gv_late_interaction_count(idx) == 1, "count should be 1 after one add");

    /* Document 1: 2 tokens */
    float doc1_tokens[] = {
        0.5f, 0.5f, 0.0f, 0.0f,
        0.0f, 0.5f, 0.5f, 0.0f
    };
    rc = gv_late_interaction_add_doc(idx, doc1_tokens, 2);
    ASSERT(rc == 0, "add_doc for document 1 should succeed");
    ASSERT(gv_late_interaction_count(idx) == 2, "count should be 2 after two adds");

    gv_late_interaction_destroy(idx);
    return 0;
}

/* --- Test: MaxSim search --- */
static int test_search(void) {
    GV_LateInteractionConfig config;
    gv_late_interaction_config_init(&config);
    config.token_dimension = TOKEN_DIM;

    GV_LateInteractionIndex *idx = gv_late_interaction_create(&config);
    ASSERT(idx != NULL, "create should succeed");

    /* Add two documents */
    float doc0[] = { 1.0f, 0.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f, 0.0f };
    float doc1[] = { 0.0f, 0.0f, 1.0f, 0.0f,  0.0f, 0.0f, 0.0f, 1.0f };
    gv_late_interaction_add_doc(idx, doc0, 2);
    gv_late_interaction_add_doc(idx, doc1, 2);

    /* Query with 2 tokens that should match doc0 better */
    float query[] = { 0.9f, 0.1f, 0.0f, 0.0f,  0.1f, 0.9f, 0.0f, 0.0f };

    GV_LateInteractionResult results[2];
    int n = gv_late_interaction_search(idx, query, 2, 2, results);
    ASSERT(n >= 1, "search should return at least 1 result");
    ASSERT(n <= 2, "search should return at most 2 results");

    /* doc0 should rank higher than doc1 for this query */
    if (n == 2) {
        ASSERT(results[0].score >= results[1].score,
               "results should be sorted by score descending");
    }

    gv_late_interaction_destroy(idx);
    return 0;
}

/* --- Test: delete document --- */
static int test_delete(void) {
    GV_LateInteractionConfig config;
    gv_late_interaction_config_init(&config);
    config.token_dimension = TOKEN_DIM;

    GV_LateInteractionIndex *idx = gv_late_interaction_create(&config);
    ASSERT(idx != NULL, "create should succeed");

    float doc0[] = { 1.0f, 0.0f, 0.0f, 0.0f };
    float doc1[] = { 0.0f, 1.0f, 0.0f, 0.0f };
    gv_late_interaction_add_doc(idx, doc0, 1);
    gv_late_interaction_add_doc(idx, doc1, 1);
    ASSERT(gv_late_interaction_count(idx) == 2, "count should be 2 before delete");

    int rc = gv_late_interaction_delete(idx, 0);
    ASSERT(rc == 0, "delete doc 0 should succeed");
    ASSERT(gv_late_interaction_count(idx) == 1, "count should be 1 after delete");

    gv_late_interaction_destroy(idx);
    return 0;
}

/* --- Test: get stats --- */
static int test_stats(void) {
    GV_LateInteractionConfig config;
    gv_late_interaction_config_init(&config);
    config.token_dimension = TOKEN_DIM;

    GV_LateInteractionIndex *idx = gv_late_interaction_create(&config);
    ASSERT(idx != NULL, "create should succeed");

    float doc0[] = { 1.0f, 0.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f, 0.0f };
    gv_late_interaction_add_doc(idx, doc0, 2);

    GV_LateInteractionStats stats;
    memset(&stats, 0, sizeof(stats));
    int rc = gv_late_interaction_get_stats(idx, &stats);
    ASSERT(rc == 0, "get_stats should succeed");
    ASSERT(stats.total_documents == 1, "stats should report 1 document");
    ASSERT(stats.total_tokens_stored == 2, "stats should report 2 tokens stored");

    gv_late_interaction_destroy(idx);
    return 0;
}

/* --- Test: search on empty index --- */
static int test_search_empty(void) {
    GV_LateInteractionConfig config;
    gv_late_interaction_config_init(&config);
    config.token_dimension = TOKEN_DIM;

    GV_LateInteractionIndex *idx = gv_late_interaction_create(&config);
    ASSERT(idx != NULL, "create should succeed");

    float query[] = { 1.0f, 0.0f, 0.0f, 0.0f };
    GV_LateInteractionResult results[5];
    int n = gv_late_interaction_search(idx, query, 1, 5, results);
    ASSERT(n == 0, "search on empty index should return 0 results");

    gv_late_interaction_destroy(idx);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing late interaction config init...", test_config_init},
        {"Testing late interaction create/destroy...", test_create_destroy},
        {"Testing late interaction add doc/count...", test_add_doc_count},
        {"Testing late interaction search...", test_search},
        {"Testing late interaction delete...", test_delete},
        {"Testing late interaction stats...", test_stats},
        {"Testing late interaction search empty...", test_search_empty},
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
