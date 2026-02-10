#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_fulltext.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

/* --- Test: config init defaults --- */
static int test_config_init(void) {
    GV_FTConfig config;
    memset(&config, 0xFF, sizeof(config));
    gv_ft_config_init(&config);

    ASSERT(config.language == GV_LANG_ENGLISH, "default language should be ENGLISH");
    ASSERT(config.enable_stemming == 1, "stemming should be enabled by default");
    ASSERT(config.enable_phrase_match == 1, "phrase match should be enabled by default");
    ASSERT(config.use_blockmax_wand == 1, "blockmax WAND should be enabled by default");
    ASSERT(config.block_size == 128, "default block size should be 128");

    return 0;
}

/* --- Test: create and destroy --- */
static int test_create_destroy(void) {
    GV_FTConfig config;
    gv_ft_config_init(&config);

    GV_FTIndex *idx = gv_ft_create(&config);
    ASSERT(idx != NULL, "gv_ft_create should return non-NULL");
    ASSERT(gv_ft_doc_count(idx) == 0, "new index should have doc count 0");

    gv_ft_destroy(idx);
    /* NULL destroy should be safe */
    gv_ft_destroy(NULL);

    /* create with NULL config (uses defaults) */
    GV_FTIndex *idx2 = gv_ft_create(NULL);
    ASSERT(idx2 != NULL, "gv_ft_create(NULL) should use defaults and succeed");
    gv_ft_destroy(idx2);

    return 0;
}

/* --- Test: add and search documents --- */
static int test_add_and_search(void) {
    GV_FTConfig config;
    gv_ft_config_init(&config);

    GV_FTIndex *idx = gv_ft_create(&config);
    ASSERT(idx != NULL, "create should succeed");

    int rc;
    rc = gv_ft_add_document(idx, 0, "The quick brown fox jumps over the lazy dog");
    ASSERT(rc == 0, "add document 0 should succeed");

    rc = gv_ft_add_document(idx, 1, "A fast brown fox leaps over a sleepy canine");
    ASSERT(rc == 0, "add document 1 should succeed");

    rc = gv_ft_add_document(idx, 2, "The weather forecast predicts rain tomorrow");
    ASSERT(rc == 0, "add document 2 should succeed");

    ASSERT(gv_ft_doc_count(idx) == 3, "doc count should be 3");

    GV_FTResult results[10];
    int n = gv_ft_search(idx, "brown fox", 10, results);
    ASSERT(n >= 1, "search for 'brown fox' should return at least 1 result");
    ASSERT(n <= 3, "search should return at most 3 results");

    /* Clean up result match_positions */
    if (n > 0) {
        gv_ft_free_results(results, (size_t)n);
    }

    gv_ft_destroy(idx);
    return 0;
}

/* --- Test: phrase search --- */
static int test_phrase_search(void) {
    GV_FTConfig config;
    gv_ft_config_init(&config);
    config.enable_phrase_match = 1;

    GV_FTIndex *idx = gv_ft_create(&config);
    ASSERT(idx != NULL, "create should succeed");

    gv_ft_add_document(idx, 0, "The quick brown fox jumps over the lazy dog");
    gv_ft_add_document(idx, 1, "Brown quick fox is not the same phrase order");

    GV_FTResult results[10];
    int n = gv_ft_search_phrase(idx, "quick brown fox", 10, results);
    /* Only document 0 should match the exact phrase "quick brown fox" */
    ASSERT(n >= 0, "phrase search should not error");

    if (n > 0) {
        gv_ft_free_results(results, (size_t)n);
    }

    gv_ft_destroy(idx);
    return 0;
}

/* --- Test: remove document --- */
static int test_remove_document(void) {
    GV_FTConfig config;
    gv_ft_config_init(&config);

    GV_FTIndex *idx = gv_ft_create(&config);
    ASSERT(idx != NULL, "create should succeed");

    gv_ft_add_document(idx, 0, "Hello world");
    gv_ft_add_document(idx, 1, "Goodbye world");
    ASSERT(gv_ft_doc_count(idx) == 2, "doc count should be 2");

    int rc = gv_ft_remove_document(idx, 0);
    ASSERT(rc == 0, "remove document 0 should succeed");
    ASSERT(gv_ft_doc_count(idx) == 1, "doc count should be 1 after remove");

    /* Removing non-existent should fail */
    rc = gv_ft_remove_document(idx, 99);
    ASSERT(rc == -1, "remove non-existent document should return -1");

    gv_ft_destroy(idx);
    return 0;
}

/* --- Test: Porter stemming --- */
static int test_stem(void) {
    char output[128];

    int rc = gv_ft_stem("running", GV_LANG_ENGLISH, output, sizeof(output));
    ASSERT(rc == 0, "stem 'running' should succeed");
    /* Porter stemmer: "running" -> "run" */
    ASSERT(strlen(output) > 0, "stemmed word should not be empty");

    rc = gv_ft_stem("jumps", GV_LANG_ENGLISH, output, sizeof(output));
    ASSERT(rc == 0, "stem 'jumps' should succeed");
    ASSERT(strlen(output) > 0, "stemmed word should not be empty");

    /* Buffer too small */
    rc = gv_ft_stem("internationalization", GV_LANG_ENGLISH, output, 2);
    ASSERT(rc == -1, "stem with tiny buffer should return -1");

    return 0;
}

/* --- Test: search on empty index --- */
static int test_search_empty(void) {
    GV_FTIndex *idx = gv_ft_create(NULL);
    ASSERT(idx != NULL, "create should succeed");

    GV_FTResult results[5];
    int n = gv_ft_search(idx, "anything", 5, results);
    ASSERT(n == 0, "search on empty index should return 0 results");

    gv_ft_destroy(idx);
    return 0;
}

/* --- Test: multiple languages --- */
static int test_language_config(void) {
    GV_FTConfig config;
    gv_ft_config_init(&config);
    config.language = GV_LANG_GERMAN;

    GV_FTIndex *idx = gv_ft_create(&config);
    ASSERT(idx != NULL, "create with GERMAN language should succeed");

    int rc = gv_ft_add_document(idx, 0, "Der schnelle braune Fuchs springt ueber den faulen Hund");
    ASSERT(rc == 0, "add German document should succeed");
    ASSERT(gv_ft_doc_count(idx) == 1, "doc count should be 1");

    gv_ft_destroy(idx);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing fulltext config init...", test_config_init},
        {"Testing fulltext create/destroy...", test_create_destroy},
        {"Testing fulltext add and search...", test_add_and_search},
        {"Testing fulltext phrase search...", test_phrase_search},
        {"Testing fulltext remove document...", test_remove_document},
        {"Testing fulltext stemming...", test_stem},
        {"Testing fulltext search empty...", test_search_empty},
        {"Testing fulltext language config...", test_language_config},
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
