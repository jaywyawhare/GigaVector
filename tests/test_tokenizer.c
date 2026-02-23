/* tests/test_tokenizer.c — In-depth tests for the tokenizer module */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_tokenizer.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while (0)

/* ------------------------------------------------------------------ */
/* 1. Config defaults                                                  */
/* ------------------------------------------------------------------ */
static int test_config_defaults(void) {
    GV_TokenizerConfig cfg;
    gv_tokenizer_config_init(&cfg);
    ASSERT(cfg.type == GV_TOKENIZER_SIMPLE, "default type should be SIMPLE");
    ASSERT(cfg.lowercase == 1, "default lowercase should be 1");
    ASSERT(cfg.remove_stopwords == 0, "default remove_stopwords should be 0");
    ASSERT(cfg.min_token_length == 1, "default min_token_length should be 1");
    ASSERT(cfg.max_token_length == 256, "default max_token_length should be 256");
    return 0;
}

/* ------------------------------------------------------------------ */
/* 2. Whitespace tokenizer                                            */
/* ------------------------------------------------------------------ */
static int test_whitespace_tokenizer(void) {
    GV_TokenizerConfig cfg;
    gv_tokenizer_config_init(&cfg);
    cfg.type = GV_TOKENIZER_WHITESPACE;
    cfg.lowercase = 0;

    GV_Tokenizer *tok = gv_tokenizer_create(&cfg);
    ASSERT(tok != NULL, "create whitespace tokenizer");

    const char *text = "Hello  World\tTab\nNewline";
    GV_TokenList list = {0};
    int rc = gv_tokenizer_tokenize(tok, text, strlen(text), &list);
    ASSERT(rc == 0, "tokenize should succeed");
    ASSERT(list.count == 4, "should produce 4 tokens");
    ASSERT(strcmp(list.tokens[0].text, "Hello") == 0, "first token");
    ASSERT(strcmp(list.tokens[1].text, "World") == 0, "second token");
    ASSERT(strcmp(list.tokens[2].text, "Tab") == 0, "third token");
    ASSERT(strcmp(list.tokens[3].text, "Newline") == 0, "fourth token");

    gv_token_list_free(&list);
    gv_tokenizer_destroy(tok);
    return 0;
}

/* ------------------------------------------------------------------ */
/* 3. Simple tokenizer (lowercase + non-alphanumeric split)           */
/* ------------------------------------------------------------------ */
static int test_simple_tokenizer(void) {
    GV_TokenizerConfig cfg;
    gv_tokenizer_config_init(&cfg);
    cfg.type = GV_TOKENIZER_SIMPLE;

    GV_Tokenizer *tok = gv_tokenizer_create(&cfg);
    ASSERT(tok != NULL, "create simple tokenizer");

    const char *text = "Hello, World! It's a test.";
    GV_TokenList list = {0};
    int rc = gv_tokenizer_tokenize(tok, text, strlen(text), &list);
    ASSERT(rc == 0, "tokenize should succeed");
    ASSERT(list.count >= 5, "should produce multiple tokens");
    /* Simple tokenizer lowercases */
    ASSERT(strcmp(list.tokens[0].text, "hello") == 0, "first token lowered");

    gv_token_list_free(&list);
    gv_tokenizer_destroy(tok);
    return 0;
}

/* ------------------------------------------------------------------ */
/* 4. Standard tokenizer with stopword removal                        */
/* ------------------------------------------------------------------ */
static int test_standard_tokenizer_stopwords(void) {
    GV_TokenizerConfig cfg;
    gv_tokenizer_config_init(&cfg);
    cfg.type = GV_TOKENIZER_STANDARD;
    cfg.remove_stopwords = 1;

    GV_Tokenizer *tok = gv_tokenizer_create(&cfg);
    ASSERT(tok != NULL, "create standard tokenizer");

    const char *text = "the cat is on the mat";
    GV_TokenList list = {0};
    int rc = gv_tokenizer_tokenize(tok, text, strlen(text), &list);
    ASSERT(rc == 0, "tokenize should succeed");
    /* "the", "is", "on" are stopwords — should be removed */
    for (size_t i = 0; i < list.count; i++) {
        ASSERT(strcmp(list.tokens[i].text, "the") != 0, "stopword 'the' removed");
        ASSERT(strcmp(list.tokens[i].text, "is") != 0, "stopword 'is' removed");
        ASSERT(strcmp(list.tokens[i].text, "on") != 0, "stopword 'on' removed");
    }
    /* "cat" and "mat" should be present */
    int found_cat = 0, found_mat = 0;
    for (size_t i = 0; i < list.count; i++) {
        if (strcmp(list.tokens[i].text, "cat") == 0) found_cat = 1;
        if (strcmp(list.tokens[i].text, "mat") == 0) found_mat = 1;
    }
    ASSERT(found_cat, "cat should survive");
    ASSERT(found_mat, "mat should survive");

    gv_token_list_free(&list);
    gv_tokenizer_destroy(tok);
    return 0;
}

/* ------------------------------------------------------------------ */
/* 5. Token positions and offsets                                     */
/* ------------------------------------------------------------------ */
static int test_token_positions(void) {
    GV_TokenizerConfig cfg;
    gv_tokenizer_config_init(&cfg);
    cfg.type = GV_TOKENIZER_WHITESPACE;
    cfg.lowercase = 0;

    GV_Tokenizer *tok = gv_tokenizer_create(&cfg);
    const char *text = "alpha beta gamma";
    GV_TokenList list = {0};
    gv_tokenizer_tokenize(tok, text, strlen(text), &list);

    ASSERT(list.count == 3, "3 tokens");
    ASSERT(list.tokens[0].position == 0, "first position is 0");
    ASSERT(list.tokens[1].position == 1, "second position is 1");
    ASSERT(list.tokens[0].offset_start == 0, "first starts at 0");
    ASSERT(list.tokens[1].offset_start == 6, "beta starts at 6");
    ASSERT(list.tokens[2].offset_start == 11, "gamma starts at 11");

    gv_token_list_free(&list);
    gv_tokenizer_destroy(tok);
    return 0;
}

/* ------------------------------------------------------------------ */
/* 6. Min/max token length filtering                                  */
/* ------------------------------------------------------------------ */
static int test_token_length_filter(void) {
    GV_TokenizerConfig cfg;
    gv_tokenizer_config_init(&cfg);
    cfg.type = GV_TOKENIZER_WHITESPACE;
    cfg.min_token_length = 4;
    cfg.max_token_length = 5;

    GV_Tokenizer *tok = gv_tokenizer_create(&cfg);
    const char *text = "a bb ccc dddd eeeee ffffff";
    GV_TokenList list = {0};
    gv_tokenizer_tokenize(tok, text, strlen(text), &list);

    /* Only "dddd" (4) and "eeeee" (5) should survive */
    ASSERT(list.count == 2, "should have 2 tokens within [4,5] length");
    ASSERT(strcmp(list.tokens[0].text, "dddd") == 0, "first survivor");
    ASSERT(strcmp(list.tokens[1].text, "eeeee") == 0, "second survivor");

    gv_token_list_free(&list);
    gv_tokenizer_destroy(tok);
    return 0;
}

/* ------------------------------------------------------------------ */
/* 7. Simple convenience function                                     */
/* ------------------------------------------------------------------ */
static int test_tokenize_simple(void) {
    GV_TokenList list = {0};
    int rc = gv_tokenize_simple("hello world foo", &list);
    ASSERT(rc == 0, "tokenize_simple should succeed");
    ASSERT(list.count == 3, "3 tokens");
    gv_token_list_free(&list);
    return 0;
}

/* ------------------------------------------------------------------ */
/* 8. Unique tokens                                                   */
/* ------------------------------------------------------------------ */
static int test_unique_tokens(void) {
    GV_TokenizerConfig cfg;
    gv_tokenizer_config_init(&cfg);
    cfg.type = GV_TOKENIZER_WHITESPACE;
    cfg.lowercase = 1;

    GV_Tokenizer *tok = gv_tokenizer_create(&cfg);
    const char *text = "dog cat dog bird cat dog";
    GV_TokenList list = {0};
    gv_tokenizer_tokenize(tok, text, strlen(text), &list);
    ASSERT(list.count == 6, "6 raw tokens");

    char **unique = NULL;
    size_t ucount = 0;
    int rc = gv_token_list_unique(&list, &unique, &ucount);
    ASSERT(rc == 0, "unique should succeed");
    ASSERT(ucount == 3, "3 unique tokens");
    gv_unique_tokens_free(unique, ucount);

    gv_token_list_free(&list);
    gv_tokenizer_destroy(tok);
    return 0;
}

/* ------------------------------------------------------------------ */
/* 9. Stopword detection                                              */
/* ------------------------------------------------------------------ */
static int test_is_stopword(void) {
    ASSERT(gv_is_stopword("the") == 1, "'the' is a stopword");
    ASSERT(gv_is_stopword("and") == 1, "'and' is a stopword");
    ASSERT(gv_is_stopword("elephant") == 0, "'elephant' is not a stopword");
    ASSERT(gv_is_stopword("") == 0, "empty is not a stopword");
    return 0;
}

/* ------------------------------------------------------------------ */
/* 10. Empty / edge cases                                             */
/* ------------------------------------------------------------------ */
static int test_empty_input(void) {
    GV_TokenizerConfig cfg;
    gv_tokenizer_config_init(&cfg);
    GV_Tokenizer *tok = gv_tokenizer_create(&cfg);

    GV_TokenList list = {0};
    int rc = gv_tokenizer_tokenize(tok, "", 0, &list);
    ASSERT(rc == 0, "empty input should succeed");
    ASSERT(list.count == 0, "no tokens from empty");
    gv_token_list_free(&list);

    /* Only whitespace */
    rc = gv_tokenizer_tokenize(tok, "   \t\n  ", 7, &list);
    ASSERT(rc == 0, "whitespace-only should succeed");
    ASSERT(list.count == 0, "no tokens from whitespace-only");
    gv_token_list_free(&list);

    gv_tokenizer_destroy(tok);
    return 0;
}

int main(void) {
    int failures = 0;
    struct { const char *name; int (*fn)(void); } tests[] = {
        {"config_defaults", test_config_defaults},
        {"whitespace_tokenizer", test_whitespace_tokenizer},
        {"simple_tokenizer", test_simple_tokenizer},
        {"standard_tokenizer_stopwords", test_standard_tokenizer_stopwords},
        {"token_positions", test_token_positions},
        {"token_length_filter", test_token_length_filter},
        {"tokenize_simple", test_tokenize_simple},
        {"unique_tokens", test_unique_tokens},
        {"is_stopword", test_is_stopword},
        {"empty_input", test_empty_input},
    };
    size_t n = sizeof(tests) / sizeof(tests[0]);
    for (size_t i = 0; i < n; i++) {
        printf("  %s ... ", tests[i].name);
        if (tests[i].fn() == 0) { printf("OK\n"); }
        else { printf("FAILED\n"); failures++; }
    }
    if (failures) { fprintf(stderr, "%d test(s) failed\n", failures); return 1; }
    printf("All tokenizer tests passed (%zu tests)\n", n);
    return 0;
}
