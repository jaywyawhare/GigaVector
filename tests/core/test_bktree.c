#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/bktree.h"
#include "storage/database.h"

#define ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
            return -1; \
        } \
    } while (0)

/* ------------------------------------------------------------------ */
/* Helper: check whether a string appears in the results array         */
/* ------------------------------------------------------------------ */
static int result_contains(const char **results, size_t n, const char *word) {
    for (size_t i = 0; i < n; i++) {
        if (results[i] && strcmp(results[i], word) == 0) return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Test: exact match (max_dist = 0)                                    */
/* ------------------------------------------------------------------ */
static int test_exact_match(void) {
    const char *words[] = {
        "apple", "apply", "apricot", "banana", "bandana",
        "mango", "melon", "grape", "grapefruit", "orange",
        "cherry", "peach", "plum", "pear", "lime",
        "lemon", "kiwi", "fig", "date", "coconut"
    };
    size_t nwords = sizeof(words) / sizeof(words[0]);

    GV_BKTree *tree = bktree_create();
    ASSERT(tree != NULL, "bktree_create");

    for (size_t i = 0; i < nwords; i++)
        bktree_insert(tree, words[i]);

    ASSERT(tree->count == nwords, "all words inserted");

    const char *results[64];
    size_t nr = 0;

    bktree_search(tree, "apple", 0, results, &nr, 64);
    ASSERT(nr == 1,                          "exact: one result");
    ASSERT(result_contains(results, nr, "apple"), "exact: 'apple' found");
    ASSERT(!result_contains(results, nr, "apply"), "exact: 'apply' not found");

    bktree_search(tree, "mango", 0, results, &nr, 64);
    ASSERT(nr == 1, "exact: mango");

    bktree_search(tree, "nothere", 0, results, &nr, 64);
    ASSERT(nr == 0, "exact: no match for unknown word");

    bktree_destroy(tree);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Test: fuzzy search (max_dist = 1)                                   */
/* ------------------------------------------------------------------ */
static int test_fuzzy_dist1(void) {
    const char *words[] = {
        "apple", "appl", "aple", "appla", "apricot",
        "banana", "apply", "orange", "mango", "grape",
        "cherry", "peach", "plum", "pear", "lime",
        "lemon", "kiwi", "fig", "date", "coconut"
    };
    size_t nwords = sizeof(words) / sizeof(words[0]);

    GV_BKTree *tree = bktree_create();
    ASSERT(tree != NULL, "bktree_create");

    for (size_t i = 0; i < nwords; i++)
        bktree_insert(tree, words[i]);

    const char *results[64];
    size_t nr = 0;

    bktree_search(tree, "apple", 1, results, &nr, 64);
    /* Within distance 1 of "apple": apple(0), appl(1), aple(1), appla(1), apply(1) */
    ASSERT(result_contains(results, nr, "apple"),  "dist1: apple");
    ASSERT(result_contains(results, nr, "appl"),   "dist1: appl");
    ASSERT(result_contains(results, nr, "aple"),   "dist1: aple");
    ASSERT(result_contains(results, nr, "appla"),  "dist1: appla");
    ASSERT(result_contains(results, nr, "apply"),  "dist1: apply");
    /* banana is far away */
    ASSERT(!result_contains(results, nr, "banana"), "dist1: banana excluded");
    ASSERT(!result_contains(results, nr, "orange"), "dist1: orange excluded");

    bktree_destroy(tree);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Test: destroy frees cleanly (run under valgrind / ASAN for leaks)   */
/* ------------------------------------------------------------------ */
static int test_destroy(void) {
    GV_BKTree *tree = bktree_create();
    ASSERT(tree != NULL, "create");

    const char *words[] = { "hello", "world", "foo", "bar", "baz" };
    for (size_t i = 0; i < 5; i++) bktree_insert(tree, words[i]);

    bktree_destroy(tree);   /* should not crash */
    return 0;
}

/* ------------------------------------------------------------------ */
/* Test: gv_db_search_fuzzy with metadata field "tag"                  */
/* ------------------------------------------------------------------ */
static int test_db_search_fuzzy(void) {
    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open");

    /* Insert 10 vectors with metadata tag values */
    const char *tags[] = {
        "apple", "appl", "aple", "banana",
        "banana", "mango", "grape", "cherry",
        "apple", "banana"
    };
    float vec[4] = {0.1f, 0.2f, 0.3f, 0.4f};

    for (int i = 0; i < 10; i++) {
        vec[0] = (float)i * 0.1f;
        int rc = db_add_vector_with_metadata(db, vec, 4, "tag", tags[i]);
        ASSERT(rc == 0, "db_add_vector_with_metadata");
    }

    /* Search fuzzy tag="apple" max_dist=1 — should match apple, appl, aple */
    GV_SearchResult results[10];
    memset(results, 0, sizeof(results));

    float query[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int got = gv_db_search_fuzzy(db, query, 10,
                                  "tag", "apple", 1,
                                  results);
    ASSERT(got > 0, "gv_db_search_fuzzy returned results");

    /* Verify that results include vectors with apple, appl, aple tags.
     * Collect the tag values from the result vectors. */
    int saw_apple = 0, saw_appl = 0, saw_aple = 0;
    for (int i = 0; i < got; i++) {
        if (!results[i].vector) continue;
        const GV_Metadata *m = results[i].vector->metadata;
        while (m) {
            if (strcmp(m->key, "tag") == 0) {
                if (strcmp(m->value, "apple") == 0) saw_apple = 1;
                if (strcmp(m->value, "appl")  == 0) saw_appl  = 1;
                if (strcmp(m->value, "aple")  == 0) saw_aple  = 1;
            }
            m = m->next;
        }
    }

    ASSERT(saw_apple, "fuzzy: 'apple' in results");
    ASSERT(saw_appl,  "fuzzy: 'appl'  in results");
    ASSERT(saw_aple,  "fuzzy: 'aple'  in results");

    gv_search_results_free(results, (size_t)got);
    db_close(db);
    return 0;
}

/* ------------------------------------------------------------------ */
/* main                                                                 */
/* ------------------------------------------------------------------ */
int main(void) {
    struct { const char *name; int (*fn)(void); } tests[] = {
        {"test_exact_match",       test_exact_match},
        {"test_fuzzy_dist1",       test_fuzzy_dist1},
        {"test_destroy",           test_destroy},
        {"test_db_search_fuzzy",   test_db_search_fuzzy},
    };

    int total  = (int)(sizeof(tests) / sizeof(tests[0]));
    int passed = 0;
    for (int i = 0; i < total; i++) {
        int r = tests[i].fn();
        if (r == 0) {
            printf("  OK   %s\n", tests[i].name);
            passed++;
        } else {
            printf("  FAIL %s\n", tests[i].name);
        }
    }

    printf("\n%d/%d tests passed\n", passed, total);
    return passed == total ? 0 : 1;
}
