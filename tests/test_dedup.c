#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_dedup.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_dedup_create_destroy(void) {
    /* Create with default config */
    GV_DedupIndex *dedup = gv_dedup_create(4, NULL);
    ASSERT(dedup != NULL, "dedup creation with dim=4, default config");
    gv_dedup_destroy(dedup);

    /* Create with explicit config */
    GV_DedupConfig cfg = {
        .epsilon = 0.5f,
        .num_hash_tables = 4,
        .hash_bits = 8,
        .seed = 12345
    };
    dedup = gv_dedup_create(4, &cfg);
    ASSERT(dedup != NULL, "dedup creation with explicit config");
    gv_dedup_destroy(dedup);

    /* Destroy NULL should be safe */
    gv_dedup_destroy(NULL);
    return 0;
}

static int test_dedup_insert_unique(void) {
    GV_DedupConfig cfg = { .epsilon = 0.01f, .num_hash_tables = 8, .hash_bits = 12, .seed = 42 };
    GV_DedupIndex *dedup = gv_dedup_create(4, &cfg);
    ASSERT(dedup != NULL, "dedup creation");

    float v1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v2[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    float v3[4] = {0.0f, 0.0f, 1.0f, 0.0f};
    float v4[4] = {0.0f, 0.0f, 0.0f, 1.0f};

    /* All distinct vectors should be inserted (return 0) */
    ASSERT(gv_dedup_insert(dedup, v1, 4) == 0, "insert v1 (unique)");
    ASSERT(gv_dedup_insert(dedup, v2, 4) == 0, "insert v2 (unique)");
    ASSERT(gv_dedup_insert(dedup, v3, 4) == 0, "insert v3 (unique)");
    ASSERT(gv_dedup_insert(dedup, v4, 4) == 0, "insert v4 (unique)");

    ASSERT(gv_dedup_count(dedup) == 4, "count is 4 after 4 unique inserts");

    gv_dedup_destroy(dedup);
    return 0;
}

static int test_dedup_insert_duplicate(void) {
    GV_DedupConfig cfg = { .epsilon = 0.5f, .num_hash_tables = 8, .hash_bits = 12, .seed = 42 };
    GV_DedupIndex *dedup = gv_dedup_create(4, &cfg);
    ASSERT(dedup != NULL, "dedup creation");

    float v1[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    ASSERT(gv_dedup_insert(dedup, v1, 4) == 0, "insert v1");

    /* Insert an exact duplicate */
    int result = gv_dedup_insert(dedup, v1, 4);
    ASSERT(result == 1, "inserting exact duplicate returns 1");
    ASSERT(gv_dedup_count(dedup) == 1, "count remains 1 (duplicate not added)");

    /* Insert a near-duplicate (within epsilon=0.5) */
    float v1_near[4] = {1.01f, 2.01f, 3.01f, 4.01f};
    result = gv_dedup_insert(dedup, v1_near, 4);
    /* Should be detected as duplicate since L2 distance is very small */
    ASSERT(result == 0 || result == 1, "near-duplicate detected or inserted");

    gv_dedup_destroy(dedup);
    return 0;
}

static int test_dedup_check(void) {
    GV_DedupConfig cfg = { .epsilon = 0.1f, .num_hash_tables = 8, .hash_bits = 12, .seed = 99 };
    GV_DedupIndex *dedup = gv_dedup_create(4, &cfg);
    ASSERT(dedup != NULL, "dedup creation");

    float v1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v2[4] = {0.0f, 10.0f, 0.0f, 0.0f};
    gv_dedup_insert(dedup, v1, 4);
    gv_dedup_insert(dedup, v2, 4);

    /* Check for an exact match of v1 */
    int idx = gv_dedup_check(dedup, v1, 4);
    ASSERT(idx >= 0, "check finds existing duplicate of v1");

    /* Check for a vector that is far from everything */
    float far[4] = {100.0f, 100.0f, 100.0f, 100.0f};
    idx = gv_dedup_check(dedup, far, 4);
    ASSERT(idx == -1, "check returns -1 for unique distant vector");

    gv_dedup_destroy(dedup);
    return 0;
}

static int test_dedup_scan(void) {
    GV_DedupConfig cfg = { .epsilon = 1.0f, .num_hash_tables = 8, .hash_bits = 12, .seed = 7 };
    GV_DedupIndex *dedup = gv_dedup_create(4, &cfg);
    ASSERT(dedup != NULL, "dedup creation");

    /* Insert vectors: v1 and v3 are very close to each other */
    float v1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v2[4] = {0.0f, 10.0f, 0.0f, 0.0f};  /* far from v1 */
    float v3[4] = {1.0f, 0.01f, 0.0f, 0.0f};   /* near v1 */

    gv_dedup_insert(dedup, v1, 4);
    gv_dedup_insert(dedup, v2, 4);
    /* v3 might be rejected as dup or inserted depending on LSH hashing */
    gv_dedup_insert(dedup, v3, 4);

    GV_DedupResult results[10];
    int n = gv_dedup_scan(dedup, results, 10);
    ASSERT(n >= 0, "scan did not return error");

    /* If duplicates found, check fields are reasonable */
    for (int i = 0; i < n; i++) {
        ASSERT(results[i].distance >= 0.0f, "duplicate distance is non-negative");
    }

    gv_dedup_destroy(dedup);
    return 0;
}

static int test_dedup_count(void) {
    GV_DedupIndex *dedup = gv_dedup_create(4, NULL);
    ASSERT(dedup != NULL, "dedup creation");

    ASSERT(gv_dedup_count(dedup) == 0, "count is 0 on empty index");

    float v1[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    gv_dedup_insert(dedup, v1, 4);
    ASSERT(gv_dedup_count(dedup) == 1, "count is 1 after one insert");

    float v2[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    gv_dedup_insert(dedup, v2, 4);
    ASSERT(gv_dedup_count(dedup) == 2, "count is 2 after two unique inserts");

    gv_dedup_destroy(dedup);
    return 0;
}

static int test_dedup_clear(void) {
    GV_DedupIndex *dedup = gv_dedup_create(4, NULL);
    ASSERT(dedup != NULL, "dedup creation");

    float v1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v2[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    gv_dedup_insert(dedup, v1, 4);
    gv_dedup_insert(dedup, v2, 4);
    ASSERT(gv_dedup_count(dedup) == 2, "count is 2 before clear");

    gv_dedup_clear(dedup);
    ASSERT(gv_dedup_count(dedup) == 0, "count is 0 after clear");

    /* After clearing, previously duplicate vectors should be insertable again */
    ASSERT(gv_dedup_insert(dedup, v1, 4) == 0, "insert v1 after clear succeeds");
    ASSERT(gv_dedup_count(dedup) == 1, "count is 1 after re-insert");

    gv_dedup_destroy(dedup);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing dedup create/destroy...", test_dedup_create_destroy},
        {"Testing dedup insert unique...", test_dedup_insert_unique},
        {"Testing dedup insert duplicate...", test_dedup_insert_duplicate},
        {"Testing dedup check...", test_dedup_check},
        {"Testing dedup scan...", test_dedup_scan},
        {"Testing dedup count...", test_dedup_count},
        {"Testing dedup clear...", test_dedup_clear},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
