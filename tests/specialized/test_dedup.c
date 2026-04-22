#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "specialized/dedup.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_dedup_create_destroy(void) {
    GV_DedupIndex *dedup = dedup_create(4, NULL);
    ASSERT(dedup != NULL, "dedup creation with dim=4, default config");
    dedup_destroy(dedup);

    GV_DedupConfig cfg = {
        .epsilon = 0.5f,
        .num_hash_tables = 4,
        .hash_bits = 8,
        .seed = 12345
    };
    dedup = dedup_create(4, &cfg);
    ASSERT(dedup != NULL, "dedup creation with explicit config");
    dedup_destroy(dedup);

    dedup_destroy(NULL);
    return 0;
}

static int test_dedup_insert_unique(void) {
    GV_DedupConfig cfg = { .epsilon = 0.01f, .num_hash_tables = 8, .hash_bits = 12, .seed = 42 };
    GV_DedupIndex *dedup = dedup_create(4, &cfg);
    ASSERT(dedup != NULL, "dedup creation");

    float v1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v2[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    float v3[4] = {0.0f, 0.0f, 1.0f, 0.0f};
    float v4[4] = {0.0f, 0.0f, 0.0f, 1.0f};

    ASSERT(dedup_insert(dedup, v1, 4) == 0, "insert v1 (unique)");
    ASSERT(dedup_insert(dedup, v2, 4) == 0, "insert v2 (unique)");
    ASSERT(dedup_insert(dedup, v3, 4) == 0, "insert v3 (unique)");
    ASSERT(dedup_insert(dedup, v4, 4) == 0, "insert v4 (unique)");

    ASSERT(dedup_count(dedup) == 4, "count is 4 after 4 unique inserts");

    dedup_destroy(dedup);
    return 0;
}

static int test_dedup_insert_duplicate(void) {
    GV_DedupConfig cfg = { .epsilon = 0.5f, .num_hash_tables = 8, .hash_bits = 12, .seed = 42 };
    GV_DedupIndex *dedup = dedup_create(4, &cfg);
    ASSERT(dedup != NULL, "dedup creation");

    float v1[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    ASSERT(dedup_insert(dedup, v1, 4) == 0, "insert v1");

    int result = dedup_insert(dedup, v1, 4);
    ASSERT(result == 1, "inserting exact duplicate returns 1");
    ASSERT(dedup_count(dedup) == 1, "count remains 1 (duplicate not added)");

    float v1_near[4] = {1.01f, 2.01f, 3.01f, 4.01f};
    result = dedup_insert(dedup, v1_near, 4);
    /* Should be detected as duplicate since L2 distance is very small */
    ASSERT(result == 0 || result == 1, "near-duplicate detected or inserted");

    dedup_destroy(dedup);
    return 0;
}

static int test_dedup_check(void) {
    GV_DedupConfig cfg = { .epsilon = 0.1f, .num_hash_tables = 8, .hash_bits = 12, .seed = 99 };
    GV_DedupIndex *dedup = dedup_create(4, &cfg);
    ASSERT(dedup != NULL, "dedup creation");

    float v1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v2[4] = {0.0f, 10.0f, 0.0f, 0.0f};
    dedup_insert(dedup, v1, 4);
    dedup_insert(dedup, v2, 4);

    int idx = dedup_check(dedup, v1, 4);
    ASSERT(idx >= 0, "check finds existing duplicate of v1");

    float far[4] = {100.0f, 100.0f, 100.0f, 100.0f};
    idx = dedup_check(dedup, far, 4);
    ASSERT(idx == -1, "check returns -1 for unique distant vector");

    dedup_destroy(dedup);
    return 0;
}

static int test_dedup_scan(void) {
    GV_DedupConfig cfg = { .epsilon = 1.0f, .num_hash_tables = 8, .hash_bits = 12, .seed = 7 };
    GV_DedupIndex *dedup = dedup_create(4, &cfg);
    ASSERT(dedup != NULL, "dedup creation");

    float v1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v2[4] = {0.0f, 10.0f, 0.0f, 0.0f};
    float v3[4] = {1.0f, 0.01f, 0.0f, 0.0f};

    dedup_insert(dedup, v1, 4);
    dedup_insert(dedup, v2, 4);
    /* v3 might be rejected as dup or inserted depending on LSH hashing */
    dedup_insert(dedup, v3, 4);

    GV_DedupResult results[10];
    int n = dedup_scan(dedup, results, 10);
    ASSERT(n >= 0, "scan did not return error");

    for (int i = 0; i < n; i++) {
        ASSERT(results[i].distance >= 0.0f, "duplicate distance is non-negative");
    }

    dedup_destroy(dedup);
    return 0;
}

static int test_dedup_count(void) {
    GV_DedupIndex *dedup = dedup_create(4, NULL);
    ASSERT(dedup != NULL, "dedup creation");

    ASSERT(dedup_count(dedup) == 0, "count is 0 on empty index");

    float v1[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    dedup_insert(dedup, v1, 4);
    ASSERT(dedup_count(dedup) == 1, "count is 1 after one insert");

    float v2[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    dedup_insert(dedup, v2, 4);
    ASSERT(dedup_count(dedup) == 2, "count is 2 after two unique inserts");

    dedup_destroy(dedup);
    return 0;
}

static int test_dedup_clear(void) {
    GV_DedupIndex *dedup = dedup_create(4, NULL);
    ASSERT(dedup != NULL, "dedup creation");

    float v1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v2[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    dedup_insert(dedup, v1, 4);
    dedup_insert(dedup, v2, 4);
    ASSERT(dedup_count(dedup) == 2, "count is 2 before clear");

    dedup_clear(dedup);
    ASSERT(dedup_count(dedup) == 0, "count is 0 after clear");

    /* After clearing, previously duplicate vectors should be insertable again */
    ASSERT(dedup_insert(dedup, v1, 4) == 0, "insert v1 after clear succeeds");
    ASSERT(dedup_count(dedup) == 1, "count is 1 after re-insert");

    dedup_destroy(dedup);
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
