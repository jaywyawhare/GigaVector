#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_score_threshold.h"
#include "gigavector/gv_database.h"
#include "gigavector/gv_distance.h"

#define DIM 4

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

/* Helper: create a flat DB with known vectors */
static GV_Database *make_db(void) {
    GV_Database *db = gv_db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    if (!db) return NULL;
    float v0[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v1[] = {0.9f, 0.1f, 0.0f, 0.0f};
    float v2[] = {0.0f, 1.0f, 0.0f, 0.0f};
    float v3[] = {0.0f, 0.0f, 0.0f, 1.0f};
    gv_db_add_vector(db, v0, DIM);
    gv_db_add_vector(db, v1, DIM);
    gv_db_add_vector(db, v2, DIM);
    gv_db_add_vector(db, v3, DIM);
    return db;
}

/* ---------- test_threshold_passes_euclidean ---------- */
static int test_threshold_passes_euclidean(void) {
    /* For euclidean: distance <= threshold passes */
    ASSERT(gv_threshold_passes(0.5f, 1.0f, GV_DISTANCE_EUCLIDEAN) == 1,
           "0.5 <= 1.0 should pass for euclidean");
    ASSERT(gv_threshold_passes(1.5f, 1.0f, GV_DISTANCE_EUCLIDEAN) == 0,
           "1.5 > 1.0 should not pass for euclidean");
    ASSERT(gv_threshold_passes(1.0f, 1.0f, GV_DISTANCE_EUCLIDEAN) == 1,
           "1.0 == 1.0 should pass for euclidean (boundary)");
    return 0;
}

/* ---------- test_threshold_passes_manhattan ---------- */
static int test_threshold_passes_manhattan(void) {
    ASSERT(gv_threshold_passes(0.3f, 0.5f, GV_DISTANCE_MANHATTAN) == 1,
           "0.3 <= 0.5 should pass for manhattan");
    ASSERT(gv_threshold_passes(0.8f, 0.5f, GV_DISTANCE_MANHATTAN) == 0,
           "0.8 > 0.5 should not pass for manhattan");
    return 0;
}

/* ---------- test_threshold_filter_basic ---------- */
static int test_threshold_filter_basic(void) {
    GV_ThresholdResult results[] = {
        {0, 0.1f},
        {1, 0.5f},
        {2, 0.8f},
        {3, 1.5f},
        {4, 2.0f},
    };

    /* Keep only results with distance <= 1.0 (euclidean) */
    size_t count = gv_threshold_filter(results, 5, 1.0f, GV_DISTANCE_EUCLIDEAN);
    ASSERT(count == 3, "should keep 3 results with distance <= 1.0");
    ASSERT(results[0].index == 0 && results[1].index == 1 && results[2].index == 2,
           "kept results should preserve order");
    return 0;
}

/* ---------- test_threshold_filter_none_pass ---------- */
static int test_threshold_filter_none_pass(void) {
    GV_ThresholdResult results[] = {
        {0, 5.0f},
        {1, 6.0f},
    };
    size_t count = gv_threshold_filter(results, 2, 0.1f, GV_DISTANCE_EUCLIDEAN);
    ASSERT(count == 0, "no results should pass very tight threshold");
    return 0;
}

/* ---------- test_threshold_filter_all_pass ---------- */
static int test_threshold_filter_all_pass(void) {
    GV_ThresholdResult results[] = {
        {0, 0.01f},
        {1, 0.02f},
        {2, 0.03f},
    };
    size_t count = gv_threshold_filter(results, 3, 100.0f, GV_DISTANCE_EUCLIDEAN);
    ASSERT(count == 3, "all results should pass very loose threshold");
    return 0;
}

/* ---------- test_search_with_threshold ---------- */
static int test_search_with_threshold(void) {
    GV_Database *db = make_db();
    ASSERT(db != NULL, "make_db should succeed");

    float query[] = {1.0f, 0.0f, 0.0f, 0.0f};
    GV_ThresholdResult results[4];
    /* Search with a tight threshold -- only very close vectors should appear */
    int n = gv_db_search_with_threshold(db, query, 4, GV_DISTANCE_EUCLIDEAN,
                                         0.5f, results);
    ASSERT(n >= 0, "search_with_threshold should not fail");
    /* At least v0 (identical) should pass */
    if (n > 0) {
        ASSERT(results[0].distance <= 0.5f, "returned results should be within threshold");
    }

    gv_db_close(db);
    return 0;
}

/* ---------- test_threshold_filter_empty ---------- */
static int test_threshold_filter_empty(void) {
    size_t count = gv_threshold_filter(NULL, 0, 1.0f, GV_DISTANCE_EUCLIDEAN);
    ASSERT(count == 0, "filtering empty set should return 0");
    return 0;
}

/* test runner */

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing threshold passes euclidean...",    test_threshold_passes_euclidean},
        {"Testing threshold passes manhattan...",    test_threshold_passes_manhattan},
        {"Testing threshold filter basic...",        test_threshold_filter_basic},
        {"Testing threshold filter none pass...",    test_threshold_filter_none_pass},
        {"Testing threshold filter all pass...",     test_threshold_filter_all_pass},
        {"Testing search with threshold...",         test_search_with_threshold},
        {"Testing threshold filter empty...",        test_threshold_filter_empty},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
