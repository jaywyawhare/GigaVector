#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gigavector/gv_hnsw_opt.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

#define DIM 8
#define MAX_ELEMENTS 200
#define M_PARAM 16
#define EF_CONSTRUCT 64
#define INSERT_COUNT 50

static void fill_vector(float *vec, size_t dim, float seed) {
    for (size_t i = 0; i < dim; i++) {
        vec[i] = sinf(seed + (float)i * 0.7f);
    }
}

/* ------------------------------------------------------------------ */
/* 1. test_hnsw_inline_create_destroy                                  */
/* ------------------------------------------------------------------ */
static int test_hnsw_inline_create_destroy(void) {
    GV_HNSWInlineConfig config;
    config.quant_bits = 8;
    config.enable_prefetch = 0;
    config.prefetch_distance = 2;

    GV_HNSWInlineIndex *idx = gv_hnsw_inline_create(DIM, MAX_ELEMENTS,
                                                      M_PARAM, EF_CONSTRUCT, &config);
    ASSERT(idx != NULL, "gv_hnsw_inline_create returned NULL");

    gv_hnsw_inline_destroy(idx);
    return 0;
}

/* ------------------------------------------------------------------ */
/* 2. test_hnsw_inline_create_defaults                                 */
/* ------------------------------------------------------------------ */
static int test_hnsw_inline_create_defaults(void) {
    /* NULL config should use defaults */
    GV_HNSWInlineIndex *idx = gv_hnsw_inline_create(DIM, MAX_ELEMENTS,
                                                      M_PARAM, EF_CONSTRUCT, NULL);
    ASSERT(idx != NULL, "create with NULL config returned NULL");

    ASSERT(gv_hnsw_inline_count(idx) == 0, "empty index should have count 0");

    gv_hnsw_inline_destroy(idx);
    return 0;
}

/* ------------------------------------------------------------------ */
/* 3. test_hnsw_inline_insert_count                                    */
/* ------------------------------------------------------------------ */
static int test_hnsw_inline_insert_count(void) {
    GV_HNSWInlineIndex *idx = gv_hnsw_inline_create(DIM, MAX_ELEMENTS,
                                                      M_PARAM, EF_CONSTRUCT, NULL);
    ASSERT(idx != NULL, "create failed");

    for (size_t i = 0; i < INSERT_COUNT; i++) {
        float vec[DIM];
        fill_vector(vec, DIM, (float)i);
        int rc = gv_hnsw_inline_insert(idx, vec, i);
        ASSERT(rc == 0, "insert failed");
    }

    ASSERT(gv_hnsw_inline_count(idx) == INSERT_COUNT,
           "count does not match number of inserts");

    gv_hnsw_inline_destroy(idx);
    return 0;
}

/* ------------------------------------------------------------------ */
/* 4. test_hnsw_inline_search                                          */
/* ------------------------------------------------------------------ */
static int test_hnsw_inline_search(void) {
    GV_HNSWInlineIndex *idx = gv_hnsw_inline_create(DIM, MAX_ELEMENTS,
                                                      M_PARAM, EF_CONSTRUCT, NULL);
    ASSERT(idx != NULL, "create failed");

    for (size_t i = 0; i < INSERT_COUNT; i++) {
        float vec[DIM];
        fill_vector(vec, DIM, (float)i);
        ASSERT(gv_hnsw_inline_insert(idx, vec, i) == 0, "insert failed");
    }

    /* Query with the first inserted vector - should find itself */
    float query[DIM];
    fill_vector(query, DIM, 0.0f);

    size_t labels[5];
    float distances[5];
    int found = gv_hnsw_inline_search(idx, query, 5, 32, labels, distances);
    ASSERT(found > 0, "search returned no results");

    /* The nearest neighbor should be the vector itself (label 0) */
    ASSERT(labels[0] == 0, "nearest neighbor should be label 0");
    ASSERT(distances[0] < 0.001f, "distance to self should be near zero");

    gv_hnsw_inline_destroy(idx);
    return 0;
}

/* ------------------------------------------------------------------ */
/* 5. test_hnsw_inline_search_ordering                                 */
/* ------------------------------------------------------------------ */
static int test_hnsw_inline_search_ordering(void) {
    GV_HNSWInlineIndex *idx = gv_hnsw_inline_create(DIM, MAX_ELEMENTS,
                                                      M_PARAM, EF_CONSTRUCT, NULL);
    ASSERT(idx != NULL, "create failed");

    for (size_t i = 0; i < INSERT_COUNT; i++) {
        float vec[DIM];
        fill_vector(vec, DIM, (float)i);
        ASSERT(gv_hnsw_inline_insert(idx, vec, i) == 0, "insert failed");
    }

    float query[DIM];
    fill_vector(query, DIM, 5.0f);

    size_t labels[10];
    float distances[10];
    int found = gv_hnsw_inline_search(idx, query, 10, 64, labels, distances);
    ASSERT(found > 1, "need at least 2 results for ordering check");

    /* Results should be sorted by ascending distance */
    for (int i = 1; i < found; i++) {
        ASSERT(distances[i] >= distances[i - 1],
               "results should be sorted by ascending distance");
    }

    gv_hnsw_inline_destroy(idx);
    return 0;
}

/* ------------------------------------------------------------------ */
/* 6. test_hnsw_inline_rebuild                                         */
/* ------------------------------------------------------------------ */
static int test_hnsw_inline_rebuild(void) {
    GV_HNSWInlineIndex *idx = gv_hnsw_inline_create(DIM, MAX_ELEMENTS,
                                                      M_PARAM, EF_CONSTRUCT, NULL);
    ASSERT(idx != NULL, "create failed");

    for (size_t i = 0; i < INSERT_COUNT; i++) {
        float vec[DIM];
        fill_vector(vec, DIM, (float)i);
        ASSERT(gv_hnsw_inline_insert(idx, vec, i) == 0, "insert failed");
    }

    /* Rebuild synchronously with defaults */
    GV_HNSWRebuildConfig rconfig;
    rconfig.connectivity_ratio = 0.8f;
    rconfig.batch_size = 1000;
    rconfig.background = 0;

    int rc = gv_hnsw_inline_rebuild(idx, &rconfig);
    ASSERT(rc == 0, "rebuild failed");

    GV_HNSWRebuildStats stats;
    rc = gv_hnsw_inline_rebuild_status(idx, &stats);
    ASSERT(rc == 0, "rebuild_status failed");
    ASSERT(stats.completed == 1, "synchronous rebuild should be completed");
    ASSERT(stats.nodes_processed > 0, "should have processed some nodes");

    gv_hnsw_inline_destroy(idx);
    return 0;
}

/* ------------------------------------------------------------------ */
/* 7. test_hnsw_inline_4bit_quant                                      */
/* ------------------------------------------------------------------ */
static int test_hnsw_inline_4bit_quant(void) {
    GV_HNSWInlineConfig config;
    config.quant_bits = 4;
    config.enable_prefetch = 1;
    config.prefetch_distance = 3;

    GV_HNSWInlineIndex *idx = gv_hnsw_inline_create(DIM, MAX_ELEMENTS,
                                                      M_PARAM, EF_CONSTRUCT, &config);
    ASSERT(idx != NULL, "create with 4-bit quant failed");

    for (size_t i = 0; i < 20; i++) {
        float vec[DIM];
        fill_vector(vec, DIM, (float)i * 2.0f);
        ASSERT(gv_hnsw_inline_insert(idx, vec, i) == 0, "insert failed");
    }

    float query[DIM];
    fill_vector(query, DIM, 0.0f);

    size_t labels[3];
    float distances[3];
    int found = gv_hnsw_inline_search(idx, query, 3, 32, labels, distances);
    ASSERT(found > 0, "search with 4-bit quant returned no results");

    gv_hnsw_inline_destroy(idx);
    return 0;
}

/* ------------------------------------------------------------------ */
/* 8. test_hnsw_inline_destroy_null                                    */
/* ------------------------------------------------------------------ */
static int test_hnsw_inline_destroy_null(void) {
    /* Should be safe to call with NULL */
    gv_hnsw_inline_destroy(NULL);
    return 0;
}

/* ================================================================== */
/* main                                                                */
/* ================================================================== */
typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing hnsw inline create/destroy...",    test_hnsw_inline_create_destroy},
        {"Testing hnsw inline create defaults...",   test_hnsw_inline_create_defaults},
        {"Testing hnsw inline insert/count...",      test_hnsw_inline_insert_count},
        {"Testing hnsw inline search...",            test_hnsw_inline_search},
        {"Testing hnsw inline search ordering...",   test_hnsw_inline_search_ordering},
        {"Testing hnsw inline rebuild...",           test_hnsw_inline_rebuild},
        {"Testing hnsw inline 4-bit quant...",       test_hnsw_inline_4bit_quant},
        {"Testing hnsw inline destroy null...",      test_hnsw_inline_destroy_null},
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
