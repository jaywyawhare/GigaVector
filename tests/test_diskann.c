#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gigavector/gv_diskann.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

#define DIM 8
#define BUILD_COUNT 64

static void fill_vector(float *vec, size_t dim, float seed) {
    for (size_t i = 0; i < dim; i++) {
        vec[i] = sinf(seed + (float)i * 0.5f);
    }
}

static void generate_batch(float *data, size_t count, size_t dim) {
    for (size_t i = 0; i < count; i++) {
        fill_vector(&data[i * dim], dim, (float)i);
    }
}

/* ------------------------------------------------------------------ */
/* 1. test_diskann_config_init                                         */
/* ------------------------------------------------------------------ */
static int test_diskann_config_init(void) {
    GV_DiskANNConfig config;
    memset(&config, 0xFF, sizeof(config));

    gv_diskann_config_init(&config);

    ASSERT(config.max_degree == 64, "default max_degree should be 64");
    ASSERT(fabsf(config.alpha - 1.2f) < 0.01f, "default alpha should be 1.2");
    ASSERT(config.build_beam_width == 128, "default build_beam_width should be 128");
    ASSERT(config.search_beam_width == 64, "default search_beam_width should be 64");
    ASSERT(config.cache_size_mb == 256, "default cache_size_mb should be 256");
    ASSERT(config.sector_size == 4096, "default sector_size should be 4096");

    return 0;
}

/* ------------------------------------------------------------------ */
/* 2. test_diskann_create_destroy                                      */
/* ------------------------------------------------------------------ */
static int test_diskann_create_destroy(void) {
    GV_DiskANNConfig config;
    gv_diskann_config_init(&config);
    config.data_path = NULL;

    GV_DiskANNIndex *idx = gv_diskann_create(DIM, &config);
    ASSERT(idx != NULL, "gv_diskann_create returned NULL");

    gv_diskann_destroy(idx);
    return 0;
}

/* ------------------------------------------------------------------ */
/* 3. test_diskann_build_and_count                                     */
/* ------------------------------------------------------------------ */
static int test_diskann_build_and_count(void) {
    GV_DiskANNConfig config;
    gv_diskann_config_init(&config);
    config.data_path = NULL;

    GV_DiskANNIndex *idx = gv_diskann_create(DIM, &config);
    ASSERT(idx != NULL, "create failed");

    float data[BUILD_COUNT * DIM];
    generate_batch(data, BUILD_COUNT, DIM);

    int rc = gv_diskann_build(idx, data, BUILD_COUNT, DIM);
    ASSERT(rc == 0, "gv_diskann_build failed");

    size_t count = gv_diskann_count(idx);
    ASSERT(count == BUILD_COUNT, "count should match BUILD_COUNT after build");

    gv_diskann_destroy(idx);
    return 0;
}

/* ------------------------------------------------------------------ */
/* 4. test_diskann_search                                              */
/* ------------------------------------------------------------------ */
static int test_diskann_search(void) {
    GV_DiskANNConfig config;
    gv_diskann_config_init(&config);
    config.data_path = NULL;

    GV_DiskANNIndex *idx = gv_diskann_create(DIM, &config);
    ASSERT(idx != NULL, "create failed");

    float data[BUILD_COUNT * DIM];
    generate_batch(data, BUILD_COUNT, DIM);
    ASSERT(gv_diskann_build(idx, data, BUILD_COUNT, DIM) == 0, "build failed");

    /* Query with the first vector */
    float query[DIM];
    fill_vector(query, DIM, 0.0f);

    GV_DiskANNResult results[5];
    memset(results, 0, sizeof(results));
    int found = gv_diskann_search(idx, query, DIM, 5, results);
    ASSERT(found > 0, "search returned no results");

    /* Nearest neighbor should be vector index 0 */
    ASSERT(results[0].index == 0, "nearest should be index 0");
    ASSERT(results[0].distance < 0.001f, "distance to self should be near zero");

    gv_diskann_destroy(idx);
    return 0;
}

/* ------------------------------------------------------------------ */
/* 5. test_diskann_search_ordering                                     */
/* ------------------------------------------------------------------ */
static int test_diskann_search_ordering(void) {
    GV_DiskANNConfig config;
    gv_diskann_config_init(&config);
    config.data_path = NULL;

    GV_DiskANNIndex *idx = gv_diskann_create(DIM, &config);
    ASSERT(idx != NULL, "create failed");

    float data[BUILD_COUNT * DIM];
    generate_batch(data, BUILD_COUNT, DIM);
    ASSERT(gv_diskann_build(idx, data, BUILD_COUNT, DIM) == 0, "build failed");

    float query[DIM];
    fill_vector(query, DIM, 10.0f);

    GV_DiskANNResult results[10];
    memset(results, 0, sizeof(results));
    int found = gv_diskann_search(idx, query, DIM, 10, results);
    ASSERT(found > 1, "need at least 2 results");

    for (int i = 1; i < found; i++) {
        ASSERT(results[i].distance >= results[i - 1].distance,
               "results should be sorted by ascending distance");
    }

    gv_diskann_destroy(idx);
    return 0;
}

/* ------------------------------------------------------------------ */
/* 6. test_diskann_incremental_insert                                  */
/* ------------------------------------------------------------------ */
static int test_diskann_incremental_insert(void) {
    GV_DiskANNConfig config;
    gv_diskann_config_init(&config);
    config.data_path = NULL;

    GV_DiskANNIndex *idx = gv_diskann_create(DIM, &config);
    ASSERT(idx != NULL, "create failed");

    /* Build initial index */
    float data[BUILD_COUNT * DIM];
    generate_batch(data, BUILD_COUNT, DIM);
    ASSERT(gv_diskann_build(idx, data, BUILD_COUNT, DIM) == 0, "build failed");

    /* Incrementally insert a new vector */
    float new_vec[DIM];
    fill_vector(new_vec, DIM, 999.0f);
    int rc = gv_diskann_insert(idx, new_vec, DIM);
    ASSERT(rc == 0, "incremental insert failed");

    ASSERT(gv_diskann_count(idx) == BUILD_COUNT + 1,
           "count should increase by 1 after insert");

    gv_diskann_destroy(idx);
    return 0;
}

/* ------------------------------------------------------------------ */
/* 7. test_diskann_delete                                              */
/* ------------------------------------------------------------------ */
static int test_diskann_delete(void) {
    GV_DiskANNConfig config;
    gv_diskann_config_init(&config);
    config.data_path = NULL;

    GV_DiskANNIndex *idx = gv_diskann_create(DIM, &config);
    ASSERT(idx != NULL, "create failed");

    float data[BUILD_COUNT * DIM];
    generate_batch(data, BUILD_COUNT, DIM);
    ASSERT(gv_diskann_build(idx, data, BUILD_COUNT, DIM) == 0, "build failed");

    /* Delete the first vector */
    int rc = gv_diskann_delete(idx, 0);
    ASSERT(rc == 0, "delete failed");

    gv_diskann_destroy(idx);
    return 0;
}

/* ------------------------------------------------------------------ */
/* 8. test_diskann_stats                                               */
/* ------------------------------------------------------------------ */
static int test_diskann_stats(void) {
    GV_DiskANNConfig config;
    gv_diskann_config_init(&config);
    config.data_path = NULL;

    GV_DiskANNIndex *idx = gv_diskann_create(DIM, &config);
    ASSERT(idx != NULL, "create failed");

    float data[BUILD_COUNT * DIM];
    generate_batch(data, BUILD_COUNT, DIM);
    ASSERT(gv_diskann_build(idx, data, BUILD_COUNT, DIM) == 0, "build failed");

    GV_DiskANNStats stats;
    memset(&stats, 0, sizeof(stats));
    int rc = gv_diskann_get_stats(idx, &stats);
    ASSERT(rc == 0, "get_stats failed");
    ASSERT(stats.total_vectors == BUILD_COUNT, "total_vectors should match BUILD_COUNT");
    ASSERT(stats.graph_edges > 0, "graph_edges should be > 0 after build");

    gv_diskann_destroy(idx);
    return 0;
}

/* ================================================================== */
/* main                                                                */
/* ================================================================== */
typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing diskann config init...",           test_diskann_config_init},
        {"Testing diskann create/destroy...",        test_diskann_create_destroy},
        {"Testing diskann build and count...",       test_diskann_build_and_count},
        {"Testing diskann search...",                test_diskann_search},
        {"Testing diskann search ordering...",       test_diskann_search_ordering},
        {"Testing diskann incremental insert...",    test_diskann_incremental_insert},
        {"Testing diskann delete...",                test_diskann_delete},
        {"Testing diskann stats...",                 test_diskann_stats},
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
