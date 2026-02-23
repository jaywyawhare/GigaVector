#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_gpu.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_gpu_available(void) {
    int avail = gv_gpu_available();
    ASSERT(avail == 0 || avail == 1, "gv_gpu_available should return 0 or 1");
    return 0;
}

static int test_gpu_device_count(void) {
    int count = gv_gpu_device_count();
    ASSERT(count >= 0, "gv_gpu_device_count should return >= 0");

    /* If no GPU available, count must be 0 */
    if (!gv_gpu_available()) {
        ASSERT(count == 0, "device_count should be 0 when GPU not available");
    }
    return 0;
}

static int test_gpu_config_init(void) {
    GV_GPUConfig config;
    memset(&config, 0xFF, sizeof(config));
    gv_gpu_config_init(&config);

    ASSERT(config.device_id == -1, "default device_id should be -1 (auto)");
    ASSERT(config.max_vectors_per_batch > 0, "default max_vectors_per_batch should be positive");
    ASSERT(config.max_query_batch_size > 0, "default max_query_batch_size should be positive");
    ASSERT(config.enable_tensor_cores == 1, "default enable_tensor_cores should be 1");
    ASSERT(config.enable_async_transfers == 1, "default enable_async_transfers should be 1");
    ASSERT(config.stream_count > 0, "default stream_count should be positive");
    ASSERT(config.memory.initial_size > 0, "default initial memory pool size should be positive");
    ASSERT(config.memory.max_size >= config.memory.initial_size,
           "max memory should be >= initial memory");
    ASSERT(config.memory.allow_growth == 1, "default allow_growth should be 1");
    return 0;
}

static int test_gpu_config_init_twice(void) {
    /* Calling config_init twice should produce identical field values */
    GV_GPUConfig c1, c2;
    memset(&c1, 0xAA, sizeof(c1));
    memset(&c2, 0x55, sizeof(c2));
    gv_gpu_config_init(&c1);
    gv_gpu_config_init(&c2);
    ASSERT(c1.device_id == c2.device_id, "device_id should match");
    ASSERT(c1.max_vectors_per_batch == c2.max_vectors_per_batch, "max_vectors_per_batch should match");
    ASSERT(c1.max_query_batch_size == c2.max_query_batch_size, "max_query_batch_size should match");
    ASSERT(c1.enable_tensor_cores == c2.enable_tensor_cores, "enable_tensor_cores should match");
    ASSERT(c1.stream_count == c2.stream_count, "stream_count should match");
    ASSERT(c1.memory.initial_size == c2.memory.initial_size, "memory.initial_size should match");
    ASSERT(c1.memory.max_size == c2.memory.max_size, "memory.max_size should match");
    return 0;
}

static int test_gpu_create_no_gpu(void) {
    /*
     * gv_gpu_create provides a CPU fallback context even when no GPU is
     * available, so it may return a valid (non-NULL) context.  We verify
     * that it does not crash and that the returned context can be cleaned up.
     */
    GV_GPUConfig config;
    gv_gpu_config_init(&config);
    GV_GPUContext *ctx = gv_gpu_create(&config);
    /* Context may be non-NULL (CPU fallback) or NULL (allocation failure) */
    if (ctx) {
        gv_gpu_destroy(ctx);
    }

    /* Also test with NULL config */
    ctx = gv_gpu_create(NULL);
    if (ctx) {
        gv_gpu_destroy(ctx);
    }
    return 0;
}

static int test_gpu_destroy_null(void) {
    /* Destroying NULL should be safe (no crash) */
    gv_gpu_destroy(NULL);
    return 0;
}

static int test_gpu_get_error_null(void) {
    const char *err = gv_gpu_get_error(NULL);
    /* Should return a non-NULL safe string or NULL without crashing */
    (void)err; /* Just ensure no crash */
    return 0;
}

static int test_gpu_get_device_info_invalid(void) {
    GV_GPUDeviceInfo info;
    memset(&info, 0, sizeof(info));

    /* NULL info pointer should fail */
    int rc = gv_gpu_get_device_info(0, NULL);
    ASSERT(rc == -1, "get_device_info with NULL info should return -1");

    /*
     * Without CUDA, get_device_info returns CPU fallback info (rc=0)
     * for any device_id.  With CUDA, invalid IDs would return -1.
     * We just verify no crash and that info is populated.
     */
    rc = gv_gpu_get_device_info(-1, &info);
    if (gv_gpu_available()) {
        ASSERT(rc == -1, "get_device_info with device_id=-1 should return -1 with GPU");
    } else {
        ASSERT(rc == 0, "get_device_info with fallback should return 0");
        ASSERT(strlen(info.name) > 0, "fallback device info should have a name");
    }

    memset(&info, 0, sizeof(info));
    rc = gv_gpu_get_device_info(9999, &info);
    if (gv_gpu_available()) {
        ASSERT(rc == -1, "get_device_info with device_id=9999 should return -1 with GPU");
    } else {
        ASSERT(rc == 0, "get_device_info fallback for any device_id returns 0");
    }
    return 0;
}

static int test_gpu_synchronize_null(void) {
    int rc = gv_gpu_synchronize(NULL);
    ASSERT(rc == -1, "gv_gpu_synchronize(NULL) should return -1");
    return 0;
}

static int test_gpu_get_stats_null(void) {
    GV_GPUStats stats;
    memset(&stats, 0xFF, sizeof(stats));
    int rc = gv_gpu_get_stats(NULL, &stats);
    ASSERT(rc == -1, "gv_gpu_get_stats(NULL, ...) should return -1");

    /* Also test NULL stats pointer (if context were valid, but we pass NULL for both) */
    rc = gv_gpu_get_stats(NULL, NULL);
    ASSERT(rc == -1, "gv_gpu_get_stats(NULL, NULL) should return -1");
    return 0;
}

static int test_gpu_reset_stats_null(void) {
    int rc = gv_gpu_reset_stats(NULL);
    ASSERT(rc == -1, "gv_gpu_reset_stats(NULL) should return -1");
    return 0;
}

static int test_gpu_index_create_null(void) {
    float vectors[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    GV_GPUIndex *idx = gv_gpu_index_create(NULL, vectors, 2, 4);
    ASSERT(idx == NULL, "gv_gpu_index_create with NULL ctx should return NULL");
    return 0;
}

static int test_gpu_index_destroy_null(void) {
    /* Should not crash */
    gv_gpu_index_destroy(NULL);
    return 0;
}

static int test_gpu_index_info_null(void) {
    size_t count, dim, mem;
    int rc = gv_gpu_index_info(NULL, &count, &dim, &mem);
    ASSERT(rc == -1, "gv_gpu_index_info(NULL, ...) should return -1");
    return 0;
}

static int test_gpu_knn_search_null(void) {
    float query[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float db_vecs[8] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    GV_GPUSearchParams params;
    memset(&params, 0, sizeof(params));
    params.k = 1;
    size_t indices[1];
    float distances[1];

    int rc = gv_gpu_knn_search(NULL, query, 1, db_vecs, 2, 4, &params, indices, distances);
    ASSERT(rc == -1, "gv_gpu_knn_search with NULL ctx should return -1");
    return 0;
}

static int test_gpu_compute_distances_null(void) {
    float q[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float db[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float out[1];

    int rc = gv_gpu_compute_distances(NULL, q, 1, db, 1, 4, GV_GPU_EUCLIDEAN, out);
    ASSERT(rc == -1, "gv_gpu_compute_distances with NULL ctx should return -1");
    return 0;
}

static int test_gpu_distance_metric_values(void) {
    /* Verify enum values are distinct and defined */
    ASSERT(GV_GPU_EUCLIDEAN == 0, "GV_GPU_EUCLIDEAN should be 0");
    ASSERT(GV_GPU_COSINE == 1, "GV_GPU_COSINE should be 1");
    ASSERT(GV_GPU_DOT_PRODUCT == 2, "GV_GPU_DOT_PRODUCT should be 2");
    ASSERT(GV_GPU_MANHATTAN == 3, "GV_GPU_MANHATTAN should be 3");
    return 0;
}

static int test_gpu_batch_add_null(void) {
    float vecs[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    int rc = gv_gpu_batch_add(NULL, NULL, vecs, 1);
    ASSERT(rc == -1, "gv_gpu_batch_add with NULL ctx and db should return -1");
    return 0;
}

static int test_gpu_index_search_null(void) {
    float query[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    GV_GPUSearchParams params;
    memset(&params, 0, sizeof(params));
    params.k = 1;
    size_t indices[1];
    float distances[1];

    int rc = gv_gpu_index_search(NULL, query, &params, indices, distances);
    ASSERT(rc == -1, "gv_gpu_index_search with NULL index should return -1");
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing gpu_available...", test_gpu_available},
        {"Testing gpu_device_count...", test_gpu_device_count},
        {"Testing gpu_config_init...", test_gpu_config_init},
        {"Testing gpu_config_init_twice...", test_gpu_config_init_twice},
        {"Testing gpu_create_no_gpu...", test_gpu_create_no_gpu},
        {"Testing gpu_destroy_null...", test_gpu_destroy_null},
        {"Testing gpu_get_error_null...", test_gpu_get_error_null},
        {"Testing gpu_get_device_info_invalid...", test_gpu_get_device_info_invalid},
        {"Testing gpu_synchronize_null...", test_gpu_synchronize_null},
        {"Testing gpu_get_stats_null...", test_gpu_get_stats_null},
        {"Testing gpu_reset_stats_null...", test_gpu_reset_stats_null},
        {"Testing gpu_index_create_null...", test_gpu_index_create_null},
        {"Testing gpu_index_destroy_null...", test_gpu_index_destroy_null},
        {"Testing gpu_index_info_null...", test_gpu_index_info_null},
        {"Testing gpu_knn_search_null...", test_gpu_knn_search_null},
        {"Testing gpu_compute_distances_null...", test_gpu_compute_distances_null},
        {"Testing gpu_distance_metric_values...", test_gpu_distance_metric_values},
        {"Testing gpu_batch_add_null...", test_gpu_batch_add_null},
        {"Testing gpu_index_search_null...", test_gpu_index_search_null},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        printf("  %s ", tests[i].name);
        if (tests[i].fn() == 0) {
            printf("OK\n");
            passed++;
        } else {
            printf("FAILED\n");
        }
    }
    printf("\n%d/%d tests passed\n", passed, n);
    return passed == n ? 0 : 1;
}
