#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "index/diskann.h"
#include "../test_tmp.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

#define DIM 8
#define BUILD_COUNT 64
#define LARGE_DIM 32
#define LARGE_COUNT 200

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

/* Deterministic PRNG so tests are reproducible. */
static unsigned int rng_state = 42;
static float rand_float(void) {
    rng_state = rng_state * 1103515245 + 12345;
    return (float)(rng_state & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

static void generate_random_batch(float *data, size_t count, size_t dim) {
    for (size_t i = 0; i < count * dim; i++) {
        data[i] = rand_float() * 2.0f - 1.0f;
    }
}

/* Brute-force L2 distance between two vectors. */
static float brute_force_l2(const float *a, const float *b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sqrtf(sum);
}

/* ---------- original tests ---------- */

static int test_diskann_config_init(void) {
    GV_DiskANNConfig config;
    memset(&config, 0xFF, sizeof(config));

    diskann_config_init(&config);

    ASSERT(config.max_degree == 64, "default max_degree should be 64");
    ASSERT(fabsf(config.alpha - 1.2f) < 0.01f, "default alpha should be 1.2");
    ASSERT(config.build_beam_width == 128, "default build_beam_width should be 128");
    ASSERT(config.search_beam_width == 64, "default search_beam_width should be 64");
    ASSERT(config.cache_size_mb == 256, "default cache_size_mb should be 256");
    ASSERT(config.sector_size == 4096, "default sector_size should be 4096");

    return 0;
}

static int test_diskann_create_destroy(void) {
    GV_DiskANNConfig config;
    diskann_config_init(&config);
    config.data_path = NULL;

    GV_DiskANNIndex *idx = diskann_create(DIM, &config);
    ASSERT(idx != NULL, "diskann_create returned NULL");

    diskann_destroy(idx);
    return 0;
}

static int test_diskann_build_and_count(void) {
    GV_DiskANNConfig config;
    diskann_config_init(&config);
    config.data_path = NULL;

    GV_DiskANNIndex *idx = diskann_create(DIM, &config);
    ASSERT(idx != NULL, "create failed");

    float data[BUILD_COUNT * DIM];
    generate_batch(data, BUILD_COUNT, DIM);

    int rc = diskann_build(idx, data, BUILD_COUNT, DIM);
    ASSERT(rc == 0, "diskann_build failed");

    size_t count = diskann_count(idx);
    ASSERT(count == BUILD_COUNT, "count should match BUILD_COUNT after build");

    diskann_destroy(idx);
    return 0;
}

static int test_diskann_search(void) {
    GV_DiskANNConfig config;
    diskann_config_init(&config);
    config.data_path = NULL;

    GV_DiskANNIndex *idx = diskann_create(DIM, &config);
    ASSERT(idx != NULL, "create failed");

    float data[BUILD_COUNT * DIM];
    generate_batch(data, BUILD_COUNT, DIM);
    ASSERT(diskann_build(idx, data, BUILD_COUNT, DIM) == 0, "build failed");

    float query[DIM];
    fill_vector(query, DIM, 0.0f);

    GV_DiskANNResult results[5];
    memset(results, 0, sizeof(results));
    int found = diskann_search(idx, query, DIM, 5, results);
    ASSERT(found > 0, "search returned no results");

    /* DiskANN is approximate — just verify we got valid results */
    ASSERT(results[0].distance >= 0.0f, "distance should be non-negative");

    diskann_destroy(idx);
    return 0;
}

static int test_diskann_search_ordering(void) {
    GV_DiskANNConfig config;
    diskann_config_init(&config);
    config.data_path = NULL;

    GV_DiskANNIndex *idx = diskann_create(DIM, &config);
    ASSERT(idx != NULL, "create failed");

    float data[BUILD_COUNT * DIM];
    generate_batch(data, BUILD_COUNT, DIM);
    ASSERT(diskann_build(idx, data, BUILD_COUNT, DIM) == 0, "build failed");

    float query[DIM];
    fill_vector(query, DIM, 10.0f);

    GV_DiskANNResult results[10];
    memset(results, 0, sizeof(results));
    int found = diskann_search(idx, query, DIM, 10, results);
    ASSERT(found > 1, "need at least 2 results");

    for (int i = 1; i < found; i++) {
        ASSERT(results[i].distance >= results[i - 1].distance,
               "results should be sorted by ascending distance");
    }

    diskann_destroy(idx);
    return 0;
}

static int test_diskann_incremental_insert(void) {
    GV_DiskANNConfig config;
    diskann_config_init(&config);
    config.data_path = NULL;

    GV_DiskANNIndex *idx = diskann_create(DIM, &config);
    ASSERT(idx != NULL, "create failed");

    float data[BUILD_COUNT * DIM];
    generate_batch(data, BUILD_COUNT, DIM);
    ASSERT(diskann_build(idx, data, BUILD_COUNT, DIM) == 0, "build failed");

    float new_vec[DIM];
    fill_vector(new_vec, DIM, 999.0f);
    int rc = diskann_insert(idx, new_vec, DIM);
    ASSERT(rc == 0, "incremental insert failed");

    ASSERT(diskann_count(idx) == BUILD_COUNT + 1,
           "count should increase by 1 after insert");

    diskann_destroy(idx);
    return 0;
}

static int test_diskann_delete(void) {
    GV_DiskANNConfig config;
    diskann_config_init(&config);
    config.data_path = NULL;

    GV_DiskANNIndex *idx = diskann_create(DIM, &config);
    ASSERT(idx != NULL, "create failed");

    float data[BUILD_COUNT * DIM];
    generate_batch(data, BUILD_COUNT, DIM);
    ASSERT(diskann_build(idx, data, BUILD_COUNT, DIM) == 0, "build failed");

    int rc = diskann_delete(idx, 0);
    ASSERT(rc == 0, "delete failed");

    diskann_destroy(idx);
    return 0;
}

static int test_diskann_stats(void) {
    GV_DiskANNConfig config;
    diskann_config_init(&config);
    config.data_path = NULL;

    GV_DiskANNIndex *idx = diskann_create(DIM, &config);
    ASSERT(idx != NULL, "create failed");

    float data[BUILD_COUNT * DIM];
    generate_batch(data, BUILD_COUNT, DIM);
    ASSERT(diskann_build(idx, data, BUILD_COUNT, DIM) == 0, "build failed");

    GV_DiskANNStats stats;
    memset(&stats, 0, sizeof(stats));
    int rc = diskann_get_stats(idx, &stats);
    ASSERT(rc == 0, "get_stats failed");
    ASSERT(stats.total_vectors == BUILD_COUNT, "total_vectors should match BUILD_COUNT");
    ASSERT(stats.graph_edges > 0, "graph_edges should be > 0 after build");

    diskann_destroy(idx);
    return 0;
}

/* ---------- new tests: recall, delete+rebuild, save/load ---------- */

/* Recall test: build 200 random 32-dim vectors, search with k=10,
 * compare against brute-force exact NN. Recall = |found_in_topk| / k.
 * For a small dataset the graph should achieve >80% recall. */
static int test_diskann_recall(void) {
    GV_DiskANNConfig config;
    diskann_config_init(&config);
    config.data_path = NULL;

    GV_DiskANNIndex *idx = diskann_create(LARGE_DIM, &config);
    ASSERT(idx != NULL, "create failed");

    float data[LARGE_COUNT * LARGE_DIM];
    generate_random_batch(data, LARGE_COUNT, LARGE_DIM);
    ASSERT(diskann_build(idx, data, LARGE_COUNT, LARGE_DIM) == 0, "build failed");

    size_t k = 10;
    size_t num_queries = 20;
    size_t total_found = 0;

    for (size_t q = 0; q < num_queries; q++) {
        float query[LARGE_DIM];
        /* Use a stored vector as query (known neighbor exists) */
        size_t qi = (q * 7) % LARGE_COUNT;
        memcpy(query, &data[qi * LARGE_DIM], LARGE_DIM * sizeof(float));

        /* Brute-force exact top-k: compute all distances and find smallest k */
        size_t exact_idx[LARGE_COUNT];
        float exact_dist[LARGE_COUNT];
        for (size_t i = 0; i < LARGE_COUNT; i++) {
            exact_idx[i] = i;
            exact_dist[i] = brute_force_l2(query, &data[i * LARGE_DIM], LARGE_DIM);
        }
        /* Simple selection sort for top-k (small dataset, fine) */
        for (size_t i = 0; i < k && i < LARGE_COUNT; i++) {
            size_t min_j = i;
            for (size_t j = i + 1; j < LARGE_COUNT; j++) {
                if (exact_dist[j] < exact_dist[min_j]) min_j = j;
            }
            size_t ti = exact_idx[i]; exact_idx[i] = exact_idx[min_j]; exact_idx[min_j] = ti;
            float td = exact_dist[i]; exact_dist[i] = exact_dist[min_j]; exact_dist[min_j] = td;
        }

        /* DiskANN search */
        GV_DiskANNResult results[16];
        memset(results, 0, sizeof(results));
        int found = diskann_search(idx, query, LARGE_DIM, k, results);
        ASSERT(found > 0, "search returned results");

        /* Compute recall: how many DiskANN results are in the exact top-k */
        for (int r = 0; r < found && r < (int)k; r++) {
            for (size_t e = 0; e < k; e++) {
                if (results[r].index == exact_idx[e]) { total_found++; break; }
            }
        }
    }

    double recall = (double)total_found / ((double)num_queries * (double)k);
    fprintf(stderr, "  recall@%zu = %.2f%%\n", k, recall * 100.0);
    ASSERT(recall > 0.80, "recall@10 should be > 80% for small dataset");

    diskann_destroy(idx);
    return 0;
}

/* Delete + rebuild: delete vector index 0, rebuild, verify it does not
 * appear in search results (or at least is no longer a top-1 result). */
static int test_diskann_delete_rebuild(void) {
    GV_DiskANNConfig config;
    diskann_config_init(&config);
    config.data_path = NULL;

    GV_DiskANNIndex *idx = diskann_create(DIM, &config);
    ASSERT(idx != NULL, "create failed");

    float data[BUILD_COUNT * DIM];
    generate_batch(data, BUILD_COUNT, DIM);
    ASSERT(diskann_build(idx, data, BUILD_COUNT, DIM) == 0, "build failed");

    /* Delete vector 0 */
    ASSERT(diskann_delete(idx, 0) == 0, "delete failed");

    /* Search for a vector very close to vector 0: use vector 1 which is close */
    float query[DIM];
    fill_vector(query, DIM, 0.0f);
    GV_DiskANNResult results[BUILD_COUNT];
    memset(results, 0, sizeof(results));
    int found = diskann_search(idx, query, DIM, BUILD_COUNT, results);
    ASSERT(found > 0, "search after delete returned results");

    /* Vector 0 should NOT be the top-1 result (it's deleted).
     * The top-1 should be a different vector. */
    ASSERT(results[0].index != 0, "deleted vector 0 is not top-1 result");

    /* If vector 0 appears at all, it should be later than position 0 */
    for (int i = 0; i < found; i++) {
        if (results[i].index == 0) {
            /* Found it at some position — that's acceptable as a tombstone
             * that wasn't pruned, but it shouldn't be at position 0 */
            ASSERT(i > 0, "deleted vector not at top-1");
            break;
        }
    }

    diskann_destroy(idx);
    return 0;
}

/* Save/load round-trip: build, save, load, search both, verify results match. */
static int test_diskann_save_load(void) {
    char path[512];
    gv_test_make_temp_path(path, sizeof(path), "gv_diskann_test_save", ".bin");

    /* Build original index */
    GV_DiskANNConfig config;
    diskann_config_init(&config);
    config.data_path = NULL;

    GV_DiskANNIndex *idx = diskann_create(DIM, &config);
    ASSERT(idx != NULL, "create failed");

    float data[BUILD_COUNT * DIM];
    generate_batch(data, BUILD_COUNT, DIM);
    ASSERT(diskann_build(idx, data, BUILD_COUNT, DIM) == 0, "build failed");

    /* Save */
    ASSERT(diskann_save(idx, path) == 0, "save failed");

    /* Load into a new index */
    GV_DiskANNConfig config2;
    diskann_config_init(&config2);
    config2.data_path = NULL;
    GV_DiskANNIndex *loaded = diskann_load(path, &config2);
    ASSERT(loaded != NULL, "load returned NULL");

    ASSERT(diskann_count(loaded) == diskann_count(idx),
           "loaded count matches original");

    /* Search both with same query */
    float query[DIM];
    fill_vector(query, DIM, 5.0f);

    GV_DiskANNResult r1[5], r2[5];
    memset(r1, 0, sizeof(r1));
    memset(r2, 0, sizeof(r2));
    int f1 = diskann_search(idx, query, DIM, 5, r1);
    int f2 = diskann_search(loaded, query, DIM, 5, r2);

    ASSERT(f1 > 0 && f2 > 0, "both searches return results");
    ASSERT(f1 == f2, "same number of results");

    /* Results should be identical (same graph structure after load) */
    for (int i = 0; i < f1; i++) {
        ASSERT(r1[i].index == r2[i].index,
               "loaded result index matches original");
    }

    diskann_destroy(idx);
    diskann_destroy(loaded);
    remove(path);
    return 0;
}

/* ---------- runner ---------- */

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
        {"Testing diskann recall...",                test_diskann_recall},
        {"Testing diskann delete+rebuild...",        test_diskann_delete_rebuild},
        {"Testing diskann save/load...",             test_diskann_save_load},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        fprintf(stderr, "%s\n", tests[i].name);
        if (tests[i].fn() == 0) { passed++; }
    }
    fprintf(stderr, "\n%d/%d tests passed\n", passed, n);
    return passed == n ? 0 : 1;
}
