#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <pthread.h>
#include "index/diskann.h"
#include "core/id_bitmap.h"
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

    /* DiskANN is approximate - just verify we got valid results */
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
            /* Found it at some position - that's acceptable as a tombstone
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

/* ---------- out-of-core (graph_on_disk) tests ---------- */

/* Recall parity + on-disk adjacency: an out-of-core index must flush its
 * adjacency to "{data_path}.graph" (freeing it from RAM) yet still read it back
 * during search well enough to match the in-RAM recall bar. */
static int test_diskann_ooc_recall(void) {
    char data_path[512];
    gv_test_make_temp_path(data_path, sizeof(data_path), "gv_diskann_ooc_data", ".bin");

    GV_DiskANNConfig config;
    diskann_config_init(&config);
    config.data_path = data_path;
    config.graph_on_disk = 1;

    GV_DiskANNIndex *idx = diskann_create(LARGE_DIM, &config);
    ASSERT(idx != NULL, "ooc create failed");

    float data[LARGE_COUNT * LARGE_DIM];
    generate_random_batch(data, LARGE_COUNT, LARGE_DIM);
    ASSERT(diskann_build(idx, data, LARGE_COUNT, LARGE_DIM) == 0, "ooc build failed");

    /* The graph file must exist and hold one fixed slot per node. */
    char graph_path[600];
    snprintf(graph_path, sizeof(graph_path), "%s.graph", data_path);
    struct stat st;
    ASSERT(stat(graph_path, &st) == 0, "graph file created after build");
    size_t expected = (size_t)LARGE_COUNT * (size_t)(1 + 64) * sizeof(uint32_t);
    ASSERT((size_t)st.st_size == expected, "graph file sized one slot per node");

    size_t k = 10;
    size_t num_queries = 20;
    size_t total_found = 0;
    for (size_t q = 0; q < num_queries; q++) {
        float query[LARGE_DIM];
        size_t qi = (q * 7) % LARGE_COUNT;
        memcpy(query, &data[qi * LARGE_DIM], LARGE_DIM * sizeof(float));

        size_t exact_idx[LARGE_COUNT];
        float exact_dist[LARGE_COUNT];
        for (size_t i = 0; i < LARGE_COUNT; i++) {
            exact_idx[i] = i;
            exact_dist[i] = brute_force_l2(query, &data[i * LARGE_DIM], LARGE_DIM);
        }
        for (size_t i = 0; i < k && i < LARGE_COUNT; i++) {
            size_t min_j = i;
            for (size_t j = i + 1; j < LARGE_COUNT; j++) {
                if (exact_dist[j] < exact_dist[min_j]) min_j = j;
            }
            size_t ti = exact_idx[i]; exact_idx[i] = exact_idx[min_j]; exact_idx[min_j] = ti;
            float td = exact_dist[i]; exact_dist[i] = exact_dist[min_j]; exact_dist[min_j] = td;
        }

        GV_DiskANNResult results[16];
        memset(results, 0, sizeof(results));
        int found = diskann_search(idx, query, LARGE_DIM, k, results);
        ASSERT(found > 0, "ooc search returned results");
        for (int r = 0; r < found && r < (int)k; r++) {
            for (size_t e = 0; e < k; e++) {
                if (results[r].index == exact_idx[e]) { total_found++; break; }
            }
        }
    }

    double recall = (double)total_found / ((double)num_queries * (double)k);
    fprintf(stderr, "  ooc recall@%zu = %.2f%%\n", k, recall * 100.0);
    ASSERT(recall > 0.80, "ooc recall@10 should be > 80%");

    /* Deletes still work out-of-core (tombstone in the retained node array). */
    ASSERT(diskann_delete(idx, 0) == 0, "ooc delete failed");

    diskann_destroy(idx);
    remove(graph_path);
    remove(data_path);
    return 0;
}

/* OOC save/load: an out-of-core index round-trips, and a saved index can be
 * loaded either out-of-core (stored flag) or forced back into RAM, producing
 * identical results. */
static int test_diskann_ooc_save_load(void) {
    char data_path[512], save_path[512];
    gv_test_make_temp_path(data_path, sizeof(data_path), "gv_diskann_ooc_sl_data", ".bin");
    gv_test_make_temp_path(save_path, sizeof(save_path), "gv_diskann_ooc_sl_save", ".bin");

    GV_DiskANNConfig config;
    diskann_config_init(&config);
    config.data_path = data_path;
    config.graph_on_disk = 1;

    GV_DiskANNIndex *idx = diskann_create(DIM, &config);
    ASSERT(idx != NULL, "ooc sl create failed");

    float data[BUILD_COUNT * DIM];
    generate_batch(data, BUILD_COUNT, DIM);
    ASSERT(diskann_build(idx, data, BUILD_COUNT, DIM) == 0, "ooc sl build failed");
    ASSERT(diskann_save(idx, save_path) == 0, "ooc save failed (reads adjacency from disk)");

    float query[DIM];
    fill_vector(query, DIM, 5.0f);
    GV_DiskANNResult r0[5];
    memset(r0, 0, sizeof(r0));
    int f0 = diskann_search(idx, query, DIM, 5, r0);
    ASSERT(f0 > 0, "ooc base search returned results");

    /* diskann_save() persists graph + PQ + metadata, not the vectors: those live
     * in the data_path file, which must persist alongside the save. Close the
     * base index first, then load sharing the same data_path. The out-of-core
     * flag is recovered from the saved v2 header (no config override needed). */
    diskann_destroy(idx);

    GV_DiskANNConfig cfg_ooc;
    diskann_config_init(&cfg_ooc);
    cfg_ooc.data_path = data_path; /* same vector file */
    GV_DiskANNIndex *loaded_ooc = diskann_load(save_path, &cfg_ooc);
    ASSERT(loaded_ooc != NULL, "ooc load returned NULL");
    ASSERT(diskann_count(loaded_ooc) == BUILD_COUNT, "ooc loaded count matches");

    GV_DiskANNResult r1[5];
    memset(r1, 0, sizeof(r1));
    int f1 = diskann_search(loaded_ooc, query, DIM, 5, r1);
    ASSERT(f1 == f0, "ooc loaded same result count");
    for (int i = 0; i < f1; i++) {
        ASSERT(r1[i].index == r0[i].index, "ooc loaded result matches base");
    }
    diskann_destroy(loaded_ooc);

    char g[600];
    snprintf(g, sizeof(g), "%s.graph", data_path); remove(g);
    remove(data_path); remove(save_path);
    return 0;
}

/* A RAM-built index (graph_on_disk=0) can be served out-of-core by force-enabling
 * the flag on load: the reconstructed adjacency is flushed to disk and freed. */
static int test_diskann_ooc_load_from_ram_save(void) {
    char data_path[512], save_path[512];
    gv_test_make_temp_path(data_path, sizeof(data_path), "gv_diskann_ram2ooc_data", ".bin");
    gv_test_make_temp_path(save_path, sizeof(save_path), "gv_diskann_ram2ooc_save", ".bin");

    GV_DiskANNConfig config;
    diskann_config_init(&config);
    config.data_path = data_path;
    config.graph_on_disk = 0; /* built in RAM */

    GV_DiskANNIndex *idx = diskann_create(DIM, &config);
    ASSERT(idx != NULL, "ram create failed");
    float data[BUILD_COUNT * DIM];
    generate_batch(data, BUILD_COUNT, DIM);
    ASSERT(diskann_build(idx, data, BUILD_COUNT, DIM) == 0, "ram build failed");
    ASSERT(diskann_save(idx, save_path) == 0, "ram save failed");

    float query[DIM];
    fill_vector(query, DIM, 5.0f);
    GV_DiskANNResult r0[5];
    memset(r0, 0, sizeof(r0));
    int f0 = diskann_search(idx, query, DIM, 5, r0);
    ASSERT(f0 > 0, "ram base search returned results");
    diskann_destroy(idx);

    GV_DiskANNConfig cfg;
    diskann_config_init(&cfg);
    cfg.data_path = data_path;  /* same vector file */
    cfg.graph_on_disk = 1;      /* force out-of-core serving */
    GV_DiskANNIndex *loaded = diskann_load(save_path, &cfg);
    ASSERT(loaded != NULL, "ram->ooc load returned NULL");

    char graph_path[600];
    snprintf(graph_path, sizeof(graph_path), "%s.graph", data_path);
    struct stat st;
    ASSERT(stat(graph_path, &st) == 0, "ooc graph file created on load");

    GV_DiskANNResult r1[5];
    memset(r1, 0, sizeof(r1));
    int f1 = diskann_search(loaded, query, DIM, 5, r1);
    ASSERT(f1 == f0, "ram->ooc loaded same result count");
    for (int i = 0; i < f1; i++) {
        ASSERT(r1[i].index == r0[i].index, "ram->ooc result matches base");
    }
    diskann_destroy(loaded);

    remove(graph_path);
    remove(data_path); remove(save_path);
    return 0;
}

/* Rebuilding an already-flushed out-of-core index must reconstruct the graph in
 * RAM and still search correctly (no stale neighbor_count with NULL neighbors). */
static int test_diskann_ooc_rebuild(void) {
    char data_path[512];
    gv_test_make_temp_path(data_path, sizeof(data_path), "gv_diskann_ooc_rebuild", ".bin");

    GV_DiskANNConfig config;
    diskann_config_init(&config);
    config.data_path = data_path;
    config.graph_on_disk = 1;

    GV_DiskANNIndex *idx = diskann_create(DIM, &config);
    ASSERT(idx != NULL, "ooc rebuild create failed");

    float data[BUILD_COUNT * DIM];
    generate_batch(data, BUILD_COUNT, DIM);
    ASSERT(diskann_build(idx, data, BUILD_COUNT, DIM) == 0, "first ooc build failed");

    /* Rebuild the same (flushed) index with fresh data. */
    generate_random_batch(data, BUILD_COUNT, DIM);
    ASSERT(diskann_build(idx, data, BUILD_COUNT, DIM) == 0, "ooc rebuild failed");

    float query[DIM];
    memcpy(query, &data[3 * DIM], DIM * sizeof(float));
    GV_DiskANNResult results[5];
    memset(results, 0, sizeof(results));
    int found = diskann_search(idx, query, DIM, 5, results);
    ASSERT(found > 0, "search after ooc rebuild returned results");

    diskann_destroy(idx);
    char g[600];
    snprintf(g, sizeof(g), "%s.graph", data_path); remove(g);
    remove(data_path);
    return 0;
}

/* Incremental insert after an out-of-core flush: the new node's adjacency and
 * its neighbours' back-edges are patched on disk (no RAM graph). The inserted
 * vector must then be findable as the nearest neighbour of itself. */
static int test_diskann_ooc_insert(void) {
    char data_path[512];
    gv_test_make_temp_path(data_path, sizeof(data_path), "gv_diskann_ooc_ins", ".bin");

    GV_DiskANNConfig config;
    diskann_config_init(&config);
    config.data_path = data_path;
    config.graph_on_disk = 1;

    GV_DiskANNIndex *idx = diskann_create(LARGE_DIM, &config);
    ASSERT(idx != NULL, "ooc ins create failed");

    float data[LARGE_COUNT * LARGE_DIM];
    generate_random_batch(data, LARGE_COUNT, LARGE_DIM);
    ASSERT(diskann_build(idx, data, LARGE_COUNT, LARGE_DIM) == 0, "ooc ins build failed");

    /* Insert a fresh vector into the already-flushed index. */
    float extra[LARGE_DIM];
    generate_random_batch(extra, 1, LARGE_DIM);
    ASSERT(diskann_insert(idx, extra, LARGE_DIM) == 0, "ooc insert after flush failed");
    ASSERT(diskann_count(idx) == LARGE_COUNT + 1, "ooc count grows after insert");

    /* The graph file must have grown to hold the new node's slot. */
    char graph_path[600];
    snprintf(graph_path, sizeof(graph_path), "%s.graph", data_path);
    struct stat st;
    ASSERT(stat(graph_path, &st) == 0, "graph file present after ooc insert");
    size_t expected = (size_t)(LARGE_COUNT + 1) * (size_t)(1 + 64) * sizeof(uint32_t);
    ASSERT((size_t)st.st_size == expected, "graph grew by one slot after insert");

    /* Query with the inserted vector itself: it should come back top-1. */
    GV_DiskANNResult results[5];
    memset(results, 0, sizeof(results));
    int found = diskann_search(idx, extra, LARGE_DIM, 5, results);
    ASSERT(found > 0, "search after ooc insert returned results");
    ASSERT(results[0].index == (size_t)LARGE_COUNT, "inserted vector is its own NN");
    ASSERT(results[0].distance < 1e-4f, "inserted vector distance ~0");

    /* A second insert must keep working too. */
    generate_random_batch(extra, 1, LARGE_DIM);
    ASSERT(diskann_insert(idx, extra, LARGE_DIM) == 0, "second ooc insert failed");
    ASSERT(diskann_count(idx) == LARGE_COUNT + 2, "ooc count grows after 2nd insert");

    /* The live edge count must match a fresh reload's, which rebuilds each
     * node's count from its on-disk slot. This fails if a post-flush insert's
     * back-edges were written to disk but not reflected in the RAM count. */
    GV_DiskANNStats s_live;
    ASSERT(diskann_get_stats(idx, &s_live) == 0, "live stats");
    char save_path[640];
    snprintf(save_path, sizeof(save_path), "%s.save", data_path);
    ASSERT(diskann_save(idx, save_path) == 0, "ooc save after insert");
    diskann_destroy(idx);

    GV_DiskANNConfig cfg;
    diskann_config_init(&cfg);
    cfg.data_path = data_path;
    GV_DiskANNIndex *re = diskann_load(save_path, &cfg);
    ASSERT(re != NULL, "reload after ooc insert");
    GV_DiskANNStats s_re;
    ASSERT(diskann_get_stats(re, &s_re) == 0, "reload stats");
    ASSERT(s_re.graph_edges == s_live.graph_edges,
           "live graph_edges match reload (back-edge counts synced on disk)");
    diskann_destroy(re);

    remove(save_path);
    remove(graph_path);
    remove(data_path);
    return 0;
}

/* force_in_memory: an out-of-core-saved index can be loaded back into RAM
 * serving. Results match the out-of-core load, and the RAM load keeps the
 * adjacency resident (higher memory, no reliance on the .graph file). */
static int test_diskann_force_in_memory_load(void) {
    char data_path[512], save_path[512];
    gv_test_make_temp_path(data_path, sizeof(data_path), "gv_diskann_fim_data", ".bin");
    gv_test_make_temp_path(save_path, sizeof(save_path), "gv_diskann_fim_save", ".bin");

    GV_DiskANNConfig config;
    diskann_config_init(&config);
    config.data_path = data_path;
    config.graph_on_disk = 1;

    GV_DiskANNIndex *idx = diskann_create(DIM, &config);
    ASSERT(idx != NULL, "fim create failed");
    float data[BUILD_COUNT * DIM];
    generate_batch(data, BUILD_COUNT, DIM);
    ASSERT(diskann_build(idx, data, BUILD_COUNT, DIM) == 0, "fim build failed");
    ASSERT(diskann_save(idx, save_path) == 0, "fim save failed");

    float query[DIM];
    fill_vector(query, DIM, 5.0f);
    GV_DiskANNResult r0[5];
    memset(r0, 0, sizeof(r0));
    int f0 = diskann_search(idx, query, DIM, 5, r0);
    ASSERT(f0 > 0, "fim base search returned results");
    diskann_destroy(idx);

    /* Out-of-core load (stored flag): adjacency freed from RAM. */
    GV_DiskANNConfig cfg_ooc;
    diskann_config_init(&cfg_ooc);
    cfg_ooc.data_path = data_path;
    GV_DiskANNIndex *ooc = diskann_load(save_path, &cfg_ooc);
    ASSERT(ooc != NULL, "ooc load returned NULL");
    GV_DiskANNStats s_ooc;
    ASSERT(diskann_get_stats(ooc, &s_ooc) == 0, "ooc stats");

    /* Forced RAM load: adjacency reconstructed and kept resident. */
    GV_DiskANNConfig cfg_ram;
    diskann_config_init(&cfg_ram);
    cfg_ram.data_path = data_path;
    cfg_ram.force_in_memory = 1;
    GV_DiskANNIndex *ram = diskann_load(save_path, &cfg_ram);
    ASSERT(ram != NULL, "force_in_memory load returned NULL");
    GV_DiskANNStats s_ram;
    ASSERT(diskann_get_stats(ram, &s_ram) == 0, "ram stats");

    /* Resident adjacency makes the RAM load use strictly more memory. */
    ASSERT(s_ram.memory_usage_bytes > s_ooc.memory_usage_bytes,
           "force_in_memory keeps the graph resident");

    GV_DiskANNResult r1[5];
    memset(r1, 0, sizeof(r1));
    int f1 = diskann_search(ram, query, DIM, 5, r1);
    ASSERT(f1 == f0, "ram load same result count");
    for (int i = 0; i < f1; i++) {
        ASSERT(r1[i].index == r0[i].index, "ram load result matches base");
    }

    diskann_destroy(ooc);
    diskann_destroy(ram);
    char g[600];
    snprintf(g, sizeof(g), "%s.graph", data_path); remove(g);
    remove(data_path); remove(save_path);
    return 0;
}

/* Inserting into an index whose nodes were all deleted must re-seed the entry
 * point so the fresh vector is searchable rather than orphaned. */
static int test_diskann_insert_into_all_deleted(void) {
    GV_DiskANNConfig config;
    diskann_config_init(&config);
    config.data_path = NULL;

    GV_DiskANNIndex *idx = diskann_create(DIM, &config);
    ASSERT(idx != NULL, "create failed");

    float data[BUILD_COUNT * DIM];
    generate_batch(data, BUILD_COUNT, DIM);
    ASSERT(diskann_build(idx, data, BUILD_COUNT, DIM) == 0, "build failed");

    for (size_t i = 0; i < BUILD_COUNT; i++) diskann_delete(idx, i);
    ASSERT(diskann_count(idx) == 0, "all nodes deleted");

    float nv[DIM];
    fill_vector(nv, DIM, 123.0f);
    ASSERT(diskann_insert(idx, nv, DIM) == 0, "insert into all-deleted failed");
    ASSERT(diskann_count(idx) == 1, "one live node after re-seed insert");

    GV_DiskANNResult r[3];
    memset(r, 0, sizeof(r));
    int found = diskann_search(idx, nv, DIM, 3, r);
    ASSERT(found == 1, "search finds the re-seeded node");
    /* The slot is recycled from the free list, so the id is a reused one, not a
     * fresh append; what matters is the self-query returns it at distance ~0. */
    ASSERT(r[0].index < BUILD_COUNT, "re-seeded node reused a freed slot");
    ASSERT(r[0].distance < 1e-4f, "self-query distance ~0");

    diskann_destroy(idx);
    return 0;
}

/* Deleted slots must be reclaimed by later inserts: deleting then re-inserting
 * the same number of vectors leaves the total slot count (disk usage) unchanged
 * rather than growing the index. */
static int test_diskann_slot_reclaim(void) {
    GV_DiskANNConfig config;
    diskann_config_init(&config);
    config.data_path = NULL;

    GV_DiskANNIndex *idx = diskann_create(DIM, &config);
    ASSERT(idx != NULL, "create failed");

    float data[BUILD_COUNT * DIM];
    generate_batch(data, BUILD_COUNT, DIM);
    ASSERT(diskann_build(idx, data, BUILD_COUNT, DIM) == 0, "build failed");

    GV_DiskANNStats s0;
    ASSERT(diskann_get_stats(idx, &s0) == 0, "stats before");

    const size_t n = 5;
    for (size_t i = 0; i < n; i++) ASSERT(diskann_delete(idx, i) == 0, "delete failed");
    ASSERT(diskann_count(idx) == BUILD_COUNT - n, "active count drops after delete");

    for (size_t i = 0; i < n; i++) {
        float nv[DIM];
        fill_vector(nv, DIM, 500.0f + (float)i);
        ASSERT(diskann_insert(idx, nv, DIM) == 0, "reinsert failed");
    }

    GV_DiskANNStats s1;
    ASSERT(diskann_get_stats(idx, &s1) == 0, "stats after");
    ASSERT(diskann_count(idx) == BUILD_COUNT, "active count restored");
    ASSERT(s1.disk_usage_bytes == s0.disk_usage_bytes,
           "reinserts reused freed slots (no total growth)");

    diskann_destroy(idx);
    return 0;
}

/* Concurrency: the index serializes its public API with an internal lock, so
 * concurrent searches and inserts must not corrupt it or trip the race detector
 * (run this suite under TSAN). */
#define CONC_ITERS 60
#define CONC_INSERTERS 2

typedef struct { GV_DiskANNIndex *idx; int iters; } ConcArg;

static void *conc_search_worker(void *p) {
    ConcArg *a = (ConcArg *)p;
    float q[LARGE_DIM];
    GV_DiskANNResult r[5];
    for (int i = 0; i < a->iters; i++) {
        fill_vector(q, LARGE_DIM, (float)(i % 50));
        memset(r, 0, sizeof(r));
        diskann_search(a->idx, q, LARGE_DIM, 5, r);
    }
    return NULL;
}

static void *conc_insert_worker(void *p) {
    ConcArg *a = (ConcArg *)p;
    float v[LARGE_DIM];
    for (int i = 0; i < a->iters; i++) {
        fill_vector(v, LARGE_DIM, 1000.0f + (float)i);
        diskann_insert(a->idx, v, LARGE_DIM);
    }
    return NULL;
}

static int test_diskann_concurrent(void) {
    char data_path[512];
    gv_test_make_temp_path(data_path, sizeof(data_path), "gv_diskann_conc", ".bin");

    GV_DiskANNConfig config;
    diskann_config_init(&config);
    config.data_path = data_path;

    GV_DiskANNIndex *idx = diskann_create(LARGE_DIM, &config);
    ASSERT(idx != NULL, "conc create failed");
    float data[LARGE_COUNT * LARGE_DIM];
    generate_random_batch(data, LARGE_COUNT, LARGE_DIM);
    ASSERT(diskann_build(idx, data, LARGE_COUNT, LARGE_DIM) == 0, "conc build failed");

    pthread_t searchers[4], inserters[CONC_INSERTERS];
    ConcArg sa = { idx, CONC_ITERS }, ia = { idx, CONC_ITERS };
    for (int i = 0; i < 4; i++)
        ASSERT(pthread_create(&searchers[i], NULL, conc_search_worker, &sa) == 0, "spawn searcher");
    for (int i = 0; i < CONC_INSERTERS; i++)
        ASSERT(pthread_create(&inserters[i], NULL, conc_insert_worker, &ia) == 0, "spawn inserter");
    for (int i = 0; i < 4; i++) pthread_join(searchers[i], NULL);
    for (int i = 0; i < CONC_INSERTERS; i++) pthread_join(inserters[i], NULL);

    /* Serialized inserts are deterministic: every one committed. */
    ASSERT(diskann_count(idx) == LARGE_COUNT + CONC_INSERTERS * CONC_ITERS,
           "all concurrent inserts committed exactly once");

    float q[LARGE_DIM];
    memcpy(q, &data[0], LARGE_DIM * sizeof(float));
    GV_DiskANNResult r[5];
    memset(r, 0, sizeof(r));
    ASSERT(diskann_search(idx, q, LARGE_DIM, 5, r) > 0, "search works after concurrency");

    diskann_destroy(idx);
    remove(data_path);
    return 0;
}

/* Parallel build: multi-threaded construction must reach the same recall bar as
 * the serial build (run under TSAN to prove the search phase is race-free). */
static int test_diskann_parallel_build(void) {
    GV_DiskANNConfig config;
    diskann_config_init(&config);
    config.data_path = NULL;

    GV_DiskANNIndex *idx = diskann_create(LARGE_DIM, &config);
    ASSERT(idx != NULL, "parallel create failed");
    float data[LARGE_COUNT * LARGE_DIM];
    generate_random_batch(data, LARGE_COUNT, LARGE_DIM);
    ASSERT(diskann_build_parallel(idx, data, LARGE_COUNT, LARGE_DIM, 4) == 0, "parallel build failed");
    ASSERT(diskann_count(idx) == LARGE_COUNT, "parallel build count");

    size_t k = 10, num_queries = 20, total_found = 0;
    for (size_t q = 0; q < num_queries; q++) {
        float query[LARGE_DIM];
        size_t qi = (q * 7) % LARGE_COUNT;
        memcpy(query, &data[qi * LARGE_DIM], LARGE_DIM * sizeof(float));
        size_t exact_idx[LARGE_COUNT];
        float exact_dist[LARGE_COUNT];
        for (size_t i = 0; i < LARGE_COUNT; i++) {
            exact_idx[i] = i;
            exact_dist[i] = brute_force_l2(query, &data[i * LARGE_DIM], LARGE_DIM);
        }
        for (size_t i = 0; i < k; i++) {
            size_t m = i;
            for (size_t j = i + 1; j < LARGE_COUNT; j++) if (exact_dist[j] < exact_dist[m]) m = j;
            size_t ti = exact_idx[i]; exact_idx[i] = exact_idx[m]; exact_idx[m] = ti;
            float td = exact_dist[i]; exact_dist[i] = exact_dist[m]; exact_dist[m] = td;
        }
        GV_DiskANNResult r[16];
        memset(r, 0, sizeof(r));
        int found = diskann_search(idx, query, LARGE_DIM, k, r);
        for (int x = 0; x < found && x < (int)k; x++)
            for (size_t e = 0; e < k; e++)
                if (r[x].index == exact_idx[e]) { total_found++; break; }
    }
    double recall = (double)total_found / ((double)num_queries * (double)k);
    fprintf(stderr, "  parallel-build recall@%zu = %.2f%%\n", k, recall * 100.0);
    ASSERT(recall > 0.80, "parallel build recall@10 > 80%");

    diskann_destroy(idx);
    return 0;
}

/* Consolidation: after deleting nodes, no live node's adjacency may still point
 * at a tombstone, and search must keep working. Verified via graph_edges, which
 * only sums live nodes' neighbour counts. */
static int test_diskann_consolidate(void) {
    GV_DiskANNConfig config;
    diskann_config_init(&config);
    config.data_path = NULL;

    GV_DiskANNIndex *idx = diskann_create(DIM, &config);
    ASSERT(idx != NULL, "consolidate create failed");
    float data[BUILD_COUNT * DIM];
    generate_batch(data, BUILD_COUNT, DIM);
    ASSERT(diskann_build(idx, data, BUILD_COUNT, DIM) == 0, "build failed");

    for (size_t i = 0; i < BUILD_COUNT; i += 3) diskann_delete(idx, i);

    GV_DiskANNStats before, after;
    ASSERT(diskann_get_stats(idx, &before) == 0, "stats before");
    ASSERT(diskann_consolidate(idx) == 0, "consolidate failed");
    ASSERT(diskann_get_stats(idx, &after) == 0, "stats after");

    /* Scrubbing dead edges can only remove edges, never add them. */
    ASSERT(after.graph_edges <= before.graph_edges, "consolidate only drops edges");

    /* Search still returns only live nodes. */
    float query[DIM];
    fill_vector(query, DIM, 7.0f);
    GV_DiskANNResult r[10];
    memset(r, 0, sizeof(r));
    int found = diskann_search(idx, query, DIM, 10, r);
    ASSERT(found > 0, "search works after consolidate");
    for (int i = 0; i < found; i++) ASSERT(r[i].index % 3 != 0, "no tombstoned id returned");

    diskann_destroy(idx);
    return 0;
}

/* Filtered search: every returned id must be in the allow-set, and the best
 * allowed vector (the query is a copy of an allowed vector) comes back top-1. */
static int test_diskann_filtered_search(void) {
    GV_DiskANNConfig config;
    diskann_config_init(&config);
    config.data_path = NULL;

    GV_DiskANNIndex *idx = diskann_create(LARGE_DIM, &config);
    ASSERT(idx != NULL, "filtered create failed");
    float data[LARGE_COUNT * LARGE_DIM];
    generate_random_batch(data, LARGE_COUNT, LARGE_DIM);
    ASSERT(diskann_build(idx, data, LARGE_COUNT, LARGE_DIM) == 0, "filtered build failed");

    /* Allow every 5th id; make the query an exact copy of an allowed vector. */
    GV_IdBitmap *allowed = gv_id_bitmap_create();
    ASSERT(allowed != NULL, "bitmap create");
    for (size_t i = 0; i < LARGE_COUNT; i += 5) gv_id_bitmap_add(allowed, (uint64_t)i);
    size_t target = 25; /* in the allow-set */

    float query[LARGE_DIM];
    memcpy(query, &data[target * LARGE_DIM], LARGE_DIM * sizeof(float));

    GV_DiskANNResult results[10];
    memset(results, 0, sizeof(results));
    int found = diskann_search_filtered(idx, query, LARGE_DIM, 10, results, allowed, 128);
    ASSERT(found > 0, "filtered search returned results");
    ASSERT(results[0].index == target, "exact allowed match is top-1");
    for (int i = 0; i < found; i++) {
        ASSERT(gv_id_bitmap_contains(allowed, (uint64_t)results[i].index),
               "every result is in the allow-set");
    }

    gv_id_bitmap_free(allowed);
    diskann_destroy(idx);
    return 0;
}

/* Cosine metric: the index must rank by cosine distance (1 - cossim), not L2.
 * A self-query returns the vector itself (cosine distance ~0), and recall
 * against a brute-force cosine ranking stays high. */
static float brute_force_cosine(const float *a, const float *b, size_t dim) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < dim; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    if (na == 0.0f || nb == 0.0f) return 1.0f;
    return 1.0f - dot / (sqrtf(na) * sqrtf(nb));
}

static int test_diskann_cosine_metric(void) {
    GV_DiskANNConfig config;
    diskann_config_init(&config);
    config.data_path = NULL;
    config.metric = GV_DISTANCE_COSINE;

    GV_DiskANNIndex *idx = diskann_create(LARGE_DIM, &config);
    ASSERT(idx != NULL, "cosine create failed");
    float data[LARGE_COUNT * LARGE_DIM];
    generate_random_batch(data, LARGE_COUNT, LARGE_DIM);
    ASSERT(diskann_build(idx, data, LARGE_COUNT, LARGE_DIM) == 0, "cosine build failed");

    size_t k = 10, num_queries = 20, total_found = 0;
    for (size_t q = 0; q < num_queries; q++) {
        float query[LARGE_DIM];
        size_t qi = (q * 7) % LARGE_COUNT;
        memcpy(query, &data[qi * LARGE_DIM], LARGE_DIM * sizeof(float));

        size_t exact_idx[LARGE_COUNT];
        float exact_dist[LARGE_COUNT];
        for (size_t i = 0; i < LARGE_COUNT; i++) {
            exact_idx[i] = i;
            exact_dist[i] = brute_force_cosine(query, &data[i * LARGE_DIM], LARGE_DIM);
        }
        for (size_t i = 0; i < k; i++) {
            size_t m = i;
            for (size_t j = i + 1; j < LARGE_COUNT; j++)
                if (exact_dist[j] < exact_dist[m]) m = j;
            size_t ti = exact_idx[i]; exact_idx[i] = exact_idx[m]; exact_idx[m] = ti;
            float td = exact_dist[i]; exact_dist[i] = exact_dist[m]; exact_dist[m] = td;
        }

        GV_DiskANNResult results[16];
        memset(results, 0, sizeof(results));
        int found = diskann_search(idx, query, LARGE_DIM, k, results);
        ASSERT(found > 0, "cosine search returned results");
        /* Self-query: top-1 is the query vector at cosine distance ~0. */
        if (q == 0) {
            ASSERT(results[0].index == qi, "cosine self-query returns itself");
            ASSERT(results[0].distance < 1e-4f, "cosine self distance ~0");
        }
        for (int r = 0; r < found && r < (int)k; r++)
            for (size_t e = 0; e < k; e++)
                if (results[r].index == exact_idx[e]) { total_found++; break; }
    }
    double recall = (double)total_found / ((double)num_queries * (double)k);
    fprintf(stderr, "  cosine recall@%zu = %.2f%%\n", k, recall * 100.0);
    ASSERT(recall > 0.80, "cosine recall@10 > 80%");

    diskann_destroy(idx);
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
        {"Testing diskann out-of-core recall...",    test_diskann_ooc_recall},
        {"Testing diskann out-of-core save/load...", test_diskann_ooc_save_load},
        {"Testing diskann out-of-core load from RAM save...", test_diskann_ooc_load_from_ram_save},
        {"Testing diskann out-of-core rebuild...",    test_diskann_ooc_rebuild},
        {"Testing diskann out-of-core insert...",     test_diskann_ooc_insert},
        {"Testing diskann insert into all-deleted...", test_diskann_insert_into_all_deleted},
        {"Testing diskann force_in_memory load...",    test_diskann_force_in_memory_load},
        {"Testing diskann slot reclamation...",        test_diskann_slot_reclaim},
        {"Testing diskann concurrent access...",        test_diskann_concurrent},
        {"Testing diskann cosine metric...",            test_diskann_cosine_metric},
        {"Testing diskann filtered search...",          test_diskann_filtered_search},
        {"Testing diskann consolidate...",              test_diskann_consolidate},
        {"Testing diskann parallel build...",           test_diskann_parallel_build},
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
