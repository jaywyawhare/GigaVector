/**
 * test_diskann_disk.c — disk-backed DiskANN build/save/reload/search.
 *
 * Unlike test_diskann.c (which only uses in-memory data_path=NULL), this test
 * sets config.data_path to a real on-disk file so the on-disk vector storage
 * path is exercised, then saves the index, reloads it from the snapshot, and
 * searches the reloaded index. Uses tests/test_tmp.h for all temp paths and
 * cleans up fully.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "index/diskann.h"
#include "../test_tmp.h"

#define ASSERT(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); return -1; } \
} while (0)

#define DIM 8
#define BUILD_COUNT 128

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

static int test_diskann_disk_roundtrip(void) {
    char dir[512];
    ASSERT(gv_test_mkdtemp(dir, sizeof(dir), "gv_diskann_disk") == 0, "mkdtemp");

    /* On-disk vector storage lives at dir/vectors.bin. */
    char data_path[640];
    int n = snprintf(data_path, sizeof(data_path), "%s/vectors.bin", dir);
    ASSERT(n > 0 && (size_t)n < sizeof(data_path), "data_path fits");

    char snap_path[512];
    int fd = gv_test_mkstemp(snap_path, sizeof(snap_path), "gv_diskann_snap");
    ASSERT(fd >= 0, "mkstemp snap");
    close(fd);
    unlink(snap_path); /* diskann_save creates it fresh */

    GV_DiskANNConfig config;
    diskann_config_init(&config);
    config.data_path = data_path;
    config.cache_size_mb = 4;

    GV_DiskANNIndex *idx = diskann_create(DIM, &config);
    ASSERT(idx != NULL, "create (disk-backed)");

    float data[BUILD_COUNT * DIM];
    generate_batch(data, BUILD_COUNT, DIM);
    ASSERT(diskann_build(idx, data, BUILD_COUNT, DIM) == 0, "build");
    ASSERT(diskann_count(idx) == BUILD_COUNT, "count after build");

    /* Baseline search on the freshly built (disk-backed) index. */
    float query[DIM];
    fill_vector(query, DIM, 3.0f);
    GV_DiskANNResult before[5];
    memset(before, 0, sizeof(before));
    int nbefore = diskann_search(idx, query, DIM, 5, before);
    ASSERT(nbefore > 0, "search before save");

    /* Save to snapshot and destroy. */
    ASSERT(diskann_save(idx, snap_path) == 0, "save");
    diskann_destroy(idx);

    /* Reload from the snapshot with the same on-disk config. */
    GV_DiskANNConfig lconfig;
    diskann_config_init(&lconfig);
    lconfig.data_path = data_path;
    lconfig.cache_size_mb = 4;
    GV_DiskANNIndex *loaded = diskann_load(snap_path, &lconfig);
    ASSERT(loaded != NULL, "load");
    ASSERT(diskann_count(loaded) == BUILD_COUNT, "count after reload");

    /* Search the reloaded index — must return results and be consistent with
     * the pre-save top-1 (deterministic build/search). */
    GV_DiskANNResult after[5];
    memset(after, 0, sizeof(after));
    int nafter = diskann_search(loaded, query, DIM, 5, after);
    ASSERT(nafter > 0, "search after reload");
    ASSERT(after[0].distance >= 0.0f, "distance non-negative");
    ASSERT(before[0].index == after[0].index, "top-1 stable across save/reload");

    diskann_destroy(loaded);

    /* Cleanup: snapshot + on-disk data dir. */
    unlink(snap_path);
    gv_test_rmrf(dir);
    return 0;
}

int main(void) {
    int rc = 0;
    rc |= test_diskann_disk_roundtrip();
    if (rc == 0) printf("All DiskANN disk-backed tests PASSED.\n");
    return rc != 0;
}
