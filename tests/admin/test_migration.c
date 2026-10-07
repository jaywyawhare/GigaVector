#include <stdio.h>
#include "core/memory.h"
#include <stdlib.h>
#include <string.h>
#include "admin/migration.h"
#include "storage/soa_storage.h"
#include "index/sparse_index.h"
#include "index/ivfdisk.h"
#include "../test_tmp.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

/* MIG_INDEX_* type tags (mirrors the GV_IndexType enum values). */
#define T_SPARSE  3
#define T_IVFDISK 11

static int test_migration_start_destroy(void) {
    float data[8] = {1.0f, 2.0f, 3.0f, 4.0f,
                      5.0f, 6.0f, 7.0f, 8.0f};
    GV_Migration *mig = migration_start(data, 2, 4, 0, NULL);
    ASSERT(mig != NULL, "migration_start returned NULL");
    migration_destroy(mig);
    return 0;
}

static int test_migration_get_info(void) {
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    GV_Migration *mig = migration_start(data, 1, 4, 0, NULL);
    ASSERT(mig != NULL, "start migration");

    GV_MigrationInfo info;
    memset(&info, 0, sizeof(info));
    int rc = migration_get_info(mig, &info);
    ASSERT(rc == 0, "get_info should succeed");
    ASSERT(info.total_vectors == 1, "total_vectors should be 1");
    ASSERT(info.status == GV_MIGRATION_PENDING ||
           info.status == GV_MIGRATION_RUNNING ||
           info.status == GV_MIGRATION_COMPLETED,
           "status should be a valid state");

    migration_destroy(mig);
    return 0;
}

static int test_migration_wait(void) {
    float data[8] = {1.0f, 2.0f, 3.0f, 4.0f,
                      5.0f, 6.0f, 7.0f, 8.0f};
    GV_Migration *mig = migration_start(data, 2, 4, 0, NULL);
    ASSERT(mig != NULL, "start migration");

    int rc = migration_wait(mig);
    ASSERT(rc == 0, "wait should succeed");

    GV_MigrationInfo info;
    migration_get_info(mig, &info);
    ASSERT(info.status == GV_MIGRATION_COMPLETED, "should be completed after wait");
    ASSERT(info.progress >= 0.99, "progress should be ~1.0 after completion");
    ASSERT(info.vectors_migrated == 2, "vectors_migrated should be 2");

    migration_destroy(mig);
    return 0;
}

static int test_migration_take_index(void) {
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    GV_Migration *mig = migration_start(data, 1, 4, 0, NULL);
    ASSERT(mig != NULL, "start migration");

    migration_wait(mig);

    void *idx = migration_take_index(mig);
    ASSERT(idx != NULL, "take_index should return non-NULL after completion");

    /* Taking again should return NULL (ownership transferred) */
    void *idx2 = migration_take_index(mig);
    ASSERT(idx2 == NULL, "second take_index should return NULL");

    /* type 0 == MIG_INDEX_KDTREE: the taken index is a GV_SoAStorage that owns
     * internal arrays, so it must be released with soa_storage_destroy(). */
    soa_storage_destroy((GV_SoAStorage *)idx);
    migration_destroy(mig);
    return 0;
}

static int test_migration_cancel(void) {
    /* Create larger data set to give cancel a chance */
    size_t count = 100;
    float *data = (float *)gv_alloc(count * 4 * sizeof(float));
    ASSERT(data != NULL, "gv_alloc data");
    for (size_t i = 0; i < count * 4; i++) {
        data[i] = (float)i * 0.01f;
    }

    GV_Migration *mig = migration_start(data, count, 4, 0, NULL);
    ASSERT(mig != NULL, "start migration");

    int rc = migration_cancel(mig);
    /* Cancel may succeed or fail if migration already completed */
    (void)rc;

    GV_MigrationInfo info;
    migration_get_info(mig, &info);
    /* After cancel, status could be CANCELLED, COMPLETED (finished before cancel),
       RUNNING (cancel not yet processed), or PENDING. All are valid. */
    ASSERT(info.status >= GV_MIGRATION_PENDING && info.status <= GV_MIGRATION_CANCELLED,
           "status should be a valid migration state");

    migration_destroy(mig);
    gv_free(data);
    return 0;
}

static int test_migration_progress(void) {
    float data[16] = {0};
    for (int i = 0; i < 16; i++) data[i] = (float)i;

    GV_Migration *mig = migration_start(data, 4, 4, 0, NULL);
    ASSERT(mig != NULL, "start migration");

    GV_MigrationInfo info;
    migration_get_info(mig, &info);
    ASSERT(info.progress >= 0.0 && info.progress <= 1.0,
           "progress should be between 0 and 1");

    migration_wait(mig);
    migration_get_info(mig, &info);
    ASSERT(info.elapsed_us > 0, "elapsed_us should be positive");

    migration_destroy(mig);
    return 0;
}

/* Migration to a SPARSE index: dense vectors convert to their non-zero
 * (index, value) components and the taken index is a GV_SparseIndex. */
static int test_migration_to_sparse(void) {
    const size_t dim = 4, n = 6;
    float data[24] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 2.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 3.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 4.0f,
        0.0f, 5.0f, 6.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, /* all-zero -> empty sparse vector */
    };
    GV_Migration *mig = migration_start(data, n, dim, T_SPARSE, NULL);
    ASSERT(mig != NULL, "start SPARSE migration");
    ASSERT(migration_wait(mig) == 0, "SPARSE migration wait succeeds");

    GV_MigrationInfo info;
    migration_get_info(mig, &info);
    ASSERT(info.status == GV_MIGRATION_COMPLETED, "SPARSE migration completes");
    ASSERT(info.vectors_migrated == n, "all vectors migrated to SPARSE");

    GV_SparseIndex *idx = (GV_SparseIndex *)migration_take_index(mig);
    ASSERT(idx != NULL, "SPARSE index produced");
    sparse_index_destroy(idx);
    migration_destroy(mig);
    return 0;
}

/* Migration to an IVFDISK index: on-disk index built under a temp data_dir
 * passed via the config. The taken index is a GV_IVFDiskIndex. */
static int test_migration_to_ivfdisk(void) {
    const size_t dim = 4, n = 16;
    float data[64];
    for (size_t i = 0; i < n; i++)
        for (size_t d = 0; d < dim; d++)
            data[i * dim + d] = (float)((i * 3u + d) % 7u);

    char dir[512];
    gv_test_make_temp_path(dir, sizeof(dir), "ivfdiskmig", "");

    GV_IVFDiskConfig cfg;
    ivfdisk_config_init(&cfg);
    cfg.nlist = 4;
    cfg.nprobe = 2;
    cfg.data_dir = dir;

    GV_Migration *mig = migration_start(data, n, dim, T_IVFDISK, &cfg);
    ASSERT(mig != NULL, "start IVFDISK migration");
    ASSERT(migration_wait(mig) == 0, "IVFDISK migration wait succeeds");

    GV_MigrationInfo info;
    migration_get_info(mig, &info);
    ASSERT(info.status == GV_MIGRATION_COMPLETED, "IVFDISK migration completes");
    ASSERT(info.vectors_migrated == n, "all vectors migrated to IVFDISK");

    GV_IVFDiskIndex *idx = (GV_IVFDiskIndex *)migration_take_index(mig);
    ASSERT(idx != NULL, "IVFDISK index produced");
    ASSERT(ivfdisk_count(idx) == n, "IVFDISK holds all vectors");
    ivfdisk_destroy(idx);
    migration_destroy(mig);
    return 0;
}

static int test_null_safety(void) {
    migration_destroy(NULL);

    GV_Migration *mig = migration_start(NULL, 0, 4, 0, NULL);
    if (mig != NULL) {
        migration_wait(mig);
        migration_destroy(mig);
    }
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing migration start/destroy...", test_migration_start_destroy},
        {"Testing migration get info...", test_migration_get_info},
        {"Testing migration wait...", test_migration_wait},
        {"Testing migration take index...", test_migration_take_index},
        {"Testing migration cancel...", test_migration_cancel},
        {"Testing migration progress...", test_migration_progress},
        {"Testing migration to SPARSE...", test_migration_to_sparse},
        {"Testing migration to IVFDISK...", test_migration_to_ivfdisk},
        {"Testing null safety...", test_null_safety},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
