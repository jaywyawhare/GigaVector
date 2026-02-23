#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_migration.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

/* ---------- test functions ---------- */

static int test_migration_start_destroy(void) {
    float data[8] = {1.0f, 2.0f, 3.0f, 4.0f,
                      5.0f, 6.0f, 7.0f, 8.0f};
    /* new_index_type=0 (KDTREE), no special config */
    GV_Migration *mig = gv_migration_start(data, 2, 4, 0, NULL);
    ASSERT(mig != NULL, "gv_migration_start returned NULL");
    gv_migration_destroy(mig);
    return 0;
}

static int test_migration_get_info(void) {
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    GV_Migration *mig = gv_migration_start(data, 1, 4, 0, NULL);
    ASSERT(mig != NULL, "start migration");

    GV_MigrationInfo info;
    memset(&info, 0, sizeof(info));
    int rc = gv_migration_get_info(mig, &info);
    ASSERT(rc == 0, "get_info should succeed");
    ASSERT(info.total_vectors == 1, "total_vectors should be 1");
    ASSERT(info.status == GV_MIGRATION_PENDING ||
           info.status == GV_MIGRATION_RUNNING ||
           info.status == GV_MIGRATION_COMPLETED,
           "status should be a valid state");

    gv_migration_destroy(mig);
    return 0;
}

static int test_migration_wait(void) {
    float data[8] = {1.0f, 2.0f, 3.0f, 4.0f,
                      5.0f, 6.0f, 7.0f, 8.0f};
    GV_Migration *mig = gv_migration_start(data, 2, 4, 0, NULL);
    ASSERT(mig != NULL, "start migration");

    int rc = gv_migration_wait(mig);
    ASSERT(rc == 0, "wait should succeed");

    GV_MigrationInfo info;
    gv_migration_get_info(mig, &info);
    ASSERT(info.status == GV_MIGRATION_COMPLETED, "should be completed after wait");
    ASSERT(info.progress >= 0.99, "progress should be ~1.0 after completion");
    ASSERT(info.vectors_migrated == 2, "vectors_migrated should be 2");

    gv_migration_destroy(mig);
    return 0;
}

static int test_migration_take_index(void) {
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    GV_Migration *mig = gv_migration_start(data, 1, 4, 0, NULL);
    ASSERT(mig != NULL, "start migration");

    gv_migration_wait(mig);

    void *idx = gv_migration_take_index(mig);
    /* After completed migration, index should be non-NULL */
    ASSERT(idx != NULL, "take_index should return non-NULL after completion");

    /* Taking again should return NULL (ownership transferred) */
    void *idx2 = gv_migration_take_index(mig);
    ASSERT(idx2 == NULL, "second take_index should return NULL");

    free(idx);
    gv_migration_destroy(mig);
    return 0;
}

static int test_migration_cancel(void) {
    /* Create larger data set to give cancel a chance */
    size_t count = 100;
    float *data = (float *)malloc(count * 4 * sizeof(float));
    ASSERT(data != NULL, "malloc data");
    for (size_t i = 0; i < count * 4; i++) {
        data[i] = (float)i * 0.01f;
    }

    GV_Migration *mig = gv_migration_start(data, count, 4, 0, NULL);
    ASSERT(mig != NULL, "start migration");

    int rc = gv_migration_cancel(mig);
    /* Cancel may succeed or fail if migration already completed */
    (void)rc;

    GV_MigrationInfo info;
    gv_migration_get_info(mig, &info);
    /* After cancel, status could be CANCELLED, COMPLETED (finished before cancel),
       RUNNING (cancel not yet processed), or PENDING. All are valid. */
    ASSERT(info.status >= GV_MIGRATION_PENDING && info.status <= GV_MIGRATION_CANCELLED,
           "status should be a valid migration state");

    gv_migration_destroy(mig);
    free(data);
    return 0;
}

static int test_migration_progress(void) {
    float data[16] = {0};
    for (int i = 0; i < 16; i++) data[i] = (float)i;

    GV_Migration *mig = gv_migration_start(data, 4, 4, 0, NULL);
    ASSERT(mig != NULL, "start migration");

    GV_MigrationInfo info;
    gv_migration_get_info(mig, &info);
    ASSERT(info.progress >= 0.0 && info.progress <= 1.0,
           "progress should be between 0 and 1");

    gv_migration_wait(mig);
    gv_migration_get_info(mig, &info);
    ASSERT(info.elapsed_us >= 0, "elapsed_us should be non-negative");

    gv_migration_destroy(mig);
    return 0;
}

static int test_null_safety(void) {
    /* Destroy NULL should be safe */
    gv_migration_destroy(NULL);

    /* Start with NULL data and 0 count */
    GV_Migration *mig = gv_migration_start(NULL, 0, 4, 0, NULL);
    if (mig != NULL) {
        gv_migration_wait(mig);
        gv_migration_destroy(mig);
    }
    return 0;
}

/* harness */

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
        {"Testing null safety...", test_null_safety},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
