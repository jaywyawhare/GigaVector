#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_snapshot.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

/* ---------- test functions ---------- */

static int test_manager_create_destroy(void) {
    GV_SnapshotManager *mgr = gv_snapshot_manager_create(10);
    ASSERT(mgr != NULL, "gv_snapshot_manager_create returned NULL");
    gv_snapshot_manager_destroy(mgr);
    return 0;
}

static int test_create_and_open_snapshot(void) {
    GV_SnapshotManager *mgr = gv_snapshot_manager_create(10);
    ASSERT(mgr != NULL, "create manager");

    float vectors[8] = {1.0f, 2.0f, 3.0f, 4.0f,
                         5.0f, 6.0f, 7.0f, 8.0f};
    uint64_t sid = gv_snapshot_create(mgr, 2, vectors, 4, "test-snap-1");
    ASSERT(sid > 0, "snapshot_create should return nonzero id");

    GV_Snapshot *snap = gv_snapshot_open(mgr, sid);
    ASSERT(snap != NULL, "snapshot_open should return non-NULL");
    ASSERT(gv_snapshot_count(snap) == 2, "snapshot count should be 2");
    ASSERT(gv_snapshot_dimension(snap) == 4, "snapshot dimension should be 4");

    gv_snapshot_close(snap);
    gv_snapshot_manager_destroy(mgr);
    return 0;
}

static int test_snapshot_get_vector(void) {
    GV_SnapshotManager *mgr = gv_snapshot_manager_create(10);
    ASSERT(mgr != NULL, "create manager");

    float vectors[8] = {10.0f, 20.0f, 30.0f, 40.0f,
                         50.0f, 60.0f, 70.0f, 80.0f};
    uint64_t sid = gv_snapshot_create(mgr, 2, vectors, 4, "get-vec-test");
    GV_Snapshot *snap = gv_snapshot_open(mgr, sid);
    ASSERT(snap != NULL, "open snapshot");

    const float *v0 = gv_snapshot_get_vector(snap, 0);
    ASSERT(v0 != NULL, "get_vector(0) should not be NULL");
    ASSERT(v0[0] == 10.0f && v0[3] == 40.0f, "vector 0 data should match");

    const float *v1 = gv_snapshot_get_vector(snap, 1);
    ASSERT(v1 != NULL, "get_vector(1) should not be NULL");
    ASSERT(v1[0] == 50.0f && v1[3] == 80.0f, "vector 1 data should match");

    /* Out-of-bounds should return NULL */
    const float *v2 = gv_snapshot_get_vector(snap, 2);
    ASSERT(v2 == NULL, "get_vector out-of-bounds should return NULL");

    gv_snapshot_close(snap);
    gv_snapshot_manager_destroy(mgr);
    return 0;
}

static int test_snapshot_list(void) {
    GV_SnapshotManager *mgr = gv_snapshot_manager_create(10);
    ASSERT(mgr != NULL, "create manager");

    float v1[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float v2[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    gv_snapshot_create(mgr, 1, v1, 4, "snap-a");
    gv_snapshot_create(mgr, 1, v2, 4, "snap-b");

    GV_SnapshotInfo infos[10];
    int count = gv_snapshot_list(mgr, infos, 10);
    ASSERT(count == 2, "should list 2 snapshots");
    ASSERT(infos[0].vector_count == 1, "first snapshot vector_count == 1");
    ASSERT(infos[1].vector_count == 1, "second snapshot vector_count == 1");

    gv_snapshot_manager_destroy(mgr);
    return 0;
}

static int test_snapshot_delete(void) {
    GV_SnapshotManager *mgr = gv_snapshot_manager_create(10);
    ASSERT(mgr != NULL, "create manager");

    float v[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    uint64_t sid = gv_snapshot_create(mgr, 1, v, 4, "to-delete");
    ASSERT(sid > 0, "create snapshot");

    int rc = gv_snapshot_delete(mgr, sid);
    ASSERT(rc == 0, "delete should succeed");

    /* Opening deleted snapshot should return NULL */
    GV_Snapshot *snap = gv_snapshot_open(mgr, sid);
    ASSERT(snap == NULL, "opening deleted snapshot should return NULL");

    gv_snapshot_manager_destroy(mgr);
    return 0;
}

static int test_snapshot_save_load(void) {
    GV_SnapshotManager *mgr = gv_snapshot_manager_create(10);
    ASSERT(mgr != NULL, "create manager");

    float v[4] = {1.5f, 2.5f, 3.5f, 4.5f};
    uint64_t sid = gv_snapshot_create(mgr, 1, v, 4, "persist-test");
    ASSERT(sid > 0, "create snapshot");

    FILE *tmp = tmpfile();
    ASSERT(tmp != NULL, "tmpfile() failed");

    int rc = gv_snapshot_save(mgr, tmp);
    ASSERT(rc == 0, "save should succeed");

    rewind(tmp);

    GV_SnapshotManager *loaded = NULL;
    rc = gv_snapshot_load(&loaded, tmp);
    ASSERT(rc == 0, "load should succeed");
    ASSERT(loaded != NULL, "loaded manager should be non-NULL");

    /* Verify loaded snapshot data */
    GV_Snapshot *snap = gv_snapshot_open(loaded, sid);
    ASSERT(snap != NULL, "open loaded snapshot");
    ASSERT(gv_snapshot_count(snap) == 1, "loaded snapshot count == 1");
    const float *vl = gv_snapshot_get_vector(snap, 0);
    ASSERT(vl != NULL, "loaded vector not NULL");
    ASSERT(vl[0] == 1.5f && vl[3] == 4.5f, "loaded data matches");

    gv_snapshot_close(snap);
    gv_snapshot_manager_destroy(loaded);
    gv_snapshot_manager_destroy(mgr);
    fclose(tmp);
    return 0;
}

static int test_snapshot_empty(void) {
    GV_SnapshotManager *mgr = gv_snapshot_manager_create(10);
    ASSERT(mgr != NULL, "create manager");

    /* Snapshot with zero vectors */
    uint64_t sid = gv_snapshot_create(mgr, 0, NULL, 4, "empty");
    ASSERT(sid > 0, "empty snapshot should get valid id");

    GV_Snapshot *snap = gv_snapshot_open(mgr, sid);
    ASSERT(snap != NULL, "open empty snapshot");
    ASSERT(gv_snapshot_count(snap) == 0, "empty snapshot count == 0");

    gv_snapshot_close(snap);
    gv_snapshot_manager_destroy(mgr);
    return 0;
}

/* ---------- harness ---------- */

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing manager create/destroy...", test_manager_create_destroy},
        {"Testing create and open snapshot...", test_create_and_open_snapshot},
        {"Testing snapshot get vector...", test_snapshot_get_vector},
        {"Testing snapshot list...", test_snapshot_list},
        {"Testing snapshot delete...", test_snapshot_delete},
        {"Testing snapshot save/load...", test_snapshot_save_load},
        {"Testing empty snapshot...", test_snapshot_empty},
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
