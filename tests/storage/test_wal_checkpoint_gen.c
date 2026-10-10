/*
 * Regression test for the checkpoint-generation fix: a crash between the
 * snapshot publish and the WAL truncate must NOT duplicate the snapshotted
 * vectors on recovery. Pre-fix, db_open loaded the snapshot (N) and then
 * replayed the still-intact WAL (the same N inserts) -> 2N. The checkpoint
 * generation (snapshot header v7 + WAL .ckptgen sidecar) lets recovery detect a
 * WAL that predates the snapshot and skip it.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gigavector.h"
#include "../test_tmp.h"

#define ASSERT(cond, msg)                       \
    do {                                        \
        if (!(cond)) {                          \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1;                          \
        }                                       \
    } while (0)

#define DIM 8

static void mkvec(float *v, int seed) {
    for (int i = 0; i < DIM; i++) v[i] = (float)(seed * 100 + i);
}

static int copy_file(const char *src, const char *dst) {
    FILE *a = fopen(src, "rb");
    if (!a) return -1;
    FILE *b = fopen(dst, "wb");
    if (!b) { fclose(a); return -1; }
    char buf[4096];
    size_t n;
    int rc = 0;
    while ((n = fread(buf, 1, sizeof(buf), a)) > 0) {
        if (fwrite(buf, 1, n, b) != n) { rc = -1; break; }
    }
    fclose(a);
    if (fclose(b) != 0) rc = -1;
    return rc;
}

/* Normal checkpoint round-trip + inserts after the checkpoint. */
static int test_normal_roundtrip(void) {
    char db_path[256], wal_path[256];
    gv_test_make_temp_path(db_path, sizeof(db_path), "gv_ckpt_db", ".gv");
    gv_test_make_temp_path(wal_path, sizeof(wal_path), "gv_ckpt_wal", ".wal");
    remove(db_path); remove(wal_path);

    GV_Database *db = db_open(db_path, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "open");
    ASSERT(db_set_wal(db, wal_path) == 0, "set wal");
    float v[DIM];
    for (int i = 0; i < 10; i++) { mkvec(v, i); (void)db_add_vector(db, v, DIM); }
    ASSERT(db_save(db, NULL) == 0, "checkpoint");
    db_close(db);

    /* Clean checkpoint round-trip: snapshot(10) + a WAL truncated for the same
     * generation => exactly 10 on reopen (no loss, no duplication). */
    GV_Database *db2 = db_open(db_path, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db2 != NULL, "reopen");
    ASSERT(database_count(db2) == 10, "clean checkpoint round-trips to exactly 10 (no duplication)");
    db_close(db2);
    remove(db_path); remove(wal_path);
    { char s[300]; snprintf(s, sizeof(s), "%s.ckptgen", wal_path); remove(s); }
    return 0;
}

/* Crash between snapshot publish and WAL truncate: reconstruct that on-disk
 * state (snapshot present, WAL still full, sidecar absent) and confirm recovery
 * does NOT duplicate. */
static int test_crash_before_truncate(void) {
    char db_path[256], wal_path[256], wal_bak[300], sidecar[300];
    gv_test_make_temp_path(db_path, sizeof(db_path), "gv_ckpt2_db", ".gv");
    gv_test_make_temp_path(wal_path, sizeof(wal_path), "gv_ckpt2_wal", ".wal");
    remove(db_path); remove(wal_path);
    snprintf(wal_bak, sizeof(wal_bak), "%s.bak", wal_path);
    snprintf(sidecar, sizeof(sidecar), "%s.ckptgen", wal_path);

    GV_Database *db = db_open(db_path, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "open");
    ASSERT(db_set_wal(db, wal_path) == 0, "set wal");
    float v[DIM];
    for (int i = 0; i < 10; i++) { mkvec(v, i); (void)db_add_vector(db, v, DIM); }

    /* The WAL now holds the 10 inserts. Snapshot the file so we can restore it
     * to its pre-checkpoint (still-full) state after db_save truncates it. */
    ASSERT(copy_file(wal_path, wal_bak) == 0, "backup WAL");
    ASSERT(db_save(db, NULL) == 0, "checkpoint (snapshot + truncate + sidecar)");
    db_close(db);

    /* Simulate the crash window: snapshot is durable (gen 1), but the WAL was
     * NOT truncated (restore the full copy) and the sidecar was NOT written
     * (remove it -> generation reads as 0). */
    ASSERT(copy_file(wal_bak, wal_path) == 0, "restore pre-truncate WAL");
    remove(sidecar);

    GV_Database *db2 = db_open(db_path, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db2 != NULL, "reopen after simulated crash");
    ASSERT(database_count(db2) == 10, "stale WAL must NOT be replayed on top of the snapshot (no duplication)");
    db_close(db2);

    remove(db_path); remove(wal_path); remove(wal_bak); remove(sidecar);
    return 0;
}

int main(void) {
    int rc = 0;
    rc |= test_normal_roundtrip();
    rc |= test_crash_before_truncate();
    if (rc == 0) printf("test_wal_checkpoint_gen: all passed\n");
    return rc;
}
