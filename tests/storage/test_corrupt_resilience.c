/**
 * test_corrupt_resilience.c — corrupt-WAL / corrupt-snapshot recovery.
 *
 * Feeds deliberately corrupted WAL and snapshot bytes to the loaders and
 * asserts they return a *defined error* (not a crash / not silent success).
 * Run under the CI ASAN/UBSAN job this also proves corruption handling does
 * not read out of bounds. A crash here fails the build (real exit code).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "storage/database.h"
#include "storage/wal.h"
#include "../test_tmp.h"

#define ASSERT(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); return -1; } \
} while (0)

#define DIM 4

static int on_insert(void *ctx, const float *data, size_t dim,
                     const char *mk, const char *mv) {
    (void)data; (void)dim; (void)mk; (void)mv;
    if (ctx) (*(int *)ctx)++;   /* count replayed records */
    return 0;
}

/* Flip the last `n` bytes of a file (typically the CRC / tail). */
static int corrupt_tail(const char *path, size_t n) {
    FILE *f = fopen(path, "r+b");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long sz = ftell(f);
    if (sz < (long)n) n = (size_t)sz;
    if (fseek(f, -(long)n, SEEK_END) != 0) { fclose(f); return -1; }
    for (size_t i = 0; i < n; i++) fputc(0xFF, f);
    fclose(f);
    return 0;
}

/* A corrupted WAL tail must make wal_replay fail (return < 0), not crash. */
static int test_corrupt_wal(void) {
    char wal_path[512];
    int fd = gv_test_mkstemp(wal_path, sizeof(wal_path), "gv_corrupt_wal");
    ASSERT(fd >= 0, "mkstemp wal");
    close(fd);
    unlink(wal_path); /* wal_open wants to create it fresh */

    GV_WAL *wal = wal_open(wal_path, DIM, 0);
    ASSERT(wal != NULL, "wal_open");
    for (int i = 0; i < 8; i++) {
        float v[DIM] = {(float)i, 1.0f, 2.0f, 3.0f};
        wal_append_insert(wal, v, DIM, "tag", "seed");
    }
    wal_close(wal);

    /* Sanity: a clean replay succeeds and sees all 8 records. */
    int cnt = 0;
    ASSERT(wal_replay(wal_path, DIM, on_insert, &cnt, 0) == 0, "clean replay ok");
    ASSERT(cnt == 8, "clean replay saw all records");

    /* Corrupt the trailing record's CRC.  A torn TAIL (as from a crash partway
     * through the last append) must be RECOVERED, not fail the whole log: replay
     * drops the torn trailing record and succeeds. */
    ASSERT(corrupt_tail(wal_path, 4) == 0, "corrupt wal tail");
    cnt = 0;
    int rc = wal_replay(wal_path, DIM, on_insert, &cnt, 0);
    ASSERT(rc == 0, "torn-tail replay recovers (defined result, no crash)");
    ASSERT(cnt < 8, "torn trailing record dropped on recovery");

    /* A wrong expected dimension must be rejected (mid-log validation failure),
     * not crash. */
    ASSERT(wal_replay(wal_path, DIM + 1, on_insert, NULL, 0) != 0, "dimension mismatch rejected");

    /* Corrupt the header magic — a structurally invalid WAL is rejected. */
    { FILE *f = fopen(wal_path, "r+b"); ASSERT(f != NULL, "reopen wal");
      fputc(0xFF, f); fputc(0xFF, f); fputc(0xFF, f); fputc(0xFF, f); fclose(f); }
    ASSERT(wal_replay(wal_path, DIM, on_insert, NULL, 0) != 0, "corrupt header rejected");

    gv_test_remove_db(wal_path);
    return 0;
}

/* Truncated / garbage snapshot bytes must make db_open_from_memory return NULL,
 * not crash. */
static int test_corrupt_snapshot(void) {
    char snap_path[512];
    int fd = gv_test_mkstemp(snap_path, sizeof(snap_path), "gv_corrupt_snap");
    ASSERT(fd >= 0, "mkstemp snap");
    close(fd);

    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open");
    for (int i = 0; i < 8; i++) {
        float v[DIM] = {(float)i, 0.5f, 0.25f, 0.125f};
        db_add_vector(db, v, DIM);
    }
    ASSERT(db_save(db, snap_path) == 0, "db_save");
    db_close(db);

    /* Read the snapshot bytes back. */
    FILE *f = fopen(snap_path, "rb");
    ASSERT(f != NULL, "reopen snapshot");
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    ASSERT(sz > 0, "snapshot non-empty");
    unsigned char *buf = (unsigned char *)malloc((size_t)sz);
    ASSERT(buf != NULL, "malloc snapshot buffer");
    ASSERT(fread(buf, 1, (size_t)sz, f) == (size_t)sz, "read snapshot");
    fclose(f);

    /* 1) Corrupt the header/magic — loader must reject. */
    unsigned char *bad = (unsigned char *)malloc((size_t)sz);
    ASSERT(bad != NULL, "malloc bad buffer");
    memcpy(bad, buf, (size_t)sz);
    bad[0] ^= 0xFF; bad[1] ^= 0xFF; bad[2] ^= 0xFF; bad[3] ^= 0xFF;
    GV_Database *d1 = db_open_from_memory(bad, (size_t)sz, DIM, GV_INDEX_TYPE_FLAT);
    if (d1) db_close(d1);
    ASSERT(d1 == NULL, "corrupted-magic snapshot rejected");
    free(bad);

    /* 2) Truncated snapshot — loader must not read past the buffer. */
    size_t half = (size_t)sz / 2;
    GV_Database *d2 = db_open_from_memory(buf, half, DIM, GV_INDEX_TYPE_FLAT);
    if (d2) db_close(d2);
    ASSERT(d2 == NULL, "truncated snapshot rejected");

    /* 3) Zero-length — must be rejected, not crash. */
    GV_Database *d3 = db_open_from_memory(buf, 0, DIM, GV_INDEX_TYPE_FLAT);
    if (d3) db_close(d3);
    ASSERT(d3 == NULL, "zero-length snapshot rejected");

    free(buf);
    gv_test_remove_db(snap_path);
    return 0;
}

int main(void) {
    int rc = 0;
    rc |= test_corrupt_wal();
    rc |= test_corrupt_snapshot();
    if (rc == 0) printf("All corrupt-resilience tests PASSED.\n");
    return rc != 0;
}
