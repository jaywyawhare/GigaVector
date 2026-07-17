#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "gigavector.h"
#include "../test_tmp.h"

#define ASSERT(cond, msg)         \
    do {                          \
        if (!(cond)) {            \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1;            \
        }                         \
    } while (0)

typedef struct {
    int *count;
} ReplayCtx;

static int on_insert_basic_cb(void *ctx, const float *data, size_t dimension,
                              const char *metadata_key, const char *metadata_value) {
    (void)data;
    (void)dimension;
    (void)metadata_key;
    (void)metadata_value;
    if (!ctx) return -1;
    ReplayCtx *rctx = (ReplayCtx *)ctx;
    if (!rctx->count) return -1;
    (*rctx->count)++;
    return 0;
}

static int on_insert_rich_cb(void *ctx, const float *data, size_t dimension,
                             const char *const *metadata_keys, const char *const *metadata_values,
                             size_t metadata_count) {
    (void)data;
    (void)dimension;
    (void)metadata_keys;
    (void)metadata_values;
    (void)metadata_count;
    if (!ctx) return -1;
    ReplayCtx *rctx = (ReplayCtx *)ctx;
    if (!rctx->count) return -1;
    (*rctx->count)++;
    return 0;
}

static int test_wal_open_close(void) {
    char wal_path[256];
    if (gv_test_make_temp_path(wal_path, sizeof(wal_path), "gv_wal_open", ".wal") != 0) return 0;
    remove(wal_path);

    GV_WAL *wal = wal_open(wal_path, 3, GV_INDEX_TYPE_KDTREE);
    ASSERT(wal != NULL, "wal open");

    wal_close(wal);
    remove(wal_path);
    return 0;
}

static int test_wal_append_insert(void) {
    char wal_path[256];
    if (gv_test_make_temp_path(wal_path, sizeof(wal_path), "gv_wal_insert", ".wal") != 0) return 0;
    remove(wal_path);

    GV_WAL *wal = wal_open(wal_path, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(wal != NULL, "wal open");

    float v[2] = {1.0f, 2.0f};
    ASSERT(wal_append_insert(wal, v, 2, "tag", "test") == 0, "append insert");
    ASSERT(wal_append_insert(wal, v, 2, NULL, NULL) == 0, "append insert without metadata");

    wal_close(wal);
    remove(wal_path);
    return 0;
}

static int test_wal_append_insert_rich(void) {
    char wal_path[256];
    if (gv_test_make_temp_path(wal_path, sizeof(wal_path), "gv_wal_rich", ".wal") != 0) return 0;
    remove(wal_path);

    GV_WAL *wal = wal_open(wal_path, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(wal != NULL, "wal open");

    float v[2] = {1.0f, 2.0f};
    const char *keys[] = {"tag", "owner", "source"};
    const char *values[] = {"a", "b", "demo"};

    ASSERT(wal_append_insert_rich(wal, v, 2, keys, values, 3) == 0, "append insert rich");

    wal_close(wal);
    remove(wal_path);
    return 0;
}

static int test_wal_append_delete(void) {
    char wal_path[256];
    if (gv_test_make_temp_path(wal_path, sizeof(wal_path), "gv_wal_delete", ".wal") != 0) return 0;
    remove(wal_path);

    GV_WAL *wal = wal_open(wal_path, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(wal != NULL, "wal open");

    ASSERT(wal_append_delete(wal, 0) == 0, "append delete");

    wal_close(wal);
    remove(wal_path);
    return 0;
}

static int test_wal_append_update(void) {
    char wal_path[256];
    if (gv_test_make_temp_path(wal_path, sizeof(wal_path), "gv_wal_update", ".wal") != 0) return 0;
    remove(wal_path);

    GV_WAL *wal = wal_open(wal_path, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(wal != NULL, "wal open");

    float v[2] = {10.0f, 20.0f};
    const char *keys[] = {"tag"};
    const char *values[] = {"updated"};

    ASSERT(wal_append_update(wal, 0, v, 2, keys, values, 1) == 0, "append update");

    wal_close(wal);
    remove(wal_path);
    return 0;
}

static int test_wal_truncate(void) {
    char wal_path[256];
    if (gv_test_make_temp_path(wal_path, sizeof(wal_path), "gv_wal_trunc", ".wal") != 0) return 0;
    remove(wal_path);

    GV_WAL *wal = wal_open(wal_path, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(wal != NULL, "wal open");

    float v[2] = {1.0f, 2.0f};
    ASSERT(wal_append_insert(wal, v, 2, NULL, NULL) == 0, "append insert");
    ASSERT(wal_truncate(wal) == 0, "truncate wal");

    wal_close(wal);
    remove(wal_path);
    return 0;
}

static int test_wal_reset(void) {
    char wal_path[256];
    if (gv_test_make_temp_path(wal_path, sizeof(wal_path), "gv_wal_reset", ".wal") != 0) return 0;
    remove(wal_path);

    GV_WAL *wal = wal_open(wal_path, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(wal != NULL, "wal open");

    float v[2] = {1.0f, 2.0f};
    ASSERT(wal_append_insert(wal, v, 2, NULL, NULL) == 0, "append insert");

    wal_close(wal);

    ASSERT(wal_reset(wal_path) == 0, "reset wal");

    remove(wal_path);
    return 0;
}

static int test_wal_dump(void) {
    char wal_path[256];
    if (gv_test_make_temp_path(wal_path, sizeof(wal_path), "gv_wal_dump", ".wal") != 0) return 0;
    remove(wal_path);

    GV_WAL *wal = wal_open(wal_path, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(wal != NULL, "wal open");

    float v[2] = {1.0f, 2.0f};
    ASSERT(wal_append_insert(wal, v, 2, "tag", "test") == 0, "append insert");

    wal_close(wal);

    FILE *out = fopen("/dev/null", "w");
    if (out != NULL) {
        int dump_result = wal_dump(wal_path, 2, GV_INDEX_TYPE_KDTREE, out);
        fclose(out);
        ASSERT(dump_result == 0, "wal dump succeeded");
    }

    remove(wal_path);
    return 0;
}

static int test_wal_replay(void) {
    char wal_path[256];
    if (gv_test_make_temp_path(wal_path, sizeof(wal_path), "gv_wal_replay", ".wal") != 0) return 0;
    remove(wal_path);

    GV_WAL *wal = wal_open(wal_path, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(wal != NULL, "wal open");

    float v[2] = {1.0f, 2.0f};
    ASSERT(wal_append_insert(wal, v, 2, "tag", "test") == 0, "append insert");

    wal_close(wal);

    int replay_count = 0;
    ReplayCtx ctx = { .count = &replay_count };

    int replay_result = wal_replay(wal_path, 2, on_insert_basic_cb, &ctx, GV_INDEX_TYPE_KDTREE);
    ASSERT(replay_result == 0, "wal replay succeeded");
    ASSERT(replay_count == 1, "replay count is 1");

    remove(wal_path);
    return 0;
}

static int test_wal_replay_rich(void) {
    char wal_path[256];
    if (gv_test_make_temp_path(wal_path, sizeof(wal_path), "gv_wal_replay_rich", ".wal") != 0) return 0;
    remove(wal_path);

    GV_WAL *wal = wal_open(wal_path, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(wal != NULL, "wal open");

    float v[2] = {1.0f, 2.0f};
    const char *keys[] = {"tag", "owner"};
    const char *values[] = {"a", "b"};
    ASSERT(wal_append_insert_rich(wal, v, 2, keys, values, 2) == 0, "append insert rich");

    wal_close(wal);

    int replay_count = 0;
    ReplayCtx ctx = { .count = &replay_count };

    ASSERT(wal_replay_rich(wal_path, 2, on_insert_rich_cb, NULL, NULL, NULL, &ctx,
                           GV_INDEX_TYPE_KDTREE) == 0,
           "wal replay rich succeeded");
    ASSERT(replay_count == 1, "replay count is 1");

    remove(wal_path);
    return 0;
}

/*
 * WAL fsync-before-truncate ORDERING reinforcement.
 *
 * wal.c durably fsync()s appended records before any ftruncate() reclaims log
 * space; the ordering guarantee is that a crash can only ever leave a *torn
 * tail* (a partially-written trailing record), never a hole in the middle of a
 * durably-acknowledged prefix. This test exercises that contract from the
 * recovery side: it writes several complete records, then physically truncates
 * the file in the MIDDLE of the last record (simulating a crash mid-append
 * before the record was fsync'd/CRC-committed). Replay must recover the
 * durable prefix and drop the torn trailing record — i.e. it must NOT fail the
 * whole log and must NOT count the torn record. Mirrors the torn-tail pattern
 * in tests/storage/test_corrupt_resilience.c.
 */
static int test_wal_fsync_truncate_ordering(void) {
    char wal_path[256];
    if (gv_test_make_temp_path(wal_path, sizeof(wal_path), "gv_wal_ordering", ".wal") != 0) return 0;
    remove(wal_path);

    GV_WAL *wal = wal_open(wal_path, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(wal != NULL, "wal open");

    /* Write several complete, durable records. */
    const int N = 6;
    for (int i = 0; i < N; i++) {
        float v[2] = {(float)i, (float)(i + 1)};
        ASSERT(wal_append_insert(wal, v, 2, "tag", "seed") == 0, "append insert");
    }
    wal_close(wal);

    /* Sanity: a clean replay sees every durable record. */
    int clean = 0;
    ReplayCtx cctx = { .count = &clean };
    ASSERT(wal_replay(wal_path, 2, on_insert_basic_cb, &cctx, GV_INDEX_TYPE_KDTREE) == 0,
           "clean replay ok");
    ASSERT(clean == N, "clean replay saw all records");

    /* Measure file size, then truncate MID-record: drop the last few bytes so
     * the trailing record is structurally incomplete (a torn tail, exactly the
     * state the fsync-then-ftruncate ordering bounds a crash to). */
    FILE *f = fopen(wal_path, "rb");
    ASSERT(f != NULL, "open for size");
    ASSERT(fseek(f, 0, SEEK_END) == 0, "seek end");
    long sz = ftell(f);
    fclose(f);
    ASSERT(sz > 8, "wal non-trivial");

    /* Cut off a handful of bytes from the end — inside the final record's
     * payload/CRC region rather than on a clean record boundary. */
    ASSERT(truncate(wal_path, sz - 5) == 0, "truncate mid-record");

    /* Replay must RECOVER: succeed (defined result, no crash) and drop the
     * torn trailing record. The durable prefix survives; the torn tail does not. */
    int torn = 0;
    ReplayCtx tctx = { .count = &torn };
    int rc = wal_replay(wal_path, 2, on_insert_basic_cb, &tctx, GV_INDEX_TYPE_KDTREE);
    ASSERT(rc == 0, "torn-tail replay recovers (no crash, defined result)");
    ASSERT(torn < N, "torn trailing record dropped on recovery");
    ASSERT(torn >= N - 1, "only the torn tail is dropped; durable prefix survives");

    remove(wal_path);
    return 0;
}

static int test_wal_in_database(void) {
    char db_path[256], wal_path[256];
    if (gv_test_make_temp_path(db_path, sizeof(db_path), "gv_wal_db", ".bin") != 0) return 0;
    if (gv_test_make_temp_path(wal_path, sizeof(wal_path), "gv_wal_db", ".wal") != 0) return 0;
    remove(db_path);
    remove(wal_path);

    GV_Database *db = db_open(db_path, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "db open");

    ASSERT(db_set_wal(db, wal_path) == 0, "set wal");

    float v[2] = {1.0f, 2.0f};
    ASSERT(db_add_vector_with_metadata(db, v, 2, "tag", "test") == 0, "add vector");

    int dump_result = db_wal_dump(db, stdout);
    if (dump_result != 0) {
        db_disable_wal(db);
        db_close(db);
        /* gv_test_remove_db also removes the "<db_path>.wal" sidecar that
         * db_open() creates, which plain remove(db_path) would leak in /tmp. */
        gv_test_remove_db(db_path);
        remove(wal_path);
        return 0;
    }

    db_disable_wal(db);
    db_close(db);

    gv_test_remove_db(db_path);
    remove(wal_path);
    return 0;
}

int main(void) {
    int rc = 0;
    rc |= test_wal_open_close();
    rc |= test_wal_append_insert();
    rc |= test_wal_append_insert_rich();
    rc |= test_wal_append_delete();
    rc |= test_wal_append_update();
    rc |= test_wal_truncate();
    rc |= test_wal_reset();
    rc |= test_wal_dump();
    rc |= test_wal_replay();
    rc |= test_wal_replay_rich();
    rc |= test_wal_fsync_truncate_ordering();
    rc |= test_wal_in_database();
    return rc;
}
