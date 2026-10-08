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

typedef struct { int ins; int del; } TxnReplayCtx;
static int txn_ins_cb(void *ctx, const float *d, size_t dim,
                      const char *const *k, const char *const *v, size_t n) {
    (void)d; (void)dim; (void)k; (void)v; (void)n;
    ((TxnReplayCtx *)ctx)->ins++;
    return 0;
}
static int txn_del_cb(void *ctx, size_t idx) {
    (void)idx;
    ((TxnReplayCtx *)ctx)->del++;
    return 0;
}

/* A single atomic TXN record replays as all its inserts + deletes. */
static int test_wal_txn_replay(void) {
    char wal_path[256];
    if (gv_test_make_temp_path(wal_path, sizeof(wal_path), "gv_wal_txn", ".wal") != 0) return 0;
    remove(wal_path);

    GV_WAL *wal = wal_open(wal_path, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(wal != NULL, "wal open");
    float a[4] = {1, 2, 3, 4}, b[4] = {5, 6, 7, 8};
    const float *ins[2] = {a, b};
    uint64_t dels[1] = {7};
    ASSERT(wal_append_txn(wal, ins, 4, 2, dels, 1) == 0, "append atomic txn record");
    wal_close(wal);

    TxnReplayCtx c = {0, 0};
    int rc = wal_replay_rich(wal_path, 4, txn_ins_cb, txn_del_cb, NULL, NULL, &c, GV_INDEX_TYPE_FLAT);
    ASSERT(rc == 0, "replay of txn record succeeds");
    ASSERT(c.ins == 2 && c.del == 1, "txn record applied all-or-nothing: 2 inserts + 1 delete");
    remove(wal_path);
    return 0;
}

/* Crash mid-commit: a partially-written TXN record must be discarded WHOLE on
 * replay, leaving everything committed before it intact (crash-atomicity). */
static int test_wal_txn_crash_atomic(void) {
    char wal_path[256];
    if (gv_test_make_temp_path(wal_path, sizeof(wal_path), "gv_wal_txncrash", ".wal") != 0) return 0;
    remove(wal_path);

    GV_WAL *wal = wal_open(wal_path, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(wal != NULL, "wal open");
    float base[4] = {9, 9, 9, 9};
    ASSERT(wal_append_insert(wal, base, 4, NULL, NULL) == 0, "base insert");
    float a0[4] = {1, 0, 0, 0}, a1[4] = {0, 1, 0, 0};
    const float *insA[2] = {a0, a1};
    ASSERT(wal_append_txn(wal, insA, 4, 2, NULL, 0) == 0, "commit txn A");
    uint64_t size_after_a = wal_size(wal);
    float b0[4] = {0, 0, 1, 0}, b1[4] = {0, 0, 0, 1};
    const float *insB[2] = {b0, b1};
    ASSERT(wal_append_txn(wal, insB, 4, 2, NULL, 0) == 0, "commit txn B");
    uint64_t size_after_b = wal_size(wal);
    wal_close(wal);
    ASSERT(size_after_b > size_after_a, "txn B grew the log");

    /* Simulate a crash partway through txn B's record. */
    long torn = (long)size_after_a + (long)((size_after_b - size_after_a) / 2);
    ASSERT(truncate(wal_path, torn) == 0, "truncate to a partial txn B");

    TxnReplayCtx c = {0, 0};
    int rc = wal_replay_rich(wal_path, 4, txn_ins_cb, txn_del_cb, NULL, NULL, &c, GV_INDEX_TYPE_FLAT);
    ASSERT(rc == 0, "replay succeeds despite torn trailing txn");
    ASSERT(c.ins == 3, "base + txn A applied; torn txn B discarded in full");
    remove(wal_path);
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
 * WAL fsync-before-truncate ordering: a crash can only leave a torn trailing
 * record, never a hole in a durable prefix. Truncating mid-last-record, replay
 * must recover the durable prefix and drop only the torn tail (not fail the log).
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

    /* Cut off a handful of bytes from the end - inside the final record's
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

static long gv_file_size(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long s = ftell(f);
    fclose(f);
    return s;
}

static int test_wal_size(void) {
    char wal_path[256];
    if (gv_test_make_temp_path(wal_path, sizeof(wal_path), "gv_wal_size", ".wal") != 0) return 0;
    remove(wal_path);

    GV_WAL *wal = wal_open(wal_path, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(wal != NULL, "wal open");
    uint64_t hdr = wal_size(wal);
    ASSERT(hdr > 0, "header-only size is non-zero");

    float v[2] = {1.0f, 2.0f};
    for (int i = 0; i < 20; i++) {
        ASSERT(wal_append_insert(wal, v, 2, NULL, NULL) == 0, "append insert");
    }
    uint64_t grown = wal_size(wal);
    ASSERT(grown > hdr, "wal_size grows with appends");

    ASSERT(wal_truncate(wal) == 0, "truncate");
    uint64_t after = wal_size(wal);
    ASSERT(after < grown, "wal_size shrinks after truncate");

    wal_close(wal);
    remove(wal_path);
    return 0;
}

static int test_db_wal_checkpoint(void) {
    char db_path[256], wal_path[256];
    if (gv_test_make_temp_path(db_path, sizeof(db_path), "gv_ckpt_db", ".bin") != 0) return 0;
    if (gv_test_make_temp_path(wal_path, sizeof(wal_path), "gv_ckpt", ".wal") != 0) return 0;
    remove(db_path); remove(wal_path);

    GV_Database *db = db_open(db_path, 2, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db open");
    ASSERT(db_set_wal(db, wal_path) == 0, "set wal");

    float v[2];
    for (int i = 0; i < 50; i++) {
        v[0] = (float)i; v[1] = (float)(i + 1);
        ASSERT(db_add_vector(db, v, 2) == 0, "add vector");
    }
    long grown = gv_file_size(wal_path);
    ASSERT(grown > 0, "wal grew on disk");

    /* Threshold 1 byte -> a checkpoint runs and truncates the WAL. */
    int ck = db_wal_checkpoint_if_needed(db, 1);
    ASSERT(ck == 1, "checkpoint ran when over threshold");
    long after = gv_file_size(wal_path);
    ASSERT(after >= 0 && after < grown, "wal shrank after checkpoint");

    /* Below threshold -> no-op. */
    ASSERT(db_wal_checkpoint_if_needed(db, (size_t)1 << 30) == 0, "no checkpoint below threshold");

    /* Data survives the checkpoint. */
    GV_SearchResult res[1];
    int n = db_search(db, v, 1, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n == 1, "data survives checkpoint");
    gv_search_results_free(res, (size_t)(n > 0 ? n : 0));

    db_disable_wal(db);
    db_close(db);
    gv_test_remove_db(db_path);
    remove(wal_path);

    /* In-memory database: nothing to checkpoint. */
    GV_Database *mem = db_open(NULL, 2, GV_INDEX_TYPE_FLAT);
    ASSERT(mem != NULL, "open in-memory db");
    ASSERT(db_wal_checkpoint(mem) == -1, "in-memory checkpoint is a no-op failure");
    ASSERT(db_wal_checkpoint_if_needed(mem, 1) == 0, "in-memory checkpoint_if_needed is 0");
    db_close(mem);
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
    rc |= test_wal_size();
    rc |= test_db_wal_checkpoint();
    rc |= test_wal_txn_replay();
    rc |= test_wal_txn_crash_atomic();
    rc |= test_wal_in_database();
    return rc;
}
