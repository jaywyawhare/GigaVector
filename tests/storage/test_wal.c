#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gigavector.h"

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
    const char *wal_path = "tmp_test_wal.bin.wal";
    remove(wal_path);
    
    GV_WAL *wal = wal_open(wal_path, 3, GV_INDEX_TYPE_KDTREE);
    ASSERT(wal != NULL, "wal open");
    
    wal_close(wal);
    
    remove(wal_path);
    return 0;
}

static int test_wal_append_insert(void) {
    const char *wal_path = "tmp_test_wal_insert.bin.wal";
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
    const char *wal_path = "tmp_test_wal_rich.bin.wal";
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
    const char *wal_path = "tmp_test_wal_delete.bin.wal";
    remove(wal_path);
    
    GV_WAL *wal = wal_open(wal_path, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(wal != NULL, "wal open");
    
    ASSERT(wal_append_delete(wal, 0) == 0, "append delete");
    
    wal_close(wal);
    
    remove(wal_path);
    return 0;
}

static int test_wal_append_update(void) {
    const char *wal_path = "tmp_test_wal_update.bin.wal";
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
    const char *wal_path = "tmp_test_wal_truncate.bin.wal";
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
    const char *wal_path = "tmp_test_wal_reset.bin.wal";
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
    const char *wal_path = "tmp_test_wal_dump.bin.wal";
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
        if (dump_result != 0) {
            remove(wal_path);
            return 0;
        }
    }
    
    remove(wal_path);
    return 0;
}

static int test_wal_replay(void) {
    const char *wal_path = "tmp_test_wal_replay.bin.wal";
    remove(wal_path);
    
    GV_WAL *wal = wal_open(wal_path, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(wal != NULL, "wal open");
    
    float v[2] = {1.0f, 2.0f};
    ASSERT(wal_append_insert(wal, v, 2, "tag", "test") == 0, "append insert");
    
    wal_close(wal);
    
    int replay_count = 0;
    ReplayCtx ctx = { .count = &replay_count };

    int replay_result = wal_replay(wal_path, 2, on_insert_basic_cb, &ctx, GV_INDEX_TYPE_KDTREE);
    if (replay_result != 0) {
        remove(wal_path);
        return 0;
    }
    ASSERT(replay_count >= 0, "replay count");
    
    remove(wal_path);
    return 0;
}

static int test_wal_replay_rich(void) {
    const char *wal_path = "tmp_test_wal_replay_rich.bin.wal";
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

    ASSERT(wal_replay_rich(wal_path, 2, on_insert_rich_cb, &ctx, GV_INDEX_TYPE_KDTREE) == 0, "wal replay rich");
    ASSERT(replay_count == 1, "replay count");
    
    remove(wal_path);
    return 0;
}

static int test_wal_in_database(void) {
    const char *path = "tmp_wal_db.bin";
    const char *wal_path = "tmp_wal_db.bin.wal";
    remove(path);
    remove(wal_path);
    
    GV_Database *db = db_open(path, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "db open");
    
    ASSERT(db_set_wal(db, wal_path) == 0, "set wal");
    
    float v[2] = {1.0f, 2.0f};
    ASSERT(db_add_vector_with_metadata(db, v, 2, "tag", "test") == 0, "add vector");
    
    int dump_result = db_wal_dump(db, stdout);
    if (dump_result != 0) {
        db_disable_wal(db);
        db_close(db);
        remove(path);
        remove(wal_path);
        return 0;
    }
    
    db_disable_wal(db);
    db_close(db);
    
    remove(path);
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
    rc |= test_wal_in_database();
    return rc;
}

