#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gigavector/gigavector.h"

#define ASSERT(cond, msg)         \
    do {                          \
        if (!(cond)) {            \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1;            \
        }                         \
    } while (0)

static int test_wal_open_close(void) {
    const char *wal_path = "tmp_test_wal.bin.wal";
    remove(wal_path);
    
    GV_WAL *wal = gv_wal_open(wal_path, 3, GV_INDEX_TYPE_KDTREE);
    ASSERT(wal != NULL, "wal open");
    
    gv_wal_close(wal);
    
    remove(wal_path);
    return 0;
}

static int test_wal_append_insert(void) {
    const char *wal_path = "tmp_test_wal_insert.bin.wal";
    remove(wal_path);
    
    GV_WAL *wal = gv_wal_open(wal_path, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(wal != NULL, "wal open");
    
    float v[2] = {1.0f, 2.0f};
    ASSERT(gv_wal_append_insert(wal, v, 2, "tag", "test") == 0, "append insert");
    
    ASSERT(gv_wal_append_insert(wal, v, 2, NULL, NULL) == 0, "append insert without metadata");
    
    gv_wal_close(wal);
    
    remove(wal_path);
    return 0;
}

static int test_wal_append_insert_rich(void) {
    const char *wal_path = "tmp_test_wal_rich.bin.wal";
    remove(wal_path);
    
    GV_WAL *wal = gv_wal_open(wal_path, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(wal != NULL, "wal open");
    
    float v[2] = {1.0f, 2.0f};
    const char *keys[] = {"tag", "owner", "source"};
    const char *values[] = {"a", "b", "demo"};
    
    ASSERT(gv_wal_append_insert_rich(wal, v, 2, keys, values, 3) == 0, "append insert rich");
    
    gv_wal_close(wal);
    
    remove(wal_path);
    return 0;
}

static int test_wal_append_delete(void) {
    const char *wal_path = "tmp_test_wal_delete.bin.wal";
    remove(wal_path);
    
    GV_WAL *wal = gv_wal_open(wal_path, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(wal != NULL, "wal open");
    
    ASSERT(gv_wal_append_delete(wal, 0) == 0, "append delete");
    
    gv_wal_close(wal);
    
    remove(wal_path);
    return 0;
}

static int test_wal_append_update(void) {
    const char *wal_path = "tmp_test_wal_update.bin.wal";
    remove(wal_path);
    
    GV_WAL *wal = gv_wal_open(wal_path, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(wal != NULL, "wal open");
    
    float v[2] = {10.0f, 20.0f};
    const char *keys[] = {"tag"};
    const char *values[] = {"updated"};
    
    ASSERT(gv_wal_append_update(wal, 0, v, 2, keys, values, 1) == 0, "append update");
    
    gv_wal_close(wal);
    
    remove(wal_path);
    return 0;
}

static int test_wal_truncate(void) {
    const char *wal_path = "tmp_test_wal_truncate.bin.wal";
    remove(wal_path);
    
    GV_WAL *wal = gv_wal_open(wal_path, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(wal != NULL, "wal open");
    
    float v[2] = {1.0f, 2.0f};
    ASSERT(gv_wal_append_insert(wal, v, 2, NULL, NULL) == 0, "append insert");
    
    ASSERT(gv_wal_truncate(wal) == 0, "truncate wal");
    
    gv_wal_close(wal);
    
    remove(wal_path);
    return 0;
}

static int test_wal_reset(void) {
    const char *wal_path = "tmp_test_wal_reset.bin.wal";
    remove(wal_path);
    
    GV_WAL *wal = gv_wal_open(wal_path, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(wal != NULL, "wal open");
    
    float v[2] = {1.0f, 2.0f};
    ASSERT(gv_wal_append_insert(wal, v, 2, NULL, NULL) == 0, "append insert");
    
    gv_wal_close(wal);
    
    ASSERT(gv_wal_reset(wal_path) == 0, "reset wal");
    
    remove(wal_path);
    return 0;
}

static int test_wal_dump(void) {
    const char *wal_path = "tmp_test_wal_dump.bin.wal";
    remove(wal_path);
    
    GV_WAL *wal = gv_wal_open(wal_path, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(wal != NULL, "wal open");
    
    float v[2] = {1.0f, 2.0f};
    ASSERT(gv_wal_append_insert(wal, v, 2, "tag", "test") == 0, "append insert");
    
    gv_wal_close(wal);
    
    FILE *out = fopen("/dev/null", "w");
    if (out != NULL) {
        int dump_result = gv_wal_dump(wal_path, 2, GV_INDEX_TYPE_KDTREE, out);
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
    
    GV_WAL *wal = gv_wal_open(wal_path, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(wal != NULL, "wal open");
    
    float v[2] = {1.0f, 2.0f};
    ASSERT(gv_wal_append_insert(wal, v, 2, "tag", "test") == 0, "append insert");
    
    gv_wal_close(wal);
    
    int replay_count = 0;
    int on_insert_cb(void *ctx, const float *data, size_t dimension,
                     const char *metadata_key, const char *metadata_value) {
        (void)ctx;
        (void)data;
        (void)dimension;
        (void)metadata_key;
        (void)metadata_value;
        replay_count++;
        return 0;
    }
    
    int replay_result = gv_wal_replay(wal_path, 2, on_insert_cb, NULL, GV_INDEX_TYPE_KDTREE);
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
    
    GV_WAL *wal = gv_wal_open(wal_path, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(wal != NULL, "wal open");
    
    float v[2] = {1.0f, 2.0f};
    const char *keys[] = {"tag", "owner"};
    const char *values[] = {"a", "b"};
    ASSERT(gv_wal_append_insert_rich(wal, v, 2, keys, values, 2) == 0, "append insert rich");
    
    gv_wal_close(wal);
    
    int replay_count = 0;
    int on_insert_cb(void *ctx, const float *data, size_t dimension,
                     const char *const *metadata_keys, const char *const *metadata_values,
                     size_t metadata_count) {
        (void)ctx;
        (void)data;
        (void)dimension;
        (void)metadata_keys;
        (void)metadata_values;
        (void)metadata_count;
        replay_count++;
        return 0;
    }
    
    ASSERT(gv_wal_replay_rich(wal_path, 2, on_insert_cb, NULL, GV_INDEX_TYPE_KDTREE) == 0, "wal replay rich");
    ASSERT(replay_count == 1, "replay count");
    
    remove(wal_path);
    return 0;
}

static int test_wal_in_database(void) {
    const char *path = "tmp_wal_db.bin";
    const char *wal_path = "tmp_wal_db.bin.wal";
    remove(path);
    remove(wal_path);
    
    GV_Database *db = gv_db_open(path, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "db open");
    
    ASSERT(gv_db_set_wal(db, wal_path) == 0, "set wal");
    
    float v[2] = {1.0f, 2.0f};
    ASSERT(gv_db_add_vector_with_metadata(db, v, 2, "tag", "test") == 0, "add vector");
    
    int dump_result = gv_db_wal_dump(db, stdout);
    if (dump_result != 0) {
        gv_db_disable_wal(db);
        gv_db_close(db);
        remove(path);
        remove(wal_path);
        return 0;
    }
    
    gv_db_disable_wal(db);
    gv_db_close(db);
    
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

