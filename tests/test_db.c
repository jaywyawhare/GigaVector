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

static int test_open_close(void) {
    GV_Database *db = gv_db_open(NULL, 3, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "db open failed");
    gv_db_close(db);
    return 0;
}

static int test_add_and_search(void) {
    GV_Database *db = gv_db_open(NULL, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "db open failed");

    float v1[2] = {1.0f, 2.0f};
    ASSERT(gv_db_add_vector(db, v1, 2) == 0, "add vector");

    GV_SearchResult res[1];
    float q[2] = {1.0f, 2.0f};
    int n = gv_db_search(db, q, 1, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n == 1, "search count");
    ASSERT(res[0].distance == 0.0f, "distance zero");

    gv_db_close(db);
    return 0;
}

static int test_save_load_and_wal(void) {
    const char *path = "tmp_db.bin";
    const char *wal_path = "tmp_db.bin.wal";
    remove(path);
    remove(wal_path);

    // create with WAL
    GV_Database *db = gv_db_open(path, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "open with path");
    ASSERT(gv_db_set_wal(db, wal_path) == 0, "enable wal");

    float v[2] = {0.1f, 0.2f};
    ASSERT(gv_db_add_vector_with_metadata(db, v, 2, "tag", "a") == 0, "add with metadata");
    ASSERT(gv_db_save(db, NULL) == 0, "save");
    gv_db_close(db);

    // reload and replay WAL
    GV_Database *db2 = gv_db_open(path, 2, GV_INDEX_TYPE_KDTREE);
    ASSERT(db2 != NULL, "reopen");
    float q[2] = {0.1f, 0.2f};
    GV_SearchResult res[1];
    int n = gv_db_search(db2, q, 1, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n == 1, "search after reload");
    ASSERT(res[0].distance == 0.0f, "distance after reload");
    gv_db_close(db2);

    remove(path);
    remove(wal_path);
    return 0;
}

int main(void) {
    int rc = 0;
    rc |= test_open_close();
    rc |= test_add_and_search();
    rc |= test_save_load_and_wal();
    if (rc == 0) {
        printf("C tests passed\n");
    }
    return rc;
}

