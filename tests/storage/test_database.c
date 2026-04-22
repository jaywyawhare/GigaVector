#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "storage/database.h"
#include "core/types.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_db_open_null(void) {
    GV_Database *db = db_open(NULL, 0, GV_INDEX_TYPE_FLAT);
    ASSERT(db == NULL, "open with zero dimension returns NULL");
    return 0;
}

static int test_db_open_flat(void) {
    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "open FLAT database");
    ASSERT(db->dimension == 4, "dimension matches");
    ASSERT(db->index_type == GV_INDEX_TYPE_FLAT, "index type matches");
    db_close(db);
    return 0;
}

static int test_db_open_hnsw(void) {
    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_HNSW);
    ASSERT(db != NULL, "open HNSW database");
    db_close(db);
    return 0;
}

static int test_db_open_kdtree(void) {
    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_KDTREE);
    ASSERT(db != NULL, "open KD-Tree database");
    db_close(db);
    return 0;
}

static int test_db_open_sparse(void) {
    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_SPARSE);
    ASSERT(db != NULL, "open SPARSE database");
    db_close(db);
    return 0;
}

static int test_db_index_suggest(void) {
    GV_IndexType t1 = index_suggest(128, 100);
    ASSERT(t1 == GV_INDEX_TYPE_FLAT, "small dataset suggests FLAT");

    GV_IndexType t2 = index_suggest(128, 1000);
    ASSERT(t2 == GV_INDEX_TYPE_HNSW, "medium dataset suggests HNSW");

    GV_IndexType t3 = index_suggest(16, 10000);
    ASSERT(t3 == GV_INDEX_TYPE_KDTREE, "low-dim large dataset suggests KDTREE");

    return 0;
}

static int test_db_cosine_normalized(void) {
    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "create db for cosine test");

    db_set_cosine_normalized(db, 1);
    ASSERT(db->cosine_normalized == 1, "cosine normalized enabled");

    db_set_cosine_normalized(db, 0);
    ASSERT(db->cosine_normalized == 0, "cosine normalized disabled");

    db_close(db);
    return 0;
}

static int test_db_get_stats(void) {
    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "create db for stats test");

    GV_DBStats stats;
    db_get_stats(db, &stats);
    ASSERT(stats.total_inserts == 0, "initial total_inserts is 0");
    ASSERT(stats.total_queries == 0, "initial total_queries is 0");

    db_close(db);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"db_open_null", test_db_open_null},
        {"db_open_flat", test_db_open_flat},
        {"db_open_hnsw", test_db_open_hnsw},
        {"db_open_kdtree", test_db_open_kdtree},
        {"db_open_sparse", test_db_open_sparse},
        {"db_index_suggest", test_db_index_suggest},
        {"db_cosine_normalized", test_db_cosine_normalized},
        {"db_get_stats", test_db_get_stats},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) passed++;
    }
    printf("%d/%d tests passed\n", passed, n);
    return (passed == n) ? 0 : 1;
}