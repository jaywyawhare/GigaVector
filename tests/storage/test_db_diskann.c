#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "storage/database.h"
#include "../test_tmp.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

#define DIM 16
#define N 120

static void fill(float *v, size_t dim, float seed) {
    for (size_t i = 0; i < dim; i++) v[i] = sinf(seed + (float)i * 0.3f);
}

/* End-to-end DiskANN collection: create, add, search, delete, then persist and
 * reopen (the graph is rebuilt from the restored vectors). */
static int test_db_diskann_lifecycle(void) {
    char path[512];
    gv_test_make_temp_path(path, sizeof(path), "gv_db_diskann", ".db");

    GV_Database *db = db_open(path, DIM, GV_INDEX_TYPE_DISKANN);
    ASSERT(db != NULL, "db_open DISKANN failed");

    float data[N * DIM];
    for (size_t i = 0; i < N; i++) {
        fill(&data[i * DIM], DIM, (float)i);
        ASSERT(db_add_vector(db, &data[i * DIM], DIM) == 0, "db_add_vector failed");
    }

    /* Search with a stored vector: it should be its own nearest neighbour. */
    GV_SearchResult res[5];
    memset(res, 0, sizeof(res));
    int found = db_search(db, &data[10 * DIM], 5, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(found > 0, "db_search returned results");
    ASSERT(res[0].id == 10, "self-query returns its own id");

    /* Delete it; it must no longer be the top hit. */
    ASSERT(db_delete_vector_by_index(db, 10) == 0, "delete failed");
    memset(res, 0, sizeof(res));
    found = db_search(db, &data[10 * DIM], 5, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(found > 0, "search after delete");
    ASSERT(res[0].id != 10, "deleted id not returned");

    ASSERT(db_save(db, path) == 0, "db_save failed");
    db_close(db);

    /* Reopen: the graph is rebuilt from persisted vectors; search still works
     * and the deleted id stays deleted. */
    GV_Database *db2 = db_open(path, DIM, GV_INDEX_TYPE_DISKANN);
    ASSERT(db2 != NULL, "reopen DISKANN failed");
    memset(res, 0, sizeof(res));
    found = db_search(db2, &data[20 * DIM], 5, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(found > 0, "search after reopen");
    ASSERT(res[0].id == 20, "reopened self-query returns its own id");
    memset(res, 0, sizeof(res));
    found = db_search(db2, &data[10 * DIM], 5, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(found > 0, "search deleted after reopen");
    ASSERT(res[0].id != 10, "deleted id stays deleted after reopen");

    db_close(db2);
    remove(path);
    char side[600];
    snprintf(side, sizeof(side), "%s.diskann", path); remove(side);
    snprintf(side, sizeof(side), "%s.wal", path); remove(side);
    return 0;
}

int main(void) {
    fprintf(stderr, "Testing db DISKANN lifecycle...\n");
    if (test_db_diskann_lifecycle() != 0) { fprintf(stderr, "\n0/1 tests passed\n"); return 1; }
    fprintf(stderr, "\n1/1 tests passed\n");
    return 0;
}
