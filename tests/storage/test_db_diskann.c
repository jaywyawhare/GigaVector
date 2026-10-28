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
    gv_search_results_free(res, (size_t)found);

    /* Delete it; it must no longer be the top hit. */
    ASSERT(db_delete_vector_by_index(db, 10) == 0, "delete failed");
    memset(res, 0, sizeof(res));
    found = db_search(db, &data[10 * DIM], 5, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(found > 0, "search after delete");
    ASSERT(res[0].id != 10, "deleted id not returned");
    gv_search_results_free(res, (size_t)found);

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
    gv_search_results_free(res, (size_t)found);
    memset(res, 0, sizeof(res));
    found = db_search(db2, &data[10 * DIM], 5, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(found > 0, "search deleted after reopen");
    ASSERT(res[0].id != 10, "deleted id stays deleted after reopen");
    gv_search_results_free(res, (size_t)found);

    db_close(db2);
    remove(path);
    char side[600];
    snprintf(side, sizeof(side), "%s.diskann", path); remove(side);
    snprintf(side, sizeof(side), "%s.wal", path); remove(side);
    return 0;
}

/* Metadata-filtered search, filter-expression search, range search, and update
 * all routed through the DiskANN db integration. Even ids carry category "a",
 * odd ids category "b". */
static int test_db_diskann_features(void) {
    char path[512];
    gv_test_make_temp_path(path, sizeof(path), "gv_db_diskann_feat", ".db");

    GV_Database *db = db_open(path, DIM, GV_INDEX_TYPE_DISKANN);
    ASSERT(db != NULL, "db_open DISKANN failed");

    float data[N * DIM];
    for (size_t i = 0; i < N; i++) {
        fill(&data[i * DIM], DIM, (float)i);
        const char *cat = (i % 2 == 0) ? "a" : "b";
        ASSERT(db_add_vector_with_metadata(db, &data[i * DIM], DIM, "category", cat) == 0,
               "metadata add failed");
    }

    /* Filtered search: every hit must be an even id (category "a"). */
    GV_SearchResult res[8];
    int found = db_search_filtered(db, &data[12 * DIM], 5, res, GV_DISTANCE_EUCLIDEAN,
                                   "category", "a");
    ASSERT(found > 0, "filtered search returned results");
    for (int i = 0; i < found; i++) ASSERT(res[i].id % 2 == 0, "filtered hit is category a");
    gv_search_results_free(res, (size_t)found);

    /* Filter-expression search: same constraint via an expression. */
    found = db_search_with_filter_expr(db, &data[12 * DIM], 5, res, GV_DISTANCE_EUCLIDEAN,
                                       "category == \"a\"");
    ASSERT(found > 0, "filter-expr search returned results");
    for (int i = 0; i < found; i++) ASSERT(res[i].id % 2 == 0, "filter-expr hit is category a");
    gv_search_results_free(res, (size_t)found);

    /* Range search: the query is a stored vector, so it is within any radius. */
    found = db_range_search(db, &data[20 * DIM], 10.0f, res, 8, GV_DISTANCE_EUCLIDEAN);
    ASSERT(found > 0, "range search returned results");
    int saw_self = 0;
    for (int i = 0; i < found; i++) if (res[i].id == 20) saw_self = 1;
    ASSERT(saw_self, "range search includes the query vector");
    gv_search_results_free(res, (size_t)found);

    /* Update: replace id 30 with id 5's vector; a self-query on the new data
     * must return id 30 as its own nearest neighbour. */
    ASSERT(db_update_vector(db, 30, &data[5 * DIM], DIM) == 0, "update failed");
    found = db_search(db, &data[5 * DIM], 3, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(found > 0, "search after update");
    int saw_updated = 0;
    for (int i = 0; i < found; i++) if (res[i].id == 30) saw_updated = 1;
    ASSERT(saw_updated, "updated vector is found at its new location");
    gv_search_results_free(res, (size_t)found);

    db_close(db);
    remove(path);
    char side[600];
    snprintf(side, sizeof(side), "%s.diskann", path); remove(side);
    snprintf(side, sizeof(side), "%s.diskann.graph", path); remove(side);
    snprintf(side, sizeof(side), "%s.wal", path); remove(side);
    return 0;
}

int main(void) {
    int passed = 0;
    fprintf(stderr, "Testing db DISKANN lifecycle...\n");
    if (test_db_diskann_lifecycle() != 0) { fprintf(stderr, "\n%d/2 tests passed\n", passed); return 1; }
    passed++;
    fprintf(stderr, "Testing db DISKANN features (filter/range/update)...\n");
    if (test_db_diskann_features() != 0) { fprintf(stderr, "\n%d/2 tests passed\n", passed); return 1; }
    passed++;
    fprintf(stderr, "\n%d/2 tests passed\n", passed);
    return 0;
}
