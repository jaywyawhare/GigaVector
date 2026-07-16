/* Phase 0 — identity seam: chunk_id <-> internal index, cascade delete, sidecar persist. */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include "storage/database.h"

static int failures = 0;
#define ASSERT(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", (msg)); failures++; } \
    else { printf("ok: %s\n", (msg)); } \
} while (0)

static void fill(float *v, size_t d, float base) {
    for (size_t i = 0; i < d; ++i) v[i] = base + (float)i;
}

int main(void) {
    const size_t D = 4;
    float a[4], b[4], c[4];
    fill(a, D, 1.0f); fill(b, D, 2.0f); fill(c, D, 3.0f);

    /* --- in-memory: add-with-id, resolve, delete-by-id --- */
    GV_Database *db = db_open(NULL, D, GV_INDEX_TYPE_HNSW);
    ASSERT(db != NULL, "db_open");

    ASSERT(db_add_vector_with_id(db, "docA:0000", a, D) == 0, "add docA:0000");
    ASSERT(db_add_vector_with_id(db, "docA:0001", b, D) == 0, "add docA:0001");
    ASSERT(db_add_vector_with_id(db, "docB:0000", c, D) == 0, "add docB:0000");

    size_t idx = 999;
    ASSERT(db_get_index_by_id(db, "docA:0001", &idx) == 0 && idx == 1, "resolve docA:0001 -> 1");
    ASSERT(db_get_index_by_id(db, "nope", &idx) != 0, "resolve missing id fails");

    /* cascading delete by doc: removes both docA chunks, keeps docB */
    int deleted = db_delete_by_doc(db, "docA");
    ASSERT(deleted == 2, "delete_by_doc(docA) removes 2");
    ASSERT(db_get_index_by_id(db, "docA:0000", &idx) != 0, "docA:0000 gone from map");
    ASSERT(db_get_index_by_id(db, "docB:0000", &idx) == 0, "docB:0000 still present");

    /* delete_by_id on the survivor */
    ASSERT(db_delete_by_id(db, "docB:0000") == 0, "delete_by_id(docB:0000)");
    ASSERT(db_get_index_by_id(db, "docB:0000", &idx) != 0, "docB:0000 gone after delete_by_id");

    db_close(db);

    /* --- file-backed: sidecar round-trip --- */
    const char *path = "test_identity_seam.db";
    char idspath[256];
    snprintf(idspath, sizeof(idspath), "%s.ids", path);
    unlink(path); unlink(idspath);

    GV_Database *fdb = db_open(path, D, GV_INDEX_TYPE_HNSW);
    ASSERT(fdb != NULL, "file db_open");
    db_add_vector_with_id(fdb, "docX:0000", a, D);
    db_add_vector_with_id(fdb, "docX:0001", b, D);
    ASSERT(db_save(fdb, path) == 0, "db_save writes sidecar");
    db_close(fdb);

    ASSERT(access(idspath, F_OK) == 0, ".ids sidecar file exists");

    GV_Database *rdb = db_open(path, D, GV_INDEX_TYPE_HNSW);
    ASSERT(rdb != NULL, "reopen file db");
    size_t ri = 999;
    ASSERT(db_get_index_by_id(rdb, "docX:0001", &ri) == 0 && ri == 1,
           "id map survived save/load (docX:0001 -> 1)");
    db_close(rdb);
    unlink(path); unlink(idspath);
    char walpath[256];
    snprintf(walpath, sizeof(walpath), "%s.wal", path); unlink(walpath);

    printf(failures ? "\nSOME TESTS FAILED (%d)\n" : "\nALL IDENTITY-SEAM TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
