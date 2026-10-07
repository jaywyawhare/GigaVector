/* Object store (filesystem backend) + native backup to/from an object store. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "storage/object_store.h"
#include "storage/backup.h"
#include "storage/database.h"
#include "schema/metadata.h"
#include "../test_tmp.h"

static int failures = 0;
#define ASSERT(c, m) do { if (!(c)) { printf("FAIL: %s\n", (m)); failures++; } \
                          else { printf("ok: %s\n", (m)); } } while (0)

static int test_object_store_fs(void) {
    char root[512];
    gv_test_make_temp_path(root, sizeof(root), "osroot", "");
    GV_ObjectStore *os = object_store_open_fs(root);
    ASSERT(os != NULL, "open fs object store");
    if (!os) return -1;

    const char *payload = "hello object store";
    size_t plen = strlen(payload);
    /* keys with '/' must stay inside the root (encoded, not path-traversal) */
    ASSERT(object_store_put(os, "a/b/c.bin", payload, plen) == 0, "put key with slashes");
    ASSERT(object_store_exists(os, "a/b/c.bin") == 1, "exists after put");
    ASSERT(object_store_exists(os, "missing") == 0, "absent key not present");

    void *got = NULL; size_t glen = 0;
    ASSERT(object_store_get(os, "a/b/c.bin", &got, &glen) == 0, "get key");
    ASSERT(got && glen == plen && memcmp(got, payload, plen) == 0, "payload round-trips");
    free(got);

    ASSERT(object_store_delete(os, "a/b/c.bin") == 0, "delete key");
    ASSERT(object_store_exists(os, "a/b/c.bin") == 0, "absent after delete");
    ASSERT(object_store_get(os, "a/b/c.bin", &got, &glen) != 0, "get after delete fails");

    object_store_destroy(os);
    return 0;
}

static int test_backup_through_object_store(void) {
    char root[512], dbpath[512];
    gv_test_make_temp_path(root, sizeof(root), "osbackup", "");
    gv_test_make_temp_path(dbpath, sizeof(dbpath), "osdb", ".gvdb");
    remove(dbpath);

    GV_ObjectStore *os = object_store_open_fs(root);
    ASSERT(os != NULL, "open object store for backup");
    if (!os) return -1;

    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "open source db");
    float v1[4] = {1, 0, 0, 0}, v2[4] = {0, 1, 0, 0};
    db_add_vector_with_metadata(db, v1, 4, "tag", "red");
    ASSERT(db_add_vector(db, v2, 4) == 0, "add v2");

    GV_BackupResult *br = backup_to_object_store(db, os, "snap1", NULL);
    ASSERT(br && br->success, "backup to object store succeeds");
    if (br) backup_result_free(br);
    db_close(db);
    ASSERT(object_store_exists(os, "snap1") == 1, "backup object present");

    GV_BackupResult *rr = backup_restore_from_object_store(os, "snap1", dbpath, NULL);
    ASSERT(rr && rr->success, "restore from object store succeeds");
    if (rr) backup_result_free(rr);

    GV_Database *db2 = db_open(dbpath, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db2 != NULL, "reopen restored db");
    if (db2) {
        GV_SearchResult res[2];
        int n = db_search(db2, v1, 1, res, GV_DISTANCE_EUCLIDEAN);
        ASSERT(n == 1, "search restored db");
        if (n == 1) {
            const char *t = vector_get_metadata(res[0].vector, "tag");
            ASSERT(t && strcmp(t, "red") == 0, "restored metadata via object store");
        }
        gv_search_results_free(res, (size_t)(n > 0 ? n : 0));
        db_close(db2);
    }
    object_store_destroy(os);
    remove(dbpath);
    return 0;
}

int main(void) {
    test_object_store_fs();
    test_backup_through_object_store();
    printf(failures ? "\nSOME TESTS FAILED (%d)\n" : "\nALL OBJECT-STORE TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
