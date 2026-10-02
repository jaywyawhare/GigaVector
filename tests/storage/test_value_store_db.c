/*
 * test_value_store_db.c — DB-level WiscKey value store integration:
 * enable, put/get/delete/gc via the GV_Database API, and clean teardown.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "storage/database.h"
#include "../test_tmp.h"

static int failures = 0;
#define ASSERT(c, m) do { if (!(c)) { printf("FAIL: %s\n", (m)); failures++; } \
                          else { printf("ok: %s\n", (m)); } } while (0)

static char PATH[512];

int main(void) {
    gv_test_make_temp_path(PATH, sizeof(PATH), "gv_test_vs_db", ".bin");
    remove(PATH);
    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open");

    /* Not enabled yet: ops fail cleanly. */
    ASSERT(db_value_store_put(db, 1, "x", 1) == -1, "put fails before enable");

    ASSERT(db_value_store_enable(db, PATH) == 0, "enable value store");
    ASSERT(db_value_store_put(db, 1, "hello", 5) == 0, "put key 1");
    ASSERT(db_value_store_put(db, 2, "world!!", 7) == 0, "put key 2");

    void *buf = NULL; size_t len = 0;
    ASSERT(db_value_store_get(db, 1, &buf, &len) == 0 && len == 5 && memcmp(buf, "hello", 5) == 0, "get key 1");
    free(buf);

    /* update + delete + gc */
    ASSERT(db_value_store_put(db, 1, "HELLO-2", 7) == 0, "update key 1");
    ASSERT(db_value_store_delete(db, 2) == 0, "delete key 2");
    ASSERT(db_value_store_gc(db) == 0, "gc");

    buf = NULL; len = 0;
    ASSERT(db_value_store_get(db, 1, &buf, &len) == 0 && len == 7 && memcmp(buf, "HELLO-2", 7) == 0, "key 1 correct post-gc");
    free(buf);
    { void *b=NULL; size_t l=0; ASSERT(db_value_store_get(db, 2, &b, &l) == -1, "key 2 gone post-gc"); free(b); }

    db_close(db);   /* must free the value store without leaks */
    remove(PATH);
    printf(failures ? "\nSOME VS-DB TESTS FAILED (%d)\n" : "\nALL VS-DB TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
