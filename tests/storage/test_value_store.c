/*
 * test_value_store.c — WiscKey key-value store: put/get/update/delete,
 * reopen durability (index rebuilt from the log), and GC reclamation.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "storage/value_store.h"

static int failures = 0;
#define ASSERT(c, m) do { if (!(c)) { printf("FAIL: %s\n", (m)); failures++; } \
                          else { printf("ok: %s\n", (m)); } } while (0)

#define PATH "/tmp/gv_test_value_store.bin"

static int get_eq(GV_ValueStore *vs, uint64_t key, const char *expect) {
    void *buf = NULL; size_t len = 0;
    if (value_store_get(vs, key, &buf, &len) != 0) return 0;
    int ok = (len == strlen(expect)) && (len == 0 || memcmp(buf, expect, len) == 0);
    free(buf);
    return ok;
}

int main(void) {
    remove(PATH);

    /* ---- put / get / update / delete ---- */
    GV_ValueStore *vs = value_store_open(PATH);
    ASSERT(vs != NULL, "value_store_open");
    ASSERT(value_store_put(vs, 1, "one", 3) == 0, "put key 1");
    ASSERT(value_store_put(vs, 2, "two", 3) == 0, "put key 2");
    ASSERT(value_store_put(vs, 3, "three", 5) == 0, "put key 3");
    ASSERT(value_store_count(vs) == 3, "count == 3");
    ASSERT(get_eq(vs, 1, "one"), "get key 1 == one");
    ASSERT(get_eq(vs, 3, "three"), "get key 3 == three");

    /* update: replaces value, count unchanged, old value orphaned */
    ASSERT(value_store_put(vs, 2, "TWO-updated", 11) == 0, "update key 2");
    ASSERT(value_store_count(vs) == 3, "count still 3 after update");
    ASSERT(get_eq(vs, 2, "TWO-updated"), "get key 2 == updated");

    /* delete */
    ASSERT(value_store_delete(vs, 3) == 0, "delete key 3");
    ASSERT(value_store_count(vs) == 2, "count == 2 after delete");
    { void *b=NULL; size_t l=0; ASSERT(value_store_get(vs, 3, &b, &l) == -1, "get deleted key fails"); free(b); }
    ASSERT(value_store_delete(vs, 999) == -1, "delete absent key fails");

    /* many keys to force map growth */
    for (uint64_t k = 100; k < 200; k++) {
        char v[32]; int n = snprintf(v, sizeof v, "val-%llu", (unsigned long long)k);
        ASSERT(value_store_put(vs, k, v, (size_t)n) == 0 || k != k, "bulk put");
    }
    ASSERT(value_store_count(vs) == 102, "count == 102 after bulk");
    ASSERT(get_eq(vs, 150, "val-150"), "bulk get key 150");
    ASSERT(value_store_sync(vs) == 0, "sync");
    value_store_close(vs);

    /* ---- reopen: index rebuilt from the durable log ---- */
    vs = value_store_open(PATH);
    ASSERT(vs != NULL, "reopen");
    ASSERT(value_store_count(vs) == 102, "count survives reopen (rebuilt from log)");
    ASSERT(get_eq(vs, 1, "one"), "key 1 survives reopen");
    ASSERT(get_eq(vs, 2, "TWO-updated"), "updated key 2 survives reopen (last write wins)");
    ASSERT(get_eq(vs, 150, "val-150"), "bulk key survives reopen");
    { void *b=NULL; size_t l=0; ASSERT(value_store_get(vs, 3, &b, &l) == -1, "deleted key stays deleted after reopen"); free(b); }

    /* ---- GC reclaims dead records (updates + deletes) ---- */
    uint64_t size_before = value_store_disk_size(vs);
    ASSERT(value_store_gc(vs) == 0, "gc runs");
    uint64_t size_after = value_store_disk_size(vs);
    ASSERT(size_after < size_before, "log shrank after GC");
    ASSERT(value_store_count(vs) == 102, "live count unchanged by GC");
    /* All live values still correct after GC (offsets remapped). */
    ASSERT(get_eq(vs, 1, "one"), "key 1 correct after GC");
    ASSERT(get_eq(vs, 2, "TWO-updated"), "key 2 correct after GC");
    ASSERT(get_eq(vs, 150, "val-150"), "bulk key correct after GC");
    { void *b=NULL; size_t l=0; ASSERT(value_store_get(vs, 3, &b, &l) == -1, "deleted key still gone after GC"); free(b); }
    value_store_close(vs);

    /* ---- reopen after GC ---- */
    vs = value_store_open(PATH);
    ASSERT(vs != NULL && value_store_count(vs) == 102, "reopen after GC, 102 live");
    ASSERT(get_eq(vs, 2, "TWO-updated"), "key 2 correct after GC+reopen");
    value_store_close(vs);

    remove(PATH);
    printf(failures ? "\nSOME VALUE_STORE TESTS FAILED (%d)\n" : "\nALL VALUE_STORE TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
