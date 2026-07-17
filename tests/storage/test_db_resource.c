/**
 * test_db_resource.c — src/storage/db_resource.c limit + accounting checks.
 *
 * Exercises the public resource-limit API (storage/database.h):
 *   db_set_resource_limits / db_get_resource_limits
 *   db_check_resource_limits
 *   db_increment_concurrent_ops / db_decrement_concurrent_ops
 *   db_get_concurrent_operations / db_get_memory_usage
 *
 * Asserts a configured limit rejects an over-limit op and that counters track
 * correctly.
 */
#include <stdio.h>
#include <stdlib.h>

#include "storage/database.h"
/* db_increment_concurrent_ops / db_decrement_concurrent_ops are internal
 * cross-module helpers declared here (exported symbols, not in the public API). */
#include "storage/db_internal.h"

#define ASSERT(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); return -1; } \
} while (0)

#define DIM 4

/* max_vectors limit: a request that would exceed it is rejected; one that fits
 * is accepted; adding real vectors up to the cap then rejects the next add. */
static int test_max_vectors_limit(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open");

    GV_ResourceLimits lim = {0};
    lim.max_vectors = 3;
    ASSERT(db_set_resource_limits(db, &lim) == 0, "set limits");

    GV_ResourceLimits got = {0};
    db_get_resource_limits(db, &got);
    ASSERT(got.max_vectors == 3, "limit read back");

    /* With 0 vectors, adding 3 fits, adding 4 does not. */
    ASSERT(db_check_resource_limits(db, 3, 0) == 0, "3 fits under cap");
    ASSERT(db_check_resource_limits(db, 4, 0) == -1, "4 rejected over cap");

    /* Fill to the cap with real vectors, then the next add-check must reject. */
    for (int i = 0; i < 3; i++) {
        float v[DIM] = {(float)i, 1.0f, 2.0f, 3.0f};
        ASSERT(db_add_vector(db, v, DIM) == 0, "add within cap");
    }
    ASSERT(db_check_resource_limits(db, 1, 0) == -1, "one more over cap rejected after fill");
    ASSERT(db_check_resource_limits(db, 0, 0) == 0, "zero-add always ok");

    db_close(db);
    return 0;
}

/* max_memory_bytes limit: an additional_memory request over budget is rejected. */
static int test_max_memory_limit(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open");

    size_t base = db_get_memory_usage(db);

    GV_ResourceLimits lim = {0};
    /* Budget just above current usage so a large add trips it. */
    lim.max_memory_bytes = base + 1024;
    ASSERT(db_set_resource_limits(db, &lim) == 0, "set mem limit");

    ASSERT(db_check_resource_limits(db, 0, 512) == 0, "512 extra fits");
    ASSERT(db_check_resource_limits(db, 0, 100000) == -1, "huge extra rejected");

    db_close(db);
    return 0;
}

/* max_concurrent_operations: increment tracks; check rejects at the cap;
 * decrement releases; counter never underflows. */
static int test_concurrent_ops(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open");

    GV_ResourceLimits lim = {0};
    lim.max_concurrent_operations = 2;
    ASSERT(db_set_resource_limits(db, &lim) == 0, "set conc limit");

    ASSERT(db_get_concurrent_operations(db) == 0, "starts at 0");
    ASSERT(db_check_resource_limits(db, 0, 0) == 0, "under cap ok at 0");

    db_increment_concurrent_ops(db);
    ASSERT(db_get_concurrent_operations(db) == 1, "counter == 1");
    ASSERT(db_check_resource_limits(db, 0, 0) == 0, "under cap ok at 1");

    db_increment_concurrent_ops(db);
    ASSERT(db_get_concurrent_operations(db) == 2, "counter == 2");
    /* At cap: current >= max -> rejected. */
    ASSERT(db_check_resource_limits(db, 0, 0) == -1, "at cap rejected");

    db_decrement_concurrent_ops(db);
    ASSERT(db_get_concurrent_operations(db) == 1, "counter back to 1");
    ASSERT(db_check_resource_limits(db, 0, 0) == 0, "below cap ok again");

    /* Underflow guard: extra decrements must not wrap below 0. */
    db_decrement_concurrent_ops(db);
    db_decrement_concurrent_ops(db);
    ASSERT(db_get_concurrent_operations(db) == 0, "no underflow past 0");

    db_close(db);
    return 0;
}

/* Zero limits mean "unlimited": nothing is rejected. */
static int test_unlimited(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open");

    GV_ResourceLimits lim = {0}; /* all zero == unlimited */
    ASSERT(db_set_resource_limits(db, &lim) == 0, "set unlimited");
    ASSERT(db_check_resource_limits(db, 1000000, 1000000000) == 0, "unlimited accepts anything");

    db_close(db);
    return 0;
}

/* NULL-arg guards. */
static int test_null_guards(void) {
    ASSERT(db_check_resource_limits(NULL, 1, 1) == -1, "NULL db check rejected");
    GV_ResourceLimits lim = {0};
    ASSERT(db_set_resource_limits(NULL, &lim) == -1, "NULL db set rejected");
    ASSERT(db_get_concurrent_operations(NULL) == 0, "NULL db conc ops 0");
    ASSERT(db_get_memory_usage(NULL) == 0, "NULL db mem 0");
    /* These are void; just ensure they do not crash. */
    db_increment_concurrent_ops(NULL);
    db_decrement_concurrent_ops(NULL);
    return 0;
}

int main(void) {
    int rc = 0;
    rc |= test_max_vectors_limit();
    rc |= test_max_memory_limit();
    rc |= test_concurrent_ops();
    rc |= test_unlimited();
    rc |= test_null_guards();
    if (rc == 0) printf("All db_resource tests PASSED.\n");
    return rc != 0;
}
