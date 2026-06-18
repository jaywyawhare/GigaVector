#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/memory.h"
#include "storage/database.h"

#define ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1; \
        } \
    } while (0)

static int test_db_alloc_tracked(void) {
    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db open");

    void *a = gv_db_alloc(db, 64);
    void *b = gv_db_calloc(db, 4, 8);
    ASSERT(a != NULL && b != NULL, "db alloc succeeds");
    ASSERT(db->memory_pool.owned_count == 2, "tracked blocks");
    ASSERT(db->memory_pool.pool_bytes == 64 + 32, "pool bytes");

    gv_db_free(db, a);
    ASSERT(db->memory_pool.owned_count == 1, "free removes tracking");

    db_close(db);
    return 0;
}

static int test_db_alloc_limit(void) {
    GV_Database *db = db_open(NULL, 2, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db open");

    GV_ResourceLimits limits = {0};
    limits.max_memory_bytes = 32;
    ASSERT(db_set_resource_limits(db, &limits) == 0, "set limits");

    void *big = gv_db_alloc(db, 64);
    ASSERT(big == NULL, "alloc over limit fails");

    db_close(db);
    return 0;
}

int main(void) {
    struct {
        const char *name;
        int (*fn)(void);
    } tests[] = {
        {"test_db_alloc_tracked", test_db_alloc_tracked},
        {"test_db_alloc_limit", test_db_alloc_limit},
    };

    for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); ++i) {
        fprintf(stderr, "RUN %s\n", tests[i].name);
        if (tests[i].fn() != 0) {
            return 1;
        }
    }
    fprintf(stderr, "OK: memory tests passed\n");
    return 0;
}
