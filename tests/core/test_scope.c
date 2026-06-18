#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/arena.h"
#include "core/scope.h"
#include "../test_tmp.h"

#define ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1; \
        } \
    } while (0)

static int test_nested_arena_scopes(void) {
    int outer_ran = 0;
    int inner_ran = 0;

    GV_WITH_ARENA(outer, 128) {
        outer_ran = 1;
        GV_WITH_ARENA(inner, 64) {
            inner_ran = 1;
            void *p = gv_arena_alloc(&inner, 8, 8);
            ASSERT(p != NULL, "inner alloc");
        }
        ASSERT(gv_arena_used(&outer) == 0, "outer arena untouched");
    }

    ASSERT(outer_ran == 1 && inner_ran == 1, "both scopes ran");
    return 0;
}

static int early_return_helper(int flag) {
    GV_WITH_ARENA(scratch, 64) {
        void *p = gv_arena_alloc(&scratch, 16, 8);
        ASSERT(p != NULL, "alloc before return");
        if (flag) {
            return 0;
        }
    }
    return -1;
}

static int test_early_return_unwinds(void) {
    ASSERT(early_return_helper(1) == 0, "early return path");
    return 0;
}

static int test_with_db_scope(void) {
    char path[512];
    ASSERT(gv_test_make_temp_path(path, sizeof(path), "gv_scope_db", ".bin") == 0,
           "temp path");
    gv_test_remove_db(path);

    int ran = 0;
    GV_WITH_DB(db, path, 4, GV_INDEX_TYPE_FLAT) {
        ran = 1;
        ASSERT(db != NULL, "db should open");
    }

    gv_test_remove_db(path);
    ASSERT(ran == 1, "body ran");
    return 0;
}

static int test_bytes_heap_copy(void) {
    const char msg[] = "export";
    GV_Bytes bytes = gv_bytes_copy_heap(msg, sizeof(msg));
    ASSERT(bytes.kind == GV_OWN_HEAP, "heap owned");
    ASSERT(bytes.len == sizeof(msg), "length");
    ASSERT(memcmp(bytes.data, msg, sizeof(msg)) == 0, "payload");
    gv_bytes_release(&bytes);
    ASSERT(bytes.data == NULL, "released");
    return 0;
}

static int test_tls_arena_reuse(void) {
    GV_Arena *a = gv_tls_arena();
    ASSERT(a != NULL, "tls arena");
    void *p = gv_arena_alloc(a, 32, 8);
    ASSERT(p != NULL, "tls alloc");
    ASSERT(gv_arena_used(a) >= 32, "tls used");

    gv_tls_arena_reset();
    ASSERT(gv_arena_used(a) == 0, "tls reset");

    void *q = gv_arena_alloc(a, 16, 8);
    ASSERT(q == p, "tls reuse from base");
    return 0;
}

int main(void) {
    struct { const char *name; int (*fn)(void); } tests[] = {
        {"test_nested_arena_scopes", test_nested_arena_scopes},
        {"test_early_return_unwinds", test_early_return_unwinds},
        {"test_with_db_scope", test_with_db_scope},
        {"test_bytes_heap_copy", test_bytes_heap_copy},
        {"test_tls_arena_reuse", test_tls_arena_reuse},
    };

    int passed = 0;
    for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); i++) {
        if (tests[i].fn() == 0) {
            printf("  OK   %s\n", tests[i].name);
            passed++;
        }
    }

    int total = (int)(sizeof(tests) / sizeof(tests[0]));
    printf("\n%d/%d tests passed\n", passed, total);
    return passed == total ? 0 : 1;
}
