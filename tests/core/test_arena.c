#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/arena.h"

#define ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1; \
        } \
    } while (0)

static int test_arena_alloc_and_reset(void) {
    GV_Arena arena;
    ASSERT(gv_arena_init(&arena, 128) == 0, "arena init");

    void *a = gv_arena_alloc(&arena, 16, 16);
    void *b = gv_arena_alloc(&arena, 32, 8);
    ASSERT(a != NULL && b != NULL, "allocations succeed");
    ASSERT(a != b, "distinct allocations");
    ASSERT(gv_arena_used(&arena) >= 48, "used bytes tracked");

    gv_arena_reset(&arena);
    ASSERT(gv_arena_used(&arena) == 0, "reset clears used");

    void *c = gv_arena_alloc(&arena, 8, 8);
    ASSERT(c == a, "reuse after reset starts from base");

    gv_arena_fini(&arena);
    return 0;
}

static int test_arena_static_backing(void) {
    uint8_t backing[256];
    GV_Arena arena;
    ASSERT(gv_arena_init_static(&arena, backing, sizeof(backing)) == 0, "static init");

    char *s = gv_arena_strdup(&arena, "gigavector");
    ASSERT(s != NULL, "strdup in static arena");
    ASSERT(strcmp(s, "gigavector") == 0, "strdup content");

    gv_arena_fini(&arena);
    ASSERT(backing[0] != 0 || s[0] == 'g', "backing untouched by fini");
    return 0;
}

static int test_arena_grow(void) {
    GV_Arena arena;
    ASSERT(gv_arena_init(&arena, 32) == 0, "small arena init");

    void *first = gv_arena_alloc(&arena, 24, 8);
    ASSERT(first != NULL, "first alloc");
    void *second = gv_arena_alloc(&arena, 24, 8);
    ASSERT(second != NULL, "grow on overflow");
    ASSERT(gv_arena_capacity(&arena) >= 48, "capacity grew");

    gv_arena_fini(&arena);
    return 0;
}

static int test_arena_calloc(void) {
    GV_Arena arena;
    ASSERT(gv_arena_init(&arena, 64) == 0, "arena init");

    int *values = (int *)gv_arena_calloc(&arena, 4, sizeof(int));
    ASSERT(values != NULL, "calloc");
    ASSERT(values[0] == 0 && values[3] == 0, "calloc zeroes");

    gv_arena_fini(&arena);
    return 0;
}

int main(void) {
    struct { const char *name; int (*fn)(void); } tests[] = {
        {"test_arena_alloc_and_reset", test_arena_alloc_and_reset},
        {"test_arena_static_backing", test_arena_static_backing},
        {"test_arena_grow", test_arena_grow},
        {"test_arena_calloc", test_arena_calloc},
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
