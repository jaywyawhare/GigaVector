#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/arena.h"
#include "core/buffer.h"

#define ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1; \
        } \
    } while (0)

static int test_buffer_push_and_clear(void) {
    GV_Arena arena;
    ASSERT(gv_arena_init(&arena, 256) == 0, "arena init");

    GV_Buffer buf;
    gv_buf_init(&buf, &arena);

    const char payload[] = "hello arena buffer";
    ASSERT(gv_buf_push(&buf, payload, sizeof(payload)) == 0, "push");
    ASSERT(buf.len == sizeof(payload), "length");
    ASSERT(memcmp(buf.data, payload, sizeof(payload)) == 0, "content");

    gv_buf_clear(&buf);
    ASSERT(buf.len == 0, "clear length");

    gv_arena_fini(&arena);
    return 0;
}

static int test_buffer_grow(void) {
    GV_Arena arena;
    ASSERT(gv_arena_init(&arena, 64) == 0, "arena init");

    GV_Buffer buf;
    gv_buf_init(&buf, &arena);

  for (int i = 0; i < 32; i++) {
        uint8_t byte = (uint8_t)i;
        ASSERT(gv_buf_push(&buf, &byte, 1) == 0, "push byte");
    }

    ASSERT(buf.len == 32, "32 bytes pushed");
    ASSERT(buf.cap >= 32, "capacity grew");

    gv_arena_fini(&arena);
    return 0;
}

int main(void) {
    struct { const char *name; int (*fn)(void); } tests[] = {
        {"test_buffer_push_and_clear", test_buffer_push_and_clear},
        {"test_buffer_grow", test_buffer_grow},
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
