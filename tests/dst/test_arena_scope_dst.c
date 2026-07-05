#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/arena.h"
#include "core/scope.h"

#define ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1; \
        } \
    } while (0)

static unsigned gv_dst_rand_state = 0xC0FFEEu;

static unsigned gv_dst_rand(void) {
    gv_dst_rand_state = gv_dst_rand_state * 1103515245u + 12345u;
    return (gv_dst_rand_state >> 16) & 0x7FFFu;
}

static int dst_random_early_exit(int depth) {
    GV_WITH_ARENA(scratch, 32 * 1024) {
        void *p = gv_arena_alloc(&scratch, 64 + (size_t)gv_dst_rand(), 8);
        ASSERT(p != NULL, "arena alloc");
        if (depth > 0) {
            int rc = dst_random_early_exit(depth - 1);
            if (rc != 0) {
                return rc;
            }
        }
        if ((gv_dst_rand() % 3u) == 0u) {
            return 0;
        }
    }
    return 0;
}

static int test_random_scope_exits(void) {
    for (int i = 0; i < 64; ++i) {
        int depth = (int)(gv_dst_rand() % 4u);
        ASSERT(dst_random_early_exit(depth) == 0, "random scope exit");
    }
    return 0;
}

int main(void) {
    if (test_random_scope_exits() != 0) {
        return 1;
    }
    fprintf(stderr, "OK: arena scope dst\n");
    return 0;
}
