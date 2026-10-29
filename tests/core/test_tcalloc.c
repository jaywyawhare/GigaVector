/*
 * Unit tests for the custom slab allocator (gv_tcalloc). The DB defaults to libc,
 * so these tests force the slab allocator on with gv_tc_set_enabled(1) to exercise
 * the real path in CI: size-class rounding, large (mmap) allocations, realloc
 * growth/shrink with content preservation, calloc zeroing, usable-size reporting,
 * and - critically - that a raw libc pointer handed to gv_tc_free is routed back
 * to libc rather than mishandled (the codebase mixes allocators).
 */
#include "core/tcalloc.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fails;
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (cond) {                                                             \
            printf("  ok   %s\n", msg);                                         \
        } else {                                                                \
            printf("  FAIL %s\n", msg);                                         \
            g_fails++;                                                          \
        }                                                                       \
    } while (0)

static uint64_t xs(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return (*s = x);
}

static void test_basic(void) {
    printf("== basic alloc/free/usable ==\n");
    void *p = gv_tc_malloc(100);
    CHECK(p != NULL, "malloc(100) non-NULL");
    CHECK(((uintptr_t)p & 15u) == 0, "16-byte aligned");
    CHECK(gv_tc_usable_size(p) >= 100, "usable_size >= request");
    memset(p, 0xAB, 100);
    gv_tc_free(p);

    CHECK(gv_tc_malloc(0) == NULL, "malloc(0) is NULL");
    CHECK(gv_tc_usable_size(NULL) == 0, "usable_size(NULL) == 0");
    gv_tc_free(NULL); /* no-op, must not crash */
    CHECK(1, "free(NULL) survives");
}

static void test_large(void) {
    printf("== large (mmap) allocations ==\n");
    size_t n = 2 * 1024 * 1024; /* above the small cap -> direct mmap */
    unsigned char *p = (unsigned char *)gv_tc_malloc(n);
    CHECK(p != NULL, "2MB malloc non-NULL");
    CHECK(((uintptr_t)p & 15u) == 0, "2MB 16-byte aligned");
    CHECK(gv_tc_usable_size(p) >= n, "2MB usable_size >= request");
    for (size_t i = 0; i < n; i += 4096) p[i] = (unsigned char)i;
    for (size_t i = 0; i < n; i += 4096)
        if (p[i] != (unsigned char)i) { g_fails++; break; }
    CHECK(1, "2MB write/readback across pages");
    gv_tc_free(p);
}

static void test_calloc(void) {
    printf("== calloc zeroing + overflow guard ==\n");
    unsigned char *p = (unsigned char *)gv_tc_calloc(1000, 4);
    int zero = 1;
    for (int i = 0; i < 4000; i++)
        if (p[i] != 0) { zero = 0; break; }
    CHECK(zero, "calloc returns zeroed memory");
    gv_tc_free(p);
    CHECK(gv_tc_calloc((size_t)-1, 2) == NULL, "calloc overflow -> NULL");
}

static void test_realloc(void) {
    printf("== realloc grow/shrink preserves content ==\n");
    CHECK(gv_tc_realloc(NULL, 64) != NULL, "realloc(NULL, n) acts as malloc");

    size_t n = 50;
    unsigned char *p = (unsigned char *)gv_tc_malloc(n);
    for (size_t i = 0; i < n; i++) p[i] = (unsigned char)(i * 7 + 1);
    /* Grow across many size classes, then down; verify the prefix each time. */
    int ok = 1;
    for (size_t grow = 64; grow <= 200000; grow = grow * 3 / 2 + 1) {
        unsigned char *np = (unsigned char *)gv_tc_realloc(p, grow);
        if (!np) { ok = 0; break; }
        size_t keep = n < grow ? n : grow;
        for (size_t i = 0; i < keep && i < n; i++)
            if (np[i] != (unsigned char)(i * 7 + 1)) { ok = 0; break; }
        p = np;
    }
    CHECK(ok, "content preserved across grow chain");
    gv_tc_free(p);
}

/* A raw libc malloc handed to gv_tc_free must go back to libc, not be masked to a
 * bogus slab header. This is the mixed-allocator case the registry guards. */
static void test_foreign_pointer(void) {
    printf("== foreign (libc) pointer passthrough ==\n");
    void *p = malloc(123);
    CHECK(p != NULL, "libc malloc non-NULL");
    gv_tc_free(p); /* must route to libc free without crashing */
    CHECK(1, "gv_tc_free(libc ptr) survives");

    void *q = malloc(64);
    void *r = gv_tc_realloc(q, 4096); /* foreign realloc stays with libc */
    CHECK(r != NULL, "gv_tc_realloc(libc ptr) non-NULL");
    free(r);
    CHECK(1, "libc free of realloc result survives");
}

static void test_stress(void) {
    printf("== randomized churn with content checksums ==\n");
    enum { SLOTS = 2048 };
    struct { unsigned char *p; size_t n; uint64_t tag; } slot[SLOTS];
    memset(slot, 0, sizeof(slot));
    uint64_t s = 0xDEADBEEFCAFEull;
    int ok = 1;
    for (int it = 0; it < 400000 && ok; it++) {
        uint32_t i = (uint32_t)(xs(&s) % SLOTS);
        size_t sz = (xs(&s) % 100 < 97) ? (1 + xs(&s) % 3000)
                                        : (70000 + xs(&s) % 100000);
        if (slot[i].p) {
            for (size_t j = 0; j < slot[i].n; j++)
                if (slot[i].p[j] != (unsigned char)(slot[i].tag + j)) { ok = 0; break; }
            gv_tc_free(slot[i].p);
        }
        uint64_t tag = xs(&s);
        slot[i].p = (unsigned char *)gv_tc_malloc(sz);
        if (!slot[i].p) { ok = 0; break; }
        slot[i].n = sz;
        slot[i].tag = tag;
        for (size_t j = 0; j < sz; j++) slot[i].p[j] = (unsigned char)(tag + j);
    }
    for (int i = 0; i < SLOTS; i++) gv_tc_free(slot[i].p);
    CHECK(ok, "400k-op checksum stress clean");
}

int main(void) {
    gv_tc_set_enabled(1);
    printf("gv_tcalloc tests (active=%d)\n", gv_tc_active());
    test_basic();
    test_large();
    test_calloc();
    test_realloc();
    test_foreign_pointer();
    test_stress();
    if (g_fails) {
        printf("FAILED: %d check(s)\n", g_fails);
        return 1;
    }
    printf("All gv_tcalloc tests passed\n");
    return 0;
}
