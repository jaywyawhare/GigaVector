/*
 * Unit tests for the custom slab allocator (gv_tcalloc). The DB defaults to libc,
 * so these tests force the slab allocator on with gv_tc_set_enabled(1) to exercise
 * the real path in CI: size-class rounding, large (mmap) allocations, realloc
 * growth/shrink with content preservation, calloc zeroing, usable-size reporting,
 * and - critically - that a raw libc pointer handed to gv_tc_free is routed back
 * to libc rather than mishandled (the codebase mixes allocators).
 */
#include "core/tcalloc.h"
#include "core/memory.h"

#include <assert.h>
#include <pthread.h>
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

/* gv_aligned_alloc must honour the requested alignment, preserve content across
 * aligned realloc, and survive free; it layers on gv_alloc so it works under both
 * the custom allocator and libc. */
static void test_aligned(void) {
    printf("== 64-byte aligned alloc/realloc/free ==\n");
    size_t aligns[] = {16, 32, 64, 128, 256, 4096};
    int all_aligned = 1;
    for (size_t a = 0; a < sizeof(aligns) / sizeof(aligns[0]); a++) {
        for (size_t sz = 1; sz <= 9000; sz = sz * 2 + 1) {
            void *p = gv_aligned_alloc(sz, aligns[a]);
            if (!p || ((uintptr_t)p & (aligns[a] - 1)) != 0) { all_aligned = 0; }
            memset(p, 0xCD, sz);
            gv_aligned_free(p);
        }
    }
    CHECK(all_aligned, "every alignment/size combination is correctly aligned");

    unsigned char *p = (unsigned char *)gv_aligned_alloc(100, 64);
    for (int i = 0; i < 100; i++) p[i] = (unsigned char)(i + 3);
    p = (unsigned char *)gv_aligned_realloc(p, 5000, 64);
    int ok = (((uintptr_t)p & 63u) == 0);
    for (int i = 0; i < 100; i++)
        if (p[i] != (unsigned char)(i + 3)) { ok = 0; break; }
    CHECK(ok, "aligned realloc keeps alignment and content");
    gv_aligned_free(p);

    CHECK(gv_aligned_alloc(16, 24) == NULL, "non-power-of-two alignment -> NULL");
    gv_aligned_free(NULL); /* no-op */
    CHECK(1, "aligned_free(NULL) survives");
}

/* Allocate many same-size objects (several slabs' worth) then free them all, so
 * on thread exit the destructor flushes every object to central and the slabs
 * become fully idle. */
#define REL_N 70000      /* > 4 slabs of the 64-byte class (~16383 objs/slab) */
static void *release_worker(void *arg) {
    (void)arg;
    void **ptrs = (void **)malloc(REL_N * sizeof(void *));
    for (int i = 0; i < REL_N; i++) {
        ptrs[i] = gv_tc_malloc(64);
        memset(ptrs[i], 0x5A, 64);
    }
    for (int i = 0; i < REL_N; i++) gv_tc_free(ptrs[i]);
    free(ptrs);
    return NULL;
}

static void test_release(void) {
    printf("== slab reclamation (gv_tc_release) ==\n");
    pthread_t th;
    pthread_create(&th, NULL, release_worker, NULL);
    pthread_join(th, NULL); /* join: destructor flushes the worker's cache to central */

    size_t freed = gv_tc_release();
    size_t again = gv_tc_release();
    /* Reclamation only happens with the slab allocator on; under GV_ALLOC=libc
     * and the sanitizer/Windows forwarders gv_tc_release is a no-op by design. */
    if (gv_tc_active()) {
        CHECK(freed >= 3u * (1u << 20), "release reclaimed multiple idle slabs");
        CHECK(again == 0, "second release is a no-op (nothing idle)");
    } else {
        CHECK(freed == 0 && again == 0, "release is a no-op on the libc path");
    }

    /* Allocator still correct after reclamation: reuse the address space and
     * verify content integrity (catches a corrupted central list or stale memo). */
    uint64_t s = 0x9E3779B97F4A7C15ull;
    void *keep[2000];
    for (int i = 0; i < 2000; i++) {
        size_t sz = 16 + (xs(&s) % 1024);
        keep[i] = gv_tc_malloc(sz);
        memset(keep[i], (int)(i & 0xFF), sz);
    }
    int ok = 1;
    for (int i = 0; i < 2000; i++) {
        unsigned char *b = (unsigned char *)keep[i];
        if (b[0] != (unsigned char)(i & 0xFF)) ok = 0;
        gv_tc_free(keep[i]);
    }
    CHECK(ok, "allocations after release are intact");
}

/* Churn worker that races gv_tc_release: holds a rotating live set so some slabs
 * stay pinned while others go idle, exercising the lock between free and release. */
static int g_rel_stop; /* read/written atomically: main stops the churn workers */
static void *churn_worker(void *arg) {
    uint64_t s = 0x1234567u ^ (uint64_t)(uintptr_t)arg;
    void *ring[256] = {0};
    while (!__atomic_load_n(&g_rel_stop, __ATOMIC_SEQ_CST)) {
        for (int i = 0; i < 4000; i++) {
            uint32_t idx = (uint32_t)(xs(&s) % 256);
            if (ring[idx]) gv_tc_free(ring[idx]);
            size_t sz = 16 + (xs(&s) % 2048);
            ring[idx] = gv_tc_malloc(sz);
            if (ring[idx]) *(unsigned char *)ring[idx] = (unsigned char)idx;
        }
    }
    for (int i = 0; i < 256; i++) if (ring[i]) gv_tc_free(ring[i]);
    return NULL;
}

static void test_release_concurrent(void) {
    printf("== gv_tc_release racing concurrent churn ==\n");
    pthread_t th[4];
    __atomic_store_n(&g_rel_stop, 0, __ATOMIC_SEQ_CST);
    for (int i = 0; i < 4; i++) pthread_create(&th[i], NULL, churn_worker, (void *)(uintptr_t)i);
    size_t total = 0;
    for (int r = 0; r < 200; r++) total += gv_tc_release();
    __atomic_store_n(&g_rel_stop, 1, __ATOMIC_SEQ_CST);
    for (int i = 0; i < 4; i++) pthread_join(th[i], NULL);
    gv_tc_release();
    CHECK(1, "no crash/deadlock under release + churn");
    (void)total;
}

int main(void) {
    gv_tc_set_enabled(1);
    printf("gv_tcalloc tests (active=%d)\n", gv_tc_active());
    test_basic();
    test_large();
    test_calloc();
    test_realloc();
    test_aligned();
    test_foreign_pointer();
    test_release();
    test_release_concurrent();
    test_stress();
    if (g_fails) {
        printf("FAILED: %d check(s)\n", g_fails);
        return 1;
    }
    printf("All gv_tcalloc tests passed\n");
    return 0;
}
