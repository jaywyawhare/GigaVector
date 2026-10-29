/*
 * bench_alloc: correctness stress + throughput benchmark for gv_tcalloc.
 *
 * The correctness pass drives a random malloc/realloc/free mix with content
 * checksums so any relocation that loses data, any overlap between live blocks,
 * or any free-list corruption shows up as a mismatch rather than a late crash.
 * The throughput pass times the hot patterns a vector DB actually hits (small
 * churn, realloc growth, multi-thread fan-out) against libc malloc so the
 * speedup is measured, not assumed.
 *
 *   Usage: bench_alloc [seconds] [threads]
 */
#include "core/tcalloc.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* Deterministic xorshift so a failure reproduces from the same seed. */
static uint64_t xs(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return (*s = x);
}

/* Byte pattern keyed by (tag,size) so a block's contents are self-describing. */
static void fill(unsigned char *p, size_t n, uint64_t tag) {
    for (size_t i = 0; i < n; i++) p[i] = (unsigned char)(tag + i * 131u);
}
static int check(const unsigned char *p, size_t n, uint64_t tag) {
    for (size_t i = 0; i < n; i++)
        if (p[i] != (unsigned char)(tag + i * 131u)) return 0;
    return 1;
}

#define SLOTS 4096

/* Random malloc/realloc/free churn with per-slot checksums. Returns 0 on OK. */
static int stress(uint64_t seed, int iters) {
    struct { unsigned char *p; size_t n; uint64_t tag; } slot[SLOTS];
    memset(slot, 0, sizeof(slot));
    uint64_t s = seed ? seed : 1;

    for (int it = 0; it < iters; it++) {
        uint32_t i = (uint32_t)(xs(&s) % SLOTS);
        uint32_t op = (uint32_t)(xs(&s) % 10);
        /* Bias sizes small (DB reality) but include large (mmap) occasionally. */
        size_t sz = (xs(&s) % 100 < 98) ? (1 + xs(&s) % 4096)
                                        : (65536 + xs(&s) % (128 * 1024));

        if (slot[i].p && !check(slot[i].p, slot[i].n, slot[i].tag)) {
            fprintf(stderr, "stress: content mismatch slot %u (n=%zu)\n", i, slot[i].n);
            return 1;
        }

        if (op < 4) { /* alloc / replace */
            gv_tc_free(slot[i].p);
            uint64_t tag = xs(&s);
            unsigned char *p = (unsigned char *)gv_tc_malloc(sz);
            if (!p) { fprintf(stderr, "stress: malloc(%zu) failed\n", sz); return 1; }
            fill(p, sz, tag);
            slot[i] = (typeof(slot[i])){p, sz, tag};
        } else if (op < 7) { /* realloc (grow or shrink), preserving overlap */
            unsigned char *p = (unsigned char *)gv_tc_realloc(slot[i].p, sz);
            if (!p) { fprintf(stderr, "stress: realloc(%zu) failed\n", sz); return 1; }
            size_t keep = slot[i].p ? (slot[i].n < sz ? slot[i].n : sz) : 0;
            if (keep && !check(p, keep, slot[i].tag)) {
                fprintf(stderr, "stress: realloc lost data slot %u\n", i);
                return 1;
            }
            uint64_t tag = xs(&s);
            fill(p, sz, tag);
            slot[i] = (typeof(slot[i])){p, sz, tag};
        } else { /* free */
            gv_tc_free(slot[i].p);
            slot[i] = (typeof(slot[i])){0, 0, 0};
        }
    }
    for (uint32_t i = 0; i < SLOTS; i++) {
        if (slot[i].p && !check(slot[i].p, slot[i].n, slot[i].tag)) {
            fprintf(stderr, "stress: final content mismatch slot %u\n", i);
            return 1;
        }
        gv_tc_free(slot[i].p);
    }
    return 0;
}

struct stress_arg { uint64_t seed; int iters; int rc; };
static void *stress_thread(void *a) {
    struct stress_arg *sa = (struct stress_arg *)a;
    sa->rc = stress(sa->seed, sa->iters);
    return NULL;
}

/* ---- throughput ---------------------------------------------------------- */

struct bench_arg { int use_libc; double secs; uint64_t ops; };

#define RING_SMALL 256
#define RING_WIDE  8192

/* Small live set: fits inside glibc's per-bin tcache, so both stay lock-free. */
static void *churn_small(void *a) {
    struct bench_arg *b = (struct bench_arg *)a;
    void *ring[RING_SMALL];
    memset(ring, 0, sizeof(ring));
    uint64_t s = 0x9e3779b97f4a7c15ull ^ (uint64_t)(uintptr_t)a;
    uint64_t ops = 0;
    double end = now_s() + b->secs;
    while (now_s() < end) {
        for (int k = 0; k < 20000; k++) {
            uint32_t i = (uint32_t)(xs(&s) & (RING_SMALL - 1));
            size_t sz = 16 + (xs(&s) % 1024);
            if (b->use_libc) { free(ring[i]); ring[i] = malloc(sz); }
            else { gv_tc_free(ring[i]); ring[i] = gv_tc_malloc(sz); }
            if (ring[i]) *(volatile char *)ring[i] = 1;
            ops++;
        }
    }
    for (int i = 0; i < RING_SMALL; i++) { if (b->use_libc) free(ring[i]); else gv_tc_free(ring[i]); }
    b->ops = ops;
    return NULL;
}

/* Wide live set (DB-like: many live same-size objects churned under load). This
 * overflows glibc's small per-bin tcache into the locked arena, which is the
 * regime the slab cache's deep per-thread free lists are built for. */
static void *churn_wide(void *a) {
    struct bench_arg *b = (struct bench_arg *)a;
    void **ring = (void **)calloc(RING_WIDE, sizeof(void *));
    uint64_t s = 0x2545f4914f6cdd1dull ^ (uint64_t)(uintptr_t)a;
    uint64_t ops = 0;
    double end = now_s() + b->secs;
    while (now_s() < end) {
        for (int k = 0; k < 20000; k++) {
            uint32_t i = (uint32_t)(xs(&s) & (RING_WIDE - 1));
            size_t sz = 16 + (xs(&s) % 512);
            if (b->use_libc) { free(ring[i]); ring[i] = malloc(sz); }
            else { gv_tc_free(ring[i]); ring[i] = gv_tc_malloc(sz); }
            if (ring[i]) *(volatile char *)ring[i] = 1;
            ops++;
        }
    }
    for (uint32_t i = 0; i < RING_WIDE; i++) { if (b->use_libc) free(ring[i]); else gv_tc_free(ring[i]); }
    free(ring);
    b->ops = ops;
    return NULL;
}

static double run_pool(int use_libc, double secs, int threads, void *(*fn)(void *)) {
    pthread_t th[64];
    struct bench_arg args[64];
    double t0 = now_s();
    for (int i = 0; i < threads; i++) {
        args[i] = (struct bench_arg){use_libc, secs, 0};
        pthread_create(&th[i], NULL, fn, &args[i]);
    }
    uint64_t total = 0;
    for (int i = 0; i < threads; i++) { pthread_join(th[i], NULL); total += args[i].ops; }
    double dt = now_s() - t0;
    return (double)total / dt / 1e6; /* Mops/s */
}

int main(int argc, char **argv) {
    /* Force the slab allocator on for this process so the benchmark always
     * exercises the custom path regardless of the compiled-in default. */
    gv_tc_set_enabled(1);

    double secs = argc > 1 ? atof(argv[1]) : 1.0;
    int threads = argc > 2 ? atoi(argv[2]) : 4;
    if (threads < 1) threads = 1;
    if (threads > 64) threads = 64;

    printf("gv_tcalloc active: %s\n", gv_tc_active() ? "yes" : "no (libc/sanitizer)");

    printf("\n== correctness stress ==\n");
    if (stress(0xC0FFEE, 600000)) { printf("FAIL single-thread stress\n"); return 1; }
    printf("  single-thread 600k ops: OK\n");
    {
        pthread_t th[8];
        struct stress_arg sa[8];
        for (int i = 0; i < 8; i++) {
            sa[i] = (struct stress_arg){0x1234 + i * 7919u, 200000, 0};
            pthread_create(&th[i], NULL, stress_thread, &sa[i]);
        }
        int bad = 0;
        for (int i = 0; i < 8; i++) { pthread_join(th[i], NULL); bad |= sa[i].rc; }
        if (bad) { printf("FAIL multi-thread stress\n"); return 1; }
        printf("  8-thread 200k ops each: OK\n");
    }

    printf("\n== throughput (%.1fs x %d threads) ==\n", secs, threads);
    double tc1 = run_pool(0, secs, 1, churn_small);
    double lc1 = run_pool(1, secs, 1, churn_small);
    printf("  small churn  1T : gv=%.1f Mops/s  libc=%.1f Mops/s  (%.2fx)\n",
           tc1, lc1, tc1 / lc1);
    double tcN = run_pool(0, secs, threads, churn_small);
    double lcN = run_pool(1, secs, threads, churn_small);
    printf("  small churn %2dT : gv=%.1f Mops/s  libc=%.1f Mops/s  (%.2fx)\n",
           threads, tcN, lcN, tcN / lcN);

    double wc1 = run_pool(0, secs, 1, churn_wide);
    double wl1 = run_pool(1, secs, 1, churn_wide);
    printf("  wide churn   1T : gv=%.1f Mops/s  libc=%.1f Mops/s  (%.2fx)\n",
           wc1, wl1, wc1 / wl1);
    double wcN = run_pool(0, secs, threads, churn_wide);
    double wlN = run_pool(1, secs, threads, churn_wide);
    printf("  wide churn  %2dT : gv=%.1f Mops/s  libc=%.1f Mops/s  (%.2fx)\n",
           threads, wcN, wlN, wcN / wlN);
    return 0;
}
