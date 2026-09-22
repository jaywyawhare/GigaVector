/*
 * CBMC harness: arena allocator pointer arithmetic
 * (src/core/arena.c, buffer.c, memory.c, scope.c).
 *
 * Arena bumps are where alignment and size arithmetic silently overflow. The
 * properties: every returned pointer is aligned as requested, lies inside the
 * arena, and no two live allocations overlap.
 *
 * MEMORY.md records that db_search() resets the TLS arena, so result buffers
 * must be heap-allocated -- that lifetime rule is a Quint/API concern, but the
 * arithmetic below is what CBMC can settle outright.
  *
 * CBMC-SOURCES: src/core/arena.c
 * CBMC-UNWIND: 6
*/
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <stdint.h>
#include "core/arena.h"

/* Dual-mode nondeterminism: symbolic under CBMC, concrete otherwise so the
 * harness still compiles and runs as an ordinary test (`make cbmc-smoke`). */
#if defined(__CPROVER__) || defined(__CPROVER)
uint64_t nondet_u64(void);
float    nondet_float(void);
size_t   nondet_size(void);
unsigned char nondet_uchar(void);
#  define ASSUME(c) __CPROVER_assume(c)
#else
static uint64_t g_seed = 0x9E3779B97F4A7C15ULL;
static uint64_t nondet_u64(void) {
    g_seed ^= g_seed << 13; g_seed ^= g_seed >> 7; g_seed ^= g_seed << 17;
    return g_seed;
}
static float nondet_float(void) {
    return (float)((int)(nondet_u64() % 2000) - 1000) / 100.0f;
}
static size_t nondet_size(void) { return (size_t)(nondet_u64() % 8); }
static unsigned char nondet_uchar(void) { return (unsigned char)(nondet_u64() & 0xFF); }
#  define ASSUME(c) do { if (!(c)) return 0; } while (0)
#endif

#define BACKING 256

int main(void) {
    unsigned char backing[BACKING];
    GV_Arena arena;
    if (gv_arena_init_static(&arena, backing, BACKING) != 0) return 0;

    size_t s1 = nondet_size();
    size_t s2 = nondet_size();
    ASSUME(s1 <= 32 && s2 <= 32);

    /* Alignments are powers of two, as the API requires. */
    size_t al1 = 1u << (nondet_u64() % 5);   /* 1,2,4,8,16 */
    size_t al2 = 1u << (nondet_u64() % 5);

    unsigned char *p1 = (unsigned char *)gv_arena_alloc(&arena, s1, al1);
    unsigned char *p2 = (unsigned char *)gv_arena_alloc(&arena, s2, al2);

    if (p1) {
        assert(((uintptr_t)p1 % al1) == 0);          /* honours alignment */
        assert(p1 >= backing);                       /* inside the arena */
        assert(p1 + s1 <= backing + BACKING);        /* and does not run past */
    }
    if (p2) {
        assert(((uintptr_t)p2 % al2) == 0);
        assert(p2 >= backing);
        assert(p2 + s2 <= backing + BACKING);
    }
    /* Two live allocations never overlap. */
    if (p1 && p2 && s1 > 0 && s2 > 0)
        assert(p1 + s1 <= p2 || p2 + s2 <= p1);

    /* used() never exceeds capacity(), whatever the request pattern. */
    assert(gv_arena_used(&arena) <= gv_arena_capacity(&arena));

    /* Reset returns the arena to empty without freeing the backing store. */
    gv_arena_reset(&arena);
    assert(gv_arena_used(&arena) == 0);

    /* A request larger than the arena must fail, not wrap around.
     * SIZE_MAX/2 rather than (size_t)-1/2: the latter is a signed-to-unsigned
     * conversion that --conversion-check flags in the harness itself. */
    void *huge = gv_arena_alloc(&arena, SIZE_MAX / 2, 8);
    assert(huge == NULL);

    gv_arena_fini(&arena);
    return 0;
}
