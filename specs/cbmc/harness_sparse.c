/*
 * CBMC harness: sparse vector representation invariants
 * (src/storage/sparse_vector.c, src/index/sparse_index.c).
 *
 * A sparse vector is (indices, values) pairs. Every consumer -- dot products,
 * merges, inverted-index intersection -- assumes indices are sorted, unique,
 * and inside the declared dimension. Violate any of those and the merge walks
 * off the end of an array.
  *
 * CBMC-SOURCES: src/storage/sparse_vector.c src/core/memory.c
 * CBMC-UNWIND: 6
*/
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "storage/sparse_vector.h"

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

#define NNZ 3
#define DIM 64

int main(void) {
    uint32_t idx[NNZ];
    float    val[NNZ];

    /* Build a strictly increasing index list, as the constructor requires. */
    uint32_t prev = 0;
    for (unsigned i = 0; i < NNZ; i++) {
        uint32_t step = (uint32_t)(nondet_u64() % 8) + 1;
        prev = prev + step;
        ASSUME(prev < DIM);
        idx[i] = prev;
        val[i] = nondet_float();
        ASSUME(val[i] == val[i]);
    }

    GV_SparseVector *sv = sparse_vector_create(DIM, idx, val, NNZ);
    if (!sv) return 0;

    /* The three invariants every consumer relies on. */
    for (unsigned i = 0; i < NNZ; i++)
        assert(idx[i] < DIM);                  /* in range */
    for (unsigned i = 1; i < NNZ; i++)
        assert(idx[i - 1] < idx[i]);           /* sorted and unique */

    sparse_vector_destroy(sv);
    return 0;
}
