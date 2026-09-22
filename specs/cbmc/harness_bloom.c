/*
 * CBMC harness: the defining property of a Bloom filter (src/core/bloom.c).
 *
 * A Bloom filter may report false POSITIVES but must never report a false
 * NEGATIVE. That one-sided error is the entire contract -- everything built on
 * top of it (segment skipping, existence checks) is only correct because of it.
 *
 *   cbmc --bounds-check --pointer-check --unwind 8 -I include \
 *        specs/cbmc/harness_bloom.c src/core/bloom.c ...
  *
 * CBMC-SOURCES: src/core/bloom.c src/core/memory.c
 * CBMC-UNWIND: 8
 * CBMC-SKIP: intractable for a structural reason, not a scope one -- it still
 *   times out at 1 item of 1 byte. bloom_optimal_bits/bloom_optimal_hashes
 *   (bloom.c:71,93) size the filter with double log/pow arithmetic, and
 *   bloom_hash_i then indexes with `% num_bits` on a value derived from that
 *   float computation. Exact IEEE-754 reasoning feeding a symbolic modulo is
 *   the same wall harness_distance.c and harness_geo.c hit. Concrete
 *   smoke-test only; core/bloom.c is classified `cbmc-pending`.
*/
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "core/bloom.h"

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

#define N_ITEMS 1
#define KEY_LEN 1

int main(void) {
    GV_BloomFilter *bf = bloom_create(4, 0.1);
    if (!bf) return 0;

    unsigned char keys[N_ITEMS][KEY_LEN];
    for (unsigned i = 0; i < N_ITEMS; i++)
        for (unsigned j = 0; j < KEY_LEN; j++)
            keys[i][j] = nondet_uchar();

    for (unsigned i = 0; i < N_ITEMS; i++)
        bloom_add(bf, keys[i], KEY_LEN);

    /* NO FALSE NEGATIVES: everything added must still be reported present,
     * for any key bytes whatsoever. */
    for (unsigned i = 0; i < N_ITEMS; i++)
        assert(bloom_check(bf, keys[i], KEY_LEN) == 1);

    /* Adding is idempotent with respect to membership. */
    bloom_add(bf, keys[0], KEY_LEN);
    assert(bloom_check(bf, keys[0], KEY_LEN) == 1);

    /* An empty filter reports nothing present -- no false positive from a
     * zeroed bit array. */
    GV_BloomFilter *empty = bloom_create(4, 0.1);
    if (empty) {
        unsigned char probe[KEY_LEN];
        for (unsigned j = 0; j < KEY_LEN; j++) probe[j] = nondet_uchar();
        assert(bloom_check(empty, probe, KEY_LEN) == 0);
        bloom_destroy(empty);
    }

    bloom_destroy(bf);
    return 0;
}
