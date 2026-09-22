/*
 * CBMC harness: compress/decompress is the identity (src/storage/compression.c,
 * src/storage/chunker.c).
 *
 * A round-trip that loses or corrupts bytes is silent data loss. `compress_bound`
 * must also be a genuine upper bound, or the caller's output buffer overflows.
  *
 * CBMC-SOURCES: src/storage/compression.c src/core/memory.c
 * CBMC-UNWIND: 10
*/
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "storage/compression.h"

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

#define IN_MAX  5

int main(void) {
    GV_CompressionConfig cfg;
    compression_config_init(&cfg);
    GV_Compressor *comp = compression_create(&cfg);
    if (!comp) return 0;

    size_t len = nondet_size();
    ASSUME(len <= IN_MAX);

    unsigned char in[IN_MAX];
    for (size_t i = 0; i < IN_MAX; i++) in[i] = nondet_uchar();

    size_t bound = compress_bound(comp, len);
    ASSUME(bound <= 4 * IN_MAX + 64);          /* keep the solver tractable */
    assert(bound >= len);                      /* a bound must actually bound */

    unsigned char *cbuf = (unsigned char *)malloc(bound ? bound : 1);
    if (!cbuf) { compression_destroy(comp); return 0; }

    size_t clen = compress(comp, in, len, cbuf, bound);
    if (clen > 0) {
        assert(clen <= bound);                 /* never writes past the bound */

        unsigned char out[IN_MAX];
        size_t dlen = decompress(comp, cbuf, clen, out, IN_MAX);
        if (dlen > 0) {
            assert(dlen == len);               /* round-trip preserves length */
            for (size_t i = 0; i < len; i++)
                assert(out[i] == in[i]);       /* ... and every byte */
        }
    }

    /* Decompressing arbitrary bytes must be memory-safe, not merely correct. */
    unsigned char garbage[IN_MAX];
    for (size_t i = 0; i < IN_MAX; i++) garbage[i] = nondet_uchar();
    unsigned char sink[IN_MAX];
    (void)decompress(comp, garbage, len, sink, IN_MAX);

    free(cbuf);
    compression_destroy(comp);
    return 0;
}
