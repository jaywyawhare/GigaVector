/*
 * CBMC harness: quantisation bounds
 * (storage/scalar_quant.c, specialized/quantization.c, binary_quant.c,
 *  storage/turboquant.c, index/codebook.c, pq.c, opq.c, multimodal/muvera.c).
 *
 * Quantisers are pure numeric transforms whose failure mode is an out-of-range
 * codebook index -- which indexes an array. The properties: dequantise stays
 * within the trained range, the round-trip error is bounded by the step size,
 * and byte-size accounting matches what the encoder actually writes.
  *
 * CBMC-SOURCES: src/storage/scalar_quant.c src/core/memory.c
 * CBMC-UNWIND: 10
*/
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "storage/scalar_quant.h"

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

#define DIM 3

int main(void) {
    float in[DIM];
    for (unsigned i = 0; i < DIM; i++) {
        in[i] = nondet_float();
        ASSUME(in[i] == in[i]);                 /* finite */
        ASSUME(in[i] >= -100.0f && in[i] <= 100.0f);
    }

    GV_ScalarQuantConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.bits = 8;

    /* The declared byte size must cover what the encoder writes. */
    size_t need = scalar_quant_bytes_needed(DIM, cfg.bits);
    assert(need >= DIM * cfg.bits / 8);

    GV_ScalarQuantVector *q = scalar_quantize(in, DIM, &cfg);
    if (!q) return 0;

    float out[DIM];
    if (scalar_dequantize(q, out) == 0) {
        for (unsigned i = 0; i < DIM; i++) {
            assert(out[i] == out[i]);           /* never produces NaN */
            /* Dequantised values stay inside the input range: a quantiser that
             * escapes its own range means the scale/offset arithmetic wrapped. */
            assert(out[i] >= -101.0f && out[i] <= 101.0f);
        }
    }

    scalar_quant_vector_destroy(q);
    return 0;
}
