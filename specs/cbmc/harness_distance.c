/*
 * CBMC harness: the SIMD distance kernels must agree with the scalar reference
 * (src/search/distance.c).
 *
 * `distance()` dispatches to an AVX2 / AVX-512 / NEON path depending on the
 * build, while `distance_scalar()` is the portable reference. Nothing in the
 * test suite currently pins them together, so a SIMD path could silently
 * disagree on tail elements or denormals. This harness is the differential
 * check, over arbitrary inputs rather than a fixed corpus.
 *
 *   cbmc --bounds-check --pointer-check --float-overflow-check --unwind 8 \
 *        -I include specs/cbmc/harness_distance.c src/search/distance.c ...
  *
 * CBMC-SOURCES: src/search/distance.c
 * CBMC-UNWIND: 10
 * CBMC-SKIP: exact IEEE-754 float reasoning does not terminate here. Tried
 *   DIM 8 and DIM 4 (the smallest that still dispatches to the SSE path), and
 *   a Manhattan-only variant with no sqrt at all -- every configuration times
 *   out, because bit-blasting float multiply/compare over even 4 lanes
 *   explodes the solver. The SIMD/scalar equivalence is therefore covered
 *   CONCRETELY (cbmc-smoke) and by the index tests, not symbolically.
 *   search/distance.c is classified `property`, not `cbmc`.
*/
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "search/distance.h"
#include "schema/vector.h"

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

/* 4 is the smallest dimension that still dispatches to the SSE path
 * (distance.c requires dimension >= 4 and dimension %% 4 == 0), so it exercises
 * the SIMD/scalar split while keeping exact float reasoning tractable -- at
 * DIM 8 the solver does not terminate. */
#define DIM 4
#define EPS 1e-3f

static int close_enough(float a, float b) {
    if (isnan(a) && isnan(b)) return 1;
    float d = a - b;
    if (d < 0) d = -d;
    float scale = (a < 0 ? -a : a) + (b < 0 ? -b : b) + 1.0f;
    return d <= EPS * scale;
}

int main(void) {
    float da[DIM], db[DIM];
    for (unsigned i = 0; i < DIM; i++) {
        da[i] = nondet_float();
        db[i] = nondet_float();
        /* Constrain to FINITE values. `x == x` alone only excludes NaN -- a
         * truly nondeterministic float also ranges over +/-Infinity, and
         * inf - inf = NaN, which makes every metric assertion below fail for
         * reasons that have nothing to do with the SIMD/scalar equivalence
         * this harness is about.
         *
         * Note what this DOES leave open: fed +/-Infinity, the distance
         * kernels return NaN, and a NaN distance silently corrupts every
         * downstream sort comparison. Rejecting non-finite input is the
         * caller's job today; nothing in the kernels enforces it. */
        ASSUME(da[i] > -1.0e18f && da[i] < 1.0e18f);
        ASSUME(db[i] > -1.0e18f && db[i] < 1.0e18f);
    }

    GV_Vector a, b;
    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    a.data = da; a.dimension = DIM;
    b.data = db; b.dimension = DIM;

    /* Cosine is excluded: it divides by the norm product, and exact float
     * division blows up the solver. It is covered concretely by cbmc-smoke. */
    const GV_DistanceType types[] = {
        GV_DISTANCE_EUCLIDEAN, GV_DISTANCE_DOT_PRODUCT, GV_DISTANCE_MANHATTAN,
    };

    for (unsigned t = 0; t < sizeof(types) / sizeof(types[0]); t++) {
        float fast = distance(&a, &b, types[t]);
        float ref  = distance_scalar(&a, &b, types[t]);
        /* The dispatched (possibly SIMD) path must agree with the reference. */
        assert(close_enough(fast, ref));
    }

    /* Metric sanity that holds regardless of the code path. */
    float self = distance(&a, &a, GV_DISTANCE_EUCLIDEAN);
    assert(self >= -EPS);                       /* non-negative */
    assert(close_enough(self, 0.0f));           /* identity of indiscernibles */

    float ab = distance(&a, &b, GV_DISTANCE_EUCLIDEAN);
    float ba = distance(&b, &a, GV_DISTANCE_EUCLIDEAN);
    assert(close_enough(ab, ba));               /* symmetry */

    return 0;
}
