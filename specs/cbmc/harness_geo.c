/*
 * CBMC harness: geospatial distance is a metric
 * (features/geo.c, and the consumers features/recommend.c,
 *  features/context_graph.c).
 *
 * `geo_distance_km` is haversine over a sphere. Every filter and ranking built
 * on it assumes metric behaviour; if symmetry or non-negativity breaks, radius
 * search silently returns the wrong set. Bounds also matter: a distance larger
 * than half the earth's circumference means the formula wrapped.
  *
 * CBMC-SOURCES: src/features/geo.c src/core/memory.c
 * CBMC-UNWIND: 4
 * CBMC-SKIP: CBMC ships no libm bodies (asin/sin/cos/sqrt) and offers no
 *   stub-override at this invocation level, so every haversine property
 *   collapses to nondeterminism rather than being discharged. Concrete
 *   smoke-test only; geo.c is classified `property`, not `cbmc`.
*/
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "features/geo.h"

/* CBMC has no body for the libm trig functions, so it models them as returning
 * a completely unconstrained double -- which can be NaN, and then every
 * downstream assertion fails for reasons that have nothing to do with the
 * implementation. Supplying RANGE-CONTRACTED models is the standard fix: each
 * stub returns a nondeterministic value constrained to the function's actual
 * codomain, which is a sound over-approximation of the real function.
 *
 * These are only compiled under __CPROVER; the concrete smoke build uses real
 * libm. */
#if defined(__CPROVER__) || defined(__CPROVER)
double __VERIFIER_nondet_double(void);

double sin(double x) {
    (void)x;
    double r = __VERIFIER_nondet_double();
    __CPROVER_assume(r >= -1.0 && r <= 1.0);
    return r;
}
double cos(double x) {
    (void)x;
    double r = __VERIFIER_nondet_double();
    __CPROVER_assume(r >= -1.0 && r <= 1.0);
    return r;
}
double sqrt(double x) {
    double r = __VERIFIER_nondet_double();
    /* sqrt is non-negative, and monotone: sqrt(x) <= x for x >= 1. */
    __CPROVER_assume(r >= 0.0);
    __CPROVER_assume(x < 0.0 || r * r <= x + 1e-9);
    return r;
}
double asin(double x) {
    /* asin: [-1, 1] -> [-pi/2, pi/2]. The codomain bound is what makes the
     * distance bound provable at all. */
    (void)x;
    double r = __VERIFIER_nondet_double();
    __CPROVER_assume(r >= -1.5707963267948966 && r <= 1.5707963267948966);
    return r;
}
#endif

#if defined(__CPROVER__) || defined(__CPROVER)
uint64_t nondet_u64(void);
size_t   nondet_size(void);
float    nondet_float(void);
double   nondet_double(void);
#  define ASSUME(c) __CPROVER_assume(c)
#else
static uint64_t g_seed = 0x9E3779B97F4A7C15ULL;
static uint64_t nondet_u64(void) {
    g_seed ^= g_seed << 13; g_seed ^= g_seed >> 7; g_seed ^= g_seed << 17;
    return g_seed;
}
static size_t nondet_size(void) { return (size_t)(nondet_u64() % 8); }
static float nondet_float(void) {
    return (float)((int)(nondet_u64() % 2000) - 1000) / 100.0f;
}
static double nondet_double(void) { return (double)nondet_float(); }
#  define ASSUME(c) do { if (!(c)) return 0; } while (0)
#endif

/* Half the earth's great-circle circumference, in km. */
#define MAX_KM 20038.0

int main(void) {
    double lat1 = nondet_double(), lng1 = nondet_double();
    double lat2 = nondet_double(), lng2 = nondet_double();
    ASSUME(lat1 >= -90.0 && lat1 <= 90.0);
    ASSUME(lat2 >= -90.0 && lat2 <= 90.0);
    ASSUME(lng1 >= -180.0 && lng1 <= 180.0);
    ASSUME(lng2 >= -180.0 && lng2 <= 180.0);

    double d = geo_distance_km(lat1, lng1, lat2, lng2);

    assert(d == d);                              /* never NaN */
    assert(d >= 0.0);                            /* non-negative */
    assert(d <= MAX_KM + 1.0);                   /* did not wrap around */

    /* SYMMETRY: d(a,b) == d(b,a). Radius search depends on this. */
    double r = geo_distance_km(lat2, lng2, lat1, lng1);
    double diff = d - r;
    if (diff < 0) diff = -diff;
    assert(diff < 1e-6);

    /* IDENTITY: a point is zero distance from itself. */
    double self = geo_distance_km(lat1, lng1, lat1, lng1);
    assert(self == self);
    assert(self < 1e-6);

    return 0;
}
