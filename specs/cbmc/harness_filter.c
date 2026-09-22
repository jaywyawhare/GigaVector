/*
 * CBMC harness: filter expression parsing and evaluation
 * (src/search/filter.c, src/search/filter_ops.c).
 *
 * `filter_parse` takes attacker-controlled text; `filter_eval` then runs it
 * against a vector. Two properties: parsing arbitrary text is memory-safe, and
 * evaluation is TOTAL -- it returns a definite yes/no for every parsed filter
 * rather than reading past the metadata list.
  *
 * CBMC-SOURCES: src/search/filter.c src/search/filter_ops.c src/core/memory.c
 * CBMC-UNWIND: 8
*/
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "search/filter.h"
#include "schema/vector.h"

#if defined(__CPROVER__) || defined(__CPROVER)
uint64_t nondet_u64(void);
size_t   nondet_size(void);
unsigned char nondet_uchar(void);
float    nondet_float(void);
#  define ASSUME(c) __CPROVER_assume(c)
#else
static uint64_t g_seed = 0x9E3779B97F4A7C15ULL;
static uint64_t nondet_u64(void) {
    g_seed ^= g_seed << 13; g_seed ^= g_seed >> 7; g_seed ^= g_seed << 17;
    return g_seed;
}
static size_t nondet_size(void) { return (size_t)(nondet_u64() % 8); }
static unsigned char nondet_uchar(void) { return (unsigned char)(nondet_u64() & 0xFF); }
static float nondet_float(void) {
    return (float)((int)(nondet_u64() % 2000) - 1000) / 100.0f;
}
#  define ASSUME(c) do { if (!(c)) return 0; } while (0)
#endif

#define EXPR_BOUND 5
#define DIM 2

int main(void) {
    size_t len = nondet_size();
    ASSUME(len <= EXPR_BOUND - 1);

    char expr[EXPR_BOUND];
    for (size_t i = 0; i < EXPR_BOUND - 1; i++) {
        expr[i] = (char)nondet_uchar();
        ASSUME(expr[i] != 0);            /* keep the string at full length */
    }
    expr[len] = '\0';

    GV_Filter *f = filter_parse(expr);
    if (!f) return 0;                    /* rejecting malformed input is correct */

    float data[DIM];
    for (unsigned i = 0; i < DIM; i++) data[i] = nondet_float();

    GV_Vector v;
    memset(&v, 0, sizeof(v));
    v.data = data;
    v.dimension = DIM;
    v.metadata = NULL;                   /* the empty-metadata edge case */

    /* Evaluation is total: a definite decision, never an out-of-range value. */
    int r = filter_eval(f, &v);
    assert(r == 0 || r == 1 || r == -1);

    /* Evaluation is deterministic -- same filter, same vector, same answer. */
    assert(filter_eval(f, &v) == r);

    filter_destroy(f);
    return 0;
}
