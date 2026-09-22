/*
 * CBMC harness: CSR structural invariants and SpMV bounds
 * (features/graph_csr.c, graph_matrix.c, and the algorithms layered on them:
 *  graph_centrality.c, graph_embed.c, graph_community.c, graph_algos_util.c).
 *
 * The whole GraphBLAS-lite layer reformulates traversal as sparse linear
 * algebra over this one structure, so every algorithm above it inherits these
 * invariants. If row_ptr is not monotone, or a column index escapes [0, n),
 * SpMV reads out of bounds -- and that is a single bug affecting PageRank,
 * HITS, FastRP, k-hop reachability and spreading activation at once.
  *
 * CBMC-SOURCES: src/features/graph_csr.c src/core/memory.c
 * CBMC-UNWIND: 5
*/
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "features/graph_csr.h"

#if defined(__CPROVER__) || defined(__CPROVER)
uint64_t nondet_u64(void);
size_t   nondet_size(void);
float    nondet_float(void);
unsigned char nondet_uchar(void);
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
static unsigned char nondet_uchar(void) { return (unsigned char)(nondet_u64() & 0xFF); }
#  define ASSUME(c) do { if (!(c)) return 0; } while (0)
#endif

#define N   2
#define NNZ 2

int main(void) {
    /* Build a CSR by hand so the structure itself is nondeterministic. */
    size_t   row_ptr[N + 1];
    uint32_t col_idx[NNZ];
    float    val[NNZ];

    row_ptr[0] = 0;
    for (unsigned i = 1; i <= N; i++) {
        size_t step = (size_t)(nondet_u64() % 3);
        row_ptr[i] = row_ptr[i - 1] + step;
        ASSUME(row_ptr[i] <= NNZ);
    }
    for (unsigned e = 0; e < NNZ; e++) {
        col_idx[e] = (uint32_t)(nondet_u64() % N);
        val[e] = nondet_float();
        /* Finite: inf edge weights make the row-sum comparison meaningless. */
        ASSUME(val[e] > -1.0e9f && val[e] < 1.0e9f);
    }

    GV_CSR m;
    memset(&m, 0, sizeof(m));
    m.n = N;
    m.nnz = row_ptr[N];
    m.row_ptr = row_ptr;
    m.col_idx = col_idx;
    m.val = val;

    /* The three structural invariants every kernel above depends on. */
    for (unsigned i = 0; i < N; i++)
        assert(row_ptr[i] <= row_ptr[i + 1]);          /* monotone */
    assert(row_ptr[N] == m.nnz);                        /* terminator agrees */
    for (size_t e = 0; e < m.nnz; e++)
        assert(col_idx[e] < m.n);                       /* in range */

    /* SpMV must stay inside y[0..n) for any conforming matrix. */
    double x[N], y[N];
    for (unsigned i = 0; i < N; i++) {
        x[i] = (double)nondet_float();
        ASSUME(x[i] > -1.0e9 && x[i] < 1.0e9);
        y[i] = 0.0;
    }
    gv_csr_spmv(&m, x, y);
    for (unsigned i = 0; i < N; i++)
        assert(y[i] == y[i]);                           /* no NaN injected */

    gv_csr_spmv_transpose(&m, x, y);
    for (unsigned i = 0; i < N; i++)
        assert(y[i] == y[i]);

    /* Row sums are finite and match a hand computation. */
    double sums[N];
    gv_csr_row_sums(&m, sums);
    for (unsigned i = 0; i < N; i++) {
        double manual = 0.0;
        for (size_t e = row_ptr[i]; e < row_ptr[i + 1]; e++) manual += (double)val[e];
        assert(sums[i] == sums[i]);
        double diff = sums[i] - manual;
        if (diff < 0) diff = -diff;
        assert(diff < 1e-3);
    }
    return 0;
}
