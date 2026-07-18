/**
 * @file graph_csr.h
 * @brief "GraphBLAS-lite" — a zero-dependency sparse adjacency-matrix layer for
 *        the GigaVector property graph, plus the sparse linear-algebra kernels
 *        (SpMV, dense SpMM, boolean reachability SpMV) that the analytics and
 *        embedding algorithms are expressed in.
 *
 * Rationale: graph traversal and iterative analytics (PageRank, eigenvector,
 * HITS, FastRP, k-hop reachability) are naturally sparse linear algebra — the
 * same insight FalkorDB gets from SuiteSparse:GraphBLAS. Rather than take on a
 * heavyweight external dependency (GigaVector's core is dependency-free), this
 * provides a compact CSR matrix + a handful of semiring kernels tuned to
 * auto-vectorize, reused across the graph-algorithm library.
 *
 * Matrices are indexed by DENSE node index [0, N) via a GV_GAContext
 * (graph_algos.h). Row i of the OUT matrix holds i's out-neighbors, etc.
 */

#ifndef GIGAVECTOR_GV_GRAPH_CSR_H
#define GIGAVECTOR_GV_GRAPH_CSR_H

#include <stddef.h>
#include <stdint.h>

#include "features/graph_db.h"
#include "features/graph_algos.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Compressed sparse row adjacency matrix over dense node indices [0, N). */
typedef struct {
    size_t    n;        /**< rows == cols == node count. */
    size_t    nnz;      /**< number of stored entries. */
    size_t   *row_ptr;  /**< length n+1; row i spans [row_ptr[i], row_ptr[i+1]). */
    uint32_t *col_idx;  /**< length nnz; dense column (neighbor) indices. */
    float    *val;      /**< length nnz edge weights, or NULL for a boolean matrix (implicit 1). */
} GV_CSR;

/** Adjacency orientation for gv_csr_build. */
typedef enum {
    GV_CSR_OUT = 0,         /**< row i = out-neighbors of i (directed). */
    GV_CSR_IN,              /**< row i = in-neighbors of i (transpose of OUT). */
    GV_CSR_UNDIRECTED       /**< row i = union of out+in neighbors (symmetric, dedup). */
} GV_CSRDir;

/**
 * Build a CSR adjacency matrix from the graph over the given dense-index context.
 * @param g   graph.
 * @param ctx dense node-index context (gv_ga_build); must match g.
 * @param dir orientation (out/in/undirected).
 * @param weighted non-zero → store edge weights in ->val; 0 → boolean (->val NULL).
 *        For UNDIRECTED weighted, parallel edges between a pair are summed.
 * @return heap CSR (free with gv_csr_free), or NULL on error.
 */
GV_CSR *gv_csr_build(const GV_GraphDB *g, const GV_GAContext *ctx,
                     GV_CSRDir dir, int weighted);

/** Free a CSR matrix (safe on NULL). */
void gv_csr_free(GV_CSR *m);

/** Number of rows (== nodes). */
size_t gv_csr_nrows(const GV_CSR *m);

/**
 * Sparse matrix–vector product y = A·x over the (+,*) semiring.
 * Boolean matrices (val==NULL) use implicit weight 1. x and y have length n; y
 * must not alias x. Auto-vectorizes on the inner accumulation.
 */
void gv_csr_spmv(const GV_CSR *A, const double *x, double *y);

/**
 * Transpose-free "column push" SpMV: y[j] += (val or 1)·x[i] for every edge
 * i→j in A (i.e. y = Aᵀ·x). Useful for PageRank-style rank distribution without
 * materializing the transpose. y must be zeroed by the caller; y != x.
 */
void gv_csr_spmv_transpose(const GV_CSR *A, const double *x, double *y);

/** Weighted row sums (out-degree / weighted degree). out has length n. */
void gv_csr_row_sums(const GV_CSR *A, double *out);

/**
 * Dense sparse matrix–matrix product C = A·B where B is a dense n×k row-major
 * matrix and C is dense n×k row-major (C must not alias B). Used by FastRP.
 */
void gv_csr_spmm_dense(const GV_CSR *A, const double *B, size_t k, double *C);

/**
 * Boolean reachability SpMV (frontier expansion over the OR-AND semiring):
 * y[i] = 1 if any neighbor j in row i has x[j] != 0, else 0. Used for k-hop
 * reachability / variable-length path expansion. y != x, both length n.
 */
void gv_csr_bool_spmv(const GV_CSR *A, const uint8_t *x, uint8_t *y);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_GRAPH_CSR_H */
