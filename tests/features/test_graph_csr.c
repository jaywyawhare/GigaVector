/**
 * test_graph_csr.c — unit tests for the GraphBLAS-lite CSR layer (graph_csr.h):
 * build (out/in/undirected, boolean/weighted) + the SpMV / SpMM / row-sums /
 * boolean-reachability kernels, checked against hand-computed values. Run under
 * ASAN/valgrind to prove the kernels + build are memory-safe.
 *
 * Graph: a -w2-> b -w3-> c   (3 nodes, 2 weighted directed edges)
 */
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "features/graph_db.h"
#include "features/graph_algos.h"
#include "features/graph_csr.h"

static int failures = 0;
#define ASSERT(c, m) do { \
    if (!(c)) { printf("FAIL: %s:%d: %s\n", __FILE__, __LINE__, (m)); failures++; } \
    else { printf("ok: %s\n", (m)); } } while (0)

int main(void) {
    GV_GraphDB *g = graph_create(NULL);
    uint64_t a = graph_add_node(g, "N");
    uint64_t b = graph_add_node(g, "N");
    uint64_t c = graph_add_node(g, "N");
    graph_add_edge(g, a, b, "E", 2.0f);
    graph_add_edge(g, b, c, "E", 3.0f);

    GV_GAContext *ctx = gv_ga_build(g);
    ASSERT(ctx && gv_ga_count(ctx) == 3, "context has 3 nodes");
    size_t ia = gv_ga_index(ctx, a), ib = gv_ga_index(ctx, b), ic = gv_ga_index(ctx, c);

    /* ---- OUT weighted ---- */
    GV_CSR *Ow = gv_csr_build(g, ctx, GV_CSR_OUT, 1);
    ASSERT(Ow && Ow->n == 3 && Ow->nnz == 2, "out-weighted: 3x3, 2 nnz");

    double rs[3];
    gv_csr_row_sums(Ow, rs);
    ASSERT((int)(rs[ia] + 0.5) == 2 && (int)(rs[ib] + 0.5) == 3 && (int)(rs[ic] + 0.5) == 0,
           "out row_sums == weighted out-degrees {2,3,0}");

    double x[3] = {1, 1, 1}, y[3] = {0};
    gv_csr_spmv(Ow, x, y);
    ASSERT((int)(y[ia] + 0.5) == 2 && (int)(y[ib] + 0.5) == 3 && (int)(y[ic] + 0.5) == 0,
           "SpMV(Ow, ones) == {2,3,0}");

    /* SpMV with x = e_c → only b (edge b->c, w3) contributes */
    double ec[3] = {0}; ec[ic] = 1.0; double yc[3] = {0};
    gv_csr_spmv(Ow, ec, yc);
    ASSERT((int)(yc[ib] + 0.5) == 3 && (int)(yc[ia] + 0.5) == 0, "SpMV(Ow, e_c) puts 3 at b");

    /* transpose SpMV: y = Owᵀ x accumulates; x=e_a should push weight 2 onto b */
    double ea[3] = {0}; ea[ia] = 1.0; double yt[3] = {0};
    gv_csr_spmv_transpose(Ow, ea, yt);
    ASSERT((int)(yt[ib] + 0.5) == 2, "SpMV_transpose(Ow, e_a) puts 2 at b");
    gv_csr_free(Ow);

    /* ---- OUT boolean ---- */
    GV_CSR *Ob = gv_csr_build(g, ctx, GV_CSR_OUT, 0);
    ASSERT(Ob && Ob->val == NULL, "boolean matrix has NULL val");
    /* bool reachability: predecessors of b (x = e_b) → a lights up */
    uint8_t xb[3] = {0}; xb[ib] = 1; uint8_t yb[3] = {0};
    gv_csr_bool_spmv(Ob, xb, yb);
    ASSERT(yb[ia] == 1 && yb[ib] == 0 && yb[ic] == 0, "bool_spmv(Ob, e_b): a is a predecessor of b");
    gv_csr_free(Ob);

    /* ---- IN boolean ---- */
    GV_CSR *Ib = gv_csr_build(g, ctx, GV_CSR_IN, 0);
    double irs[3];
    gv_csr_row_sums(Ib, irs);
    ASSERT((int)(irs[ia] + 0.5) == 0 && (int)(irs[ib] + 0.5) == 1 && (int)(irs[ic] + 0.5) == 1,
           "in row_sums == in-degrees {0,1,1}");
    gv_csr_free(Ib);

    /* ---- UNDIRECTED weighted: b neighbors a(2) and c(3) ---- */
    GV_CSR *Uw = gv_csr_build(g, ctx, GV_CSR_UNDIRECTED, 1);
    double urs[3];
    gv_csr_row_sums(Uw, urs);
    ASSERT((int)(urs[ib] + 0.5) == 5, "undirected weighted degree of b == 2+3 == 5");
    ASSERT((int)(urs[ia] + 0.5) == 2 && (int)(urs[ic] + 0.5) == 3, "undirected degrees of a,c");
    gv_csr_free(Uw);

    /* ---- SpMM: C = A * B, B = 3x2 identity-ish, check propagation ---- */
    GV_CSR *A = gv_csr_build(g, ctx, GV_CSR_OUT, 1);
    /* B is indexed by DENSE index, so fill rows explicitly per node. */
    double B[6] = {0};
    B[ia*2+0] = 1; B[ia*2+1] = 0;   /* row a = (1,0) */
    B[ib*2+0] = 0; B[ib*2+1] = 1;   /* row b = (0,1) */
    B[ic*2+0] = 1; B[ic*2+1] = 1;   /* row c = (1,1) */
    double C[6] = {0};
    gv_csr_spmm_dense(A, B, 2, C);
    /* row a = w2 * B[b] = 2*(0,1) = (0,2); row b = w3 * B[c] = 3*(1,1) = (3,3); row c = 0 */
    ASSERT((int)(C[ia*2+0]+0.5) == 0 && (int)(C[ia*2+1]+0.5) == 2, "SpMM row a == (0,2)");
    ASSERT((int)(C[ib*2+0]+0.5) == 3 && (int)(C[ib*2+1]+0.5) == 3, "SpMM row b == (3,3)");
    ASSERT((int)(C[ic*2+0]+0.5) == 0 && (int)(C[ic*2+1]+0.5) == 0, "SpMM row c == (0,0)");
    gv_csr_free(A);

    /* empty graph → valid empty matrix */
    GV_GraphDB *e = graph_create(NULL);
    GV_GAContext *ectx = gv_ga_build(e);
    GV_CSR *em = gv_csr_build(e, ectx, GV_CSR_OUT, 1);
    ASSERT(em && em->n == 0 && em->nnz == 0, "empty graph -> empty CSR");
    gv_csr_free(em); gv_ga_free(ectx); graph_destroy(e);

    gv_ga_free(ctx);
    graph_destroy(g);

    if (failures == 0) printf("All CSR-layer tests PASSED.\n");
    else printf("%d CSR test(s) FAILED.\n", failures);
    return failures ? 1 : 0;
}
