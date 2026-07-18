/**
 * graph_csr.c — "GraphBLAS-lite" CSR adjacency matrix + sparse linear-algebra
 * kernels. Zero external dependencies; kernels use restrict + simple loops so
 * the compiler auto-vectorizes the inner accumulations at -O3 (and -mavx2 when
 * the build enables SIMD_FLAGS).
 */
#include "features/graph_csr.h"
#include "core/memory.h"

#include <string.h>

size_t gv_csr_nrows(const GV_CSR *m) { return m ? m->n : 0; }

void gv_csr_free(GV_CSR *m) {
    if (!m) return;
    gv_free(m->row_ptr);
    gv_free(m->col_idx);
    gv_free(m->val);
    gv_free(m);
}

/* Visit the neighbors (dense index + weight) of node index i for the chosen
 * orientation, invoking cb(user, j, w). Returns -1 if the node vanished. */
typedef void (*csr_nb_cb)(void *user, uint32_t j, float w);

static int csr_visit(const GV_GraphDB *g, const GV_GAContext *ctx,
                     uint64_t node_id, GV_CSRDir dir, int weighted,
                     csr_nb_cb cb, void *user) {
    const GV_GraphNode *n = graph_get_node(g, node_id);
    if (!n) return -1;
    if (dir == GV_CSR_OUT || dir == GV_CSR_UNDIRECTED) {
        for (size_t e = 0; e < n->out_count; e++) {
            size_t j = gv_ga_index(ctx, n->out_edges[e].neighbor_id);
            if (j == (size_t)-1) continue;
            float w = 1.0f;
            if (weighted) {
                const GV_GraphEdge *ed = graph_get_edge(g, n->out_edges[e].edge_id);
                w = ed ? ed->weight : 1.0f;
            }
            cb(user, (uint32_t)j, w);
        }
    }
    if (dir == GV_CSR_IN || dir == GV_CSR_UNDIRECTED) {
        for (size_t e = 0; e < n->in_count; e++) {
            size_t j = gv_ga_index(ctx, n->in_edges[e].neighbor_id);
            if (j == (size_t)-1) continue;
            float w = 1.0f;
            if (weighted) {
                const GV_GraphEdge *ed = graph_get_edge(g, n->in_edges[e].edge_id);
                w = ed ? ed->weight : 1.0f;
            }
            cb(user, (uint32_t)j, w);
        }
    }
    return 0;
}

/* Dedup state for one row: `stamp[j]==gen` means column j is already present in
 * the current row at nnz slot `pos[j]`. */
typedef struct {
    uint64_t *stamp;
    size_t   *pos;
    uint64_t  gen;
    /* pass-1: count; pass-2: fill */
    size_t    count;      /* deduped entries in current row */
    uint32_t *col_idx;    /* pass-2 target (or NULL in pass-1) */
    float    *val;        /* pass-2 target (weighted) or NULL */
    size_t    base;       /* row start offset in pass-2 */
    int       weighted;
} CsrRowState;

static void csr_count_cb(void *user, uint32_t j, float w) {
    CsrRowState *s = (CsrRowState *)user;
    (void)w;
    if (s->stamp[j] != s->gen) { s->stamp[j] = s->gen; s->count++; }
}

static void csr_fill_cb(void *user, uint32_t j, float w) {
    CsrRowState *s = (CsrRowState *)user;
    if (s->stamp[j] != s->gen) {
        s->stamp[j] = s->gen;
        size_t slot = s->base + s->count;
        s->pos[j] = slot;
        s->col_idx[slot] = j;
        if (s->weighted) s->val[slot] = w;
        s->count++;
    } else if (s->weighted) {
        s->val[s->pos[j]] += w;   /* accumulate parallel edges */
    }
}

GV_CSR *gv_csr_build(const GV_GraphDB *g, const GV_GAContext *ctx,
                     GV_CSRDir dir, int weighted) {
    if (!g || !ctx) return NULL;
    size_t N = gv_ga_count(ctx);

    GV_CSR *m = (GV_CSR *)gv_calloc(1, sizeof(GV_CSR));
    if (!m) return NULL;
    m->n = N;
    m->row_ptr = (size_t *)gv_calloc(N + 1, sizeof(size_t));
    if (!m->row_ptr) { gv_csr_free(m); return NULL; }
    if (N == 0) return m;  /* valid empty matrix */

    uint64_t *stamp = (uint64_t *)gv_calloc(N, sizeof(uint64_t));
    size_t   *pos   = (size_t *)gv_alloc(N * sizeof(size_t));
    if (!stamp || !pos) { gv_free(stamp); gv_free(pos); gv_csr_free(m); return NULL; }

    CsrRowState st;
    memset(&st, 0, sizeof(st));
    st.stamp = stamp; st.pos = pos; st.weighted = weighted;

    /* Pass 1: count deduped nnz per row -> row_ptr. */
    for (size_t i = 0; i < N; i++) {
        st.gen = (uint64_t)i + 1;
        st.count = 0;
        csr_visit(g, ctx, gv_ga_id(ctx, i), dir, weighted, csr_count_cb, &st);
        m->row_ptr[i + 1] = m->row_ptr[i] + st.count;
    }
    m->nnz = m->row_ptr[N];

    if (m->nnz > 0) {
        m->col_idx = (uint32_t *)gv_alloc(m->nnz * sizeof(uint32_t));
        if (!m->col_idx) { gv_free(stamp); gv_free(pos); gv_csr_free(m); return NULL; }
        if (weighted) {
            m->val = (float *)gv_alloc(m->nnz * sizeof(float));
            if (!m->val) { gv_free(stamp); gv_free(pos); gv_csr_free(m); return NULL; }
        }
        st.col_idx = m->col_idx;
        st.val = m->val;
        /* Pass 2: fill (reuse stamp with a fresh generation range). */
        for (size_t i = 0; i < N; i++) {
            st.gen = (uint64_t)N + i + 2;  /* disjoint from pass-1 gens */
            st.count = 0;
            st.base = m->row_ptr[i];
            csr_visit(g, ctx, gv_ga_id(ctx, i), dir, weighted, csr_fill_cb, &st);
        }
    }

    gv_free(stamp);
    gv_free(pos);
    return m;
}

/* ── kernels ────────────────────────────────────────────────────────────────*/

void gv_csr_spmv(const GV_CSR *A, const double *x, double *y) {
    if (!A || !x || !y) return;
    const size_t *rp = A->row_ptr;
    const uint32_t * restrict ci = A->col_idx;
    const float * restrict v = A->val;
    for (size_t i = 0; i < A->n; i++) {
        double acc = 0.0;
        size_t s = rp[i], e = rp[i + 1];
        if (v) {
            for (size_t k = s; k < e; k++) acc += (double)v[k] * x[ci[k]];
        } else {
            for (size_t k = s; k < e; k++) acc += x[ci[k]];
        }
        y[i] = acc;
    }
}

void gv_csr_spmv_transpose(const GV_CSR *A, const double *x, double *y) {
    if (!A || !x || !y) return;
    const size_t *rp = A->row_ptr;
    const uint32_t *ci = A->col_idx;
    const float *v = A->val;
    for (size_t i = 0; i < A->n; i++) {
        double xi = x[i];
        if (xi == 0.0) continue;
        size_t s = rp[i], e = rp[i + 1];
        if (v) {
            for (size_t k = s; k < e; k++) y[ci[k]] += (double)v[k] * xi;
        } else {
            for (size_t k = s; k < e; k++) y[ci[k]] += xi;
        }
    }
}

void gv_csr_row_sums(const GV_CSR *A, double *out) {
    if (!A || !out) return;
    const size_t *rp = A->row_ptr;
    const float *v = A->val;
    for (size_t i = 0; i < A->n; i++) {
        size_t s = rp[i], e = rp[i + 1];
        if (v) {
            double acc = 0.0;
            for (size_t k = s; k < e; k++) acc += (double)v[k];
            out[i] = acc;
        } else {
            out[i] = (double)(e - s);
        }
    }
}

void gv_csr_spmm_dense(const GV_CSR *A, const double *B, size_t k, double *C) {
    if (!A || !B || !C || k == 0) return;
    const size_t *rp = A->row_ptr;
    const uint32_t *ci = A->col_idx;
    const float *v = A->val;
    memset(C, 0, A->n * k * sizeof(double));
    for (size_t i = 0; i < A->n; i++) {
        double * restrict crow = C + i * k;
        size_t s = rp[i], e = rp[i + 1];
        for (size_t p = s; p < e; p++) {
            const double * restrict brow = B + (size_t)ci[p] * k;
            double w = v ? (double)v[p] : 1.0;
            for (size_t c = 0; c < k; c++) crow[c] += w * brow[c];  /* axpy, auto-vectorized */
        }
    }
}

void gv_csr_bool_spmv(const GV_CSR *A, const uint8_t *x, uint8_t *y) {
    if (!A || !x || !y) return;
    const size_t *rp = A->row_ptr;
    const uint32_t *ci = A->col_idx;
    for (size_t i = 0; i < A->n; i++) {
        uint8_t r = 0;
        size_t s = rp[i], e = rp[i + 1];
        for (size_t kk = s; kk < e; kk++) { if (x[ci[kk]]) { r = 1; break; } }
        y[i] = r;
    }
}
