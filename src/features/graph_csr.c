/**
 * graph_csr.c — "GraphBLAS-lite" CSR adjacency matrix + sparse linear-algebra
 * kernels. Zero external dependencies; kernels use restrict + simple loops so
 * the compiler auto-vectorizes the inner accumulations at -O3 (and -mavx2 when
 * the build enables SIMD_FLAGS).
 */
#include "features/graph_csr.h"
#include "features/graph_algos.h"
#include "core/memory.h"

#include <string.h>
#include <pthread.h>
#ifndef _WIN32
#include <unistd.h>
#else
#include <windows.h>
#endif

/* Row-parallel SpMV worker: rows [start,end) of A x -> y. */
typedef struct {
    const GV_CSR *A;
    const double *x;
    double *y;
    size_t start, end;
} CsrSpmvJob;

static void *csr_spmv_worker(void *arg) {
    CsrSpmvJob *j = (CsrSpmvJob *)arg;
    const GV_CSR *A = j->A;
    const size_t *rp = A->row_ptr;
    const uint32_t *ci = A->col_idx;
    const float *v = A->val;
    for (size_t i = j->start; i < j->end; i++) {
        double acc = 0.0;
        size_t s = rp[i], e = rp[i + 1];
        if (v) {
            for (size_t k = s; k < e; k++) acc += (double)v[k] * j->x[ci[k]];
        } else {
            for (size_t k = s; k < e; k++) acc += j->x[ci[k]];
        }
        j->y[i] = acc;
    }
    return NULL;
}

#define CSR_PARALLEL_MIN_ROWS 4096

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
    (void)g;
    const GV_GraphNode *n = gv_ga_node(ctx, node_id);
    if (!n) return -1;
    if (dir == GV_CSR_OUT || dir == GV_CSR_UNDIRECTED) {
        for (size_t e = 0; e < n->out_count; e++) {
            size_t j = gv_ga_index(ctx, n->out_edges[e].neighbor_id);
            if (j == (size_t)-1) continue;
            float w = 1.0f;
            if (weighted) {
                const GV_GraphEdge *ed = gv_ga_edge(ctx, n->out_edges[e].edge_id);
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
                const GV_GraphEdge *ed = gv_ga_edge(ctx, n->in_edges[e].edge_id);
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

    /* Snapshot consistency: a pinned context already holds the graph read
     * lock for its whole lifetime; an external context needs us to take it
     * around the build so concurrent mutations can't tear the snapshot. */
    int external = !gv_ga_pinned(ctx);
    if (external) graph_read_lock(g);

    GV_CSR *m = (GV_CSR *)gv_calloc(1, sizeof(GV_CSR));
    if (!m) return NULL;
    m->n = N;
    m->row_ptr = (size_t *)gv_calloc(N + 1, sizeof(size_t));
    if (!m->row_ptr) { gv_csr_free(m); if (external) graph_read_unlock(g); return NULL; }
    if (N == 0) { if (external) graph_read_unlock(g); return m; }  /* valid empty matrix */

    uint64_t *stamp = (uint64_t *)gv_calloc(N, sizeof(uint64_t));
    size_t   *pos   = (size_t *)gv_alloc(N * sizeof(size_t));
    if (!stamp || !pos) {
        gv_free(stamp);
        gv_free(pos);
        gv_csr_free(m);
        if (external) graph_read_unlock(g);
        return NULL;
    }

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
        if (!m->col_idx) {
            gv_free(stamp);
            gv_free(pos);
            gv_csr_free(m);
            if (external) graph_read_unlock(g);
            return NULL;
        }
        if (weighted) {
            m->val = (float *)gv_alloc(m->nnz * sizeof(float));
            if (!m->val) {
                gv_free(stamp);
                gv_free(pos);
                gv_csr_free(m);
                if (external) graph_read_unlock(g);
                return NULL;
            }
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
    if (external) graph_read_unlock(g);
    return m;
}

void gv_csr_spmv(const GV_CSR *A, const double *x, double *y) {
    if (!A || !x || !y) return;

    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    size_t nthreads = (ncpu > 0 && A->n >= CSR_PARALLEL_MIN_ROWS)
                          ? (size_t)ncpu : 1;
    if (nthreads <= 1) {
        CsrSpmvJob j = { A, x, y, 0, A->n };
        csr_spmv_worker(&j);
        return;
    }

    pthread_t *threads = (pthread_t *)gv_alloc((nthreads - 1) * sizeof(pthread_t));
    CsrSpmvJob *jobs = (CsrSpmvJob *)gv_alloc(nthreads * sizeof(CsrSpmvJob));
    if (!threads || !jobs) {
        gv_free(threads);
        gv_free(jobs);
        CsrSpmvJob j = { A, x, y, 0, A->n };
        csr_spmv_worker(&j);
        return;
    }

    size_t chunk = (A->n + nthreads - 1) / nthreads;
    for (size_t t = 0; t < nthreads; t++) {
        jobs[t].A = A; jobs[t].x = x; jobs[t].y = y;
        jobs[t].start = t * chunk;
        jobs[t].end = jobs[t].start + chunk < A->n ? jobs[t].start + chunk : A->n;
        if (t == nthreads - 1) continue;
        pthread_create(&threads[t], NULL, csr_spmv_worker, &jobs[t]);
    }
    csr_spmv_worker(&jobs[nthreads - 1]);
    for (size_t t = 0; t < nthreads - 1; t++) pthread_join(threads[t], NULL);

    gv_free(threads);
    gv_free(jobs);
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


/* ── Incremental patching ──────────────────────────────────────────────────
 * Rebuild only the rows touched by a batch of edge add/remove deltas,
 * copying untouched rows verbatim. For small mutation batches this avoids
 * re-enumerating every node's adjacency (the dominant cost of a full
 * build: per-node hash lookups + pointer chasing). Returns 0 on success,
 * -1 if any delta endpoint is absent from ctx (caller must fall back to a
 * full rebuild). */
typedef struct {
    uint64_t *stamp;    /* dedup stamps, length N */
    size_t   *pos;      /* nnz slot per column, length N */
} CsrPatchScratch;

static int csr_patch_rows(const GV_GraphDB *g, const GV_GAContext *ctx,
                          GV_CSRDir dir, int weighted, GV_CSR *m,
                          const GV_GraphEdgeDelta *deltas, size_t nd,
                          const uint8_t *row_touched) {
    size_t N = m->n;
    if (N == 0) return 0;

    /* Resolve endpoints to dense indices up front; unknown node -> rebuild. */
    for (size_t d = 0; d < nd; d++) {
        if (gv_ga_index(ctx, deltas[d].source) == (size_t)-1 ||
            gv_ga_index(ctx, deltas[d].target) == (size_t)-1)
            return -1;
    }

    CsrPatchScratch sc;
    sc.stamp = (uint64_t *)gv_calloc(N, sizeof(uint64_t));
    sc.pos   = (size_t *)gv_alloc(N * sizeof(size_t));

    GV_CSR *out = (GV_CSR *)gv_calloc(1, sizeof(GV_CSR));
    out->n = N;
    if (!sc.stamp || !sc.pos || !out) {
        gv_free(sc.stamp); gv_free(sc.pos); gv_csr_free(out);
        return -1;
    }
    out->row_ptr = (size_t *)gv_calloc(N + 1, sizeof(size_t));
    if (!out->row_ptr) {
        gv_csr_free(out);
        gv_free(sc.stamp); gv_free(sc.pos);
        return -1;
    }

    CsrRowState st;
    memset(&st, 0, sizeof(st));
    st.stamp = sc.stamp;
    st.pos = sc.pos;
    st.weighted = weighted;

    int external = !gv_ga_pinned(ctx);
    if (external) graph_read_lock(g);

    /* count pass over touched rows only */
    for (size_t i = 0; i < N; i++) {
        if (!row_touched[i]) { out->row_ptr[i + 1] = m->row_ptr[i + 1] - m->row_ptr[i]; continue; }
        st.gen = (uint64_t)i + 1;
        st.count = 0;
        csr_visit(g, ctx, gv_ga_id(ctx, i), dir, weighted, csr_count_cb, &st);
        out->row_ptr[i + 1] = st.count;
    }
    for (size_t i = 0; i < N; i++) out->row_ptr[i + 1] += out->row_ptr[i];
    out->nnz = out->row_ptr[N];

    if (out->nnz > 0) {
        out->col_idx = (uint32_t *)gv_alloc(out->nnz * sizeof(uint32_t));
        if (weighted) out->val = (float *)gv_alloc(out->nnz * sizeof(float));
        if (!out->col_idx || (weighted && !out->val)) {
            gv_csr_free(out);
            gv_free(sc.stamp); gv_free(sc.pos);
            graph_read_unlock(g);
            return -1;
        }
        /* fill pass: touched rows from live graph, rest copied verbatim */
        for (size_t i = 0; i < N; i++) {
            size_t s = out->row_ptr[i], e = out->row_ptr[i + 1];
            size_t len = e - s;
            if (!row_touched[i]) {
                if (len) memcpy(out->col_idx + s, m->col_idx + m->row_ptr[i],
                                len * sizeof(uint32_t));
                if (len && weighted)
                    memcpy(out->val + s, m->val + m->row_ptr[i],
                           len * sizeof(float));
                continue;
            }
            st.gen = (uint64_t)N + i + 2;
            st.count = 0;
            st.base = s;
            st.col_idx = out->col_idx;
            st.val = out->val;
            csr_visit(g, ctx, gv_ga_id(ctx, i), dir, weighted, csr_fill_cb, &st);
        }
    }
    if (external) graph_read_unlock(g);

    /* Swap arrays into m so cache identity survives. */
    gv_free(m->row_ptr); gv_free(m->col_idx); gv_free(m->val);
    m->row_ptr = out->row_ptr;
    m->col_idx = out->col_idx;
    m->val     = out->val;
    m->nnz     = out->nnz;
    gv_free(out);   /* struct shell only; arrays now owned by m */

    gv_free(sc.stamp);
    gv_free(sc.pos);
    return 0;
}

#define GV_CSR_PATCH_MAX_DELTAS 1024

/* ── Memoized builds ───────────────────────────────────────────────────────
 * Analytics repeatedly rebuild the same CSR while the graph is unchanged;
 * key the cache by the graph's mutation version and drop it on change.
 * Single global entry per orientation/weighted combo — enough for pipelines
 * that run several algorithms back-to-back over one snapshot. */
#define GV_CSR_CACHE_SLOTS 6

typedef struct {
    const GV_GraphDB *g;
    uint64_t   version;
    GV_CSRDir  dir;
    int        weighted;
    GV_CSR    *m;
} CsrCacheSlot;

static CsrCacheSlot g_csr_cache[GV_CSR_CACHE_SLOTS];
static pthread_mutex_t g_csr_cache_mu = PTHREAD_MUTEX_INITIALIZER;

void gv_csr_cache_clear(void) {
    pthread_mutex_lock(&g_csr_cache_mu);
    for (int i = 0; i < GV_CSR_CACHE_SLOTS; i++) {
        gv_csr_free(g_csr_cache[i].m);
        memset(&g_csr_cache[i], 0, sizeof(CsrCacheSlot));
    }
    pthread_mutex_unlock(&g_csr_cache_mu);
}

/* Install a caller-owned, current-version matrix into the cache (evicting as
 * needed) and return it to the caller under the do-not-free contract. */
static GV_CSR *csr_cache_install(const GV_GraphDB *g, uint64_t ver,
                                 GV_CSRDir dir, int weighted, GV_CSR *m) {
    pthread_mutex_lock(&g_csr_cache_mu);
    CsrCacheSlot *slot = NULL;
    for (int i = 0; i < GV_CSR_CACHE_SLOTS; i++) {
        CsrCacheSlot *s = &g_csr_cache[i];
        if (s->m && (s->g != g || s->version != ver)) {
            gv_csr_free(s->m);
            memset(s, 0, sizeof(CsrCacheSlot));
        }
        if (!s->m && !slot) slot = s;
    }
    if (!slot) slot = &g_csr_cache[0];   /* all current: evict one */
    gv_csr_free(slot->m);
    slot->g = g;
    slot->version = ver;
    slot->dir = dir;
    slot->weighted = weighted ? 1 : 0;
    slot->m = m;
    pthread_mutex_unlock(&g_csr_cache_mu);
    return m;
}

GV_CSR *gv_csr_build_cached(const GV_GraphDB *g, const GV_GAContext *ctx,
                            GV_CSRDir dir, int weighted) {

    if (!g || !ctx) return NULL;
    uint64_t ver = graph_version(g);

    GV_CSR *base = NULL;      /* freshest stale matrix of this combo */
    uint64_t base_ver = 0;

    pthread_mutex_lock(&g_csr_cache_mu);
    for (int i = 0; i < GV_CSR_CACHE_SLOTS; i++) {
        CsrCacheSlot *s = &g_csr_cache[i];
        int match_combo = s->m && s->g == g && s->dir == dir &&
                          s->weighted == (weighted ? 1 : 0);
        if (s->m && s->g == g && s->version == ver) {
            /* current-version entry: serve any matching combo, and NEVER
             * expire other combos of this snapshot — callers may still hold
             * those pointers (e.g. PageRank keeps IN while building OUT). */
            if (match_combo) {
                pthread_mutex_unlock(&g_csr_cache_mu);
                return s->m;
            }
            continue;
        }
        if (!s->m) continue;
        if (match_combo && s->version > base_ver) {
            /* take ownership out of the cache so no one else frees it */
            gv_csr_free(base);
            base = s->m;
            base_ver = s->version;
            memset(s, 0, sizeof(CsrCacheSlot));
        } else {
            gv_csr_free(s->m);
            memset(s, 0, sizeof(CsrCacheSlot));
        }
    }
    pthread_mutex_unlock(&g_csr_cache_mu);

    /* Incremental fast path: small edge-only delta -> patch touched rows.
     * Falls back to a full rebuild whenever coverage or endpoints are off. */
    if (base && base->n == gv_ga_count(ctx)) {
        size_t nd = 0;
        if (graph_edge_deltas_since(g, base_ver, NULL, 0, &nd) == 0 &&
            nd > 0 && nd <= GV_CSR_PATCH_MAX_DELTAS) {
            GV_GraphEdgeDelta *dl =
                (GV_GraphEdgeDelta *)gv_alloc(nd * sizeof(GV_GraphEdgeDelta));
            size_t nd2 = 0;
            if (dl && graph_edge_deltas_since(g, base_ver, dl, nd, &nd2) == 0 &&
                nd2 == nd) {
                size_t N = base->n;
                uint8_t *touched = (uint8_t *)gv_calloc(N ? N : 1, 1);
                int ok_rows = touched != NULL;
                for (size_t d = 0; d < nd && ok_rows; d++) {
                    size_t ru = gv_ga_index(ctx, dl[d].source);
                    size_t rv = gv_ga_index(ctx, dl[d].target);
                    if (ru == (size_t)-1 || rv == (size_t)-1 ||
                        ru >= N || rv >= N) { ok_rows = 0; break; }
                    if (dir == GV_CSR_OUT || dir == GV_CSR_UNDIRECTED)
                        touched[ru] = 1;
                    if (dir == GV_CSR_IN || dir == GV_CSR_UNDIRECTED)
                        touched[rv] = 1;
                }
                if (ok_rows &&
                    csr_patch_rows(g, ctx, dir, weighted,
                                   base, dl, nd, touched) == 0 &&
                    graph_version(g) == ver) {
                    gv_free(touched);
                    gv_free(dl);
                    /* patched matrix is current: install and serve it */
                    return csr_cache_install(g, ver, dir, weighted, base);
                }
                gv_free(touched);
            }
            gv_free(dl);
        }
    }

    if (base) { gv_csr_free(base); base = NULL; }

    /* Build outside the mutex: gv_csr_build takes the graph read lock and
     * may be slow. Concurrent duplicate builds are possible but harmless. */
    GV_CSR *m = gv_csr_build(g, ctx, dir, weighted);
    if (!m) return NULL;

    if (graph_version(g) != ver) return m;  /* raced: caller-owned fallback */
    return csr_cache_install(g, ver, dir, weighted, m);
}
