#define _POSIX_C_SOURCE 200112L

#include "index/rabitq.h"
#include "core/memory.h"
#include "schema/vector.h"
#include "schema/metadata.h"
#include "storage/soa_storage.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
/* Internal structure                                                   */
/* ------------------------------------------------------------------ */

typedef struct {
    size_t           dim;         /* original dimension */
    size_t           padded_dim;  /* next power-of-2 >= max(dim, 64) */
    size_t           code_words;  /* padded_dim / 64 */
    float           *signs;       /* [padded_dim] random ±1 from seed */
    GV_SoAStorage   *storage;     /* shared soa – float vectors + metadata */
    int              owns_storage;
    uint64_t        *codes;       /* [cap * code_words] binarised codes */
    uint8_t         *deleted_arr; /* [cap] per-slot deletion flag */
    size_t           count;
    size_t           cap;
    GV_RaBitQConfig  config;
} GV_RaBitQIndex;

/* ------------------------------------------------------------------ */
/* Helpers                                                              */
/* ------------------------------------------------------------------ */

static size_t next_pow2_ge64(size_t n) {
    size_t p = 64;
    while (p < n) p <<= 1;
    return p;
}

/* Deterministic sign vector from seed (LCG). */
static void generate_signs(float *signs, size_t n, uint64_t seed) {
    uint64_t s = seed ^ 0x6c62272e07bb0142ULL;
    for (size_t i = 0; i < n; i++) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        signs[i] = (s >> 63) ? 1.0f : -1.0f;
    }
}

/* In-place normalised Walsh-Hadamard transform (n must be power of 2). */
static void wht(float *x, size_t n) {
    for (size_t len = 1; len < n; len <<= 1) {
        for (size_t i = 0; i < n; i += len << 1) {
            for (size_t j = i; j < i + len; j++) {
                float u = x[j], v = x[j + len];
                x[j]       = u + v;
                x[j + len] = u - v;
            }
        }
    }
    float inv = 1.0f / sqrtf((float)n);
    for (size_t i = 0; i < n; i++) x[i] *= inv;
}

/* Apply the Randomised Hadamard Transform to `src` (length dim),
   writing padded_dim floats into `dst` (caller-allocated). */
static void rht(const float *src, float *dst, size_t dim, size_t padded_dim,
                const float *signs) {
    memcpy(dst, src, dim * sizeof(float));
    if (padded_dim > dim) memset(dst + dim, 0, (padded_dim - dim) * sizeof(float));
    for (size_t i = 0; i < padded_dim; i++) dst[i] *= signs[i];
    wht(dst, padded_dim);
}

/* L2-normalise in-place; no-op if norm is zero. */
static void l2_normalize(float *v, size_t n) {
    float sq = 0.0f;
    for (size_t i = 0; i < n; i++) sq += v[i] * v[i];
    if (sq <= 0.0f) return;
    float inv = 1.0f / sqrtf(sq);
    for (size_t i = 0; i < n; i++) v[i] *= inv;
}

/* Pack sign bits (>=0 → 1) into code_words uint64_t words. */
static void binarize(const float *x, uint64_t *codes, size_t padded_dim) {
    size_t code_words = padded_dim / 64;
    for (size_t w = 0; w < code_words; w++) {
        uint64_t word = 0;
        const float *base = x + w * 64;
        for (int b = 0; b < 64; b++) {
            if (base[b] >= 0.0f) word |= (1ULL << b);
        }
        codes[w] = word;
    }
}

/* XNOR-popcount inner product (number of matching bits). */
static int popcount_xnor(const uint64_t *a, const uint64_t *b,
                          size_t code_words) {
    int cnt = 0;
    for (size_t w = 0; w < code_words; w++)
        cnt += __builtin_popcountll(~(a[w] ^ b[w]));
    return cnt;
}

/* metadata_match helper – check if a metadata list satisfies a k/v filter. */
static int meta_match(GV_Metadata *m, const char *fk, const char *fv) {
    if (!fk || !fv) return 1;
    for (; m; m = m->next)
        if (m->key && m->value &&
            strcmp(m->key, fk) == 0 && strcmp(m->value, fv) == 0) return 1;
    return 0;
}

/* Build a GV_SearchResult from a stored vector and fill the copy. */
static void fill_result(GV_RaBitQIndex *idx, size_t vi, float dist,
                        GV_SearchResult *r) {
    r->id       = vi;
    r->distance = dist;
    r->is_sparse = 0;
    r->sparse_vector = NULL;

    GV_Vector view;
    soa_storage_get_vector_view(idx->storage, vi, &view);
    GV_Vector *copy = vector_create_from_data(view.dimension, view.data);
    if (copy) {
        GV_Metadata *m = soa_storage_get_metadata(idx->storage, vi);
        for (; m; m = m->next)
            if (m->key && m->value)
                vector_set_metadata(copy, m->key, m->value);
    }
    r->vector = copy;
}

/* ------------------------------------------------------------------ */
/* Candidate heap (max-heap on distance for k-NN maintenance)          */
/* ------------------------------------------------------------------ */

typedef struct { float dist; size_t idx; } RBQCandidate;

static void heap_push(RBQCandidate *h, size_t *hsz, size_t cap,
                      float dist, size_t idx) {
    if (*hsz < cap) {
        size_t i = (*hsz)++;
        h[i] = (RBQCandidate){dist, idx};
        /* sift up */
        while (i > 0) {
            size_t p = (i - 1) / 2;
            if (h[p].dist >= h[i].dist) break;
            RBQCandidate tmp = h[p]; h[p] = h[i]; h[i] = tmp;
            i = p;
        }
    } else if (dist < h[0].dist) {
        h[0] = (RBQCandidate){dist, idx};
        /* sift down */
        size_t i = 0;
        for (;;) {
            size_t l = 2*i+1, r = 2*i+2, mx = i;
            if (l < cap && h[l].dist > h[mx].dist) mx = l;
            if (r < cap && h[r].dist > h[mx].dist) mx = r;
            if (mx == i) break;
            RBQCandidate tmp = h[i]; h[i] = h[mx]; h[mx] = tmp;
            i = mx;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

void *rabitq_create(size_t dimension, const GV_RaBitQConfig *config,
                    GV_SoAStorage *soa_storage) {
    if (dimension == 0) return NULL;

    GV_RaBitQIndex *idx = (GV_RaBitQIndex *)gv_calloc(1, sizeof(GV_RaBitQIndex));
    if (!idx) return NULL;

    idx->dim        = dimension;
    idx->padded_dim = next_pow2_ge64(dimension);
    idx->code_words = idx->padded_dim / 64;

    if (config) {
        idx->config = *config;
    } else {
        idx->config.seed          = 42;
        idx->config.rerank_factor = 0;
    }
    if (idx->config.rerank_factor == 0) idx->config.rerank_factor = 4;

    idx->signs = (float *)gv_alloc(idx->padded_dim * sizeof(float));
    if (!idx->signs) { gv_free(idx); return NULL; }
    generate_signs(idx->signs, idx->padded_dim, idx->config.seed);

    if (soa_storage) {
        idx->storage      = soa_storage;
        idx->owns_storage = 0;
    } else {
        idx->storage = soa_storage_create(dimension, 0);
        if (!idx->storage) { gv_free(idx->signs); gv_free(idx); return NULL; }
        idx->owns_storage = 1;
    }

    return idx;
}

int rabitq_insert(void *index, GV_Vector *vector) {
    if (!index || !vector) return -1;
    GV_RaBitQIndex *idx = (GV_RaBitQIndex *)index;
    if (vector->dimension != idx->dim) return -1;

    /* Grow codes / deleted arrays if needed. */
    if (idx->count >= idx->cap) {
        size_t new_cap = idx->cap == 0 ? 256 : idx->cap * 2;
        uint64_t *nc = (uint64_t *)gv_realloc(idx->codes,
            new_cap * idx->code_words * sizeof(uint64_t));
        uint8_t  *nd = (uint8_t  *)gv_realloc(idx->deleted_arr,
            new_cap * sizeof(uint8_t));
        if (!nc || !nd) {
            if (nc) idx->codes       = nc;
            if (nd) idx->deleted_arr = nd;
            return -1;
        }
        idx->codes       = nc;
        idx->deleted_arr = nd;
        memset(idx->deleted_arr + idx->cap, 0, new_cap - idx->cap);
        idx->cap = new_cap;
    }

    /* Add raw float to soa_storage. */
    size_t vi = soa_storage_add(idx->storage, vector->data, vector->metadata);
    if (vi == (size_t)-1) return -1;
    vector->metadata = NULL;
    vector_destroy(vector);

    /* Compute binarised code: normalise → RHT → binarise. */
    float *tmp = (float *)gv_alloc(idx->padded_dim * sizeof(float));
    if (!tmp) return -1;  /* vector already committed to storage */

    float norm_buf[idx->dim];  /* VLA – dim is runtime but bounded by caller */
    memcpy(norm_buf, soa_storage_get_data(idx->storage, vi),
           idx->dim * sizeof(float));
    l2_normalize(norm_buf, idx->dim);
    rht(norm_buf, tmp, idx->dim, idx->padded_dim, idx->signs);
    binarize(tmp, idx->codes + vi * idx->code_words, idx->padded_dim);
    gv_free(tmp);

    idx->count++;
    return 0;
}

int rabitq_search(void *index, const GV_Vector *query, size_t k,
                  GV_SearchResult *results, GV_DistanceType distance_type,
                  const char *filter_key, const char *filter_value) {
    if (!index || !query || !results || k == 0) return -1;
    GV_RaBitQIndex *idx = (GV_RaBitQIndex *)index;
    if (query->dimension != idx->dim) return -1;

    size_t n = idx->storage->count;
    if (n == 0) return 0;

    /* Prepare binarised query. */
    float *qbuf = (float *)gv_alloc(idx->padded_dim * sizeof(float));
    if (!qbuf) return -1;

    float *qnorm = (float *)gv_alloc(idx->dim * sizeof(float));
    if (!qnorm) { gv_free(qbuf); return -1; }
    memcpy(qnorm, query->data, idx->dim * sizeof(float));
    l2_normalize(qnorm, idx->dim);
    rht(qnorm, qbuf, idx->dim, idx->padded_dim, idx->signs);

    uint64_t *qcodes = (uint64_t *)gv_alloc(idx->code_words * sizeof(uint64_t));
    if (!qcodes) { gv_free(qbuf); gv_free(qnorm); return -1; }
    binarize(qbuf, qcodes, idx->padded_dim);
    gv_free(qbuf);
    gv_free(qnorm);

    /* First pass: binary scan → top (rerank_factor * k) candidates. */
    size_t rerank_k = idx->config.rerank_factor * k;
    if (rerank_k < k) rerank_k = k;
    if (rerank_k > n) rerank_k = n;

    /* We want highest popcount_xnor → store negated score in max-heap so we
       keep the worst of the best and pop it when we find something better.
       Actually: smaller -score = better candidate, so use a max-heap on
       (-score) to maintain the top rerank_k.  But we already wrote
       heap_push to keep smallest dist, so we invert: cost = -popcount. */
    RBQCandidate *heap = (RBQCandidate *)gv_alloc(
        rerank_k * sizeof(RBQCandidate));
    if (!heap) { gv_free(qcodes); return -1; }
    size_t heap_sz = 0;

    for (size_t i = 0; i < n; i++) {
        if (soa_storage_is_deleted(idx->storage, i) == 1) continue;
        if (filter_key && filter_value) {
            GV_Metadata *m = soa_storage_get_metadata(idx->storage, i);
            if (!meta_match(m, filter_key, filter_value)) continue;
        }
        int matches = popcount_xnor(idx->codes + i * idx->code_words,
                                    qcodes, idx->code_words);
        /* Use negative matches as "distance" so heap_push keeps highest. */
        float pseudo_dist = -(float)matches;
        heap_push(heap, &heap_sz, rerank_k, pseudo_dist, i);
    }
    gv_free(qcodes);

    if (heap_sz == 0) { gv_free(heap); return 0; }

    /* Second pass: exact rerank of heap_sz candidates. */
    GV_Vector query_view;
    query_view.dimension  = idx->dim;
    query_view.data       = (float *)query->data;
    query_view.metadata   = NULL;

    RBQCandidate *rerank_heap = (RBQCandidate *)gv_alloc(
        k * sizeof(RBQCandidate));
    if (!rerank_heap) { gv_free(heap); return -1; }
    size_t rerank_sz = 0;

    /* Import from distance.h */
    extern float distance(const GV_Vector *a, const GV_Vector *b,
                          GV_DistanceType dt);
    GV_Vector tmp_vec;
    tmp_vec.dimension = idx->dim;
    tmp_vec.metadata  = NULL;

    for (size_t i = 0; i < heap_sz; i++) {
        size_t vi = heap[i].idx;
        tmp_vec.data = (float *)soa_storage_get_data(idx->storage, vi);
        float d = distance(&query_view, &tmp_vec, distance_type);
        heap_push(rerank_heap, &rerank_sz, k, d, vi);
    }
    gv_free(heap);

    /* Extract results in sorted order. */
    int found = (int)rerank_sz;
    for (int i = found - 1; i >= 0; i--) {
        size_t vi   = rerank_heap[0].idx;
        float  dist = rerank_heap[0].dist;
        fill_result(idx, vi, dist, &results[i]);
        rerank_heap[0] = rerank_heap[rerank_sz - 1];
        rerank_sz--;
        /* sift down */
        size_t cur = 0;
        for (;;) {
            size_t l = 2*cur+1, r = 2*cur+2, mx = cur;
            if (l < rerank_sz && rerank_heap[l].dist > rerank_heap[mx].dist) mx = l;
            if (r < rerank_sz && rerank_heap[r].dist > rerank_heap[mx].dist) mx = r;
            if (mx == cur) break;
            RBQCandidate tmp = rerank_heap[cur];
            rerank_heap[cur] = rerank_heap[mx];
            rerank_heap[mx]  = tmp;
            cur = mx;
        }
    }
    gv_free(rerank_heap);
    return found;
}

int rabitq_range_search(void *index, const GV_Vector *query, float radius,
                        GV_SearchResult *results, size_t max_results,
                        GV_DistanceType distance_type,
                        const char *filter_key, const char *filter_value) {
    if (!index || !query || !results || max_results == 0 || radius < 0.0f) return -1;
    GV_RaBitQIndex *idx = (GV_RaBitQIndex *)index;
    if (query->dimension != idx->dim) return -1;

    extern float distance(const GV_Vector *a, const GV_Vector *b,
                          GV_DistanceType dt);
    GV_Vector query_view;
    query_view.dimension = idx->dim;
    query_view.data      = (float *)query->data;
    query_view.metadata  = NULL;

    GV_Vector tmp_vec;
    tmp_vec.dimension = idx->dim;
    tmp_vec.metadata  = NULL;

    size_t found = 0;
    size_t n = idx->storage->count;
    for (size_t i = 0; i < n && found < max_results; i++) {
        if (soa_storage_is_deleted(idx->storage, i) == 1) continue;
        if (filter_key && filter_value) {
            GV_Metadata *m = soa_storage_get_metadata(idx->storage, i);
            if (!meta_match(m, filter_key, filter_value)) continue;
        }
        tmp_vec.data = (float *)soa_storage_get_data(idx->storage, i);
        float d = distance(&query_view, &tmp_vec, distance_type);
        if (d <= radius) {
            fill_result(idx, i, d, &results[found]);
            found++;
        }
    }
    return (int)found;
}

int rabitq_delete(void *index, size_t vector_index) {
    if (!index) return -1;
    GV_RaBitQIndex *idx = (GV_RaBitQIndex *)index;
    if (vector_index >= idx->storage->count) return -1;
    int rc = soa_storage_mark_deleted(idx->storage, vector_index);
    if (rc == 0 && vector_index < idx->cap)
        idx->deleted_arr[vector_index] = 1;
    return rc;
}

int rabitq_update(void *index, size_t vector_index,
                  const float *new_data, size_t dimension) {
    if (!index || !new_data) return -1;
    GV_RaBitQIndex *idx = (GV_RaBitQIndex *)index;
    if (dimension != idx->dim) return -1;
    if (vector_index >= idx->storage->count) return -1;

    int rc = soa_storage_update_data(idx->storage, vector_index, new_data);
    if (rc != 0) return rc;

    if (vector_index < idx->cap) {
        float *tmp = (float *)gv_alloc(idx->padded_dim * sizeof(float));
        if (tmp) {
            float norm_buf[idx->dim];
            memcpy(norm_buf, new_data, idx->dim * sizeof(float));
            l2_normalize(norm_buf, idx->dim);
            rht(norm_buf, tmp, idx->dim, idx->padded_dim, idx->signs);
            binarize(tmp, idx->codes + vector_index * idx->code_words,
                     idx->padded_dim);
            gv_free(tmp);
        }
    }
    return 0;
}

size_t rabitq_count(const void *index) {
    if (!index) return 0;
    const GV_RaBitQIndex *idx = (const GV_RaBitQIndex *)index;
    return idx->storage ? soa_storage_count(idx->storage) : idx->count;
}

void rabitq_destroy(void *index) {
    if (!index) return;
    GV_RaBitQIndex *idx = (GV_RaBitQIndex *)index;
    gv_free(idx->signs);
    gv_free(idx->codes);
    gv_free(idx->deleted_arr);
    if (idx->owns_storage && idx->storage)
        soa_storage_destroy(idx->storage);
    gv_free(idx);
}

/* ------------------------------------------------------------------ */
/* Persistence                                                          */
/* ------------------------------------------------------------------ */

#define RABITQ_MAGIC 0x52425451UL  /* "RBTQ" */

int rabitq_save(const void *index, FILE *out, uint32_t version) {
    if (!index || !out) return -1;
    const GV_RaBitQIndex *idx = (const GV_RaBitQIndex *)index;

    uint32_t magic = RABITQ_MAGIC;
    uint64_t dim   = (uint64_t)idx->dim;
    uint64_t seed  = idx->config.seed;
    uint64_t rf    = idx->config.rerank_factor;
    uint64_t cnt   = (uint64_t)idx->count;
    uint64_t cap   = (uint64_t)idx->cap;
    uint64_t cw    = (uint64_t)idx->code_words;

    if (fwrite(&magic, sizeof magic, 1, out) != 1) return -1;
    if (fwrite(&dim,   sizeof dim,   1, out) != 1) return -1;
    if (fwrite(&seed,  sizeof seed,  1, out) != 1) return -1;
    if (fwrite(&rf,    sizeof rf,    1, out) != 1) return -1;
    if (fwrite(&cnt,   sizeof cnt,   1, out) != 1) return -1;
    if (fwrite(&cap,   sizeof cap,   1, out) != 1) return -1;
    if (fwrite(&cw,    sizeof cw,    1, out) != 1) return -1;

    if (idx->cap > 0 && idx->codes) {
        if (fwrite(idx->codes, sizeof(uint64_t),
                   idx->cap * idx->code_words, out) !=
            idx->cap * idx->code_words) return -1;
    }
    if (idx->cap > 0 && idx->deleted_arr) {
        if (fwrite(idx->deleted_arr, 1, idx->cap, out) != idx->cap) return -1;
    }

    /* Save soa_storage (owns-storage case only — shared storage saved by DB). */
    if (idx->owns_storage && idx->storage) {
        if (soa_storage_save(idx->storage, out, version) != 0) return -1;
    }
    return 0;
}

int rabitq_load(void **index_ptr, FILE *in, size_t dimension,
                uint32_t version) {
    if (!index_ptr || !in) return -1;

    uint32_t magic = 0;
    if (fread(&magic, sizeof magic, 1, in) != 1 || magic != RABITQ_MAGIC)
        return -1;

    uint64_t dim, seed, rf, cnt, cap, cw;
    if (fread(&dim,  sizeof dim,  1, in) != 1) return -1;
    if (fread(&seed, sizeof seed, 1, in) != 1) return -1;
    if (fread(&rf,   sizeof rf,   1, in) != 1) return -1;
    if (fread(&cnt,  sizeof cnt,  1, in) != 1) return -1;
    if (fread(&cap,  sizeof cap,  1, in) != 1) return -1;
    if (fread(&cw,   sizeof cw,   1, in) != 1) return -1;

    if (dimension != 0 && (size_t)dim != dimension) return -1;

    GV_RaBitQConfig cfg = { .seed = seed, .rerank_factor = (size_t)rf };
    GV_RaBitQIndex *idx = (GV_RaBitQIndex *)rabitq_create((size_t)dim, &cfg, NULL);
    if (!idx) return -1;

    if (cap > 0) {
        idx->codes = (uint64_t *)gv_alloc((size_t)cap * (size_t)cw * sizeof(uint64_t));
        idx->deleted_arr = (uint8_t *)gv_calloc((size_t)cap, 1);
        if (!idx->codes || !idx->deleted_arr) {
            rabitq_destroy(idx);
            return -1;
        }
        if (fread(idx->codes, sizeof(uint64_t),
                  (size_t)cap * (size_t)cw, in) != (size_t)cap * (size_t)cw) {
            rabitq_destroy(idx);
            return -1;
        }
        if (fread(idx->deleted_arr, 1, (size_t)cap, in) != (size_t)cap) {
            rabitq_destroy(idx);
            return -1;
        }
        idx->cap   = (size_t)cap;
        idx->count = (size_t)cnt;
    }

    if (idx->owns_storage) {
        soa_storage_destroy(idx->storage);
        idx->storage = NULL;
        if (soa_storage_load(idx->storage, in, (uint32_t)dim) != 0) {
            rabitq_destroy(idx);
            return -1;
        }
    }

    *index_ptr = idx;
    return 0;
}
