/**
 * @file gv_late_interaction.c
 * @brief ColBERT-style late interaction / MaxSim scoring for sparse-dense
 *        hybrid search.
 *
 * Each document is stored as a variable-length sequence of token embeddings.
 * A contiguous token pool avoids per-token allocations and is cache-friendly.
 * Search uses a two-stage pipeline:
 *   1. Rank all documents by dot(avg_query, avg_doc) to select a candidate pool.
 *   2. Compute full MaxSim on the top candidates and return the final top-k.
 * All public functions are protected by a pthread_rwlock_t.
 */

#include "gigavector/gv_late_interaction.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <pthread.h>

/* ============================================================================
 * SIMD headers (compile-time detection)
 * ============================================================================ */

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __SSE4_2__
#include <nmmintrin.h>
#include <emmintrin.h>
#endif

/* ============================================================================
 * Internal Constants
 * ============================================================================ */

#define GV_LI_MAGIC       "GV_LINT"
#define GV_LI_MAGIC_LEN   7
#define GV_LI_VERSION      1
#define GV_LI_INITIAL_DOC_CAPACITY   64
#define GV_LI_INITIAL_POOL_CAPACITY  (64 * 128)  /* tokens */

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

/**
 * @brief Per-document metadata stored in a dense array.
 */
typedef struct {
    size_t token_offset;   /**< Start position in the token pool (in floats: offset * dim). */
    size_t num_tokens;     /**< Number of token embeddings for this document. */
    int    deleted;        /**< Non-zero when logically deleted. */
    float *avg_embedding;  /**< Pre-computed average embedding (token_dimension floats). */
} GV_LIDocMeta;

/**
 * @brief Full index state (opaque to callers).
 */
struct GV_LateInteractionIndex {
    GV_LateInteractionConfig config;

    /* Token pool: contiguous buffer of (pool_used * dim) floats. */
    float  *token_pool;
    size_t  pool_used;        /**< Number of tokens currently stored. */
    size_t  pool_capacity;    /**< Allocated capacity (in tokens). */

    /* Document metadata array. */
    GV_LIDocMeta *docs;
    size_t        doc_count;
    size_t        doc_capacity;

    /* Bookkeeping. */
    size_t active_docs;       /**< Number of non-deleted documents. */
    size_t active_tokens;     /**< Total tokens across non-deleted documents. */

    pthread_rwlock_t rwlock;
};

/* ============================================================================
 * Dot-product helpers (scalar + SIMD)
 * ============================================================================ */

/**
 * @brief Scalar dot product fallback.
 */
static float gv_li_dot_scalar(const float *a, const float *b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

#ifdef __AVX2__
/**
 * @brief AVX2-accelerated dot product.
 */
static float gv_li_dot_avx2(const float *a, const float *b, size_t dim) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }

    /* Horizontal sum of 256-bit accumulator. */
    __m128 lo = _mm256_extractf128_ps(acc, 0);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float sum = _mm_cvtss_f32(s);

    /* Scalar tail. */
    for (; i < dim; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}
#endif /* __AVX2__ */

/**
 * @brief Best-available dot product for the current build.
 */
static float gv_li_dot(const float *a, const float *b, size_t dim) {
#ifdef __AVX2__
    return gv_li_dot_avx2(a, b, dim);
#else
    return gv_li_dot_scalar(a, b, dim);
#endif
}

/* ============================================================================
 * Max-heap helpers for top-k selection (max-heap on score, keeps lowest k)
 *
 * For MaxSim scoring, higher is better.  We use a min-heap so the root is
 * the *smallest* score in the current top-k set.  When a new candidate has
 * a score greater than the root it replaces it.
 * ============================================================================ */

typedef struct {
    float  score;
    size_t doc_idx;
} GV_LIHeapItem;

static void gv_li_heap_sift_down(GV_LIHeapItem *heap, size_t size, size_t i) {
    while (1) {
        size_t l = 2 * i + 1;
        size_t r = l + 1;
        size_t smallest = i;
        if (l < size && heap[l].score < heap[smallest].score) smallest = l;
        if (r < size && heap[r].score < heap[smallest].score) smallest = r;
        if (smallest == i) break;
        GV_LIHeapItem tmp = heap[i];
        heap[i] = heap[smallest];
        heap[smallest] = tmp;
        i = smallest;
    }
}

static void gv_li_heap_push(GV_LIHeapItem *heap, size_t *size, size_t capacity,
                             float score, size_t doc_idx) {
    if (*size < capacity) {
        heap[*size].score   = score;
        heap[*size].doc_idx = doc_idx;
        (*size)++;
        /* Sift up. */
        size_t i = *size - 1;
        while (i > 0) {
            size_t parent = (i - 1) / 2;
            if (heap[i].score < heap[parent].score) {
                GV_LIHeapItem tmp = heap[i];
                heap[i] = heap[parent];
                heap[parent] = tmp;
                i = parent;
            } else {
                break;
            }
        }
    } else if (score > heap[0].score) {
        heap[0].score   = score;
        heap[0].doc_idx = doc_idx;
        gv_li_heap_sift_down(heap, *size, 0);
    }
}

/* ============================================================================
 * Internal: compute average embedding for a document
 * ============================================================================ */

static float *gv_li_compute_avg(const float *tokens, size_t num_tokens, size_t dim) {
    float *avg = (float *)calloc(dim, sizeof(float));
    if (!avg) return NULL;

    for (size_t t = 0; t < num_tokens; t++) {
        const float *tok = tokens + t * dim;
        for (size_t d = 0; d < dim; d++) {
            avg[d] += tok[d];
        }
    }

    if (num_tokens > 0) {
        float inv = 1.0f / (float)num_tokens;
        for (size_t d = 0; d < dim; d++) {
            avg[d] *= inv;
        }
    }

    return avg;
}

/* ============================================================================
 * Internal: compute MaxSim score between query tokens and a document
 *
 * MaxSim(Q, D) = sum_{q in Q} max_{d in D} dot(q, d)
 * ============================================================================ */

static float gv_li_maxsim(const float *query_tokens, size_t num_query,
                           const float *doc_tokens, size_t num_doc,
                           size_t dim) {
    float total = 0.0f;

    for (size_t q = 0; q < num_query; q++) {
        const float *qvec = query_tokens + q * dim;
        float best = -FLT_MAX;

        for (size_t d = 0; d < num_doc; d++) {
            const float *dvec = doc_tokens + d * dim;
            float sim = gv_li_dot(qvec, dvec, dim);
            if (sim > best) {
                best = sim;
            }
        }

        total += best;
    }

    return total;
}

/* ============================================================================
 * Internal: grow the token pool
 * ============================================================================ */

static int gv_li_grow_pool(GV_LateInteractionIndex *idx, size_t needed_tokens) {
    if (idx->pool_used + needed_tokens <= idx->pool_capacity) return 0;

    size_t new_cap = idx->pool_capacity;
    while (new_cap < idx->pool_used + needed_tokens) {
        new_cap = new_cap < 1024 ? new_cap * 2 : new_cap + new_cap / 2;
    }

    float *new_pool = (float *)realloc(idx->token_pool,
                                       new_cap * idx->config.token_dimension * sizeof(float));
    if (!new_pool) return -1;

    idx->token_pool    = new_pool;
    idx->pool_capacity = new_cap;
    return 0;
}

/* ============================================================================
 * Internal: grow the document metadata array
 * ============================================================================ */

static int gv_li_grow_docs(GV_LateInteractionIndex *idx) {
    if (idx->doc_count < idx->doc_capacity) return 0;

    size_t new_cap = idx->doc_capacity * 2;
    GV_LIDocMeta *new_docs = (GV_LIDocMeta *)realloc(idx->docs,
                                                       new_cap * sizeof(GV_LIDocMeta));
    if (!new_docs) return -1;

    idx->docs         = new_docs;
    idx->doc_capacity = new_cap;
    return 0;
}

/* ============================================================================
 * Configuration
 * ============================================================================ */

static const GV_LateInteractionConfig DEFAULT_CONFIG = {
    .token_dimension = 128,
    .max_doc_tokens  = 512,
    .max_query_tokens = 32,
    .candidate_pool  = 1000
};

void gv_late_interaction_config_init(GV_LateInteractionConfig *config) {
    if (!config) return;
    *config = DEFAULT_CONFIG;
}

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

GV_LateInteractionIndex *gv_late_interaction_create(const GV_LateInteractionConfig *config) {
    GV_LateInteractionConfig cfg = config ? *config : DEFAULT_CONFIG;

    /* Sanity-check / apply defaults. */
    if (cfg.token_dimension == 0) cfg.token_dimension = 128;
    if (cfg.max_doc_tokens  == 0) cfg.max_doc_tokens  = 512;
    if (cfg.max_query_tokens == 0) cfg.max_query_tokens = 32;
    if (cfg.candidate_pool  == 0) cfg.candidate_pool  = 1000;

    GV_LateInteractionIndex *idx = (GV_LateInteractionIndex *)calloc(
        1, sizeof(GV_LateInteractionIndex));
    if (!idx) return NULL;

    idx->config = cfg;

    /* Allocate token pool. */
    idx->pool_capacity = GV_LI_INITIAL_POOL_CAPACITY;
    idx->token_pool = (float *)malloc(
        idx->pool_capacity * cfg.token_dimension * sizeof(float));
    if (!idx->token_pool) {
        free(idx);
        return NULL;
    }
    idx->pool_used = 0;

    /* Allocate document metadata array. */
    idx->doc_capacity = GV_LI_INITIAL_DOC_CAPACITY;
    idx->docs = (GV_LIDocMeta *)calloc(idx->doc_capacity, sizeof(GV_LIDocMeta));
    if (!idx->docs) {
        free(idx->token_pool);
        free(idx);
        return NULL;
    }
    idx->doc_count     = 0;
    idx->active_docs   = 0;
    idx->active_tokens = 0;

    if (pthread_rwlock_init(&idx->rwlock, NULL) != 0) {
        free(idx->docs);
        free(idx->token_pool);
        free(idx);
        return NULL;
    }

    return idx;
}

void gv_late_interaction_destroy(GV_LateInteractionIndex *index) {
    if (!index) return;

    /* Free per-document average embeddings. */
    for (size_t i = 0; i < index->doc_count; i++) {
        free(index->docs[i].avg_embedding);
    }

    free(index->docs);
    free(index->token_pool);
    pthread_rwlock_destroy(&index->rwlock);
    free(index);
}

/* ============================================================================
 * Add Document
 * ============================================================================ */

int gv_late_interaction_add_doc(GV_LateInteractionIndex *index,
                                 const float *token_embeddings, size_t num_tokens) {
    if (!index || !token_embeddings || num_tokens == 0) return -1;
    if (num_tokens > index->config.max_doc_tokens) return -1;

    pthread_rwlock_wrlock(&index->rwlock);

    /* Ensure pool capacity. */
    if (gv_li_grow_pool(index, num_tokens) != 0) {
        pthread_rwlock_unlock(&index->rwlock);
        return -1;
    }

    /* Ensure document array capacity. */
    if (gv_li_grow_docs(index) != 0) {
        pthread_rwlock_unlock(&index->rwlock);
        return -1;
    }

    size_t dim = index->config.token_dimension;
    size_t offset = index->pool_used;

    /* Copy token embeddings into the pool. */
    memcpy(index->token_pool + offset * dim,
           token_embeddings,
           num_tokens * dim * sizeof(float));

    /* Pre-compute average embedding for first-stage retrieval. */
    float *avg = gv_li_compute_avg(token_embeddings, num_tokens, dim);
    if (!avg) {
        pthread_rwlock_unlock(&index->rwlock);
        return -1;
    }

    GV_LIDocMeta *doc = &index->docs[index->doc_count];
    doc->token_offset  = offset;
    doc->num_tokens    = num_tokens;
    doc->deleted       = 0;
    doc->avg_embedding = avg;

    index->pool_used     += num_tokens;
    index->doc_count++;
    index->active_docs++;
    index->active_tokens += num_tokens;

    pthread_rwlock_unlock(&index->rwlock);
    return 0;
}

/* ============================================================================
 * Delete Document
 * ============================================================================ */

int gv_late_interaction_delete(GV_LateInteractionIndex *index, size_t doc_index) {
    if (!index) return -1;

    pthread_rwlock_wrlock(&index->rwlock);

    if (doc_index >= index->doc_count || index->docs[doc_index].deleted) {
        pthread_rwlock_unlock(&index->rwlock);
        return -1;
    }

    index->docs[doc_index].deleted = 1;
    index->active_docs--;
    index->active_tokens -= index->docs[doc_index].num_tokens;

    /* Free average embedding. */
    free(index->docs[doc_index].avg_embedding);
    index->docs[doc_index].avg_embedding = NULL;

    pthread_rwlock_unlock(&index->rwlock);
    return 0;
}

/* ============================================================================
 * Search (two-stage: avg-dot candidate selection, then full MaxSim)
 * ============================================================================ */

int gv_late_interaction_search(const GV_LateInteractionIndex *index,
                                const float *query_tokens, size_t num_query_tokens,
                                size_t k, GV_LateInteractionResult *results) {
    if (!index || !query_tokens || num_query_tokens == 0 || k == 0 || !results) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&index->rwlock);

    if (index->active_docs == 0) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&index->rwlock);
        return 0;
    }

    size_t dim = index->config.token_dimension;

    /* --- Compute average query embedding for first-stage scoring. --- */
    float *avg_query = gv_li_compute_avg(query_tokens, num_query_tokens, dim);
    if (!avg_query) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&index->rwlock);
        return -1;
    }

    /* Determine candidate pool size.  If the index has fewer active docs
     * than the configured pool, or k >= pool, skip the first stage. */
    size_t pool_size = index->config.candidate_pool;
    int skip_first_stage = (index->active_docs <= pool_size) || (k >= pool_size);

    /* --- Stage 1: select candidate documents by avg-embedding dot product. --- */
    size_t  *candidates     = NULL;
    size_t   num_candidates = 0;

    if (skip_first_stage) {
        /* Use all non-deleted documents directly. */
        candidates = (size_t *)malloc(index->active_docs * sizeof(size_t));
        if (!candidates) {
            free(avg_query);
            pthread_rwlock_unlock((pthread_rwlock_t *)&index->rwlock);
            return -1;
        }
        for (size_t i = 0; i < index->doc_count; i++) {
            if (!index->docs[i].deleted) {
                candidates[num_candidates++] = i;
            }
        }
    } else {
        /* Use a min-heap of size pool_size to keep the top-scoring candidates
         * by average-embedding dot product. */
        GV_LIHeapItem *stage1_heap = (GV_LIHeapItem *)malloc(
            pool_size * sizeof(GV_LIHeapItem));
        if (!stage1_heap) {
            free(avg_query);
            pthread_rwlock_unlock((pthread_rwlock_t *)&index->rwlock);
            return -1;
        }
        size_t heap_size = 0;

        for (size_t i = 0; i < index->doc_count; i++) {
            if (index->docs[i].deleted) continue;

            float s = gv_li_dot(avg_query, index->docs[i].avg_embedding, dim);
            gv_li_heap_push(stage1_heap, &heap_size, pool_size, s, i);
        }

        /* Extract candidate indices from heap. */
        num_candidates = heap_size;
        candidates = (size_t *)malloc(num_candidates * sizeof(size_t));
        if (!candidates) {
            free(stage1_heap);
            free(avg_query);
            pthread_rwlock_unlock((pthread_rwlock_t *)&index->rwlock);
            return -1;
        }
        for (size_t i = 0; i < num_candidates; i++) {
            candidates[i] = stage1_heap[i].doc_idx;
        }
        free(stage1_heap);
    }

    free(avg_query);

    /* --- Stage 2: full MaxSim on candidates, keep top k. --- */
    size_t effective_k = k < num_candidates ? k : num_candidates;

    GV_LIHeapItem *result_heap = (GV_LIHeapItem *)malloc(
        effective_k * sizeof(GV_LIHeapItem));
    if (!result_heap) {
        free(candidates);
        pthread_rwlock_unlock((pthread_rwlock_t *)&index->rwlock);
        return -1;
    }
    size_t result_heap_size = 0;

    for (size_t c = 0; c < num_candidates; c++) {
        size_t di = candidates[c];
        const GV_LIDocMeta *doc = &index->docs[di];
        const float *doc_tokens = index->token_pool + doc->token_offset * dim;

        float score = gv_li_maxsim(query_tokens, num_query_tokens,
                                    doc_tokens, doc->num_tokens, dim);

        gv_li_heap_push(result_heap, &result_heap_size, effective_k, score, di);
    }

    free(candidates);

    /* Extract results from min-heap in descending score order. */
    int n = (int)result_heap_size;
    for (int i = n - 1; i >= 0; i--) {
        results[i].doc_index = result_heap[0].doc_idx;
        results[i].score     = result_heap[0].score;

        /* Pop root. */
        result_heap[0] = result_heap[result_heap_size - 1];
        result_heap_size--;
        if (result_heap_size > 0) {
            gv_li_heap_sift_down(result_heap, result_heap_size, 0);
        }
    }

    free(result_heap);

    pthread_rwlock_unlock((pthread_rwlock_t *)&index->rwlock);
    return n;
}

/* ============================================================================
 * Stats & Count
 * ============================================================================ */

int gv_late_interaction_get_stats(const GV_LateInteractionIndex *index,
                                   GV_LateInteractionStats *stats) {
    if (!index || !stats) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&index->rwlock);

    stats->total_documents    = index->active_docs;
    stats->total_tokens_stored = index->active_tokens;

    size_t dim = index->config.token_dimension;

    /* Estimate memory usage. */
    stats->memory_bytes = sizeof(GV_LateInteractionIndex);
    stats->memory_bytes += index->pool_capacity * dim * sizeof(float);
    stats->memory_bytes += index->doc_capacity * sizeof(GV_LIDocMeta);
    /* Average embeddings for non-deleted docs. */
    for (size_t i = 0; i < index->doc_count; i++) {
        if (index->docs[i].avg_embedding) {
            stats->memory_bytes += dim * sizeof(float);
        }
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&index->rwlock);
    return 0;
}

size_t gv_late_interaction_count(const GV_LateInteractionIndex *index) {
    if (!index) return 0;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&index->rwlock);
    size_t count = index->active_docs;
    pthread_rwlock_unlock((pthread_rwlock_t *)&index->rwlock);

    return count;
}

/* ============================================================================
 * Serialization helpers
 * ============================================================================ */

static int gv_li_write_u32(FILE *f, uint32_t v) {
    return fwrite(&v, sizeof(uint32_t), 1, f) == 1 ? 0 : -1;
}

static int gv_li_read_u32(FILE *f, uint32_t *v) {
    return (v && fread(v, sizeof(uint32_t), 1, f) == 1) ? 0 : -1;
}

static int gv_li_write_u64(FILE *f, uint64_t v) {
    return fwrite(&v, sizeof(uint64_t), 1, f) == 1 ? 0 : -1;
}

static int gv_li_read_u64(FILE *f, uint64_t *v) {
    return (v && fread(v, sizeof(uint64_t), 1, f) == 1) ? 0 : -1;
}

/* ============================================================================
 * Save
 * ============================================================================ */

int gv_late_interaction_save(const GV_LateInteractionIndex *index,
                              const char *filepath) {
    if (!index || !filepath) return -1;

    FILE *fp = fopen(filepath, "wb");
    if (!fp) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&index->rwlock);

    /* Magic + version. */
    if (fwrite(GV_LI_MAGIC, 1, GV_LI_MAGIC_LEN, fp) != GV_LI_MAGIC_LEN) goto fail;
    if (gv_li_write_u32(fp, GV_LI_VERSION) != 0) goto fail;

    /* Configuration. */
    if (gv_li_write_u64(fp, (uint64_t)index->config.token_dimension) != 0) goto fail;
    if (gv_li_write_u64(fp, (uint64_t)index->config.max_doc_tokens) != 0) goto fail;
    if (gv_li_write_u64(fp, (uint64_t)index->config.max_query_tokens) != 0) goto fail;
    if (gv_li_write_u64(fp, (uint64_t)index->config.candidate_pool) != 0) goto fail;

    /* Count non-deleted documents. */
    uint32_t active = (uint32_t)index->active_docs;
    if (gv_li_write_u32(fp, active) != 0) goto fail;

    size_t dim = index->config.token_dimension;

    /* Write each non-deleted document. */
    for (size_t i = 0; i < index->doc_count; i++) {
        const GV_LIDocMeta *doc = &index->docs[i];
        if (doc->deleted) continue;

        if (gv_li_write_u32(fp, (uint32_t)doc->num_tokens) != 0) goto fail;

        size_t floats = doc->num_tokens * dim;
        const float *tok_data = index->token_pool + doc->token_offset * dim;
        if (floats > 0) {
            if (fwrite(tok_data, sizeof(float), floats, fp) != floats) goto fail;
        }
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&index->rwlock);
    fclose(fp);
    return 0;

fail:
    pthread_rwlock_unlock((pthread_rwlock_t *)&index->rwlock);
    fclose(fp);
    return -1;
}

/* ============================================================================
 * Load
 * ============================================================================ */

GV_LateInteractionIndex *gv_late_interaction_load(const char *filepath) {
    if (!filepath) return NULL;

    FILE *fp = fopen(filepath, "rb");
    if (!fp) return NULL;

    /* Verify magic. */
    char magic[GV_LI_MAGIC_LEN];
    if (fread(magic, 1, GV_LI_MAGIC_LEN, fp) != GV_LI_MAGIC_LEN ||
        memcmp(magic, GV_LI_MAGIC, GV_LI_MAGIC_LEN) != 0) {
        fclose(fp);
        return NULL;
    }

    /* Verify version. */
    uint32_t version = 0;
    if (gv_li_read_u32(fp, &version) != 0 || version != GV_LI_VERSION) {
        fclose(fp);
        return NULL;
    }

    /* Read configuration. */
    uint64_t td = 0, mdt = 0, mqt = 0, cp = 0;
    if (gv_li_read_u64(fp, &td)  != 0) { fclose(fp); return NULL; }
    if (gv_li_read_u64(fp, &mdt) != 0) { fclose(fp); return NULL; }
    if (gv_li_read_u64(fp, &mqt) != 0) { fclose(fp); return NULL; }
    if (gv_li_read_u64(fp, &cp)  != 0) { fclose(fp); return NULL; }

    GV_LateInteractionConfig cfg;
    cfg.token_dimension  = (size_t)td;
    cfg.max_doc_tokens   = (size_t)mdt;
    cfg.max_query_tokens = (size_t)mqt;
    cfg.candidate_pool   = (size_t)cp;

    /* Read document count. */
    uint32_t doc_count = 0;
    if (gv_li_read_u32(fp, &doc_count) != 0) { fclose(fp); return NULL; }

    /* Create the index. */
    GV_LateInteractionIndex *idx = gv_late_interaction_create(&cfg);
    if (!idx) { fclose(fp); return NULL; }

    size_t dim = cfg.token_dimension;

    /* Read documents. */
    for (uint32_t i = 0; i < doc_count; i++) {
        uint32_t num_tokens = 0;
        if (gv_li_read_u32(fp, &num_tokens) != 0) {
            gv_late_interaction_destroy(idx);
            fclose(fp);
            return NULL;
        }

        size_t floats = (size_t)num_tokens * dim;
        float *tokens = NULL;
        if (floats > 0) {
            tokens = (float *)malloc(floats * sizeof(float));
            if (!tokens) {
                gv_late_interaction_destroy(idx);
                fclose(fp);
                return NULL;
            }
            if (fread(tokens, sizeof(float), floats, fp) != floats) {
                free(tokens);
                gv_late_interaction_destroy(idx);
                fclose(fp);
                return NULL;
            }
        }

        /* gv_late_interaction_add_doc acquires wrlock internally.
         * Since we are the sole owner at this point (just created the index),
         * this is safe. */
        int rc = gv_late_interaction_add_doc(idx, tokens, (size_t)num_tokens);
        free(tokens);

        if (rc != 0) {
            gv_late_interaction_destroy(idx);
            fclose(fp);
            return NULL;
        }
    }

    fclose(fp);
    return idx;
}
