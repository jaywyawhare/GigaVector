/**
 * @file late_interaction.c
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

#include "multimodal/late_interaction.h"
#include "core/heap.h"
#include "core/utils.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include <pthread.h>

/* SIMD headers (compile-time detection) */

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __SSE4_2__
#include <nmmintrin.h>
#include <emmintrin.h>
#endif

#define GV_LI_MAGIC       "GV_LINT"
#define GV_LI_MAGIC_LEN   7
#define GV_LI_VERSION      1
#define GV_LI_INITIAL_DOC_CAPACITY   64
#define GV_LI_INITIAL_POOL_CAPACITY  (64 * 128)  /* tokens */

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

    float  *token_pool;
    size_t  pool_used;        /**< Number of tokens currently stored. */
    size_t  pool_capacity;    /**< Allocated capacity (in tokens). */

    GV_LIDocMeta *docs;
    size_t        doc_count;
    size_t        doc_capacity;

    size_t active_docs;       /**< Number of non-deleted documents. */
    size_t active_tokens;     /**< Total tokens across non-deleted documents. */

    pthread_rwlock_t rwlock;
};

/**
 * @brief Scalar dot product fallback.
 */
static float li_dot_scalar(const float *a, const float *b, size_t dim) {
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
static float li_dot_avx2(const float *a, const float *b, size_t dim) {
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
static float li_dot(const float *a, const float *b, size_t dim) {
#ifdef __AVX2__
    return li_dot_avx2(a, b, dim);
#else
    return li_dot_scalar(a, b, dim);
#endif
}

/* Max-heap helpers for top-k selection (max-heap on dist, keeps lowest k)
 *
 * For MaxSim scoring, higher is better.  The macro-generated heap keeps the
 * largest dist at the root; when a new candidate's dist is smaller than the
 * root it replaces it, maintaining the top-k highest-scoring entries.
 */

typedef struct {
    float  dist;
    size_t doc_idx;
} GV_LIHeapItem;

GV_MIN_HEAP_DEFINE(li_heap, GV_LIHeapItem)

static float *li_compute_avg(const float *tokens, size_t num_tokens, size_t dim) {
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

/* Internal: compute MaxSim score between query tokens and a document
 *
 * MaxSim(Q, D) = sum_{q in Q} max_{d in D} dot(q, d)
 */

static float li_maxsim(const float *query_tokens, size_t num_query,
                           const float *doc_tokens, size_t num_doc,
                           size_t dim) {
    float total = 0.0f;

    for (size_t q = 0; q < num_query; q++) {
        const float *qvec = query_tokens + q * dim;
        float best = -FLT_MAX;

        for (size_t d = 0; d < num_doc; d++) {
            const float *dvec = doc_tokens + d * dim;
            float sim = li_dot(qvec, dvec, dim);
            if (sim > best) {
                best = sim;
            }
        }

        total += best;
    }

    return total;
}

static int li_grow_pool(GV_LateInteractionIndex *idx, size_t needed_tokens) {
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

static int li_grow_docs(GV_LateInteractionIndex *idx) {
    if (idx->doc_count < idx->doc_capacity) return 0;

    size_t new_cap = idx->doc_capacity * 2;
    GV_LIDocMeta *new_docs = (GV_LIDocMeta *)realloc(idx->docs,
                                                       new_cap * sizeof(GV_LIDocMeta));
    if (!new_docs) return -1;

    idx->docs         = new_docs;
    idx->doc_capacity = new_cap;
    return 0;
}

static const GV_LateInteractionConfig DEFAULT_CONFIG = {
    .token_dimension = 128,
    .max_doc_tokens  = 512,
    .max_query_tokens = 32,
    .candidate_pool  = 1000
};

void late_interaction_config_init(GV_LateInteractionConfig *config) {
    if (!config) return;
    *config = DEFAULT_CONFIG;
}

GV_LateInteractionIndex *late_interaction_create(const GV_LateInteractionConfig *config) {
    GV_LateInteractionConfig cfg = config ? *config : DEFAULT_CONFIG;

    if (cfg.token_dimension == 0) cfg.token_dimension = 128;
    if (cfg.max_doc_tokens  == 0) cfg.max_doc_tokens  = 512;
    if (cfg.max_query_tokens == 0) cfg.max_query_tokens = 32;
    if (cfg.candidate_pool  == 0) cfg.candidate_pool  = 1000;

    GV_LateInteractionIndex *idx = (GV_LateInteractionIndex *)calloc(
        1, sizeof(GV_LateInteractionIndex));
    if (!idx) return NULL;

    idx->config = cfg;

    idx->pool_capacity = GV_LI_INITIAL_POOL_CAPACITY;
    idx->token_pool = (float *)malloc(
        idx->pool_capacity * cfg.token_dimension * sizeof(float));
    if (!idx->token_pool) {
        free(idx);
        return NULL;
    }
    idx->pool_used = 0;

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

void late_interaction_destroy(GV_LateInteractionIndex *index) {
    if (!index) return;

    for (size_t i = 0; i < index->doc_count; i++) {
        free(index->docs[i].avg_embedding);
    }

    free(index->docs);
    free(index->token_pool);
    pthread_rwlock_destroy(&index->rwlock);
    free(index);
}

int late_interaction_add_doc(GV_LateInteractionIndex *index,
                                 const float *token_embeddings, size_t num_tokens) {
    if (!index || !token_embeddings || num_tokens == 0) return -1;
    if (num_tokens > index->config.max_doc_tokens) return -1;

    pthread_rwlock_wrlock(&index->rwlock);

    if (li_grow_pool(index, num_tokens) != 0) {
        pthread_rwlock_unlock(&index->rwlock);
        return -1;
    }

    if (li_grow_docs(index) != 0) {
        pthread_rwlock_unlock(&index->rwlock);
        return -1;
    }

    size_t dim = index->config.token_dimension;
    size_t offset = index->pool_used;

    memcpy(index->token_pool + offset * dim,
           token_embeddings,
           num_tokens * dim * sizeof(float));

    float *avg = li_compute_avg(token_embeddings, num_tokens, dim);
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

int late_interaction_delete(GV_LateInteractionIndex *index, size_t doc_index) {
    if (!index) return -1;

    pthread_rwlock_wrlock(&index->rwlock);

    if (doc_index >= index->doc_count || index->docs[doc_index].deleted) {
        pthread_rwlock_unlock(&index->rwlock);
        return -1;
    }

    index->docs[doc_index].deleted = 1;
    index->active_docs--;
    index->active_tokens -= index->docs[doc_index].num_tokens;

    free(index->docs[doc_index].avg_embedding);
    index->docs[doc_index].avg_embedding = NULL;

    pthread_rwlock_unlock(&index->rwlock);
    return 0;
}

int late_interaction_search(const GV_LateInteractionIndex *index,
                                const float *query_tokens, size_t num_query_tokens,
                                size_t k, GV_LateInteractionResult *results) {
    if (!index || !query_tokens || num_query_tokens == 0 || k == 0 || !results) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&index->rwlock);

    if (index->active_docs == 0) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&index->rwlock);
        return 0;
    }

    size_t dim = index->config.token_dimension;

    float *avg_query = li_compute_avg(query_tokens, num_query_tokens, dim);
    if (!avg_query) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&index->rwlock);
        return -1;
    }

    /* Determine candidate pool size.  If the index has fewer active docs
     * than the configured pool, or k >= pool, skip the first stage. */
    size_t pool_size = index->config.candidate_pool;
    int skip_first_stage = (index->active_docs <= pool_size) || (k >= pool_size);

    size_t  *candidates     = NULL;
    size_t   num_candidates = 0;

    if (skip_first_stage) {
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

            float s = li_dot(avg_query, index->docs[i].avg_embedding, dim);
            li_heap_push(stage1_heap, &heap_size, pool_size, (GV_LIHeapItem){s, i});
        }

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

        float score = li_maxsim(query_tokens, num_query_tokens,
                                    doc_tokens, doc->num_tokens, dim);

        li_heap_push(result_heap, &result_heap_size, effective_k, (GV_LIHeapItem){score, di});
    }

    free(candidates);

    int n = (int)result_heap_size;
    for (int i = n - 1; i >= 0; i--) {
        results[i].doc_index = result_heap[0].doc_idx;
        results[i].score     = result_heap[0].dist;

        result_heap[0] = result_heap[result_heap_size - 1];
        result_heap_size--;
        if (result_heap_size > 0) {
            li_heap_sift_down(result_heap, result_heap_size, 0);
        }
    }

    free(result_heap);

    pthread_rwlock_unlock((pthread_rwlock_t *)&index->rwlock);
    return n;
}

int late_interaction_get_stats(const GV_LateInteractionIndex *index,
                                   GV_LateInteractionStats *stats) {
    if (!index || !stats) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&index->rwlock);

    stats->total_documents    = index->active_docs;
    stats->total_tokens_stored = index->active_tokens;

    size_t dim = index->config.token_dimension;

    stats->memory_bytes = sizeof(GV_LateInteractionIndex);
    stats->memory_bytes += index->pool_capacity * dim * sizeof(float);
    stats->memory_bytes += index->doc_capacity * sizeof(GV_LIDocMeta);
    for (size_t i = 0; i < index->doc_count; i++) {
        if (index->docs[i].avg_embedding) {
            stats->memory_bytes += dim * sizeof(float);
        }
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&index->rwlock);
    return 0;
}

size_t late_interaction_count(const GV_LateInteractionIndex *index) {
    if (!index) return 0;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&index->rwlock);
    size_t count = index->active_docs;
    pthread_rwlock_unlock((pthread_rwlock_t *)&index->rwlock);

    return count;
}


int late_interaction_save(const GV_LateInteractionIndex *index,
                              const char *filepath) {
    if (!index || !filepath) return -1;

    FILE *fp = fopen(filepath, "wb");
    if (!fp) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&index->rwlock);

    if (fwrite(GV_LI_MAGIC, 1, GV_LI_MAGIC_LEN, fp) != GV_LI_MAGIC_LEN) goto fail;
    if (write_u32(fp, GV_LI_VERSION) != 0) goto fail;

    if (write_u64(fp, (uint64_t)index->config.token_dimension) != 0) goto fail;
    if (write_u64(fp, (uint64_t)index->config.max_doc_tokens) != 0) goto fail;
    if (write_u64(fp, (uint64_t)index->config.max_query_tokens) != 0) goto fail;
    if (write_u64(fp, (uint64_t)index->config.candidate_pool) != 0) goto fail;

    uint32_t active = (uint32_t)index->active_docs;
    if (write_u32(fp, active) != 0) goto fail;

    size_t dim = index->config.token_dimension;

    for (size_t i = 0; i < index->doc_count; i++) {
        const GV_LIDocMeta *doc = &index->docs[i];
        if (doc->deleted) continue;

        if (write_u32(fp, (uint32_t)doc->num_tokens) != 0) goto fail;

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

GV_LateInteractionIndex *late_interaction_load(const char *filepath) {
    if (!filepath) return NULL;

    FILE *fp = fopen(filepath, "rb");
    if (!fp) return NULL;

    char magic[GV_LI_MAGIC_LEN];
    if (fread(magic, 1, GV_LI_MAGIC_LEN, fp) != GV_LI_MAGIC_LEN ||
        memcmp(magic, GV_LI_MAGIC, GV_LI_MAGIC_LEN) != 0) {
        fclose(fp);
        return NULL;
    }

    uint32_t version = 0;
    if (read_u32(fp, &version) != 0 || version != GV_LI_VERSION) {
        fclose(fp);
        return NULL;
    }

    uint64_t td = 0, mdt = 0, mqt = 0, cp = 0;
    if (read_u64(fp, &td)  != 0) { fclose(fp); return NULL; }
    if (read_u64(fp, &mdt) != 0) { fclose(fp); return NULL; }
    if (read_u64(fp, &mqt) != 0) { fclose(fp); return NULL; }
    if (read_u64(fp, &cp)  != 0) { fclose(fp); return NULL; }

    GV_LateInteractionConfig cfg;
    cfg.token_dimension  = (size_t)td;
    cfg.max_doc_tokens   = (size_t)mdt;
    cfg.max_query_tokens = (size_t)mqt;
    cfg.candidate_pool   = (size_t)cp;

    uint32_t doc_count = 0;
    if (read_u32(fp, &doc_count) != 0) { fclose(fp); return NULL; }

    GV_LateInteractionIndex *idx = late_interaction_create(&cfg);
    if (!idx) { fclose(fp); return NULL; }

    size_t dim = cfg.token_dimension;

    for (uint32_t i = 0; i < doc_count; i++) {
        uint32_t num_tokens = 0;
        if (read_u32(fp, &num_tokens) != 0) {
            late_interaction_destroy(idx);
            fclose(fp);
            return NULL;
        }

        size_t floats = (size_t)num_tokens * dim;
        float *tokens = NULL;
        if (floats > 0) {
            tokens = (float *)malloc(floats * sizeof(float));
            if (!tokens) {
                late_interaction_destroy(idx);
                fclose(fp);
                return NULL;
            }
            if (fread(tokens, sizeof(float), floats, fp) != floats) {
                free(tokens);
                late_interaction_destroy(idx);
                fclose(fp);
                return NULL;
            }
        }

        /* late_interaction_add_doc acquires wrlock internally.
         * Since we are the sole owner at this point (just created the index),
         * this is safe. */
        int rc = late_interaction_add_doc(idx, tokens, (size_t)num_tokens);
        free(tokens);

        if (rc != 0) {
            late_interaction_destroy(idx);
            fclose(fp);
            return NULL;
        }
    }

    fclose(fp);
    return idx;
}
