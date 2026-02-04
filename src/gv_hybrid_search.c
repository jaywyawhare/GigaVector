/**
 * @file gv_hybrid_search.c
 * @brief Hybrid search implementation.
 */

#include "gigavector/gv_hybrid_search.h"
#include "gigavector/gv_database.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <pthread.h>
#include <sys/time.h>

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

struct GV_HybridSearcher {
    GV_Database *db;
    GV_BM25Index *bm25;
    GV_HybridConfig config;
    pthread_mutex_t mutex;
};

/* ============================================================================
 * Time Helpers
 * ============================================================================ */

static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* ============================================================================
 * Configuration
 * ============================================================================ */

static const GV_HybridConfig DEFAULT_CONFIG = {
    .fusion_type = GV_FUSION_LINEAR,
    .vector_weight = 0.5,
    .text_weight = 0.5,
    .rrf_k = 60.0,
    .distance_type = GV_DISTANCE_COSINE,
    .prefetch_k = 0
};

void gv_hybrid_config_init(GV_HybridConfig *config) {
    if (!config) return;
    *config = DEFAULT_CONFIG;
}

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

GV_HybridSearcher *gv_hybrid_create(GV_Database *db, GV_BM25Index *bm25,
                                     const GV_HybridConfig *config) {
    if (!db || !bm25) return NULL;

    GV_HybridSearcher *searcher = calloc(1, sizeof(GV_HybridSearcher));
    if (!searcher) return NULL;

    searcher->db = db;
    searcher->bm25 = bm25;
    searcher->config = config ? *config : DEFAULT_CONFIG;

    if (pthread_mutex_init(&searcher->mutex, NULL) != 0) {
        free(searcher);
        return NULL;
    }

    return searcher;
}

void gv_hybrid_destroy(GV_HybridSearcher *searcher) {
    if (!searcher) return;
    pthread_mutex_destroy(&searcher->mutex);
    free(searcher);
}

/* ============================================================================
 * Fusion Functions
 * ============================================================================ */

double gv_hybrid_linear_fusion(double vector_score, double text_score,
                                double vector_weight, double text_weight) {
    return vector_weight * vector_score + text_weight * text_score;
}

double gv_hybrid_rrf_fusion(size_t vector_rank, size_t text_rank, double k) {
    double score = 0.0;
    if (vector_rank > 0) {
        score += 1.0 / (k + (double)vector_rank);
    }
    if (text_rank > 0) {
        score += 1.0 / (k + (double)text_rank);
    }
    return score;
}

double gv_hybrid_normalize_score(double score, double min_score, double max_score) {
    if (max_score <= min_score) return 0.5;
    return (score - min_score) / (max_score - min_score);
}

/* ============================================================================
 * Internal Search Helpers
 * ============================================================================ */

typedef struct {
    size_t id;
    double vector_score;
    double text_score;
    size_t vector_rank;
    size_t text_rank;
    double combined_score;
} CandidateEntry;

static int compare_candidates(const void *a, const void *b) {
    const CandidateEntry *ca = (const CandidateEntry *)a;
    const CandidateEntry *cb = (const CandidateEntry *)b;
    if (cb->combined_score > ca->combined_score) return 1;
    if (cb->combined_score < ca->combined_score) return -1;
    return 0;
}

static CandidateEntry *find_or_add_candidate(CandidateEntry *candidates, size_t *count,
                                              size_t capacity, size_t id) {
    /* Find existing */
    for (size_t i = 0; i < *count; i++) {
        if (candidates[i].id == id) {
            return &candidates[i];
        }
    }

    /* Add new if capacity allows */
    if (*count >= capacity) return NULL;

    CandidateEntry *entry = &candidates[*count];
    memset(entry, 0, sizeof(*entry));
    entry->id = id;
    (*count)++;
    return entry;
}

/* ============================================================================
 * Search Operations
 * ============================================================================ */

int gv_hybrid_search(GV_HybridSearcher *searcher, const float *query_vector,
                     const char *query_text, size_t k, GV_HybridResult *results) {
    return gv_hybrid_search_with_stats(searcher, query_vector, query_text, k,
                                        results, NULL);
}

int gv_hybrid_search_with_stats(GV_HybridSearcher *searcher, const float *query_vector,
                                 const char *query_text, size_t k,
                                 GV_HybridResult *results, GV_HybridStats *stats) {
    if (!searcher || k == 0 || !results) return -1;
    if (!query_vector && !query_text) return -1;

    double start_time = get_time_ms();
    double vector_time = 0.0, text_time = 0.0;

    pthread_mutex_lock(&searcher->mutex);

    GV_HybridConfig *cfg = &searcher->config;

    /* Determine prefetch size */
    size_t prefetch_k = cfg->prefetch_k > 0 ? cfg->prefetch_k : k * 3;
    if (prefetch_k < k) prefetch_k = k;

    /* Allocate candidates (max 2 * prefetch_k) */
    size_t max_candidates = prefetch_k * 2;
    CandidateEntry *candidates = calloc(max_candidates, sizeof(CandidateEntry));
    if (!candidates) {
        pthread_mutex_unlock(&searcher->mutex);
        return -1;
    }
    size_t candidate_count = 0;

    /* Vector search */
    double vec_min = DBL_MAX, vec_max = -DBL_MAX;
    if (query_vector) {
        double vec_start = get_time_ms();

        GV_SearchResult *vec_results = malloc(prefetch_k * sizeof(GV_SearchResult));
        if (vec_results) {
            int vec_found = gv_db_search(searcher->db, query_vector, prefetch_k,
                                          vec_results, cfg->distance_type);

            for (int i = 0; i < vec_found; i++) {
                /* For distance, lower is better. Convert to similarity. */
                double sim = 1.0 / (1.0 + vec_results[i].distance);

                if (sim < vec_min) vec_min = sim;
                if (sim > vec_max) vec_max = sim;

                /* Use rank position as ID since GV_SearchResult doesn't store index */
                CandidateEntry *entry = find_or_add_candidate(candidates, &candidate_count,
                                                               max_candidates, (size_t)i);
                if (entry) {
                    entry->vector_score = sim;
                    entry->vector_rank = i + 1;
                }
            }
            free(vec_results);
        }

        vector_time = get_time_ms() - vec_start;
    }

    /* Text search */
    double txt_min = DBL_MAX, txt_max = -DBL_MAX;
    if (query_text) {
        double txt_start = get_time_ms();

        GV_BM25Result *txt_results = malloc(prefetch_k * sizeof(GV_BM25Result));
        if (txt_results) {
            int txt_found = gv_bm25_search(searcher->bm25, query_text, prefetch_k, txt_results);

            for (int i = 0; i < txt_found; i++) {
                if (txt_results[i].score < txt_min) txt_min = txt_results[i].score;
                if (txt_results[i].score > txt_max) txt_max = txt_results[i].score;

                CandidateEntry *entry = find_or_add_candidate(candidates, &candidate_count,
                                                               max_candidates, txt_results[i].doc_id);
                if (entry) {
                    entry->text_score = txt_results[i].score;
                    entry->text_rank = i + 1;
                }
            }
            free(txt_results);
        }

        text_time = get_time_ms() - txt_start;
    }

    /* Normalize scores and compute fusion */
    double fusion_start = get_time_ms();

    for (size_t i = 0; i < candidate_count; i++) {
        CandidateEntry *c = &candidates[i];

        double norm_vec = 0.0, norm_txt = 0.0;

        if (query_vector && c->vector_rank > 0) {
            norm_vec = gv_hybrid_normalize_score(c->vector_score, vec_min, vec_max);
        }
        if (query_text && c->text_rank > 0) {
            norm_txt = gv_hybrid_normalize_score(c->text_score, txt_min, txt_max);
        }

        switch (cfg->fusion_type) {
            case GV_FUSION_LINEAR:
                c->combined_score = gv_hybrid_linear_fusion(norm_vec, norm_txt,
                                                             cfg->vector_weight, cfg->text_weight);
                break;

            case GV_FUSION_RRF:
                c->combined_score = gv_hybrid_rrf_fusion(c->vector_rank, c->text_rank, cfg->rrf_k);
                break;

            case GV_FUSION_WEIGHTED_RRF:
                c->combined_score = cfg->vector_weight * (c->vector_rank > 0 ?
                    1.0 / (cfg->rrf_k + c->vector_rank) : 0.0) +
                    cfg->text_weight * (c->text_rank > 0 ?
                    1.0 / (cfg->rrf_k + c->text_rank) : 0.0);
                break;
        }
    }

    double fusion_time = get_time_ms() - fusion_start;

    /* Sort by combined score */
    qsort(candidates, candidate_count, sizeof(CandidateEntry), compare_candidates);

    /* Copy top k to results */
    size_t result_count = candidate_count < k ? candidate_count : k;
    for (size_t i = 0; i < result_count; i++) {
        results[i].vector_index = candidates[i].id;
        results[i].combined_score = candidates[i].combined_score;
        results[i].vector_score = candidates[i].vector_score;
        results[i].text_score = candidates[i].text_score;
        results[i].vector_rank = candidates[i].vector_rank;
        results[i].text_rank = candidates[i].text_rank;
    }

    free(candidates);

    pthread_mutex_unlock(&searcher->mutex);

    /* Fill statistics */
    if (stats) {
        stats->vector_candidates = query_vector ? prefetch_k : 0;
        stats->text_candidates = query_text ? prefetch_k : 0;
        stats->unique_candidates = candidate_count;
        stats->vector_search_time_ms = vector_time;
        stats->text_search_time_ms = text_time;
        stats->fusion_time_ms = fusion_time;
        stats->total_time_ms = get_time_ms() - start_time;
    }

    return (int)result_count;
}

int gv_hybrid_search_vector_only(GV_HybridSearcher *searcher, const float *query_vector,
                                  size_t k, GV_HybridResult *results) {
    return gv_hybrid_search(searcher, query_vector, NULL, k, results);
}

int gv_hybrid_search_text_only(GV_HybridSearcher *searcher, const char *query_text,
                                size_t k, GV_HybridResult *results) {
    return gv_hybrid_search(searcher, NULL, query_text, k, results);
}

/* ============================================================================
 * Configuration Updates
 * ============================================================================ */

int gv_hybrid_set_config(GV_HybridSearcher *searcher, const GV_HybridConfig *config) {
    if (!searcher || !config) return -1;

    pthread_mutex_lock(&searcher->mutex);
    searcher->config = *config;
    pthread_mutex_unlock(&searcher->mutex);

    return 0;
}

int gv_hybrid_get_config(const GV_HybridSearcher *searcher, GV_HybridConfig *config) {
    if (!searcher || !config) return -1;

    pthread_mutex_lock((pthread_mutex_t *)&searcher->mutex);
    *config = searcher->config;
    pthread_mutex_unlock((pthread_mutex_t *)&searcher->mutex);

    return 0;
}

int gv_hybrid_set_weights(GV_HybridSearcher *searcher, double vector_weight,
                           double text_weight) {
    if (!searcher) return -1;
    if (vector_weight < 0.0 || text_weight < 0.0) return -1;

    /* Normalize weights */
    double sum = vector_weight + text_weight;
    if (sum <= 0.0) return -1;

    pthread_mutex_lock(&searcher->mutex);
    searcher->config.vector_weight = vector_weight / sum;
    searcher->config.text_weight = text_weight / sum;
    pthread_mutex_unlock(&searcher->mutex);

    return 0;
}
