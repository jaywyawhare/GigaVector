/**
 * @file gv_phased_ranking.c
 * @brief Multi-stage phased ranking pipeline implementation.
 *
 * Executes a configurable sequence of ranking phases:
 *   1. ANN -- fast approximate nearest-neighbor retrieval via gv_db_search().
 *   2. RERANK_EXPR -- expression-based reranking via gv_ranking.h.
 *   3. RERANK_MMR  -- diversity reranking via gv_mmr.h.
 *   4. RERANK_CALLBACK -- user-supplied scoring function.
 *   5. FILTER -- metadata filter expression via gv_filter.h.
 *
 * Each phase consumes the previous phase's output candidates, refines or
 * reorders them, and passes forward at most output_k results.
 *
 * Thread safety is ensured via a pthread_mutex_t on the pipeline.
 * Per-phase timing uses CLOCK_MONOTONIC via clock_gettime().
 */

#include "gigavector/gv_phased_ranking.h"
#include "gigavector/gv_database.h"
#include "gigavector/gv_distance.h"
#include "gigavector/gv_ranking.h"
#include "gigavector/gv_mmr.h"
#include "gigavector/gv_filter.h"
#include "gigavector/gv_types.h"
#include "gigavector/gv_vector.h"
#include "gigavector/gv_soa_storage.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <pthread.h>

/* ============================================================================
 * Internal Constants
 * ============================================================================ */

/** Maximum number of phases in a single pipeline. */
#define GV_PIPELINE_MAX_PHASES 8

/* ============================================================================
 * Internal Types
 * ============================================================================ */

/**
 * @brief Internal per-phase statistics recorded during execution.
 */
typedef struct {
    size_t input_count;
    size_t output_count;
    double latency_ms;
} PhaseStats;

/**
 * @brief Opaque pipeline structure.
 */
struct GV_Pipeline {
    const GV_Database *db;

    GV_PhaseConfig phases[GV_PIPELINE_MAX_PHASES];
    size_t         phase_count;

    PhaseStats     stats[GV_PIPELINE_MAX_PHASES];
    double         total_latency_ms;

    pthread_mutex_t mutex;
};

/**
 * @brief Internal candidate carried between phases.
 */
typedef struct {
    size_t index;       /**< Database vector index. */
    float  score;       /**< Current score. */
    int    phase_id;    /**< Last phase that touched this candidate. */
} Candidate;

/* ============================================================================
 * Timing Helpers
 * ============================================================================ */

static double timespec_to_ms(const struct timespec *ts) {
    return (double)ts->tv_sec * 1000.0 + (double)ts->tv_nsec / 1.0e6;
}

static double elapsed_ms(const struct timespec *start, const struct timespec *end) {
    return timespec_to_ms(end) - timespec_to_ms(start);
}

/* ============================================================================
 * Sorting Helpers
 * ============================================================================ */

static int compare_candidates_desc(const void *a, const void *b) {
    const Candidate *ca = (const Candidate *)a;
    const Candidate *cb = (const Candidate *)b;
    if (cb->score > ca->score) return  1;
    if (cb->score < ca->score) return -1;
    return 0;
}

/* ============================================================================
 * SoA Index Recovery
 * ============================================================================ */

/**
 * @brief Recover the SoA storage index from a GV_SearchResult.
 *
 * GV_SearchResult.vector->data points into contiguous SoA storage at
 * offset (index * dimension).  We recover the index by pointer arithmetic.
 */
static size_t result_to_soa_index(const GV_Database *db, const GV_SearchResult *sr) {
    if (!sr || !sr->vector || !sr->vector->data) return (size_t)-1;

    size_t dim = gv_database_dimension(db);
    if (dim == 0) return (size_t)-1;

    const float *base = gv_database_get_vector(db, 0);
    if (!base) return (size_t)-1;

    ptrdiff_t diff = sr->vector->data - base;
    if (diff < 0) return (size_t)-1;

    size_t idx = (size_t)diff / dim;
    if (idx >= gv_database_count(db)) return (size_t)-1;

    return idx;
}

/* ============================================================================
 * Pipeline Lifecycle
 * ============================================================================ */

GV_Pipeline *gv_pipeline_create(const void *db) {
    if (!db) return NULL;

    GV_Pipeline *pipe = calloc(1, sizeof(GV_Pipeline));
    if (!pipe) return NULL;

    pipe->db = (const GV_Database *)db;
    pipe->phase_count = 0;
    pipe->total_latency_ms = 0.0;

    if (pthread_mutex_init(&pipe->mutex, NULL) != 0) {
        free(pipe);
        return NULL;
    }

    return pipe;
}

void gv_pipeline_destroy(GV_Pipeline *pipe) {
    if (!pipe) return;
    pthread_mutex_destroy(&pipe->mutex);
    free(pipe);
}

/* ============================================================================
 * Phase Management
 * ============================================================================ */

int gv_pipeline_add_phase(GV_Pipeline *pipe, const GV_PhaseConfig *config) {
    if (!pipe || !config) return -1;

    pthread_mutex_lock(&pipe->mutex);

    if (pipe->phase_count >= GV_PIPELINE_MAX_PHASES) {
        pthread_mutex_unlock(&pipe->mutex);
        return -1;
    }

    /* First phase must be ANN. */
    if (pipe->phase_count == 0 && config->type != GV_PHASE_ANN) {
        pthread_mutex_unlock(&pipe->mutex);
        return -1;
    }

    int phase_id = (int)pipe->phase_count;
    pipe->phases[pipe->phase_count] = *config;
    pipe->phase_count++;

    pthread_mutex_unlock(&pipe->mutex);
    return phase_id;
}

void gv_pipeline_clear_phases(GV_Pipeline *pipe) {
    if (!pipe) return;

    pthread_mutex_lock(&pipe->mutex);
    pipe->phase_count = 0;
    memset(pipe->stats, 0, sizeof(pipe->stats));
    pipe->total_latency_ms = 0.0;
    pthread_mutex_unlock(&pipe->mutex);
}

size_t gv_pipeline_phase_count(const GV_Pipeline *pipe) {
    if (!pipe) return 0;
    return pipe->phase_count;
}

/* ============================================================================
 * Phase Executors (Internal)
 * ============================================================================ */

/**
 * @brief Execute the ANN phase: retrieve candidates from the database.
 *
 * @return Number of candidates produced, or -1 on error.
 */
static int execute_ann_phase(const GV_Database *db,
                             const GV_PhaseConfig *config,
                             const float *query, size_t dimension,
                             Candidate **out_candidates) {
    size_t fetch_k = config->output_k;
    if (fetch_k == 0) fetch_k = 100;

    GV_SearchResult *search_res = calloc(fetch_k, sizeof(GV_SearchResult));
    if (!search_res) return -1;

    GV_DistanceType dist = (GV_DistanceType)config->params.ann.distance_type;

    int found = gv_db_search(db, query, fetch_k, search_res, dist);
    if (found <= 0) {
        free(search_res);
        return (found == 0) ? 0 : -1;
    }

    Candidate *candidates = malloc((size_t)found * sizeof(Candidate));
    if (!candidates) {
        free(search_res);
        return -1;
    }

    size_t valid = 0;
    for (int i = 0; i < found; i++) {
        size_t soa_idx = result_to_soa_index(db, &search_res[i]);
        if (soa_idx == (size_t)-1) continue;

        candidates[valid].index    = soa_idx;
        candidates[valid].score    = search_res[i].distance;
        candidates[valid].phase_id = 0;
        valid++;
    }

    free(search_res);

    if (valid == 0) {
        free(candidates);
        *out_candidates = NULL;
        return 0;
    }

    *out_candidates = candidates;
    return (int)valid;
}

/**
 * @brief Execute expression-based reranking phase.
 *
 * Parses the expression, evaluates it on each candidate using the current
 * score as _score, sorts descending, and keeps top output_k.
 */
static int execute_rerank_expr_phase(const GV_PhaseConfig *config,
                                     int phase_id,
                                     Candidate *candidates, size_t count,
                                     size_t *out_count) {
    if (!config->params.expr.expression) return -1;

    GV_RankExpr *expr = gv_rank_expr_parse(config->params.expr.expression);
    if (!expr) return -1;

    /* Evaluate the expression for each candidate. */
    for (size_t i = 0; i < count; i++) {
        float vector_score = candidates[i].score;
        double new_score = gv_rank_expr_eval(expr, vector_score, NULL, 0);
        candidates[i].score = (float)new_score;
        candidates[i].phase_id = phase_id;
    }

    gv_rank_expr_destroy(expr);

    /* Sort descending by new score. */
    qsort(candidates, count, sizeof(Candidate), compare_candidates_desc);

    /* Truncate to output_k. */
    size_t keep = config->output_k;
    if (keep == 0 || keep > count) keep = count;
    *out_count = keep;

    return 0;
}

/**
 * @brief Execute MMR diversity reranking phase.
 *
 * Uses gv_mmr_rerank() to select diverse candidates from the current set.
 */
static int execute_rerank_mmr_phase(const GV_Database *db,
                                    const GV_PhaseConfig *config,
                                    const float *query, size_t dimension,
                                    int phase_id,
                                    Candidate *candidates, size_t count,
                                    Candidate **out_candidates,
                                    size_t *out_count) {
    size_t keep = config->output_k;
    if (keep == 0 || keep > count) keep = count;

    /* Build contiguous candidate vectors and index/distance arrays. */
    float  *cand_vectors   = malloc(count * dimension * sizeof(float));
    size_t *cand_indices   = malloc(count * sizeof(size_t));
    float  *cand_distances = malloc(count * sizeof(float));

    if (!cand_vectors || !cand_indices || !cand_distances) {
        free(cand_vectors);
        free(cand_indices);
        free(cand_distances);
        return -1;
    }

    size_t valid = 0;
    for (size_t i = 0; i < count; i++) {
        const float *vec = gv_database_get_vector(db, candidates[i].index);
        if (!vec) continue;

        memcpy(cand_vectors + valid * dimension, vec, dimension * sizeof(float));
        cand_indices[valid]   = candidates[i].index;
        cand_distances[valid] = candidates[i].score;
        valid++;
    }

    if (valid == 0) {
        free(cand_vectors);
        free(cand_indices);
        free(cand_distances);
        *out_count = 0;
        return 0;
    }

    if (keep > valid) keep = valid;

    GV_MMRConfig mmr_cfg;
    gv_mmr_config_init(&mmr_cfg);
    mmr_cfg.lambda = config->params.mmr.lambda;

    GV_MMRResult *mmr_results = malloc(keep * sizeof(GV_MMRResult));
    if (!mmr_results) {
        free(cand_vectors);
        free(cand_indices);
        free(cand_distances);
        return -1;
    }

    int mmr_count = gv_mmr_rerank(query, dimension,
                                   cand_vectors, cand_indices,
                                   cand_distances, valid,
                                   keep, &mmr_cfg, mmr_results);

    free(cand_vectors);
    free(cand_indices);
    free(cand_distances);

    if (mmr_count < 0) {
        free(mmr_results);
        return -1;
    }

    /* Rebuild candidate list from MMR results. */
    Candidate *new_candidates = malloc((size_t)mmr_count * sizeof(Candidate));
    if (!new_candidates) {
        free(mmr_results);
        return -1;
    }

    for (int i = 0; i < mmr_count; i++) {
        new_candidates[i].index    = mmr_results[i].index;
        new_candidates[i].score    = mmr_results[i].score;
        new_candidates[i].phase_id = phase_id;
    }

    free(mmr_results);

    *out_candidates = new_candidates;
    *out_count = (size_t)mmr_count;
    return 0;
}

/**
 * @brief Execute user-callback reranking phase.
 *
 * Calls the user function on each candidate to produce a new score, then
 * sorts descending and keeps top output_k.
 */
static int execute_rerank_callback_phase(const GV_PhaseConfig *config,
                                         int phase_id,
                                         Candidate *candidates, size_t count,
                                         size_t *out_count) {
    if (!config->params.callback.fn) return -1;

    GV_RerankCallback fn = config->params.callback.fn;
    const void *user_data = config->params.callback.data;

    for (size_t i = 0; i < count; i++) {
        candidates[i].score = fn(candidates[i].index,
                                 candidates[i].score,
                                 user_data);
        candidates[i].phase_id = phase_id;
    }

    /* Sort descending by new score. */
    qsort(candidates, count, sizeof(Candidate), compare_candidates_desc);

    /* Truncate to output_k. */
    size_t keep = config->output_k;
    if (keep == 0 || keep > count) keep = count;
    *out_count = keep;

    return 0;
}

/**
 * @brief Execute metadata filter phase.
 *
 * Evaluates a filter expression against each candidate's metadata and
 * removes non-matching candidates.
 */
static int execute_filter_phase(const GV_Database *db,
                                const GV_PhaseConfig *config,
                                int phase_id,
                                Candidate *candidates, size_t count,
                                size_t *out_count) {
    if (!config->params.filter.filter_expr) return -1;

    GV_Filter *filter = gv_filter_parse(config->params.filter.filter_expr);
    if (!filter) return -1;

    size_t write_idx = 0;
    for (size_t i = 0; i < count; i++) {
        /* Build a lightweight GV_Vector view for the filter evaluator. */
        GV_Vector view;
        memset(&view, 0, sizeof(view));

        const GV_SoAStorage *storage = db->soa_storage;
        if (storage) {
            gv_soa_storage_get_vector_view(storage, candidates[i].index, &view);
        }

        int match = gv_filter_eval(filter, &view);
        if (match == 1) {
            candidates[write_idx] = candidates[i];
            candidates[write_idx].phase_id = phase_id;
            write_idx++;
        }
    }

    gv_filter_destroy(filter);
    *out_count = write_idx;
    return 0;
}

/* ============================================================================
 * Pipeline Execution
 * ============================================================================ */

int gv_pipeline_execute(GV_Pipeline *pipe, const float *query,
                        size_t dimension, size_t final_k,
                        GV_PhasedResult *results) {
    if (!pipe || !query || !results || final_k == 0) return -1;

    pthread_mutex_lock(&pipe->mutex);

    if (pipe->phase_count == 0) {
        pthread_mutex_unlock(&pipe->mutex);
        return -1;
    }

    /* Validate first phase is ANN. */
    if (pipe->phases[0].type != GV_PHASE_ANN) {
        pthread_mutex_unlock(&pipe->mutex);
        return -1;
    }

    /* Validate dimension matches database. */
    if (dimension != gv_database_dimension(pipe->db)) {
        pthread_mutex_unlock(&pipe->mutex);
        return -1;
    }

    /* Reset stats. */
    memset(pipe->stats, 0, sizeof(pipe->stats));
    pipe->total_latency_ms = 0.0;

    Candidate *candidates = NULL;
    size_t     cand_count = 0;
    int        rc = 0;

    for (size_t p = 0; p < pipe->phase_count; p++) {
        const GV_PhaseConfig *cfg = &pipe->phases[p];
        struct timespec t_start, t_end;

        clock_gettime(CLOCK_MONOTONIC, &t_start);

        pipe->stats[p].input_count = cand_count;

        switch (cfg->type) {
        case GV_PHASE_ANN: {
            /* ANN must be first phase; produces the initial candidate set. */
            Candidate *ann_cands = NULL;
            int ann_count = execute_ann_phase(pipe->db, cfg, query, dimension,
                                              &ann_cands);
            if (ann_count < 0) {
                rc = -1;
            } else {
                candidates = ann_cands;
                cand_count = (size_t)ann_count;
                pipe->stats[p].input_count = 0; /* ANN has no input candidates. */
            }
            break;
        }

        case GV_PHASE_RERANK_EXPR: {
            if (!candidates || cand_count == 0) break;
            size_t new_count = cand_count;
            if (execute_rerank_expr_phase(cfg, (int)p, candidates, cand_count,
                                          &new_count) < 0) {
                rc = -1;
            } else {
                cand_count = new_count;
            }
            break;
        }

        case GV_PHASE_RERANK_MMR: {
            if (!candidates || cand_count == 0) break;
            Candidate *new_cands = NULL;
            size_t new_count = 0;
            if (execute_rerank_mmr_phase(pipe->db, cfg, query, dimension,
                                         (int)p, candidates, cand_count,
                                         &new_cands, &new_count) < 0) {
                rc = -1;
            } else {
                free(candidates);
                candidates = new_cands;
                cand_count = new_count;
            }
            break;
        }

        case GV_PHASE_RERANK_CALLBACK: {
            if (!candidates || cand_count == 0) break;
            size_t new_count = cand_count;
            if (execute_rerank_callback_phase(cfg, (int)p, candidates,
                                              cand_count, &new_count) < 0) {
                rc = -1;
            } else {
                cand_count = new_count;
            }
            break;
        }

        case GV_PHASE_FILTER: {
            if (!candidates || cand_count == 0) break;
            size_t new_count = cand_count;
            if (execute_filter_phase(pipe->db, cfg, (int)p, candidates,
                                     cand_count, &new_count) < 0) {
                rc = -1;
            } else {
                cand_count = new_count;
            }
            break;
        }

        default:
            rc = -1;
            break;
        }

        clock_gettime(CLOCK_MONOTONIC, &t_end);

        double phase_ms = elapsed_ms(&t_start, &t_end);
        pipe->stats[p].output_count = cand_count;
        pipe->stats[p].latency_ms   = phase_ms;
        pipe->total_latency_ms      += phase_ms;

        if (rc < 0) break;
    }

    /* Copy final candidates to output, truncated to final_k. */
    int result_count = 0;
    if (rc == 0 && candidates && cand_count > 0) {
        size_t copy_count = cand_count < final_k ? cand_count : final_k;
        for (size_t i = 0; i < copy_count; i++) {
            results[i].index         = candidates[i].index;
            results[i].score         = candidates[i].score;
            results[i].phase_reached = candidates[i].phase_id;
        }
        result_count = (int)copy_count;
    }

    free(candidates);
    pthread_mutex_unlock(&pipe->mutex);

    return (rc < 0) ? -1 : result_count;
}

/* ============================================================================
 * Statistics
 * ============================================================================ */

int gv_pipeline_get_stats(const GV_Pipeline *pipe, GV_PipelineStats *stats) {
    if (!pipe || !stats) return -1;

    memset(stats, 0, sizeof(*stats));

    size_t n = pipe->phase_count;
    if (n == 0) {
        stats->phase_count = 0;
        stats->total_latency_ms = 0.0;
        return 0;
    }

    stats->phase_input_counts  = calloc(n, sizeof(size_t));
    stats->phase_output_counts = calloc(n, sizeof(size_t));
    stats->phase_latencies_ms  = calloc(n, sizeof(double));

    if (!stats->phase_input_counts || !stats->phase_output_counts ||
        !stats->phase_latencies_ms) {
        free(stats->phase_input_counts);
        free(stats->phase_output_counts);
        free(stats->phase_latencies_ms);
        memset(stats, 0, sizeof(*stats));
        return -1;
    }

    for (size_t i = 0; i < n; i++) {
        stats->phase_input_counts[i]  = pipe->stats[i].input_count;
        stats->phase_output_counts[i] = pipe->stats[i].output_count;
        stats->phase_latencies_ms[i]  = pipe->stats[i].latency_ms;
    }

    stats->phase_count      = n;
    stats->total_latency_ms = pipe->total_latency_ms;

    return 0;
}

void gv_pipeline_free_stats(GV_PipelineStats *stats) {
    if (!stats) return;

    free(stats->phase_input_counts);
    free(stats->phase_output_counts);
    free(stats->phase_latencies_ms);

    stats->phase_input_counts  = NULL;
    stats->phase_output_counts = NULL;
    stats->phase_latencies_ms  = NULL;
    stats->phase_count         = 0;
    stats->total_latency_ms    = 0.0;
}
