/**
 * @file gv_phased_ranking.h
 * @brief Multi-stage phased ranking pipeline for progressive candidate refinement.
 *
 * Provides a configurable pipeline of ranking stages: fast ANN retrieval
 * first, then progressively more expensive scoring phases such as
 * expression-based reranking, MMR diversity reranking, user-supplied
 * callbacks, and metadata filtering.
 *
 * Typical usage:
 * @code
 *   GV_Pipeline *pipe = gv_pipeline_create(db);
 *
 *   GV_PhaseConfig ann = { .type = GV_PHASE_ANN, .output_k = 100 };
 *   ann.params.ann.distance_type = GV_DISTANCE_COSINE;
 *   ann.params.ann.ef_search = 200;
 *   gv_pipeline_add_phase(pipe, &ann);
 *
 *   GV_PhaseConfig rerank = { .type = GV_PHASE_RERANK_EXPR, .output_k = 20 };
 *   rerank.params.expr.expression = "0.7 * _score + 0.3 * popularity";
 *   gv_pipeline_add_phase(pipe, &rerank);
 *
 *   GV_PhasedResult results[10];
 *   int n = gv_pipeline_execute(pipe, query, dim, 10, results);
 *
 *   GV_PipelineStats stats;
 *   gv_pipeline_get_stats(pipe, &stats);
 *   gv_pipeline_free_stats(&stats);
 *
 *   gv_pipeline_destroy(pipe);
 * @endcode
 */

#ifndef GIGAVECTOR_GV_PHASED_RANKING_H
#define GIGAVECTOR_GV_PHASED_RANKING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Enumerations */

/**
 * @brief Phase types for the ranking pipeline.
 */
typedef enum {
    GV_PHASE_ANN              = 0,  /**< Vector ANN search (must be the first phase). */
    GV_PHASE_RERANK_EXPR      = 1,  /**< Re-rank with a ranking expression (gv_ranking.h). */
    GV_PHASE_RERANK_MMR       = 2,  /**< Re-rank for diversity via MMR (gv_mmr.h). */
    GV_PHASE_RERANK_CALLBACK  = 3,  /**< Re-rank with a user-supplied scoring callback. */
    GV_PHASE_FILTER           = 4   /**< Filter candidates by metadata expression. */
} GV_PhaseType;

/* Callback Type */

/**
 * @brief User-supplied reranking callback.
 *
 * Called once per candidate during a GV_PHASE_RERANK_CALLBACK phase.
 *
 * @param index         Vector index of the candidate in the database.
 * @param current_score Current score from the previous phase.
 * @param user_data     Opaque pointer supplied in GV_PhaseConfig.params.callback.data.
 * @return New score for this candidate.
 */
typedef float (*GV_RerankCallback)(size_t index, float current_score,
                                   const void *user_data);

/* Data Structures */

/**
 * @brief Configuration for a single pipeline phase.
 */
typedef struct {
    GV_PhaseType type;      /**< Which phase type to execute. */
    size_t output_k;        /**< Maximum number of results this phase emits. */

    /** Phase-specific parameters. */
    union {
        /** Parameters for GV_PHASE_ANN. */
        struct {
            int    distance_type;   /**< GV_DistanceType value. */
            size_t ef_search;       /**< HNSW ef_search override (0 = use default). */
        } ann;

        /** Parameters for GV_PHASE_RERANK_EXPR. */
        struct {
            const char *expression; /**< Ranking expression string (see gv_ranking.h). */
        } expr;

        /** Parameters for GV_PHASE_RERANK_MMR. */
        struct {
            float lambda;           /**< MMR trade-off: 0.0 = diversity, 1.0 = relevance. */
        } mmr;

        /** Parameters for GV_PHASE_RERANK_CALLBACK. */
        struct {
            GV_RerankCallback fn;   /**< Scoring function. */
            void *data;             /**< Opaque user data passed to fn. */
        } callback;

        /** Parameters for GV_PHASE_FILTER. */
        struct {
            const char *filter_expr; /**< Metadata filter expression (see gv_filter.h). */
        } filter;
    } params;
} GV_PhaseConfig;

/**
 * @brief A single result from the phased ranking pipeline.
 */
typedef struct {
    size_t index;           /**< Vector index in the database. */
    float  score;           /**< Final score after all phases that touched this result. */
    int    phase_reached;   /**< Index of the last phase that processed this result. */
} GV_PhasedResult;

/**
 * @brief Opaque pipeline handle.
 */
typedef struct GV_Pipeline GV_Pipeline;

/**
 * @brief Per-execution statistics for the pipeline.
 */
typedef struct {
    size_t *phase_input_counts;     /**< Number of candidates entering each phase. */
    size_t *phase_output_counts;    /**< Number of candidates leaving each phase. */
    double *phase_latencies_ms;     /**< Wall-clock time in milliseconds per phase. */
    size_t  phase_count;            /**< Number of phases in this snapshot. */
    double  total_latency_ms;       /**< Sum of all phase latencies. */
} GV_PipelineStats;

/* Pipeline Lifecycle */

/**
 * @brief Create a new phased ranking pipeline.
 *
 * @param db  Database handle (cast to GV_Database* internally); must be non-NULL.
 * @return Allocated pipeline, or NULL on error.
 */
GV_Pipeline *gv_pipeline_create(const void *db);

/**
 * @brief Destroy a pipeline and free all associated resources.
 *
 * Safe to call with NULL.
 *
 * @param pipe  Pipeline to destroy.
 */
void gv_pipeline_destroy(GV_Pipeline *pipe);

/* Phase Management */

/**
 * @brief Append a phase to the pipeline.
 *
 * The first phase added must be GV_PHASE_ANN. Up to 8 phases may be added.
 *
 * @param pipe    Pipeline handle; must be non-NULL.
 * @param config  Phase configuration; must be non-NULL. The structure is copied.
 * @return Phase index (>= 0) on success, or -1 on error.
 */
int gv_pipeline_add_phase(GV_Pipeline *pipe, const GV_PhaseConfig *config);

/**
 * @brief Remove all phases from the pipeline.
 *
 * @param pipe  Pipeline handle; must be non-NULL.
 */
void gv_pipeline_clear_phases(GV_Pipeline *pipe);

/**
 * @brief Return the number of phases currently in the pipeline.
 *
 * @param pipe  Pipeline handle; must be non-NULL.
 * @return Phase count.
 */
size_t gv_pipeline_phase_count(const GV_Pipeline *pipe);

/* Execution */

/**
 * @brief Execute the full ranking pipeline against a query vector.
 *
 * The ANN phase is executed first; each subsequent phase refines the
 * candidates produced by its predecessor. The final output is truncated
 * to @p final_k results.
 *
 * @param pipe      Pipeline handle; must be non-NULL and have at least one ANN phase.
 * @param query     Query vector data.
 * @param dimension Query vector dimension; must match the database dimension.
 * @param final_k   Maximum number of results to return.
 * @param results   Output array of at least @p final_k elements.
 * @return Number of results written (0 to final_k), or -1 on error.
 */
int gv_pipeline_execute(GV_Pipeline *pipe, const float *query,
                        size_t dimension, size_t final_k,
                        GV_PhasedResult *results);

/* Statistics */

/**
 * @brief Retrieve statistics from the most recent pipeline execution.
 *
 * The arrays inside @p stats are heap-allocated and must be released
 * with gv_pipeline_free_stats().
 *
 * @param pipe   Pipeline handle; must be non-NULL.
 * @param stats  Output structure; must be non-NULL.
 * @return 0 on success, -1 on error.
 */
int gv_pipeline_get_stats(const GV_Pipeline *pipe, GV_PipelineStats *stats);

/**
 * @brief Free heap memory owned by a GV_PipelineStats structure.
 *
 * Safe to call with NULL fields.
 *
 * @param stats  Stats structure whose arrays will be freed.
 */
void gv_pipeline_free_stats(GV_PipelineStats *stats);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_PHASED_RANKING_H */
