#ifndef GIGAVECTOR_GV_HYBRID_SEARCH_H
#define GIGAVECTOR_GV_HYBRID_SEARCH_H

#include <stddef.h>
#include <stdint.h>

#include "gv_types.h"
#include "gv_distance.h"
#include "gv_bm25.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_hybrid_search.h
 * @brief Hybrid search combining vector similarity and BM25 text relevance.
 *
 * Provides fusion algorithms to combine dense vector search with sparse
 * text search for improved retrieval quality.
 */

/* Forward declarations */
struct GV_Database;
typedef struct GV_Database GV_Database;

/**
 * @brief Score fusion method.
 */
typedef enum {
    GV_FUSION_LINEAR = 0,           /**< Weighted linear combination. */
    GV_FUSION_RRF = 1,              /**< Reciprocal Rank Fusion. */
    GV_FUSION_WEIGHTED_RRF = 2      /**< RRF with custom weights. */
} GV_FusionType;

/**
 * @brief Hybrid search configuration.
 */
typedef struct {
    GV_FusionType fusion_type;      /**< Score fusion method (default: LINEAR). */
    double vector_weight;           /**< Weight for vector scores (default: 0.5). */
    double text_weight;             /**< Weight for text scores (default: 0.5). */
    double rrf_k;                   /**< RRF constant k (default: 60). */
    GV_DistanceType distance_type;  /**< Vector distance metric (default: COSINE). */
    size_t prefetch_k;              /**< Results to fetch from each source (default: k*3). */
} GV_HybridConfig;

/**
 * @brief Hybrid search result.
 */
typedef struct {
    size_t vector_index;            /**< Vector/Document index. */
    double combined_score;          /**< Combined fusion score. */
    double vector_score;            /**< Original vector similarity score. */
    double text_score;              /**< Original BM25 text score. */
    size_t vector_rank;             /**< Rank from vector search (0 if not found). */
    size_t text_rank;               /**< Rank from text search (0 if not found). */
} GV_HybridResult;

/**
 * @brief Hybrid search statistics.
 */
typedef struct {
    size_t vector_candidates;       /**< Candidates from vector search. */
    size_t text_candidates;         /**< Candidates from text search. */
    size_t unique_candidates;       /**< Unique candidates after merge. */
    double vector_search_time_ms;   /**< Vector search time. */
    double text_search_time_ms;     /**< Text search time. */
    double fusion_time_ms;          /**< Fusion time. */
    double total_time_ms;           /**< Total search time. */
} GV_HybridStats;

/**
 * @brief Opaque hybrid searcher handle.
 */
typedef struct GV_HybridSearcher GV_HybridSearcher;

/* ============================================================================
 * Configuration
 * ============================================================================ */

/**
 * @brief Initialize hybrid configuration with defaults.
 *
 * Default values:
 * - fusion_type: GV_FUSION_LINEAR
 * - vector_weight: 0.5
 * - text_weight: 0.5
 * - rrf_k: 60
 * - distance_type: GV_DIST_COSINE
 * - prefetch_k: 0 (auto: k * 3)
 *
 * @param config Configuration to initialize.
 */
void gv_hybrid_config_init(GV_HybridConfig *config);

/* ============================================================================
 * Hybrid Searcher Lifecycle
 * ============================================================================ */

/**
 * @brief Create a hybrid searcher.
 *
 * @param db Vector database.
 * @param bm25 BM25 text index.
 * @param config Hybrid configuration (NULL for defaults).
 * @return Hybrid searcher instance, or NULL on error.
 */
GV_HybridSearcher *gv_hybrid_create(GV_Database *db, GV_BM25Index *bm25,
                                     const GV_HybridConfig *config);

/**
 * @brief Destroy a hybrid searcher.
 *
 * Does not destroy the underlying database or BM25 index.
 *
 * @param searcher Hybrid searcher instance (safe to call with NULL).
 */
void gv_hybrid_destroy(GV_HybridSearcher *searcher);

/* ============================================================================
 * Search Operations
 * ============================================================================ */

/**
 * @brief Perform hybrid search with vector and text query.
 *
 * @param searcher Hybrid searcher.
 * @param query_vector Query vector (can be NULL for text-only).
 * @param query_text Query text (can be NULL for vector-only).
 * @param k Number of results to return.
 * @param results Output results array (must be pre-allocated with k elements).
 * @return Number of results found, or -1 on error.
 */
int gv_hybrid_search(GV_HybridSearcher *searcher, const float *query_vector,
                     const char *query_text, size_t k, GV_HybridResult *results);

/**
 * @brief Perform hybrid search with statistics.
 *
 * @param searcher Hybrid searcher.
 * @param query_vector Query vector.
 * @param query_text Query text.
 * @param k Number of results.
 * @param results Output results array.
 * @param stats Output statistics.
 * @return Number of results found, or -1 on error.
 */
int gv_hybrid_search_with_stats(GV_HybridSearcher *searcher, const float *query_vector,
                                 const char *query_text, size_t k,
                                 GV_HybridResult *results, GV_HybridStats *stats);

/**
 * @brief Perform vector-only search through hybrid searcher.
 *
 * @param searcher Hybrid searcher.
 * @param query_vector Query vector.
 * @param k Number of results.
 * @param results Output results array.
 * @return Number of results found, or -1 on error.
 */
int gv_hybrid_search_vector_only(GV_HybridSearcher *searcher, const float *query_vector,
                                  size_t k, GV_HybridResult *results);

/**
 * @brief Perform text-only search through hybrid searcher.
 *
 * @param searcher Hybrid searcher.
 * @param query_text Query text.
 * @param k Number of results.
 * @param results Output results array.
 * @return Number of results found, or -1 on error.
 */
int gv_hybrid_search_text_only(GV_HybridSearcher *searcher, const char *query_text,
                                size_t k, GV_HybridResult *results);

/* ============================================================================
 * Configuration Updates
 * ============================================================================ */

/**
 * @brief Update hybrid configuration.
 *
 * @param searcher Hybrid searcher.
 * @param config New configuration.
 * @return 0 on success, -1 on error.
 */
int gv_hybrid_set_config(GV_HybridSearcher *searcher, const GV_HybridConfig *config);

/**
 * @brief Get current hybrid configuration.
 *
 * @param searcher Hybrid searcher.
 * @param config Output configuration.
 * @return 0 on success, -1 on error.
 */
int gv_hybrid_get_config(const GV_HybridSearcher *searcher, GV_HybridConfig *config);

/**
 * @brief Set fusion weights.
 *
 * Convenience function to update weights without full config.
 * Weights will be normalized to sum to 1.0.
 *
 * @param searcher Hybrid searcher.
 * @param vector_weight Weight for vector scores.
 * @param text_weight Weight for text scores.
 * @return 0 on success, -1 on error.
 */
int gv_hybrid_set_weights(GV_HybridSearcher *searcher, double vector_weight,
                           double text_weight);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * @brief Compute linear fusion score.
 *
 * @param vector_score Normalized vector score (0-1).
 * @param text_score Normalized text score (0-1).
 * @param vector_weight Vector weight.
 * @param text_weight Text weight.
 * @return Combined score.
 */
double gv_hybrid_linear_fusion(double vector_score, double text_score,
                                double vector_weight, double text_weight);

/**
 * @brief Compute RRF (Reciprocal Rank Fusion) score.
 *
 * @param vector_rank Rank from vector search (1-based, 0 if not found).
 * @param text_rank Rank from text search (1-based, 0 if not found).
 * @param k RRF constant (typically 60).
 * @return Combined RRF score.
 */
double gv_hybrid_rrf_fusion(size_t vector_rank, size_t text_rank, double k);

/**
 * @brief Normalize a score to 0-1 range.
 *
 * @param score Raw score.
 * @param min_score Minimum score in result set.
 * @param max_score Maximum score in result set.
 * @return Normalized score (0-1).
 */
double gv_hybrid_normalize_score(double score, double min_score, double max_score);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_HYBRID_SEARCH_H */
