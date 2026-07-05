#ifndef GIGAVECTOR_GV_FRESHNESS_H
#define GIGAVECTOR_GV_FRESHNESS_H

#include <stddef.h>
#include <stdint.h>
#include "search/ranking.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file freshness.h
 * @brief Vector Freshness Scoring — blend vector similarity with time-decay signals.
 *
 * Enables ranking search results by a combination of vector similarity and
 * how recently a vector was inserted (or last updated). Newer vectors receive
 * a higher freshness score, which can be blended with the vector score.
 */

/**
 * @brief Parameters controlling freshness scoring.
 */
typedef struct {
    double score_weight;      /**< Weight for the vector similarity score (e.g. 0.7). */
    double freshness_weight;  /**< Weight for the freshness score (e.g. 0.3). */
    double half_life_sec;     /**< Half-life of decay in seconds (default 86400 = 1 day). */
    GV_RankOp decay_type;     /**< Decay function: GV_RANK_DECAY_EXP, _GAUSS, or _LINEAR. */
} GV_FreshnessParams;

/**
 * @brief Per-result struct exposing decomposed scores for debuggability.
 */
typedef struct {
    size_t   index;               /**< Vector index in the database. */
    float    final_score;         /**< Combined score: score_weight*vector_score + freshness_weight*freshness_score. */
    float    vector_score;        /**< Raw vector similarity/distance score. */
    float    freshness_score;     /**< Time-decay freshness score in [0, 1]. */
    uint64_t insert_timestamp_ms; /**< Insertion timestamp in milliseconds since epoch. */
} GV_FreshnessResult;

/**
 * @brief Compute the freshness score for a vector.
 *
 * Uses the selected decay function to compute a score in [0, 1] that
 * decreases with age. A vector inserted right now scores 1.0; after
 * one half-life it scores 0.5.
 *
 * @param insert_time_ms Insertion timestamp in milliseconds.
 * @param now_ms         Current time in milliseconds.
 * @param half_life_ms   Half-life in milliseconds.
 * @param decay_type     GV_RANK_DECAY_EXP, GV_RANK_DECAY_GAUSS, or GV_RANK_DECAY_LINEAR.
 * @return Freshness score in [0, 1].
 */
float freshness_score(uint64_t insert_time_ms, uint64_t now_ms,
                      double half_life_ms, GV_RankOp decay_type);

/**
 * @brief Linear blend of vector score and freshness.
 *
 * @param vector_score    Raw vector similarity score.
 * @param freshness       Freshness score in [0, 1].
 * @param score_weight    Weight for vector_score.
 * @param freshness_weight Weight for freshness.
 * @return Blended score.
 */
float freshness_blend(float vector_score, float freshness,
                      float score_weight, float freshness_weight);

/**
 * @brief Search with freshness re-ranking.
 *
 * Oversamples candidates via db_search(), computes freshness for each, blends
 * scores, and returns the top-k results sorted by final_score descending.
 *
 * @param db     Database handle; must be non-NULL.
 * @param query  Query vector of dimension db->dimension.
 * @param k      Number of results to return.
 * @param params Freshness parameters; must be non-NULL.
 * @param out    Output array of at least k GV_FreshnessResult elements.
 * @return Number of results written (0..k), or -1 on error.
 */
int db_search_with_freshness(const void *db, const float *query, size_t k,
                              const GV_FreshnessParams *params,
                              GV_FreshnessResult *out);

/**
 * @brief Filtered search with freshness re-ranking.
 *
 * Same as db_search_with_freshness() but only considers vectors matching
 * the given metadata key/value filter.
 *
 * @param db           Database handle; must be non-NULL.
 * @param query        Query vector.
 * @param k            Number of results.
 * @param params       Freshness parameters; must be non-NULL.
 * @param filter_key   Metadata key to filter by; NULL to disable.
 * @param filter_value Metadata value to match.
 * @param out          Output array of at least k elements.
 * @return Number of results written (0..k), or -1 on error.
 */
int db_search_filtered_with_freshness(const void *db, const float *query, size_t k,
                                       const GV_FreshnessParams *params,
                                       const char *filter_key, const char *filter_value,
                                       GV_FreshnessResult *out);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_FRESHNESS_H */
