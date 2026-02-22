/**
 * @file gv_mmr.h
 * @brief Maximal Marginal Relevance (MMR) reranking for diversity-aware search.
 *
 * MMR iteratively selects results that balance relevance to the query against
 * diversity (dissimilarity to already-selected items).  The trade-off is
 * controlled by the lambda parameter:
 *   score = lambda * relevance(d, q) - (1 - lambda) * max_similarity(d, S)
 * where S is the set of already-selected documents.
 */

#ifndef GIGAVECTOR_GV_MMR_H
#define GIGAVECTOR_GV_MMR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GV_Database GV_Database;

/* Configuration */

/**
 * @brief Configuration for MMR reranking.
 */
typedef struct {
    float lambda;       /**< Trade-off: 0.0 = full diversity, 1.0 = full relevance (default: 0.7). */
    int distance_type;  /**< Distance metric (GV_DistanceType value, default: GV_DISTANCE_COSINE). */
} GV_MMRConfig;

/* Result */

/**
 * @brief A single MMR-reranked result.
 */
typedef struct {
    size_t index;       /**< Original candidate index (from candidate_indices). */
    float score;        /**< Combined MMR score. */
    float relevance;    /**< Relevance component (similarity to query). */
    float diversity;    /**< Diversity component (dissimilarity to selected set). */
} GV_MMRResult;

/* API */

/**
 * @brief Initialize an MMR configuration with default values.
 *
 * Sets lambda to 0.7 and distance_type to GV_DISTANCE_COSINE (1).
 *
 * @param config Configuration to initialize; must be non-NULL.
 */
void gv_mmr_config_init(GV_MMRConfig *config);

/**
 * @brief Rerank a pre-fetched set of candidate vectors using MMR.
 *
 * Given an initial set of search results (candidate vectors with their
 * distances to the query), this function iteratively selects up to @p k
 * results that maximise the MMR objective.
 *
 * Distances are normalised to [0, 1] internally so that the lambda
 * weighting is consistent regardless of the distance metric scale.
 *
 * @param query               Query vector data (dimension floats).
 * @param dimension           Number of components in each vector.
 * @param candidates          Contiguous candidate vector data
 *                            (candidate_count * dimension floats).
 * @param candidate_indices   Original database indices for each candidate.
 * @param candidate_distances Pre-computed distances from the query for each candidate.
 * @param candidate_count     Number of candidates.
 * @param k                   Number of results to select (<= candidate_count).
 * @param config              MMR configuration; NULL to use defaults.
 * @param results             Output array of at least @p k elements.
 * @return Number of results written (0 to k), or -1 on error.
 */
int gv_mmr_rerank(const float *query, size_t dimension,
                  const float *candidates, const size_t *candidate_indices,
                  const float *candidate_distances, size_t candidate_count,
                  size_t k, const GV_MMRConfig *config,
                  GV_MMRResult *results);

/**
 * @brief Convenience: search a database and apply MMR reranking in one call.
 *
 * Performs a standard k-NN search with oversampling (k * oversample candidates),
 * then applies MMR reranking to select the final @p k diverse results.
 *
 * @param db        Database to search; must be non-NULL.
 * @param query     Query vector data (dimension floats).
 * @param dimension Number of components in the query; must match db dimension.
 * @param k         Number of diverse results to return.
 * @param oversample Oversampling factor (e.g. 4 fetches 4*k candidates).
 * @param config    MMR configuration; NULL to use defaults.
 * @param results   Output array of at least @p k elements.
 * @return Number of results written (0 to k), or -1 on error.
 */
int gv_mmr_search(const void *db, const float *query, size_t dimension,
                  size_t k, size_t oversample, const GV_MMRConfig *config,
                  GV_MMRResult *results);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_MMR_H */
