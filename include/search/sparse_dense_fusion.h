#ifndef GIGAVECTOR_GV_SPARSE_DENSE_FUSION_H
#define GIGAVECTOR_GV_SPARSE_DENSE_FUSION_H

#include <stddef.h>
#include <stdint.h>
#include "search/distance.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file sparse_dense_fusion.h
 * @brief Learned sparse + dense vector fusion for hybrid retrieval.
 *
 * Combines results from a learned sparse index (SPLADE-style) with a dense
 * vector index (HNSW, flat, etc.) using configurable fusion strategies.
 */

typedef enum {
    GV_SPARSE_DENSE_FUSION_RRF,          /**< Reciprocal Rank Fusion. */
    GV_SPARSE_DENSE_FUSION_WEIGHTED_RRF, /**< Weighted RRF with separate sparse/dense weights. */
    GV_SPARSE_DENSE_FUSION_LINEAR,       /**< Linear combination of normalised scores. */
    GV_SPARSE_DENSE_FUSION_CONVEX        /**< Convex combination: alpha*dense + (1-alpha)*sparse. */
} GV_SparseDenseFusionType;

typedef struct {
    GV_SparseDenseFusionType type;   /**< Fusion strategy. */
    float sparse_weight;             /**< Weight for sparse results (default 0.5). */
    float dense_weight;              /**< Weight for dense results (default 0.5). */
    int   rrf_k;                     /**< RRF constant k (default 60). */
} GV_SparseDenseFusionConfig;

/**
 * @brief Initialise a fusion config with sensible defaults.
 *
 * Defaults: RRF, sparse_weight 0.5, dense_weight 0.5, rrf_k 60.
 *
 * @param config Configuration to initialise.
 */
void sparse_dense_fusion_config_init(GV_SparseDenseFusionConfig *config);

/**
 * @brief Fuse sparse and dense search results into a unified ranking.
 *
 * Results from each source must be sorted by score descending.
 * Duplicate vector_ids are deduplicated; the entry with the higher
 * individual source score is kept (and receives the fused score).
 *
 * @param sparse_results  Sparse search results (sorted by score desc).
 * @param sparse_count    Number of sparse results.
 * @param dense_results   Dense search results (sorted by score desc).
 * @param dense_count     Number of dense results.
 * @param fused           Output fused results (pre-allocated, capacity >= max_fused).
 * @param max_fused       Capacity of fused array.
 * @param config          Fusion configuration (NULL for defaults).
 * @return Number of fused results written, or -1 on error.
 */
int sparse_dense_fuse(const GV_SearchResult *sparse_results, size_t sparse_count,
                      const GV_SearchResult *dense_results, size_t dense_count,
                      GV_SearchResult *fused, size_t max_fused,
                      const GV_SparseDenseFusionConfig *config);

/**
 * @brief Convenience: run both sparse and dense search, then fuse.
 *
 * This is the "one-call" hybrid search for learned sparse + dense.
 * Requires external index handles typed as void* to avoid coupling to
 * specific index implementations.
 *
 * @param sparse_index    Learned sparse index handle.
 * @param dense_index     Dense vector index handle (HNSW, etc.).
 * @param sparse_terms    Term strings for the sparse query.
 * @param sparse_weights  Learned weights for each term.
 * @param sparse_term_count Number of terms.
 * @param dense_query     Dense query vector.
 * @param dimension       Dense query vector dimension.
 * @param k               Number of final results desired.
 * @param fused           Output fused results (pre-allocated with >= k elements).
 * @param config          Fusion configuration (NULL for defaults).
 * @return Number of results written, or -1 on error.
 */
int sparse_dense_search(void *sparse_index, void *dense_index,
                        const char **sparse_terms, const float *sparse_weights,
                        size_t sparse_term_count,
                        const float *dense_query, size_t dimension,
                        size_t k, GV_SearchResult *fused,
                        const GV_SparseDenseFusionConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_SPARSE_DENSE_FUSION_H */
