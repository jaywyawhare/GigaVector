/**
 * @file gv_exact_search.h
 * @brief Brute-force exact k-NN search helpers.
 */

#ifndef GIGAVECTOR_GV_EXACT_SEARCH_H
#define GIGAVECTOR_GV_EXACT_SEARCH_H

#include <stddef.h>

#include "gv_distance.h"
#include "gv_kdtree.h"
#include "gv_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Perform brute-force exact k-NN search over an array of vectors.
 *
 * Distances are computed with @ref gv_distance, which uses SIMD where
 * available. Results are returned sorted by increasing distance.
 *
 * @param vectors Array of vector pointers; must be non-NULL when count > 0.
 * @param count Number of vectors in @p vectors.
 * @param query Query vector; must be non-NULL and dimension-compatible.
 * @param k Number of nearest neighbors to return (up to @p count).
 * @param results Output array of at least @p k elements; must be non-NULL.
 * @param distance_type Distance metric to use.
 * @return Number of neighbors found (0 to min(k, count)), or -1 on error.
 */
int gv_exact_knn_search_vectors(GV_Vector *const *vectors, size_t count,
                                const GV_Vector *query, size_t k,
                                GV_SearchResult *results, GV_DistanceType distance_type);

/**
 * @brief Perform brute-force exact k-NN search over all nodes in a K-D tree.
 *
 * The caller provides the total number of vectors stored in the tree so that
 * an internal array can be allocated once and reused for distance evaluation.
 * This is intended for small collections where a linear scan is competitive.
 *
 * @param root Root of the K-D tree; may be NULL.
 * @param total_count Expected number of vectors stored in the tree.
 * @param query Query vector; must be non-NULL and dimension-compatible.
 * @param k Number of nearest neighbors to return (up to @p total_count).
 * @param results Output array of at least @p k elements; must be non-NULL.
 * @param distance_type Distance metric to use.
 * @return Number of neighbors found (0 to min(k, total_count)), or -1 on error.
 */
int gv_exact_knn_search_kdtree(const GV_KDNode *root, size_t total_count,
                               const GV_Vector *query, size_t k,
                               GV_SearchResult *results, GV_DistanceType distance_type);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_EXACT_SEARCH_H */



