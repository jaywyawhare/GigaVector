/**
 * @file gv_sparse_index.h
 * @brief Inverted index for sparse vectors (dot product search).
 */

#ifndef GIGAVECTOR_GV_SPARSE_INDEX_H
#define GIGAVECTOR_GV_SPARSE_INDEX_H

#include <stddef.h>

#include "gv_types.h"
#include "gv_sparse_vector.h"
#include "gv_distance.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GV_SparseIndex GV_SparseIndex;

/**
 * @brief Create a sparse inverted index.
 *
 * @param dimension Vector dimensionality.
 * @return Allocated index or NULL on error.
 */
GV_SparseIndex *gv_sparse_index_create(size_t dimension);

/**
 * @brief Destroy a sparse index and all stored vectors.
 *
 * @param index Index to destroy; safe to call with NULL.
 */
void gv_sparse_index_destroy(GV_SparseIndex *index);

/**
 * @brief Add a sparse vector to the index.
 *
 * Ownership of @p vector is transferred to the index on success.
 *
 * @param index Sparse index; must be non-NULL.
 * @param vector Sparse vector to add; must be non-NULL.
 * @return 0 on success, -1 on error.
 */
int gv_sparse_index_add(GV_SparseIndex *index, GV_SparseVector *vector);

/**
 * @brief Search top-k by dot product (or cosine if normalized) for a sparse query.
 *
 * Currently supports GV_DISTANCE_DOT_PRODUCT or GV_DISTANCE_COSINE. For
 * cosine, the query should be pre-normalized; the index does not store norms.
 *
 * @param index Sparse index; must be non-NULL.
 * @param query Sparse query vector; must be non-NULL.
 * @param k Number of results to return.
 * @param results Output array of at least @p k elements.
 * @param distance_type GV_DISTANCE_DOT_PRODUCT or GV_DISTANCE_COSINE.
 * @return Number of results (0..k) or -1 on error.
 */
int gv_sparse_index_search(const GV_SparseIndex *index, const GV_SparseVector *query,
                           size_t k, GV_SearchResult *results, GV_DistanceType distance_type);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_SPARSE_INDEX_H */



