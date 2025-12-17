/**
 * @file gv_sparse_vector.h
 * @brief Sparse vector utilities (COO format).
 */

#ifndef GIGAVECTOR_GV_SPARSE_VECTOR_H
#define GIGAVECTOR_GV_SPARSE_VECTOR_H

#include <stddef.h>
#include <stdint.h>

#include "gv_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Create a sparse vector by copying indices/values.
 *
 * @param dimension Total dimensionality.
 * @param indices Array of length nnz containing dimension indices.
 * @param values Array of length nnz containing corresponding values.
 * @param nnz Number of non-zero entries.
 * @return Allocated sparse vector or NULL on error.
 */
GV_SparseVector *gv_sparse_vector_create(size_t dimension,
                                         const uint32_t *indices,
                                         const float *values,
                                         size_t nnz);

/**
 * @brief Destroy a sparse vector and free its memory.
 *
 * @param sv Sparse vector to destroy; safe to call with NULL.
 */
void gv_sparse_vector_destroy(GV_SparseVector *sv);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_SPARSE_VECTOR_H */



