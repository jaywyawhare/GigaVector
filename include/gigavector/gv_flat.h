#ifndef GIGAVECTOR_GV_FLAT_H
#define GIGAVECTOR_GV_FLAT_H

#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

#include "gv_types.h"
#include "gv_distance.h"
#include "gv_soa_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Flat (brute-force) index configuration.
 */
typedef struct {
    int use_simd;  /**< Prefer SIMD distance computation (default: 1). */
} GV_FlatConfig;

/**
 * @brief Create a new Flat index.
 *
 * @param dimension Vector dimensionality.
 * @param config Configuration parameters; NULL for defaults.
 * @param soa_storage Optional SoA storage to use; if NULL, creates a new one.
 * @return Allocated Flat index, or NULL on error.
 */
void *gv_flat_create(size_t dimension, const GV_FlatConfig *config, GV_SoAStorage *soa_storage);

/**
 * @brief Insert a vector into the Flat index.
 *
 * @param index Flat index instance; must be non-NULL.
 * @param vector Vector to insert; ownership transferred to index.
 * @return 0 on success, -1 on error.
 */
int gv_flat_insert(void *index, GV_Vector *vector);

/**
 * @brief Search for k nearest neighbors in Flat index (brute-force).
 *
 * @param index Flat index instance; must be non-NULL.
 * @param query Query vector.
 * @param k Number of neighbors to find.
 * @param results Output array of at least k elements.
 * @param distance_type Distance metric to use.
 * @param filter_key Optional metadata filter key; NULL to disable.
 * @param filter_value Optional metadata filter value; NULL if key is NULL.
 * @return Number of neighbors found (0 to k), or -1 on error.
 */
int gv_flat_search(void *index, const GV_Vector *query, size_t k,
                   GV_SearchResult *results, GV_DistanceType distance_type,
                   const char *filter_key, const char *filter_value);

/**
 * @brief Range search: find all vectors within a distance threshold.
 *
 * @param index Flat index instance; must be non-NULL.
 * @param query Query vector.
 * @param radius Maximum distance threshold (inclusive).
 * @param results Output array to store results; must be pre-allocated.
 * @param max_results Maximum number of results to return.
 * @param distance_type Distance metric to use.
 * @param filter_key Optional metadata filter key; NULL to disable.
 * @param filter_value Optional metadata filter value; NULL if key is NULL.
 * @return Number of vectors found within radius (0 to max_results), or -1 on error.
 */
int gv_flat_range_search(void *index, const GV_Vector *query, float radius,
                         GV_SearchResult *results, size_t max_results,
                         GV_DistanceType distance_type,
                         const char *filter_key, const char *filter_value);

/**
 * @brief Destroy Flat index and free all resources.
 *
 * @param index Flat index instance; safe to call with NULL.
 */
void gv_flat_destroy(void *index);

/**
 * @brief Get the number of vectors in the Flat index.
 *
 * @param index Flat index instance; must be non-NULL.
 * @return Number of vectors, or 0 if index is NULL.
 */
size_t gv_flat_count(const void *index);

/**
 * @brief Delete a vector from the Flat index by its index.
 *
 * @param index Flat index instance; must be non-NULL.
 * @param vector_index Index of the vector to delete (0-based).
 * @return 0 on success, -1 on error.
 */
int gv_flat_delete(void *index, size_t vector_index);

/**
 * @brief Update a vector in the Flat index by its index.
 *
 * @param index Flat index instance; must be non-NULL.
 * @param vector_index Index of the vector to update (0-based).
 * @param new_data New vector data array.
 * @param dimension Vector dimension; must match index dimension.
 * @return 0 on success, -1 on error.
 */
int gv_flat_update(void *index, size_t vector_index, const float *new_data, size_t dimension);

/**
 * @brief Save Flat index to file.
 *
 * @param index Flat index instance; must be non-NULL.
 * @param out File stream opened for writing.
 * @param version File format version.
 * @return 0 on success, -1 on error.
 */
int gv_flat_save(const void *index, FILE *out, uint32_t version);

/**
 * @brief Load Flat index from file.
 *
 * @param index_ptr Pointer to Flat index pointer (will be allocated).
 * @param in File stream opened for reading.
 * @param dimension Vector dimensionality.
 * @param version File format version.
 * @return 0 on success, -1 on error.
 */
int gv_flat_load(void **index_ptr, FILE *in, size_t dimension, uint32_t version);

#ifdef __cplusplus
}
#endif

#endif
