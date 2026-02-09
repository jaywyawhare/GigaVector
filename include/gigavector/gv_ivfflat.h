#ifndef GIGAVECTOR_GV_IVFFLAT_H
#define GIGAVECTOR_GV_IVFFLAT_H

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
 * @brief IVF-Flat index configuration.
 */
typedef struct {
    size_t nlist;       /**< Number of coarse centroids (inverted lists). */
    size_t nprobe;      /**< Number of lists to probe at search time. */
    size_t train_iters; /**< K-means iterations for coarse centroids. */
    int use_cosine;     /**< Normalize query for cosine distance. */
} GV_IVFFlatConfig;

/**
 * @brief Create a new IVF-Flat index.
 *
 * @param dimension Vector dimensionality.
 * @param config Configuration parameters; NULL for defaults.
 * @return Allocated IVF-Flat index, or NULL on error.
 */
void *gv_ivfflat_create(size_t dimension, const GV_IVFFlatConfig *config);

/**
 * @brief Train coarse centroids via K-means.
 *
 * @param index IVF-Flat index (from gv_ivfflat_create).
 * @param data Contiguous training vectors (count * dimension floats).
 * @param count Number of training vectors.
 * @return 0 on success, -1 on error.
 */
int gv_ivfflat_train(void *index, const float *data, size_t count);

/**
 * @brief Insert a vector into IVF-Flat (requires training first).
 *
 * @param index IVF-Flat index.
 * @param vector Ownership of the vector transfers to the index.
 * @return 0 on success, -1 on error.
 */
int gv_ivfflat_insert(void *index, GV_Vector *vector);

/**
 * @brief Search IVF-Flat for k nearest neighbors.
 *
 * @param index IVF-Flat index.
 * @param query Query vector.
 * @param k Number of neighbors.
 * @param results Output array of at least k elements.
 * @param distance_type Distance metric.
 * @param filter_key Optional metadata filter key; NULL to disable.
 * @param filter_value Optional metadata filter value; NULL if key is NULL.
 * @return Number of neighbors found (0..k) or -1 on error.
 */
int gv_ivfflat_search(void *index, const GV_Vector *query, size_t k,
                      GV_SearchResult *results, GV_DistanceType distance_type,
                      const char *filter_key, const char *filter_value);

/**
 * @brief Range search in IVF-Flat.
 */
int gv_ivfflat_range_search(void *index, const GV_Vector *query, float radius,
                            GV_SearchResult *results, size_t max_results,
                            GV_DistanceType distance_type,
                            const char *filter_key, const char *filter_value);

/**
 * @brief Check whether the IVF-Flat index has been trained.
 */
int gv_ivfflat_is_trained(const void *index);

/**
 * @brief Destroy IVF-Flat index.
 */
void gv_ivfflat_destroy(void *index);

/**
 * @brief Count vectors in IVF-Flat.
 */
size_t gv_ivfflat_count(const void *index);

/**
 * @brief Delete a vector by its entry index.
 */
int gv_ivfflat_delete(void *index, size_t entry_index);

/**
 * @brief Update a vector by its entry index.
 */
int gv_ivfflat_update(void *index, size_t entry_index, const float *new_data, size_t dimension);

/**
 * @brief Save IVF-Flat index to file.
 */
int gv_ivfflat_save(const void *index, FILE *out, uint32_t version);

/**
 * @brief Load IVF-Flat index from file.
 */
int gv_ivfflat_load(void **index_ptr, FILE *in, size_t dimension, uint32_t version);

#ifdef __cplusplus
}
#endif

#endif
