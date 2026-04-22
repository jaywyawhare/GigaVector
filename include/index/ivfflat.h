#ifndef GIGAVECTOR_GV_IVFFLAT_H
#define GIGAVECTOR_GV_IVFFLAT_H

#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

#include "core/types.h"
#include "search/distance.h"
#include "storage/soa_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

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
void *ivfflat_create(size_t dimension, const GV_IVFFlatConfig *config);

/**
 * @brief Train coarse centroids via K-means.
 *
 * @param index IVF-Flat index (from ivfflat_create).
 * @param data Contiguous training vectors (count * dimension floats).
 * @param count Number of training vectors.
 * @return 0 on success, -1 on error.
 */
int ivfflat_train(void *index, const float *data, size_t count);

/**
 * @brief Insert a vector into IVF-Flat (requires training first).
 *
 * @param index IVF-Flat index.
 * @param vector Ownership of the vector transfers to the index.
 * @return 0 on success, -1 on error.
 */
int ivfflat_insert(void *index, GV_Vector *vector);

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
int ivfflat_search(void *index, const GV_Vector *query, size_t k,
                      GV_SearchResult *results, GV_DistanceType distance_type,
                      const char *filter_key, const char *filter_value);

int ivfflat_range_search(void *index, const GV_Vector *query, float radius,
                            GV_SearchResult *results, size_t max_results,
                            GV_DistanceType distance_type,
                            const char *filter_key, const char *filter_value);

/**
 * @brief Query a boolean condition.
 *
 * @param index Index instance.
 * @return 1 if true, 0 if false, -1 on error.
 */
int ivfflat_is_trained(const void *index);

/**
 * @brief Destroy an instance and free associated resources.
 *
 * @param index Index instance.
 */
void ivfflat_destroy(void *index);

/**
 * @brief Return the number of stored items.
 *
 * @param index Index instance.
 * @return Count value.
 */
size_t ivfflat_count(const void *index);

/**
 * @brief Delete an item.
 *
 * @param index Index instance.
 * @param entry_index Index value.
 * @return 0 on success, -1 on error.
 */
int ivfflat_delete(void *index, size_t entry_index);

/**
 * @brief Update an item.
 *
 * @param index Index instance.
 * @param entry_index Index value.
 * @param new_data new_data.
 * @param dimension Vector dimensionality.
 * @return 0 on success, -1 on error.
 */
int ivfflat_update(void *index, size_t entry_index, const float *new_data, size_t dimension);

/**
 * @brief Save state to a file.
 *
 * @param index Index instance.
 * @param out Output buffer.
 * @param version version.
 * @return 0 on success, -1 on error.
 */
int ivfflat_save(const void *index, FILE *out, uint32_t version);

/**
 * @brief Load state from a file.
 *
 * @param index_ptr index_ptr.
 * @param in Input file stream.
 * @param dimension Vector dimensionality.
 * @param version version.
 * @return 0 on success, -1 on error.
 */
int ivfflat_load(void **index_ptr, FILE *in, size_t dimension, uint32_t version);

#ifdef __cplusplus
}
#endif

#endif
