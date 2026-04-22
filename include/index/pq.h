#ifndef GIGAVECTOR_GV_PQ_H
#define GIGAVECTOR_GV_PQ_H

#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

#include "core/types.h"
#include "search/distance.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Standalone Product Quantization index configuration.
 */
typedef struct {
    size_t m;           /**< Number of sub-quantizers (must divide dimension). */
    uint8_t nbits;      /**< Bits per sub-quantizer code (typically 8). */
    size_t train_iters; /**< K-means iterations for codebook training. */
} GV_PQConfig;

void *pq_create(size_t dimension, const GV_PQConfig *config);
/**
 * @brief Perform the operation.
 *
 * @param index Index instance.
 * @param data Input data buffer.
 * @param count Number of items.
 * @return 0 on success, -1 on error.
 */
int pq_train(void *index, const float *data, size_t count);

/**
 * @brief Insert a vector into PQ index (requires training first).
 */
int pq_insert(void *index, GV_Vector *vector);

/**
 * @brief Search PQ index for k nearest neighbors using ADC.
 */
int pq_search(void *index, const GV_Vector *query, size_t k,
                 GV_SearchResult *results, GV_DistanceType distance_type,
                 const char *filter_key, const char *filter_value);

int pq_range_search(void *index, const GV_Vector *query, float radius,
                       GV_SearchResult *results, size_t max_results,
                       GV_DistanceType distance_type,
                       const char *filter_key, const char *filter_value);
/**
 * @brief Query a boolean condition.
 *
 * @param index Index instance.
 * @return 1 if true, 0 if false, -1 on error.
 */
int pq_is_trained(const void *index);
/**
 * @brief Destroy an instance and free associated resources.
 *
 * @param index Index instance.
 */
void pq_destroy(void *index);
/**
 * @brief Return the number of stored items.
 *
 * @param index Index instance.
 * @return Count value.
 */
size_t pq_count(const void *index);
/**
 * @brief Delete an item.
 *
 * @param index Index instance.
 * @param entry_index Index value.
 * @return 0 on success, -1 on error.
 */
int pq_delete(void *index, size_t entry_index);
/**
 * @brief Update an item.
 *
 * @param index Index instance.
 * @param entry_index Index value.
 * @param new_data new_data.
 * @param dimension Vector dimensionality.
 * @return 0 on success, -1 on error.
 */
int pq_update(void *index, size_t entry_index, const float *new_data, size_t dimension);
/**
 * @brief Save state to a file.
 *
 * @param index Index instance.
 * @param out Output buffer.
 * @param version version.
 * @return 0 on success, -1 on error.
 */
int pq_save(const void *index, FILE *out, uint32_t version);
/**
 * @brief Load state from a file.
 *
 * @param index_ptr index_ptr.
 * @param in Input file stream.
 * @param dimension Vector dimensionality.
 * @param version version.
 * @return 0 on success, -1 on error.
 */
int pq_load(void **index_ptr, FILE *in, size_t dimension, uint32_t version);

#ifdef __cplusplus
}
#endif

#endif
