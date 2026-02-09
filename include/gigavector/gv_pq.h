#ifndef GIGAVECTOR_GV_PQ_H
#define GIGAVECTOR_GV_PQ_H

#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

#include "gv_types.h"
#include "gv_distance.h"

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

/**
 * @brief Create a new standalone PQ index.
 */
void *gv_pq_create(size_t dimension, const GV_PQConfig *config);

/**
 * @brief Train PQ codebooks via K-means.
 */
int gv_pq_train(void *index, const float *data, size_t count);

/**
 * @brief Insert a vector into PQ index (requires training first).
 */
int gv_pq_insert(void *index, GV_Vector *vector);

/**
 * @brief Search PQ index for k nearest neighbors using ADC.
 */
int gv_pq_search(void *index, const GV_Vector *query, size_t k,
                 GV_SearchResult *results, GV_DistanceType distance_type,
                 const char *filter_key, const char *filter_value);

/**
 * @brief Range search in PQ index.
 */
int gv_pq_range_search(void *index, const GV_Vector *query, float radius,
                       GV_SearchResult *results, size_t max_results,
                       GV_DistanceType distance_type,
                       const char *filter_key, const char *filter_value);

/**
 * @brief Check whether the PQ index has been trained.
 */
int gv_pq_is_trained(const void *index);

/**
 * @brief Destroy PQ index.
 */
void gv_pq_destroy(void *index);

/**
 * @brief Count vectors in PQ index.
 */
size_t gv_pq_count(const void *index);

/**
 * @brief Delete a vector by its entry index.
 */
int gv_pq_delete(void *index, size_t entry_index);

/**
 * @brief Update a vector by its entry index.
 */
int gv_pq_update(void *index, size_t entry_index, const float *new_data, size_t dimension);

/**
 * @brief Save PQ index to file.
 */
int gv_pq_save(const void *index, FILE *out, uint32_t version);

/**
 * @brief Load PQ index from file.
 */
int gv_pq_load(void **index_ptr, FILE *in, size_t dimension, uint32_t version);

#ifdef __cplusplus
}
#endif

#endif
