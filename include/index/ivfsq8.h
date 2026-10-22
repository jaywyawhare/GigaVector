#ifndef GIGAVECTOR_GV_IVFSQ8_H
#define GIGAVECTOR_GV_IVFSQ8_H

#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

#include "core/types.h"
#include "search/distance.h"
#include "storage/scalar_quant.h"
#include "storage/soa_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t nlist;          /**< Number of coarse centroids (inverted lists). */
    size_t nprobe;         /**< Number of lists to probe at search time. */
    size_t train_iters;    /**< K-means iterations for coarse centroids. */
    int use_cosine;        /**< Normalize query for cosine distance. */
    int per_dimension;     /**< Use per-dimension min/max for scalar quant (0 = global). */
    size_t default_rerank; /**< Rerank top-N candidates with exact float distance (0 = disabled). */
} GV_IVFSQ8Config;

void *ivfsq8_create(size_t dimension, const GV_IVFSQ8Config *config);

int ivfsq8_train(void *index, const float *data, size_t count);

int ivfsq8_insert(void *index, GV_Vector *vector);

int ivfsq8_search(void *index, const GV_Vector *query, size_t k,
                  GV_SearchResult *results, GV_DistanceType distance_type,
                  const char *filter_key, const char *filter_value);

int ivfsq8_range_search(void *index, const GV_Vector *query, float radius,
                        GV_SearchResult *results, size_t max_results,
                        GV_DistanceType distance_type,
                        const char *filter_key, const char *filter_value);

int ivfsq8_is_trained(const void *index);

void ivfsq8_destroy(void *index);

size_t ivfsq8_count(const void *index);

/** @brief Set the number of lists probed at search time (no clamping). */
void ivfsq8_set_nprobe(void *index, size_t nprobe);

/** @brief Get the number of lists probed at search time; 0 if index is NULL. */
size_t ivfsq8_get_nprobe(const void *index);

int ivfsq8_delete(void *index, size_t entry_index);

int ivfsq8_update(void *index, size_t entry_index, const float *new_data, size_t dimension);

int ivfsq8_save(const void *index, FILE *out, uint32_t version);

int ivfsq8_load(void **index_ptr, FILE *in, size_t dimension, uint32_t version);

#ifdef __cplusplus
}
#endif

#endif
