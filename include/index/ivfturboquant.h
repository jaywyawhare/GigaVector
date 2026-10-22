#ifndef GIGAVECTOR_GV_IVFTURBOQUANT_H
#define GIGAVECTOR_GV_IVFTURBOQUANT_H

#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

#include "core/types.h"
#include "search/distance.h"
#include "storage/turboquant.h"
#include "storage/soa_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t nlist;          /**< Number of coarse centroids (inverted lists). */
    size_t nprobe;         /**< Number of lists to probe at search time. */
    size_t train_iters;    /**< K-means iterations for coarse centroids. */
    int use_cosine;        /**< Normalize query for cosine distance. */
    size_t default_rerank; /**< Rerank top-N candidates with exact float distance (0 = disabled). */
    GV_TurboQuantConfig turbo;
} GV_IVFTurboQuantConfig;

void *ivfturboquant_create(size_t dimension, const GV_IVFTurboQuantConfig *config);

int ivfturboquant_train(void *index, const float *data, size_t count);

int ivfturboquant_insert(void *index, GV_Vector *vector);

int ivfturboquant_search(void *index, const GV_Vector *query, size_t k,
                         GV_SearchResult *results, GV_DistanceType distance_type,
                         const char *filter_key, const char *filter_value);

int ivfturboquant_range_search(void *index, const GV_Vector *query, float radius,
                               GV_SearchResult *results, size_t max_results,
                               GV_DistanceType distance_type,
                               const char *filter_key, const char *filter_value);

int ivfturboquant_is_trained(const void *index);

void ivfturboquant_destroy(void *index);

size_t ivfturboquant_count(const void *index);

/** @brief Set the number of lists probed at search time (no clamping). */
void ivfturboquant_set_nprobe(void *index, size_t nprobe);

/** @brief Get the number of lists probed at search time; 0 if index is NULL. */
size_t ivfturboquant_get_nprobe(const void *index);

int ivfturboquant_delete(void *index, size_t entry_index);

int ivfturboquant_update(void *index, size_t entry_index, const float *new_data, size_t dimension);

int ivfturboquant_save(const void *index, FILE *out, uint32_t version);

int ivfturboquant_load(void **index_ptr, FILE *in, size_t dimension, uint32_t version);

#ifdef __cplusplus
}
#endif

#endif
