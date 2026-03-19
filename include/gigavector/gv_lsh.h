#ifndef GIGAVECTOR_GV_LSH_H
#define GIGAVECTOR_GV_LSH_H

#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

#include "gv_types.h"
#include "gv_distance.h"
#include "gv_soa_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t num_tables;    /**< Number of hash tables (default: 8). */
    size_t num_hash_bits; /**< Number of hash functions per table (K in E2LSH). */
    uint64_t seed;        /**< Random seed for hash function generation. */
    float bucket_width;   /**< w parameter for E2LSH (default: 4.0). */
} GV_LSHConfig;

void *gv_lsh_create(size_t dimension, const GV_LSHConfig *config, GV_SoAStorage *soa_storage);

int gv_lsh_insert(void *index, GV_Vector *vector);

int gv_lsh_search(void *index, const GV_Vector *query, size_t k,
                  GV_SearchResult *results, GV_DistanceType distance_type,
                  const char *filter_key, const char *filter_value);

int gv_lsh_range_search(void *index, const GV_Vector *query, float radius,
                        GV_SearchResult *results, size_t max_results,
                        GV_DistanceType distance_type,
                        const char *filter_key, const char *filter_value);

void gv_lsh_destroy(void *index);

size_t gv_lsh_count(const void *index);

int gv_lsh_delete(void *index, size_t vector_index);

int gv_lsh_update(void *index, size_t vector_index, const float *new_data, size_t dimension);

int gv_lsh_save(const void *index, FILE *out, uint32_t version);

int gv_lsh_load(void **index_ptr, FILE *in, size_t dimension, uint32_t version);

#ifdef __cplusplus
}
#endif

#endif
