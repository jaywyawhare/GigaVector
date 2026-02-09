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

/**
 * @brief LSH (Locality-Sensitive Hashing) index configuration.
 */
typedef struct {
    size_t num_tables;    /**< Number of hash tables (default: 8). */
    size_t num_hash_bits; /**< Number of hash bits per table (default: 16). */
    uint64_t seed;        /**< Random seed for hash function generation. */
} GV_LSHConfig;

/**
 * @brief Create a new LSH index.
 */
void *gv_lsh_create(size_t dimension, const GV_LSHConfig *config, GV_SoAStorage *soa_storage);

/**
 * @brief Insert a vector into the LSH index.
 */
int gv_lsh_insert(void *index, GV_Vector *vector);

/**
 * @brief Search LSH index for k nearest neighbors.
 */
int gv_lsh_search(void *index, const GV_Vector *query, size_t k,
                  GV_SearchResult *results, GV_DistanceType distance_type,
                  const char *filter_key, const char *filter_value);

/**
 * @brief Range search in LSH index.
 */
int gv_lsh_range_search(void *index, const GV_Vector *query, float radius,
                        GV_SearchResult *results, size_t max_results,
                        GV_DistanceType distance_type,
                        const char *filter_key, const char *filter_value);

/**
 * @brief Destroy LSH index.
 */
void gv_lsh_destroy(void *index);

/**
 * @brief Count vectors in LSH index.
 */
size_t gv_lsh_count(const void *index);

/**
 * @brief Delete a vector by its index.
 */
int gv_lsh_delete(void *index, size_t vector_index);

/**
 * @brief Update a vector by its index.
 */
int gv_lsh_update(void *index, size_t vector_index, const float *new_data, size_t dimension);

/**
 * @brief Save LSH index to file.
 */
int gv_lsh_save(const void *index, FILE *out, uint32_t version);

/**
 * @brief Load LSH index from file.
 */
int gv_lsh_load(void **index_ptr, FILE *in, size_t dimension, uint32_t version);

#ifdef __cplusplus
}
#endif

#endif
