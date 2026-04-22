#ifndef GIGAVECTOR_GV_LSH_H
#define GIGAVECTOR_GV_LSH_H

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
    size_t num_tables;    /**< Number of hash tables (default: 8). */
    size_t num_hash_bits; /**< Number of hash functions per table (K in E2LSH). */
    uint64_t seed;        /**< Random seed for hash function generation. */
    float bucket_width;   /**< w parameter for E2LSH (default: 4.0). */
} GV_LSHConfig;

void *lsh_create(size_t dimension, const GV_LSHConfig *config, GV_SoAStorage *soa_storage);

/**
 * @brief Perform the operation.
 *
 * @param index Index instance.
 * @param vector vector.
 * @return 0 on success, -1 on error.
 */
int lsh_insert(void *index, GV_Vector *vector);

int lsh_search(void *index, const GV_Vector *query, size_t k,
                  GV_SearchResult *results, GV_DistanceType distance_type,
                  const char *filter_key, const char *filter_value);

int lsh_range_search(void *index, const GV_Vector *query, float radius,
                        GV_SearchResult *results, size_t max_results,
                        GV_DistanceType distance_type,
                        const char *filter_key, const char *filter_value);

/**
 * @brief Destroy an instance and free associated resources.
 *
 * @param index Index instance.
 */
void lsh_destroy(void *index);

/**
 * @brief Return the number of stored items.
 *
 * @param index Index instance.
 * @return Count value.
 */
size_t lsh_count(const void *index);

/**
 * @brief Delete an item.
 *
 * @param index Index instance.
 * @param vector_index Index value.
 * @return 0 on success, -1 on error.
 */
int lsh_delete(void *index, size_t vector_index);

/**
 * @brief Update an item.
 *
 * @param index Index instance.
 * @param vector_index Index value.
 * @param new_data new_data.
 * @param dimension Vector dimensionality.
 * @return 0 on success, -1 on error.
 */
int lsh_update(void *index, size_t vector_index, const float *new_data, size_t dimension);

/**
 * @brief Save state to a file.
 *
 * @param index Index instance.
 * @param out Output buffer.
 * @param version version.
 * @return 0 on success, -1 on error.
 */
int lsh_save(const void *index, FILE *out, uint32_t version);

/**
 * @brief Load state from a file.
 *
 * @param index_ptr index_ptr.
 * @param in Input file stream.
 * @param dimension Vector dimensionality.
 * @param version version.
 * @return 0 on success, -1 on error.
 */
int lsh_load(void **index_ptr, FILE *in, size_t dimension, uint32_t version);

#ifdef __cplusplus
}
#endif

#endif
