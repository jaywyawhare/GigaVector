#ifndef GIGAVECTOR_GV_DEDUP_H
#define GIGAVECTOR_GV_DEDUP_H

#include <stddef.h>
#include <stdint.h>

#include "gv_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Configuration for the deduplication index.
 */
typedef struct {
    float epsilon;           /**< Maximum L2 distance to consider vectors as duplicates. */
    size_t num_hash_tables;  /**< Number of LSH hash tables (default: 8). */
    size_t hash_bits;        /**< Number of hash bits per table (default: 12). */
    uint64_t seed;           /**< Random seed for hyperplane generation. */
} GV_DedupConfig;

/**
 * @brief Result describing a pair of duplicate vectors.
 */
typedef struct {
    size_t original_index;   /**< Index of the first (earlier) vector. */
    size_t duplicate_index;  /**< Index of the second (later) vector. */
    float distance;          /**< L2 distance between the pair. */
} GV_DedupResult;

/**
 * @brief Opaque deduplication index handle.
 */
typedef struct GV_DedupIndex GV_DedupIndex;

/**
 * @brief Create a new deduplication index.
 *
 * @param dimension Dimensionality of the vectors.
 * @param config    Configuration parameters (NULL for defaults).
 * @return Pointer to the new index, or NULL on failure.
 */
GV_DedupIndex *gv_dedup_create(size_t dimension, const GV_DedupConfig *config);

/**
 * @brief Destroy a deduplication index and free all resources.
 *
 * @param dedup The index to destroy.
 */
void gv_dedup_destroy(GV_DedupIndex *dedup);

/**
 * @brief Check whether a vector already exists as a near-duplicate.
 *
 * @param dedup     The deduplication index.
 * @param data      Pointer to the vector data (length = dimension).
 * @param dimension Dimensionality of the query vector.
 * @return Index of an existing duplicate, or -1 if the vector is unique.
 */
int gv_dedup_check(GV_DedupIndex *dedup, const float *data, size_t dimension);

/**
 * @brief Insert a vector only if it is not a near-duplicate of an existing one.
 *
 * @param dedup     The deduplication index.
 * @param data      Pointer to the vector data (length = dimension).
 * @param dimension Dimensionality of the vector.
 * @return 0 if inserted, 1 if a duplicate was found, -1 on error.
 */
int gv_dedup_insert(GV_DedupIndex *dedup, const float *data, size_t dimension);

/**
 * @brief Scan for all duplicate pairs in the index.
 *
 * @param dedup       The deduplication index.
 * @param results     Output array for duplicate pairs.
 * @param max_results Maximum number of pairs to return.
 * @return Number of duplicate pairs found, or -1 on error.
 */
int gv_dedup_scan(GV_DedupIndex *dedup, GV_DedupResult *results, size_t max_results);

/**
 * @brief Return the number of vectors stored in the index.
 *
 * @param dedup The deduplication index.
 * @return Number of stored vectors.
 */
size_t gv_dedup_count(const GV_DedupIndex *dedup);

/**
 * @brief Remove all vectors from the index.
 *
 * @param dedup The deduplication index.
 */
void gv_dedup_clear(GV_DedupIndex *dedup);

#ifdef __cplusplus
}
#endif

#endif
