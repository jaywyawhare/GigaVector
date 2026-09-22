#ifndef GIGAVECTOR_GV_COMPACT_STORE_H
#define GIGAVECTOR_GV_COMPACT_STORE_H

#include <stddef.h>

#include "search/distance.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file compact_store.h
 * @brief Low-precision (fp16 / int8) vector store with brute-force search.
 *
 * Stores vectors at half (fp16) or a quarter (int8) of float32 RAM, decoding on
 * the fly for distance computation. For fp16 the loss is negligible for
 * normalized embeddings; int8 uses a per-vector symmetric scale. Use this when
 * memory is the constraint and exact/brute-force search over a compact set is
 * acceptable (e.g. a cold tier, or a memory-budgeted collection).
 */

typedef enum {
    GV_COMPACT_F16 = 0,   /**< IEEE binary16, 2 bytes/element. */
    GV_COMPACT_I8  = 1    /**< Symmetric int8 + per-vector scale, ~1 byte/element. */
} GV_CompactPrecision;

typedef struct GV_CompactStore GV_CompactStore;

/** @brief Create a compact store for @p dimension-dimensional vectors. */
GV_CompactStore *compact_store_create(size_t dimension, GV_CompactPrecision precision);

/** @brief Free a compact store (safe with NULL). */
void compact_store_free(GV_CompactStore *s);

/**
 * @brief Encode and append a vector.
 * @return Its 0-based index, or (size_t)-1 on error.
 */
size_t compact_store_add(GV_CompactStore *s, const float *data);

/** @brief Number of stored vectors. */
size_t compact_store_count(const GV_CompactStore *s);

/** @brief Bytes used by the encoded vectors (excluding bookkeeping). */
size_t compact_store_memory_bytes(const GV_CompactStore *s);

/**
 * @brief Decode vector @p index into @p out (dimension floats).
 * @return 0 on success, -1 on invalid arguments.
 */
int compact_store_get(const GV_CompactStore *s, size_t index, float *out);

/**
 * @brief Brute-force top-k search over the compact store.
 *
 * @param query  Query vector (dimension floats).
 * @param k      Neighbours to return (clamped to count).
 * @param metric Distance metric.
 * @param ids    Output array (length >= k) of vector indices, ascending distance.
 * @param dists  Output array (length >= k) of distances.
 * @return Number of results written (0..k), or -1 on invalid arguments.
 */
int compact_store_search(const GV_CompactStore *s, const float *query, size_t k,
                         GV_DistanceType metric, size_t *ids, float *dists);

#ifdef __cplusplus
}
#endif

#endif
