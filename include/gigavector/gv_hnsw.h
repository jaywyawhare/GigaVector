#ifndef GIGAVECTOR_GV_HNSW_H
#define GIGAVECTOR_GV_HNSW_H

#include <stddef.h>

#include "gv_distance.h"
#include "gv_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief HNSW index configuration parameters.
 */
typedef struct {
    size_t M;              /**< Number of connections per node (default: 16) */
    size_t efConstruction; /**< Candidate list size during construction (default: 200) */
    size_t efSearch;       /**< Candidate list size during search (default: 50) */
    size_t maxLevel;       /**< Maximum level in hierarchy (auto-calculated if 0) */
} GV_HNSWConfig;

/**
 * @brief Create a new HNSW index.
 *
 * @param dimension Vector dimensionality.
 * @param config Configuration parameters; NULL for defaults.
 * @return Allocated HNSW index, or NULL on error.
 */
void *gv_hnsw_create(size_t dimension, const GV_HNSWConfig *config);

/**
 * @brief Insert a vector into the HNSW index.
 *
 * @param index HNSW index instance; must be non-NULL.
 * @param vector Vector to insert; ownership transferred to index.
 * @return 0 on success, -1 on error.
 */
int gv_hnsw_insert(void *index, GV_Vector *vector);

/**
 * @brief Search for k nearest neighbors in HNSW.
 *
 * @param index HNSW index instance; must be non-NULL.
 * @param query Query vector.
 * @param k Number of neighbors to find.
 * @param results Output array of at least k elements.
 * @param distance_type Distance metric to use.
 * @param filter_key Optional metadata filter key; NULL to disable.
 * @param filter_value Optional metadata filter value; NULL if key is NULL.
 * @return Number of neighbors found (0 to k), or -1 on error.
 */
int gv_hnsw_search(void *index, const GV_Vector *query, size_t k,
                   GV_SearchResult *results, GV_DistanceType distance_type,
                   const char *filter_key, const char *filter_value);

/**
 * @brief Destroy HNSW index and free all resources.
 *
 * @param index HNSW index instance; safe to call with NULL.
 */
void gv_hnsw_destroy(void *index);

/**
 * @brief Get the number of vectors in the HNSW index.
 *
 * @param index HNSW index instance; must be non-NULL.
 * @return Number of vectors, or 0 if index is NULL.
 */
size_t gv_hnsw_count(const void *index);

/**
 * @brief Save HNSW index to file.
 *
 * @param index HNSW index instance; must be non-NULL.
 * @param out File stream opened for writing.
 * @param version File format version.
 * @return 0 on success, -1 on error.
 */
int gv_hnsw_save(const void *index, FILE *out, uint32_t version);

/**
 * @brief Load HNSW index from file.
 *
 * @param index_ptr Pointer to HNSW index pointer (will be allocated).
 * @param in File stream opened for reading.
 * @param dimension Vector dimensionality.
 * @param version File format version.
 * @return 0 on success, -1 on error.
 */
int gv_hnsw_load(void **index_ptr, FILE *in, size_t dimension, uint32_t version);

#ifdef __cplusplus
}
#endif

#endif

