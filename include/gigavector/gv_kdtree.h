#ifndef GIGAVECTOR_GV_KDTREE_H
#define GIGAVECTOR_GV_KDTREE_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "gv_distance.h"
#include "gv_types.h"
#include "gv_soa_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Insert a vector into the K-D tree using SoA storage.
 *
 * @param root Pointer to the root node pointer; will be updated on first insert.
 * @param storage SoA storage containing vectors; must be non-NULL.
 * @param vector_index Index of the vector in SoA storage to insert.
 * @param depth Current recursion depth; pass 0 for the initial call.
 * @return 0 on success, -1 on invalid arguments or allocation failure.
 */
int gv_kdtree_insert(GV_KDNode **root, GV_SoAStorage *storage, size_t vector_index, size_t depth);

/**
 * @brief Recursively serialize a K-D tree (including metadata) to an open FILE stream.
 *
 * The format uses a pre-order traversal with presence flags for null nodes.
 * A byte flag is written for each node (1 = present, 0 = null). For present
 * nodes, the axis (uint32_t) and all vector components (float) are written
 * before recursing into children. For format version >= 2, metadata is also
 * written: uint32_t metadata_count, followed by key/value length-prefixed
 * pairs (uint32_t key_len, key bytes, uint32_t value_len, value bytes).
 *
 * @param node Root of the subtree to serialize; may be NULL.
 * @param storage SoA storage containing vectors; must be non-NULL.
 * @param out Open FILE stream in binary mode.
 * @param version Format version; metadata is included when version >= 2.
 * @return 0 on success, -1 on invalid arguments or write failure.
 */
int gv_kdtree_save_recursive(const GV_KDNode *node, const GV_SoAStorage *storage, FILE *out, uint32_t version);

/**
 * @brief Recursively load a K-D tree from an open FILE stream.
 *
 * Expects the same pre-order format emitted by gv_kdtree_save_recursive().
 * Vectors are added to the SoA storage and nodes reference them by index.
 *
 * @param root Pointer to receive the root pointer; must be non-NULL.
 * @param storage SoA storage to add vectors to; must be non-NULL.
 * @param in Open FILE stream positioned at the start of a node record.
 * @param dimension Expected vector dimensionality.
 * @param version Format version read from file; metadata is loaded when version >= 2.
 * @return 0 on success, -1 on invalid arguments or read/allocation failure.
 */
int gv_kdtree_load_recursive(GV_KDNode **root, GV_SoAStorage *storage, FILE *in, size_t dimension, uint32_t version);

/**
 * @brief Recursively destroy a K-D tree and all its nodes.
 *
 * Frees all nodes. Vectors are stored in SoA storage and not freed here.
 * Safe to call with NULL.
 *
 * @param node Root of the subtree to destroy; may be NULL.
 */
void gv_kdtree_destroy_recursive(GV_KDNode *node);

/**
 * @brief Find k nearest neighbors in the K-D tree.
 *
 * @param root Root of the K-D tree to search; may be NULL.
 * @param storage SoA storage containing vectors; must be non-NULL.
 * @param query Query vector to find neighbors for.
 * @param k Number of nearest neighbors to find.
 * @param results Output array of at least @p k elements.
 * @param distance_type Distance metric to use.
 * @return Number of neighbors found (0 to k), or -1 on error.
 */
int gv_kdtree_knn_search(const GV_KDNode *root, const GV_SoAStorage *storage, const GV_Vector *query, size_t k,
                          GV_SearchResult *results, GV_DistanceType distance_type);

/**
 * @brief Find k nearest neighbors with metadata filtering.
 *
 * @param root Root of the K-D tree to search; may be NULL.
 * @param storage SoA storage containing vectors; must be non-NULL.
 * @param query Query vector to find neighbors for.
 * @param k Number of nearest neighbors to find.
 * @param results Output array of at least @p k elements.
 * @param distance_type Distance metric to use.
 * @param filter_key Metadata key to filter by; NULL to disable filtering.
 * @param filter_value Metadata value to match; NULL if filter_key is NULL.
 * @return Number of neighbors found (0 to k), or -1 on error.
 */
int gv_kdtree_knn_search_filtered(const GV_KDNode *root, const GV_SoAStorage *storage, const GV_Vector *query, size_t k,
                                   GV_SearchResult *results, GV_DistanceType distance_type,
                                   const char *filter_key, const char *filter_value);

/**
 * @brief Range search: find all vectors within a distance threshold.
 *
 * @param root Root of the K-D tree to search; may be NULL.
 * @param storage SoA storage containing vectors; must be non-NULL.
 * @param query Query vector to find neighbors for.
 * @param radius Maximum distance threshold (inclusive).
 * @param results Output array to store results; must be pre-allocated.
 * @param max_results Maximum number of results to return (capacity of results array).
 * @param distance_type Distance metric to use.
 * @return Number of vectors found within radius (0 to max_results), or -1 on error.
 */
int gv_kdtree_range_search(const GV_KDNode *root, const GV_SoAStorage *storage, const GV_Vector *query, float radius,
                            GV_SearchResult *results, size_t max_results, GV_DistanceType distance_type);

/**
 * @brief Range search with metadata filtering.
 *
 * @param root Root of the K-D tree to search; may be NULL.
 * @param storage SoA storage containing vectors; must be non-NULL.
 * @param query Query vector to find neighbors for.
 * @param radius Maximum distance threshold (inclusive).
 * @param results Output array to store results; must be pre-allocated.
 * @param max_results Maximum number of results to return (capacity of results array).
 * @param distance_type Distance metric to use.
 * @param filter_key Metadata key to filter by; NULL to disable filtering.
 * @param filter_value Metadata value to match; NULL if filter_key is NULL.
 * @return Number of vectors found within radius (0 to max_results), or -1 on error.
 */
int gv_kdtree_range_search_filtered(const GV_KDNode *root, const GV_SoAStorage *storage, const GV_Vector *query, float radius,
                                     GV_SearchResult *results, size_t max_results,
                                     GV_DistanceType distance_type,
                                     const char *filter_key, const char *filter_value);

/**
 * @brief Delete a vector from the K-D tree by its index in SoA storage.
 *
 * @param root Pointer to the root node pointer; will be updated if root is deleted.
 * @param storage SoA storage containing vectors; must be non-NULL.
 * @param vector_index Index of the vector to delete in SoA storage.
 * @return 0 on success, -1 on invalid arguments or vector not found.
 */
int gv_kdtree_delete(GV_KDNode **root, GV_SoAStorage *storage, size_t vector_index);

/**
 * @brief Update a vector in the K-D tree by its index in SoA storage.
 *
 * This function updates the vector data and rebuilds the tree structure
 * if necessary. The tree is updated by deleting the old node and inserting
 * a new one with the updated data.
 *
 * @param root Pointer to the root node pointer; will be updated if needed.
 * @param storage SoA storage containing vectors; must be non-NULL.
 * @param vector_index Index of the vector to update in SoA storage.
 * @param new_data New vector data to store.
 * @return 0 on success, -1 on invalid arguments or vector not found.
 */
int gv_kdtree_update(GV_KDNode **root, GV_SoAStorage *storage, size_t vector_index, const float *new_data);

#ifdef __cplusplus
}
#endif

#endif

