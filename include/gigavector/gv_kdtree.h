#ifndef GIGAVECTOR_GV_KDTREE_H
#define GIGAVECTOR_GV_KDTREE_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "gv_distance.h"
#include "gv_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Insert a vector into the K-D tree.
 *
 * Ownership of @p point is transferred to the tree on success.
 *
 * @param root Pointer to the root node pointer; will be updated on first insert.
 * @param point Vector to insert; must be non-NULL with a non-zero dimension.
 * @param depth Current recursion depth; pass 0 for the initial call.
 * @return 0 on success, -1 on invalid arguments or allocation failure.
 */
int gv_kdtree_insert(GV_KDNode **root, GV_Vector *point, size_t depth);

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
 * @param out Open FILE stream in binary mode.
 * @param version Format version; metadata is included when version >= 2.
 * @return 0 on success, -1 on invalid arguments or write failure.
 */
int gv_kdtree_save_recursive(const GV_KDNode *node, FILE *out, uint32_t version);

/**
 * @brief Recursively load a K-D tree from an open FILE stream.
 *
 * Expects the same pre-order format emitted by gv_kdtree_save_recursive().
 * On success, allocates nodes and vectors; ownership is given to the caller.
 *
 * @param root Pointer to receive the root pointer; must be non-NULL.
 * @param in Open FILE stream positioned at the start of a node record.
 * @param dimension Expected vector dimensionality.
 * @param version Format version read from file; metadata is loaded when version >= 2.
 * @return 0 on success, -1 on invalid arguments or read/allocation failure.
 */
int gv_kdtree_load_recursive(GV_KDNode **root, FILE *in, size_t dimension, uint32_t version);

/**
 * @brief Recursively destroy a K-D tree and all its nodes.
 *
 * Frees all nodes and their associated vectors. Safe to call with NULL.
 *
 * @param node Root of the subtree to destroy; may be NULL.
 */
void gv_kdtree_destroy_recursive(GV_KDNode *node);

/**
 * @brief Find k nearest neighbors in the K-D tree.
 *
 * @param root Root of the K-D tree to search; may be NULL.
 * @param query Query vector to find neighbors for.
 * @param k Number of nearest neighbors to find.
 * @param results Output array of at least @p k elements.
 * @param distance_type Distance metric to use.
 * @return Number of neighbors found (0 to k), or -1 on error.
 */
int gv_kdtree_knn_search(const GV_KDNode *root, const GV_Vector *query, size_t k,
                          GV_SearchResult *results, GV_DistanceType distance_type);

/**
 * @brief Find k nearest neighbors with metadata filtering.
 *
 * @param root Root of the K-D tree to search; may be NULL.
 * @param query Query vector to find neighbors for.
 * @param k Number of nearest neighbors to find.
 * @param results Output array of at least @p k elements.
 * @param distance_type Distance metric to use.
 * @param filter_key Metadata key to filter by; NULL to disable filtering.
 * @param filter_value Metadata value to match; NULL if filter_key is NULL.
 * @return Number of neighbors found (0 to k), or -1 on error.
 */
int gv_kdtree_knn_search_filtered(const GV_KDNode *root, const GV_Vector *query, size_t k,
                                   GV_SearchResult *results, GV_DistanceType distance_type,
                                   const char *filter_key, const char *filter_value);

#ifdef __cplusplus
}
#endif

#endif

