#ifndef GIGAVECTOR_GV_KDTREE_H
#define GIGAVECTOR_GV_KDTREE_H

#include <stddef.h>
#include <stdio.h>

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
 * @brief Recursively serialize a K-D tree to an open FILE stream.
 *
 * The format uses a pre-order traversal with presence flags for null nodes.
 * A byte flag is written for each node (1 = present, 0 = null). For present
 * nodes, the axis (uint32_t) and all vector components (float) are written
 * before recursing into children.
 *
 * @param node Root of the subtree to serialize; may be NULL.
 * @param out Open FILE stream in binary mode.
 * @return 0 on success, -1 on invalid arguments or write failure.
 */
int gv_kdtree_save_recursive(const GV_KDNode *node, FILE *out);

/**
 * @brief Recursively load a K-D tree from an open FILE stream.
 *
 * Expects the same pre-order format emitted by gv_kdtree_save_recursive().
 * On success, allocates nodes and vectors; ownership is given to the caller.
 *
 * @param root Pointer to receive the root pointer; must be non-NULL.
 * @param in Open FILE stream positioned at the start of a node record.
 * @param dimension Expected vector dimensionality.
 * @return 0 on success, -1 on invalid arguments or read/allocation failure.
 */
int gv_kdtree_load_recursive(GV_KDNode **root, FILE *in, size_t dimension);

#ifdef __cplusplus
}
#endif

#endif

