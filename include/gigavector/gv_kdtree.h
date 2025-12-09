#ifndef GIGAVECTOR_GV_KDTREE_H
#define GIGAVECTOR_GV_KDTREE_H

#include <stddef.h>

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

#ifdef __cplusplus
}
#endif

#endif

