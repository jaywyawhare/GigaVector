#ifndef GIGAVECTOR_GV_TYPES_H
#define GIGAVECTOR_GV_TYPES_H

#include <stddef.h>

/**
 * @brief Represents a single floating-point vector.
 */
typedef struct {
    size_t dimension;
    float *data;
} GV_Vector;

/**
 * @brief Node for a simple K-D tree storing vectors.
 */
typedef struct GV_KDNode {
    GV_Vector *point;
    size_t axis;
    struct GV_KDNode *left;
    struct GV_KDNode *right;
} GV_KDNode;

#endif

