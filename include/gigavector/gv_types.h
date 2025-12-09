#ifndef GIGAVECTOR_GV_TYPES_H
#define GIGAVECTOR_GV_TYPES_H

#include <stddef.h>

/**
 * @brief Represents a key-value metadata pair.
 */
typedef struct GV_Metadata {
    char *key;
    char *value;
    struct GV_Metadata *next;
} GV_Metadata;

/**
 * @brief Represents a single floating-point vector with optional metadata.
 */
typedef struct {
    size_t dimension;
    float *data;
    GV_Metadata *metadata;
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

/**
 * @brief Represents a single search result with distance.
 */
typedef struct {
    const GV_Vector *vector;
    float distance;
} GV_SearchResult;

#endif

