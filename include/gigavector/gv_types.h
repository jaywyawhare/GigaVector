#ifndef GIGAVECTOR_GV_TYPES_H
#define GIGAVECTOR_GV_TYPES_H

#include <stddef.h>
#include <stdint.h>

typedef struct GV_Metadata {
    char *key;
    char *value;
    struct GV_Metadata *next;
} GV_Metadata;

typedef struct {
    size_t dimension;
    float *data;
    GV_Metadata *metadata;
} GV_Vector;

/**
 * @brief Node for a simple K-D tree storing vectors.
 * 
 * Uses Structure-of-Arrays storage: stores vector index instead of pointer.
 */
typedef struct GV_KDNode {
    size_t vector_index;  /**< Index into SoA storage. */
    size_t axis;
    struct GV_KDNode *left;
    struct GV_KDNode *right;
} GV_KDNode;

typedef struct {
    uint32_t index;
    float value;
} GV_SparseEntry;

typedef struct GV_SparseVector {
    size_t dimension;
    size_t nnz;
    GV_SparseEntry *entries;
    GV_Metadata *metadata;
} GV_SparseVector;

typedef struct {
    const GV_Vector *vector;
    const GV_SparseVector *sparse_vector;
    int is_sparse;
    float distance;
    size_t id;
} GV_SearchResult;

#endif

