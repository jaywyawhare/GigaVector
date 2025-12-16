#ifndef GIGAVECTOR_GV_DISTANCE_H
#define GIGAVECTOR_GV_DISTANCE_H

#include <stddef.h>

#include "gv_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Distance metric types.
 */
typedef enum {
    GV_DISTANCE_EUCLIDEAN = 0,
    GV_DISTANCE_COSINE = 1,
    GV_DISTANCE_DOT_PRODUCT = 2,
    GV_DISTANCE_MANHATTAN = 3
} GV_DistanceType;

/**
 * @brief Calculate Euclidean distance between two vectors.
 *
 * @param a First vector; must be non-NULL with matching dimension.
 * @param b Second vector; must be non-NULL with matching dimension.
 * @return Euclidean distance (non-negative), or -1.0f on invalid arguments.
 */
float gv_distance_euclidean(const GV_Vector *a, const GV_Vector *b);

/**
 * @brief Calculate cosine similarity between two vectors.
 *
 * Returns a value in [-1, 1] where 1 means identical direction.
 *
 * @param a First vector; must be non-NULL with matching dimension.
 * @param b Second vector; must be non-NULL with matching dimension.
 * @return Cosine similarity in [-1, 1], or -2.0f on invalid arguments.
 */
float gv_distance_cosine(const GV_Vector *a, const GV_Vector *b);

/**
 * @brief Calculate dot product between two vectors.
 *
 * Returns negative dot product as distance (higher dot product = lower distance).
 * This allows dot product to be used in similarity search where lower distance
 * indicates higher similarity.
 *
 * @param a First vector; must be non-NULL with matching dimension.
 * @param b Second vector; must be non-NULL with matching dimension.
 * @return Negative dot product (distance), or -1.0f on invalid arguments.
 */
float gv_distance_dot_product(const GV_Vector *a, const GV_Vector *b);

/**
 * @brief Calculate Manhattan (L1) distance between two vectors.
 *
 * @param a First vector; must be non-NULL with matching dimension.
 * @param b Second vector; must be non-NULL with matching dimension.
 * @return Manhattan distance (non-negative), or -1.0f on invalid arguments.
 */
float gv_distance_manhattan(const GV_Vector *a, const GV_Vector *b);

/**
 * @brief Calculate distance using the specified metric type.
 *
 * @param a First vector; must be non-NULL.
 * @param b Second vector; must be non-NULL.
 * @param type Distance metric to use.
 * @return Distance value (interpretation depends on metric), or negative on error.
 */
float gv_distance(const GV_Vector *a, const GV_Vector *b, GV_DistanceType type);

#ifdef __cplusplus
}
#endif

#endif

