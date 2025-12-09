#ifndef GIGAVECTOR_GV_VECTOR_H
#define GIGAVECTOR_GV_VECTOR_H

#include <stddef.h>

#include "gv_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Allocate a vector with the given dimensionality.
 *
 * The vector's components are initialized to zero.
 *
 * @param dimension Number of floating-point components.
 * @return Pointer to the allocated vector, or NULL on failure or zero dimension.
 */
GV_Vector *gv_vector_create(size_t dimension);

/**
 * @brief Create a vector by copying data from a provided array.
 *
 * @param dimension Number of floating-point components.
 * @param data Source array containing at least @p dimension elements.
 * @return Pointer to the allocated vector, or NULL on failure or invalid input.
 */
GV_Vector *gv_vector_create_from_data(size_t dimension, const float *data);

/**
 * @brief Release memory owned by a vector and its data buffer.
 *
 * Safe to call with NULL; no action is taken.
 *
 * @param vector Vector to destroy.
 */
void gv_vector_destroy(GV_Vector *vector);

/**
 * @brief Set a component value at the given index.
 *
 * @param vector Target vector.
 * @param index Zero-based component index to modify.
 * @param value New value to write.
 * @return 0 on success, -1 on invalid arguments or index out of range.
 */
int gv_vector_set(GV_Vector *vector, size_t index, float value);

/**
 * @brief Retrieve a component value at the given index.
 *
 * @param vector Target vector.
 * @param index Zero-based component index to read.
 * @param out_value Destination for the retrieved value.
 * @return 0 on success, -1 on invalid arguments or index out of range.
 */
int gv_vector_get(const GV_Vector *vector, size_t index, float *out_value);

/**
 * @brief Set all components of the vector to zero.
 *
 * @param vector Target vector.
 * @return 0 on success, -1 if @p vector is NULL.
 */
int gv_vector_clear(GV_Vector *vector);

#ifdef __cplusplus
}
#endif

#endif

