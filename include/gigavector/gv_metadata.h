#ifndef GIGAVECTOR_GV_METADATA_H
#define GIGAVECTOR_GV_METADATA_H

#include <stddef.h>

#include "gv_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Add or update a metadata key-value pair for a vector.
 *
 * @param vector Target vector; must be non-NULL.
 * @param key Metadata key string; must be non-NULL.
 * @param value Metadata value string; must be non-NULL.
 * @return 0 on success, -1 on invalid arguments or allocation failure.
 */
int gv_vector_set_metadata(GV_Vector *vector, const char *key, const char *value);

/**
 * @brief Retrieve a metadata value by key.
 *
 * @param vector Source vector; must be non-NULL.
 * @param key Metadata key to look up; must be non-NULL.
 * @return Pointer to the value string, or NULL if not found.
 */
const char *gv_vector_get_metadata(const GV_Vector *vector, const char *key);

/**
 * @brief Remove a metadata entry by key.
 *
 * @param vector Target vector; must be non-NULL.
 * @param key Metadata key to remove; must be non-NULL.
 * @return 0 on success (or if key not found), -1 on invalid arguments.
 */
int gv_vector_remove_metadata(GV_Vector *vector, const char *key);

/**
 * @brief Clear all metadata from a vector.
 *
 * @param vector Target vector; must be non-NULL.
 */
void gv_vector_clear_metadata(GV_Vector *vector);

#ifdef __cplusplus
}
#endif

#endif

