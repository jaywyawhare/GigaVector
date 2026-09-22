#ifndef GIGAVECTOR_GV_METADATA_H
#define GIGAVECTOR_GV_METADATA_H

#include <stddef.h>
#include <stdio.h>

#include "core/types.h"

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
int vector_set_metadata(GV_Vector *vector, const char *key, const char *value);

/**
 * @brief Retrieve a metadata value by key.
 *
 * @param vector Source vector; must be non-NULL.
 * @param key Metadata key to look up; must be non-NULL.
 * @return Pointer to the value string, or NULL if not found.
 */
const char *vector_get_metadata(const GV_Vector *vector, const char *key);

/**
 * @brief Remove a metadata entry by key.
 *
 * @param vector Target vector; must be non-NULL.
 * @param key Metadata key to remove; must be non-NULL.
 * @return 0 on success (or if key not found), -1 on invalid arguments.
 */
int vector_remove_metadata(GV_Vector *vector, const char *key);

/**
 * @brief Clear all metadata from a vector.
 *
 * @param vector Target vector; must be non-NULL.
 */
void vector_clear_metadata(GV_Vector *vector);

/**
 * @brief Create metadata from key-value pairs.
 *
 * @param keys Array of key strings.
 * @param values Array of value strings.
 * @param count Number of key-value pairs.
 * @return New metadata linked list, or NULL on failure.
 */
GV_Metadata *metadata_from_keys_values(const char **keys, const char **values, size_t count);

/**
 * @brief Free a metadata linked list.
 *
 * @param meta Metadata to free; safe to call with NULL.
 */
void metadata_free(GV_Metadata *meta);

/**
 * @brief Read a metadata block written by write_metadata() and attach each
 *        key/value pair to @p vector (count + length-prefixed strings).
 *
 * The inverse of core/utils.h write_metadata(); shared by every index's load
 * path so the (de)serialization format lives in one place.
 *
 * @param in Input stream positioned at the metadata block.
 * @param vector Target vector; pairs are added via vector_set_metadata.
 * @return 0 on success, -1 on read or allocation failure.
 */
int read_metadata_into_vector(FILE *in, GV_Vector *vector);

/**
 * @brief Apply every key/value pair in the @p src metadata list onto @p dst
 *        (skipping entries with a NULL key or value).
 *
 * @param dst Target vector; unchanged if NULL.
 * @param src Metadata list head; no-op if NULL.
 */
void vector_apply_metadata(GV_Vector *dst, const GV_Metadata *src);

/**
 * @brief Copy every key/value pair from @p src's metadata onto @p dst.
 *
 * @param dst Target vector; unchanged if NULL.
 * @param src Source vector; no-op if NULL or has no metadata.
 */
void vector_copy_metadata(GV_Vector *dst, const GV_Vector *src);

#ifdef __cplusplus
}
#endif

#endif

