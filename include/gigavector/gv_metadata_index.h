#ifndef GIGAVECTOR_GV_METADATA_INDEX_H
#define GIGAVECTOR_GV_METADATA_INDEX_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque handle for metadata inverted index.
 */
typedef struct GV_MetadataIndex GV_MetadataIndex;

/**
 * @brief Create a new metadata inverted index.
 *
 * @return Allocated index or NULL on failure.
 */
GV_MetadataIndex *gv_metadata_index_create(void);

/**
 * @brief Destroy a metadata inverted index and free all resources.
 *
 * @param index Index to destroy; safe to call with NULL.
 */
void gv_metadata_index_destroy(GV_MetadataIndex *index);

/**
 * @brief Add a vector index to the inverted index for a given key-value pair.
 *
 * @param index Metadata index; must be non-NULL.
 * @param key Metadata key; must be non-NULL.
 * @param value Metadata value; must be non-NULL.
 * @param vector_index Vector index to add.
 * @return 0 on success, -1 on error.
 */
int gv_metadata_index_add(GV_MetadataIndex *index, const char *key, const char *value, size_t vector_index);

/**
 * @brief Remove a vector index from the inverted index for a given key-value pair.
 *
 * @param index Metadata index; must be non-NULL.
 * @param key Metadata key; must be non-NULL.
 * @param value Metadata value; must be non-NULL.
 * @param vector_index Vector index to remove.
 * @return 0 on success, -1 on error.
 */
int gv_metadata_index_remove(GV_MetadataIndex *index, const char *key, const char *value, size_t vector_index);

/**
 * @brief Query the inverted index to get all vector indices matching a key-value pair.
 *
 * @param index Metadata index; must be non-NULL.
 * @param key Metadata key; must be non-NULL.
 * @param value Metadata value; must be non-NULL.
 * @param out_indices Output array to store vector indices; must be pre-allocated.
 * @param max_indices Maximum number of indices to return.
 * @return Number of indices found and written to out_indices, or -1 on error.
 */
int gv_metadata_index_query(const GV_MetadataIndex *index, const char *key, const char *value,
                            size_t *out_indices, size_t max_indices);

/**
 * @brief Get the count of vector indices matching a key-value pair.
 *
 * @param index Metadata index; must be non-NULL.
 * @param key Metadata key; must be non-NULL.
 * @param value Metadata value; must be non-NULL.
 * @return Number of matching vector indices, or -1 on error.
 */
size_t gv_metadata_index_count(const GV_MetadataIndex *index, const char *key, const char *value);

/**
 * @brief Remove all entries for a given vector index (used when vector is deleted).
 *
 * @param index Metadata index; must be non-NULL.
 * @param vector_index Vector index to remove all entries for.
 * @return 0 on success, -1 on error.
 */
int gv_metadata_index_remove_vector(GV_MetadataIndex *index, size_t vector_index);

/**
 * @brief Update metadata for a vector (remove old entries, add new ones).
 *
 * @param index Metadata index; must be non-NULL.
 * @param vector_index Vector index to update.
 * @param old_metadata Old metadata (to remove).
 * @param new_metadata New metadata (to add).
 * @return 0 on success, -1 on error.
 */
int gv_metadata_index_update(GV_MetadataIndex *index, size_t vector_index,
                             const void *old_metadata, const void *new_metadata);

#ifdef __cplusplus
}
#endif

#endif


