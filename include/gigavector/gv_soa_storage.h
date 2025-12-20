#ifndef GIGAVECTOR_GV_SOA_STORAGE_H
#define GIGAVECTOR_GV_SOA_STORAGE_H

#include <stddef.h>
#include "gv_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Structure-of-Arrays storage for dense vectors.
 * 
 * Stores all vector data in a contiguous packed buffer for better cache locality.
 * Metadata is stored separately per vector since it's variable size.
 */
typedef struct {
    size_t dimension;        /**< Dimensionality of all vectors. */
    size_t count;            /**< Current number of vectors stored. */
    size_t capacity;         /**< Allocated capacity (number of vectors). */
    float *data;             /**< Contiguous array: [vec0_dim0, vec0_dim1, ..., vec1_dim0, ...]. */
    GV_Metadata **metadata;  /**< Array of metadata pointers, one per vector (may be NULL). */
} GV_SoAStorage;

/**
 * @brief Create a new SoA storage with the given dimension and initial capacity.
 *
 * @param dimension Vector dimensionality; must be > 0.
 * @param initial_capacity Initial capacity (number of vectors); 0 uses default (1024).
 * @return Allocated storage or NULL on failure.
 */
GV_SoAStorage *gv_soa_storage_create(size_t dimension, size_t initial_capacity);

/**
 * @brief Destroy SoA storage and free all resources.
 *
 * @param storage Storage to destroy; safe to call with NULL.
 */
void gv_soa_storage_destroy(GV_SoAStorage *storage);

/**
 * @brief Add a vector to the storage.
 *
 * @param storage Target storage; must be non-NULL.
 * @param data Pointer to dimension floats to copy.
 * @param metadata Optional metadata to attach; ownership transferred if non-NULL.
 * @return Vector index (>= 0) on success, or (size_t)-1 on failure.
 */
size_t gv_soa_storage_add(GV_SoAStorage *storage, const float *data, GV_Metadata *metadata);

/**
 * @brief Get a pointer to the data for a vector at the given index.
 *
 * The returned pointer is valid until the storage is destroyed or resized.
 * For a vector at index i, the data starts at storage->data[i * storage->dimension].
 *
 * @param storage Storage to query; must be non-NULL.
 * @param index Vector index; must be < storage->count.
 * @return Pointer to the vector's data, or NULL on invalid index.
 */
const float *gv_soa_storage_get_data(const GV_SoAStorage *storage, size_t index);

/**
 * @brief Get metadata for a vector at the given index.
 *
 * @param storage Storage to query; must be non-NULL.
 * @param index Vector index; must be < storage->count.
 * @return Metadata pointer (may be NULL if no metadata), or NULL on invalid index.
 */
GV_Metadata *gv_soa_storage_get_metadata(const GV_SoAStorage *storage, size_t index);

/**
 * @brief Create a GV_Vector view of a vector in SoA storage (for compatibility).
 *
 * This creates a lightweight view that points into the SoA storage.
 * The returned vector should NOT be destroyed with gv_vector_destroy.
 * It is only valid while the storage exists.
 *
 * @param storage Storage to query; must be non-NULL.
 * @param index Vector index; must be < storage->count.
 * @param out_vector Output pointer to fill; must be non-NULL.
 * @return 0 on success, -1 on invalid arguments.
 */
int gv_soa_storage_get_vector_view(const GV_SoAStorage *storage, size_t index, GV_Vector *out_vector);

/**
 * @brief Get the current count of vectors in storage.
 *
 * @param storage Storage to query; must be non-NULL.
 * @return Number of vectors stored.
 */
size_t gv_soa_storage_count(const GV_SoAStorage *storage);

/**
 * @brief Get the dimension of vectors in storage.
 *
 * @param storage Storage to query; must be non-NULL.
 * @return Vector dimension.
 */
size_t gv_soa_storage_dimension(const GV_SoAStorage *storage);

#ifdef __cplusplus
}
#endif

#endif


