#ifndef GIGAVECTOR_GV_NAMESPACE_H
#define GIGAVECTOR_GV_NAMESPACE_H

#include <stddef.h>
#include <stdint.h>

#include "gv_types.h"
#include "gv_distance.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_namespace.h
 * @brief Namespace/Collection support for multi-tenancy.
 *
 * This module provides namespace isolation, allowing multiple logical
 * vector collections within a single database instance.
 */

/* Forward declarations */
struct GV_Database;
typedef struct GV_Database GV_Database;

/**
 * @brief Index type enumeration (from gv_database.h).
 */
typedef enum {
    GV_NS_INDEX_KDTREE = 0,
    GV_NS_INDEX_HNSW = 1,
    GV_NS_INDEX_IVFPQ = 2,
    GV_NS_INDEX_SPARSE = 3
} GV_NSIndexType;

/**
 * @brief Namespace configuration.
 */
typedef struct {
    const char *name;                  /**< Namespace name (required, max 64 chars). */
    size_t dimension;                  /**< Vector dimension (required). */
    GV_NSIndexType index_type;         /**< Index type (default: HNSW). */
    size_t max_vectors;                /**< Maximum vectors (0 = unlimited). */
    size_t max_memory_bytes;           /**< Maximum memory (0 = unlimited). */
} GV_NamespaceConfig;

/**
 * @brief Namespace information.
 */
typedef struct {
    char *name;                        /**< Namespace name. */
    size_t dimension;                  /**< Vector dimension. */
    GV_NSIndexType index_type;         /**< Index type. */
    size_t vector_count;               /**< Current vector count. */
    size_t memory_bytes;               /**< Current memory usage. */
    uint64_t created_at;               /**< Creation timestamp. */
    uint64_t last_modified;            /**< Last modification timestamp. */
} GV_NamespaceInfo;

/**
 * @brief Opaque namespace handle.
 */
typedef struct GV_Namespace GV_Namespace;

/**
 * @brief Opaque namespace manager handle.
 */
typedef struct GV_NamespaceManager GV_NamespaceManager;

/* ============================================================================
 * Namespace Manager Lifecycle
 * ============================================================================ */

/**
 * @brief Create a namespace manager.
 *
 * @param base_path Base directory for namespace persistence (NULL for in-memory only).
 * @return Namespace manager instance, or NULL on error.
 */
GV_NamespaceManager *gv_namespace_manager_create(const char *base_path);

/**
 * @brief Destroy a namespace manager and all namespaces.
 *
 * @param mgr Namespace manager instance (safe to call with NULL).
 */
void gv_namespace_manager_destroy(GV_NamespaceManager *mgr);

/* ============================================================================
 * Namespace Operations
 * ============================================================================ */

/**
 * @brief Create a new namespace.
 *
 * @param mgr Namespace manager.
 * @param config Namespace configuration.
 * @return Namespace handle, or NULL on error.
 */
GV_Namespace *gv_namespace_create(GV_NamespaceManager *mgr, const GV_NamespaceConfig *config);

/**
 * @brief Get an existing namespace by name.
 *
 * @param mgr Namespace manager.
 * @param name Namespace name.
 * @return Namespace handle, or NULL if not found.
 */
GV_Namespace *gv_namespace_get(GV_NamespaceManager *mgr, const char *name);

/**
 * @brief Delete a namespace and all its data.
 *
 * @param mgr Namespace manager.
 * @param name Namespace name.
 * @return 0 on success, -1 on error.
 */
int gv_namespace_delete(GV_NamespaceManager *mgr, const char *name);

/**
 * @brief List all namespaces.
 *
 * @param mgr Namespace manager.
 * @param names Output array of namespace names (caller must free each string and array).
 * @param count Output number of namespaces.
 * @return 0 on success, -1 on error.
 */
int gv_namespace_list(GV_NamespaceManager *mgr, char ***names, size_t *count);

/**
 * @brief Get namespace information.
 *
 * @param ns Namespace handle.
 * @param info Output information structure.
 * @return 0 on success, -1 on error.
 */
int gv_namespace_get_info(const GV_Namespace *ns, GV_NamespaceInfo *info);

/**
 * @brief Free namespace information structure.
 *
 * @param info Information structure to free.
 */
void gv_namespace_free_info(GV_NamespaceInfo *info);

/**
 * @brief Check if a namespace exists.
 *
 * @param mgr Namespace manager.
 * @param name Namespace name.
 * @return 1 if exists, 0 if not, -1 on error.
 */
int gv_namespace_exists(GV_NamespaceManager *mgr, const char *name);

/* ============================================================================
 * Vector Operations within Namespace
 * ============================================================================ */

/**
 * @brief Add a vector to a namespace.
 *
 * @param ns Namespace handle.
 * @param data Vector data.
 * @param dimension Vector dimension (must match namespace dimension).
 * @return 0 on success, -1 on error.
 */
int gv_namespace_add_vector(GV_Namespace *ns, const float *data, size_t dimension);

/**
 * @brief Add a vector with metadata to a namespace.
 *
 * @param ns Namespace handle.
 * @param data Vector data.
 * @param dimension Vector dimension.
 * @param keys Metadata keys array.
 * @param values Metadata values array.
 * @param meta_count Number of metadata entries.
 * @return 0 on success, -1 on error.
 */
int gv_namespace_add_vector_with_metadata(GV_Namespace *ns, const float *data, size_t dimension,
                                           const char *const *keys, const char *const *values,
                                           size_t meta_count);

/**
 * @brief Search for k nearest neighbors in a namespace.
 *
 * @param ns Namespace handle.
 * @param query Query vector.
 * @param k Number of neighbors.
 * @param results Output results array (must be pre-allocated with k elements).
 * @param distance_type Distance metric.
 * @return Number of results found, or -1 on error.
 */
int gv_namespace_search(const GV_Namespace *ns, const float *query, size_t k,
                        GV_SearchResult *results, GV_DistanceType distance_type);

/**
 * @brief Search with metadata filter in a namespace.
 *
 * @param ns Namespace handle.
 * @param query Query vector.
 * @param k Number of neighbors.
 * @param results Output results array.
 * @param distance_type Distance metric.
 * @param filter_key Metadata key to filter by.
 * @param filter_value Metadata value to match.
 * @return Number of results found, or -1 on error.
 */
int gv_namespace_search_filtered(const GV_Namespace *ns, const float *query, size_t k,
                                  GV_SearchResult *results, GV_DistanceType distance_type,
                                  const char *filter_key, const char *filter_value);

/**
 * @brief Delete a vector by index in a namespace.
 *
 * @param ns Namespace handle.
 * @param vector_index Index of vector to delete.
 * @return 0 on success, -1 on error.
 */
int gv_namespace_delete_vector(GV_Namespace *ns, size_t vector_index);

/**
 * @brief Get vector count in a namespace.
 *
 * @param ns Namespace handle.
 * @return Vector count, or 0 on error.
 */
size_t gv_namespace_count(const GV_Namespace *ns);

/* ============================================================================
 * Persistence
 * ============================================================================ */

/**
 * @brief Save a namespace to disk.
 *
 * @param ns Namespace handle.
 * @return 0 on success, -1 on error.
 */
int gv_namespace_save(GV_Namespace *ns);

/**
 * @brief Save all namespaces to disk.
 *
 * @param mgr Namespace manager.
 * @return 0 on success, -1 on error.
 */
int gv_namespace_manager_save_all(GV_NamespaceManager *mgr);

/**
 * @brief Load all namespaces from disk.
 *
 * @param mgr Namespace manager.
 * @return Number of namespaces loaded, or -1 on error.
 */
int gv_namespace_manager_load_all(GV_NamespaceManager *mgr);

/* ============================================================================
 * Utility
 * ============================================================================ */

/**
 * @brief Get the underlying database handle for a namespace.
 *
 * This allows using the full GV_Database API on a namespace.
 * Use with caution as modifications bypass namespace tracking.
 *
 * @param ns Namespace handle.
 * @return Database handle, or NULL on error.
 */
GV_Database *gv_namespace_get_db(GV_Namespace *ns);

/**
 * @brief Initialize namespace configuration with defaults.
 *
 * Default values:
 * - name: NULL (must be set)
 * - dimension: 0 (must be set)
 * - index_type: GV_NS_INDEX_HNSW
 * - max_vectors: 0 (unlimited)
 * - max_memory_bytes: 0 (unlimited)
 *
 * @param config Configuration to initialize.
 */
void gv_namespace_config_init(GV_NamespaceConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_NAMESPACE_H */
