#ifndef GIGAVECTOR_GV_TTL_H
#define GIGAVECTOR_GV_TTL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_ttl.h
 * @brief Time-to-Live (TTL) support for automatic data expiration.
 *
 * This module provides TTL management for vectors, allowing automatic
 * expiration and cleanup of stale data.
 */

/* Forward declaration */
struct GV_Database;

/**
 * @brief TTL configuration.
 */
typedef struct {
    uint64_t default_ttl_seconds;      /**< Default TTL for new vectors (0 = no expiration). */
    uint64_t cleanup_interval_seconds; /**< Background cleanup interval (default: 60). */
    int lazy_expiration;               /**< Check expiration on access vs background (default: 1). */
    size_t max_expired_per_cleanup;    /**< Max vectors to expire per cleanup cycle (default: 1000). */
} GV_TTLConfig;

/**
 * @brief TTL statistics.
 */
typedef struct {
    uint64_t total_vectors_with_ttl;   /**< Vectors with TTL set. */
    uint64_t total_expired;            /**< Total vectors expired. */
    uint64_t next_expiration_time;     /**< Unix timestamp of next expiration. */
    uint64_t last_cleanup_time;        /**< Unix timestamp of last cleanup. */
} GV_TTLStats;

/**
 * @brief Opaque TTL manager handle.
 */
typedef struct GV_TTLManager GV_TTLManager;

/* TTL Manager Lifecycle */

/**
 * @brief Create a TTL manager.
 *
 * @param config TTL configuration; NULL to use defaults.
 * @return TTL manager instance, or NULL on error.
 */
GV_TTLManager *gv_ttl_create(const GV_TTLConfig *config);

/**
 * @brief Destroy a TTL manager and free all resources.
 *
 * @param mgr TTL manager instance (safe to call with NULL).
 */
void gv_ttl_destroy(GV_TTLManager *mgr);

/**
 * @brief Initialize TTL configuration with default values.
 *
 * Default values:
 * - default_ttl_seconds: 0 (no expiration)
 * - cleanup_interval_seconds: 60
 * - lazy_expiration: 1
 * - max_expired_per_cleanup: 1000
 *
 * @param config Configuration structure to initialize.
 */
void gv_ttl_config_init(GV_TTLConfig *config);

/* TTL Operations */

/**
 * @brief Set TTL for a vector.
 *
 * @param mgr TTL manager.
 * @param vector_index Index of the vector.
 * @param ttl_seconds TTL in seconds from now (0 to remove TTL).
 * @return 0 on success, -1 on error.
 */
int gv_ttl_set(GV_TTLManager *mgr, size_t vector_index, uint64_t ttl_seconds);

/**
 * @brief Set absolute expiration time for a vector.
 *
 * @param mgr TTL manager.
 * @param vector_index Index of the vector.
 * @param expire_at_unix Unix timestamp when vector expires (0 to remove TTL).
 * @return 0 on success, -1 on error.
 */
int gv_ttl_set_absolute(GV_TTLManager *mgr, size_t vector_index, uint64_t expire_at_unix);

/**
 * @brief Get expiration time for a vector.
 *
 * @param mgr TTL manager.
 * @param vector_index Index of the vector.
 * @param expire_at Output unix timestamp (0 if no TTL).
 * @return 0 on success, -1 on error.
 */
int gv_ttl_get(const GV_TTLManager *mgr, size_t vector_index, uint64_t *expire_at);

/**
 * @brief Remove TTL from a vector.
 *
 * @param mgr TTL manager.
 * @param vector_index Index of the vector.
 * @return 0 on success, -1 on error.
 */
int gv_ttl_remove(GV_TTLManager *mgr, size_t vector_index);

/**
 * @brief Check if a vector has expired.
 *
 * @param mgr TTL manager.
 * @param vector_index Index of the vector.
 * @return 1 if expired, 0 if not expired or no TTL, -1 on error.
 */
int gv_ttl_is_expired(const GV_TTLManager *mgr, size_t vector_index);

/**
 * @brief Get remaining time before expiration.
 *
 * @param mgr TTL manager.
 * @param vector_index Index of the vector.
 * @param remaining_seconds Output remaining seconds (0 if expired or no TTL).
 * @return 0 on success, -1 on error.
 */
int gv_ttl_get_remaining(const GV_TTLManager *mgr, size_t vector_index, uint64_t *remaining_seconds);

/* Cleanup Operations */

/**
 * @brief Cleanup expired vectors (synchronous).
 *
 * Scans for expired vectors and deletes them from the database.
 *
 * @param mgr TTL manager.
 * @param db Database to cleanup (must be non-NULL).
 * @return Number of vectors expired, or -1 on error.
 */
int gv_ttl_cleanup_expired(GV_TTLManager *mgr, struct GV_Database *db);

/**
 * @brief Start background cleanup thread.
 *
 * The cleanup thread periodically scans for and removes expired vectors.
 *
 * @param mgr TTL manager.
 * @param db Database to cleanup.
 * @return 0 on success, -1 on error.
 */
int gv_ttl_start_background_cleanup(GV_TTLManager *mgr, struct GV_Database *db);

/**
 * @brief Stop background cleanup thread.
 *
 * @param mgr TTL manager.
 */
void gv_ttl_stop_background_cleanup(GV_TTLManager *mgr);

/**
 * @brief Check if background cleanup is running.
 *
 * @param mgr TTL manager.
 * @return 1 if running, 0 if not, -1 on error.
 */
int gv_ttl_is_background_cleanup_running(const GV_TTLManager *mgr);

/* Statistics */

/**
 * @brief Get TTL statistics.
 *
 * @param mgr TTL manager.
 * @param stats Output statistics structure.
 * @return 0 on success, -1 on error.
 */
int gv_ttl_get_stats(const GV_TTLManager *mgr, GV_TTLStats *stats);

/* Bulk Operations */

/**
 * @brief Set TTL for multiple vectors.
 *
 * @param mgr TTL manager.
 * @param indices Array of vector indices.
 * @param count Number of indices.
 * @param ttl_seconds TTL in seconds from now.
 * @return Number of TTLs set, or -1 on error.
 */
int gv_ttl_set_bulk(GV_TTLManager *mgr, const size_t *indices, size_t count, uint64_t ttl_seconds);

/**
 * @brief Get all vectors expiring before a given time.
 *
 * @param mgr TTL manager.
 * @param before_unix Unix timestamp.
 * @param indices Output array (caller allocates).
 * @param max_indices Maximum number of indices to return.
 * @return Number of indices found, or -1 on error.
 */
int gv_ttl_get_expiring_before(const GV_TTLManager *mgr, uint64_t before_unix,
                                size_t *indices, size_t max_indices);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_TTL_H */
