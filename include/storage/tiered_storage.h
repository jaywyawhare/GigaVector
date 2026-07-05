#ifndef GIGAVECTOR_TIERED_STORAGE_H
#define GIGAVECTOR_TIERED_STORAGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Storage tier levels for vector data.
 *
 * HOT:  Recently accessed/inserted vectors, kept in full float32 in-memory SoA storage.
 * WARM: Older vectors that have not been accessed recently.
 * COLD: Oldest vectors, infrequently or never accessed.
 */
typedef enum {
    GV_TIER_HOT  = 0,
    GV_TIER_WARM = 1,
    GV_TIER_COLD = 2
} GV_StorageTier;

/**
 * @brief Opaque tiered storage manager.
 *
 * Maintains per-vector insertion timestamps and tracks tier assignments.
 * One instance lives inside GV_Database when tiering_enabled != 0.
 */
typedef struct GV_TieredStorageManager GV_TieredStorageManager;

/* Forward declaration to avoid circular includes */
struct GV_Database;

/**
 * @brief Create a new tiered storage manager.
 *
 * @param initial_capacity Initial number of vector slots to allocate.
 * @return Allocated manager or NULL on failure.
 */
GV_TieredStorageManager *tiered_storage_create(size_t initial_capacity);

/**
 * @brief Destroy a tiered storage manager and free all resources.
 *
 * @param mgr Manager to destroy; safe to call with NULL.
 */
void tiered_storage_destroy(GV_TieredStorageManager *mgr);

/**
 * @brief Record an insertion timestamp for a vector slot.
 *
 * Called immediately after a vector is inserted at @p vec_id.
 *
 * @param mgr Manager; must be non-NULL.
 * @param vec_id Vector slot index in soa_storage.
 * @param insert_time_us Insertion time in microseconds since epoch.
 * @return 0 on success, -1 on failure (e.g., allocation error).
 */
int tiered_storage_record_insert(GV_TieredStorageManager *mgr,
                                  size_t vec_id,
                                  uint64_t insert_time_us);

/**
 * @brief Override the insertion timestamp for a vector (useful for testing).
 *
 * @param mgr Manager; must be non-NULL.
 * @param vec_id Vector slot index.
 * @param insert_time_us New timestamp in microseconds since epoch.
 * @return 0 on success, -1 if vec_id is out of range.
 */
int tiered_storage_set_insert_time(GV_TieredStorageManager *mgr,
                                    size_t vec_id,
                                    uint64_t insert_time_us);

/**
 * @brief Classify a single vector into its tier based on age.
 *
 * Uses the insertion timestamp stored in the manager and the age thresholds
 * in @p db to determine the appropriate tier.
 *
 * @param db Database with tiering config; must be non-NULL with tiering_enabled != 0.
 * @param vec_id Vector slot index.
 * @return Tier level for the vector.
 */
GV_StorageTier tiered_storage_classify(const struct GV_Database *db, size_t vec_id);

/**
 * @brief Get the current tier assignment for a vector.
 *
 * @param db Database; must be non-NULL with tiering_enabled != 0.
 * @param vec_id Vector slot index.
 * @param out Pointer to store the tier; must be non-NULL.
 * @return 0 on success, -1 on invalid arguments.
 */
int tiered_storage_get_tier(const struct GV_Database *db, size_t vec_id,
                              GV_StorageTier *out);

/**
 * @brief Scan all vectors and count by tier (for stats).
 *
 * @param db Database; must be non-NULL with tiering_enabled != 0.
 * @param hot_count  Output: number of hot vectors (may be NULL).
 * @param warm_count Output: number of warm vectors (may be NULL).
 * @param cold_count Output: number of cold vectors (may be NULL).
 */
void tiered_storage_promote(const struct GV_Database *db,
                             size_t *hot_count,
                             size_t *warm_count,
                             size_t *cold_count);

/* ---- Public API (exposed in gigavector.h) ---- */

/**
 * @brief Configure tiered storage thresholds for a database.
 *
 * Enables tiering if not already enabled. Vectors older than
 * @p hot_max_age_sec are classified as WARM; those older than
 * @p warm_max_age_sec are classified as COLD.
 *
 * @param db Database; must be non-NULL.
 * @param hot_max_age_sec   Age threshold (seconds) for HOT->WARM demotion.
 * @param warm_max_age_sec  Age threshold (seconds) for WARM->COLD demotion.
 * @param hot_max_vectors   Maximum number of HOT vectors (0 = unlimited).
 * @return 0 on success, -1 on error.
 */
int gv_db_set_tiering_config(struct GV_Database *db,
                               uint64_t hot_max_age_sec,
                               uint64_t warm_max_age_sec,
                               size_t hot_max_vectors);

/**
 * @brief Get the storage tier for a specific vector.
 *
 * @param db     Database; must be non-NULL.
 * @param vec_id Vector slot index.
 * @param out    Pointer to store the tier; must be non-NULL.
 * @return 0 on success, -1 on error or if tiering is disabled.
 */
int gv_db_get_vector_tier(const struct GV_Database *db, size_t vec_id,
                            GV_StorageTier *out);

/**
 * @brief Get tiering statistics for the database.
 *
 * @param db         Database; must be non-NULL.
 * @param hot_count  Output: number of hot vectors (may be NULL).
 * @param warm_count Output: number of warm vectors (may be NULL).
 * @param cold_count Output: number of cold vectors (may be NULL).
 * @return 0 on success, -1 on error or if tiering is disabled.
 */
int gv_db_tiering_stats(const struct GV_Database *db,
                          size_t *hot_count,
                          size_t *warm_count,
                          size_t *cold_count);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_TIERED_STORAGE_H */
