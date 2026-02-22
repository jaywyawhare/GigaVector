#ifndef GIGAVECTOR_GV_TIMETRAVEL_H
#define GIGAVECTOR_GV_TIMETRAVEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_timetravel.h
 * @brief Time-Travel / Auto-Versioning for GigaVector.
 *
 * Every mutation automatically creates a new version. Users can query any
 * historical snapshot without explicit snapshot management. Change records
 * are stored in an append-only log; point-in-time reconstruction replays
 * changes backwards from the current state.
 */

/* Configuration */

/**
 * @brief Time-travel configuration.
 */
typedef struct {
    size_t max_versions;    /**< Maximum change records to retain (default: 1000). */
    size_t max_storage_mb;  /**< Maximum storage budget in MiB (default: 512). */
    int    auto_gc;         /**< Auto garbage-collect when limits exceeded (default: 1). */
    size_t gc_keep_count;   /**< Minimum recent versions to keep during GC (default: 100). */
} GV_TimeTravelConfig;

/* Opaque Types */

/**
 * @brief Opaque time-travel manager handle.
 */
typedef struct GV_TimeTravelManager GV_TimeTravelManager;

/* Version Entry */

/**
 * @brief Metadata for a single version in the change log.
 */
typedef struct {
    uint64_t version_id;        /**< Monotonically increasing version identifier. */
    uint64_t timestamp;         /**< Creation time in microseconds since epoch. */
    size_t   vector_count;      /**< Total live vector count at this version. */
    char     description[128];  /**< Human-readable mutation description. */
} GV_VersionEntry;

/* Lifecycle */

/**
 * @brief Initialize a TimeTravelConfig with default values.
 *
 * Default values:
 * - max_versions: 1000
 * - max_storage_mb: 512
 * - auto_gc: 1
 * - gc_keep_count: 100
 *
 * @param config Configuration structure to initialize.
 */
void gv_tt_config_init(GV_TimeTravelConfig *config);

/**
 * @brief Create a time-travel manager.
 *
 * @param config Configuration; NULL to use defaults.
 * @return Manager instance, or NULL on error.
 */
GV_TimeTravelManager *gv_tt_create(const GV_TimeTravelConfig *config);

/**
 * @brief Destroy a time-travel manager and free all resources.
 *
 * @param mgr Manager instance (safe to call with NULL).
 */
void gv_tt_destroy(GV_TimeTravelManager *mgr);

/* Mutation Recording */

/**
 * @brief Record a vector insertion.
 *
 * @param mgr   Manager instance.
 * @param index Vector index that was inserted.
 * @param vector   Pointer to the inserted vector data.
 * @param dimension Number of float components.
 * @return New version ID, or 0 on error.
 */
uint64_t gv_tt_record_insert(GV_TimeTravelManager *mgr, size_t index,
                              const float *vector, size_t dimension);

/**
 * @brief Record a vector update (in-place replacement).
 *
 * @param mgr        Manager instance.
 * @param index      Vector index that was updated.
 * @param old_vector Previous vector data.
 * @param new_vector New vector data.
 * @param dimension  Number of float components.
 * @return New version ID, or 0 on error.
 */
uint64_t gv_tt_record_update(GV_TimeTravelManager *mgr, size_t index,
                              const float *old_vector, const float *new_vector,
                              size_t dimension);

/**
 * @brief Record a vector deletion.
 *
 * @param mgr       Manager instance.
 * @param index     Vector index that was deleted.
 * @param vector    The deleted vector data (for undo capability).
 * @param dimension Number of float components.
 * @return New version ID, or 0 on error.
 */
uint64_t gv_tt_record_delete(GV_TimeTravelManager *mgr, size_t index,
                              const float *vector, size_t dimension);

/* Point-in-Time Queries */

/**
 * @brief Retrieve the state of a vector at a specific version.
 *
 * Reconstructs the vector by starting from the most recent state and
 * undoing changes with version > the requested version.
 *
 * @param mgr        Manager instance.
 * @param version_id Target version.
 * @param index      Vector index to query.
 * @param output     Output buffer (must hold at least @p dimension floats).
 * @param dimension  Expected dimension.
 * @return 1 if found, 0 if not found or deleted at that version, -1 on error.
 */
int gv_tt_query_at_version(const GV_TimeTravelManager *mgr, uint64_t version_id,
                            size_t index, float *output, size_t dimension);

/**
 * @brief Retrieve the state of a vector at a specific timestamp.
 *
 * Finds the latest version whose timestamp <= the given timestamp, then
 * delegates to gv_tt_query_at_version.
 *
 * @param mgr       Manager instance.
 * @param timestamp  Target timestamp in microseconds since epoch.
 * @param index      Vector index to query.
 * @param output     Output buffer.
 * @param dimension  Expected dimension.
 * @return 1 if found, 0 if not found or deleted at that time, -1 on error.
 */
int gv_tt_query_at_timestamp(const GV_TimeTravelManager *mgr, uint64_t timestamp,
                              size_t index, float *output, size_t dimension);

/* Version Inspection */

/**
 * @brief Get the total live vector count at a specific version.
 *
 * @param mgr        Manager instance.
 * @param version_id Target version.
 * @return Live vector count, or 0 if the version is unknown.
 */
size_t gv_tt_count_at_version(const GV_TimeTravelManager *mgr, uint64_t version_id);

/**
 * @brief Get the current (latest) version ID.
 *
 * @param mgr Manager instance.
 * @return Current version ID, or 0 if no mutations have been recorded.
 */
uint64_t gv_tt_current_version(const GV_TimeTravelManager *mgr);

/**
 * @brief List recorded versions.
 *
 * Fills the output array with version metadata, ordered oldest to newest.
 *
 * @param mgr       Manager instance.
 * @param out       Output array of GV_VersionEntry.
 * @param max_count Maximum entries to write.
 * @return Number of entries written, or -1 on error.
 */
int gv_tt_list_versions(const GV_TimeTravelManager *mgr, GV_VersionEntry *out,
                         size_t max_count);

/* Garbage Collection */

/**
 * @brief Run garbage collection on the change log.
 *
 * Removes the oldest change records when version count exceeds max_versions
 * or storage exceeds max_storage_mb. Always keeps at least gc_keep_count
 * recent versions.
 *
 * @param mgr Manager instance.
 * @return Number of versions removed, or -1 on error.
 */
int gv_tt_gc(GV_TimeTravelManager *mgr);

/* Persistence */

/**
 * @brief Save the time-travel state to a binary file.
 *
 * Format: magic + header + change records array.
 *
 * @param mgr  Manager instance.
 * @param path Output file path.
 * @return 0 on success, -1 on error.
 */
int gv_tt_save(const GV_TimeTravelManager *mgr, const char *path);

/**
 * @brief Load a time-travel manager from a binary file.
 *
 * @param path Input file path.
 * @return Manager instance, or NULL on error.
 */
GV_TimeTravelManager *gv_tt_load(const char *path);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_TIMETRAVEL_H */
