#ifndef GIGAVECTOR_GV_CONDITIONAL_H
#define GIGAVECTOR_GV_CONDITIONAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_conditional.h
 * @brief CAS-style conditional mutations for safe concurrent updates.
 *
 * Provides compare-and-swap semantics for vector and metadata updates,
 * supporting optimistic concurrency control and safe embedding model
 * migrations.
 */

/* Forward declaration */
struct GV_Database;

/* ============================================================================
 * Condition Types
 * ============================================================================ */

/**
 * @brief Type of condition to evaluate before applying a mutation.
 */
typedef enum {
    GV_COND_VERSION_EQ,           /**< Tracked version must equal expected value. */
    GV_COND_VERSION_LT,           /**< Tracked version must be less than expected value. */
    GV_COND_METADATA_EQ,          /**< Metadata field must equal specified value. */
    GV_COND_METADATA_EXISTS,      /**< Metadata field must exist. */
    GV_COND_METADATA_NOT_EXISTS,  /**< Metadata field must not exist. */
    GV_COND_NOT_DELETED           /**< Vector must not be marked as deleted. */
} GV_ConditionType;

/**
 * @brief A single condition to evaluate against a vector's current state.
 */
typedef struct {
    GV_ConditionType type;        /**< Type of condition. */
    const char *field_name;       /**< Metadata field name (for METADATA_* conditions). */
    const char *field_value;      /**< Expected metadata value (for METADATA_EQ). */
    uint64_t version;             /**< Expected version (for VERSION_EQ / VERSION_LT). */
} GV_Condition;

/* ============================================================================
 * Result Codes
 * ============================================================================ */

/**
 * @brief Result of a conditional mutation.
 */
typedef enum {
    GV_COND_OK        =  0,      /**< All conditions passed; mutation applied. */
    GV_COND_FAILED    = -1,      /**< One or more conditions did not hold. */
    GV_COND_NOT_FOUND = -2,      /**< Target vector index does not exist. */
    GV_COND_CONFLICT  = -3       /**< Version conflict detected (concurrent modification). */
} GV_ConditionalResult;

/* ============================================================================
 * Versioned Vector Info
 * ============================================================================ */

/**
 * @brief Tracked version information for a single vector.
 */
typedef struct {
    size_t index;                 /**< Vector index in SoA storage. */
    uint64_t version;             /**< Current version counter. */
    uint64_t updated_at;          /**< Timestamp of last update (microseconds since epoch). */
} GV_VersionedVector;

/* ============================================================================
 * Opaque Manager Handle
 * ============================================================================ */

/**
 * @brief Opaque conditional-update manager handle.
 */
typedef struct GV_CondManager GV_CondManager;

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

/**
 * @brief Create a conditional-update manager bound to a database.
 *
 * @param db Database handle (cast to void* for header independence); must be non-NULL.
 * @return Manager instance, or NULL on error.
 */
GV_CondManager *gv_cond_create(void *db);

/**
 * @brief Destroy a conditional-update manager and free all resources.
 *
 * @param mgr Manager instance (safe to call with NULL).
 */
void gv_cond_destroy(GV_CondManager *mgr);

/* ============================================================================
 * Conditional Mutations
 * ============================================================================ */

/**
 * @brief Conditionally update vector data.
 *
 * Acquires the database write lock, evaluates all conditions against the
 * vector's current state, and if every condition holds, replaces the vector
 * data and increments the tracked version.
 *
 * @param mgr Manager instance; must be non-NULL.
 * @param index Vector index in SoA storage.
 * @param new_data Replacement vector data (dimension floats).
 * @param dimension Number of floats in new_data; must equal db->dimension.
 * @param conditions Array of conditions to evaluate (may be NULL if count is 0).
 * @param condition_count Number of conditions.
 * @return GV_COND_OK on success, or an error code.
 */
GV_ConditionalResult gv_cond_update_vector(GV_CondManager *mgr, size_t index,
                                            const float *new_data, size_t dimension,
                                            const GV_Condition *conditions,
                                            size_t condition_count);

/**
 * @brief Conditionally update a single metadata key-value pair.
 *
 * Acquires the database write lock, evaluates all conditions, and if all
 * pass, sets the metadata key to the given value and increments the version.
 *
 * @param mgr Manager instance; must be non-NULL.
 * @param index Vector index in SoA storage.
 * @param key Metadata key to set; must be non-NULL.
 * @param value Metadata value to set; must be non-NULL.
 * @param conditions Array of conditions to evaluate (may be NULL if count is 0).
 * @param condition_count Number of conditions.
 * @return GV_COND_OK on success, or an error code.
 */
GV_ConditionalResult gv_cond_update_metadata(GV_CondManager *mgr, size_t index,
                                              const char *key, const char *value,
                                              const GV_Condition *conditions,
                                              size_t condition_count);

/**
 * @brief Conditionally delete a vector.
 *
 * Acquires the database write lock, evaluates all conditions, and if all
 * pass, marks the vector as deleted and increments the version.
 *
 * @param mgr Manager instance; must be non-NULL.
 * @param index Vector index in SoA storage.
 * @param conditions Array of conditions to evaluate (may be NULL if count is 0).
 * @param condition_count Number of conditions.
 * @return GV_COND_OK on success, or an error code.
 */
GV_ConditionalResult gv_cond_delete(GV_CondManager *mgr, size_t index,
                                     const GV_Condition *conditions,
                                     size_t condition_count);

/* ============================================================================
 * Version Queries
 * ============================================================================ */

/**
 * @brief Get the current tracked version of a vector.
 *
 * @param mgr Manager instance; must be non-NULL.
 * @param index Vector index.
 * @return Current version, or 0 if the index is not tracked.
 */
uint64_t gv_cond_get_version(const GV_CondManager *mgr, size_t index);

/* ============================================================================
 * Batch Operations
 * ============================================================================ */

/**
 * @brief Conditionally update multiple vectors in a single pass.
 *
 * Each vector is evaluated and updated independently under the database
 * write lock. Per-vector results are written to the results array.
 *
 * @param mgr Manager instance; must be non-NULL.
 * @param indices Array of vector indices (batch_size elements).
 * @param vectors Array of pointers to replacement data (batch_size elements).
 * @param conditions Array of per-vector condition arrays (batch_size elements).
 * @param condition_counts Array of per-vector condition counts (batch_size elements).
 * @param batch_size Number of vectors in the batch.
 * @param results Output array for per-vector results (batch_size elements).
 * @return Number of vectors successfully updated, or -1 on invalid arguments.
 */
int gv_cond_batch_update(GV_CondManager *mgr,
                          const size_t *indices,
                          const float **vectors,
                          const GV_Condition **conditions,
                          const size_t *condition_counts,
                          size_t batch_size,
                          GV_ConditionalResult *results);

/* ============================================================================
 * Convenience Wrappers
 * ============================================================================ */

/**
 * @brief Conditionally migrate a vector's embedding to a new model output.
 *
 * Convenience wrapper around gv_cond_update_vector that checks the vector's
 * tracked version equals expected_version before replacing the embedding data.
 * Useful for safe embedding model migrations where stale writes must be rejected.
 *
 * @param mgr Manager instance; must be non-NULL.
 * @param index Vector index in SoA storage.
 * @param new_embedding Replacement embedding data (dimension floats).
 * @param dimension Number of floats in new_embedding; must equal db->dimension.
 * @param expected_version Version that the vector must currently be at.
 * @return GV_COND_OK on success, or an error code.
 */
GV_ConditionalResult gv_cond_migrate_embedding(GV_CondManager *mgr, size_t index,
                                                const float *new_embedding,
                                                size_t dimension,
                                                uint64_t expected_version);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_CONDITIONAL_H */
