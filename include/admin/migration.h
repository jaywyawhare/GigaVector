#ifndef GIGAVECTOR_GV_MIGRATION_H
#define GIGAVECTOR_GV_MIGRATION_H
#include <stddef.h>
#include <stdint.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GV_MIGRATION_PENDING = 0,
    GV_MIGRATION_RUNNING = 1,
    GV_MIGRATION_COMPLETED = 2,
    GV_MIGRATION_FAILED = 3,
    GV_MIGRATION_CANCELLED = 4
} GV_MigrationStatus;

typedef struct {
    GV_MigrationStatus status;
    double progress;          /* 0.0 to 1.0 */
    size_t vectors_migrated;
    size_t total_vectors;
    uint64_t start_time_us;
    uint64_t elapsed_us;
    char error_message[256];
} GV_MigrationInfo;

typedef struct GV_Migration GV_Migration;

/* Start migration: builds new index type in background from existing vector data.
 * source_data: pointer to contiguous vector data (count * dimension floats)
 * count: number of vectors
 * dimension: vector dimension
 * new_index_type: target index type (as int, maps to GV_IndexType)
 * new_index_config: opaque pointer to config struct for the new index type (can be NULL for defaults)
 */
GV_Migration *migration_start(const float *source_data, size_t count,
                                  size_t dimension, int new_index_type,
                                  const void *new_index_config);

/**
 * @brief Retrieve information.
 *
 * @param mig mig.
 * @param info Output information structure.
 * @return 0 on success, -1 on error.
 */
int migration_get_info(const GV_Migration *mig, GV_MigrationInfo *info);
/**
 * @brief Perform the operation.
 *
 * @param mig mig.
 * @return 0 on success, -1 on error.
 */
int migration_wait(GV_Migration *mig);
/**
 * @brief Perform the operation.
 *
 * @param mig mig.
 * @return 0 on success, -1 on error.
 */
int migration_cancel(GV_Migration *mig);

/* Only valid after COMPLETED status. Caller takes ownership of the returned index. */
void *migration_take_index(GV_Migration *mig);

/**
 * @brief Destroy an instance and free associated resources.
 *
 * @param mig mig.
 */
void migration_destroy(GV_Migration *mig);

#ifdef __cplusplus
}
#endif
#endif
