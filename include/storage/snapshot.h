#ifndef GIGAVECTOR_GV_SNAPSHOT_H
#define GIGAVECTOR_GV_SNAPSHOT_H
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t snapshot_id;
    uint64_t timestamp_us;      /* creation time */
    size_t vector_count;        /* count at snapshot time */
    char label[64];             /* user label */
} GV_SnapshotInfo;

typedef struct GV_SnapshotManager GV_SnapshotManager;
typedef struct GV_Snapshot GV_Snapshot;

GV_SnapshotManager *snapshot_manager_create(size_t max_snapshots);
/**
 * @brief Destroy an instance and free associated resources.
 *
 * @param mgr Manager instance.
 */
void snapshot_manager_destroy(GV_SnapshotManager *mgr);

uint64_t snapshot_create(GV_SnapshotManager *mgr, size_t vector_count,
                            const float *vector_data, size_t dimension,
                            const char *label);

GV_Snapshot *snapshot_open(GV_SnapshotManager *mgr, uint64_t snapshot_id);
/**
 * @brief Perform the operation.
 *
 * @param snap snap.
 */
void snapshot_close(GV_Snapshot *snap);

/**
 * @brief Return the number of stored items.
 *
 * @param snap snap.
 * @return Count value.
 */
size_t snapshot_count(const GV_Snapshot *snap);
const float *snapshot_get_vector(const GV_Snapshot *snap, size_t index);
/**
 * @brief Perform the operation.
 *
 * @param snap snap.
 * @return Count value.
 */
size_t snapshot_dimension(const GV_Snapshot *snap);

/**
 * @brief List items.
 *
 * @param mgr Manager instance.
 * @param infos infos.
 * @param max_infos max_infos.
 * @return 0 on success, -1 on error.
 */
int snapshot_list(const GV_SnapshotManager *mgr, GV_SnapshotInfo *infos, size_t max_infos);
/**
 * @brief Delete an item.
 *
 * @param mgr Manager instance.
 * @param snapshot_id Identifier.
 * @return 0 on success, -1 on error.
 */
int snapshot_delete(GV_SnapshotManager *mgr, uint64_t snapshot_id);
/**
 * @brief Save state to a file.
 *
 * @param mgr Manager instance.
 * @param out Output buffer.
 * @return 0 on success, -1 on error.
 */
int snapshot_save(const GV_SnapshotManager *mgr, FILE *out);
/**
 * @brief Load state from a file.
 *
 * @param mgr_ptr mgr_ptr.
 * @param in Input file stream.
 * @return 0 on success, -1 on error.
 */
int snapshot_load(GV_SnapshotManager **mgr_ptr, FILE *in);

#ifdef __cplusplus
}
#endif
#endif
