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

/* Manager manages all snapshots for a database */
GV_SnapshotManager *gv_snapshot_manager_create(size_t max_snapshots);
void gv_snapshot_manager_destroy(GV_SnapshotManager *mgr);

/* Create a snapshot capturing current DB state */
uint64_t gv_snapshot_create(GV_SnapshotManager *mgr, size_t vector_count,
                            const float *vector_data, size_t dimension,
                            const char *label);

/* Open a read-only snapshot for queries */
GV_Snapshot *gv_snapshot_open(GV_SnapshotManager *mgr, uint64_t snapshot_id);
void gv_snapshot_close(GV_Snapshot *snap);

/* Query snapshot data */
size_t gv_snapshot_count(const GV_Snapshot *snap);
const float *gv_snapshot_get_vector(const GV_Snapshot *snap, size_t index);
size_t gv_snapshot_dimension(const GV_Snapshot *snap);

/* Management */
int gv_snapshot_list(const GV_SnapshotManager *mgr, GV_SnapshotInfo *infos, size_t max_infos);
int gv_snapshot_delete(GV_SnapshotManager *mgr, uint64_t snapshot_id);
int gv_snapshot_save(const GV_SnapshotManager *mgr, FILE *out);
int gv_snapshot_load(GV_SnapshotManager **mgr_ptr, FILE *in);

#ifdef __cplusplus
}
#endif
#endif
