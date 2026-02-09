#ifndef GIGAVECTOR_GV_VACUUM_H
#define GIGAVECTOR_GV_VACUUM_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct GV_Database GV_Database;

typedef enum {
    GV_VACUUM_IDLE = 0,
    GV_VACUUM_RUNNING = 1,
    GV_VACUUM_COMPLETED = 2,
    GV_VACUUM_FAILED = 3
} GV_VacuumState;

typedef struct {
    size_t min_deleted_count;        /* Min deleted vectors to trigger vacuum (default: 100) */
    double min_fragmentation_ratio;  /* Min fragmentation ratio to trigger (default: 0.1) */
    size_t batch_size;               /* Vectors to process per batch (default: 1000) */
    int priority;                    /* 0=low (yield often), 1=normal, 2=high (default: 0) */
    size_t interval_sec;             /* Auto-vacuum interval seconds (default: 600) */
} GV_VacuumConfig;

typedef struct {
    GV_VacuumState state;
    size_t vectors_compacted;
    size_t bytes_reclaimed;
    double fragmentation_before;
    double fragmentation_after;
    uint64_t started_at;
    uint64_t completed_at;
    uint64_t duration_ms;
    size_t total_runs;
} GV_VacuumStats;

typedef struct GV_VacuumManager GV_VacuumManager;

void gv_vacuum_config_init(GV_VacuumConfig *config);
GV_VacuumManager *gv_vacuum_create(GV_Database *db, const GV_VacuumConfig *config);
void gv_vacuum_destroy(GV_VacuumManager *mgr);

/* Manual vacuum */
int gv_vacuum_run(GV_VacuumManager *mgr);

/* Background auto-vacuum */
int gv_vacuum_start_auto(GV_VacuumManager *mgr);
int gv_vacuum_stop_auto(GV_VacuumManager *mgr);

/* Analyze fragmentation without vacuuming */
double gv_vacuum_get_fragmentation(const GV_VacuumManager *mgr);

/* Stats */
int gv_vacuum_get_stats(const GV_VacuumManager *mgr, GV_VacuumStats *stats);
GV_VacuumState gv_vacuum_get_state(const GV_VacuumManager *mgr);

#ifdef __cplusplus
}
#endif
#endif
