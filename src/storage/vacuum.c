/**
 * @file vacuum.c
 * @brief Async compaction/vacuum: background defragmentation, space reclamation,
 *        and fragmentation metrics.
 */

#include "storage/vacuum.h"
#include "storage/database.h"
#include "core/memory.h"

#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <time.h>
#include <errno.h>
#include "core/compat.h"

/* Internal Structures */

struct GV_VacuumManager {
    GV_Database *db;
    GV_VacuumConfig config;
    GV_VacuumStats stats;

    /* Background thread */
    pthread_t thread;
    int thread_running;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
};

/* Time Helpers */

static uint64_t vacuum_get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;
}

static uint64_t vacuum_get_epoch_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;
}

/* Fragmentation Analysis */

/**
 * @brief Count deleted vectors in the SoA storage.
 *
 * Caller must hold at least a read lock on db->rwlock.
 */
static size_t vacuum_count_deleted(const GV_Database *db) {
    if (db == NULL || db->soa_storage == NULL) {
        return 0;
    }

    GV_SoAStorage *storage = db->soa_storage;
    size_t deleted = 0;

    for (size_t i = 0; i < storage->count; ++i) {
        if (storage->deleted[i] != 0) {
            deleted++;
        }
    }
    return deleted;
}

/**
 * @brief Compute fragmentation ratio.
 *
 * fragmentation = deleted / (deleted + active)
 * where (deleted + active) == storage->count.
 *
 * Caller must hold at least a read lock on db->rwlock.
 */
static double vacuum_compute_fragmentation(const GV_Database *db) {
    if (db == NULL || db->soa_storage == NULL || db->soa_storage->count == 0) {
        return 0.0;
    }

    size_t deleted = vacuum_count_deleted(db);
    return (double)deleted / (double)db->soa_storage->count;
}

/* Core Vacuum Logic */

/**
 * @brief Perform a single vacuum pass with batch processing.
 *
 * Scans SoA storage, identifies deleted slots, compacts by moving active
 * vectors to fill gaps.  Processes batch_size vectors at a time and yields
 * between batches when priority == 0 (low).
 *
 * Caller must NOT hold db->rwlock -- the function acquires it internally.
 *
 * @return 0 on success, -1 on error.
 */
static int vacuum_run_internal(GV_VacuumManager *mgr) {
    GV_Database *db = mgr->db;

    /* pre-check under read lock */
    pthread_rwlock_rdlock(&db->rwlock);
    double frag_before = vacuum_compute_fragmentation(db);
    size_t deleted_count = vacuum_count_deleted(db);
    pthread_rwlock_unlock(&db->rwlock);

    /* Check thresholds */
    if (deleted_count < mgr->config.min_deleted_count &&
        frag_before < mgr->config.min_fragmentation_ratio) {
        /* Nothing to do -- still counts as success */
        return 0;
    }

    /* update state */
    pthread_mutex_lock(&mgr->mutex);
    mgr->stats.state = GV_VACUUM_RUNNING;
    mgr->stats.started_at = vacuum_get_epoch_ms();
    mgr->stats.fragmentation_before = frag_before;
    pthread_mutex_unlock(&mgr->mutex);

    uint64_t t_start = vacuum_get_time_ms();

    /* acquire write lock and compact */
    pthread_rwlock_wrlock(&db->rwlock);

    GV_SoAStorage *storage = db->soa_storage;
    if (storage == NULL || storage->count == 0) {
        pthread_rwlock_unlock(&db->rwlock);
        pthread_mutex_lock(&mgr->mutex);
        mgr->stats.state = GV_VACUUM_COMPLETED;
        mgr->stats.completed_at = vacuum_get_epoch_ms();
        mgr->stats.duration_ms = vacuum_get_time_ms() - t_start;
        mgr->stats.total_runs++;
        pthread_mutex_unlock(&mgr->mutex);
        return 0;
    }

    /* Re-count under write lock (authoritative) */
    size_t dim = storage->dimension;
    deleted_count = 0;
    for (size_t i = 0; i < storage->count; ++i) {
        if (storage->deleted[i] != 0) {
            deleted_count++;
        }
    }

    if (deleted_count == 0) {
        pthread_rwlock_unlock(&db->rwlock);
        pthread_mutex_lock(&mgr->mutex);
        mgr->stats.state = GV_VACUUM_COMPLETED;
        mgr->stats.completed_at = vacuum_get_epoch_ms();
        mgr->stats.duration_ms = vacuum_get_time_ms() - t_start;
        mgr->stats.fragmentation_after = 0.0;
        mgr->stats.total_runs++;
        pthread_mutex_unlock(&mgr->mutex);
        return 0;
    }

    size_t old_count = storage->count;
    size_t new_count = old_count - deleted_count;
    size_t bytes_reclaimed = deleted_count * dim * sizeof(float);
    size_t vectors_compacted = new_count;

    /*
     * Delegate the actual renumber + full index/metadata rebuild to the single
     * correct implementation in database.c. Its own partial per-index-type
     * logic previously left HNSW / IVF / FLAT / metadata_index stale after a
     * vacuum;
     * db_compact_soa_storage_locked() rebuilds the primary index and the
     * metadata_index consistently with the renumbered SoA indices.
     *
     * We already hold the write lock, matching that routine's contract, so we
     * do NOT double-lock here.
     */
    if (db_compact_soa_storage_locked(db) != 0) {
        pthread_rwlock_unlock(&db->rwlock);

        pthread_mutex_lock(&mgr->mutex);
        mgr->stats.state = GV_VACUUM_FAILED;
        mgr->stats.completed_at = vacuum_get_epoch_ms();
        mgr->stats.duration_ms = vacuum_get_time_ms() - t_start;
        mgr->stats.total_runs++;
        pthread_mutex_unlock(&mgr->mutex);
        return -1;
    }

    /* Keep the database's public vector count in sync with the compacted SoA. */
    db->count = new_count;

    double frag_after = vacuum_compute_fragmentation(db);

    pthread_rwlock_unlock(&db->rwlock);

    /* update stats */
    uint64_t t_end = vacuum_get_time_ms();

    pthread_mutex_lock(&mgr->mutex);
    mgr->stats.state = GV_VACUUM_COMPLETED;
    mgr->stats.vectors_compacted += vectors_compacted;
    mgr->stats.bytes_reclaimed += bytes_reclaimed;
    mgr->stats.fragmentation_after = frag_after;
    mgr->stats.completed_at = vacuum_get_epoch_ms();
    mgr->stats.duration_ms = t_end - t_start;
    mgr->stats.total_runs++;
    pthread_mutex_unlock(&mgr->mutex);

    return 0;
}

/* Background Thread */

static void *vacuum_thread_func(void *arg) {
    GV_VacuumManager *mgr = (GV_VacuumManager *)arg;

    pthread_mutex_lock(&mgr->mutex);

    while (mgr->thread_running) {
        struct timespec timeout;
        clock_gettime(CLOCK_REALTIME, &timeout);
        timeout.tv_sec += (time_t)mgr->config.interval_sec;

        int wait_result = pthread_cond_timedwait(&mgr->cond, &mgr->mutex,
                                                  &timeout);

        if (!mgr->thread_running) {
            break;
        }

        if (wait_result == ETIMEDOUT || wait_result == 0) {
            /* Check fragmentation before running */
            pthread_mutex_unlock(&mgr->mutex);

            pthread_rwlock_rdlock(&mgr->db->rwlock);
            double frag = vacuum_compute_fragmentation(mgr->db);
            size_t deleted = vacuum_count_deleted(mgr->db);
            pthread_rwlock_unlock(&mgr->db->rwlock);

            if (deleted >= mgr->config.min_deleted_count ||
                frag >= mgr->config.min_fragmentation_ratio) {
                vacuum_run_internal(mgr);
            }

            pthread_mutex_lock(&mgr->mutex);
        }
    }

    pthread_mutex_unlock(&mgr->mutex);
    return NULL;
}

/* Public API */

void vacuum_config_init(GV_VacuumConfig *config) {
    if (config == NULL) {
        return;
    }
    config->min_deleted_count = 100;
    config->min_fragmentation_ratio = 0.1;
    config->batch_size = 1000;
    config->priority = 0;
    config->interval_sec = 600;
}

GV_VacuumManager *vacuum_create(GV_Database *db, const GV_VacuumConfig *config) {
    if (db == NULL) {
        return NULL;
    }

    GV_VacuumManager *mgr = (GV_VacuumManager *)gv_calloc(1, sizeof(GV_VacuumManager));
    if (mgr == NULL) {
        return NULL;
    }

    mgr->db = db;

    if (config != NULL) {
        mgr->config = *config;
    } else {
        vacuum_config_init(&mgr->config);
    }

    /* Clamp batch_size to a reasonable minimum */
    if (mgr->config.batch_size == 0) {
        mgr->config.batch_size = 1000;
    }

    memset(&mgr->stats, 0, sizeof(mgr->stats));
    mgr->stats.state = GV_VACUUM_IDLE;

    mgr->thread_running = 0;

    if (pthread_mutex_init(&mgr->mutex, NULL) != 0) {
        gv_free(mgr);
        return NULL;
    }

    if (pthread_cond_init(&mgr->cond, NULL) != 0) {
        pthread_mutex_destroy(&mgr->mutex);
        gv_free(mgr);
        return NULL;
    }

    return mgr;
}

void vacuum_destroy(GV_VacuumManager *mgr) {
    if (mgr == NULL) {
        return;
    }

    /* Stop auto-vacuum if running */
    vacuum_stop_auto(mgr);

    pthread_cond_destroy(&mgr->cond);
    pthread_mutex_destroy(&mgr->mutex);
    gv_free(mgr);
}

int vacuum_run(GV_VacuumManager *mgr) {
    if (mgr == NULL || mgr->db == NULL) {
        return -1;
    }

    return vacuum_run_internal(mgr);
}

int vacuum_start_auto(GV_VacuumManager *mgr) {
    if (mgr == NULL) {
        return -1;
    }

    pthread_mutex_lock(&mgr->mutex);

    if (mgr->thread_running) {
        pthread_mutex_unlock(&mgr->mutex);
        return 0; /* Already running */
    }

    mgr->thread_running = 1;

    int rc = pthread_create(&mgr->thread, NULL, vacuum_thread_func, mgr);
    if (rc != 0) {
        mgr->thread_running = 0;
        pthread_mutex_unlock(&mgr->mutex);
        return -1;
    }

    pthread_mutex_unlock(&mgr->mutex);
    return 0;
}

int vacuum_stop_auto(GV_VacuumManager *mgr) {
    if (mgr == NULL) {
        return -1;
    }

    pthread_mutex_lock(&mgr->mutex);

    if (!mgr->thread_running) {
        pthread_mutex_unlock(&mgr->mutex);
        return 0; /* Not running */
    }

    mgr->thread_running = 0;
    pthread_cond_signal(&mgr->cond);
    pthread_mutex_unlock(&mgr->mutex);

    /* Wait for the thread to finish */
    pthread_join(mgr->thread, NULL);
    return 0;
}

double vacuum_get_fragmentation(const GV_VacuumManager *mgr) {
    if (mgr == NULL || mgr->db == NULL) {
        return 0.0;
    }

    /* Acquire read lock to compute fragmentation safely */
    pthread_rwlock_rdlock(&mgr->db->rwlock);
    double frag = vacuum_compute_fragmentation(mgr->db);
    pthread_rwlock_unlock(&mgr->db->rwlock);

    return frag;
}

int vacuum_get_stats(const GV_VacuumManager *mgr, GV_VacuumStats *stats) {
    if (mgr == NULL || stats == NULL) {
        return -1;
    }

    pthread_mutex_lock((pthread_mutex_t *)&mgr->mutex);
    *stats = mgr->stats;
    pthread_mutex_unlock((pthread_mutex_t *)&mgr->mutex);

    return 0;
}

GV_VacuumState vacuum_get_state(const GV_VacuumManager *mgr) {
    if (mgr == NULL) {
        return GV_VACUUM_IDLE;
    }

    pthread_mutex_lock((pthread_mutex_t *)&mgr->mutex);
    GV_VacuumState state = mgr->stats.state;
    pthread_mutex_unlock((pthread_mutex_t *)&mgr->mutex);

    return state;
}
