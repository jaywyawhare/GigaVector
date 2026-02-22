/**
 * @file gv_vacuum.c
 * @brief Async compaction/vacuum: background defragmentation, space reclamation,
 *        and fragmentation metrics.
 */

#include "gigavector/gv_vacuum.h"
#include "gigavector/gv_database.h"

#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>

/*  Internal Structures  */

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

/*  Time Helpers  */

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

/*  Fragmentation Analysis  */

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

/*  Core Vacuum Logic  */

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

    /* ------- pre-check under read lock ------- */
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

    /* ------- update state ------- */
    pthread_mutex_lock(&mgr->mutex);
    mgr->stats.state = GV_VACUUM_RUNNING;
    mgr->stats.started_at = vacuum_get_epoch_ms();
    mgr->stats.fragmentation_before = frag_before;
    pthread_mutex_unlock(&mgr->mutex);

    uint64_t t_start = vacuum_get_time_ms();

    /* ------- acquire write lock and compact ------- */
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

    /* Allocate compacted arrays */
    float *new_data = (float *)malloc(new_count * dim * sizeof(float));
    GV_Metadata **new_metadata = (GV_Metadata **)calloc(new_count, sizeof(GV_Metadata *));
    int *new_deleted = (int *)calloc(new_count, sizeof(int));

    if (new_data == NULL || new_metadata == NULL || new_deleted == NULL) {
        free(new_data);
        free(new_metadata);
        free(new_deleted);
        pthread_rwlock_unlock(&db->rwlock);

        pthread_mutex_lock(&mgr->mutex);
        mgr->stats.state = GV_VACUUM_FAILED;
        mgr->stats.completed_at = vacuum_get_epoch_ms();
        mgr->stats.duration_ms = vacuum_get_time_ms() - t_start;
        mgr->stats.total_runs++;
        pthread_mutex_unlock(&mgr->mutex);
        return -1;
    }

    /*
     * Batch-process: copy active vectors into compacted arrays.
     * For low-priority, we release the write lock and yield between batches
     * to avoid starving readers.  However, since we are building new arrays
     * and have not swapped them yet, we can safely drop the lock only BEFORE
     * the swap.  To keep the implementation simple and correct, we do the
     * full scan under the write lock but insert micro-yields via usleep.
     */
    size_t batch_size = mgr->config.batch_size;
    int low_priority = (mgr->config.priority == 0);
    size_t vectors_compacted = 0;
    size_t new_idx = 0;
    size_t batch_counter = 0;

    for (size_t old_idx = 0; old_idx < old_count; ++old_idx) {
        if (storage->deleted[old_idx] == 0) {
            /* Copy vector data */
            memcpy(new_data + (new_idx * dim),
                   storage->data + (old_idx * dim),
                   dim * sizeof(float));
            new_metadata[new_idx] = storage->metadata[old_idx];
            storage->metadata[old_idx] = NULL; /* Transfer ownership */
            new_deleted[new_idx] = 0;
            new_idx++;
            vectors_compacted++;
        } else {
            /* Free metadata for deleted vectors */
            GV_Metadata *md = storage->metadata[old_idx];
            while (md != NULL) {
                GV_Metadata *next = md->next;
                free(md->key);
                free(md->value);
                free(md);
                md = next;
            }
            storage->metadata[old_idx] = NULL;
        }

        batch_counter++;
        if (batch_counter >= batch_size && low_priority) {
            batch_counter = 0;
            /* Yield to allow other threads to proceed */
            usleep(100); /* 100 microseconds */
        }
    }

    /* Swap arrays */
    free(storage->data);
    free(storage->metadata);
    free(storage->deleted);

    storage->data = new_data;
    storage->metadata = new_metadata;
    storage->deleted = new_deleted;
    storage->count = new_count;
    storage->capacity = new_count;

    /* Update database vector count */
    db->count = new_count;

    /* Rebuild index structures to reflect new positions.
     * Use gv_db_compact() approach: rebuild KD-tree or HNSW from scratch.
     * Since we already hold the write lock and have compacted storage,
     * we rebuild the primary index in-place. */
    if (db->index_type == GV_INDEX_TYPE_KDTREE && db->root != NULL) {
        gv_kdtree_destroy_recursive(db->root);
        db->root = NULL;
        for (size_t i = 0; i < new_count; ++i) {
            gv_kdtree_insert(&db->root, storage, i, 0);
        }
    }
    /* For HNSW, IVFPQ, and other index types the caller should trigger a
     * separate index rebuild or use gv_db_compact() which already handles
     * those cases.  The vacuum module focuses on SoA storage defragmentation. */

    double frag_after = vacuum_compute_fragmentation(db);

    pthread_rwlock_unlock(&db->rwlock);

    /* ------- update stats ------- */
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

/*  Background Thread  */

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

/*  Public API  */

void gv_vacuum_config_init(GV_VacuumConfig *config) {
    if (config == NULL) {
        return;
    }
    config->min_deleted_count = 100;
    config->min_fragmentation_ratio = 0.1;
    config->batch_size = 1000;
    config->priority = 0;
    config->interval_sec = 600;
}

GV_VacuumManager *gv_vacuum_create(GV_Database *db, const GV_VacuumConfig *config) {
    if (db == NULL) {
        return NULL;
    }

    GV_VacuumManager *mgr = (GV_VacuumManager *)calloc(1, sizeof(GV_VacuumManager));
    if (mgr == NULL) {
        return NULL;
    }

    mgr->db = db;

    if (config != NULL) {
        mgr->config = *config;
    } else {
        gv_vacuum_config_init(&mgr->config);
    }

    /* Clamp batch_size to a reasonable minimum */
    if (mgr->config.batch_size == 0) {
        mgr->config.batch_size = 1000;
    }

    memset(&mgr->stats, 0, sizeof(mgr->stats));
    mgr->stats.state = GV_VACUUM_IDLE;

    mgr->thread_running = 0;

    if (pthread_mutex_init(&mgr->mutex, NULL) != 0) {
        free(mgr);
        return NULL;
    }

    if (pthread_cond_init(&mgr->cond, NULL) != 0) {
        pthread_mutex_destroy(&mgr->mutex);
        free(mgr);
        return NULL;
    }

    return mgr;
}

void gv_vacuum_destroy(GV_VacuumManager *mgr) {
    if (mgr == NULL) {
        return;
    }

    /* Stop auto-vacuum if running */
    gv_vacuum_stop_auto(mgr);

    pthread_cond_destroy(&mgr->cond);
    pthread_mutex_destroy(&mgr->mutex);
    free(mgr);
}

int gv_vacuum_run(GV_VacuumManager *mgr) {
    if (mgr == NULL || mgr->db == NULL) {
        return -1;
    }

    return vacuum_run_internal(mgr);
}

int gv_vacuum_start_auto(GV_VacuumManager *mgr) {
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

int gv_vacuum_stop_auto(GV_VacuumManager *mgr) {
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

double gv_vacuum_get_fragmentation(const GV_VacuumManager *mgr) {
    if (mgr == NULL || mgr->db == NULL) {
        return 0.0;
    }

    /* Acquire read lock to compute fragmentation safely */
    pthread_rwlock_rdlock(&mgr->db->rwlock);
    double frag = vacuum_compute_fragmentation(mgr->db);
    pthread_rwlock_unlock(&mgr->db->rwlock);

    return frag;
}

int gv_vacuum_get_stats(const GV_VacuumManager *mgr, GV_VacuumStats *stats) {
    if (mgr == NULL || stats == NULL) {
        return -1;
    }

    pthread_mutex_lock((pthread_mutex_t *)&mgr->mutex);
    *stats = mgr->stats;
    pthread_mutex_unlock((pthread_mutex_t *)&mgr->mutex);

    return 0;
}

GV_VacuumState gv_vacuum_get_state(const GV_VacuumManager *mgr) {
    if (mgr == NULL) {
        return GV_VACUUM_IDLE;
    }

    pthread_mutex_lock((pthread_mutex_t *)&mgr->mutex);
    GV_VacuumState state = mgr->stats.state;
    pthread_mutex_unlock((pthread_mutex_t *)&mgr->mutex);

    return state;
}
