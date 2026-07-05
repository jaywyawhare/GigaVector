#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include "index/ivf_retrain.h"
#include "index/ivfflat.h"
#include "index/ivfpq.h"
#include "index/ivfsq8.h"
#include "index/ivfturboquant.h"
#include "storage/database.h"

/* ------------------------------------------------------------------ */
/* Internal helpers                                                     */
/* ------------------------------------------------------------------ */

/**
 * Compute current inertia for the IVF index stored in db->hnsw_index.
 * Returns -1.0f if the index type is not a supported IVF variant.
 */
static float db_compute_ivf_inertia(GV_Database *db) {
    if (!db) return -1.0f;
    switch (db->index_type) {
        case GV_INDEX_TYPE_IVFFLAT:
            return ivfflat_compute_inertia(db->hnsw_index);
        /* Other IVF types: return 0.0f to indicate "no drift data available"
         * which means no automatic retrain is triggered for those types via
         * the drift path (they can still be triggered manually). */
        default:
            return -1.0f;
    }
}

/**
 * Run a synchronous retrain on the IVF index using the configured iters.
 * Caller must NOT hold db->rwlock when calling this.
 */
static int db_do_ivf_retrain(GV_Database *db) {
    if (!db) return -1;

    /* Use the database read lock to protect the index during retrain setup,
     * then upgrade to write lock for the centroid swap. For IVFFLAT the
     * retrain is fully in-place under the write lock. */
    pthread_rwlock_wrlock(&db->rwlock);

    int rc = -1;
    switch (db->index_type) {
        case GV_INDEX_TYPE_IVFFLAT: {
            /* Default to 10 iterations; can be made configurable later */
            rc = ivfflat_retrain(db->hnsw_index, 10);
            if (rc == 0) {
                /* Record new inertia as the post-retrain baseline */
                float new_inertia = ivfflat_compute_inertia(db->hnsw_index);
                pthread_mutex_lock(&db->retrain_mutex);
                db->initial_inertia = (new_inertia > 0.0f) ? new_inertia : db->initial_inertia;
                db->inserts_since_retrain = 0;
                db->last_retrain_drift = 1.0f;
                pthread_mutex_unlock(&db->retrain_mutex);
            }
            break;
        }
        default:
            rc = -1;
            break;
    }

    pthread_rwlock_unlock(&db->rwlock);
    return rc;
}

/* ------------------------------------------------------------------ */
/* Background retrain thread                                            */
/* ------------------------------------------------------------------ */

static void *retrain_thread_fn(void *arg) {
    GV_Database *db = (GV_Database *)arg;
    db_do_ivf_retrain(db);

    pthread_mutex_lock(&db->retrain_mutex);
    db->retrain_running = 0;
    pthread_mutex_unlock(&db->retrain_mutex);

    return NULL;
}

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

float ivf_retrain_check_drift(GV_Database *db) {
    if (!db) return -1.0f;

    float current_inertia = db_compute_ivf_inertia(db);
    if (current_inertia < 0.0f) return -1.0f;

    pthread_mutex_lock(&db->retrain_mutex);
    float initial = db->initial_inertia;
    pthread_mutex_unlock(&db->retrain_mutex);

    if (initial <= 0.0f) {
        /* Record initial inertia on first call */
        pthread_mutex_lock(&db->retrain_mutex);
        db->initial_inertia = current_inertia;
        pthread_mutex_unlock(&db->retrain_mutex);
        return 1.0f; /* drift ratio = 1.0 (no drift yet) */
    }

    float drift = current_inertia / initial;

    pthread_mutex_lock(&db->retrain_mutex);
    db->last_retrain_drift = drift;
    pthread_mutex_unlock(&db->retrain_mutex);

    return drift;
}

int ivf_retrain_trigger(GV_Database *db) {
    if (!db) return -1;

    pthread_mutex_lock(&db->retrain_mutex);
    if (db->retrain_running) {
        pthread_mutex_unlock(&db->retrain_mutex);
        return 0; /* already in progress */
    }
    db->retrain_running = 1;
    pthread_mutex_unlock(&db->retrain_mutex);

    int rc = pthread_create(&db->retrain_thread, NULL, retrain_thread_fn, db);
    if (rc != 0) {
        pthread_mutex_lock(&db->retrain_mutex);
        db->retrain_running = 0;
        pthread_mutex_unlock(&db->retrain_mutex);
        return -1;
    }
    return 0;
}

void ivf_retrain_stop(GV_Database *db) {
    if (!db) return;

    pthread_mutex_lock(&db->retrain_mutex);
    int running = db->retrain_running;
    pthread_mutex_unlock(&db->retrain_mutex);

    if (running) {
        pthread_join(db->retrain_thread, NULL);
        pthread_mutex_lock(&db->retrain_mutex);
        db->retrain_running = 0;
        pthread_mutex_unlock(&db->retrain_mutex);
    }
}

/* ------------------------------------------------------------------ */
/* User-facing wrapper API                                              */
/* ------------------------------------------------------------------ */

void gv_db_set_retrain_config(GV_Database *db, float drift_threshold,
                               size_t min_new_vectors) {
    if (!db) return;
    pthread_mutex_lock(&db->retrain_mutex);
    db->retrain_enabled = 1;
    db->retrain_drift_threshold = drift_threshold;
    db->retrain_min_new_vectors = min_new_vectors;
    pthread_mutex_unlock(&db->retrain_mutex);
}

int gv_db_trigger_retrain(GV_Database *db) {
    if (!db) return -1;

    pthread_mutex_lock(&db->retrain_mutex);
    int running = db->retrain_running;
    pthread_mutex_unlock(&db->retrain_mutex);

    if (running) {
        /* Wait for existing retrain to complete, then run synchronously */
        ivf_retrain_stop(db);
    }

    /* Run synchronously on the calling thread */
    return db_do_ivf_retrain(db);
}

void gv_db_retrain_status(const GV_Database *db, int *is_running,
                           float *last_drift) {
    if (!db) return;

    /* Cast away const for mutex — the mutex is logically const here */
    GV_Database *mdb = (GV_Database *)(uintptr_t)db;
    pthread_mutex_lock(&mdb->retrain_mutex);
    if (is_running) *is_running = db->retrain_running;
    if (last_drift) *last_drift = db->last_retrain_drift;
    pthread_mutex_unlock(&mdb->retrain_mutex);
}
