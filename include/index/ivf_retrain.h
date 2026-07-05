#ifndef GIGAVECTOR_INDEX_IVF_RETRAIN_H
#define GIGAVECTOR_INDEX_IVF_RETRAIN_H

#include <stddef.h>
#include "storage/database.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute the current drift of the IVF index relative to its training inertia.
 *
 * Returns the mean squared distance of live vectors to their assigned centroids
 * divided by the stored initial_inertia.  A value > 1.0 + retrain_drift_threshold
 * indicates significant drift.  Returns -1.0f if the database has no IVF index or
 * is not yet trained.
 *
 * @param db Database instance; must be non-NULL.
 * @return Drift ratio, or -1.0f on error.
 */
float ivf_retrain_check_drift(GV_Database *db);

/**
 * @brief Spawn a background thread to retrain the IVF index.
 *
 * No-op if a retrain is already in progress or if the database has no
 * trainable IVF index.  Sets db->retrain_running = 1 before returning.
 *
 * @param db Database instance; must be non-NULL.
 * @return 0 on success (thread spawned), -1 on error.
 */
int ivf_retrain_trigger(GV_Database *db);

/**
 * @brief Join the background retrain thread (called from db_close).
 *
 * Waits for the background thread to finish and resets retrain_running to 0.
 * Safe to call when no retrain is running.
 *
 * @param db Database instance; must be non-NULL.
 */
void ivf_retrain_stop(GV_Database *db);

/* ------------------------------------------------------------------ */
/* Public API — also declared in gigavector.h via this header          */
/* ------------------------------------------------------------------ */

/**
 * @brief Configure automatic IVF retraining for a database.
 *
 * Retraining is triggered automatically after each insert once
 * inserts_since_retrain >= min_new_vectors AND
 * computed_drift > drift_threshold.
 *
 * @param db                Database instance; must be non-NULL.
 * @param drift_threshold   Fractional drift above initial inertia that
 *                          triggers retrain (e.g. 0.15 = 15 %).
 * @param min_new_vectors   Minimum number of inserts before drift is
 *                          checked (default 50000).
 */
void gv_db_set_retrain_config(GV_Database *db, float drift_threshold,
                               size_t min_new_vectors);

/**
 * @brief Manually trigger IVF retraining (runs synchronously on the
 *        calling thread if no background retrain is in progress).
 *
 * @param db Database instance; must be non-NULL.
 * @return 0 on success, -1 on error.
 */
int gv_db_trigger_retrain(GV_Database *db);

/**
 * @brief Query the current retrain status.
 *
 * @param db         Database instance; must be non-NULL.
 * @param is_running Output: 1 if a retrain thread is active, else 0.
 *                   May be NULL.
 * @param last_drift Output: last drift ratio returned by
 *                   ivf_retrain_check_drift().  May be NULL.
 */
void gv_db_retrain_status(const GV_Database *db, int *is_running,
                           float *last_drift);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_INDEX_IVF_RETRAIN_H */
