/*
 * tiered_storage.c — Tiered hot/warm/cold storage management for GigaVector.
 *
 * Strategy:
 *   HOT:  age < hot_max_age_seconds
 *   WARM: hot_max_age_seconds <= age < warm_max_age_seconds
 *   COLD: age >= warm_max_age_seconds
 *
 * The manager keeps a per-slot insertion timestamp (microseconds since epoch).
 * Classification is performed on-demand from the current wall-clock time.
 */

#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>

#include "storage/tiered_storage.h"
#include "storage/database.h"
#include "storage/soa_storage.h"

/* ---------------------------------------------------------------------- */
/* Internal manager struct                                                 */
/* ---------------------------------------------------------------------- */

struct GV_TieredStorageManager {
    uint64_t      *insert_times_us; /**< Per-slot insertion timestamp (us since epoch). */
    size_t         capacity;        /**< Allocated slots. */
    pthread_mutex_t mutex;
};

/* ---------------------------------------------------------------------- */
/* Helpers                                                                 */
/* ---------------------------------------------------------------------- */

static uint64_t ts_now_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
}

/** Grow the manager's timestamp array to cover at least @p required slots. */
static int ts_grow(GV_TieredStorageManager *mgr, size_t required) {
    if (required <= mgr->capacity) return 0;
    size_t new_cap = mgr->capacity ? mgr->capacity * 2 : 64;
    while (new_cap < required) new_cap *= 2;
    uint64_t *new_arr = (uint64_t *)realloc(mgr->insert_times_us,
                                             new_cap * sizeof(uint64_t));
    if (!new_arr) return -1;
    /* Zero-initialise new slots so they are classified as HOT by default
     * (a zero timestamp means "unknown / just inserted"). */
    memset(new_arr + mgr->capacity, 0,
           (new_cap - mgr->capacity) * sizeof(uint64_t));
    mgr->insert_times_us = new_arr;
    mgr->capacity = new_cap;
    return 0;
}

/* ---------------------------------------------------------------------- */
/* Lifecycle                                                               */
/* ---------------------------------------------------------------------- */

GV_TieredStorageManager *tiered_storage_create(size_t initial_capacity) {
    GV_TieredStorageManager *mgr =
        (GV_TieredStorageManager *)calloc(1, sizeof(GV_TieredStorageManager));
    if (!mgr) return NULL;

    if (initial_capacity == 0) initial_capacity = 64;
    mgr->insert_times_us = (uint64_t *)calloc(initial_capacity, sizeof(uint64_t));
    if (!mgr->insert_times_us) {
        free(mgr);
        return NULL;
    }
    mgr->capacity = initial_capacity;
    pthread_mutex_init(&mgr->mutex, NULL);
    return mgr;
}

void tiered_storage_destroy(GV_TieredStorageManager *mgr) {
    if (!mgr) return;
    pthread_mutex_destroy(&mgr->mutex);
    free(mgr->insert_times_us);
    free(mgr);
}

/* ---------------------------------------------------------------------- */
/* Timestamp management                                                    */
/* ---------------------------------------------------------------------- */

int tiered_storage_record_insert(GV_TieredStorageManager *mgr,
                                  size_t vec_id,
                                  uint64_t insert_time_us) {
    if (!mgr) return -1;
    pthread_mutex_lock(&mgr->mutex);
    if (ts_grow(mgr, vec_id + 1) != 0) {
        pthread_mutex_unlock(&mgr->mutex);
        return -1;
    }
    mgr->insert_times_us[vec_id] = insert_time_us;
    pthread_mutex_unlock(&mgr->mutex);
    return 0;
}

int tiered_storage_set_insert_time(GV_TieredStorageManager *mgr,
                                    size_t vec_id,
                                    uint64_t insert_time_us) {
    if (!mgr) return -1;
    pthread_mutex_lock(&mgr->mutex);
    if (vec_id >= mgr->capacity) {
        pthread_mutex_unlock(&mgr->mutex);
        return -1;
    }
    mgr->insert_times_us[vec_id] = insert_time_us;
    pthread_mutex_unlock(&mgr->mutex);
    return 0;
}

/* ---------------------------------------------------------------------- */
/* Classification                                                          */
/* ---------------------------------------------------------------------- */

GV_StorageTier tiered_storage_classify(const GV_Database *db, size_t vec_id) {
    if (!db || !db->tiered_storage || !db->tiering_enabled)
        return GV_TIER_HOT;

    GV_TieredStorageManager *mgr = db->tiered_storage;

    pthread_mutex_lock(&mgr->mutex);
    if (vec_id >= mgr->capacity) {
        pthread_mutex_unlock(&mgr->mutex);
        return GV_TIER_HOT;
    }
    uint64_t insert_us = mgr->insert_times_us[vec_id];
    pthread_mutex_unlock(&mgr->mutex);

    /* A zero timestamp means we have no record — treat as HOT. */
    if (insert_us == 0) return GV_TIER_HOT;

    uint64_t now_us = ts_now_us();
    uint64_t age_sec = (now_us > insert_us) ? (now_us - insert_us) / 1000000ULL : 0;

    if (db->warm_max_age_seconds > 0 && age_sec >= db->warm_max_age_seconds)
        return GV_TIER_COLD;
    if (db->hot_max_age_seconds > 0 && age_sec >= db->hot_max_age_seconds)
        return GV_TIER_WARM;
    return GV_TIER_HOT;
}

int tiered_storage_get_tier(const GV_Database *db, size_t vec_id,
                              GV_StorageTier *out) {
    if (!db || !out) return -1;
    if (!db->tiering_enabled || !db->tiered_storage) return -1;
    *out = tiered_storage_classify(db, vec_id);
    return 0;
}

/* ---------------------------------------------------------------------- */
/* Stats scan                                                              */
/* ---------------------------------------------------------------------- */

void tiered_storage_promote(const GV_Database *db,
                             size_t *hot_count,
                             size_t *warm_count,
                             size_t *cold_count) {
    size_t h = 0, w = 0, c = 0;

    if (!db || !db->tiering_enabled || !db->tiered_storage ||
        !db->soa_storage) {
        if (hot_count)  *hot_count  = h;
        if (warm_count) *warm_count = w;
        if (cold_count) *cold_count = c;
        return;
    }

    size_t n = soa_storage_count(db->soa_storage);
    for (size_t i = 0; i < n; ++i) {
        if (soa_storage_is_deleted(db->soa_storage, i) == 1) continue;
        GV_StorageTier t = tiered_storage_classify(db, i);
        if      (t == GV_TIER_HOT)  ++h;
        else if (t == GV_TIER_WARM) ++w;
        else                         ++c;
    }

    if (hot_count)  *hot_count  = h;
    if (warm_count) *warm_count = w;
    if (cold_count) *cold_count = c;
}

/* ---------------------------------------------------------------------- */
/* Public API                                                              */
/* ---------------------------------------------------------------------- */

int gv_db_set_tiering_config(GV_Database *db,
                               uint64_t hot_max_age_sec,
                               uint64_t warm_max_age_sec,
                               size_t hot_max_vectors) {
    if (!db) return -1;

    if (!db->tiered_storage) {
        size_t cap = db->soa_storage ? soa_storage_count(db->soa_storage) : 64;
        if (cap == 0) cap = 64;
        db->tiered_storage = tiered_storage_create(cap);
        if (!db->tiered_storage) return -1;
    }

    db->hot_max_age_seconds  = hot_max_age_sec;
    db->warm_max_age_seconds = warm_max_age_sec;
    db->hot_max_vectors      = hot_max_vectors;
    db->tiering_enabled      = 1;
    return 0;
}

int gv_db_get_vector_tier(const GV_Database *db, size_t vec_id,
                            GV_StorageTier *out) {
    return tiered_storage_get_tier(db, vec_id, out);
}

int gv_db_tiering_stats(const GV_Database *db,
                          size_t *hot_count,
                          size_t *warm_count,
                          size_t *cold_count) {
    if (!db) return -1;
    if (!db->tiering_enabled || !db->tiered_storage) return -1;
    tiered_storage_promote(db, hot_count, warm_count, cold_count);
    return 0;
}
