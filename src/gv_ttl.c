/**
 * @file gv_ttl.c
 * @brief TTL (Time-to-Live) implementation.
 */

#include "gigavector/gv_ttl.h"
#include "gigavector/gv_database.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

/**
 * @brief TTL entry for a vector.
 */
typedef struct GV_TTLEntry {
    size_t vector_index;           /**< Index of the vector. */
    uint64_t expire_at;            /**< Unix timestamp when vector expires. */
    struct GV_TTLEntry *next;      /**< Next entry in hash bucket. */
} GV_TTLEntry;

/**
 * @brief Hash table bucket count.
 */
#define TTL_HASH_BUCKETS 1024

/**
 * @brief TTL manager internal structure.
 */
struct GV_TTLManager {
    GV_TTLConfig config;

    /* Hash table for TTL entries */
    GV_TTLEntry *buckets[TTL_HASH_BUCKETS];
    size_t entry_count;
    pthread_mutex_t mutex;

    /* Statistics */
    uint64_t total_expired;
    uint64_t last_cleanup_time;

    /* Background cleanup */
    pthread_t cleanup_thread;
    int cleanup_running;
    int cleanup_stop_requested;
    pthread_cond_t cleanup_cond;
    GV_Database *cleanup_db;
};

/* ============================================================================
 * Hash Function
 * ============================================================================ */

static size_t hash_index(size_t vector_index) {
    return vector_index % TTL_HASH_BUCKETS;
}

/* ============================================================================
 * Configuration
 * ============================================================================ */

static const GV_TTLConfig DEFAULT_CONFIG = {
    .default_ttl_seconds = 0,
    .cleanup_interval_seconds = 60,
    .lazy_expiration = 1,
    .max_expired_per_cleanup = 1000
};

void gv_ttl_config_init(GV_TTLConfig *config) {
    if (!config) return;
    *config = DEFAULT_CONFIG;
}

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

GV_TTLManager *gv_ttl_create(const GV_TTLConfig *config) {
    GV_TTLManager *mgr = calloc(1, sizeof(GV_TTLManager));
    if (!mgr) return NULL;

    mgr->config = config ? *config : DEFAULT_CONFIG;

    if (pthread_mutex_init(&mgr->mutex, NULL) != 0) {
        free(mgr);
        return NULL;
    }

    if (pthread_cond_init(&mgr->cleanup_cond, NULL) != 0) {
        pthread_mutex_destroy(&mgr->mutex);
        free(mgr);
        return NULL;
    }

    return mgr;
}

void gv_ttl_destroy(GV_TTLManager *mgr) {
    if (!mgr) return;

    /* Stop background cleanup if running */
    gv_ttl_stop_background_cleanup(mgr);

    /* Free all entries */
    for (size_t i = 0; i < TTL_HASH_BUCKETS; i++) {
        GV_TTLEntry *entry = mgr->buckets[i];
        while (entry) {
            GV_TTLEntry *next = entry->next;
            free(entry);
            entry = next;
        }
    }

    pthread_cond_destroy(&mgr->cleanup_cond);
    pthread_mutex_destroy(&mgr->mutex);
    free(mgr);
}

/* ============================================================================
 * Internal Helpers
 * ============================================================================ */

static GV_TTLEntry *find_entry(GV_TTLManager *mgr, size_t vector_index) {
    size_t bucket = hash_index(vector_index);
    GV_TTLEntry *entry = mgr->buckets[bucket];
    while (entry) {
        if (entry->vector_index == vector_index) {
            return entry;
        }
        entry = entry->next;
    }
    return NULL;
}

static int remove_entry(GV_TTLManager *mgr, size_t vector_index) {
    size_t bucket = hash_index(vector_index);
    GV_TTLEntry **pp = &mgr->buckets[bucket];

    while (*pp) {
        if ((*pp)->vector_index == vector_index) {
            GV_TTLEntry *to_free = *pp;
            *pp = (*pp)->next;
            free(to_free);
            mgr->entry_count--;
            return 0;
        }
        pp = &(*pp)->next;
    }
    return -1;
}

static uint64_t current_time_unix(void) {
    return (uint64_t)time(NULL);
}

/* ============================================================================
 * TTL Operations
 * ============================================================================ */

int gv_ttl_set(GV_TTLManager *mgr, size_t vector_index, uint64_t ttl_seconds) {
    if (!mgr) return -1;

    if (ttl_seconds == 0) {
        return gv_ttl_remove(mgr, vector_index);
    }

    uint64_t expire_at = current_time_unix() + ttl_seconds;
    return gv_ttl_set_absolute(mgr, vector_index, expire_at);
}

int gv_ttl_set_absolute(GV_TTLManager *mgr, size_t vector_index, uint64_t expire_at_unix) {
    if (!mgr) return -1;

    if (expire_at_unix == 0) {
        return gv_ttl_remove(mgr, vector_index);
    }

    pthread_mutex_lock(&mgr->mutex);

    /* Check if entry already exists */
    GV_TTLEntry *entry = find_entry(mgr, vector_index);
    if (entry) {
        entry->expire_at = expire_at_unix;
        pthread_mutex_unlock(&mgr->mutex);
        return 0;
    }

    /* Create new entry */
    entry = malloc(sizeof(GV_TTLEntry));
    if (!entry) {
        pthread_mutex_unlock(&mgr->mutex);
        return -1;
    }

    entry->vector_index = vector_index;
    entry->expire_at = expire_at_unix;

    /* Insert into hash table */
    size_t bucket = hash_index(vector_index);
    entry->next = mgr->buckets[bucket];
    mgr->buckets[bucket] = entry;
    mgr->entry_count++;

    pthread_mutex_unlock(&mgr->mutex);
    return 0;
}

int gv_ttl_get(const GV_TTLManager *mgr, size_t vector_index, uint64_t *expire_at) {
    if (!mgr || !expire_at) return -1;

    pthread_mutex_lock((pthread_mutex_t *)&mgr->mutex);

    GV_TTLEntry *entry = find_entry((GV_TTLManager *)mgr, vector_index);
    if (entry) {
        *expire_at = entry->expire_at;
    } else {
        *expire_at = 0;
    }

    pthread_mutex_unlock((pthread_mutex_t *)&mgr->mutex);
    return 0;
}

int gv_ttl_remove(GV_TTLManager *mgr, size_t vector_index) {
    if (!mgr) return -1;

    pthread_mutex_lock(&mgr->mutex);
    int result = remove_entry(mgr, vector_index);
    pthread_mutex_unlock(&mgr->mutex);

    return result;
}

int gv_ttl_is_expired(const GV_TTLManager *mgr, size_t vector_index) {
    if (!mgr) return -1;

    pthread_mutex_lock((pthread_mutex_t *)&mgr->mutex);

    GV_TTLEntry *entry = find_entry((GV_TTLManager *)mgr, vector_index);
    int expired = 0;
    if (entry) {
        expired = (entry->expire_at <= current_time_unix()) ? 1 : 0;
    }

    pthread_mutex_unlock((pthread_mutex_t *)&mgr->mutex);
    return expired;
}

int gv_ttl_get_remaining(const GV_TTLManager *mgr, size_t vector_index, uint64_t *remaining_seconds) {
    if (!mgr || !remaining_seconds) return -1;

    pthread_mutex_lock((pthread_mutex_t *)&mgr->mutex);

    GV_TTLEntry *entry = find_entry((GV_TTLManager *)mgr, vector_index);
    if (entry) {
        uint64_t now = current_time_unix();
        if (entry->expire_at > now) {
            *remaining_seconds = entry->expire_at - now;
        } else {
            *remaining_seconds = 0;
        }
    } else {
        *remaining_seconds = 0;
    }

    pthread_mutex_unlock((pthread_mutex_t *)&mgr->mutex);
    return 0;
}

/* ============================================================================
 * Cleanup Operations
 * ============================================================================ */

int gv_ttl_cleanup_expired(GV_TTLManager *mgr, GV_Database *db) {
    if (!mgr || !db) return -1;

    pthread_mutex_lock(&mgr->mutex);

    uint64_t now = current_time_unix();
    size_t expired_count = 0;
    size_t max_expire = mgr->config.max_expired_per_cleanup;

    /* Collect expired indices */
    size_t *expired_indices = malloc(max_expire * sizeof(size_t));
    if (!expired_indices) {
        pthread_mutex_unlock(&mgr->mutex);
        return -1;
    }

    for (size_t bucket = 0; bucket < TTL_HASH_BUCKETS && expired_count < max_expire; bucket++) {
        GV_TTLEntry *entry = mgr->buckets[bucket];
        while (entry && expired_count < max_expire) {
            if (entry->expire_at <= now) {
                expired_indices[expired_count++] = entry->vector_index;
            }
            entry = entry->next;
        }
    }

    pthread_mutex_unlock(&mgr->mutex);

    /* Delete expired vectors from database */
    for (size_t i = 0; i < expired_count; i++) {
        if (gv_db_delete_vector_by_index(db, expired_indices[i]) == 0) {
            /* Remove from TTL tracking */
            pthread_mutex_lock(&mgr->mutex);
            remove_entry(mgr, expired_indices[i]);
            mgr->total_expired++;
            pthread_mutex_unlock(&mgr->mutex);
        }
    }

    free(expired_indices);

    pthread_mutex_lock(&mgr->mutex);
    mgr->last_cleanup_time = now;
    pthread_mutex_unlock(&mgr->mutex);

    return (int)expired_count;
}

/* ============================================================================
 * Background Cleanup Thread
 * ============================================================================ */

static void *cleanup_thread_func(void *arg) {
    GV_TTLManager *mgr = (GV_TTLManager *)arg;

    pthread_mutex_lock(&mgr->mutex);

    while (!mgr->cleanup_stop_requested) {
        /* Wait for cleanup interval */
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += mgr->config.cleanup_interval_seconds;

        int rc = pthread_cond_timedwait(&mgr->cleanup_cond, &mgr->mutex, &ts);

        if (mgr->cleanup_stop_requested) {
            break;
        }

        if (rc == 0 || rc == 110 /* ETIMEDOUT */) {
            /* Perform cleanup (release lock during cleanup) */
            pthread_mutex_unlock(&mgr->mutex);
            gv_ttl_cleanup_expired(mgr, mgr->cleanup_db);
            pthread_mutex_lock(&mgr->mutex);
        }
    }

    pthread_mutex_unlock(&mgr->mutex);
    return NULL;
}

int gv_ttl_start_background_cleanup(GV_TTLManager *mgr, GV_Database *db) {
    if (!mgr || !db) return -1;

    pthread_mutex_lock(&mgr->mutex);

    if (mgr->cleanup_running) {
        pthread_mutex_unlock(&mgr->mutex);
        return -1;
    }

    mgr->cleanup_db = db;
    mgr->cleanup_stop_requested = 0;
    mgr->cleanup_running = 1;

    pthread_mutex_unlock(&mgr->mutex);

    if (pthread_create(&mgr->cleanup_thread, NULL, cleanup_thread_func, mgr) != 0) {
        pthread_mutex_lock(&mgr->mutex);
        mgr->cleanup_running = 0;
        pthread_mutex_unlock(&mgr->mutex);
        return -1;
    }

    return 0;
}

void gv_ttl_stop_background_cleanup(GV_TTLManager *mgr) {
    if (!mgr) return;

    pthread_mutex_lock(&mgr->mutex);

    if (!mgr->cleanup_running) {
        pthread_mutex_unlock(&mgr->mutex);
        return;
    }

    mgr->cleanup_stop_requested = 1;
    pthread_cond_signal(&mgr->cleanup_cond);

    pthread_mutex_unlock(&mgr->mutex);

    pthread_join(mgr->cleanup_thread, NULL);

    pthread_mutex_lock(&mgr->mutex);
    mgr->cleanup_running = 0;
    mgr->cleanup_db = NULL;
    pthread_mutex_unlock(&mgr->mutex);
}

int gv_ttl_is_background_cleanup_running(const GV_TTLManager *mgr) {
    if (!mgr) return -1;

    pthread_mutex_lock((pthread_mutex_t *)&mgr->mutex);
    int running = mgr->cleanup_running;
    pthread_mutex_unlock((pthread_mutex_t *)&mgr->mutex);

    return running;
}

/* ============================================================================
 * Statistics
 * ============================================================================ */

int gv_ttl_get_stats(const GV_TTLManager *mgr, GV_TTLStats *stats) {
    if (!mgr || !stats) return -1;

    pthread_mutex_lock((pthread_mutex_t *)&mgr->mutex);

    stats->total_vectors_with_ttl = mgr->entry_count;
    stats->total_expired = mgr->total_expired;
    stats->last_cleanup_time = mgr->last_cleanup_time;

    /* Find next expiration time */
    uint64_t min_expire = UINT64_MAX;
    for (size_t bucket = 0; bucket < TTL_HASH_BUCKETS; bucket++) {
        GV_TTLEntry *entry = mgr->buckets[bucket];
        while (entry) {
            if (entry->expire_at < min_expire) {
                min_expire = entry->expire_at;
            }
            entry = entry->next;
        }
    }
    stats->next_expiration_time = (min_expire == UINT64_MAX) ? 0 : min_expire;

    pthread_mutex_unlock((pthread_mutex_t *)&mgr->mutex);
    return 0;
}

/* ============================================================================
 * Bulk Operations
 * ============================================================================ */

int gv_ttl_set_bulk(GV_TTLManager *mgr, const size_t *indices, size_t count, uint64_t ttl_seconds) {
    if (!mgr || !indices || count == 0) return -1;

    int success_count = 0;
    for (size_t i = 0; i < count; i++) {
        if (gv_ttl_set(mgr, indices[i], ttl_seconds) == 0) {
            success_count++;
        }
    }

    return success_count;
}

int gv_ttl_get_expiring_before(const GV_TTLManager *mgr, uint64_t before_unix,
                                size_t *indices, size_t max_indices) {
    if (!mgr || !indices || max_indices == 0) return -1;

    pthread_mutex_lock((pthread_mutex_t *)&mgr->mutex);

    size_t found = 0;
    for (size_t bucket = 0; bucket < TTL_HASH_BUCKETS && found < max_indices; bucket++) {
        GV_TTLEntry *entry = mgr->buckets[bucket];
        while (entry && found < max_indices) {
            if (entry->expire_at < before_unix) {
                indices[found++] = entry->vector_index;
            }
            entry = entry->next;
        }
    }

    pthread_mutex_unlock((pthread_mutex_t *)&mgr->mutex);
    return (int)found;
}
