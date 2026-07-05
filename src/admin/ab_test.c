/**
 * @file ab_test.c
 * @brief Index-level A/B testing implementation.
 *
 * Provides a shadow index alongside the primary index, routing a configurable
 * fraction of queries to each and accumulating per-variant latency statistics.
 */

#define _POSIX_C_SOURCE 200809L

#include "admin/ab_test.h"
#include "storage/database.h"
#include "core/utils.h"
#include "admin/tracing.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <inttypes.h>

/* -------------------------------------------------------------------------
 * Internal helpers
 * ---------------------------------------------------------------------- */

/**
 * @brief Return current monotonic time in microseconds.
 */
static uint64_t ab_now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

/* -------------------------------------------------------------------------
 * ab_test_create
 * ---------------------------------------------------------------------- */

GV_ABTest *ab_test_create(GV_Database *db, const char *name, int type_b, float split) {
    if (!db || !name) {
        return NULL;
    }
    if (split < 0.0f) split = 0.0f;
    if (split > 1.0f) split = 1.0f;

    GV_ABTest *test = (GV_ABTest *)calloc(1, sizeof(GV_ABTest));
    if (!test) {
        return NULL;
    }

    /* Populate fields. */
    strncpy(test->name, name, sizeof(test->name) - 1);
    test->name[sizeof(test->name) - 1] = '\0';
    test->index_type_a   = (int)db->index_type;
    test->index_type_b   = type_b;
    test->traffic_split  = split;
    test->queries_a      = 0;
    test->queries_b      = 0;
    test->latency_sum_a  = 0.0;
    test->latency_sum_b  = 0.0;
    test->route_counter  = 0;
    test->creation_time_us = ab_now_us();

    /* Create the shadow database with the same dimension but different type. */
    GV_Database *shadow = db_open(NULL, db->dimension, (GV_IndexType)type_b);
    if (!shadow) {
        free(test);
        return NULL;
    }

    /* Copy all existing vectors from the primary into the shadow. */
    size_t count = database_count(db);
    for (size_t i = 0; i < count; i++) {
        const float *vec = database_get_vector(db, i);
        if (vec) {
            db_add_vector(shadow, vec, db->dimension);
        }
    }

    test->shadow_db = shadow;
    return test;
}

/* -------------------------------------------------------------------------
 * ab_test_route — must be called with db->ab_mutex held
 * ---------------------------------------------------------------------- */

int ab_test_route(GV_Database *db) {
    if (!db || !db->ab_test) {
        return 0; /* default to A */
    }
    GV_ABTest *test = (GV_ABTest *)db->ab_test;
    uint64_t counter = test->route_counter++;

    /*
     * Mix the counter with a multiplicative hash to spread consecutive
     * query numbers evenly across the [0, 1000) space. This ensures that
     * even small batches (e.g. 100 queries) observe a traffic_split close
     * to the configured fraction rather than routing all traffic to one
     * variant until the first 'period' boundary is crossed.
     *
     * Knuth's multiplicative hash constant for 64-bit integers:
     *   h = counter * 2654435761
     *   h = h ^ (h >> 16)   (additional mixing)
     */
    uint64_t h = counter * 2654435761ULL;
    h ^= (h >> 16);
    uint64_t slot = h % 1000ULL;
    uint64_t threshold = (uint64_t)(test->traffic_split * 1000.0f + 0.5f);
    return slot < threshold ? 1 : 0;
}

/* -------------------------------------------------------------------------
 * ab_test_record — must be called with db->ab_mutex held
 * ---------------------------------------------------------------------- */

void ab_test_record(GV_Database *db, int which, double latency_us) {
    if (!db || !db->ab_test) {
        return;
    }
    GV_ABTest *test = (GV_ABTest *)db->ab_test;
    if (which == 1) {
        test->queries_b++;
        test->latency_sum_b += latency_us;
    } else {
        test->queries_a++;
        test->latency_sum_a += latency_us;
    }
}

/* -------------------------------------------------------------------------
 * ab_test_report — reads ab_test under caller's lock
 * ---------------------------------------------------------------------- */

int ab_test_report(const GV_Database *db, char *buf, size_t len) {
    if (!db || !db->ab_test || !buf || len == 0) {
        return -1;
    }
    const GV_ABTest *test = (const GV_ABTest *)db->ab_test;

    double avg_lat_a = (test->queries_a > 0)
                       ? test->latency_sum_a / (double)test->queries_a
                       : 0.0;
    double avg_lat_b = (test->queries_b > 0)
                       ? test->latency_sum_b / (double)test->queries_b
                       : 0.0;

    int n = snprintf(buf, len,
        "{"
        "\"name\":\"%s\","
        "\"index_type_a\":%d,"
        "\"index_type_b\":%d,"
        "\"traffic_split\":%.4f,"
        "\"queries_a\":%" PRIu64 ","
        "\"queries_b\":%" PRIu64 ","
        "\"avg_latency_a_us\":%.3f,"
        "\"avg_latency_b_us\":%.3f,"
        "\"creation_time_us\":%" PRIu64
        "}",
        test->name,
        test->index_type_a,
        test->index_type_b,
        (double)test->traffic_split,
        test->queries_a,
        test->queries_b,
        avg_lat_a,
        avg_lat_b,
        test->creation_time_us);

    return (n > 0 && (size_t)n < len) ? 0 : -1;
}

/* -------------------------------------------------------------------------
 * ab_test_destroy
 * ---------------------------------------------------------------------- */

void ab_test_destroy(GV_ABTest *test) {
    if (!test) {
        return;
    }
    if (test->shadow_db) {
        db_close(test->shadow_db);
        test->shadow_db = NULL;
    }
    free(test);
}

/* -------------------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------------- */

int gv_db_ab_test_start(GV_Database *db, const char *name, int type_b, float split) {
    if (!db || !name) {
        return -1;
    }
    pthread_mutex_lock(&db->ab_mutex);
    if (db->ab_test != NULL) {
        /* A test is already running. */
        pthread_mutex_unlock(&db->ab_mutex);
        return -1;
    }
    pthread_mutex_unlock(&db->ab_mutex);

    /* Create the test outside the lock (may do DB I/O). */
    GV_ABTest *test = ab_test_create(db, name, type_b, split);
    if (!test) {
        return -1;
    }

    pthread_mutex_lock(&db->ab_mutex);
    if (db->ab_test != NULL) {
        /* Race: another thread started a test between our checks. */
        pthread_mutex_unlock(&db->ab_mutex);
        ab_test_destroy(test);
        return -1;
    }
    db->ab_test = test;
    pthread_mutex_unlock(&db->ab_mutex);
    return 0;
}

int gv_db_ab_test_search(GV_Database *db, const float *query, size_t k,
                          GV_SearchResult *results, GV_DistanceType distance_type) {
    if (!db || !query || !results || k == 0) {
        return -1;
    }

    pthread_mutex_lock(&db->ab_mutex);
    if (!db->ab_test) {
        pthread_mutex_unlock(&db->ab_mutex);
        /* Fall back to primary search. */
        return db_search(db, query, k, results, distance_type);
    }

    int which = ab_test_route(db);
    pthread_mutex_unlock(&db->ab_mutex);

    uint64_t t0 = ab_now_us();
    int found;
    if (which == 1) {
        /* Shadow index (B). */
        GV_ABTest *test = (GV_ABTest *)db->ab_test;
        found = db_search(test->shadow_db, query, k, results, distance_type);
    } else {
        /* Primary index (A). */
        found = db_search(db, query, k, results, distance_type);
    }
    uint64_t t1 = ab_now_us();
    double latency_us = (double)(t1 - t0);

    pthread_mutex_lock(&db->ab_mutex);
    ab_test_record(db, which, latency_us);
    pthread_mutex_unlock(&db->ab_mutex);

    return found;
}

int gv_db_ab_test_report(const GV_Database *db, char *buf, size_t len) {
    if (!db || !buf || len == 0) {
        return -1;
    }
    pthread_mutex_lock((pthread_mutex_t *)&db->ab_mutex);
    int rc = ab_test_report(db, buf, len);
    pthread_mutex_unlock((pthread_mutex_t *)&db->ab_mutex);
    return rc;
}

void gv_db_ab_test_stop(GV_Database *db) {
    if (!db) {
        return;
    }
    pthread_mutex_lock(&db->ab_mutex);
    GV_ABTest *test = (GV_ABTest *)db->ab_test;
    db->ab_test = NULL;
    pthread_mutex_unlock(&db->ab_mutex);
    ab_test_destroy(test);
}
