/**
 * @file gv_consistency.c
 * @brief Per-query configurable consistency levels implementation.
 *
 * Supports STRONG, EVENTUAL, BOUNDED_STALENESS, and SESSION consistency
 * levels.  The manager maintains a default level and a session table that
 * maps session tokens to the last observed write position so that SESSION
 * consistency can guarantee read-your-writes semantics.
 */

#include "gigavector/gv_consistency.h"

#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

/**
 * @brief Maximum number of concurrent sessions tracked by the manager.
 */
#define GV_MAX_SESSIONS 1024

/**
 * @brief A single entry in the session table.
 *
 * A token value of 0 indicates an unused slot.
 */
typedef struct {
    uint64_t token;             /* Session token (0 = unused) */
    uint64_t write_position;    /* Last write position seen in this session */
} GV_SessionEntry;

/**
 * @brief Consistency manager internal structure.
 */
struct GV_ConsistencyManager {
    GV_ConsistencyLevel default_level;

    /* Session table (fixed-size array) */
    GV_SessionEntry sessions[GV_MAX_SESSIONS];
    size_t session_count;

    /* Atomic counter for generating unique session tokens */
    _Atomic uint64_t next_token;

    /* Mutex protecting all shared state */
    pthread_mutex_t mutex;
};

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

GV_ConsistencyManager *gv_consistency_create(GV_ConsistencyLevel default_level)
{
    GV_ConsistencyManager *mgr = calloc(1, sizeof(*mgr));
    if (!mgr) {
        return NULL;
    }

    mgr->default_level = default_level;
    mgr->session_count = 0;
    atomic_store(&mgr->next_token, 1); /* Tokens start at 1; 0 means unused */

    if (pthread_mutex_init(&mgr->mutex, NULL) != 0) {
        free(mgr);
        return NULL;
    }

    return mgr;
}

void gv_consistency_destroy(GV_ConsistencyManager *mgr)
{
    if (!mgr) {
        return;
    }
    pthread_mutex_destroy(&mgr->mutex);
    free(mgr);
}

/* ============================================================================
 * Default Level
 * ============================================================================ */

int gv_consistency_set_default(GV_ConsistencyManager *mgr, GV_ConsistencyLevel level)
{
    if (!mgr) {
        return -1;
    }
    if (level < GV_CONSISTENCY_STRONG || level > GV_CONSISTENCY_SESSION) {
        return -1;
    }

    pthread_mutex_lock(&mgr->mutex);
    mgr->default_level = level;
    pthread_mutex_unlock(&mgr->mutex);

    return 0;
}

GV_ConsistencyLevel gv_consistency_get_default(const GV_ConsistencyManager *mgr)
{
    if (!mgr) {
        return GV_CONSISTENCY_STRONG;
    }
    /* default_level is read atomically (single word); safe without lock for
     * readers that tolerate a slightly stale value.  We cast away const to
     * lock, but the logical constness is preserved. */
    return mgr->default_level;
}

/* ============================================================================
 * Consistency Check
 * ============================================================================ */

/**
 * @brief Look up the write position recorded for a session token.
 *
 * Must be called with mgr->mutex held.
 *
 * @return The write position, or 0 if the token is not found.
 */
static uint64_t session_lookup_locked(const GV_ConsistencyManager *mgr,
                                      uint64_t session_token)
{
    for (size_t i = 0; i < mgr->session_count; i++) {
        if (mgr->sessions[i].token == session_token) {
            return mgr->sessions[i].write_position;
        }
    }
    return 0;
}

int gv_consistency_check(const GV_ConsistencyManager *mgr,
                          const GV_ConsistencyConfig *config,
                          uint64_t replica_lag_ms,
                          uint64_t replica_position)
{
    if (!mgr || !config) {
        return -1;
    }

    GV_ConsistencyLevel level = config->level;

    switch (level) {
    case GV_CONSISTENCY_STRONG:
        /*
         * STRONG consistency: the caller must read from the leader.
         * Return 0 to indicate that this replica does NOT satisfy the
         * requirement (the caller should redirect to the leader).
         */
        return 0;

    case GV_CONSISTENCY_EVENTUAL:
        /*
         * EVENTUAL consistency: any replica is acceptable regardless of
         * its current lag.
         */
        return 1;

    case GV_CONSISTENCY_BOUNDED_STALENESS:
        /*
         * BOUNDED_STALENESS: the replica is acceptable only if its
         * replication lag is within the configured bound.
         */
        return (replica_lag_ms <= config->max_staleness_ms) ? 1 : 0;

    case GV_CONSISTENCY_SESSION: {
        /*
         * SESSION consistency: the replica is acceptable only if its
         * position has caught up to the last write position recorded for
         * this session.  We need the lock to read the session table.
         */
        GV_ConsistencyManager *mutable_mgr = (GV_ConsistencyManager *)mgr;
        pthread_mutex_lock(&mutable_mgr->mutex);
        uint64_t required_pos = session_lookup_locked(mgr, config->session_token);
        pthread_mutex_unlock(&mutable_mgr->mutex);

        if (required_pos == 0) {
            /* Unknown session or no writes recorded yet -- allow read */
            return 1;
        }
        return (replica_position >= required_pos) ? 1 : 0;
    }

    default:
        return -1;
    }
}

/* ============================================================================
 * Session Token Management
 * ============================================================================ */

uint64_t gv_consistency_new_session(GV_ConsistencyManager *mgr)
{
    if (!mgr) {
        return 0;
    }

    uint64_t token = atomic_fetch_add(&mgr->next_token, 1);

    pthread_mutex_lock(&mgr->mutex);

    if (mgr->session_count < GV_MAX_SESSIONS) {
        mgr->sessions[mgr->session_count].token = token;
        mgr->sessions[mgr->session_count].write_position = 0;
        mgr->session_count++;
    } else {
        /*
         * Session table is full.  Evict the oldest session (index 0) by
         * shifting the array and reusing the last slot.
         */
        memmove(&mgr->sessions[0], &mgr->sessions[1],
                (GV_MAX_SESSIONS - 1) * sizeof(GV_SessionEntry));
        mgr->sessions[GV_MAX_SESSIONS - 1].token = token;
        mgr->sessions[GV_MAX_SESSIONS - 1].write_position = 0;
        /* session_count stays at GV_MAX_SESSIONS */
    }

    pthread_mutex_unlock(&mgr->mutex);

    return token;
}

int gv_consistency_update_session(GV_ConsistencyManager *mgr,
                                   uint64_t session_token,
                                   uint64_t write_position)
{
    if (!mgr || session_token == 0) {
        return -1;
    }

    pthread_mutex_lock(&mgr->mutex);

    for (size_t i = 0; i < mgr->session_count; i++) {
        if (mgr->sessions[i].token == session_token) {
            /* Only advance the position forward (monotonic) */
            if (write_position > mgr->sessions[i].write_position) {
                mgr->sessions[i].write_position = write_position;
            }
            pthread_mutex_unlock(&mgr->mutex);
            return 0;
        }
    }

    pthread_mutex_unlock(&mgr->mutex);
    return -1; /* Session not found */
}

uint64_t gv_consistency_get_session_position(const GV_ConsistencyManager *mgr,
                                              uint64_t session_token)
{
    if (!mgr || session_token == 0) {
        return 0;
    }

    GV_ConsistencyManager *mutable_mgr = (GV_ConsistencyManager *)mgr;
    pthread_mutex_lock(&mutable_mgr->mutex);
    uint64_t pos = session_lookup_locked(mgr, session_token);
    pthread_mutex_unlock(&mutable_mgr->mutex);

    return pos;
}

/* ============================================================================
 * Config Helpers
 * ============================================================================ */

void gv_consistency_config_init(GV_ConsistencyConfig *config)
{
    if (!config) {
        return;
    }
    memset(config, 0, sizeof(*config));
    config->level = GV_CONSISTENCY_STRONG;
}

GV_ConsistencyConfig gv_consistency_strong(void)
{
    GV_ConsistencyConfig config;
    memset(&config, 0, sizeof(config));
    config.level = GV_CONSISTENCY_STRONG;
    return config;
}

GV_ConsistencyConfig gv_consistency_eventual(void)
{
    GV_ConsistencyConfig config;
    memset(&config, 0, sizeof(config));
    config.level = GV_CONSISTENCY_EVENTUAL;
    return config;
}

GV_ConsistencyConfig gv_consistency_bounded(uint64_t max_staleness_ms)
{
    GV_ConsistencyConfig config;
    memset(&config, 0, sizeof(config));
    config.level = GV_CONSISTENCY_BOUNDED_STALENESS;
    config.max_staleness_ms = max_staleness_ms;
    return config;
}

GV_ConsistencyConfig gv_consistency_session(uint64_t token)
{
    GV_ConsistencyConfig config;
    memset(&config, 0, sizeof(config));
    config.level = GV_CONSISTENCY_SESSION;
    config.session_token = token;
    return config;
}
