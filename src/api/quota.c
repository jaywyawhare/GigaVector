#define _POSIX_C_SOURCE 200809L

/**
 * @file quota.c
 * @brief Per-namespace tenant isolation quotas implementation.
 *
 * Provides per-tenant resource limits: max vectors, memory, QPS, IPS,
 * storage, and collections.  Rate limiting uses a token bucket algorithm.
 * Thread-safe via pthread_mutex_t.
 */

#include "api/quota.h"
#include "core/memory.h"

#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include "core/compat.h"

#define QUOTA_MAX_TENANTS   256
#define TENANT_ID_MAX_LEN   128

/**
 * @brief Token bucket state for rate limiting.
 *
 * Refill rate  = configured max (tokens per second).
 * Burst        = max * 2 (allows short bursts).
 * Each allowed operation consumes one token.
 */
typedef struct {
    double tokens;          /**< Current number of tokens available. */
    double max_tokens;      /**< Burst capacity (max_rate * 2). */
    double refill_rate;     /**< Tokens added per second. */
    double last_refill;     /**< Timestamp (seconds) of last refill. */
} TokenBucket;

static void token_bucket_init(TokenBucket *tb, double max_rate) {
    if (max_rate <= 0.0) {
        tb->tokens      = 0.0;
        tb->max_tokens   = 0.0;
        tb->refill_rate  = 0.0;
        tb->last_refill  = 0.0;
        return;
    }
    tb->max_tokens   = max_rate * 2.0;
    tb->tokens       = tb->max_tokens;   /* Start full */
    tb->refill_rate  = max_rate;
    tb->last_refill  = 0.0;              /* Set on first use */
}

/**
 * @brief Return current monotonic time in seconds (fractional).
 */
static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/**
 * @brief Refill the bucket based on elapsed time, then try to consume
 *        @p count tokens.
 * @return 1 if tokens were consumed, 0 if not enough tokens.
 */
static int token_bucket_try_consume(TokenBucket *tb, double count) {
    if (tb->refill_rate <= 0.0) {
        return 1;  /* Rate limiting disabled (unlimited) */
    }

    double now = monotonic_seconds();

    if (tb->last_refill == 0.0) {
        tb->last_refill = now;
    }

    double elapsed = now - tb->last_refill;
    if (elapsed > 0.0) {
        tb->tokens += elapsed * tb->refill_rate;
        if (tb->tokens > tb->max_tokens) {
            tb->tokens = tb->max_tokens;
        }
        tb->last_refill = now;
    }

    if (tb->tokens >= count) {
        tb->tokens -= count;
        return 1;
    }
    return 0;
}

/**
 * @brief Return the instantaneous rate (tokens consumed per second recently).
 *
 * Approximated as: (max_tokens - current_tokens) normalised over a 1-second
 * window.  This is a rough estimate useful for reporting only.
 */
static double token_bucket_current_rate(const TokenBucket *tb) {
    if (tb->refill_rate <= 0.0) {
        return 0.0;
    }
    double used = tb->max_tokens - tb->tokens;
    if (used < 0.0) used = 0.0;
    return used;
}

typedef struct {
    char tenant_id[TENANT_ID_MAX_LEN];
    int  active;                         /**< 1 if slot is in use. */

    GV_QuotaConfig config;
    GV_QuotaUsage  usage;

    /* Capacity claimed by in-flight requests that have been admitted but have
     * not yet recorded their insert. Counted against the cap alongside
     * usage.current_vectors so that two concurrent admissions cannot both fit
     * where only one does. */
    size_t reserved_vectors;

    TokenBucket qps_bucket;              /**< Query rate limiter. */
    TokenBucket ips_bucket;              /**< Insert rate limiter. */
} QuotaTenantEntry;

struct GV_QuotaManager {
    QuotaTenantEntry tenants[QUOTA_MAX_TENANTS];
    size_t           tenant_count;
    pthread_mutex_t  mutex;
};

/**
 * @brief Find the slot index for a given tenant_id.
 * @return Index into tenants[], or -1 if not found.
 * @note Caller must hold the mutex.
 */
static int find_tenant(const GV_QuotaManager *mgr, const char *tenant_id) {
    for (size_t i = 0; i < QUOTA_MAX_TENANTS; i++) {
        if (mgr->tenants[i].active &&
            strncmp(mgr->tenants[i].tenant_id, tenant_id, TENANT_ID_MAX_LEN) == 0) {
            return (int)i;
        }
    }
    return -1;
}

/**
 * @brief Find an unused slot.
 * @return Index, or -1 if full.
 * @note Caller must hold the mutex.
 */
static int find_free_slot(const GV_QuotaManager *mgr) {
    for (size_t i = 0; i < QUOTA_MAX_TENANTS; i++) {
        if (!mgr->tenants[i].active) {
            return (int)i;
        }
    }
    return -1;
}

void gv_quota_config_init(GV_QuotaConfig *config) {
    if (!config) return;
    memset(config, 0, sizeof(*config));
    /* All zeros means "unlimited" for every field */
}

GV_QuotaManager *gv_quota_create(void) {
    GV_QuotaManager *mgr = gv_calloc(1, sizeof(GV_QuotaManager));
    if (!mgr) return NULL;

    if (pthread_mutex_init(&mgr->mutex, NULL) != 0) {
        gv_free(mgr);
        return NULL;
    }

    return mgr;
}

void gv_quota_destroy(GV_QuotaManager *mgr) {
    if (!mgr) return;
    pthread_mutex_destroy(&mgr->mutex);
    gv_free(mgr);
}

int gv_quota_set(GV_QuotaManager *mgr, const char *tenant_id,
                 const GV_QuotaConfig *config) {
    if (!mgr || !tenant_id || !config) return -1;
    if (strlen(tenant_id) == 0 || strlen(tenant_id) >= TENANT_ID_MAX_LEN) return -1;

    pthread_mutex_lock(&mgr->mutex);

    int idx = find_tenant(mgr, tenant_id);
    if (idx >= 0) {
        mgr->tenants[idx].config = *config;
        token_bucket_init(&mgr->tenants[idx].qps_bucket, config->max_qps);
        token_bucket_init(&mgr->tenants[idx].ips_bucket, config->max_ips);
        pthread_mutex_unlock(&mgr->mutex);
        return 0;
    }

    idx = find_free_slot(mgr);
    if (idx < 0) {
        pthread_mutex_unlock(&mgr->mutex);
        return -1;  /* Table full */
    }

    QuotaTenantEntry *entry = &mgr->tenants[idx];
    memset(entry, 0, sizeof(*entry));
    strncpy(entry->tenant_id, tenant_id, TENANT_ID_MAX_LEN - 1);
    entry->tenant_id[TENANT_ID_MAX_LEN - 1] = '\0';
    entry->active = 1;
    entry->config = *config;

    token_bucket_init(&entry->qps_bucket, config->max_qps);
    token_bucket_init(&entry->ips_bucket, config->max_ips);

    mgr->tenant_count++;

    pthread_mutex_unlock(&mgr->mutex);
    return 0;
}

int gv_quota_get(const GV_QuotaManager *mgr, const char *tenant_id,
                 GV_QuotaConfig *config) {
    if (!mgr || !tenant_id || !config) return -1;

    pthread_mutex_lock((pthread_mutex_t *)&mgr->mutex);

    int idx = find_tenant(mgr, tenant_id);
    if (idx < 0) {
        pthread_mutex_unlock((pthread_mutex_t *)&mgr->mutex);
        return -1;
    }

    *config = mgr->tenants[idx].config;

    pthread_mutex_unlock((pthread_mutex_t *)&mgr->mutex);
    return 0;
}

int gv_quota_remove(GV_QuotaManager *mgr, const char *tenant_id) {
    if (!mgr || !tenant_id) return -1;

    pthread_mutex_lock(&mgr->mutex);

    int idx = find_tenant(mgr, tenant_id);
    if (idx < 0) {
        pthread_mutex_unlock(&mgr->mutex);
        return -1;
    }

    mgr->tenants[idx].active = 0;
    memset(&mgr->tenants[idx], 0, sizeof(QuotaTenantEntry));
    mgr->tenant_count--;

    pthread_mutex_unlock(&mgr->mutex);
    return 0;
}

GV_QuotaResult gv_quota_check_insert(GV_QuotaManager *mgr, const char *tenant_id,
                                     size_t vector_count) {
    if (!mgr || !tenant_id) return GV_QUOTA_ERROR;

    pthread_mutex_lock(&mgr->mutex);

    int idx = find_tenant(mgr, tenant_id);
    if (idx < 0) {
        pthread_mutex_unlock(&mgr->mutex);
        return GV_QUOTA_ERROR;
    }

    QuotaTenantEntry *entry = &mgr->tenants[idx];
    const GV_QuotaConfig *cfg = &entry->config;

    /* Count in-flight reservations: ignoring them is what let two callers be
     * admitted against the same snapshot. */
    if (cfg->max_vectors > 0 &&
        entry->usage.current_vectors + entry->reserved_vectors + vector_count
            > cfg->max_vectors) {
        entry->usage.total_rejected++;
        pthread_mutex_unlock(&mgr->mutex);
        return GV_QUOTA_EXCEEDED;
    }

    /* Cannot predict exact bytes here; if already at the cap, reject further inserts. */
    if (cfg->max_memory_bytes > 0 &&
        entry->usage.current_memory_bytes >= cfg->max_memory_bytes) {
        entry->usage.total_rejected++;
        pthread_mutex_unlock(&mgr->mutex);
        return GV_QUOTA_EXCEEDED;
    }

    if (cfg->max_storage_bytes > 0 &&
        entry->usage.current_storage_bytes >= cfg->max_storage_bytes) {
        entry->usage.total_rejected++;
        pthread_mutex_unlock(&mgr->mutex);
        return GV_QUOTA_EXCEEDED;
    }

    if (cfg->max_ips > 0.0) {
        if (!token_bucket_try_consume(&entry->ips_bucket, (double)vector_count)) {
            entry->usage.total_throttled++;
            pthread_mutex_unlock(&mgr->mutex);
            return GV_QUOTA_THROTTLED;
        }
    }

    pthread_mutex_unlock(&mgr->mutex);
    return GV_QUOTA_OK;
}

GV_QuotaResult gv_quota_check_query(GV_QuotaManager *mgr, const char *tenant_id) {
    if (!mgr || !tenant_id) return GV_QUOTA_ERROR;

    pthread_mutex_lock(&mgr->mutex);

    int idx = find_tenant(mgr, tenant_id);
    if (idx < 0) {
        pthread_mutex_unlock(&mgr->mutex);
        return GV_QUOTA_ERROR;
    }

    QuotaTenantEntry *entry = &mgr->tenants[idx];
    const GV_QuotaConfig *cfg = &entry->config;

    if (cfg->max_qps > 0.0) {
        if (!token_bucket_try_consume(&entry->qps_bucket, 1.0)) {
            entry->usage.total_throttled++;
            pthread_mutex_unlock(&mgr->mutex);
            return GV_QUOTA_THROTTLED;
        }
    }

    pthread_mutex_unlock(&mgr->mutex);
    return GV_QUOTA_OK;
}

int gv_quota_record_insert(GV_QuotaManager *mgr, const char *tenant_id,
                           size_t count, size_t bytes) {
    if (!mgr || !tenant_id) return -1;

    pthread_mutex_lock(&mgr->mutex);

    int idx = find_tenant(mgr, tenant_id);
    if (idx < 0) {
        pthread_mutex_unlock(&mgr->mutex);
        return -1;
    }

    GV_QuotaUsage *u = &mgr->tenants[idx].usage;
    u->current_vectors      += count;
    u->current_memory_bytes += bytes;
    u->current_storage_bytes += bytes;

    u->current_ips = token_bucket_current_rate(&mgr->tenants[idx].ips_bucket);

    pthread_mutex_unlock(&mgr->mutex);
    return 0;
}

int gv_quota_record_query(GV_QuotaManager *mgr, const char *tenant_id) {
    if (!mgr || !tenant_id) return -1;

    pthread_mutex_lock(&mgr->mutex);

    int idx = find_tenant(mgr, tenant_id);
    if (idx < 0) {
        pthread_mutex_unlock(&mgr->mutex);
        return -1;
    }

    GV_QuotaUsage *u = &mgr->tenants[idx].usage;

    u->current_qps = token_bucket_current_rate(&mgr->tenants[idx].qps_bucket);

    pthread_mutex_unlock(&mgr->mutex);
    return 0;
}

int gv_quota_record_delete(GV_QuotaManager *mgr, const char *tenant_id,
                           size_t count, size_t bytes) {
    if (!mgr || !tenant_id) return -1;

    pthread_mutex_lock(&mgr->mutex);

    int idx = find_tenant(mgr, tenant_id);
    if (idx < 0) {
        pthread_mutex_unlock(&mgr->mutex);
        return -1;
    }

    GV_QuotaUsage *u = &mgr->tenants[idx].usage;

    /* Saturating subtraction to avoid unsigned underflow. */
    if (u->current_vectors >= count) {
        u->current_vectors -= count;
    } else {
        u->current_vectors = 0;
    }

    if (u->current_memory_bytes >= bytes) {
        u->current_memory_bytes -= bytes;
    } else {
        u->current_memory_bytes = 0;
    }

    if (u->current_storage_bytes >= bytes) {
        u->current_storage_bytes -= bytes;
    } else {
        u->current_storage_bytes = 0;
    }

    pthread_mutex_unlock(&mgr->mutex);
    return 0;
}

int gv_quota_get_usage(const GV_QuotaManager *mgr, const char *tenant_id,
                       GV_QuotaUsage *usage) {
    if (!mgr || !tenant_id || !usage) return -1;

    pthread_mutex_lock((pthread_mutex_t *)&mgr->mutex);

    int idx = find_tenant(mgr, tenant_id);
    if (idx < 0) {
        pthread_mutex_unlock((pthread_mutex_t *)&mgr->mutex);
        return -1;
    }

    *usage = mgr->tenants[idx].usage;

    usage->current_qps = token_bucket_current_rate(&mgr->tenants[idx].qps_bucket);
    usage->current_ips = token_bucket_current_rate(&mgr->tenants[idx].ips_bucket);

    pthread_mutex_unlock((pthread_mutex_t *)&mgr->mutex);
    return 0;
}

int gv_quota_reset_usage(GV_QuotaManager *mgr, const char *tenant_id) {
    if (!mgr || !tenant_id) return -1;

    pthread_mutex_lock(&mgr->mutex);

    int idx = find_tenant(mgr, tenant_id);
    if (idx < 0) {
        pthread_mutex_unlock(&mgr->mutex);
        return -1;
    }

    QuotaTenantEntry *entry = &mgr->tenants[idx];
    memset(&entry->usage, 0, sizeof(GV_QuotaUsage));

    token_bucket_init(&entry->qps_bucket, entry->config.max_qps);
    token_bucket_init(&entry->ips_bucket, entry->config.max_ips);

    pthread_mutex_unlock(&mgr->mutex);
    return 0;
}


/* ---- atomic reserve / commit / release ---------------------------------- *
 *
 * gv_quota_check_insert asks "would this fit?" and answers from a snapshot the
 * caller then races against. These three keep the check and the claim under a
 * single lock acquisition, which is what makes the answer binding.
 */

GV_QuotaResult gv_quota_reserve_insert(GV_QuotaManager *mgr, const char *tenant_id,
                                       size_t vector_count) {
    if (!mgr || !tenant_id) return GV_QUOTA_ERROR;

    pthread_mutex_lock(&mgr->mutex);

    int idx = find_tenant(mgr, tenant_id);
    if (idx < 0) {
        pthread_mutex_unlock(&mgr->mutex);
        return GV_QUOTA_ERROR;
    }

    QuotaTenantEntry *entry = &mgr->tenants[idx];
    const GV_QuotaConfig *cfg = &entry->config;

    /* Reserved capacity counts against the cap exactly like recorded usage. */
    if (cfg->max_vectors > 0 &&
        entry->usage.current_vectors + entry->reserved_vectors + vector_count
            > cfg->max_vectors) {
        entry->usage.total_rejected++;
        pthread_mutex_unlock(&mgr->mutex);
        return GV_QUOTA_EXCEEDED;
    }
    if (cfg->max_memory_bytes > 0 &&
        entry->usage.current_memory_bytes >= cfg->max_memory_bytes) {
        entry->usage.total_rejected++;
        pthread_mutex_unlock(&mgr->mutex);
        return GV_QUOTA_EXCEEDED;
    }
    if (cfg->max_storage_bytes > 0 &&
        entry->usage.current_storage_bytes >= cfg->max_storage_bytes) {
        entry->usage.total_rejected++;
        pthread_mutex_unlock(&mgr->mutex);
        return GV_QUOTA_EXCEEDED;
    }
    if (cfg->max_ips > 0.0 &&
        !token_bucket_try_consume(&entry->ips_bucket, (double)vector_count)) {
        entry->usage.total_throttled++;
        pthread_mutex_unlock(&mgr->mutex);
        return GV_QUOTA_THROTTLED;
    }

    /* Claim it under the same lock that admitted it. */
    entry->reserved_vectors += vector_count;

    pthread_mutex_unlock(&mgr->mutex);
    return GV_QUOTA_OK;
}

int gv_quota_commit_insert(GV_QuotaManager *mgr, const char *tenant_id,
                           size_t count, size_t bytes) {
    if (!mgr || !tenant_id) return -1;

    pthread_mutex_lock(&mgr->mutex);
    int idx = find_tenant(mgr, tenant_id);
    if (idx < 0) {
        pthread_mutex_unlock(&mgr->mutex);
        return -1;
    }
    QuotaTenantEntry *entry = &mgr->tenants[idx];

    /* Saturate rather than wrap if a caller commits more than it reserved. */
    entry->reserved_vectors -= (count <= entry->reserved_vectors)
                                 ? count : entry->reserved_vectors;

    entry->usage.current_vectors       += count;
    entry->usage.current_memory_bytes  += bytes;
    entry->usage.current_storage_bytes += bytes;
    entry->usage.current_ips = token_bucket_current_rate(&entry->ips_bucket);

    pthread_mutex_unlock(&mgr->mutex);
    return 0;
}

int gv_quota_release_insert(GV_QuotaManager *mgr, const char *tenant_id, size_t count) {
    if (!mgr || !tenant_id) return -1;

    pthread_mutex_lock(&mgr->mutex);
    int idx = find_tenant(mgr, tenant_id);
    if (idx < 0) {
        pthread_mutex_unlock(&mgr->mutex);
        return -1;
    }
    QuotaTenantEntry *entry = &mgr->tenants[idx];
    entry->reserved_vectors -= (count <= entry->reserved_vectors)
                                 ? count : entry->reserved_vectors;
    pthread_mutex_unlock(&mgr->mutex);
    return 0;
}
