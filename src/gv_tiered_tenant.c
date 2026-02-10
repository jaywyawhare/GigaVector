#define _POSIX_C_SOURCE 200809L

/**
 * @file gv_tiered_tenant.c
 * @brief Tiered multitenancy implementation for GigaVector.
 *
 * Manages tenants across shared, dedicated, and premium tiers with
 * automatic promotion/demotion based on usage thresholds.  Uses a
 * hash table keyed by tenant_id for O(1) lookups.  Thread-safe via
 * pthread_rwlock_t.
 */

#include "gigavector/gv_tiered_tenant.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>
#include <time.h>

/* ============================================================================
 * Constants
 * ============================================================================ */

#define TENANT_ID_MAX_LEN   128
#define HASH_BUCKETS        1024
#define QPS_WINDOW_SECONDS  60       /**< Sliding window for QPS averaging. */
#define QPS_SLOTS           60       /**< One slot per second. */
#define DEMOTE_DAYS         7        /**< Days below threshold before demotion. */
#define DEMOTE_SECONDS      (DEMOTE_DAYS * 86400ULL)
#define DEMOTE_RATIO        0.5      /**< Must be below 50% of lower threshold. */

/* Binary persistence magic / version */
#define TIERED_MAGIC        "GVTIER"
#define TIERED_MAGIC_LEN    6
#define TIERED_VERSION      1

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

/**
 * @brief Sliding window QPS tracker.
 *
 * Keeps per-second query counts over the last QPS_WINDOW_SECONDS seconds.
 */
typedef struct {
    uint32_t counts[QPS_SLOTS];   /**< Query count per slot. */
    uint64_t base_time;           /**< Epoch second of slot 0. */
    uint64_t total;               /**< Running total in window. */
} QPSTracker;

/**
 * @brief Per-tenant entry in the hash table.
 */
typedef struct TenantEntry {
    char             tenant_id[TENANT_ID_MAX_LEN];
    int              active;          /**< 1 if slot is in use. */
    GV_TenantTier    tier;
    size_t           vector_count;
    size_t           memory_bytes;
    uint64_t         created_at;      /**< Epoch seconds. */
    uint64_t         last_active;     /**< Epoch seconds. */
    QPSTracker       qps;
    struct TenantEntry *next;         /**< Hash chain pointer. */
} TenantEntry;

/**
 * @brief Tiered tenant manager.
 */
struct GV_TieredManager {
    GV_TieredTenantConfig config;
    TenantEntry          *buckets[HASH_BUCKETS];
    size_t                tenant_count;
    pthread_rwlock_t      rwlock;
};

/* ============================================================================
 * Helpers
 * ============================================================================ */

/**
 * @brief Return current wall-clock time in epoch seconds.
 */
static uint64_t now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec;
}

/**
 * @brief djb2 hash of a string, reduced to HASH_BUCKETS range.
 */
static size_t hash_tenant_id(const char *id) {
    unsigned long h = 5381;
    int c;
    while ((c = (unsigned char)*id++) != 0) {
        h = ((h << 5) + h) + (unsigned long)c;  /* h * 33 + c */
    }
    return h % HASH_BUCKETS;
}

/**
 * @brief Find a tenant entry in the hash table.
 * @note Caller must hold at least a read lock.
 */
static TenantEntry *find_tenant(const GV_TieredManager *mgr, const char *tenant_id) {
    size_t idx = hash_tenant_id(tenant_id);
    TenantEntry *e = mgr->buckets[idx];
    while (e) {
        if (e->active && strncmp(e->tenant_id, tenant_id, TENANT_ID_MAX_LEN) == 0) {
            return e;
        }
        e = e->next;
    }
    return NULL;
}

/* ============================================================================
 * QPS Tracker
 * ============================================================================ */

static void qps_tracker_init(QPSTracker *q) {
    memset(q, 0, sizeof(*q));
}

/**
 * @brief Advance the sliding window to the current time, clearing stale slots.
 */
static void qps_tracker_advance(QPSTracker *q, uint64_t now) {
    if (q->base_time == 0) {
        q->base_time = now;
        return;
    }

    uint64_t elapsed = now - q->base_time;
    if (elapsed == 0) return;

    if (elapsed >= QPS_SLOTS) {
        /* Entire window expired -- reset everything */
        memset(q->counts, 0, sizeof(q->counts));
        q->total     = 0;
        q->base_time = now;
        return;
    }

    /* Shift window: clear slots that have fallen outside the window */
    for (uint64_t i = 0; i < elapsed; i++) {
        size_t slot = (size_t)((q->base_time + i) % QPS_SLOTS);
        q->total -= q->counts[slot];
        q->counts[slot] = 0;
    }
    q->base_time = now;
}

/**
 * @brief Record one query in the sliding window.
 */
static void qps_tracker_record(QPSTracker *q) {
    uint64_t now = now_seconds();
    qps_tracker_advance(q, now);
    size_t slot = (size_t)(now % QPS_SLOTS);
    q->counts[slot]++;
    q->total++;
}

/**
 * @brief Compute average QPS over the sliding window.
 */
static double qps_tracker_average(QPSTracker *q) {
    uint64_t now = now_seconds();
    qps_tracker_advance(q, now);
    return (double)q->total / (double)QPS_SLOTS;
}

/* ============================================================================
 * Configuration
 * ============================================================================ */

void gv_tiered_config_init(GV_TieredTenantConfig *config) {
    if (!config) return;
    memset(config, 0, sizeof(*config));
    config->thresholds.shared_max_vectors      = 10000;
    config->thresholds.dedicated_max_vectors   = 1000000;
    config->thresholds.shared_max_memory_mb    = 64;
    config->thresholds.dedicated_max_memory_mb = 1024;
    config->auto_promote       = 1;
    config->auto_demote        = 0;
    config->max_shared_tenants = 1000;
    config->max_total_tenants  = 10000;
}

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

GV_TieredManager *gv_tiered_create(const GV_TieredTenantConfig *config) {
    GV_TieredManager *mgr = calloc(1, sizeof(GV_TieredManager));
    if (!mgr) return NULL;

    if (pthread_rwlock_init(&mgr->rwlock, NULL) != 0) {
        free(mgr);
        return NULL;
    }

    if (config) {
        mgr->config = *config;
    } else {
        gv_tiered_config_init(&mgr->config);
    }

    return mgr;
}

void gv_tiered_destroy(GV_TieredManager *mgr) {
    if (!mgr) return;

    /* Free all hash chain entries */
    for (size_t i = 0; i < HASH_BUCKETS; i++) {
        TenantEntry *e = mgr->buckets[i];
        while (e) {
            TenantEntry *next = e->next;
            free(e);
            e = next;
        }
    }

    pthread_rwlock_destroy(&mgr->rwlock);
    free(mgr);
}

/* ============================================================================
 * Tenant Operations
 * ============================================================================ */

int gv_tiered_add_tenant(GV_TieredManager *mgr, const char *tenant_id,
                          GV_TenantTier initial_tier) {
    if (!mgr || !tenant_id) return -1;
    if (strlen(tenant_id) == 0 || strlen(tenant_id) >= TENANT_ID_MAX_LEN) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    /* Check capacity */
    if (mgr->tenant_count >= mgr->config.max_total_tenants) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    /* Check shared tier capacity */
    if (initial_tier == GV_TIER_SHARED) {
        size_t shared_count = 0;
        for (size_t i = 0; i < HASH_BUCKETS; i++) {
            for (TenantEntry *e = mgr->buckets[i]; e; e = e->next) {
                if (e->active && e->tier == GV_TIER_SHARED) shared_count++;
            }
        }
        if (shared_count >= mgr->config.max_shared_tenants) {
            pthread_rwlock_unlock(&mgr->rwlock);
            return -1;
        }
    }

    /* Check for duplicate */
    if (find_tenant(mgr, tenant_id)) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    /* Allocate new entry */
    TenantEntry *entry = calloc(1, sizeof(TenantEntry));
    if (!entry) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    strncpy(entry->tenant_id, tenant_id, TENANT_ID_MAX_LEN - 1);
    entry->tenant_id[TENANT_ID_MAX_LEN - 1] = '\0';
    entry->active      = 1;
    entry->tier        = initial_tier;
    entry->created_at  = now_seconds();
    entry->last_active = entry->created_at;
    qps_tracker_init(&entry->qps);

    /* Insert at head of hash chain */
    size_t idx = hash_tenant_id(tenant_id);
    entry->next = mgr->buckets[idx];
    mgr->buckets[idx] = entry;
    mgr->tenant_count++;

    pthread_rwlock_unlock(&mgr->rwlock);
    return 0;
}

int gv_tiered_remove_tenant(GV_TieredManager *mgr, const char *tenant_id) {
    if (!mgr || !tenant_id) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    size_t idx = hash_tenant_id(tenant_id);
    TenantEntry *prev = NULL;
    TenantEntry *e = mgr->buckets[idx];

    while (e) {
        if (e->active && strncmp(e->tenant_id, tenant_id, TENANT_ID_MAX_LEN) == 0) {
            /* Unlink from chain */
            if (prev) {
                prev->next = e->next;
            } else {
                mgr->buckets[idx] = e->next;
            }
            free(e);
            mgr->tenant_count--;
            pthread_rwlock_unlock(&mgr->rwlock);
            return 0;
        }
        prev = e;
        e = e->next;
    }

    pthread_rwlock_unlock(&mgr->rwlock);
    return -1;  /* Not found */
}

int gv_tiered_promote(GV_TieredManager *mgr, const char *tenant_id,
                       GV_TenantTier new_tier) {
    if (!mgr || !tenant_id) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    TenantEntry *entry = find_tenant(mgr, tenant_id);
    if (!entry) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    entry->tier        = new_tier;
    entry->last_active = now_seconds();

    pthread_rwlock_unlock(&mgr->rwlock);
    return 0;
}

int gv_tiered_get_info(const GV_TieredManager *mgr, const char *tenant_id,
                        GV_TenantInfo *info) {
    if (!mgr || !tenant_id || !info) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&mgr->rwlock);

    TenantEntry *entry = find_tenant(mgr, tenant_id);
    if (!entry) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
        return -1;
    }

    info->tenant_id    = entry->tenant_id;
    info->tier         = entry->tier;
    info->vector_count = entry->vector_count;
    info->memory_bytes = entry->memory_bytes;
    info->created_at   = entry->created_at;
    info->last_active  = entry->last_active;
    info->qps_avg      = qps_tracker_average(&entry->qps);

    pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
    return 0;
}

/* ============================================================================
 * Usage Tracking
 * ============================================================================ */

int gv_tiered_record_usage(GV_TieredManager *mgr, const char *tenant_id,
                            size_t vectors_delta, size_t memory_delta) {
    if (!mgr || !tenant_id) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    TenantEntry *entry = find_tenant(mgr, tenant_id);
    if (!entry) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    entry->vector_count += vectors_delta;
    entry->memory_bytes += memory_delta;
    entry->last_active   = now_seconds();

    /* Record a query hit for QPS tracking */
    qps_tracker_record(&entry->qps);

    pthread_rwlock_unlock(&mgr->rwlock);
    return 0;
}

/* ============================================================================
 * Auto-Promotion / Demotion
 * ============================================================================ */

int gv_tiered_check_promote(GV_TieredManager *mgr) {
    if (!mgr) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    const GV_TierThresholds *th = &mgr->config.thresholds;
    int promoted = 0;
    uint64_t now = now_seconds();

    size_t shared_max_mem_bytes    = th->shared_max_memory_mb * 1024ULL * 1024ULL;
    size_t dedicated_max_mem_bytes = th->dedicated_max_memory_mb * 1024ULL * 1024ULL;

    for (size_t i = 0; i < HASH_BUCKETS; i++) {
        for (TenantEntry *e = mgr->buckets[i]; e; e = e->next) {
            if (!e->active) continue;

            /* --- Auto-promotion ------------------------------------------ */
            if (mgr->config.auto_promote) {
                if (e->tier == GV_TIER_SHARED) {
                    if (e->vector_count > th->shared_max_vectors ||
                        e->memory_bytes > shared_max_mem_bytes) {
                        e->tier = GV_TIER_DEDICATED;
                        e->last_active = now;
                        promoted++;
                        continue;  /* Skip demotion check for just-promoted */
                    }
                }
                if (e->tier == GV_TIER_DEDICATED) {
                    if (e->vector_count > th->dedicated_max_vectors ||
                        e->memory_bytes > dedicated_max_mem_bytes) {
                        e->tier = GV_TIER_PREMIUM;
                        e->last_active = now;
                        promoted++;
                        continue;
                    }
                }
            }

            /* --- Auto-demotion ------------------------------------------- */
            if (mgr->config.auto_demote) {
                uint64_t inactive_duration = (now > e->last_active)
                                             ? (now - e->last_active) : 0;

                if (e->tier == GV_TIER_DEDICATED) {
                    /* Demote to shared if below 50% of shared thresholds
                       for DEMOTE_DAYS. */
                    size_t vec_thresh = (size_t)((double)th->shared_max_vectors * DEMOTE_RATIO);
                    size_t mem_thresh = (size_t)((double)shared_max_mem_bytes * DEMOTE_RATIO);

                    if (e->vector_count < vec_thresh &&
                        e->memory_bytes < mem_thresh &&
                        inactive_duration >= DEMOTE_SECONDS) {
                        e->tier = GV_TIER_SHARED;
                        e->last_active = now;
                        promoted++;
                    }
                } else if (e->tier == GV_TIER_PREMIUM) {
                    /* Demote to dedicated if below 50% of dedicated thresholds
                       for DEMOTE_DAYS. */
                    size_t vec_thresh = (size_t)((double)th->dedicated_max_vectors * DEMOTE_RATIO);
                    size_t mem_thresh = (size_t)((double)dedicated_max_mem_bytes * DEMOTE_RATIO);

                    if (e->vector_count < vec_thresh &&
                        e->memory_bytes < mem_thresh &&
                        inactive_duration >= DEMOTE_SECONDS) {
                        e->tier = GV_TIER_DEDICATED;
                        e->last_active = now;
                        promoted++;
                    }
                }
            }
        }
    }

    pthread_rwlock_unlock(&mgr->rwlock);
    return promoted;
}

/* ============================================================================
 * Enumeration
 * ============================================================================ */

int gv_tiered_list_tenants(const GV_TieredManager *mgr, GV_TenantTier tier,
                            GV_TenantInfo *out, size_t max_count) {
    if (!mgr || !out || max_count == 0) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&mgr->rwlock);

    int count = 0;
    for (size_t i = 0; i < HASH_BUCKETS && (size_t)count < max_count; i++) {
        for (TenantEntry *e = mgr->buckets[i]; e && (size_t)count < max_count; e = e->next) {
            if (!e->active) continue;
            if (e->tier != tier) continue;

            out[count].tenant_id    = e->tenant_id;
            out[count].tier         = e->tier;
            out[count].vector_count = e->vector_count;
            out[count].memory_bytes = e->memory_bytes;
            out[count].created_at   = e->created_at;
            out[count].last_active  = e->last_active;
            out[count].qps_avg      = qps_tracker_average(&e->qps);
            count++;
        }
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
    return count;
}

size_t gv_tiered_tenant_count(const GV_TieredManager *mgr) {
    if (!mgr) return 0;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&mgr->rwlock);
    size_t count = mgr->tenant_count;
    pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);

    return count;
}

/* ============================================================================
 * Binary Persistence
 * ============================================================================ */

/**
 * @brief On-disk header for the tiered tenant file.
 */
typedef struct {
    char     magic[TIERED_MAGIC_LEN];
    uint32_t version;
    uint32_t tenant_count;
    /* Followed by GV_TieredTenantConfig, then tenant_count TenantRecords */
} TieredFileHeader;

/**
 * @brief On-disk per-tenant record.
 */
typedef struct {
    char     tenant_id[TENANT_ID_MAX_LEN];
    uint32_t tier;
    uint64_t vector_count;
    uint64_t memory_bytes;
    uint64_t created_at;
    uint64_t last_active;
} TenantRecord;

int gv_tiered_save(const GV_TieredManager *mgr, const char *path) {
    if (!mgr || !path) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&mgr->rwlock);

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
        return -1;
    }

    /* Write header */
    TieredFileHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, TIERED_MAGIC, TIERED_MAGIC_LEN);
    hdr.version      = TIERED_VERSION;
    hdr.tenant_count = (uint32_t)mgr->tenant_count;

    if (fwrite(&hdr, sizeof(hdr), 1, fp) != 1) goto fail;

    /* Write config */
    if (fwrite(&mgr->config, sizeof(GV_TieredTenantConfig), 1, fp) != 1) goto fail;

    /* Write each tenant record */
    for (size_t i = 0; i < HASH_BUCKETS; i++) {
        for (TenantEntry *e = mgr->buckets[i]; e; e = e->next) {
            if (!e->active) continue;

            TenantRecord rec;
            memset(&rec, 0, sizeof(rec));
            memcpy(rec.tenant_id, e->tenant_id, TENANT_ID_MAX_LEN);
            rec.tier         = (uint32_t)e->tier;
            rec.vector_count = (uint64_t)e->vector_count;
            rec.memory_bytes = (uint64_t)e->memory_bytes;
            rec.created_at   = e->created_at;
            rec.last_active  = e->last_active;

            if (fwrite(&rec, sizeof(rec), 1, fp) != 1) goto fail;
        }
    }

    fclose(fp);
    pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
    return 0;

fail:
    fclose(fp);
    pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
    return -1;
}

GV_TieredManager *gv_tiered_load(const char *path) {
    if (!path) return NULL;

    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;

    /* Read header */
    TieredFileHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, fp) != 1) goto fail;

    /* Validate magic and version */
    if (memcmp(hdr.magic, TIERED_MAGIC, TIERED_MAGIC_LEN) != 0) goto fail;
    if (hdr.version != TIERED_VERSION) goto fail;

    /* Read config */
    GV_TieredTenantConfig config;
    if (fread(&config, sizeof(config), 1, fp) != 1) goto fail;

    /* Create manager with loaded config */
    GV_TieredManager *mgr = gv_tiered_create(&config);
    if (!mgr) goto fail;

    /* Read tenant records */
    for (uint32_t i = 0; i < hdr.tenant_count; i++) {
        TenantRecord rec;
        if (fread(&rec, sizeof(rec), 1, fp) != 1) {
            gv_tiered_destroy(mgr);
            goto fail;
        }

        rec.tenant_id[TENANT_ID_MAX_LEN - 1] = '\0';

        /* Allocate entry and insert directly (bypass capacity checks for load) */
        TenantEntry *entry = calloc(1, sizeof(TenantEntry));
        if (!entry) {
            gv_tiered_destroy(mgr);
            goto fail;
        }

        memcpy(entry->tenant_id, rec.tenant_id, TENANT_ID_MAX_LEN);
        entry->active       = 1;
        entry->tier         = (GV_TenantTier)rec.tier;
        entry->vector_count = (size_t)rec.vector_count;
        entry->memory_bytes = (size_t)rec.memory_bytes;
        entry->created_at   = rec.created_at;
        entry->last_active  = rec.last_active;
        qps_tracker_init(&entry->qps);

        size_t idx = hash_tenant_id(entry->tenant_id);
        entry->next = mgr->buckets[idx];
        mgr->buckets[idx] = entry;
        mgr->tenant_count++;
    }

    fclose(fp);
    return mgr;

fail:
    fclose(fp);
    return NULL;
}
