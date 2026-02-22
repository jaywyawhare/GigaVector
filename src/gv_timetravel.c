/**
 * @file gv_timetravel.c
 * @brief Time-Travel / Auto-Versioning implementation for GigaVector.
 *
 * Every mutation appends a change record to an append-only log. Point-in-time
 * queries reconstruct vector state by replaying changes backwards from the
 * latest known state. Thread safety is provided via pthread_rwlock_t.
 */

#define _POSIX_C_SOURCE 200112L

#include "gigavector/gv_timetravel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

/*  Constants  */

#define TT_MAGIC        "GVTT"
#define TT_MAGIC_LEN    4
#define TT_FILE_VERSION 1
#define TT_INIT_CAP     64

/*  Internal Types  */

/**
 * @brief Type of mutation stored in a change record.
 */
typedef enum {
    TT_CHANGE_INSERT = 0,
    TT_CHANGE_UPDATE = 1,
    TT_CHANGE_DELETE = 2
} TT_ChangeType;

/**
 * @brief A single entry in the append-only change log.
 *
 * For INSERT: old_data is NULL, new_data holds the inserted vector.
 * For UPDATE: old_data holds the previous vector, new_data holds the new one.
 * For DELETE: old_data holds the deleted vector, new_data is NULL.
 */
typedef struct {
    uint64_t     version_id;
    uint64_t     timestamp;     /* microseconds since epoch */
    TT_ChangeType type;
    size_t       index;         /* affected vector index */
    size_t       dimension;
    float       *old_data;      /* NULL for inserts */
    float       *new_data;      /* NULL for deletes */
    size_t       vector_count;  /* live vector count after this mutation */
} TT_ChangeRecord;

/**
 * @brief Time-travel manager internal structure.
 */
struct GV_TimeTravelManager {
    /* Configuration */
    GV_TimeTravelConfig config;

    /* Change log (dynamic array, append-only) */
    TT_ChangeRecord *log;
    size_t           log_count;
    size_t           log_capacity;

    /* Version counter (next version to assign) */
    uint64_t         next_version;

    /* Current live vector count (maintained incrementally) */
    size_t           current_vector_count;

    /* Reader-writer lock for thread safety */
    pthread_rwlock_t rwlock;
};

/*  Internal Helpers  */

/**
 * @brief Get current time in microseconds since epoch.
 */
static uint64_t now_microseconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

/**
 * @brief Duplicate a float array. Returns NULL if src is NULL.
 */
static float *float_dup(const float *src, size_t count)
{
    if (!src || count == 0)
        return NULL;
    float *copy = malloc(count * sizeof(float));
    if (copy)
        memcpy(copy, src, count * sizeof(float));
    return copy;
}

/**
 * @brief Ensure the change log has room for one more record.
 *
 * Must be called while the write lock is held.
 */
static int ensure_log_capacity(GV_TimeTravelManager *mgr)
{
    if (mgr->log_count < mgr->log_capacity)
        return 0;

    size_t new_cap = mgr->log_capacity == 0 ? TT_INIT_CAP : mgr->log_capacity * 2;
    TT_ChangeRecord *tmp = realloc(mgr->log, new_cap * sizeof(TT_ChangeRecord));
    if (!tmp)
        return -1;

    mgr->log = tmp;
    mgr->log_capacity = new_cap;
    return 0;
}

/**
 * @brief Compute approximate storage used by the change log in bytes.
 */
static size_t compute_storage_bytes(const GV_TimeTravelManager *mgr)
{
    size_t total = mgr->log_count * sizeof(TT_ChangeRecord);
    for (size_t i = 0; i < mgr->log_count; i++) {
        const TT_ChangeRecord *rec = &mgr->log[i];
        if (rec->old_data)
            total += rec->dimension * sizeof(float);
        if (rec->new_data)
            total += rec->dimension * sizeof(float);
    }
    return total;
}

/**
 * @brief Free a single change record's heap data.
 */
static void free_change_record(TT_ChangeRecord *rec)
{
    free(rec->old_data);
    free(rec->new_data);
    rec->old_data = NULL;
    rec->new_data = NULL;
}

/**
 * @brief Append a change record to the log (internal, assumes write lock held).
 *
 * Handles auto-GC if configured. Returns the assigned version ID, or 0 on error.
 */
static uint64_t append_change(GV_TimeTravelManager *mgr, TT_ChangeType type,
                               size_t index, const float *old_data,
                               const float *new_data, size_t dimension)
{
    if (ensure_log_capacity(mgr) != 0)
        return 0;

    uint64_t ver = mgr->next_version++;
    uint64_t ts = now_microseconds();

    /* Update live vector count */
    if (type == TT_CHANGE_INSERT)
        mgr->current_vector_count++;
    else if (type == TT_CHANGE_DELETE && mgr->current_vector_count > 0)
        mgr->current_vector_count--;

    TT_ChangeRecord *rec = &mgr->log[mgr->log_count];
    rec->version_id   = ver;
    rec->timestamp     = ts;
    rec->type          = type;
    rec->index         = index;
    rec->dimension     = dimension;
    rec->old_data      = float_dup(old_data, dimension);
    rec->new_data      = float_dup(new_data, dimension);
    rec->vector_count  = mgr->current_vector_count;

    /* Verify copies succeeded where expected */
    if ((old_data && !rec->old_data) || (new_data && !rec->new_data)) {
        free(rec->old_data);
        free(rec->new_data);
        rec->old_data = NULL;
        rec->new_data = NULL;
        /* Revert vector count change */
        if (type == TT_CHANGE_INSERT && mgr->current_vector_count > 0)
            mgr->current_vector_count--;
        else if (type == TT_CHANGE_DELETE)
            mgr->current_vector_count++;
        mgr->next_version--;
        return 0;
    }

    mgr->log_count++;
    return ver;
}

/**
 * @brief Internal GC implementation (assumes write lock held).
 *
 * Returns the number of versions removed.
 */
static int gc_internal(GV_TimeTravelManager *mgr)
{
    if (mgr->log_count <= mgr->config.gc_keep_count)
        return 0;

    int need_gc = 0;

    /* Check version count limit */
    if (mgr->config.max_versions > 0 && mgr->log_count > mgr->config.max_versions)
        need_gc = 1;

    /* Check storage limit */
    if (mgr->config.max_storage_mb > 0) {
        size_t storage = compute_storage_bytes(mgr);
        size_t limit = mgr->config.max_storage_mb * 1024ULL * 1024ULL;
        if (storage > limit)
            need_gc = 1;
    }

    if (!need_gc)
        return 0;

    /* Determine how many records to remove. Keep at least gc_keep_count. */
    size_t keep = mgr->config.gc_keep_count;
    if (keep > mgr->log_count)
        keep = mgr->log_count;

    size_t remove_count = mgr->log_count - keep;

    /* Also honour max_versions: if after keeping gc_keep_count we are still
     * over max_versions, only remove enough to reach max_versions. But we
     * must remove at least enough to get under the limit. */
    if (mgr->config.max_versions > 0 && mgr->log_count > mgr->config.max_versions) {
        size_t needed = mgr->log_count - mgr->config.max_versions;
        if (needed > remove_count)
            remove_count = needed;
        /* But never remove so many that fewer than gc_keep_count remain */
        if (mgr->log_count - remove_count < keep)
            remove_count = mgr->log_count - keep;
    }

    if (remove_count == 0)
        return 0;

    /* Free the oldest records */
    for (size_t i = 0; i < remove_count; i++) {
        free_change_record(&mgr->log[i]);
    }

    /* Shift remaining records to the front */
    size_t remaining = mgr->log_count - remove_count;
    if (remaining > 0) {
        memmove(mgr->log, mgr->log + remove_count,
                remaining * sizeof(TT_ChangeRecord));
    }
    mgr->log_count = remaining;

    return (int)remove_count;
}

/**
 * @brief Find the version ID corresponding to a timestamp.
 *
 * Returns the highest version_id whose timestamp <= the given timestamp,
 * or 0 if no such version exists.
 *
 * Must be called while a read lock (or write lock) is held.
 */
static uint64_t version_at_timestamp(const GV_TimeTravelManager *mgr, uint64_t timestamp)
{
    uint64_t best = 0;
    for (size_t i = 0; i < mgr->log_count; i++) {
        if (mgr->log[i].timestamp <= timestamp)
            best = mgr->log[i].version_id;
    }
    return best;
}

/*  Lifecycle  */

void gv_tt_config_init(GV_TimeTravelConfig *config)
{
    if (!config)
        return;
    config->max_versions   = 1000;
    config->max_storage_mb = 512;
    config->auto_gc        = 1;
    config->gc_keep_count  = 100;
}

GV_TimeTravelManager *gv_tt_create(const GV_TimeTravelConfig *config)
{
    GV_TimeTravelManager *mgr = calloc(1, sizeof(GV_TimeTravelManager));
    if (!mgr)
        return NULL;

    if (config) {
        mgr->config = *config;
    } else {
        gv_tt_config_init(&mgr->config);
    }

    mgr->log          = NULL;
    mgr->log_count    = 0;
    mgr->log_capacity = 0;
    mgr->next_version = 1;
    mgr->current_vector_count = 0;

    if (pthread_rwlock_init(&mgr->rwlock, NULL) != 0) {
        free(mgr);
        return NULL;
    }

    return mgr;
}

void gv_tt_destroy(GV_TimeTravelManager *mgr)
{
    if (!mgr)
        return;

    for (size_t i = 0; i < mgr->log_count; i++) {
        free_change_record(&mgr->log[i]);
    }
    free(mgr->log);

    pthread_rwlock_destroy(&mgr->rwlock);
    free(mgr);
}

/*  Mutation Recording  */

uint64_t gv_tt_record_insert(GV_TimeTravelManager *mgr, size_t index,
                              const float *vector, size_t dimension)
{
    if (!mgr || !vector || dimension == 0)
        return 0;

    pthread_rwlock_wrlock(&mgr->rwlock);

    uint64_t ver = append_change(mgr, TT_CHANGE_INSERT, index,
                                  NULL, vector, dimension);

    /* Auto-GC if configured */
    if (ver != 0 && mgr->config.auto_gc)
        gc_internal(mgr);

    pthread_rwlock_unlock(&mgr->rwlock);
    return ver;
}

uint64_t gv_tt_record_update(GV_TimeTravelManager *mgr, size_t index,
                              const float *old_vector, const float *new_vector,
                              size_t dimension)
{
    if (!mgr || !old_vector || !new_vector || dimension == 0)
        return 0;

    pthread_rwlock_wrlock(&mgr->rwlock);

    uint64_t ver = append_change(mgr, TT_CHANGE_UPDATE, index,
                                  old_vector, new_vector, dimension);

    if (ver != 0 && mgr->config.auto_gc)
        gc_internal(mgr);

    pthread_rwlock_unlock(&mgr->rwlock);
    return ver;
}

uint64_t gv_tt_record_delete(GV_TimeTravelManager *mgr, size_t index,
                              const float *vector, size_t dimension)
{
    if (!mgr || !vector || dimension == 0)
        return 0;

    pthread_rwlock_wrlock(&mgr->rwlock);

    uint64_t ver = append_change(mgr, TT_CHANGE_DELETE, index,
                                  vector, NULL, dimension);

    if (ver != 0 && mgr->config.auto_gc)
        gc_internal(mgr);

    pthread_rwlock_unlock(&mgr->rwlock);
    return ver;
}

/*  Point-in-Time Queries  */

int gv_tt_query_at_version(const GV_TimeTravelManager *mgr, uint64_t version_id,
                            size_t index, float *output, size_t dimension)
{
    if (!mgr || !output || dimension == 0)
        return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&mgr->rwlock);

    if (mgr->log_count == 0 || version_id == 0) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
        return -1;
    }

    /* Strategy: walk the log backwards from the most recent record.
     *
     * 1. First, determine the current state of the vector at the latest
     *    version by scanning the entire log forward for the most recent
     *    change affecting this index.
     * 2. Then, undo changes in reverse order for records whose version_id
     *    is > the target version.
     *
     * State tracking:
     *   - exists: whether the vector is considered to exist
     *   - state[]: the current vector data
     */

    float *state = malloc(dimension * sizeof(float));
    if (!state) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
        return -1;
    }

    int exists = 0;

    /* Pass 1: build the current (latest) state by scanning forward. */
    for (size_t i = 0; i < mgr->log_count; i++) {
        const TT_ChangeRecord *rec = &mgr->log[i];
        if (rec->index != index)
            continue;

        if (rec->type == TT_CHANGE_INSERT) {
            if (rec->new_data && rec->dimension == dimension) {
                memcpy(state, rec->new_data, dimension * sizeof(float));
                exists = 1;
            }
        } else if (rec->type == TT_CHANGE_UPDATE) {
            if (rec->new_data && rec->dimension == dimension) {
                memcpy(state, rec->new_data, dimension * sizeof(float));
                exists = 1;
            }
        } else if (rec->type == TT_CHANGE_DELETE) {
            exists = 0;
        }
    }

    /* Pass 2: undo changes with version_id > target, walking backwards. */
    for (size_t i = mgr->log_count; i > 0; i--) {
        const TT_ChangeRecord *rec = &mgr->log[i - 1];

        if (rec->version_id <= version_id)
            break; /* all remaining records are at or before the target */

        if (rec->index != index)
            continue;

        /* Undo this change */
        if (rec->type == TT_CHANGE_INSERT) {
            /* Undo insert: the vector did not exist before this record */
            exists = 0;
        } else if (rec->type == TT_CHANGE_UPDATE) {
            /* Undo update: revert to old_data */
            if (rec->old_data && rec->dimension == dimension) {
                memcpy(state, rec->old_data, dimension * sizeof(float));
                exists = 1;
            }
        } else if (rec->type == TT_CHANGE_DELETE) {
            /* Undo delete: the vector existed before with old_data */
            if (rec->old_data && rec->dimension == dimension) {
                memcpy(state, rec->old_data, dimension * sizeof(float));
                exists = 1;
            }
        }
    }

    int result;
    if (exists) {
        memcpy(output, state, dimension * sizeof(float));
        result = 1;
    } else {
        result = 0;
    }

    free(state);
    pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
    return result;
}

int gv_tt_query_at_timestamp(const GV_TimeTravelManager *mgr, uint64_t timestamp,
                              size_t index, float *output, size_t dimension)
{
    if (!mgr || !output || dimension == 0)
        return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&mgr->rwlock);

    uint64_t ver = version_at_timestamp(mgr, timestamp);
    if (ver == 0) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
        return 0; /* no version at or before this timestamp */
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);

    /* Delegate to version-based query (it acquires its own read lock) */
    return gv_tt_query_at_version(mgr, ver, index, output, dimension);
}

/*  Version Inspection  */

size_t gv_tt_count_at_version(const GV_TimeTravelManager *mgr, uint64_t version_id)
{
    if (!mgr)
        return 0;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&mgr->rwlock);

    /* Find the change record matching the requested version */
    for (size_t i = 0; i < mgr->log_count; i++) {
        if (mgr->log[i].version_id == version_id) {
            size_t count = mgr->log[i].vector_count;
            pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
            return count;
        }
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
    return 0;
}

uint64_t gv_tt_current_version(const GV_TimeTravelManager *mgr)
{
    if (!mgr)
        return 0;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&mgr->rwlock);

    uint64_t ver = 0;
    if (mgr->log_count > 0)
        ver = mgr->log[mgr->log_count - 1].version_id;

    pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
    return ver;
}

int gv_tt_list_versions(const GV_TimeTravelManager *mgr, GV_VersionEntry *out,
                         size_t max_count)
{
    if (!mgr || !out || max_count == 0)
        return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&mgr->rwlock);

    size_t to_write = mgr->log_count < max_count ? mgr->log_count : max_count;

    for (size_t i = 0; i < to_write; i++) {
        const TT_ChangeRecord *rec = &mgr->log[i];
        out[i].version_id   = rec->version_id;
        out[i].timestamp     = rec->timestamp;
        out[i].vector_count  = rec->vector_count;

        /* Build a human-readable description */
        const char *type_str;
        switch (rec->type) {
            case TT_CHANGE_INSERT: type_str = "insert"; break;
            case TT_CHANGE_UPDATE: type_str = "update"; break;
            case TT_CHANGE_DELETE: type_str = "delete"; break;
            default:               type_str = "unknown"; break;
        }
        memset(out[i].description, 0, sizeof(out[i].description));
        snprintf(out[i].description, sizeof(out[i].description),
                 "%s at index %zu (dim=%zu)", type_str, rec->index, rec->dimension);
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
    return (int)to_write;
}

/*  Garbage Collection  */

int gv_tt_gc(GV_TimeTravelManager *mgr)
{
    if (!mgr)
        return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);
    int removed = gc_internal(mgr);
    pthread_rwlock_unlock(&mgr->rwlock);

    return removed;
}

/*  Persistence Helpers  */

static int write_uint64(FILE *f, uint64_t v)
{
    return fwrite(&v, sizeof(v), 1, f) == 1 ? 0 : -1;
}

static int read_uint64(FILE *f, uint64_t *v)
{
    return fread(v, sizeof(*v), 1, f) == 1 ? 0 : -1;
}

static int write_size(FILE *f, size_t v)
{
    uint64_t tmp = (uint64_t)v;
    return write_uint64(f, tmp);
}

static int read_size(FILE *f, size_t *v)
{
    uint64_t tmp;
    if (read_uint64(f, &tmp) != 0)
        return -1;
    *v = (size_t)tmp;
    return 0;
}

static int write_int(FILE *f, int v)
{
    int32_t tmp = (int32_t)v;
    return fwrite(&tmp, sizeof(tmp), 1, f) == 1 ? 0 : -1;
}

static int read_int(FILE *f, int *v)
{
    int32_t tmp;
    if (fread(&tmp, sizeof(tmp), 1, f) != 1)
        return -1;
    *v = (int)tmp;
    return 0;
}

/*  Persistence  */

int gv_tt_save(const GV_TimeTravelManager *mgr, const char *path)
{
    if (!mgr || !path)
        return -1;

    FILE *f = fopen(path, "wb");
    if (!f)
        return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&mgr->rwlock);

    /* Magic */
    if (fwrite(TT_MAGIC, 1, TT_MAGIC_LEN, f) != TT_MAGIC_LEN)
        goto fail;

    /* File version */
    uint32_t file_ver = TT_FILE_VERSION;
    if (fwrite(&file_ver, sizeof(file_ver), 1, f) != 1)
        goto fail;

    /* Config */
    if (write_size(f, mgr->config.max_versions)   != 0) goto fail;
    if (write_size(f, mgr->config.max_storage_mb)  != 0) goto fail;
    if (write_int(f, mgr->config.auto_gc)          != 0) goto fail;
    if (write_size(f, mgr->config.gc_keep_count)   != 0) goto fail;

    /* State */
    if (write_uint64(f, mgr->next_version)         != 0) goto fail;
    if (write_size(f, mgr->current_vector_count)   != 0) goto fail;
    if (write_size(f, mgr->log_count)              != 0) goto fail;

    /* Change records */
    for (size_t i = 0; i < mgr->log_count; i++) {
        const TT_ChangeRecord *rec = &mgr->log[i];

        if (write_uint64(f, rec->version_id) != 0) goto fail;
        if (write_uint64(f, rec->timestamp)  != 0) goto fail;
        if (write_int(f, (int)rec->type)     != 0) goto fail;
        if (write_size(f, rec->index)        != 0) goto fail;
        if (write_size(f, rec->dimension)    != 0) goto fail;
        if (write_size(f, rec->vector_count) != 0) goto fail;

        /* Flags indicating presence of old_data and new_data */
        uint8_t has_old = rec->old_data ? 1 : 0;
        uint8_t has_new = rec->new_data ? 1 : 0;
        if (fwrite(&has_old, 1, 1, f) != 1) goto fail;
        if (fwrite(&has_new, 1, 1, f) != 1) goto fail;

        if (has_old) {
            size_t bytes = rec->dimension * sizeof(float);
            if (fwrite(rec->old_data, 1, bytes, f) != bytes) goto fail;
        }
        if (has_new) {
            size_t bytes = rec->dimension * sizeof(float);
            if (fwrite(rec->new_data, 1, bytes, f) != bytes) goto fail;
        }
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
    fclose(f);
    return 0;

fail:
    pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
    fclose(f);
    return -1;
}

GV_TimeTravelManager *gv_tt_load(const char *path)
{
    if (!path)
        return NULL;

    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;

    /* Read and verify magic */
    char magic[TT_MAGIC_LEN];
    if (fread(magic, 1, TT_MAGIC_LEN, f) != TT_MAGIC_LEN)
        goto fail_early;
    if (memcmp(magic, TT_MAGIC, TT_MAGIC_LEN) != 0)
        goto fail_early;

    /* Read file version */
    uint32_t file_ver;
    if (fread(&file_ver, sizeof(file_ver), 1, f) != 1)
        goto fail_early;
    if (file_ver != TT_FILE_VERSION)
        goto fail_early;

    /* Read config */
    GV_TimeTravelConfig config;
    if (read_size(f, &config.max_versions)   != 0) goto fail_early;
    if (read_size(f, &config.max_storage_mb)  != 0) goto fail_early;
    if (read_int(f, &config.auto_gc)          != 0) goto fail_early;
    if (read_size(f, &config.gc_keep_count)   != 0) goto fail_early;

    /* Create the manager */
    GV_TimeTravelManager *mgr = gv_tt_create(&config);
    if (!mgr)
        goto fail_early;

    /* Read state */
    if (read_uint64(f, &mgr->next_version)       != 0) goto fail;
    if (read_size(f, &mgr->current_vector_count)  != 0) goto fail;

    size_t record_count;
    if (read_size(f, &record_count)                != 0) goto fail;

    /* Read change records */
    for (size_t i = 0; i < record_count; i++) {
        if (ensure_log_capacity(mgr) != 0)
            goto fail;

        TT_ChangeRecord *rec = &mgr->log[mgr->log_count];
        memset(rec, 0, sizeof(*rec));

        if (read_uint64(f, &rec->version_id) != 0) goto fail;
        if (read_uint64(f, &rec->timestamp)  != 0) goto fail;

        int type_int;
        if (read_int(f, &type_int)           != 0) goto fail;
        rec->type = (TT_ChangeType)type_int;

        if (read_size(f, &rec->index)        != 0) goto fail;
        if (read_size(f, &rec->dimension)    != 0) goto fail;
        if (read_size(f, &rec->vector_count) != 0) goto fail;

        uint8_t has_old, has_new;
        if (fread(&has_old, 1, 1, f) != 1) goto fail;
        if (fread(&has_new, 1, 1, f) != 1) goto fail;

        if (has_old) {
            size_t bytes = rec->dimension * sizeof(float);
            rec->old_data = malloc(bytes);
            if (!rec->old_data) goto fail;
            if (fread(rec->old_data, 1, bytes, f) != bytes) goto fail;
        }
        if (has_new) {
            size_t bytes = rec->dimension * sizeof(float);
            rec->new_data = malloc(bytes);
            if (!rec->new_data) goto fail;
            if (fread(rec->new_data, 1, bytes, f) != bytes) goto fail;
        }

        mgr->log_count++;
    }

    fclose(f);
    return mgr;

fail:
    gv_tt_destroy(mgr);
fail_early:
    fclose(f);
    return NULL;
}
