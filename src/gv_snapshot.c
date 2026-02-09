/**
 * @file gv_snapshot.c
 * @brief Point-in-time snapshot implementation for GigaVector.
 */

#define _POSIX_C_SOURCE 199309L

#include "gigavector/gv_snapshot.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ============================================================================
 * Internal structures
 * ============================================================================ */

typedef struct {
    uint64_t snapshot_id;
    uint64_t timestamp_us;
    size_t   vector_count;
    size_t   dimension;
    float   *data;
    char     label[64];
    int      active;
} GV_SnapshotEntry;

struct GV_SnapshotManager {
    GV_SnapshotEntry *entries;
    size_t            count;
    size_t            capacity;
    size_t            max_snapshots;
    uint64_t          next_id;
};

struct GV_Snapshot {
    const GV_SnapshotEntry *entry;
};

/* ============================================================================
 * Constants
 * ============================================================================ */

#define SNAPSHOT_MAGIC      "GVSNAP"
#define SNAPSHOT_MAGIC_LEN  6
#define SNAPSHOT_VERSION    1
#define INITIAL_CAPACITY    8

/* ============================================================================
 * Helpers
 * ============================================================================ */

static uint64_t now_microseconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

static int ensure_capacity(GV_SnapshotManager *mgr)
{
    if (mgr->count < mgr->capacity) {
        return 0;
    }

    size_t new_cap = mgr->capacity == 0 ? INITIAL_CAPACITY : mgr->capacity * 2;
    GV_SnapshotEntry *new_entries = realloc(mgr->entries,
                                            new_cap * sizeof(GV_SnapshotEntry));
    if (!new_entries) {
        return -1;
    }
    mgr->entries  = new_entries;
    mgr->capacity = new_cap;
    return 0;
}

static GV_SnapshotEntry *find_entry(const GV_SnapshotManager *mgr,
                                    uint64_t snapshot_id)
{
    for (size_t i = 0; i < mgr->count; i++) {
        if (mgr->entries[i].active && mgr->entries[i].snapshot_id == snapshot_id) {
            return &mgr->entries[i];
        }
    }
    return NULL;
}

/* ============================================================================
 * Manager lifecycle
 * ============================================================================ */

GV_SnapshotManager *gv_snapshot_manager_create(size_t max_snapshots)
{
    GV_SnapshotManager *mgr = calloc(1, sizeof(GV_SnapshotManager));
    if (!mgr) {
        return NULL;
    }
    mgr->max_snapshots = max_snapshots;
    mgr->next_id       = 1;
    return mgr;
}

void gv_snapshot_manager_destroy(GV_SnapshotManager *mgr)
{
    if (!mgr) {
        return;
    }
    for (size_t i = 0; i < mgr->count; i++) {
        free(mgr->entries[i].data);
    }
    free(mgr->entries);
    free(mgr);
}

/* ============================================================================
 * Snapshot creation
 * ============================================================================ */

uint64_t gv_snapshot_create(GV_SnapshotManager *mgr, size_t vector_count,
                            const float *vector_data, size_t dimension,
                            const char *label)
{
    if (!mgr) {
        return 0;
    }

    /* Count active snapshots to enforce the limit */
    size_t active_count = 0;
    for (size_t i = 0; i < mgr->count; i++) {
        if (mgr->entries[i].active) {
            active_count++;
        }
    }
    if (active_count >= mgr->max_snapshots) {
        return 0;
    }

    if (ensure_capacity(mgr) != 0) {
        return 0;
    }

    size_t total_floats = vector_count * dimension;
    float *data_copy = NULL;

    if (total_floats > 0) {
        data_copy = malloc(total_floats * sizeof(float));
        if (!data_copy) {
            return 0;
        }
        memcpy(data_copy, vector_data, total_floats * sizeof(float));
    }

    GV_SnapshotEntry *entry = &mgr->entries[mgr->count];
    entry->snapshot_id  = mgr->next_id;
    entry->timestamp_us = now_microseconds();
    entry->vector_count = vector_count;
    entry->dimension    = dimension;
    entry->data         = data_copy;
    entry->active       = 1;

    memset(entry->label, 0, sizeof(entry->label));
    if (label) {
        strncpy(entry->label, label, sizeof(entry->label) - 1);
    }

    mgr->count++;
    return mgr->next_id++;
}

/* ============================================================================
 * Snapshot open / close
 * ============================================================================ */

GV_Snapshot *gv_snapshot_open(GV_SnapshotManager *mgr, uint64_t snapshot_id)
{
    if (!mgr) {
        return NULL;
    }

    GV_SnapshotEntry *entry = find_entry(mgr, snapshot_id);
    if (!entry) {
        return NULL;
    }

    GV_Snapshot *snap = malloc(sizeof(GV_Snapshot));
    if (!snap) {
        return NULL;
    }
    snap->entry = entry;
    return snap;
}

void gv_snapshot_close(GV_Snapshot *snap)
{
    free(snap);
}

/* ============================================================================
 * Query snapshot data
 * ============================================================================ */

size_t gv_snapshot_count(const GV_Snapshot *snap)
{
    if (!snap || !snap->entry) {
        return 0;
    }
    return snap->entry->vector_count;
}

const float *gv_snapshot_get_vector(const GV_Snapshot *snap, size_t index)
{
    if (!snap || !snap->entry) {
        return NULL;
    }
    if (index >= snap->entry->vector_count) {
        return NULL;
    }
    return snap->entry->data + (index * snap->entry->dimension);
}

size_t gv_snapshot_dimension(const GV_Snapshot *snap)
{
    if (!snap || !snap->entry) {
        return 0;
    }
    return snap->entry->dimension;
}

/* ============================================================================
 * Management: list, delete
 * ============================================================================ */

int gv_snapshot_list(const GV_SnapshotManager *mgr, GV_SnapshotInfo *infos,
                     size_t max_infos)
{
    if (!mgr) {
        return -1;
    }

    size_t written = 0;
    for (size_t i = 0; i < mgr->count && written < max_infos; i++) {
        if (!mgr->entries[i].active) {
            continue;
        }
        GV_SnapshotInfo *info = &infos[written];
        info->snapshot_id  = mgr->entries[i].snapshot_id;
        info->timestamp_us = mgr->entries[i].timestamp_us;
        info->vector_count = mgr->entries[i].vector_count;
        memset(info->label, 0, sizeof(info->label));
        memcpy(info->label, mgr->entries[i].label, sizeof(info->label));
        written++;
    }
    return (int)written;
}

int gv_snapshot_delete(GV_SnapshotManager *mgr, uint64_t snapshot_id)
{
    if (!mgr) {
        return -1;
    }

    GV_SnapshotEntry *entry = find_entry(mgr, snapshot_id);
    if (!entry) {
        return -1;
    }

    free(entry->data);
    entry->data   = NULL;
    entry->active = 0;
    return 0;
}

/* ============================================================================
 * Persistence: save / load
 * ============================================================================ */

int gv_snapshot_save(const GV_SnapshotManager *mgr, FILE *out)
{
    if (!mgr || !out) {
        return -1;
    }

    /* Write magic and version */
    if (fwrite(SNAPSHOT_MAGIC, 1, SNAPSHOT_MAGIC_LEN, out) != SNAPSHOT_MAGIC_LEN) {
        return -1;
    }
    uint32_t version = SNAPSHOT_VERSION;
    if (fwrite(&version, sizeof(version), 1, out) != 1) {
        return -1;
    }

    /* Count active entries */
    size_t active_count = 0;
    for (size_t i = 0; i < mgr->count; i++) {
        if (mgr->entries[i].active) {
            active_count++;
        }
    }

    /* Write header: active_count, max_snapshots, next_id */
    if (fwrite(&active_count, sizeof(active_count), 1, out) != 1) {
        return -1;
    }
    if (fwrite(&mgr->max_snapshots, sizeof(mgr->max_snapshots), 1, out) != 1) {
        return -1;
    }
    if (fwrite(&mgr->next_id, sizeof(mgr->next_id), 1, out) != 1) {
        return -1;
    }

    /* Write each active entry */
    for (size_t i = 0; i < mgr->count; i++) {
        const GV_SnapshotEntry *e = &mgr->entries[i];
        if (!e->active) {
            continue;
        }

        if (fwrite(&e->snapshot_id, sizeof(e->snapshot_id), 1, out) != 1) {
            return -1;
        }
        if (fwrite(&e->timestamp_us, sizeof(e->timestamp_us), 1, out) != 1) {
            return -1;
        }
        if (fwrite(&e->vector_count, sizeof(e->vector_count), 1, out) != 1) {
            return -1;
        }
        if (fwrite(&e->dimension, sizeof(e->dimension), 1, out) != 1) {
            return -1;
        }
        if (fwrite(e->label, 1, sizeof(e->label), out) != sizeof(e->label)) {
            return -1;
        }

        /* Write vector data */
        size_t total_floats = e->vector_count * e->dimension;
        if (total_floats > 0) {
            if (fwrite(e->data, sizeof(float), total_floats, out) != total_floats) {
                return -1;
            }
        }
    }

    return 0;
}

int gv_snapshot_load(GV_SnapshotManager **mgr_ptr, FILE *in)
{
    if (!mgr_ptr || !in) {
        return -1;
    }

    /* Read and verify magic */
    char magic[SNAPSHOT_MAGIC_LEN];
    if (fread(magic, 1, SNAPSHOT_MAGIC_LEN, in) != SNAPSHOT_MAGIC_LEN) {
        return -1;
    }
    if (memcmp(magic, SNAPSHOT_MAGIC, SNAPSHOT_MAGIC_LEN) != 0) {
        return -1;
    }

    /* Read version */
    uint32_t version;
    if (fread(&version, sizeof(version), 1, in) != 1) {
        return -1;
    }
    if (version != SNAPSHOT_VERSION) {
        return -1;
    }

    /* Read header */
    size_t active_count;
    size_t max_snapshots;
    uint64_t next_id;

    if (fread(&active_count, sizeof(active_count), 1, in) != 1) {
        return -1;
    }
    if (fread(&max_snapshots, sizeof(max_snapshots), 1, in) != 1) {
        return -1;
    }
    if (fread(&next_id, sizeof(next_id), 1, in) != 1) {
        return -1;
    }

    /* Create the manager */
    GV_SnapshotManager *mgr = gv_snapshot_manager_create(max_snapshots);
    if (!mgr) {
        return -1;
    }
    mgr->next_id = next_id;

    /* Read each entry */
    for (size_t i = 0; i < active_count; i++) {
        if (ensure_capacity(mgr) != 0) {
            gv_snapshot_manager_destroy(mgr);
            return -1;
        }

        GV_SnapshotEntry *e = &mgr->entries[mgr->count];
        memset(e, 0, sizeof(*e));

        if (fread(&e->snapshot_id, sizeof(e->snapshot_id), 1, in) != 1) {
            gv_snapshot_manager_destroy(mgr);
            return -1;
        }
        if (fread(&e->timestamp_us, sizeof(e->timestamp_us), 1, in) != 1) {
            gv_snapshot_manager_destroy(mgr);
            return -1;
        }
        if (fread(&e->vector_count, sizeof(e->vector_count), 1, in) != 1) {
            gv_snapshot_manager_destroy(mgr);
            return -1;
        }
        if (fread(&e->dimension, sizeof(e->dimension), 1, in) != 1) {
            gv_snapshot_manager_destroy(mgr);
            return -1;
        }
        if (fread(e->label, 1, sizeof(e->label), in) != sizeof(e->label)) {
            gv_snapshot_manager_destroy(mgr);
            return -1;
        }

        /* Read vector data */
        size_t total_floats = e->vector_count * e->dimension;
        if (total_floats > 0) {
            e->data = malloc(total_floats * sizeof(float));
            if (!e->data) {
                gv_snapshot_manager_destroy(mgr);
                return -1;
            }
            if (fread(e->data, sizeof(float), total_floats, in) != total_floats) {
                gv_snapshot_manager_destroy(mgr);
                return -1;
            }
        }

        e->active = 1;
        mgr->count++;
    }

    *mgr_ptr = mgr;
    return 0;
}
