#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include "gigavector/gv_versioning.h"

/* ---------------------------------------------------------------------------
 * Internal types
 * --------------------------------------------------------------------------- */

typedef struct {
    uint64_t version_id;
    uint64_t timestamp_us;
    size_t   count;
    size_t   dimension;
    char     label[128];
    float   *data;
    int      active;
} GV_VersionEntry;

struct GV_VersionManager {
    GV_VersionEntry *entries;
    size_t           entry_count;
    size_t           entry_capacity;
    size_t           max_versions;
    uint64_t         next_id;
};

/* ---------------------------------------------------------------------------
 * Helpers
 * --------------------------------------------------------------------------- */

static uint64_t now_microseconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

static GV_VersionEntry *find_entry(const GV_VersionManager *mgr, uint64_t version_id)
{
    if (!mgr) return NULL;
    for (size_t i = 0; i < mgr->entry_count; i++) {
        if (mgr->entries[i].active && mgr->entries[i].version_id == version_id) {
            return &mgr->entries[i];
        }
    }
    return NULL;
}

static int active_count(const GV_VersionManager *mgr)
{
    int n = 0;
    for (size_t i = 0; i < mgr->entry_count; i++) {
        if (mgr->entries[i].active) n++;
    }
    return n;
}

static void fill_info(const GV_VersionEntry *e, GV_VersionInfo *info)
{
    info->version_id     = e->version_id;
    info->timestamp_us   = e->timestamp_us;
    info->vector_count   = e->count;
    info->dimension      = e->dimension;
    info->data_size_bytes = e->count * e->dimension * sizeof(float);
    memcpy(info->label, e->label, sizeof(info->label));
}

static int ensure_capacity(GV_VersionManager *mgr)
{
    if (mgr->entry_count < mgr->entry_capacity) return 0;

    size_t new_cap = mgr->entry_capacity == 0 ? 4 : mgr->entry_capacity * 2;
    GV_VersionEntry *tmp = realloc(mgr->entries, new_cap * sizeof(GV_VersionEntry));
    if (!tmp) return -1;

    memset(tmp + mgr->entry_capacity, 0, (new_cap - mgr->entry_capacity) * sizeof(GV_VersionEntry));
    mgr->entries        = tmp;
    mgr->entry_capacity = new_cap;
    return 0;
}

/* ---------------------------------------------------------------------------
 * Public API
 * --------------------------------------------------------------------------- */

GV_VersionManager *gv_version_manager_create(size_t max_versions)
{
    if (max_versions == 0) return NULL;

    GV_VersionManager *mgr = calloc(1, sizeof(GV_VersionManager));
    if (!mgr) return NULL;

    mgr->max_versions = max_versions;
    mgr->next_id      = 1;
    return mgr;
}

void gv_version_manager_destroy(GV_VersionManager *mgr)
{
    if (!mgr) return;
    for (size_t i = 0; i < mgr->entry_count; i++) {
        free(mgr->entries[i].data);
    }
    free(mgr->entries);
    free(mgr);
}

uint64_t gv_version_create(GV_VersionManager *mgr, const float *data,
                            size_t count, size_t dimension, const char *label)
{
    if (!mgr || !data || count == 0 || dimension == 0) return 0;

    /* Reject if we have already reached the maximum number of active versions */
    if ((size_t)active_count(mgr) >= mgr->max_versions) return 0;

    if (ensure_capacity(mgr) != 0) return 0;

    size_t total_floats = count * dimension;
    float *copy = malloc(total_floats * sizeof(float));
    if (!copy) return 0;
    memcpy(copy, data, total_floats * sizeof(float));

    GV_VersionEntry *e = &mgr->entries[mgr->entry_count];
    e->version_id   = mgr->next_id++;
    e->timestamp_us = now_microseconds();
    e->count        = count;
    e->dimension    = dimension;
    e->data         = copy;
    e->active       = 1;

    memset(e->label, 0, sizeof(e->label));
    if (label) {
        strncpy(e->label, label, sizeof(e->label) - 1);
    }

    mgr->entry_count++;
    return e->version_id;
}

int gv_version_list(const GV_VersionManager *mgr, GV_VersionInfo *infos, size_t max_infos)
{
    if (!mgr || !infos || max_infos == 0) return -1;

    int written = 0;
    for (size_t i = 0; i < mgr->entry_count && (size_t)written < max_infos; i++) {
        if (mgr->entries[i].active) {
            fill_info(&mgr->entries[i], &infos[written]);
            written++;
        }
    }
    return written;
}

int gv_version_count(const GV_VersionManager *mgr)
{
    if (!mgr) return -1;
    return active_count(mgr);
}

int gv_version_get_info(const GV_VersionManager *mgr, uint64_t version_id, GV_VersionInfo *info)
{
    if (!mgr || !info) return -1;

    const GV_VersionEntry *e = find_entry(mgr, version_id);
    if (!e) return -1;

    fill_info(e, info);
    return 0;
}

float *gv_version_get_data(const GV_VersionManager *mgr, uint64_t version_id,
                            size_t *count_out, size_t *dimension_out)
{
    if (!mgr) return NULL;

    const GV_VersionEntry *e = find_entry(mgr, version_id);
    if (!e) return NULL;

    size_t total_floats = e->count * e->dimension;
    float *copy = malloc(total_floats * sizeof(float));
    if (!copy) return NULL;
    memcpy(copy, e->data, total_floats * sizeof(float));

    if (count_out)     *count_out     = e->count;
    if (dimension_out) *dimension_out = e->dimension;
    return copy;
}

int gv_version_delete(GV_VersionManager *mgr, uint64_t version_id)
{
    if (!mgr) return -1;

    GV_VersionEntry *e = find_entry(mgr, version_id);
    if (!e) return -1;

    free(e->data);
    e->data   = NULL;
    e->active = 0;
    return 0;
}

int gv_version_compare(const GV_VersionManager *mgr, uint64_t v1, uint64_t v2,
                        size_t *added, size_t *removed, size_t *modified)
{
    if (!mgr || !added || !removed || !modified) return -1;

    const GV_VersionEntry *e1 = find_entry(mgr, v1);
    const GV_VersionEntry *e2 = find_entry(mgr, v2);
    if (!e1 || !e2) return -1;

    /* Dimensions must match for a meaningful comparison */
    if (e1->dimension != e2->dimension) return -1;

    size_t dim        = e1->dimension;
    size_t min_count  = e1->count < e2->count ? e1->count : e2->count;
    size_t mod_count  = 0;
    int    total_diff = 0;

    /* Compare overlapping vectors element-wise */
    for (size_t i = 0; i < min_count; i++) {
        const float *vec1 = e1->data + i * dim;
        const float *vec2 = e2->data + i * dim;
        int differs = 0;
        for (size_t d = 0; d < dim; d++) {
            if (vec1[d] != vec2[d]) {
                differs = 1;
                break;
            }
        }
        if (differs) mod_count++;
    }

    /* Vectors present in v2 but beyond v1's count are "added" */
    *added    = (e2->count > e1->count) ? (e2->count - e1->count) : 0;
    /* Vectors present in v1 but beyond v2's count are "removed" */
    *removed  = (e1->count > e2->count) ? (e1->count - e2->count) : 0;
    *modified = mod_count;

    total_diff = (int)(*added + *removed + *modified);
    return total_diff;
}

/* ---------------------------------------------------------------------------
 * Persistence helpers
 * --------------------------------------------------------------------------- */

static int write_uint64(FILE *out, uint64_t v)
{
    return fwrite(&v, sizeof(v), 1, out) == 1 ? 0 : -1;
}

static int read_uint64(FILE *in, uint64_t *v)
{
    return fread(v, sizeof(*v), 1, in) == 1 ? 0 : -1;
}

static int write_size(FILE *out, size_t v)
{
    uint64_t tmp = (uint64_t)v;
    return write_uint64(out, tmp);
}

static int read_size(FILE *in, size_t *v)
{
    uint64_t tmp;
    if (read_uint64(in, &tmp) != 0) return -1;
    *v = (size_t)tmp;
    return 0;
}

int gv_version_save(const GV_VersionManager *mgr, FILE *out)
{
    if (!mgr || !out) return -1;

    /* Header: active entry count, max_versions, next_id */
    size_t act = (size_t)active_count(mgr);
    if (write_size(out, act)              != 0) return -1;
    if (write_size(out, mgr->max_versions) != 0) return -1;
    if (write_uint64(out, mgr->next_id)   != 0) return -1;

    for (size_t i = 0; i < mgr->entry_count; i++) {
        const GV_VersionEntry *e = &mgr->entries[i];
        if (!e->active) continue;

        if (write_uint64(out, e->version_id)   != 0) return -1;
        if (write_uint64(out, e->timestamp_us)  != 0) return -1;
        if (write_size(out, e->count)           != 0) return -1;
        if (write_size(out, e->dimension)       != 0) return -1;

        if (fwrite(e->label, 1, sizeof(e->label), out) != sizeof(e->label)) return -1;

        size_t data_bytes = e->count * e->dimension * sizeof(float);
        if (data_bytes > 0) {
            if (fwrite(e->data, 1, data_bytes, out) != data_bytes) return -1;
        }
    }

    return 0;
}

int gv_version_load(GV_VersionManager **mgr_ptr, FILE *in)
{
    if (!mgr_ptr || !in) return -1;

    size_t   act, max_ver;
    uint64_t next;

    if (read_size(in, &act)     != 0) return -1;
    if (read_size(in, &max_ver) != 0) return -1;
    if (read_uint64(in, &next)  != 0) return -1;

    GV_VersionManager *mgr = gv_version_manager_create(max_ver);
    if (!mgr) return -1;
    mgr->next_id = next;

    for (size_t i = 0; i < act; i++) {
        if (ensure_capacity(mgr) != 0) {
            gv_version_manager_destroy(mgr);
            return -1;
        }

        GV_VersionEntry *e = &mgr->entries[mgr->entry_count];
        memset(e, 0, sizeof(*e));

        if (read_uint64(in, &e->version_id)  != 0) goto fail;
        if (read_uint64(in, &e->timestamp_us) != 0) goto fail;
        if (read_size(in, &e->count)          != 0) goto fail;
        if (read_size(in, &e->dimension)      != 0) goto fail;

        if (fread(e->label, 1, sizeof(e->label), in) != sizeof(e->label)) goto fail;

        size_t data_bytes = e->count * e->dimension * sizeof(float);
        if (data_bytes > 0) {
            e->data = malloc(data_bytes);
            if (!e->data) goto fail;
            if (fread(e->data, 1, data_bytes, in) != data_bytes) goto fail;
        }

        e->active = 1;
        mgr->entry_count++;
    }

    *mgr_ptr = mgr;
    return 0;

fail:
    gv_version_manager_destroy(mgr);
    return -1;
}
