/*
 * value_store.c - WiscKey key-value separation over the append-only value log.
 *
 * An open-addressing hash map holds key -> current vlog offset. Values live in
 * the vlog; updates/deletes orphan old records, which value_store_gc() reclaims.
 */

#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include "storage/value_store.h"
#include "storage/vlog.h"
#include "core/utils.h"

/* A map slot. state: 0=empty, 1=occupied, 2=tombstone (deleted, may probe past). */
typedef struct {
    uint64_t key;
    uint64_t offset;   /* vlog offset of the current value */
    uint8_t  state;
} VSSlot;

struct GV_ValueStore {
    GV_VLog  *vlog;
    VSSlot   *slots;
    size_t    cap;      /* power of two */
    size_t    count;    /* live (occupied) entries */
    size_t    tombstones;
    pthread_mutex_t mutex;
};

/* Each vlog record wraps the value with a self-describing prefix so the in-memory
 * key->offset map can be rebuilt by scanning the log after a reopen:
 *   [u64 key little-endian][u8 type: 0=put 1=delete][value bytes] */
#define VS_PREFIX 9
#define VS_TYPE_PUT 0
#define VS_TYPE_DEL 1

/* Append a wrapped record to the log; returns its offset via @p off_out. */
static int vs_write_record(GV_ValueStore *vs, uint64_t key, uint8_t type,
                           const void *value, size_t len, uint64_t *off_out) {
    /* Reject a length that would overflow VS_PREFIX + len (a tiny allocation the
     * following memcpy would then overrun). vlog_append enforces its own upper
     * bound; this only guards the size arithmetic here. */
    if (len > SIZE_MAX - VS_PREFIX) return -1;
    unsigned char *rec = (unsigned char *)malloc(VS_PREFIX + len);
    if (!rec) return -1;
    gv_put_u64(rec, key);
    rec[8] = type;
    if (len) memcpy(rec + VS_PREFIX, value, len);
    int rc = vlog_append(vs->vlog, rec, VS_PREFIX + len, off_out);
    free(rec);
    return rc;
}

static int vs_resize(GV_ValueStore *vs, size_t new_cap) {
    VSSlot *ns = (VSSlot *)calloc(new_cap, sizeof(VSSlot));
    if (!ns) return -1;
    for (size_t i = 0; i < vs->cap; i++) {
        if (vs->slots[i].state != 1) continue;
        uint64_t h = gv_mix64(vs->slots[i].key) & (new_cap - 1);
        while (ns[h].state == 1) h = (h + 1) & (new_cap - 1);
        ns[h] = vs->slots[i];
    }
    free(vs->slots);
    vs->slots = ns;
    vs->cap = new_cap;
    vs->tombstones = 0;
    return 0;
}

/* Find the slot for @p key. Returns index of the occupied slot, or SIZE_MAX.
 * If @p insert_at is non-NULL, stores the index where the key could be inserted. */
static size_t vs_find(const GV_ValueStore *vs, uint64_t key, size_t *insert_at) {
    size_t mask = vs->cap - 1;
    size_t h = gv_mix64(key) & mask;
    size_t first_free = SIZE_MAX;
    for (size_t probes = 0; probes <= vs->cap; probes++) {
        VSSlot *s = &vs->slots[h];
        if (s->state == 0) {
            if (insert_at) *insert_at = (first_free != SIZE_MAX) ? first_free : h;
            return SIZE_MAX;
        }
        if (s->state == 2) {
            if (first_free == SIZE_MAX) first_free = h;
        } else if (s->key == key) {
            if (insert_at) *insert_at = h;
            return h;
        }
        h = (h + 1) & mask;
    }
    if (insert_at) *insert_at = first_free;
    return SIZE_MAX;
}

/* Map upsert/erase (caller holds the mutex, or is single-threaded during open). */
static int vs_map_upsert(GV_ValueStore *vs, uint64_t key, uint64_t off) {
    if ((vs->count + vs->tombstones + 1) * 10 >= vs->cap * 7)
        if (vs_resize(vs, vs->cap * 2) != 0) return -1;
    size_t at = SIZE_MAX;
    size_t found = vs_find(vs, key, &at);
    if (found != SIZE_MAX) { vs->slots[found].offset = off; return 0; }
    if (at == SIZE_MAX) return -1;
    if (vs->slots[at].state == 2 && vs->tombstones > 0) vs->tombstones--;
    vs->slots[at].key = key; vs->slots[at].offset = off; vs->slots[at].state = 1;
    vs->count++;
    return 0;
}
static void vs_map_erase(GV_ValueStore *vs, uint64_t key) {
    size_t idx = vs_find(vs, key, NULL);
    if (idx == SIZE_MAX) return;
    vs->slots[idx].state = 2;
    vs->count--; vs->tombstones++;
}

/* Rebuild the key->offset map by replaying the log in append order (last wins). */
static int vs_rebuild_cb(uint64_t offset, const void *value, size_t len, void *ctx) {
    GV_ValueStore *vs = (GV_ValueStore *)ctx;
    if (len < VS_PREFIX) return 0;   /* skip malformed record */
    const unsigned char *p = (const unsigned char *)value;
    uint64_t key = gv_get_u64(p);
    if (p[8] == VS_TYPE_DEL) vs_map_erase(vs, key);
    else vs_map_upsert(vs, key, offset);
    return 0;
}

GV_ValueStore *value_store_open(const char *path) {
    if (!path) return NULL;
    GV_ValueStore *vs = (GV_ValueStore *)calloc(1, sizeof(GV_ValueStore));
    if (!vs) return NULL;
    vs->cap = 16;
    vs->slots = (VSSlot *)calloc(vs->cap, sizeof(VSSlot));
    if (!vs->slots) { free(vs); return NULL; }
    vs->vlog = vlog_open(path);
    if (!vs->vlog) { free(vs->slots); free(vs); return NULL; }
    pthread_mutex_init(&vs->mutex, NULL);
    /* Reconstruct the index from the durable log (empty for a new file). */
    if (vlog_scan(vs->vlog, vs_rebuild_cb, vs) < 0) { value_store_close(vs); return NULL; }
    return vs;
}

int value_store_put(GV_ValueStore *vs, uint64_t key, const void *value, size_t len) {
    if (!vs) return -1;
    /* Append + map-upsert under one lock so concurrent puts to the same key can't
     * interleave (which would leave the map pointing at an older record than the
     * log's last write). Lock order vs->mutex -> vlog mutex matches value_store_gc. */
    pthread_mutex_lock(&vs->mutex);
    /* Reserve map capacity before the durable append so an OOM can't leave a
     * record in the log that we then report as a failed put. After this the
     * post-append vs_map_upsert cannot need to resize, so it cannot fail. */
    if ((vs->count + vs->tombstones + 1) * 10 >= vs->cap * 7) {
        if (vs_resize(vs, vs->cap * 2) != 0) { pthread_mutex_unlock(&vs->mutex); return -1; }
    }
    uint64_t off;
    int rc = vs_write_record(vs, key, VS_TYPE_PUT, value, len, &off);
    if (rc == 0) rc = vs_map_upsert(vs, key, off);
    pthread_mutex_unlock(&vs->mutex);
    return rc;
}

int value_store_get(GV_ValueStore *vs, uint64_t key, void **value_out, size_t *len_out) {
    if (!vs || !len_out) return -1;
    /* Hold the map lock across the vlog read: otherwise a concurrent GC could
     * compact the log and remap offsets, making the captured offset point at a
     * different record (wrong value or spurious CRC failure). */
    pthread_mutex_lock(&vs->mutex);
    size_t idx = vs_find(vs, key, NULL);
    if (idx == SIZE_MAX) { pthread_mutex_unlock(&vs->mutex); return -1; }
    uint64_t off = vs->slots[idx].offset;

    void *raw = NULL; size_t rawlen = 0;
    if (vlog_read(vs->vlog, off, &raw, &rawlen) != 0 || rawlen < VS_PREFIX) {
        pthread_mutex_unlock(&vs->mutex); free(raw); return -1;
    }
    size_t vlen = rawlen - VS_PREFIX;
    void *out = NULL;
    if (vlen) {
        out = malloc(vlen);
        if (!out) { pthread_mutex_unlock(&vs->mutex); free(raw); return -1; }
        memcpy(out, (unsigned char *)raw + VS_PREFIX, vlen);
    }
    pthread_mutex_unlock(&vs->mutex);
    free(raw);
    if (value_out) *value_out = out; else free(out);
    *len_out = vlen;
    return 0;
}

int value_store_delete(GV_ValueStore *vs, uint64_t key) {
    if (!vs) return -1;
    pthread_mutex_lock(&vs->mutex);
    size_t idx = vs_find(vs, key, NULL);
    if (idx == SIZE_MAX) { pthread_mutex_unlock(&vs->mutex); return -1; }
    /* Append the durable delete marker FIRST. If it fails, leave the map intact
     * (both in-memory and log still hold the key) rather than tombstone only in
     * memory - which would resurrect the key on the next reopen/rebuild. */
    uint64_t off;
    if (vs_write_record(vs, key, VS_TYPE_DEL, NULL, 0, &off) != 0) {
        pthread_mutex_unlock(&vs->mutex);
        return -1;
    }
    vs->slots[idx].state = 2;   /* tombstone; value orphaned in vlog */
    vs->count--;
    vs->tombstones++;
    pthread_mutex_unlock(&vs->mutex);
    return 0;
}

size_t   value_store_count(const GV_ValueStore *vs)     { return vs ? vs->count : 0; }
uint64_t value_store_disk_size(const GV_ValueStore *vs) { return vs ? vlog_size(vs->vlog) : 0; }
int      value_store_sync(GV_ValueStore *vs)            { return vs ? vlog_sync(vs->vlog) : -1; }

/* GC liveness: an offset is live if some occupied slot points at it. Built as a
 * sorted array of live offsets for O(log n) lookup during the vlog walk. */
typedef struct { uint64_t *offs; size_t n; } LiveOffs;
static int off_cmp(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x < y) ? -1 : (x > y) ? 1 : 0;
}
static int gc_is_live(uint64_t off, void *ctx) {
    LiveOffs *lo = (LiveOffs *)ctx;
    size_t lo_i = 0, hi = lo->n;
    while (lo_i < hi) {
        size_t mid = lo_i + (hi - lo_i) / 2;
        if (lo->offs[mid] == off) return 1;
        if (lo->offs[mid] < off) lo_i = mid + 1; else hi = mid;
    }
    return 0;
}

int value_store_gc(GV_ValueStore *vs) {
    if (!vs) return -1;
    pthread_mutex_lock(&vs->mutex);

    /* Snapshot live offsets from the map. */
    LiveOffs lo; lo.n = 0; lo.offs = NULL;
    if (vs->count > 0) {
        lo.offs = (uint64_t *)malloc(vs->count * sizeof(uint64_t));
        if (!lo.offs) { pthread_mutex_unlock(&vs->mutex); return -1; }
        for (size_t i = 0; i < vs->cap; i++)
            if (vs->slots[i].state == 1) lo.offs[lo.n++] = vs->slots[i].offset;
        qsort(lo.offs, lo.n, sizeof(uint64_t), off_cmp);
    }

    uint64_t *old_off = NULL, *new_off = NULL; size_t nrem = 0;
    int rc = vlog_gc(vs->vlog, gc_is_live, &lo, &old_off, &new_off, &nrem);
    free(lo.offs);
    if (rc != 0) { pthread_mutex_unlock(&vs->mutex); return -1; }

    /* Remap every live slot's offset old->new. Build a sorted old->new lookup. */
    for (size_t i = 0; i < vs->cap; i++) {
        if (vs->slots[i].state != 1) continue;
        uint64_t cur = vs->slots[i].offset;
        for (size_t j = 0; j < nrem; j++) {
            if (old_off[j] == cur) { vs->slots[i].offset = new_off[j]; break; }
        }
    }
    free(old_off); free(new_off);
    pthread_mutex_unlock(&vs->mutex);
    return 0;
}

void value_store_close(GV_ValueStore *vs) {
    if (!vs) return;
    if (vs->vlog) vlog_close(vs->vlog);
    pthread_mutex_destroy(&vs->mutex);
    free(vs->slots);
    free(vs);
}
