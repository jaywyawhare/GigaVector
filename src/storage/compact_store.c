#include "storage/compact_store.h"
#include "core/half.h"
#include "core/memory.h"
#include "core/types.h"

#include <stdint.h>

struct GV_CompactStore {
    size_t dimension;
    size_t count;
    size_t capacity;
    GV_CompactPrecision precision;
    void   *codes;   /* uint16_t[cap*dim] (F16) or int8_t[cap*dim] (I8) */
    float  *scales;  /* per-vector scale (I8 only; NULL for F16) */
};

static size_t elem_bytes(GV_CompactPrecision p) { return p == GV_COMPACT_F16 ? 2 : 1; }

GV_CompactStore *compact_store_create(size_t dimension, GV_CompactPrecision precision) {
    if (dimension == 0) return NULL;
    GV_CompactStore *s = (GV_CompactStore *)gv_calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->dimension = dimension;
    s->precision = precision;
    return s;
}

void compact_store_free(GV_CompactStore *s) {
    if (!s) return;
    gv_free(s->codes);
    gv_free(s->scales);
    gv_free(s);
}

static int grow(GV_CompactStore *s, size_t need) {
    if (s->capacity >= need) return 0;
    size_t nc = s->capacity ? s->capacity * 2 : 1024;
    while (nc < need) nc *= 2;
    void *nco = gv_realloc(s->codes, nc * s->dimension * elem_bytes(s->precision));
    if (!nco) return -1;
    s->codes = nco;
    if (s->precision == GV_COMPACT_I8) {
        float *ns = (float *)gv_realloc(s->scales, nc * sizeof(float));
        if (!ns) return -1;
        s->scales = ns;
    }
    s->capacity = nc;
    return 0;
}

size_t compact_store_add(GV_CompactStore *s, const float *data) {
    if (!s || !data) return (size_t)-1;
    if (grow(s, s->count + 1) != 0) return (size_t)-1;
    size_t off = s->count * s->dimension;
    if (s->precision == GV_COMPACT_F16) {
        uint16_t *c = (uint16_t *)s->codes + off;
        for (size_t i = 0; i < s->dimension; i++) c[i] = gv_f32_to_f16(data[i]);
    } else {
        int8_t *c = (int8_t *)s->codes + off;
        s->scales[s->count] = gv_i8_encode(data, c, s->dimension);
    }
    return s->count++;
}

size_t compact_store_count(const GV_CompactStore *s) { return s ? s->count : 0; }

size_t compact_store_memory_bytes(const GV_CompactStore *s) {
    if (!s) return 0;
    size_t b = s->count * s->dimension * elem_bytes(s->precision);
    if (s->precision == GV_COMPACT_I8) b += s->count * sizeof(float);
    return b;
}

int compact_store_get(const GV_CompactStore *s, size_t index, float *out) {
    if (!s || !out || index >= s->count) return -1;
    size_t off = index * s->dimension;
    if (s->precision == GV_COMPACT_F16) {
        const uint16_t *c = (const uint16_t *)s->codes + off;
        for (size_t i = 0; i < s->dimension; i++) out[i] = gv_f16_to_f32(c[i]);
    } else {
        const int8_t *c = (const int8_t *)s->codes + off;
        gv_i8_decode(c, s->scales[index], out, s->dimension);
    }
    return 0;
}

static void topk_push(float *ds, size_t *ids, size_t *n, size_t k, float d, size_t id) {
    if (*n < k) {
        size_t p = *n;
        while (p > 0 && ds[p - 1] > d) { ds[p] = ds[p - 1]; ids[p] = ids[p - 1]; p--; }
        ds[p] = d; ids[p] = id; (*n)++;
    } else if (k > 0 && d < ds[k - 1]) {
        size_t p = k - 1;
        while (p > 0 && ds[p - 1] > d) { ds[p] = ds[p - 1]; ids[p] = ids[p - 1]; p--; }
        ds[p] = d; ids[p] = id;
    }
}

int compact_store_search(const GV_CompactStore *s, const float *query, size_t k,
                         GV_DistanceType metric, size_t *ids, float *dists) {
    if (!s || !query || !ids || !dists || k == 0) return -1;
    if (k > s->count) k = s->count;
    if (k == 0) return 0;

    float *buf = (float *)gv_alloc(s->dimension * sizeof(float));
    if (!buf) return -1;
    GV_Vector cand; cand.dimension = s->dimension; cand.data = buf; cand.metadata = NULL;
    GV_Vector q;    q.dimension = s->dimension; q.data = (float *)query; q.metadata = NULL;

    size_t n = 0;
    for (size_t i = 0; i < s->count; i++) {
        compact_store_get(s, i, buf);
        float d = distance(&cand, &q, metric);
        if (d < 0.0f) continue;
        topk_push(dists, ids, &n, k, d, i);
    }
    gv_free(buf);
    return (int)n;
}
