#include "features/graph_prop_index.h"
#include "core/memory.h"
#include "core/utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INITIAL_CAPACITY 64
#define LOAD_FACTOR_NUM  7
#define LOAD_FACTOR_DEN 10

typedef struct {
    char         *key;
    GV_PropValue  value;
    uint64_t     *node_ids;
    size_t        node_count;
    size_t        node_cap;
    uint64_t     *edge_ids;
    size_t        edge_count;
    size_t        edge_cap;
    uint32_t      hash;
    int           occupied;
} PropEntry;

struct GV_GraphPropIndex {
    PropEntry *entries;
    size_t     capacity;
    size_t     count;
};

static uint32_t fnv1a_hash(const char *data, size_t len)
{
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < len; i++) {
        h ^= (uint8_t)data[i];
        h *= 16777619u;
    }
    return h;
}

static uint32_t hash_prop_value(GV_PropValue v)
{
    union {
        int64_t  i;
        double   f;
        int      b;
        uint64_t u;
    } conv = {0};

    switch (v.type) {
    case GV_PROP_NULL:
        return 0;
    case GV_PROP_STRING:
        return v.as.s ? fnv1a_hash(v.as.s, strlen(v.as.s)) : 0;
    case GV_PROP_INT64:
        conv.i = v.as.i;
        return fnv1a_hash((const char *)&conv.u, sizeof(conv.u));
    case GV_PROP_FLOAT64:
        conv.f = v.as.f;
        return fnv1a_hash((const char *)&conv.u, sizeof(conv.u));
    case GV_PROP_BOOL:
        conv.b = v.as.b;
        return fnv1a_hash((const char *)&conv.u, sizeof(conv.u));
    case GV_PROP_BLOB:
        return v.as.blob.data && v.as.blob.len > 0
                   ? fnv1a_hash((const char *)v.as.blob.data, v.as.blob.len)
                   : 0;
    default:
        return 0;
    }
}

static uint32_t compute_hash(const char *key, GV_PropValue value)
{
    uint32_t h = fnv1a_hash(key, strlen(key));
    h ^= hash_prop_value(value) * 0x01000193u;
    return h;
}

static void prop_entry_free(PropEntry *e)
{
    if (!e) return;
    gv_free(e->key);
    e->key = NULL;
    gv_prop_free(&e->value);
    gv_free(e->node_ids);
    e->node_ids = NULL;
    e->node_count = 0;
    e->node_cap = 0;
    gv_free(e->edge_ids);
    e->edge_ids = NULL;
    e->edge_count = 0;
    e->edge_cap = 0;
    e->hash = 0;
    e->occupied = 0;
}

static void grow(GV_GraphPropIndex *idx)
{
    size_t old_cap = idx->capacity;
    size_t old_count = idx->count;
    PropEntry *old = idx->entries;

    idx->capacity = old_cap == 0 ? INITIAL_CAPACITY : old_cap * 2;
    idx->entries = (PropEntry *)gv_calloc(idx->capacity, sizeof(PropEntry));
    idx->count = 0;

    if (!idx->entries) {
        idx->entries = old;
        idx->capacity = old_cap;
        idx->count = old_count;
        return;
    }

    for (size_t i = 0; i < old_cap; i++) {
        if (old[i].occupied) {
            size_t pos = old[i].hash % idx->capacity;
            while (idx->entries[pos].occupied) {
                pos = (pos + 1) % idx->capacity;
            }
            idx->entries[pos] = old[i];
            idx->count++;
            memset(&old[i], 0, sizeof(PropEntry));
        }
    }
    gv_free(old);
}

static PropEntry *find_slot(const GV_GraphPropIndex *idx, uint32_t hash,
                            const char *key, GV_PropValue value)
{
    if (idx->capacity == 0) return NULL;
    size_t pos = hash % idx->capacity;
    for (size_t i = 0; i < idx->capacity; i++) {
        PropEntry *e = &idx->entries[pos];
        if (!e->occupied) return e;
        if (e->hash == hash && strcmp(e->key, key) == 0 &&
            gv_prop_compare(e->value, value) == 0) {
            return e;
        }
        pos = (pos + 1) % idx->capacity;
    }
    return NULL;
}

static PropEntry *find_existing(const GV_GraphPropIndex *idx, uint32_t hash,
                                const char *key, GV_PropValue value)
{
    if (idx->capacity == 0) return NULL;
    size_t pos = hash % idx->capacity;
    for (size_t i = 0; i < idx->capacity; i++) {
        PropEntry *e = &idx->entries[pos];
        if (!e->occupied) return NULL;
        if (e->hash == hash && strcmp(e->key, key) == 0 &&
            gv_prop_compare(e->value, value) == 0) {
            return e;
        }
        pos = (pos + 1) % idx->capacity;
    }
    return NULL;
}

static int add_id(uint64_t **ids, size_t *count, size_t *cap, uint64_t id)
{
    if (*count >= *cap) {
        size_t new_cap = *cap == 0 ? 8 : *cap * 2;
        uint64_t *tmp = (uint64_t *)gv_realloc(*ids, new_cap * sizeof(uint64_t));
        if (!tmp) return -1;
        *ids = tmp;
        *cap = new_cap;
    }
    (*ids)[(*count)++] = id;
    return 0;
}

static int remove_id(uint64_t *ids, size_t *count, uint64_t id)
{
    for (size_t i = 0; i < *count; i++) {
        if (ids[i] == id) {
            ids[i] = ids[--(*count)];
            return 0;
        }
    }
    return -1;
}

GV_GraphPropIndex *graph_prop_index_create(void)
{
    GV_GraphPropIndex *idx = (GV_GraphPropIndex *)gv_calloc(1, sizeof(GV_GraphPropIndex));
    return idx;
}

void graph_prop_index_destroy(GV_GraphPropIndex *idx)
{
    if (!idx) return;
    if (idx->entries) {
        for (size_t i = 0; i < idx->capacity; i++) {
            if (idx->entries[i].occupied) {
                prop_entry_free(&idx->entries[i]);
            }
        }
        gv_free(idx->entries);
    }
    gv_free(idx);
}

static int add_entry(GV_GraphPropIndex *idx, const char *key, GV_PropValue value,
                     uint64_t id, int is_edge)
{
    if (!idx || !key) return -1;

    if (idx->capacity == 0 ||
        idx->count >= idx->capacity * LOAD_FACTOR_NUM / LOAD_FACTOR_DEN) {
        grow(idx);
    }

    uint32_t h = compute_hash(key, value);
    PropEntry *e = find_slot(idx, h, key, value);

    if (!e) return -1;

    if (e->occupied) {
        if (is_edge) {
            return add_id(&e->edge_ids, &e->edge_count, &e->edge_cap, id);
        } else {
            return add_id(&e->node_ids, &e->node_count, &e->node_cap, id);
        }
    }

    e->key = gv_strdup(key);
    if (!e->key) return -1;
    e->value = gv_prop_clone(value);
    e->hash = h;
    e->occupied = 1;
    idx->count++;

    if (is_edge) {
        return add_id(&e->edge_ids, &e->edge_count, &e->edge_cap, id);
    } else {
        return add_id(&e->node_ids, &e->node_count, &e->node_cap, id);
    }
}

int graph_prop_index_add_node(GV_GraphPropIndex *idx, const char *key,
                              GV_PropValue value, uint64_t node_id)
{
    return add_entry(idx, key, value, node_id, 0);
}

int graph_prop_index_add_edge(GV_GraphPropIndex *idx, const char *key,
                              GV_PropValue value, uint64_t edge_id)
{
    return add_entry(idx, key, value, edge_id, 1);
}

static int remove_entry(GV_GraphPropIndex *idx, const char *key, GV_PropValue value,
                        uint64_t id, int is_edge)
{
    if (!idx || !key || idx->capacity == 0) return -1;

    uint32_t h = compute_hash(key, value);
    PropEntry *e = find_existing(idx, h, key, value);
    if (!e) return -1;

    int rc;
    if (is_edge) {
        rc = remove_id(e->edge_ids, &e->edge_count, id);
    } else {
        rc = remove_id(e->node_ids, &e->node_count, id);
    }
    if (rc != 0) return rc;

    if (e->node_count == 0 && e->edge_count == 0) {
        prop_entry_free(e);
        idx->count--;
    }
    return 0;
}

int graph_prop_index_remove_node(GV_GraphPropIndex *idx, const char *key,
                                 GV_PropValue value, uint64_t node_id)
{
    return remove_entry(idx, key, value, node_id, 0);
}

int graph_prop_index_remove_edge(GV_GraphPropIndex *idx, const char *key,
                                 GV_PropValue value, uint64_t edge_id)
{
    return remove_entry(idx, key, value, edge_id, 1);
}

static int find_entries(const GV_GraphPropIndex *idx, const char *key,
                        GV_PropValue value, uint64_t *out_ids, size_t max_count,
                        int is_edge)
{
    if (!idx || !key || idx->capacity == 0) return 0;

    uint32_t h = compute_hash(key, value);
    PropEntry *e = find_existing(idx, h, key, value);
    if (!e) return 0;

    size_t count = is_edge ? e->edge_count : e->node_count;
    const uint64_t *src = is_edge ? e->edge_ids : e->node_ids;
    size_t n = count < max_count ? count : max_count;
    if (out_ids) {
        for (size_t i = 0; i < n; i++) out_ids[i] = src[i];
    }
    return (int)n;
}

int graph_prop_index_find_exact(const GV_GraphPropIndex *idx, const char *key,
                                GV_PropValue value, uint64_t *out_ids,
                                size_t max_count)
{
    return find_entries(idx, key, value, out_ids, max_count, 0);
}

int graph_prop_index_find_edges_exact(const GV_GraphPropIndex *idx, const char *key,
                                      GV_PropValue value, uint64_t *out_ids,
                                      size_t max_count)
{
    return find_entries(idx, key, value, out_ids, max_count, 1);
}

typedef struct {
    GV_PropValue value;
    uint64_t     id;
} RangeEntry;

static int range_entry_cmp(const void *a, const void *b)
{
    const RangeEntry *ra = (const RangeEntry *)a;
    const RangeEntry *rb = (const RangeEntry *)b;
    return gv_prop_compare(ra->value, rb->value);
}

static int find_range_impl(const GV_GraphPropIndex *idx, const char *key,
                           GV_PropValue min_val, GV_PropValue max_val,
                           uint64_t *out_ids, size_t max_count, int is_edge)
{
    if (!idx || !key || idx->capacity == 0) return 0;

    /* If both bounds are NULL, no range is valid */
    int has_min = (min_val.type != GV_PROP_NULL);
    int has_max = (max_val.type != GV_PROP_NULL);
    if (!has_min && !has_max) return 0;

    size_t collect_cap = 64;
    RangeEntry *collected = (RangeEntry *)gv_alloc(collect_cap * sizeof(RangeEntry));
    if (!collected) return 0;
    size_t collected_count = 0;

    for (size_t i = 0; i < idx->capacity; i++) {
        const PropEntry *e = &idx->entries[i];
        if (!e->occupied) continue;
        if (strcmp(e->key, key) != 0) continue;

        int in_range = 1;
        if (has_min && gv_prop_compare(e->value, min_val) < 0) in_range = 0;
        if (has_max && gv_prop_compare(e->value, max_val) > 0) in_range = 0;
        if (!in_range) continue;

        uint64_t *ids = is_edge ? e->edge_ids : e->node_ids;
        size_t count = is_edge ? e->edge_count : e->node_count;

        if (collected_count + count > collect_cap) {
            collect_cap = (collected_count + count) * 2;
            RangeEntry *tmp = (RangeEntry *)gv_realloc(
                collected, collect_cap * sizeof(RangeEntry));
            if (!tmp) { gv_free(collected); return 0; }
            collected = tmp;
        }

        for (size_t j = 0; j < count; j++) {
            collected[collected_count].value = e->value;
            collected[collected_count].id = ids[j];
            collected_count++;
        }
    }

    /* Sort by value for deterministic output */
    qsort(collected, collected_count, sizeof(RangeEntry), range_entry_cmp);

    size_t n = collected_count < max_count ? collected_count : max_count;
    if (out_ids) {
        for (size_t i = 0; i < n; i++) out_ids[i] = collected[i].id;
    }

    gv_free(collected);
    return (int)n;
}

int graph_prop_index_find_range(const GV_GraphPropIndex *idx, const char *key,
                                GV_PropValue min_val, GV_PropValue max_val,
                                uint64_t *out_ids, size_t max_count)
{
    return find_range_impl(idx, key, min_val, max_val, out_ids, max_count, 0);
}

int graph_prop_index_find_edges_range(const GV_GraphPropIndex *idx, const char *key,
                                      GV_PropValue min_val, GV_PropValue max_val,
                                      uint64_t *out_ids, size_t max_count)
{
    return find_range_impl(idx, key, min_val, max_val, out_ids, max_count, 1);
}

size_t graph_prop_index_node_count(const GV_GraphPropIndex *idx)
{
    if (!idx) return 0;
    size_t total = 0;
    for (size_t i = 0; i < idx->capacity; i++) {
        if (idx->entries[i].occupied) {
            total += idx->entries[i].node_count;
        }
    }
    return total;
}

size_t graph_prop_index_edge_count(const GV_GraphPropIndex *idx)
{
    if (!idx) return 0;
    size_t total = 0;
    for (size_t i = 0; i < idx->capacity; i++) {
        if (idx->entries[i].occupied) {
            total += idx->entries[i].edge_count;
        }
    }
    return total;
}

/* Persistence: write format per entry:
 *   uint32_t hash
 *   GV_PropType type
 *   (value payload varies by type)
 *   uint64_t node_count
 *   uint64_t[] node_ids
 *   uint64_t edge_count
 *   uint64_t[] edge_ids
 *   uint32_t key_len
 *   char[] key
 */

static int write_prop_value(FILE *out, GV_PropValue v)
{
    int t = (int)v.type;
    if (fwrite(&t, sizeof(t), 1, out) != 1) return -1;
    switch (v.type) {
    case GV_PROP_NULL:
        break;
    case GV_PROP_INT64:
        if (fwrite(&v.as.i, sizeof(v.as.i), 1, out) != 1) return -1;
        break;
    case GV_PROP_FLOAT64:
        if (fwrite(&v.as.f, sizeof(v.as.f), 1, out) != 1) return -1;
        break;
    case GV_PROP_BOOL:
        if (fwrite(&v.as.b, sizeof(v.as.b), 1, out) != 1) return -1;
        break;
    case GV_PROP_STRING: {
        uint32_t len = v.as.s ? (uint32_t)strlen(v.as.s) : 0;
        if (fwrite(&len, sizeof(len), 1, out) != 1) return -1;
        if (len > 0 && fwrite(v.as.s, 1, len, out) != len) return -1;
        break;
    }
    case GV_PROP_BLOB: {
        uint32_t len = (uint32_t)v.as.blob.len;
        if (fwrite(&len, sizeof(len), 1, out) != 1) return -1;
        if (len > 0 && fwrite(v.as.blob.data, 1, len, out) != len) return -1;
        break;
    }
    }
    return 0;
}

static int read_prop_value(FILE *in, GV_PropValue *v)
{
    int t;
    if (fread(&t, sizeof(t), 1, in) != 1) return -1;
    v->type = (GV_PropType)t;
    switch (v->type) {
    case GV_PROP_NULL:
        v->as.i = 0;
        break;
    case GV_PROP_INT64:
        if (fread(&v->as.i, sizeof(v->as.i), 1, in) != 1) return -1;
        break;
    case GV_PROP_FLOAT64:
        if (fread(&v->as.f, sizeof(v->as.f), 1, in) != 1) return -1;
        break;
    case GV_PROP_BOOL:
        if (fread(&v->as.b, sizeof(v->as.b), 1, in) != 1) return -1;
        break;
    case GV_PROP_STRING: {
        uint32_t len;
        if (fread(&len, sizeof(len), 1, in) != 1) return -1;
        if (len > 0) {
            v->as.s = (char *)gv_alloc(len + 1);
            if (!v->as.s) return -1;
            if (fread(v->as.s, 1, len, in) != len) {
                gv_free(v->as.s);
                v->as.s = NULL;
                return -1;
            }
            v->as.s[len] = '\0';
        } else {
            v->as.s = NULL;
        }
        break;
    }
    case GV_PROP_BLOB: {
        uint32_t len;
        if (fread(&len, sizeof(len), 1, in) != 1) return -1;
        v->as.blob.len = len;
        if (len > 0) {
            v->as.blob.data = (uint8_t *)gv_alloc(len);
            if (!v->as.blob.data) return -1;
            if (fread(v->as.blob.data, 1, len, in) != len) {
                gv_free(v->as.blob.data);
                v->as.blob.data = NULL;
                v->as.blob.len = 0;
                return -1;
            }
        } else {
            v->as.blob.data = NULL;
        }
        break;
    }
    default:
        v->type = GV_PROP_NULL;
        v->as.i = 0;
        break;
    }
    return 0;
}

int graph_prop_index_save(const GV_GraphPropIndex *idx, FILE *out)
{
    if (!idx || !out) return -1;

    /* Write slot count and occupied count */
    uint64_t cap = idx->capacity;
    uint64_t cnt = idx->count;
    if (fwrite(&cap, sizeof(cap), 1, out) != 1) return -1;
    if (fwrite(&cnt, sizeof(cnt), 1, out) != 1) return -1;

    for (size_t i = 0; i < idx->capacity; i++) {
        const PropEntry *e = &idx->entries[i];
        int occ = e->occupied;
        if (fwrite(&occ, sizeof(occ), 1, out) != 1) return -1;
        if (!occ) continue;

        if (fwrite(&e->hash, sizeof(e->hash), 1, out) != 1) return -1;

        if (write_prop_value(out, e->value) != 0) return -1;

        uint64_t nc = e->node_count;
        uint64_t ec = e->edge_count;
        if (fwrite(&nc, sizeof(nc), 1, out) != 1) return -1;
        if (nc > 0 && fwrite(e->node_ids, sizeof(uint64_t), nc, out) != nc) return -1;
        if (fwrite(&ec, sizeof(ec), 1, out) != 1) return -1;
        if (ec > 0 && fwrite(e->edge_ids, sizeof(uint64_t), ec, out) != ec) return -1;

        uint32_t klen = e->key ? (uint32_t)strlen(e->key) : 0;
        if (fwrite(&klen, sizeof(klen), 1, out) != 1) return -1;
        if (klen > 0 && fwrite(e->key, 1, klen, out) != klen) return -1;
    }
    return 0;
}

int graph_prop_index_load(GV_GraphPropIndex **out, FILE *in)
{
    if (!out || !in) return -1;

    GV_GraphPropIndex *idx = graph_prop_index_create();
    if (!idx) return -1;

    uint64_t cap, cnt;
    if (fread(&cap, sizeof(cap), 1, in) != 1) goto fail;
    if (fread(&cnt, sizeof(cnt), 1, in) != 1) goto fail;

    /* Reallocate to the saved capacity */
    if (cap > 0) {
        idx->capacity = cap;
        idx->entries = (PropEntry *)gv_calloc(cap, sizeof(PropEntry));
        if (!idx->entries) goto fail;
    }

    for (size_t i = 0; i < cap; i++) {
        int occ;
        if (fread(&occ, sizeof(occ), 1, in) != 1) goto fail;
        if (!occ) continue;

        PropEntry *e = &idx->entries[i];
        e->occupied = 1;

        if (fread(&e->hash, sizeof(e->hash), 1, in) != 1) goto fail;

        if (read_prop_value(in, &e->value) != 0) goto fail;

        uint64_t nc, ec;
        if (fread(&nc, sizeof(nc), 1, in) != 1) goto fail;
        if (nc > 0) {
            e->node_ids = (uint64_t *)gv_alloc(nc * sizeof(uint64_t));
            if (!e->node_ids) goto fail;
            e->node_count = nc;
            e->node_cap = nc;
            if (fread(e->node_ids, sizeof(uint64_t), nc, in) != nc) goto fail;
        }
        if (fread(&ec, sizeof(ec), 1, in) != 1) goto fail;
        if (ec > 0) {
            e->edge_ids = (uint64_t *)gv_alloc(ec * sizeof(uint64_t));
            if (!e->edge_ids) goto fail;
            e->edge_count = ec;
            e->edge_cap = ec;
            if (fread(e->edge_ids, sizeof(uint64_t), ec, in) != ec) goto fail;
        }

        uint32_t klen;
        if (fread(&klen, sizeof(klen), 1, in) != 1) goto fail;
        if (klen > 0) {
            e->key = (char *)gv_alloc(klen + 1);
            if (!e->key) goto fail;
            if (fread(e->key, 1, klen, in) != klen) goto fail;
            e->key[klen] = '\0';
        }

        idx->count++;
    }

    *out = idx;
    return 0;

fail:
    graph_prop_index_destroy(idx);
    return -1;
}
