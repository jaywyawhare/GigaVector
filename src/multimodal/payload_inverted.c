#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "core/memory.h"
#include "core/utils.h"
#include "multimodal/payload_inverted.h"

/* ---- Hash map: string key -> GV_IdBitmap* ---- */

#define BUCKET_EMPTY NULL

typedef struct {
    char          *key;
    GV_IdBitmap   *bitmap;
    uint32_t       hash;
    int            occupied;
} InvBucket;

typedef struct {
    InvBucket *buckets;
    size_t     capacity;
    size_t     count;
} InvHashMap;

static InvHashMap *hashmap_create(size_t initial_cap) {
    if (initial_cap < 16) initial_cap = 16;
    InvHashMap *hm = (InvHashMap *)gv_calloc(1, sizeof(InvHashMap));
    if (!hm) return NULL;
    hm->buckets  = (InvBucket *)gv_calloc(initial_cap, sizeof(InvBucket));
    if (!hm->buckets) { gv_free(hm); return NULL; }
    hm->capacity = initial_cap;
    hm->count    = 0;
    return hm;
}

static void hashmap_destroy(InvHashMap *hm) {
    if (!hm) return;
    for (size_t i = 0; i < hm->capacity; ++i) {
        if (hm->buckets[i].occupied) {
            gv_free(hm->buckets[i].key);
            gv_id_bitmap_free(hm->buckets[i].bitmap);
        }
    }
    gv_free(hm->buckets);
    gv_free(hm);
}

static void hashmap_grow(InvHashMap *hm) {
    size_t old_cap = hm->capacity;
    InvBucket *old  = hm->buckets;
    size_t new_cap  = old_cap * 2;
    InvBucket *nb   = (InvBucket *)gv_calloc(new_cap, sizeof(InvBucket));
    if (!nb) return;
    hm->buckets  = nb;
    hm->capacity = new_cap;
    hm->count    = 0;
    for (size_t i = 0; i < old_cap; ++i) {
        if (old[i].occupied) {
            size_t idx = old[i].hash & (new_cap - 1);
            while (hm->buckets[idx].occupied) idx = (idx + 1) & (new_cap - 1);
            hm->buckets[idx] = old[i];
            hm->count++;
        }
    }
    gv_free(old);
}

static InvBucket *hashmap_get_or_insert(InvHashMap *hm, const char *key) {
    uint32_t h = gv_fnv1a(key, strlen(key));
    if (hm->count * 4 >= hm->capacity * 3) hashmap_grow(hm);
    size_t idx = h & (hm->capacity - 1);
    while (hm->buckets[idx].occupied) {
        if (hm->buckets[idx].hash == h && strcmp(hm->buckets[idx].key, key) == 0)
            return &hm->buckets[idx];
        idx = (idx + 1) & (hm->capacity - 1);
    }
    hm->buckets[idx].key      = gv_strdup(key);
    hm->buckets[idx].bitmap   = gv_id_bitmap_create();
    hm->buckets[idx].hash     = h;
    hm->buckets[idx].occupied = 1;
    hm->count++;
    return &hm->buckets[idx];
}

static InvBucket *hashmap_find(const InvHashMap *hm, const char *key) {
    if (!hm) return NULL;
    uint32_t h = gv_fnv1a(key, strlen(key));
    size_t idx = h & (hm->capacity - 1);
    while (hm->buckets[idx].occupied) {
        if (hm->buckets[idx].hash == h && strcmp(hm->buckets[idx].key, key) == 0)
            return &hm->buckets[idx];
        if (hm->buckets[idx].key == NULL) break;
        idx = (idx + 1) & (hm->capacity - 1);
    }
    return NULL;
}

/* ---- Per-field index ---- */

typedef struct {
    char       name[64];
    int        type;       /* 0=string, 1=int64, 2=float64, 3=bool */
    InvHashMap *exact;     /* full value -> bitmap */
    InvHashMap *tokens;    /* individual tokens -> bitmap (string fields only) */
    size_t      entry_count;
} InvField;

/* ---- Per-vector reverse index (for removal) ---- */

typedef struct {
    uint64_t vector_id;
    char   **field_names;
    char   **values;
    size_t   count;
    size_t   capacity;
} InvVectorRecord;

/* ---- Top-level index ---- */

struct GV_PayloadInvertedIndex {
    InvField        *fields;
    size_t           field_count;
    size_t           field_capacity;
    InvVectorRecord *records;
    size_t           record_count;
    size_t           record_capacity;
};

/* ---- Tokenizer (whitespace split) ---- */

static size_t tokenize(const char *str, char ***out_tokens) {
    *out_tokens = NULL;
    if (!str || !*str) return 0;

    size_t cap = 8;
    size_t n   = 0;
    char **tokens = (char **)gv_alloc(cap * sizeof(char *));
    if (!tokens) return 0;

    const char *p = str;
    while (*p) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;
        const char *start = p;
        while (*p && !isspace((unsigned char)*p)) p++;
        size_t len = (size_t)(p - start);
        char *tok  = (char *)gv_alloc(len + 1);
        if (!tok) break;
        memcpy(tok, start, len);
        tok[len] = '\0';
        if (n == cap) {
            cap *= 2;
            char **nt = (char **)gv_realloc(tokens, cap * sizeof(char *));
            if (!nt) { gv_free(tok); break; }
            tokens = nt;
        }
        tokens[n++] = tok;
    }
    *out_tokens = tokens;
    return n;
}

static void free_tokens(char **tokens, size_t n) {
    if (!tokens) return;
    for (size_t i = 0; i < n; ++i) gv_free(tokens[i]);
    gv_free(tokens);
}

/* ---- Public API ---- */

GV_PayloadInvertedIndex *payload_inverted_create(void) {
    GV_PayloadInvertedIndex *idx = (GV_PayloadInvertedIndex *)gv_calloc(1, sizeof(GV_PayloadInvertedIndex));
    if (!idx) return NULL;
    idx->field_capacity = 16;
    idx->fields = (InvField *)gv_calloc(idx->field_capacity, sizeof(InvField));
    if (!idx->fields) { gv_free(idx); return NULL; }
    idx->record_capacity = 1024;
    idx->records = (InvVectorRecord *)gv_calloc(idx->record_capacity, sizeof(InvVectorRecord));
    if (!idx->records) { gv_free(idx->fields); gv_free(idx); return NULL; }
    return idx;
}

void payload_inverted_destroy(GV_PayloadInvertedIndex *idx) {
    if (!idx) return;
    for (size_t i = 0; i < idx->field_count; ++i) {
        hashmap_destroy(idx->fields[i].exact);
        hashmap_destroy(idx->fields[i].tokens);
    }
    gv_free(idx->fields);
    for (size_t i = 0; i < idx->record_count; ++i) {
        for (size_t j = 0; j < idx->records[i].count; ++j) {
            gv_free(idx->records[i].field_names[j]);
            gv_free(idx->records[i].values[j]);
        }
        gv_free(idx->records[i].field_names);
        gv_free(idx->records[i].values);
    }
    gv_free(idx->records);
    gv_free(idx);
}

int payload_inverted_add_field(GV_PayloadInvertedIndex *idx, const char *field, int type) {
    if (!idx || !field) return -1;
    for (size_t i = 0; i < idx->field_count; ++i) {
        if (strcmp(idx->fields[i].name, field) == 0) return 0;
    }
    if (idx->field_count == idx->field_capacity) {
        size_t nc = idx->field_capacity * 2;
        InvField *nf = (InvField *)gv_realloc(idx->fields, nc * sizeof(InvField));
        if (!nf) return -1;
        memset(nf + idx->field_capacity, 0, (nc - idx->field_capacity) * sizeof(InvField));
        idx->fields = nf;
        idx->field_capacity = nc;
    }
    InvField *f = &idx->fields[idx->field_count];
    snprintf(f->name, sizeof(f->name), "%s", field);
    f->type = type;
    f->exact  = hashmap_create(64);
    f->tokens = hashmap_create(64);
    f->entry_count = 0;
    idx->field_count++;
    return 0;
}

/* Find or grow the reverse record for a vector_id */
static InvVectorRecord *get_or_create_record(GV_PayloadInvertedIndex *idx, uint64_t vector_id) {
    for (size_t i = 0; i < idx->record_count; ++i) {
        if (idx->records[i].vector_id == vector_id) return &idx->records[i];
    }
    if (idx->record_count == idx->record_capacity) {
        size_t nc = idx->record_capacity * 2;
        InvVectorRecord *nr = (InvVectorRecord *)gv_realloc(idx->records, nc * sizeof(InvVectorRecord));
        if (!nr) return NULL;
        memset(nr + idx->record_capacity, 0, (nc - idx->record_capacity) * sizeof(InvVectorRecord));
        idx->records = nr;
        idx->record_capacity = nc;
    }
    InvVectorRecord *rec = &idx->records[idx->record_count];
    memset(rec, 0, sizeof(*rec));
    rec->vector_id = vector_id;
    rec->capacity  = 8;
    rec->field_names = (char **)gv_calloc(rec->capacity, sizeof(char *));
    rec->values      = (char **)gv_calloc(rec->capacity, sizeof(char *));
    idx->record_count++;
    return rec;
}

static void record_add(InvVectorRecord *rec, const char *field, const char *value) {
    if (rec->count == rec->capacity) {
        size_t nc = rec->capacity * 2;
        char **nf = (char **)gv_realloc(rec->field_names, nc * sizeof(char *));
        char **nv = (char **)gv_realloc(rec->values, nc * sizeof(char *));
        if (nf) rec->field_names = nf;
        if (nv) rec->values = nv;
        rec->capacity = nc;
    }
    rec->field_names[rec->count] = gv_strdup(field);
    rec->values[rec->count]      = gv_strdup(value);
    rec->count++;
}

static InvField *find_field(const GV_PayloadInvertedIndex *idx, const char *field) {
    for (size_t i = 0; i < idx->field_count; ++i) {
        if (strcmp(idx->fields[i].name, field) == 0) return &idx->fields[i];
    }
    return NULL;
}

int payload_inverted_index(GV_PayloadInvertedIndex *idx, const char *field,
                           uint64_t vector_id, const char *value) {
    if (!idx || !field || !value) return -1;
    InvField *f = find_field(idx, field);
    if (!f) return -1;

    /* Add to exact-match inverted list */
    InvBucket *b = hashmap_get_or_insert(f->exact, value);
    if (!b->bitmap) b->bitmap = gv_id_bitmap_create();
    gv_id_bitmap_add(b->bitmap, vector_id);

    /* For string fields, also tokenize and index each token */
    if (f->type == 0) {
        char **tokens = NULL;
        size_t ntok   = tokenize(value, &tokens);
        for (size_t i = 0; i < ntok; ++i) {
            InvBucket *tb = hashmap_get_or_insert(f->tokens, tokens[i]);
            if (!tb->bitmap) tb->bitmap = gv_id_bitmap_create();
            gv_id_bitmap_add(tb->bitmap, vector_id);
        }
        free_tokens(tokens, ntok);
    }

    f->entry_count++;

    /* Update reverse record */
    InvVectorRecord *rec = get_or_create_record(idx, vector_id);
    if (rec) record_add(rec, field, value);

    return 0;
}

int payload_inverted_remove(GV_PayloadInvertedIndex *idx, uint64_t vector_id) {
    if (!idx) return -1;

    /* Find the reverse record to know which field/value pairs to remove */
    for (size_t ri = 0; ri < idx->record_count; ++ri) {
        if (idx->records[ri].vector_id != vector_id) continue;
        InvVectorRecord *rec = &idx->records[ri];

        for (size_t vi = 0; vi < rec->count; ++vi) {
            InvField *f = find_field(idx, rec->field_names[vi]);
            if (!f) continue;

            /* Remove from exact match */
            InvBucket *b = hashmap_find(f->exact, rec->values[vi]);
            if (b && b->bitmap) {
                gv_id_bitmap_remove(b->bitmap, vector_id);
                if (f->entry_count > 0) f->entry_count--;
            }

            /* Remove from tokens */
            if (f->type == 0 && f->tokens) {
                char **tokens = NULL;
                size_t ntok   = tokenize(rec->values[vi], &tokens);
                for (size_t ti = 0; ti < ntok; ++ti) {
                    InvBucket *tb = hashmap_find(f->tokens, tokens[ti]);
                    if (tb && tb->bitmap) {
                        gv_id_bitmap_remove(tb->bitmap, vector_id);
                    }
                }
                free_tokens(tokens, ntok);
            }
        }

        /* Free and compact the record */
        for (size_t vi = 0; vi < rec->count; ++vi) {
            gv_free(rec->field_names[vi]);
            gv_free(rec->values[vi]);
        }
        gv_free(rec->field_names);
        gv_free(rec->values);

        /* Move last record into this slot */
        idx->records[ri] = idx->records[idx->record_count - 1];
        memset(&idx->records[idx->record_count - 1], 0, sizeof(InvVectorRecord));
        idx->record_count--;
        return 0;
    }
    return -1;
}

GV_IdBitmap *payload_inverted_find(const GV_PayloadInvertedIndex *idx,
                                   const char *field, const char *value) {
    if (!idx || !field || !value) return NULL;
    InvField *f = find_field(idx, field);
    if (!f) return NULL;

    InvBucket *b = hashmap_find(f->exact, value);
    if (!b || !b->bitmap) return NULL;
    return gv_id_bitmap_clone(b->bitmap);
}

GV_IdBitmap *payload_inverted_find_prefix(const GV_PayloadInvertedIndex *idx,
                                          const char *field, const char *prefix) {
    if (!idx || !field || !prefix) return NULL;
    InvField *f = find_field(idx, field);
    if (!f) return NULL;

    GV_IdBitmap *result = gv_id_bitmap_create();
    if (!result) return NULL;

    size_t prefix_len = strlen(prefix);
    InvHashMap *hm = (f->type == 0 && f->tokens) ? f->tokens : f->exact;

    for (size_t i = 0; i < hm->capacity; ++i) {
        if (!hm->buckets[i].occupied || !hm->buckets[i].key) continue;
        if (strncmp(hm->buckets[i].key, prefix, prefix_len) == 0) {
            gv_id_bitmap_or_into(result, hm->buckets[i].bitmap);
        }
    }
    return result;
}

GV_IdBitmap *payload_inverted_find_range(const GV_PayloadInvertedIndex *idx,
                                         const char *field,
                                         double min_val, double max_val) {
    if (!idx || !field) return NULL;
    InvField *f = find_field(idx, field);
    if (!f) return NULL;

    GV_IdBitmap *result = gv_id_bitmap_create();
    if (!result) return NULL;

    /* For numeric fields, the exact hash map stores stringified numbers.
     * Parse each key and check if it falls in [min_val, max_val]. */
    InvHashMap *hm = f->exact;
    for (size_t i = 0; i < hm->capacity; ++i) {
        if (!hm->buckets[i].occupied || !hm->buckets[i].key) continue;
        char *endptr = NULL;
        double val = strtod(hm->buckets[i].key, &endptr);
        if (endptr != hm->buckets[i].key && *endptr == '\0') {
            if (val >= min_val && val <= max_val) {
                gv_id_bitmap_or_into(result, hm->buckets[i].bitmap);
            }
        }
    }
    return result;
}

/* ---- Persistence ---- */

int payload_inverted_save(const GV_PayloadInvertedIndex *idx, FILE *out) {
    if (!idx || !out) return -1;

    /* Write field count */
    if (write_u32(out, (uint32_t)idx->field_count) != 0) return -1;

    /* Write field schemas */
    for (size_t i = 0; i < idx->field_count; ++i) {
        if (write_string(out, idx->fields[i].name) != 0) return -1;
        if (write_u32(out, (uint32_t)idx->fields[i].type) != 0) return -1;
    }

    /* Write record count */
    if (write_u64(out, (uint64_t)idx->record_count) != 0) return -1;

    /* Write each record */
    for (size_t ri = 0; ri < idx->record_count; ++ri) {
        const InvVectorRecord *rec = &idx->records[ri];
        if (write_u64(out, rec->vector_id) != 0) return -1;
        if (write_u32(out, (uint32_t)rec->count) != 0) return -1;
        for (size_t vi = 0; vi < rec->count; ++vi) {
            if (write_string(out, rec->field_names[vi]) != 0) return -1;
            if (write_string(out, rec->values[vi]) != 0) return -1;
        }
    }

    return 0;
}

int payload_inverted_load(GV_PayloadInvertedIndex **idx_out, FILE *in) {
    if (!idx_out || !in) return -1;

    GV_PayloadInvertedIndex *idx = payload_inverted_create();
    if (!idx) return -1;

    uint32_t field_count = 0;
    if (read_u32(in, &field_count) != 0) { payload_inverted_destroy(idx); return -1; }

    /* Read field schemas and add them */
    for (uint32_t i = 0; i < field_count; ++i) {
        char *name = read_string(in);
        uint32_t type = 0;
        if (!name || read_u32(in, &type) != 0) { gv_free(name); payload_inverted_destroy(idx); return -1; }
        payload_inverted_add_field(idx, name, (int)type);
        gv_free(name);
    }

    uint64_t record_count = 0;
    if (read_u64(in, &record_count) != 0) { payload_inverted_destroy(idx); return -1; }

    /* Read and re-index records */
    for (uint64_t ri = 0; ri < record_count; ++ri) {
        uint64_t vid = 0;
        if (read_u64(in, &vid) != 0) { payload_inverted_destroy(idx); return -1; }
        uint32_t pair_count = 0;
        if (read_u32(in, &pair_count) != 0) { payload_inverted_destroy(idx); return -1; }
        for (uint32_t pi = 0; pi < pair_count; ++pi) {
            char *fname = read_string(in);
            char *fval  = read_string(in);
            if (!fname || !fval) { gv_free(fname); gv_free(fval); payload_inverted_destroy(idx); return -1; }
            payload_inverted_index(idx, fname, vid, fval);
            gv_free(fname);
            gv_free(fval);
        }
    }

    *idx_out = idx;
    return 0;
}

size_t payload_inverted_field_count(const GV_PayloadInvertedIndex *idx) {
    return idx ? idx->field_count : 0;
}

size_t payload_inverted_entry_count(const GV_PayloadInvertedIndex *idx) {
    if (!idx) return 0;
    size_t total = 0;
    for (size_t i = 0; i < idx->field_count; ++i) {
        total += idx->fields[i].entry_count;
    }
    return total;
}
