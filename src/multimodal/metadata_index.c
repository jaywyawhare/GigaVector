/**
 * metadata_index.c — inverted index mapping metadata (key=value) → the set of
 * vectors carrying it, used to accelerate filtered search.
 *
 * Posting lists are stored as roaring-lite bitmaps (GV_IdBitmap, core/id_bitmap.h),
 * the Dgraph model: each posting is a compressed sorted UID set, so membership is
 * O(1)-ish, dedup/growth is automatic, and multi-filter queries reduce to bitmap
 * intersections (see metadata_index_query_bitmap + gv_id_bitmap_and).
 */
#include <stdlib.h>
#include "core/memory.h"
#include <string.h>
#include <stdint.h>

#include "multimodal/metadata_index.h"
#include "core/id_bitmap.h"
#include "core/utils.h"

#define GV_METADATA_INDEX_HASH_SIZE 1024

typedef struct GV_MetadataKVEntry {
    char *key;
    char *value;
    GV_IdBitmap *ids;                 /* roaring posting: set of vector indices */
    struct GV_MetadataKVEntry *next;  /* hash-chain */
} GV_MetadataKVEntry;

typedef struct {
    GV_MetadataKVEntry *head;
} GV_MetadataBucket;

struct GV_MetadataIndex {
    GV_MetadataBucket *buckets;
    size_t bucket_count;
    size_t total_entries;
};

static uint32_t metadata_index_hash_pair(const char *key, const char *value) {
    uint32_t key_hash = hash_str(key);
    uint32_t value_hash = hash_str(value);
    return key_hash ^ (value_hash << 1);
}

GV_MetadataIndex *metadata_index_create(void) {
    GV_MetadataIndex *index = (GV_MetadataIndex *)gv_alloc(sizeof(GV_MetadataIndex));
    if (index == NULL) {
        return NULL;
    }
    index->bucket_count = GV_METADATA_INDEX_HASH_SIZE;
    index->buckets = (GV_MetadataBucket *)gv_calloc(index->bucket_count, sizeof(GV_MetadataBucket));
    if (index->buckets == NULL) {
        gv_free(index);
        return NULL;
    }
    index->total_entries = 0;
    return index;
}

void metadata_index_destroy(GV_MetadataIndex *index) {
    if (index == NULL) {
        return;
    }
    if (index->buckets != NULL) {
        for (size_t i = 0; i < index->bucket_count; ++i) {
            GV_MetadataKVEntry *entry = index->buckets[i].head;
            while (entry != NULL) {
                GV_MetadataKVEntry *next = entry->next;
                gv_free(entry->key);
                gv_free(entry->value);
                gv_id_bitmap_free(entry->ids);
                gv_free(entry);
                entry = next;
            }
        }
        gv_free(index->buckets);
        index->buckets = NULL;
    }
    index->bucket_count = 0;
    gv_free(index);
}

static GV_MetadataKVEntry *metadata_index_find_or_create(GV_MetadataIndex *index,
                                                         const char *key, const char *value,
                                                         int create) {
    if (index == NULL || key == NULL || value == NULL) {
        return NULL;
    }
    uint32_t hash = metadata_index_hash_pair(key, value);
    size_t bucket_idx = hash % index->bucket_count;

    GV_MetadataKVEntry *entry = index->buckets[bucket_idx].head;
    while (entry != NULL) {
        if (strcmp(entry->key, key) == 0 && strcmp(entry->value, value) == 0) {
            return entry;
        }
        entry = entry->next;
    }
    if (!create) {
        return NULL;
    }

    entry = (GV_MetadataKVEntry *)gv_calloc(1, sizeof(GV_MetadataKVEntry));
    if (entry == NULL) {
        return NULL;
    }
    entry->key = gv_dup_cstr(key);
    entry->value = gv_dup_cstr(value);
    entry->ids = gv_id_bitmap_create();
    if (entry->key == NULL || entry->value == NULL || entry->ids == NULL) {
        gv_free(entry->key);
        gv_free(entry->value);
        gv_id_bitmap_free(entry->ids);
        gv_free(entry);
        return NULL;
    }

    entry->next = index->buckets[bucket_idx].head;
    index->buckets[bucket_idx].head = entry;
    index->total_entries++;
    return entry;
}

int metadata_index_add(GV_MetadataIndex *index, const char *key, const char *value, size_t vector_index) {
    if (index == NULL || key == NULL || value == NULL) {
        return -1;
    }
    GV_MetadataKVEntry *entry = metadata_index_find_or_create(index, key, value, 1);
    if (entry == NULL) {
        return -1;
    }
    /* the bitmap dedups + grows internally */
    return gv_id_bitmap_add(entry->ids, (uint64_t)vector_index);
}

int metadata_index_remove(GV_MetadataIndex *index, const char *key, const char *value, size_t vector_index) {
    if (index == NULL || key == NULL || value == NULL) {
        return -1;
    }
    GV_MetadataKVEntry *entry = metadata_index_find_or_create(index, key, value, 0);
    if (entry == NULL) {
        return 0; /* nothing to remove */
    }
    gv_id_bitmap_remove(entry->ids, (uint64_t)vector_index);
    return 0;
}

typedef struct {
    size_t *out;
    size_t  max;
    size_t  n;
} MetaQueryCtx;

static int metadata_query_collect(uint64_t id, void *ctx) {
    MetaQueryCtx *q = (MetaQueryCtx *)ctx;
    if (q->n >= q->max) return 1;   /* stop iteration: output full */
    q->out[q->n++] = (size_t)id;
    return 0;
}

int metadata_index_query(const GV_MetadataIndex *index, const char *key, const char *value,
                         size_t *out_indices, size_t max_indices) {
    if (index == NULL || key == NULL || value == NULL || out_indices == NULL || max_indices == 0) {
        return -1;
    }
    GV_MetadataKVEntry *entry = metadata_index_find_or_create((GV_MetadataIndex *)index, key, value, 0);
    if (entry == NULL) {
        return 0;
    }
    MetaQueryCtx q = { out_indices, max_indices, 0 };
    gv_id_bitmap_iterate(entry->ids, metadata_query_collect, &q);
    return (int)q.n;
}

const GV_IdBitmap *metadata_index_query_bitmap(const GV_MetadataIndex *index,
                                               const char *key, const char *value) {
    if (index == NULL || key == NULL || value == NULL) {
        return NULL;
    }
    GV_MetadataKVEntry *entry = metadata_index_find_or_create((GV_MetadataIndex *)index, key, value, 0);
    return entry ? entry->ids : NULL;
}

size_t metadata_index_count(const GV_MetadataIndex *index, const char *key, const char *value) {
    if (index == NULL || key == NULL || value == NULL) {
        return 0;
    }
    GV_MetadataKVEntry *entry = metadata_index_find_or_create((GV_MetadataIndex *)index, key, value, 0);
    if (entry == NULL) {
        return 0;
    }
    return (size_t)gv_id_bitmap_cardinality(entry->ids);
}

int metadata_index_remove_vector(GV_MetadataIndex *index, size_t vector_index) {
    if (index == NULL) {
        return -1;
    }
    for (size_t i = 0; i < index->bucket_count; ++i) {
        for (GV_MetadataKVEntry *entry = index->buckets[i].head; entry != NULL; entry = entry->next) {
            gv_id_bitmap_remove(entry->ids, (uint64_t)vector_index);
        }
    }
    return 0;
}

int metadata_index_copy_vector(const GV_MetadataIndex *from_index, size_t from_vector_index,
                               GV_MetadataIndex *to_index, size_t to_vector_index) {
    if (from_index == NULL || to_index == NULL) {
        return -1;
    }
    for (size_t i = 0; i < from_index->bucket_count; ++i) {
        for (GV_MetadataKVEntry *entry = from_index->buckets[i].head; entry != NULL; entry = entry->next) {
            if (gv_id_bitmap_contains(entry->ids, (uint64_t)from_vector_index)) {
                if (metadata_index_add(to_index, entry->key, entry->value, to_vector_index) != 0) {
                    return -1;
                }
            }
        }
    }
    return 0;
}

int metadata_index_update(GV_MetadataIndex *index, size_t vector_index,
                          const void *old_metadata, const void *new_metadata) {
    if (index == NULL) {
        return -1;
    }
    if (old_metadata != NULL) {
        for (GV_Metadata *cur = (GV_Metadata *)old_metadata; cur != NULL; cur = cur->next) {
            metadata_index_remove(index, cur->key, cur->value, vector_index);
        }
    }
    if (new_metadata != NULL) {
        for (GV_Metadata *cur = (GV_Metadata *)new_metadata; cur != NULL; cur = cur->next) {
            metadata_index_add(index, cur->key, cur->value, vector_index);
        }
    }
    return 0;
}
