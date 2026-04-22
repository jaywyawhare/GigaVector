#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "multimodal/metadata_index.h"
#include "core/utils.h"

#define GV_METADATA_INDEX_HASH_SIZE 1024
#define GV_METADATA_INDEX_LOAD_FACTOR 0.75

/**
 * @brief Hash table entry for key-value pair.
 */
typedef struct GV_MetadataKVEntry {
    char *key;
    char *value;
    size_t *vector_indices;
    size_t count;
    size_t capacity;
    struct GV_MetadataKVEntry *next;  /* For chaining */
} GV_MetadataKVEntry;

/**
 * @brief Hash table bucket.
 */
typedef struct {
    GV_MetadataKVEntry *head;
} GV_MetadataBucket;

/**
 * @brief Metadata inverted index structure.
 */
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
    GV_MetadataIndex *index = (GV_MetadataIndex *)malloc(sizeof(GV_MetadataIndex));
    if (index == NULL) {
        return NULL;
    }

    index->bucket_count = GV_METADATA_INDEX_HASH_SIZE;
    index->buckets = (GV_MetadataBucket *)calloc(index->bucket_count, sizeof(GV_MetadataBucket));
    if (index->buckets == NULL) {
        free(index);
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
        size_t bucket_count = index->bucket_count;
        if (bucket_count == 0 || bucket_count >= 1000000) {
            free(index->buckets);
        } else {
            for (size_t i = 0; i < bucket_count; ++i) {
                GV_MetadataKVEntry *entry = index->buckets[i].head;
                while (entry != NULL) {
                    GV_MetadataKVEntry *next = entry->next;
                    if (entry->key != NULL) {
                        free(entry->key);
                    }
                    if (entry->value != NULL) {
                        free(entry->value);
                    }
                    if (entry->vector_indices != NULL) {
                        free(entry->vector_indices);
                    }
                    free(entry);
                    entry = next;
                }
            }
            free(index->buckets);
        }
        index->buckets = NULL;
    }
    index->bucket_count = 0;
    free(index);
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

    entry = (GV_MetadataKVEntry *)malloc(sizeof(GV_MetadataKVEntry));
    if (entry == NULL) {
        return NULL;
    }

    entry->key = (char *)malloc(strlen(key) + 1);
    entry->value = (char *)malloc(strlen(value) + 1);
    if (entry->key == NULL || entry->value == NULL) {
        free(entry->key);
        free(entry->value);
        free(entry);
        return NULL;
    }

    strcpy(entry->key, key);
    strcpy(entry->value, value);
    entry->count = 0;
    entry->capacity = 16;
    entry->vector_indices = (size_t *)malloc(entry->capacity * sizeof(size_t));
    if (entry->vector_indices == NULL) {
        free(entry->key);
        free(entry->value);
        free(entry);
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

    for (size_t i = 0; i < entry->count; ++i) {
        if (entry->vector_indices[i] == vector_index) {
            return 0; /* Already exists */
        }
    }

    if (entry->count >= entry->capacity) {
        size_t new_capacity = entry->capacity * 2;
        size_t *new_indices = (size_t *)realloc(entry->vector_indices, new_capacity * sizeof(size_t));
        if (new_indices == NULL) {
            return -1;
        }
        entry->vector_indices = new_indices;
        entry->capacity = new_capacity;
    }

    entry->vector_indices[entry->count++] = vector_index;
    return 0;
}

int metadata_index_remove(GV_MetadataIndex *index, const char *key, const char *value, size_t vector_index) {
    if (index == NULL || key == NULL || value == NULL) {
        return -1;
    }

    GV_MetadataKVEntry *entry = metadata_index_find_or_create(index, key, value, 0);
    if (entry == NULL) {
        return 0; /* Entry doesn't exist, nothing to remove */
    }

    for (size_t i = 0; i < entry->count; ++i) {
        if (entry->vector_indices[i] == vector_index) {
            for (size_t j = i; j < entry->count - 1; ++j) {
                entry->vector_indices[j] = entry->vector_indices[j + 1];
            }
            entry->count--;
            return 0;
        }
    }

    return 0; /* Vector index not found, but that's okay */
}

int metadata_index_query(const GV_MetadataIndex *index, const char *key, const char *value,
                            size_t *out_indices, size_t max_indices) {
    if (index == NULL || key == NULL || value == NULL || out_indices == NULL || max_indices == 0) {
        return -1;
    }

    GV_MetadataKVEntry *entry = metadata_index_find_or_create((GV_MetadataIndex *)index, key, value, 0);
    if (entry == NULL) {
        return 0; /* No matching entries */
    }

    size_t copy_count = (entry->count < max_indices) ? entry->count : max_indices;
    for (size_t i = 0; i < copy_count; ++i) {
        out_indices[i] = entry->vector_indices[i];
    }

    return (int)copy_count;
}

size_t metadata_index_count(const GV_MetadataIndex *index, const char *key, const char *value) {
    if (index == NULL || key == NULL || value == NULL) {
        return 0;
    }

    GV_MetadataKVEntry *entry = metadata_index_find_or_create((GV_MetadataIndex *)index, key, value, 0);
    if (entry == NULL) {
        return 0;
    }

    return entry->count;
}

int metadata_index_remove_vector(GV_MetadataIndex *index, size_t vector_index) {
    if (index == NULL) {
        return -1;
    }

    for (size_t i = 0; i < index->bucket_count; ++i) {
        GV_MetadataKVEntry *entry = index->buckets[i].head;
        while (entry != NULL) {
            for (size_t j = 0; j < entry->count; ++j) {
                if (entry->vector_indices[j] == vector_index) {
                    for (size_t k = j; k < entry->count - 1; ++k) {
                        entry->vector_indices[k] = entry->vector_indices[k + 1];
                    }
                    entry->count--;
                    j--; /* Check same position again */
                }
            }
            entry = entry->next;
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
        GV_Metadata *old_meta = (GV_Metadata *)old_metadata;
        GV_Metadata *current = old_meta;
        while (current != NULL) {
            metadata_index_remove(index, current->key, current->value, vector_index);
            current = current->next;
        }
    }

    if (new_metadata != NULL) {
        GV_Metadata *new_meta = (GV_Metadata *)new_metadata;
        GV_Metadata *current = new_meta;
        while (current != NULL) {
            metadata_index_add(index, current->key, current->value, vector_index);
            current = current->next;
        }
    }

    return 0;
}

