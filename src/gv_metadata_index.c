#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "gigavector/gv_metadata_index.h"
#include "gigavector/gv_metadata.h"

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

/**
 * @brief Simple hash function for strings (djb2).
 */
static uint32_t gv_metadata_index_hash(const char *str) {
    uint32_t hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
    }
    return hash;
}

/**
 * @brief Combine two strings for hashing.
 */
static uint32_t gv_metadata_index_hash_pair(const char *key, const char *value) {
    uint32_t key_hash = gv_metadata_index_hash(key);
    uint32_t value_hash = gv_metadata_index_hash(value);
    return key_hash ^ (value_hash << 1);
}

/**
 * @brief Create a new metadata inverted index.
 */
GV_MetadataIndex *gv_metadata_index_create(void) {
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

/**
 * @brief Destroy a metadata inverted index and free all resources.
 */
void gv_metadata_index_destroy(GV_MetadataIndex *index) {
    if (index == NULL) {
        return;
    }

    for (size_t i = 0; i < index->bucket_count; ++i) {
        GV_MetadataKVEntry *entry = index->buckets[i].head;
        while (entry != NULL) {
            GV_MetadataKVEntry *next = entry->next;
            free(entry->key);
            free(entry->value);
            free(entry->vector_indices);
            free(entry);
            entry = next;
        }
    }

    free(index->buckets);
    free(index);
}

/**
 * @brief Find or create an entry for a key-value pair.
 */
static GV_MetadataKVEntry *gv_metadata_index_find_or_create(GV_MetadataIndex *index,
                                                             const char *key, const char *value,
                                                             int create) {
    if (index == NULL || key == NULL || value == NULL) {
        return NULL;
    }

    uint32_t hash = gv_metadata_index_hash_pair(key, value);
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

    /* Create new entry */
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

/**
 * @brief Add a vector index to the inverted index for a given key-value pair.
 */
int gv_metadata_index_add(GV_MetadataIndex *index, const char *key, const char *value, size_t vector_index) {
    if (index == NULL || key == NULL || value == NULL) {
        return -1;
    }

    GV_MetadataKVEntry *entry = gv_metadata_index_find_or_create(index, key, value, 1);
    if (entry == NULL) {
        return -1;
    }

    /* Check if vector_index already exists */
    for (size_t i = 0; i < entry->count; ++i) {
        if (entry->vector_indices[i] == vector_index) {
            return 0; /* Already exists */
        }
    }

    /* Resize if needed */
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

/**
 * @brief Remove a vector index from the inverted index for a given key-value pair.
 */
int gv_metadata_index_remove(GV_MetadataIndex *index, const char *key, const char *value, size_t vector_index) {
    if (index == NULL || key == NULL || value == NULL) {
        return -1;
    }

    GV_MetadataKVEntry *entry = gv_metadata_index_find_or_create(index, key, value, 0);
    if (entry == NULL) {
        return 0; /* Entry doesn't exist, nothing to remove */
    }

    /* Find and remove the vector index */
    for (size_t i = 0; i < entry->count; ++i) {
        if (entry->vector_indices[i] == vector_index) {
            /* Shift remaining indices */
            for (size_t j = i; j < entry->count - 1; ++j) {
                entry->vector_indices[j] = entry->vector_indices[j + 1];
            }
            entry->count--;
            return 0;
        }
    }

    return 0; /* Vector index not found, but that's okay */
}

/**
 * @brief Query the inverted index to get all vector indices matching a key-value pair.
 */
int gv_metadata_index_query(const GV_MetadataIndex *index, const char *key, const char *value,
                            size_t *out_indices, size_t max_indices) {
    if (index == NULL || key == NULL || value == NULL || out_indices == NULL || max_indices == 0) {
        return -1;
    }

    GV_MetadataKVEntry *entry = gv_metadata_index_find_or_create((GV_MetadataIndex *)index, key, value, 0);
    if (entry == NULL) {
        return 0; /* No matching entries */
    }

    size_t copy_count = (entry->count < max_indices) ? entry->count : max_indices;
    for (size_t i = 0; i < copy_count; ++i) {
        out_indices[i] = entry->vector_indices[i];
    }

    return (int)copy_count;
}

/**
 * @brief Get the count of vector indices matching a key-value pair.
 */
size_t gv_metadata_index_count(const GV_MetadataIndex *index, const char *key, const char *value) {
    if (index == NULL || key == NULL || value == NULL) {
        return 0;
    }

    GV_MetadataKVEntry *entry = gv_metadata_index_find_or_create((GV_MetadataIndex *)index, key, value, 0);
    if (entry == NULL) {
        return 0;
    }

    return entry->count;
}

/**
 * @brief Remove all entries for a given vector index (used when vector is deleted).
 */
int gv_metadata_index_remove_vector(GV_MetadataIndex *index, size_t vector_index) {
    if (index == NULL) {
        return -1;
    }

    /* Iterate through all buckets and entries */
    for (size_t i = 0; i < index->bucket_count; ++i) {
        GV_MetadataKVEntry *entry = index->buckets[i].head;
        while (entry != NULL) {
            /* Remove vector_index from this entry if present */
            for (size_t j = 0; j < entry->count; ++j) {
                if (entry->vector_indices[j] == vector_index) {
                    /* Shift remaining indices */
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

/**
 * @brief Update metadata for a vector (remove old entries, add new ones).
 */
int gv_metadata_index_update(GV_MetadataIndex *index, size_t vector_index,
                             const void *old_metadata, const void *new_metadata) {
    if (index == NULL) {
        return -1;
    }

    /* Remove old metadata entries */
    if (old_metadata != NULL) {
        GV_Metadata *old_meta = (GV_Metadata *)old_metadata;
        GV_Metadata *current = old_meta;
        while (current != NULL) {
            gv_metadata_index_remove(index, current->key, current->value, vector_index);
            current = current->next;
        }
    }

    /* Add new metadata entries */
    if (new_metadata != NULL) {
        GV_Metadata *new_meta = (GV_Metadata *)new_metadata;
        GV_Metadata *current = new_meta;
        while (current != NULL) {
            gv_metadata_index_add(index, current->key, current->value, vector_index);
            current = current->next;
        }
    }

    return 0;
}


