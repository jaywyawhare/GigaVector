#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include "storage/soa_storage.h"
#include "schema/metadata.h"
#include "schema/vector.h"
#include "core/memory.h"
#include "core/utils.h"
#include "storage/database.h"

#define GV_SOA_SAVE_MAGIC 0x534F4153u /* "SOAS" */

static void *soa_alloc(GV_SoAStorage *storage, size_t size) {
    if (storage != NULL && storage->owner_db != NULL) {
        return gv_db_alloc(storage->owner_db, size);
    }
    return malloc(size);
}

static void *soa_calloc(GV_SoAStorage *storage, size_t nmemb, size_t size) {
    if (storage != NULL && storage->owner_db != NULL) {
        return gv_db_calloc(storage->owner_db, nmemb, size);
    }
    return calloc(nmemb, size);
}

static void *soa_realloc(GV_SoAStorage *storage, void *ptr, size_t size) {
    if (storage != NULL && storage->owner_db != NULL) {
        return gv_db_realloc(storage->owner_db, ptr, size);
    }
    return realloc(ptr, size);
}

static void soa_free(GV_SoAStorage *storage, void *ptr) {
    if (ptr == NULL) {
        return;
    }
    if (storage != NULL && storage->owner_db != NULL) {
        gv_db_free(storage->owner_db, ptr);
        return;
    }
    free(ptr);
}

static int soa_read_metadata(FILE *in, GV_Metadata **out)
{
    *out = NULL;
    uint32_t count = 0;
    if (read_u32(in, &count) != 0) return -1;
    GV_Vector tmp = { .dimension = 0, .data = NULL, .metadata = NULL };
    for (uint32_t i = 0; i < count; ++i) {
        char *key = read_string(in);
        char *value = read_string(in);
        if (!key || !value) {
            free(key);
            free(value);
            vector_clear_metadata(&tmp);
            return -1;
        }
        if (vector_set_metadata(&tmp, key, value) != 0) {
            free(key);
            free(value);
            vector_clear_metadata(&tmp);
            return -1;
        }
        free(key);
        free(value);
    }
    *out = tmp.metadata;
    return 0;
}

static int soa_storage_grow(GV_SoAStorage *storage, size_t min_capacity)
{
    if (storage->capacity >= min_capacity) return 0;
    size_t new_capacity = storage->capacity ? storage->capacity : 1024;
    while (new_capacity < min_capacity) {
        if (new_capacity > SIZE_MAX / 2) return -1;
        new_capacity *= 2;
    }
    if (storage->dimension == 0 ||
        new_capacity > SIZE_MAX / storage->dimension / sizeof(float)) {
        return -1;
    }
    size_t new_data_size = new_capacity * storage->dimension * sizeof(float);
    float *tmp_data = (float *)soa_realloc(storage, storage->data, new_data_size);
    GV_Metadata **tmp_meta =
        (GV_Metadata **)soa_realloc(storage, storage->metadata, new_capacity * sizeof(GV_Metadata *));
    int *tmp_del = (int *)soa_realloc(storage, storage->deleted, new_capacity * sizeof(int));
    if (!tmp_data || !tmp_meta || !tmp_del) {
        if (tmp_data) storage->data = tmp_data;
        if (tmp_meta) storage->metadata = tmp_meta;
        if (tmp_del) storage->deleted = tmp_del;
        return -1;
    }
    if (new_capacity > storage->capacity) {
        memset(tmp_meta + storage->capacity, 0,
               (new_capacity - storage->capacity) * sizeof(GV_Metadata *));
        memset(tmp_del + storage->capacity, 0,
               (new_capacity - storage->capacity) * sizeof(int));
    }
    storage->data = tmp_data;
    storage->metadata = tmp_meta;
    storage->deleted = tmp_del;
    storage->capacity = new_capacity;
    return 0;
}

GV_SoAStorage *soa_storage_create(size_t dimension, size_t initial_capacity) {
    if (dimension == 0) {
        return NULL;
    }

    GV_SoAStorage *storage = (GV_SoAStorage *)malloc(sizeof(GV_SoAStorage));
    if (storage == NULL) {
        return NULL;
    }
    memset(storage, 0, sizeof(*storage));

    storage->dimension = dimension;
    storage->count = 0;
    storage->capacity = (initial_capacity > 0) ? initial_capacity : 1024;

    if (storage->capacity > SIZE_MAX / dimension / sizeof(float)) {
        free(storage);
        return NULL;
    }
    size_t data_size = storage->capacity * dimension * sizeof(float);
    storage->data = (float *)malloc(data_size);
    if (storage->data == NULL) {
        free(storage);
        return NULL;
    }

    storage->metadata = (GV_Metadata **)calloc(storage->capacity, sizeof(GV_Metadata *));
    if (storage->metadata == NULL) {
        free(storage->data);
        free(storage);
        return NULL;
    }

    storage->deleted = (int *)calloc(storage->capacity, sizeof(int));
    if (storage->deleted == NULL) {
        free(storage->metadata);
        free(storage->data);
        free(storage);
        return NULL;
    }

    return storage;
}

void soa_storage_bind_database(GV_SoAStorage *storage, GV_Database *db) {
    if (storage == NULL || db == NULL) {
        return;
    }
    size_t data_bytes = storage->capacity * storage->dimension * sizeof(float);
    float *data = (float *)gv_db_alloc(db, data_bytes);
    GV_Metadata **meta = (GV_Metadata **)gv_db_calloc(
        db, storage->capacity, sizeof(GV_Metadata *));
    int *del = (int *)gv_db_calloc(db, storage->capacity, sizeof(int));
    if (data == NULL || meta == NULL || del == NULL) {
        if (data != NULL) gv_db_free(db, data);
        if (meta != NULL) gv_db_free(db, meta);
        if (del != NULL) gv_db_free(db, del);
        return;
    }
    if (storage->data != NULL) {
        memcpy(data, storage->data, storage->count * storage->dimension * sizeof(float));
        free(storage->data);
    }
    if (storage->metadata != NULL) {
        memcpy(meta, storage->metadata, storage->capacity * sizeof(GV_Metadata *));
        free(storage->metadata);
    }
    if (storage->deleted != NULL) {
        memcpy(del, storage->deleted, storage->capacity * sizeof(int));
        free(storage->deleted);
    }
    storage->data = data;
    storage->metadata = meta;
    storage->deleted = del;
    storage->owner_db = db;
}

void soa_storage_destroy(GV_SoAStorage *storage) {
    if (storage == NULL) {
        return;
    }

    if (storage->metadata != NULL) {
        for (size_t i = 0; i < storage->count; i++) {
            if (storage->metadata[i] != NULL) {
                GV_Vector temp_vector;
                temp_vector.dimension = storage->dimension;
                temp_vector.data = NULL;
                temp_vector.metadata = storage->metadata[i];
                vector_clear_metadata(&temp_vector);
            }
        }
        if (storage->owner_db == NULL) {
            soa_free(storage, storage->metadata);
        }
    }

    if (storage->owner_db == NULL) {
        soa_free(storage, storage->deleted);
        soa_free(storage, storage->data);
    } else {
        /* Arrays are owned by GV_Memory; db_close tears them down via gv_memory_fini. */
        storage->data = NULL;
        storage->metadata = NULL;
        storage->deleted = NULL;
        storage->owner_db = NULL;
    }
    free(storage);
}

size_t soa_storage_add(GV_SoAStorage *storage, const float *data, GV_Metadata *metadata) {
    if (storage == NULL || data == NULL) {
        return (size_t)-1;
    }

    if (storage->count >= storage->capacity) {
        if (soa_storage_grow(storage, storage->capacity * 2) != 0) {
            return (size_t)-1;
        }
    }

    size_t index = storage->count;
    float *dest = storage->data + (index * storage->dimension);
    memcpy(dest, data, storage->dimension * sizeof(float));
    storage->metadata[index] = metadata;
    storage->deleted[index] = 0;
    storage->count++;

    return index;
}

const float *soa_storage_get_data(const GV_SoAStorage *storage, size_t index) {
    if (storage == NULL || index >= storage->count) {
        return NULL;
    }
    return storage->data + (index * storage->dimension);
}

GV_Metadata *soa_storage_get_metadata(const GV_SoAStorage *storage, size_t index) {
    if (storage == NULL || index >= storage->count) {
        return NULL;
    }
    return storage->metadata[index];
}

int soa_storage_get_vector_view(const GV_SoAStorage *storage, size_t index, GV_Vector *out_vector) {
    if (storage == NULL || out_vector == NULL || index >= storage->count) {
        return -1;
    }

    out_vector->dimension = storage->dimension;
    out_vector->data = (float *)(storage->data + (index * storage->dimension));
    out_vector->metadata = storage->metadata[index];
    return 0;
}

size_t soa_storage_count(const GV_SoAStorage *storage) {
    if (storage == NULL) {
        return 0;
    }
    return storage->count;
}

size_t soa_storage_dimension(const GV_SoAStorage *storage) {
    if (storage == NULL) {
        return 0;
    }
    return storage->dimension;
}

int soa_storage_mark_deleted(GV_SoAStorage *storage, size_t index) {
    if (storage == NULL || index >= storage->count) {
        return -1;
    }
    storage->deleted[index] = 1;
    return 0;
}

int soa_storage_is_deleted(const GV_SoAStorage *storage, size_t index) {
    if (storage == NULL || index >= storage->count) {
        return -1;
    }
    return storage->deleted[index];
}

int soa_storage_update_data(GV_SoAStorage *storage, size_t index, const float *data) {
    if (storage == NULL || data == NULL || index >= storage->count) {
        return -1;
    }
    if (storage->deleted[index] != 0) {
        return -1;
    }
    float *dest = storage->data + (index * storage->dimension);
    memcpy(dest, data, storage->dimension * sizeof(float));
    return 0;
}

int soa_storage_update_metadata(GV_SoAStorage *storage, size_t index, GV_Metadata *metadata) {
    if (storage == NULL || index >= storage->count) {
        return -1;
    }
    if (storage->deleted[index] != 0) {
        return -1;
    }
    if (storage->metadata[index] != NULL) {
        GV_Vector temp_vec = { .dimension = storage->dimension, .data = NULL, .metadata = storage->metadata[index] };
        vector_clear_metadata(&temp_vec);
    }
    storage->metadata[index] = metadata;
    return 0;
}

int soa_storage_save(const GV_SoAStorage *storage, FILE *out, uint32_t version)
{
    if (!storage || !out) return -1;
    (void)version;
    if (write_u32(out, GV_SOA_SAVE_MAGIC) != 0) return -1;
    if (write_u32(out, (uint32_t)storage->dimension) != 0) return -1;
    if (write_u64(out, (uint64_t)storage->count) != 0) return -1;

    for (size_t i = 0; i < storage->count; ++i) {
        if (write_u32(out, (uint32_t)storage->deleted[i]) != 0) return -1;
        if (fwrite(storage->data + i * storage->dimension, sizeof(float),
                   storage->dimension, out) != storage->dimension) {
            return -1;
        }
        if (write_metadata(out, storage->metadata[i]) != 0) return -1;
    }
    return 0;
}

int soa_storage_load(GV_SoAStorage *storage, FILE *in, uint32_t version)
{
    if (!storage || !in) return -1;
    (void)version;

    uint32_t magic = 0, dim = 0;
    uint64_t count = 0;
    if (read_u32(in, &magic) != 0 || magic != GV_SOA_SAVE_MAGIC) return -1;
    if (read_u32(in, &dim) != 0) return -1;
    if (read_u64(in, &count) != 0) return -1;
    if (dim == 0 || count > 100000000u) return -1;
    if (storage->dimension != 0 && storage->dimension != (size_t)dim) return -1;
    storage->dimension = (size_t)dim;

    if (soa_storage_grow(storage, (size_t)count) != 0) return -1;

    for (size_t i = 0; i < storage->count; ++i) {
        if (storage->metadata[i]) {
            GV_Vector tmp = { .dimension = storage->dimension, .data = NULL,
                              .metadata = storage->metadata[i] };
            vector_clear_metadata(&tmp);
            storage->metadata[i] = NULL;
        }
    }

    storage->count = (size_t)count;
    for (size_t i = 0; i < storage->count; ++i) {
        uint32_t deleted = 0;
        if (read_u32(in, &deleted) != 0) return -1;
        storage->deleted[i] = (int)deleted;
        if (fread(storage->data + i * storage->dimension, sizeof(float),
                  storage->dimension, in) != storage->dimension) {
            return -1;
        }
        if (soa_read_metadata(in, &storage->metadata[i]) != 0) return -1;
    }
    return 0;
}

