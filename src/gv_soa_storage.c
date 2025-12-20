#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_soa_storage.h"
#include "gigavector/gv_metadata.h"

GV_SoAStorage *gv_soa_storage_create(size_t dimension, size_t initial_capacity) {
    if (dimension == 0) {
        return NULL;
    }

    GV_SoAStorage *storage = (GV_SoAStorage *)malloc(sizeof(GV_SoAStorage));
    if (storage == NULL) {
        return NULL;
    }

    storage->dimension = dimension;
    storage->count = 0;
    storage->capacity = (initial_capacity > 0) ? initial_capacity : 1024;

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

void gv_soa_storage_destroy(GV_SoAStorage *storage) {
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
                gv_vector_clear_metadata(&temp_vector);
            }
        }
        free(storage->metadata);
    }

    free(storage->deleted);
    free(storage->data);
    free(storage);
}

size_t gv_soa_storage_add(GV_SoAStorage *storage, const float *data, GV_Metadata *metadata) {
    if (storage == NULL || data == NULL) {
        return (size_t)-1;
    }

    if (storage->count >= storage->capacity) {
        size_t new_capacity = storage->capacity * 2;
        size_t new_data_size = new_capacity * storage->dimension * sizeof(float);
        float *new_data = (float *)realloc(storage->data, new_data_size);
        if (new_data == NULL) {
            return (size_t)-1;
        }
        storage->data = new_data;

        GV_Metadata **new_metadata = (GV_Metadata **)realloc(storage->metadata, new_capacity * sizeof(GV_Metadata *));
        if (new_metadata == NULL) {
            return (size_t)-1;
        }
        memset(new_metadata + storage->capacity, 0, (new_capacity - storage->capacity) * sizeof(GV_Metadata *));
        storage->metadata = new_metadata;

        int *new_deleted = (int *)realloc(storage->deleted, new_capacity * sizeof(int));
        if (new_deleted == NULL) {
            return (size_t)-1;
        }
        memset(new_deleted + storage->capacity, 0, (new_capacity - storage->capacity) * sizeof(int));
        storage->deleted = new_deleted;
        storage->capacity = new_capacity;
    }

    size_t index = storage->count;
    float *dest = storage->data + (index * storage->dimension);
    memcpy(dest, data, storage->dimension * sizeof(float));
    storage->metadata[index] = metadata;
    storage->deleted[index] = 0;
    storage->count++;

    return index;
}

const float *gv_soa_storage_get_data(const GV_SoAStorage *storage, size_t index) {
    if (storage == NULL || index >= storage->count) {
        return NULL;
    }
    return storage->data + (index * storage->dimension);
}

GV_Metadata *gv_soa_storage_get_metadata(const GV_SoAStorage *storage, size_t index) {
    if (storage == NULL || index >= storage->count) {
        return NULL;
    }
    return storage->metadata[index];
}

int gv_soa_storage_get_vector_view(const GV_SoAStorage *storage, size_t index, GV_Vector *out_vector) {
    if (storage == NULL || out_vector == NULL || index >= storage->count) {
        return -1;
    }

    out_vector->dimension = storage->dimension;
    out_vector->data = (float *)(storage->data + (index * storage->dimension));
    out_vector->metadata = storage->metadata[index];
    return 0;
}

size_t gv_soa_storage_count(const GV_SoAStorage *storage) {
    if (storage == NULL) {
        return 0;
    }
    return storage->count;
}

size_t gv_soa_storage_dimension(const GV_SoAStorage *storage) {
    if (storage == NULL) {
        return 0;
    }
    return storage->dimension;
}

int gv_soa_storage_mark_deleted(GV_SoAStorage *storage, size_t index) {
    if (storage == NULL || index >= storage->count) {
        return -1;
    }
    storage->deleted[index] = 1;
    return 0;
}

int gv_soa_storage_is_deleted(const GV_SoAStorage *storage, size_t index) {
    if (storage == NULL || index >= storage->count) {
        return -1;
    }
    return storage->deleted[index];
}

