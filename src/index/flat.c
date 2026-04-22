#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "index/flat.h"
#include "search/distance.h"
#include "schema/vector.h"
#include "schema/metadata.h"
#include "storage/soa_storage.h"
#include "core/utils.h"
#include "core/heap.h"

typedef struct {
    size_t dimension;
    GV_FlatConfig config;
    GV_SoAStorage *storage;
    int owns_storage;
} GV_FlatIndex;

typedef struct { float dist; size_t idx; } GV_FlatHeapItem;
GV_HEAP_DEFINE(flat_heap, GV_FlatHeapItem)


void *flat_create(size_t dimension, const GV_FlatConfig *config, GV_SoAStorage *soa_storage) {
    if (dimension == 0) return NULL;

    GV_FlatIndex *idx = (GV_FlatIndex *)calloc(1, sizeof(GV_FlatIndex));
    if (!idx) return NULL;

    idx->dimension = dimension;
    if (config) {
        idx->config = *config;
    } else {
        idx->config.use_simd = 1;
    }

    if (soa_storage) {
        idx->storage = soa_storage;
        idx->owns_storage = 0;
    } else {
        idx->storage = soa_storage_create(dimension, 0);
        if (!idx->storage) {
            free(idx);
            return NULL;
        }
        idx->owns_storage = 1;
    }

    return idx;
}

int flat_insert(void *index, GV_Vector *vector) {
    if (!index || !vector) return -1;
    GV_FlatIndex *idx = (GV_FlatIndex *)index;

    if (vector->dimension != idx->dimension) return -1;

    size_t vi = soa_storage_add(idx->storage, vector->data, vector->metadata);
    if (vi == (size_t)-1) return -1;

    /* Storage took ownership of metadata, clear from vector to prevent double free */
    vector->metadata = NULL;
    vector_destroy(vector);
    return 0;
}


int flat_search(void *index, const GV_Vector *query, size_t k,
                   GV_SearchResult *results, GV_DistanceType distance_type,
                   const char *filter_key, const char *filter_value) {
    if (!index || !query || !results || k == 0) return -1;
    GV_FlatIndex *idx = (GV_FlatIndex *)index;

    if (query->dimension != idx->dimension) return -1;

    size_t count = idx->storage->count;
    if (count == 0) return 0;

    GV_FlatHeapItem *heap = (GV_FlatHeapItem *)malloc(k * sizeof(GV_FlatHeapItem));
    if (!heap) return -1;
    size_t heap_size = 0;

    GV_Vector tmp_vec;
    tmp_vec.dimension = idx->dimension;
    tmp_vec.metadata = NULL;

    for (size_t i = 0; i < count; i++) {
        if (soa_storage_is_deleted(idx->storage, i) == 1) continue;

        if (filter_key && filter_value) {
            GV_Metadata *meta = soa_storage_get_metadata(idx->storage, i);
            if (!metadata_match(meta, filter_key, filter_value)) continue;
        }

        tmp_vec.data = (float *)soa_storage_get_data(idx->storage, i);
        float dist = distance(query, &tmp_vec, distance_type);

        flat_heap_push(heap, &heap_size, k, (GV_FlatHeapItem){dist, i});
    }

    int n = (int)heap_size;
    for (int i = n - 1; i >= 0; i--) {
        size_t vi = heap[0].idx;
        float dist = heap[0].dist;

        GV_Vector view;
        soa_storage_get_vector_view(idx->storage, vi, &view);
        results[i].vector = &view; /* Will be overwritten below */
        results[i].distance = dist;
        results[i].is_sparse = 0;
        results[i].sparse_vector = NULL;
        results[i].id = vi;

        /* Copy vector so result outlives storage */
        GV_Vector *copy = vector_create_from_data(view.dimension, view.data);
        if (copy) {
            GV_Metadata *meta = soa_storage_get_metadata(idx->storage, vi);
            if (meta) {
                GV_Metadata *cur = meta;
                while (cur) {
                    if (cur->key && cur->value) {
                        vector_set_metadata(copy, cur->key, cur->value);
                    }
                    cur = cur->next;
                }
            }
            results[i].vector = copy;
        } else {
            results[i].vector = NULL;
        }

        heap[0] = heap[heap_size - 1];
        heap_size--;
        if (heap_size > 0) {
            flat_heap_sift_down(heap, heap_size, 0);
        }
    }

    free(heap);
    return n;
}

int flat_range_search(void *index, const GV_Vector *query, float radius,
                         GV_SearchResult *results, size_t max_results,
                         GV_DistanceType distance_type,
                         const char *filter_key, const char *filter_value) {
    if (!index || !query || !results || max_results == 0 || radius < 0.0f) return -1;
    GV_FlatIndex *idx = (GV_FlatIndex *)index;

    if (query->dimension != idx->dimension) return -1;

    size_t count = idx->storage->count;
    size_t found = 0;

    GV_Vector tmp_vec;
    tmp_vec.dimension = idx->dimension;
    tmp_vec.metadata = NULL;

    for (size_t i = 0; i < count && found < max_results; i++) {
        if (soa_storage_is_deleted(idx->storage, i) == 1) continue;

        if (filter_key && filter_value) {
            GV_Metadata *meta = soa_storage_get_metadata(idx->storage, i);
            if (!metadata_match(meta, filter_key, filter_value)) continue;
        }

        tmp_vec.data = (float *)soa_storage_get_data(idx->storage, i);
        float dist = distance(query, &tmp_vec, distance_type);

        if (dist <= radius) {
            GV_Vector view;
            soa_storage_get_vector_view(idx->storage, i, &view);
            GV_Vector *copy = vector_create_from_data(view.dimension, view.data);
            if (copy) {
                GV_Metadata *meta = soa_storage_get_metadata(idx->storage, i);
                if (meta) {
                    GV_Metadata *cur = meta;
                    while (cur) {
                        if (cur->key && cur->value) {
                            vector_set_metadata(copy, cur->key, cur->value);
                        }
                        cur = cur->next;
                    }
                }
            }
            results[found].vector = copy;
            results[found].distance = dist;
            results[found].is_sparse = 0;
            results[found].sparse_vector = NULL;
            results[found].id = i;
            found++;
        }
    }

    return (int)found;
}

void flat_destroy(void *index) {
    if (!index) return;
    GV_FlatIndex *idx = (GV_FlatIndex *)index;
    if (idx->owns_storage && idx->storage) {
        soa_storage_destroy(idx->storage);
    }
    free(idx);
}

size_t flat_count(const void *index) {
    if (!index) return 0;
    const GV_FlatIndex *idx = (const GV_FlatIndex *)index;
    size_t total = 0;
    for (size_t i = 0; i < idx->storage->count; i++) {
        if (soa_storage_is_deleted(idx->storage, i) != 1) {
            total++;
        }
    }
    return total;
}

int flat_delete(void *index, size_t vector_index) {
    if (!index) return -1;
    GV_FlatIndex *idx = (GV_FlatIndex *)index;
    return soa_storage_mark_deleted(idx->storage, vector_index);
}

int flat_update(void *index, size_t vector_index, const float *new_data, size_t dimension) {
    if (!index || !new_data) return -1;
    GV_FlatIndex *idx = (GV_FlatIndex *)index;
    if (dimension != idx->dimension) return -1;
    if (soa_storage_is_deleted(idx->storage, vector_index) == 1) return -1;
    return soa_storage_update_data(idx->storage, vector_index, new_data);
}

int flat_save(const void *index, FILE *out, uint32_t version) {
    if (!index || !out) return -1;
    const GV_FlatIndex *idx = (const GV_FlatIndex *)index;
    (void)version;

    if (write_u32(out, (uint32_t)idx->dimension) != 0) return -1;
    if (write_u32(out, (uint32_t)idx->config.use_simd) != 0) return -1;

    uint32_t count = (uint32_t)idx->storage->count;
    if (write_u32(out, count) != 0) return -1;

    for (uint32_t i = 0; i < count; i++) {
        uint32_t deleted = (uint32_t)(soa_storage_is_deleted(idx->storage, i) == 1 ? 1 : 0);
        if (write_u32(out, deleted) != 0) return -1;

        const float *data = soa_storage_get_data(idx->storage, i);
        if (fwrite(data, sizeof(float), idx->dimension, out) != idx->dimension) return -1;

        GV_Metadata *meta = soa_storage_get_metadata(idx->storage, i);
        uint32_t meta_count = 0;
        GV_Metadata *cur = meta;
        while (cur) { meta_count++; cur = cur->next; }
        if (write_u32(out, meta_count) != 0) return -1;
        cur = meta;
        while (cur) {
            uint32_t klen = cur->key ? (uint32_t)strlen(cur->key) : 0;
            uint32_t vlen = cur->value ? (uint32_t)strlen(cur->value) : 0;
            if (write_str(out, cur->key ? cur->key : "", klen) != 0) return -1;
            if (write_str(out, cur->value ? cur->value : "", vlen) != 0) return -1;
            cur = cur->next;
        }
    }

    return 0;
}

int flat_load(void **index_ptr, FILE *in, size_t dimension, uint32_t version) {
    if (!index_ptr || !in) return -1;
    (void)version;

    uint32_t file_dim = 0, use_simd = 0, count = 0;
    if (read_u32(in, &file_dim) != 0) return -1;
    if (read_u32(in, &use_simd) != 0) return -1;
    if (read_u32(in, &count) != 0) return -1;

    if (dimension != 0 && dimension != (size_t)file_dim) return -1;

    GV_FlatConfig config = { .use_simd = (int)use_simd };
    void *index = flat_create((size_t)file_dim, &config, NULL);
    if (!index) return -1;

    GV_FlatIndex *idx = (GV_FlatIndex *)index;

    for (uint32_t i = 0; i < count; i++) {
        uint32_t deleted = 0;
        if (read_u32(in, &deleted) != 0) { flat_destroy(index); return -1; }

        float *data = (float *)malloc(file_dim * sizeof(float));
        if (!data) { flat_destroy(index); return -1; }
        if (fread(data, sizeof(float), file_dim, in) != file_dim) {
            free(data);
            flat_destroy(index);
            return -1;
        }

        uint32_t meta_count = 0;
        if (read_u32(in, &meta_count) != 0) { free(data); flat_destroy(index); return -1; }

        GV_Metadata *metadata = NULL;
        for (uint32_t m = 0; m < meta_count; m++) {
            uint32_t klen = 0, vlen = 0;
            char *key = NULL, *value = NULL;
            if (read_u32(in, &klen) != 0) { free(data); flat_destroy(index); return -1; }
            if (read_str(in, &key, klen) != 0) { free(data); flat_destroy(index); return -1; }
            if (read_u32(in, &vlen) != 0) { free(key); free(data); flat_destroy(index); return -1; }
            if (read_str(in, &value, vlen) != 0) { free(key); free(data); flat_destroy(index); return -1; }

            GV_Metadata *node = (GV_Metadata *)malloc(sizeof(GV_Metadata));
            if (!node) { free(key); free(value); free(data); flat_destroy(index); return -1; }
            node->key = key;
            node->value = value;
            node->next = metadata;
            metadata = node;
        }

        size_t vi = soa_storage_add(idx->storage, data, metadata);
        free(data);
        if (vi == (size_t)-1) { flat_destroy(index); return -1; }

        if (deleted) {
            soa_storage_mark_deleted(idx->storage, vi);
        }
    }

    *index_ptr = index;
    return 0;
}
