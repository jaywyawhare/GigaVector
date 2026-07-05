#include <stdio.h>
#include "core/memory.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "index/ivfturboquant.h"
#include "search/distance.h"
#include "schema/vector.h"
#include "schema/metadata.h"
#include "core/heap.h"
#include "core/utils.h"

typedef struct GV_IVFTurboQuantEntry {
    GV_Vector *vector;
    GV_TurboQuantCode *code;
    size_t id;
    int deleted;
    struct GV_IVFTurboQuantEntry *next;
} GV_IVFTurboQuantEntry;

typedef struct {
    size_t dimension;
    GV_IVFTurboQuantConfig config;
    float *centroids;
    GV_TurboQuantizer *quantizer;
    GV_IVFTurboQuantEntry **lists;
    size_t *list_sizes;
    int trained;
    size_t total_count;
    size_t next_id;
} GV_IVFTurboQuantIndex;

typedef struct {
    float dist;
    size_t id;
    GV_IVFTurboQuantEntry *entry;
} GV_IVFTurboQuantHeapItem;

GV_HEAP_DEFINE(ivfturboquant_heap, GV_IVFTurboQuantHeapItem)

static void ivfturboquant_default_config(size_t dimension, GV_IVFTurboQuantConfig *cfg) {
    cfg->nlist = 64;
    cfg->nprobe = 4;
    cfg->train_iters = 15;
    cfg->use_cosine = 0;
    cfg->default_rerank = 200;
    cfg->turbo.bits = 8;
    cfg->turbo.projections = dimension / 4;
    if (cfg->turbo.projections == 0) {
        cfg->turbo.projections = 2;
    }
    cfg->turbo.seed = 42;
    cfg->turbo.use_qjl = 1;
    cfg->turbo.rotation = GV_TURBOQUANT_ROTATION_AUTO;
}

static void ivfturboquant_argmin(const float *data, size_t count, size_t dim,
                                 const float *centroids, size_t k, int *assign) {
    for (size_t i = 0; i < count; i++) {
        const float *vec = data + i * dim;
        float best_dist = INFINITY;
        int best_idx = -1;
        for (size_t c = 0; c < k; c++) {
            const float *centroid = centroids + c * dim;
            float dist = 0.0f;
            for (size_t d = 0; d < dim; d++) {
                float diff = vec[d] - centroid[d];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = (int)c;
            }
        }
        assign[i] = best_idx;
    }
}

static int ivfturboquant_kmeans(const float *data, size_t count, size_t dim,
                                size_t k, size_t iters, float *out_centroids) {
    if (count < k || !data || !out_centroids) {
        return -1;
    }

    memcpy(out_centroids, data, k * dim * sizeof(float));

    int *assign = (int *)gv_alloc(count * sizeof(int));
    float *new_centroids = (float *)gv_calloc(k * dim, sizeof(float));
    size_t *counts = (size_t *)gv_calloc(k, sizeof(size_t));

    if (!assign || !new_centroids || !counts) {
        gv_free(assign);
        gv_free(new_centroids);
        gv_free(counts);
        return -1;
    }

    for (size_t iter = 0; iter < iters; iter++) {
        ivfturboquant_argmin(data, count, dim, out_centroids, k, assign);

        memset(new_centroids, 0, k * dim * sizeof(float));
        memset(counts, 0, k * sizeof(size_t));

        for (size_t i = 0; i < count; i++) {
            int c = assign[i];
            if (c < 0) {
                continue;
            }
            const float *vec = data + i * dim;
            for (size_t d = 0; d < dim; d++) {
                new_centroids[c * dim + d] += vec[d];
            }
            counts[c]++;
        }

        for (size_t c = 0; c < k; c++) {
            if (counts[c] > 0) {
                for (size_t d = 0; d < dim; d++) {
                    new_centroids[c * dim + d] /= (float)counts[c];
                }
            }
        }

        memcpy(out_centroids, new_centroids, k * dim * sizeof(float));
    }

    gv_free(assign);
    gv_free(new_centroids);
    gv_free(counts);
    return 0;
}

static float ivfturboquant_entry_distance(const GV_IVFTurboQuantIndex *idx,
                                          const GV_Vector *query,
                                          const GV_IVFTurboQuantEntry *entry,
                                          GV_DistanceType distance_type,
                                          const GV_TurboQuantQuery *prepared,
                                          int exact) {
    if (entry == NULL) {
        return -1.0f;
    }
    if (exact && entry->vector != NULL && entry->vector->data != NULL) {
        return distance(query, entry->vector, distance_type);
    }
    if (entry->code != NULL && idx->quantizer != NULL && prepared != NULL) {
        return turboquant_distance_prepared(idx->quantizer, entry->code, prepared, distance_type);
    }
    if (entry->vector != NULL) {
        return distance(query, entry->vector, distance_type);
    }
    return -1.0f;
}

static void ivfturboquant_copy_result(GV_SearchResult *out, const GV_IVFTurboQuantEntry *entry,
                                      float dist) {
    GV_Vector *copy = vector_create_from_data(entry->vector->dimension, entry->vector->data);
    if (copy) {
        GV_Metadata *meta = entry->vector->metadata;
        while (meta) {
            if (meta->key && meta->value) {
                vector_set_metadata(copy, meta->key, meta->value);
            }
            meta = meta->next;
        }
        out->vector = copy;
    } else {
        out->vector = NULL;
    }
    out->distance = dist;
    out->is_sparse = 0;
    out->sparse_vector = NULL;
    out->id = entry->id;
}

void *ivfturboquant_create(size_t dimension, const GV_IVFTurboQuantConfig *config) {
    if (dimension == 0 || dimension % 2 != 0) {
        return NULL;
    }

    GV_IVFTurboQuantIndex *idx = (GV_IVFTurboQuantIndex *)gv_calloc(1, sizeof(GV_IVFTurboQuantIndex));
    if (!idx) {
        return NULL;
    }

    idx->dimension = dimension;

    if (config) {
        idx->config = *config;
    } else {
        ivfturboquant_default_config(dimension, &idx->config);
    }

    if (idx->config.nprobe > idx->config.nlist) {
        idx->config.nprobe = idx->config.nlist;
    }

    idx->centroids = (float *)gv_alloc(idx->config.nlist * dimension * sizeof(float));
    idx->lists = (GV_IVFTurboQuantEntry **)gv_calloc(idx->config.nlist, sizeof(GV_IVFTurboQuantEntry *));
    idx->list_sizes = (size_t *)gv_calloc(idx->config.nlist, sizeof(size_t));

    if (!idx->centroids || !idx->lists || !idx->list_sizes) {
        gv_free(idx->centroids);
        gv_free(idx->lists);
        gv_free(idx->list_sizes);
        gv_free(idx);
        return NULL;
    }

    idx->quantizer = turboquant_create(dimension, &idx->config.turbo);
    if (idx->quantizer == NULL) {
        gv_free(idx->centroids);
        gv_free(idx->lists);
        gv_free(idx->list_sizes);
        gv_free(idx);
        return NULL;
    }

    idx->trained = 0;
    idx->total_count = 0;
    idx->next_id = 0;

    return idx;
}

int ivfturboquant_train(void *index, const float *data, size_t count) {
    if (!index || !data || count == 0) {
        return -1;
    }

    GV_IVFTurboQuantIndex *idx = (GV_IVFTurboQuantIndex *)index;
    if (count < idx->config.nlist) {
        return -1;
    }

    size_t total = count * idx->dimension;
    float *train_buf = (float *)gv_alloc(total * sizeof(float));
    if (!train_buf) {
        return -1;
    }
    memcpy(train_buf, data, total * sizeof(float));

    if (idx->config.use_cosine) {
        for (size_t i = 0; i < count; ++i) {
            float norm = 0.0f;
            float *v = train_buf + i * idx->dimension;
            for (size_t j = 0; j < idx->dimension; ++j) {
                norm += v[j] * v[j];
            }
            if (norm > 0.0f) {
                norm = 1.0f / sqrtf(norm);
                for (size_t j = 0; j < idx->dimension; ++j) {
                    v[j] *= norm;
                }
            }
        }
    }

    if (ivfturboquant_kmeans(train_buf, count, idx->dimension, idx->config.nlist,
                             idx->config.train_iters, idx->centroids) != 0) {
        gv_free(train_buf);
        return -1;
    }

    gv_free(train_buf);
    idx->trained = 1;
    return 0;
}

int ivfturboquant_insert(void *index, GV_Vector *vector) {
    if (!index || !vector) {
        return -1;
    }

    GV_IVFTurboQuantIndex *idx = (GV_IVFTurboQuantIndex *)index;
    if (!idx->trained || vector->dimension != idx->dimension || idx->quantizer == NULL) {
        return -1;
    }

    float best_dist = INFINITY;
    size_t best_list = 0;

    for (size_t i = 0; i < idx->config.nlist; i++) {
        const float *centroid = idx->centroids + i * idx->dimension;
        float dist = 0.0f;
        for (size_t d = 0; d < idx->dimension; d++) {
            float diff = vector->data[d] - centroid[d];
            dist += diff * diff;
        }
        if (dist < best_dist) {
            best_dist = dist;
            best_list = i;
        }
    }

    GV_IVFTurboQuantEntry *entry = (GV_IVFTurboQuantEntry *)gv_alloc(sizeof(GV_IVFTurboQuantEntry));
    if (!entry) {
        return -1;
    }

    entry->vector = vector;
    entry->code = turboquant_encode(idx->quantizer, vector->data);
    if (entry->code == NULL) {
        gv_free(entry);
        return -1;
    }

    entry->id = idx->next_id++;
    entry->deleted = 0;
    entry->next = idx->lists[best_list];
    idx->lists[best_list] = entry;
    idx->list_sizes[best_list]++;
    idx->total_count++;

    return 0;
}

int ivfturboquant_search(void *index, const GV_Vector *query, size_t k,
                         GV_SearchResult *results, GV_DistanceType distance_type,
                         const char *filter_key, const char *filter_value) {
    if (!index || !query || !results || k == 0) {
        return -1;
    }

    GV_IVFTurboQuantIndex *idx = (GV_IVFTurboQuantIndex *)index;
    if (!idx->trained || query->dimension != idx->dimension || idx->quantizer == NULL) {
        return -1;
    }

    GV_TurboQuantQuery prepared;
    if (turboquant_prepare_query(idx->quantizer, query->data, &prepared) != 0) {
        return -1;
    }

    size_t nprobe = idx->config.nprobe;
    if (nprobe > idx->config.nlist) {
        nprobe = idx->config.nlist;
    }

    size_t heap_cap = k;
    if (idx->config.default_rerank > heap_cap) {
        heap_cap = idx->config.default_rerank;
    }

    GV_IVFTurboQuantHeapItem *centroid_heap = (GV_IVFTurboQuantHeapItem *)gv_alloc(
        nprobe * sizeof(GV_IVFTurboQuantHeapItem));
    if (!centroid_heap) {
        turboquant_query_destroy(&prepared);
        return -1;
    }

    size_t heap_size = 0;
    for (size_t i = 0; i < idx->config.nlist; i++) {
        const float *centroid = idx->centroids + i * idx->dimension;
        float dist = 0.0f;
        for (size_t d = 0; d < idx->dimension; d++) {
            float diff = query->data[d] - centroid[d];
            dist += diff * diff;
        }
        ivfturboquant_heap_push(centroid_heap, &heap_size, nprobe,
                                (GV_IVFTurboQuantHeapItem){dist, i, NULL});
    }

    size_t *probe_lists = (size_t *)gv_alloc(nprobe * sizeof(size_t));
    if (!probe_lists) {
        gv_free(centroid_heap);
        turboquant_query_destroy(&prepared);
        return -1;
    }

    for (size_t i = nprobe; i > 0; i--) {
        probe_lists[i - 1] = centroid_heap[0].id;
        centroid_heap[0] = centroid_heap[heap_size - 1];
        heap_size--;
        if (heap_size > 0) {
            ivfturboquant_heap_sift_down(centroid_heap, heap_size, 0);
        }
    }
    gv_free(centroid_heap);

    GV_IVFTurboQuantHeapItem *heap = (GV_IVFTurboQuantHeapItem *)gv_alloc(
        heap_cap * sizeof(GV_IVFTurboQuantHeapItem));
    if (!heap) {
        gv_free(probe_lists);
        turboquant_query_destroy(&prepared);
        return -1;
    }

    heap_size = 0;
    for (size_t i = 0; i < nprobe; i++) {
        GV_IVFTurboQuantEntry *entry = idx->lists[probe_lists[i]];
        while (entry) {
            if (!entry->deleted &&
                metadata_match(entry->vector->metadata, filter_key, filter_value)) {
                float dist = ivfturboquant_entry_distance(idx, query, entry, distance_type,
                                                          &prepared, 0);
                if (dist >= 0.0f) {
                    ivfturboquant_heap_push(heap, &heap_size, heap_cap,
                                            (GV_IVFTurboQuantHeapItem){dist, entry->id, entry});
                }
            }
            entry = entry->next;
        }
    }
    gv_free(probe_lists);

    size_t found = heap_size;
    GV_IVFTurboQuantHeapItem *candidates = (GV_IVFTurboQuantHeapItem *)gv_alloc(
        found * sizeof(GV_IVFTurboQuantHeapItem));
    if (!candidates) {
        gv_free(heap);
        turboquant_query_destroy(&prepared);
        return -1;
    }

    for (size_t i = found; i > 0; i--) {
        candidates[i - 1] = heap[0];
        heap[0] = heap[heap_size - 1];
        heap_size--;
        if (heap_size > 0) {
            ivfturboquant_heap_sift_down(heap, heap_size, 0);
        }
    }
    gv_free(heap);

    if (idx->config.default_rerank > 0 && found > 0) {
        size_t rr = idx->config.default_rerank;
        if (rr > found) {
            rr = found;
        }
        for (size_t i = 0; i < rr; ++i) {
            if (candidates[i].entry == NULL) {
                continue;
            }
            float exact = ivfturboquant_entry_distance(idx, query, candidates[i].entry,
                                                       distance_type, &prepared, 1);
            if (exact >= 0.0f) {
                candidates[i].dist = exact;
            }
        }
    }

    turboquant_query_destroy(&prepared);

    size_t result_count = (found < k) ? found : k;
    for (size_t i = 0; i < result_count; ++i) {
        size_t minj = i;
        for (size_t j = i + 1; j < found; ++j) {
            if (candidates[j].dist < candidates[minj].dist) {
                minj = j;
            }
        }
        if (minj != i) {
            GV_IVFTurboQuantHeapItem tmp = candidates[i];
            candidates[i] = candidates[minj];
            candidates[minj] = tmp;
        }
        ivfturboquant_copy_result(&results[i], candidates[i].entry, candidates[i].dist);
    }

    gv_free(candidates);
    return (int)result_count;
}

int ivfturboquant_range_search(void *index, const GV_Vector *query, float radius,
                               GV_SearchResult *results, size_t max_results,
                               GV_DistanceType distance_type,
                               const char *filter_key, const char *filter_value) {
    if (!index || !query || !results || max_results == 0 || radius < 0.0f) {
        return -1;
    }

    GV_IVFTurboQuantIndex *idx = (GV_IVFTurboQuantIndex *)index;
    if (!idx->trained || query->dimension != idx->dimension) {
        return -1;
    }

    size_t nprobe = idx->config.nprobe;
    if (nprobe > idx->config.nlist) {
        nprobe = idx->config.nlist;
    }

    GV_IVFTurboQuantHeapItem *centroid_heap = (GV_IVFTurboQuantHeapItem *)gv_alloc(
        nprobe * sizeof(GV_IVFTurboQuantHeapItem));
    if (!centroid_heap) {
        return -1;
    }

    size_t heap_size = 0;
    for (size_t i = 0; i < idx->config.nlist; i++) {
        const float *centroid = idx->centroids + i * idx->dimension;
        float dist = 0.0f;
        for (size_t d = 0; d < idx->dimension; d++) {
            float diff = query->data[d] - centroid[d];
            dist += diff * diff;
        }
        ivfturboquant_heap_push(centroid_heap, &heap_size, nprobe,
                                (GV_IVFTurboQuantHeapItem){dist, i, NULL});
    }

    size_t *probe_lists = (size_t *)gv_alloc(nprobe * sizeof(size_t));
    if (!probe_lists) {
        gv_free(centroid_heap);
        return -1;
    }

    for (size_t i = nprobe; i > 0; i--) {
        probe_lists[i - 1] = centroid_heap[0].id;
        centroid_heap[0] = centroid_heap[heap_size - 1];
        heap_size--;
        if (heap_size > 0) {
            ivfturboquant_heap_sift_down(centroid_heap, heap_size, 0);
        }
    }
    gv_free(centroid_heap);

    size_t found = 0;
    for (size_t i = 0; i < nprobe && found < max_results; i++) {
        GV_IVFTurboQuantEntry *entry = idx->lists[probe_lists[i]];
        while (entry && found < max_results) {
            if (!entry->deleted &&
                metadata_match(entry->vector->metadata, filter_key, filter_value)) {
                float dist = ivfturboquant_entry_distance(idx, query, entry, distance_type, NULL, 1);
                if (dist >= 0.0f && dist <= radius) {
                    ivfturboquant_copy_result(&results[found], entry, dist);
                    found++;
                }
            }
            entry = entry->next;
        }
    }

    gv_free(probe_lists);
    return (int)found;
}

int ivfturboquant_is_trained(const void *index) {
    if (!index) {
        return 0;
    }
    const GV_IVFTurboQuantIndex *idx = (const GV_IVFTurboQuantIndex *)index;
    return idx->trained;
}

void ivfturboquant_destroy(void *index) {
    if (!index) {
        return;
    }

    GV_IVFTurboQuantIndex *idx = (GV_IVFTurboQuantIndex *)index;

    if (idx->lists) {
        for (size_t i = 0; i < idx->config.nlist; i++) {
            GV_IVFTurboQuantEntry *entry = idx->lists[i];
            while (entry) {
                GV_IVFTurboQuantEntry *next = entry->next;
                if (entry->vector) {
                    vector_destroy(entry->vector);
                }
                if (entry->code) {
                    turboquant_code_destroy(entry->code);
                }
                gv_free(entry);
                entry = next;
            }
        }
        gv_free(idx->lists);
    }

    if (idx->quantizer) {
        turboquant_destroy(idx->quantizer);
    }

    gv_free(idx->centroids);
    gv_free(idx->list_sizes);
    gv_free(idx);
}

size_t ivfturboquant_count(const void *index) {
    if (!index) {
        return 0;
    }
    const GV_IVFTurboQuantIndex *idx = (const GV_IVFTurboQuantIndex *)index;

    size_t count = 0;
    for (size_t i = 0; i < idx->config.nlist; i++) {
        GV_IVFTurboQuantEntry *entry = idx->lists[i];
        while (entry) {
            if (!entry->deleted) {
                count++;
            }
            entry = entry->next;
        }
    }
    return count;
}

int ivfturboquant_delete(void *index, size_t entry_index) {
    if (!index) {
        return -1;
    }

    GV_IVFTurboQuantIndex *idx = (GV_IVFTurboQuantIndex *)index;
    for (size_t i = 0; i < idx->config.nlist; i++) {
        GV_IVFTurboQuantEntry *entry = idx->lists[i];
        while (entry) {
            if (entry->id == entry_index) {
                if (entry->deleted) {
                    return -1;
                }
                entry->deleted = 1;
                return 0;
            }
            entry = entry->next;
        }
    }
    return -1;
}

int ivfturboquant_update(void *index, size_t entry_index, const float *new_data, size_t dimension) {
    if (!index || !new_data) {
        return -1;
    }

    GV_IVFTurboQuantIndex *idx = (GV_IVFTurboQuantIndex *)index;
    if (dimension != idx->dimension || idx->quantizer == NULL) {
        return -1;
    }

    for (size_t i = 0; i < idx->config.nlist; i++) {
        GV_IVFTurboQuantEntry *entry = idx->lists[i];
        while (entry) {
            if (entry->id == entry_index) {
                if (entry->deleted || entry->vector == NULL || entry->vector->data == NULL) {
                    return -1;
                }
                memcpy(entry->vector->data, new_data, dimension * sizeof(float));
                if (entry->code) {
                    turboquant_code_destroy(entry->code);
                }
                entry->code = turboquant_encode(idx->quantizer, new_data);
                return entry->code != NULL ? 0 : -1;
            }
            entry = entry->next;
        }
    }
    return -1;
}

int ivfturboquant_save(const void *index, FILE *out, uint32_t version) {
    if (!index || !out) {
        return -1;
    }
    (void)version;

    const GV_IVFTurboQuantIndex *idx = (const GV_IVFTurboQuantIndex *)index;

    if (write_u32(out, (uint32_t)idx->dimension) != 0) return -1;
    if (write_u32(out, (uint32_t)idx->config.nlist) != 0) return -1;
    if (write_u32(out, (uint32_t)idx->config.nprobe) != 0) return -1;
    if (write_u32(out, (uint32_t)idx->config.train_iters) != 0) return -1;
    if (write_u32(out, (uint32_t)idx->config.use_cosine) != 0) return -1;
    if (write_u32(out, (uint32_t)idx->config.default_rerank) != 0) return -1;
    if (write_u32(out, (uint32_t)idx->config.turbo.bits) != 0) return -1;
    if (write_u32(out, (uint32_t)idx->config.turbo.projections) != 0) return -1;
    if (write_u32(out, (uint32_t)(idx->config.turbo.seed & 0xFFFFFFFFu)) != 0) return -1;
    if (write_u32(out, (uint32_t)(idx->config.turbo.seed >> 32)) != 0) return -1;
    if (write_u32(out, (uint32_t)idx->config.turbo.use_qjl) != 0) return -1;
    if (write_u32(out, (uint32_t)idx->config.turbo.rotation) != 0) return -1;
    if (write_u32(out, (uint32_t)idx->trained) != 0) return -1;
    if (write_u32(out, (uint32_t)idx->next_id) != 0) return -1;

    if (idx->trained) {
        size_t centroid_floats = idx->config.nlist * idx->dimension;
        if (fwrite(idx->centroids, sizeof(float), centroid_floats, out) != centroid_floats) {
            return -1;
        }
    }

    if (write_u32(out, (uint32_t)idx->config.nlist) != 0) return -1;

    for (size_t i = 0; i < idx->config.nlist; i++) {
        uint32_t list_count = 0;
        GV_IVFTurboQuantEntry *entry = idx->lists[i];
        while (entry) {
            list_count++;
            entry = entry->next;
        }

        if (write_u32(out, list_count) != 0) return -1;

        entry = idx->lists[i];
        while (entry) {
            if (write_u32(out, (uint32_t)entry->id) != 0) return -1;
            if (write_u32(out, (uint32_t)entry->deleted) != 0) return -1;

            if (fwrite(entry->vector->data, sizeof(float), idx->dimension, out) != idx->dimension) {
                return -1;
            }

            uint32_t meta_count = 0;
            GV_Metadata *meta = entry->vector->metadata;
            while (meta) {
                meta_count++;
                meta = meta->next;
            }
            if (write_u32(out, meta_count) != 0) return -1;

            meta = entry->vector->metadata;
            while (meta) {
                uint32_t klen = meta->key ? (uint32_t)strlen(meta->key) : 0;
                uint32_t vlen = meta->value ? (uint32_t)strlen(meta->value) : 0;
                if (write_str(out, meta->key ? meta->key : "", klen) != 0) return -1;
                if (write_str(out, meta->value ? meta->value : "", vlen) != 0) return -1;
                meta = meta->next;
            }

            entry = entry->next;
        }
    }

    return 0;
}

int ivfturboquant_load(void **index_ptr, FILE *in, size_t dimension, uint32_t version) {
    if (!index_ptr || !in) {
        return -1;
    }
    (void)version;

    uint32_t file_dim = 0, nlist = 0, nprobe = 0, train_iters = 0;
    uint32_t use_cosine = 0, default_rerank = 0;
    uint32_t turbo_bits = 0, turbo_projections = 0;
    uint32_t turbo_seed_lo = 0, turbo_seed_hi = 0;
    uint32_t turbo_use_qjl = 0, turbo_rotation = 0;
    uint32_t trained = 0, next_id = 0;

    if (read_u32(in, &file_dim) != 0) return -1;
    if (read_u32(in, &nlist) != 0) return -1;
    if (read_u32(in, &nprobe) != 0) return -1;
    if (read_u32(in, &train_iters) != 0) return -1;
    if (read_u32(in, &use_cosine) != 0) return -1;
    if (read_u32(in, &default_rerank) != 0) return -1;
    if (read_u32(in, &turbo_bits) != 0) return -1;
    if (read_u32(in, &turbo_projections) != 0) return -1;
    if (read_u32(in, &turbo_seed_lo) != 0) return -1;
    if (read_u32(in, &turbo_seed_hi) != 0) return -1;
    if (read_u32(in, &turbo_use_qjl) != 0) return -1;
    if (read_u32(in, &turbo_rotation) != 0) return -1;
    if (read_u32(in, &trained) != 0) return -1;
    if (read_u32(in, &next_id) != 0) return -1;

    if (dimension != 0 && dimension != (size_t)file_dim) {
        return -1;
    }

    GV_IVFTurboQuantConfig config = {
        .nlist = nlist,
        .nprobe = nprobe,
        .train_iters = train_iters,
        .use_cosine = (int)use_cosine,
        .default_rerank = default_rerank,
        .turbo = {
            .bits = (uint8_t)turbo_bits,
            .projections = turbo_projections,
            .seed = ((uint64_t)turbo_seed_hi << 32) | turbo_seed_lo,
            .use_qjl = (int)turbo_use_qjl,
            .rotation = (GV_TurboQuantRotation)turbo_rotation
        }
    };

    void *index = ivfturboquant_create((size_t)file_dim, &config);
    if (!index) {
        return -1;
    }

    GV_IVFTurboQuantIndex *idx = (GV_IVFTurboQuantIndex *)index;
    idx->trained = (int)trained;
    idx->next_id = (size_t)next_id;

    if (trained) {
        size_t centroid_floats = idx->config.nlist * idx->dimension;
        if (fread(idx->centroids, sizeof(float), centroid_floats, in) != centroid_floats) {
            ivfturboquant_destroy(index);
            return -1;
        }
    }

    uint32_t num_lists = 0;
    if (read_u32(in, &num_lists) != 0) {
        ivfturboquant_destroy(index);
        return -1;
    }
    if (num_lists != nlist) {
        ivfturboquant_destroy(index);
        return -1;
    }

    for (size_t i = 0; i < nlist; i++) {
        uint32_t list_count = 0;
        if (read_u32(in, &list_count) != 0) {
            ivfturboquant_destroy(index);
            return -1;
        }

        GV_IVFTurboQuantEntry **tail = &idx->lists[i];
        for (uint32_t j = 0; j < list_count; j++) {
            uint32_t entry_id = 0, deleted = 0;
            if (read_u32(in, &entry_id) != 0) {
                ivfturboquant_destroy(index);
                return -1;
            }
            if (read_u32(in, &deleted) != 0) {
                ivfturboquant_destroy(index);
                return -1;
            }

            float *data = (float *)gv_alloc(idx->dimension * sizeof(float));
            if (!data) {
                ivfturboquant_destroy(index);
                return -1;
            }
            if (fread(data, sizeof(float), idx->dimension, in) != idx->dimension) {
                gv_free(data);
                ivfturboquant_destroy(index);
                return -1;
            }

            GV_Vector *vec = vector_create_from_data(idx->dimension, data);
            gv_free(data);
            if (!vec) {
                ivfturboquant_destroy(index);
                return -1;
            }

            GV_IVFTurboQuantEntry *entry = (GV_IVFTurboQuantEntry *)gv_alloc(sizeof(GV_IVFTurboQuantEntry));
            if (!entry) {
                vector_destroy(vec);
                ivfturboquant_destroy(index);
                return -1;
            }

            entry->vector = vec;
            entry->id = (size_t)entry_id;
            entry->deleted = (int)deleted;
            entry->next = NULL;
            entry->code = turboquant_encode(idx->quantizer, vec->data);
            if (entry->code == NULL) {
                vector_destroy(vec);
                gv_free(entry);
                ivfturboquant_destroy(index);
                return -1;
            }

            uint32_t meta_count = 0;
            if (read_u32(in, &meta_count) != 0) {
                turboquant_code_destroy(entry->code);
                vector_destroy(vec);
                gv_free(entry);
                ivfturboquant_destroy(index);
                return -1;
            }

            for (uint32_t m = 0; m < meta_count; m++) {
                uint32_t klen = 0, vlen = 0;
                char *key = NULL, *value = NULL;

                if (read_u32(in, &klen) != 0) goto meta_fail;
                if (read_str(in, &key, klen) != 0) goto meta_fail;
                if (read_u32(in, &vlen) != 0) goto meta_fail;
                if (read_str(in, &value, vlen) != 0) goto meta_fail;

                if (vector_set_metadata(vec, key, value) != 0) {
                    gv_free(key);
                    gv_free(value);
                    goto meta_fail;
                }
                gv_free(key);
                gv_free(value);
                continue;

            meta_fail:
                if (key) gv_free(key);
                if (value) gv_free(value);
                turboquant_code_destroy(entry->code);
                vector_destroy(vec);
                gv_free(entry);
                ivfturboquant_destroy(index);
                return -1;
            }

            *tail = entry;
            tail = &entry->next;
            idx->list_sizes[i]++;
            idx->total_count++;
        }
    }

    *index_ptr = index;
    return 0;
}
