#include <stdio.h>
#include "core/memory.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "index/ivfsq8.h"
#include "search/distance.h"
#include "schema/vector.h"
#include "schema/metadata.h"
#include "core/heap.h"
#include "core/utils.h"

typedef struct GV_IVFSQ8Entry {
    GV_Vector *vector;
    GV_ScalarQuantVector *scalar_quant;
    size_t id;
    int deleted;
    struct GV_IVFSQ8Entry *next;
} GV_IVFSQ8Entry;

typedef struct {
    size_t dimension;
    GV_IVFSQ8Config config;
    float *centroids;
    GV_ScalarQuantVector *scalar_quant_template;
    GV_IVFSQ8Entry **lists;
    size_t *list_sizes;
    int trained;
    size_t total_count;
    size_t next_id;
} GV_IVFSQ8Index;

typedef struct {
    float dist;
    size_t id;
    GV_IVFSQ8Entry *entry;
} GV_IVFSQ8HeapItem;

GV_HEAP_DEFINE(ivfsq8_heap, GV_IVFSQ8HeapItem)

static void ivfsq8_argmin(const float *data, size_t count, size_t dim,
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

static int ivfsq8_kmeans(const float *data, size_t count, size_t dim,
                         size_t k, size_t iters, float *out_centroids) {
    if (count < k || !data || !out_centroids) return -1;

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
        ivfsq8_argmin(data, count, dim, out_centroids, k, assign);

        memset(new_centroids, 0, k * dim * sizeof(float));
        memset(counts, 0, k * sizeof(size_t));

        for (size_t i = 0; i < count; i++) {
            int c = assign[i];
            if (c < 0) continue;
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

static GV_ScalarQuantVector *ivfsq8_quantize_vector(const float *data, const GV_IVFSQ8Index *idx) {
    const GV_ScalarQuantVector *tmpl = idx->scalar_quant_template;
    if (tmpl == NULL || data == NULL) {
        return NULL;
    }

    GV_ScalarQuantVector *sqv = (GV_ScalarQuantVector *)gv_alloc(sizeof(GV_ScalarQuantVector));
    if (sqv == NULL) {
        return NULL;
    }

    sqv->dimension = idx->dimension;
    sqv->bits = 8;
    sqv->per_dimension = idx->config.per_dimension;
    sqv->bytes_per_vector = tmpl->bytes_per_vector;

    size_t nvals = sqv->per_dimension ? idx->dimension : 1;
    sqv->min_vals = (float *)gv_alloc(nvals * sizeof(float));
    sqv->max_vals = (float *)gv_alloc(nvals * sizeof(float));
    if (sqv->min_vals == NULL || sqv->max_vals == NULL) {
        gv_free(sqv->min_vals);
        gv_free(sqv->max_vals);
        gv_free(sqv);
        return NULL;
    }
    memcpy(sqv->min_vals, tmpl->min_vals, nvals * sizeof(float));
    memcpy(sqv->max_vals, tmpl->max_vals, nvals * sizeof(float));

    sqv->quantized = (uint8_t *)gv_calloc(sqv->bytes_per_vector, sizeof(uint8_t));
    if (sqv->quantized == NULL) {
        gv_free(sqv->min_vals);
        gv_free(sqv->max_vals);
        gv_free(sqv);
        return NULL;
    }

    size_t max_quant = 255;
    for (size_t i = 0; i < idx->dimension; ++i) {
        float min_val = sqv->per_dimension ? sqv->min_vals[i] : sqv->min_vals[0];
        float max_val = sqv->per_dimension ? sqv->max_vals[i] : sqv->max_vals[0];
        float range = max_val - min_val;
        if (range <= 0.0f) {
            continue;
        }
        float normalized = (data[i] - min_val) / range;
        normalized = (normalized < 0.0f) ? 0.0f : (normalized > 1.0f) ? 1.0f : normalized;
        size_t quantized_val = (size_t)(normalized * max_quant + 0.5f);
        if (quantized_val > max_quant) {
            quantized_val = max_quant;
        }
        sqv->quantized[i] = (uint8_t)quantized_val;
    }

    return sqv;
}

static float ivfsq8_entry_distance(const GV_IVFSQ8Index *idx, const GV_Vector *query,
                                   const GV_IVFSQ8Entry *entry, GV_DistanceType distance_type,
                                   int exact) {
    if (entry == NULL) {
        return -1.0f;
    }
    if (exact && entry->vector != NULL && entry->vector->data != NULL) {
        return distance(query, entry->vector, distance_type);
    }
    if (entry->scalar_quant != NULL) {
        return scalar_quant_distance(query->data, entry->scalar_quant, (int)distance_type);
    }
    if (entry->vector != NULL) {
        return distance(query, entry->vector, distance_type);
    }
    return -1.0f;
}

static void ivfsq8_copy_result(GV_SearchResult *out, const GV_IVFSQ8Entry *entry, float dist) {
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

void *ivfsq8_create(size_t dimension, const GV_IVFSQ8Config *config) {
    if (dimension == 0) {
        return NULL;
    }

    GV_IVFSQ8Index *idx = (GV_IVFSQ8Index *)gv_calloc(1, sizeof(GV_IVFSQ8Index));
    if (!idx) {
        return NULL;
    }

    idx->dimension = dimension;

    if (config) {
        idx->config = *config;
    } else {
        idx->config.nlist = 64;
        idx->config.nprobe = 4;
        idx->config.train_iters = 15;
        idx->config.use_cosine = 0;
        idx->config.per_dimension = 0;
        idx->config.default_rerank = 200;
    }

    if (idx->config.nprobe > idx->config.nlist) {
        idx->config.nprobe = idx->config.nlist;
    }

    idx->centroids = (float *)gv_alloc(idx->config.nlist * dimension * sizeof(float));
    idx->lists = (GV_IVFSQ8Entry **)gv_calloc(idx->config.nlist, sizeof(GV_IVFSQ8Entry *));
    idx->list_sizes = (size_t *)gv_calloc(idx->config.nlist, sizeof(size_t));

    if (!idx->centroids || !idx->lists || !idx->list_sizes) {
        gv_free(idx->centroids);
        gv_free(idx->lists);
        gv_free(idx->list_sizes);
        gv_free(idx);
        return NULL;
    }

    idx->scalar_quant_template = NULL;
    idx->trained = 0;
    idx->total_count = 0;
    idx->next_id = 0;

    return idx;
}

int ivfsq8_train(void *index, const float *data, size_t count) {
    if (!index || !data || count == 0) {
        return -1;
    }

    GV_IVFSQ8Index *idx = (GV_IVFSQ8Index *)index;
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

    if (ivfsq8_kmeans(train_buf, count, idx->dimension, idx->config.nlist,
                      idx->config.train_iters, idx->centroids) != 0) {
        gv_free(train_buf);
        return -1;
    }

    GV_ScalarQuantConfig sq_cfg = {
        .bits = 8,
        .per_dimension = idx->config.per_dimension ? 1 : 0
    };

    if (idx->scalar_quant_template != NULL) {
        scalar_quant_vector_destroy(idx->scalar_quant_template);
        idx->scalar_quant_template = NULL;
    }

    idx->scalar_quant_template = scalar_quantize_train(train_buf, count, idx->dimension, &sq_cfg);
    gv_free(train_buf);

    if (idx->scalar_quant_template == NULL) {
        return -1;
    }

    idx->trained = 1;
    return 0;
}

int ivfsq8_insert(void *index, GV_Vector *vector) {
    if (!index || !vector) {
        return -1;
    }

    GV_IVFSQ8Index *idx = (GV_IVFSQ8Index *)index;
    if (!idx->trained || vector->dimension != idx->dimension) {
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

    GV_IVFSQ8Entry *entry = (GV_IVFSQ8Entry *)gv_alloc(sizeof(GV_IVFSQ8Entry));
    if (!entry) {
        return -1;
    }

    entry->vector = vector;
    entry->scalar_quant = ivfsq8_quantize_vector(vector->data, idx);
    if (entry->scalar_quant == NULL) {
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

int ivfsq8_search(void *index, const GV_Vector *query, size_t k,
                  GV_SearchResult *results, GV_DistanceType distance_type,
                  const char *filter_key, const char *filter_value) {
    if (!index || !query || !results || k == 0) {
        return -1;
    }

    GV_IVFSQ8Index *idx = (GV_IVFSQ8Index *)index;
    if (!idx->trained || query->dimension != idx->dimension) {
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

    GV_IVFSQ8HeapItem *centroid_heap = (GV_IVFSQ8HeapItem *)gv_alloc(
        nprobe * sizeof(GV_IVFSQ8HeapItem));
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
        ivfsq8_heap_push(centroid_heap, &heap_size, nprobe, (GV_IVFSQ8HeapItem){dist, i, NULL});
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
            ivfsq8_heap_sift_down(centroid_heap, heap_size, 0);
        }
    }
    gv_free(centroid_heap);

    GV_IVFSQ8HeapItem *heap = (GV_IVFSQ8HeapItem *)gv_alloc(heap_cap * sizeof(GV_IVFSQ8HeapItem));
    if (!heap) {
        gv_free(probe_lists);
        return -1;
    }

    heap_size = 0;
    for (size_t i = 0; i < nprobe; i++) {
        GV_IVFSQ8Entry *entry = idx->lists[probe_lists[i]];
        while (entry) {
            if (!entry->deleted &&
                metadata_match(entry->vector->metadata, filter_key, filter_value)) {
                float dist = ivfsq8_entry_distance(idx, query, entry, distance_type, 0);
                if (dist >= 0.0f) {
                    ivfsq8_heap_push(heap, &heap_size, heap_cap,
                                     (GV_IVFSQ8HeapItem){dist, entry->id, entry});
                }
            }
            entry = entry->next;
        }
    }
    gv_free(probe_lists);

    size_t found = heap_size;
    GV_IVFSQ8HeapItem *candidates = (GV_IVFSQ8HeapItem *)gv_alloc(found * sizeof(GV_IVFSQ8HeapItem));
    if (!candidates) {
        gv_free(heap);
        return -1;
    }

    for (size_t i = found; i > 0; i--) {
        candidates[i - 1] = heap[0];
        heap[0] = heap[heap_size - 1];
        heap_size--;
        if (heap_size > 0) {
            ivfsq8_heap_sift_down(heap, heap_size, 0);
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
            float exact = ivfsq8_entry_distance(idx, query, candidates[i].entry, distance_type, 1);
            if (exact >= 0.0f) {
                candidates[i].dist = exact;
            }
        }
    }

    size_t result_count = (found < k) ? found : k;
    for (size_t i = 0; i < result_count; ++i) {
        size_t minj = i;
        for (size_t j = i + 1; j < found; ++j) {
            if (candidates[j].dist < candidates[minj].dist) {
                minj = j;
            }
        }
        if (minj != i) {
            GV_IVFSQ8HeapItem tmp = candidates[i];
            candidates[i] = candidates[minj];
            candidates[minj] = tmp;
        }
        ivfsq8_copy_result(&results[i], candidates[i].entry, candidates[i].dist);
    }

    gv_free(candidates);
    return (int)result_count;
}

int ivfsq8_range_search(void *index, const GV_Vector *query, float radius,
                        GV_SearchResult *results, size_t max_results,
                        GV_DistanceType distance_type,
                        const char *filter_key, const char *filter_value) {
    if (!index || !query || !results || max_results == 0 || radius < 0.0f) {
        return -1;
    }

    GV_IVFSQ8Index *idx = (GV_IVFSQ8Index *)index;
    if (!idx->trained || query->dimension != idx->dimension) {
        return -1;
    }

    size_t nprobe = idx->config.nprobe;
    if (nprobe > idx->config.nlist) {
        nprobe = idx->config.nlist;
    }

    GV_IVFSQ8HeapItem *centroid_heap = (GV_IVFSQ8HeapItem *)gv_alloc(
        nprobe * sizeof(GV_IVFSQ8HeapItem));
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
        ivfsq8_heap_push(centroid_heap, &heap_size, nprobe, (GV_IVFSQ8HeapItem){dist, i, NULL});
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
            ivfsq8_heap_sift_down(centroid_heap, heap_size, 0);
        }
    }
    gv_free(centroid_heap);

    size_t found = 0;
    for (size_t i = 0; i < nprobe && found < max_results; i++) {
        GV_IVFSQ8Entry *entry = idx->lists[probe_lists[i]];
        while (entry && found < max_results) {
            if (!entry->deleted &&
                metadata_match(entry->vector->metadata, filter_key, filter_value)) {
                float dist = ivfsq8_entry_distance(idx, query, entry, distance_type, 1);
                if (dist >= 0.0f && dist <= radius) {
                    ivfsq8_copy_result(&results[found], entry, dist);
                    found++;
                }
            }
            entry = entry->next;
        }
    }

    gv_free(probe_lists);
    return (int)found;
}

int ivfsq8_is_trained(const void *index) {
    if (!index) {
        return 0;
    }
    const GV_IVFSQ8Index *idx = (const GV_IVFSQ8Index *)index;
    return idx->trained;
}

void ivfsq8_destroy(void *index) {
    if (!index) {
        return;
    }

    GV_IVFSQ8Index *idx = (GV_IVFSQ8Index *)index;

    if (idx->lists) {
        for (size_t i = 0; i < idx->config.nlist; i++) {
            GV_IVFSQ8Entry *entry = idx->lists[i];
            while (entry) {
                GV_IVFSQ8Entry *next = entry->next;
                if (entry->vector) {
                    vector_destroy(entry->vector);
                }
                if (entry->scalar_quant) {
                    scalar_quant_vector_destroy(entry->scalar_quant);
                }
                gv_free(entry);
                entry = next;
            }
        }
        gv_free(idx->lists);
    }

    if (idx->scalar_quant_template) {
        scalar_quant_vector_destroy(idx->scalar_quant_template);
    }

    gv_free(idx->centroids);
    gv_free(idx->list_sizes);
    gv_free(idx);
}

size_t ivfsq8_count(const void *index) {
    if (!index) {
        return 0;
    }
    const GV_IVFSQ8Index *idx = (const GV_IVFSQ8Index *)index;

    size_t count = 0;
    for (size_t i = 0; i < idx->config.nlist; i++) {
        GV_IVFSQ8Entry *entry = idx->lists[i];
        while (entry) {
            if (!entry->deleted) {
                count++;
            }
            entry = entry->next;
        }
    }
    return count;
}

int ivfsq8_delete(void *index, size_t entry_index) {
    if (!index) {
        return -1;
    }

    GV_IVFSQ8Index *idx = (GV_IVFSQ8Index *)index;
    for (size_t i = 0; i < idx->config.nlist; i++) {
        GV_IVFSQ8Entry *entry = idx->lists[i];
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

int ivfsq8_update(void *index, size_t entry_index, const float *new_data, size_t dimension) {
    if (!index || !new_data) {
        return -1;
    }

    GV_IVFSQ8Index *idx = (GV_IVFSQ8Index *)index;
    if (dimension != idx->dimension) {
        return -1;
    }

    for (size_t i = 0; i < idx->config.nlist; i++) {
        GV_IVFSQ8Entry *entry = idx->lists[i];
        while (entry) {
            if (entry->id == entry_index) {
                if (entry->deleted || entry->vector == NULL || entry->vector->data == NULL) {
                    return -1;
                }
                memcpy(entry->vector->data, new_data, dimension * sizeof(float));
                if (entry->scalar_quant) {
                    scalar_quant_vector_destroy(entry->scalar_quant);
                }
                entry->scalar_quant = ivfsq8_quantize_vector(new_data, idx);
                return entry->scalar_quant != NULL ? 0 : -1;
            }
            entry = entry->next;
        }
    }
    return -1;
}

static int ivfsq8_write_scalar_template(FILE *out, const GV_ScalarQuantVector *tmpl) {
    if (write_u32(out, (uint32_t)tmpl->dimension) != 0) return -1;
    if (write_u32(out, (uint32_t)tmpl->bits) != 0) return -1;
    if (write_u32(out, (uint32_t)tmpl->per_dimension) != 0) return -1;
    if (write_u32(out, (uint32_t)tmpl->bytes_per_vector) != 0) return -1;

    size_t nvals = tmpl->per_dimension ? tmpl->dimension : 1;
    if (fwrite(tmpl->min_vals, sizeof(float), nvals, out) != nvals) return -1;
    if (fwrite(tmpl->max_vals, sizeof(float), nvals, out) != nvals) return -1;
    return 0;
}

static int ivfsq8_read_scalar_template(FILE *in, GV_ScalarQuantVector **out) {
    uint32_t dim = 0, bits = 0, per_dim = 0, bytes = 0;
    if (read_u32(in, &dim) != 0) return -1;
    if (read_u32(in, &bits) != 0) return -1;
    if (read_u32(in, &per_dim) != 0) return -1;
    if (read_u32(in, &bytes) != 0) return -1;

    GV_ScalarQuantVector *tmpl = (GV_ScalarQuantVector *)gv_calloc(1, sizeof(GV_ScalarQuantVector));
    if (!tmpl) return -1;

    tmpl->dimension = dim;
    tmpl->bits = (uint8_t)bits;
    tmpl->per_dimension = (int)per_dim;
    tmpl->bytes_per_vector = bytes;
    tmpl->quantized = NULL;

    size_t nvals = tmpl->per_dimension ? tmpl->dimension : 1;
    tmpl->min_vals = (float *)gv_alloc(nvals * sizeof(float));
    tmpl->max_vals = (float *)gv_alloc(nvals * sizeof(float));
    if (!tmpl->min_vals || !tmpl->max_vals) {
        scalar_quant_vector_destroy(tmpl);
        return -1;
    }
    if (fread(tmpl->min_vals, sizeof(float), nvals, in) != nvals) {
        scalar_quant_vector_destroy(tmpl);
        return -1;
    }
    if (fread(tmpl->max_vals, sizeof(float), nvals, in) != nvals) {
        scalar_quant_vector_destroy(tmpl);
        return -1;
    }

    *out = tmpl;
    return 0;
}

int ivfsq8_save(const void *index, FILE *out, uint32_t version) {
    if (!index || !out) {
        return -1;
    }
    (void)version;

    const GV_IVFSQ8Index *idx = (const GV_IVFSQ8Index *)index;

    if (write_u32(out, (uint32_t)idx->dimension) != 0) return -1;
    if (write_u32(out, (uint32_t)idx->config.nlist) != 0) return -1;
    if (write_u32(out, (uint32_t)idx->config.nprobe) != 0) return -1;
    if (write_u32(out, (uint32_t)idx->config.train_iters) != 0) return -1;
    if (write_u32(out, (uint32_t)idx->config.use_cosine) != 0) return -1;
    if (write_u32(out, (uint32_t)idx->config.per_dimension) != 0) return -1;
    if (write_u32(out, (uint32_t)idx->config.default_rerank) != 0) return -1;
    if (write_u32(out, (uint32_t)idx->trained) != 0) return -1;
    if (write_u32(out, (uint32_t)idx->next_id) != 0) return -1;

    if (idx->trained) {
        size_t centroid_floats = idx->config.nlist * idx->dimension;
        if (fwrite(idx->centroids, sizeof(float), centroid_floats, out) != centroid_floats) {
            return -1;
        }
        if (idx->scalar_quant_template == NULL ||
            ivfsq8_write_scalar_template(out, idx->scalar_quant_template) != 0) {
            return -1;
        }
    }

    if (write_u32(out, (uint32_t)idx->config.nlist) != 0) return -1;

    for (size_t i = 0; i < idx->config.nlist; i++) {
        uint32_t list_count = 0;
        GV_IVFSQ8Entry *entry = idx->lists[i];
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

            if (entry->scalar_quant == NULL ||
                fwrite(entry->scalar_quant->quantized, 1, entry->scalar_quant->bytes_per_vector, out) !=
                    entry->scalar_quant->bytes_per_vector) {
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

int ivfsq8_load(void **index_ptr, FILE *in, size_t dimension, uint32_t version) {
    if (!index_ptr || !in) {
        return -1;
    }
    (void)version;

    uint32_t file_dim = 0, nlist = 0, nprobe = 0, train_iters = 0;
    uint32_t use_cosine = 0, per_dimension = 0, default_rerank = 0;
    uint32_t trained = 0, next_id = 0;

    if (read_u32(in, &file_dim) != 0) return -1;
    if (read_u32(in, &nlist) != 0) return -1;
    if (read_u32(in, &nprobe) != 0) return -1;
    if (read_u32(in, &train_iters) != 0) return -1;
    if (read_u32(in, &use_cosine) != 0) return -1;
    if (read_u32(in, &per_dimension) != 0) return -1;
    if (read_u32(in, &default_rerank) != 0) return -1;
    if (read_u32(in, &trained) != 0) return -1;
    if (read_u32(in, &next_id) != 0) return -1;

    if (dimension != 0 && dimension != (size_t)file_dim) {
        return -1;
    }

    GV_IVFSQ8Config config = {
        .nlist = nlist,
        .nprobe = nprobe,
        .train_iters = train_iters,
        .use_cosine = (int)use_cosine,
        .per_dimension = (int)per_dimension,
        .default_rerank = default_rerank
    };

    void *index = ivfsq8_create((size_t)file_dim, &config);
    if (!index) {
        return -1;
    }

    GV_IVFSQ8Index *idx = (GV_IVFSQ8Index *)index;
    idx->trained = (int)trained;
    idx->next_id = (size_t)next_id;

    if (trained) {
        size_t centroid_floats = idx->config.nlist * idx->dimension;
        if (fread(idx->centroids, sizeof(float), centroid_floats, in) != centroid_floats) {
            ivfsq8_destroy(index);
            return -1;
        }
        if (ivfsq8_read_scalar_template(in, &idx->scalar_quant_template) != 0) {
            ivfsq8_destroy(index);
            return -1;
        }
    }

    uint32_t num_lists = 0;
    if (read_u32(in, &num_lists) != 0) {
        ivfsq8_destroy(index);
        return -1;
    }
    if (num_lists != nlist) {
        ivfsq8_destroy(index);
        return -1;
    }

    for (size_t i = 0; i < nlist; i++) {
        uint32_t list_count = 0;
        if (read_u32(in, &list_count) != 0) {
            ivfsq8_destroy(index);
            return -1;
        }

        GV_IVFSQ8Entry **tail = &idx->lists[i];
        for (uint32_t j = 0; j < list_count; j++) {
            uint32_t entry_id = 0, deleted = 0;
            if (read_u32(in, &entry_id) != 0) {
                ivfsq8_destroy(index);
                return -1;
            }
            if (read_u32(in, &deleted) != 0) {
                ivfsq8_destroy(index);
                return -1;
            }

            float *data = (float *)gv_alloc(idx->dimension * sizeof(float));
            if (!data) {
                ivfsq8_destroy(index);
                return -1;
            }
            if (fread(data, sizeof(float), idx->dimension, in) != idx->dimension) {
                gv_free(data);
                ivfsq8_destroy(index);
                return -1;
            }

            GV_Vector *vec = vector_create_from_data(idx->dimension, data);
            gv_free(data);
            if (!vec) {
                ivfsq8_destroy(index);
                return -1;
            }

            GV_IVFSQ8Entry *entry = (GV_IVFSQ8Entry *)gv_alloc(sizeof(GV_IVFSQ8Entry));
            if (!entry) {
                vector_destroy(vec);
                ivfsq8_destroy(index);
                return -1;
            }

            entry->vector = vec;
            entry->id = (size_t)entry_id;
            entry->deleted = (int)deleted;
            entry->next = NULL;

            if (idx->scalar_quant_template == NULL) {
                vector_destroy(vec);
                gv_free(entry);
                ivfsq8_destroy(index);
                return -1;
            }

            entry->scalar_quant = (GV_ScalarQuantVector *)gv_alloc(sizeof(GV_ScalarQuantVector));
            if (entry->scalar_quant == NULL) {
                vector_destroy(vec);
                gv_free(entry);
                ivfsq8_destroy(index);
                return -1;
            }
            entry->scalar_quant->dimension = idx->dimension;
            entry->scalar_quant->bits = 8;
            entry->scalar_quant->per_dimension = idx->config.per_dimension;
            entry->scalar_quant->bytes_per_vector = idx->scalar_quant_template->bytes_per_vector;
            size_t nvals = entry->scalar_quant->per_dimension ? idx->dimension : 1;
            entry->scalar_quant->min_vals = (float *)gv_alloc(nvals * sizeof(float));
            entry->scalar_quant->max_vals = (float *)gv_alloc(nvals * sizeof(float));
            entry->scalar_quant->quantized = (uint8_t *)gv_alloc(entry->scalar_quant->bytes_per_vector);
            if (!entry->scalar_quant->min_vals || !entry->scalar_quant->max_vals ||
                !entry->scalar_quant->quantized) {
                scalar_quant_vector_destroy(entry->scalar_quant);
                vector_destroy(vec);
                gv_free(entry);
                ivfsq8_destroy(index);
                return -1;
            }
            memcpy(entry->scalar_quant->min_vals, idx->scalar_quant_template->min_vals,
                   nvals * sizeof(float));
            memcpy(entry->scalar_quant->max_vals, idx->scalar_quant_template->max_vals,
                   nvals * sizeof(float));
            if (fread(entry->scalar_quant->quantized, 1, entry->scalar_quant->bytes_per_vector, in) !=
                entry->scalar_quant->bytes_per_vector) {
                scalar_quant_vector_destroy(entry->scalar_quant);
                vector_destroy(vec);
                gv_free(entry);
                ivfsq8_destroy(index);
                return -1;
            }

            uint32_t meta_count = 0;
            if (read_u32(in, &meta_count) != 0) {
                scalar_quant_vector_destroy(entry->scalar_quant);
                vector_destroy(vec);
                gv_free(entry);
                ivfsq8_destroy(index);
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
                scalar_quant_vector_destroy(entry->scalar_quant);
                vector_destroy(vec);
                gv_free(entry);
                ivfsq8_destroy(index);
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
