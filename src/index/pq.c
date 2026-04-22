#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <float.h>

#include "index/pq.h"
#include "search/distance.h"
#include "schema/vector.h"
#include "schema/metadata.h"
#include "core/utils.h"
#include "core/heap.h"

typedef struct {
    uint8_t *codes;        /* m bytes for nbits=8 */
    float *raw_data;       /* dimension floats for exact reranking */
    GV_Metadata *metadata;
    int deleted;
    size_t id;
} GV_PQEntry;

typedef struct {
    size_t dimension;
    size_t m;              /* Number of sub-quantizers */
    uint8_t nbits;         /* Bits per sub-quantizer code */
    size_t ksub;           /* 2^nbits = codes per sub-quantizer */
    size_t dsub;           /* dimension / m */
    float *codebooks;      /* m * ksub * dsub floats */
    GV_PQEntry *entries;   /* Dynamic array of entries */
    size_t entry_count;
    size_t entry_capacity;
    int trained;
    size_t train_iters;
} GV_PQIndex;

typedef struct { float dist; size_t idx; } GV_PQHeapItem;
GV_HEAP_DEFINE(pq_heap, GV_PQHeapItem)

/* Squared Euclidean distance between sub-vectors */
static float pq_subvec_distance_sq(const float *a, const float *b, size_t dsub) {
    float sum = 0.0f;
    for (size_t i = 0; i < dsub; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

#define GV_PQ_RERANK_FACTOR 20

static void pq_train_subquantizer(float *codebook, const float *subvecs,
                                     size_t count, size_t dsub, size_t ksub,
                                     size_t train_iters) {
    if (count == 0 || ksub == 0) return;

    for (size_t k = 0; k < ksub && k < count; k++) {
        size_t idx = (k * count) / ksub; /* Spread evenly */
        memcpy(&codebook[k * dsub], &subvecs[idx * dsub], dsub * sizeof(float));
    }

    for (size_t k = count; k < ksub; k++) {
        memset(&codebook[k * dsub], 0, dsub * sizeof(float));
    }

    uint32_t *assignments = (uint32_t *)malloc(count * sizeof(uint32_t));
    if (!assignments) return;

    for (size_t iter = 0; iter < train_iters; iter++) {
        for (size_t i = 0; i < count; i++) {
            float min_dist = FLT_MAX;
            uint32_t best_k = 0;
            for (size_t k = 0; k < ksub; k++) {
                float dist = pq_subvec_distance_sq(&subvecs[i * dsub], &codebook[k * dsub], dsub);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_k = k;
                }
            }
            assignments[i] = best_k;
        }

        float *new_centroids = (float *)calloc(ksub * dsub, sizeof(float));
        uint32_t *counts = (uint32_t *)calloc(ksub, sizeof(uint32_t));
        if (!new_centroids || !counts) {
            free(new_centroids);
            free(counts);
            break;
        }

        for (size_t i = 0; i < count; i++) {
            uint32_t k = assignments[i];
            counts[k]++;
            for (size_t d = 0; d < dsub; d++) {
                new_centroids[k * dsub + d] += subvecs[i * dsub + d];
            }
        }

        for (size_t k = 0; k < ksub; k++) {
            if (counts[k] > 0) {
                for (size_t d = 0; d < dsub; d++) {
                    codebook[k * dsub + d] = new_centroids[k * dsub + d] / counts[k];
                }
            }
        }

        free(new_centroids);
        free(counts);
    }

    free(assignments);
}

static void pq_encode(const GV_PQIndex *idx, const float *data, uint8_t *codes) {
    for (size_t m_i = 0; m_i < idx->m; m_i++) {
        const float *subvec = &data[m_i * idx->dsub];
        const float *subcodebook = &idx->codebooks[m_i * idx->ksub * idx->dsub];

        float min_dist = FLT_MAX;
        uint8_t best_code = 0;

        for (size_t k = 0; k < idx->ksub; k++) {
            float dist = pq_subvec_distance_sq(subvec, &subcodebook[k * idx->dsub], idx->dsub);
            if (dist < min_dist) {
                min_dist = dist;
                best_code = (uint8_t)k;
            }
        }

        codes[m_i] = best_code;
    }
}

void *pq_create(size_t dimension, const GV_PQConfig *config) {
    if (dimension == 0) return NULL;

    GV_PQIndex *idx = (GV_PQIndex *)calloc(1, sizeof(GV_PQIndex));
    if (!idx) return NULL;

    idx->dimension = dimension;

    if (config) {
        idx->m = config->m;
        idx->nbits = config->nbits;
        idx->train_iters = config->train_iters;
    } else {
        idx->m = 8;
        idx->nbits = 8;
        idx->train_iters = 15;
    }

    if (idx->m == 0 || idx->nbits == 0 || idx->nbits > 8) {
        free(idx);
        return NULL;
    }

    if (dimension % idx->m != 0) {
        free(idx);
        return NULL;
    }

    idx->ksub = 1 << idx->nbits; /* 2^nbits */
    idx->dsub = dimension / idx->m;
    idx->trained = 0;

    idx->codebooks = (float *)calloc(idx->m * idx->ksub * idx->dsub, sizeof(float));
    if (!idx->codebooks) {
        free(idx);
        return NULL;
    }

    idx->entry_capacity = 128;
    idx->entries = (GV_PQEntry *)calloc(idx->entry_capacity, sizeof(GV_PQEntry));
    if (!idx->entries) {
        free(idx->codebooks);
        free(idx);
        return NULL;
    }
    idx->entry_count = 0;

    return idx;
}

int pq_train(void *index, const float *data, size_t count) {
    if (!index || !data || count == 0) return -1;
    GV_PQIndex *idx = (GV_PQIndex *)index;

    float *subvecs = (float *)malloc(count * idx->dsub * sizeof(float));
    if (!subvecs) return -1;

    for (size_t m_i = 0; m_i < idx->m; m_i++) {
        for (size_t i = 0; i < count; i++) {
            memcpy(&subvecs[i * idx->dsub],
                   &data[i * idx->dimension + m_i * idx->dsub],
                   idx->dsub * sizeof(float));
        }

        float *subcodebook = &idx->codebooks[m_i * idx->ksub * idx->dsub];
        pq_train_subquantizer(subcodebook, subvecs, count, idx->dsub, idx->ksub, idx->train_iters);
    }

    free(subvecs);
    idx->trained = 1;
    return 0;
}

int pq_insert(void *index, GV_Vector *vector) {
    if (!index || !vector) return -1;
    GV_PQIndex *idx = (GV_PQIndex *)index;

    if (!idx->trained) return -1;
    if (vector->dimension != idx->dimension) return -1;

    if (idx->entry_count >= idx->entry_capacity) {
        size_t new_capacity = idx->entry_capacity * 2;
        GV_PQEntry *new_entries = (GV_PQEntry *)realloc(idx->entries, new_capacity * sizeof(GV_PQEntry));
        if (!new_entries) return -1;
        memset(&new_entries[idx->entry_capacity], 0, (new_capacity - idx->entry_capacity) * sizeof(GV_PQEntry));
        idx->entries = new_entries;
        idx->entry_capacity = new_capacity;
    }

    GV_PQEntry *entry = &idx->entries[idx->entry_count];

    entry->codes = (uint8_t *)malloc(idx->m * sizeof(uint8_t));
    if (!entry->codes) return -1;

    pq_encode(idx, vector->data, entry->codes);

    entry->raw_data = (float *)malloc(idx->dimension * sizeof(float));
    if (!entry->raw_data) {
        free(entry->codes);
        return -1;
    }
    memcpy(entry->raw_data, vector->data, idx->dimension * sizeof(float));

    entry->metadata = vector->metadata;
    vector->metadata = NULL;

    entry->deleted = 0;
    entry->id = idx->entry_count;

    idx->entry_count++;

    vector_destroy(vector);
    return 0;
}

int pq_search(void *index, const GV_Vector *query, size_t k,
                 GV_SearchResult *results, GV_DistanceType distance_type,
                 const char *filter_key, const char *filter_value) {
    if (!index || !query || !results || k == 0) return -1;
    GV_PQIndex *idx = (GV_PQIndex *)index;

    if (!idx->trained) return -1;
    if (query->dimension != idx->dimension) return -1;

    float *distance_table = (float *)malloc(idx->m * idx->ksub * sizeof(float));
    if (!distance_table) return -1;

    for (size_t m_i = 0; m_i < idx->m; m_i++) {
        const float *query_subvec = &query->data[m_i * idx->dsub];
        const float *subcodebook = &idx->codebooks[m_i * idx->ksub * idx->dsub];

        for (size_t k_i = 0; k_i < idx->ksub; k_i++) {
            float dist_sq = pq_subvec_distance_sq(query_subvec, &subcodebook[k_i * idx->dsub], idx->dsub);
            distance_table[m_i * idx->ksub + k_i] = dist_sq;
        }
    }

    size_t oversample_k = k * GV_PQ_RERANK_FACTOR;
    if (oversample_k > idx->entry_count) oversample_k = idx->entry_count;
    if (oversample_k < k) oversample_k = k;

    GV_PQHeapItem *heap = (GV_PQHeapItem *)malloc(oversample_k * sizeof(GV_PQHeapItem));
    if (!heap) {
        free(distance_table);
        return -1;
    }
    size_t heap_size = 0;

    for (size_t i = 0; i < idx->entry_count; i++) {
        if (idx->entries[i].deleted) continue;

        if (filter_key && filter_value) {
            if (!metadata_match(idx->entries[i].metadata, filter_key, filter_value)) {
                continue;
            }
        }

        float dist_squared = 0.0f;
        for (size_t m_i = 0; m_i < idx->m; m_i++) {
            uint8_t code = idx->entries[i].codes[m_i];
            dist_squared += distance_table[m_i * idx->ksub + code];
        }
        float dist = sqrtf(dist_squared);

        pq_heap_push(heap, &heap_size, oversample_k, (GV_PQHeapItem){dist, i});
    }

    free(distance_table);

    size_t cand_count = heap_size;
    GV_PQHeapItem *candidates = (GV_PQHeapItem *)malloc(cand_count * sizeof(GV_PQHeapItem));
    if (!candidates) {
        free(heap);
        return -1;
    }

    for (size_t i = cand_count; i > 0; --i) {
        candidates[i - 1] = heap[0];
        heap[0] = heap[heap_size - 1];
        heap_size--;
        if (heap_size > 0) {
            pq_heap_sift_down(heap, heap_size, 0);
        }
    }
    free(heap);

    for (size_t i = 0; i < cand_count; i++) {
        size_t entry_idx = candidates[i].idx;
        GV_PQEntry *entry = &idx->entries[entry_idx];

        GV_Vector temp_vec = {
            .data = entry->raw_data,
            .dimension = idx->dimension,
            .metadata = NULL
        };
        float exact_dist = distance(&temp_vec, query, distance_type);
        if (exact_dist >= 0.0f) {
            candidates[i].dist = exact_dist;
        }
    }

    for (size_t i = 0; i + 1 < cand_count; i++) {
        size_t minj = i;
        for (size_t j = i + 1; j < cand_count; j++) {
            if (candidates[j].dist < candidates[minj].dist) minj = j;
        }
        if (minj != i) {
            GV_PQHeapItem tmp = candidates[i];
            candidates[i] = candidates[minj];
            candidates[minj] = tmp;
        }
    }

    int n = (int)(cand_count < k ? cand_count : k);
    for (int i = 0; i < n; i++) {
        size_t entry_idx = candidates[i].idx;
        GV_PQEntry *entry = &idx->entries[entry_idx];

        GV_Vector *result_vec = vector_create_from_data(idx->dimension, entry->raw_data);
        if (result_vec) {
            GV_Metadata *cur = entry->metadata;
            while (cur) {
                if (cur->key && cur->value) {
                    vector_set_metadata(result_vec, cur->key, cur->value);
                }
                cur = cur->next;
            }

            results[i].vector = result_vec;
            results[i].distance = candidates[i].dist;
            results[i].is_sparse = 0;
            results[i].sparse_vector = NULL;
            results[i].id = entry_idx;
        } else {
            results[i].vector = NULL;
            results[i].distance = candidates[i].dist;
            results[i].is_sparse = 0;
            results[i].sparse_vector = NULL;
            results[i].id = entry_idx;
        }
    }

    free(candidates);
    return n;
}

int pq_range_search(void *index, const GV_Vector *query, float radius,
                       GV_SearchResult *results, size_t max_results,
                       GV_DistanceType distance_type,
                       const char *filter_key, const char *filter_value) {
    if (!index || !query || !results || max_results == 0 || radius < 0.0f) return -1;
    GV_PQIndex *idx = (GV_PQIndex *)index;

    if (!idx->trained) return -1;
    if (query->dimension != idx->dimension) return -1;

    float *distance_table = (float *)malloc(idx->m * idx->ksub * sizeof(float));
    if (!distance_table) return -1;

    for (size_t m_i = 0; m_i < idx->m; m_i++) {
        const float *query_subvec = &query->data[m_i * idx->dsub];
        const float *subcodebook = &idx->codebooks[m_i * idx->ksub * idx->dsub];

        for (size_t k_i = 0; k_i < idx->ksub; k_i++) {
            float dist = pq_subvec_distance_sq(query_subvec, &subcodebook[k_i * idx->dsub], idx->dsub);
            distance_table[m_i * idx->ksub + k_i] = dist * dist;
        }
    }

    size_t found = 0;

    for (size_t i = 0; i < idx->entry_count && found < max_results; i++) {
        if (idx->entries[i].deleted) continue;

        if (filter_key && filter_value) {
            if (!metadata_match(idx->entries[i].metadata, filter_key, filter_value)) {
                continue;
            }
        }

        float dist_squared = 0.0f;
        for (size_t m_i = 0; m_i < idx->m; m_i++) {
            uint8_t code = idx->entries[i].codes[m_i];
            dist_squared += distance_table[m_i * idx->ksub + code];
        }
        float approx_dist = sqrtf(dist_squared);

        /* Early rejection if approximate distance exceeds radius */
        if (approx_dist > radius * 1.5f) continue; /* Allow some slack for approximation error */

        GV_PQEntry *entry = &idx->entries[i];
        GV_Vector *result_vec = vector_create_from_data(idx->dimension, entry->raw_data);
        if (result_vec) {
            GV_Metadata *cur = entry->metadata;
            while (cur) {
                if (cur->key && cur->value) {
                    vector_set_metadata(result_vec, cur->key, cur->value);
                }
                cur = cur->next;
            }

            float exact_dist = distance(query, result_vec, distance_type);

            if (exact_dist <= radius) {
                results[found].vector = result_vec;
                results[found].distance = exact_dist;
                results[found].is_sparse = 0;
                results[found].sparse_vector = NULL;
                results[found].id = i;
                found++;
            } else {
                vector_destroy(result_vec);
            }
        }
    }

    free(distance_table);
    return (int)found;
}

int pq_is_trained(const void *index) {
    if (!index) return 0;
    const GV_PQIndex *idx = (const GV_PQIndex *)index;
    return idx->trained;
}

size_t pq_count(const void *index) {
    if (!index) return 0;
    const GV_PQIndex *idx = (const GV_PQIndex *)index;
    size_t count = 0;
    for (size_t i = 0; i < idx->entry_count; i++) {
        if (!idx->entries[i].deleted) count++;
    }
    return count;
}

int pq_delete(void *index, size_t entry_index) {
    if (!index) return -1;
    GV_PQIndex *idx = (GV_PQIndex *)index;
    if (entry_index >= idx->entry_count) return -1;
    idx->entries[entry_index].deleted = 1;
    return 0;
}

int pq_update(void *index, size_t entry_index, const float *new_data, size_t dimension) {
    if (!index || !new_data) return -1;
    GV_PQIndex *idx = (GV_PQIndex *)index;

    if (!idx->trained) return -1;
    if (dimension != idx->dimension) return -1;
    if (entry_index >= idx->entry_count) return -1;
    if (idx->entries[entry_index].deleted) return -1;

    GV_PQEntry *entry = &idx->entries[entry_index];

    pq_encode(idx, new_data, entry->codes);
    memcpy(entry->raw_data, new_data, idx->dimension * sizeof(float));

    return 0;
}

int pq_save(const void *index, FILE *out, uint32_t version) {
    if (!index || !out) return -1;
    const GV_PQIndex *idx = (const GV_PQIndex *)index;
    (void)version;

    if (write_u32(out, (uint32_t)idx->dimension) != 0) return -1;
    if (write_u32(out, (uint32_t)idx->m) != 0) return -1;
    if (write_u8(out, idx->nbits) != 0) return -1;
    if (write_u32(out, (uint32_t)idx->train_iters) != 0) return -1;
    if (write_u32(out, (uint32_t)idx->trained) != 0) return -1;

    size_t codebook_size = idx->m * idx->ksub * idx->dsub;
    if (fwrite(idx->codebooks, sizeof(float), codebook_size, out) != codebook_size) return -1;

    if (write_u32(out, (uint32_t)idx->entry_count) != 0) return -1;

    for (size_t i = 0; i < idx->entry_count; i++) {
        const GV_PQEntry *entry = &idx->entries[i];

        if (write_u32(out, (uint32_t)entry->deleted) != 0) return -1;
        if (write_u32(out, (uint32_t)entry->id) != 0) return -1;

        if (fwrite(entry->codes, sizeof(uint8_t), idx->m, out) != idx->m) return -1;

        if (fwrite(entry->raw_data, sizeof(float), idx->dimension, out) != idx->dimension) return -1;

        uint32_t meta_count = 0;
        GV_Metadata *cur = entry->metadata;
        while (cur) { meta_count++; cur = cur->next; }
        if (write_u32(out, meta_count) != 0) return -1;

        cur = entry->metadata;
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

int pq_load(void **index_ptr, FILE *in, size_t dimension, uint32_t version) {
    if (!index_ptr || !in) return -1;
    (void)version;

    uint32_t file_dim = 0, m = 0, train_iters = 0, trained = 0, entry_count = 0;
    uint8_t nbits = 0;

    if (read_u32(in, &file_dim) != 0) return -1;
    if (read_u32(in, &m) != 0) return -1;
    if (read_u8(in, &nbits) != 0) return -1;
    if (read_u32(in, &train_iters) != 0) return -1;
    if (read_u32(in, &trained) != 0) return -1;

    if (dimension != 0 && dimension != (size_t)file_dim) return -1;

    GV_PQConfig config = {
        .m = (size_t)m,
        .nbits = nbits,
        .train_iters = (size_t)train_iters
    };

    void *index = pq_create((size_t)file_dim, &config);
    if (!index) return -1;

    GV_PQIndex *idx = (GV_PQIndex *)index;

    size_t codebook_size = idx->m * idx->ksub * idx->dsub;
    if (fread(idx->codebooks, sizeof(float), codebook_size, in) != codebook_size) {
        pq_destroy(index);
        return -1;
    }

    idx->trained = (int)trained;

    if (read_u32(in, &entry_count) != 0) {
        pq_destroy(index);
        return -1;
    }

    if (entry_count > idx->entry_capacity) {
        GV_PQEntry *new_entries = (GV_PQEntry *)realloc(idx->entries, entry_count * sizeof(GV_PQEntry));
        if (!new_entries) {
            pq_destroy(index);
            return -1;
        }
        memset(&new_entries[idx->entry_capacity], 0, (entry_count - idx->entry_capacity) * sizeof(GV_PQEntry));
        idx->entries = new_entries;
        idx->entry_capacity = entry_count;
    }

    for (uint32_t i = 0; i < entry_count; i++) {
        GV_PQEntry *entry = &idx->entries[i];

        uint32_t deleted = 0, id = 0;
        if (read_u32(in, &deleted) != 0) { pq_destroy(index); return -1; }
        if (read_u32(in, &id) != 0) { pq_destroy(index); return -1; }

        entry->deleted = (int)deleted;
        entry->id = (size_t)id;

        entry->codes = (uint8_t *)malloc(idx->m * sizeof(uint8_t));
        if (!entry->codes) { pq_destroy(index); return -1; }
        if (fread(entry->codes, sizeof(uint8_t), idx->m, in) != idx->m) {
            pq_destroy(index);
            return -1;
        }

        entry->raw_data = (float *)malloc(idx->dimension * sizeof(float));
        if (!entry->raw_data) { pq_destroy(index); return -1; }
        if (fread(entry->raw_data, sizeof(float), idx->dimension, in) != idx->dimension) {
            pq_destroy(index);
            return -1;
        }

        uint32_t meta_count = 0;
        if (read_u32(in, &meta_count) != 0) { pq_destroy(index); return -1; }

        entry->metadata = NULL;
        for (uint32_t m_idx = 0; m_idx < meta_count; m_idx++) {
            uint32_t klen = 0, vlen = 0;
            char *key = NULL, *value = NULL;
            if (read_u32(in, &klen) != 0) { pq_destroy(index); return -1; }
            if (read_str(in, &key, klen) != 0) { pq_destroy(index); return -1; }
            if (read_u32(in, &vlen) != 0) { free(key); pq_destroy(index); return -1; }
            if (read_str(in, &value, vlen) != 0) { free(key); pq_destroy(index); return -1; }

            GV_Metadata *node = (GV_Metadata *)malloc(sizeof(GV_Metadata));
            if (!node) { free(key); free(value); pq_destroy(index); return -1; }
            node->key = key;
            node->value = value;
            node->next = entry->metadata;
            entry->metadata = node;
        }

        idx->entry_count++;
    }

    *index_ptr = index;
    return 0;
}

void pq_destroy(void *index) {
    if (!index) return;
    GV_PQIndex *idx = (GV_PQIndex *)index;

    free(idx->codebooks);

    for (size_t i = 0; i < idx->entry_count; i++) {
        free(idx->entries[i].codes);
        free(idx->entries[i].raw_data);
        metadata_free(idx->entries[i].metadata);
    }
    free(idx->entries);

    free(idx);
}
