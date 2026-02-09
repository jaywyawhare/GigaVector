#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <float.h>

#include "gigavector/gv_pq.h"
#include "gigavector/gv_distance.h"
#include "gigavector/gv_vector.h"
#include "gigavector/gv_metadata.h"

/* Internal PQ entry structure */
typedef struct {
    uint8_t *codes;        /* m bytes for nbits=8 */
    float *raw_data;       /* dimension floats for exact reranking */
    GV_Metadata *metadata;
    int deleted;
    size_t id;
} GV_PQEntry;

/* Internal PQ index structure */
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

/* Max-heap for top-k selection */
typedef struct {
    float dist;
    size_t idx;
} GV_PQHeapItem;

static void gv_pq_heap_sift_down(GV_PQHeapItem *heap, size_t size, size_t i) {
    while (1) {
        size_t l = 2 * i + 1;
        size_t r = l + 1;
        size_t largest = i;
        if (l < size && heap[l].dist > heap[largest].dist) largest = l;
        if (r < size && heap[r].dist > heap[largest].dist) largest = r;
        if (largest == i) break;
        GV_PQHeapItem tmp = heap[i];
        heap[i] = heap[largest];
        heap[largest] = tmp;
        i = largest;
    }
}

static void gv_pq_heap_push(GV_PQHeapItem *heap, size_t *size, size_t capacity, float dist, size_t idx) {
    if (*size < capacity) {
        heap[*size].dist = dist;
        heap[*size].idx = idx;
        (*size)++;
        /* Sift up */
        size_t i = *size - 1;
        while (i > 0) {
            size_t parent = (i - 1) / 2;
            if (heap[i].dist > heap[parent].dist) {
                GV_PQHeapItem tmp = heap[i];
                heap[i] = heap[parent];
                heap[parent] = tmp;
                i = parent;
            } else {
                break;
            }
        }
    } else if (dist < heap[0].dist) {
        heap[0].dist = dist;
        heap[0].idx = idx;
        gv_pq_heap_sift_down(heap, *size, 0);
    }
}

/* I/O helpers */
static int gv_pq_write_u32(FILE *f, uint32_t v) {
    return fwrite(&v, sizeof(uint32_t), 1, f) == 1 ? 0 : -1;
}

static int gv_pq_read_u32(FILE *f, uint32_t *v) {
    return (v && fread(v, sizeof(uint32_t), 1, f) == 1) ? 0 : -1;
}

static int gv_pq_write_u8(FILE *f, uint8_t v) {
    return fwrite(&v, sizeof(uint8_t), 1, f) == 1 ? 0 : -1;
}

static int gv_pq_read_u8(FILE *f, uint8_t *v) {
    return (v && fread(v, sizeof(uint8_t), 1, f) == 1) ? 0 : -1;
}

static int gv_pq_write_str(FILE *f, const char *s, uint32_t len) {
    if (gv_pq_write_u32(f, len) != 0) return -1;
    if (len == 0) return 0;
    return fwrite(s, 1, len, f) == len ? 0 : -1;
}

static int gv_pq_read_str(FILE *f, char **s, uint32_t len) {
    *s = NULL;
    if (len == 0) {
        *s = (char *)malloc(1);
        if (!*s) return -1;
        (*s)[0] = '\0';
        return 0;
    }
    char *buf = (char *)malloc(len + 1);
    if (!buf) return -1;
    if (fread(buf, 1, len, f) != len) { free(buf); return -1; }
    buf[len] = '\0';
    *s = buf;
    return 0;
}

/* Metadata matching helper */
static int gv_pq_metadata_match(const GV_Metadata *meta, const char *key, const char *value) {
    if (!key || !value) return 1; /* No filter = match */
    const GV_Metadata *cur = meta;
    while (cur) {
        if (cur->key && cur->value && strcmp(cur->key, key) == 0 && strcmp(cur->value, value) == 0) {
            return 1;
        }
        cur = cur->next;
    }
    return 0;
}

/* Euclidean distance between sub-vectors */
static float gv_pq_subvec_distance(const float *a, const float *b, size_t dsub) {
    float sum = 0.0f;
    for (size_t i = 0; i < dsub; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

/* K-means training for a single sub-quantizer */
static void gv_pq_train_subquantizer(float *codebook, const float *subvecs,
                                     size_t count, size_t dsub, size_t ksub,
                                     size_t train_iters) {
    if (count == 0 || ksub == 0) return;

    /* Initialize centroids from random sub-vectors */
    for (size_t k = 0; k < ksub && k < count; k++) {
        size_t idx = (k * count) / ksub; /* Spread evenly */
        memcpy(&codebook[k * dsub], &subvecs[idx * dsub], dsub * sizeof(float));
    }

    /* If we have fewer training vectors than centroids, pad with zeros */
    for (size_t k = count; k < ksub; k++) {
        memset(&codebook[k * dsub], 0, dsub * sizeof(float));
    }

    /* Lloyd's algorithm iterations */
    uint32_t *assignments = (uint32_t *)malloc(count * sizeof(uint32_t));
    if (!assignments) return;

    for (size_t iter = 0; iter < train_iters; iter++) {
        /* Assignment step */
        for (size_t i = 0; i < count; i++) {
            float min_dist = FLT_MAX;
            uint32_t best_k = 0;
            for (size_t k = 0; k < ksub; k++) {
                float dist = gv_pq_subvec_distance(&subvecs[i * dsub], &codebook[k * dsub], dsub);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_k = k;
                }
            }
            assignments[i] = best_k;
        }

        /* Update step */
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

/* Encode a vector using the trained codebook */
static void gv_pq_encode(const GV_PQIndex *idx, const float *data, uint8_t *codes) {
    for (size_t m_i = 0; m_i < idx->m; m_i++) {
        const float *subvec = &data[m_i * idx->dsub];
        const float *subcodebook = &idx->codebooks[m_i * idx->ksub * idx->dsub];

        float min_dist = FLT_MAX;
        uint8_t best_code = 0;

        for (size_t k = 0; k < idx->ksub; k++) {
            float dist = gv_pq_subvec_distance(subvec, &subcodebook[k * idx->dsub], idx->dsub);
            if (dist < min_dist) {
                min_dist = dist;
                best_code = (uint8_t)k;
            }
        }

        codes[m_i] = best_code;
    }
}

/* Create PQ index */
void *gv_pq_create(size_t dimension, const GV_PQConfig *config) {
    if (dimension == 0) return NULL;

    GV_PQIndex *idx = (GV_PQIndex *)calloc(1, sizeof(GV_PQIndex));
    if (!idx) return NULL;

    idx->dimension = dimension;

    /* Set config or use defaults */
    if (config) {
        idx->m = config->m;
        idx->nbits = config->nbits;
        idx->train_iters = config->train_iters;
    } else {
        idx->m = 8;
        idx->nbits = 8;
        idx->train_iters = 15;
    }

    /* Validate configuration */
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

    /* Allocate codebook */
    idx->codebooks = (float *)calloc(idx->m * idx->ksub * idx->dsub, sizeof(float));
    if (!idx->codebooks) {
        free(idx);
        return NULL;
    }

    /* Initialize entry storage */
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

/* Train PQ codebooks */
int gv_pq_train(void *index, const float *data, size_t count) {
    if (!index || !data || count == 0) return -1;
    GV_PQIndex *idx = (GV_PQIndex *)index;

    /* Allocate sub-vector storage */
    float *subvecs = (float *)malloc(count * idx->dsub * sizeof(float));
    if (!subvecs) return -1;

    /* Train each sub-quantizer */
    for (size_t m_i = 0; m_i < idx->m; m_i++) {
        /* Extract sub-vectors for this sub-quantizer */
        for (size_t i = 0; i < count; i++) {
            memcpy(&subvecs[i * idx->dsub],
                   &data[i * idx->dimension + m_i * idx->dsub],
                   idx->dsub * sizeof(float));
        }

        /* Train codebook for this sub-quantizer */
        float *subcodebook = &idx->codebooks[m_i * idx->ksub * idx->dsub];
        gv_pq_train_subquantizer(subcodebook, subvecs, count, idx->dsub, idx->ksub, idx->train_iters);
    }

    free(subvecs);
    idx->trained = 1;
    return 0;
}

/* Insert vector into PQ index */
int gv_pq_insert(void *index, GV_Vector *vector) {
    if (!index || !vector) return -1;
    GV_PQIndex *idx = (GV_PQIndex *)index;

    if (!idx->trained) return -1;
    if (vector->dimension != idx->dimension) return -1;

    /* Expand capacity if needed */
    if (idx->entry_count >= idx->entry_capacity) {
        size_t new_capacity = idx->entry_capacity * 2;
        GV_PQEntry *new_entries = (GV_PQEntry *)realloc(idx->entries, new_capacity * sizeof(GV_PQEntry));
        if (!new_entries) return -1;
        memset(&new_entries[idx->entry_capacity], 0, (new_capacity - idx->entry_capacity) * sizeof(GV_PQEntry));
        idx->entries = new_entries;
        idx->entry_capacity = new_capacity;
    }

    GV_PQEntry *entry = &idx->entries[idx->entry_count];

    /* Allocate and encode */
    entry->codes = (uint8_t *)malloc(idx->m * sizeof(uint8_t));
    if (!entry->codes) return -1;

    gv_pq_encode(idx, vector->data, entry->codes);

    /* Store raw data for exact reranking */
    entry->raw_data = (float *)malloc(idx->dimension * sizeof(float));
    if (!entry->raw_data) {
        free(entry->codes);
        return -1;
    }
    memcpy(entry->raw_data, vector->data, idx->dimension * sizeof(float));

    /* Take ownership of metadata */
    entry->metadata = vector->metadata;
    vector->metadata = NULL;

    entry->deleted = 0;
    entry->id = idx->entry_count;

    idx->entry_count++;

    /* Clean up vector */
    gv_vector_destroy(vector);
    return 0;
}

/* Search using Asymmetric Distance Computation (ADC) */
int gv_pq_search(void *index, const GV_Vector *query, size_t k,
                 GV_SearchResult *results, GV_DistanceType distance_type,
                 const char *filter_key, const char *filter_value) {
    if (!index || !query || !results || k == 0) return -1;
    GV_PQIndex *idx = (GV_PQIndex *)index;

    if (!idx->trained) return -1;
    if (query->dimension != idx->dimension) return -1;

    /* Build distance lookup table for ADC */
    float *distance_table = (float *)malloc(idx->m * idx->ksub * sizeof(float));
    if (!distance_table) return -1;

    for (size_t m_i = 0; m_i < idx->m; m_i++) {
        const float *query_subvec = &query->data[m_i * idx->dsub];
        const float *subcodebook = &idx->codebooks[m_i * idx->ksub * idx->dsub];

        for (size_t k_i = 0; k_i < idx->ksub; k_i++) {
            float dist = gv_pq_subvec_distance(query_subvec, &subcodebook[k_i * idx->dsub], idx->dsub);
            distance_table[m_i * idx->ksub + k_i] = dist * dist; /* Store squared distance */
        }
    }

    /* Use max-heap for top-k */
    GV_PQHeapItem *heap = (GV_PQHeapItem *)malloc(k * sizeof(GV_PQHeapItem));
    if (!heap) {
        free(distance_table);
        return -1;
    }
    size_t heap_size = 0;

    /* Scan all entries using ADC */
    for (size_t i = 0; i < idx->entry_count; i++) {
        if (idx->entries[i].deleted) continue;

        /* Metadata filter */
        if (filter_key && filter_value) {
            if (!gv_pq_metadata_match(idx->entries[i].metadata, filter_key, filter_value)) {
                continue;
            }
        }

        /* Compute approximate distance using lookup table */
        float dist_squared = 0.0f;
        for (size_t m_i = 0; m_i < idx->m; m_i++) {
            uint8_t code = idx->entries[i].codes[m_i];
            dist_squared += distance_table[m_i * idx->ksub + code];
        }
        float dist = sqrtf(dist_squared);

        gv_pq_heap_push(heap, &heap_size, k, dist, i);
    }

    /* Extract results from heap in sorted order */
    int n = (int)heap_size;
    for (int i = n - 1; i >= 0; i--) {
        size_t entry_idx = heap[0].idx;
        float approx_dist = heap[0].dist;

        GV_PQEntry *entry = &idx->entries[entry_idx];

        /* Create result vector with exact distance if needed */
        GV_Vector *result_vec = gv_vector_create_from_data(idx->dimension, entry->raw_data);
        if (result_vec) {
            /* Copy metadata */
            GV_Metadata *cur = entry->metadata;
            while (cur) {
                if (cur->key && cur->value) {
                    gv_vector_set_metadata(result_vec, cur->key, cur->value);
                }
                cur = cur->next;
            }

            /* Compute exact distance for reranking */
            float exact_dist = gv_distance(query, result_vec, distance_type);

            results[i].vector = result_vec;
            results[i].distance = exact_dist;
            results[i].is_sparse = 0;
            results[i].sparse_vector = NULL;
        } else {
            results[i].vector = NULL;
            results[i].distance = approx_dist;
            results[i].is_sparse = 0;
            results[i].sparse_vector = NULL;
        }

        /* Remove from heap */
        heap[0] = heap[heap_size - 1];
        heap_size--;
        if (heap_size > 0) {
            gv_pq_heap_sift_down(heap, heap_size, 0);
        }
    }

    free(heap);
    free(distance_table);

    /* Re-sort by exact distance for better accuracy */
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (results[i].distance > results[j].distance) {
                GV_SearchResult tmp = results[i];
                results[i] = results[j];
                results[j] = tmp;
            }
        }
    }

    return n;
}

/* Range search */
int gv_pq_range_search(void *index, const GV_Vector *query, float radius,
                       GV_SearchResult *results, size_t max_results,
                       GV_DistanceType distance_type,
                       const char *filter_key, const char *filter_value) {
    if (!index || !query || !results || max_results == 0 || radius < 0.0f) return -1;
    GV_PQIndex *idx = (GV_PQIndex *)index;

    if (!idx->trained) return -1;
    if (query->dimension != idx->dimension) return -1;

    /* Build distance lookup table for ADC */
    float *distance_table = (float *)malloc(idx->m * idx->ksub * sizeof(float));
    if (!distance_table) return -1;

    for (size_t m_i = 0; m_i < idx->m; m_i++) {
        const float *query_subvec = &query->data[m_i * idx->dsub];
        const float *subcodebook = &idx->codebooks[m_i * idx->ksub * idx->dsub];

        for (size_t k_i = 0; k_i < idx->ksub; k_i++) {
            float dist = gv_pq_subvec_distance(query_subvec, &subcodebook[k_i * idx->dsub], idx->dsub);
            distance_table[m_i * idx->ksub + k_i] = dist * dist;
        }
    }

    size_t found = 0;

    /* Scan all entries */
    for (size_t i = 0; i < idx->entry_count && found < max_results; i++) {
        if (idx->entries[i].deleted) continue;

        /* Metadata filter */
        if (filter_key && filter_value) {
            if (!gv_pq_metadata_match(idx->entries[i].metadata, filter_key, filter_value)) {
                continue;
            }
        }

        /* Compute approximate distance */
        float dist_squared = 0.0f;
        for (size_t m_i = 0; m_i < idx->m; m_i++) {
            uint8_t code = idx->entries[i].codes[m_i];
            dist_squared += distance_table[m_i * idx->ksub + code];
        }
        float approx_dist = sqrtf(dist_squared);

        /* Early rejection if approximate distance exceeds radius */
        if (approx_dist > radius * 1.5f) continue; /* Allow some slack for approximation error */

        GV_PQEntry *entry = &idx->entries[i];
        GV_Vector *result_vec = gv_vector_create_from_data(idx->dimension, entry->raw_data);
        if (result_vec) {
            /* Copy metadata */
            GV_Metadata *cur = entry->metadata;
            while (cur) {
                if (cur->key && cur->value) {
                    gv_vector_set_metadata(result_vec, cur->key, cur->value);
                }
                cur = cur->next;
            }

            /* Compute exact distance */
            float exact_dist = gv_distance(query, result_vec, distance_type);

            if (exact_dist <= radius) {
                results[found].vector = result_vec;
                results[found].distance = exact_dist;
                results[found].is_sparse = 0;
                results[found].sparse_vector = NULL;
                found++;
            } else {
                gv_vector_destroy(result_vec);
            }
        }
    }

    free(distance_table);
    return (int)found;
}

/* Check if trained */
int gv_pq_is_trained(const void *index) {
    if (!index) return 0;
    const GV_PQIndex *idx = (const GV_PQIndex *)index;
    return idx->trained;
}

/* Count vectors */
size_t gv_pq_count(const void *index) {
    if (!index) return 0;
    const GV_PQIndex *idx = (const GV_PQIndex *)index;
    size_t count = 0;
    for (size_t i = 0; i < idx->entry_count; i++) {
        if (!idx->entries[i].deleted) count++;
    }
    return count;
}

/* Delete vector */
int gv_pq_delete(void *index, size_t entry_index) {
    if (!index) return -1;
    GV_PQIndex *idx = (GV_PQIndex *)index;
    if (entry_index >= idx->entry_count) return -1;
    idx->entries[entry_index].deleted = 1;
    return 0;
}

/* Update vector */
int gv_pq_update(void *index, size_t entry_index, const float *new_data, size_t dimension) {
    if (!index || !new_data) return -1;
    GV_PQIndex *idx = (GV_PQIndex *)index;

    if (!idx->trained) return -1;
    if (dimension != idx->dimension) return -1;
    if (entry_index >= idx->entry_count) return -1;
    if (idx->entries[entry_index].deleted) return -1;

    GV_PQEntry *entry = &idx->entries[entry_index];

    /* Re-encode with new data */
    gv_pq_encode(idx, new_data, entry->codes);

    /* Update raw data */
    memcpy(entry->raw_data, new_data, idx->dimension * sizeof(float));

    return 0;
}

/* Save to file */
int gv_pq_save(const void *index, FILE *out, uint32_t version) {
    if (!index || !out) return -1;
    const GV_PQIndex *idx = (const GV_PQIndex *)index;
    (void)version;

    /* Write PQ index header */
    if (gv_pq_write_u32(out, (uint32_t)idx->dimension) != 0) return -1;
    if (gv_pq_write_u32(out, (uint32_t)idx->m) != 0) return -1;
    if (gv_pq_write_u8(out, idx->nbits) != 0) return -1;
    if (gv_pq_write_u32(out, (uint32_t)idx->train_iters) != 0) return -1;
    if (gv_pq_write_u32(out, (uint32_t)idx->trained) != 0) return -1;

    /* Write codebooks */
    size_t codebook_size = idx->m * idx->ksub * idx->dsub;
    if (fwrite(idx->codebooks, sizeof(float), codebook_size, out) != codebook_size) return -1;

    /* Write entry count */
    if (gv_pq_write_u32(out, (uint32_t)idx->entry_count) != 0) return -1;

    /* Write entries */
    for (size_t i = 0; i < idx->entry_count; i++) {
        const GV_PQEntry *entry = &idx->entries[i];

        /* Write deleted flag and id */
        if (gv_pq_write_u32(out, (uint32_t)entry->deleted) != 0) return -1;
        if (gv_pq_write_u32(out, (uint32_t)entry->id) != 0) return -1;

        /* Write codes */
        if (fwrite(entry->codes, sizeof(uint8_t), idx->m, out) != idx->m) return -1;

        /* Write raw data */
        if (fwrite(entry->raw_data, sizeof(float), idx->dimension, out) != idx->dimension) return -1;

        /* Write metadata */
        uint32_t meta_count = 0;
        GV_Metadata *cur = entry->metadata;
        while (cur) { meta_count++; cur = cur->next; }
        if (gv_pq_write_u32(out, meta_count) != 0) return -1;

        cur = entry->metadata;
        while (cur) {
            uint32_t klen = cur->key ? (uint32_t)strlen(cur->key) : 0;
            uint32_t vlen = cur->value ? (uint32_t)strlen(cur->value) : 0;
            if (gv_pq_write_str(out, cur->key ? cur->key : "", klen) != 0) return -1;
            if (gv_pq_write_str(out, cur->value ? cur->value : "", vlen) != 0) return -1;
            cur = cur->next;
        }
    }

    return 0;
}

/* Load from file */
int gv_pq_load(void **index_ptr, FILE *in, size_t dimension, uint32_t version) {
    if (!index_ptr || !in) return -1;
    (void)version;

    uint32_t file_dim = 0, m = 0, train_iters = 0, trained = 0, entry_count = 0;
    uint8_t nbits = 0;

    if (gv_pq_read_u32(in, &file_dim) != 0) return -1;
    if (gv_pq_read_u32(in, &m) != 0) return -1;
    if (gv_pq_read_u8(in, &nbits) != 0) return -1;
    if (gv_pq_read_u32(in, &train_iters) != 0) return -1;
    if (gv_pq_read_u32(in, &trained) != 0) return -1;

    if (dimension != 0 && dimension != (size_t)file_dim) return -1;

    /* Create index */
    GV_PQConfig config = {
        .m = (size_t)m,
        .nbits = nbits,
        .train_iters = (size_t)train_iters
    };

    void *index = gv_pq_create((size_t)file_dim, &config);
    if (!index) return -1;

    GV_PQIndex *idx = (GV_PQIndex *)index;

    /* Read codebooks */
    size_t codebook_size = idx->m * idx->ksub * idx->dsub;
    if (fread(idx->codebooks, sizeof(float), codebook_size, in) != codebook_size) {
        gv_pq_destroy(index);
        return -1;
    }

    idx->trained = (int)trained;

    /* Read entry count */
    if (gv_pq_read_u32(in, &entry_count) != 0) {
        gv_pq_destroy(index);
        return -1;
    }

    /* Ensure capacity */
    if (entry_count > idx->entry_capacity) {
        GV_PQEntry *new_entries = (GV_PQEntry *)realloc(idx->entries, entry_count * sizeof(GV_PQEntry));
        if (!new_entries) {
            gv_pq_destroy(index);
            return -1;
        }
        memset(&new_entries[idx->entry_capacity], 0, (entry_count - idx->entry_capacity) * sizeof(GV_PQEntry));
        idx->entries = new_entries;
        idx->entry_capacity = entry_count;
    }

    /* Read entries */
    for (uint32_t i = 0; i < entry_count; i++) {
        GV_PQEntry *entry = &idx->entries[i];

        uint32_t deleted = 0, id = 0;
        if (gv_pq_read_u32(in, &deleted) != 0) { gv_pq_destroy(index); return -1; }
        if (gv_pq_read_u32(in, &id) != 0) { gv_pq_destroy(index); return -1; }

        entry->deleted = (int)deleted;
        entry->id = (size_t)id;

        /* Read codes */
        entry->codes = (uint8_t *)malloc(idx->m * sizeof(uint8_t));
        if (!entry->codes) { gv_pq_destroy(index); return -1; }
        if (fread(entry->codes, sizeof(uint8_t), idx->m, in) != idx->m) {
            gv_pq_destroy(index);
            return -1;
        }

        /* Read raw data */
        entry->raw_data = (float *)malloc(idx->dimension * sizeof(float));
        if (!entry->raw_data) { gv_pq_destroy(index); return -1; }
        if (fread(entry->raw_data, sizeof(float), idx->dimension, in) != idx->dimension) {
            gv_pq_destroy(index);
            return -1;
        }

        /* Read metadata */
        uint32_t meta_count = 0;
        if (gv_pq_read_u32(in, &meta_count) != 0) { gv_pq_destroy(index); return -1; }

        entry->metadata = NULL;
        for (uint32_t m_idx = 0; m_idx < meta_count; m_idx++) {
            uint32_t klen = 0, vlen = 0;
            char *key = NULL, *value = NULL;
            if (gv_pq_read_u32(in, &klen) != 0) { gv_pq_destroy(index); return -1; }
            if (gv_pq_read_str(in, &key, klen) != 0) { gv_pq_destroy(index); return -1; }
            if (gv_pq_read_u32(in, &vlen) != 0) { free(key); gv_pq_destroy(index); return -1; }
            if (gv_pq_read_str(in, &value, vlen) != 0) { free(key); gv_pq_destroy(index); return -1; }

            GV_Metadata *node = (GV_Metadata *)malloc(sizeof(GV_Metadata));
            if (!node) { free(key); free(value); gv_pq_destroy(index); return -1; }
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

/* Destroy PQ index */
void gv_pq_destroy(void *index) {
    if (!index) return;
    GV_PQIndex *idx = (GV_PQIndex *)index;

    /* Free codebooks */
    free(idx->codebooks);

    /* Free entries */
    for (size_t i = 0; i < idx->entry_count; i++) {
        free(idx->entries[i].codes);
        free(idx->entries[i].raw_data);
        gv_metadata_free(idx->entries[i].metadata);
    }
    free(idx->entries);

    free(idx);
}
