#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "index/lsh.h"
#include "search/distance.h"
#include "schema/vector.h"
#include "schema/metadata.h"
#include "storage/soa_storage.h"
#include "core/utils.h"

typedef struct GV_LSHBucket {
    size_t *indices;
    size_t count;
    size_t capacity;
} GV_LSHBucket;

typedef struct GV_LSHIndex {
    size_t dimension;
    GV_LSHConfig config;
    float effective_bucket_width; /* bucket_width * sqrt(dim) for dimension-aware scaling */
    GV_SoAStorage *storage;
    int owns_storage;
    float **hyperplanes;
    float *offsets;        /* E2LSH random offsets b_i, one per hash function */
    GV_LSHBucket **tables;
    size_t num_buckets;
} GV_LSHIndex;

typedef struct {
    const GV_Vector *vector;
    float distance;
    size_t vec_idx;
} GV_LSHCandidate;


static uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static float uniform_random(uint64_t *state) {
    return (float)xorshift64(state) / (float)UINT64_MAX;
}

static float gaussian_random(uint64_t *state) {
    float u1 = uniform_random(state);
    float u2 = uniform_random(state);
    if (u1 < 1e-9f) u1 = 1e-9f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
}

static void generate_hyperplanes(float **hyperplanes, float *offsets,
                                 size_t num_tables, size_t num_hash_bits,
                                 size_t dimension, uint64_t seed, float effective_bucket_width) {
    uint64_t state = seed;
    size_t total_planes = num_tables * num_hash_bits;

    for (size_t i = 0; i < total_planes; ++i) {
        hyperplanes[i] = (float *)malloc(dimension * sizeof(float));
        if (hyperplanes[i] != NULL) {
            for (size_t d = 0; d < dimension; ++d) {
                hyperplanes[i][d] = gaussian_random(&state);
            }
        }
        offsets[i] = uniform_random(&state) * effective_bucket_width;
    }
}

static uint32_t hash_vector(const float *data, size_t dimension, float **hyperplanes,
                            const float *offsets, float effective_bucket_width,
                            size_t table_idx, size_t num_hash_bits) {
    uint32_t hash = 0;
    size_t base_idx = table_idx * num_hash_bits;

    for (size_t b = 0; b < num_hash_bits; ++b) {
        float *plane = hyperplanes[base_idx + b];
        if (plane == NULL) continue;

        float dot = 0.0f;
        for (size_t d = 0; d < dimension; ++d) {
            dot += data[d] * plane[d];
        }

        /* E2LSH: h(v) = floor((a·v + b) / w) */
        int32_t h = (int32_t)floorf((dot + offsets[base_idx + b]) / effective_bucket_width);
        hash = hash * 2654435761U + (uint32_t)h;
    }

    return hash;
}

static int bucket_add(GV_LSHBucket *bucket, size_t index) {
    if (bucket->count >= bucket->capacity) {
        size_t new_capacity = bucket->capacity == 0 ? 8 : bucket->capacity * 2;
        size_t *new_indices = (size_t *)realloc(bucket->indices, new_capacity * sizeof(size_t));
        if (new_indices == NULL) {
            return -1;
        }
        bucket->indices = new_indices;
        bucket->capacity = new_capacity;
    }
    bucket->indices[bucket->count++] = index;
    return 0;
}

void *lsh_create(size_t dimension, const GV_LSHConfig *config, GV_SoAStorage *soa_storage) {
    if (dimension == 0) {
        return NULL;
    }

    GV_LSHIndex *index = (GV_LSHIndex *)calloc(1, sizeof(GV_LSHIndex));
    if (index == NULL) {
        return NULL;
    }

    index->dimension = dimension;
    index->config.num_tables = (config && config->num_tables > 0) ? config->num_tables : 8;
    index->config.num_hash_bits = (config && config->num_hash_bits > 0) ? config->num_hash_bits : 4;
    index->config.seed = (config && config->seed != 0) ? config->seed : 42;
    index->config.bucket_width = (config && config->bucket_width > 0.0f) ? config->bucket_width : 4.0f;
    index->effective_bucket_width = index->config.bucket_width * sqrtf((float)dimension);

    /* E2LSH uses modular hashing, not power-of-2 buckets */
    size_t num_buckets = 65536;
    index->num_buckets = num_buckets;

    if (soa_storage != NULL) {
        index->storage = soa_storage;
        index->owns_storage = 0;
    } else {
        index->storage = soa_storage_create(dimension, 1024);
        if (index->storage == NULL) {
            free(index);
            return NULL;
        }
        index->owns_storage = 1;
    }

    size_t total_planes = index->config.num_tables * index->config.num_hash_bits;
    index->hyperplanes = (float **)calloc(total_planes, sizeof(float *));
    if (index->hyperplanes == NULL) {
        if (index->owns_storage) {
            soa_storage_destroy(index->storage);
        }
        free(index);
        return NULL;
    }

    index->offsets = (float *)malloc(total_planes * sizeof(float));
    if (index->offsets == NULL) {
        free(index->hyperplanes);
        if (index->owns_storage) {
            soa_storage_destroy(index->storage);
        }
        free(index);
        return NULL;
    }

    generate_hyperplanes(index->hyperplanes, index->offsets,
                        index->config.num_tables,
                        index->config.num_hash_bits, dimension,
                        index->config.seed, index->effective_bucket_width);

    index->tables = (GV_LSHBucket **)calloc(index->config.num_tables, sizeof(GV_LSHBucket *));
    if (index->tables == NULL) {
        for (size_t i = 0; i < total_planes; ++i) {
            free(index->hyperplanes[i]);
        }
        free(index->hyperplanes);
        if (index->owns_storage) {
            soa_storage_destroy(index->storage);
        }
        free(index);
        return NULL;
    }

    for (size_t t = 0; t < index->config.num_tables; ++t) {
        index->tables[t] = (GV_LSHBucket *)calloc(num_buckets, sizeof(GV_LSHBucket));
        if (index->tables[t] == NULL) {
            for (size_t i = 0; i < t; ++i) {
                for (size_t b = 0; b < num_buckets; ++b) {
                    free(index->tables[i][b].indices);
                }
                free(index->tables[i]);
            }
            free(index->tables);
            for (size_t i = 0; i < total_planes; ++i) {
                free(index->hyperplanes[i]);
            }
            free(index->hyperplanes);
            if (index->owns_storage) {
                soa_storage_destroy(index->storage);
            }
            free(index);
            return NULL;
        }
    }

    return index;
}

int lsh_insert(void *index, GV_Vector *vector) {
    if (index == NULL || vector == NULL) {
        return -1;
    }

    GV_LSHIndex *lsh = (GV_LSHIndex *)index;
    if (vector->dimension != lsh->dimension) {
        return -1;
    }

    GV_Metadata *metadata = vector->metadata;
    vector->metadata = NULL;
    size_t vector_index = soa_storage_add(lsh->storage, vector->data, metadata);
    if (vector_index == (size_t)-1) {
        vector->metadata = metadata;
        return -1;
    }

    const float *data = soa_storage_get_data(lsh->storage, vector_index);
    if (data == NULL) {
        return -1;
    }

    for (size_t t = 0; t < lsh->config.num_tables; ++t) {
        uint32_t hash = hash_vector(data, lsh->dimension, lsh->hyperplanes,
                                     lsh->offsets, lsh->effective_bucket_width,
                                     t, lsh->config.num_hash_bits);
        uint32_t bucket_idx = hash % lsh->num_buckets;

        if (bucket_add(&lsh->tables[t][bucket_idx], vector_index) != 0) {
            vector_destroy(vector);
            return -1;
        }
    }

    vector_destroy(vector);
    return 0;
}

static int compare_candidates(const void *a, const void *b) {
    const GV_LSHCandidate *ca = (const GV_LSHCandidate *)a;
    const GV_LSHCandidate *cb = (const GV_LSHCandidate *)b;
    if (ca->distance < cb->distance) return -1;
    if (ca->distance > cb->distance) return 1;
    return 0;
}

static void heap_sift_down(GV_LSHCandidate *heap, size_t size, size_t idx) {
    size_t largest = idx;
    size_t left = 2 * idx + 1;
    size_t right = 2 * idx + 2;

    if (left < size && heap[left].distance > heap[largest].distance) {
        largest = left;
    }
    if (right < size && heap[right].distance > heap[largest].distance) {
        largest = right;
    }

    if (largest != idx) {
        GV_LSHCandidate tmp = heap[idx];
        heap[idx] = heap[largest];
        heap[largest] = tmp;
        heap_sift_down(heap, size, largest);
    }
}

static void heap_push(GV_LSHCandidate *heap, size_t *heap_size, size_t k,
                     const GV_Vector *vector, float distance, size_t vec_idx) {
    if (*heap_size < k) {
        heap[*heap_size].vector = vector;
        heap[*heap_size].distance = distance;
        heap[*heap_size].vec_idx = vec_idx;
        (*heap_size)++;

        for (int i = (int)*heap_size / 2 - 1; i >= 0; --i) {
            heap_sift_down(heap, *heap_size, (size_t)i);
        }
    } else if (distance < heap[0].distance) {
        heap[0].vector = vector;
        heap[0].distance = distance;
        heap[0].vec_idx = vec_idx;
        heap_sift_down(heap, *heap_size, 0);
    }
}

int lsh_search(void *index, const GV_Vector *query, size_t k,
                  GV_SearchResult *results, GV_DistanceType distance_type,
                  const char *filter_key, const char *filter_value) {
    if (index == NULL || query == NULL || results == NULL || k == 0) {
        return -1;
    }

    GV_LSHIndex *lsh = (GV_LSHIndex *)index;
    if (query->dimension != lsh->dimension) {
        return -1;
    }

    size_t storage_count = soa_storage_count(lsh->storage);
    if (storage_count == 0) {
        return 0;
    }

    int *seen = (int *)calloc(storage_count, sizeof(int));
    if (seen == NULL) {
        return -1;
    }

    size_t *candidates = (size_t *)malloc(storage_count * sizeof(size_t));
    if (candidates == NULL) {
        free(seen);
        return -1;
    }
    size_t candidate_count = 0;

    for (size_t t = 0; t < lsh->config.num_tables; ++t) {
        uint32_t hash = hash_vector(query->data, lsh->dimension, lsh->hyperplanes,
                                     lsh->offsets, lsh->effective_bucket_width,
                                     t, lsh->config.num_hash_bits);
        uint32_t bucket_idx = hash % lsh->num_buckets;

        GV_LSHBucket *bucket = &lsh->tables[t][bucket_idx];
        for (size_t i = 0; i < bucket->count; ++i) {
            size_t vec_idx = bucket->indices[i];
            if (vec_idx < storage_count && !seen[vec_idx]) {
                if (soa_storage_is_deleted(lsh->storage, vec_idx) == 0) {
                    seen[vec_idx] = 1;
                    candidates[candidate_count++] = vec_idx;
                }
            }
        }
    }

    free(seen);

    if (candidate_count == 0) {
        free(candidates);
        return 0;
    }

    GV_LSHCandidate *heap = (GV_LSHCandidate *)malloc(k * sizeof(GV_LSHCandidate));
    if (heap == NULL) {
        free(candidates);
        return -1;
    }
    size_t heap_size = 0;

    for (size_t i = 0; i < candidate_count; ++i) {
        size_t vec_idx = candidates[i];
        const float *data = soa_storage_get_data(lsh->storage, vec_idx);
        GV_Metadata *metadata = soa_storage_get_metadata(lsh->storage, vec_idx);

        if (data == NULL) continue;

        if (filter_key != NULL && filter_value != NULL) {
            const char *meta_val = metadata_get_direct(metadata, filter_key);
            if (meta_val == NULL || strcmp(meta_val, filter_value) != 0) {
                continue;
            }
        }

        GV_Vector temp_vec = {
            .data = (float *)data,
            .dimension = lsh->dimension,
            .metadata = NULL
        };

        float dist = distance(&temp_vec, query, distance_type);
        if (dist < 0.0f) continue;

        heap_push(heap, &heap_size, k, &temp_vec, dist, vec_idx);
    }

    free(candidates);

    GV_LSHCandidate *sorted = (GV_LSHCandidate *)malloc(heap_size * sizeof(GV_LSHCandidate));
    if (sorted == NULL) {
        free(heap);
        return -1;
    }
    memcpy(sorted, heap, heap_size * sizeof(GV_LSHCandidate));
    free(heap);

    qsort(sorted, heap_size, sizeof(GV_LSHCandidate), compare_candidates);

    for (size_t i = 0; i < heap_size; ++i) {
        size_t sidx = sorted[i].vec_idx;
        const float *sdata = soa_storage_get_data(lsh->storage, sidx);
        if (sdata == NULL) continue;

        GV_Vector *result_vec = vector_create_from_data(lsh->dimension, sdata);
        if (result_vec == NULL) {
            for (size_t j = 0; j < i; ++j) {
                if (results[j].vector != NULL) {
                    vector_destroy((GV_Vector *)results[j].vector);
                }
            }
            free(sorted);
            return -1;
        }

        GV_Metadata *orig_metadata = soa_storage_get_metadata(lsh->storage, sidx);
        result_vec->metadata = metadata_copy(orig_metadata);

        results[i].vector = result_vec;
        results[i].sparse_vector = NULL;
        results[i].is_sparse = 0;
        results[i].distance = sorted[i].distance;
        results[i].id = sidx;
    }

    free(sorted);
    return (int)heap_size;
}

int lsh_range_search(void *index, const GV_Vector *query, float radius,
                        GV_SearchResult *results, size_t max_results,
                        GV_DistanceType distance_type,
                        const char *filter_key, const char *filter_value) {
    if (index == NULL || query == NULL || results == NULL || max_results == 0) {
        return -1;
    }

    GV_LSHIndex *lsh = (GV_LSHIndex *)index;
    if (query->dimension != lsh->dimension) {
        return -1;
    }

    size_t storage_count = soa_storage_count(lsh->storage);
    if (storage_count == 0) {
        return 0;
    }

    int *seen = (int *)calloc(storage_count, sizeof(int));
    if (seen == NULL) {
        return -1;
    }

    size_t *candidates = (size_t *)malloc(storage_count * sizeof(size_t));
    if (candidates == NULL) {
        free(seen);
        return -1;
    }
    size_t candidate_count = 0;

    for (size_t t = 0; t < lsh->config.num_tables; ++t) {
        uint32_t hash = hash_vector(query->data, lsh->dimension, lsh->hyperplanes,
                                     lsh->offsets, lsh->effective_bucket_width,
                                     t, lsh->config.num_hash_bits);
        uint32_t bucket_idx = hash % lsh->num_buckets;

        GV_LSHBucket *bucket = &lsh->tables[t][bucket_idx];
        for (size_t i = 0; i < bucket->count; ++i) {
            size_t vec_idx = bucket->indices[i];
            if (vec_idx < storage_count && !seen[vec_idx]) {
                if (soa_storage_is_deleted(lsh->storage, vec_idx) == 0) {
                    seen[vec_idx] = 1;
                    candidates[candidate_count++] = vec_idx;
                }
            }
        }
    }

    free(seen);

    if (candidate_count == 0) {
        free(candidates);
        return 0;
    }

    size_t result_count = 0;
    for (size_t i = 0; i < candidate_count && result_count < max_results; ++i) {
        size_t vec_idx = candidates[i];
        const float *data = soa_storage_get_data(lsh->storage, vec_idx);
        GV_Metadata *metadata = soa_storage_get_metadata(lsh->storage, vec_idx);

        if (data == NULL) continue;

        if (filter_key != NULL && filter_value != NULL) {
            const char *meta_val = metadata_get_direct(metadata, filter_key);
            if (meta_val == NULL || strcmp(meta_val, filter_value) != 0) {
                continue;
            }
        }

        GV_Vector temp_vec = {
            .data = (float *)data,
            .dimension = lsh->dimension,
            .metadata = NULL
        };

        float dist = distance(&temp_vec, query, distance_type);
        if (dist < 0.0f || dist > radius) continue;

        GV_Vector *result_vec = vector_create_from_data(lsh->dimension, data);
        if (result_vec == NULL) {
            for (size_t j = 0; j < result_count; ++j) {
                if (results[j].vector != NULL) {
                    vector_destroy((GV_Vector *)results[j].vector);
                }
            }
            free(candidates);
            return -1;
        }

        result_vec->metadata = metadata_copy(metadata);

        results[result_count].vector = result_vec;
        results[result_count].sparse_vector = NULL;
        results[result_count].is_sparse = 0;
        results[result_count].distance = dist;
        results[result_count].id = vec_idx;
        result_count++;
    }

    free(candidates);

    /* Sort GV_SearchResult by distance(not using compare_candidates which is for GV_LSHCandidate) */
    for (size_t i = 0; i + 1 < (size_t)result_count; ++i) {
        size_t minj = i;
        for (size_t j = i + 1; j < (size_t)result_count; ++j) {
            if (results[j].distance < results[minj].distance) minj = j;
        }
        if (minj != i) {
            GV_SearchResult tmp = results[i];
            results[i] = results[minj];
            results[minj] = tmp;
        }
    }

    return (int)result_count;
}

void lsh_destroy(void *index) {
    if (index == NULL) {
        return;
    }

    GV_LSHIndex *lsh = (GV_LSHIndex *)index;

    if (lsh->tables != NULL) {
        for (size_t t = 0; t < lsh->config.num_tables; ++t) {
            if (lsh->tables[t] != NULL) {
                for (size_t b = 0; b < lsh->num_buckets; ++b) {
                    free(lsh->tables[t][b].indices);
                }
                free(lsh->tables[t]);
            }
        }
        free(lsh->tables);
    }

    if (lsh->hyperplanes != NULL) {
        size_t total_planes = lsh->config.num_tables * lsh->config.num_hash_bits;
        for (size_t i = 0; i < total_planes; ++i) {
            free(lsh->hyperplanes[i]);
        }
        free(lsh->hyperplanes);
    }

    free(lsh->offsets);

    if (lsh->storage != NULL && lsh->owns_storage) {
        soa_storage_destroy(lsh->storage);
    }

    free(lsh);
}

size_t lsh_count(const void *index) {
    if (index == NULL) {
        return 0;
    }

    GV_LSHIndex *lsh = (GV_LSHIndex *)index;
    return soa_storage_count(lsh->storage);
}

int lsh_delete(void *index, size_t vector_index) {
    if (index == NULL) {
        return -1;
    }

    GV_LSHIndex *lsh = (GV_LSHIndex *)index;
    return soa_storage_mark_deleted(lsh->storage, vector_index);
}

int lsh_update(void *index, size_t vector_index, const float *new_data, size_t dimension) {
    if (index == NULL || new_data == NULL) {
        return -1;
    }

    GV_LSHIndex *lsh = (GV_LSHIndex *)index;
    if (dimension != lsh->dimension) {
        return -1;
    }

    size_t storage_count = soa_storage_count(lsh->storage);
    if (vector_index >= storage_count) {
        return -1;
    }

    if (soa_storage_is_deleted(lsh->storage, vector_index) != 0) {
        return -1;
    }

    const float *old_data = soa_storage_get_data(lsh->storage, vector_index);
    if (old_data == NULL) {
        return -1;
    }

    for (size_t t = 0; t < lsh->config.num_tables; ++t) {
        uint32_t old_hash = hash_vector(old_data, lsh->dimension, lsh->hyperplanes,
                                        lsh->offsets, lsh->effective_bucket_width,
                                        t, lsh->config.num_hash_bits);
        uint32_t old_bucket_idx = old_hash % lsh->num_buckets;

        GV_LSHBucket *old_bucket = &lsh->tables[t][old_bucket_idx];
        size_t write_pos = 0;
        for (size_t i = 0; i < old_bucket->count; ++i) {
            if (old_bucket->indices[i] != vector_index) {
                old_bucket->indices[write_pos++] = old_bucket->indices[i];
            }
        }
        old_bucket->count = write_pos;
    }

    if (soa_storage_update_data(lsh->storage, vector_index, new_data) != 0) {
        return -1;
    }

    for (size_t t = 0; t < lsh->config.num_tables; ++t) {
        uint32_t new_hash = hash_vector(new_data, lsh->dimension, lsh->hyperplanes,
                                        lsh->offsets, lsh->effective_bucket_width,
                                        t, lsh->config.num_hash_bits);
        uint32_t new_bucket_idx = new_hash % lsh->num_buckets;

        if (bucket_add(&lsh->tables[t][new_bucket_idx], vector_index) != 0) {
            return -1;
        }
    }

    return 0;
}

static int lsh_read_metadata_into(FILE *in, GV_Metadata **metadata_out) {
    if (metadata_out == NULL) {
        return -1;
    }

    uint32_t count = 0;
    if (read_u32(in, &count) != 0) {
        return -1;
    }

    GV_Metadata *head = NULL;
    GV_Metadata *tail = NULL;

    for (uint32_t i = 0; i < count; ++i) {
        uint32_t key_len = 0;
        uint32_t val_len = 0;
        char *key = NULL;
        char *value = NULL;

        if (read_u32(in, &key_len) != 0) {
            metadata_free(head);
            return -1;
        }
        if (read_str(in, &key, key_len) != 0) {
            free(key);
            metadata_free(head);
            return -1;
        }

        if (read_u32(in, &val_len) != 0) {
            free(key);
            metadata_free(head);
            return -1;
        }
        if (read_str(in, &value, val_len) != 0) {
            free(key);
            metadata_free(head);
            return -1;
        }

        GV_Metadata *new_meta = (GV_Metadata *)malloc(sizeof(GV_Metadata));
        if (new_meta == NULL) {
            free(key);
            free(value);
            metadata_free(head);
            return -1;
        }
        new_meta->key = key;
        new_meta->value = value;
        new_meta->next = NULL;

        if (head == NULL) {
            head = tail = new_meta;
        } else {
            tail->next = new_meta;
            tail = new_meta;
        }
    }

    *metadata_out = head;
    return 0;
}

int lsh_save(const void *index, FILE *out, uint32_t version) {
    if (index == NULL || out == NULL) {
        return -1;
    }

    GV_LSHIndex *lsh = (GV_LSHIndex *)index;

    if (write_size(out, lsh->config.num_tables) != 0) {
        return -1;
    }
    if (write_size(out, lsh->config.num_hash_bits) != 0) {
        return -1;
    }
    if (write_u64(out, lsh->config.seed) != 0) {
        return -1;
    }
    if (write_floats(out, &lsh->config.bucket_width, 1) != 0) {
        return -1;
    }
    if (write_floats(out, &lsh->effective_bucket_width, 1) != 0) {
        return -1;
    }

    size_t total_planes = lsh->config.num_tables * lsh->config.num_hash_bits;

    /* Write offsets array */
    if (write_floats(out, lsh->offsets, total_planes) != 0) {
        return -1;
    }

    for (size_t i = 0; i < total_planes; ++i) {
        if (lsh->hyperplanes[i] == NULL) {
            return -1;
        }
        if (write_floats(out, lsh->hyperplanes[i], lsh->dimension) != 0) {
            return -1;
        }
    }

    size_t storage_count = soa_storage_count(lsh->storage);
    if (write_size(out, storage_count) != 0) {
        return -1;
    }

    for (size_t i = 0; i < storage_count; ++i) {
        const float *data = soa_storage_get_data(lsh->storage, i);
        if (data == NULL) {
            return -1;
        }
        if (write_floats(out, data, lsh->dimension) != 0) {
            return -1;
        }

        GV_Metadata *metadata = soa_storage_get_metadata(lsh->storage, i);
        if (write_metadata(out, metadata) != 0) {
            return -1;
        }

        int deleted = soa_storage_is_deleted(lsh->storage, i);
        uint32_t deleted_flag = (deleted == 1) ? 1 : 0;
        if (write_u32(out, deleted_flag) != 0) {
            return -1;
        }
    }

    (void)version;
    return 0;
}

int lsh_load(void **index_ptr, FILE *in, size_t dimension, uint32_t version) {
    if (index_ptr == NULL || in == NULL || dimension == 0) {
        return -1;
    }

    GV_LSHConfig config;
    if (read_size(in, &config.num_tables) != 0) {
        return -1;
    }
    if (read_size(in, &config.num_hash_bits) != 0) {
        return -1;
    }
    if (read_u64(in, &config.seed) != 0) {
        return -1;
    }
    if (read_floats(in, &config.bucket_width, 1) != 0) {
        return -1;
    }

    GV_LSHIndex *lsh = (GV_LSHIndex *)lsh_create(dimension, &config, NULL);
    if (lsh == NULL) {
        return -1;
    }

    /* Read saved effective_bucket_width and override the computed one */
    if (read_floats(in, &lsh->effective_bucket_width, 1) != 0) {
        lsh_destroy(lsh);
        return -1;
    }

    size_t total_planes = config.num_tables * config.num_hash_bits;

    /* Read offsets (overwrite generated ones) */
    if (read_floats(in, lsh->offsets, total_planes) != 0) {
        lsh_destroy(lsh);
        return -1;
    }

    for (size_t i = 0; i < total_planes; ++i) {
        if (lsh->hyperplanes[i] == NULL) {
            lsh->hyperplanes[i] = (float *)malloc(dimension * sizeof(float));
            if (lsh->hyperplanes[i] == NULL) {
                lsh_destroy(lsh);
                return -1;
            }
        }
        if (read_floats(in, lsh->hyperplanes[i], dimension) != 0) {
            lsh_destroy(lsh);
            return -1;
        }
    }

    size_t vector_count = 0;
    if (read_size(in, &vector_count) != 0) {
        lsh_destroy(lsh);
        return -1;
    }

    for (size_t i = 0; i < vector_count; ++i) {
        float *data = (float *)malloc(dimension * sizeof(float));
        if (data == NULL) {
            lsh_destroy(lsh);
            return -1;
        }

        if (read_floats(in, data, dimension) != 0) {
            free(data);
            lsh_destroy(lsh);
            return -1;
        }

        GV_Metadata *metadata = NULL;
        if (lsh_read_metadata_into(in, &metadata) != 0) {
            free(data);
            lsh_destroy(lsh);
            return -1;
        }

        uint32_t deleted_flag = 0;
        if (read_u32(in, &deleted_flag) != 0) {
            free(data);
            metadata_free(metadata);
            lsh_destroy(lsh);
            return -1;
        }

        size_t vec_idx = soa_storage_add(lsh->storage, data, metadata);
        free(data);

        if (vec_idx == (size_t)-1) {
            lsh_destroy(lsh);
            return -1;
        }

        if (deleted_flag != 0) {
            soa_storage_mark_deleted(lsh->storage, vec_idx);
        } else {
            const float *stored_data = soa_storage_get_data(lsh->storage, vec_idx);
            if (stored_data != NULL) {
                for (size_t t = 0; t < lsh->config.num_tables; ++t) {
                    uint32_t hash = hash_vector(stored_data, dimension, lsh->hyperplanes,
                                               lsh->offsets, lsh->effective_bucket_width,
                                               t, lsh->config.num_hash_bits);
                    uint32_t bucket_idx = hash % lsh->num_buckets;
                    bucket_add(&lsh->tables[t][bucket_idx], vec_idx);
                }
            }
        }
    }

    *index_ptr = lsh;
    (void)version;
    return 0;
}
