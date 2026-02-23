#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "gigavector/gv_ivfflat.h"
#include "gigavector/gv_distance.h"
#include "gigavector/gv_vector.h"
#include "gigavector/gv_metadata.h"

/* Internal entry structure for IVF-Flat inverted lists */
typedef struct GV_IVFFlatEntry {
    GV_Vector *vector;               /* Full unquantized vector (owns it) */
    size_t id;                       /* Global insertion order ID */
    int deleted;                     /* Deletion flag: 1 if deleted, 0 if active */
    struct GV_IVFFlatEntry *next;    /* Next entry in linked list */
} GV_IVFFlatEntry;

/* Internal IVF-Flat index structure */
typedef struct {
    size_t dimension;                /* Vector dimensionality */
    GV_IVFFlatConfig config;         /* Index configuration */
    float *centroids;                /* Coarse centroids: nlist * dimension */
    GV_IVFFlatEntry **lists;         /* Array of linked list heads, one per centroid */
    size_t *list_sizes;              /* Count of entries per list */
    int trained;                     /* Training status flag */
    size_t total_count;              /* Total number of vectors (including deleted) */
    size_t next_id;                  /* Next ID to assign */
} GV_IVFFlatIndex;

/* Max-heap for top-k selection */
typedef struct {
    float dist;
    size_t id;
    GV_IVFFlatEntry *entry;
} GV_IVFFlatHeapItem;

static void gv_ivfflat_heap_sift_down(GV_IVFFlatHeapItem *heap, size_t size, size_t i) {
    while (1) {
        size_t l = 2 * i + 1;
        size_t r = l + 1;
        size_t largest = i;
        if (l < size && heap[l].dist > heap[largest].dist) largest = l;
        if (r < size && heap[r].dist > heap[largest].dist) largest = r;
        if (largest == i) break;
        GV_IVFFlatHeapItem tmp = heap[i];
        heap[i] = heap[largest];
        heap[largest] = tmp;
        i = largest;
    }
}

static void gv_ivfflat_heap_push(GV_IVFFlatHeapItem *heap, size_t *size, size_t capacity,
                                  float dist, size_t id, GV_IVFFlatEntry *entry) {
    if (*size < capacity) {
        heap[*size].dist = dist;
        heap[*size].id = id;
        heap[*size].entry = entry;
        (*size)++;
        /* Sift up */
        size_t i = *size - 1;
        while (i > 0) {
            size_t parent = (i - 1) / 2;
            if (heap[i].dist > heap[parent].dist) {
                GV_IVFFlatHeapItem tmp = heap[i];
                heap[i] = heap[parent];
                heap[parent] = tmp;
                i = parent;
            } else {
                break;
            }
        }
    } else if (dist < heap[0].dist) {
        heap[0].dist = dist;
        heap[0].id = id;
        heap[0].entry = entry;
        gv_ivfflat_heap_sift_down(heap, *size, 0);
    }
}

/* I/O helpers */
static int gv_ivfflat_write_u32(FILE *f, uint32_t v) {
    return fwrite(&v, sizeof(uint32_t), 1, f) == 1 ? 0 : -1;
}

static int gv_ivfflat_read_u32(FILE *f, uint32_t *v) {
    return (v && fread(v, sizeof(uint32_t), 1, f) == 1) ? 0 : -1;
}

static int gv_ivfflat_write_str(FILE *f, const char *s, uint32_t len) {
    if (gv_ivfflat_write_u32(f, len) != 0) return -1;
    if (len == 0) return 0;
    return fwrite(s, 1, len, f) == len ? 0 : -1;
}

static int gv_ivfflat_read_str(FILE *f, char **s, uint32_t len) {
    *s = NULL;
    if (len == 0) {
        *s = (char *)malloc(1);
        if (!*s) return -1;
        (*s)[0] = '\0';
        return 0;
    }
    char *buf = (char *)malloc(len + 1);
    if (!buf) return -1;
    if (fread(buf, 1, len, f) != len) {
        free(buf);
        return -1;
    }
    buf[len] = '\0';
    *s = buf;
    return 0;
}

/* K-means helper: assign vectors to nearest centroids */
static void gv_ivfflat_argmin(const float *data, size_t count, size_t dim,
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

/* Simple Lloyd's K-means algorithm */
static int gv_ivfflat_kmeans(const float *data, size_t count, size_t dim,
                              size_t k, size_t iters, float *out_centroids) {
    if (count < k || !data || !out_centroids) return -1;

    /* Initialize: pick first k vectors as initial centroids */
    memcpy(out_centroids, data, k * dim * sizeof(float));

    int *assign = (int *)malloc(count * sizeof(int));
    float *new_centroids = (float *)calloc(k * dim, sizeof(float));
    size_t *counts = (size_t *)calloc(k, sizeof(size_t));

    if (!assign || !new_centroids || !counts) {
        free(assign);
        free(new_centroids);
        free(counts);
        return -1;
    }

    /* Lloyd's iterations */
    for (size_t iter = 0; iter < iters; iter++) {
        /* Assignment step */
        gv_ivfflat_argmin(data, count, dim, out_centroids, k, assign);

        /* Update step */
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

        /* Average to get new centroids */
        for (size_t c = 0; c < k; c++) {
            if (counts[c] > 0) {
                for (size_t d = 0; d < dim; d++) {
                    new_centroids[c * dim + d] /= (float)counts[c];
                }
            }
        }

        memcpy(out_centroids, new_centroids, k * dim * sizeof(float));
    }

    free(assign);
    free(new_centroids);
    free(counts);
    return 0;
}

void *gv_ivfflat_create(size_t dimension, const GV_IVFFlatConfig *config) {
    if (dimension == 0) return NULL;

    GV_IVFFlatIndex *idx = (GV_IVFFlatIndex *)calloc(1, sizeof(GV_IVFFlatIndex));
    if (!idx) return NULL;

    idx->dimension = dimension;

    /* Apply defaults or user config */
    if (config) {
        idx->config = *config;
    } else {
        idx->config.nlist = 64;
        idx->config.nprobe = 4;
        idx->config.train_iters = 15;
        idx->config.use_cosine = 0;
    }

    /* Ensure nprobe doesn't exceed nlist */
    if (idx->config.nprobe > idx->config.nlist) {
        idx->config.nprobe = idx->config.nlist;
    }

    /* Allocate centroids */
    idx->centroids = (float *)malloc(idx->config.nlist * dimension * sizeof(float));
    if (!idx->centroids) {
        free(idx);
        return NULL;
    }

    /* Allocate inverted lists */
    idx->lists = (GV_IVFFlatEntry **)calloc(idx->config.nlist, sizeof(GV_IVFFlatEntry *));
    idx->list_sizes = (size_t *)calloc(idx->config.nlist, sizeof(size_t));

    if (!idx->lists || !idx->list_sizes) {
        free(idx->centroids);
        free(idx->lists);
        free(idx->list_sizes);
        free(idx);
        return NULL;
    }

    idx->trained = 0;
    idx->total_count = 0;
    idx->next_id = 0;

    return idx;
}

int gv_ivfflat_train(void *index, const float *data, size_t count) {
    if (!index || !data || count == 0) return -1;

    GV_IVFFlatIndex *idx = (GV_IVFFlatIndex *)index;

    if (count < idx->config.nlist) return -1;

    /* Train coarse centroids using K-means */
    if (gv_ivfflat_kmeans(data, count, idx->dimension, idx->config.nlist,
                          idx->config.train_iters, idx->centroids) != 0) {
        return -1;
    }

    idx->trained = 1;
    return 0;
}

int gv_ivfflat_insert(void *index, GV_Vector *vector) {
    if (!index || !vector) return -1;

    GV_IVFFlatIndex *idx = (GV_IVFFlatIndex *)index;

    if (!idx->trained) return -1;
    if (vector->dimension != idx->dimension) return -1;

    /* Find nearest centroid */
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

    /* Create new entry */
    GV_IVFFlatEntry *entry = (GV_IVFFlatEntry *)malloc(sizeof(GV_IVFFlatEntry));
    if (!entry) return -1;

    entry->vector = vector;
    entry->id = idx->next_id++;
    entry->deleted = 0;
    entry->next = idx->lists[best_list];

    /* Prepend to list */
    idx->lists[best_list] = entry;
    idx->list_sizes[best_list]++;
    idx->total_count++;

    return 0;
}

/* Metadata filter helper */
static int gv_ivfflat_metadata_match(const GV_Metadata *meta, const char *key, const char *value) {
    if (!key || !value) return 1; /* No filter = match all */

    const GV_Metadata *cur = meta;
    while (cur) {
        if (cur->key && cur->value &&
            strcmp(cur->key, key) == 0 && strcmp(cur->value, value) == 0) {
            return 1;
        }
        cur = cur->next;
    }
    return 0;
}

int gv_ivfflat_search(void *index, const GV_Vector *query, size_t k,
                      GV_SearchResult *results, GV_DistanceType distance_type,
                      const char *filter_key, const char *filter_value) {
    if (!index || !query || !results || k == 0) return -1;

    GV_IVFFlatIndex *idx = (GV_IVFFlatIndex *)index;

    if (!idx->trained) return -1;
    if (query->dimension != idx->dimension) return -1;

    /* Cap nprobe to nlist */
    size_t nprobe = idx->config.nprobe;
    if (nprobe > idx->config.nlist) nprobe = idx->config.nlist;

    /* Find nprobe closest centroids */
    GV_IVFFlatHeapItem *centroid_heap = (GV_IVFFlatHeapItem *)malloc(
        nprobe * sizeof(GV_IVFFlatHeapItem));
    if (!centroid_heap) return -1;

    size_t heap_size = 0;

    for (size_t i = 0; i < idx->config.nlist; i++) {
        const float *centroid = idx->centroids + i * idx->dimension;
        float dist = 0.0f;
        for (size_t d = 0; d < idx->dimension; d++) {
            float diff = query->data[d] - centroid[d];
            dist += diff * diff;
        }

        gv_ivfflat_heap_push(centroid_heap, &heap_size, nprobe, dist, i, NULL);
    }

    /* Extract probe lists */
    size_t *probe_lists = (size_t *)malloc(nprobe * sizeof(size_t));
    if (!probe_lists) {
        free(centroid_heap);
        return -1;
    }

    for (size_t i = nprobe; i > 0; i--) {
        probe_lists[i - 1] = centroid_heap[0].id;
        centroid_heap[0] = centroid_heap[heap_size - 1];
        heap_size--;
        if (heap_size > 0) {
            gv_ivfflat_heap_sift_down(centroid_heap, heap_size, 0);
        }
    }
    free(centroid_heap);

    /* Max-heap for top-k results */
    GV_IVFFlatHeapItem *heap = (GV_IVFFlatHeapItem *)malloc(k * sizeof(GV_IVFFlatHeapItem));
    if (!heap) {
        free(probe_lists);
        return -1;
    }

    heap_size = 0;

    /* Scan entries in selected lists */
    for (size_t i = 0; i < nprobe; i++) {
        size_t list_idx = probe_lists[i];
        GV_IVFFlatEntry *entry = idx->lists[list_idx];

        while (entry) {
            if (!entry->deleted) {
                /* Apply metadata filter */
                if (gv_ivfflat_metadata_match(entry->vector->metadata, filter_key, filter_value)) {
                    /* Compute distance */
                    float dist = gv_distance(query, entry->vector, distance_type);

                    if (dist >= 0.0f) {
                        gv_ivfflat_heap_push(heap, &heap_size, k, dist, entry->id, entry);
                    }
                }
            }
            entry = entry->next;
        }
    }

    free(probe_lists);

    /* Extract results from heap in sorted order (nearest first) */
    int n = (int)heap_size;
    for (int i = n - 1; i >= 0; i--) {
        GV_IVFFlatEntry *entry = heap[0].entry;
        float dist = heap[0].dist;

        /* Copy vector with metadata */
        GV_Vector *copy = gv_vector_create_from_data(entry->vector->dimension,
                                                      entry->vector->data);
        if (copy) {
            /* Copy metadata */
            GV_Metadata *meta = entry->vector->metadata;
            while (meta) {
                if (meta->key && meta->value) {
                    gv_vector_set_metadata(copy, meta->key, meta->value);
                }
                meta = meta->next;
            }
            results[i].vector = copy;
        } else {
            results[i].vector = NULL;
        }

        results[i].distance = dist;
        results[i].is_sparse = 0;
        results[i].sparse_vector = NULL;
        results[i].id = entry->id;

        /* Remove from heap */
        heap[0] = heap[heap_size - 1];
        heap_size--;
        if (heap_size > 0) {
            gv_ivfflat_heap_sift_down(heap, heap_size, 0);
        }
    }

    free(heap);
    return n;
}

int gv_ivfflat_range_search(void *index, const GV_Vector *query, float radius,
                            GV_SearchResult *results, size_t max_results,
                            GV_DistanceType distance_type,
                            const char *filter_key, const char *filter_value) {
    if (!index || !query || !results || max_results == 0 || radius < 0.0f) return -1;

    GV_IVFFlatIndex *idx = (GV_IVFFlatIndex *)index;

    if (!idx->trained) return -1;
    if (query->dimension != idx->dimension) return -1;

    /* Cap nprobe to nlist */
    size_t nprobe = idx->config.nprobe;
    if (nprobe > idx->config.nlist) nprobe = idx->config.nlist;

    /* Find nprobe closest centroids */
    GV_IVFFlatHeapItem *centroid_heap = (GV_IVFFlatHeapItem *)malloc(
        nprobe * sizeof(GV_IVFFlatHeapItem));
    if (!centroid_heap) return -1;

    size_t heap_size = 0;

    for (size_t i = 0; i < idx->config.nlist; i++) {
        const float *centroid = idx->centroids + i * idx->dimension;
        float dist = 0.0f;
        for (size_t d = 0; d < idx->dimension; d++) {
            float diff = query->data[d] - centroid[d];
            dist += diff * diff;
        }

        gv_ivfflat_heap_push(centroid_heap, &heap_size, nprobe, dist, i, NULL);
    }

    /* Extract probe lists */
    size_t *probe_lists = (size_t *)malloc(nprobe * sizeof(size_t));
    if (!probe_lists) {
        free(centroid_heap);
        return -1;
    }

    for (size_t i = nprobe; i > 0; i--) {
        probe_lists[i - 1] = centroid_heap[0].id;
        centroid_heap[0] = centroid_heap[heap_size - 1];
        heap_size--;
        if (heap_size > 0) {
            gv_ivfflat_heap_sift_down(centroid_heap, heap_size, 0);
        }
    }
    free(centroid_heap);

    /* Scan entries in selected lists and collect within radius */
    size_t found = 0;

    for (size_t i = 0; i < idx->config.nprobe && found < max_results; i++) {
        size_t list_idx = probe_lists[i];
        GV_IVFFlatEntry *entry = idx->lists[list_idx];

        while (entry && found < max_results) {
            if (!entry->deleted) {
                /* Apply metadata filter */
                if (gv_ivfflat_metadata_match(entry->vector->metadata, filter_key, filter_value)) {
                    /* Compute distance */
                    float dist = gv_distance(query, entry->vector, distance_type);

                    if (dist >= 0.0f && dist <= radius) {
                        /* Copy vector with metadata */
                        GV_Vector *copy = gv_vector_create_from_data(entry->vector->dimension,
                                                                      entry->vector->data);
                        if (copy) {
                            /* Copy metadata */
                            GV_Metadata *meta = entry->vector->metadata;
                            while (meta) {
                                if (meta->key && meta->value) {
                                    gv_vector_set_metadata(copy, meta->key, meta->value);
                                }
                                meta = meta->next;
                            }
                        }

                        results[found].vector = copy;
                        results[found].distance = dist;
                        results[found].is_sparse = 0;
                        results[found].sparse_vector = NULL;
                        results[found].id = entry->id;
                        found++;
                    }
                }
            }
            entry = entry->next;
        }
    }

    free(probe_lists);
    return (int)found;
}

int gv_ivfflat_is_trained(const void *index) {
    if (!index) return 0;
    const GV_IVFFlatIndex *idx = (const GV_IVFFlatIndex *)index;
    return idx->trained;
}

void gv_ivfflat_destroy(void *index) {
    if (!index) return;

    GV_IVFFlatIndex *idx = (GV_IVFFlatIndex *)index;

    /* Free all entries in all lists */
    if (idx->lists) {
        for (size_t i = 0; i < idx->config.nlist; i++) {
            GV_IVFFlatEntry *entry = idx->lists[i];
            while (entry) {
                GV_IVFFlatEntry *next = entry->next;
                if (entry->vector) {
                    gv_vector_destroy(entry->vector);
                }
                free(entry);
                entry = next;
            }
        }
        free(idx->lists);
    }

    free(idx->centroids);
    free(idx->list_sizes);
    free(idx);
}

size_t gv_ivfflat_count(const void *index) {
    if (!index) return 0;
    const GV_IVFFlatIndex *idx = (const GV_IVFFlatIndex *)index;

    size_t count = 0;
    for (size_t i = 0; i < idx->config.nlist; i++) {
        GV_IVFFlatEntry *entry = idx->lists[i];
        while (entry) {
            if (!entry->deleted) {
                count++;
            }
            entry = entry->next;
        }
    }
    return count;
}

int gv_ivfflat_delete(void *index, size_t entry_index) {
    if (!index) return -1;

    GV_IVFFlatIndex *idx = (GV_IVFFlatIndex *)index;

    /* Find entry by global ID */
    for (size_t i = 0; i < idx->config.nlist; i++) {
        GV_IVFFlatEntry *entry = idx->lists[i];
        while (entry) {
            if (entry->id == entry_index) {
                if (entry->deleted) return -1; /* Already deleted */
                entry->deleted = 1;
                return 0;
            }
            entry = entry->next;
        }
    }

    return -1; /* Not found */
}

int gv_ivfflat_update(void *index, size_t entry_index, const float *new_data, size_t dimension) {
    if (!index || !new_data) return -1;

    GV_IVFFlatIndex *idx = (GV_IVFFlatIndex *)index;

    if (dimension != idx->dimension) return -1;

    /* Find entry by global ID */
    for (size_t i = 0; i < idx->config.nlist; i++) {
        GV_IVFFlatEntry *entry = idx->lists[i];
        while (entry) {
            if (entry->id == entry_index) {
                if (entry->deleted) return -1; /* Cannot update deleted entry */

                /* Update vector data */
                if (entry->vector && entry->vector->data) {
                    memcpy(entry->vector->data, new_data, dimension * sizeof(float));
                    return 0;
                }
                return -1;
            }
            entry = entry->next;
        }
    }

    return -1; /* Not found */
}

int gv_ivfflat_save(const void *index, FILE *out, uint32_t version) {
    if (!index || !out) return -1;

    const GV_IVFFlatIndex *idx = (const GV_IVFFlatIndex *)index;
    (void)version;

    /* Write dimension and config */
    if (gv_ivfflat_write_u32(out, (uint32_t)idx->dimension) != 0) return -1;
    if (gv_ivfflat_write_u32(out, (uint32_t)idx->config.nlist) != 0) return -1;
    if (gv_ivfflat_write_u32(out, (uint32_t)idx->config.nprobe) != 0) return -1;
    if (gv_ivfflat_write_u32(out, (uint32_t)idx->config.train_iters) != 0) return -1;
    if (gv_ivfflat_write_u32(out, (uint32_t)idx->config.use_cosine) != 0) return -1;
    if (gv_ivfflat_write_u32(out, (uint32_t)idx->trained) != 0) return -1;
    if (gv_ivfflat_write_u32(out, (uint32_t)idx->next_id) != 0) return -1;

    /* Write centroids if trained */
    if (idx->trained) {
        size_t centroid_floats = idx->config.nlist * idx->dimension;
        if (fwrite(idx->centroids, sizeof(float), centroid_floats, out) != centroid_floats) {
            return -1;
        }
    }

    /* Write number of lists */
    if (gv_ivfflat_write_u32(out, (uint32_t)idx->config.nlist) != 0) return -1;

    /* Write each list */
    for (size_t i = 0; i < idx->config.nlist; i++) {
        /* Count entries in this list */
        uint32_t list_count = 0;
        GV_IVFFlatEntry *entry = idx->lists[i];
        while (entry) {
            list_count++;
            entry = entry->next;
        }

        if (gv_ivfflat_write_u32(out, list_count) != 0) return -1;

        /* Write each entry */
        entry = idx->lists[i];
        while (entry) {
            /* Write ID and deleted flag */
            if (gv_ivfflat_write_u32(out, (uint32_t)entry->id) != 0) return -1;
            if (gv_ivfflat_write_u32(out, (uint32_t)entry->deleted) != 0) return -1;

            /* Write vector data */
            if (fwrite(entry->vector->data, sizeof(float), idx->dimension, out) != idx->dimension) {
                return -1;
            }

            /* Write metadata */
            uint32_t meta_count = 0;
            GV_Metadata *meta = entry->vector->metadata;
            while (meta) {
                meta_count++;
                meta = meta->next;
            }

            if (gv_ivfflat_write_u32(out, meta_count) != 0) return -1;

            meta = entry->vector->metadata;
            while (meta) {
                uint32_t klen = meta->key ? (uint32_t)strlen(meta->key) : 0;
                uint32_t vlen = meta->value ? (uint32_t)strlen(meta->value) : 0;

                if (gv_ivfflat_write_str(out, meta->key ? meta->key : "", klen) != 0) return -1;
                if (gv_ivfflat_write_str(out, meta->value ? meta->value : "", vlen) != 0) return -1;

                meta = meta->next;
            }

            entry = entry->next;
        }
    }

    return 0;
}

int gv_ivfflat_load(void **index_ptr, FILE *in, size_t dimension, uint32_t version) {
    if (!index_ptr || !in) return -1;
    (void)version;

    uint32_t file_dim = 0, nlist = 0, nprobe = 0, train_iters = 0, use_cosine = 0;
    uint32_t trained = 0, next_id = 0;

    if (gv_ivfflat_read_u32(in, &file_dim) != 0) return -1;
    if (gv_ivfflat_read_u32(in, &nlist) != 0) return -1;
    if (gv_ivfflat_read_u32(in, &nprobe) != 0) return -1;
    if (gv_ivfflat_read_u32(in, &train_iters) != 0) return -1;
    if (gv_ivfflat_read_u32(in, &use_cosine) != 0) return -1;
    if (gv_ivfflat_read_u32(in, &trained) != 0) return -1;
    if (gv_ivfflat_read_u32(in, &next_id) != 0) return -1;

    if (dimension != 0 && dimension != (size_t)file_dim) return -1;

    /* Create index with loaded config */
    GV_IVFFlatConfig config = {
        .nlist = nlist,
        .nprobe = nprobe,
        .train_iters = train_iters,
        .use_cosine = (int)use_cosine
    };

    void *index = gv_ivfflat_create((size_t)file_dim, &config);
    if (!index) return -1;

    GV_IVFFlatIndex *idx = (GV_IVFFlatIndex *)index;
    idx->trained = (int)trained;
    idx->next_id = (size_t)next_id;

    /* Load centroids if trained */
    if (trained) {
        size_t centroid_floats = idx->config.nlist * idx->dimension;
        if (fread(idx->centroids, sizeof(float), centroid_floats, in) != centroid_floats) {
            gv_ivfflat_destroy(index);
            return -1;
        }
    }

    /* Read number of lists */
    uint32_t num_lists = 0;
    if (gv_ivfflat_read_u32(in, &num_lists) != 0) {
        gv_ivfflat_destroy(index);
        return -1;
    }

    if (num_lists != nlist) {
        gv_ivfflat_destroy(index);
        return -1;
    }

    /* Load each list */
    for (size_t i = 0; i < nlist; i++) {
        uint32_t list_count = 0;
        if (gv_ivfflat_read_u32(in, &list_count) != 0) {
            gv_ivfflat_destroy(index);
            return -1;
        }

        /* Load entries in reverse order to maintain original order */
        GV_IVFFlatEntry **tail = &idx->lists[i];

        for (uint32_t j = 0; j < list_count; j++) {
            uint32_t entry_id = 0, deleted = 0;

            if (gv_ivfflat_read_u32(in, &entry_id) != 0) {
                gv_ivfflat_destroy(index);
                return -1;
            }
            if (gv_ivfflat_read_u32(in, &deleted) != 0) {
                gv_ivfflat_destroy(index);
                return -1;
            }

            /* Read vector data */
            float *data = (float *)malloc(idx->dimension * sizeof(float));
            if (!data) {
                gv_ivfflat_destroy(index);
                return -1;
            }

            if (fread(data, sizeof(float), idx->dimension, in) != idx->dimension) {
                free(data);
                gv_ivfflat_destroy(index);
                return -1;
            }

            /* Create vector */
            GV_Vector *vec = gv_vector_create_from_data(idx->dimension, data);
            free(data);

            if (!vec) {
                gv_ivfflat_destroy(index);
                return -1;
            }

            /* Read metadata */
            uint32_t meta_count = 0;
            if (gv_ivfflat_read_u32(in, &meta_count) != 0) {
                gv_vector_destroy(vec);
                gv_ivfflat_destroy(index);
                return -1;
            }

            for (uint32_t m = 0; m < meta_count; m++) {
                uint32_t klen = 0, vlen = 0;
                char *key = NULL, *value = NULL;

                if (gv_ivfflat_read_u32(in, &klen) != 0) {
                    gv_vector_destroy(vec);
                    gv_ivfflat_destroy(index);
                    return -1;
                }
                if (gv_ivfflat_read_str(in, &key, klen) != 0) {
                    gv_vector_destroy(vec);
                    gv_ivfflat_destroy(index);
                    return -1;
                }
                if (gv_ivfflat_read_u32(in, &vlen) != 0) {
                    free(key);
                    gv_vector_destroy(vec);
                    gv_ivfflat_destroy(index);
                    return -1;
                }
                if (gv_ivfflat_read_str(in, &value, vlen) != 0) {
                    free(key);
                    gv_vector_destroy(vec);
                    gv_ivfflat_destroy(index);
                    return -1;
                }

                if (gv_vector_set_metadata(vec, key, value) != 0) {
                    free(key);
                    free(value);
                    gv_vector_destroy(vec);
                    gv_ivfflat_destroy(index);
                    return -1;
                }

                free(key);
                free(value);
            }

            /* Create entry */
            GV_IVFFlatEntry *entry = (GV_IVFFlatEntry *)malloc(sizeof(GV_IVFFlatEntry));
            if (!entry) {
                gv_vector_destroy(vec);
                gv_ivfflat_destroy(index);
                return -1;
            }

            entry->vector = vec;
            entry->id = (size_t)entry_id;
            entry->deleted = (int)deleted;
            entry->next = NULL;

            /* Append to maintain order */
            *tail = entry;
            tail = &entry->next;

            idx->list_sizes[i]++;
            idx->total_count++;
        }
    }

    *index_ptr = index;
    return 0;
}
