#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gigavector/gv_embedded.h"
#include "gigavector/gv_distance.h"

/* ------------------------------------------------------------------ */
/* Magic header for the binary save format                            */
/* ------------------------------------------------------------------ */
#define GV_EMBEDDED_MAGIC "GVEM"
#define GV_EMBEDDED_MAGIC_LEN 4
#define GV_EMBEDDED_FILE_VERSION 1

/* ------------------------------------------------------------------ */
/* Internal: HNSW simplified graph (single-level, no locks)           */
/* ------------------------------------------------------------------ */
#define GV_EMBEDDED_HNSW_M_DEFAULT 16

typedef struct {
    size_t *neighbors;     /**< Neighbor indices (up to M entries). */
    size_t neighbor_count;
} GV_EmbeddedHNSWNode;

typedef struct {
    GV_EmbeddedHNSWNode *nodes;
    size_t capacity;
    size_t M;             /**< Max neighbors per node. */
    size_t ef_construction;
} GV_EmbeddedHNSW;

/* ------------------------------------------------------------------ */
/* Internal: LSH tables                                               */
/* ------------------------------------------------------------------ */
#define GV_EMBEDDED_LSH_TABLES_DEFAULT 8
#define GV_EMBEDDED_LSH_BITS_DEFAULT   12

typedef struct {
    size_t *indices;
    size_t count;
    size_t capacity;
} GV_EmbeddedLSHBucket;

typedef struct {
    float **hyperplanes;         /**< [num_tables * num_bits][dimension] */
    GV_EmbeddedLSHBucket **tables; /**< [num_tables][num_buckets] */
    size_t num_tables;
    size_t num_bits;
    size_t num_buckets;
    size_t dimension;
} GV_EmbeddedLSH;

/* ------------------------------------------------------------------ */
/* Internal: Quantization parameters                                  */
/* ------------------------------------------------------------------ */
typedef struct {
    float *min_vals;   /**< Per-dimension minimum (dimension floats). */
    float *max_vals;   /**< Per-dimension maximum (dimension floats). */
    uint8_t *data;     /**< Quantized vector storage. */
    size_t bytes_per_vector;
} GV_EmbeddedQuant;

/* ------------------------------------------------------------------ */
/* Internal: Max-heap for top-k selection                             */
/* ------------------------------------------------------------------ */
typedef struct {
    float dist;
    size_t idx;
} GV_EmbeddedHeapItem;

/* ------------------------------------------------------------------ */
/* The main embedded database structure                               */
/* ------------------------------------------------------------------ */
struct GV_EmbeddedDB {
    /* Configuration snapshot */
    size_t dimension;
    int index_type;
    size_t max_vectors;
    size_t memory_limit_mb;
    int quantize;

    /* Vector storage (contiguous float array) */
    float *vectors;          /**< [capacity * dimension] floats. */
    size_t count;            /**< Next insertion slot (total slots used). */
    size_t capacity;         /**< Allocated slot count. */

    /* Deleted bitmap: 1 bit per slot */
    uint8_t *deleted;
    size_t deleted_bytes;

    /* Memory tracking */
    size_t memory_used;

    /* Quantization (optional) */
    GV_EmbeddedQuant *quant;

    /* Index-specific data */
    GV_EmbeddedHNSW *hnsw;
    GV_EmbeddedLSH  *lsh;
};

/* ================================================================== */
/* Helper: memory tracking                                            */
/* ================================================================== */

static void *emb_tracked_malloc(GV_EmbeddedDB *db, size_t size) {
    if (db->memory_limit_mb > 0) {
        size_t limit = db->memory_limit_mb * 1024UL * 1024UL;
        if (db->memory_used + size > limit) {
            return NULL;
        }
    }
    void *ptr = malloc(size);
    if (ptr) {
        db->memory_used += size;
    }
    return ptr;
}

static void *emb_tracked_calloc(GV_EmbeddedDB *db, size_t nmemb, size_t size) {
    size_t total = nmemb * size;
    if (db->memory_limit_mb > 0) {
        size_t limit = db->memory_limit_mb * 1024UL * 1024UL;
        if (db->memory_used + total > limit) {
            return NULL;
        }
    }
    void *ptr = calloc(nmemb, size);
    if (ptr) {
        db->memory_used += total;
    }
    return ptr;
}

static void *emb_tracked_realloc(GV_EmbeddedDB *db, void *ptr, size_t old_size, size_t new_size) {
    if (new_size > old_size && db->memory_limit_mb > 0) {
        size_t limit = db->memory_limit_mb * 1024UL * 1024UL;
        if (db->memory_used + (new_size - old_size) > limit) {
            return NULL;
        }
    }
    void *new_ptr = realloc(ptr, new_size);
    if (new_ptr) {
        if (new_size > old_size) {
            db->memory_used += (new_size - old_size);
        } else {
            db->memory_used -= (old_size - new_size);
        }
    }
    return new_ptr;
}

static void emb_tracked_free(GV_EmbeddedDB *db, void *ptr, size_t size) {
    if (ptr) {
        free(ptr);
        if (db->memory_used >= size) {
            db->memory_used -= size;
        } else {
            db->memory_used = 0;
        }
    }
}

/* ================================================================== */
/* Helper: deleted bitmap                                             */
/* ================================================================== */

static int emb_is_deleted(const GV_EmbeddedDB *db, size_t idx) {
    if (idx >= db->count) return 1;
    return (db->deleted[idx / 8] >> (idx % 8)) & 1;
}

static void emb_mark_deleted(GV_EmbeddedDB *db, size_t idx) {
    db->deleted[idx / 8] |= (uint8_t)(1U << (idx % 8));
}

static void emb_clear_deleted(GV_EmbeddedDB *db, size_t idx) {
    db->deleted[idx / 8] &= (uint8_t)~(1U << (idx % 8));
}

/* ================================================================== */
/* Helper: distance computation (raw float arrays)                    */
/* ================================================================== */

static float emb_distance_euclidean(const float *a, const float *b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sqrtf(sum);
}

static float emb_distance_cosine(const float *a, const float *b, size_t dim) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    na = sqrtf(na);
    nb = sqrtf(nb);
    if (na == 0.0f || nb == 0.0f) return 0.0f;
    return 1.0f - (dot / (na * nb));
}

static float emb_distance_dot(const float *a, const float *b, size_t dim) {
    float dot = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
    }
    return -dot;
}

static float emb_distance_manhattan(const float *a, const float *b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += (d < 0.0f) ? -d : d;
    }
    return sum;
}

static float emb_compute_distance(const float *a, const float *b, size_t dim, int dist_type) {
    switch (dist_type) {
        case GV_DISTANCE_EUCLIDEAN:  return emb_distance_euclidean(a, b, dim);
        case GV_DISTANCE_COSINE:     return emb_distance_cosine(a, b, dim);
        case GV_DISTANCE_DOT_PRODUCT:return emb_distance_dot(a, b, dim);
        case GV_DISTANCE_MANHATTAN:  return emb_distance_manhattan(a, b, dim);
        default:                     return emb_distance_euclidean(a, b, dim);
    }
}

/* ================================================================== */
/* Helper: max-heap for top-k                                         */
/* ================================================================== */

static void emb_heap_sift_down(GV_EmbeddedHeapItem *heap, size_t size, size_t i) {
    while (1) {
        size_t l = 2 * i + 1;
        size_t r = l + 1;
        size_t largest = i;
        if (l < size && heap[l].dist > heap[largest].dist) largest = l;
        if (r < size && heap[r].dist > heap[largest].dist) largest = r;
        if (largest == i) break;
        GV_EmbeddedHeapItem tmp = heap[i];
        heap[i] = heap[largest];
        heap[largest] = tmp;
        i = largest;
    }
}

static void emb_heap_push(GV_EmbeddedHeapItem *heap, size_t *size, size_t cap,
                           float dist, size_t idx) {
    if (*size < cap) {
        heap[*size].dist = dist;
        heap[*size].idx  = idx;
        (*size)++;
        /* Sift up */
        size_t i = *size - 1;
        while (i > 0) {
            size_t parent = (i - 1) / 2;
            if (heap[i].dist > heap[parent].dist) {
                GV_EmbeddedHeapItem tmp = heap[i];
                heap[i] = heap[parent];
                heap[parent] = tmp;
                i = parent;
            } else {
                break;
            }
        }
    } else if (dist < heap[0].dist) {
        heap[0].dist = dist;
        heap[0].idx  = idx;
        emb_heap_sift_down(heap, *size, 0);
    }
}

/* ================================================================== */
/* Quantization helpers                                               */
/* ================================================================== */

static size_t emb_quant_bytes_per_vector(size_t dimension, int bits) {
    return (dimension * (size_t)bits + 7) / 8;
}

static GV_EmbeddedQuant *emb_quant_create(GV_EmbeddedDB *db, size_t capacity) {
    GV_EmbeddedQuant *q = (GV_EmbeddedQuant *)emb_tracked_calloc(db, 1, sizeof(GV_EmbeddedQuant));
    if (!q) return NULL;

    q->min_vals = (float *)emb_tracked_calloc(db, db->dimension, sizeof(float));
    q->max_vals = (float *)emb_tracked_calloc(db, db->dimension, sizeof(float));
    if (!q->min_vals || !q->max_vals) {
        emb_tracked_free(db, q->min_vals, db->dimension * sizeof(float));
        emb_tracked_free(db, q->max_vals, db->dimension * sizeof(float));
        emb_tracked_free(db, q, sizeof(GV_EmbeddedQuant));
        return NULL;
    }

    /* Initialize min/max to neutral values */
    for (size_t d = 0; d < db->dimension; ++d) {
        q->min_vals[d] =  FLT_MAX;
        q->max_vals[d] = -FLT_MAX;
    }

    q->bytes_per_vector = emb_quant_bytes_per_vector(db->dimension, db->quantize);
    size_t data_size = capacity * q->bytes_per_vector;
    if (data_size > 0) {
        q->data = (uint8_t *)emb_tracked_calloc(db, data_size, 1);
        if (!q->data) {
            emb_tracked_free(db, q->min_vals, db->dimension * sizeof(float));
            emb_tracked_free(db, q->max_vals, db->dimension * sizeof(float));
            emb_tracked_free(db, q, sizeof(GV_EmbeddedQuant));
            return NULL;
        }
    }
    return q;
}

static void emb_quant_destroy(GV_EmbeddedDB *db, GV_EmbeddedQuant *q) {
    if (!q) return;
    emb_tracked_free(db, q->data, db->capacity * q->bytes_per_vector);
    emb_tracked_free(db, q->min_vals, db->dimension * sizeof(float));
    emb_tracked_free(db, q->max_vals, db->dimension * sizeof(float));
    emb_tracked_free(db, q, sizeof(GV_EmbeddedQuant));
}

static void emb_quant_update_minmax(GV_EmbeddedQuant *q, const float *vec, size_t dim) {
    for (size_t d = 0; d < dim; ++d) {
        if (vec[d] < q->min_vals[d]) q->min_vals[d] = vec[d];
        if (vec[d] > q->max_vals[d]) q->max_vals[d] = vec[d];
    }
}

static void emb_quant_encode(const GV_EmbeddedQuant *q, const float *vec,
                              size_t dim, int bits, uint8_t *out) {
    int max_val = (1 << bits) - 1;
    memset(out, 0, q->bytes_per_vector);

    size_t bit_offset = 0;
    for (size_t d = 0; d < dim; ++d) {
        float range = q->max_vals[d] - q->min_vals[d];
        float normalized;
        if (range < 1e-9f) {
            normalized = 0.0f;
        } else {
            normalized = (vec[d] - q->min_vals[d]) / range;
        }
        if (normalized < 0.0f) normalized = 0.0f;
        if (normalized > 1.0f) normalized = 1.0f;

        uint8_t quantized = (uint8_t)(normalized * max_val + 0.5f);

        /* Pack bits into output */
        for (int b = 0; b < bits; ++b) {
            if (quantized & (1U << b)) {
                out[bit_offset / 8] |= (uint8_t)(1U << (bit_offset % 8));
            }
            bit_offset++;
        }
    }
}

static void emb_quant_decode(const GV_EmbeddedQuant *q, const uint8_t *encoded,
                              size_t dim, int bits, float *out) {
    int max_val = (1 << bits) - 1;
    size_t bit_offset = 0;

    for (size_t d = 0; d < dim; ++d) {
        uint8_t quantized = 0;
        for (int b = 0; b < bits; ++b) {
            if (encoded[bit_offset / 8] & (1U << (bit_offset % 8))) {
                quantized |= (uint8_t)(1U << b);
            }
            bit_offset++;
        }

        float range = q->max_vals[d] - q->min_vals[d];
        if (range < 1e-9f) {
            out[d] = q->min_vals[d];
        } else {
            out[d] = q->min_vals[d] + ((float)quantized / (float)max_val) * range;
        }
    }
}

/* ================================================================== */
/* HNSW (simplified single-level) helpers                             */
/* ================================================================== */

static GV_EmbeddedHNSW *emb_hnsw_create(GV_EmbeddedDB *db, size_t capacity) {
    GV_EmbeddedHNSW *h = (GV_EmbeddedHNSW *)emb_tracked_calloc(db, 1, sizeof(GV_EmbeddedHNSW));
    if (!h) return NULL;

    h->M = GV_EMBEDDED_HNSW_M_DEFAULT;
    h->ef_construction = 64;
    h->capacity = capacity;

    h->nodes = (GV_EmbeddedHNSWNode *)emb_tracked_calloc(db, capacity, sizeof(GV_EmbeddedHNSWNode));
    if (!h->nodes) {
        emb_tracked_free(db, h, sizeof(GV_EmbeddedHNSW));
        return NULL;
    }

    return h;
}

static void emb_hnsw_destroy(GV_EmbeddedDB *db, GV_EmbeddedHNSW *h) {
    if (!h) return;
    for (size_t i = 0; i < h->capacity; ++i) {
        if (h->nodes[i].neighbors) {
            emb_tracked_free(db, h->nodes[i].neighbors, h->M * sizeof(size_t));
        }
    }
    emb_tracked_free(db, h->nodes, h->capacity * sizeof(GV_EmbeddedHNSWNode));
    emb_tracked_free(db, h, sizeof(GV_EmbeddedHNSW));
}

static int emb_hnsw_ensure_capacity(GV_EmbeddedDB *db, GV_EmbeddedHNSW *h, size_t needed) {
    if (needed <= h->capacity) return 0;

    size_t new_cap = h->capacity == 0 ? 64 : h->capacity;
    while (new_cap < needed) new_cap *= 2;

    GV_EmbeddedHNSWNode *new_nodes = (GV_EmbeddedHNSWNode *)emb_tracked_realloc(
        db, h->nodes, h->capacity * sizeof(GV_EmbeddedHNSWNode),
        new_cap * sizeof(GV_EmbeddedHNSWNode));
    if (!new_nodes) return -1;

    memset(&new_nodes[h->capacity], 0, (new_cap - h->capacity) * sizeof(GV_EmbeddedHNSWNode));
    h->nodes = new_nodes;
    h->capacity = new_cap;
    return 0;
}

static int emb_hnsw_insert(GV_EmbeddedDB *db, size_t vec_idx) {
    GV_EmbeddedHNSW *h = db->hnsw;

    if (emb_hnsw_ensure_capacity(db, h, vec_idx + 1) != 0) return -1;

    GV_EmbeddedHNSWNode *node = &h->nodes[vec_idx];
    if (!node->neighbors) {
        node->neighbors = (size_t *)emb_tracked_calloc(db, h->M, sizeof(size_t));
        if (!node->neighbors) return -1;
    }
    node->neighbor_count = 0;

    const float *new_vec = db->vectors + vec_idx * db->dimension;

    /* Find ef_construction nearest active nodes via brute-force scan
     * (single-level HNSW -- we keep it simple for embedded mode). */
    size_t ef = h->ef_construction;
    if (ef > db->count) ef = db->count;

    GV_EmbeddedHeapItem *candidates = NULL;
    size_t cand_count = 0;

    if (ef > 0) {
        candidates = (GV_EmbeddedHeapItem *)malloc(ef * sizeof(GV_EmbeddedHeapItem));
        if (!candidates) return -1;

        for (size_t i = 0; i < db->count; ++i) {
            if (i == vec_idx) continue;
            if (emb_is_deleted(db, i)) continue;

            float dist = emb_distance_euclidean(new_vec, db->vectors + i * db->dimension, db->dimension);
            emb_heap_push(candidates, &cand_count, ef, dist, i);
        }
    }

    /* Connect to up to M nearest */
    size_t connect_count = cand_count < h->M ? cand_count : h->M;

    /* Sort candidates (extract from max-heap) for deterministic selection */
    if (cand_count > 0) {
        /* Simple selection of smallest-distance items from the max-heap */
        GV_EmbeddedHeapItem *sorted = (GV_EmbeddedHeapItem *)malloc(cand_count * sizeof(GV_EmbeddedHeapItem));
        if (sorted) {
            memcpy(sorted, candidates, cand_count * sizeof(GV_EmbeddedHeapItem));
            /* Sort ascending by distance */
            for (size_t i = 0; i < cand_count; ++i) {
                for (size_t j = i + 1; j < cand_count; ++j) {
                    if (sorted[j].dist < sorted[i].dist) {
                        GV_EmbeddedHeapItem tmp = sorted[i];
                        sorted[i] = sorted[j];
                        sorted[j] = tmp;
                    }
                }
            }

            for (size_t c = 0; c < connect_count; ++c) {
                size_t neighbor_idx = sorted[c].idx;
                node->neighbors[node->neighbor_count++] = neighbor_idx;

                /* Bidirectional edge: add reverse connection if capacity allows */
                GV_EmbeddedHNSWNode *nb = &h->nodes[neighbor_idx];
                if (!nb->neighbors) {
                    nb->neighbors = (size_t *)emb_tracked_calloc(db, h->M, sizeof(size_t));
                }
                if (nb->neighbors && nb->neighbor_count < h->M) {
                    nb->neighbors[nb->neighbor_count++] = vec_idx;
                }
            }

            free(sorted);
        }
    }

    free(candidates);
    return 0;
}

static int emb_hnsw_search(const GV_EmbeddedDB *db, const float *query, size_t k,
                            int dist_type, GV_EmbeddedHeapItem *heap, size_t *heap_size) {
    GV_EmbeddedHNSW *h = db->hnsw;

    /* Find an entry point: first non-deleted node */
    size_t entry = (size_t)-1;
    for (size_t i = 0; i < db->count; ++i) {
        if (!emb_is_deleted(db, i)) { entry = i; break; }
    }
    if (entry == (size_t)-1) return 0;

    /* Greedy walk + expansion (best-first search on the graph) */
    size_t ef = k < 64 ? 64 : k * 2;
    if (ef > db->count) ef = db->count;

    uint8_t *visited = (uint8_t *)calloc((db->count + 7) / 8, 1);
    if (!visited) return -1;

    /* Seed with entry point */
    GV_EmbeddedHeapItem *candidates = (GV_EmbeddedHeapItem *)malloc(ef * sizeof(GV_EmbeddedHeapItem));
    if (!candidates) { free(visited); return -1; }
    size_t cand_size = 0;

    float entry_dist = emb_compute_distance(query, db->vectors + entry * db->dimension,
                                             db->dimension, dist_type);
    emb_heap_push(heap, heap_size, k, entry_dist, entry);
    candidates[cand_size].dist = entry_dist;
    candidates[cand_size].idx  = entry;
    cand_size++;
    visited[entry / 8] |= (uint8_t)(1U << (entry % 8));

    /* Best-first expansion */
    size_t cursor = 0;
    while (cursor < cand_size) {
        size_t cur_idx = candidates[cursor].idx;
        cursor++;

        if (cur_idx >= h->capacity) continue;
        GV_EmbeddedHNSWNode *node = &h->nodes[cur_idx];
        if (!node->neighbors) continue;

        for (size_t n = 0; n < node->neighbor_count; ++n) {
            size_t nb_idx = node->neighbors[n];
            if (nb_idx >= db->count) continue;
            if (visited[nb_idx / 8] & (1U << (nb_idx % 8))) continue;
            visited[nb_idx / 8] |= (uint8_t)(1U << (nb_idx % 8));

            if (emb_is_deleted(db, nb_idx)) continue;

            float dist = emb_compute_distance(query, db->vectors + nb_idx * db->dimension,
                                               db->dimension, dist_type);
            emb_heap_push(heap, heap_size, k, dist, nb_idx);

            /* Add to expansion list */
            if (cand_size < ef) {
                candidates[cand_size].dist = dist;
                candidates[cand_size].idx  = nb_idx;
                cand_size++;
            }
        }
    }

    free(candidates);
    free(visited);
    return 0;
}

/* ================================================================== */
/* LSH helpers                                                        */
/* ================================================================== */

static uint64_t emb_xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static float emb_gaussian_random(uint64_t *state) {
    float u1 = (float)emb_xorshift64(state) / (float)UINT64_MAX;
    float u2 = (float)emb_xorshift64(state) / (float)UINT64_MAX;
    if (u1 < 1e-9f) u1 = 1e-9f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
}

static GV_EmbeddedLSH *emb_lsh_create(GV_EmbeddedDB *db) {
    GV_EmbeddedLSH *l = (GV_EmbeddedLSH *)emb_tracked_calloc(db, 1, sizeof(GV_EmbeddedLSH));
    if (!l) return NULL;

    l->num_tables = GV_EMBEDDED_LSH_TABLES_DEFAULT;
    l->num_bits   = GV_EMBEDDED_LSH_BITS_DEFAULT;
    l->dimension  = db->dimension;
    l->num_buckets = 1UL << l->num_bits;
    if (l->num_buckets > 4096) l->num_buckets = 4096;

    /* Allocate hyperplanes */
    size_t total_planes = l->num_tables * l->num_bits;
    l->hyperplanes = (float **)emb_tracked_calloc(db, total_planes, sizeof(float *));
    if (!l->hyperplanes) {
        emb_tracked_free(db, l, sizeof(GV_EmbeddedLSH));
        return NULL;
    }

    uint64_t rng_state = 42;
    for (size_t i = 0; i < total_planes; ++i) {
        l->hyperplanes[i] = (float *)emb_tracked_malloc(db, db->dimension * sizeof(float));
        if (!l->hyperplanes[i]) {
            for (size_t j = 0; j < i; ++j) {
                emb_tracked_free(db, l->hyperplanes[j], db->dimension * sizeof(float));
            }
            emb_tracked_free(db, l->hyperplanes, total_planes * sizeof(float *));
            emb_tracked_free(db, l, sizeof(GV_EmbeddedLSH));
            return NULL;
        }
        for (size_t d = 0; d < db->dimension; ++d) {
            l->hyperplanes[i][d] = emb_gaussian_random(&rng_state);
        }
    }

    /* Allocate tables */
    l->tables = (GV_EmbeddedLSHBucket **)emb_tracked_calloc(db, l->num_tables, sizeof(GV_EmbeddedLSHBucket *));
    if (!l->tables) {
        for (size_t i = 0; i < total_planes; ++i) {
            emb_tracked_free(db, l->hyperplanes[i], db->dimension * sizeof(float));
        }
        emb_tracked_free(db, l->hyperplanes, total_planes * sizeof(float *));
        emb_tracked_free(db, l, sizeof(GV_EmbeddedLSH));
        return NULL;
    }

    for (size_t t = 0; t < l->num_tables; ++t) {
        l->tables[t] = (GV_EmbeddedLSHBucket *)emb_tracked_calloc(db, l->num_buckets, sizeof(GV_EmbeddedLSHBucket));
        if (!l->tables[t]) {
            for (size_t j = 0; j < t; ++j) {
                for (size_t b = 0; b < l->num_buckets; ++b) {
                    emb_tracked_free(db, l->tables[j][b].indices,
                                     l->tables[j][b].capacity * sizeof(size_t));
                }
                emb_tracked_free(db, l->tables[j], l->num_buckets * sizeof(GV_EmbeddedLSHBucket));
            }
            emb_tracked_free(db, l->tables, l->num_tables * sizeof(GV_EmbeddedLSHBucket *));
            for (size_t i = 0; i < total_planes; ++i) {
                emb_tracked_free(db, l->hyperplanes[i], db->dimension * sizeof(float));
            }
            emb_tracked_free(db, l->hyperplanes, total_planes * sizeof(float *));
            emb_tracked_free(db, l, sizeof(GV_EmbeddedLSH));
            return NULL;
        }
    }

    return l;
}

static void emb_lsh_destroy(GV_EmbeddedDB *db, GV_EmbeddedLSH *l) {
    if (!l) return;

    if (l->tables) {
        for (size_t t = 0; t < l->num_tables; ++t) {
            if (l->tables[t]) {
                for (size_t b = 0; b < l->num_buckets; ++b) {
                    emb_tracked_free(db, l->tables[t][b].indices,
                                     l->tables[t][b].capacity * sizeof(size_t));
                }
                emb_tracked_free(db, l->tables[t], l->num_buckets * sizeof(GV_EmbeddedLSHBucket));
            }
        }
        emb_tracked_free(db, l->tables, l->num_tables * sizeof(GV_EmbeddedLSHBucket *));
    }

    if (l->hyperplanes) {
        size_t total_planes = l->num_tables * l->num_bits;
        for (size_t i = 0; i < total_planes; ++i) {
            emb_tracked_free(db, l->hyperplanes[i], l->dimension * sizeof(float));
        }
        emb_tracked_free(db, l->hyperplanes, total_planes * sizeof(float *));
    }

    emb_tracked_free(db, l, sizeof(GV_EmbeddedLSH));
}

static uint32_t emb_lsh_hash(const GV_EmbeddedLSH *l, const float *data, size_t table_idx) {
    uint32_t hash = 0;
    size_t base = table_idx * l->num_bits;

    for (size_t b = 0; b < l->num_bits; ++b) {
        const float *plane = l->hyperplanes[base + b];
        float dot = 0.0f;
        for (size_t d = 0; d < l->dimension; ++d) {
            dot += data[d] * plane[d];
        }
        if (dot >= 0.0f) {
            hash |= (1U << b);
        }
    }

    return hash;
}

static int emb_lsh_bucket_add(GV_EmbeddedDB *db, GV_EmbeddedLSHBucket *bucket, size_t idx) {
    if (bucket->count >= bucket->capacity) {
        size_t new_cap = bucket->capacity == 0 ? 8 : bucket->capacity * 2;
        size_t *new_indices = (size_t *)emb_tracked_realloc(
            db, bucket->indices, bucket->capacity * sizeof(size_t),
            new_cap * sizeof(size_t));
        if (!new_indices) return -1;
        bucket->indices = new_indices;
        bucket->capacity = new_cap;
    }
    bucket->indices[bucket->count++] = idx;
    return 0;
}

static int emb_lsh_insert(GV_EmbeddedDB *db, size_t vec_idx) {
    GV_EmbeddedLSH *l = db->lsh;
    const float *data = db->vectors + vec_idx * db->dimension;

    for (size_t t = 0; t < l->num_tables; ++t) {
        uint32_t hash = emb_lsh_hash(l, data, t);
        uint32_t bucket_idx = hash % (uint32_t)l->num_buckets;
        if (emb_lsh_bucket_add(db, &l->tables[t][bucket_idx], vec_idx) != 0) {
            return -1;
        }
    }

    return 0;
}

static int emb_lsh_search(const GV_EmbeddedDB *db, const float *query, size_t k,
                           int dist_type, GV_EmbeddedHeapItem *heap, size_t *heap_size) {
    GV_EmbeddedLSH *l = db->lsh;

    /* Collect candidate set from all tables */
    uint8_t *seen = (uint8_t *)calloc((db->count + 7) / 8, 1);
    if (!seen) return -1;

    for (size_t t = 0; t < l->num_tables; ++t) {
        uint32_t hash = emb_lsh_hash(l, query, t);
        uint32_t bucket_idx = hash % (uint32_t)l->num_buckets;
        GV_EmbeddedLSHBucket *bucket = &l->tables[t][bucket_idx];

        for (size_t i = 0; i < bucket->count; ++i) {
            size_t vec_idx = bucket->indices[i];
            if (vec_idx >= db->count) continue;
            if (seen[vec_idx / 8] & (1U << (vec_idx % 8))) continue;
            seen[vec_idx / 8] |= (uint8_t)(1U << (vec_idx % 8));

            if (emb_is_deleted(db, vec_idx)) continue;

            float dist = emb_compute_distance(query, db->vectors + vec_idx * db->dimension,
                                               db->dimension, dist_type);
            emb_heap_push(heap, heap_size, k, dist, vec_idx);
        }
    }

    free(seen);
    return 0;
}

/* ================================================================== */
/* Storage growth                                                     */
/* ================================================================== */

static int emb_ensure_capacity(GV_EmbeddedDB *db, size_t needed) {
    if (needed <= db->capacity) return 0;

    /* Enforce hard max_vectors limit */
    if (db->max_vectors > 0 && needed > db->max_vectors) return -1;

    size_t new_cap = db->capacity == 0 ? 64 : db->capacity;
    while (new_cap < needed) new_cap *= 2;
    if (db->max_vectors > 0 && new_cap > db->max_vectors) new_cap = db->max_vectors;

    /* Grow vector storage */
    size_t old_vec_bytes = db->capacity * db->dimension * sizeof(float);
    size_t new_vec_bytes = new_cap * db->dimension * sizeof(float);
    float *new_vecs = (float *)emb_tracked_realloc(db, db->vectors, old_vec_bytes, new_vec_bytes);
    if (!new_vecs) return -1;
    memset(new_vecs + db->capacity * db->dimension, 0,
           (new_cap - db->capacity) * db->dimension * sizeof(float));
    db->vectors = new_vecs;

    /* Grow deleted bitmap */
    size_t old_del_bytes = db->deleted_bytes;
    size_t new_del_bytes = (new_cap + 7) / 8;
    uint8_t *new_del = (uint8_t *)emb_tracked_realloc(db, db->deleted, old_del_bytes, new_del_bytes);
    if (!new_del) return -1;
    if (new_del_bytes > old_del_bytes) {
        memset(new_del + old_del_bytes, 0, new_del_bytes - old_del_bytes);
    }
    db->deleted = new_del;
    db->deleted_bytes = new_del_bytes;

    /* Grow quantization buffer */
    if (db->quant && db->quant->data) {
        size_t old_q = db->capacity * db->quant->bytes_per_vector;  /* use old capacity before update */
        size_t new_q = new_cap * db->quant->bytes_per_vector;
        uint8_t *new_qdata = (uint8_t *)emb_tracked_realloc(db, db->quant->data, old_q, new_q);
        if (!new_qdata) return -1;
        if (new_q > old_q) memset(new_qdata + old_q, 0, new_q - old_q);
        db->quant->data = new_qdata;
    }

    db->capacity = new_cap;
    return 0;
}

/* ================================================================== */
/* Public API                                                         */
/* ================================================================== */

void gv_embedded_config_init(GV_EmbeddedConfig *config) {
    if (!config) return;
    memset(config, 0, sizeof(GV_EmbeddedConfig));
    config->dimension       = 0;
    config->index_type      = GV_EMBEDDED_INDEX_FLAT;
    config->max_vectors     = 0;
    config->memory_limit_mb = 64;
    config->mmap_storage    = 0;
    config->storage_path    = NULL;
    config->quantize        = 0;
}

GV_EmbeddedDB *gv_embedded_open(const GV_EmbeddedConfig *config) {
    if (!config || config->dimension == 0) return NULL;

    /* Validate index type */
    if (config->index_type != GV_EMBEDDED_INDEX_FLAT &&
        config->index_type != GV_EMBEDDED_INDEX_HNSW &&
        config->index_type != GV_EMBEDDED_INDEX_LSH) {
        return NULL;
    }

    /* Validate quantization */
    if (config->quantize != 0 && config->quantize != 4 && config->quantize != 8) {
        return NULL;
    }

    GV_EmbeddedDB *db = (GV_EmbeddedDB *)calloc(1, sizeof(GV_EmbeddedDB));
    if (!db) return NULL;

    db->dimension       = config->dimension;
    db->index_type      = config->index_type;
    db->max_vectors     = config->max_vectors;
    db->memory_limit_mb = config->memory_limit_mb > 0 ? config->memory_limit_mb : 64;
    db->quantize        = config->quantize;
    db->memory_used     = sizeof(GV_EmbeddedDB);

    /* Initial capacity */
    size_t init_cap = 64;
    if (db->max_vectors > 0 && init_cap > db->max_vectors) {
        init_cap = db->max_vectors;
    }

    if (emb_ensure_capacity(db, init_cap) != 0) {
        free(db);
        return NULL;
    }

    /* Quantization */
    if (db->quantize > 0) {
        db->quant = emb_quant_create(db, db->capacity);
        if (!db->quant) {
            emb_tracked_free(db, db->vectors, db->capacity * db->dimension * sizeof(float));
            emb_tracked_free(db, db->deleted, db->deleted_bytes);
            free(db);
            return NULL;
        }
    }

    /* Index structures */
    if (db->index_type == GV_EMBEDDED_INDEX_HNSW) {
        db->hnsw = emb_hnsw_create(db, init_cap);
        if (!db->hnsw) {
            if (db->quant) emb_quant_destroy(db, db->quant);
            emb_tracked_free(db, db->vectors, db->capacity * db->dimension * sizeof(float));
            emb_tracked_free(db, db->deleted, db->deleted_bytes);
            free(db);
            return NULL;
        }
    } else if (db->index_type == GV_EMBEDDED_INDEX_LSH) {
        db->lsh = emb_lsh_create(db);
        if (!db->lsh) {
            if (db->quant) emb_quant_destroy(db, db->quant);
            emb_tracked_free(db, db->vectors, db->capacity * db->dimension * sizeof(float));
            emb_tracked_free(db, db->deleted, db->deleted_bytes);
            free(db);
            return NULL;
        }
    }

    return db;
}

void gv_embedded_close(GV_EmbeddedDB *db) {
    if (!db) return;

    if (db->hnsw) emb_hnsw_destroy(db, db->hnsw);
    if (db->lsh)  emb_lsh_destroy(db, db->lsh);
    if (db->quant) emb_quant_destroy(db, db->quant);

    emb_tracked_free(db, db->vectors, db->capacity * db->dimension * sizeof(float));
    emb_tracked_free(db, db->deleted, db->deleted_bytes);
    free(db);
}

int gv_embedded_add(GV_EmbeddedDB *db, const float *vector) {
    if (!db || !vector) return -1;

    /* Enforce hard limit */
    if (db->max_vectors > 0 && db->count >= db->max_vectors) return -1;

    if (emb_ensure_capacity(db, db->count + 1) != 0) return -1;

    size_t idx = db->count;
    memcpy(db->vectors + idx * db->dimension, vector, db->dimension * sizeof(float));
    emb_clear_deleted(db, idx);
    db->count++;

    /* Quantize */
    if (db->quant) {
        emb_quant_update_minmax(db->quant, vector, db->dimension);
        emb_quant_encode(db->quant, vector, db->dimension, db->quantize,
                         db->quant->data + idx * db->quant->bytes_per_vector);
    }

    /* Update index */
    if (db->index_type == GV_EMBEDDED_INDEX_HNSW) {
        if (emb_hnsw_insert(db, idx) != 0) return -1;
    } else if (db->index_type == GV_EMBEDDED_INDEX_LSH) {
        if (emb_lsh_insert(db, idx) != 0) return -1;
    }

    return (int)idx;
}

int gv_embedded_add_with_id(GV_EmbeddedDB *db, size_t id, const float *vector) {
    if (!db || !vector) return -1;

    /* Enforce hard limit */
    if (db->max_vectors > 0 && id >= db->max_vectors) return -1;

    size_t needed = id + 1;
    if (emb_ensure_capacity(db, needed) != 0) return -1;

    /* Extend count if necessary, marking intermediate slots as deleted */
    while (db->count < needed) {
        emb_mark_deleted(db, db->count);
        db->count++;
    }

    memcpy(db->vectors + id * db->dimension, vector, db->dimension * sizeof(float));
    emb_clear_deleted(db, id);

    /* Quantize */
    if (db->quant) {
        emb_quant_update_minmax(db->quant, vector, db->dimension);
        emb_quant_encode(db->quant, vector, db->dimension, db->quantize,
                         db->quant->data + id * db->quant->bytes_per_vector);
    }

    /* Update index */
    if (db->index_type == GV_EMBEDDED_INDEX_HNSW) {
        if (emb_hnsw_insert(db, id) != 0) return -1;
    } else if (db->index_type == GV_EMBEDDED_INDEX_LSH) {
        if (emb_lsh_insert(db, id) != 0) return -1;
    }

    return 0;
}

int gv_embedded_search(const GV_EmbeddedDB *db, const float *query, size_t k,
                       int distance_type, GV_EmbeddedResult *results) {
    if (!db || !query || !results || k == 0) return -1;
    if (db->count == 0) return 0;

    GV_EmbeddedHeapItem *heap = (GV_EmbeddedHeapItem *)malloc(k * sizeof(GV_EmbeddedHeapItem));
    if (!heap) return -1;
    size_t heap_size = 0;

    if (db->index_type == GV_EMBEDDED_INDEX_HNSW && db->hnsw) {
        if (emb_hnsw_search(db, query, k, distance_type, heap, &heap_size) != 0) {
            free(heap);
            return -1;
        }
    } else if (db->index_type == GV_EMBEDDED_INDEX_LSH && db->lsh) {
        if (emb_lsh_search(db, query, k, distance_type, heap, &heap_size) != 0) {
            free(heap);
            return -1;
        }
    } else {
        /* FLAT: brute-force linear scan */
        for (size_t i = 0; i < db->count; ++i) {
            if (emb_is_deleted(db, i)) continue;
            float dist = emb_compute_distance(query, db->vectors + i * db->dimension,
                                               db->dimension, distance_type);
            emb_heap_push(heap, &heap_size, k, dist, i);
        }
    }

    /* Extract results from max-heap in ascending distance order */
    int n = (int)heap_size;
    for (int i = n - 1; i >= 0; --i) {
        results[i].index    = heap[0].idx;
        results[i].distance = heap[0].dist;
        /* Remove max from heap */
        heap[0] = heap[heap_size - 1];
        heap_size--;
        if (heap_size > 0) {
            emb_heap_sift_down(heap, heap_size, 0);
        }
    }

    free(heap);
    return n;
}

int gv_embedded_delete(GV_EmbeddedDB *db, size_t index) {
    if (!db) return -1;
    if (index >= db->count) return -1;
    if (emb_is_deleted(db, index)) return -1;

    emb_mark_deleted(db, index);
    return 0;
}

int gv_embedded_get(const GV_EmbeddedDB *db, size_t index, float *output) {
    if (!db || !output) return -1;
    if (index >= db->count) return -1;
    if (emb_is_deleted(db, index)) return -1;

    memcpy(output, db->vectors + index * db->dimension, db->dimension * sizeof(float));
    return 0;
}

size_t gv_embedded_count(const GV_EmbeddedDB *db) {
    if (!db) return 0;

    size_t active = 0;
    for (size_t i = 0; i < db->count; ++i) {
        if (!emb_is_deleted(db, i)) active++;
    }
    return active;
}

size_t gv_embedded_memory_usage(const GV_EmbeddedDB *db) {
    if (!db) return 0;
    return db->memory_used;
}

/* ================================================================== */
/* Save / Load (binary format)                                        */
/* ================================================================== */

static int emb_write_u32(FILE *f, uint32_t v) {
    return fwrite(&v, sizeof(uint32_t), 1, f) == 1 ? 0 : -1;
}

static int emb_read_u32(FILE *f, uint32_t *v) {
    return (v && fread(v, sizeof(uint32_t), 1, f) == 1) ? 0 : -1;
}

static int emb_write_u64(FILE *f, uint64_t v) {
    return fwrite(&v, sizeof(uint64_t), 1, f) == 1 ? 0 : -1;
}

static int emb_read_u64(FILE *f, uint64_t *v) {
    return (v && fread(v, sizeof(uint64_t), 1, f) == 1) ? 0 : -1;
}

int gv_embedded_save(const GV_EmbeddedDB *db, const char *path) {
    if (!db || !path) return -1;

    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    /* Header: magic */
    if (fwrite(GV_EMBEDDED_MAGIC, 1, GV_EMBEDDED_MAGIC_LEN, f) != GV_EMBEDDED_MAGIC_LEN) {
        fclose(f);
        return -1;
    }

    /* Header: version, dimension, count, index_type, quantize */
    if (emb_write_u32(f, GV_EMBEDDED_FILE_VERSION) != 0) { fclose(f); return -1; }
    if (emb_write_u64(f, (uint64_t)db->dimension) != 0)  { fclose(f); return -1; }
    if (emb_write_u64(f, (uint64_t)db->count) != 0)      { fclose(f); return -1; }
    if (emb_write_u32(f, (uint32_t)db->index_type) != 0)  { fclose(f); return -1; }
    if (emb_write_u32(f, (uint32_t)db->quantize) != 0)    { fclose(f); return -1; }
    if (emb_write_u64(f, (uint64_t)db->max_vectors) != 0) { fclose(f); return -1; }
    if (emb_write_u64(f, (uint64_t)db->memory_limit_mb) != 0) { fclose(f); return -1; }

    /* Deleted bitmap */
    size_t del_bytes = (db->count + 7) / 8;
    if (del_bytes > 0) {
        if (fwrite(db->deleted, 1, del_bytes, f) != del_bytes) { fclose(f); return -1; }
    }

    /* Vector data */
    size_t vec_floats = db->count * db->dimension;
    if (vec_floats > 0) {
        if (fwrite(db->vectors, sizeof(float), vec_floats, f) != vec_floats) { fclose(f); return -1; }
    }

    /* Quantization data */
    if (db->quantize > 0 && db->quant) {
        if (fwrite(db->quant->min_vals, sizeof(float), db->dimension, f) != db->dimension) {
            fclose(f);
            return -1;
        }
        if (fwrite(db->quant->max_vals, sizeof(float), db->dimension, f) != db->dimension) {
            fclose(f);
            return -1;
        }
        size_t q_bytes = db->count * db->quant->bytes_per_vector;
        if (q_bytes > 0) {
            if (fwrite(db->quant->data, 1, q_bytes, f) != q_bytes) { fclose(f); return -1; }
        }
    }

    /* HNSW graph data */
    if (db->index_type == GV_EMBEDDED_INDEX_HNSW && db->hnsw) {
        if (emb_write_u64(f, (uint64_t)db->hnsw->M) != 0) { fclose(f); return -1; }
        if (emb_write_u64(f, (uint64_t)db->hnsw->ef_construction) != 0) { fclose(f); return -1; }

        for (size_t i = 0; i < db->count; ++i) {
            GV_EmbeddedHNSWNode *node = (i < db->hnsw->capacity) ? &db->hnsw->nodes[i] : NULL;
            uint32_t nc = (node && node->neighbors) ? (uint32_t)node->neighbor_count : 0;
            if (emb_write_u32(f, nc) != 0) { fclose(f); return -1; }
            for (uint32_t n = 0; n < nc; ++n) {
                if (emb_write_u64(f, (uint64_t)node->neighbors[n]) != 0) { fclose(f); return -1; }
            }
        }
    }

    /* LSH hyperplanes */
    if (db->index_type == GV_EMBEDDED_INDEX_LSH && db->lsh) {
        if (emb_write_u64(f, (uint64_t)db->lsh->num_tables) != 0) { fclose(f); return -1; }
        if (emb_write_u64(f, (uint64_t)db->lsh->num_bits) != 0)   { fclose(f); return -1; }
        if (emb_write_u64(f, (uint64_t)db->lsh->num_buckets) != 0){ fclose(f); return -1; }

        size_t total_planes = db->lsh->num_tables * db->lsh->num_bits;
        for (size_t i = 0; i < total_planes; ++i) {
            if (fwrite(db->lsh->hyperplanes[i], sizeof(float), db->dimension, f) != db->dimension) {
                fclose(f);
                return -1;
            }
        }
    }

    fclose(f);
    return 0;
}

GV_EmbeddedDB *gv_embedded_load(const char *path) {
    if (!path) return NULL;

    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    /* Verify magic */
    char magic[GV_EMBEDDED_MAGIC_LEN];
    if (fread(magic, 1, GV_EMBEDDED_MAGIC_LEN, f) != GV_EMBEDDED_MAGIC_LEN ||
        memcmp(magic, GV_EMBEDDED_MAGIC, GV_EMBEDDED_MAGIC_LEN) != 0) {
        fclose(f);
        return NULL;
    }

    uint32_t version = 0;
    if (emb_read_u32(f, &version) != 0 || version != GV_EMBEDDED_FILE_VERSION) {
        fclose(f);
        return NULL;
    }

    uint64_t dim64 = 0, count64 = 0, max_vec64 = 0, mem_limit64 = 0;
    uint32_t idx_type = 0, quant = 0;

    if (emb_read_u64(f, &dim64) != 0)       { fclose(f); return NULL; }
    if (emb_read_u64(f, &count64) != 0)      { fclose(f); return NULL; }
    if (emb_read_u32(f, &idx_type) != 0)     { fclose(f); return NULL; }
    if (emb_read_u32(f, &quant) != 0)        { fclose(f); return NULL; }
    if (emb_read_u64(f, &max_vec64) != 0)    { fclose(f); return NULL; }
    if (emb_read_u64(f, &mem_limit64) != 0)  { fclose(f); return NULL; }

    GV_EmbeddedConfig config;
    gv_embedded_config_init(&config);
    config.dimension       = (size_t)dim64;
    config.index_type      = (int)idx_type;
    config.quantize        = (int)quant;
    config.max_vectors     = (size_t)max_vec64;
    config.memory_limit_mb = (size_t)mem_limit64;

    /* Create db but we will fill in data manually, so use a large memory limit temporarily */
    size_t saved_limit = config.memory_limit_mb;
    config.memory_limit_mb = 0; /* Disable limit during load */
    GV_EmbeddedDB *db = gv_embedded_open(&config);
    if (!db) { fclose(f); return NULL; }
    db->memory_limit_mb = saved_limit;

    size_t count = (size_t)count64;
    if (count > 0) {
        if (emb_ensure_capacity(db, count) != 0) {
            gv_embedded_close(db);
            fclose(f);
            return NULL;
        }

        /* Read deleted bitmap */
        size_t del_bytes = (count + 7) / 8;
        if (fread(db->deleted, 1, del_bytes, f) != del_bytes) {
            gv_embedded_close(db);
            fclose(f);
            return NULL;
        }

        /* Read vector data */
        size_t vec_floats = count * db->dimension;
        if (fread(db->vectors, sizeof(float), vec_floats, f) != vec_floats) {
            gv_embedded_close(db);
            fclose(f);
            return NULL;
        }

        db->count = count;
    }

    /* Read quantization data */
    if (db->quantize > 0 && db->quant && count > 0) {
        if (fread(db->quant->min_vals, sizeof(float), db->dimension, f) != db->dimension) {
            gv_embedded_close(db);
            fclose(f);
            return NULL;
        }
        if (fread(db->quant->max_vals, sizeof(float), db->dimension, f) != db->dimension) {
            gv_embedded_close(db);
            fclose(f);
            return NULL;
        }
        size_t q_bytes = count * db->quant->bytes_per_vector;
        if (q_bytes > 0) {
            if (fread(db->quant->data, 1, q_bytes, f) != q_bytes) {
                gv_embedded_close(db);
                fclose(f);
                return NULL;
            }
        }
    }

    /* Rebuild HNSW graph */
    if (db->index_type == GV_EMBEDDED_INDEX_HNSW && db->hnsw && count > 0) {
        uint64_t M64 = 0, ef64 = 0;
        if (emb_read_u64(f, &M64) != 0 || emb_read_u64(f, &ef64) != 0) {
            gv_embedded_close(db);
            fclose(f);
            return NULL;
        }
        db->hnsw->M = (size_t)M64;
        db->hnsw->ef_construction = (size_t)ef64;

        if (emb_hnsw_ensure_capacity(db, db->hnsw, count) != 0) {
            gv_embedded_close(db);
            fclose(f);
            return NULL;
        }

        for (size_t i = 0; i < count; ++i) {
            uint32_t nc = 0;
            if (emb_read_u32(f, &nc) != 0) {
                gv_embedded_close(db);
                fclose(f);
                return NULL;
            }

            GV_EmbeddedHNSWNode *node = &db->hnsw->nodes[i];
            if (nc > 0) {
                node->neighbors = (size_t *)emb_tracked_calloc(db, db->hnsw->M, sizeof(size_t));
                if (!node->neighbors) {
                    gv_embedded_close(db);
                    fclose(f);
                    return NULL;
                }
                node->neighbor_count = nc;
                for (uint32_t n = 0; n < nc; ++n) {
                    uint64_t nb64 = 0;
                    if (emb_read_u64(f, &nb64) != 0) {
                        gv_embedded_close(db);
                        fclose(f);
                        return NULL;
                    }
                    node->neighbors[n] = (size_t)nb64;
                }
            }
        }
    }

    /* Rebuild LSH tables */
    if (db->index_type == GV_EMBEDDED_INDEX_LSH && db->lsh && count > 0) {
        uint64_t nt64 = 0, nb64 = 0, nbk64 = 0;
        if (emb_read_u64(f, &nt64) != 0 || emb_read_u64(f, &nb64) != 0 ||
            emb_read_u64(f, &nbk64) != 0) {
            gv_embedded_close(db);
            fclose(f);
            return NULL;
        }

        /* Read saved hyperplanes (overwrite the randomly generated ones) */
        size_t total_planes = (size_t)nt64 * (size_t)nb64;
        for (size_t i = 0; i < total_planes && i < db->lsh->num_tables * db->lsh->num_bits; ++i) {
            if (fread(db->lsh->hyperplanes[i], sizeof(float), db->dimension, f) != db->dimension) {
                gv_embedded_close(db);
                fclose(f);
                return NULL;
            }
        }

        /* Re-insert all non-deleted vectors into LSH tables */
        for (size_t i = 0; i < count; ++i) {
            if (!emb_is_deleted(db, i)) {
                emb_lsh_insert(db, i);
            }
        }
    }

    fclose(f);
    return db;
}

/* ================================================================== */
/* Compact                                                            */
/* ================================================================== */

int gv_embedded_compact(GV_EmbeddedDB *db) {
    if (!db) return -1;

    /* Count active vectors */
    size_t active = 0;
    for (size_t i = 0; i < db->count; ++i) {
        if (!emb_is_deleted(db, i)) active++;
    }

    if (active == db->count) return 0; /* Nothing to compact */

    /* Build mapping old -> new and shift vectors */
    size_t write_pos = 0;
    for (size_t read_pos = 0; read_pos < db->count; ++read_pos) {
        if (emb_is_deleted(db, read_pos)) continue;

        if (write_pos != read_pos) {
            memcpy(db->vectors + write_pos * db->dimension,
                   db->vectors + read_pos * db->dimension,
                   db->dimension * sizeof(float));
        }

        /* Re-encode quantized data */
        if (db->quant && db->quant->data) {
            emb_quant_encode(db->quant, db->vectors + write_pos * db->dimension,
                             db->dimension, db->quantize,
                             db->quant->data + write_pos * db->quant->bytes_per_vector);
        }

        write_pos++;
    }

    db->count = active;

    /* Clear deleted bitmap */
    size_t del_bytes = (db->count + 7) / 8;
    memset(db->deleted, 0, db->deleted_bytes);
    (void)del_bytes;

    /* Rebuild index structures */
    if (db->index_type == GV_EMBEDDED_INDEX_HNSW && db->hnsw) {
        /* Destroy old graph */
        emb_hnsw_destroy(db, db->hnsw);
        db->hnsw = emb_hnsw_create(db, db->capacity);
        if (!db->hnsw) return -1;

        /* Re-insert all vectors */
        for (size_t i = 0; i < db->count; ++i) {
            if (emb_hnsw_insert(db, i) != 0) return -1;
        }
    } else if (db->index_type == GV_EMBEDDED_INDEX_LSH && db->lsh) {
        /* Clear all buckets */
        for (size_t t = 0; t < db->lsh->num_tables; ++t) {
            for (size_t b = 0; b < db->lsh->num_buckets; ++b) {
                db->lsh->tables[t][b].count = 0;
            }
        }

        /* Re-insert all vectors */
        for (size_t i = 0; i < db->count; ++i) {
            if (emb_lsh_insert(db, i) != 0) return -1;
        }
    }

    return 0;
}
