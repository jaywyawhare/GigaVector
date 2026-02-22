#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

#include "gigavector/gv_multivec.h"
#include "gigavector/gv_vector.h"
#include "gigavector/gv_distance.h"

/*  Internal data structures                                          */

/**
 * @brief A single document entry holding its chunk vectors.
 */
typedef struct {
    uint64_t doc_id;      /**< Unique document identifier. */
    float   *chunks;      /**< Contiguous chunk data (num_chunks * dimension floats). */
    size_t   num_chunks;  /**< Number of chunks stored for this document. */
    int      deleted;     /**< Non-zero when logically deleted. */
} GV_DocEntry;

/**
 * @brief Internal representation of a multi-vector index.
 */
typedef struct {
    size_t             dimension;
    GV_MultiVecConfig  config;
    GV_DocEntry       *docs;
    size_t             doc_count;
    size_t             doc_capacity;
    size_t             total_chunks;
} GV_MultiVecIndex;

/*  Max-heap helpers for top-k document selection                     */

typedef struct {
    float    dist;
    size_t   doc_idx;
} GV_MVHeapItem;

static void gv_mv_heap_sift_down(GV_MVHeapItem *heap, size_t size, size_t i) {
    while (1) {
        size_t l = 2 * i + 1;
        size_t r = l + 1;
        size_t largest = i;
        if (l < size && heap[l].dist > heap[largest].dist) largest = l;
        if (r < size && heap[r].dist > heap[largest].dist) largest = r;
        if (largest == i) break;
        GV_MVHeapItem tmp = heap[i];
        heap[i] = heap[largest];
        heap[largest] = tmp;
        i = largest;
    }
}

static void gv_mv_heap_push(GV_MVHeapItem *heap, size_t *size, size_t capacity,
                             float dist, size_t doc_idx) {
    if (*size < capacity) {
        heap[*size].dist = dist;
        heap[*size].doc_idx = doc_idx;
        (*size)++;
        /* Sift up */
        size_t i = *size - 1;
        while (i > 0) {
            size_t parent = (i - 1) / 2;
            if (heap[i].dist > heap[parent].dist) {
                GV_MVHeapItem tmp = heap[i];
                heap[i] = heap[parent];
                heap[parent] = tmp;
                i = parent;
            } else {
                break;
            }
        }
    } else if (dist < heap[0].dist) {
        heap[0].dist = dist;
        heap[0].doc_idx = doc_idx;
        gv_mv_heap_sift_down(heap, *size, 0);
    }
}

/*  Default configuration                                             */

static const GV_MultiVecConfig gv_multivec_default_config = {
    .max_chunks_per_doc = 256,
    .aggregation        = GV_DOC_AGG_MAX_SIM
};

/*  Lifecycle                                                         */

void *gv_multivec_create(size_t dimension, const GV_MultiVecConfig *config) {
    if (dimension == 0) return NULL;

    GV_MultiVecIndex *idx = (GV_MultiVecIndex *)calloc(1, sizeof(GV_MultiVecIndex));
    if (!idx) return NULL;

    idx->dimension = dimension;
    idx->config = config ? *config : gv_multivec_default_config;

    if (idx->config.max_chunks_per_doc == 0) {
        idx->config.max_chunks_per_doc = 256;
    }

    idx->doc_capacity = 16;
    idx->docs = (GV_DocEntry *)calloc(idx->doc_capacity, sizeof(GV_DocEntry));
    if (!idx->docs) {
        free(idx);
        return NULL;
    }

    idx->doc_count    = 0;
    idx->total_chunks = 0;

    return idx;
}

void gv_multivec_destroy(void *index) {
    if (!index) return;
    GV_MultiVecIndex *idx = (GV_MultiVecIndex *)index;

    for (size_t i = 0; i < idx->doc_count; i++) {
        free(idx->docs[i].chunks);
    }
    free(idx->docs);
    free(idx);
}

/*  Document management                                               */

int gv_multivec_add_document(void *index, uint64_t doc_id,
                             const float *chunks, size_t num_chunks,
                             size_t dimension) {
    if (!index || !chunks || num_chunks == 0) return -1;

    GV_MultiVecIndex *idx = (GV_MultiVecIndex *)index;

    if (dimension != idx->dimension) return -1;
    if (num_chunks > idx->config.max_chunks_per_doc) return -1;

    /* Check for duplicate doc_id */
    for (size_t i = 0; i < idx->doc_count; i++) {
        if (!idx->docs[i].deleted && idx->docs[i].doc_id == doc_id) {
            return -1;
        }
    }

    /* Grow array if necessary */
    if (idx->doc_count >= idx->doc_capacity) {
        size_t new_cap = idx->doc_capacity * 2;
        GV_DocEntry *new_docs = (GV_DocEntry *)realloc(idx->docs,
                                                       new_cap * sizeof(GV_DocEntry));
        if (!new_docs) return -1;
        idx->docs = new_docs;
        idx->doc_capacity = new_cap;
    }

    /* Copy chunk data */
    size_t data_size = num_chunks * dimension * sizeof(float);
    float *chunk_copy = (float *)malloc(data_size);
    if (!chunk_copy) return -1;
    memcpy(chunk_copy, chunks, data_size);

    GV_DocEntry *entry = &idx->docs[idx->doc_count];
    entry->doc_id     = doc_id;
    entry->chunks     = chunk_copy;
    entry->num_chunks = num_chunks;
    entry->deleted    = 0;

    idx->doc_count++;
    idx->total_chunks += num_chunks;

    return 0;
}

int gv_multivec_delete_document(void *index, uint64_t doc_id) {
    if (!index) return -1;
    GV_MultiVecIndex *idx = (GV_MultiVecIndex *)index;

    for (size_t i = 0; i < idx->doc_count; i++) {
        if (!idx->docs[i].deleted && idx->docs[i].doc_id == doc_id) {
            idx->docs[i].deleted = 1;
            idx->total_chunks -= idx->docs[i].num_chunks;
            free(idx->docs[i].chunks);
            idx->docs[i].chunks = NULL;
            idx->docs[i].num_chunks = 0;
            return 0;
        }
    }

    return -1; /* not found */
}

/*  Search                                                            */

/**
 * @brief Compute the aggregated distance from a query to all chunks of a
 *        document, returning the aggregate score and the index of the
 *        best (lowest-distance) chunk.
 */
static float gv_multivec_aggregate(const float *query,
                                   const GV_DocEntry *doc,
                                   size_t dimension,
                                   GV_DistanceType distance_type,
                                   GV_DocAggregation aggregation,
                                   size_t *best_chunk_out) {
    /* Build a temporary GV_Vector for the query so we can reuse gv_distance(). */
    GV_Vector q_vec;
    q_vec.dimension = dimension;
    q_vec.data      = (float *)query;   /* const-cast safe: gv_distance only reads */
    q_vec.metadata  = NULL;

    GV_Vector c_vec;
    c_vec.dimension = dimension;
    c_vec.metadata  = NULL;

    float best_dist = FLT_MAX;
    size_t best_idx = 0;
    float sum_dist  = 0.0f;

    for (size_t c = 0; c < doc->num_chunks; c++) {
        c_vec.data = doc->chunks + c * dimension;
        float d = gv_distance(&q_vec, &c_vec, distance_type);

        sum_dist += d;

        if (d < best_dist) {
            best_dist = d;
            best_idx  = c;
        }
    }

    if (best_chunk_out) {
        *best_chunk_out = best_idx;
    }

    switch (aggregation) {
        case GV_DOC_AGG_MAX_SIM:
            return best_dist;
        case GV_DOC_AGG_AVG_SIM:
            return (doc->num_chunks > 0) ? (sum_dist / (float)doc->num_chunks) : 0.0f;
        case GV_DOC_AGG_SUM_SIM:
            return sum_dist;
        default:
            return best_dist;
    }
}

int gv_multivec_search(void *index, const float *query, size_t k,
                       GV_DocSearchResult *results,
                       GV_DistanceType distance_type) {
    if (!index || !query || !results || k == 0) return -1;

    GV_MultiVecIndex *idx = (GV_MultiVecIndex *)index;

    GV_MVHeapItem *heap = (GV_MVHeapItem *)malloc(k * sizeof(GV_MVHeapItem));
    if (!heap) return -1;
    size_t heap_size = 0;

    for (size_t i = 0; i < idx->doc_count; i++) {
        if (idx->docs[i].deleted) continue;

        size_t best_chunk = 0;
        float agg_dist = gv_multivec_aggregate(query, &idx->docs[i],
                                               idx->dimension,
                                               distance_type,
                                               idx->config.aggregation,
                                               &best_chunk);

        gv_mv_heap_push(heap, &heap_size, k, agg_dist, i);
    }

    /* Extract results from the max-heap in ascending distance order. */
    int n = (int)heap_size;

    for (int i = n - 1; i >= 0; i--) {
        size_t di = heap[0].doc_idx;
        const GV_DocEntry *doc = &idx->docs[di];

        /* Recompute best chunk for the result record. */
        size_t best_chunk = 0;
        gv_multivec_aggregate(query, doc, idx->dimension,
                              distance_type, idx->config.aggregation,
                              &best_chunk);

        results[i].doc_id           = doc->doc_id;
        results[i].score            = heap[0].dist;
        results[i].num_chunks       = doc->num_chunks;
        results[i].best_chunk_index = best_chunk;

        /* Pop top of heap */
        heap[0] = heap[heap_size - 1];
        heap_size--;
        if (heap_size > 0) {
            gv_mv_heap_sift_down(heap, heap_size, 0);
        }
    }

    free(heap);
    return n;
}

/*  Statistics                                                        */

size_t gv_multivec_count_documents(const void *index) {
    if (!index) return 0;
    const GV_MultiVecIndex *idx = (const GV_MultiVecIndex *)index;

    size_t count = 0;
    for (size_t i = 0; i < idx->doc_count; i++) {
        if (!idx->docs[i].deleted) count++;
    }
    return count;
}

size_t gv_multivec_count_chunks(const void *index) {
    if (!index) return 0;
    const GV_MultiVecIndex *idx = (const GV_MultiVecIndex *)index;
    return idx->total_chunks;
}

/*  Serialization helpers                                             */

static int gv_mv_write_u64(FILE *f, uint64_t v) {
    return fwrite(&v, sizeof(uint64_t), 1, f) == 1 ? 0 : -1;
}

static int gv_mv_read_u64(FILE *f, uint64_t *v) {
    return (v && fread(v, sizeof(uint64_t), 1, f) == 1) ? 0 : -1;
}

static int gv_mv_write_u32(FILE *f, uint32_t v) {
    return fwrite(&v, sizeof(uint32_t), 1, f) == 1 ? 0 : -1;
}

static int gv_mv_read_u32(FILE *f, uint32_t *v) {
    return (v && fread(v, sizeof(uint32_t), 1, f) == 1) ? 0 : -1;
}

/*  Save / Load                                                       */

int gv_multivec_save(const void *index, FILE *out) {
    if (!index || !out) return -1;
    const GV_MultiVecIndex *idx = (const GV_MultiVecIndex *)index;

    /* Write header: dimension, config */
    if (gv_mv_write_u32(out, (uint32_t)idx->dimension) != 0) return -1;
    if (gv_mv_write_u32(out, (uint32_t)idx->config.max_chunks_per_doc) != 0) return -1;
    if (gv_mv_write_u32(out, (uint32_t)idx->config.aggregation) != 0) return -1;

    /* Count non-deleted documents */
    uint32_t active_count = 0;
    for (size_t i = 0; i < idx->doc_count; i++) {
        if (!idx->docs[i].deleted) active_count++;
    }

    if (gv_mv_write_u32(out, active_count) != 0) return -1;

    /* Write each non-deleted document */
    for (size_t i = 0; i < idx->doc_count; i++) {
        const GV_DocEntry *doc = &idx->docs[i];
        if (doc->deleted) continue;

        if (gv_mv_write_u64(out, doc->doc_id) != 0) return -1;
        if (gv_mv_write_u32(out, (uint32_t)doc->num_chunks) != 0) return -1;

        size_t floats = doc->num_chunks * idx->dimension;
        if (floats > 0) {
            if (fwrite(doc->chunks, sizeof(float), floats, out) != floats) return -1;
        }
    }

    return 0;
}

int gv_multivec_load(void **index_ptr, FILE *in, size_t dimension) {
    if (!index_ptr || !in) return -1;

    uint32_t file_dim = 0, max_chunks = 0, aggregation = 0, doc_count = 0;

    if (gv_mv_read_u32(in, &file_dim) != 0) return -1;
    if (gv_mv_read_u32(in, &max_chunks) != 0) return -1;
    if (gv_mv_read_u32(in, &aggregation) != 0) return -1;
    if (gv_mv_read_u32(in, &doc_count) != 0) return -1;

    if (dimension != 0 && dimension != (size_t)file_dim) return -1;

    GV_MultiVecConfig config;
    config.max_chunks_per_doc = (size_t)max_chunks;
    config.aggregation        = (GV_DocAggregation)aggregation;

    void *index = gv_multivec_create((size_t)file_dim, &config);
    if (!index) return -1;

    for (uint32_t i = 0; i < doc_count; i++) {
        uint64_t doc_id = 0;
        uint32_t num_chunks = 0;

        if (gv_mv_read_u64(in, &doc_id) != 0) {
            gv_multivec_destroy(index);
            return -1;
        }
        if (gv_mv_read_u32(in, &num_chunks) != 0) {
            gv_multivec_destroy(index);
            return -1;
        }

        size_t floats = (size_t)num_chunks * (size_t)file_dim;
        float *chunks = NULL;

        if (floats > 0) {
            chunks = (float *)malloc(floats * sizeof(float));
            if (!chunks) {
                gv_multivec_destroy(index);
                return -1;
            }
            if (fread(chunks, sizeof(float), floats, in) != floats) {
                free(chunks);
                gv_multivec_destroy(index);
                return -1;
            }
        }

        int rc = gv_multivec_add_document(index, doc_id, chunks,
                                          (size_t)num_chunks, (size_t)file_dim);
        free(chunks);

        if (rc != 0) {
            gv_multivec_destroy(index);
            return -1;
        }
    }

    *index_ptr = index;
    return 0;
}
