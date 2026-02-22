/**
 * @file gv_diskann.c
 * @brief DiskANN: SSD-backed Vamana graph index for datasets exceeding RAM.
 *
 * Implements the Vamana graph algorithm with:
 *   - Greedy search + robust pruning (alpha-controlled)
 *   - Beam search starting from the medoid entry point
 *   - PQ-compressed vectors for in-memory neighbor distance estimates
 *   - Full vectors stored in sector-aligned disk pages, read via pread()
 *   - LRU cache for frequently accessed disk pages
 *   - Lazy deletion with tombstone flags
 *   - Persistence: graph adjacency lists + PQ codebooks + metadata header
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <stdint.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <time.h>

#include "gigavector/gv_diskann.h"

/*  Constants  */

#define DISKANN_MAGIC           0x44414E4E  /* "DANN" */
#define DISKANN_VERSION         1
#define DISKANN_DEFAULT_DEGREE  64
#define DISKANN_DEFAULT_ALPHA   1.2f
#define DISKANN_DEFAULT_BUILD_BW  128
#define DISKANN_DEFAULT_SEARCH_BW 64
#define DISKANN_DEFAULT_CACHE_MB  256
#define DISKANN_DEFAULT_SECTOR    4096
#define DISKANN_PQ_NBITS          8
#define DISKANN_PQ_KSUB           256  /* 2^8 */
#define DISKANN_INITIAL_CAPACITY  1024

/*  PQ Codebook (in-memory compressed navigation)  */

typedef struct {
    size_t m;            /* Number of sub-quantizers */
    size_t dsub;         /* Sub-vector dimension = dimension / m */
    size_t ksub;         /* Centroids per sub-quantizer (256 for 8-bit) */
    float *codebooks;    /* m * ksub * dsub floats */
    int trained;
} DiskANN_PQ;

/*  LRU Disk Page Cache  */

typedef struct DiskANN_CachePage {
    size_t page_id;                     /* Which vector page this caches */
    float *data;                        /* Decoded vector data */
    struct DiskANN_CachePage *lru_prev;
    struct DiskANN_CachePage *lru_next;
    struct DiskANN_CachePage *hash_next;
} DiskANN_CachePage;

#define DISKANN_CACHE_BUCKETS 1024

typedef struct {
    DiskANN_CachePage *buckets[DISKANN_CACHE_BUCKETS];
    DiskANN_CachePage *lru_head;  /* Most recently used */
    DiskANN_CachePage *lru_tail;  /* Least recently used */
    size_t count;
    size_t max_pages;
    size_t page_data_bytes;       /* Bytes per cached page (vectors_per_page * dim * sizeof(float)) */
    size_t vectors_per_page;
    size_t hits;
    size_t misses;
} DiskANN_Cache;

/*  Graph Node  */

typedef struct {
    size_t *neighbors;       /* Adjacency list (indices into nodes array) */
    size_t neighbor_count;
    uint8_t *pq_code;        /* PQ-compressed representation (m bytes) */
    int deleted;             /* Tombstone flag */
} DiskANN_Node;

/*  Index Structure  */

struct GV_DiskANNIndex {
    /* Configuration */
    size_t dimension;
    size_t max_degree;
    float alpha;
    size_t build_beam_width;
    size_t search_beam_width;
    size_t sector_size;

    /* Graph */
    DiskANN_Node *nodes;
    size_t count;
    size_t capacity;
    size_t medoid;           /* Entry point: vector closest to centroid */

    /* PQ for in-memory navigation */
    DiskANN_PQ pq;

    /* Disk storage */
    char *data_path;
    int data_fd;             /* File descriptor for on-disk vector storage */
    size_t vectors_per_page; /* How many vectors fit in one sector-aligned page */

    /* LRU cache */
    DiskANN_Cache cache;

    /* Statistics */
    size_t disk_reads;
    double total_search_latency_us;
    size_t total_searches;
};

/*  Forward Declarations  */

static float diskann_l2_distance(const float *a, const float *b, size_t dim);
static void diskann_pq_init(DiskANN_PQ *pq);
static int diskann_pq_train(DiskANN_PQ *pq, const float *data, size_t count, size_t dimension, size_t pq_dim);
static void diskann_pq_encode(const DiskANN_PQ *pq, const float *vec, uint8_t *code);
static float diskann_pq_distance(const DiskANN_PQ *pq, const float *query, const uint8_t *code);
static void diskann_pq_destroy(DiskANN_PQ *pq);

static void diskann_cache_init(DiskANN_Cache *cache, size_t max_mb, size_t vectors_per_page, size_t dimension);
static float *diskann_cache_get(DiskANN_Cache *cache, size_t page_id);
static void diskann_cache_put(DiskANN_Cache *cache, size_t page_id, const float *data);
static void diskann_cache_destroy(DiskANN_Cache *cache);

static int diskann_disk_open(GV_DiskANNIndex *index);
static int diskann_disk_write_vector(GV_DiskANNIndex *index, size_t vec_index, const float *data);
static int diskann_disk_read_vector(GV_DiskANNIndex *index, size_t vec_index, float *out);
static void diskann_disk_close(GV_DiskANNIndex *index);

static size_t diskann_compute_medoid(const float *data, size_t count, size_t dimension);
static int diskann_greedy_search(const GV_DiskANNIndex *index, const float *query,
                                  size_t beam_width, size_t *visited, size_t *visited_count,
                                  size_t max_visited);
static void diskann_robust_prune(GV_DiskANNIndex *index, size_t node_id,
                                  size_t *candidates, float *distances, size_t cand_count);

/*  Utility: L2 Squared Distance  */

static float diskann_l2_distance(const float *a, const float *b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

/*  Min-Heap for Beam Search  */

typedef struct {
    size_t index;
    float distance;
} DiskANN_Candidate;

static int diskann_cand_compare(const void *a, const void *b) {
    const DiskANN_Candidate *ca = (const DiskANN_Candidate *)a;
    const DiskANN_Candidate *cb = (const DiskANN_Candidate *)b;
    if (ca->distance < cb->distance) return -1;
    if (ca->distance > cb->distance) return 1;
    return 0;
}

/*  PQ Codebook Implementation  */

static void diskann_pq_init(DiskANN_PQ *pq) {
    memset(pq, 0, sizeof(DiskANN_PQ));
    pq->ksub = DISKANN_PQ_KSUB;
}

static int diskann_pq_train(DiskANN_PQ *pq, const float *data, size_t count,
                             size_t dimension, size_t pq_dim) {
    if (count == 0 || dimension == 0) return -1;

    /* Determine number of sub-quantizers */
    if (pq_dim == 0) {
        /* Auto: aim for roughly dimension/4 sub-quantizers, min 1 */
        pq->m = dimension / 4;
        if (pq->m == 0) pq->m = 1;
        /* Ensure m divides dimension evenly */
        while (pq->m > 1 && dimension % pq->m != 0) {
            pq->m--;
        }
    } else {
        pq->m = pq_dim;
        if (dimension % pq->m != 0) {
            /* Adjust to nearest valid divisor */
            while (pq->m > 1 && dimension % pq->m != 0) {
                pq->m--;
            }
        }
    }

    pq->dsub = dimension / pq->m;
    pq->ksub = DISKANN_PQ_KSUB;
    if (count < pq->ksub) {
        pq->ksub = count;
    }

    /* Allocate codebooks: m sub-quantizers, each with ksub centroids of dsub dimensions */
    pq->codebooks = (float *)calloc(pq->m * pq->ksub * pq->dsub, sizeof(float));
    if (!pq->codebooks) return -1;

    /* Temporary storage for sub-vectors */
    float *subvecs = (float *)malloc(count * pq->dsub * sizeof(float));
    if (!subvecs) {
        free(pq->codebooks);
        pq->codebooks = NULL;
        return -1;
    }

    uint32_t *assignments = (uint32_t *)malloc(count * sizeof(uint32_t));
    if (!assignments) {
        free(subvecs);
        free(pq->codebooks);
        pq->codebooks = NULL;
        return -1;
    }

    /* Train each sub-quantizer with K-means (Lloyd's algorithm) */
    for (size_t mi = 0; mi < pq->m; mi++) {
        float *subcodebook = &pq->codebooks[mi * pq->ksub * pq->dsub];

        /* Extract sub-vectors */
        for (size_t i = 0; i < count; i++) {
            memcpy(&subvecs[i * pq->dsub],
                   &data[i * dimension + mi * pq->dsub],
                   pq->dsub * sizeof(float));
        }

        /* Initialize centroids: spread evenly across training data */
        for (size_t k = 0; k < pq->ksub && k < count; k++) {
            size_t idx = (k * count) / pq->ksub;
            memcpy(&subcodebook[k * pq->dsub], &subvecs[idx * pq->dsub],
                   pq->dsub * sizeof(float));
        }

        /* Pad remaining centroids with zeros */
        for (size_t k = count; k < pq->ksub; k++) {
            memset(&subcodebook[k * pq->dsub], 0, pq->dsub * sizeof(float));
        }

        /* 10 iterations of Lloyd's algorithm */
        size_t train_iters = 10;
        for (size_t iter = 0; iter < train_iters; iter++) {
            /* Assignment step */
            for (size_t i = 0; i < count; i++) {
                float min_dist = FLT_MAX;
                uint32_t best_k = 0;
                for (size_t k = 0; k < pq->ksub; k++) {
                    float dist = diskann_l2_distance(&subvecs[i * pq->dsub],
                                                     &subcodebook[k * pq->dsub],
                                                     pq->dsub);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_k = (uint32_t)k;
                    }
                }
                assignments[i] = best_k;
            }

            /* Update step */
            float *new_centroids = (float *)calloc(pq->ksub * pq->dsub, sizeof(float));
            uint32_t *counts = (uint32_t *)calloc(pq->ksub, sizeof(uint32_t));
            if (!new_centroids || !counts) {
                free(new_centroids);
                free(counts);
                break;
            }

            for (size_t i = 0; i < count; i++) {
                uint32_t k = assignments[i];
                counts[k]++;
                for (size_t d = 0; d < pq->dsub; d++) {
                    new_centroids[k * pq->dsub + d] += subvecs[i * pq->dsub + d];
                }
            }

            for (size_t k = 0; k < pq->ksub; k++) {
                if (counts[k] > 0) {
                    for (size_t d = 0; d < pq->dsub; d++) {
                        subcodebook[k * pq->dsub + d] = new_centroids[k * pq->dsub + d] / counts[k];
                    }
                }
            }

            free(new_centroids);
            free(counts);
        }
    }

    free(assignments);
    free(subvecs);
    pq->trained = 1;
    return 0;
}

static void diskann_pq_encode(const DiskANN_PQ *pq, const float *vec, uint8_t *code) {
    for (size_t mi = 0; mi < pq->m; mi++) {
        const float *subvec = &vec[mi * pq->dsub];
        const float *subcodebook = &pq->codebooks[mi * pq->ksub * pq->dsub];

        float min_dist = FLT_MAX;
        uint8_t best_code = 0;

        for (size_t k = 0; k < pq->ksub; k++) {
            float dist = diskann_l2_distance(subvec, &subcodebook[k * pq->dsub], pq->dsub);
            if (dist < min_dist) {
                min_dist = dist;
                best_code = (uint8_t)k;
            }
        }

        code[mi] = best_code;
    }
}

static float diskann_pq_distance(const DiskANN_PQ *pq, const float *query, const uint8_t *code) {
    float total = 0.0f;
    for (size_t mi = 0; mi < pq->m; mi++) {
        const float *subquery = &query[mi * pq->dsub];
        const float *centroid = &pq->codebooks[mi * pq->ksub * pq->dsub + code[mi] * pq->dsub];
        total += diskann_l2_distance(subquery, centroid, pq->dsub);
    }
    return total;
}

static void diskann_pq_destroy(DiskANN_PQ *pq) {
    free(pq->codebooks);
    pq->codebooks = NULL;
    pq->trained = 0;
}

/*  LRU Disk Page Cache Implementation  */

static void diskann_cache_init(DiskANN_Cache *cache, size_t max_mb,
                                size_t vectors_per_page, size_t dimension) {
    memset(cache, 0, sizeof(DiskANN_Cache));
    cache->page_data_bytes = vectors_per_page * dimension * sizeof(float);
    cache->vectors_per_page = vectors_per_page;
    if (cache->page_data_bytes > 0) {
        cache->max_pages = (max_mb * 1024 * 1024) / cache->page_data_bytes;
    }
    if (cache->max_pages == 0) {
        cache->max_pages = 1;
    }
}

static size_t diskann_cache_bucket(size_t page_id) {
    return page_id % DISKANN_CACHE_BUCKETS;
}

static void diskann_cache_lru_remove(DiskANN_Cache *cache, DiskANN_CachePage *page) {
    if (page->lru_prev) page->lru_prev->lru_next = page->lru_next;
    else cache->lru_head = page->lru_next;

    if (page->lru_next) page->lru_next->lru_prev = page->lru_prev;
    else cache->lru_tail = page->lru_prev;

    page->lru_prev = NULL;
    page->lru_next = NULL;
}

static void diskann_cache_lru_push_front(DiskANN_Cache *cache, DiskANN_CachePage *page) {
    page->lru_prev = NULL;
    page->lru_next = cache->lru_head;
    if (cache->lru_head) cache->lru_head->lru_prev = page;
    cache->lru_head = page;
    if (!cache->lru_tail) cache->lru_tail = page;
}

static float *diskann_cache_get(DiskANN_Cache *cache, size_t page_id) {
    size_t bi = diskann_cache_bucket(page_id);
    DiskANN_CachePage *cur = cache->buckets[bi];

    while (cur) {
        if (cur->page_id == page_id) {
            /* Cache hit: move to front */
            diskann_cache_lru_remove(cache, cur);
            diskann_cache_lru_push_front(cache, cur);
            cache->hits++;
            return cur->data;
        }
        cur = cur->hash_next;
    }

    cache->misses++;
    return NULL;
}

static void diskann_cache_evict_lru(DiskANN_Cache *cache) {
    DiskANN_CachePage *victim = cache->lru_tail;
    if (!victim) return;

    /* Remove from LRU list */
    diskann_cache_lru_remove(cache, victim);

    /* Remove from hash bucket */
    size_t bi = diskann_cache_bucket(victim->page_id);
    DiskANN_CachePage *prev = NULL;
    DiskANN_CachePage *cur = cache->buckets[bi];
    while (cur) {
        if (cur == victim) {
            if (prev) prev->hash_next = cur->hash_next;
            else cache->buckets[bi] = cur->hash_next;
            break;
        }
        prev = cur;
        cur = cur->hash_next;
    }

    free(victim->data);
    free(victim);
    cache->count--;
}

static void diskann_cache_put(DiskANN_Cache *cache, size_t page_id, const float *data) {
    /* Check if already cached */
    size_t bi = diskann_cache_bucket(page_id);
    DiskANN_CachePage *cur = cache->buckets[bi];
    while (cur) {
        if (cur->page_id == page_id) {
            /* Update existing */
            memcpy(cur->data, data, cache->page_data_bytes);
            diskann_cache_lru_remove(cache, cur);
            diskann_cache_lru_push_front(cache, cur);
            return;
        }
        cur = cur->hash_next;
    }

    /* Evict if full */
    while (cache->count >= cache->max_pages) {
        diskann_cache_evict_lru(cache);
    }

    /* Allocate new page */
    DiskANN_CachePage *page = (DiskANN_CachePage *)calloc(1, sizeof(DiskANN_CachePage));
    if (!page) return;

    page->page_id = page_id;
    page->data = (float *)malloc(cache->page_data_bytes);
    if (!page->data) {
        free(page);
        return;
    }
    memcpy(page->data, data, cache->page_data_bytes);

    /* Insert into hash bucket */
    page->hash_next = cache->buckets[bi];
    cache->buckets[bi] = page;

    /* Add to LRU front */
    diskann_cache_lru_push_front(cache, page);
    cache->count++;
}

static void diskann_cache_destroy(DiskANN_Cache *cache) {
    for (size_t i = 0; i < DISKANN_CACHE_BUCKETS; i++) {
        DiskANN_CachePage *cur = cache->buckets[i];
        while (cur) {
            DiskANN_CachePage *next = cur->hash_next;
            free(cur->data);
            free(cur);
            cur = next;
        }
        cache->buckets[i] = NULL;
    }
    cache->lru_head = NULL;
    cache->lru_tail = NULL;
    cache->count = 0;
}

/*  Disk Storage (Sector-Aligned Pages via pread/pwrite)  */

static int diskann_disk_open(GV_DiskANNIndex *index) {
    if (!index->data_path) return -1;

    index->data_fd = open(index->data_path, O_RDWR | O_CREAT, 0644);
    if (index->data_fd < 0) return -1;

    return 0;
}

static int diskann_disk_write_vector(GV_DiskANNIndex *index, size_t vec_index, const float *data) {
    if (index->data_fd < 0) return -1;

    size_t vec_bytes = index->dimension * sizeof(float);
    /* Align offset to sector boundaries: each vector occupies a full sector-aligned slot */
    size_t slot_size = ((vec_bytes + index->sector_size - 1) / index->sector_size) * index->sector_size;
    off_t offset = (off_t)(vec_index * slot_size);

    /* Write via pwrite for thread safety */
    size_t written = 0;
    while (written < vec_bytes) {
        ssize_t ret = pwrite(index->data_fd, (const char *)data + written,
                             vec_bytes - written, offset + (off_t)written);
        if (ret < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        written += (size_t)ret;
    }

    return 0;
}

static int diskann_disk_read_vector(GV_DiskANNIndex *index, size_t vec_index, float *out) {
    if (index->data_fd < 0) return -1;

    /* Check the LRU cache first */
    size_t page_id = vec_index / index->vectors_per_page;
    size_t offset_in_page = vec_index % index->vectors_per_page;

    float *cached_page = diskann_cache_get(&index->cache, page_id);
    if (cached_page) {
        memcpy(out, &cached_page[offset_in_page * index->dimension],
               index->dimension * sizeof(float));
        return 0;
    }

    /* Cache miss: read the full page from disk */
    index->disk_reads++;

    size_t vec_bytes = index->dimension * sizeof(float);
    size_t slot_size = ((vec_bytes + index->sector_size - 1) / index->sector_size) * index->sector_size;
    size_t page_vectors = index->vectors_per_page;
    size_t page_data_floats = page_vectors * index->dimension;

    float *page_buf = (float *)calloc(page_data_floats, sizeof(float));
    if (!page_buf) {
        /* Fallback: read single vector */
        off_t file_offset = (off_t)(vec_index * slot_size);
        size_t nread = 0;
        while (nread < vec_bytes) {
            ssize_t ret = pread(index->data_fd, (char *)out + nread,
                                vec_bytes - nread, file_offset + (off_t)nread);
            if (ret < 0) {
                if (errno == EINTR) continue;
                return -1;
            }
            if (ret == 0) break; /* EOF */
            nread += (size_t)ret;
        }
        return (nread >= vec_bytes) ? 0 : -1;
    }

    /* Read all vectors in this page */
    for (size_t vi = 0; vi < page_vectors; vi++) {
        size_t global_vi = page_id * page_vectors + vi;
        if (global_vi >= index->count) break;

        off_t file_offset = (off_t)(global_vi * slot_size);
        size_t nread = 0;
        while (nread < vec_bytes) {
            ssize_t ret = pread(index->data_fd, (char *)&page_buf[vi * index->dimension] + nread,
                                vec_bytes - nread, file_offset + (off_t)nread);
            if (ret < 0) {
                if (errno == EINTR) continue;
                free(page_buf);
                return -1;
            }
            if (ret == 0) break;
            nread += (size_t)ret;
        }
    }

    /* Store page in cache */
    diskann_cache_put(&index->cache, page_id, page_buf);

    /* Copy out the requested vector */
    memcpy(out, &page_buf[offset_in_page * index->dimension],
           index->dimension * sizeof(float));

    free(page_buf);
    return 0;
}

static void diskann_disk_close(GV_DiskANNIndex *index) {
    if (index->data_fd >= 0) {
        close(index->data_fd);
        index->data_fd = -1;
    }
}

/*  Medoid Computation  */

static size_t diskann_compute_medoid(const float *data, size_t count, size_t dimension) {
    if (count == 0) return 0;

    /* Compute centroid */
    float *centroid = (float *)calloc(dimension, sizeof(float));
    if (!centroid) return 0;

    for (size_t i = 0; i < count; i++) {
        for (size_t d = 0; d < dimension; d++) {
            centroid[d] += data[i * dimension + d];
        }
    }
    for (size_t d = 0; d < dimension; d++) {
        centroid[d] /= (float)count;
    }

    /* Find vector closest to centroid */
    size_t best = 0;
    float best_dist = FLT_MAX;
    for (size_t i = 0; i < count; i++) {
        float dist = diskann_l2_distance(&data[i * dimension], centroid, dimension);
        if (dist < best_dist) {
            best_dist = dist;
            best = i;
        }
    }

    free(centroid);
    return best;
}

/*  Greedy Search on Vamana Graph  */

/**
 * Beam search starting from medoid. Returns indices of visited nodes sorted by
 * distance to query. The visited array is caller-allocated and sized to max_visited.
 * Returns the number of results written to visited[].
 */
static int diskann_greedy_search(const GV_DiskANNIndex *index, const float *query,
                                  size_t beam_width, size_t *visited, size_t *visited_count,
                                  size_t max_visited) {
    if (index->count == 0) {
        *visited_count = 0;
        return 0;
    }

    /* Bitset for tracking visited nodes */
    size_t bitset_size = (index->count + 63) / 64;
    uint64_t *seen = (uint64_t *)calloc(bitset_size, sizeof(uint64_t));
    if (!seen) return -1;

    /* Candidate list (sorted by distance) */
    size_t cand_cap = beam_width * 4;
    if (cand_cap < 256) cand_cap = 256;
    DiskANN_Candidate *candidates = (DiskANN_Candidate *)malloc(cand_cap * sizeof(DiskANN_Candidate));
    if (!candidates) {
        free(seen);
        return -1;
    }
    size_t cand_count = 0;

    /* Temporary buffer for reading vectors from disk */
    float *vec_buf = (float *)malloc(index->dimension * sizeof(float));
    if (!vec_buf) {
        free(candidates);
        free(seen);
        return -1;
    }

    /* Start from medoid */
    size_t start = index->medoid;
    if (start >= index->count) start = 0;

    /* Compute distance to medoid using PQ for initial estimate,
       then read exact vector for candidates that get close */
    float start_dist;
    if (index->pq.trained && index->nodes[start].pq_code) {
        start_dist = diskann_pq_distance(&index->pq, query, index->nodes[start].pq_code);
    } else {
        /* Read from disk */
        if (diskann_disk_read_vector((GV_DiskANNIndex *)index, start, vec_buf) == 0) {
            start_dist = diskann_l2_distance(query, vec_buf, index->dimension);
        } else {
            start_dist = FLT_MAX;
        }
    }

    candidates[0].index = start;
    candidates[0].distance = start_dist;
    cand_count = 1;
    seen[start / 64] |= (1ULL << (start % 64));

    size_t result_count = 0;
    size_t explore_idx = 0;

    while (explore_idx < cand_count) {
        size_t curr = candidates[explore_idx].index;
        explore_idx++;

        if (index->nodes[curr].deleted) continue;

        /* Record this as a visited/result node */
        if (result_count < max_visited) {
            visited[result_count++] = curr;
        }

        /* Expand neighbors */
        const DiskANN_Node *node = &index->nodes[curr];
        for (size_t ni = 0; ni < node->neighbor_count; ni++) {
            size_t neighbor = node->neighbors[ni];
            if (neighbor >= index->count) continue;

            /* Check if already visited */
            if (seen[neighbor / 64] & (1ULL << (neighbor % 64))) continue;
            seen[neighbor / 64] |= (1ULL << (neighbor % 64));

            if (index->nodes[neighbor].deleted) continue;

            /* Compute distance using PQ codes for fast estimate */
            float dist;
            if (index->pq.trained && index->nodes[neighbor].pq_code) {
                dist = diskann_pq_distance(&index->pq, query, index->nodes[neighbor].pq_code);
            } else {
                if (diskann_disk_read_vector((GV_DiskANNIndex *)index, neighbor, vec_buf) == 0) {
                    dist = diskann_l2_distance(query, vec_buf, index->dimension);
                } else {
                    continue;
                }
            }

            /* Insert into sorted candidate list if within beam width */
            if (cand_count < cand_cap) {
                candidates[cand_count].index = neighbor;
                candidates[cand_count].distance = dist;
                cand_count++;
                qsort(candidates, cand_count, sizeof(DiskANN_Candidate), diskann_cand_compare);
            } else if (dist < candidates[cand_count - 1].distance) {
                candidates[cand_count - 1].index = neighbor;
                candidates[cand_count - 1].distance = dist;
                qsort(candidates, cand_count, sizeof(DiskANN_Candidate), diskann_cand_compare);
            }

            /* Trim candidate list to beam_width * 2 to keep exploration bounded */
            if (cand_count > beam_width * 2) {
                cand_count = beam_width * 2;
            }
        }
    }

    *visited_count = result_count;

    free(vec_buf);
    free(candidates);
    free(seen);
    return 0;
}

/*  Robust Pruning (Vamana Algorithm)  */

/**
 * Prune the neighbor list of node_id using the alpha-based robust pruning rule.
 * candidates[] and distances[] contain candidate neighbors and their distances
 * to node_id. After pruning, node_id's neighbor list is updated in-place.
 */
static void diskann_robust_prune(GV_DiskANNIndex *index, size_t node_id,
                                  size_t *candidates, float *distances, size_t cand_count) {
    DiskANN_Node *node = &index->nodes[node_id];
    size_t max_degree = index->max_degree;
    float alpha = index->alpha;

    /* Sort candidates by distance to node_id */
    /* Simple selection sort (candidate lists are small) */
    for (size_t i = 0; i < cand_count; i++) {
        size_t min_idx = i;
        for (size_t j = i + 1; j < cand_count; j++) {
            if (distances[j] < distances[min_idx]) {
                min_idx = j;
            }
        }
        if (min_idx != i) {
            size_t tmp_idx = candidates[i];
            candidates[i] = candidates[min_idx];
            candidates[min_idx] = tmp_idx;

            float tmp_dist = distances[i];
            distances[i] = distances[min_idx];
            distances[min_idx] = tmp_dist;
        }
    }

    /* Prune: greedily select neighbors */
    size_t *pruned = (size_t *)malloc(max_degree * sizeof(size_t));
    if (!pruned) return;

    size_t pruned_count = 0;
    int *removed = (int *)calloc(cand_count, sizeof(int));
    if (!removed) {
        free(pruned);
        return;
    }

    /* Temporary buffer for reading vectors for inter-neighbor distance computation */
    float *vec_a = (float *)malloc(index->dimension * sizeof(float));
    float *vec_b = (float *)malloc(index->dimension * sizeof(float));
    if (!vec_a || !vec_b) {
        free(vec_a);
        free(vec_b);
        free(removed);
        free(pruned);
        return;
    }

    for (size_t i = 0; i < cand_count && pruned_count < max_degree; i++) {
        if (removed[i]) continue;
        if (candidates[i] == node_id) continue;
        if (index->nodes[candidates[i]].deleted) continue;

        pruned[pruned_count++] = candidates[i];

        /* Remove candidates that are closer to this newly added neighbor
           than to node_id (scaled by alpha) */
        for (size_t j = i + 1; j < cand_count; j++) {
            if (removed[j]) continue;
            if (candidates[j] == node_id) continue;

            /* Compute distance between candidate[j] and the new neighbor */
            float inter_dist;
            if (index->pq.trained &&
                index->nodes[candidates[i]].pq_code &&
                index->nodes[candidates[j]].pq_code) {
                /* Use PQ codes read into vec_a to compute approximate distance */
                if (diskann_disk_read_vector(index, candidates[i], vec_a) == 0 &&
                    diskann_disk_read_vector(index, candidates[j], vec_b) == 0) {
                    inter_dist = diskann_l2_distance(vec_a, vec_b, index->dimension);
                } else {
                    continue;
                }
            } else {
                if (diskann_disk_read_vector(index, candidates[i], vec_a) == 0 &&
                    diskann_disk_read_vector(index, candidates[j], vec_b) == 0) {
                    inter_dist = diskann_l2_distance(vec_a, vec_b, index->dimension);
                } else {
                    continue;
                }
            }

            /* Robust pruning condition: if dist(candidate_j, new_neighbor) * alpha <= dist(candidate_j, node),
               then candidate_j is redundant */
            if (inter_dist * alpha <= distances[j]) {
                removed[j] = 1;
            }
        }
    }

    free(vec_a);
    free(vec_b);
    free(removed);

    /* Update node's neighbor list */
    free(node->neighbors);
    node->neighbors = pruned;
    node->neighbor_count = pruned_count;
}

/*  Configuration  */

void gv_diskann_config_init(GV_DiskANNConfig *config) {
    if (!config) return;
    config->max_degree = DISKANN_DEFAULT_DEGREE;
    config->alpha = DISKANN_DEFAULT_ALPHA;
    config->build_beam_width = DISKANN_DEFAULT_BUILD_BW;
    config->search_beam_width = DISKANN_DEFAULT_SEARCH_BW;
    config->pq_dim = 0;
    config->data_path = NULL;
    config->cache_size_mb = DISKANN_DEFAULT_CACHE_MB;
    config->sector_size = DISKANN_DEFAULT_SECTOR;
}

/*  Lifecycle  */

GV_DiskANNIndex *gv_diskann_create(size_t dimension, const GV_DiskANNConfig *config) {
    if (dimension == 0) return NULL;

    GV_DiskANNIndex *index = (GV_DiskANNIndex *)calloc(1, sizeof(GV_DiskANNIndex));
    if (!index) return NULL;

    index->dimension = dimension;
    index->data_fd = -1;

    /* Apply configuration */
    if (config) {
        index->max_degree = config->max_degree > 0 ? config->max_degree : DISKANN_DEFAULT_DEGREE;
        index->alpha = config->alpha > 0.0f ? config->alpha : DISKANN_DEFAULT_ALPHA;
        index->build_beam_width = config->build_beam_width > 0 ? config->build_beam_width : DISKANN_DEFAULT_BUILD_BW;
        index->search_beam_width = config->search_beam_width > 0 ? config->search_beam_width : DISKANN_DEFAULT_SEARCH_BW;
        index->sector_size = config->sector_size > 0 ? config->sector_size : DISKANN_DEFAULT_SECTOR;
    } else {
        index->max_degree = DISKANN_DEFAULT_DEGREE;
        index->alpha = DISKANN_DEFAULT_ALPHA;
        index->build_beam_width = DISKANN_DEFAULT_BUILD_BW;
        index->search_beam_width = DISKANN_DEFAULT_SEARCH_BW;
        index->sector_size = DISKANN_DEFAULT_SECTOR;
    }

    /* Compute vectors per page (how many vectors fit in one sector-aligned page) */
    size_t vec_bytes = dimension * sizeof(float);
    size_t slot_size = ((vec_bytes + index->sector_size - 1) / index->sector_size) * index->sector_size;
    index->vectors_per_page = index->sector_size / slot_size;
    if (index->vectors_per_page == 0) index->vectors_per_page = 1;

    /* Set up data path */
    if (config && config->data_path) {
        index->data_path = (char *)malloc(strlen(config->data_path) + 1);
        if (!index->data_path) {
            free(index);
            return NULL;
        }
        strcpy(index->data_path, config->data_path);
    } else {
        /* Default: create a temp path */
        const char *default_path = "diskann_data.bin";
        index->data_path = (char *)malloc(strlen(default_path) + 1);
        if (!index->data_path) {
            free(index);
            return NULL;
        }
        strcpy(index->data_path, default_path);
    }

    /* Open disk storage */
    if (diskann_disk_open(index) != 0) {
        free(index->data_path);
        free(index);
        return NULL;
    }

    /* Initialize LRU cache */
    size_t cache_mb = (config && config->cache_size_mb > 0) ? config->cache_size_mb : DISKANN_DEFAULT_CACHE_MB;
    diskann_cache_init(&index->cache, cache_mb, index->vectors_per_page, dimension);

    /* Initialize PQ */
    diskann_pq_init(&index->pq);

    /* Initialize node storage */
    index->capacity = DISKANN_INITIAL_CAPACITY;
    index->nodes = (DiskANN_Node *)calloc(index->capacity, sizeof(DiskANN_Node));
    if (!index->nodes) {
        diskann_disk_close(index);
        free(index->data_path);
        free(index);
        return NULL;
    }

    index->count = 0;
    index->medoid = 0;

    return index;
}

void gv_diskann_destroy(GV_DiskANNIndex *index) {
    if (!index) return;

    /* Free graph nodes */
    for (size_t i = 0; i < index->count; i++) {
        free(index->nodes[i].neighbors);
        free(index->nodes[i].pq_code);
    }
    free(index->nodes);

    /* Destroy PQ codebooks */
    diskann_pq_destroy(&index->pq);

    /* Destroy LRU cache */
    diskann_cache_destroy(&index->cache);

    /* Close disk storage */
    diskann_disk_close(index);

    free(index->data_path);
    free(index);
}

/*  Build (Batch Construction with Vamana Algorithm)  */

int gv_diskann_build(GV_DiskANNIndex *index, const float *data, size_t count, size_t dimension) {
    if (!index || !data || count == 0) return -1;
    if (dimension != index->dimension) return -1;

    /* Ensure capacity */
    if (count > index->capacity) {
        DiskANN_Node *new_nodes = (DiskANN_Node *)realloc(index->nodes, count * sizeof(DiskANN_Node));
        if (!new_nodes) return -1;
        memset(&new_nodes[index->capacity], 0, (count - index->capacity) * sizeof(DiskANN_Node));
        index->nodes = new_nodes;
        index->capacity = count;
    }

    index->count = count;

    /* Step 1: Write all vectors to disk */
    for (size_t i = 0; i < count; i++) {
        if (diskann_disk_write_vector(index, i, &data[i * dimension]) != 0) {
            return -1;
        }
    }

    /* Step 2: Compute medoid (entry point) */
    index->medoid = diskann_compute_medoid(data, count, dimension);

    /* Step 3: Train PQ codebooks for in-memory navigation */
    size_t pq_dim = 0;
    /* Determine pq_dim: auto or from config is already handled in pq_train */
    if (diskann_pq_train(&index->pq, data, count, dimension, pq_dim) != 0) {
        /* PQ training failed; continue without PQ (slower but functional) */
    }

    /* Step 4: Encode all vectors with PQ */
    if (index->pq.trained) {
        for (size_t i = 0; i < count; i++) {
            index->nodes[i].pq_code = (uint8_t *)malloc(index->pq.m);
            if (index->nodes[i].pq_code) {
                diskann_pq_encode(&index->pq, &data[i * dimension], index->nodes[i].pq_code);
            }
        }
    }

    /* Step 5: Initialize random graph (each node connects to random neighbors) */
    for (size_t i = 0; i < count; i++) {
        size_t initial_degree = index->max_degree < count ? index->max_degree : count - 1;
        index->nodes[i].neighbors = (size_t *)malloc(initial_degree * sizeof(size_t));
        if (!index->nodes[i].neighbors) {
            index->nodes[i].neighbor_count = 0;
            continue;
        }

        size_t nc = 0;
        /* Connect to nearby nodes (simple initialization: sequential neighbors) */
        for (size_t j = 1; j <= initial_degree && nc < initial_degree; j++) {
            size_t neighbor = (i + j) % count;
            if (neighbor != i) {
                index->nodes[i].neighbors[nc++] = neighbor;
            }
        }
        index->nodes[i].neighbor_count = nc;
        index->nodes[i].deleted = 0;
    }

    /* Step 6: Vamana iteration - for each vector, do greedy search + robust prune */
    size_t *search_results = (size_t *)malloc(index->build_beam_width * 2 * sizeof(size_t));
    float *search_distances = (float *)malloc(index->build_beam_width * 2 * sizeof(float));
    float *vec_buf = (float *)malloc(dimension * sizeof(float));
    if (!search_results || !search_distances || !vec_buf) {
        free(search_results);
        free(search_distances);
        free(vec_buf);
        return -1;
    }

    /* Two passes for better graph quality as per the Vamana paper */
    for (int pass = 0; pass < 2; pass++) {
        for (size_t i = 0; i < count; i++) {
            const float *query = &data[i * dimension];

            /* Greedy search to find neighbors */
            size_t visited_count = 0;
            size_t max_results = index->build_beam_width * 2;
            if (max_results > count) max_results = count;

            if (diskann_greedy_search(index, query, index->build_beam_width,
                                       search_results, &visited_count, max_results) != 0) {
                continue;
            }

            if (visited_count == 0) continue;

            /* Compute distances for all visited nodes */
            size_t valid_count = 0;
            for (size_t vi = 0; vi < visited_count; vi++) {
                if (search_results[vi] == i) continue; /* Skip self */
                if (index->nodes[search_results[vi]].deleted) continue;

                float dist;
                if (index->pq.trained && index->nodes[search_results[vi]].pq_code) {
                    dist = diskann_pq_distance(&index->pq, query, index->nodes[search_results[vi]].pq_code);
                } else {
                    dist = diskann_l2_distance(query, &data[search_results[vi] * dimension], dimension);
                }

                search_results[valid_count] = search_results[vi];
                search_distances[valid_count] = dist;
                valid_count++;
            }

            if (valid_count == 0) continue;

            /* Also include current neighbors as candidates */
            DiskANN_Node *node = &index->nodes[i];
            size_t total_cand = valid_count + node->neighbor_count;
            size_t *all_cands = (size_t *)malloc(total_cand * sizeof(size_t));
            float *all_dists = (float *)malloc(total_cand * sizeof(float));
            if (!all_cands || !all_dists) {
                free(all_cands);
                free(all_dists);
                continue;
            }

            memcpy(all_cands, search_results, valid_count * sizeof(size_t));
            memcpy(all_dists, search_distances, valid_count * sizeof(float));

            size_t ac = valid_count;
            for (size_t ni = 0; ni < node->neighbor_count; ni++) {
                size_t nb = node->neighbors[ni];
                /* Check for duplicate */
                int dup = 0;
                for (size_t ci = 0; ci < ac; ci++) {
                    if (all_cands[ci] == nb) { dup = 1; break; }
                }
                if (dup || nb == i) continue;

                float dist;
                if (index->pq.trained && index->nodes[nb].pq_code) {
                    dist = diskann_pq_distance(&index->pq, query, index->nodes[nb].pq_code);
                } else {
                    dist = diskann_l2_distance(query, &data[nb * dimension], dimension);
                }
                all_cands[ac] = nb;
                all_dists[ac] = dist;
                ac++;
            }

            /* Robust prune to select final neighbors */
            diskann_robust_prune(index, i, all_cands, all_dists, ac);

            free(all_cands);
            free(all_dists);

            /* Ensure bidirectional edges: for each new neighbor, add back-edge */
            for (size_t ni = 0; ni < node->neighbor_count; ni++) {
                size_t nb = node->neighbors[ni];
                DiskANN_Node *nb_node = &index->nodes[nb];

                /* Check if back-edge already exists */
                int exists = 0;
                for (size_t j = 0; j < nb_node->neighbor_count; j++) {
                    if (nb_node->neighbors[j] == i) { exists = 1; break; }
                }

                if (!exists && nb_node->neighbor_count < index->max_degree) {
                    size_t *new_neighbors = (size_t *)realloc(nb_node->neighbors,
                                                               (nb_node->neighbor_count + 1) * sizeof(size_t));
                    if (new_neighbors) {
                        nb_node->neighbors = new_neighbors;
                        nb_node->neighbors[nb_node->neighbor_count++] = i;
                    }
                }
            }
        }
    }

    free(search_results);
    free(search_distances);
    free(vec_buf);

    return 0;
}

/*  Incremental Insert  */

int gv_diskann_insert(GV_DiskANNIndex *index, const float *data, size_t dimension) {
    if (!index || !data) return -1;
    if (dimension != index->dimension) return -1;

    /* Ensure capacity */
    if (index->count >= index->capacity) {
        size_t new_cap = index->capacity * 2;
        DiskANN_Node *new_nodes = (DiskANN_Node *)realloc(index->nodes, new_cap * sizeof(DiskANN_Node));
        if (!new_nodes) return -1;
        memset(&new_nodes[index->capacity], 0, (new_cap - index->capacity) * sizeof(DiskANN_Node));
        index->nodes = new_nodes;
        index->capacity = new_cap;
    }

    size_t new_id = index->count;

    /* Write vector to disk */
    if (diskann_disk_write_vector(index, new_id, data) != 0) return -1;

    /* Initialize node */
    DiskANN_Node *node = &index->nodes[new_id];
    memset(node, 0, sizeof(DiskANN_Node));
    node->deleted = 0;

    /* PQ encode */
    if (index->pq.trained) {
        node->pq_code = (uint8_t *)malloc(index->pq.m);
        if (node->pq_code) {
            diskann_pq_encode(&index->pq, data, node->pq_code);
        }
    }

    index->count++;

    /* If this is the first vector, it becomes the medoid */
    if (index->count == 1) {
        index->medoid = 0;
        node->neighbors = NULL;
        node->neighbor_count = 0;
        return 0;
    }

    /* Greedy search to find nearest neighbors */
    size_t max_results = index->build_beam_width * 2;
    if (max_results > index->count) max_results = index->count;

    size_t *search_results = (size_t *)malloc(max_results * sizeof(size_t));
    float *search_distances = (float *)malloc(max_results * sizeof(float));
    if (!search_results || !search_distances) {
        free(search_results);
        free(search_distances);
        return -1;
    }

    size_t visited_count = 0;
    diskann_greedy_search(index, data, index->build_beam_width,
                           search_results, &visited_count, max_results);

    /* Compute distances */
    size_t valid_count = 0;
    float *vec_buf = (float *)malloc(dimension * sizeof(float));
    if (!vec_buf) {
        free(search_results);
        free(search_distances);
        return -1;
    }

    for (size_t vi = 0; vi < visited_count; vi++) {
        if (search_results[vi] == new_id) continue;
        if (index->nodes[search_results[vi]].deleted) continue;

        float dist;
        if (index->pq.trained && index->nodes[search_results[vi]].pq_code) {
            dist = diskann_pq_distance(&index->pq, data, index->nodes[search_results[vi]].pq_code);
        } else if (diskann_disk_read_vector(index, search_results[vi], vec_buf) == 0) {
            dist = diskann_l2_distance(data, vec_buf, dimension);
        } else {
            continue;
        }

        search_results[valid_count] = search_results[vi];
        search_distances[valid_count] = dist;
        valid_count++;
    }

    free(vec_buf);

    /* Robust prune to select neighbors */
    if (valid_count > 0) {
        diskann_robust_prune(index, new_id, search_results, search_distances, valid_count);
    } else {
        node->neighbors = NULL;
        node->neighbor_count = 0;
    }

    /* Add back-edges */
    for (size_t ni = 0; ni < node->neighbor_count; ni++) {
        size_t nb = node->neighbors[ni];
        if (nb >= index->count) continue;
        DiskANN_Node *nb_node = &index->nodes[nb];

        int exists = 0;
        for (size_t j = 0; j < nb_node->neighbor_count; j++) {
            if (nb_node->neighbors[j] == new_id) { exists = 1; break; }
        }

        if (!exists) {
            if (nb_node->neighbor_count < index->max_degree) {
                size_t *new_neighbors = (size_t *)realloc(nb_node->neighbors,
                                                           (nb_node->neighbor_count + 1) * sizeof(size_t));
                if (new_neighbors) {
                    nb_node->neighbors = new_neighbors;
                    nb_node->neighbors[nb_node->neighbor_count++] = new_id;
                }
            }
            /* If neighbor is already at max degree, we could prune it too,
               but for simplicity we skip the back-edge in that case */
        }
    }

    free(search_results);
    free(search_distances);

    return 0;
}

/*  Search  */

int gv_diskann_search(const GV_DiskANNIndex *index, const float *query, size_t dimension,
                       size_t k, GV_DiskANNResult *results) {
    if (!index || !query || !results || k == 0) return -1;
    if (dimension != index->dimension) return -1;
    if (index->count == 0) return 0;

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    /* Beam search */
    size_t max_visited = index->search_beam_width * 4;
    if (max_visited > index->count) max_visited = index->count;
    if (max_visited < k) max_visited = k;

    size_t *visited = (size_t *)malloc(max_visited * sizeof(size_t));
    if (!visited) return -1;

    size_t visited_count = 0;
    if (diskann_greedy_search(index, query, index->search_beam_width,
                               visited, &visited_count, max_visited) != 0) {
        free(visited);
        return -1;
    }

    /* Refine: read exact vectors from disk for visited candidates and compute true L2 */
    DiskANN_Candidate *refined = (DiskANN_Candidate *)malloc(visited_count * sizeof(DiskANN_Candidate));
    if (!refined) {
        free(visited);
        return -1;
    }

    float *vec_buf = (float *)malloc(dimension * sizeof(float));
    if (!vec_buf) {
        free(refined);
        free(visited);
        return -1;
    }

    size_t refined_count = 0;
    for (size_t i = 0; i < visited_count; i++) {
        size_t vid = visited[i];
        if (vid >= index->count) continue;
        if (index->nodes[vid].deleted) continue;

        if (diskann_disk_read_vector((GV_DiskANNIndex *)index, vid, vec_buf) != 0) continue;

        float dist = diskann_l2_distance(query, vec_buf, dimension);
        refined[refined_count].index = vid;
        refined[refined_count].distance = dist;
        refined_count++;
    }

    free(vec_buf);
    free(visited);

    /* Sort by exact distance */
    qsort(refined, refined_count, sizeof(DiskANN_Candidate), diskann_cand_compare);

    /* Copy top-k results */
    size_t result_count = refined_count < k ? refined_count : k;
    for (size_t i = 0; i < result_count; i++) {
        results[i].index = refined[i].index;
        results[i].distance = refined[i].distance;
    }

    free(refined);

    /* Update latency stats (cast away const for stats tracking) */
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double latency_us = (double)(ts_end.tv_sec - ts_start.tv_sec) * 1e6 +
                         (double)(ts_end.tv_nsec - ts_start.tv_nsec) / 1e3;
    GV_DiskANNIndex *mutable_idx = (GV_DiskANNIndex *)index;
    mutable_idx->total_search_latency_us += latency_us;
    mutable_idx->total_searches++;

    return (int)result_count;
}

/*  Delete (Lazy Tombstone)  */

int gv_diskann_delete(GV_DiskANNIndex *index, size_t vector_index) {
    if (!index) return -1;
    if (vector_index >= index->count) return -1;
    if (index->nodes[vector_index].deleted) return -1;

    index->nodes[vector_index].deleted = 1;

    /* If deleting the medoid, find a new one */
    if (vector_index == index->medoid) {
        for (size_t i = 0; i < index->count; i++) {
            if (!index->nodes[i].deleted) {
                index->medoid = i;
                break;
            }
        }
    }

    return 0;
}

/*  Statistics  */

int gv_diskann_get_stats(const GV_DiskANNIndex *index, GV_DiskANNStats *stats) {
    if (!index || !stats) return -1;

    memset(stats, 0, sizeof(GV_DiskANNStats));

    /* Count active vectors and total graph edges */
    size_t active = 0;
    size_t edges = 0;
    for (size_t i = 0; i < index->count; i++) {
        if (!index->nodes[i].deleted) {
            active++;
            edges += index->nodes[i].neighbor_count;
        }
    }

    stats->total_vectors = active;
    stats->graph_edges = edges;
    stats->cache_hits = index->cache.hits;
    stats->cache_misses = index->cache.misses;
    stats->disk_reads = index->disk_reads;
    stats->avg_search_latency_us = index->total_searches > 0
        ? index->total_search_latency_us / (double)index->total_searches
        : 0.0;

    /* Memory usage estimate: nodes + PQ codes + adjacency lists + cache */
    size_t mem = index->count * sizeof(DiskANN_Node);
    for (size_t i = 0; i < index->count; i++) {
        mem += index->nodes[i].neighbor_count * sizeof(size_t);
        if (index->nodes[i].pq_code && index->pq.trained) {
            mem += index->pq.m;
        }
    }
    if (index->pq.trained) {
        mem += index->pq.m * index->pq.ksub * index->pq.dsub * sizeof(float);
    }
    mem += index->cache.count * (sizeof(DiskANN_CachePage) + index->cache.page_data_bytes);
    stats->memory_usage_bytes = mem;

    /* Disk usage: sector-aligned slots * count */
    size_t vec_bytes = index->dimension * sizeof(float);
    size_t slot_size = ((vec_bytes + index->sector_size - 1) / index->sector_size) * index->sector_size;
    stats->disk_usage_bytes = index->count * slot_size;

    return 0;
}

/*  Count  */

size_t gv_diskann_count(const GV_DiskANNIndex *index) {
    if (!index) return 0;
    size_t active = 0;
    for (size_t i = 0; i < index->count; i++) {
        if (!index->nodes[i].deleted) active++;
    }
    return active;
}

/*  Persistence: Save  */

static int diskann_write_u32(FILE *f, uint32_t v) {
    return fwrite(&v, sizeof(uint32_t), 1, f) == 1 ? 0 : -1;
}

static int diskann_write_u64(FILE *f, uint64_t v) {
    return fwrite(&v, sizeof(uint64_t), 1, f) == 1 ? 0 : -1;
}

static int diskann_write_f32(FILE *f, float v) {
    return fwrite(&v, sizeof(float), 1, f) == 1 ? 0 : -1;
}

static int diskann_write_floats(FILE *f, const float *data, size_t count) {
    return fwrite(data, sizeof(float), count, f) == count ? 0 : -1;
}

static int diskann_write_bytes(FILE *f, const uint8_t *data, size_t count) {
    return fwrite(data, sizeof(uint8_t), count, f) == count ? 0 : -1;
}

int gv_diskann_save(const GV_DiskANNIndex *index, const char *filepath) {
    if (!index || !filepath) return -1;

    FILE *f = fopen(filepath, "wb");
    if (!f) return -1;

    /* Header */
    if (diskann_write_u32(f, DISKANN_MAGIC) != 0) goto fail;
    if (diskann_write_u32(f, DISKANN_VERSION) != 0) goto fail;
    if (diskann_write_u64(f, (uint64_t)index->dimension) != 0) goto fail;
    if (diskann_write_u64(f, (uint64_t)index->count) != 0) goto fail;
    if (diskann_write_u64(f, (uint64_t)index->max_degree) != 0) goto fail;
    if (diskann_write_f32(f, index->alpha) != 0) goto fail;
    if (diskann_write_u64(f, (uint64_t)index->build_beam_width) != 0) goto fail;
    if (diskann_write_u64(f, (uint64_t)index->search_beam_width) != 0) goto fail;
    if (diskann_write_u64(f, (uint64_t)index->sector_size) != 0) goto fail;
    if (diskann_write_u64(f, (uint64_t)index->medoid) != 0) goto fail;

    /* PQ metadata */
    if (diskann_write_u32(f, (uint32_t)index->pq.trained) != 0) goto fail;
    if (index->pq.trained) {
        if (diskann_write_u64(f, (uint64_t)index->pq.m) != 0) goto fail;
        if (diskann_write_u64(f, (uint64_t)index->pq.dsub) != 0) goto fail;
        if (diskann_write_u64(f, (uint64_t)index->pq.ksub) != 0) goto fail;

        /* Write codebooks */
        size_t cb_size = index->pq.m * index->pq.ksub * index->pq.dsub;
        if (diskann_write_floats(f, index->pq.codebooks, cb_size) != 0) goto fail;
    }

    /* Graph: for each node write adjacency list and PQ code */
    for (size_t i = 0; i < index->count; i++) {
        const DiskANN_Node *node = &index->nodes[i];

        if (diskann_write_u32(f, (uint32_t)node->deleted) != 0) goto fail;
        if (diskann_write_u64(f, (uint64_t)node->neighbor_count) != 0) goto fail;

        for (size_t j = 0; j < node->neighbor_count; j++) {
            if (diskann_write_u64(f, (uint64_t)node->neighbors[j]) != 0) goto fail;
        }

        /* PQ code */
        uint32_t has_pq = (node->pq_code && index->pq.trained) ? 1 : 0;
        if (diskann_write_u32(f, has_pq) != 0) goto fail;
        if (has_pq) {
            if (diskann_write_bytes(f, node->pq_code, index->pq.m) != 0) goto fail;
        }
    }

    /* Write data path length and path for reference */
    if (index->data_path) {
        uint32_t path_len = (uint32_t)strlen(index->data_path);
        if (diskann_write_u32(f, path_len) != 0) goto fail;
        if (fwrite(index->data_path, 1, path_len, f) != path_len) goto fail;
    } else {
        if (diskann_write_u32(f, 0) != 0) goto fail;
    }

    fclose(f);
    return 0;

fail:
    fclose(f);
    return -1;
}

/*  Persistence: Load  */

static int diskann_read_u32(FILE *f, uint32_t *v) {
    return (v && fread(v, sizeof(uint32_t), 1, f) == 1) ? 0 : -1;
}

static int diskann_read_u64(FILE *f, uint64_t *v) {
    return (v && fread(v, sizeof(uint64_t), 1, f) == 1) ? 0 : -1;
}

static int diskann_read_f32(FILE *f, float *v) {
    return (v && fread(v, sizeof(float), 1, f) == 1) ? 0 : -1;
}

static int diskann_read_floats(FILE *f, float *data, size_t count) {
    return (data && fread(data, sizeof(float), count, f) == count) ? 0 : -1;
}

static int diskann_read_bytes(FILE *f, uint8_t *data, size_t count) {
    return (data && fread(data, sizeof(uint8_t), count, f) == count) ? 0 : -1;
}

GV_DiskANNIndex *gv_diskann_load(const char *filepath, const GV_DiskANNConfig *config) {
    if (!filepath) return NULL;

    FILE *f = fopen(filepath, "rb");
    if (!f) return NULL;

    /* Read header */
    uint32_t magic = 0, version = 0;
    if (diskann_read_u32(f, &magic) != 0 || magic != DISKANN_MAGIC) goto fail;
    if (diskann_read_u32(f, &version) != 0 || version != DISKANN_VERSION) goto fail;

    uint64_t dimension = 0, count = 0, max_degree = 0;
    uint64_t build_bw = 0, search_bw = 0, sector_size = 0, medoid = 0;
    float alpha = 0.0f;

    if (diskann_read_u64(f, &dimension) != 0) goto fail;
    if (diskann_read_u64(f, &count) != 0) goto fail;
    if (diskann_read_u64(f, &max_degree) != 0) goto fail;
    if (diskann_read_f32(f, &alpha) != 0) goto fail;
    if (diskann_read_u64(f, &build_bw) != 0) goto fail;
    if (diskann_read_u64(f, &search_bw) != 0) goto fail;
    if (diskann_read_u64(f, &sector_size) != 0) goto fail;
    if (diskann_read_u64(f, &medoid) != 0) goto fail;

    if (dimension == 0) goto fail;

    /* Build a config from the file data, overridden by user config where provided */
    GV_DiskANNConfig load_config;
    gv_diskann_config_init(&load_config);
    load_config.max_degree = (size_t)max_degree;
    load_config.alpha = alpha;
    load_config.build_beam_width = (size_t)build_bw;
    load_config.search_beam_width = (size_t)search_bw;
    load_config.sector_size = (size_t)sector_size;

    /* Use caller-provided data_path and cache_size if available */
    if (config) {
        if (config->data_path) load_config.data_path = config->data_path;
        if (config->cache_size_mb > 0) load_config.cache_size_mb = config->cache_size_mb;
        if (config->search_beam_width > 0) load_config.search_beam_width = config->search_beam_width;
    }

    /* Read PQ metadata */
    uint32_t pq_trained = 0;
    if (diskann_read_u32(f, &pq_trained) != 0) goto fail;

    uint64_t pq_m = 0, pq_dsub = 0, pq_ksub = 0;
    float *pq_codebooks = NULL;

    if (pq_trained) {
        if (diskann_read_u64(f, &pq_m) != 0) goto fail;
        if (diskann_read_u64(f, &pq_dsub) != 0) goto fail;
        if (diskann_read_u64(f, &pq_ksub) != 0) goto fail;

        size_t cb_size = (size_t)(pq_m * pq_ksub * pq_dsub);
        pq_codebooks = (float *)malloc(cb_size * sizeof(float));
        if (!pq_codebooks) goto fail;
        if (diskann_read_floats(f, pq_codebooks, cb_size) != 0) {
            free(pq_codebooks);
            goto fail;
        }
    }

    /* We need to read the data_path at the end before creating the index,
       but the graph data comes first. Read graph data into temp storage. */

    /* Allocate temporary arrays for graph data */
    typedef struct {
        uint32_t deleted;
        uint64_t neighbor_count;
        size_t *neighbors;
        uint8_t *pq_code;
    } TempNode;

    TempNode *temp_nodes = NULL;
    if (count > 0) {
        temp_nodes = (TempNode *)calloc((size_t)count, sizeof(TempNode));
        if (!temp_nodes) {
            free(pq_codebooks);
            goto fail;
        }
    }

    for (size_t i = 0; i < (size_t)count; i++) {
        if (diskann_read_u32(f, &temp_nodes[i].deleted) != 0) goto fail_temp;
        if (diskann_read_u64(f, &temp_nodes[i].neighbor_count) != 0) goto fail_temp;

        if (temp_nodes[i].neighbor_count > 0) {
            temp_nodes[i].neighbors = (size_t *)malloc((size_t)temp_nodes[i].neighbor_count * sizeof(size_t));
            if (!temp_nodes[i].neighbors) goto fail_temp;

            for (size_t j = 0; j < (size_t)temp_nodes[i].neighbor_count; j++) {
                uint64_t nb = 0;
                if (diskann_read_u64(f, &nb) != 0) goto fail_temp;
                temp_nodes[i].neighbors[j] = (size_t)nb;
            }
        }

        uint32_t has_pq = 0;
        if (diskann_read_u32(f, &has_pq) != 0) goto fail_temp;
        if (has_pq && pq_trained) {
            temp_nodes[i].pq_code = (uint8_t *)malloc((size_t)pq_m);
            if (!temp_nodes[i].pq_code) goto fail_temp;
            if (diskann_read_bytes(f, temp_nodes[i].pq_code, (size_t)pq_m) != 0) goto fail_temp;
        }
    }

    /* Read stored data path */
    uint32_t stored_path_len = 0;
    if (diskann_read_u32(f, &stored_path_len) != 0) goto fail_temp;

    char *stored_path = NULL;
    if (stored_path_len > 0) {
        stored_path = (char *)malloc(stored_path_len + 1);
        if (!stored_path) goto fail_temp;
        if (fread(stored_path, 1, stored_path_len, f) != stored_path_len) {
            free(stored_path);
            goto fail_temp;
        }
        stored_path[stored_path_len] = '\0';
    }

    /* Use stored path if caller didn't provide one */
    if (!load_config.data_path && stored_path) {
        load_config.data_path = stored_path;
    }

    fclose(f);
    f = NULL;

    /* Create the index */
    GV_DiskANNIndex *index = gv_diskann_create((size_t)dimension, &load_config);
    free(stored_path);

    if (!index) {
        /* Clean up temp nodes */
        if (temp_nodes) {
            for (size_t i = 0; i < (size_t)count; i++) {
                free(temp_nodes[i].neighbors);
                free(temp_nodes[i].pq_code);
            }
            free(temp_nodes);
        }
        free(pq_codebooks);
        return NULL;
    }

    /* Restore PQ */
    if (pq_trained) {
        index->pq.m = (size_t)pq_m;
        index->pq.dsub = (size_t)pq_dsub;
        index->pq.ksub = (size_t)pq_ksub;
        index->pq.codebooks = pq_codebooks;
        index->pq.trained = 1;
    } else {
        free(pq_codebooks);
    }

    /* Ensure capacity */
    if ((size_t)count > index->capacity) {
        DiskANN_Node *new_nodes = (DiskANN_Node *)realloc(index->nodes, (size_t)count * sizeof(DiskANN_Node));
        if (!new_nodes) {
            if (temp_nodes) {
                for (size_t i = 0; i < (size_t)count; i++) {
                    free(temp_nodes[i].neighbors);
                    free(temp_nodes[i].pq_code);
                }
                free(temp_nodes);
            }
            gv_diskann_destroy(index);
            return NULL;
        }
        memset(&new_nodes[index->capacity], 0, ((size_t)count - index->capacity) * sizeof(DiskANN_Node));
        index->nodes = new_nodes;
        index->capacity = (size_t)count;
    }

    /* Copy graph from temp storage */
    index->count = (size_t)count;
    index->medoid = (size_t)medoid;

    for (size_t i = 0; i < (size_t)count; i++) {
        index->nodes[i].deleted = (int)temp_nodes[i].deleted;
        index->nodes[i].neighbor_count = (size_t)temp_nodes[i].neighbor_count;
        index->nodes[i].neighbors = temp_nodes[i].neighbors;
        index->nodes[i].pq_code = temp_nodes[i].pq_code;
        temp_nodes[i].neighbors = NULL; /* Ownership transferred */
        temp_nodes[i].pq_code = NULL;
    }

    free(temp_nodes);
    return index;

fail_temp:
    if (temp_nodes) {
        for (size_t i = 0; i < (size_t)count; i++) {
            free(temp_nodes[i].neighbors);
            free(temp_nodes[i].pq_code);
        }
        free(temp_nodes);
    }
    free(pq_codebooks);
fail:
    if (f) fclose(f);
    return NULL;
}
