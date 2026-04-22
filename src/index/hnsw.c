#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <stdint.h>
#include <limits.h>
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__SSE4_2__)
#include <nmmintrin.h>
#endif

#include "index/hnsw.h"
#include "search/distance.h"
#include "schema/metadata.h"
#include "schema/vector.h"
#include "specialized/binary_quant.h"
#include "storage/soa_storage.h"
#include "core/utils.h"

typedef struct {
    size_t vector_index;             /**< Index into GV_SoAStorage */
    GV_BinaryVector *binary_vector;  /**< Binary quantized version for fast search */
    size_t level;
    int deleted;                     /**< Deletion flag: 1 if deleted, 0 if active */
} GV_HNSWNode;

typedef struct {
    size_t node_idx;     /**< Index into nodes[] array */
    float distance;
} GV_HNSWCandidate;

typedef struct {
    size_t dimension;
    size_t M;
    size_t efConstruction;
    size_t efSearch;
    size_t maxLevel;
    int use_binary_quant;
    size_t quant_rerank;
    int use_acorn;
    size_t acorn_hops;
    GV_DistanceType distance_type;
    size_t entry_point;              /**< Node index of entry point (SIZE_MAX = none) */
    size_t count;
    size_t nodes_capacity;
    GV_HNSWNode *nodes;              /**< Array of node structs */

    int32_t *neighbors;              /**< Flat neighbor array, -1 = empty sentinel */
    size_t *offsets;                 /**< offsets[i] = start of node i's neighbors in neighbors[] */
    size_t neighbors_size;           /**< Total used int32_t slots in neighbors[] */
    size_t neighbors_capacity;       /**< Total allocated int32_t slots in neighbors[] */
    size_t *cum_nb_per_level;        /**< Cumulative neighbor slots per level */
    size_t max_level_alloc;          /**< Size of cum_nb_per_level array */

    GV_SoAStorage *soa_storage;
    int soa_storage_owned;

    uint32_t *visited_epoch;
    uint32_t current_epoch;
    size_t visited_capacity;

    float *search_dis;
    size_t *search_ids;
    uint8_t *search_proc;
    size_t search_buf_size;
    float *insert_dis;
    size_t *insert_ids;
    uint8_t *insert_proc;
    size_t insert_buf_size;
} GV_HNSWIndex;


static size_t calculate_level(size_t maxLevel, size_t M) {
    double r = (double)rand() / ((double)RAND_MAX + 1.0);
    if (r == 0.0) r = 1e-18;
    double mL = 1.0 / log((double)M);
    size_t level = (size_t)(-log(r) * mL);
    return (level > maxLevel) ? maxLevel : level;
}

static int compare_candidates(const void *a, const void *b) {
    const GV_HNSWCandidate *ca = (const GV_HNSWCandidate *)a;
    const GV_HNSWCandidate *cb = (const GV_HNSWCandidate *)b;
    if (ca->distance < cb->distance) return -1;
    if (ca->distance > cb->distance) return 1;
    return 0;
}

static void build_cum_nb_per_level(GV_HNSWIndex *index) {
    for (size_t l = 0; l <= index->max_level_alloc; ++l) {
        if (l == 0) {
            index->cum_nb_per_level[l] = 0;
        } else if (l == 1) {
            index->cum_nb_per_level[l] = 2 * index->M;
        } else {
            index->cum_nb_per_level[l] = index->cum_nb_per_level[l - 1] + index->M;
        }
    }
}

static inline size_t nb_slots_for_level(const GV_HNSWIndex *index, size_t level) {
    if (level + 1 <= index->max_level_alloc) {
        return index->cum_nb_per_level[level + 1];
    }
    return 2 * index->M + level * index->M;
}

static inline int32_t *nb_begin(const GV_HNSWIndex *index, size_t node_idx, size_t level) {
    return index->neighbors + index->offsets[node_idx] + index->cum_nb_per_level[level];
}

static inline size_t max_nb_at_level(const GV_HNSWIndex *index, size_t level) {
    return (level == 0) ? 2 * index->M : index->M;
}

static inline size_t count_neighbors(const GV_HNSWIndex *index, size_t node_idx, size_t level) {
    int32_t *start = nb_begin(index, node_idx, level);
    size_t max_n = max_nb_at_level(index, level);
    size_t cnt = 0;
    for (size_t i = 0; i < max_n; ++i) {
        if (start[i] < 0) break;
        cnt++;
    }
    return cnt;
}

static inline float hnsw_l2_scalar(const float *a, const float *b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

#ifdef __AVX2__
static inline float hnsw_l2_avx2(const float *a, const float *b, size_t dim) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(diff, diff, acc);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    float sum = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    for (; i < dim; ++i) { float d = a[i] - b[i]; sum += d * d; }
    return sum;
}

static inline float hnsw_dot_avx2(const float *a, const float *b, size_t dim) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    float sum = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    for (; i < dim; ++i) sum += a[i] * b[i];
    return sum;
}
#endif

static float hnsw_raw_distance(const float *a, const float *b, size_t dim,
                                   GV_DistanceType dtype) {
    switch (dtype) {
    case GV_DISTANCE_EUCLIDEAN:
#ifdef __AVX2__
        return hnsw_l2_avx2(a, b, dim);
#else
        return hnsw_l2_scalar(a, b, dim);
#endif
    case GV_DISTANCE_COSINE: {
        float dot = 0.0f, na = 0.0f, nb = 0.0f;
#ifdef __AVX2__
        __m256 vdot = _mm256_setzero_ps(), vna = _mm256_setzero_ps(), vnb = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 8 <= dim; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            vdot = _mm256_fmadd_ps(va, vb, vdot);
            vna = _mm256_fmadd_ps(va, va, vna);
            vnb = _mm256_fmadd_ps(vb, vb, vnb);
        }
        float td[8], tna[8], tnb[8];
        _mm256_storeu_ps(td, vdot); _mm256_storeu_ps(tna, vna); _mm256_storeu_ps(tnb, vnb);
        for (int j = 0; j < 8; ++j) { dot += td[j]; na += tna[j]; nb += tnb[j]; }
        for (; i < dim; ++i) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
#else
        for (size_t i = 0; i < dim; ++i) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
#endif
        float denom = sqrtf(na) * sqrtf(nb);
        return (denom > 0.0f) ? (1.0f - dot / denom) : 1.0f;
    }
    case GV_DISTANCE_DOT_PRODUCT:
#ifdef __AVX2__
        return -hnsw_dot_avx2(a, b, dim);
#else
        { float d = 0; for (size_t i = 0; i < dim; ++i) d += a[i]*b[i]; return -d; }
#endif
    case GV_DISTANCE_MANHATTAN: {
        float sum = 0.0f;
        for (size_t i = 0; i < dim; ++i) { float d = a[i] - b[i]; sum += (d < 0) ? -d : d; }
        return sum;
    }
    default:
#ifdef __AVX2__
        return hnsw_l2_avx2(a, b, dim);
#else
        return hnsw_l2_scalar(a, b, dim);
#endif
    }
}

static inline void hnsw_l2_batch4(const float *x,
                                       const float *y0, const float *y1,
                                       const float *y2, const float *y3,
                                       size_t dim,
                                       float *d0, float *d1, float *d2, float *d3) {
#ifdef __AVX2__
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 v0 = _mm256_sub_ps(vx, _mm256_loadu_ps(y0 + i));
        __m256 v1 = _mm256_sub_ps(vx, _mm256_loadu_ps(y1 + i));
        __m256 v2 = _mm256_sub_ps(vx, _mm256_loadu_ps(y2 + i));
        __m256 v3 = _mm256_sub_ps(vx, _mm256_loadu_ps(y3 + i));
        acc0 = _mm256_fmadd_ps(v0, v0, acc0);
        acc1 = _mm256_fmadd_ps(v1, v1, acc1);
        acc2 = _mm256_fmadd_ps(v2, v2, acc2);
        acc3 = _mm256_fmadd_ps(v3, v3, acc3);
    }
    float t0[8], t1[8], t2[8], t3[8];
    _mm256_storeu_ps(t0, acc0); _mm256_storeu_ps(t1, acc1);
    _mm256_storeu_ps(t2, acc2); _mm256_storeu_ps(t3, acc3);
    *d0 = t0[0]+t0[1]+t0[2]+t0[3]+t0[4]+t0[5]+t0[6]+t0[7];
    *d1 = t1[0]+t1[1]+t1[2]+t1[3]+t1[4]+t1[5]+t1[6]+t1[7];
    *d2 = t2[0]+t2[1]+t2[2]+t2[3]+t2[4]+t2[5]+t2[6]+t2[7];
    *d3 = t3[0]+t3[1]+t3[2]+t3[3]+t3[4]+t3[5]+t3[6]+t3[7];
    for (; i < dim; ++i) {
        float q0 = x[i]-y0[i], q1 = x[i]-y1[i], q2 = x[i]-y2[i], q3 = x[i]-y3[i];
        *d0 += q0*q0; *d1 += q1*q1; *d2 += q2*q2; *d3 += q3*q3;
    }
#else
    *d0 = *d1 = *d2 = *d3 = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float q0 = x[i]-y0[i], q1 = x[i]-y1[i], q2 = x[i]-y2[i], q3 = x[i]-y3[i];
        *d0 += q0*q0; *d1 += q1*q1; *d2 += q2*q2; *d3 += q3*q3;
    }
#endif
}

static inline void prefetch_L2(const void *addr) {
#if defined(__SSE4_2__) || defined(__AVX2__)
    _mm_prefetch((const char *)addr, _MM_HINT_T1);
#else
    (void)addr;
#endif
}

void *gv_hnsw_create(size_t dimension, const GV_HNSWConfig *config, GV_SoAStorage *soa_storage) {
    if (dimension == 0) return NULL;

    GV_HNSWIndex *index = (GV_HNSWIndex *)calloc(1, sizeof(GV_HNSWIndex));
    if (!index) return NULL;

    index->dimension = dimension;
    index->M = (config && config->M > 0) ? config->M : 16;
    index->efConstruction = (config && config->efConstruction > 0) ? config->efConstruction : 200;
    index->efSearch = (config && config->efSearch > 0) ? config->efSearch : 50;
    index->maxLevel = (config && config->maxLevel > 0) ? config->maxLevel : 16;
    index->use_binary_quant = (config && config->use_binary_quant) ? 1 : 0;
    index->quant_rerank = (config && config->quant_rerank > 0) ? config->quant_rerank : 0;
    index->use_acorn = (config && config->use_acorn) ? 1 : 0;
    index->acorn_hops = (config && config->acorn_hops > 0 && config->acorn_hops <= 2) ? config->acorn_hops : 1;
    index->distance_type = config ? config->distance_type : GV_DISTANCE_EUCLIDEAN;
    index->entry_point = SIZE_MAX;
    index->count = 0;
    index->nodes_capacity = 1024;

    index->nodes = (GV_HNSWNode *)calloc(index->nodes_capacity, sizeof(GV_HNSWNode));
    if (!index->nodes) { free(index); return NULL; }

    index->max_level_alloc = index->maxLevel + 1;
    index->cum_nb_per_level = (size_t *)calloc(index->max_level_alloc + 1, sizeof(size_t));
    if (!index->cum_nb_per_level) { free(index->nodes); free(index); return NULL; }
    build_cum_nb_per_level(index);

    size_t nb_per_node_l0 = 2 * index->M;
    index->neighbors_capacity = index->nodes_capacity * nb_per_node_l0;
    index->neighbors_size = 0;
    index->neighbors = (int32_t *)malloc(index->neighbors_capacity * sizeof(int32_t));
    index->offsets = (size_t *)calloc(index->nodes_capacity, sizeof(size_t));
    if (!index->neighbors || !index->offsets) {
        free(index->neighbors); free(index->offsets);
        free(index->cum_nb_per_level); free(index->nodes); free(index);
        return NULL;
    }
    memset(index->neighbors, 0xFF, index->neighbors_capacity * sizeof(int32_t));

    if (soa_storage) {
        index->soa_storage = soa_storage;
        index->soa_storage_owned = 0;
    } else {
        index->soa_storage = soa_storage_create(dimension, 1024);
        if (!index->soa_storage) {
            free(index->neighbors); free(index->offsets);
            free(index->cum_nb_per_level); free(index->nodes); free(index);
            return NULL;
        }
        index->soa_storage_owned = 1;
    }

    index->visited_capacity = index->nodes_capacity;
    index->visited_epoch = (uint32_t *)calloc(index->visited_capacity, sizeof(uint32_t));
    if (!index->visited_epoch) {
        if (index->soa_storage_owned) soa_storage_destroy(index->soa_storage);
        free(index->neighbors); free(index->offsets);
        free(index->cum_nb_per_level); free(index->nodes); free(index);
        return NULL;
    }
    index->current_epoch = 0;

    index->search_buf_size = 2 * index->efSearch;
    index->search_dis = (float *)malloc(index->search_buf_size * sizeof(float));
    index->search_ids = (size_t *)malloc(index->search_buf_size * sizeof(size_t));
    index->search_proc = (uint8_t *)malloc(index->search_buf_size * sizeof(uint8_t));
    if (!index->search_dis || !index->search_ids || !index->search_proc) {
        free(index->search_dis); free(index->search_ids); free(index->search_proc);
        free(index->visited_epoch);
        if (index->soa_storage_owned) soa_storage_destroy(index->soa_storage);
        free(index->neighbors); free(index->offsets);
        free(index->cum_nb_per_level); free(index->nodes); free(index);
        return NULL;
    }

    index->insert_buf_size = index->efConstruction;
    index->insert_dis = (float *)malloc(index->insert_buf_size * sizeof(float));
    index->insert_ids = (size_t *)malloc(index->insert_buf_size * sizeof(size_t));
    index->insert_proc = (uint8_t *)malloc(index->insert_buf_size * sizeof(uint8_t));
    if (!index->insert_dis || !index->insert_ids || !index->insert_proc) {
        free(index->insert_dis); free(index->insert_ids); free(index->insert_proc);
        free(index->search_dis); free(index->search_ids); free(index->search_proc);
        free(index->visited_epoch);
        if (index->soa_storage_owned) soa_storage_destroy(index->soa_storage);
        free(index->neighbors); free(index->offsets);
        free(index->cum_nb_per_level); free(index->nodes); free(index);
        return NULL;
    }

    return index;
}

int gv_hnsw_reserve(void *index_ptr, size_t n) {
    if (!index_ptr || n == 0) return -1;
    GV_HNSWIndex *index = (GV_HNSWIndex *)index_ptr;
    size_t total = index->count + n;

    /* Grow nodes + offsets */
    if (total > index->nodes_capacity) {
        size_t new_cap = total + (total >> 2); /* 25% headroom */
        GV_HNSWNode *new_nodes = (GV_HNSWNode *)realloc(index->nodes, new_cap * sizeof(GV_HNSWNode));
        if (!new_nodes) return -1;
        memset(new_nodes + index->nodes_capacity, 0, (new_cap - index->nodes_capacity) * sizeof(GV_HNSWNode));
        index->nodes = new_nodes;

        size_t *new_off = (size_t *)realloc(index->offsets, new_cap * sizeof(size_t));
        if (!new_off) return -1;
        index->offsets = new_off;

        index->nodes_capacity = new_cap;
    }

    /* Grow visited_epoch */
    if (total > index->visited_capacity) {
        size_t new_cap = total + (total >> 2);
        uint32_t *new_vis = (uint32_t *)realloc(index->visited_epoch, new_cap * sizeof(uint32_t));
        if (!new_vis) return -1;
        memset(new_vis + index->visited_capacity, 0, (new_cap - index->visited_capacity) * sizeof(uint32_t));
        index->visited_epoch = new_vis;
        index->visited_capacity = new_cap;
    }

    /* Pre-allocate neighbor slots: most nodes will be level 0 (2*M slots each).
     * Higher levels are rare, so we under-estimate slightly. */
    size_t nb_per_node = 2 * index->M; /* level 0 */
    size_t nb_needed = index->neighbors_size + n * nb_per_node;
    if (nb_needed > index->neighbors_capacity) {
        size_t new_cap = nb_needed + (nb_needed >> 2);
        int32_t *new_nb = (int32_t *)realloc(index->neighbors, new_cap * sizeof(int32_t));
        if (!new_nb) return -1;
        memset(new_nb + index->neighbors_capacity, 0xFF, (new_cap - index->neighbors_capacity) * sizeof(int32_t));
        index->neighbors = new_nb;
        index->neighbors_capacity = new_cap;
    }

    /* Pre-allocate SoA storage */
    if (index->soa_storage_owned && index->soa_storage) {
        GV_SoAStorage *s = index->soa_storage;
        if (total > s->capacity) {
            size_t new_cap = total + (total >> 2);
            float *new_data = (float *)realloc(s->data, new_cap * s->dimension * sizeof(float));
            if (!new_data) return -1;
            s->data = new_data;
            GV_Metadata **new_meta = (GV_Metadata **)realloc(s->metadata, new_cap * sizeof(GV_Metadata *));
            if (!new_meta) return -1;
            memset(new_meta + s->capacity, 0, (new_cap - s->capacity) * sizeof(GV_Metadata *));
            s->metadata = new_meta;
            int *new_del = (int *)realloc(s->deleted, new_cap * sizeof(int));
            if (!new_del) return -1;
            memset(new_del + s->capacity, 0, (new_cap - s->capacity) * sizeof(int));
            s->deleted = new_del;
            s->capacity = new_cap;
        }
    }

    return 0;
}

static int alloc_node_neighbors(GV_HNSWIndex *index, size_t node_idx, size_t level) {
    size_t slots_needed = nb_slots_for_level(index, level);
    size_t offset = index->neighbors_size;
    size_t new_size = offset + slots_needed;

    /* Amortized growth: double capacity when exceeded */
    if (new_size > index->neighbors_capacity) {
        size_t new_cap = index->neighbors_capacity * 2;
        if (new_cap < new_size) new_cap = new_size;
        int32_t *new_nb = (int32_t *)realloc(index->neighbors, new_cap * sizeof(int32_t));
        if (!new_nb) return -1;
        /* Fill new capacity with -1 sentinel */
        memset(new_nb + index->neighbors_capacity, 0xFF, (new_cap - index->neighbors_capacity) * sizeof(int32_t));
        index->neighbors = new_nb;
        index->neighbors_capacity = new_cap;
    }

    index->offsets[node_idx] = offset;
    index->neighbors_size = new_size;
    return 0;
}

static void add_link(GV_HNSWIndex *index, size_t from, size_t to, size_t level,
                      const float *soa_base, size_t soa_dim) {
    int32_t *start = nb_begin(index, from, level);
    size_t max_n = max_nb_at_level(index, level);

    for (size_t i = 0; i < max_n; ++i) {
        if (start[i] < 0) {
            start[i] = (int32_t)to;
            return;
        }
    }

    #define LINK_VEC(vidx) (soa_base + (vidx) * soa_dim)
    const float *from_data = LINK_VEC(index->nodes[from].vector_index);
    float new_dist = hnsw_raw_distance(from_data, LINK_VEC(index->nodes[to].vector_index),
                                            soa_dim, index->distance_type);

    float worst_dist = -1.0f;
    size_t worst_i = 0;

    if (index->distance_type == GV_DISTANCE_EUCLIDEAN && max_n >= 4) {
        const float *bp[4];
        size_t bi_arr[4];
        size_t bi = 0;
        for (size_t i = 0; i < max_n; ++i) {
            int32_t nb = start[i];
            if (nb < 0 || index->nodes[nb].deleted) {
                start[i] = (int32_t)to;
                goto add_link_done;
            }
            bp[bi] = LINK_VEC(index->nodes[nb].vector_index);
            bi_arr[bi] = i;
            bi++;
            if (bi == 4) {
                float d0, d1, d2, d3;
                hnsw_l2_batch4(from_data, bp[0], bp[1], bp[2], bp[3], soa_dim, &d0, &d1, &d2, &d3);
                float dd[4] = {d0, d1, d2, d3};
                for (int r = 0; r < 4; r++) {
                    if (dd[r] > worst_dist) { worst_dist = dd[r]; worst_i = bi_arr[r]; }
                }
                bi = 0;
            }
        }
        for (size_t r = 0; r < bi; ++r) {
            float d = hnsw_raw_distance(from_data, bp[r], soa_dim, index->distance_type);
            if (d > worst_dist) { worst_dist = d; worst_i = bi_arr[r]; }
        }
    } else {
        for (size_t i = 0; i < max_n; ++i) {
            int32_t nb = start[i];
            if (nb < 0 || index->nodes[nb].deleted) {
                start[i] = (int32_t)to;
                goto add_link_done;
            }
            float d = hnsw_raw_distance(from_data, LINK_VEC(index->nodes[nb].vector_index),
                                             soa_dim, index->distance_type);
            if (d > worst_dist) { worst_dist = d; worst_i = i; }
        }
    }

    if (new_dist < worst_dist) {
        start[worst_i] = (int32_t)to;
    }
add_link_done:
    #undef LINK_VEC
    (void)0;
}

static inline void heap_sift_up(float *dis, size_t *ids, uint8_t *proc, size_t i) {
    float val = dis[i];
    size_t id = ids[i];
    uint8_t p = proc[i];
    while (i > 0) {
        size_t parent = (i - 1) >> 1;
        if (dis[parent] >= val) break;
        dis[i] = dis[parent]; ids[i] = ids[parent]; proc[i] = proc[parent];
        i = parent;
    }
    dis[i] = val; ids[i] = id; proc[i] = p;
}

static inline void heap_sift_down(float *dis, size_t *ids, uint8_t *proc, size_t k, size_t i) {
    float val = dis[i];
    size_t id = ids[i];
    uint8_t p = proc[i];
    while (1) {
        size_t c1 = 2 * i + 1;
        size_t c2 = c1 + 1;
        if (c1 >= k) break;
        size_t larger = (c2 < k && dis[c2] > dis[c1]) ? c2 : c1;
        if (val >= dis[larger]) break;
        dis[i] = dis[larger]; ids[i] = ids[larger]; proc[i] = proc[larger];
        i = larger;
    }
    dis[i] = val; ids[i] = id; proc[i] = p;
}

static inline int mmheap_push(float *dis, size_t *ids, uint8_t *proc,
                               size_t *k, size_t capacity,
                               size_t node_idx, float dist) {
    if (*k < capacity) {
        dis[*k] = dist;
        ids[*k] = node_idx;
        proc[*k] = 0;
        heap_sift_up(dis, ids, proc, *k);
        (*k)++;
        return 1;
    } else if (dist < dis[0]) {
        dis[0] = dist;
        ids[0] = node_idx;
        proc[0] = 0;
        heap_sift_down(dis, ids, proc, *k, 0);
        return 1;
    }
    return 0;
}

static inline size_t mmheap_pop_min(float *dis, size_t *ids, uint8_t *proc,
                                     size_t k, float *out_dist) {
    float best_dist = FLT_MAX;
    size_t best_pos = SIZE_MAX;

#ifdef __AVX2__
    __m256 vmin = _mm256_set1_ps(FLT_MAX);
    __m256i vmin_idx = _mm256_set1_epi32(-1);
    __m256i vbase = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i vstep = _mm256_set1_epi32(8);

    size_t i = 0;
    for (; i + 8 <= k; i += 8) {
        __m256 vdis = _mm256_loadu_ps(dis + i);
        uint64_t proc8;
        memcpy(&proc8, proc + i, 8);
        __m256i vproc32 = _mm256_setr_epi32(
            (proc8 >>  0) & 0xFF, (proc8 >>  8) & 0xFF,
            (proc8 >> 16) & 0xFF, (proc8 >> 24) & 0xFF,
            (proc8 >> 32) & 0xFF, (proc8 >> 40) & 0xFF,
            (proc8 >> 48) & 0xFF, (proc8 >> 56) & 0xFF);
        __m256i proc_mask = _mm256_cmpeq_epi32(vproc32, _mm256_setzero_si256());
        __m256 masked_dis = _mm256_blendv_ps(_mm256_set1_ps(FLT_MAX), vdis,
                                              _mm256_castsi256_ps(proc_mask));
        __m256 cmp = _mm256_cmp_ps(masked_dis, vmin, _CMP_LT_OS);
        vmin = _mm256_blendv_ps(vmin, masked_dis, cmp);
        vmin_idx = _mm256_castps_si256(
            _mm256_blendv_ps(_mm256_castsi256_ps(vmin_idx),
                             _mm256_castsi256_ps(vbase), cmp));
        vbase = _mm256_add_epi32(vbase, vstep);
    }
    float lane_dis[8]; int32_t lane_idx[8];
    _mm256_storeu_ps(lane_dis, vmin);
    _mm256_storeu_si256((__m256i *)lane_idx, vmin_idx);
    for (int j = 0; j < 8; j++) {
        if (lane_dis[j] < best_dist) {
            best_dist = lane_dis[j];
            best_pos = (size_t)lane_idx[j];
        }
    }
    for (; i < k; ++i) {
        if (!proc[i] && dis[i] < best_dist) {
            best_dist = dis[i];
            best_pos = i;
        }
    }
#else
    for (size_t i = 0; i < k; ++i) {
        if (!proc[i] && dis[i] < best_dist) {
            best_dist = dis[i];
            best_pos = i;
        }
    }
#endif

    if (best_pos == SIZE_MAX) return SIZE_MAX;
    proc[best_pos] = 1;
    *out_dist = best_dist;
    return ids[best_pos];
}

static int hnsw_insert_impl(GV_HNSWIndex *index, size_t vector_index, size_t dimension);

int gv_hnsw_insert(void *index_ptr, GV_Vector *vector) {
    if (!index_ptr || !vector) return -1;
    GV_HNSWIndex *index = (GV_HNSWIndex *)index_ptr;
    if (vector->dimension != index->dimension) return -1;

    GV_Metadata *metadata = vector->metadata;
    vector->metadata = NULL;
    size_t vector_index = soa_storage_add(index->soa_storage, vector->data, metadata);
    if (vector_index == (size_t)-1) { vector->metadata = metadata; return -1; }

    return hnsw_insert_impl(index, vector_index, vector->dimension);
}

static int hnsw_insert_impl(GV_HNSWIndex *index, size_t vector_index, size_t dimension) {
    size_t level = calculate_level(index->maxLevel, index->M);
    size_t node_idx = index->count;

    if (node_idx >= index->nodes_capacity) {
        size_t new_cap = index->nodes_capacity * 2;
        GV_HNSWNode *new_nodes = (GV_HNSWNode *)realloc(index->nodes, new_cap * sizeof(GV_HNSWNode));
        if (!new_nodes) { soa_storage_mark_deleted(index->soa_storage, vector_index); return -1; }
        index->nodes = new_nodes;
        memset(index->nodes + index->nodes_capacity, 0, (new_cap - index->nodes_capacity) * sizeof(GV_HNSWNode));

        size_t *new_off = (size_t *)realloc(index->offsets, new_cap * sizeof(size_t));
        if (!new_off) { soa_storage_mark_deleted(index->soa_storage, vector_index); return -1; }
        index->offsets = new_off;

        uint32_t *new_vis = (uint32_t *)realloc(index->visited_epoch, new_cap * sizeof(uint32_t));
        if (!new_vis) { soa_storage_mark_deleted(index->soa_storage, vector_index); return -1; }
        memset(new_vis + index->visited_capacity, 0, (new_cap - index->visited_capacity) * sizeof(uint32_t));
        index->visited_epoch = new_vis;
        index->visited_capacity = new_cap;
        index->nodes_capacity = new_cap;
    }

    index->nodes[node_idx].vector_index = vector_index;
    index->nodes[node_idx].binary_vector = NULL;
    index->nodes[node_idx].level = level;
    index->nodes[node_idx].deleted = 0;

    if (index->use_binary_quant) {
        const float *vd = soa_storage_get_data(index->soa_storage, vector_index);
        if (vd) index->nodes[node_idx].binary_vector = binary_quantize(vd, dimension);
    }

    if (alloc_node_neighbors(index, node_idx, level) != 0) {
        soa_storage_mark_deleted(index->soa_storage, vector_index);
        return -1;
    }

    size_t old_entry = index->entry_point;
    if (index->entry_point == SIZE_MAX) {
        index->entry_point = node_idx;
    } else if (level > index->nodes[index->entry_point].level) {
        index->entry_point = node_idx;
    }

    index->count = node_idx + 1;

    if (node_idx == 0) return 0;

    const float *soa_base = index->soa_storage->data;
    const size_t soa_dim = index->dimension;
    #define SOA_VEC_IMPL(vidx) (soa_base + (vidx) * soa_dim)

    const float *new_vec = SOA_VEC_IMPL(vector_index);

    size_t cur = old_entry;
    if (cur == SIZE_MAX) goto insert_impl_done;

    size_t curLevel = index->nodes[cur].level;
    if (curLevel > index->maxLevel) curLevel = index->maxLevel;

    for (int lc = (int)curLevel; lc > (int)level; --lc) {
        if ((size_t)lc > index->nodes[cur].level) continue;
        float cur_dist = hnsw_raw_distance(new_vec, SOA_VEC_IMPL(index->nodes[cur].vector_index), soa_dim, index->distance_type);

        int improved = 1;
        while (improved) {
            improved = 0;
            int32_t *nbs = nb_begin(index, cur, (size_t)lc);
            size_t max_n = max_nb_at_level(index, (size_t)lc);
            for (size_t i = 0; i < max_n; ++i) {
                int32_t nb = nbs[i];
                if (nb < 0) break;
                if (index->nodes[nb].deleted) continue;
                float dist = hnsw_raw_distance(new_vec, SOA_VEC_IMPL(index->nodes[nb].vector_index), soa_dim, index->distance_type);
                if (dist < cur_dist) { cur_dist = dist; cur = (size_t)nb; improved = 1; }
            }
        }
        curLevel = index->nodes[cur].level;
        if (curLevel > index->maxLevel) curLevel = index->maxLevel;
    }

    size_t searchLevel = (curLevel < level) ? curLevel : level;
    if (searchLevel > index->nodes[cur].level) searchLevel = index->nodes[cur].level;
    if (searchLevel > level) searchLevel = level;

    for (int lc = (int)searchLevel; lc >= 0; --lc) {
        if ((size_t)lc > index->nodes[cur].level || (size_t)lc > level) continue;

        index->current_epoch++;
        if (index->current_epoch == 0) {
            memset(index->visited_epoch, 0, index->visited_capacity * sizeof(uint32_t));
            index->current_epoch = 1;
        }
        index->visited_epoch[node_idx] = index->current_epoch;

        float *heap_dis = index->insert_dis;
        size_t *heap_ids = index->insert_ids;
        uint8_t *heap_proc = index->insert_proc;
        size_t heap_k = 0;

        float seed_dist = hnsw_raw_distance(new_vec, SOA_VEC_IMPL(index->nodes[cur].vector_index), soa_dim, index->distance_type);
        mmheap_push(heap_dis, heap_ids, heap_proc, &heap_k, index->efConstruction, cur, seed_dist);
        index->visited_epoch[cur] = index->current_epoch;

        for (;;) {
            float cand_dist;
            size_t cand_node = mmheap_pop_min(heap_dis, heap_ids, heap_proc, heap_k, &cand_dist);
            if (cand_node == SIZE_MAX) break;
            if (index->nodes[cand_node].deleted) continue;
            if ((size_t)lc > index->nodes[cand_node].level) continue;
            if (heap_k >= index->efConstruction && cand_dist > heap_dis[0]) break;

            int32_t *nbs = nb_begin(index, cand_node, (size_t)lc);
            size_t max_n = max_nb_at_level(index, (size_t)lc);

            size_t jmax = 0;
            int32_t valid_nbs[64];
            for (size_t i = 0; i < max_n; ++i) {
                int32_t nb = nbs[i];
                if (nb < 0) break;
                valid_nbs[jmax] = nb;
                prefetch_L2(&index->visited_epoch[nb]);
                jmax++;
            }

            size_t batch_buf_idx = 0;
            size_t batch_nodes[4];
            const float *batch_ptrs[4];

            for (size_t i = 0; i < jmax; ++i) {
                int32_t nb = valid_nbs[i];
                if (index->nodes[nb].deleted) continue;
                if ((size_t)nb >= index->visited_capacity ||
                    index->visited_epoch[nb] == index->current_epoch) continue;
                index->visited_epoch[nb] = index->current_epoch;

                if (i + 1 < jmax) {
                    prefetch_L2(SOA_VEC_IMPL(index->nodes[valid_nbs[i+1]].vector_index));
                }

                batch_nodes[batch_buf_idx] = (size_t)nb;
                batch_ptrs[batch_buf_idx] = SOA_VEC_IMPL(index->nodes[nb].vector_index);
                batch_buf_idx++;

                if (batch_buf_idx == 4 && index->distance_type == GV_DISTANCE_EUCLIDEAN) {
                    float d0, d1, d2, d3;
                    hnsw_l2_batch4(new_vec, batch_ptrs[0], batch_ptrs[1],
                                       batch_ptrs[2], batch_ptrs[3], soa_dim,
                                       &d0, &d1, &d2, &d3);
                    float thresh = (heap_k >= index->efConstruction) ? heap_dis[0] : FLT_MAX;
                    if (d0 < thresh) mmheap_push(heap_dis, heap_ids, heap_proc, &heap_k, index->efConstruction, batch_nodes[0], d0);
                    if (d1 < thresh) mmheap_push(heap_dis, heap_ids, heap_proc, &heap_k, index->efConstruction, batch_nodes[1], d1);
                    if (d2 < thresh) mmheap_push(heap_dis, heap_ids, heap_proc, &heap_k, index->efConstruction, batch_nodes[2], d2);
                    if (d3 < thresh) mmheap_push(heap_dis, heap_ids, heap_proc, &heap_k, index->efConstruction, batch_nodes[3], d3);
                    batch_buf_idx = 0;
                }
            }
            for (size_t r = 0; r < batch_buf_idx; ++r) {
                float dist = hnsw_raw_distance(new_vec, batch_ptrs[r], soa_dim, index->distance_type);
                mmheap_push(heap_dis, heap_ids, heap_proc, &heap_k, index->efConstruction, batch_nodes[r], dist);
            }
        }

        if (heap_k == 0) continue;

        size_t max_nbrs = max_nb_at_level(index, (size_t)lc);
        size_t extract_need = max_nbrs * 3;
        if (extract_need > heap_k) extract_need = heap_k;

        float tmp_dis2[512];
        size_t tmp_ids2[512];
        size_t copy_n = (heap_k < 512) ? heap_k : 512;
        memcpy(tmp_dis2, heap_dis, copy_n * sizeof(float));
        memcpy(tmp_ids2, heap_ids, copy_n * sizeof(size_t));

        GV_HNSWCandidate sorted_cands[128];
        size_t cand_count = 0;
        for (size_t e = 0; e < extract_need && e < 128; ++e) {
            float best_d = FLT_MAX;
            size_t best_i = SIZE_MAX;
            for (size_t i = 0; i < copy_n; ++i) {
                if (tmp_dis2[i] < best_d) { best_d = tmp_dis2[i]; best_i = i; }
            }
            if (best_i == SIZE_MAX) break;
            sorted_cands[cand_count].node_idx = tmp_ids2[best_i];
            sorted_cands[cand_count].distance = best_d;
            cand_count++;
            tmp_dis2[best_i] = FLT_MAX;
        }

        size_t sel_buf[64];
        size_t sel_count = 0;
        for (size_t ci = 0; ci < cand_count && sel_count < max_nbrs; ++ci) {
            size_t cn = sorted_cands[ci].node_idx;
            if (cn == node_idx || index->nodes[cn].deleted) continue;
            float dq = sorted_cands[ci].distance;
            const float *cd = SOA_VEC_IMPL(index->nodes[cn].vector_index);
            int keep = 1;
            for (size_t si = 0; si < sel_count; ++si) {
                float ds = hnsw_raw_distance(cd, SOA_VEC_IMPL(index->nodes[sel_buf[si]].vector_index), soa_dim, index->distance_type);
                if (ds < dq) { keep = 0; break; }
            }
            if (keep) sel_buf[sel_count++] = cn;
        }
        if (sel_count < max_nbrs) {
            for (size_t ci = 0; ci < cand_count && sel_count < max_nbrs; ++ci) {
                size_t cn = sorted_cands[ci].node_idx;
                if (cn == node_idx || index->nodes[cn].deleted) continue;
                int dup = 0;
                for (size_t si = 0; si < sel_count; ++si) {
                    if (sel_buf[si] == cn) { dup = 1; break; }
                }
                if (!dup) sel_buf[sel_count++] = cn;
            }
        }

        for (size_t i = 0; i < sel_count; ++i) {
            size_t sel = sel_buf[i];
            add_link(index, node_idx, sel, (size_t)lc, soa_base, soa_dim);
            if ((size_t)lc <= index->nodes[sel].level) {
                add_link(index, sel, node_idx, (size_t)lc, soa_base, soa_dim);
            }
        }
    }

insert_impl_done:
    #undef SOA_VEC_IMPL
    return 0;
}

/* Raw insert: skip GV_Vector allocation */
int gv_hnsw_insert_raw(void *index_ptr, const float *data, size_t dimension) {
    if (!index_ptr || !data || dimension == 0) return -1;
    GV_HNSWIndex *index = (GV_HNSWIndex *)index_ptr;
    if (dimension != index->dimension) return -1;

    size_t vector_index = soa_storage_add(index->soa_storage, data, NULL);
    if (vector_index == (size_t)-1) return -1;

    return hnsw_insert_impl(index, vector_index, dimension);
}

int gv_hnsw_search(void *index_ptr, const GV_Vector *query, size_t k,
                   GV_SearchResult *results, GV_DistanceType distance_type,
                   const char *filter_key, const char *filter_value) {
    if (!index_ptr || !query || !results || k == 0) return -1;
    GV_HNSWIndex *index = (GV_HNSWIndex *)index_ptr;
    if (query->dimension != index->dimension || index->entry_point == SIZE_MAX) return 0;

    memset(results, 0, k * sizeof(GV_SearchResult));

    GV_BinaryVector *query_binary = NULL;
    if (index->use_binary_quant) {
        query_binary = binary_quantize(query->data, query->dimension);
        if (!query_binary) return -1;
    }

    const float *soa_base = index->soa_storage->data;
    const size_t soa_dim = index->dimension;
    const float *qdata = query->data;
    #define SRCH_VEC(vidx) (soa_base + (vidx) * soa_dim)

    /* Greedy descent through upper layers */
    size_t cur = index->entry_point;
    size_t curLevel = index->nodes[cur].level;

    for (int lc = (int)curLevel; lc > 0; --lc) {
        if ((size_t)lc > index->nodes[cur].level) continue;
        float cur_dist = hnsw_raw_distance(SRCH_VEC(index->nodes[cur].vector_index), qdata, soa_dim, distance_type);

        int improved = 1;
        while (improved) {
            improved = 0;
            int32_t *nbs = nb_begin(index, cur, (size_t)lc);
            size_t max_n = max_nb_at_level(index, (size_t)lc);
            for (size_t i = 0; i < max_n; ++i) {
                int32_t nb = nbs[i];
                if (nb < 0) break;
                if (index->nodes[nb].deleted) continue;
                float dist;
                if (index->use_binary_quant && query_binary && index->nodes[nb].binary_vector) {
                    dist = (float)binary_hamming_distance_fast(query_binary, index->nodes[nb].binary_vector);
                } else {
                    dist = hnsw_raw_distance(SRCH_VEC(index->nodes[nb].vector_index), qdata, soa_dim, distance_type);
                }
                if (dist < cur_dist) { cur_dist = dist; cur = (size_t)nb; improved = 1; }
            }
        }
        curLevel = index->nodes[cur].level;
    }

    /* Level-0 beam search */
    size_t ef = index->efSearch;
    if (filter_key && index->use_acorn) {
        size_t factor = index->acorn_hops + 1;
        if (factor > 3) factor = 3;
        if (ef > 0 && ef <= SIZE_MAX / factor) ef *= factor;
    }

    index->current_epoch++;
    if (index->current_epoch == 0) {
        memset(index->visited_epoch, 0, index->visited_capacity * sizeof(uint32_t));
        index->current_epoch = 1;
    }

    size_t buf_need = ef;
    if (buf_need > index->search_buf_size) {
        float *nd = (float *)realloc(index->search_dis, buf_need * sizeof(float));
        size_t *ni = (size_t *)realloc(index->search_ids, buf_need * sizeof(size_t));
        uint8_t *np = (uint8_t *)realloc(index->search_proc, buf_need * sizeof(uint8_t));
        if (!nd || !ni || !np) {
            if (nd) index->search_dis = nd;
            if (ni) index->search_ids = ni;
            if (np) index->search_proc = np;
            if (query_binary) binary_vector_destroy(query_binary);
            return -1;
        }
        index->search_dis = nd;
        index->search_ids = ni;
        index->search_proc = np;
        index->search_buf_size = buf_need;
    }

    float *heap_dis = index->search_dis;
    size_t *heap_ids = index->search_ids;
    uint8_t *heap_proc = index->search_proc;
    size_t heap_k = 0;

    float cur_dist;
    if (index->use_binary_quant && query_binary && index->nodes[cur].binary_vector) {
        cur_dist = (float)binary_hamming_distance_fast(query_binary, index->nodes[cur].binary_vector);
    } else {
        cur_dist = hnsw_raw_distance(SRCH_VEC(index->nodes[cur].vector_index), qdata, soa_dim, distance_type);
    }
    mmheap_push(heap_dis, heap_ids, heap_proc, &heap_k, buf_need,
                cur, cur_dist);
    index->visited_epoch[cur] = index->current_epoch;

    for (;;) {
        float cand_dist;
        size_t cand_node = mmheap_pop_min(heap_dis, heap_ids, heap_proc,
                                          heap_k, &cand_dist);
        if (cand_node == SIZE_MAX) break;

        if (index->nodes[cand_node].deleted) continue;

        /* Termination: if this candidate is worse than the heap max, stop */
        if (heap_k >= ef && cand_dist > heap_dis[0]) break;

        int32_t *nbs = nb_begin(index, cand_node, 0);
        size_t max_n = max_nb_at_level(index, 0);

        /* Pass 1: collect valid neighbors + prefetch visited table */
        size_t jmax = 0;
        int32_t valid_nbs[64];
        for (size_t i = 0; i < max_n; ++i) {
            int32_t nb = nbs[i];
            if (nb < 0) break;
            valid_nbs[jmax] = nb;
            prefetch_L2(&index->visited_epoch[nb]);
            jmax++;
        }

        /* Pass 2: batch-4 distance computation with visited check */
        size_t batch_cnt = 0;
        size_t batch_nids[4];
        const float *batch_ptrs[4];

        for (size_t i = 0; i < jmax; ++i) {
            int32_t nb = valid_nbs[i];
            if (index->nodes[nb].deleted) continue;
            if ((size_t)nb >= index->visited_capacity ||
                index->visited_epoch[nb] == index->current_epoch) continue;
            index->visited_epoch[nb] = index->current_epoch;

            /* Prefetch next vector data */
            if (i + 1 < jmax) {
                prefetch_L2(SRCH_VEC(index->nodes[valid_nbs[i+1]].vector_index));
            }

            if (index->use_binary_quant && query_binary && index->nodes[nb].binary_vector) {
                float dist = (float)binary_hamming_distance_fast(query_binary, index->nodes[nb].binary_vector);
                mmheap_push(heap_dis, heap_ids, heap_proc, &heap_k, buf_need,
                            (size_t)nb, dist);
                continue;
            }

            batch_nids[batch_cnt] = (size_t)nb;
            batch_ptrs[batch_cnt] = SRCH_VEC(index->nodes[nb].vector_index);
            batch_cnt++;

            if (batch_cnt == 4 && distance_type == GV_DISTANCE_EUCLIDEAN) {
                float d0, d1, d2, d3;
                hnsw_l2_batch4(qdata, batch_ptrs[0], batch_ptrs[1],
                                   batch_ptrs[2], batch_ptrs[3], soa_dim,
                                   &d0, &d1, &d2, &d3);
                /* Quick reject: skip heap push if distance >= max and heap full */
                float thresh = (heap_k >= buf_need) ? heap_dis[0] : FLT_MAX;
                if (d0 < thresh) mmheap_push(heap_dis, heap_ids, heap_proc, &heap_k, buf_need,
                            batch_nids[0], d0);
                if (d1 < thresh) mmheap_push(heap_dis, heap_ids, heap_proc, &heap_k, buf_need,
                            batch_nids[1], d1);
                if (d2 < thresh) mmheap_push(heap_dis, heap_ids, heap_proc, &heap_k, buf_need,
                            batch_nids[2], d2);
                if (d3 < thresh) mmheap_push(heap_dis, heap_ids, heap_proc, &heap_k, buf_need,
                            batch_nids[3], d3);
                batch_cnt = 0;
            }
        }
        /* Remainder */
        for (size_t r = 0; r < batch_cnt; ++r) {
            float dist = hnsw_raw_distance(batch_ptrs[r], qdata, soa_dim, distance_type);
            mmheap_push(heap_dis, heap_ids, heap_proc, &heap_k, buf_need,
                        batch_nids[r], dist);
        }
    }

    if (heap_k == 0) {
        if (query_binary) binary_vector_destroy(query_binary);
        return 0;
    }

    /* Extract top-k from max-heap via repeated min-extraction */
    size_t need = k;
    if (index->use_binary_quant && index->quant_rerank > 0 && query_binary) {
        need = (index->quant_rerank > k) ? index->quant_rerank : k;
    }
    if (need > heap_k) need = heap_k;

    /* Copy heap data to temp arrays for extraction (don't modify the pre-allocated buffers) */
    float tmp_dis[1024];
    size_t tmp_ids[1024];
    size_t cand_count = (heap_k < 1024) ? heap_k : 1024;
    memcpy(tmp_dis, heap_dis, cand_count * sizeof(float));
    memcpy(tmp_ids, heap_ids, cand_count * sizeof(size_t));

    GV_HNSWCandidate sorted_cands[512]; /* enough for k + rerank */
    size_t sorted_count = 0;

    /* Extract 'need' smallest elements */
    for (size_t e = 0; e < need && e < 512; ++e) {
        float best_d = FLT_MAX;
        size_t best_i = SIZE_MAX;
        for (size_t i = 0; i < cand_count; ++i) {
            if (tmp_dis[i] < best_d) {
                best_d = tmp_dis[i];
                best_i = i;
            }
        }
        if (best_i == SIZE_MAX) break;
        sorted_cands[sorted_count].node_idx = tmp_ids[best_i];
        sorted_cands[sorted_count].distance = best_d;
        sorted_count++;
        tmp_dis[best_i] = FLT_MAX; /* mark as extracted */
    }

    if (index->use_binary_quant && index->quant_rerank > 0 && query_binary) {
        for (size_t i = 0; i < sorted_count; ++i) {
            if (!index->nodes[sorted_cands[i].node_idx].deleted) {
                sorted_cands[i].distance = hnsw_raw_distance(
                    SRCH_VEC(index->nodes[sorted_cands[i].node_idx].vector_index), qdata, soa_dim, distance_type);
            }
        }
        qsort(sorted_cands, sorted_count, sizeof(GV_HNSWCandidate), compare_candidates);
    }

    /* Build results */
    size_t result_count = 0;
    for (size_t i = 0; i < sorted_count && result_count < k; ++i) {
        size_t cn = sorted_cands[i].node_idx;
        if (index->nodes[cn].deleted) continue;

        const float *node_data = SRCH_VEC(index->nodes[cn].vector_index);
        GV_Metadata *node_meta = soa_storage_get_metadata(index->soa_storage, index->nodes[cn].vector_index);

        if (filter_key && !metadata_get_direct(node_meta, filter_key)) continue;
        if (filter_key && strcmp(metadata_get_direct(node_meta, filter_key), filter_value) != 0) continue;

        GV_Vector *result_vec = (GV_Vector *)malloc(sizeof(GV_Vector));
        if (!result_vec) continue;
        result_vec->data = (float *)malloc(soa_dim * sizeof(float));
        if (!result_vec->data) { free(result_vec); continue; }
        memcpy(result_vec->data, node_data, soa_dim * sizeof(float));
        result_vec->dimension = soa_dim;
        result_vec->metadata = metadata_copy(node_meta);
        results[result_count].vector = result_vec;
        results[result_count].distance = sorted_cands[i].distance;
        results[result_count].id = index->nodes[cn].vector_index;
        result_count++;
    }

    if (query_binary) binary_vector_destroy(query_binary);
    #undef SRCH_VEC
    return (int)result_count;
}

void gv_hnsw_destroy(void *index_ptr) {
    if (!index_ptr) return;
    GV_HNSWIndex *index = (GV_HNSWIndex *)index_ptr;
    for (size_t i = 0; i < index->count; ++i) {
        if (index->nodes[i].binary_vector)
            binary_vector_destroy(index->nodes[i].binary_vector);
    }
    free(index->nodes);
    free(index->neighbors);
    free(index->offsets);
    free(index->cum_nb_per_level);
    if (index->soa_storage && index->soa_storage_owned)
        soa_storage_destroy(index->soa_storage);
    free(index->visited_epoch);
    free(index->search_dis);
    free(index->search_ids);
    free(index->search_proc);
    free(index->insert_dis);
    free(index->insert_ids);
    free(index->insert_proc);
    free(index);
}

size_t gv_hnsw_count(const void *index_ptr) {
    if (!index_ptr) return 0;
    return ((GV_HNSWIndex *)index_ptr)->count;
}

int gv_hnsw_delete(void *index_ptr, size_t node_index) {
    if (!index_ptr) return -1;
    GV_HNSWIndex *index = (GV_HNSWIndex *)index_ptr;
    if (node_index >= index->count) return -1;
    if (index->nodes[node_index].deleted) return -1;

    index->nodes[node_index].deleted = 1;

    /* Remove from neighbor lists of other nodes */
    for (size_t i = 0; i < index->count; ++i) {
        if (index->nodes[i].deleted || i == node_index) continue;
        for (size_t l = 0; l <= index->nodes[i].level; ++l) {
            int32_t *start = nb_begin(index, i, l);
            size_t max_n = max_nb_at_level(index, l);
            size_t wp = 0;
            for (size_t j = 0; j < max_n; ++j) {
                if (start[j] < 0) break;
                if ((size_t)start[j] != node_index) {
                    start[wp++] = start[j];
                }
            }
            for (size_t j = wp; j < max_n; ++j) start[j] = -1;
        }
    }

    if (index->entry_point == node_index) {
        index->entry_point = SIZE_MAX;
        for (size_t i = 0; i < index->count; ++i) {
            if (!index->nodes[i].deleted) { index->entry_point = i; break; }
        }
    }
    return 0;
}

int gv_hnsw_update(void *index_ptr, size_t node_index, const float *new_data, size_t dimension) {
    if (!index_ptr || !new_data) return -1;
    GV_HNSWIndex *index = (GV_HNSWIndex *)index_ptr;
    if (node_index >= index->count || dimension != index->dimension) return -1;
    if (index->nodes[node_index].deleted) return -1;

    if (soa_storage_update_data(index->soa_storage, index->nodes[node_index].vector_index, new_data) != 0)
        return -1;

    if (index->use_binary_quant) {
        if (index->nodes[node_index].binary_vector)
            binary_vector_destroy(index->nodes[node_index].binary_vector);
        index->nodes[node_index].binary_vector = binary_quantize(new_data, dimension);
        if (!index->nodes[node_index].binary_vector) return -1;
    }
    return 0;
}

static int read_metadata(FILE *in, GV_Vector *vec) {
    if (!vec) return -1;
    uint32_t count = 0;
    if (read_u32(in, &count) != 0) return -1;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t kl = 0, vl = 0;
        char *key = NULL, *value = NULL;
        if (read_u32(in, &kl) != 0 || read_str(in, &key, kl) != 0) { free(key); return -1; }
        if (read_u32(in, &vl) != 0 || read_str(in, &value, vl) != 0) { free(key); free(value); return -1; }
        if (vector_set_metadata(vec, key, value) != 0) { free(key); free(value); return -1; }
        free(key); free(value);
    }
    return 0;
}

int gv_hnsw_save(const void *index_ptr, FILE *out, uint32_t version) {
    if (!index_ptr || !out) return -1;
    GV_HNSWIndex *index = (GV_HNSWIndex *)index_ptr;

    if (write_u32(out, (uint32_t)index->M) != 0) return -1;
    if (write_u32(out, (uint32_t)index->efConstruction) != 0) return -1;
    if (write_u32(out, (uint32_t)index->efSearch) != 0) return -1;
    if (write_u32(out, (uint32_t)index->maxLevel) != 0) return -1;
    if (write_u64(out, (uint64_t)index->count) != 0) return -1;

    uint64_t ep = (index->entry_point == SIZE_MAX) ? UINT64_MAX : (uint64_t)index->entry_point;
    if (write_u64(out, ep) != 0) return -1;

    /* Pass 1: node data */
    for (size_t i = 0; i < index->count; ++i) {
        if (write_u32(out, (uint32_t)index->nodes[i].level) != 0) return -1;
        const float *vd = soa_storage_get_data(index->soa_storage, index->nodes[i].vector_index);
        if (!vd) return -1;
        if (write_floats(out, vd, index->dimension) != 0) return -1;
        if (version >= 2) {
            GV_Metadata *meta = soa_storage_get_metadata(index->soa_storage, index->nodes[i].vector_index);
            if (write_metadata(out, meta) != 0) return -1;
        }
    }

    /* Pass 2: neighbor connectivity */
    for (size_t i = 0; i < index->count; ++i) {
        for (size_t l = 0; l <= index->nodes[i].level; ++l) {
            size_t nc = count_neighbors(index, i, l);
            if (write_u32(out, (uint32_t)nc) != 0) return -1;
            int32_t *start = nb_begin(index, i, l);
            for (size_t j = 0; j < nc; ++j) {
                if (write_u64(out, (uint64_t)start[j]) != 0) return -1;
            }
        }
    }
    return 0;
}

int gv_hnsw_load(void **index_ptr, FILE *in, size_t dimension, uint32_t version) {
    if (!index_ptr || !in || dimension == 0) return -1;

    uint32_t M = 0, efConstruction = 0, efSearch = 0, maxLevel = 0;
    uint64_t count = 0, entry_point_idx = 0;
    if (read_u32(in, &M) != 0) return -1;
    if (read_u32(in, &efConstruction) != 0) return -1;
    if (read_u32(in, &efSearch) != 0) return -1;
    if (read_u32(in, &maxLevel) != 0) return -1;
    if (read_u64(in, &count) != 0) return -1;
    if (read_u64(in, &entry_point_idx) != 0) return -1;

    GV_HNSWConfig config = {.M = M, .efConstruction = efConstruction, .efSearch = efSearch, .maxLevel = maxLevel};
    void *idx = gv_hnsw_create(dimension, &config, NULL);
    if (!idx) return -1;
    GV_HNSWIndex *hnsw = (GV_HNSWIndex *)idx;

    if (count == 0) { *index_ptr = idx; return 0; }

    /* Grow if needed */
    if (count > hnsw->nodes_capacity) {
        size_t nc = (size_t)count;
        GV_HNSWNode *nn = (GV_HNSWNode *)realloc(hnsw->nodes, nc * sizeof(GV_HNSWNode));
        if (!nn) { gv_hnsw_destroy(idx); return -1; }
        hnsw->nodes = nn;
        size_t *no = (size_t *)realloc(hnsw->offsets, nc * sizeof(size_t));
        if (!no) { gv_hnsw_destroy(idx); return -1; }
        hnsw->offsets = no;
        uint32_t *nv = (uint32_t *)realloc(hnsw->visited_epoch, nc * sizeof(uint32_t));
        if (!nv) { gv_hnsw_destroy(idx); return -1; }
        memset(nv + hnsw->visited_capacity, 0, (nc - hnsw->visited_capacity) * sizeof(uint32_t));
        hnsw->visited_epoch = nv;
        hnsw->visited_capacity = nc;
        hnsw->nodes_capacity = nc;
    }

    /* Read nodes */
    for (size_t i = 0; i < (size_t)count; ++i) {
        uint32_t lev = 0;
        if (read_u32(in, &lev) != 0) { gv_hnsw_destroy(idx); return -1; }

        float *vd = (float *)malloc(dimension * sizeof(float));
        if (!vd) { gv_hnsw_destroy(idx); return -1; }
        if (read_floats(in, vd, dimension) != 0) { free(vd); gv_hnsw_destroy(idx); return -1; }

        GV_Metadata *meta = NULL;
        if (version >= 2) {
            GV_Vector tmp = {.data = vd, .dimension = dimension, .metadata = NULL};
            if (read_metadata(in, &tmp) != 0) { free(vd); gv_hnsw_destroy(idx); return -1; }
            meta = tmp.metadata;
        }

        size_t vi = soa_storage_add(hnsw->soa_storage, vd, meta);
        free(vd);
        if (vi == (size_t)-1) { gv_hnsw_destroy(idx); return -1; }

        hnsw->nodes[i].vector_index = vi;
        hnsw->nodes[i].binary_vector = NULL;
        hnsw->nodes[i].level = lev;
        hnsw->nodes[i].deleted = 0;

        if (alloc_node_neighbors(hnsw, i, lev) != 0) { gv_hnsw_destroy(idx); return -1; }
    }
    hnsw->count = (size_t)count;

    /* Read neighbor connectivity */
    for (size_t i = 0; i < (size_t)count; ++i) {
        for (size_t l = 0; l <= hnsw->nodes[i].level; ++l) {
            uint32_t nc = 0;
            if (read_u32(in, &nc) != 0) { gv_hnsw_destroy(idx); return -1; }
            int32_t *start = nb_begin(hnsw, i, l);
            size_t max_n = max_nb_at_level(hnsw, l);
            for (uint32_t j = 0; j < nc && j < (uint32_t)max_n; ++j) {
                uint64_t ni = 0;
                if (read_u64(in, &ni) != 0) { gv_hnsw_destroy(idx); return -1; }
                if (ni >= count) { gv_hnsw_destroy(idx); return -1; }
                start[j] = (int32_t)ni;
            }
            /* Skip excess */
            for (uint32_t j = (uint32_t)max_n; j < nc; ++j) {
                uint64_t dummy;
                read_u64(in, &dummy);
            }
        }
    }

    if (entry_point_idx != UINT64_MAX && entry_point_idx < count) {
        hnsw->entry_point = (size_t)entry_point_idx;
    } else if (count > 0) {
        hnsw->entry_point = 0;
    }

    *index_ptr = idx;
    return 0;
}

int gv_hnsw_range_search(void *index_ptr, const GV_Vector *query, float radius,
                         GV_SearchResult *results, size_t max_results,
                         GV_DistanceType distance_type,
                         const char *filter_key, const char *filter_value) {
    GV_HNSWIndex *index = (GV_HNSWIndex *)index_ptr;
    if (!index || !query || !results || max_results == 0 || radius < 0.0f ||
        query->dimension != index->dimension || !query->data) return -1;
    if (index->count == 0 || index->entry_point == SIZE_MAX) return 0;

    GV_SearchResult *tmp = (GV_SearchResult *)malloc(max_results * sizeof(GV_SearchResult));
    if (!tmp) return -1;

    int n = gv_hnsw_search(index_ptr, query, max_results, tmp, distance_type, filter_key, filter_value);
    if (n <= 0) { free(tmp); return n; }

    size_t out = 0;
    for (int i = 0; i < n; ++i) {
        if (tmp[i].distance <= radius && out < max_results) {
            results[out++] = tmp[i];
        } else {
            if (tmp[i].vector) vector_destroy((GV_Vector *)tmp[i].vector);
        }
    }
    free(tmp);
    return (int)out;
}
