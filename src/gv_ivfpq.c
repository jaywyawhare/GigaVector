#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

#include "gigavector/gv_ivfpq.h"
#include "gigavector/gv_distance.h"
#include "gigavector/gv_config.h"
#include "gigavector/gv_vector.h"
#include "gigavector/gv_metadata.h"

/* Bitwise CRC32 (poly 0xEDB88320) */
static uint32_t gv_crc32_init(void) { return 0xFFFFFFFFu; }
static uint32_t gv_crc32_update(uint32_t crc, const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    for (size_t i = 0; i < len; ++i) {
        crc ^= p[i];
        for (int k = 0; k < 8; ++k) {
            crc = (crc >> 1) ^ (0xEDB88320u & -(int)(crc & 1u));
        }
    }
    return crc;
}
static uint32_t gv_crc32_finish(uint32_t crc) { return crc ^ 0xFFFFFFFFu; }

static int gv_ivfpq_write_u32(FILE *f, uint32_t v) {
    return fwrite(&v, sizeof(uint32_t), 1, f) == 1 ? 0 : -1;
}

static int gv_ivfpq_write_str(FILE *f, const char *s, uint32_t len) {
    if (gv_ivfpq_write_u32(f, len) != 0) return -1;
    if (len == 0) return 0;
    return fwrite(s, 1, len, f) == len ? 0 : -1;
}

static int gv_ivfpq_read_u32(FILE *f, uint32_t *v) {
    return (v && fread(v, sizeof(uint32_t), 1, f) == 1) ? 0 : -1;
}

static int gv_ivfpq_read_str(FILE *f, char **s, uint32_t len) {
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

struct GV_IVFPQEntry;
typedef struct {
    float dist;
    struct GV_IVFPQEntry *entry;
} GV_IVFPQHeapItem;

static void gv_ivfpq_heap_sift_down(GV_IVFPQHeapItem *heap, size_t size, size_t idx) {
    while (1) {
        size_t l = idx * 2 + 1;
        size_t r = l + 1;
        size_t largest = idx;
        if (l < size && heap[l].dist > heap[largest].dist) largest = l;
        if (r < size && heap[r].dist > heap[largest].dist) largest = r;
        if (largest == idx) break;
        GV_IVFPQHeapItem tmp = heap[idx];
        heap[idx] = heap[largest];
        heap[largest] = tmp;
        idx = largest;
    }
}

#if defined(__AVX512F__)
static inline float gv_ivfpq_l2_avx512(const float *a, const float *b, size_t dim) {
    __m512 acc = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 diff = _mm512_sub_ps(va, vb);
        acc = _mm512_fmadd_ps(diff, diff, acc);
    }
    float tmp[16];
    _mm512_storeu_ps(tmp, acc);
    float sum = 0.0f;
    for (int t = 0; t < 16; ++t) sum += tmp[t];
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

static inline float gv_ivfpq_dot_avx512(const float *a, const float *b, size_t dim) {
    __m512 acc = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        acc = _mm512_fmadd_ps(va, vb, acc);
    }
    float tmp[16];
    _mm512_storeu_ps(tmp, acc);
    float sum = 0.0f;
    for (int t = 0; t < 16; ++t) sum += tmp[t];
    for (; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}
#endif

#if defined(__AVX2__)
static inline float gv_ivfpq_l2_avx2(const float *a, const float *b, size_t dim) {
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
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

static inline float gv_ivfpq_dot_avx2(const float *a, const float *b, size_t dim) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}
#endif

static inline unsigned int gv_ivfpq_cpu_features(void) {
    static unsigned int cached = 0;
    if (cached == 0) {
        cached = gv_cpu_detect_features();
    }
    return cached;
}

static inline float gv_ivfpq_l2_scalar(const float *a, const float *b, size_t dim) {
    float dist = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        dist += d * d;
    }
    return dist;
}

static inline float gv_ivfpq_dot_scalar(const float *a, const float *b, size_t dim) {
    float v = 0.0f;
    for (size_t i = 0; i < dim; ++i) v += a[i] * b[i];
    return v;
}

static inline float gv_ivfpq_l2_runtime(const float *a, const float *b, size_t dim) {
    unsigned int feats = gv_ivfpq_cpu_features();
    if (feats & GV_CPU_FEATURE_AVX512F) {
#ifdef __AVX512F__
        return gv_ivfpq_l2_avx512(a, b, dim);
#else
        return gv_ivfpq_l2_scalar(a, b, dim);
#endif
    }
    if (feats & GV_CPU_FEATURE_AVX2) {
#ifdef __AVX2__
        return gv_ivfpq_l2_avx2(a, b, dim);
#else
        return gv_ivfpq_l2_scalar(a, b, dim);
#endif
    }
    return gv_ivfpq_l2_scalar(a, b, dim);
}

static inline float gv_ivfpq_dot_runtime(const float *a, const float *b, size_t dim) {
    unsigned int feats = gv_ivfpq_cpu_features();
    if (feats & GV_CPU_FEATURE_AVX512F) {
#ifdef __AVX512F__
        return gv_ivfpq_dot_avx512(a, b, dim);
#else
        return gv_ivfpq_dot_scalar(a, b, dim);
#endif
    }
    if (feats & GV_CPU_FEATURE_AVX2) {
#ifdef __AVX2__
        return gv_ivfpq_dot_avx2(a, b, dim);
#else
        return gv_ivfpq_dot_scalar(a, b, dim);
#endif
    }
    return gv_ivfpq_dot_scalar(a, b, dim);
}

static void gv_ivfpq_heap_sift_up(GV_IVFPQHeapItem *heap, size_t idx) {
    while (idx > 0) {
        size_t parent = (idx - 1) / 2;
        if (heap[parent].dist >= heap[idx].dist) break;
        GV_IVFPQHeapItem tmp = heap[parent];
        heap[parent] = heap[idx];
        heap[idx] = tmp;
        idx = parent;
    }
}


typedef struct GV_IVFPQEntry {
    uint8_t *codes;        /* length m */
    GV_Vector *vector;     /* original vector for output */
    GV_ScalarQuantVector *scalar_quant; /* scalar quantized version if enabled */
    int deleted;           /* Deletion flag: 1 if deleted, 0 if active */
} GV_IVFPQEntry;

typedef struct {
    GV_IVFPQEntry *entries; /* AoS for metadata + vector */
    uint8_t *codes_soa;     /* SoA codes: layout m blocks, each block length capacity */
    size_t count;
    size_t capacity;
} GV_IVFPQList;

typedef struct {
    size_t dimension;
    size_t nlist;
    size_t m;
    size_t subdim;
    uint8_t nbits;
    size_t codebook_size; /* 1 << nbits */
    size_t nprobe;
    size_t train_iters;
    int trained;
    size_t default_rerank;
    int use_cosine;
    int use_scalar_quant;
    GV_ScalarQuantConfig scalar_quant_config;
    GV_ScalarQuantVector *scalar_quant_template; /* template with min/max from training */
    float oversampling_factor; /* Factor to oversample candidates before reranking */
    float *coarse;   /* nlist * dimension */
    float *pq;       /* m * codebook_size * subdim */
    GV_IVFPQList *lists;
    pthread_rwlock_t rwlock;
    pthread_mutex_t *list_mutex;
    size_t count;
    /* scratch buffers reused across searches to reduce allocs */
    float *lut_buf;
    size_t lut_buf_size; /* elements */
} GV_IVFPQIndex;

static int gv_ivfpq_argmin(const float *queries, size_t qcount, size_t dim, const float *centroids, size_t ccount, int *assign) {
    for (size_t qi = 0; qi < qcount; ++qi) {
        const float *q = queries + qi * dim;
        float best = INFINITY;
        int best_id = -1;
        for (size_t ci = 0; ci < ccount; ++ci) {
            const float *c = centroids + ci * dim;
            float d = 0.0f;
            for (size_t j = 0; j < dim; ++j) {
                float diff = q[j] - c[j];
                d += diff * diff;
            }
            if (d < best) {
                best = d;
                best_id = (int)ci;
            }
        }
        assign[qi] = best_id;
    }
    return 0;
}

static int gv_ivfpq_kmeans(float *data, size_t n, size_t dim, size_t k, size_t iters, float *out_centroids) {
    if (n < k) return -1;
    /* init: pick first k vectors */
    memcpy(out_centroids, data, k * dim * sizeof(float));
    int *assign = (int *)malloc(n * sizeof(int));
    if (!assign) return -1;
    float *newc = (float *)calloc(k * dim, sizeof(float));
    size_t *counts = (size_t *)calloc(k, sizeof(size_t));
    if (!newc || !counts) {
        free(assign);
        free(newc);
        free(counts);
        return -1;
    }

    for (size_t it = 0; it < iters; ++it) {
        /* assign */
        gv_ivfpq_argmin(data, n, dim, out_centroids, k, assign);
        memset(newc, 0, k * dim * sizeof(float));
        memset(counts, 0, k * sizeof(size_t));
        for (size_t i = 0; i < n; ++i) {
            int c = assign[i];
            if (c < 0) continue;
            const float *v = data + i * dim;
            for (size_t j = 0; j < dim; ++j) newc[c * dim + j] += v[j];
            counts[c]++;
        }
        for (size_t c = 0; c < k; ++c) {
            if (counts[c] == 0) continue;
            for (size_t j = 0; j < dim; ++j) newc[c * dim + j] /= (float)counts[c];
        }
        memcpy(out_centroids, newc, k * dim * sizeof(float));
    }

    free(assign);
    free(newc);
    free(counts);
    return 0;
}

void *gv_ivfpq_create(size_t dimension, const GV_IVFPQConfig *config) {
    if (dimension == 0) return NULL;
    GV_IVFPQIndex *idx = (GV_IVFPQIndex *)calloc(1, sizeof(GV_IVFPQIndex));
    if (!idx) return NULL;
    idx->dimension = dimension;
    /* heuristic defaults */
    size_t def_m = (dimension >= 64) ? 8 : (dimension >= 32 ? 4 : 2);
    size_t def_nlist = 64;
    idx->nlist = (config && config->nlist) ? config->nlist : def_nlist;
    idx->m = (config && config->m) ? config->m : def_m;
    idx->nbits = (config && config->nbits) ? config->nbits : 8;
    idx->nprobe = (config && config->nprobe) ? config->nprobe : 4;
    idx->train_iters = (config && config->train_iters) ? config->train_iters : 15;
    idx->default_rerank = (config && config->default_rerank) ? config->default_rerank : 32;
    idx->use_cosine = (config ? config->use_cosine : 0);
    idx->use_scalar_quant = (config && config->use_scalar_quant) ? 1 : 0;
    idx->oversampling_factor = (config && config->oversampling_factor > 0.0f) ? config->oversampling_factor : 1.0f;
    if (idx->use_scalar_quant && config) {
        idx->scalar_quant_config = config->scalar_quant_config;
        idx->scalar_quant_template = NULL;
    } else {
        idx->scalar_quant_config.bits = 0;
        idx->scalar_quant_config.per_dimension = 0;
        idx->scalar_quant_template = NULL;
    }
    if (idx->m == 0 || idx->dimension % idx->m != 0) {
        free(idx);
        return NULL;
    }
    idx->subdim = idx->dimension / idx->m;
    if (idx->nbits == 0 || idx->nbits > 16) {
        free(idx);
        return NULL;
    }
    idx->codebook_size = 1u << idx->nbits;
    if (idx->nprobe == 0 || idx->nprobe > idx->nlist) {
        idx->nprobe = idx->nlist;
    }
    idx->coarse = (float *)malloc(idx->nlist * idx->dimension * sizeof(float));
    /* PQ codebooks stored AoS (unchanged) */
    idx->pq = (float *)malloc(idx->m * idx->codebook_size * idx->subdim * sizeof(float));
    idx->lists = (GV_IVFPQList *)calloc(idx->nlist, sizeof(GV_IVFPQList));
    idx->list_mutex = (pthread_mutex_t *)calloc(idx->nlist, sizeof(pthread_mutex_t));
    idx->lut_buf = NULL;
    idx->lut_buf_size = 0;
    if (!idx->coarse || !idx->pq || !idx->lists || !idx->list_mutex) {
        free(idx->coarse);
        free(idx->pq);
        free(idx->lists);
        free(idx->list_mutex);
        free(idx);
        return NULL;
    }
    pthread_rwlock_init(&idx->rwlock, NULL);
    for (size_t i = 0; i < idx->nlist; ++i) {
        pthread_mutex_init(&idx->list_mutex[i], NULL);
    }
    return idx;
}

int gv_ivfpq_train(void *index_ptr, const float *data, size_t count) {
    GV_IVFPQIndex *idx = (GV_IVFPQIndex *)index_ptr;
    if (idx == NULL || data == NULL) {
        return -1;
    }
    pthread_rwlock_wrlock(&idx->rwlock);
    if (count < idx->nlist || count < idx->codebook_size || count < idx->m) {
        pthread_rwlock_unlock(&idx->rwlock);
        return -1;
    }
    size_t total = count * idx->dimension;
    float *train_buf = (float *)malloc(total * sizeof(float));
    if (!train_buf) {
        pthread_rwlock_unlock(&idx->rwlock);
        return -1;
    }
    memcpy(train_buf, data, total * sizeof(float));
    if (idx->use_cosine) {
        for (size_t i = 0; i < count; ++i) {
            float norm = 0.0f;
            float *v = train_buf + i * idx->dimension;
            for (size_t j = 0; j < idx->dimension; ++j) norm += v[j] * v[j];
            if (norm > 0.0f) {
                norm = 1.0f / sqrtf(norm);
                for (size_t j = 0; j < idx->dimension; ++j) v[j] *= norm;
            }
        }
    }
    if (gv_ivfpq_kmeans(train_buf, count, idx->dimension, idx->nlist, idx->train_iters, idx->coarse) != 0) {
        free(train_buf);
        pthread_rwlock_unlock(&idx->rwlock);
        return -1;
    }
    /* train each subquantizer */
    float *subbuf = (float *)malloc(count * idx->subdim * sizeof(float));
    if (!subbuf) return -1;
    for (size_t m = 0; m < idx->m; ++m) {
        for (size_t i = 0; i < count; ++i) {
            memcpy(subbuf + i * idx->subdim, train_buf + i * idx->dimension + m * idx->subdim, idx->subdim * sizeof(float));
        }
        if (gv_ivfpq_kmeans(subbuf, count, idx->subdim, idx->codebook_size, idx->train_iters,
                            idx->pq + m * idx->codebook_size * idx->subdim) != 0) {
            free(subbuf);
            free(train_buf);
            pthread_rwlock_unlock(&idx->rwlock);
            return -1;
        }
    }
    free(subbuf);
    
    if (idx->use_scalar_quant) {
        if (idx->scalar_quant_template != NULL) {
            gv_scalar_quant_vector_destroy(idx->scalar_quant_template);
        }
        idx->scalar_quant_template = gv_scalar_quantize_train(train_buf, count, idx->dimension, &idx->scalar_quant_config);
        if (idx->scalar_quant_template == NULL) {
            free(train_buf);
            pthread_rwlock_unlock(&idx->rwlock);
            return -1;
        }
    }
    
    free(train_buf);
    idx->trained = 1;
    pthread_rwlock_unlock(&idx->rwlock);
    return 0;
}

static int gv_ivfpq_encode(const GV_IVFPQIndex *idx, const float *vec, uint8_t *codes) {
    /* coarse assign */
    float best = INFINITY;
    int best_id = -1;
    for (size_t c = 0; c < idx->nlist; ++c) {
        const float *cent = idx->coarse + c * idx->dimension;
        float d = gv_ivfpq_l2_runtime(vec, cent, idx->dimension);
        if (d < best) {
            best = d;
            best_id = (int)c;
        }
    }
    if (best_id < 0) return -1;

    const float *centroid = idx->coarse + best_id * idx->dimension;
    float *res = (float *)malloc(idx->dimension * sizeof(float));
    if (!res) return -1;
    for (size_t j = 0; j < idx->dimension; ++j) {
        res[j] = vec[j] - centroid[j];
    }

    for (size_t m = 0; m < idx->m; ++m) {
        const float *codebook = idx->pq + m * idx->codebook_size * idx->subdim;
        const float *subvec = res + m * idx->subdim;
        float bestd = INFINITY;
        uint8_t bestc = 0;
        for (size_t c = 0; c < idx->codebook_size; ++c) {
            const float *cb = codebook + c * idx->subdim;
            float d = gv_ivfpq_l2_runtime(subvec, cb, idx->subdim);
            if (d < bestd) {
                bestd = d;
                bestc = (uint8_t)c;
            }
        }
        codes[m] = bestc;
    }

    free(res);
    return best_id;
}

int gv_ivfpq_insert(void *index_ptr, GV_Vector *vector) {
    GV_IVFPQIndex *idx = (GV_IVFPQIndex *)index_ptr;
    if (idx == NULL || vector == NULL || vector->dimension != idx->dimension || idx->trained == 0) {
        return -1;
    }
    pthread_rwlock_rdlock(&idx->rwlock);
    if (idx->use_cosine) {
        float norm = 0.0f;
        for (size_t i = 0; i < idx->dimension; ++i) norm += vector->data[i] * vector->data[i];
        if (norm > 0.0f) {
            norm = 1.0f / sqrtf(norm);
            for (size_t i = 0; i < idx->dimension; ++i) vector->data[i] *= norm;
        }
    }
    uint8_t *codes = (uint8_t *)malloc(idx->m);
    if (!codes) return -1;
    int list_id = gv_ivfpq_encode(idx, vector->data, codes);
    if (list_id < 0) {
        free(codes);
        pthread_rwlock_unlock(&idx->rwlock);
        return -1;
    }
    GV_IVFPQList *list = &idx->lists[list_id];
    pthread_mutex_lock(&idx->list_mutex[list_id]);
    if (list->count >= list->capacity) {
        size_t newcap = list->capacity ? list->capacity * 2 : 16;
        GV_IVFPQEntry *newents = (GV_IVFPQEntry *)realloc(list->entries, newcap * sizeof(GV_IVFPQEntry));
        uint8_t *newcodes = (uint8_t *)realloc(list->codes_soa, newcap * idx->m * sizeof(uint8_t));
        if (!newents) {
            free(codes);
            pthread_mutex_unlock(&idx->list_mutex[list_id]);
            pthread_rwlock_unlock(&idx->rwlock);
            return -1;
        }
        if (!newcodes) {
            free(codes);
            pthread_mutex_unlock(&idx->list_mutex[list_id]);
            pthread_rwlock_unlock(&idx->rwlock);
            return -1;
        }
        list->entries = newents;
        list->codes_soa = newcodes;
        list->capacity = newcap;
    }
    list->entries[list->count].codes = codes;
    list->entries[list->count].vector = vector;
    list->entries[list->count].scalar_quant = NULL;
    list->entries[list->count].deleted = 0;
    
    if (idx->use_scalar_quant && idx->scalar_quant_template != NULL) {
        GV_ScalarQuantVector *sqv = (GV_ScalarQuantVector *)malloc(sizeof(GV_ScalarQuantVector));
        if (sqv != NULL) {
            sqv->dimension = idx->dimension;
            sqv->bits = idx->scalar_quant_config.bits;
            sqv->per_dimension = idx->scalar_quant_config.per_dimension;
            sqv->bytes_per_vector = idx->scalar_quant_template->bytes_per_vector;
            sqv->min_vals = idx->scalar_quant_template->min_vals;
            sqv->max_vals = idx->scalar_quant_template->max_vals;
            sqv->quantized = (uint8_t *)calloc(sqv->bytes_per_vector, sizeof(uint8_t));
            if (sqv->quantized != NULL) {
                size_t max_quant = (1ULL << sqv->bits) - 1;
                for (size_t i = 0; i < idx->dimension; ++i) {
                    float min_val = sqv->per_dimension ? sqv->min_vals[i] : sqv->min_vals[0];
                    float max_val = sqv->per_dimension ? sqv->max_vals[i] : sqv->max_vals[0];
                    float range = max_val - min_val;
                    if (range <= 0.0f) continue;
                    float normalized = (vector->data[i] - min_val) / range;
                    normalized = (normalized < 0.0f) ? 0.0f : (normalized > 1.0f) ? 1.0f : normalized;
                    size_t quantized_val = (size_t)(normalized * max_quant + 0.5f);
                    if (quantized_val > max_quant) quantized_val = max_quant;
                    if (sqv->bits == 4) {
                        size_t byte_idx = i / 2;
                        size_t bit_offset = (i % 2) * 4;
                        sqv->quantized[byte_idx] |= (uint8_t)(quantized_val << (4 - bit_offset));
                    } else if (sqv->bits == 8) {
                        sqv->quantized[i] = (uint8_t)quantized_val;
                    } else if (sqv->bits == 16) {
                        ((uint16_t *)sqv->quantized)[i] = (uint16_t)quantized_val;
                    }
                }
                list->entries[list->count].scalar_quant = sqv;
            } else {
                free(sqv);
            }
        }
    }
    
    /* copy into SoA */
    for (size_t m = 0; m < idx->m; ++m) {
        list->codes_soa[m * list->capacity + list->count] = codes[m];
    }
    list->count++;
    idx->count++;
    pthread_mutex_unlock(&idx->list_mutex[list_id]);
    pthread_rwlock_unlock(&idx->rwlock);
    return 0;
}

size_t gv_ivfpq_count(const void *index_ptr) {
    if (index_ptr == NULL) return 0;
    const GV_IVFPQIndex *idx = (const GV_IVFPQIndex *)index_ptr;
    return idx->count;
}

int gv_ivfpq_is_trained(const void *index_ptr) {
    if (index_ptr == NULL) return 0;
    const GV_IVFPQIndex *idx = (const GV_IVFPQIndex *)index_ptr;
    return idx->trained;
}

int gv_ivfpq_search(void *index_ptr, const GV_Vector *query, size_t k,
                    GV_SearchResult *results, GV_DistanceType distance_type,
                    size_t nprobe_override, size_t rerank_top) {
    GV_IVFPQIndex *idx = (GV_IVFPQIndex *)index_ptr;
    if (idx == NULL || query == NULL || results == NULL || k == 0 ||
        query->dimension != idx->dimension || idx->trained == 0) {
        return -1;
    }
    pthread_rwlock_rdlock(&idx->rwlock);
    int cosine = idx->use_cosine || (distance_type == GV_DISTANCE_COSINE);
    /* coarse scores + heap-select top nprobe */
    size_t nprobe = (nprobe_override > 0) ? nprobe_override : idx->nprobe;
    if (nprobe > idx->nlist) nprobe = idx->nlist;
    GV_IVFPQHeapItem *cheap = (GV_IVFPQHeapItem *)malloc(nprobe * sizeof(GV_IVFPQHeapItem));
    size_t chsize = 0;
    if (!cheap) {
        pthread_rwlock_unlock(&idx->rwlock);
        return -1;
    }
    for (size_t c = 0; c < idx->nlist; ++c) {
        float d = gv_ivfpq_l2_runtime(query->data, idx->coarse + c * idx->dimension, idx->dimension);
        if (chsize < nprobe) {
            cheap[chsize].dist = d;
            cheap[chsize].entry = (GV_IVFPQEntry *)(uintptr_t)c;
            gv_ivfpq_heap_sift_up(cheap, chsize);
            chsize++;
        } else if (d < cheap[0].dist) {
            cheap[0].dist = d;
            cheap[0].entry = (GV_IVFPQEntry *)(uintptr_t)c;
            gv_ivfpq_heap_sift_down(cheap, chsize, 0);
        }
    }
    int *probe_ids = (int *)malloc(nprobe * sizeof(int));
    if (!probe_ids) {
        free(cheap);
        pthread_rwlock_unlock(&idx->rwlock);
        return -1;
    }
    for (size_t i = nprobe; i-- > 0;) {
        probe_ids[i] = (int)(uintptr_t)cheap[0].entry;
        cheap[0] = cheap[chsize - 1];
        chsize--;
        if (chsize > 0) gv_ivfpq_heap_sift_down(cheap, chsize, 0);
    }
    free(cheap);

    /* precompute LUT */
    size_t lut_need = idx->m * idx->codebook_size;
    if (idx->lut_buf_size < lut_need) {
        float *nbuf = (float *)realloc(idx->lut_buf, lut_need * sizeof(float));
        if (!nbuf) {
            free(probe_ids);
            pthread_rwlock_unlock(&idx->rwlock);
            return -1;
        }
        idx->lut_buf = nbuf;
        idx->lut_buf_size = lut_need;
    }
    float *lut = idx->lut_buf;
    /* cosine: normalize query once */
    float *qbuf = NULL;
    const float *qdata = query->data;
    if (idx->use_cosine || cosine) {
        float norm = 0.0f;
        for (size_t i = 0; i < idx->dimension; ++i) norm += query->data[i] * query->data[i];
        if (norm > 0.0f) {
            norm = 1.0f / sqrtf(norm);
            qbuf = (float *)malloc(idx->dimension * sizeof(float));
            if (!qbuf) {
                free(probe_ids);
                pthread_rwlock_unlock(&idx->rwlock);
                return -1;
            }
            for (size_t i = 0; i < idx->dimension; ++i) qbuf[i] = query->data[i] * norm;
            qdata = qbuf;
        }
    }

    for (size_t m = 0; m < idx->m; ++m) {
        const float *cb = idx->pq + m * idx->codebook_size * idx->subdim;
        const float *subq = qdata + m * idx->subdim;
        float *lut_row = lut + m * idx->codebook_size;
        for (size_t c = 0; c < idx->codebook_size; ++c) {
            const float *code = cb + c * idx->subdim;
            float d;
            if (cosine) {
                float dot = gv_ivfpq_dot_runtime(subq, code, idx->subdim);
                float cq = gv_ivfpq_dot_runtime(code, code, idx->subdim);
                float denom = sqrtf(cq);
                d = (denom > 0.0f) ? (1.0f - dot / denom) : 1.0f;
            } else {
                d = gv_ivfpq_l2_runtime(subq, code, idx->subdim);
            }
            lut_row[c] = d;
        }
    }

    /* Calculate oversampled candidate count */
    size_t oversampled_k = (size_t)(k * idx->oversampling_factor + 0.5f);
    if (oversampled_k < k) oversampled_k = k;
    
    /* bounded max-heap for top-k (or oversampled-k) */
    GV_IVFPQHeapItem *heap = (GV_IVFPQHeapItem *)malloc(oversampled_k * sizeof(GV_IVFPQHeapItem));
    size_t hsize = 0;
    if (!heap) {
        free(probe_ids);
        free(lut);
        pthread_rwlock_unlock(&idx->rwlock);
        return -1;
    }

    for (size_t pi = 0; pi < nprobe; ++pi) {
        int lid = probe_ids[pi];
        if (lid < 0) continue;
            GV_IVFPQList *list = &idx->lists[lid];
            const uint8_t *codes_soa = list->codes_soa;
            size_t lcount = list->count;
            if (codes_soa) {
                const size_t cap = list->capacity;
                for (size_t e = 0; e < lcount; ++e) {
                    float d = 0.0f;
                    const uint8_t *base = codes_soa + e;
                    size_t m = 0;
                    /* unroll by 4 for better ILP */
                    for (; m + 4 <= idx->m; m += 4) {
                        /* Prefetch upcoming codes/LUT rows to hide latency */
                        if (m + 8 < idx->m) {
                            __builtin_prefetch(base + (m + 8) * cap, 0, 1);
                            __builtin_prefetch(lut + (m + 8) * idx->codebook_size, 0, 1);
                        }
                        d += lut[(m + 0) * idx->codebook_size + base[(m + 0) * cap]];
                        d += lut[(m + 1) * idx->codebook_size + base[(m + 1) * cap]];
                        d += lut[(m + 2) * idx->codebook_size + base[(m + 2) * cap]];
                        d += lut[(m + 3) * idx->codebook_size + base[(m + 3) * cap]];
                    }
                    for (; m < idx->m; ++m) {
                        d += lut[m * idx->codebook_size + base[m * cap]];
                    }
                    GV_IVFPQEntry *ent = &list->entries[e];
                    if (ent->deleted != 0) continue;
                    if (hsize < oversampled_k) {
                        heap[hsize].dist = d;
                        heap[hsize].entry = ent;
                        gv_ivfpq_heap_sift_up(heap, hsize);
                        hsize++;
                    } else if (d < heap[0].dist) {
                        heap[0].dist = d;
                        heap[0].entry = ent;
                        gv_ivfpq_heap_sift_down(heap, hsize, 0);
                    }
                }
            } else {
                for (size_t e = 0; e < lcount; ++e) {
                    GV_IVFPQEntry *ent = &list->entries[e];
                    if (ent->deleted != 0) continue;
                    float d = 0.0f;
                    size_t m = 0;
                    for (; m + 4 <= idx->m; m += 4) {
                        if (m + 8 < idx->m) {
                            __builtin_prefetch(ent->codes + m + 8, 0, 1);
                            __builtin_prefetch(lut + (m + 8) * idx->codebook_size, 0, 1);
                        }
                        d += lut[(m + 0) * idx->codebook_size + ent->codes[m + 0]];
                        d += lut[(m + 1) * idx->codebook_size + ent->codes[m + 1]];
                        d += lut[(m + 2) * idx->codebook_size + ent->codes[m + 2]];
                        d += lut[(m + 3) * idx->codebook_size + ent->codes[m + 3]];
                    }
                    for (; m < idx->m; ++m) {
                        d += lut[m * idx->codebook_size + ent->codes[m]];
                    }
                    if (hsize < oversampled_k) {
                        heap[hsize].dist = d;
                        heap[hsize].entry = ent;
                        gv_ivfpq_heap_sift_up(heap, hsize);
                        hsize++;
                    } else if (d < heap[0].dist) {
                        heap[0].dist = d;
                        heap[0].entry = ent;
                        gv_ivfpq_heap_sift_down(heap, hsize, 0);
                    }
                }
            }
    }

    /* extract heap into arrays sorted ascending */
    size_t found = hsize;
    float *bestd = (float *)malloc(found * sizeof(float));
    GV_IVFPQEntry **beste = (GV_IVFPQEntry **)malloc(found * sizeof(GV_IVFPQEntry *));
    if (!bestd || !beste) {
        free(heap);
        free(probe_ids);
        free(lut);
        free(bestd);
        free(beste);
        pthread_rwlock_unlock(&idx->rwlock);
        return -1;
    }
    for (size_t i = found; i-- > 0;) {
        bestd[i] = heap[0].dist;
        beste[i] = heap[0].entry;
        heap[0] = heap[hsize - 1];
        hsize--;
        if (hsize > 0) gv_ivfpq_heap_sift_down(heap, hsize, 0);
    }
    free(heap);

    /* Optional rerank with exact distance on oversampled candidates */
    size_t rr = rerank_top > 0 ? rerank_top : idx->default_rerank;
    if (rr > found) rr = found;
    if (rr == 0 && idx->oversampling_factor > 1.0f) {
        rr = found;
    }
    if (rr > 0) {
        /* Rerank oversampled candidates with exact distances */
        for (size_t i = 0; i < rr; ++i) {
            if (beste[i] == NULL || beste[i]->deleted != 0) continue;
            
            float dist;
            if (idx->use_scalar_quant && beste[i]->scalar_quant != NULL) {
                dist = gv_scalar_quant_distance(query->data, beste[i]->scalar_quant, 
                                                 (int)(cosine ? GV_DISTANCE_COSINE : GV_DISTANCE_EUCLIDEAN));
            } else {
                dist = gv_distance(query, beste[i]->vector, cosine ? GV_DISTANCE_COSINE : GV_DISTANCE_EUCLIDEAN);
            }
            
            if (cosine && dist > -1.5f) {
                dist = 1.0f - dist; /* convert similarity to distance */
            }
            if (dist < 0) dist = bestd[i]; /* fallback */
            bestd[i] = dist;
        }
        /* re-sort first rr */
        for (size_t i = 0; i + 1 < rr; ++i) {
            size_t minj = i;
            for (size_t j = i + 1; j < rr; ++j) {
                if (bestd[j] < bestd[minj]) minj = j;
            }
            if (minj != i) {
                float td = bestd[i];
                bestd[i] = bestd[minj];
                bestd[minj] = td;
                GV_IVFPQEntry *te = beste[i];
                beste[i] = beste[minj];
                beste[minj] = te;
            }
        }
    }

    /* Limit results to k after reranking */
    size_t result_count = (found < k) ? found : k;
    for (size_t i = 0; i < result_count; ++i) {
        results[i].distance = bestd[i];
        results[i].vector = beste[i] ? beste[i]->vector : NULL;
    }

    free(probe_ids);
    /* lut is cached in idx->lut_buf; do not free */
    free(bestd);
    free(beste);
    free(qbuf);
    pthread_rwlock_unlock(&idx->rwlock);
    return (int)result_count;
}

void gv_ivfpq_destroy(void *index_ptr) {
    GV_IVFPQIndex *idx = (GV_IVFPQIndex *)index_ptr;
    if (!idx) return;
    pthread_rwlock_wrlock(&idx->rwlock);
    if (idx->lists) {
        for (size_t i = 0; i < idx->nlist; ++i) {
            GV_IVFPQList *list = &idx->lists[i];
            if (list->entries) {
                for (size_t e = 0; e < list->count; ++e) {
                    free(list->entries[e].codes);
                    if (list->entries[e].scalar_quant != NULL) {
                        GV_ScalarQuantVector *sqv = list->entries[e].scalar_quant;
                        if (sqv->quantized != NULL) {
                            free(sqv->quantized);
                        }
                        sqv->min_vals = NULL;
                        sqv->max_vals = NULL;
                        free(sqv);
                    }
                    gv_vector_destroy(list->entries[e].vector);
                }
                free(list->entries);
            }
            free(list->codes_soa);
        }
    }
    if (idx->scalar_quant_template != NULL) {
        gv_scalar_quant_vector_destroy(idx->scalar_quant_template);
    }
    free(idx->lists);
    /* list_mutex freed separately */
    if (idx->list_mutex) {
        for (size_t i = 0; i < idx->nlist; ++i) {
            pthread_mutex_destroy(&idx->list_mutex[i]);
        }
    }
    pthread_rwlock_unlock(&idx->rwlock);
    pthread_rwlock_destroy(&idx->rwlock);
    free(idx->list_mutex);
    free(idx->coarse);
    free(idx->pq);
    free(idx->lut_buf);
    free(idx);
}

int gv_ivfpq_save(const void *index_ptr, FILE *out, uint32_t version) {
    (void)version;
    const GV_IVFPQIndex *idx = (const GV_IVFPQIndex *)index_ptr;
    if (idx == NULL || out == NULL) return -1;
    uint32_t crc = gv_crc32_init();
    /* header */
    if (fwrite(&idx->dimension, sizeof(size_t), 1, out) != 1) return -1;
    crc = gv_crc32_update(crc, &idx->dimension, sizeof(size_t));
    if (fwrite(&idx->nlist, sizeof(size_t), 1, out) != 1) return -1;
    crc = gv_crc32_update(crc, &idx->nlist, sizeof(size_t));
    if (fwrite(&idx->m, sizeof(size_t), 1, out) != 1) return -1;
    crc = gv_crc32_update(crc, &idx->m, sizeof(size_t));
    if (fwrite(&idx->nbits, sizeof(uint8_t), 1, out) != 1) return -1;
    crc = gv_crc32_update(crc, &idx->nbits, sizeof(uint8_t));
    if (fwrite(&idx->nprobe, sizeof(size_t), 1, out) != 1) return -1;
    crc = gv_crc32_update(crc, &idx->nprobe, sizeof(size_t));
    if (fwrite(&idx->train_iters, sizeof(size_t), 1, out) != 1) return -1;
    crc = gv_crc32_update(crc, &idx->train_iters, sizeof(size_t));
    if (fwrite(&idx->default_rerank, sizeof(size_t), 1, out) != 1) return -1;
    crc = gv_crc32_update(crc, &idx->default_rerank, sizeof(size_t));
    if (fwrite(&idx->use_cosine, sizeof(int), 1, out) != 1) return -1;
    crc = gv_crc32_update(crc, &idx->use_cosine, sizeof(int));
    if (fwrite(&idx->oversampling_factor, sizeof(float), 1, out) != 1) return -1;
    crc = gv_crc32_update(crc, &idx->oversampling_factor, sizeof(float));
    if (fwrite(&idx->trained, sizeof(int), 1, out) != 1) return -1;
    crc = gv_crc32_update(crc, &idx->trained, sizeof(int));
    if (idx->trained) {
        size_t coarse_sz = idx->nlist * idx->dimension;
        size_t pq_sz = idx->m * idx->codebook_size * idx->subdim;
        if (fwrite(idx->coarse, sizeof(float), coarse_sz, out) != coarse_sz) return -1;
        crc = gv_crc32_update(crc, idx->coarse, coarse_sz * sizeof(float));
        if (fwrite(idx->pq, sizeof(float), pq_sz, out) != pq_sz) return -1;
        crc = gv_crc32_update(crc, idx->pq, pq_sz * sizeof(float));
    }
    if (fwrite(&idx->count, sizeof(size_t), 1, out) != 1) return -1;
    crc = gv_crc32_update(crc, &idx->count, sizeof(size_t));
    for (size_t i = 0; i < idx->nlist; ++i) {
        GV_IVFPQList *list = &idx->lists[i];
        if (fwrite(&list->count, sizeof(size_t), 1, out) != 1) return -1;
        crc = gv_crc32_update(crc, &list->count, sizeof(size_t));
        for (size_t e = 0; e < list->count; ++e) {
            GV_IVFPQEntry *ent = &list->entries[e];
        if (fwrite(ent->codes, sizeof(uint8_t), idx->m, out) != idx->m) return -1;
            crc = gv_crc32_update(crc, ent->codes, idx->m * sizeof(uint8_t));
            if (fwrite(ent->vector->data, sizeof(float), ent->vector->dimension, out) != ent->vector->dimension) return -1;
            crc = gv_crc32_update(crc, ent->vector->data, ent->vector->dimension * sizeof(float));
            /* metadata persist */
            const GV_Metadata *meta = ent->vector->metadata;
            uint32_t mcount = 0;
            for (const GV_Metadata *c = meta; c != NULL; c = c->next) mcount++;
            if (gv_ivfpq_write_u32(out, mcount) != 0) return -1;
            crc = gv_crc32_update(crc, &mcount, sizeof(uint32_t));
            for (const GV_Metadata *c = meta; c != NULL; c = c->next) {
                uint32_t klen = (uint32_t)strlen(c->key);
                uint32_t vlen = (uint32_t)strlen(c->value);
                if (gv_ivfpq_write_str(out, c->key, klen) != 0) return -1;
                if (gv_ivfpq_write_str(out, c->value, vlen) != 0) return -1;
                crc = gv_crc32_update(crc, &klen, sizeof(uint32_t));
                crc = gv_crc32_update(crc, c->key, klen);
                crc = gv_crc32_update(crc, &vlen, sizeof(uint32_t));
                crc = gv_crc32_update(crc, c->value, vlen);
            }
        }
    }
    crc = gv_crc32_finish(crc);
    if (fwrite(&crc, sizeof(uint32_t), 1, out) != 1) return -1;
    return 0;
}

int gv_ivfpq_load(void **index_ptr, FILE *in, size_t dimension, uint32_t version) {
    (void)version;
    if (index_ptr == NULL || in == NULL) return -1;
    size_t dim = 0, nlist = 0, m = 0, nprobe = 0, train_iters = 0, default_rerank = 0;
    uint8_t nbits = 0;
    int trained = 0, use_cosine = 0;
    float oversampling_factor = 1.0f;
    uint32_t crc = gv_crc32_init();
    if (fread(&dim, sizeof(size_t), 1, in) != 1) return -1;
    crc = gv_crc32_update(crc, &dim, sizeof(size_t));
    if (dim != dimension) return -1;
    if (fread(&nlist, sizeof(size_t), 1, in) != 1) return -1;
    crc = gv_crc32_update(crc, &nlist, sizeof(size_t));
    if (fread(&m, sizeof(size_t), 1, in) != 1) return -1;
    crc = gv_crc32_update(crc, &m, sizeof(size_t));
    if (fread(&nbits, sizeof(uint8_t), 1, in) != 1) return -1;
    crc = gv_crc32_update(crc, &nbits, sizeof(uint8_t));
    if (fread(&nprobe, sizeof(size_t), 1, in) != 1) return -1;
    crc = gv_crc32_update(crc, &nprobe, sizeof(size_t));
    if (fread(&train_iters, sizeof(size_t), 1, in) != 1) return -1;
    crc = gv_crc32_update(crc, &train_iters, sizeof(size_t));
    if (fread(&default_rerank, sizeof(size_t), 1, in) != 1) return -1;
    crc = gv_crc32_update(crc, &default_rerank, sizeof(size_t));
    if (fread(&use_cosine, sizeof(int), 1, in) != 1) return -1;
    crc = gv_crc32_update(crc, &use_cosine, sizeof(int));
    if (fread(&oversampling_factor, sizeof(float), 1, in) != 1) {
        oversampling_factor = 1.0f;
    } else {
        crc = gv_crc32_update(crc, &oversampling_factor, sizeof(float));
    }
    if (fread(&trained, sizeof(int), 1, in) != 1) return -1;
    crc = gv_crc32_update(crc, &trained, sizeof(int));

    GV_IVFPQConfig cfg = {.nlist = nlist, .m = m, .nbits = nbits, .nprobe = nprobe, .train_iters = train_iters, .default_rerank = default_rerank, .use_cosine = use_cosine, .oversampling_factor = oversampling_factor};
    void *idx_ptr = gv_ivfpq_create(dim, &cfg);
    if (!idx_ptr) return -1;
    GV_IVFPQIndex *idx = (GV_IVFPQIndex *)idx_ptr;
    idx->trained = trained;
    if (trained) {
        size_t coarse_sz = idx->nlist * idx->dimension;
        size_t pq_sz = idx->m * idx->codebook_size * idx->subdim;
        if (fread(idx->coarse, sizeof(float), coarse_sz, in) != coarse_sz) {
            gv_ivfpq_destroy(idx_ptr);
            return -1;
        }
        crc = gv_crc32_update(crc, idx->coarse, coarse_sz * sizeof(float));
        if (fread(idx->pq, sizeof(float), pq_sz, in) != pq_sz) {
            gv_ivfpq_destroy(idx_ptr);
            return -1;
        }
        crc = gv_crc32_update(crc, idx->pq, pq_sz * sizeof(float));
    }
    size_t total = 0;
    if (fread(&total, sizeof(size_t), 1, in) != 1) {
        gv_ivfpq_destroy(idx_ptr);
        return -1;
    }
    crc = gv_crc32_update(crc, &total, sizeof(size_t));
    size_t loaded_total = 0;
    for (size_t i = 0; i < idx->nlist; ++i) {
        size_t lcount = 0;
        if (fread(&lcount, sizeof(size_t), 1, in) != 1) {
            gv_ivfpq_destroy(idx_ptr);
            return -1;
        }
        crc = gv_crc32_update(crc, &lcount, sizeof(size_t));
        GV_IVFPQList *list = &idx->lists[i];
        if (lcount > 0) {
            list->entries = (GV_IVFPQEntry *)malloc(lcount * sizeof(GV_IVFPQEntry));
            list->codes_soa = (uint8_t *)malloc(lcount * idx->m * sizeof(uint8_t));
            if (!list->entries || !list->codes_soa) {
                gv_ivfpq_destroy(idx_ptr);
                return -1;
            }
            list->capacity = lcount;
        }
        for (size_t e = 0; e < lcount; ++e) {
            GV_IVFPQEntry *ent = &list->entries[e];
            ent->codes = (uint8_t *)malloc(idx->m);
            if (!ent->codes) {
                gv_ivfpq_destroy(idx_ptr);
                return -1;
            }
            if (fread(ent->codes, sizeof(uint8_t), idx->m, in) != idx->m) {
                gv_ivfpq_destroy(idx_ptr);
                return -1;
            }
            crc = gv_crc32_update(crc, ent->codes, idx->m * sizeof(uint8_t));
            for (size_t m = 0; m < idx->m; ++m) {
                list->codes_soa[m * list->capacity + e] = ent->codes[m];
            }
            GV_Vector *vec = gv_vector_create(dim);
            if (!vec) {
                gv_ivfpq_destroy(idx_ptr);
                return -1;
            }
            if (fread(vec->data, sizeof(float), dim, in) != dim) {
                gv_vector_destroy(vec);
                gv_ivfpq_destroy(idx_ptr);
                return -1;
            }
            crc = gv_crc32_update(crc, vec->data, dim * sizeof(float));
            if (idx->use_cosine) {
                float norm = 0.0f;
                for (size_t j = 0; j < dim; ++j) norm += vec->data[j] * vec->data[j];
                if (norm > 0.0f) {
                    norm = 1.0f / sqrtf(norm);
                    for (size_t j = 0; j < dim; ++j) vec->data[j] *= norm;
                }
            }
            uint32_t mcount = 0;
            if (gv_ivfpq_read_u32(in, &mcount) != 0) {
                gv_vector_destroy(vec);
                gv_ivfpq_destroy(idx_ptr);
                return -1;
            }
            crc = gv_crc32_update(crc, &mcount, sizeof(uint32_t));
            for (uint32_t mi = 0; mi < mcount; ++mi) {
                uint32_t klen = 0, vlen = 0;
                if (gv_ivfpq_read_u32(in, &klen) != 0) {
                    gv_vector_destroy(vec);
                    gv_ivfpq_destroy(idx_ptr);
                    return -1;
                }
                char *k = NULL;
                if (gv_ivfpq_read_str(in, &k, klen) != 0) {
                    gv_vector_destroy(vec);
                    gv_ivfpq_destroy(idx_ptr);
                    return -1;
                }
                if (gv_ivfpq_read_u32(in, &vlen) != 0) {
                    free(k);
                    gv_vector_destroy(vec);
                    gv_ivfpq_destroy(idx_ptr);
                    return -1;
                }
                char *v = NULL;
                if (gv_ivfpq_read_str(in, &v, vlen) != 0) {
                    free(k);
                    gv_vector_destroy(vec);
                    gv_ivfpq_destroy(idx_ptr);
                    return -1;
                }
                if (gv_vector_set_metadata(vec, k, v) != 0) {
                    free(k);
                    free(v);
                    gv_vector_destroy(vec);
                    gv_ivfpq_destroy(idx_ptr);
                    return -1;
                }
                crc = gv_crc32_update(crc, &klen, sizeof(uint32_t));
                crc = gv_crc32_update(crc, k, klen);
                crc = gv_crc32_update(crc, &vlen, sizeof(uint32_t));
                crc = gv_crc32_update(crc, v, vlen);
                free(k);
                free(v);
            }
            ent->vector = vec;
            ent->deleted = 0;
        }
        list->count = lcount;
        loaded_total += lcount;
    }
    if (loaded_total != total) {
        gv_ivfpq_destroy(idx_ptr);
        return -1;
    }
    idx->count = total;
    uint32_t stored_crc = 0;
    if (fread(&stored_crc, sizeof(uint32_t), 1, in) != 1) {
        gv_ivfpq_destroy(idx_ptr);
        return -1;
    }
    crc = gv_crc32_finish(crc);
    if (crc != stored_crc) {
        gv_ivfpq_destroy(idx_ptr);
        return -1;
    }
    *index_ptr = idx_ptr;
    return 0;
}


int gv_ivfpq_range_search(void *index_ptr, const GV_Vector *query, float radius,
                           GV_SearchResult *results, size_t max_results,
                           GV_DistanceType distance_type) {
    GV_IVFPQIndex *idx = (GV_IVFPQIndex *)index_ptr;
    if (idx == NULL || query == NULL || results == NULL || max_results == 0 || radius < 0.0f ||
        query->dimension != idx->dimension || query->data == NULL || idx->trained == 0) {
        return -1;
    }

    pthread_rwlock_rdlock(&idx->rwlock);

    int cosine = (distance_type == GV_DISTANCE_COSINE) ? 1 : 0;
    float *qbuf = (float *)malloc(idx->dimension * sizeof(float));
    if (!qbuf) {
        pthread_rwlock_unlock(&idx->rwlock);
        return -1;
    }
    memcpy(qbuf, query->data, idx->dimension * sizeof(float));

    if (cosine) {
        float norm = 0.0f;
        for (size_t i = 0; i < idx->dimension; ++i) norm += qbuf[i] * qbuf[i];
        if (norm > 0.0f) {
            norm = 1.0f / sqrtf(norm);
            for (size_t i = 0; i < idx->dimension; ++i) qbuf[i] *= norm;
        }
    }

    int *probe_ids = (int *)malloc(idx->nprobe * sizeof(int));
    if (!probe_ids) {
        free(qbuf);
        pthread_rwlock_unlock(&idx->rwlock);
        return -1;
    }

    for (size_t i = 0; i < idx->nprobe; ++i) probe_ids[i] = -1;

    for (size_t i = 0; i < idx->nlist; ++i) {
        float d = 0.0f;
        for (size_t j = 0; j < idx->dimension; ++j) {
            float diff = qbuf[j] - idx->coarse[i * idx->dimension + j];
            d += diff * diff;
        }
        for (size_t p = 0; p < idx->nprobe; ++p) {
            if (probe_ids[p] < 0 || d < probe_ids[p]) {
                for (size_t q = idx->nprobe - 1; q > p; --q) {
                    probe_ids[q] = probe_ids[q - 1];
                }
                probe_ids[p] = (int)i;
                break;
            }
        }
    }

    size_t codebook_size = idx->codebook_size;
    float *lut = idx->lut_buf;
    size_t lut_needed = idx->m * codebook_size;
    if (lut == NULL || idx->lut_buf_size < lut_needed) {
        lut = (float *)realloc(lut, lut_needed * sizeof(float));
        if (!lut) {
            free(probe_ids);
            free(qbuf);
            pthread_rwlock_unlock(&idx->rwlock);
            return -1;
        }
        idx->lut_buf = lut;
        idx->lut_buf_size = lut_needed;
    }

    for (size_t m = 0; m < idx->m; ++m) {
        float *lut_row = lut + m * codebook_size;
        float *pq_row = idx->pq + m * codebook_size * idx->subdim;
        for (size_t c = 0; c < codebook_size; ++c) {
            float d = 0.0f;
            for (size_t s = 0; s < idx->subdim; ++s) {
                float diff = qbuf[m * idx->subdim + s] - pq_row[c * idx->subdim + s];
                d += diff * diff;
            }
            lut_row[c] = d;
        }
    }

    typedef struct {
        GV_Vector *vector;
        GV_IVFPQEntry *entry;
        float pq_distance;
    } RangeCandidate;
    
    RangeCandidate *candidates = (RangeCandidate *)malloc(max_results * 2 * sizeof(RangeCandidate));
    if (!candidates) {
        free(probe_ids);
        free(qbuf);
        pthread_rwlock_unlock(&idx->rwlock);
        return -1;
    }
    size_t found = 0;

    for (size_t pi = 0; pi < idx->nprobe; ++pi) {
        int lid = probe_ids[pi];
        if (lid < 0) continue;
        GV_IVFPQList *list = &idx->lists[lid];
        const uint8_t *codes_soa = list->codes_soa;
        size_t lcount = list->count;
        if (codes_soa) {
            const size_t cap = list->capacity;
            for (size_t e = 0; e < lcount && found < max_results * 2; ++e) {
                float d = 0.0f;
                const uint8_t *base = codes_soa + e;
                for (size_t m = 0; m < idx->m; ++m) {
                    d += lut[m * idx->codebook_size + base[m * cap]];
                }
                if (d <= radius) {
                    GV_IVFPQEntry *ent = &list->entries[e];
                    if (ent->vector != NULL) {
                        candidates[found].vector = ent->vector;
                        candidates[found].entry = ent;
                        candidates[found].pq_distance = d;
                        found++;
                    }
                }
            }
        } else {
            for (size_t e = 0; e < lcount && found < max_results * 2; ++e) {
                GV_IVFPQEntry *ent = &list->entries[e];
                float d = 0.0f;
                for (size_t m = 0; m < idx->m; ++m) {
                    d += lut[m * idx->codebook_size + ent->codes[m]];
                }
                if (d <= radius) {
                    if (ent->vector != NULL && ent->deleted == 0) {
                        candidates[found].vector = ent->vector;
                        candidates[found].entry = ent;
                        candidates[found].pq_distance = d;
                        found++;
                    }
                }
            }
        }
    }

    size_t result_count = 0;
    for (size_t i = 0; i < found && result_count < max_results; ++i) {
        if (candidates[i].vector != NULL) {
            float dist;
            if (idx->use_scalar_quant && candidates[i].entry != NULL && 
                candidates[i].entry->scalar_quant != NULL) {
                dist = gv_scalar_quant_distance(query->data, candidates[i].entry->scalar_quant,
                    (int)(cosine ? GV_DISTANCE_COSINE : GV_DISTANCE_EUCLIDEAN));
                if (cosine && dist > -1.5f) {
                    dist = 1.0f - dist;
                }
            } else {
                dist = gv_distance(query, candidates[i].vector, cosine ? GV_DISTANCE_COSINE : GV_DISTANCE_EUCLIDEAN);
                if (cosine && dist > -1.5f) {
                    dist = 1.0f - dist;
                }
            }
            if (dist <= radius) {
                results[result_count].vector = candidates[i].vector;
                results[result_count].distance = dist;
                result_count++;
            }
        }
    }

    free(candidates);
    free(probe_ids);
    free(qbuf);
    pthread_rwlock_unlock(&idx->rwlock);
    return (int)result_count;
}

int gv_ivfpq_delete(void *index_ptr, size_t entry_index) {
    if (index_ptr == NULL) {
        return -1;
    }

    GV_IVFPQIndex *idx = (GV_IVFPQIndex *)index_ptr;
    pthread_rwlock_wrlock(&idx->rwlock);

    size_t current_index = 0;
    for (size_t list_id = 0; list_id < idx->nlist; ++list_id) {
        GV_IVFPQList *list = &idx->lists[list_id];
        pthread_mutex_lock(&idx->list_mutex[list_id]);
        
        for (size_t e = 0; e < list->count; ++e) {
            if (current_index == entry_index) {
                if (list->entries[e].deleted != 0) {
                    pthread_mutex_unlock(&idx->list_mutex[list_id]);
                    pthread_rwlock_unlock(&idx->rwlock);
                    return -1;
                }
                list->entries[e].deleted = 1;
                pthread_mutex_unlock(&idx->list_mutex[list_id]);
                pthread_rwlock_unlock(&idx->rwlock);
                return 0;
            }
            current_index++;
        }
        
        pthread_mutex_unlock(&idx->list_mutex[list_id]);
    }

    pthread_rwlock_unlock(&idx->rwlock);
    return -1;
}

int gv_ivfpq_update(void *index_ptr, size_t entry_index, const float *new_data, size_t dimension) {
    if (index_ptr == NULL || new_data == NULL) {
        return -1;
    }

    GV_IVFPQIndex *idx = (GV_IVFPQIndex *)index_ptr;
    if (dimension != idx->dimension || idx->trained == 0) {
        return -1;
    }

    pthread_rwlock_wrlock(&idx->rwlock);

    size_t current_index = 0;
    for (size_t list_id = 0; list_id < idx->nlist; ++list_id) {
        GV_IVFPQList *list = &idx->lists[list_id];
        pthread_mutex_lock(&idx->list_mutex[list_id]);
        
        for (size_t e = 0; e < list->count; ++e) {
            if (current_index == entry_index) {
                if (list->entries[e].deleted != 0) {
                    pthread_mutex_unlock(&idx->list_mutex[list_id]);
                    pthread_rwlock_unlock(&idx->rwlock);
                    return -1;
                }
                
                GV_IVFPQEntry *ent = &list->entries[e];
                
                /* Update vector data */
                if (ent->vector != NULL) {
                    memcpy(ent->vector->data, new_data, dimension * sizeof(float));
                }
                
                /* Recompute PQ codes */
                if (ent->codes != NULL && idx->pq != NULL && idx->coarse != NULL) {
                    /* Find closest coarse centroid */
                    float best = INFINITY;
                    int best_coarse_id = -1;
                    for (size_t c = 0; c < idx->nlist; ++c) {
                        const float *cent = idx->coarse + c * idx->dimension;
                        float d = 0.0f;
                        for (size_t j = 0; j < idx->dimension; ++j) {
                            float diff = new_data[j] - cent[j];
                            d += diff * diff;
                        }
                        if (d < best) {
                            best = d;
                            best_coarse_id = (int)c;
                        }
                    }
                    
                    if (best_coarse_id >= 0) {
                        const float *centroid = idx->coarse + best_coarse_id * idx->dimension;
                        float *residual = (float *)malloc(idx->dimension * sizeof(float));
                        if (residual != NULL) {
                            for (size_t j = 0; j < idx->dimension; ++j) {
                                residual[j] = new_data[j] - centroid[j];
                            }
                            
                            for (size_t m = 0; m < idx->m; ++m) {
                                const float *codebook = idx->pq + m * idx->codebook_size * idx->subdim;
                                const float *subvec = residual + m * idx->subdim;
                                float bestd = INFINITY;
                                uint8_t bestc = 0;
                                for (size_t c = 0; c < idx->codebook_size; ++c) {
                                    const float *cb = codebook + c * idx->subdim;
                                    float d = 0.0f;
                                    for (size_t d_idx = 0; d_idx < idx->subdim; ++d_idx) {
                                        float diff = subvec[d_idx] - cb[d_idx];
                                        d += diff * diff;
                                    }
                                    if (d < bestd) {
                                        bestd = d;
                                        bestc = (uint8_t)c;
                                    }
                                }
                                ent->codes[m] = bestc;
                                list->codes_soa[m * list->capacity + e] = bestc;
                            }
                            free(residual);
                        }
                    }
                }
                
                /* Update scalar quantization if enabled */
                if (idx->use_scalar_quant && ent->scalar_quant != NULL && idx->scalar_quant_template != NULL) {
                    GV_ScalarQuantVector *sqv = ent->scalar_quant;
                    size_t max_quant = (1ULL << sqv->bits) - 1;
                    for (size_t i = 0; i < dimension; ++i) {
                        float min_val = sqv->per_dimension ? sqv->min_vals[i] : sqv->min_vals[0];
                        float max_val = sqv->per_dimension ? sqv->max_vals[i] : sqv->max_vals[0];
                        float range = max_val - min_val;
                        if (range > 0.0f) {
                            float normalized = (new_data[i] - min_val) / range;
                            uint8_t quantized = (uint8_t)(normalized * max_quant + 0.5f);
                            if (quantized > max_quant) quantized = max_quant;
                            size_t byte_idx = i * sqv->bits / 8;
                            size_t bit_offset = (i * sqv->bits) % 8;
                            if (sqv->bits == 8) {
                                sqv->quantized[byte_idx] = quantized;
                            } else {
                                /* Handle bit packing for non-8-bit quantization */
                                uint8_t mask = (1 << sqv->bits) - 1;
                                sqv->quantized[byte_idx] = (sqv->quantized[byte_idx] & ~(mask << bit_offset)) | (quantized << bit_offset);
                            }
                        }
                    }
                }
                
                pthread_mutex_unlock(&idx->list_mutex[list_id]);
                pthread_rwlock_unlock(&idx->rwlock);
                return 0;
            }
            current_index++;
        }
        
        pthread_mutex_unlock(&idx->list_mutex[list_id]);
    }

    pthread_rwlock_unlock(&idx->rwlock);
    return -1;
}
