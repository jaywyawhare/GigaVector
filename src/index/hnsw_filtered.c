#include <math.h>
#include <float.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "index/hnsw_filtered.h"
#include "index/hnsw.h"
#include "core/memory.h"
#include "core/utils.h"
#include "schema/metadata.h"
#include "schema/vector.h"
#include "storage/soa_storage.h"
#include "multimodal/payload_index.h"

/* ---- Internal HNSW struct mirror (must stay in sync with hnsw.c) ---- */

typedef struct {
    size_t vector_index;
    void   *binary_vector;
    size_t level;
    int    deleted;
} GV_HNSWNode;

typedef struct {
    size_t dimension;
    size_t M;
    size_t efConstruction;
    size_t efSearch;
    size_t maxLevel;
    int    use_binary_quant;
    size_t quant_rerank;
    int    use_acorn;
    size_t acorn_hops;
    GV_DistanceType distance_type;
    unsigned int rand_seed;
    size_t entry_point;
    size_t count;
    size_t nodes_capacity;
    GV_HNSWNode *nodes;
    int32_t     *neighbors;
    size_t      *offsets;
    size_t       neighbors_size;
    size_t       neighbors_capacity;
    size_t      *cum_nb_per_level;
    size_t       max_level_alloc;
    GV_SoAStorage *soa_storage;
    int           soa_storage_owned;
    uint32_t     *visited_epoch;
    uint32_t      current_epoch;
    size_t        visited_capacity;
    float  *search_dis;
    size_t *search_ids;
    uint8_t *search_proc;
    size_t   search_buf_size;
    float  *insert_dis;
    size_t *insert_ids;
    uint8_t *insert_proc;
    size_t   insert_buf_size;
} GV_HNSWIndex;

/* ---- Internal helpers (mirrored from hnsw.c) ---- */

static inline size_t nb_slots_for_level_i(const GV_HNSWIndex *index, size_t level) {
    if (level + 1 <= index->max_level_alloc)
        return index->cum_nb_per_level[level + 1];
    return 2 * index->M + level * index->M;
}

static inline int32_t *nb_begin_i(const GV_HNSWIndex *index, size_t node_idx, size_t level) {
    return index->neighbors + index->offsets[node_idx] + index->cum_nb_per_level[level];
}

static inline size_t max_nb_at_level_i(const GV_HNSWIndex *index, size_t level) {
    return (level == 0) ? 2 * index->M : index->M;
}

static inline float hnsw_l2_scalar_i(const float *a, const float *b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

static inline float hnsw_dot_i(const float *a, const float *b, size_t dim) {
    float d = 0.0f;
    for (size_t i = 0; i < dim; ++i) d += a[i] * b[i];
    return d;
}

static inline float hnsw_cosine_i(const float *a, const float *b, size_t dim) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    float denom = sqrtf(na) * sqrtf(nb);
    return (denom > 0.0f) ? (1.0f - dot / denom) : 1.0f;
}

static inline float hnsw_manhattan_i(const float *a, const float *b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) { float d = a[i] - b[i]; sum += (d < 0) ? -d : d; }
    return sum;
}

static float hnsw_raw_distance_i(const float *a, const float *b, size_t dim,
                                  GV_DistanceType dtype) {
    switch (dtype) {
    case GV_DISTANCE_COSINE:     return hnsw_cosine_i(a, b, dim);
    case GV_DISTANCE_DOT_PRODUCT: return -hnsw_dot_i(a, b, dim);
    case GV_DISTANCE_MANHATTAN:  return hnsw_manhattan_i(a, b, dim);
    case GV_DISTANCE_EUCLIDEAN:
    default:                     return hnsw_l2_scalar_i(a, b, dim);
    }
}

/* ---- Max-heap priority queue (from hnsw.c) ---- */

static inline void heap_sift_up_f(float *dis, size_t *ids, uint8_t *proc, size_t i) {
    float val = dis[i]; size_t id = ids[i]; uint8_t p = proc[i];
    while (i > 0) {
        size_t parent = (i - 1) >> 1;
        if (dis[parent] >= val) break;
        dis[i] = dis[parent]; ids[i] = ids[parent]; proc[i] = proc[parent];
        i = parent;
    }
    dis[i] = val; ids[i] = id; proc[i] = p;
}

static inline void heap_sift_down_f(float *dis, size_t *ids, uint8_t *proc, size_t k, size_t i) {
    float val = dis[i]; size_t id = ids[i]; uint8_t p = proc[i];
    while (1) {
        size_t c1 = 2 * i + 1, c2 = c1 + 1;
        if (c1 >= k) break;
        size_t larger = (c2 < k && dis[c2] > dis[c1]) ? c2 : c1;
        if (val >= dis[larger]) break;
        dis[i] = dis[larger]; ids[i] = ids[larger]; proc[i] = proc[larger];
        i = larger;
    }
    dis[i] = val; ids[i] = id; proc[i] = p;
}

static inline int mmheap_push_f(float *dis, size_t *ids, uint8_t *proc,
                                size_t *k, size_t capacity, size_t node_idx, float dist) {
    if (*k < capacity) {
        dis[*k] = dist; ids[*k] = node_idx; proc[*k] = 0;
        heap_sift_up_f(dis, ids, proc, *k);
        (*k)++;
        return 1;
    } else if (dist < dis[0]) {
        dis[0] = dist; ids[0] = node_idx; proc[0] = 0;
        heap_sift_down_f(dis, ids, proc, *k, 0);
        return 1;
    }
    return 0;
}

static inline size_t mmheap_pop_min_f(float *dis, size_t *ids, uint8_t *proc,
                                      size_t k, float *out_dist) {
    float best_dist = FLT_MAX;
    size_t best_pos = SIZE_MAX;
    for (size_t i = 0; i < k; ++i) {
        if (!proc[i] && dis[i] < best_dist) {
            best_dist = dis[i];
            best_pos = i;
        }
    }
    if (best_pos == SIZE_MAX) return SIZE_MAX;
    proc[best_pos] = 1;
    *out_dist = best_dist;
    return ids[best_pos];
}

/* ---- Filtered search implementation ---- */

static inline int is_allowed(const GV_IdBitmap *allowed, size_t node_id) {
    return (allowed == NULL) || gv_id_bitmap_contains(allowed, (uint64_t)node_id);
}

int gv_hnsw_search_filtered(void *index_ptr, const float *query, size_t dimension,
                            size_t k, const GV_IdBitmap *allowed_set,
                            GV_SearchResult *results,
                            const GV_HNSWFilteredConfig *config) {
    if (!index_ptr || !query || !results || k == 0) return -1;
    GV_HNSWIndex *index = (GV_HNSWIndex *)index_ptr;
    if (dimension != index->dimension || index->entry_point == SIZE_MAX) return 0;

    memset(results, 0, k * sizeof(GV_SearchResult));

    const GV_DistanceType dtype = (config && config->distance_type != GV_DISTANCE_EUCLIDEAN)
                                      ? config->distance_type
                                      : index->distance_type;
    const size_t ef = (config && config->ef_search > 0) ? config->ef_search : index->efSearch;

    const float *soa_base = index->soa_storage->data;
    const size_t soa_dim  = index->dimension;
    #define SF_VEC(vidx) (soa_base + (vidx) * soa_dim)

    /* Find the highest-level allowed node as entry point.
     * Walk from the global entry_point downward; if it's not allowed,
     * search neighbors at each level for a better (allowed) starting point. */
    size_t cur = index->entry_point;
    size_t curLevel = index->nodes[cur].level;
    if (curLevel > index->maxLevel) curLevel = index->maxLevel;

    /* If entry point itself is not in the filter, try to find an allowed one
     * by descending through the graph. If none found, fall back to brute-force
     * scan for the closest allowed node. */
    if (!is_allowed(allowed_set, cur)) {
        int found = 0;
        /* Greedy descent: at each level, scan all neighbors of current node */
        for (int lc = (int)curLevel; lc >= 0; --lc) {
            if ((size_t)lc > index->nodes[cur].level) continue;
            float cur_dist = hnsw_raw_distance_i(SF_VEC(index->nodes[cur].vector_index),
                                                  query, soa_dim, dtype);
            int improved = 1;
            while (improved) {
                improved = 0;
                int32_t *nbs = nb_begin_i(index, cur, (size_t)lc);
                size_t max_n = max_nb_at_level_i(index, (size_t)lc);
                for (size_t i = 0; i < max_n; ++i) {
                    int32_t nb = nbs[i];
                    if (nb < 0) break;
                    if (index->nodes[nb].deleted) continue;
                    if (!is_allowed(allowed_set, (size_t)nb)) continue;
                    float dist = hnsw_raw_distance_i(SF_VEC(index->nodes[nb].vector_index),
                                                      query, soa_dim, dtype);
                    if (dist < cur_dist) {
                        cur_dist = dist;
                        cur = (size_t)nb;
                        found = 1;
                        improved = 1;
                    }
                }
            }
            if (found) break;
        }
        /* If still not found, brute-force scan all nodes for the closest allowed one */
        if (!found) {
            float best_dist = FLT_MAX;
            size_t best_idx = SIZE_MAX;
            for (size_t i = 0; i < index->count; ++i) {
                if (index->nodes[i].deleted) continue;
                if (!is_allowed(allowed_set, i)) continue;
                float dist = hnsw_raw_distance_i(SF_VEC(index->nodes[i].vector_index),
                                                  query, soa_dim, dtype);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = i;
                }
            }
            if (best_idx == SIZE_MAX) return 0;  /* no allowed nodes at all */
            cur = best_idx;
            curLevel = index->nodes[cur].level;
            if (curLevel > index->maxLevel) curLevel = index->maxLevel;
        }
    }

    /* Greedy descent through upper layers (only allowed nodes) */
    for (int lc = (int)curLevel; lc > 0; --lc) {
        if ((size_t)lc > index->nodes[cur].level) continue;
        float cur_dist = hnsw_raw_distance_i(SF_VEC(index->nodes[cur].vector_index),
                                              query, soa_dim, dtype);
        int improved = 1;
        while (improved) {
            improved = 0;
            int32_t *nbs = nb_begin_i(index, cur, (size_t)lc);
            size_t max_n = max_nb_at_level_i(index, (size_t)lc);
            for (size_t i = 0; i < max_n; ++i) {
                int32_t nb = nbs[i];
                if (nb < 0) break;
                if (index->nodes[nb].deleted) continue;
                if (!is_allowed(allowed_set, (size_t)nb)) continue;
                float dist = hnsw_raw_distance_i(SF_VEC(index->nodes[nb].vector_index),
                                                  query, soa_dim, dtype);
                if (dist < cur_dist) {
                    cur_dist = dist;
                    cur = (size_t)nb;
                    improved = 1;
                }
            }
        }
    }

    /* Level-0 beam search */
    index->current_epoch++;
    if (index->current_epoch == 0) {
        memset(index->visited_epoch, 0, index->visited_capacity * sizeof(uint32_t));
        index->current_epoch = 1;
    }

    size_t buf_need = ef;
    if (buf_need > index->search_buf_size) {
        float *nd  = (float *)  gv_realloc(index->search_dis,  buf_need * sizeof(float));
        size_t *ni = (size_t *) gv_realloc(index->search_ids,  buf_need * sizeof(size_t));
        uint8_t *np = (uint8_t*)gv_realloc(index->search_proc, buf_need * sizeof(uint8_t));
        if (!nd || !ni || !np) {
            if (nd) index->search_dis  = nd;
            if (ni) index->search_ids  = ni;
            if (np) index->search_proc = np;
            return -1;
        }
        index->search_dis  = nd;
        index->search_ids  = ni;
        index->search_proc = np;
        index->search_buf_size = buf_need;
    }

    float  *heap_dis = index->search_dis;
    size_t *heap_ids = index->search_ids;
    uint8_t *heap_proc = index->search_proc;
    size_t heap_k = 0;

    /* Seed: the current node (guaranteed allowed after descent) */
    float cur_dist = hnsw_raw_distance_i(SF_VEC(index->nodes[cur].vector_index),
                                          query, soa_dim, dtype);
    mmheap_push_f(heap_dis, heap_ids, heap_proc, &heap_k, buf_need, cur, cur_dist);
    index->visited_epoch[cur] = index->current_epoch;

    for (;;) {
        float cand_dist;
        size_t cand_node = mmheap_pop_min_f(heap_dis, heap_ids, heap_proc, heap_k, &cand_dist);
        if (cand_node == SIZE_MAX) break;
        if (index->nodes[cand_node].deleted) continue;
        if (!is_allowed(allowed_set, cand_node)) continue;
        if (heap_k >= ef && cand_dist > heap_dis[0]) break;

        int32_t *nbs  = nb_begin_i(index, cand_node, 0);
        size_t   max_n = max_nb_at_level_i(index, 0);

        for (size_t i = 0; i < max_n; ++i) {
            int32_t nb = nbs[i];
            if (nb < 0) break;
            if (index->nodes[nb].deleted) continue;
            if (!is_allowed(allowed_set, (size_t)nb)) continue;
            if ((size_t)nb >= index->visited_capacity ||
                index->visited_epoch[nb] == index->current_epoch) continue;
            index->visited_epoch[nb] = index->current_epoch;

            float dist = hnsw_raw_distance_i(SF_VEC(index->nodes[nb].vector_index),
                                              query, soa_dim, dtype);
            mmheap_push_f(heap_dis, heap_ids, heap_proc, &heap_k, buf_need, (size_t)nb, dist);
        }
    }

    if (heap_k == 0) return 0;

    /* Extract top-k from max-heap via repeated min-extraction */
    size_t need = (heap_k < k) ? heap_k : k;
    size_t cand_count = (heap_k < 1024) ? heap_k : 1024;

    float  tmp_dis[1024];
    size_t tmp_ids[1024];
    memcpy(tmp_dis, heap_dis, cand_count * sizeof(float));
    memcpy(tmp_ids, heap_ids, cand_count * sizeof(size_t));

    typedef struct { size_t node_idx; float distance; } Cand;
    Cand sorted[512];
    size_t sorted_count = 0;

    for (size_t e = 0; e < need && e < 512; ++e) {
        float best_d = FLT_MAX;
        size_t best_i = SIZE_MAX;
        for (size_t i = 0; i < cand_count; ++i) {
            if (tmp_dis[i] < best_d) { best_d = tmp_dis[i]; best_i = i; }
        }
        if (best_i == SIZE_MAX) break;
        sorted[sorted_count].node_idx = tmp_ids[best_i];
        sorted[sorted_count].distance = best_d;
        sorted_count++;
        tmp_dis[best_i] = FLT_MAX;
    }

    size_t result_count = 0;
    for (size_t i = 0; i < sorted_count && result_count < k; ++i) {
        size_t cn = sorted[i].node_idx;
        if (index->nodes[cn].deleted) continue;

        const float *node_data = SF_VEC(index->nodes[cn].vector_index);

        GV_Vector *result_vec = (GV_Vector *)gv_alloc(sizeof(GV_Vector));
        if (!result_vec) continue;
        result_vec->data = (float *)gv_alloc(soa_dim * sizeof(float));
        if (!result_vec->data) { gv_free(result_vec); continue; }
        memcpy(result_vec->data, node_data, soa_dim * sizeof(float));
        result_vec->dimension = soa_dim;
        result_vec->metadata  = NULL;
        results[result_count].vector   = result_vec;
        results[result_count].distance = sorted[i].distance;
        results[result_count].id       = index->nodes[cn].vector_index;
        result_count++;
    }

    #undef SF_VEC
    return (int)result_count;
}

int gv_hnsw_range_search_filtered(void *index_ptr, const float *query, size_t dimension,
                                  float radius, const GV_IdBitmap *allowed_set,
                                  GV_SearchResult *results, size_t max_results,
                                  const GV_HNSWFilteredConfig *config) {
    if (!index_ptr || !query || !results || max_results == 0 || radius < 0.0f) return -1;
    GV_HNSWIndex *index = (GV_HNSWIndex *)index_ptr;
    if (dimension != index->dimension || index->entry_point == SIZE_MAX) return 0;

    /* Use a generous ef for range search to ensure we explore enough of the graph */
    GV_HNSWFilteredConfig range_cfg;
    if (config) {
        range_cfg = *config;
    } else {
        memset(&range_cfg, 0, sizeof(range_cfg));
    }
    if (range_cfg.ef_search == 0) {
        range_cfg.ef_search = (index->efSearch > 200) ? index->efSearch : 200;
    }
    size_t fetch_k = max_results;
    if (fetch_k < 100) fetch_k = 100;

    GV_SearchResult *tmp = (GV_SearchResult *)gv_alloc(fetch_k * sizeof(GV_SearchResult));
    if (!tmp) return -1;

    int n = gv_hnsw_search_filtered(index_ptr, query, dimension, fetch_k,
                                    allowed_set, tmp, &range_cfg);
    if (n <= 0) { gv_free(tmp); return n; }

    size_t out = 0;
    for (int i = 0; i < n; ++i) {
        if (tmp[i].distance <= radius && out < max_results) {
            results[out++] = tmp[i];
        } else {
            if (tmp[i].vector) vector_destroy((GV_Vector *)tmp[i].vector);
        }
    }
    gv_free(tmp);
    return (int)out;
}

/* ---- Filter bitmap builder ---- */

GV_IdBitmap *gv_build_filter_bitmap(void *payload_index_ptr,
                                    const char *filter_key, int filter_op,
                                    const char *filter_value) {
    if (!payload_index_ptr || !filter_key || !filter_value) return NULL;
    GV_PayloadIndex *pi = (GV_PayloadIndex *)payload_index_ptr;

    GV_IdBitmap *bitmap = gv_id_bitmap_create();
    if (!bitmap) return NULL;

    /* Determine field type from the payload index by doing an EQ query with each type
     * and see which yields results. We iterate over possible field types. */
    GV_PayloadQuery q;
    memset(&q, 0, sizeof(q));
    q.field_name = filter_key;

    /* Try string first (most common for filter expressions) */
    q.field_type = GV_FIELD_STRING;

    switch (filter_op) {
    case 0: /* EQ */
        q.op = GV_PAYLOAD_OP_EQ;
        q.value.string_val = filter_value;
        break;
    case 1: /* NE */
        q.op = GV_PAYLOAD_OP_NE;
        q.value.string_val = filter_value;
        break;
    case 2: /* GT */
        q.op = GV_PAYLOAD_OP_GT;
        q.value.string_val = filter_value;
        break;
    case 3: /* GE */
        q.op = GV_PAYLOAD_OP_GE;
        q.value.string_val = filter_value;
        break;
    case 4: /* LT */
        q.op = GV_PAYLOAD_OP_LT;
        q.value.string_val = filter_value;
        break;
    case 5: /* LE */
        q.op = GV_PAYLOAD_OP_LE;
        q.value.string_val = filter_value;
        break;
    case 6: /* CONTAINS */
        q.op = GV_PAYLOAD_OP_CONTAINS;
        q.value.string_val = filter_value;
        break;
    case 7: /* PREFIX */
        q.op = GV_PAYLOAD_OP_PREFIX;
        q.value.string_val = filter_value;
        break;
    default:
        gv_id_bitmap_free(bitmap);
        return NULL;
    }

    /* Query into a temporary array, then copy IDs into bitmap */
    size_t total = payload_index_total_entries(pi);
    if (total == 0) return bitmap;

    size_t *ids = (size_t *)gv_alloc(total * sizeof(size_t));
    if (!ids) { gv_id_bitmap_free(bitmap); return NULL; }

    int count = payload_index_query(pi, &q, ids, total);
    if (count > 0) {
        for (int i = 0; i < count; ++i) {
            gv_id_bitmap_add(bitmap, (uint64_t)ids[i]);
        }
    }

    gv_free(ids);
    return bitmap;
}
