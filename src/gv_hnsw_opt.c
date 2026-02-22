/**
 * @file gv_hnsw_opt.c
 * @brief Optimized HNSW index: inline quantized storage + incremental rebuild.
 *
 * Each graph node stores a scalar-quantized (4-bit or 8-bit) copy of its
 * vector inline, so candidate distance computation during traversal avoids
 * chasing a pointer to separate storage.  Full-precision vectors are kept
 * in a flat array for final reranking.
 *
 * The incremental rebuild walks nodes in batches, searches for better
 * neighbor candidates using the live graph, and prunes with the standard
 * HNSW heuristic.  It can optionally run in a background pthread.
 *
 * Thread safety: all public functions acquire a pthread_rwlock_t (read for
 * search/count/status, write for insert/rebuild/save).
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>
#include <errno.h>

#include "gigavector/gv_hnsw_opt.h"

/*  Constants  */

#define HNSW_OPT_MAGIC           0x484E5357  /* "HNSW" */
#define HNSW_OPT_VERSION         1
#define HNSW_OPT_INITIAL_CAP     1024
#define HNSW_OPT_MAX_LEVEL       32

/*  Inline quantized vector  */

/**
 * Per-dimension quantization parameters stored once in the index header.
 * Each dimension has its own min/max so the quantized byte (or nibble)
 * spans the tightest possible range.
 */
typedef struct {
    float *min_vals;   /**< Per-dimension minimums (dimension floats) */
    float *max_vals;   /**< Per-dimension maximums (dimension floats) */
    int bits;          /**< 4 or 8 */
    size_t bytes_per_vec; /**< Packed quantized bytes per vector */
} QuantParams;

/*  Graph node with inline storage  */

typedef struct {
    uint8_t *quant_vec;        /**< Quantized vector (bytes_per_vec bytes) */
    size_t *neighbors;         /**< Flat neighbor index array per level */
    size_t *neighbor_counts;   /**< Number of neighbors at each level */
    size_t *neighbor_caps;     /**< Capacity of neighbor array at each level */
    size_t level;              /**< Maximum level this node participates in */
    size_t label;              /**< User-supplied label */
    size_t flat_index;         /**< Index into full-precision flat array */
} InlineNode;

/*  Index structure (opaque)  */

struct GV_HNSWInlineIndex {
    /* Parameters */
    size_t dimension;
    size_t max_elements;
    size_t M;                  /**< Connections per node (upper layers) */
    size_t M0;                 /**< Connections at layer 0 (2 * M) */
    size_t ef_construction;
    double level_mult;         /**< 1.0 / ln(M) */

    /* Inline config */
    int quant_bits;
    int enable_prefetch;
    size_t prefetch_distance;

    /* Quantization */
    QuantParams qparams;

    /* Nodes */
    InlineNode *nodes;
    size_t count;
    size_t capacity;

    /* Full-precision vectors (count * dimension floats) */
    float *vectors;
    size_t vectors_cap;

    /* Entry point */
    size_t entry_point;        /**< Index into nodes[], SIZE_MAX when empty */
    size_t max_level_cur;      /**< Current maximum level in the graph */

    /* Rebuild state */
    GV_HNSWRebuildStats rebuild_stats;
    int rebuild_running;
    pthread_t rebuild_thread;

    /* Thread safety */
    pthread_rwlock_t rwlock;
};

/*  Internal helpers: quantization  */

static size_t quant_bytes_needed(size_t dimension, int bits) {
    return (dimension * (size_t)bits + 7) / 8;
}

/**
 * Initialize quantization params with neutral min/max (will be updated on
 * first inserts).
 */
static int qparams_init(QuantParams *qp, size_t dimension, int bits) {
    qp->bits = bits;
    qp->bytes_per_vec = quant_bytes_needed(dimension, bits);
    qp->min_vals = (float *)malloc(dimension * sizeof(float));
    qp->max_vals = (float *)malloc(dimension * sizeof(float));
    if (qp->min_vals == NULL || qp->max_vals == NULL) {
        free(qp->min_vals);
        free(qp->max_vals);
        qp->min_vals = NULL;
        qp->max_vals = NULL;
        return -1;
    }
    for (size_t i = 0; i < dimension; ++i) {
        qp->min_vals[i] = FLT_MAX;
        qp->max_vals[i] = -FLT_MAX;
    }
    return 0;
}

static void qparams_destroy(QuantParams *qp) {
    free(qp->min_vals);
    free(qp->max_vals);
    qp->min_vals = NULL;
    qp->max_vals = NULL;
}

/** Update per-dimension min/max from a new vector. */
static void qparams_update(QuantParams *qp, const float *vec, size_t dimension) {
    for (size_t i = 0; i < dimension; ++i) {
        if (vec[i] < qp->min_vals[i]) qp->min_vals[i] = vec[i];
        if (vec[i] > qp->max_vals[i]) qp->max_vals[i] = vec[i];
    }
}

/** Quantize a float vector into a caller-allocated uint8_t buffer. */
static void quantize_vector(const QuantParams *qp, const float *vec,
                            size_t dimension, uint8_t *out) {
    size_t max_quant = ((size_t)1 << qp->bits) - 1;
    memset(out, 0, qp->bytes_per_vec);

    for (size_t i = 0; i < dimension; ++i) {
        float minv = qp->min_vals[i];
        float maxv = qp->max_vals[i];
        float range = maxv - minv;
        size_t qval = 0;

        if (range > 0.0f) {
            float norm = (vec[i] - minv) / range;
            if (norm < 0.0f) norm = 0.0f;
            if (norm > 1.0f) norm = 1.0f;
            qval = (size_t)(norm * (float)max_quant + 0.5f);
            if (qval > max_quant) qval = max_quant;
        }

        if (qp->bits == 8) {
            out[i] = (uint8_t)qval;
        } else { /* 4-bit */
            size_t byte_idx = i / 2;
            if (i % 2 == 0) {
                out[byte_idx] |= (uint8_t)(qval << 4);
            } else {
                out[byte_idx] |= (uint8_t)(qval & 0x0F);
            }
        }
    }
}

/**
 * Compute approximate L2 distance between a float query and a quantized
 * vector.  Dequantizes on-the-fly without allocating.
 */
static float distance_query_quantized(const float *query,
                                      const uint8_t *qvec,
                                      const QuantParams *qp,
                                      size_t dimension) {
    size_t max_quant = ((size_t)1 << qp->bits) - 1;
    float dist = 0.0f;

    for (size_t i = 0; i < dimension; ++i) {
        size_t qval;
        if (qp->bits == 8) {
            qval = qvec[i];
        } else { /* 4-bit */
            size_t byte_idx = i / 2;
            qval = (i % 2 == 0) ? (qvec[byte_idx] >> 4) & 0x0F
                                 : qvec[byte_idx] & 0x0F;
        }

        float minv = qp->min_vals[i];
        float range = qp->max_vals[i] - minv;
        float dequant = minv + ((float)qval / (float)max_quant) * range;
        float diff = query[i] - dequant;
        dist += diff * diff;
    }
    return dist;
}

/** Full-precision L2 squared distance. */
static float distance_l2(const float *a, const float *b, size_t dimension) {
    float dist = 0.0f;
    for (size_t i = 0; i < dimension; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

/*  Internal helpers: level assignment  */

static size_t assign_level(double level_mult) {
    double r = (double)rand() / ((double)RAND_MAX + 1.0);
    if (r <= 0.0) r = 1e-12;
    size_t level = (size_t)(-log(r) * level_mult);
    if (level > HNSW_OPT_MAX_LEVEL) level = HNSW_OPT_MAX_LEVEL;
    return level;
}

/*  Internal helpers: node management  */

static int node_init(InlineNode *node, size_t level, size_t label,
                     size_t flat_index, size_t M, size_t M0,
                     size_t bytes_per_vec) {
    node->level = level;
    node->label = label;
    node->flat_index = flat_index;

    node->quant_vec = (uint8_t *)calloc(bytes_per_vec, sizeof(uint8_t));
    node->neighbor_counts = (size_t *)calloc(level + 1, sizeof(size_t));
    node->neighbor_caps = (size_t *)malloc((level + 1) * sizeof(size_t));
    node->neighbors = NULL;

    if (node->quant_vec == NULL || node->neighbor_counts == NULL ||
        node->neighbor_caps == NULL) {
        free(node->quant_vec);
        free(node->neighbor_counts);
        free(node->neighbor_caps);
        return -1;
    }

    /* Allocate per-level neighbor arrays.  Layer 0 gets M0 slots,
     * upper layers get M slots. */
    size_t total_slots = 0;
    for (size_t l = 0; l <= level; ++l) {
        size_t cap = (l == 0) ? M0 : M;
        node->neighbor_caps[l] = cap;
        total_slots += cap;
    }

    node->neighbors = (size_t *)malloc(total_slots * sizeof(size_t));
    if (node->neighbors == NULL) {
        free(node->quant_vec);
        free(node->neighbor_counts);
        free(node->neighbor_caps);
        return -1;
    }

    return 0;
}

/** Return pointer to the start of the neighbor array for a given level. */
static size_t *node_neighbors_at(const InlineNode *node, size_t level) {
    size_t offset = 0;
    for (size_t l = 0; l < level; ++l) {
        offset += node->neighbor_caps[l];
    }
    return node->neighbors + offset;
}

static void node_destroy(InlineNode *node) {
    if (node == NULL) return;
    free(node->quant_vec);
    free(node->neighbors);
    free(node->neighbor_counts);
    free(node->neighbor_caps);
    node->quant_vec = NULL;
    node->neighbors = NULL;
    node->neighbor_counts = NULL;
    node->neighbor_caps = NULL;
}

/*  Internal helpers: min-heap for candidate list  */

typedef struct {
    size_t node_idx;
    float distance;
} Candidate;

static int candidate_cmp_asc(const void *a, const void *b) {
    float da = ((const Candidate *)a)->distance;
    float db = ((const Candidate *)b)->distance;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

/*  Internal helpers: search layer (greedy with ef candidates)  */

/**
 * Search a single layer starting from entry_id, collecting up to ef
 * candidates.  Returns the number of candidates found.
 *
 * @param use_quant  If non-zero, use quantized inline distance for speed.
 */
static size_t search_layer(const GV_HNSWInlineIndex *idx,
                           const float *query, size_t entry_id,
                           size_t ef, size_t layer, int use_quant,
                           Candidate *results, size_t results_cap) {
    if (idx->count == 0 || entry_id >= idx->count) return 0;

    /* Visited set (bitmap) */
    size_t visited_bytes = (idx->count + 7) / 8;
    uint8_t *visited = (uint8_t *)calloc(visited_bytes, 1);
    if (visited == NULL) return 0;

    /* Working candidate list (sorted ascending by distance) */
    size_t cand_cap = ef + 1;
    Candidate *cand = (Candidate *)malloc(cand_cap * sizeof(Candidate));
    if (cand == NULL) {
        free(visited);
        return 0;
    }

    /* Seed with entry */
    const InlineNode *ep = &idx->nodes[entry_id];
    float ep_dist;
    if (use_quant) {
        ep_dist = distance_query_quantized(query, ep->quant_vec,
                                           &idx->qparams, idx->dimension);
    } else {
        ep_dist = distance_l2(query,
                              idx->vectors + ep->flat_index * idx->dimension,
                              idx->dimension);
    }
    cand[0].node_idx = entry_id;
    cand[0].distance = ep_dist;
    size_t cand_count = 1;
    visited[entry_id / 8] |= (uint8_t)(1 << (entry_id % 8));

    size_t scan_pos = 0;

    while (scan_pos < cand_count) {
        size_t cur_idx = cand[scan_pos].node_idx;
        scan_pos++;

        const InlineNode *cur = &idx->nodes[cur_idx];
        if (layer > cur->level) continue;

        size_t *nbrs = node_neighbors_at(cur, layer);
        size_t nbr_count = cur->neighbor_counts[layer];

        for (size_t i = 0; i < nbr_count; ++i) {
            size_t nbr_id = nbrs[i];
            if (nbr_id >= idx->count) continue;

            /* Prefetch next neighbor data ahead of time */
            if (idx->enable_prefetch && i + idx->prefetch_distance < nbr_count) {
                size_t pf_id = nbrs[i + idx->prefetch_distance];
                if (pf_id < idx->count) {
                    __builtin_prefetch(idx->nodes[pf_id].quant_vec, 0, 1);
                }
            }

            if (visited[nbr_id / 8] & (1 << (nbr_id % 8))) continue;
            visited[nbr_id / 8] |= (uint8_t)(1 << (nbr_id % 8));

            const InlineNode *nbr = &idx->nodes[nbr_id];
            float d;
            if (use_quant) {
                d = distance_query_quantized(query, nbr->quant_vec,
                                             &idx->qparams, idx->dimension);
            } else {
                d = distance_l2(query,
                                idx->vectors + nbr->flat_index * idx->dimension,
                                idx->dimension);
            }

            /* Insert into candidate list if better than the worst, or list
             * is not yet full. */
            if (cand_count < ef) {
                cand[cand_count].node_idx = nbr_id;
                cand[cand_count].distance = d;
                cand_count++;
                qsort(cand, cand_count, sizeof(Candidate), candidate_cmp_asc);
            } else if (d < cand[cand_count - 1].distance) {
                cand[cand_count - 1].node_idx = nbr_id;
                cand[cand_count - 1].distance = d;
                qsort(cand, cand_count, sizeof(Candidate), candidate_cmp_asc);
            }
        }
    }

    /* Copy out */
    size_t out_count = (cand_count < results_cap) ? cand_count : results_cap;
    memcpy(results, cand, out_count * sizeof(Candidate));

    free(cand);
    free(visited);
    return out_count;
}

/*  Internal helpers: neighbor selection heuristic  */

/**
 * Standard HNSW neighbor selection: from a candidate list, greedily pick
 * neighbors that are closer to the target than to any already-selected
 * neighbor.  Fills selected[] with up to max_count indices.
 */
static size_t select_neighbors(const GV_HNSWInlineIndex *idx,
                               const float *target_vec,
                               const Candidate *candidates, size_t cand_count,
                               size_t max_count,
                               size_t *selected) {
    (void)target_vec; /* dist_to_target comes from the candidate struct */
    if (cand_count == 0 || max_count == 0) return 0;

    /* Work on a sorted copy */
    Candidate *sorted = (Candidate *)malloc(cand_count * sizeof(Candidate));
    if (sorted == NULL) return 0;
    memcpy(sorted, candidates, cand_count * sizeof(Candidate));
    qsort(sorted, cand_count, sizeof(Candidate), candidate_cmp_asc);

    size_t sel_count = 0;
    for (size_t i = 0; i < cand_count && sel_count < max_count; ++i) {
        size_t cid = sorted[i].node_idx;
        float dist_to_target = sorted[i].distance;

        /* Check if this candidate is closer to target than to any already
         * selected neighbor (the heuristic prune). */
        int keep = 1;
        for (size_t j = 0; j < sel_count; ++j) {
            float dist_between = distance_l2(
                idx->vectors + idx->nodes[cid].flat_index * idx->dimension,
                idx->vectors + idx->nodes[selected[j]].flat_index * idx->dimension,
                idx->dimension);
            if (dist_between < dist_to_target) {
                keep = 0;
                break;
            }
        }

        if (keep) {
            selected[sel_count++] = cid;
        }
    }

    free(sorted);
    return sel_count;
}

/*  Internal helpers: connect a new node at a given level  */

static void connect_node(GV_HNSWInlineIndex *idx, size_t new_id,
                         const Candidate *candidates, size_t cand_count,
                         size_t level) {
    size_t max_conn = (level == 0) ? idx->M0 : idx->M;

    /* Select best neighbors for the new node */
    const float *new_vec = idx->vectors + idx->nodes[new_id].flat_index * idx->dimension;
    size_t *selected = (size_t *)malloc(max_conn * sizeof(size_t));
    if (selected == NULL) return;

    size_t sel_count = select_neighbors(idx, new_vec, candidates, cand_count,
                                        max_conn, selected);

    /* Set outgoing edges from new node */
    InlineNode *new_node = &idx->nodes[new_id];
    size_t *nbrs = node_neighbors_at(new_node, level);
    new_node->neighbor_counts[level] = sel_count;
    memcpy(nbrs, selected, sel_count * sizeof(size_t));

    /* Add reverse edges and prune if over capacity */
    for (size_t i = 0; i < sel_count; ++i) {
        size_t nbr_id = selected[i];
        InlineNode *nbr_node = &idx->nodes[nbr_id];
        if (level > nbr_node->level) continue;

        size_t *nbr_nbrs = node_neighbors_at(nbr_node, level);
        size_t nbr_cnt = nbr_node->neighbor_counts[level];
        size_t nbr_cap = nbr_node->neighbor_caps[level];

        if (nbr_cnt < nbr_cap) {
            /* Simply append */
            nbr_nbrs[nbr_cnt] = new_id;
            nbr_node->neighbor_counts[level] = nbr_cnt + 1;
        } else {
            /* Need to prune: build candidate list with the new node included,
             * then re-select. */
            size_t prune_count = nbr_cnt + 1;
            Candidate *prune_cands = (Candidate *)malloc(prune_count * sizeof(Candidate));
            if (prune_cands == NULL) continue;

            const float *nbr_vec = idx->vectors + nbr_node->flat_index * idx->dimension;
            for (size_t j = 0; j < nbr_cnt; ++j) {
                prune_cands[j].node_idx = nbr_nbrs[j];
                prune_cands[j].distance = distance_l2(
                    nbr_vec,
                    idx->vectors + idx->nodes[nbr_nbrs[j]].flat_index * idx->dimension,
                    idx->dimension);
            }
            prune_cands[nbr_cnt].node_idx = new_id;
            prune_cands[nbr_cnt].distance = distance_l2(nbr_vec, new_vec,
                                                         idx->dimension);

            size_t *pruned = (size_t *)malloc(nbr_cap * sizeof(size_t));
            if (pruned == NULL) {
                free(prune_cands);
                continue;
            }

            size_t pruned_count = select_neighbors(idx, nbr_vec, prune_cands,
                                                   prune_count, nbr_cap, pruned);
            memcpy(nbr_nbrs, pruned, pruned_count * sizeof(size_t));
            nbr_node->neighbor_counts[level] = pruned_count;

            free(pruned);
            free(prune_cands);
        }
    }

    free(selected);
}

/*  Internal helpers: time measurement  */

static double time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/*  Public API: lifecycle  */

GV_HNSWInlineIndex *gv_hnsw_inline_create(size_t dimension, size_t max_elements,
                                           size_t M, size_t ef_construction,
                                           const GV_HNSWInlineConfig *config) {
    if (dimension == 0 || max_elements == 0 || M == 0) {
        return NULL;
    }

    GV_HNSWInlineIndex *idx = (GV_HNSWInlineIndex *)calloc(1, sizeof(GV_HNSWInlineIndex));
    if (idx == NULL) {
        return NULL;
    }

    idx->dimension = dimension;
    idx->max_elements = max_elements;
    idx->M = M;
    idx->M0 = 2 * M;
    idx->ef_construction = (ef_construction > 0) ? ef_construction : 200;
    idx->level_mult = 1.0 / log((double)M);

    /* Inline config */
    idx->quant_bits = 8;
    idx->enable_prefetch = 0;
    idx->prefetch_distance = 2;
    if (config != NULL) {
        if (config->quant_bits == 4 || config->quant_bits == 8) {
            idx->quant_bits = config->quant_bits;
        }
        idx->enable_prefetch = config->enable_prefetch ? 1 : 0;
        if (config->prefetch_distance > 0) {
            idx->prefetch_distance = config->prefetch_distance;
        }
    }

    /* Quantization params */
    if (qparams_init(&idx->qparams, dimension, idx->quant_bits) != 0) {
        free(idx);
        return NULL;
    }

    /* Node array */
    idx->capacity = (max_elements < HNSW_OPT_INITIAL_CAP)
                        ? max_elements : HNSW_OPT_INITIAL_CAP;
    idx->nodes = (InlineNode *)calloc(idx->capacity, sizeof(InlineNode));
    if (idx->nodes == NULL) {
        qparams_destroy(&idx->qparams);
        free(idx);
        return NULL;
    }

    /* Full-precision vector storage */
    idx->vectors_cap = idx->capacity;
    idx->vectors = (float *)malloc(idx->vectors_cap * dimension * sizeof(float));
    if (idx->vectors == NULL) {
        free(idx->nodes);
        qparams_destroy(&idx->qparams);
        free(idx);
        return NULL;
    }

    idx->count = 0;
    idx->entry_point = SIZE_MAX;
    idx->max_level_cur = 0;

    /* Rebuild state */
    memset(&idx->rebuild_stats, 0, sizeof(GV_HNSWRebuildStats));
    idx->rebuild_running = 0;

    /* Thread safety */
    if (pthread_rwlock_init(&idx->rwlock, NULL) != 0) {
        free(idx->vectors);
        free(idx->nodes);
        qparams_destroy(&idx->qparams);
        free(idx);
        return NULL;
    }

    return idx;
}

void gv_hnsw_inline_destroy(GV_HNSWInlineIndex *idx) {
    if (idx == NULL) return;

    /* Wait for background rebuild to finish if running */
    if (idx->rebuild_running) {
        pthread_join(idx->rebuild_thread, NULL);
        idx->rebuild_running = 0;
    }

    for (size_t i = 0; i < idx->count; ++i) {
        node_destroy(&idx->nodes[i]);
    }
    free(idx->nodes);
    free(idx->vectors);
    qparams_destroy(&idx->qparams);
    pthread_rwlock_destroy(&idx->rwlock);
    free(idx);
}

/*  Public API: insert  */

int gv_hnsw_inline_insert(GV_HNSWInlineIndex *idx, const float *vector,
                           size_t label) {
    if (idx == NULL || vector == NULL) return -1;

    pthread_rwlock_wrlock(&idx->rwlock);

    if (idx->count >= idx->max_elements) {
        pthread_rwlock_unlock(&idx->rwlock);
        return -1;
    }

    /* Grow node array if needed */
    if (idx->count >= idx->capacity) {
        size_t new_cap = idx->capacity * 2;
        if (new_cap > idx->max_elements) new_cap = idx->max_elements;

        InlineNode *new_nodes = (InlineNode *)realloc(
            idx->nodes, new_cap * sizeof(InlineNode));
        if (new_nodes == NULL) {
            pthread_rwlock_unlock(&idx->rwlock);
            return -1;
        }
        memset(new_nodes + idx->capacity, 0,
               (new_cap - idx->capacity) * sizeof(InlineNode));
        idx->nodes = new_nodes;
        idx->capacity = new_cap;
    }

    /* Grow vector storage if needed */
    if (idx->count >= idx->vectors_cap) {
        size_t new_vcap = idx->vectors_cap * 2;
        if (new_vcap > idx->max_elements) new_vcap = idx->max_elements;

        float *new_vecs = (float *)realloc(
            idx->vectors, new_vcap * idx->dimension * sizeof(float));
        if (new_vecs == NULL) {
            pthread_rwlock_unlock(&idx->rwlock);
            return -1;
        }
        idx->vectors = new_vecs;
        idx->vectors_cap = new_vcap;
    }

    /* Store full-precision vector */
    size_t new_id = idx->count;
    memcpy(idx->vectors + new_id * idx->dimension, vector,
           idx->dimension * sizeof(float));

    /* Update quantization min/max and quantize */
    qparams_update(&idx->qparams, vector, idx->dimension);

    /* Assign level */
    size_t level = assign_level(idx->level_mult);

    /* Initialize node */
    if (node_init(&idx->nodes[new_id], level, label, new_id,
                  idx->M, idx->M0, idx->qparams.bytes_per_vec) != 0) {
        pthread_rwlock_unlock(&idx->rwlock);
        return -1;
    }

    quantize_vector(&idx->qparams, vector, idx->dimension,
                    idx->nodes[new_id].quant_vec);

    idx->count++;

    /* First node: just set as entry point */
    if (idx->count == 1) {
        idx->entry_point = new_id;
        idx->max_level_cur = level;
        pthread_rwlock_unlock(&idx->rwlock);
        return 0;
    }

    /* Greedy descent from top level to the new node's level + 1 */
    size_t cur_ep = idx->entry_point;
    for (size_t lc = idx->max_level_cur; lc > level && lc > 0; --lc) {
        Candidate best;
        best.node_idx = cur_ep;
        best.distance = distance_l2(vector,
                                    idx->vectors + idx->nodes[cur_ep].flat_index * idx->dimension,
                                    idx->dimension);

        int changed = 1;
        while (changed) {
            changed = 0;
            const InlineNode *cur = &idx->nodes[cur_ep];
            if (lc > cur->level) break;

            size_t *nbrs = node_neighbors_at(cur, lc);
            size_t nbr_cnt = cur->neighbor_counts[lc];

            for (size_t i = 0; i < nbr_cnt; ++i) {
                size_t nbr_id = nbrs[i];
                if (nbr_id >= idx->count) continue;
                float d = distance_l2(vector,
                                      idx->vectors + idx->nodes[nbr_id].flat_index * idx->dimension,
                                      idx->dimension);
                if (d < best.distance) {
                    best.node_idx = nbr_id;
                    best.distance = d;
                    changed = 1;
                }
            }
            cur_ep = best.node_idx;
        }
    }

    /* For each level from min(level, max_level_cur) down to 0, search and
     * connect */
    size_t insert_top = (level < idx->max_level_cur) ? level : idx->max_level_cur;
    Candidate *cands = (Candidate *)malloc(idx->ef_construction * sizeof(Candidate));
    if (cands == NULL) {
        pthread_rwlock_unlock(&idx->rwlock);
        return -1;
    }

    for (size_t lc = insert_top + 1; lc > 0; --lc) {
        size_t layer = lc - 1;
        size_t found = search_layer(idx, vector, cur_ep, idx->ef_construction,
                                    layer, 1 /* use quantized */, cands,
                                    idx->ef_construction);
        if (found > 0) {
            connect_node(idx, new_id, cands, found, layer);
            cur_ep = cands[0].node_idx; /* closest becomes next entry */
        }
    }

    free(cands);

    /* Update entry point if new node has a higher level */
    if (level > idx->max_level_cur) {
        idx->entry_point = new_id;
        idx->max_level_cur = level;
    }

    pthread_rwlock_unlock(&idx->rwlock);
    return 0;
}

/*  Public API: search  */

int gv_hnsw_inline_search(const GV_HNSWInlineIndex *idx, const float *query,
                           size_t k, size_t ef_search,
                           size_t *labels, float *distances) {
    if (idx == NULL || query == NULL || labels == NULL || distances == NULL ||
        k == 0) {
        return -1;
    }

    /* Cast away const for the rwlock (readers don't modify logical state) */
    pthread_rwlock_rdlock((pthread_rwlock_t *)&idx->rwlock);

    if (idx->count == 0 || idx->entry_point == SIZE_MAX) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);
        return 0;
    }

    if (ef_search < k) ef_search = k;

    /* Greedy descent from top level to layer 1 */
    size_t cur_ep = idx->entry_point;
    for (size_t lc = idx->max_level_cur; lc > 0; --lc) {
        float best_dist = distance_query_quantized(
            query, idx->nodes[cur_ep].quant_vec, &idx->qparams, idx->dimension);
        size_t best_id = cur_ep;

        int changed = 1;
        while (changed) {
            changed = 0;
            const InlineNode *cur = &idx->nodes[cur_ep];
            if (lc > cur->level) break;

            size_t *nbrs = node_neighbors_at(cur, lc);
            size_t nbr_cnt = cur->neighbor_counts[lc];

            for (size_t i = 0; i < nbr_cnt; ++i) {
                size_t nbr_id = nbrs[i];
                if (nbr_id >= idx->count) continue;

                /* Prefetch */
                if (idx->enable_prefetch &&
                    i + idx->prefetch_distance < nbr_cnt) {
                    size_t pf_id = nbrs[i + idx->prefetch_distance];
                    if (pf_id < idx->count) {
                        __builtin_prefetch(idx->nodes[pf_id].quant_vec, 0, 1);
                    }
                }

                float d = distance_query_quantized(
                    query, idx->nodes[nbr_id].quant_vec, &idx->qparams,
                    idx->dimension);
                if (d < best_dist) {
                    best_dist = d;
                    best_id = nbr_id;
                    changed = 1;
                }
            }
            cur_ep = best_id;
        }
    }

    /* Layer 0 search with ef candidates (quantized distances) */
    Candidate *cands = (Candidate *)malloc(ef_search * sizeof(Candidate));
    if (cands == NULL) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);
        return -1;
    }

    size_t found = search_layer(idx, query, cur_ep, ef_search, 0,
                                1 /* use quantized */, cands, ef_search);

    /* Rerank top candidates with full-precision vectors */
    size_t rerank_count = (found < ef_search) ? found : ef_search;
    for (size_t i = 0; i < rerank_count; ++i) {
        size_t nid = cands[i].node_idx;
        cands[i].distance = distance_l2(
            query, idx->vectors + idx->nodes[nid].flat_index * idx->dimension,
            idx->dimension);
    }
    qsort(cands, rerank_count, sizeof(Candidate), candidate_cmp_asc);

    /* Output top-k */
    size_t out_count = (rerank_count < k) ? rerank_count : k;
    for (size_t i = 0; i < out_count; ++i) {
        labels[i] = idx->nodes[cands[i].node_idx].label;
        distances[i] = cands[i].distance;
    }

    free(cands);
    pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);
    return (int)out_count;
}

/*  Internal: incremental rebuild worker  */

typedef struct {
    GV_HNSWInlineIndex *idx;
    float connectivity_ratio;
    size_t batch_size;
} RebuildArgs;

static void *rebuild_worker(void *arg) {
    RebuildArgs *rargs = (RebuildArgs *)arg;
    GV_HNSWInlineIndex *idx = rargs->idx;
    float conn_ratio = rargs->connectivity_ratio;
    size_t batch_size = rargs->batch_size;
    free(rargs);

    double start = time_ms();
    size_t edges_added = 0;
    size_t edges_removed = 0;
    size_t nodes_processed = 0;

    /* Process nodes in batches */
    for (size_t batch_start = 0; batch_start < idx->count;
         batch_start += batch_size) {
        size_t batch_end = batch_start + batch_size;
        if (batch_end > idx->count) batch_end = idx->count;

        pthread_rwlock_wrlock(&idx->rwlock);

        for (size_t ni = batch_start; ni < batch_end; ++ni) {
            InlineNode *node = &idx->nodes[ni];
            const float *node_vec = idx->vectors + node->flat_index * idx->dimension;

            /* Process layer 0 (the most impactful layer) */
            if (node->level == SIZE_MAX) continue; /* defensive */

            size_t *nbrs = node_neighbors_at(node, 0);
            size_t old_count = node->neighbor_counts[0];
            size_t max_conn = idx->M0;

            /* Determine how many existing edges to retain */
            size_t retain = (size_t)((float)old_count * conn_ratio + 0.5f);
            if (retain > old_count) retain = old_count;

            /* Search the graph for new candidates */
            size_t search_ef = max_conn * 2;
            if (search_ef < 32) search_ef = 32;
            Candidate *cands = (Candidate *)malloc(search_ef * sizeof(Candidate));
            if (cands == NULL) continue;

            /* Start search from the current entry point */
            size_t found = search_layer(idx, node_vec, idx->entry_point,
                                        search_ef, 0, 1, cands, search_ef);

            /* Remove self from candidates */
            size_t filtered = 0;
            for (size_t c = 0; c < found; ++c) {
                if (cands[c].node_idx != ni) {
                    cands[filtered++] = cands[c];
                }
            }

            /* Select neighbors using heuristic */
            size_t *selected = (size_t *)malloc(max_conn * sizeof(size_t));
            if (selected == NULL) {
                free(cands);
                continue;
            }

            size_t new_count = select_neighbors(idx, node_vec, cands, filtered,
                                                max_conn, selected);

            /* Count changes */
            for (size_t j = 0; j < new_count; ++j) {
                int was_neighbor = 0;
                for (size_t k = 0; k < old_count; ++k) {
                    if (nbrs[k] == selected[j]) {
                        was_neighbor = 1;
                        break;
                    }
                }
                if (!was_neighbor) edges_added++;
            }
            for (size_t k = 0; k < old_count; ++k) {
                int still_neighbor = 0;
                for (size_t j = 0; j < new_count; ++j) {
                    if (selected[j] == nbrs[k]) {
                        still_neighbor = 1;
                        break;
                    }
                }
                if (!still_neighbor) edges_removed++;
            }

            /* Apply new neighbor list */
            memcpy(nbrs, selected, new_count * sizeof(size_t));
            node->neighbor_counts[0] = new_count;

            free(selected);
            free(cands);
            nodes_processed++;
        }

        /* Update stats while we hold the lock */
        idx->rebuild_stats.nodes_processed = nodes_processed;
        idx->rebuild_stats.edges_added = edges_added;
        idx->rebuild_stats.edges_removed = edges_removed;
        idx->rebuild_stats.elapsed_ms = time_ms() - start;

        pthread_rwlock_unlock(&idx->rwlock);
    }

    /* Final stats */
    pthread_rwlock_wrlock(&idx->rwlock);
    idx->rebuild_stats.nodes_processed = nodes_processed;
    idx->rebuild_stats.edges_added = edges_added;
    idx->rebuild_stats.edges_removed = edges_removed;
    idx->rebuild_stats.elapsed_ms = time_ms() - start;
    idx->rebuild_stats.completed = 1;
    idx->rebuild_running = 0;
    pthread_rwlock_unlock(&idx->rwlock);

    return NULL;
}

/*  Public API: incremental rebuild  */

int gv_hnsw_inline_rebuild(GV_HNSWInlineIndex *idx,
                            const GV_HNSWRebuildConfig *config) {
    if (idx == NULL) return -1;

    pthread_rwlock_wrlock(&idx->rwlock);

    if (idx->rebuild_running) {
        pthread_rwlock_unlock(&idx->rwlock);
        return -1; /* rebuild already in progress */
    }

    /* Parse config */
    float conn_ratio = 0.8f;
    size_t batch_size = 1000;
    int background = 0;
    if (config != NULL) {
        if (config->connectivity_ratio > 0.0f && config->connectivity_ratio <= 1.0f) {
            conn_ratio = config->connectivity_ratio;
        }
        if (config->batch_size > 0) {
            batch_size = config->batch_size;
        }
        background = config->background ? 1 : 0;
    }

    /* Reset stats */
    memset(&idx->rebuild_stats, 0, sizeof(GV_HNSWRebuildStats));

    if (idx->count == 0) {
        idx->rebuild_stats.completed = 1;
        pthread_rwlock_unlock(&idx->rwlock);
        return 0;
    }

    RebuildArgs *rargs = (RebuildArgs *)malloc(sizeof(RebuildArgs));
    if (rargs == NULL) {
        pthread_rwlock_unlock(&idx->rwlock);
        return -1;
    }
    rargs->idx = idx;
    rargs->connectivity_ratio = conn_ratio;
    rargs->batch_size = batch_size;

    if (background) {
        idx->rebuild_running = 1;
        pthread_rwlock_unlock(&idx->rwlock);

        if (pthread_create(&idx->rebuild_thread, NULL, rebuild_worker, rargs) != 0) {
            free(rargs);
            pthread_rwlock_wrlock(&idx->rwlock);
            idx->rebuild_running = 0;
            pthread_rwlock_unlock(&idx->rwlock);
            return -1;
        }
        return 0;
    }

    /* Synchronous rebuild: release lock, worker will re-acquire per batch */
    pthread_rwlock_unlock(&idx->rwlock);
    rebuild_worker(rargs); /* rargs freed inside worker */
    return 0;
}

int gv_hnsw_inline_rebuild_status(const GV_HNSWInlineIndex *idx,
                                   GV_HNSWRebuildStats *stats) {
    if (idx == NULL || stats == NULL) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&idx->rwlock);
    *stats = idx->rebuild_stats;
    pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);
    return 0;
}

/*  Public API: utility  */

size_t gv_hnsw_inline_count(const GV_HNSWInlineIndex *idx) {
    if (idx == NULL) return 0;
    return idx->count;
}

/*  Internal helpers: file I/O  */

static int write_u32(FILE *f, uint32_t v) {
    return (fwrite(&v, sizeof(uint32_t), 1, f) == 1) ? 0 : -1;
}

static int write_u64(FILE *f, uint64_t v) {
    return (fwrite(&v, sizeof(uint64_t), 1, f) == 1) ? 0 : -1;
}

static int write_floats(FILE *f, const float *data, size_t count) {
    return (fwrite(data, sizeof(float), count, f) == count) ? 0 : -1;
}

static int write_bytes(FILE *f, const uint8_t *data, size_t count) {
    return (fwrite(data, 1, count, f) == count) ? 0 : -1;
}

static int read_u32(FILE *f, uint32_t *v) {
    return (fread(v, sizeof(uint32_t), 1, f) == 1) ? 0 : -1;
}

static int read_u64(FILE *f, uint64_t *v) {
    return (fread(v, sizeof(uint64_t), 1, f) == 1) ? 0 : -1;
}

static int read_floats(FILE *f, float *data, size_t count) {
    return (fread(data, sizeof(float), count, f) == count) ? 0 : -1;
}

static int read_bytes(FILE *f, uint8_t *data, size_t count) {
    return (fread(data, 1, count, f) == count) ? 0 : -1;
}

/*  Public API: save / load  */

int gv_hnsw_inline_save(const GV_HNSWInlineIndex *idx, const char *path) {
    if (idx == NULL || path == NULL) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&idx->rwlock);

    FILE *f = fopen(path, "wb");
    if (f == NULL) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);
        return -1;
    }

    int rc = 0;

    /* Header: magic, version, parameters */
    if (write_u32(f, HNSW_OPT_MAGIC) != 0) { rc = -1; goto done; }
    if (write_u32(f, HNSW_OPT_VERSION) != 0) { rc = -1; goto done; }
    if (write_u64(f, (uint64_t)idx->dimension) != 0) { rc = -1; goto done; }
    if (write_u64(f, (uint64_t)idx->max_elements) != 0) { rc = -1; goto done; }
    if (write_u64(f, (uint64_t)idx->M) != 0) { rc = -1; goto done; }
    if (write_u64(f, (uint64_t)idx->ef_construction) != 0) { rc = -1; goto done; }
    if (write_u32(f, (uint32_t)idx->quant_bits) != 0) { rc = -1; goto done; }
    if (write_u64(f, (uint64_t)idx->count) != 0) { rc = -1; goto done; }
    if (write_u64(f, (uint64_t)idx->entry_point) != 0) { rc = -1; goto done; }
    if (write_u64(f, (uint64_t)idx->max_level_cur) != 0) { rc = -1; goto done; }

    /* Quantization params: min/max per dimension */
    if (write_floats(f, idx->qparams.min_vals, idx->dimension) != 0) { rc = -1; goto done; }
    if (write_floats(f, idx->qparams.max_vals, idx->dimension) != 0) { rc = -1; goto done; }

    /* Full-precision vectors */
    if (write_floats(f, idx->vectors, idx->count * idx->dimension) != 0) { rc = -1; goto done; }

    /* Nodes: level, label, quantized vector, neighbor lists */
    for (size_t i = 0; i < idx->count; ++i) {
        const InlineNode *node = &idx->nodes[i];
        if (write_u64(f, (uint64_t)node->level) != 0) { rc = -1; goto done; }
        if (write_u64(f, (uint64_t)node->label) != 0) { rc = -1; goto done; }
        if (write_bytes(f, node->quant_vec, idx->qparams.bytes_per_vec) != 0) { rc = -1; goto done; }

        for (size_t l = 0; l <= node->level; ++l) {
            size_t cnt = node->neighbor_counts[l];
            if (write_u64(f, (uint64_t)cnt) != 0) { rc = -1; goto done; }
            size_t *nbrs = node_neighbors_at(node, l);
            for (size_t j = 0; j < cnt; ++j) {
                if (write_u64(f, (uint64_t)nbrs[j]) != 0) { rc = -1; goto done; }
            }
        }
    }

done:
    fclose(f);
    pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);
    return rc;
}

GV_HNSWInlineIndex *gv_hnsw_inline_load(const char *path) {
    if (path == NULL) return NULL;

    FILE *f = fopen(path, "rb");
    if (f == NULL) return NULL;

    uint32_t magic = 0, version = 0, qbits = 0;
    uint64_t dimension = 0, max_elements = 0, M = 0, ef_construction = 0;
    uint64_t count = 0, entry_point = 0, max_level_cur = 0;

    if (read_u32(f, &magic) != 0 || magic != HNSW_OPT_MAGIC) goto fail;
    if (read_u32(f, &version) != 0 || version != HNSW_OPT_VERSION) goto fail;
    if (read_u64(f, &dimension) != 0) goto fail;
    if (read_u64(f, &max_elements) != 0) goto fail;
    if (read_u64(f, &M) != 0) goto fail;
    if (read_u64(f, &ef_construction) != 0) goto fail;
    if (read_u32(f, &qbits) != 0) goto fail;
    if (read_u64(f, &count) != 0) goto fail;
    if (read_u64(f, &entry_point) != 0) goto fail;
    if (read_u64(f, &max_level_cur) != 0) goto fail;

    if (qbits != 4 && qbits != 8) goto fail;

    GV_HNSWInlineConfig cfg = {
        .quant_bits = (int)qbits,
        .enable_prefetch = 0,
        .prefetch_distance = 2
    };

    GV_HNSWInlineIndex *idx = gv_hnsw_inline_create(
        (size_t)dimension, (size_t)max_elements, (size_t)M,
        (size_t)ef_construction, &cfg);
    if (idx == NULL) goto fail;

    /* Read quantization params */
    if (read_floats(f, idx->qparams.min_vals, (size_t)dimension) != 0) {
        gv_hnsw_inline_destroy(idx);
        goto fail;
    }
    if (read_floats(f, idx->qparams.max_vals, (size_t)dimension) != 0) {
        gv_hnsw_inline_destroy(idx);
        goto fail;
    }

    /* Ensure capacity */
    if ((size_t)count > idx->capacity) {
        InlineNode *new_nodes = (InlineNode *)realloc(
            idx->nodes, (size_t)count * sizeof(InlineNode));
        if (new_nodes == NULL) { gv_hnsw_inline_destroy(idx); goto fail; }
        idx->nodes = new_nodes;
        idx->capacity = (size_t)count;
    }
    if ((size_t)count > idx->vectors_cap) {
        float *new_vecs = (float *)realloc(
            idx->vectors, (size_t)count * (size_t)dimension * sizeof(float));
        if (new_vecs == NULL) { gv_hnsw_inline_destroy(idx); goto fail; }
        idx->vectors = new_vecs;
        idx->vectors_cap = (size_t)count;
    }

    /* Read full-precision vectors */
    if (read_floats(f, idx->vectors, (size_t)count * (size_t)dimension) != 0) {
        gv_hnsw_inline_destroy(idx);
        goto fail;
    }

    /* Read nodes */
    for (size_t i = 0; i < (size_t)count; ++i) {
        uint64_t level64 = 0, label64 = 0;
        if (read_u64(f, &level64) != 0) { gv_hnsw_inline_destroy(idx); goto fail; }
        if (read_u64(f, &label64) != 0) { gv_hnsw_inline_destroy(idx); goto fail; }

        size_t level = (size_t)level64;
        if (node_init(&idx->nodes[i], level, (size_t)label64, i,
                      idx->M, idx->M0, idx->qparams.bytes_per_vec) != 0) {
            gv_hnsw_inline_destroy(idx);
            goto fail;
        }

        if (read_bytes(f, idx->nodes[i].quant_vec,
                       idx->qparams.bytes_per_vec) != 0) {
            gv_hnsw_inline_destroy(idx);
            goto fail;
        }

        for (size_t l = 0; l <= level; ++l) {
            uint64_t cnt64 = 0;
            if (read_u64(f, &cnt64) != 0) { gv_hnsw_inline_destroy(idx); goto fail; }
            size_t cnt = (size_t)cnt64;
            size_t cap = idx->nodes[i].neighbor_caps[l];

            size_t *nbrs = node_neighbors_at(&idx->nodes[i], l);
            size_t actual = (cnt < cap) ? cnt : cap;
            for (size_t j = 0; j < cnt; ++j) {
                uint64_t nbr_idx = 0;
                if (read_u64(f, &nbr_idx) != 0) { gv_hnsw_inline_destroy(idx); goto fail; }
                if (j < actual) {
                    nbrs[j] = (size_t)nbr_idx;
                }
            }
            idx->nodes[i].neighbor_counts[l] = actual;
        }

        idx->count++;
    }

    idx->entry_point = (size_t)entry_point;
    idx->max_level_cur = (size_t)max_level_cur;

    fclose(f);
    return idx;

fail:
    fclose(f);
    return NULL;
}
