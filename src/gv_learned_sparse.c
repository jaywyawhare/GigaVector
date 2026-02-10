/**
 * @file gv_learned_sparse.c
 * @brief Learned sparse vector index implementation (SPLADE, BGE-M3 sparse).
 *
 * Inverted index with float-weighted posting lists and optional WAND
 * (Weighted AND) optimization for efficient top-k retrieval.  Scoring is
 * a dot product over shared non-zero token dimensions.
 *
 * All public functions are protected by a pthread_rwlock_t.
 */

#include "gigavector/gv_learned_sparse.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <pthread.h>

/* ============================================================================
 * Internal Constants
 * ============================================================================ */

#define GV_LS_MAGIC       "GV_LSPA"
#define GV_LS_MAGIC_LEN   7
#define GV_LS_VERSION     1

#define GV_LS_INITIAL_DOC_CAPACITY      64
#define GV_LS_INITIAL_POSTING_CAPACITY  16
#define GV_LS_SCORE_MAP_BUCKETS         4096

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

/**
 * @brief A single entry in a posting list: (doc_id, weight).
 */
typedef struct {
    size_t doc_id;
    float  weight;
} GV_LSPosting;

/**
 * @brief Posting list for a single vocabulary token.
 *
 * Postings are stored in a sorted array (by doc_id, ascending).
 * When WAND is enabled, block-level maximum weights are maintained
 * for pruning during search.
 */
typedef struct {
    GV_LSPosting *postings;
    size_t        count;
    size_t        capacity;

    /* WAND block-level max weights.  block_maxw[i] is the maximum weight
     * among postings in the range [i * block_size, (i+1) * block_size). */
    float  *block_maxw;
    size_t  block_maxw_count;
    size_t  block_maxw_capacity;
} GV_LSPostingList;

/**
 * @brief Per-document metadata.
 */
typedef struct {
    size_t entry_count;         /**< Number of non-zero entries. */
    int    deleted;             /**< Non-zero when logically deleted. */
} GV_LSDocMeta;

/**
 * @brief Full learned sparse index state (opaque to callers).
 */
struct GV_LearnedSparseIndex {
    GV_LearnedSparseConfig config;

    /* Inverted index: one posting list per token_id (array of vocab_size). */
    GV_LSPostingList *posting_lists;

    /* Document metadata array. */
    GV_LSDocMeta *docs;
    size_t        doc_count;
    size_t        doc_capacity;

    /* Bookkeeping. */
    size_t active_docs;         /**< Non-deleted document count. */
    size_t total_postings;      /**< Sum of all posting list lengths. */
    size_t total_entry_count;   /**< Sum of entry_count across active docs. */

    pthread_rwlock_t rwlock;
};

/* ============================================================================
 * Internal: posting list helpers
 * ============================================================================ */

/**
 * @brief Append a posting to a posting list, maintaining doc_id sort order.
 *
 * New documents are always appended with ascending doc_ids during insertion,
 * so we append at the end.
 */
static int gv_ls_posting_append(GV_LSPostingList *pl, size_t doc_id, float weight,
                                size_t block_size) {
    /* Grow postings array if needed. */
    if (pl->count >= pl->capacity) {
        size_t new_cap = pl->capacity == 0 ? GV_LS_INITIAL_POSTING_CAPACITY
                                           : pl->capacity * 2;
        GV_LSPosting *new_arr = (GV_LSPosting *)realloc(
            pl->postings, new_cap * sizeof(GV_LSPosting));
        if (!new_arr) return -1;
        pl->postings = new_arr;
        pl->capacity = new_cap;
    }

    pl->postings[pl->count].doc_id = doc_id;
    pl->postings[pl->count].weight = weight;
    pl->count++;

    /* Update WAND block max weight. */
    if (block_size > 0) {
        size_t block_idx = (pl->count - 1) / block_size;

        /* Grow block_maxw array if needed. */
        if (block_idx >= pl->block_maxw_capacity) {
            size_t new_cap = pl->block_maxw_capacity == 0
                                 ? 8
                                 : pl->block_maxw_capacity * 2;
            while (new_cap <= block_idx) new_cap *= 2;

            float *new_bm = (float *)realloc(
                pl->block_maxw, new_cap * sizeof(float));
            if (!new_bm) return -1;

            /* Zero-init new entries. */
            for (size_t i = pl->block_maxw_capacity; i < new_cap; i++) {
                new_bm[i] = 0.0f;
            }
            pl->block_maxw = new_bm;
            pl->block_maxw_capacity = new_cap;
        }

        if (block_idx >= pl->block_maxw_count) {
            pl->block_maxw_count = block_idx + 1;
        }

        if (weight > pl->block_maxw[block_idx]) {
            pl->block_maxw[block_idx] = weight;
        }
    }

    return 0;
}

/**
 * @brief Free a posting list's internal arrays.
 */
static void gv_ls_posting_list_free(GV_LSPostingList *pl) {
    free(pl->postings);
    free(pl->block_maxw);
    pl->postings = NULL;
    pl->block_maxw = NULL;
    pl->count = 0;
    pl->capacity = 0;
    pl->block_maxw_count = 0;
    pl->block_maxw_capacity = 0;
}

/* ============================================================================
 * Internal: min-heap for top-k selection (root = smallest score)
 * ============================================================================ */

typedef struct {
    float  score;
    size_t doc_id;
} GV_LSHeapItem;

static void gv_ls_heap_sift_down(GV_LSHeapItem *heap, size_t size, size_t i) {
    while (1) {
        size_t l = 2 * i + 1;
        size_t r = l + 1;
        size_t smallest = i;
        if (l < size && heap[l].score < heap[smallest].score) smallest = l;
        if (r < size && heap[r].score < heap[smallest].score) smallest = r;
        if (smallest == i) break;
        GV_LSHeapItem tmp = heap[i];
        heap[i] = heap[smallest];
        heap[smallest] = tmp;
        i = smallest;
    }
}

static void gv_ls_heap_push(GV_LSHeapItem *heap, size_t *size, size_t capacity,
                             float score, size_t doc_id) {
    if (*size < capacity) {
        heap[*size].score  = score;
        heap[*size].doc_id = doc_id;
        (*size)++;
        /* Sift up. */
        size_t i = *size - 1;
        while (i > 0) {
            size_t parent = (i - 1) / 2;
            if (heap[i].score < heap[parent].score) {
                GV_LSHeapItem tmp = heap[i];
                heap[i] = heap[parent];
                heap[parent] = tmp;
                i = parent;
            } else {
                break;
            }
        }
    } else if (score > heap[0].score) {
        heap[0].score  = score;
        heap[0].doc_id = doc_id;
        gv_ls_heap_sift_down(heap, *size, 0);
    }
}

/* ============================================================================
 * Internal: simple hash map (doc_id -> float score) for non-WAND accumulation
 * ============================================================================ */

typedef struct GV_LSScoreEntry {
    size_t doc_id;
    float  score;
    int    used;
    struct GV_LSScoreEntry *next;
} GV_LSScoreEntry;

typedef struct {
    GV_LSScoreEntry **buckets;
    size_t            num_buckets;
} GV_LSScoreMap;

static GV_LSScoreMap *gv_ls_score_map_create(size_t num_buckets) {
    GV_LSScoreMap *map = (GV_LSScoreMap *)calloc(1, sizeof(GV_LSScoreMap));
    if (!map) return NULL;

    map->num_buckets = num_buckets;
    map->buckets = (GV_LSScoreEntry **)calloc(num_buckets, sizeof(GV_LSScoreEntry *));
    if (!map->buckets) {
        free(map);
        return NULL;
    }
    return map;
}

static void gv_ls_score_map_destroy(GV_LSScoreMap *map) {
    if (!map) return;
    for (size_t i = 0; i < map->num_buckets; i++) {
        GV_LSScoreEntry *e = map->buckets[i];
        while (e) {
            GV_LSScoreEntry *next = e->next;
            free(e);
            e = next;
        }
    }
    free(map->buckets);
    free(map);
}

static float *gv_ls_score_map_get_or_insert(GV_LSScoreMap *map, size_t doc_id) {
    size_t bucket = doc_id % map->num_buckets;
    GV_LSScoreEntry *e = map->buckets[bucket];
    while (e) {
        if (e->doc_id == doc_id) {
            return &e->score;
        }
        e = e->next;
    }

    /* Insert new entry. */
    e = (GV_LSScoreEntry *)calloc(1, sizeof(GV_LSScoreEntry));
    if (!e) return NULL;
    e->doc_id = doc_id;
    e->score  = 0.0f;
    e->used   = 1;
    e->next   = map->buckets[bucket];
    map->buckets[bucket] = e;
    return &e->score;
}

/* ============================================================================
 * Internal: WAND term cursor
 * ============================================================================ */

typedef struct {
    uint32_t token_id;
    float    query_weight;
    const GV_LSPostingList *pl;
    size_t   cursor;            /**< Current position in posting list. */
    float    max_contribution;  /**< query_weight * global_max_weight for this term. */
} GV_LSWandCursor;

/**
 * @brief Get the current doc_id a cursor points to, or SIZE_MAX if exhausted.
 */
static size_t gv_ls_cursor_doc(const GV_LSWandCursor *c) {
    if (c->cursor >= c->pl->count) return SIZE_MAX;
    return c->pl->postings[c->cursor].doc_id;
}

/**
 * @brief Advance cursor to the first posting with doc_id >= target.
 */
static void gv_ls_cursor_advance_to(GV_LSWandCursor *c, size_t target,
                                     size_t block_size) {
    if (c->cursor >= c->pl->count) return;

    /* Use block-level skip if WAND block info is available. */
    if (block_size > 0 && c->pl->block_maxw != NULL) {
        size_t block_idx = c->cursor / block_size;
        while (block_idx < c->pl->block_maxw_count) {
            size_t block_end_pos = (block_idx + 1) * block_size;
            if (block_end_pos > c->pl->count) {
                block_end_pos = c->pl->count;
            }
            /* If the last doc in this block is < target, skip the entire block. */
            if (block_end_pos > 0 &&
                c->pl->postings[block_end_pos - 1].doc_id < target) {
                c->cursor = block_end_pos;
                block_idx++;
                continue;
            }
            break;
        }
    }

    /* Linear scan within the block. */
    while (c->cursor < c->pl->count &&
           c->pl->postings[c->cursor].doc_id < target) {
        c->cursor++;
    }
}

/**
 * @brief Comparison for sorting cursors by current doc_id (ascending).
 */
static int gv_ls_cursor_cmp(const void *a, const void *b) {
    size_t da = gv_ls_cursor_doc((const GV_LSWandCursor *)a);
    size_t db = gv_ls_cursor_doc((const GV_LSWandCursor *)b);
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

/* ============================================================================
 * Internal: compute global max weight per posting list
 * ============================================================================ */

static float gv_ls_posting_list_max_weight(const GV_LSPostingList *pl) {
    float mx = 0.0f;
    for (size_t i = 0; i < pl->count; i++) {
        if (pl->postings[i].weight > mx) {
            mx = pl->postings[i].weight;
        }
    }
    return mx;
}

/* ============================================================================
 * Internal: WAND search
 * ============================================================================ */

static int gv_ls_search_wand(const GV_LearnedSparseIndex *idx,
                              const GV_SparseEntry *query, size_t query_count,
                              float min_score, size_t k,
                              GV_LearnedSparseResult *results) {
    /* Build cursors for query terms that have non-empty posting lists. */
    GV_LSWandCursor *cursors = (GV_LSWandCursor *)calloc(
        query_count, sizeof(GV_LSWandCursor));
    if (!cursors) return -1;

    size_t num_cursors = 0;
    for (size_t i = 0; i < query_count; i++) {
        uint32_t tid = query[i].token_id;
        if (tid >= idx->config.vocab_size) continue;

        const GV_LSPostingList *pl = &idx->posting_lists[tid];
        if (pl->count == 0) continue;

        float global_max = gv_ls_posting_list_max_weight(pl);
        if (global_max <= 0.0f) continue;

        cursors[num_cursors].token_id       = tid;
        cursors[num_cursors].query_weight   = query[i].weight;
        cursors[num_cursors].pl             = pl;
        cursors[num_cursors].cursor         = 0;
        cursors[num_cursors].max_contribution = query[i].weight * global_max;
        num_cursors++;
    }

    if (num_cursors == 0) {
        free(cursors);
        return 0;
    }

    /* Allocate result min-heap. */
    GV_LSHeapItem *heap = (GV_LSHeapItem *)malloc(k * sizeof(GV_LSHeapItem));
    if (!heap) {
        free(cursors);
        return -1;
    }
    size_t heap_size = 0;

    size_t block_size = idx->config.wand_block_size;

    /* Main WAND loop. */
    while (1) {
        /* Sort cursors by current doc_id. */
        qsort(cursors, num_cursors, sizeof(GV_LSWandCursor), gv_ls_cursor_cmp);

        /* Remove exhausted cursors from the end. */
        while (num_cursors > 0 &&
               gv_ls_cursor_doc(&cursors[num_cursors - 1]) == SIZE_MAX) {
            num_cursors--;
        }
        if (num_cursors == 0) break;

        /* Compute the threshold: the current k-th best score. */
        float threshold = min_score;
        if (heap_size >= k && heap[0].score > threshold) {
            threshold = heap[0].score;
        }

        /* Find the pivot: the smallest prefix of sorted cursors whose
         * cumulative max_contribution exceeds the threshold. */
        float upper_bound = 0.0f;
        size_t pivot = 0;
        for (pivot = 0; pivot < num_cursors; pivot++) {
            upper_bound += cursors[pivot].max_contribution;
            if (upper_bound > threshold) break;
        }

        /* If no pivot found, no more candidates can exceed threshold. */
        if (pivot >= num_cursors) break;

        size_t pivot_doc = gv_ls_cursor_doc(&cursors[pivot]);
        if (pivot_doc == SIZE_MAX) break;

        /* Check if all cursors up to pivot point to the same doc. */
        size_t first_doc = gv_ls_cursor_doc(&cursors[0]);

        if (first_doc == pivot_doc) {
            /* All terms up to pivot share the same doc: fully score it. */
            if (!idx->docs[pivot_doc].deleted) {
                float score = 0.0f;
                for (size_t c = 0; c < num_cursors; c++) {
                    if (gv_ls_cursor_doc(&cursors[c]) != pivot_doc) break;
                    score += cursors[c].query_weight *
                             cursors[c].pl->postings[cursors[c].cursor].weight;
                }

                if (score >= min_score) {
                    gv_ls_heap_push(heap, &heap_size, k, score, pivot_doc);
                }
            }

            /* Advance all cursors past pivot_doc. */
            for (size_t c = 0; c < num_cursors; c++) {
                if (gv_ls_cursor_doc(&cursors[c]) != pivot_doc) break;
                cursors[c].cursor++;
            }
        } else {
            /* Not all cursors at pivot_doc: advance cursors before pivot
             * to at least pivot_doc. */
            for (size_t c = 0; c < pivot; c++) {
                if (gv_ls_cursor_doc(&cursors[c]) < pivot_doc) {
                    gv_ls_cursor_advance_to(&cursors[c], pivot_doc, block_size);
                }
            }
        }
    }

    /* Extract results from min-heap in descending score order. */
    int n = (int)heap_size;
    for (int i = n - 1; i >= 0; i--) {
        results[i].doc_index = heap[0].doc_id;
        results[i].score     = heap[0].score;

        /* Pop root. */
        heap[0] = heap[heap_size - 1];
        heap_size--;
        if (heap_size > 0) {
            gv_ls_heap_sift_down(heap, heap_size, 0);
        }
    }

    free(heap);
    free(cursors);
    return n;
}

/* ============================================================================
 * Internal: non-WAND (accumulator) search
 * ============================================================================ */

static int gv_ls_search_accumulate(const GV_LearnedSparseIndex *idx,
                                    const GV_SparseEntry *query,
                                    size_t query_count,
                                    float min_score, size_t k,
                                    GV_LearnedSparseResult *results) {
    GV_LSScoreMap *map = gv_ls_score_map_create(GV_LS_SCORE_MAP_BUCKETS);
    if (!map) return -1;

    /* Accumulate scores: for each query term, iterate its posting list. */
    for (size_t q = 0; q < query_count; q++) {
        uint32_t tid = query[q].token_id;
        if (tid >= idx->config.vocab_size) continue;

        const GV_LSPostingList *pl = &idx->posting_lists[tid];
        float qw = query[q].weight;

        for (size_t p = 0; p < pl->count; p++) {
            size_t doc_id = pl->postings[p].doc_id;

            /* Skip deleted documents. */
            if (idx->docs[doc_id].deleted) continue;

            float *score_ptr = gv_ls_score_map_get_or_insert(map, doc_id);
            if (!score_ptr) {
                gv_ls_score_map_destroy(map);
                return -1;
            }
            *score_ptr += qw * pl->postings[p].weight;
        }
    }

    /* Extract top-k from the score map using a min-heap. */
    GV_LSHeapItem *heap = (GV_LSHeapItem *)malloc(k * sizeof(GV_LSHeapItem));
    if (!heap) {
        gv_ls_score_map_destroy(map);
        return -1;
    }
    size_t heap_size = 0;

    for (size_t b = 0; b < map->num_buckets; b++) {
        GV_LSScoreEntry *e = map->buckets[b];
        while (e) {
            if (e->used && e->score >= min_score) {
                gv_ls_heap_push(heap, &heap_size, k, e->score, e->doc_id);
            }
            e = e->next;
        }
    }

    /* Extract results from min-heap in descending score order. */
    int n = (int)heap_size;
    for (int i = n - 1; i >= 0; i--) {
        results[i].doc_index = heap[0].doc_id;
        results[i].score     = heap[0].score;

        heap[0] = heap[heap_size - 1];
        heap_size--;
        if (heap_size > 0) {
            gv_ls_heap_sift_down(heap, heap_size, 0);
        }
    }

    free(heap);
    gv_ls_score_map_destroy(map);
    return n;
}

/* ============================================================================
 * Internal: serialization helpers
 * ============================================================================ */

static int gv_ls_write_u32(FILE *f, uint32_t v) {
    return fwrite(&v, sizeof(uint32_t), 1, f) == 1 ? 0 : -1;
}

static int gv_ls_read_u32(FILE *f, uint32_t *v) {
    return (v && fread(v, sizeof(uint32_t), 1, f) == 1) ? 0 : -1;
}

static int gv_ls_write_u64(FILE *f, uint64_t v) {
    return fwrite(&v, sizeof(uint64_t), 1, f) == 1 ? 0 : -1;
}

static int gv_ls_read_u64(FILE *f, uint64_t *v) {
    return (v && fread(v, sizeof(uint64_t), 1, f) == 1) ? 0 : -1;
}

static int gv_ls_write_float(FILE *f, float v) {
    return fwrite(&v, sizeof(float), 1, f) == 1 ? 0 : -1;
}

static int gv_ls_read_float(FILE *f, float *v) {
    return (v && fread(v, sizeof(float), 1, f) == 1) ? 0 : -1;
}

/* ============================================================================
 * Configuration
 * ============================================================================ */

static const GV_LearnedSparseConfig DEFAULT_CONFIG = {
    .vocab_size      = 30522,
    .max_nonzeros    = 256,
    .use_wand        = 1,
    .wand_block_size = 128
};

void gv_ls_config_init(GV_LearnedSparseConfig *config) {
    if (!config) return;
    *config = DEFAULT_CONFIG;
}

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

GV_LearnedSparseIndex *gv_ls_create(const GV_LearnedSparseConfig *config) {
    GV_LearnedSparseConfig cfg = config ? *config : DEFAULT_CONFIG;

    /* Apply defaults for zero values. */
    if (cfg.vocab_size == 0)      cfg.vocab_size      = 30522;
    if (cfg.max_nonzeros == 0)    cfg.max_nonzeros    = 256;
    if (cfg.wand_block_size == 0) cfg.wand_block_size = 128;

    GV_LearnedSparseIndex *idx = (GV_LearnedSparseIndex *)calloc(
        1, sizeof(GV_LearnedSparseIndex));
    if (!idx) return NULL;

    idx->config = cfg;

    /* Allocate posting lists array (one per vocab token). */
    idx->posting_lists = (GV_LSPostingList *)calloc(
        cfg.vocab_size, sizeof(GV_LSPostingList));
    if (!idx->posting_lists) {
        free(idx);
        return NULL;
    }

    /* Allocate document metadata array. */
    idx->doc_capacity = GV_LS_INITIAL_DOC_CAPACITY;
    idx->docs = (GV_LSDocMeta *)calloc(idx->doc_capacity, sizeof(GV_LSDocMeta));
    if (!idx->docs) {
        free(idx->posting_lists);
        free(idx);
        return NULL;
    }

    idx->doc_count       = 0;
    idx->active_docs     = 0;
    idx->total_postings  = 0;
    idx->total_entry_count = 0;

    if (pthread_rwlock_init(&idx->rwlock, NULL) != 0) {
        free(idx->docs);
        free(idx->posting_lists);
        free(idx);
        return NULL;
    }

    return idx;
}

void gv_ls_destroy(GV_LearnedSparseIndex *idx) {
    if (!idx) return;

    /* Free all posting lists. */
    if (idx->posting_lists) {
        for (size_t i = 0; i < idx->config.vocab_size; i++) {
            gv_ls_posting_list_free(&idx->posting_lists[i]);
        }
        free(idx->posting_lists);
    }

    free(idx->docs);
    pthread_rwlock_destroy(&idx->rwlock);
    free(idx);
}

/* ============================================================================
 * Indexing Operations
 * ============================================================================ */

int gv_ls_insert(GV_LearnedSparseIndex *idx, const GV_SparseEntry *entries,
                 size_t count) {
    if (!idx || !entries || count == 0) return -1;
    if (count > idx->config.max_nonzeros) return -1;

    pthread_rwlock_wrlock(&idx->rwlock);

    /* Grow document array if needed. */
    if (idx->doc_count >= idx->doc_capacity) {
        size_t new_cap = idx->doc_capacity * 2;
        GV_LSDocMeta *new_docs = (GV_LSDocMeta *)realloc(
            idx->docs, new_cap * sizeof(GV_LSDocMeta));
        if (!new_docs) {
            pthread_rwlock_unlock(&idx->rwlock);
            return -1;
        }
        memset(new_docs + idx->doc_capacity, 0,
               (new_cap - idx->doc_capacity) * sizeof(GV_LSDocMeta));
        idx->docs = new_docs;
        idx->doc_capacity = new_cap;
    }

    size_t doc_id = idx->doc_count;
    size_t block_size = idx->config.use_wand ? idx->config.wand_block_size : 0;

    /* Append to posting lists for each non-zero entry. */
    for (size_t i = 0; i < count; i++) {
        uint32_t tid = entries[i].token_id;
        if (tid >= idx->config.vocab_size) continue;
        if (entries[i].weight <= 0.0f) continue;

        if (gv_ls_posting_append(&idx->posting_lists[tid], doc_id,
                                  entries[i].weight, block_size) != 0) {
            /* Partial insert: posting lists may be inconsistent,
             * but the doc_count hasn't been bumped yet, so the
             * partially-inserted postings point to a doc_id that
             * will never be returned in searches. */
            pthread_rwlock_unlock(&idx->rwlock);
            return -1;
        }
        idx->total_postings++;
    }

    /* Record document metadata. */
    idx->docs[doc_id].entry_count = count;
    idx->docs[doc_id].deleted     = 0;

    idx->doc_count++;
    idx->active_docs++;
    idx->total_entry_count += count;

    pthread_rwlock_unlock(&idx->rwlock);
    return (int)doc_id;
}

int gv_ls_delete(GV_LearnedSparseIndex *idx, size_t doc_id) {
    if (!idx) return -1;

    pthread_rwlock_wrlock(&idx->rwlock);

    if (doc_id >= idx->doc_count || idx->docs[doc_id].deleted) {
        pthread_rwlock_unlock(&idx->rwlock);
        return -1;
    }

    idx->docs[doc_id].deleted = 1;
    idx->active_docs--;
    idx->total_entry_count -= idx->docs[doc_id].entry_count;

    pthread_rwlock_unlock(&idx->rwlock);
    return 0;
}

/* ============================================================================
 * Search Operations
 * ============================================================================ */

int gv_ls_search(const GV_LearnedSparseIndex *idx, const GV_SparseEntry *query,
                 size_t query_count, size_t k, GV_LearnedSparseResult *results) {
    if (!idx || !query || query_count == 0 || k == 0 || !results) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&idx->rwlock);

    if (idx->active_docs == 0) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);
        return 0;
    }

    int n;
    if (idx->config.use_wand) {
        n = gv_ls_search_wand(idx, query, query_count, 0.0f, k, results);
    } else {
        n = gv_ls_search_accumulate(idx, query, query_count, 0.0f, k, results);
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);
    return n;
}

int gv_ls_search_with_threshold(const GV_LearnedSparseIndex *idx,
                                const GV_SparseEntry *query, size_t query_count,
                                float min_score, size_t k,
                                GV_LearnedSparseResult *results) {
    if (!idx || !query || query_count == 0 || k == 0 || !results) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&idx->rwlock);

    if (idx->active_docs == 0) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);
        return 0;
    }

    int n;
    if (idx->config.use_wand) {
        n = gv_ls_search_wand(idx, query, query_count, min_score, k, results);
    } else {
        n = gv_ls_search_accumulate(idx, query, query_count, min_score, k,
                                     results);
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);
    return n;
}

/* ============================================================================
 * Index Information
 * ============================================================================ */

int gv_ls_get_stats(const GV_LearnedSparseIndex *idx,
                    GV_LearnedSparseStats *stats) {
    if (!idx || !stats) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&idx->rwlock);

    stats->doc_count      = idx->active_docs;
    stats->total_postings = idx->total_postings;
    stats->avg_doc_length = idx->active_docs > 0
        ? (double)idx->total_entry_count / (double)idx->active_docs
        : 0.0;

    /* Count distinct tokens that have at least one posting. */
    size_t vocab_used = 0;
    for (size_t i = 0; i < idx->config.vocab_size; i++) {
        if (idx->posting_lists[i].count > 0) {
            vocab_used++;
        }
    }
    stats->vocab_used = vocab_used;

    pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);
    return 0;
}

size_t gv_ls_count(const GV_LearnedSparseIndex *idx) {
    if (!idx) return 0;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&idx->rwlock);
    size_t count = idx->active_docs;
    pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);

    return count;
}

/* ============================================================================
 * Persistence: Save
 * ============================================================================ */

int gv_ls_save(const GV_LearnedSparseIndex *idx, const char *path) {
    if (!idx || !path) return -1;

    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&idx->rwlock);

    /* Magic + version. */
    if (fwrite(GV_LS_MAGIC, 1, GV_LS_MAGIC_LEN, fp) != GV_LS_MAGIC_LEN) goto fail;
    if (gv_ls_write_u32(fp, GV_LS_VERSION) != 0) goto fail;

    /* Configuration. */
    if (gv_ls_write_u64(fp, (uint64_t)idx->config.vocab_size) != 0) goto fail;
    if (gv_ls_write_u64(fp, (uint64_t)idx->config.max_nonzeros) != 0) goto fail;
    if (gv_ls_write_u32(fp, (uint32_t)idx->config.use_wand) != 0) goto fail;
    if (gv_ls_write_u64(fp, (uint64_t)idx->config.wand_block_size) != 0) goto fail;

    /* Document count (total, including deleted). */
    if (gv_ls_write_u64(fp, (uint64_t)idx->doc_count) != 0) goto fail;

    /* Document metadata. */
    for (size_t i = 0; i < idx->doc_count; i++) {
        if (gv_ls_write_u64(fp, (uint64_t)idx->docs[i].entry_count) != 0) goto fail;
        if (gv_ls_write_u32(fp, (uint32_t)idx->docs[i].deleted) != 0) goto fail;
    }

    /* Posting lists: write only non-empty ones. */
    /* First, count non-empty posting lists. */
    uint64_t non_empty_count = 0;
    for (size_t i = 0; i < idx->config.vocab_size; i++) {
        if (idx->posting_lists[i].count > 0) non_empty_count++;
    }
    if (gv_ls_write_u64(fp, non_empty_count) != 0) goto fail;

    for (size_t i = 0; i < idx->config.vocab_size; i++) {
        const GV_LSPostingList *pl = &idx->posting_lists[i];
        if (pl->count == 0) continue;

        /* Token ID. */
        if (gv_ls_write_u32(fp, (uint32_t)i) != 0) goto fail;
        /* Posting count. */
        if (gv_ls_write_u64(fp, (uint64_t)pl->count) != 0) goto fail;

        /* Postings. */
        for (size_t j = 0; j < pl->count; j++) {
            if (gv_ls_write_u64(fp, (uint64_t)pl->postings[j].doc_id) != 0) goto fail;
            if (gv_ls_write_float(fp, pl->postings[j].weight) != 0) goto fail;
        }
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);
    fclose(fp);
    return 0;

fail:
    pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);
    fclose(fp);
    return -1;
}

/* ============================================================================
 * Persistence: Load
 * ============================================================================ */

GV_LearnedSparseIndex *gv_ls_load(const char *path) {
    if (!path) return NULL;

    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;

    /* Verify magic. */
    char magic[GV_LS_MAGIC_LEN];
    if (fread(magic, 1, GV_LS_MAGIC_LEN, fp) != GV_LS_MAGIC_LEN ||
        memcmp(magic, GV_LS_MAGIC, GV_LS_MAGIC_LEN) != 0) {
        fclose(fp);
        return NULL;
    }

    /* Verify version. */
    uint32_t version = 0;
    if (gv_ls_read_u32(fp, &version) != 0 || version != GV_LS_VERSION) {
        fclose(fp);
        return NULL;
    }

    /* Read configuration. */
    uint64_t vocab_size = 0, max_nonzeros = 0, wand_block_size = 0;
    uint32_t use_wand = 0;
    if (gv_ls_read_u64(fp, &vocab_size) != 0)      { fclose(fp); return NULL; }
    if (gv_ls_read_u64(fp, &max_nonzeros) != 0)    { fclose(fp); return NULL; }
    if (gv_ls_read_u32(fp, &use_wand) != 0)        { fclose(fp); return NULL; }
    if (gv_ls_read_u64(fp, &wand_block_size) != 0) { fclose(fp); return NULL; }

    GV_LearnedSparseConfig cfg;
    cfg.vocab_size      = (size_t)vocab_size;
    cfg.max_nonzeros    = (size_t)max_nonzeros;
    cfg.use_wand        = (int)use_wand;
    cfg.wand_block_size = (size_t)wand_block_size;

    /* Read document count. */
    uint64_t doc_count_raw = 0;
    if (gv_ls_read_u64(fp, &doc_count_raw) != 0) { fclose(fp); return NULL; }

    /* Create the index. */
    GV_LearnedSparseIndex *idx = gv_ls_create(&cfg);
    if (!idx) { fclose(fp); return NULL; }

    /* Grow document array to fit loaded doc count. */
    size_t doc_count = (size_t)doc_count_raw;
    if (doc_count > 0) {
        while (idx->doc_capacity < doc_count) {
            size_t new_cap = idx->doc_capacity * 2;
            GV_LSDocMeta *new_docs = (GV_LSDocMeta *)realloc(
                idx->docs, new_cap * sizeof(GV_LSDocMeta));
            if (!new_docs) {
                gv_ls_destroy(idx);
                fclose(fp);
                return NULL;
            }
            memset(new_docs + idx->doc_capacity, 0,
                   (new_cap - idx->doc_capacity) * sizeof(GV_LSDocMeta));
            idx->docs = new_docs;
            idx->doc_capacity = new_cap;
        }
    }

    /* Read document metadata. */
    idx->doc_count         = doc_count;
    idx->active_docs       = 0;
    idx->total_entry_count = 0;
    for (size_t i = 0; i < doc_count; i++) {
        uint64_t ec = 0;
        uint32_t del = 0;
        if (gv_ls_read_u64(fp, &ec) != 0 || gv_ls_read_u32(fp, &del) != 0) {
            gv_ls_destroy(idx);
            fclose(fp);
            return NULL;
        }
        idx->docs[i].entry_count = (size_t)ec;
        idx->docs[i].deleted     = (int)del;
        if (!del) {
            idx->active_docs++;
            idx->total_entry_count += (size_t)ec;
        }
    }

    /* Read posting lists. */
    uint64_t non_empty_count = 0;
    if (gv_ls_read_u64(fp, &non_empty_count) != 0) {
        gv_ls_destroy(idx);
        fclose(fp);
        return NULL;
    }

    size_t block_size = cfg.use_wand ? cfg.wand_block_size : 0;
    idx->total_postings = 0;

    for (uint64_t t = 0; t < non_empty_count; t++) {
        uint32_t tid = 0;
        uint64_t pcount = 0;
        if (gv_ls_read_u32(fp, &tid) != 0 || gv_ls_read_u64(fp, &pcount) != 0) {
            gv_ls_destroy(idx);
            fclose(fp);
            return NULL;
        }

        if ((size_t)tid >= cfg.vocab_size) {
            gv_ls_destroy(idx);
            fclose(fp);
            return NULL;
        }

        for (uint64_t p = 0; p < pcount; p++) {
            uint64_t did = 0;
            float    w   = 0.0f;
            if (gv_ls_read_u64(fp, &did) != 0 || gv_ls_read_float(fp, &w) != 0) {
                gv_ls_destroy(idx);
                fclose(fp);
                return NULL;
            }

            if (gv_ls_posting_append(&idx->posting_lists[tid], (size_t)did, w,
                                      block_size) != 0) {
                gv_ls_destroy(idx);
                fclose(fp);
                return NULL;
            }
            idx->total_postings++;
        }
    }

    fclose(fp);
    return idx;
}
