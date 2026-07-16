/**
 * @file quant_rerank.c
 * @brief Quantization-aware two-stage reranking pipeline implementation.
 *
 * Stage 1: coarse ANN retrieval via db_search() with oversample_factor*k candidates.
 * Stage 2: asymmetric quantized distance refinement — raw float query vs stored
 *          quantized codes via quant_distance(); sort by rerank_score ascending.
 *
 * quant_rerank_search() mirrors the pattern of rank_search() in ranking.c:
 * it calls db_search() with oversample_factor*k candidates, then for each
 * candidate looks up the pre-encoded code at codes[id * code_stride] and calls
 * quant_distance() to compute a refined rerank_score.
 *
 * gv_db_search_with_rerank() is a simplified variant that reranks using exact
 * float32 distances without requiring a pre-trained codebook.
 */

#include "search/quant_rerank.h"
#include "storage/database.h"
#include "storage/soa_storage.h"
#include "schema/vector.h"
#include "search/distance.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/** Default oversample factor when caller passes 0. */
#define GV_QUANT_RERANK_DEFAULT_OVERSAMPLE 4

/* ------------------------------------------------------------------ */
/* Internal helpers                                                     */
/* ------------------------------------------------------------------ */

/**
 * @brief Comparison function for ascending rerank_score sort.
 *
 * Lower rerank_score = more similar, so sort ascending.
 */
static int compare_rerank_asc(const void *a, const void *b) {
    const GV_QuantRerankResult *ra = (const GV_QuantRerankResult *)a;
    const GV_QuantRerankResult *rb = (const GV_QuantRerankResult *)b;
    if (ra->rerank_score < rb->rerank_score) return -1;
    if (ra->rerank_score > rb->rerank_score) return  1;
    return 0;
}

/**
 * @brief Helper: compute squared Euclidean distance between two float vectors.
 *
 * Used by gv_db_search_with_rerank() for exact reranking without a codebook.
 */
static float l2_squared(const float *a, const float *b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

/* ------------------------------------------------------------------ */
/* Internal candidate struct for exact reranking                        */
/* ------------------------------------------------------------------ */

typedef struct {
    size_t index;
    float  approx_score;
    float  exact_score;
} ExactRerankCandidate;

static int compare_exact_asc(const void *a, const void *b) {
    const ExactRerankCandidate *ra = (const ExactRerankCandidate *)a;
    const ExactRerankCandidate *rb = (const ExactRerankCandidate *)b;
    if (ra->exact_score < rb->exact_score) return -1;
    if (ra->exact_score > rb->exact_score) return  1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

void quant_rerank_config_init(GV_QuantRerankConfig *cfg) {
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->oversample_factor = GV_QUANT_RERANK_DEFAULT_OVERSAMPLE;
    cfg->distance_type     = GV_DISTANCE_EUCLIDEAN; /* 0 */
}

int quant_rerank_search(GV_Database *db, const float *query, size_t k,
                        const GV_QuantRerankConfig *cfg,
                        GV_QuantRerankResult *out) {
    if (!db || !query || k == 0 || !cfg || !out) return -1;
    if (!cfg->codebook || !cfg->codes || cfg->code_stride == 0) return -1;

    /* Determine how many candidates to retrieve in stage 1. */
    size_t oversample = cfg->oversample_factor > 0
                            ? cfg->oversample_factor
                            : GV_QUANT_RERANK_DEFAULT_OVERSAMPLE;
    size_t fetch_k = oversample * k;
    if (fetch_k < k) fetch_k = k; /* overflow guard */

    size_t dim = database_dimension(db);
    if (dim == 0) return -1;

    /* Stage 1: coarse ANN retrieval. */
    GV_SearchResult *search_res = calloc(fetch_k, sizeof(GV_SearchResult));
    if (!search_res) return -1;

    int found = db_search(db, query, fetch_k, search_res,
                          (GV_DistanceType)cfg->distance_type);
    if (found <= 0) {
        free(search_res);
        return (found == 0) ? 0 : -1;
    }

    /* Stage 2: rerank with asymmetric quantized distance. */
    GV_QuantRerankResult *candidates = malloc((size_t)found * sizeof(GV_QuantRerankResult));
    if (!candidates) {
        free(search_res);
        return -1;
    }

    size_t valid = 0;
    for (int i = 0; i < found; i++) {
        size_t vid = search_res[i].id;
        /* Skip candidates whose SoA id is beyond the encoded codes array
         * (e.g. inserted after encoding) to avoid an OOB read. */
        if (vid >= cfg->codes_count) continue;
        const uint8_t *code = cfg->codes + vid * cfg->code_stride;
        float refined = quant_distance(cfg->codebook, query, dim, code);
        if (refined < 0.0f) {
            /* quant_distance reported an error; skip this candidate. */
            continue;
        }
        candidates[valid].index        = vid;
        candidates[valid].approx_score = search_res[i].distance;
        candidates[valid].rerank_score = refined;
        valid++;
    }

    /* Free each result's owned vector (data+metadata), then the array. */
    gv_search_results_free(search_res, (size_t)found);
    free(search_res);

    if (valid == 0) {
        free(candidates);
        return 0;
    }

    /* Sort ascending by rerank_score (lower = more similar). */
    qsort(candidates, valid, sizeof(GV_QuantRerankResult), compare_rerank_asc);

    /* Copy top-k into the output array. */
    size_t result_count = valid < k ? valid : k;
    memcpy(out, candidates, result_count * sizeof(GV_QuantRerankResult));

    free(candidates);
    return (int)result_count;
}

int gv_db_search_with_rerank(GV_Database *db, const float *query, size_t k,
                              size_t oversample, GV_SearchResult *out) {
    if (!db || !query || k == 0 || !out) return -1;

    if (oversample == 0) oversample = GV_QUANT_RERANK_DEFAULT_OVERSAMPLE;
    size_t fetch_k = oversample * k;
    if (fetch_k < k) fetch_k = k;

    size_t dim = database_dimension(db);
    if (dim == 0) return -1;

    /* Stage 1: coarse ANN retrieval. */
    GV_SearchResult *search_res = calloc(fetch_k, sizeof(GV_SearchResult));
    if (!search_res) return -1;

    int found = db_search(db, query, fetch_k, search_res, GV_DISTANCE_EUCLIDEAN);
    if (found <= 0) {
        free(search_res);
        return (found == 0) ? 0 : -1;
    }

    /* Stage 2: rerank with exact float32 distances. */
    ExactRerankCandidate *candidates = malloc((size_t)found * sizeof(ExactRerankCandidate));
    if (!candidates) {
        free(search_res);
        return -1;
    }

    size_t valid = 0;
    for (int i = 0; i < found; i++) {
        size_t vid = search_res[i].id;
        const float *vec = database_get_vector(db, vid);
        if (!vec) continue;

        float exact = l2_squared(query, vec, dim);
        candidates[valid].index        = vid;
        candidates[valid].approx_score = search_res[i].distance;
        candidates[valid].exact_score  = exact;
        valid++;
    }

    /* Free each result's owned vector (data+metadata), then the array. */
    gv_search_results_free(search_res, (size_t)found);
    free(search_res);

    if (valid == 0) {
        free(candidates);
        return 0;
    }

    /* Sort ascending by exact_score. */
    qsort(candidates, valid, sizeof(ExactRerankCandidate), compare_exact_asc);

    /* Fill output GV_SearchResult array. */
    size_t result_count = valid < k ? valid : k;
    for (size_t i = 0; i < result_count; i++) {
        memset(&out[i], 0, sizeof(GV_SearchResult));
        out[i].id       = candidates[i].index;
        out[i].distance = candidates[i].exact_score;
        /* Populate the vector pointer so callers can read the data. */
        const GV_SoAStorage *storage = db->soa_storage;
        if (storage) {
            GV_Vector view;
            memset(&view, 0, sizeof(view));
            if (soa_storage_get_vector_view(storage, candidates[i].index, &view) == 0) {
                out[i].vector = vector_create_from_data(view.dimension, view.data);
            }
        }
    }

    free(candidates);
    return (int)result_count;
}
