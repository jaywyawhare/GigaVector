/**
 * @file gv_mmr.c
 * @brief Maximal Marginal Relevance (MMR) reranking implementation.
 *
 * MMR iteratively selects results that balance relevance to the query against
 * diversity (dissimilarity to already-selected items):
 *
 *   score = lambda * relevance(d, q) - (1 - lambda) * max_similarity(d, S)
 *
 * where S is the set of documents already selected.
 */

#include "gigavector/gv_mmr.h"
#include "gigavector/gv_distance.h"
#include "gigavector/gv_database.h"
#include "gigavector/gv_vector.h"
#include "gigavector/gv_types.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/*  Configuration Defaults  */

static const GV_MMRConfig DEFAULT_CONFIG = {
    .lambda        = 0.7f,
    .distance_type = 1       /* GV_DISTANCE_COSINE */
};

void gv_mmr_config_init(GV_MMRConfig *config) {
    if (!config) return;
    *config = DEFAULT_CONFIG;
}

/*  Internal Helpers  */

/**
 * @brief Compute distance between two raw float vectors using gv_distance().
 *
 * Wraps the raw float pointers in temporary stack-allocated GV_Vector structs
 * so they can be passed to the existing gv_distance() API.
 */
static float compute_distance(const float *a, const float *b,
                              size_t dimension, GV_DistanceType type) {
    GV_Vector va = { .dimension = dimension, .data = (float *)a, .metadata = NULL };
    GV_Vector vb = { .dimension = dimension, .data = (float *)b, .metadata = NULL };
    return gv_distance(&va, &vb, type);
}

/**
 * @brief Convert a raw distance value to a similarity score in [0, 1].
 *
 * The conversion depends on the distance metric:
 *   - Cosine: cosine similarity is in [-1, 1]; map to [0, 1] via (sim + 1) / 2.
 *   - Dot product: gv_distance returns -dot; similarity = -distance, clamped.
 *   - Euclidean/Manhattan/Hamming: similarity = 1 / (1 + distance).
 */
static float distance_to_similarity(float dist, GV_DistanceType type) {
    switch (type) {
        case GV_DISTANCE_COSINE:
            /* gv_distance_cosine returns cosine similarity in [-1, 1] */
            return (dist + 1.0f) / 2.0f;

        case GV_DISTANCE_DOT_PRODUCT:
            /* gv_distance_dot_product returns -dot; higher dot = more similar */
            /* Use sigmoid-style mapping: 1/(1+exp(dist)) where dist = -dot */
            return 1.0f / (1.0f + expf(dist));

        case GV_DISTANCE_EUCLIDEAN:
        case GV_DISTANCE_MANHATTAN:
        case GV_DISTANCE_HAMMING:
        default:
            /* Lower distance = higher similarity */
            if (dist < 0.0f) dist = 0.0f;
            return 1.0f / (1.0f + dist);
    }
}

/**
 * @brief Normalise an array of similarity scores to [0, 1] using min-max scaling.
 *
 * If all values are identical, they are set to 1.0.
 */
static void normalize_scores(float *scores, size_t count) {
    if (count == 0) return;

    float min_val = scores[0];
    float max_val = scores[0];
    for (size_t i = 1; i < count; i++) {
        if (scores[i] < min_val) min_val = scores[i];
        if (scores[i] > max_val) max_val = scores[i];
    }

    float range = max_val - min_val;
    if (range < 1e-12f) {
        /* All scores are effectively equal */
        for (size_t i = 0; i < count; i++) {
            scores[i] = 1.0f;
        }
        return;
    }

    for (size_t i = 0; i < count; i++) {
        scores[i] = (scores[i] - min_val) / range;
    }
}

/**
 * @brief Recover the SoA storage index from a GV_SearchResult.
 *
 * GV_SearchResult.vector->data points directly into SoA contiguous storage
 * at offset (index * dimension).  We recover the index by computing the
 * pointer difference from the base data pointer.
 *
 * Returns (size_t)-1 if the index cannot be determined.
 */
static size_t result_to_soa_index(const GV_Database *db, const GV_SearchResult *sr) {
    if (!sr || !sr->vector || !sr->vector->data) return (size_t)-1;

    size_t dim = gv_database_dimension(db);
    if (dim == 0) return (size_t)-1;

    const float *base = gv_database_get_vector(db, 0);
    if (!base) return (size_t)-1;

    ptrdiff_t diff = sr->vector->data - base;
    if (diff < 0) return (size_t)-1;

    size_t idx = (size_t)diff / dim;
    if (idx >= gv_database_count(db)) return (size_t)-1;

    return idx;
}

/*  gv_mmr_rerank  */

int gv_mmr_rerank(const float *query, size_t dimension,
                  const float *candidates, const size_t *candidate_indices,
                  const float *candidate_distances, size_t candidate_count,
                  size_t k, const GV_MMRConfig *config,
                  GV_MMRResult *results) {
    /* ------ Argument validation ------ */
    if (!query || !candidates || !candidate_indices || !candidate_distances || !results) {
        return -1;
    }
    if (dimension == 0 || candidate_count == 0 || k == 0) {
        return -1;
    }

    GV_MMRConfig cfg;
    if (config) {
        cfg = *config;
    } else {
        cfg = DEFAULT_CONFIG;
    }

    /* Clamp lambda to [0, 1] */
    if (cfg.lambda < 0.0f) cfg.lambda = 0.0f;
    if (cfg.lambda > 1.0f) cfg.lambda = 1.0f;

    GV_DistanceType dist_type = (GV_DistanceType)cfg.distance_type;

    /* Limit k to candidate_count */
    if (k > candidate_count) k = candidate_count;

    /* ------ Compute and normalise relevance scores ------ */
    float *relevance = (float *)malloc(candidate_count * sizeof(float));
    if (!relevance) return -1;

    for (size_t i = 0; i < candidate_count; i++) {
        relevance[i] = distance_to_similarity(candidate_distances[i], dist_type);
    }
    normalize_scores(relevance, candidate_count);

    /* ------ Greedy iterative MMR selection ------ */

    /* Track which candidates have been selected */
    int *selected = (int *)calloc(candidate_count, sizeof(int));
    if (!selected) {
        free(relevance);
        return -1;
    }

    size_t selected_count = 0;

    for (size_t step = 0; step < k; step++) {
        float best_mmr = -FLT_MAX;
        size_t best_idx = 0;
        int found = 0;

        for (size_t i = 0; i < candidate_count; i++) {
            if (selected[i]) continue;

            /* Relevance component */
            float rel = relevance[i];

            /* Diversity component: max similarity to any already-selected item */
            float max_sim = 0.0f;
            for (size_t j = 0; j < step; j++) {
                size_t sel_idx = results[j].index;

                /* Find the candidate buffer position for this selected item */
                size_t sel_pos = 0;
                for (size_t c = 0; c < candidate_count; c++) {
                    if (candidate_indices[c] == sel_idx) {
                        sel_pos = c;
                        break;
                    }
                }

                const float *vec_i = candidates + i * dimension;
                const float *vec_s = candidates + sel_pos * dimension;

                float dist = compute_distance(vec_i, vec_s, dimension, dist_type);
                float sim = distance_to_similarity(dist, dist_type);
                if (sim > max_sim) max_sim = sim;
            }

            /* MMR score */
            float mmr = cfg.lambda * rel - (1.0f - cfg.lambda) * max_sim;

            if (!found || mmr > best_mmr) {
                best_mmr = mmr;
                best_idx = i;
                found = 1;
            }
        }

        if (!found) break;

        /* Record the selected candidate */
        selected[best_idx] = 1;
        results[selected_count].index     = candidate_indices[best_idx];
        results[selected_count].score     = best_mmr;
        results[selected_count].relevance = relevance[best_idx];

        /* Compute diversity for the result we just picked */
        float max_sim_final = 0.0f;
        for (size_t j = 0; j < selected_count; j++) {
            size_t sel_idx = results[j].index;
            size_t sel_pos = 0;
            for (size_t c = 0; c < candidate_count; c++) {
                if (candidate_indices[c] == sel_idx) {
                    sel_pos = c;
                    break;
                }
            }
            const float *vec_i = candidates + best_idx * dimension;
            const float *vec_s = candidates + sel_pos * dimension;
            float dist = compute_distance(vec_i, vec_s, dimension, dist_type);
            float sim = distance_to_similarity(dist, dist_type);
            if (sim > max_sim_final) max_sim_final = sim;
        }
        results[selected_count].diversity = 1.0f - max_sim_final;

        selected_count++;
    }

    free(relevance);
    free(selected);

    return (int)selected_count;
}

/*  gv_mmr_search  */

int gv_mmr_search(const void *db_ptr, const float *query, size_t dimension,
                  size_t k, size_t oversample, const GV_MMRConfig *config,
                  GV_MMRResult *results) {
    /* ------ Argument validation ------ */
    const GV_Database *db = (const GV_Database *)db_ptr;
    if (!db || !query || !results) return -1;
    if (dimension == 0 || k == 0) return -1;
    if (dimension != gv_database_dimension(db)) return -1;

    GV_MMRConfig cfg;
    if (config) {
        cfg = *config;
    } else {
        cfg = DEFAULT_CONFIG;
    }

    /* Ensure a sane oversample factor */
    if (oversample < 1) oversample = 1;

    /* ------ Fetch oversampled candidates via gv_db_search ------ */
    size_t fetch_k = k * oversample;
    if (fetch_k < k) fetch_k = k; /* overflow guard */

    GV_SearchResult *search_res = (GV_SearchResult *)calloc(fetch_k, sizeof(GV_SearchResult));
    if (!search_res) return -1;

    int found = gv_db_search(db, query, fetch_k, search_res, (GV_DistanceType)cfg.distance_type);
    if (found <= 0) {
        free(search_res);
        return (found == 0) ? 0 : -1;
    }

    size_t candidate_count = (size_t)found;

    /* ------ Extract candidate data for MMR reranking ------ */
    float *cand_vectors   = (float *)malloc(candidate_count * dimension * sizeof(float));
    size_t *cand_indices  = (size_t *)malloc(candidate_count * sizeof(size_t));
    float *cand_distances = (float *)malloc(candidate_count * sizeof(float));

    if (!cand_vectors || !cand_indices || !cand_distances) {
        free(cand_vectors);
        free(cand_indices);
        free(cand_distances);
        free(search_res);
        return -1;
    }

    size_t valid = 0;
    for (size_t i = 0; i < candidate_count; i++) {
        const GV_SearchResult *sr = &search_res[i];
        if (!sr->vector || !sr->vector->data) continue;

        size_t soa_idx = result_to_soa_index(db, sr);
        if (soa_idx == (size_t)-1) continue;

        memcpy(cand_vectors + valid * dimension, sr->vector->data, dimension * sizeof(float));
        cand_indices[valid]  = soa_idx;
        cand_distances[valid] = sr->distance;
        valid++;
    }

    free(search_res);

    if (valid == 0) {
        free(cand_vectors);
        free(cand_indices);
        free(cand_distances);
        return 0;
    }

    /* ------ Apply MMR reranking ------ */
    int result_count = gv_mmr_rerank(query, dimension,
                                     cand_vectors, cand_indices,
                                     cand_distances, valid,
                                     k, &cfg, results);

    free(cand_vectors);
    free(cand_indices);
    free(cand_distances);

    return result_count;
}
