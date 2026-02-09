/**
 * @file gv_recommend.c
 * @brief Recommendation API: find similar vectors based on positive/negative examples.
 */

#include "gigavector/gv_recommend.h"
#include "gigavector/gv_database.h"
#include "gigavector/gv_types.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ============================================================================
 * Configuration Defaults
 * ============================================================================ */

static const GV_RecommendConfig DEFAULT_CONFIG = {
    .positive_weight = 1.0f,
    .negative_weight = 0.5f,
    .distance_type   = 1,      /* GV_DISTANCE_COSINE */
    .oversample      = 2,
    .exclude_input   = 1
};

void gv_recommend_config_init(GV_RecommendConfig *config) {
    if (!config) return;
    *config = DEFAULT_CONFIG;
}

/* ============================================================================
 * Internal Helpers
 * ============================================================================ */

/**
 * @brief L2-normalize a vector in place.
 *
 * If the norm is zero (or very close), the vector is left unchanged.
 */
static void l2_normalize(float *vec, size_t dim) {
    double norm_sq = 0.0;
    for (size_t i = 0; i < dim; i++) {
        norm_sq += (double)vec[i] * (double)vec[i];
    }
    if (norm_sq < 1e-30) return;
    double inv_norm = 1.0 / sqrt(norm_sq);
    for (size_t i = 0; i < dim; i++) {
        vec[i] = (float)((double)vec[i] * inv_norm);
    }
}

/**
 * @brief Recover the SoA storage index from a GV_SearchResult.
 *
 * GV_SearchResult.vector->data points directly into SoA contiguous storage
 * at offset (index * dimension). We recover the index by computing the
 * pointer difference from the base data pointer obtained via
 * gv_database_get_vector(db, 0).
 *
 * Returns (size_t)-1 if the index cannot be determined.
 */
static size_t result_to_index(const GV_Database *db, const GV_SearchResult *sr) {
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

/**
 * @brief Convert GV_SearchResult array to GV_RecommendResult array,
 *        optionally excluding a set of input IDs.
 *
 * @param db           Database handle.
 * @param search_res   Raw search results from gv_db_search.
 * @param search_count Number of raw results.
 * @param exclude_ids  Array of IDs to exclude (may be NULL).
 * @param exclude_count Number of IDs to exclude.
 * @param k            Maximum number of results to output.
 * @param out          Output array with at least k elements.
 * @return Number of results written to out.
 */
static int convert_results(const GV_Database *db,
                           const GV_SearchResult *search_res, int search_count,
                           const size_t *exclude_ids, size_t exclude_count,
                           size_t k, GV_RecommendResult *out) {
    size_t written = 0;
    for (int i = 0; i < search_count && written < k; i++) {
        size_t idx = result_to_index(db, &search_res[i]);
        if (idx == (size_t)-1) continue;

        /* Check exclusion list */
        int excluded = 0;
        if (exclude_ids && exclude_count > 0) {
            for (size_t e = 0; e < exclude_count; e++) {
                if (exclude_ids[e] == idx) {
                    excluded = 1;
                    break;
                }
            }
        }
        if (excluded) continue;

        out[written].index = idx;
        out[written].score = search_res[i].distance;
        written++;
    }
    return (int)written;
}

/* ============================================================================
 * gv_recommend_by_vector
 * ============================================================================ */

int gv_recommend_by_vector(const GV_Database *db,
                            const float *positive_vectors, size_t positive_count,
                            const float *negative_vectors, size_t negative_count,
                            size_t dimension, size_t k, const GV_RecommendConfig *config,
                            GV_RecommendResult *results) {
    if (!db || !results || k == 0) return -1;
    if (!positive_vectors || positive_count == 0) return -1;
    if (dimension == 0) return -1;
    if (dimension != gv_database_dimension(db)) return -1;

    GV_RecommendConfig cfg;
    if (config) {
        cfg = *config;
    } else {
        cfg = DEFAULT_CONFIG;
    }

    /* Allocate query vector: positive centroid minus weighted negative centroid */
    float *query = (float *)calloc(dimension, sizeof(float));
    if (!query) return -1;

    /* Compute positive centroid (element-wise average, weighted) */
    for (size_t p = 0; p < positive_count; p++) {
        const float *vec = positive_vectors + p * dimension;
        for (size_t d = 0; d < dimension; d++) {
            query[d] += vec[d];
        }
    }
    float pos_scale = cfg.positive_weight / (float)positive_count;
    for (size_t d = 0; d < dimension; d++) {
        query[d] *= pos_scale;
    }

    /* Subtract weighted negative centroid */
    if (negative_vectors && negative_count > 0) {
        float *neg_centroid = (float *)calloc(dimension, sizeof(float));
        if (!neg_centroid) {
            free(query);
            return -1;
        }
        for (size_t n = 0; n < negative_count; n++) {
            const float *vec = negative_vectors + n * dimension;
            for (size_t d = 0; d < dimension; d++) {
                neg_centroid[d] += vec[d];
            }
        }
        float neg_scale = cfg.negative_weight / (float)negative_count;
        for (size_t d = 0; d < dimension; d++) {
            query[d] -= neg_centroid[d] * neg_scale;
        }
        free(neg_centroid);
    }

    /* L2-normalize the query vector */
    l2_normalize(query, dimension);

    /* Determine search count: oversample to allow for filtering */
    size_t search_k = k * cfg.oversample;
    if (search_k < k) search_k = k; /* overflow guard */

    GV_SearchResult *search_res = (GV_SearchResult *)calloc(search_k, sizeof(GV_SearchResult));
    if (!search_res) {
        free(query);
        return -1;
    }

    int found = gv_db_search(db, query, search_k, search_res, (GV_DistanceType)cfg.distance_type);
    free(query);

    if (found < 0) {
        free(search_res);
        return -1;
    }

    /* Convert to recommend results (no exclusion for raw-vector path) */
    int result_count = convert_results(db, search_res, found,
                                       NULL, 0,
                                       k, results);
    free(search_res);
    return result_count;
}

/* ============================================================================
 * gv_recommend_by_id
 * ============================================================================ */

int gv_recommend_by_id(const GV_Database *db,
                        const size_t *positive_ids, size_t positive_count,
                        const size_t *negative_ids, size_t negative_count,
                        size_t k, const GV_RecommendConfig *config,
                        GV_RecommendResult *results) {
    if (!db || !results || k == 0) return -1;
    if (!positive_ids || positive_count == 0) return -1;

    GV_RecommendConfig cfg;
    if (config) {
        cfg = *config;
    } else {
        cfg = DEFAULT_CONFIG;
    }

    size_t dim = gv_database_dimension(db);
    if (dim == 0) return -1;

    /* Fetch positive vectors */
    float *pos_buf = (float *)malloc(positive_count * dim * sizeof(float));
    if (!pos_buf) return -1;

    for (size_t i = 0; i < positive_count; i++) {
        const float *vec = gv_database_get_vector(db, positive_ids[i]);
        if (!vec) {
            free(pos_buf);
            return -1;
        }
        memcpy(pos_buf + i * dim, vec, dim * sizeof(float));
    }

    /* Fetch negative vectors */
    float *neg_buf = NULL;
    if (negative_ids && negative_count > 0) {
        neg_buf = (float *)malloc(negative_count * dim * sizeof(float));
        if (!neg_buf) {
            free(pos_buf);
            return -1;
        }
        for (size_t i = 0; i < negative_count; i++) {
            const float *vec = gv_database_get_vector(db, negative_ids[i]);
            if (!vec) {
                free(pos_buf);
                free(neg_buf);
                return -1;
            }
            memcpy(neg_buf + i * dim, vec, dim * sizeof(float));
        }
    }

    /* Build exclusion set from input IDs if configured */
    size_t *exclude_ids = NULL;
    size_t exclude_count = 0;
    if (cfg.exclude_input) {
        exclude_count = positive_count + negative_count;
        exclude_ids = (size_t *)malloc(exclude_count * sizeof(size_t));
        if (!exclude_ids) {
            free(pos_buf);
            free(neg_buf);
            return -1;
        }
        memcpy(exclude_ids, positive_ids, positive_count * sizeof(size_t));
        if (negative_ids && negative_count > 0) {
            memcpy(exclude_ids + positive_count, negative_ids, negative_count * sizeof(size_t));
        }
    }

    /* Compute the recommendation query vector */
    float *query = (float *)calloc(dim, sizeof(float));
    if (!query) {
        free(pos_buf);
        free(neg_buf);
        free(exclude_ids);
        return -1;
    }

    /* Positive centroid */
    for (size_t p = 0; p < positive_count; p++) {
        const float *vec = pos_buf + p * dim;
        for (size_t d = 0; d < dim; d++) {
            query[d] += vec[d];
        }
    }
    float pos_scale = cfg.positive_weight / (float)positive_count;
    for (size_t d = 0; d < dim; d++) {
        query[d] *= pos_scale;
    }

    /* Subtract negative centroid */
    if (neg_buf && negative_count > 0) {
        float *neg_centroid = (float *)calloc(dim, sizeof(float));
        if (!neg_centroid) {
            free(pos_buf);
            free(neg_buf);
            free(exclude_ids);
            free(query);
            return -1;
        }
        for (size_t n = 0; n < negative_count; n++) {
            const float *vec = neg_buf + n * dim;
            for (size_t d = 0; d < dim; d++) {
                neg_centroid[d] += vec[d];
            }
        }
        float neg_scale = cfg.negative_weight / (float)negative_count;
        for (size_t d = 0; d < dim; d++) {
            query[d] -= neg_centroid[d] * neg_scale;
        }
        free(neg_centroid);
    }

    free(pos_buf);
    free(neg_buf);

    /* L2-normalize */
    l2_normalize(query, dim);

    /* Oversample to compensate for exclusion filtering */
    size_t search_k = k;
    if (cfg.exclude_input && exclude_count > 0) {
        search_k = (k + exclude_count) * cfg.oversample;
    } else {
        search_k = k * cfg.oversample;
    }
    if (search_k < k) search_k = k; /* overflow guard */

    GV_SearchResult *search_res = (GV_SearchResult *)calloc(search_k, sizeof(GV_SearchResult));
    if (!search_res) {
        free(query);
        free(exclude_ids);
        return -1;
    }

    int found = gv_db_search(db, query, search_k, search_res, (GV_DistanceType)cfg.distance_type);
    free(query);

    if (found < 0) {
        free(search_res);
        free(exclude_ids);
        return -1;
    }

    int result_count = convert_results(db, search_res, found,
                                       exclude_ids, exclude_count,
                                       k, results);
    free(search_res);
    free(exclude_ids);
    return result_count;
}

/* ============================================================================
 * gv_recommend_discover
 * ============================================================================ */

int gv_recommend_discover(const GV_Database *db,
                           const float *target, const float *context,
                           size_t dimension, size_t k, const GV_RecommendConfig *config,
                           GV_RecommendResult *results) {
    if (!db || !results || k == 0) return -1;
    if (!target || !context) return -1;
    if (dimension == 0) return -1;
    if (dimension != gv_database_dimension(db)) return -1;

    GV_RecommendConfig cfg;
    if (config) {
        cfg = *config;
    } else {
        cfg = DEFAULT_CONFIG;
    }

    /* Compute direction vector: target - context */
    float *direction = (float *)malloc(dimension * sizeof(float));
    if (!direction) return -1;

    for (size_t d = 0; d < dimension; d++) {
        direction[d] = target[d] - context[d];
    }

    /* L2-normalize the direction */
    l2_normalize(direction, dimension);

    /* Search using the direction as query */
    size_t search_k = k * cfg.oversample;
    if (search_k < k) search_k = k; /* overflow guard */

    GV_SearchResult *search_res = (GV_SearchResult *)calloc(search_k, sizeof(GV_SearchResult));
    if (!search_res) {
        free(direction);
        return -1;
    }

    int found = gv_db_search(db, direction, search_k, search_res, (GV_DistanceType)cfg.distance_type);
    free(direction);

    if (found < 0) {
        free(search_res);
        return -1;
    }

    /* Convert results (no exclusion for discover) */
    int result_count = convert_results(db, search_res, found,
                                       NULL, 0,
                                       k, results);
    free(search_res);
    return result_count;
}
