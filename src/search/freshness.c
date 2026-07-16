/**
 * @file freshness.c
 * @brief Vector Freshness Scoring implementation.
 *
 * Blends vector similarity scores with time-decay freshness signals so that
 * recently inserted (or updated) vectors rank higher when freshness_weight > 0.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "search/freshness.h"
#include "search/ranking.h"
#include "search/distance.h"
#include "storage/database.h"
#include "storage/soa_storage.h"
#include "core/sim_time.h"
#include "core/types.h"

/** Default oversample multiplier: retrieve this many candidates before re-ranking. */
#define GV_FRESHNESS_OVERSAMPLE_FACTOR 4

/* M_LN2 may not be available on all platforms. */
#ifndef M_LN2
#define M_LN2 0.6931471805599453
#endif

/* ---------------------------------------------------------------------------
 * Internal helpers
 * ------------------------------------------------------------------------ */

/**
 * Compute a normalised age in units of half_life_ms.
 * Returns 0.0 if the vector was inserted now; 1.0 after one half-life.
 */
static double compute_age_ratio(uint64_t insert_time_ms, uint64_t now_ms,
                                double half_life_ms)
{
    if (half_life_ms <= 0.0) return 0.0;
    double age_ms = (now_ms >= insert_time_ms)
                    ? (double)(now_ms - insert_time_ms)
                    : 0.0;
    return age_ms / half_life_ms;
}

/* ---------------------------------------------------------------------------
 * Public API
 * ------------------------------------------------------------------------ */

float freshness_score(uint64_t insert_time_ms, uint64_t now_ms,
                      double half_life_ms, GV_RankOp decay_type)
{
    double t = compute_age_ratio(insert_time_ms, now_ms, half_life_ms);

    switch (decay_type) {
        case GV_RANK_DECAY_GAUSS: {
            /* exp(-0.5 * t^2 * ln2) — equals 0.5 at t=1 */
            double v = exp(-0.5 * t * t * M_LN2);
            return (float)(v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v));
        }
        case GV_RANK_DECAY_LINEAR: {
            /* max(0, 1 - t * ln2) — equals 0 before t = 1/ln2 */
            double v = 1.0 - t * M_LN2;
            return (float)(v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v));
        }
        case GV_RANK_DECAY_EXP:
        default: {
            /* exp(-ln2 * t) — equals 0.5 at t=1 */
            double v = exp(-M_LN2 * t);
            return (float)(v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v));
        }
    }
}

float freshness_blend(float vector_score, float freshness,
                      float score_weight, float freshness_weight)
{
    return score_weight * vector_score + freshness_weight * freshness;
}

/* ---------------------------------------------------------------------------
 * Comparison function for qsort (sort by final_score descending)
 * ------------------------------------------------------------------------ */

static int cmp_freshness_result_desc(const void *a, const void *b)
{
    const GV_FreshnessResult *ra = (const GV_FreshnessResult *)a;
    const GV_FreshnessResult *rb = (const GV_FreshnessResult *)b;
    if (ra->final_score > rb->final_score) return -1;
    if (ra->final_score < rb->final_score) return  1;
    return 0;
}

/* ---------------------------------------------------------------------------
 * Core implementation shared by filtered and unfiltered variants
 * ------------------------------------------------------------------------ */

static int freshness_search_impl(const GV_Database *db,
                                  const float *query, size_t k,
                                  const GV_FreshnessParams *params,
                                  const char *filter_key, const char *filter_value,
                                  GV_FreshnessResult *out)
{
    if (!db || !query || k == 0 || !params || !out) return -1;
    if (!db->soa_storage) return -1;

    /* Oversample to get more candidates before re-ranking. */
    size_t fetch_k = k * GV_FRESHNESS_OVERSAMPLE_FACTOR;
    if (fetch_k < k) fetch_k = k; /* overflow guard */

    GV_SearchResult *raw = (GV_SearchResult *)calloc(fetch_k, sizeof(GV_SearchResult));
    if (!raw) return -1;

    int nfound;
    if (filter_key != NULL) {
        nfound = db_search_filtered(db, query, fetch_k, raw,
                                    GV_DISTANCE_EUCLIDEAN, filter_key, filter_value);
    } else {
        nfound = db_search(db, query, fetch_k, raw, GV_DISTANCE_EUCLIDEAN);
    }

    if (nfound < 0) {
        gv_search_results_free(raw, (size_t)fetch_k);
        free(raw);
        return -1;
    }

    uint64_t now_ms = gv_time_now_ms();
    double half_life_ms = params->half_life_sec * 1000.0;

    /* Allocate a temporary array for all candidates. */
    GV_FreshnessResult *tmp = (GV_FreshnessResult *)malloc(
        (size_t)nfound * sizeof(GV_FreshnessResult));
    if (!tmp) {
        gv_search_results_free(raw, (size_t)nfound);
        free(raw);
        return -1;
    }

    for (int i = 0; i < nfound; ++i) {
        size_t vec_id = raw[i].id;
        float  vs     = raw[i].distance; /* lower distance = more similar */
        uint64_t ts   = 0;
        if (vec_id < db->soa_storage->count) {
            ts = db->soa_storage->insert_timestamps[vec_id];
        }

        float fs = freshness_score(ts, now_ms, half_life_ms, params->decay_type);

        /* Convert distance to similarity: use 1/(1+d) so score is in (0,1]. */
        float similarity = 1.0f / (1.0f + (vs < 0.0f ? -vs : vs));

        float final = freshness_blend(similarity, fs,
                                      (float)params->score_weight,
                                      (float)params->freshness_weight);

        tmp[i].index               = vec_id;
        tmp[i].final_score         = final;
        tmp[i].vector_score        = similarity;
        tmp[i].freshness_score     = fs;
        tmp[i].insert_timestamp_ms = ts;
    }

    /* Free each result's owned vector (data+metadata), then the array. */
    gv_search_results_free(raw, (size_t)nfound);
    free(raw);

    /* Sort descending by final_score. */
    qsort(tmp, (size_t)nfound, sizeof(GV_FreshnessResult),
          cmp_freshness_result_desc);

    /* Write top-k into out. */
    size_t write_count = (size_t)nfound < k ? (size_t)nfound : k;
    memcpy(out, tmp, write_count * sizeof(GV_FreshnessResult));
    free(tmp);

    return (int)write_count;
}

/* ---------------------------------------------------------------------------
 * Public entry points
 * ------------------------------------------------------------------------ */

int db_search_with_freshness(const void *db, const float *query, size_t k,
                              const GV_FreshnessParams *params,
                              GV_FreshnessResult *out)
{
    return freshness_search_impl((const GV_Database *)db, query, k,
                                  params, NULL, NULL, out);
}

int db_search_filtered_with_freshness(const void *db, const float *query, size_t k,
                                       const GV_FreshnessParams *params,
                                       const char *filter_key, const char *filter_value,
                                       GV_FreshnessResult *out)
{
    return freshness_search_impl((const GV_Database *)db, query, k,
                                  params, filter_key, filter_value, out);
}
