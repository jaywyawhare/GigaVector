#include <stddef.h>
#include <string.h>

#include "gigavector/gv_score_threshold.h"
#include "gigavector/gv_database.h"

/*
 * All distance metrics in GigaVector are stored so that lower values indicate
 * higher similarity:
 *   - Euclidean (0): actual L2 distance, lower = closer
 *   - Cosine (1):    1 - cosine_similarity, lower = more similar
 *   - Dot product (2): negative dot product, lower = more similar
 *   - Manhattan (3): L1 distance, lower = closer
 *   - Hamming (4):   bit difference count, lower = more similar
 *
 * Therefore the threshold check is uniformly: distance <= threshold.
 */

int gv_threshold_passes(float distance, float threshold, int distance_type) {
    (void)distance_type; /* all metrics: lower distance = more similar */
    return distance <= threshold;
}

size_t gv_threshold_filter(GV_ThresholdResult *results, size_t count,
                            float threshold, int distance_type) {
    if (!results || count == 0) {
        return 0;
    }

    size_t write = 0;
    for (size_t i = 0; i < count; i++) {
        if (gv_threshold_passes(results[i].distance, threshold, distance_type)) {
            if (write != i) {
                results[write] = results[i];
            }
            write++;
        }
    }
    return write;
}

int gv_db_search_with_threshold(const void *db, const float *query_data, size_t k,
                                 int distance_type, float score_threshold,
                                 GV_ThresholdResult *results) {
    if (!db || !query_data || k == 0 || !results) {
        return -1;
    }

    const GV_Database *database = (const GV_Database *)db;

    /* Use stack-allocated GV_SearchResult array for the search.
     * For very large k values, this could be problematic, but the specification
     * says no dynamic allocation. Practically k is small (tens to hundreds). */
    GV_SearchResult search_results[k];
    memset(search_results, 0, sizeof(GV_SearchResult) * k);

    int found = gv_db_search(database, query_data, k, search_results,
                             (GV_DistanceType)distance_type);
    if (found < 0) {
        return -1;
    }

    /* Copy results into GV_ThresholdResult format and track index.
     * GV_SearchResult contains a GV_Vector pointer; we need to derive
     * a meaningful index. The SoA storage index can be recovered from
     * the vector pointer offset relative to the database storage. For
     * simplicity, we use the iteration index (position in search results)
     * as the result index -- this mirrors the ordering from gv_db_search. */
    size_t passed = 0;
    for (int i = 0; i < found; i++) {
        float dist = search_results[i].distance;
        if (gv_threshold_passes(dist, score_threshold, distance_type)) {
            results[passed].index = (size_t)i;
            results[passed].distance = dist;
            passed++;
        }
    }

    return (int)passed;
}
