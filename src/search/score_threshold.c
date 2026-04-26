#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "search/score_threshold.h"
#include "storage/database.h"

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

int threshold_passes(float distance, float threshold, int distance_type) {
    (void)distance_type; /* all metrics: lower distance = more similar */
    return distance <= threshold;
}

size_t threshold_filter(GV_ThresholdResult *results, size_t count,
                            float threshold, int distance_type) {
    if (!results || count == 0) {
        return 0;
    }

    size_t write = 0;
    for (size_t i = 0; i < count; i++) {
        if (threshold_passes(results[i].distance, threshold, distance_type)) {
            if (write != i) {
                results[write] = results[i];
            }
            write++;
        }
    }
    return write;
}

int db_search_with_threshold(const void *db, const float *query_data, size_t k,
                                 int distance_type, float score_threshold,
                                 GV_ThresholdResult *results) {
    if (!db || !query_data || k == 0 || !results) {
        return -1;
    }

    const GV_Database *database = (const GV_Database *)db;

    GV_SearchResult *search_results = (GV_SearchResult *)calloc(k, sizeof(GV_SearchResult));
    if (!search_results) return -1;

    int found = db_search(database, query_data, k, search_results,
                             (GV_DistanceType)distance_type);
    if (found < 0) {
        free(search_results);
        return -1;
    }

    size_t passed = 0;
    for (int i = 0; i < found; i++) {
        float dist = search_results[i].distance;
        if (threshold_passes(dist, score_threshold, distance_type)) {
            results[passed].index = (size_t)i;
            results[passed].distance = dist;
            passed++;
        }
    }

    free(search_results);
    return (int)passed;
}
