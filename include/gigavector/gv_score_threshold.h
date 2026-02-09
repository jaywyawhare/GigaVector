#ifndef GIGAVECTOR_GV_SCORE_THRESHOLD_H
#define GIGAVECTOR_GV_SCORE_THRESHOLD_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct GV_Database GV_Database;

/* Search result (matches GV_SearchResult from gv_types.h) */
typedef struct {
    size_t index;
    float distance;
} GV_ThresholdResult;

/**
 * Search with score threshold: only return results with distance <= threshold.
 * For similarity metrics (dot product, cosine), threshold is minimum similarity.
 * For distance metrics (euclidean, manhattan), threshold is maximum distance.
 *
 * @param db Database to search.
 * @param query_data Query vector.
 * @param k Maximum results to return.
 * @param distance_type Distance metric.
 * @param score_threshold Maximum distance (or minimum similarity) threshold.
 * @param results Output array of at least k elements.
 * @return Number of results passing threshold (0 to k), or -1 on error.
 */
int gv_db_search_with_threshold(const void *db, const float *query_data, size_t k,
                                 int distance_type, float score_threshold,
                                 GV_ThresholdResult *results);

/**
 * Apply threshold filter to existing search results in-place.
 * Returns new count after filtering.
 */
size_t gv_threshold_filter(GV_ThresholdResult *results, size_t count,
                            float threshold, int distance_type);

/**
 * Check if a distance value passes the threshold for a given metric.
 * For euclidean/manhattan: distance <= threshold
 * For cosine/dot_product: distance >= threshold (higher = more similar, but these
 * are stored as 1-cosine or -dot, so distance <= threshold still applies)
 */
int gv_threshold_passes(float distance, float threshold, int distance_type);

#ifdef __cplusplus
}
#endif
#endif
