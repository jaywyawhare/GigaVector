#ifndef GIGAVECTOR_GV_HNSW_FILTERED_H
#define GIGAVECTOR_GV_HNSW_FILTERED_H

#include <stddef.h>
#include <stdint.h>
#include "core/id_bitmap.h"
#include "search/distance.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t k;
    GV_DistanceType distance_type;
    size_t ef_search;
    size_t max_candidates;
} GV_HNSWFilteredConfig;

/*
 * Pre-filtered HNSW search: the graph walk ONLY visits nodes whose IDs
 * are in the allowed_set bitmap. This is dramatically better than
 * post-filtering because it doesn't waste search budget on non-matching nodes.
 */
int gv_hnsw_search_filtered(void *index, const float *query, size_t dimension,
                            size_t k, const GV_IdBitmap *allowed_set,
                            GV_SearchResult *results,
                            const GV_HNSWFilteredConfig *config);

int gv_hnsw_range_search_filtered(void *index, const float *query, size_t dimension,
                                  float radius, const GV_IdBitmap *allowed_set,
                                  GV_SearchResult *results, size_t max_results,
                                  const GV_HNSWFilteredConfig *config);

GV_IdBitmap *gv_build_filter_bitmap(void *payload_index,
                                    const char *filter_key, int filter_op,
                                    const char *filter_value);

#ifdef __cplusplus
}
#endif
#endif
