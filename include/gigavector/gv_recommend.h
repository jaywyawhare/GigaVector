#ifndef GIGAVECTOR_GV_RECOMMEND_H
#define GIGAVECTOR_GV_RECOMMEND_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct GV_Database GV_Database;

typedef struct {
    float positive_weight;     /* Weight for positive examples (default: 1.0) */
    float negative_weight;     /* Weight for negative examples (default: 0.5) */
    int distance_type;         /* Distance metric (default: GV_DISTANCE_COSINE) */
    size_t oversample;         /* Oversample factor for filtering (default: 2) */
    int exclude_input;         /* Exclude input vectors from results (default: 1) */
} GV_RecommendConfig;

typedef struct {
    size_t index;
    float score;
} GV_RecommendResult;

void gv_recommend_config_init(GV_RecommendConfig *config);

/* Recommend by vector indices: average positive vectors, subtract negatives, search */
int gv_recommend_by_id(const GV_Database *db,
                        const size_t *positive_ids, size_t positive_count,
                        const size_t *negative_ids, size_t negative_count,
                        size_t k, const GV_RecommendConfig *config,
                        GV_RecommendResult *results);

/* Recommend by raw vectors */
int gv_recommend_by_vector(const GV_Database *db,
                            const float *positive_vectors, size_t positive_count,
                            const float *negative_vectors, size_t negative_count,
                            size_t dimension, size_t k, const GV_RecommendConfig *config,
                            GV_RecommendResult *results);

/* Discovery: find vectors similar to target but different from context */
int gv_recommend_discover(const GV_Database *db,
                           const float *target, const float *context,
                           size_t dimension, size_t k, const GV_RecommendConfig *config,
                           GV_RecommendResult *results);

#ifdef __cplusplus
}
#endif
#endif
