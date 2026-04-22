#ifndef GIGAVECTOR_GV_OPTIMIZER_H
#define GIGAVECTOR_GV_OPTIMIZER_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GV_PLAN_EXACT_SCAN = 0,       /* Brute-force exact search */
    GV_PLAN_INDEX_SEARCH = 1,     /* Use configured index */
    GV_PLAN_OVERSAMPLE_FILTER = 2 /* Oversample then post-filter */
} GV_PlanStrategy;

typedef struct {
    GV_PlanStrategy strategy;
    size_t ef_search;              /* Recommended ef_search for HNSW */
    size_t nprobe;                 /* Recommended nprobe for IVF */
    size_t rerank_top;             /* Recommended rerank for PQ */
    double estimated_cost;         /* Estimated relative cost (lower = faster) */
    double estimated_recall;       /* Estimated recall (0.0 to 1.0) */
    int use_metadata_index;        /* 1 if should use metadata index pre-filter */
    size_t oversample_k;           /* If oversampling, how many to fetch before filter */
    char explanation[256];         /* Human-readable plan explanation */
} GV_QueryPlan;

typedef struct {
    size_t total_vectors;
    size_t dimension;
    int index_type;
    double deleted_ratio;
    double avg_vectors_per_filter_match; /* estimated filter selectivity */
    size_t last_search_latency_us;
} GV_CollectionStats;

typedef struct GV_QueryOptimizer GV_QueryOptimizer;

GV_QueryOptimizer *optimizer_create(void);
/**
 * @brief Destroy an instance and free associated resources.
 *
 * @param opt opt.
 */
void optimizer_destroy(GV_QueryOptimizer *opt);

/**
 * @brief Perform the operation.
 *
 * @param opt opt.
 * @param stats Output statistics structure.
 */
void optimizer_update_stats(GV_QueryOptimizer *opt, const GV_CollectionStats *stats);

int optimizer_plan(const GV_QueryOptimizer *opt, size_t k,
                      int has_filter, double filter_selectivity,
                      GV_QueryPlan *plan);

void optimizer_record_result(GV_QueryOptimizer *opt, const GV_QueryPlan *plan,
                                 uint64_t actual_latency_us, double actual_recall);

/**
 * @brief Perform the operation.
 *
 * @param opt opt.
 * @param k k.
 * @return Count value.
 */
size_t optimizer_recommend_ef_search(const GV_QueryOptimizer *opt, size_t k);
/**
 * @brief Perform the operation.
 *
 * @param opt opt.
 * @param k k.
 * @return Count value.
 */
size_t optimizer_recommend_nprobe(const GV_QueryOptimizer *opt, size_t k);

#ifdef __cplusplus
}
#endif
#endif
