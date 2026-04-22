#ifndef GIGAVECTOR_GV_LATE_INTERACTION_H
#define GIGAVECTOR_GV_LATE_INTERACTION_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct GV_LateInteractionIndex GV_LateInteractionIndex;

typedef struct {
    size_t token_dimension;    /* Per-token embedding dimension (e.g., 128) */
    size_t max_doc_tokens;     /* Max tokens per document (default: 512) */
    size_t max_query_tokens;   /* Max tokens per query (default: 32) */
    size_t candidate_pool;     /* Candidate pool size for two-stage retrieval (default: 1000) */
} GV_LateInteractionConfig;

typedef struct {
    size_t doc_index;
    float score;               /* MaxSim score */
} GV_LateInteractionResult;

typedef struct {
    size_t total_documents;
    size_t total_tokens_stored;
    size_t memory_bytes;
} GV_LateInteractionStats;

/**
 * @brief Initialize a configuration structure with default values.
 *
 * @param config Configuration to apply/output.
 */
void late_interaction_config_init(GV_LateInteractionConfig *config);
GV_LateInteractionIndex *late_interaction_create(const GV_LateInteractionConfig *config);
/**
 * @brief Destroy an instance and free associated resources.
 *
 * @param index Index instance.
 */
void late_interaction_destroy(GV_LateInteractionIndex *index);

int late_interaction_add_doc(GV_LateInteractionIndex *index,
                                 const float *token_embeddings, size_t num_tokens);

/* MaxSim search: for each query token, find max similarity with any doc token, then sum across query tokens. */
int late_interaction_search(const GV_LateInteractionIndex *index,
                                const float *query_tokens, size_t num_query_tokens,
                                size_t k, GV_LateInteractionResult *results);

/**
 * @brief Delete an item.
 *
 * @param index Index instance.
 * @param doc_index Index value.
 * @return 0 on success, -1 on error.
 */
int late_interaction_delete(GV_LateInteractionIndex *index, size_t doc_index);
/**
 * @brief Retrieve statistics.
 *
 * @param index Index instance.
 * @param stats Output statistics structure.
 * @return 0 on success, -1 on error.
 */
int late_interaction_get_stats(const GV_LateInteractionIndex *index, GV_LateInteractionStats *stats);
/**
 * @brief Return the number of stored items.
 *
 * @param index Index instance.
 * @return Count value.
 */
size_t late_interaction_count(const GV_LateInteractionIndex *index);
/**
 * @brief Save state to a file.
 *
 * @param index Index instance.
 * @param filepath Filesystem path.
 * @return 0 on success, -1 on error.
 */
int late_interaction_save(const GV_LateInteractionIndex *index, const char *filepath);
GV_LateInteractionIndex *late_interaction_load(const char *filepath);

#ifdef __cplusplus
}
#endif
#endif
