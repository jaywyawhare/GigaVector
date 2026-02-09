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

void gv_late_interaction_config_init(GV_LateInteractionConfig *config);
GV_LateInteractionIndex *gv_late_interaction_create(const GV_LateInteractionConfig *config);
void gv_late_interaction_destroy(GV_LateInteractionIndex *index);

/* Add a document as multiple token embeddings */
int gv_late_interaction_add_doc(GV_LateInteractionIndex *index,
                                 const float *token_embeddings, size_t num_tokens);

/* MaxSim search: for each query token, find max similarity with any doc token, sum across query tokens */
int gv_late_interaction_search(const GV_LateInteractionIndex *index,
                                const float *query_tokens, size_t num_query_tokens,
                                size_t k, GV_LateInteractionResult *results);

/* Delete document */
int gv_late_interaction_delete(GV_LateInteractionIndex *index, size_t doc_index);

/* Stats and count */
int gv_late_interaction_get_stats(const GV_LateInteractionIndex *index, GV_LateInteractionStats *stats);
size_t gv_late_interaction_count(const GV_LateInteractionIndex *index);

/* Save/load */
int gv_late_interaction_save(const GV_LateInteractionIndex *index, const char *filepath);
GV_LateInteractionIndex *gv_late_interaction_load(const char *filepath);

#ifdef __cplusplus
}
#endif
#endif
