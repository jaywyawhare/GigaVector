/**
 * @file document_search.h
 * @brief Unified cross-layer search — fans a query out to the embedding, graph,
 *        and memory layers, joins the results on their shared chunk_id, fuses
 *        the per-layer ranks with Reciprocal Rank Fusion, and returns enriched
 *        results (chunk + its triplets + its facts).
 */
#ifndef GIGAVECTOR_GV_DOCUMENT_SEARCH_H
#define GIGAVECTOR_GV_DOCUMENT_SEARCH_H

#include <stddef.h>
#include "storage/database.h"
#include "features/knowledge_graph.h"
#include "storage/memory_layer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char          chunk_id[64];      /**< The shared key that joined the layers. */
    char         *chunk_text;        /**< Chunk text (heap; NULL if unavailable). */
    float         score;             /**< Fused RRF score. */
    float         vector_score;      /**< Cosine similarity from the embedding layer. */
    int           triplet_hit_count; /**< Graph facts attached to this chunk. */
    int           memory_hit_count;  /**< Memory facts attached to this chunk. */
    GV_KGTriple  *triplets;          /**< Graph triples for this chunk (heap). */
    size_t        triplet_count;
    char        **facts;             /**< Memory fact texts for this chunk (heap). */
    size_t        fact_count;
} GV_EnrichedResult;

/**
 * @brief Unified search across the three layers, joined on chunk_id.
 *
 * @param db              Embedding layer (required).
 * @param kg              Graph layer (NULL to skip enrichment).
 * @param mem             Memory layer (NULL to skip).
 * @param query_text      Query text (used for graph/keyword signals; may be NULL).
 * @param query_embedding Query vector (dimension must match db; may be NULL to
 *                        skip vector ranking).
 * @param k               Number of enriched results to return.
 * @param out             Receives a heap array of GV_EnrichedResult.
 * @param out_count       Receives the number of results.
 * @return 0 on success, -1 on error.
 */
int gv_search_document(GV_Database *db, GV_KnowledgeGraph *kg, GV_MemoryLayer *mem,
                       const char *query_text, const float *query_embedding, size_t k,
                       GV_EnrichedResult **out, size_t *out_count);

/** @brief Free enriched results returned by gv_search_document. */
void gv_enriched_results_free(GV_EnrichedResult *r, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_DOCUMENT_SEARCH_H */
