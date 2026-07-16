/**
 * @file document_ingest.h
 * @brief Document ingest coordinator — fans one document into the embedding,
 *        graph, and memory layers under a shared chunk_id (the cross-layer PK).
 *
 * Embeddings come from a caller-supplied callback (primary) or an optional
 * GV_AutoEmbedder default. Graph triples and memory facts come from optional
 * extractor callbacks; when none are provided (or kg/mem are NULL), that layer
 * is skipped — a valid embeddings-only ingest (no LLM required).
 */
#ifndef GIGAVECTOR_GV_DOCUMENT_INGEST_H
#define GIGAVECTOR_GV_DOCUMENT_INGEST_H

#include <stddef.h>
#include "storage/database.h"
#include "features/knowledge_graph.h"
#include "storage/memory_layer.h"
#include "multimodal/auto_embed.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Embed a batch of texts. Write n*dim floats into @p out (row-major).
 * @return 0 on success, -1 on error.
 */
typedef int (*GV_EmbedFn)(void *ctx, const char **texts, size_t n,
                          size_t dim, float *out);

/** @brief A triple extracted from chunk text. */
typedef struct {
    const char *subject;       /**< Subject entity name. */
    const char *subject_type;  /**< Subject type (NULL -> "Entity"). */
    const char *predicate;     /**< Relation label. */
    const char *object;        /**< Object entity name. */
    const char *object_type;   /**< Object type (NULL -> "Entity"). */
} GV_ExtractedTriple;

/**
 * @brief Extract triples from a chunk. Write up to @p max into @p out.
 *        Returned strings must remain valid until the call returns.
 * @return Number of triples, or -1 on error.
 */
typedef int (*GV_TripleExtractFn)(void *ctx, const char *chunk_text,
                                  GV_ExtractedTriple *out, size_t max);

/**
 * @brief Extract facts from a chunk. Write up to @p max heap-allocated strings
 *        into @p out_facts (the ingest coordinator frees them).
 * @return Number of facts, or -1 on error.
 */
typedef int (*GV_FactExtractFn)(void *ctx, const char *chunk_text,
                                char **out_facts, size_t max);

typedef struct {
    GV_EmbedFn         embed;             /**< Primary embedder (required unless default set). */
    void              *embed_ctx;
    GV_AutoEmbedder   *default_embedder;  /**< Used when embed == NULL. */
    GV_TripleExtractFn extract_triples;   /**< Optional (needs kg). */
    void              *triples_ctx;
    GV_FactExtractFn   extract_facts;     /**< Optional (needs mem). */
    void              *facts_ctx;
    size_t chunk_tokens;                  /**< 0 -> 512. */
    size_t overlap_tokens;                /**< 0 -> 128. */
    size_t max_triples_per_chunk;         /**< 0 -> 8. */
    size_t max_facts_per_chunk;           /**< 0 -> 4. */
} GV_IngestConfig;

typedef struct {
    size_t chunks;   /**< Chunks produced. */
    size_t vectors;  /**< Vectors added to the embedding layer. */
    size_t triples;  /**< Triples added to the graph layer. */
    size_t facts;    /**< Facts added to the memory layer. */
} GV_IngestStats;

/**
 * @brief Ingest a document into all attached layers under a shared chunk_id.
 *
 * @param db   Embedding layer (required).
 * @param kg   Graph layer (NULL to skip).
 * @param mem  Memory layer (NULL to skip).
 * @param text Document text.
 * @param doc_id Document id (NULL -> generated).
 * @param cfg  Ingest configuration (embedder + optional extractors).
 * @param out  Optional statistics.
 * @return 0 on success, -1 on error.
 */
int gv_ingest_document(GV_Database *db, GV_KnowledgeGraph *kg, GV_MemoryLayer *mem,
                       const char *text, const char *doc_id,
                       const GV_IngestConfig *cfg, GV_IngestStats *out);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_DOCUMENT_INGEST_H */
