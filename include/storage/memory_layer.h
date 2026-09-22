#ifndef GIGAVECTOR_GV_MEMORY_LAYER_H
#define GIGAVECTOR_GV_MEMORY_LAYER_H

#include <stddef.h>
#include <stdint.h>
#include <time.h>

#include "storage/database.h"
#include "core/types.h"
#include "multimodal/llm.h"
#include "features/context_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GV_MEMORY_TYPE_FACT = 0,        /**< Factual information. */
    GV_MEMORY_TYPE_PREFERENCE = 1,  /**< User preference. */
    GV_MEMORY_TYPE_RELATIONSHIP = 2,/**< Relationship between entities. */
    GV_MEMORY_TYPE_EVENT = 3        /**< Event or occurrence. */
} GV_MemoryType;

typedef enum {
    GV_CONSOLIDATION_MERGE = 0,     /**< Merge similar memories into one. */
    GV_CONSOLIDATION_UPDATE = 1,     /**< Update existing memory with new info. */
    GV_CONSOLIDATION_LINK = 2,      /**< Create relationship link. */
    GV_CONSOLIDATION_ARCHIVE = 3    /**< Archive redundant memory. */
} GV_ConsolidationStrategy;

/**
 * @brief Memory link/relationship type enumeration.
 *
 * Typed connections between memories that help build a knowledge graph
 * and improve retrieval.
 */
typedef enum {
    GV_LINK_SIMILAR = 0,            /**< Memories are semantically similar. */
    GV_LINK_SUPPORTS = 1,           /**< This memory supports/reinforces target. */
    GV_LINK_CONTRADICTS = 2,        /**< This memory contradicts target. */
    GV_LINK_EXTENDS = 3,            /**< This memory extends/elaborates target. */
    GV_LINK_CAUSAL = 4,             /**< This memory is caused by target. */
    GV_LINK_EXAMPLE = 5,            /**< This memory is an example of target. */
    GV_LINK_PREREQUISITE = 6,       /**< Target depends on this memory. */
    GV_LINK_TEMPORAL = 7            /**< Temporal relationship (before/after). */
} GV_MemoryLinkType;

/**
 * @brief Memory link structure.
 *
 * Represents a directed, typed connection between two memories.
 * Links are bidirectional - when A->B is created, B->A is also created
 * with reciprocal type and slightly reduced strength (0.9x).
 */
typedef struct {
    char *target_memory_id;         /**< ID of the linked memory. */
    GV_MemoryLinkType link_type;    /**< Type of relationship. */
    float strength;                 /**< Link strength (0.0-1.0). */
    time_t created_at;              /**< When link was created. */
    char *reason;                   /**< Optional: why link was created. */
} GV_MemoryLink;

typedef struct {
    char *memory_id;                /**< Unique memory identifier. */
    GV_MemoryType memory_type;       /**< Type of memory. */
    char *source;                   /**< Original source identifier. */
    time_t timestamp;               /**< Creation timestamp. */
    time_t last_accessed;           /**< Last access timestamp for decay. */
    uint32_t access_count;          /**< Number of times accessed. */
    double importance_score;        /**< Importance score (0.0-1.0). */
    char *extraction_metadata;       /**< JSON string with extraction details. */
    char **related_memory_ids;      /**< Array of related memory IDs (legacy). */
    size_t related_count;           /**< Number of related memories (legacy). */
    GV_MemoryLink *links;           /**< Array of typed memory links. */
    size_t link_count;              /**< Number of links. */
    int consolidated;               /**< 1 if consolidated, 0 otherwise. */
    time_t valid_from;              /**< Fact valid from (0 = unset). */
    time_t valid_to;                /**< Fact valid until (0 = unset). */
} GV_MemoryMetadata;

/**
 * @brief Search options for memory retrieval.
 *
 * Provides fine-grained control over search behavior including
 * temporal weighting of semantic vs. recency scores.
 */
typedef struct {
    float temporal_weight;          /**< Blend factor: 0.0=semantic only, 1.0=recency only. */
    float importance_weight;        /**< Weight for importance in final score (default: 0.4). */
    int include_linked;             /**< Include linked memories in results (1) or not (0). */
    float link_boost;               /**< Score boost for linked memories (default: 0.1). */
    time_t min_timestamp;           /**< Filter: minimum creation timestamp. */
    time_t max_timestamp;           /**< Filter: maximum creation timestamp. */
    int memory_type;                /**< Filter: specific memory type (-1 = all). */
    const char *source;             /**< Filter: specific source (NULL = all). */
    const size_t *candidate_vector_indices; /**< Optional vector index allow-list (NULL = all). */
    size_t candidate_count;         /**< Length of candidate_vector_indices (0 = all). */
} GV_MemorySearchOptions;

/**
 * @brief Create default search options.
 *
 * @return Default search options with balanced weighting.
 */
GV_MemorySearchOptions memory_search_options_default(void);

typedef struct {
    char *memory_id;                /**< Memory identifier. */
    char *content;                  /**< Memory content text. */
    float relevance_score;          /**< Relevance score (0.0-1.0). */
    float distance;                 /**< Vector distance. */
    GV_MemoryMetadata *metadata;   /**< Memory metadata. */
    GV_MemoryMetadata **related;   /**< Related memories. */
    size_t related_count;           /**< Number of related memories. */
} GV_MemoryResult;

typedef struct {
    double extraction_threshold;    /**< Minimum importance for extraction (0.0-1.0). */
    double consolidation_threshold; /**< Similarity threshold for consolidation (0.0-1.0). */
    GV_ConsolidationStrategy default_strategy; /**< Default consolidation strategy. */
    int enable_temporal_weighting;  /**< Enable temporal relevance weighting. */
    int enable_relationship_retrieval; /**< Include related memories in results. */
    size_t max_related_memories;    /**< Maximum related memories to return. */
    void *llm_config;               /**< LLM configuration (GV_LLMConfig*), NULL to disable LLM. */
    int use_llm_extraction;         /**< Use LLM for extraction (1) or heuristics only (0). */
    int use_llm_consolidation;      /**< Use LLM for consolidation (1) or similarity only (0). */
    void *context_graph_config;     /**< Context graph configuration (GV_ContextGraphConfig*), NULL to disable. */
    int enable_context_graph;        /**< Enable context graph (1) or disable (0). */
} GV_MemoryLayerConfig;

typedef struct GV_MemoryLayer {
    GV_Database *db;                /**< Underlying vector database. */
    GV_MemoryLayerConfig config;    /**< Configuration. */
    uint64_t next_memory_id;        /**< Next memory ID counter. */
    pthread_mutex_t mutex;          /**< Mutex for thread safety. */
    void *llm;                      /**< LLM handle (GV_LLM*), NULL if not configured. */
    void *context_graph;           /**< Context graph handle (GV_ContextGraph*), NULL if not configured. */
} GV_MemoryLayer;

/**
 * @brief Create a new memory layer.
 *
 * @param db Vector database to use; must be non-NULL.
 * @param config Configuration structure; NULL uses defaults.
 * @return Allocated memory layer or NULL on failure.
 */
GV_MemoryLayer *memory_layer_create(GV_Database *db, const GV_MemoryLayerConfig *config);

/**
 * @brief Destroy a memory layer.
 *
 * @param layer Memory layer to destroy; safe to call with NULL.
 */
void memory_layer_destroy(GV_MemoryLayer *layer);

/**
 * @brief Add a memory directly with content and metadata.
 *
 * @param layer Memory layer; must be non-NULL.
 * @param content Memory content text; must be non-NULL.
 * @param embedding Vector embedding for content; must match db dimension.
 * @param metadata Memory metadata; ownership transferred if non-NULL.
 * @param out_vector_index Optional output: vector index assigned in the database.
 * @return Memory ID string (caller must free) or NULL on failure.
 */
char *memory_add(GV_MemoryLayer *layer, const char *content, 
                     const float *embedding, GV_MemoryMetadata *metadata,
                     size_t *out_vector_index);

/**
 * @brief Add a memory with optional context-graph ingest control.
 *
 * @param ingest_context When non-zero, extract entities into the context graph on add.
 */
char *memory_add_opts(GV_MemoryLayer *layer, const char *content,
                          const float *embedding, GV_MemoryMetadata *metadata,
                          size_t *out_vector_index, int ingest_context);

/**
 * @brief Extract memories from conversation text.
 *
 * @param layer Memory layer; must be non-NULL.
 * @param conversation Conversation text to extract from; must be non-NULL.
 * @param conversation_id Conversation identifier; can be NULL.
 * @param embeddings Array of embeddings for extracted memories; must be pre-allocated.
 * @param memory_count Output: number of memories extracted.
 * @return Array of memory IDs (caller must free) or NULL on failure.
 */
char **memory_extract_from_conversation(GV_MemoryLayer *layer,
                                             const char *conversation,
                                             const char *conversation_id,
                                             float **embeddings,
                                             size_t *memory_count);

/**
 * @brief Extract memories from plain text document.
 *
 * @param layer Memory layer; must be non-NULL.
 * @param text Document text; must be non-NULL.
 * @param source Source identifier; can be NULL.
 * @param embeddings Array of embeddings for extracted memories; must be pre-allocated.
 * @param memory_count Output: number of memories extracted.
 * @return Array of memory IDs (caller must free) or NULL on failure.
 */
char **memory_extract_from_text(GV_MemoryLayer *layer,
                                    const char *text,
                                    const char *source,
                                    float **embeddings,
                                    size_t *memory_count);

/**
 * @brief Consolidate similar memories.
 *
 * @param layer Memory layer; must be non-NULL.
 * @param threshold Similarity threshold (0.0-1.0); if <= 0, uses config threshold.
 * @param strategy Consolidation strategy; if negative, uses config strategy.
 * @return Number of memories consolidated, or -1 on error.
 */
int memory_consolidate(GV_MemoryLayer *layer, double threshold, 
                           int strategy);

/**
 * @brief Search for memories by query.
 *
 * @param layer Memory layer; must be non-NULL.
 * @param query_embedding Query vector embedding; must match db dimension.
 * @param k Number of results to return.
 * @param results Output array; must be pre-allocated with k elements.
 * @param distance_type Distance metric to use.
 * @return Number of results found (0 to k), or -1 on error.
 */
int memory_search(GV_MemoryLayer *layer, const float *query_embedding,
                      size_t k, GV_MemoryResult *results,
                      GV_DistanceType distance_type);

/**
 * @brief Search for memories with advanced options.
 *
 * Provides fine-grained control over search behavior including:
 * - temporal_weight: Blend semantic similarity with recency
 * - importance_weight: How much importance score affects ranking
 * - link_boost: Boost for memories connected to top results
 *
 * @param layer Memory layer; must be non-NULL.
 * @param query_embedding Query vector embedding; must match db dimension.
 * @param k Number of results to return.
 * @param results Output array; must be pre-allocated with k elements.
 * @param distance_type Distance metric to use.
 * @param options Search options; NULL uses defaults.
 * @return Number of results found (0 to k), or -1 on error.
 */
int memory_search_advanced(GV_MemoryLayer *layer, const float *query_embedding,
                               size_t k, GV_MemoryResult *results,
                               GV_DistanceType distance_type,
                               const GV_MemorySearchOptions *options);

/**
 * @brief Search for memories with metadata filtering.
 *
 * @param layer Memory layer; must be non-NULL.
 * @param query_embedding Query vector embedding; must match db dimension.
 * @param k Number of results to return.
 * @param results Output array; must be pre-allocated with k elements.
 * @param distance_type Distance metric to use.
 * @param memory_type Filter by memory type; negative to ignore.
 * @param source Filter by source; NULL to ignore.
 * @param min_timestamp Minimum timestamp; 0 to ignore.
 * @param max_timestamp Maximum timestamp; 0 to ignore.
 * @return Number of results found (0 to k), or -1 on error.
 */
int memory_search_filtered(GV_MemoryLayer *layer, const float *query_embedding,
                               size_t k, GV_MemoryResult *results,
                               GV_DistanceType distance_type,
                               int memory_type, const char *source,
                               time_t min_timestamp, time_t max_timestamp);

/**
 * @brief Get related memories for a given memory ID.
 *
 * @param layer Memory layer; must be non-NULL.
 * @param memory_id Memory identifier; must be non-NULL.
 * @param k Maximum number of related memories to return.
 * @param results Output array; must be pre-allocated with k elements.
 * @return Number of related memories found, or -1 on error.
 */
int memory_get_related(GV_MemoryLayer *layer, const char *memory_id,
                          size_t k, GV_MemoryResult *results);

/**
 * @brief Get a memory by ID.
 *
 * @param layer Memory layer; must be non-NULL.
 * @param memory_id Memory identifier; must be non-NULL.
 * @param result Output structure; must be non-NULL.
 * @return 0 on success, -1 if not found, -2 on error.
 */
int memory_get(GV_MemoryLayer *layer, const char *memory_id,
                   GV_MemoryResult *result);

/**
 * @brief Copy the stored embedding vector for a memory.
 *
 * @param layer Memory layer; must be non-NULL.
 * @param memory_id Memory identifier; must be non-NULL.
 * @param out_embedding Output buffer; must hold @p dimension floats.
 * @param dimension Expected embedding dimension.
 * @return 0 on success, -1 if not found, -2 on error.
 */
int memory_get_embedding(GV_MemoryLayer *layer, const char *memory_id,
                             float *out_embedding, size_t dimension);

/**
 * @brief Update a memory's content and metadata.
 *
 * @param layer Memory layer; must be non-NULL.
 * @param memory_id Memory identifier; must be non-NULL.
 * @param new_embedding Updated vector embedding; NULL to keep existing.
 * @param new_metadata Updated metadata; NULL to keep existing.
 * @return 0 on success, -1 on error.
 */
int memory_update(GV_MemoryLayer *layer, const char *memory_id,
                      const float *new_embedding, GV_MemoryMetadata *new_metadata);

/**
 * @brief Delete a memory by ID.
 *
 * @param layer Memory layer; must be non-NULL.
 * @param memory_id Memory identifier; must be non-NULL.
 * @return 0 on success, -1 on error.
 */
int memory_delete(GV_MemoryLayer *layer, const char *memory_id);

/**
 * @brief Delete every memory whose source matches @p source.
 *
 * Cross-layer rollback primitive: removes all facts ingested from a given
 * chunk/doc id (memories store their provenance as the "source" metadata key).
 *
 * @param layer Memory layer; must be non-NULL.
 * @param source Source identifier to match; must be non-NULL.
 * @return Number of memories deleted, or -1 on error.
 */
int memory_delete_by_source(GV_MemoryLayer *layer, const char *source);

/**
 * @brief Free memory result structure.
 *
 * @param result Result to free; safe to call with NULL.
 */
void memory_result_free(GV_MemoryResult *result);

/**
 * @brief Free memory metadata structure.
 *
 * @param metadata Metadata to free; safe to call with NULL.
 */
void memory_metadata_free(GV_MemoryMetadata *metadata);

/**
 * @brief Create default memory layer configuration.
 *
 * @return Default configuration structure.
 */
GV_MemoryLayerConfig memory_layer_config_default(void);

/**
 * @brief Create a link between two memories.
 *
 * Creates a directed link from source to target memory. Automatically
 * creates a reciprocal link (target->source) with:
 * - Reciprocal link type (supports <-> supported_by, etc.)
 * - Reduced strength (0.9x of original)
 *
 * @param layer Memory layer; must be non-NULL.
 * @param source_id Source memory ID; must be non-NULL.
 * @param target_id Target memory ID; must be non-NULL.
 * @param link_type Type of relationship.
 * @param strength Link strength (0.0-1.0); clamped if out of range.
 * @param reason Optional reason for link creation; can be NULL.
 * @return 0 on success, -1 on error.
 */
int memory_link_create(GV_MemoryLayer *layer,
                           const char *source_id,
                           const char *target_id,
                           GV_MemoryLinkType link_type,
                           float strength,
                           const char *reason);

/**
 * @brief Remove a link between two memories.
 *
 * Removes both the forward and reciprocal links.
 *
 * @param layer Memory layer; must be non-NULL.
 * @param source_id Source memory ID; must be non-NULL.
 * @param target_id Target memory ID; must be non-NULL.
 * @return 0 on success, -1 on error.
 */
int memory_link_remove(GV_MemoryLayer *layer,
                           const char *source_id,
                           const char *target_id);

/**
 * @brief Get all links for a memory.
 *
 * @param layer Memory layer; must be non-NULL.
 * @param memory_id Memory ID; must be non-NULL.
 * @param links Output array of links; must be pre-allocated.
 * @param max_links Maximum links to return.
 * @return Number of links found, or -1 on error.
 */
int memory_link_get(GV_MemoryLayer *layer,
                        const char *memory_id,
                        GV_MemoryLink *links,
                        size_t max_links);

/**
 * @brief Get the reciprocal link type.
 *
 * Returns the inverse relationship type:
 * - SUPPORTS <-> SUPPORTED_BY (returns SUPPORTS for bidirectional)
 * - CONTRADICTS <-> CONTRADICTS (symmetric)
 * - EXTENDS <-> EXTENDED_BY (returns EXTENDS for bidirectional)
 * - CAUSAL <-> EFFECT_OF (returns CAUSAL)
 * - EXAMPLE <-> EXEMPLIFIED_BY (returns EXAMPLE)
 * - PREREQUISITE <-> DEPENDS_ON (returns PREREQUISITE)
 *
 * @param link_type Original link type.
 * @return Reciprocal link type.
 */
GV_MemoryLinkType memory_link_reciprocal(GV_MemoryLinkType link_type);

/**
 * @brief Free a memory link structure.
 *
 * @param link Link to free; safe to call with NULL.
 */
void memory_link_free(GV_MemoryLink *link);

/**
 * @brief Record a memory access (for access-based scoring).
 *
 * Updates the memory's access count and last_accessed timestamp.
 * This information is used by the importance scoring system.
 *
 * @param layer Memory layer; must be non-NULL.
 * @param memory_id Memory ID; must be non-NULL.
 * @param relevance Relevance score at time of access (0.0-1.0).
 * @return 0 on success, -1 on error.
 */
int memory_record_access(GV_MemoryLayer *layer,
                             const char *memory_id,
                             float relevance);

/**
 * @brief Extract entities from text, ingest into the layer context graph, return names.
 *
 * When context graph is disabled or extraction fails, returns 0 with an empty list.
 *
 * @param layer Memory layer; must be non-NULL.
 * @param text Text to extract from; must be non-NULL.
 * @param out_names Output array of allocated entity name strings; caller frees via
 *                  memory_layer_free_context_entity_names().
 * @param out_count Output number of names.
 * @return 0 on success, -1 on error.
 */
int memory_layer_extract_context_entities(GV_MemoryLayer *layer, const char *text,
                                          char ***out_names, size_t *out_count);

/**
 * @brief Free entity names returned by memory_layer_extract_context_entities().
 *
 * @param names Name array; safe to call with NULL.
 * @param count Number of names.
 */
void memory_layer_free_context_entity_names(char **names, size_t count);

#ifdef __cplusplus
}
#endif

#endif

