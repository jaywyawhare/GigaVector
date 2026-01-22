#ifndef GIGAVECTOR_GV_CONTEXT_GRAPH_H
#define GIGAVECTOR_GV_CONTEXT_GRAPH_H

#include <stddef.h>
#include <stdint.h>
#include <time.h>

#include "gv_llm.h"
#include "gv_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Entity type enumeration.
 */
typedef enum {
    GV_ENTITY_TYPE_PERSON = 0,      /**< Person entity. */
    GV_ENTITY_TYPE_ORGANIZATION = 1,/**< Organization entity. */
    GV_ENTITY_TYPE_LOCATION = 2,     /**< Location entity. */
    GV_ENTITY_TYPE_EVENT = 3,       /**< Event entity. */
    GV_ENTITY_TYPE_OBJECT = 4,      /**< Object entity. */
    GV_ENTITY_TYPE_CONCEPT = 5,     /**< Concept entity. */
    GV_ENTITY_TYPE_USER = 6         /**< User entity (self-reference). */
} GV_EntityType;

/**
 * @brief Graph entity structure.
 */
typedef struct {
    char *entity_id;                /**< Unique entity identifier. */
    char *name;                     /**< Entity name. */
    GV_EntityType entity_type;      /**< Entity type. */
    float *embedding;               /**< Entity embedding vector. */
    size_t embedding_dim;          /**< Embedding dimension. */
    time_t created;                 /**< Creation timestamp. */
    time_t updated;                 /**< Last update timestamp. */
    uint64_t mentions;              /**< Number of times entity is mentioned. */
    char *user_id;                  /**< User ID filter. */
    char *agent_id;                 /**< Agent ID filter (optional). */
    char *run_id;                   /**< Run ID filter (optional). */
} GV_GraphEntity;

/**
 * @brief Graph relationship structure.
 */
typedef struct {
    char *relationship_id;          /**< Unique relationship identifier. */
    char *source_entity_id;         /**< Source entity ID. */
    char *destination_entity_id;    /**< Destination entity ID. */
    char *relationship_type;        /**< Relationship type (e.g., "knows", "works_with"). */
    time_t created;                 /**< Creation timestamp. */
    time_t updated;                 /**< Last update timestamp. */
    uint64_t mentions;              /**< Number of times relationship is mentioned. */
} GV_GraphRelationship;

/**
 * @brief Graph query result structure.
 */
typedef struct {
    char *source_name;              /**< Source entity name. */
    char *relationship_type;        /**< Relationship type. */
    char *destination_name;         /**< Destination entity name. */
    float similarity;               /**< Similarity score (0.0-1.0). */
} GV_GraphQueryResult;

/**
 * @brief Context graph structure (opaque).
 */
typedef struct GV_ContextGraph GV_ContextGraph;

/**
 * @brief Embedding generation callback function type.
 * 
 * @param text Text to generate embedding for.
 * @param embedding_dim Output: dimension of the embedding vector.
 * @param user_data User-provided data pointer.
 * @return Allocated embedding vector (caller must free) or NULL on failure.
 */
typedef float *(*GV_EmbeddingCallback)(const char *text, size_t *embedding_dim, void *user_data);

/**
 * @brief Context graph configuration.
 */
typedef struct {
    void *llm;                      /**< LLM instance for entity extraction (GV_LLM*), NULL to disable. */
    void *embedding_service;        /**< Embedding service instance (GV_EmbeddingService*), NULL to disable. */
    double similarity_threshold;    /**< Similarity threshold for entity matching (0.0-1.0). */
    int enable_entity_extraction;   /**< Enable entity extraction (1) or manual only (0). */
    int enable_relationship_extraction; /**< Enable relationship extraction (1) or manual only (0). */
    size_t max_traversal_depth;     /**< Maximum graph traversal depth. */
    size_t max_results;             /**< Maximum results per query. */
    GV_EmbeddingCallback embedding_callback; /**< Callback to generate embeddings (NULL to disable auto-generation). */
    void *embedding_user_data;      /**< User data passed to embedding callback. */
    size_t embedding_dimension;     /**< Expected embedding dimension (0 if unknown). */
} GV_ContextGraphConfig;

/**
 * @brief Create a new context graph.
 *
 * @param config Configuration structure; NULL uses defaults.
 * @return Allocated context graph or NULL on failure.
 */
GV_ContextGraph *gv_context_graph_create(const GV_ContextGraphConfig *config);

/**
 * @brief Destroy a context graph.
 *
 * @param graph Context graph to destroy; safe to call with NULL.
 */
void gv_context_graph_destroy(GV_ContextGraph *graph);

/**
 * @brief Extract entities and relationships from text.
 *
 * @param graph Context graph; must be non-NULL.
 * @param text Text to extract from; must be non-NULL.
 * @param user_id User ID filter; can be NULL.
 * @param agent_id Agent ID filter; can be NULL.
 * @param run_id Run ID filter; can be NULL.
 * @param entities Output array of entities; must be pre-allocated.
 * @param entity_count Output: number of entities extracted.
 * @param relationships Output array of relationships; must be pre-allocated.
 * @param relationship_count Output: number of relationships extracted.
 * @return 0 on success, -1 on error.
 */
int gv_context_graph_extract(GV_ContextGraph *graph,
                             const char *text,
                             const char *user_id,
                             const char *agent_id,
                             const char *run_id,
                             GV_GraphEntity **entities,
                             size_t *entity_count,
                             GV_GraphRelationship **relationships,
                             size_t *relationship_count);

/**
 * @brief Add entities to the graph.
 *
 * @param graph Context graph; must be non-NULL.
 * @param entities Array of entities to add; must be non-NULL.
 * @param entity_count Number of entities.
 * @return 0 on success, -1 on error.
 */
int gv_context_graph_add_entities(GV_ContextGraph *graph,
                                  const GV_GraphEntity *entities,
                                  size_t entity_count);

/**
 * @brief Add relationships to the graph.
 *
 * @param graph Context graph; must be non-NULL.
 * @param relationships Array of relationships to add; must be non-NULL.
 * @param relationship_count Number of relationships.
 * @return 0 on success, -1 on error.
 */
int gv_context_graph_add_relationships(GV_ContextGraph *graph,
                                      const GV_GraphRelationship *relationships,
                                      size_t relationship_count);

/**
 * @brief Search for related entities in the graph.
 *
 * @param graph Context graph; must be non-NULL.
 * @param query_embedding Query embedding vector; must be non-NULL.
 * @param embedding_dim Embedding dimension.
 * @param user_id User ID filter; can be NULL.
 * @param agent_id Agent ID filter; can be NULL.
 * @param run_id Run ID filter; can be NULL.
 * @param results Output array; must be pre-allocated.
 * @param max_results Maximum number of results to return.
 * @return Number of results found, or -1 on error.
 */
int gv_context_graph_search(GV_ContextGraph *graph,
                             const float *query_embedding,
                             size_t embedding_dim,
                             const char *user_id,
                             const char *agent_id,
                             const char *run_id,
                             GV_GraphQueryResult *results,
                             size_t max_results);

/**
 * @brief Get related entities for a given entity.
 *
 * @param graph Context graph; must be non-NULL.
 * @param entity_id Entity identifier; must be non-NULL.
 * @param max_depth Maximum traversal depth.
 * @param results Output array; must be pre-allocated.
 * @param max_results Maximum number of results to return.
 * @return Number of results found, or -1 on error.
 */
int gv_context_graph_get_related(GV_ContextGraph *graph,
                                  const char *entity_id,
                                  size_t max_depth,
                                  GV_GraphQueryResult *results,
                                  size_t max_results);

/**
 * @brief Delete entities and relationships from the graph.
 *
 * @param graph Context graph; must be non-NULL.
 * @param entities Array of entity identifiers to delete; must be non-NULL.
 * @param entity_count Number of entities.
 * @return 0 on success, -1 on error.
 */
int gv_context_graph_delete_entities(GV_ContextGraph *graph,
                                      const char **entity_ids,
                                      size_t entity_count);

/**
 * @brief Delete relationships from the graph.
 *
 * @param graph Context graph; must be non-NULL.
 * @param relationships Array of relationship identifiers to delete; must be non-NULL.
 * @param relationship_count Number of relationships.
 * @return 0 on success, -1 on error.
 */
int gv_context_graph_delete_relationships(GV_ContextGraph *graph,
                                          const char **relationship_ids,
                                          size_t relationship_count);

/**
 * @brief Free graph entity structure.
 *
 * @param entity Entity to free; safe to call with NULL.
 */
void gv_graph_entity_free(GV_GraphEntity *entity);

/**
 * @brief Free graph relationship structure.
 *
 * @param relationship Relationship to free; safe to call with NULL.
 */
void gv_graph_relationship_free(GV_GraphRelationship *relationship);

/**
 * @brief Free graph query result structure.
 *
 * @param result Result to free; safe to call with NULL.
 */
void gv_graph_query_result_free(GV_GraphQueryResult *result);

/**
 * @brief Create default context graph configuration.
 *
 * @return Default configuration structure.
 */
GV_ContextGraphConfig gv_context_graph_config_default(void);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_CONTEXT_GRAPH_H */

