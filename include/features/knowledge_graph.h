/**
 * @file knowledge_graph.h
 * @brief Knowledge graph layer combining graph structure, vector embeddings,
 *        and a triple store (Subject-Predicate-Object) for semantic reasoning.
 *
 * The knowledge graph provides:
 *  - Entity and relation management with property bags
 *  - SPO triple-pattern queries with wildcard support
 *  - Cosine-similarity semantic search over entity embeddings
 *  - Entity resolution / deduplication
 *  - Link prediction via embedding similarity + structural patterns
 *  - BFS traversal, shortest-path, and subgraph extraction
 *  - Hybrid (vector + graph) search
 *  - Degree-centrality analytics
 *  - Binary persistence with magic header "GVKG"
 */

#ifndef GIGAVECTOR_GV_KNOWLEDGE_GRAPH_H
#define GIGAVECTOR_GV_KNOWLEDGE_GRAPH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GV_KnowledgeGraph GV_KnowledgeGraph;
struct GV_Database; /* fwd decl: KG can resolve entity similarity via db_search */

/**
 * @brief Key-value property node (singly-linked list).
 */
typedef struct GV_KGProp {
    char *key;                      /**< Property key (heap-allocated). */
    char *value;                    /**< Property value (heap-allocated). */
    struct GV_KGProp *next;         /**< Next property in the list. */
} GV_KGProp;

typedef struct {
    uint64_t entity_id;             /**< Unique entity identifier. */
    char *name;                     /**< Human-readable name. */
    char *type;                     /**< Entity type (e.g. "Person", "Organization"). */
    float *embedding;               /**< Optional embedding vector (NULL if absent). */
    size_t dimension;               /**< Embedding dimension (0 when no embedding). */
    GV_KGProp *properties;          /**< Linked list of key-value properties. */
    size_t prop_count;              /**< Number of properties. */
    uint64_t created_at;            /**< Creation timestamp (epoch seconds). */
    float confidence;               /**< Extraction confidence (0.0 - 1.0). */
} GV_KGEntity;

/**
 * @brief Directed relation between two entities (triple edge).
 */
typedef struct {
    uint64_t relation_id;           /**< Unique relation identifier. */
    uint64_t subject_id;            /**< Source entity ID. */
    uint64_t object_id;             /**< Target entity ID. */
    char *predicate;                /**< Relation label (e.g. "works_at"). */
    float weight;                   /**< Strength / confidence of the relation. */
    GV_KGProp *properties;          /**< Linked list of key-value properties. */
    uint64_t created_at;            /**< Creation timestamp (epoch seconds). */
} GV_KGRelation;

/**
 * @brief Materialised triple returned by SPO queries.
 */
typedef struct {
    uint64_t subject_id;            /**< Subject entity ID. */
    char *subject_name;             /**< Subject entity name (heap-allocated). */
    char *predicate;                /**< Predicate label (heap-allocated). */
    uint64_t object_id;             /**< Object entity ID. */
    char *object_name;              /**< Object entity name (heap-allocated). */
    float score;                    /**< Relevance / confidence score. */
} GV_KGTriple;

typedef struct {
    size_t entity_bucket_count;     /**< Entity hash-table buckets (default 4096). */
    size_t relation_bucket_count;   /**< Relation hash-table buckets (default 8192). */
    size_t embedding_dimension;     /**< Embedding vector dimension (default 128, 0 disables). */
    float similarity_threshold;     /**< Cosine threshold for entity dedup (default 0.7). */
    float link_prediction_threshold;/**< Threshold for link prediction (default 0.8). */
    size_t max_entities;            /**< Hard cap on entity count (default 1000000). */
} GV_KGConfig;

typedef struct {
    uint64_t *entity_ids;           /**< Array of entity IDs in the subgraph. */
    size_t entity_count;            /**< Number of entities. */
    uint64_t *relation_ids;         /**< Array of relation IDs in the subgraph. */
    size_t relation_count;          /**< Number of relations. */
} GV_KGSubgraph;

typedef struct {
    uint64_t entity_id;             /**< Matched entity ID. */
    char *name;                     /**< Entity name (heap-allocated). */
    char *type;                     /**< Entity type (heap-allocated). */
    float similarity;               /**< Cosine similarity to query. */
} GV_KGSearchResult;

/**
 * @brief Predicted (or duplicate-candidate) link between two entities.
 */
typedef struct {
    uint64_t entity_a;              /**< First entity ID. */
    uint64_t entity_b;              /**< Second entity ID. */
    char *predicted_predicate;      /**< Predicted predicate label (heap-allocated). */
    float confidence;               /**< Prediction confidence. */
} GV_KGLinkPrediction;

typedef struct {
    size_t entity_count;            /**< Total entities. */
    size_t relation_count;          /**< Total relations. */
    size_t triple_count;            /**< Total triples (== relation_count). */
    size_t type_count;              /**< Distinct entity types. */
    size_t predicate_count;         /**< Distinct predicate labels. */
    size_t embedding_count;         /**< Entities that carry embeddings. */
} GV_KGStats;

/**
 * @brief Initialise a configuration struct with default values.
 *
 * @param config  Configuration to initialise; must be non-NULL.
 */
void kg_config_init(GV_KGConfig *config);

/**
 * @brief Create a new knowledge graph.
 *
 * @param config  Configuration; NULL uses defaults.
 * @return Opaque handle, or NULL on allocation failure.
 */
GV_KnowledgeGraph *kg_create(const GV_KGConfig *config);

/**
 * @brief Destroy a knowledge graph and free all resources.
 *
 * @param kg  Knowledge graph to destroy; safe to call with NULL.
 */
void kg_destroy(GV_KnowledgeGraph *kg);

/**
 * @brief Add an entity to the knowledge graph.
 *
 * @param kg         Knowledge graph handle.
 * @param name       Entity name; must be non-NULL.
 * @param type       Entity type string; must be non-NULL.
 * @param embedding  Optional embedding vector (NULL to skip).
 * @param dimension  Dimension of the embedding (ignored when embedding is NULL).
 * @return Assigned entity_id (>0), or 0 on failure.
 */
uint64_t kg_add_entity(GV_KnowledgeGraph *kg, const char *name,
                           const char *type, const float *embedding,
                           size_t dimension);

/**
 * @brief Remove an entity and cascade-delete its relations.
 *
 * @param kg         Knowledge graph handle.
 * @param entity_id  Entity to remove.
 * @return 0 on success, -1 on failure (not found or NULL kg).
 */
int kg_remove_entity(GV_KnowledgeGraph *kg, uint64_t entity_id);

/**
 * @brief Look up an entity by ID.
 *
 * @param kg         Knowledge graph handle.
 * @param entity_id  Entity identifier.
 * @return Pointer to internal entity (do NOT free), or NULL if not found.
 */
const GV_KGEntity *kg_get_entity(const GV_KnowledgeGraph *kg,
                                     uint64_t entity_id);

/**
 * @brief Set (or overwrite) a property on an entity.
 *
 * @param kg         Knowledge graph handle.
 * @param entity_id  Target entity.
 * @param key        Property key; must be non-NULL.
 * @param value      Property value; must be non-NULL.
 * @return 0 on success, -1 on failure.
 */
int kg_set_entity_prop(GV_KnowledgeGraph *kg, uint64_t entity_id,
                           const char *key, const char *value);

/**
 * @brief Get the value of an entity property.
 *
 * @param kg         Knowledge graph handle.
 * @param entity_id  Target entity.
 * @param key        Property key.
 * @return Property value (internal pointer, do NOT free), or NULL if missing.
 */
const char *kg_get_entity_prop(const GV_KnowledgeGraph *kg,
                                   uint64_t entity_id, const char *key);

/**
 * @brief Find entities matching a given type.
 *
 * @param kg         Knowledge graph handle.
 * @param type       Entity type to match.
 * @param out_ids    Output array for matching entity IDs.
 * @param max_count  Capacity of out_ids.
 * @return Number of matches written, or -1 on error.
 */
int kg_find_entities_by_type(const GV_KnowledgeGraph *kg, const char *type,
                                 uint64_t *out_ids, size_t max_count);

/**
 * @brief Find entities by exact name match.
 *
 * @param kg         Knowledge graph handle.
 * @param name       Name to match.
 * @param out_ids    Output array for matching entity IDs.
 * @param max_count  Capacity of out_ids.
 * @return Number of matches written, or -1 on error.
 */
int kg_find_entities_by_name(const GV_KnowledgeGraph *kg, const char *name,
                                 uint64_t *out_ids, size_t max_count);

/**
 * @brief Add a directed relation (triple) between two entities.
 *
 * @param kg         Knowledge graph handle.
 * @param subject    Subject (source) entity ID; must exist.
 * @param predicate  Predicate label; must be non-NULL.
 * @param object     Object (target) entity ID; must exist.
 * @param weight     Relation weight / confidence.
 * @return Assigned relation_id (>0), or 0 on failure.
 */
uint64_t kg_add_relation(GV_KnowledgeGraph *kg, uint64_t subject,
                             const char *predicate, uint64_t object,
                             float weight);

/**
 * @brief Remove a relation by ID.
 *
 * @param kg          Knowledge graph handle.
 * @param relation_id Relation to remove.
 * @return 0 on success, -1 on failure.
 */
int kg_remove_relation(GV_KnowledgeGraph *kg, uint64_t relation_id);

/**
 * @brief Look up a relation by ID.
 *
 * @param kg          Knowledge graph handle.
 * @param relation_id Relation identifier.
 * @return Pointer to internal relation (do NOT free), or NULL if not found.
 */
const GV_KGRelation *kg_get_relation(const GV_KnowledgeGraph *kg,
                                         uint64_t relation_id);

/**
 * @brief Set (or overwrite) a property on a relation.
 *
 * @param kg          Knowledge graph handle.
 * @param relation_id Target relation.
 * @param key         Property key; must be non-NULL.
 * @param value       Property value; must be non-NULL.
 * @return 0 on success, -1 on failure.
 */
int kg_set_relation_prop(GV_KnowledgeGraph *kg, uint64_t relation_id,
                             const char *key, const char *value);

/**
 * @brief Query triples using an SPO pattern.  Pass NULL for any parameter to
 *        treat it as a wildcard.
 *
 * @param kg         Knowledge graph handle.
 * @param subject    Pointer to subject ID filter (NULL = wildcard).
 * @param predicate  Predicate string filter (NULL = wildcard).
 * @param object     Pointer to object ID filter (NULL = wildcard).
 * @param out        Output array for matching triples.
 * @param max_count  Capacity of out.
 * @return Number of triples written, or -1 on error.
 */
int kg_query_triples(const GV_KnowledgeGraph *kg, const uint64_t *subject,
                         const char *predicate, const uint64_t *object,
                         GV_KGTriple *out, size_t max_count);

/**
 * @brief Free an array of triples returned by kg_query_triples.
 *
 * @param triples  Array to free; safe to call with NULL.
 * @param count    Number of elements.
 */
void kg_free_triples(GV_KGTriple *triples, size_t count);

/**
 * @brief Find the k most similar entities by cosine similarity.
 *
 * @param kg              Knowledge graph handle.
 * @param query_embedding Query vector.
 * @param dimension       Query dimension (must match config).
 * @param k               Maximum results to return.
 * @param results         Output array (pre-allocated, at least k elements).
 * @return Number of results written, or -1 on error.
 */
int kg_search_similar(const GV_KnowledgeGraph *kg,
                          const float *query_embedding, size_t dimension,
                          size_t k, GV_KGSearchResult *results);

/**
 * @brief Combined name + embedding search.
 *
 * @param kg              Knowledge graph handle.
 * @param text            Text to match against entity names.
 * @param text_embedding  Embedding of the text (may be NULL).
 * @param dimension       Embedding dimension.
 * @param k               Maximum results.
 * @param results         Output array.
 * @return Number of results written, or -1 on error.
 */
int kg_search_by_text(const GV_KnowledgeGraph *kg, const char *text,
                          const float *text_embedding, size_t dimension,
                          size_t k, GV_KGSearchResult *results);

/**
 * @brief Free an array of search results.
 *
 * @param results  Array to free; safe to call with NULL.
 * @param count    Number of elements.
 */
void kg_free_search_results(GV_KGSearchResult *results, size_t count);

/**
 * @brief Resolve an entity: find an existing match or create a new one.
 *
 * Returns the full 64-bit entity_id. Entity IDs start at 1, so 0 is
 * reserved as the "error"/"not found" sentinel. Prefer this over
 * kg_resolve_entity() which narrows the id to int.
 *
 * @param kg         Knowledge graph handle.
 * @param name       Entity name.
 * @param type       Entity type.
 * @param embedding  Optional embedding (NULL to skip similarity check).
 * @param dimension  Embedding dimension.
 * @return entity_id of the resolved (existing or newly created) entity, or 0 on error.
 */
uint64_t kg_resolve_entity_id(GV_KnowledgeGraph *kg, const char *name,
                          const char *type, const float *embedding,
                          size_t dimension);

/**
 * @brief Resolve an entity, returning the id narrowed to int (legacy API).
 *
 * Kept for backwards-compatible callers that expect an int return. Because
 * entity IDs are 64-bit, a full id may not fit in an int; rather than
 * silently truncating (which could corrupt the id or alias a different
 * entity), this clamps any id that does not fit in a non-negative int to
 * INT_MAX. New code should call kg_resolve_entity_id() instead.
 *
 * @param kg         Knowledge graph handle.
 * @param name       Entity name.
 * @param type       Entity type.
 * @param embedding  Optional embedding (NULL to skip similarity check).
 * @param dimension  Embedding dimension.
 * @return entity_id clamped to [0, INT_MAX], or 0 on error.
 */
int kg_resolve_entity(GV_KnowledgeGraph *kg, const char *name,
                          const char *type, const float *embedding,
                          size_t dimension);

/**
 * @brief Find potential duplicate entities by embedding similarity.
 *
 * @param kg         Knowledge graph handle.
 * @param threshold  Cosine similarity threshold for duplicates.
 * @param out        Output array for duplicate pairs.
 * @param max_count  Capacity of out.
 * @return Number of pairs written, or -1 on error.
 */
int kg_find_duplicates(const GV_KnowledgeGraph *kg, float threshold,
                           GV_KGLinkPrediction *out, size_t max_count);

/**
 * @brief Merge two entities: move relations, copy properties, delete donor.
 *
 * @param kg        Knowledge graph handle.
 * @param keep_id   Entity to keep.
 * @param merge_id  Entity to merge into keep_id (will be deleted).
 * @return 0 on success, -1 on failure.
 */
int kg_merge_entities(GV_KnowledgeGraph *kg, uint64_t keep_id,
                          uint64_t merge_id);

/**
 * @brief Predict missing links for an entity.
 *
 * @param kg         Knowledge graph handle.
 * @param entity_id  Source entity.
 * @param k          Maximum predictions.
 * @param results    Output array.
 * @return Number of predictions written, or -1 on error.
 */
int kg_predict_links(const GV_KnowledgeGraph *kg, uint64_t entity_id,
                         size_t k, GV_KGLinkPrediction *results);

/**
 * @brief Get immediate neighbours of an entity (outgoing + incoming).
 *
 * @param kg         Knowledge graph handle.
 * @param entity_id  Centre entity.
 * @param out_ids    Output array for neighbour IDs.
 * @param max_count  Capacity of out_ids.
 * @return Number of neighbours written, or -1 on error.
 */
int kg_get_neighbors(const GV_KnowledgeGraph *kg, uint64_t entity_id,
                         uint64_t *out_ids, size_t max_count);

/**
 * @brief BFS traversal from a start entity.
 *
 * @param kg         Knowledge graph handle.
 * @param start      Start entity ID.
 * @param max_depth  Maximum BFS depth.
 * @param out_ids    Output array for reachable entity IDs.
 * @param max_count  Capacity of out_ids.
 * @return Number of entities written, or -1 on error.
 */
int kg_traverse(const GV_KnowledgeGraph *kg, uint64_t start,
                    size_t max_depth, uint64_t *out_ids, size_t max_count);

typedef struct {
    uint64_t entity_id;
    float activation;
} GV_KGActivation;

/**
 * @brief Spreading activation from seed entities with optional causal edge boost.
 */
int kg_spreading_activation(const GV_KnowledgeGraph *kg,
                            const uint64_t *seed_entities, size_t seed_count,
                            size_t max_depth, float causal_boost,
                            GV_KGActivation *out, size_t max_out);

/**
 * @brief Find the shortest path between two entities (BFS).
 *
 * @param kg        Knowledge graph handle.
 * @param from      Source entity ID.
 * @param to        Destination entity ID.
 * @param path_ids  Output array for entity IDs along the path (including from and to).
 * @param max_len   Capacity of path_ids.
 * @return Path length (number of entities), or -1 if unreachable / error.
 */
int kg_shortest_path(const GV_KnowledgeGraph *kg, uint64_t from,
                         uint64_t to, uint64_t *path_ids, size_t max_len);

/**
 * @brief Extract a subgraph within a given radius of a centre entity.
 *
 * @param kg        Knowledge graph handle.
 * @param center    Centre entity ID.
 * @param radius    Maximum hop distance.
 * @param subgraph  Output subgraph structure (caller provides, function fills).
 * @return 0 on success, -1 on error.
 */
int kg_extract_subgraph(const GV_KnowledgeGraph *kg, uint64_t center,
                            size_t radius, GV_KGSubgraph *subgraph);

/**
 * @brief Free resources within a subgraph structure.
 *
 * @param subgraph  Subgraph to free; safe to call with NULL.
 */
void kg_free_subgraph(GV_KGSubgraph *subgraph);

/**
 * @brief Hybrid search: embedding similarity filtered by entity type and predicate.
 *
 * @param kg               Knowledge graph handle.
 * @param query_embedding  Query vector.
 * @param dimension        Embedding dimension.
 * @param entity_type      Required entity type (NULL = any).
 * @param predicate_filter Required predicate on at least one relation (NULL = any).
 * @param k                Maximum results.
 * @param results          Output array.
 * @return Number of results written, or -1 on error.
 */
int kg_hybrid_search(const GV_KnowledgeGraph *kg,
                         const float *query_embedding, size_t dimension,
                         const char *entity_type, const char *predicate_filter,
                         size_t k, GV_KGSearchResult *results);

/**
 * @brief Compute aggregate statistics.
 *
 * @param kg     Knowledge graph handle.
 * @param stats  Output structure.
 * @return 0 on success, -1 on error.
 */
int kg_get_stats(const GV_KnowledgeGraph *kg, GV_KGStats *stats);

/**
 * @brief Compute degree centrality for an entity.
 *
 * Centrality = (in_degree + out_degree) / (total_entities - 1).
 *
 * @param kg         Knowledge graph handle.
 * @param entity_id  Entity to evaluate.
 * @return Centrality value in [0,1], or -1.0f on error.
 */
float kg_entity_centrality(const GV_KnowledgeGraph *kg, uint64_t entity_id);

/**
 * @brief Get distinct entity types.
 *
 * @param kg         Knowledge graph handle.
 * @param out_types  Output array of heap-allocated type strings (caller frees).
 * @param max_count  Capacity of out_types.
 * @return Number of types written, or -1 on error.
 */
int kg_get_entity_types(const GV_KnowledgeGraph *kg, char **out_types,
                            size_t max_count);

/**
 * @brief Get distinct predicate labels.
 *
 * @param kg             Knowledge graph handle.
 * @param out_predicates Output array of heap-allocated predicate strings (caller frees).
 * @param max_count      Capacity of out_predicates.
 * @return Number of predicates written, or -1 on error.
 */
int kg_get_predicates(const GV_KnowledgeGraph *kg, char **out_predicates,
                          size_t max_count);

/**
 * @brief Collect all entity IDs (up to max_count). Used by query engines that
 *        must enumerate unlabeled nodes.
 * @return Number written, or -1 on error.
 */
int kg_all_entity_ids(const GV_KnowledgeGraph *kg, uint64_t *out_ids, size_t max_count);

/* ---- Combined-DB integration (Phase 1) ---- */

/**
 * @brief Attach a vector database so entity similarity resolves via db_search
 *        instead of the KG's private embedding store ("vectors as a predicate").
 *
 * When attached, entities added WITH an embedding are mirrored into the DB under
 * the id "kgent:{entity_id}", and kg_search_similar delegates to db_search. Pass
 * NULL to detach. The KG does not own the database. In combined-pipeline mode,
 * add entities WITHOUT embeddings so vectors live only in the DB (no duplication).
 *
 * @param kg  Knowledge graph handle.
 * @param db  Vector database to attach, or NULL to detach.
 */
void kg_attach_vector_db(GV_KnowledgeGraph *kg, struct GV_Database *db);

/**
 * @brief Add a relation carrying a chunk_id facet (provenance for the shared join).
 *
 * Convenience over kg_add_relation + kg_set_relation_prop(rel, "chunk_id", ...).
 * @return relation_id (>0), or 0 on failure.
 */
uint64_t kg_add_relation_with_chunk(GV_KnowledgeGraph *kg, uint64_t subject,
                                     const char *predicate, uint64_t object,
                                     float weight, const char *chunk_id);

/**
 * @brief Query all triples whose relation carries the given chunk_id facet.
 *
 * The cross-layer join primitive: given a chunk_id, return its graph facts.
 * @return Number of triples written, or -1 on error.
 */
int kg_query_triples_by_chunk(const GV_KnowledgeGraph *kg, const char *chunk_id,
                              GV_KGTriple *out, size_t max_count);

/**
 * @brief Reverse-edge query: triples where @p object is the object of @p predicate.
 *
 * Thin convenience over kg_query_triples using the existing object index
 * ("who relates to X via p?"). @p predicate may be NULL (any).
 * @return Number of triples written, or -1 on error.
 */
int kg_query_reverse(const GV_KnowledgeGraph *kg, uint64_t object,
                     const char *predicate, GV_KGTriple *out, size_t max_count);

/**
 * @brief Save the knowledge graph to a binary file (magic "GVKG").
 *
 * @param kg    Knowledge graph handle.
 * @param path  File path.
 * @return 0 on success, -1 on failure.
 */
int kg_save(const GV_KnowledgeGraph *kg, const char *path);

/**
 * @brief Load a knowledge graph from a binary file.
 *
 * @param path  File path.
 * @return Loaded knowledge graph handle, or NULL on failure.
 */
GV_KnowledgeGraph *kg_load(const char *path);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_KNOWLEDGE_GRAPH_H */
