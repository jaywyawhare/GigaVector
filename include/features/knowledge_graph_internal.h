/**
  * @file knowledge_graph_internal.h
  * @brief Internal knowledge-graph storage structs, WAL macros and cross-file prototypes.
  *
  * Shared between knowledge_graph.c (core), knowledge_graph_query.c and
  * knowledge_graph_persist.c. Not a public API.
  */
#ifndef GV_FEATURES_KNOWLEDGE_GRAPH_INTERNAL_H
#define GV_FEATURES_KNOWLEDGE_GRAPH_INTERNAL_H

#include "features/knowledge_graph.h"
#include "core/types.h"
#include "core/id_bitmap.h"
#include "storage/database.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <pthread.h>

#define KG_MAGIC         "GVKG"
#define KG_MAGIC_LEN     4
#define KG_VERSION       1
#define KG_INITIAL_IDX   64   /* initial capacity for index lists */

typedef struct {
    uint64_t *ids;
    size_t    count;
    size_t    capacity;
} KG_IdList;

struct KG_RelationNode; /* fwd decl for adjacency edges */

/*
 * A materialised adjacency edge. Each edge stores *raw pointers* to the
 * relation node and to the entity node at the far end, so following a hop is a
 * pure pointer dereference - no hashing, no id->node lookup, no triple copy.
 *
 * The referenced nodes live in the entity/relation hash tables, which use
 * separate-chaining: nodes are heap-allocated once and never moved for the
 * lifetime of the graph, so these pointers stay valid until the node is
 * removed. Every code path that frees a relation node also unlinks the edges
 * that reference it (kg_unlink_adjacency), so a dangling edge is never observed.
 */
typedef struct KG_Edge {
    struct KG_RelationNode *rel;      /* the relation (edge) */
    struct KG_EntityNode   *neighbor; /* entity at the far end of the edge */
} KG_Edge;

typedef struct KG_EntityNode {
    GV_KGEntity             entity;
    struct KG_EntityNode   *next;

    /* Direct adjacency: O(1)-per-hop traversal via pointer dereference.
     * out_edges: relations where this node is the subject (this -> neighbor).
     * in_edges:  relations where this node is the object  (neighbor -> this). */
    KG_Edge                *out_edges;
    size_t                  out_count;
    size_t                  out_cap;
    KG_Edge                *in_edges;
    size_t                  in_count;
    size_t                  in_cap;
} KG_EntityNode;

typedef struct KG_RelationNode {
    GV_KGRelation             relation;
    struct KG_RelationNode   *next;
} KG_RelationNode;

typedef struct KG_IndexEntry {
    uint64_t               key;     /* entity_id or hash(predicate) */
    KG_IdList              list;
    struct KG_IndexEntry  *next;
} KG_IndexEntry;

struct GV_KnowledgeGraph {
    GV_KGConfig config;

    KG_EntityNode  **entity_buckets;
    size_t           entity_bucket_count;
    size_t           entity_count;
    uint64_t         next_entity_id;

    KG_RelationNode **relation_buckets;
    size_t            relation_bucket_count;
    size_t            relation_count;
    uint64_t          next_relation_id;

    /* SPO indexes: subject_index[subject_id] -> list of relation_ids
     *              object_index[object_id]   -> list of relation_ids
     *              predicate_index[hash]     -> list of relation_ids */
    KG_IndexEntry **subject_index;
    KG_IndexEntry **object_index;
    KG_IndexEntry **predicate_index;
    KG_IndexEntry **chunk_index;     /* hash(chunk_id) -> relation_ids */
    KG_IndexEntry **name_index;      /* hash(name) -> entity_ids */
    KG_IndexEntry **type_index;      /* hash(type) -> entity_ids */
    size_t          spo_bucket_count;

    float     *all_embeddings;       /* dim * embedding_cap floats */
    uint64_t  *embedding_entity_ids;
    size_t     embedding_count;
    size_t     embedding_cap;

    GV_Database *vdb;                /* optional: resolve entity similarity via db_search */

    /* Write-ahead log: mutations are appended + fsync'd before the mutating
     * call returns; kg_load replays the log over the last snapshot. */
    FILE *wal_file;
    char *wal_path;
    int   wal_replaying;

    pthread_rwlock_t rwlock;
};

/* ── WAL record format (little-endian, framed via gv_wal_codec.h) ────────
 *   1 ADD_ENTITY      u64 id, str name, str type, u8 has_emb, u32 dim,
 *                     dim*float (only when has_emb)
 *   2 REMOVE_ENTITY   u64 id
 *   3 ADD_RELATION    u64 id, u64 subject, str predicate, u64 object, f32 weight
 *   4 REMOVE_RELATION u64 id
 *   5 SET_ENTITY_PROP u64 id, str key, str value
 *   6 SET_RELATION_PROP u64 id, str key, str value */
#define KG_WAL_OP_ADD_ENTITY      1
#define KG_WAL_OP_REMOVE_ENTITY   2
#define KG_WAL_OP_ADD_RELATION    3
#define KG_WAL_OP_REMOVE_RELATION 4
#define KG_WAL_OP_SET_ENTITY_PROP 5
#define KG_WAL_OP_SET_RELATION_PROP 6
#define KG_WAL_OP_REMOVE_ENTITY_PROP 7
#define KG_WAL_OP_REMOVE_RELATION_PROP 8

/* --- Cross-file prototypes (split from knowledge_graph.c) --- */
/* Core storage/index/prop/embedding/WAL helpers kept in knowledge_graph.c,
 * used by knowledge_graph_query.c and knowledge_graph_persist.c. */
int kg_idlist_push(KG_IdList *list, uint64_t id);
int kg_embedding_add(GV_KnowledgeGraph *kg, uint64_t entity_id,
                             const float *emb, size_t dim);
const float *kg_embedding_get(const GV_KnowledgeGraph *kg,
                                      uint64_t entity_id, size_t *out_idx);
void kg_embedding_remove(GV_KnowledgeGraph *kg, uint64_t entity_id);
KG_EntityNode *kg_find_entity_node(const GV_KnowledgeGraph *kg,
                                           uint64_t entity_id);
KG_RelationNode *kg_find_relation_node(const GV_KnowledgeGraph *kg,
                                               uint64_t relation_id);
uint64_t kg_hash_string(const char *str);
uint64_t kg_hash_uint64(uint64_t key, size_t buckets);
KG_IndexEntry *kg_index_find(KG_IndexEntry **table,
                                     size_t buckets, uint64_t key);
KG_IndexEntry *kg_index_get_or_create(KG_IndexEntry **table,
                                              size_t buckets, uint64_t key);
void kg_index_remove_id(KG_IndexEntry **table, size_t buckets,
                                uint64_t key, uint64_t relation_id);
int kg_insert_entity_with_id(GV_KnowledgeGraph *kg, uint64_t eid,
                                    const char *name, const char *type,
                                    const float *embedding, size_t dimension);
int kg_insert_relation_with_id(GV_KnowledgeGraph *kg, uint64_t rid,
                                      uint64_t subject, const char *predicate,
                                      uint64_t object, float weight);
GV_KGProp *kg_prop_find(GV_KGProp *head, const char *key);
void kg_prop_free_list(GV_KGProp *head);
uint64_t kg_prop_hash(GV_PropValue v);
int kg_prop_remove(GV_KGProp **head, size_t *count, const char *key);
int kg_prop_set(GV_KGProp **head, size_t *count,
                       const char *key, const char *value);
void kg_rebuild_adjacency(GV_KnowledgeGraph *kg);
int kg_remove_entity_internal(GV_KnowledgeGraph *kg, uint64_t entity_id);
int kg_remove_relation_internal(GV_KnowledgeGraph *kg,
                                        uint64_t relation_id);
void kg_str_index_add(GV_KnowledgeGraph *kg, KG_IndexEntry **table,
                             size_t buckets, const char *s, uint64_t id);
void kg_wal_close(GV_KnowledgeGraph *kg);
size_t kg_collect_relations_for_entity(const GV_KnowledgeGraph *kg,
                                               uint64_t entity_id,
                                               uint64_t **out_ids);
void kg_entity_adjacency_free(KG_EntityNode *n);
void kg_entity_data_free(GV_KGEntity *e);
void kg_fill_triple(const GV_KnowledgeGraph *kg,
                            const GV_KGRelation *rel, GV_KGTriple *t);
/* Already external-linkage set helpers. */
GV_IdBitmap *kg_predicate_relation_set(const GV_KnowledgeGraph *kg, const char *predicate);
GV_IdBitmap *kg_entity_neighbor_set(const GV_KnowledgeGraph *kg, uint64_t entity_id,
                                    int direction);
#endif /* GV_FEATURES_KNOWLEDGE_GRAPH_INTERNAL_H */
