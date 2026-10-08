#define _POSIX_C_SOURCE 200112L

/**
 * @file knowledge_graph.c
 * @brief Knowledge graph implementation: entity/relation storage, SPO triple
 *        queries, cosine-similarity search, entity resolution, link prediction,
 *        BFS traversal, subgraph extraction, hybrid search, and persistence.
 */

#include "features/knowledge_graph.h"
#include "gv_wal_codec.h"
#include "core/memory.h"
#include "core/id_bitmap.h"
#include "core/utils.h"
#include "core/types.h"
#include "storage/database.h"
#include "search/distance.h"
#include "schema/vector.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <pthread.h>
#include <limits.h>
#ifndef _WIN32
#include <unistd.h>
#else
#include <io.h>
#define fsync(fd) _commit(fd)
#endif

#define KG_MAGIC         "GVKG"
#define KG_MAGIC_LEN     4
#define KG_VERSION       1
#define KG_INITIAL_IDX   64   /* initial capacity for index lists */

typedef struct {
    uint64_t *ids;
    size_t    count;
    size_t    capacity;
} KG_IdList;

static int kg_idlist_init(KG_IdList *list) {
    list->ids = (uint64_t *)gv_alloc(KG_INITIAL_IDX * sizeof(uint64_t));
    if (!list->ids) return -1;
    list->count = 0;
    list->capacity = KG_INITIAL_IDX;
    return 0;
}

static int kg_idlist_push(KG_IdList *list, uint64_t id) {
    if (list->count >= list->capacity) {
        size_t new_cap = list->capacity * 2;
        uint64_t *tmp = (uint64_t *)gv_realloc(list->ids,
                                             new_cap * sizeof(uint64_t));
        if (!tmp) return -1;
        list->ids = tmp;
        list->capacity = new_cap;
    }
    list->ids[list->count++] = id;
    return 0;
}

static void kg_idlist_remove(KG_IdList *list, uint64_t id) {
    for (size_t i = 0; i < list->count; i++) {
        if (list->ids[i] == id) {
            list->ids[i] = list->ids[list->count - 1];
            list->count--;
            return;
        }
    }
}

static void kg_idlist_free(KG_IdList *list) {
    gv_free(list->ids);
    list->ids = NULL;
    list->count = 0;
    list->capacity = 0;
}

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

static void kg_wal_close(GV_KnowledgeGraph *kg)
{
    if (kg->wal_file) {
        fclose(kg->wal_file);
        kg->wal_file = NULL;
    }
    gv_free(kg->wal_path);
    kg->wal_path = NULL;
}

/* Append+fsync one record. Called with the write lock held or during
 * single-threaded replay. Returns 0 on success. */
static int kg_wal_append(GV_KnowledgeGraph *kg, const GV_WalBuf *b)
{
    if (!kg->wal_file || kg->wal_replaying) return 0;
    return gv_wal_append_frame(kg->wal_file, b);
}

static uint64_t kg_hash_uint64(uint64_t key, size_t buckets) {
    /* Splitmix-style finaliser */
    key ^= key >> 30;
    key *= 0xbf58476d1ce4e5b9ULL;
    key ^= key >> 27;
    key *= 0x94d049bb133111ebULL;
    key ^= key >> 31;
    return buckets ? key % buckets : 0;
}

static uint64_t kg_hash_string(const char *str) {
    uint64_t h = 14695981039346656037ULL; /* FNV-1a offset basis */
    while (*str) {
        h ^= (uint64_t)(unsigned char)(*str++);
        h *= 1099511628211ULL; /* FNV-1a prime */
    }
    return h;
}

/** @brief Hash of a property value's string form (used for chunk-index keying). */
static uint64_t kg_prop_hash(GV_PropValue v) {
    char *s = gv_prop_to_string(v);
    uint64_t h = kg_hash_string(s ? s : "");
    gv_free(s);
    return h;
}

/** @brief True when a property value's string form equals @p s. */
static int kg_prop_str_eq(GV_PropValue v, const char *s) {
    char *ps = gv_prop_to_string(v);
    int eq = ps && s && strcmp(ps, s) == 0;
    gv_free(ps);
    return eq;
}

static uint64_t kg_now_epoch(void) {
    return (uint64_t)time(NULL);
}

static int kg_prop_remove(GV_KGProp **head, size_t *count, const char *key) {
    GV_KGProp *prev = NULL;
    for (GV_KGProp *p = *head; p; prev = p, p = p->next) {
        if (strcmp(p->key, key) == 0) {
            if (prev) prev->next = p->next;
            else *head = p->next;
            gv_free(p->key);
            gv_prop_free(&p->value);
            gv_free(p);
            (*count)--;
            return 0;
        }
    }
    return -1;
}

static void kg_prop_free_list(GV_KGProp *head) {
    while (head) {
        GV_KGProp *next = head->next;
        gv_free(head->key);
        gv_prop_free(&head->value);
        gv_free(head);
        head = next;
    }
}

static GV_KGProp *kg_prop_find(GV_KGProp *head, const char *key) {
    for (GV_KGProp *p = head; p; p = p->next) {
        if (p->key && strcmp(p->key, key) == 0) return p;
    }
    return NULL;
}

static int kg_prop_set(GV_KGProp **head, size_t *count,
                       const char *key, const char *value) {
    GV_KGProp *existing = kg_prop_find(*head, key);
    if (existing) {
        GV_PropValue newval = gv_prop_parse(value, GV_PROP_STRING);
        gv_prop_free(&existing->value);
        existing->value = newval;
        return 0;
    }
    GV_KGProp *node = (GV_KGProp *)gv_calloc(1, sizeof(GV_KGProp));
    if (!node) return -1;
    node->key = gv_dup_cstr(key);
    node->value = gv_prop_string(value);
    if (!node->key || node->value.type == GV_PROP_NULL) {
        gv_free(node->key);
        gv_prop_free(&node->value);
        gv_free(node);
        return -1;
    }
    node->next = *head;
    *head = node;
    (*count)++;
    return 0;
}



static KG_EntityNode *kg_find_entity_node(const GV_KnowledgeGraph *kg,
                                           uint64_t entity_id) {
    size_t idx = (size_t)kg_hash_uint64(entity_id, kg->entity_bucket_count);
    for (KG_EntityNode *n = kg->entity_buckets[idx]; n; n = n->next) {
        if (n->entity.entity_id == entity_id) return n;
    }
    return NULL;
}

static KG_RelationNode *kg_find_relation_node(const GV_KnowledgeGraph *kg,
                                               uint64_t relation_id) {
    size_t idx = (size_t)kg_hash_uint64(relation_id,
                                         kg->relation_bucket_count);
    for (KG_RelationNode *n = kg->relation_buckets[idx]; n; n = n->next) {
        if (n->relation.relation_id == relation_id) return n;
    }
    return NULL;
}

/* Direct adjacency (constant-time hops) */

static int kg_edge_push(KG_Edge **arr, size_t *count, size_t *cap,
                        KG_RelationNode *rel, KG_EntityNode *neighbor) {
    if (*count >= *cap) {
        size_t nc = (*cap == 0) ? 4 : (*cap * 2);
        KG_Edge *tmp = (KG_Edge *)gv_realloc(*arr, nc * sizeof(KG_Edge));
        if (!tmp) return -1;
        *arr = tmp;
        *cap = nc;
    }
    (*arr)[*count].rel = rel;
    (*arr)[*count].neighbor = neighbor;
    (*count)++;
    return 0;
}

static void kg_edge_remove_by_rel(KG_Edge *arr, size_t *count,
                                  const KG_RelationNode *rel) {
    for (size_t i = 0; i < *count; i++) {
        if (arr[i].rel == rel) {
            arr[i] = arr[*count - 1]; /* order is irrelevant; swap-remove is O(1) */
            (*count)--;
            return;
        }
    }
}

/* Link a freshly created relation into both endpoints' adjacency arrays.
 * subj/obj must be the entity nodes for rel->subject_id / rel->object_id.
 * A self-loop (subj == obj) is correctly recorded as both an out and in edge. */
static void kg_link_adjacency(KG_EntityNode *subj, KG_EntityNode *obj,
                              KG_RelationNode *rel) {
    if (!subj || !obj || !rel) return;
    kg_edge_push(&subj->out_edges, &subj->out_count, &subj->out_cap, rel, obj);
    kg_edge_push(&obj->in_edges, &obj->in_count, &obj->in_cap, rel, subj);
}

/* Unlink a relation from both endpoints' adjacency arrays before it is freed. */
static void kg_unlink_adjacency(const GV_KnowledgeGraph *kg,
                                const KG_RelationNode *rel) {
    KG_EntityNode *subj = kg_find_entity_node(kg, rel->relation.subject_id);
    KG_EntityNode *obj = kg_find_entity_node(kg, rel->relation.object_id);
    if (subj) kg_edge_remove_by_rel(subj->out_edges, &subj->out_count, rel);
    if (obj) kg_edge_remove_by_rel(obj->in_edges, &obj->in_count, rel);
}

static void kg_entity_adjacency_free(KG_EntityNode *n) {
    gv_free(n->out_edges);
    gv_free(n->in_edges);
    n->out_edges = n->in_edges = NULL;
    n->out_count = n->out_cap = 0;
    n->in_count = n->in_cap = 0;
}

/* Rebuild every node's adjacency from the relation table. O(entities + edges).
 * Used after bulk load and after structural rewrites (merge) where incremental
 * maintenance would be error-prone. */
static void kg_rebuild_adjacency(GV_KnowledgeGraph *kg) {
    for (size_t b = 0; b < kg->entity_bucket_count; b++) {
        for (KG_EntityNode *n = kg->entity_buckets[b]; n; n = n->next) {
            n->out_count = 0;
            n->in_count = 0;
        }
    }
    for (size_t b = 0; b < kg->relation_bucket_count; b++) {
        for (KG_RelationNode *rn = kg->relation_buckets[b]; rn; rn = rn->next) {
            KG_EntityNode *subj = kg_find_entity_node(kg,
                                      rn->relation.subject_id);
            KG_EntityNode *obj = kg_find_entity_node(kg,
                                     rn->relation.object_id);
            kg_link_adjacency(subj, obj, rn);
        }
    }
}

static KG_IndexEntry *kg_index_find(KG_IndexEntry **table,
                                     size_t buckets, uint64_t key) {
    size_t idx = (size_t)(key % buckets);
    for (KG_IndexEntry *e = table[idx]; e; e = e->next) {
        if (e->key == key) return e;
    }
    return NULL;
}

static KG_IndexEntry *kg_index_get_or_create(KG_IndexEntry **table,
                                              size_t buckets, uint64_t key) {
    size_t idx = (size_t)(key % buckets);
    for (KG_IndexEntry *e = table[idx]; e; e = e->next) {
        if (e->key == key) return e;
    }
    KG_IndexEntry *entry = (KG_IndexEntry *)gv_calloc(1, sizeof(KG_IndexEntry));
    if (!entry) return NULL;
    entry->key = key;
    if (kg_idlist_init(&entry->list) != 0) {
        gv_free(entry);
        return NULL;
    }
    entry->next = table[idx];
    table[idx] = entry;
    return entry;
}

static void kg_index_remove_id(KG_IndexEntry **table, size_t buckets,
                                uint64_t key, uint64_t relation_id) {
    KG_IndexEntry *e = kg_index_find(table, buckets, key);
    if (e) kg_idlist_remove(&e->list, relation_id);
}

/* hash(string)-keyed index maintenance shared by the name/type/chunk indexes */
static void kg_str_index_add(GV_KnowledgeGraph *kg, KG_IndexEntry **table,
                             size_t buckets, const char *s, uint64_t id) {
    (void)kg;
    if (!s) return;
    KG_IndexEntry *e = kg_index_get_or_create(table, buckets, kg_hash_string(s));
    if (e) kg_idlist_push(&e->list, id);
}

static void kg_str_index_remove(KG_IndexEntry **table, size_t buckets,
                                const char *s, uint64_t id) {
    if (!s) return;
    kg_index_remove_id(table, buckets, kg_hash_string(s), id);
}

static void kg_index_free_table(KG_IndexEntry **table, size_t buckets) {
    if (!table) return;
    for (size_t i = 0; i < buckets; i++) {
        KG_IndexEntry *e = table[i];
        while (e) {
            KG_IndexEntry *next = e->next;
            kg_idlist_free(&e->list);
            gv_free(e);
            e = next;
        }
    }
    gv_free(table);
}

static int kg_embedding_add(GV_KnowledgeGraph *kg, uint64_t entity_id,
                             const float *emb, size_t dim) {
    if (dim != kg->config.embedding_dimension) return -1;
    if (kg->embedding_count >= kg->embedding_cap) {
        size_t new_cap = kg->embedding_cap == 0 ? 256 : kg->embedding_cap * 2;
        float *new_emb = (float *)gv_realloc(kg->all_embeddings,
                                           new_cap * dim * sizeof(float));
        uint64_t *new_ids = (uint64_t *)gv_realloc(kg->embedding_entity_ids,
                                                  new_cap * sizeof(uint64_t));
        if (!new_emb || !new_ids) {
            /* Rollback on partial allocation */
            if (new_emb) kg->all_embeddings = new_emb;
            if (new_ids) kg->embedding_entity_ids = new_ids;
            return -1;
        }
        kg->all_embeddings = new_emb;
        kg->embedding_entity_ids = new_ids;
        kg->embedding_cap = new_cap;
    }
    size_t off = kg->embedding_count * dim;
    memcpy(kg->all_embeddings + off, emb, dim * sizeof(float));
    kg->embedding_entity_ids[kg->embedding_count] = entity_id;
    kg->embedding_count++;
    return 0;
}

static void kg_embedding_remove(GV_KnowledgeGraph *kg, uint64_t entity_id) {
    size_t dim = kg->config.embedding_dimension;
    for (size_t i = 0; i < kg->embedding_count; i++) {
        if (kg->embedding_entity_ids[i] == entity_id) {
            size_t last = kg->embedding_count - 1;
            if (i != last) {
                kg->embedding_entity_ids[i] = kg->embedding_entity_ids[last];
                memcpy(kg->all_embeddings + i * dim,
                       kg->all_embeddings + last * dim,
                       dim * sizeof(float));
            }
            kg->embedding_count--;
            return;
        }
    }
}

static const float *kg_embedding_get(const GV_KnowledgeGraph *kg,
                                      uint64_t entity_id, size_t *out_idx) {
    size_t dim = kg->config.embedding_dimension;
    for (size_t i = 0; i < kg->embedding_count; i++) {
        if (kg->embedding_entity_ids[i] == entity_id) {
            if (out_idx) *out_idx = i;
            return kg->all_embeddings + i * dim;
        }
    }
    return NULL;
}

static void kg_entity_data_free(GV_KGEntity *e) {
    gv_free(e->name);
    gv_free(e->type);
    gv_free(e->embedding);
    kg_prop_free_list(e->properties);
    e->name = NULL;
    e->type = NULL;
    e->embedding = NULL;
    e->properties = NULL;
}

static void kg_relation_data_free(GV_KGRelation *r) {
    gv_free(r->predicate);
    kg_prop_free_list(r->properties);
    r->predicate = NULL;
    r->properties = NULL;
}

static size_t kg_collect_relations_for_entity(const GV_KnowledgeGraph *kg,
                                               uint64_t entity_id,
                                               uint64_t **out_ids) {
    size_t total = 0;
    size_t cap = 64;
    uint64_t *ids = (uint64_t *)gv_alloc(cap * sizeof(uint64_t));
    if (!ids) { *out_ids = NULL; return 0; }

    KG_IndexEntry *se = kg_index_find(kg->subject_index,
                                       kg->spo_bucket_count, entity_id);
    if (se) {
        for (size_t i = 0; i < se->list.count; i++) {
            if (total >= cap) {
                size_t new_cap = cap * 2;
                uint64_t *tmp = (uint64_t *)gv_realloc(ids,
                                                      new_cap * sizeof(uint64_t));
                if (!tmp) break;
                ids = tmp;
                cap = new_cap;  /* only after success; else a later block writes OOB */
            }
            ids[total++] = se->list.ids[i];
        }
    }

    KG_IndexEntry *oe = kg_index_find(kg->object_index,
                                       kg->spo_bucket_count, entity_id);
    if (oe) {
        for (size_t i = 0; i < oe->list.count; i++) {
            int dup = 0;
            for (size_t j = 0; j < total; j++) {
                if (ids[j] == oe->list.ids[i]) { dup = 1; break; }
            }
            if (dup) continue;
            if (total >= cap) {
                size_t new_cap = cap * 2;
                uint64_t *tmp = (uint64_t *)gv_realloc(ids,
                                                      new_cap * sizeof(uint64_t));
                if (!tmp) break;
                ids = tmp;
                cap = new_cap;  /* only after success; else a later block writes OOB */
            }
            ids[total++] = oe->list.ids[i];
        }
    }

    *out_ids = ids;
    return total;
}

/* Dgraph-style typed roaring sets:
 * Materialize typed relation / neighbor sets as roaring bitmaps (id_bitmap.h)
 * so typed queries compose as set operations - the Dgraph model where a
 * <predicate> is a UID set and a typed join is a bitmap AND. Callers own the
 * returned bitmap (free with gv_id_bitmap_free) and combine several with
 * gv_id_bitmap_and / gv_id_bitmap_or. */

GV_IdBitmap *kg_predicate_relation_set(const GV_KnowledgeGraph *kg, const char *predicate) {
    if (!kg || !predicate) return NULL;
    GV_IdBitmap *bm = gv_id_bitmap_create();
    if (!bm) return NULL;
    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);
    uint64_t ph = kg_hash_string(predicate);
    KG_IndexEntry *e = kg_index_find(kg->predicate_index, kg->spo_bucket_count, ph);
    if (e) {
        for (size_t i = 0; i < e->list.count; i++)
            gv_id_bitmap_add(bm, e->list.ids[i]);
    }
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return bm;
}

GV_IdBitmap *kg_entity_neighbor_set(const GV_KnowledgeGraph *kg, uint64_t entity_id,
                                    int direction) {
    if (!kg) return NULL;
    GV_IdBitmap *bm = gv_id_bitmap_create();
    if (!bm) return NULL;
    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);
    KG_EntityNode *n = kg_find_entity_node(kg, entity_id);
    if (n) {
        if (direction == 0 || direction == 2) {  /* outgoing */
            for (size_t i = 0; i < n->out_count; i++)
                if (n->out_edges[i].neighbor)
                    gv_id_bitmap_add(bm, n->out_edges[i].neighbor->entity.entity_id);
        }
        if (direction == 1 || direction == 2) {  /* incoming */
            for (size_t i = 0; i < n->in_count; i++)
                if (n->in_edges[i].neighbor)
                    gv_id_bitmap_add(bm, n->in_edges[i].neighbor->entity.entity_id);
        }
    }
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return bm;
}

static int kg_remove_relation_internal(GV_KnowledgeGraph *kg,
                                        uint64_t relation_id) {
    size_t idx = (size_t)kg_hash_uint64(relation_id,
                                         kg->relation_bucket_count);
    KG_RelationNode *prev = NULL;
    for (KG_RelationNode *n = kg->relation_buckets[idx]; n; n = n->next) {
        if (n->relation.relation_id == relation_id) {
            kg_index_remove_id(kg->subject_index, kg->spo_bucket_count,
                               n->relation.subject_id, relation_id);
            kg_index_remove_id(kg->object_index, kg->spo_bucket_count,
                               n->relation.object_id, relation_id);
            uint64_t pred_hash = kg_hash_string(n->relation.predicate);
            kg_index_remove_id(kg->predicate_index, kg->spo_bucket_count,
                               pred_hash, relation_id);

            GV_KGProp *cp = kg_prop_find(n->relation.properties, "chunk_id");
            if (cp && cp->value.type == GV_PROP_STRING && cp->value.as.s) {
                kg_index_remove_id(kg->chunk_index, kg->spo_bucket_count,
                                   kg_hash_string(cp->value.as.s), relation_id);
            }

            /* Drop the edge from both endpoints' adjacency before freeing it. */
            kg_unlink_adjacency(kg, n);

            if (prev) prev->next = n->next;
            else kg->relation_buckets[idx] = n->next;
            kg_relation_data_free(&n->relation);
            gv_free(n);
            kg->relation_count--;
            return 0;
        }
        prev = n;
    }
    return -1;
}

static int kg_are_connected(const GV_KnowledgeGraph *kg,
                             uint64_t a, uint64_t b) {
    KG_IndexEntry *se = kg_index_find(kg->subject_index,
                                       kg->spo_bucket_count, a);
    if (se) {
        for (size_t i = 0; i < se->list.count; i++) {
            KG_RelationNode *rn = kg_find_relation_node(kg, se->list.ids[i]);
            if (rn && rn->relation.object_id == b) return 1;
        }
    }
    KG_IndexEntry *oe = kg_index_find(kg->object_index,
                                       kg->spo_bucket_count, a);
    if (oe) {
        for (size_t i = 0; i < oe->list.count; i++) {
            KG_RelationNode *rn = kg_find_relation_node(kg, oe->list.ids[i]);
            if (rn && rn->relation.subject_id == b) return 1;
        }
    }
    return 0;
}

static size_t kg_shared_neighbors(const GV_KnowledgeGraph *kg,
                                   uint64_t a, uint64_t b) {
    uint64_t *na = NULL, *nb = NULL;
    size_t ca = kg_collect_relations_for_entity(kg, a, &na);
    size_t cb = kg_collect_relations_for_entity(kg, b, &nb);

    size_t na_cap = 64, nb_cap = 64;
    uint64_t *neigh_a = (uint64_t *)gv_alloc(na_cap * sizeof(uint64_t));
    uint64_t *neigh_b = (uint64_t *)gv_alloc(nb_cap * sizeof(uint64_t));
    size_t neigh_a_count = 0, neigh_b_count = 0;

    if (!neigh_a || !neigh_b) {
        gv_free(na); gv_free(nb); gv_free(neigh_a); gv_free(neigh_b);
        return 0;
    }

    for (size_t i = 0; i < ca; i++) {
        KG_RelationNode *rn = kg_find_relation_node(kg, na[i]);
        if (!rn) continue;
        uint64_t other = (rn->relation.subject_id == a) ?
                          rn->relation.object_id : rn->relation.subject_id;
        if (neigh_a_count >= na_cap) {
            na_cap *= 2;
            uint64_t *tmp = (uint64_t *)gv_realloc(neigh_a,
                                                  na_cap * sizeof(uint64_t));
            if (!tmp) break;
            neigh_a = tmp;
        }
        neigh_a[neigh_a_count++] = other;
    }

    for (size_t i = 0; i < cb; i++) {
        KG_RelationNode *rn = kg_find_relation_node(kg, nb[i]);
        if (!rn) continue;
        uint64_t other = (rn->relation.subject_id == b) ?
                          rn->relation.object_id : rn->relation.subject_id;
        if (neigh_b_count >= nb_cap) {
            nb_cap *= 2;
            uint64_t *tmp = (uint64_t *)gv_realloc(neigh_b,
                                                  nb_cap * sizeof(uint64_t));
            if (!tmp) break;
            neigh_b = tmp;
        }
        neigh_b[neigh_b_count++] = other;
    }

    size_t shared = 0;
    for (size_t i = 0; i < neigh_a_count; i++) {
        for (size_t j = 0; j < neigh_b_count; j++) {
            if (neigh_a[i] == neigh_b[j]) { shared++; break; }
        }
    }

    gv_free(na); gv_free(nb);
    gv_free(neigh_a); gv_free(neigh_b);
    return shared;
}

/*
 * Internal: BFS helper (used by traverse, shortest_path, subgraph)
 *
 * Returns number of entities discovered.  visited[] holds discovered IDs,
 * depths[] holds their BFS depth.  parent[] holds predecessor entity_id
 * (0 = root / no parent).
 */

typedef struct {
    uint64_t *visited;
    size_t   *depths;
    uint64_t *parent;
    size_t    count;
    size_t    cap;
} KG_BFSState;

static int kg_bfs_init(KG_BFSState *bfs, size_t cap) {
    bfs->visited = (uint64_t *)gv_alloc(cap * sizeof(uint64_t));
    bfs->depths  = (size_t *)gv_alloc(cap * sizeof(size_t));
    bfs->parent  = (uint64_t *)gv_calloc(cap, sizeof(uint64_t));
    if (!bfs->visited || !bfs->depths || !bfs->parent) {
        gv_free(bfs->visited); gv_free(bfs->depths); gv_free(bfs->parent);
        return -1;
    }
    bfs->count = 0;
    bfs->cap = cap;
    return 0;
}

static int kg_bfs_seen(const KG_BFSState *bfs, uint64_t id) {
    for (size_t i = 0; i < bfs->count; i++) {
        if (bfs->visited[i] == id) return 1;
    }
    return 0;
}

static int kg_bfs_push(KG_BFSState *bfs, uint64_t id, size_t depth,
                        uint64_t parent) {
    if (bfs->count >= bfs->cap) {
        size_t new_cap = bfs->cap * 2;
        uint64_t *v = (uint64_t *)gv_realloc(bfs->visited,
                                            new_cap * sizeof(uint64_t));
        size_t *d = (size_t *)gv_realloc(bfs->depths,
                                       new_cap * sizeof(size_t));
        uint64_t *p = (uint64_t *)gv_realloc(bfs->parent,
                                            new_cap * sizeof(uint64_t));
        if (!v || !d || !p) {
            if (v) bfs->visited = v;
            if (d) bfs->depths = d;
            if (p) bfs->parent = p;
            return -1;
        }
        bfs->visited = v;
        bfs->depths = d;
        bfs->parent = p;
        bfs->cap = new_cap;
    }
    bfs->visited[bfs->count] = id;
    bfs->depths[bfs->count] = depth;
    bfs->parent[bfs->count] = parent;
    bfs->count++;
    return 0;
}

static void kg_bfs_free(KG_BFSState *bfs) {
    gv_free(bfs->visited);
    gv_free(bfs->depths);
    gv_free(bfs->parent);
    bfs->visited = NULL;
    bfs->depths = NULL;
    bfs->parent = NULL;
    bfs->count = 0;
}

static void kg_bfs_run(const GV_KnowledgeGraph *kg, uint64_t start,
                        size_t max_depth, KG_BFSState *bfs) {
    kg_bfs_push(bfs, start, 0, 0);
    size_t front = 0;

    while (front < bfs->count) {
        uint64_t cur = bfs->visited[front];
        size_t   dep = bfs->depths[front];
        front++;

        if (dep >= max_depth) continue;

        KG_IndexEntry *se = kg_index_find(kg->subject_index,
                                           kg->spo_bucket_count, cur);
        if (se) {
            for (size_t i = 0; i < se->list.count; i++) {
                KG_RelationNode *rn = kg_find_relation_node(kg,
                                                             se->list.ids[i]);
                if (!rn) continue;
                uint64_t nbr = rn->relation.object_id;
                if (!kg_bfs_seen(bfs, nbr)) {
                    kg_bfs_push(bfs, nbr, dep + 1, cur);
                }
            }
        }

        KG_IndexEntry *oe = kg_index_find(kg->object_index,
                                           kg->spo_bucket_count, cur);
        if (oe) {
            for (size_t i = 0; i < oe->list.count; i++) {
                KG_RelationNode *rn = kg_find_relation_node(kg,
                                                             oe->list.ids[i]);
                if (!rn) continue;
                uint64_t nbr = rn->relation.subject_id;
                if (!kg_bfs_seen(bfs, nbr)) {
                    kg_bfs_push(bfs, nbr, dep + 1, cur);
                }
            }
        }
    }
}

/*
 * Internal: entity_has_predicate - check if entity participates in
 *           at least one relation with the given predicate
 */

static int kg_entity_has_predicate(const GV_KnowledgeGraph *kg,
                                    uint64_t entity_id,
                                    const char *predicate) {
    KG_IndexEntry *se = kg_index_find(kg->subject_index,
                                       kg->spo_bucket_count, entity_id);
    if (se) {
        for (size_t i = 0; i < se->list.count; i++) {
            KG_RelationNode *rn = kg_find_relation_node(kg, se->list.ids[i]);
            if (rn && strcmp(rn->relation.predicate, predicate) == 0)
                return 1;
        }
    }
    KG_IndexEntry *oe = kg_index_find(kg->object_index,
                                       kg->spo_bucket_count, entity_id);
    if (oe) {
        for (size_t i = 0; i < oe->list.count; i++) {
            KG_RelationNode *rn = kg_find_relation_node(kg, oe->list.ids[i]);
            if (rn && strcmp(rn->relation.predicate, predicate) == 0)
                return 1;
        }
    }
    return 0;
}

typedef struct {
    uint64_t id;
    float    score;
} KG_ScorePair;

static int kg_score_cmp_desc(const void *a, const void *b) {
    float sa = ((const KG_ScorePair *)a)->score;
    float sb = ((const KG_ScorePair *)b)->score;
    if (sa > sb) return -1;
    if (sa < sb) return  1;
    return 0;
}

void kg_config_init(GV_KGConfig *config) {
    if (!config) return;
    config->entity_bucket_count     = 4096;
    config->relation_bucket_count   = 8192;
    config->embedding_dimension     = 128;
    config->similarity_threshold    = 0.7f;
    config->link_prediction_threshold = 0.8f;
    config->max_entities            = 1000000;
}

GV_KnowledgeGraph *kg_create(const GV_KGConfig *config) {
    GV_KGConfig cfg;
    if (config) {
        cfg = *config;
    } else {
        kg_config_init(&cfg);
    }

    GV_KnowledgeGraph *kg = (GV_KnowledgeGraph *)gv_calloc(1,
                                sizeof(GV_KnowledgeGraph));
    if (!kg) return NULL;

    kg->config = cfg;
    kg->entity_bucket_count  = cfg.entity_bucket_count;
    kg->relation_bucket_count = cfg.relation_bucket_count;
    kg->spo_bucket_count = cfg.relation_bucket_count;
    kg->next_entity_id = 1;
    kg->next_relation_id = 1;

    kg->entity_buckets = (KG_EntityNode **)gv_calloc(kg->entity_bucket_count,
                                                   sizeof(KG_EntityNode *));
    if (!kg->entity_buckets) goto fail;

    kg->relation_buckets = (KG_RelationNode **)gv_calloc(
        kg->relation_bucket_count, sizeof(KG_RelationNode *));
    if (!kg->relation_buckets) goto fail;

    kg->subject_index = (KG_IndexEntry **)gv_calloc(kg->spo_bucket_count,
                                                  sizeof(KG_IndexEntry *));
    kg->object_index = (KG_IndexEntry **)gv_calloc(kg->spo_bucket_count,
                                                 sizeof(KG_IndexEntry *));
    kg->predicate_index = (KG_IndexEntry **)gv_calloc(kg->spo_bucket_count,
                                                    sizeof(KG_IndexEntry *));
    kg->chunk_index = (KG_IndexEntry **)gv_calloc(kg->spo_bucket_count,
                                                  sizeof(KG_IndexEntry *));
    kg->name_index = (KG_IndexEntry **)gv_calloc(kg->entity_bucket_count,
                                                  sizeof(KG_IndexEntry *));
    kg->type_index = (KG_IndexEntry **)gv_calloc(kg->entity_bucket_count,
                                                  sizeof(KG_IndexEntry *));
    if (!kg->subject_index || !kg->object_index || !kg->predicate_index ||
        !kg->chunk_index || !kg->name_index || !kg->type_index)
        goto fail;

    if (pthread_rwlock_init(&kg->rwlock, NULL) != 0) goto fail;

    return kg;

fail:
    gv_free(kg->entity_buckets);
    gv_free(kg->relation_buckets);
    gv_free(kg->subject_index);
    gv_free(kg->object_index);
    gv_free(kg->predicate_index);
    gv_free(kg->chunk_index);
    gv_free(kg->name_index);
    gv_free(kg->type_index);
    gv_free(kg);
    return NULL;
}

void kg_destroy(GV_KnowledgeGraph *kg) {
    if (!kg) return;

    for (size_t i = 0; i < kg->entity_bucket_count; i++) {
        KG_EntityNode *n = kg->entity_buckets[i];
        while (n) {
            KG_EntityNode *next = n->next;
            kg_entity_adjacency_free(n);
            kg_entity_data_free(&n->entity);
            gv_free(n);
            n = next;
        }
    }
    gv_free(kg->entity_buckets);

    for (size_t i = 0; i < kg->relation_bucket_count; i++) {
        KG_RelationNode *n = kg->relation_buckets[i];
        while (n) {
            KG_RelationNode *next = n->next;
            kg_relation_data_free(&n->relation);
            gv_free(n);
            n = next;
        }
    }
    gv_free(kg->relation_buckets);

    kg_index_free_table(kg->subject_index, kg->spo_bucket_count);
    kg_index_free_table(kg->object_index, kg->spo_bucket_count);
    kg_index_free_table(kg->predicate_index, kg->spo_bucket_count);
    kg_index_free_table(kg->chunk_index, kg->spo_bucket_count);
    kg_index_free_table(kg->name_index, kg->entity_bucket_count);
    kg_index_free_table(kg->type_index, kg->entity_bucket_count);

    gv_free(kg->all_embeddings);
    gv_free(kg->embedding_entity_ids);

    kg_wal_close(kg);

    pthread_rwlock_destroy(&kg->rwlock);
    gv_free(kg);
}

/* Insert an entity with an explicit id (WAL replay). Caller holds the write
 * lock. Unlike kg_add_entity this does not mirror into an attached vector DB -
 * the vector DB has its own durability story. Returns 0 on success. */
static int kg_insert_entity_with_id(GV_KnowledgeGraph *kg, uint64_t eid,
                                    const char *name, const char *type,
                                    const float *embedding, size_t dimension) {
    if (!name || !type) return -1;
    KG_EntityNode *node = (KG_EntityNode *)gv_calloc(1, sizeof(KG_EntityNode));
    if (!node) return -1;

    GV_KGEntity *e = &node->entity;
    e->entity_id  = eid;
    e->name       = gv_dup_cstr(name);
    e->type       = gv_dup_cstr(type);
    e->created_at = kg_now_epoch();
    e->confidence = 1.0f;
    if (!e->name || !e->type) {
        gv_free(e->name);
        gv_free(e->type);
        gv_free(node);
        return -1;
    }

    if (embedding && dimension > 0 && kg->config.embedding_dimension > 0) {
        e->embedding = (float *)gv_alloc(dimension * sizeof(float));
        if (e->embedding) {
            memcpy(e->embedding, embedding, dimension * sizeof(float));
            e->dimension = dimension;
            kg_embedding_add(kg, eid, embedding, dimension);
        }
    }

    size_t bucket = (size_t)kg_hash_uint64(eid, kg->entity_bucket_count);
    node->next = kg->entity_buckets[bucket];
    kg->entity_buckets[bucket] = node;
    kg->entity_count++;

    kg_str_index_add(kg, kg->name_index, kg->entity_bucket_count, e->name, eid);
    kg_str_index_add(kg, kg->type_index, kg->entity_bucket_count, e->type, eid);

    if (eid >= kg->next_entity_id) kg->next_entity_id = eid + 1;
    return 0;
}

uint64_t kg_add_entity(GV_KnowledgeGraph *kg, const char *name,
                           const char *type, const float *embedding,
                           size_t dimension) {
    if (!kg || !name || !type) return 0;

    pthread_rwlock_wrlock(&kg->rwlock);

    if (kg->entity_count >= kg->config.max_entities) {
        pthread_rwlock_unlock(&kg->rwlock);
        return 0;
    }

    uint64_t eid = kg->next_entity_id++;

    if (kg_insert_entity_with_id(kg, eid, name, type, embedding, dimension) != 0) {
        /* id consumed; bump past it so it is never reused after failure */
        pthread_rwlock_unlock(&kg->rwlock);
        return 0;
    }

    /* Vectors-as-a-predicate: if a vector DB is attached, mirror the embedding
     * there under "kgent:{id}" so similarity resolves via db_search. */
    if (kg->vdb != NULL && embedding && dimension > 0 &&
        kg->config.embedding_dimension > 0) {
        char kid[48];
        snprintf(kid, sizeof(kid), "kgent:%llu", (unsigned long long)eid);
        db_add_vector_with_id(kg->vdb, kid, embedding, dimension);
    }

    {
        GV_WalBuf b;
        gv_wal_init(&b);
        gv_wal_put_u8(&b, KG_WAL_OP_ADD_ENTITY);
        gv_wal_put_u64(&b, eid);
        gv_wal_put_str(&b, name);
        gv_wal_put_str(&b, type);
        uint8_t has_emb = (embedding && dimension > 0 &&
                           kg->config.embedding_dimension > 0) ? 1 : 0;
        gv_wal_put_u8(&b, has_emb);
        if (has_emb) {
            gv_wal_put_u32(&b, (uint32_t)dimension);
            gv_wal_put_bytes(&b, embedding, dimension * sizeof(float));
        }
        kg_wal_append(kg, &b);
        gv_wal_free(&b);
    }

    pthread_rwlock_unlock(&kg->rwlock);
    return eid;
}

static int kg_remove_entity_internal(GV_KnowledgeGraph *kg, uint64_t entity_id) {
    uint64_t *rel_ids = NULL;
    size_t rel_count = kg_collect_relations_for_entity(kg, entity_id,
                                                        &rel_ids);
    for (size_t i = 0; i < rel_count; i++) {
        kg_remove_relation_internal(kg, rel_ids[i]);
    }
    gv_free(rel_ids);

    kg_embedding_remove(kg, entity_id);

    size_t bucket = (size_t)kg_hash_uint64(entity_id,
                                            kg->entity_bucket_count);
    KG_EntityNode *prev = NULL;
    for (KG_EntityNode *n = kg->entity_buckets[bucket]; n; n = n->next) {
        if (n->entity.entity_id == entity_id) {
            kg_str_index_remove(kg->name_index, kg->entity_bucket_count,
                                n->entity.name, entity_id);
            kg_str_index_remove(kg->type_index, kg->entity_bucket_count,
                                n->entity.type, entity_id);
            if (prev) prev->next = n->next;
            else kg->entity_buckets[bucket] = n->next;
            kg_entity_adjacency_free(n);
            kg_entity_data_free(&n->entity);
            gv_free(n);
            kg->entity_count--;
            return 0;
        }
        prev = n;
    }
    return -1;
}

int kg_remove_entity(GV_KnowledgeGraph *kg, uint64_t entity_id) {
    if (!kg) return -1;

    pthread_rwlock_wrlock(&kg->rwlock);

    int rc = kg_remove_entity_internal(kg, entity_id);
    if (rc == 0) {
        GV_WalBuf b;
        gv_wal_init(&b);
        gv_wal_put_u8(&b, KG_WAL_OP_REMOVE_ENTITY);
        gv_wal_put_u64(&b, entity_id);
        kg_wal_append(kg, &b);
        gv_wal_free(&b);
    }

    pthread_rwlock_unlock(&kg->rwlock);
    return rc;
}

const GV_KGEntity *kg_get_entity(const GV_KnowledgeGraph *kg,
                                     uint64_t entity_id) {
    if (!kg) return NULL;
    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);
    KG_EntityNode *n = kg_find_entity_node(kg, entity_id);
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return n ? &n->entity : NULL;
}

static int kg_remove_prop_internal(GV_KGProp **head, size_t *count,
                                   const char *key) {
    return kg_prop_remove(head, count, key);
}

int kg_remove_entity_prop(GV_KnowledgeGraph *kg, uint64_t entity_id,
                          const char *key) {
    if (!kg || !key) return -1;
    pthread_rwlock_wrlock(&kg->rwlock);
    KG_EntityNode *n = kg_find_entity_node(kg, entity_id);
    if (!n) { pthread_rwlock_unlock(&kg->rwlock); return -1; }
    int rc = kg_remove_prop_internal(&n->entity.properties,
                                     &n->entity.prop_count, key);
    if (rc == 0) {
        GV_WalBuf b;
        gv_wal_init(&b);
        gv_wal_put_u8(&b, KG_WAL_OP_REMOVE_ENTITY_PROP);
        gv_wal_put_u64(&b, entity_id);
        gv_wal_put_str(&b, key);
        kg_wal_append(kg, &b);
        gv_wal_free(&b);
    }
    pthread_rwlock_unlock(&kg->rwlock);
    return rc;
}

int kg_remove_relation_prop(GV_KnowledgeGraph *kg, uint64_t relation_id,
                            const char *key) {
    if (!kg || !key) return -1;
    pthread_rwlock_wrlock(&kg->rwlock);
    KG_RelationNode *n = kg_find_relation_node(kg, relation_id);
    if (!n) { pthread_rwlock_unlock(&kg->rwlock); return -1; }
    /* Keep the chunk index consistent when deleting a chunk_id facet. */
    int is_chunk_key = strcmp(key, "chunk_id") == 0;
    GV_KGProp *old = is_chunk_key
        ? kg_prop_find(n->relation.properties, "chunk_id") : NULL;
    if (old && old->value.type != GV_PROP_NULL) {
        kg_index_remove_id(kg->chunk_index, kg->spo_bucket_count,
                           kg_prop_hash(old->value), relation_id);
    }
    size_t dummy_count = 0;
    GV_KGProp *p = n->relation.properties;
    while (p) { dummy_count++; p = p->next; }
    int rc = kg_remove_prop_internal(&n->relation.properties, &dummy_count, key);
    if (rc == 0) {
        GV_WalBuf b;
        gv_wal_init(&b);
        gv_wal_put_u8(&b, KG_WAL_OP_REMOVE_RELATION_PROP);
        gv_wal_put_u64(&b, relation_id);
        gv_wal_put_str(&b, key);
        kg_wal_append(kg, &b);
        gv_wal_free(&b);
    }
    pthread_rwlock_unlock(&kg->rwlock);
    return rc;
}

int kg_set_entity_prop(GV_KnowledgeGraph *kg, uint64_t entity_id,
                           const char *key, const char *value) {
    if (!kg || !key || !value) return -1;

    pthread_rwlock_wrlock(&kg->rwlock);
    KG_EntityNode *n = kg_find_entity_node(kg, entity_id);
    if (!n) {
        pthread_rwlock_unlock(&kg->rwlock);
        return -1;
    }
    int rc = kg_prop_set(&n->entity.properties, &n->entity.prop_count,
                          key, value);
    if (rc == 0) {
        GV_WalBuf b;
        gv_wal_init(&b);
        gv_wal_put_u8(&b, KG_WAL_OP_SET_ENTITY_PROP);
        gv_wal_put_u64(&b, entity_id);
        gv_wal_put_str(&b, key);
        gv_wal_put_str(&b, value);
        kg_wal_append(kg, &b);
        gv_wal_free(&b);
    }
    pthread_rwlock_unlock(&kg->rwlock);
    return rc;
}

const char *kg_get_entity_prop(const GV_KnowledgeGraph *kg,
                                   uint64_t entity_id, const char *key) {
    if (!kg || !key) return NULL;
    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);
    KG_EntityNode *n = kg_find_entity_node(kg, entity_id);
    if (!n) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return NULL;
    }
    GV_KGProp *p = kg_prop_find(n->entity.properties, key);
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return p && p->value.type == GV_PROP_STRING ? p->value.as.s : NULL;
}

int kg_find_entities_by_type(const GV_KnowledgeGraph *kg, const char *type,
                                 uint64_t *out_ids, size_t max_count) {
    if (!kg || !type || !out_ids || max_count == 0) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);
    size_t found = 0;
    KG_IndexEntry *e = kg_index_find(kg->type_index,
                                     kg->entity_bucket_count,
                                     kg_hash_string(type));
    if (e) {
        for (size_t i = 0; i < e->list.count && found < max_count; i++) {
            KG_EntityNode *n = kg_find_entity_node(kg, e->list.ids[i]);
            /* Hash collision check: verify the type actually matches */
            if (n && n->entity.type && strcmp(n->entity.type, type) == 0) {
                out_ids[found++] = n->entity.entity_id;
            }
        }
    }
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return (int)found;
}

int kg_find_entities_by_name(const GV_KnowledgeGraph *kg, const char *name,
                                 uint64_t *out_ids, size_t max_count) {
    if (!kg || !name || !out_ids || max_count == 0) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);
    size_t found = 0;
    KG_IndexEntry *e = kg_index_find(kg->name_index,
                                     kg->entity_bucket_count,
                                     kg_hash_string(name));
    if (e) {
        for (size_t i = 0; i < e->list.count && found < max_count; i++) {
            KG_EntityNode *n = kg_find_entity_node(kg, e->list.ids[i]);
            /* Hash collision check: verify the name actually matches */
            if (n && n->entity.name && strcmp(n->entity.name, name) == 0) {
                out_ids[found++] = n->entity.entity_id;
            }
        }
    }
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return (int)found;
}

/* Insert a relation with an explicit id (WAL replay). Caller holds the write
 * lock. Returns 0 on success. */
static int kg_insert_relation_with_id(GV_KnowledgeGraph *kg, uint64_t rid,
                                      uint64_t subject, const char *predicate,
                                      uint64_t object, float weight) {
    KG_EntityNode *subj_node = kg_find_entity_node(kg, subject);
    KG_EntityNode *obj_node = kg_find_entity_node(kg, object);
    if (!subj_node || !obj_node || !predicate) return -1;

    KG_RelationNode *node = (KG_RelationNode *)gv_calloc(1,
                                sizeof(KG_RelationNode));
    if (!node) return -1;

    GV_KGRelation *r = &node->relation;
    r->relation_id = rid;
    r->subject_id  = subject;
    r->object_id   = object;
    r->predicate   = gv_dup_cstr(predicate);
    r->weight      = weight;
    r->created_at  = kg_now_epoch();
    if (!r->predicate) {
        gv_free(node);
        return -1;
    }

    size_t bucket = (size_t)kg_hash_uint64(rid, kg->relation_bucket_count);
    node->next = kg->relation_buckets[bucket];
    kg->relation_buckets[bucket] = node;
    kg->relation_count++;

    KG_IndexEntry *se = kg_index_get_or_create(kg->subject_index,
                                                kg->spo_bucket_count,
                                                subject);
    if (se) kg_idlist_push(&se->list, rid);

    KG_IndexEntry *oe = kg_index_get_or_create(kg->object_index,
                                                kg->spo_bucket_count,
                                                object);
    if (oe) kg_idlist_push(&oe->list, rid);

    uint64_t pred_hash = kg_hash_string(predicate);
    KG_IndexEntry *pe = kg_index_get_or_create(kg->predicate_index,
                                                kg->spo_bucket_count,
                                                pred_hash);
    if (pe) kg_idlist_push(&pe->list, rid);

    /* Wire the new edge into both endpoints for O(1) pointer-deref hops. */
    kg_link_adjacency(subj_node, obj_node, node);

    if (rid >= kg->next_relation_id) kg->next_relation_id = rid + 1;
    return 0;
}

uint64_t kg_add_relation(GV_KnowledgeGraph *kg, uint64_t subject,
                             const char *predicate, uint64_t object,
                             float weight) {
    if (!kg || !predicate) return 0;

    pthread_rwlock_wrlock(&kg->rwlock);

    uint64_t rid = kg->next_relation_id++;

    if (kg_insert_relation_with_id(kg, rid, subject, predicate, object,
                                   weight) != 0) {
        pthread_rwlock_unlock(&kg->rwlock);
        return 0;
    }

    {
        GV_WalBuf b;
        gv_wal_init(&b);
        gv_wal_put_u8(&b, KG_WAL_OP_ADD_RELATION);
        gv_wal_put_u64(&b, rid);
        gv_wal_put_u64(&b, subject);
        gv_wal_put_str(&b, predicate);
        gv_wal_put_u64(&b, object);
        gv_wal_put_f32(&b, weight);
        kg_wal_append(kg, &b);
        gv_wal_free(&b);
    }

    pthread_rwlock_unlock(&kg->rwlock);
    return rid;
}

int kg_remove_relation(GV_KnowledgeGraph *kg, uint64_t relation_id) {
    if (!kg) return -1;
    pthread_rwlock_wrlock(&kg->rwlock);
    int rc = kg_remove_relation_internal(kg, relation_id);
    if (rc == 0) {
        GV_WalBuf b;
        gv_wal_init(&b);
        gv_wal_put_u8(&b, KG_WAL_OP_REMOVE_RELATION);
        gv_wal_put_u64(&b, relation_id);
        kg_wal_append(kg, &b);
        gv_wal_free(&b);
    }
    pthread_rwlock_unlock(&kg->rwlock);
    return rc;
}

const GV_KGRelation *kg_get_relation(const GV_KnowledgeGraph *kg,
                                         uint64_t relation_id) {
    if (!kg) return NULL;
    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);
    KG_RelationNode *n = kg_find_relation_node(kg, relation_id);
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return n ? &n->relation : NULL;
}

int kg_set_relation_prop(GV_KnowledgeGraph *kg, uint64_t relation_id,
                             const char *key, const char *value) {
    if (!kg || !key || !value) return -1;

    pthread_rwlock_wrlock(&kg->rwlock);
    KG_RelationNode *n = kg_find_relation_node(kg, relation_id);
    if (!n) {
        pthread_rwlock_unlock(&kg->rwlock);
        return -1;
    }
    /* Keep the chunk_id -> relations index in sync with the property. */
    int is_chunk_key = strcmp(key, "chunk_id") == 0;
    GV_KGProp *old = is_chunk_key
        ? kg_prop_find(n->relation.properties, "chunk_id") : NULL;
    if (old && old->value.type != GV_PROP_NULL) {
        kg_index_remove_id(kg->chunk_index, kg->spo_bucket_count,
                           kg_prop_hash(old->value), relation_id);
    }
    /* Relation does not track prop_count in the public struct, use a local */
    size_t dummy_count = 0;
    GV_KGProp *p = n->relation.properties;
    while (p) { dummy_count++; p = p->next; }
    int rc = kg_prop_set(&n->relation.properties, &dummy_count, key, value);
    if (rc == 0 && is_chunk_key) {
        KG_IndexEntry *ce = kg_index_get_or_create(kg->chunk_index,
                                                   kg->spo_bucket_count,
                                                   kg_hash_string(value));
        if (ce) kg_idlist_push(&ce->list, relation_id);
    }
    if (rc == 0) {
        GV_WalBuf b;
        gv_wal_init(&b);
        gv_wal_put_u8(&b, KG_WAL_OP_SET_RELATION_PROP);
        gv_wal_put_u64(&b, relation_id);
        gv_wal_put_str(&b, key);
        gv_wal_put_str(&b, value);
        kg_wal_append(kg, &b);
        gv_wal_free(&b);
    }
    pthread_rwlock_unlock(&kg->rwlock);
    return rc;
}

/**
 * @brief Internal helper: fill a GV_KGTriple from a relation.
 */
static void kg_fill_triple(const GV_KnowledgeGraph *kg,
                            const GV_KGRelation *rel, GV_KGTriple *t) {
    t->subject_id = rel->subject_id;
    t->object_id  = rel->object_id;
    t->predicate  = gv_dup_cstr(rel->predicate);
    t->score      = rel->weight;

    KG_EntityNode *sn = kg_find_entity_node(kg, rel->subject_id);
    t->subject_name = sn ? gv_dup_cstr(sn->entity.name) : gv_dup_cstr("?");

    KG_EntityNode *on = kg_find_entity_node(kg, rel->object_id);
    t->object_name = on ? gv_dup_cstr(on->entity.name) : gv_dup_cstr("?");
}

static int kg_triple_matches(const GV_KGRelation *r,
                              const uint64_t *subject,
                              const char *predicate,
                              const uint64_t *object) {
    if (subject && r->subject_id != *subject) return 0;
    if (object  && r->object_id  != *object)  return 0;
    if (predicate && strcmp(r->predicate, predicate) != 0) return 0;
    return 1;
}

int kg_query_triples(const GV_KnowledgeGraph *kg, const uint64_t *subject,
                         const char *predicate, const uint64_t *object,
                         GV_KGTriple *out, size_t max_count) {
    if (!kg || !out || max_count == 0) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);

    size_t found = 0;

    /*
     * Optimisation: if subject is given, use subject_index.
     * If object given (no subject), use object_index.
     * If only predicate given, use predicate_index.
     * Otherwise full scan.
     */
    if (subject) {
        KG_IndexEntry *se = kg_index_find(kg->subject_index,
                                           kg->spo_bucket_count, *subject);
        if (se) {
            for (size_t i = 0; i < se->list.count && found < max_count; i++) {
                KG_RelationNode *rn = kg_find_relation_node(kg,
                                                             se->list.ids[i]);
                if (!rn) continue;
                if (kg_triple_matches(&rn->relation, subject,
                                       predicate, object)) {
                    kg_fill_triple(kg, &rn->relation, &out[found++]);
                }
            }
        }
    } else if (object) {
        KG_IndexEntry *oe = kg_index_find(kg->object_index,
                                           kg->spo_bucket_count, *object);
        if (oe) {
            for (size_t i = 0; i < oe->list.count && found < max_count; i++) {
                KG_RelationNode *rn = kg_find_relation_node(kg,
                                                             oe->list.ids[i]);
                if (!rn) continue;
                if (kg_triple_matches(&rn->relation, subject,
                                       predicate, object)) {
                    kg_fill_triple(kg, &rn->relation, &out[found++]);
                }
            }
        }
    } else if (predicate) {
        uint64_t ph = kg_hash_string(predicate);
        KG_IndexEntry *pe = kg_index_find(kg->predicate_index,
                                           kg->spo_bucket_count, ph);
        if (pe) {
            for (size_t i = 0; i < pe->list.count && found < max_count; i++) {
                KG_RelationNode *rn = kg_find_relation_node(kg,
                                                             pe->list.ids[i]);
                if (!rn) continue;
                /* Hash collision check: verify predicate actually matches */
                if (strcmp(rn->relation.predicate, predicate) == 0) {
                    kg_fill_triple(kg, &rn->relation, &out[found++]);
                }
            }
        }
    } else {
        /* Full scan (all wildcards) */
        for (size_t b = 0; b < kg->relation_bucket_count &&
             found < max_count; b++) {
            for (KG_RelationNode *n = kg->relation_buckets[b];
                 n && found < max_count; n = n->next) {
                kg_fill_triple(kg, &n->relation, &out[found++]);
            }
        }
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return (int)found;
}

void kg_free_triples(GV_KGTriple *triples, size_t count) {
    if (!triples) return;
    for (size_t i = 0; i < count; i++) {
        gv_free(triples[i].subject_name);
        gv_free(triples[i].predicate);
        gv_free(triples[i].object_name);
    }
}

/* Combined-DB integration (Phase 1) */

void kg_attach_vector_db(GV_KnowledgeGraph *kg, struct GV_Database *db) {
    if (!kg) return;
    pthread_rwlock_wrlock(&kg->rwlock);
    kg->vdb = db;
    pthread_rwlock_unlock(&kg->rwlock);
}

int kg_all_entity_ids(const GV_KnowledgeGraph *kg, uint64_t *out_ids, size_t max_count) {
    if (!kg || !out_ids || max_count == 0) return -1;
    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);
    size_t found = 0;
    for (size_t i = 0; i < kg->entity_bucket_count && found < max_count; i++) {
        for (KG_EntityNode *n = kg->entity_buckets[i]; n && found < max_count; n = n->next) {
            out_ids[found++] = n->entity.entity_id;
        }
    }
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return (int)found;
}

uint64_t kg_add_relation_with_chunk(GV_KnowledgeGraph *kg, uint64_t subject,
                                    const char *predicate, uint64_t object,
                                    float weight, const char *chunk_id) {
    uint64_t rid = kg_add_relation(kg, subject, predicate, object, weight);
    if (rid != 0 && chunk_id != NULL) {
        kg_set_relation_prop(kg, rid, "chunk_id", chunk_id);
    }
    return rid;
}

uint64_t kg_add_relation_with_props(GV_KnowledgeGraph *kg, uint64_t subject,
                                    const char *predicate, uint64_t object,
                                    float weight,
                                    const GV_KGPropKV *props, size_t n_props) {
    uint64_t rid = kg_add_relation(kg, subject, predicate, object, weight);
    if (rid != 0 && props != NULL) {
        for (size_t i = 0; i < n_props; i++) {
            if (props[i].key && props[i].value) {
                kg_set_relation_prop(kg, rid, props[i].key, props[i].value);
            }
        }
    }
    return rid;
}

int kg_query_triples_by_chunk(const GV_KnowledgeGraph *kg, const char *chunk_id,
                              GV_KGTriple *out, size_t max_count) {
    if (!kg || !chunk_id || !out || max_count == 0) return -1;
    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);
    size_t found = 0;
    uint64_t ch = kg_hash_string(chunk_id);
    KG_IndexEntry *ce = kg_index_find(kg->chunk_index, kg->spo_bucket_count, ch);
    if (ce) {
        for (size_t i = 0; i < ce->list.count && found < max_count; i++) {
            KG_RelationNode *rn = kg_find_relation_node(kg, ce->list.ids[i]);
            if (!rn) continue;
            /* Hash collision check: verify the relation's chunk_id actually matches */
            GV_KGProp *p = kg_prop_find(rn->relation.properties, "chunk_id");
            if (p && kg_prop_str_eq(p->value, chunk_id)) {
                kg_fill_triple(kg, &rn->relation, &out[found++]);
            }
        }
    }
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return (int)found;
}

int kg_remove_relations_by_chunk(GV_KnowledgeGraph *kg, const char *chunk_id) {
    if (!kg || !chunk_id) return -1;
    pthread_rwlock_wrlock(&kg->rwlock);
    uint64_t ch = kg_hash_string(chunk_id);
    KG_IndexEntry *ce = kg_index_find(kg->chunk_index, kg->spo_bucket_count, ch);
    if (!ce) {
        pthread_rwlock_unlock(&kg->rwlock);
        return 0;
    }
    size_t removed = 0;
    /* Copy ids out first: removal mutates the index list we are iterating. */
    size_t n = ce->list.count;
    uint64_t *ids = (uint64_t *)gv_alloc((n ? n : 1) * sizeof(uint64_t));
    if (!ids) {
        pthread_rwlock_unlock(&kg->rwlock);
        return -1;
    }
    memcpy(ids, ce->list.ids, n * sizeof(uint64_t));
    for (size_t i = 0; i < n; i++) {
        /* Verify the relation's chunk_id actually matches (hash collisions). */
        KG_RelationNode *rn = kg_find_relation_node(kg, ids[i]);
        if (!rn) continue;
        GV_KGProp *p = kg_prop_find(rn->relation.properties, "chunk_id");
        if (!p || !kg_prop_str_eq(p->value, chunk_id)) continue;
        if (kg_remove_relation_internal(kg, ids[i]) == 0) removed++;
    }
    gv_free(ids);
    pthread_rwlock_unlock(&kg->rwlock);
    return (int)removed;
}

int kg_expand_context(const GV_KnowledgeGraph *kg,
                      const uint64_t *seeds, size_t n_seeds,
                      size_t radius, GV_KGTriple *out, size_t max_count) {
    if (!kg || !out || max_count == 0 || radius == 0) return -1;
    if (n_seeds > 0 && !seeds) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);

    /* Visited set over entity ids: open addressing, power-of-two capacity. */
    size_t vcap = 1024;
    while (vcap < (n_seeds + 8) * 4) vcap <<= 1;
    uint64_t *vkeys = (uint64_t *)gv_calloc(vcap, sizeof(uint64_t));
    uint8_t *vocc = (uint8_t *)gv_calloc(vcap, sizeof(uint8_t));
    /* Relation-id set so a triple is not re-emitted when reached from both
     * endpoints during undirected expansion. */
    uint64_t *rkeys = (uint64_t *)gv_calloc(vcap, sizeof(uint64_t));
    uint8_t *rocc = (uint8_t *)gv_calloc(vcap, sizeof(uint8_t));
    uint64_t *frontier = (uint64_t *)gv_alloc((n_seeds ? n_seeds : 1)
                                              * sizeof(uint64_t));
    uint64_t *next_f = (uint64_t *)gv_alloc(vcap * sizeof(uint64_t));
    if (!vkeys || !vocc || !rkeys || !rocc || !frontier || !next_f) {
        gv_free(vkeys); gv_free(vocc); gv_free(rkeys); gv_free(rocc);
        gv_free(frontier); gv_free(next_f);
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return -1;
    }

#define VX_H(id) ((size_t)((id) * 0x9E3779B97F4A7C15ULL) & (vcap - 1))
#define VX_ADD(id) do { \
        size_t h = VX_H(id); \
        for (size_t j = 0; j < vcap; j++) { \
            size_t pp = (h + j) & (vcap - 1); \
            if (!vocc[pp]) { vocc[pp] = 1; vkeys[pp] = (id); break; } \
            if (vkeys[pp] == (id)) break; \
        } \
    } while (0)
#define RX_HAS(id) ({ \
    int _r = 0; \
    size_t h = VX_H(id); \
    for (size_t j = 0; j < vcap; j++) { \
        size_t pp = (h + j) & (vcap - 1); \
        if (!rocc[pp]) break; \
        if (rkeys[pp] == (id)) { _r = 1; break; } \
    } \
    _r; })
#define RX_ADD(id) do { \
    size_t h = VX_H(id); \
    for (size_t j = 0; j < vcap; j++) { \
        size_t pp = (h + j) & (vcap - 1); \
        if (!rocc[pp]) { rocc[pp] = 1; rkeys[pp] = (id); break; } \
        if (rkeys[pp] == (id)) break; \
    } \
} while (0)
#define VX_HAS(id) ({ \
    int _r = 0; \
    size_t h = VX_H(id); \
    for (size_t j = 0; j < vcap; j++) { \
        size_t pp = (h + j) & (vcap - 1); \
        if (!vocc[pp]) break; \
        if (vkeys[pp] == (id)) { _r = 1; break; } \
    } \
    _r; })

    size_t f_len = 0;
    for (size_t i = 0; i < n_seeds; i++) {
        uint64_t id = seeds[i];
        if (!kg_find_entity_node(kg, id)) continue;   /* unknown seed: skip */
        if (VX_HAS(id)) continue;
        VX_ADD(id);
        frontier[f_len++] = id;
    }

    size_t total_found = 0;
    for (size_t hop = 0; hop < radius && total_found < max_count; hop++) {
        size_t nf_len = 0;
        for (size_t i = 0; i < f_len && total_found < max_count; i++) {
            KG_EntityNode *en = kg_find_entity_node(kg, frontier[i]);
            if (!en) continue;
            /* Incident relations via O(1)-per-hop adjacency pointers. */
            for (int dir = 0; dir < 2 && total_found < max_count; dir++) {
                size_t ecnt = dir == 0 ? en->out_count : en->in_count;
                KG_Edge *edges = dir == 0 ? en->out_edges : en->in_edges;
                for (size_t k = 0; k < ecnt && total_found < max_count; k++) {
                    if (!edges[k].rel) continue;
                    if (RX_HAS(edges[k].rel->relation.relation_id)) continue;
                    RX_ADD(edges[k].rel->relation.relation_id);
                    kg_fill_triple(kg, &edges[k].rel->relation,
                                   &out[total_found]);
                    total_found++;
                    KG_EntityNode *nb = edges[k].neighbor;
                    if (nb && !VX_HAS(nb->entity.entity_id)) {
                        VX_ADD(nb->entity.entity_id);
                        next_f[nf_len++] = nb->entity.entity_id;
                    }
                }
            }
        }
        uint64_t *tmp = frontier; frontier = next_f; next_f = tmp;
        f_len = nf_len;
        if (f_len == 0) break;
    }

#undef VX_H

    gv_free(vkeys);
    gv_free(vocc);
    gv_free(rkeys);
    gv_free(rocc);
    gv_free(frontier);
    gv_free(next_f);
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return (int)total_found;
}

int kg_query_reverse(const GV_KnowledgeGraph *kg, uint64_t object,
                     const char *predicate, GV_KGTriple *out, size_t max_count) {
    /* Reuse the object index via kg_query_triples with the object bound. */
    return kg_query_triples(kg, NULL, predicate, &object, out, max_count);
}

int kg_search_similar(const GV_KnowledgeGraph *kg,
                          const float *query_embedding, size_t dimension,
                          size_t k, GV_KGSearchResult *results) {
    if (!kg || !query_embedding || !results || k == 0) return -1;
    if (kg->config.embedding_dimension == 0) return 0;
    if (dimension != kg->config.embedding_dimension) return -1;

    /* If a vector DB is attached, resolve similarity via db_search (filtering to
     * entity vectors keyed "kgent:", which may be interleaved with chunks). */
    GV_Database *vdb = kg->vdb;
    if (vdb != NULL) {
        size_t over = k * 8;
        if (over < 32) over = 32;
        GV_SearchResult *sr = (GV_SearchResult *)gv_alloc(over * sizeof(GV_SearchResult));
        if (!sr) return -1;
        int n_sr = db_search(vdb, query_embedding, over, sr, GV_DISTANCE_COSINE);
        if (n_sr < 0) { gv_free(sr); return -1; }
        size_t out = 0;
        pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);
        for (int i = 0; i < n_sr && out < k; i++) {
            const char *sid = db_get_id_by_index(vdb, sr[i].id);
            if (!sid || strncmp(sid, "kgent:", 6) != 0) continue;
            uint64_t eid = strtoull(sid + 6, NULL, 10);
            KG_EntityNode *en = kg_find_entity_node(kg, eid);
            results[out].entity_id  = eid;
            results[out].name       = en ? gv_dup_cstr(en->entity.name) : NULL;
            results[out].type       = en ? gv_dup_cstr(en->entity.type) : NULL;
            results[out].similarity = 1.0f - sr[i].distance; /* cosine distance -> similarity */
            out++;
        }
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        /* db_search hands back owned result vectors - free all of them. */
        for (int i = 0; i < n_sr; i++)
            if (sr[i].vector) vector_destroy((GV_Vector *)sr[i].vector);
        gv_free(sr);
        return (int)out;
    }

    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);

    size_t n = kg->embedding_count;
    if (n == 0) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return 0;
    }

    KG_ScorePair *pairs = (KG_ScorePair *)gv_alloc(n * sizeof(KG_ScorePair));
    if (!pairs) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return -1;
    }

    size_t dim = kg->config.embedding_dimension;
    for (size_t i = 0; i < n; i++) {
        pairs[i].id = kg->embedding_entity_ids[i];
        pairs[i].score = cosine_similarity(query_embedding,
                                               kg->all_embeddings + i * dim,
                                               dim);
    }

    qsort(pairs, n, sizeof(KG_ScorePair), kg_score_cmp_desc);

    size_t result_count = (k < n) ? k : n;
    for (size_t i = 0; i < result_count; i++) {
        KG_EntityNode *en = kg_find_entity_node(kg, pairs[i].id);
        results[i].entity_id  = pairs[i].id;
        results[i].name       = en ? gv_dup_cstr(en->entity.name) : NULL;
        results[i].type       = en ? gv_dup_cstr(en->entity.type) : NULL;
        results[i].similarity = pairs[i].score;
    }

    gv_free(pairs);
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return (int)result_count;
}

int kg_search_by_text(const GV_KnowledgeGraph *kg, const char *text,
                          const float *text_embedding, size_t dimension,
                          size_t k, GV_KGSearchResult *results) {
    if (!kg || !text || !results || k == 0) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);

    size_t cap = 256;
    KG_ScorePair *pairs = (KG_ScorePair *)gv_alloc(cap * sizeof(KG_ScorePair));
    if (!pairs) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return -1;
    }
    size_t pair_count = 0;

    size_t dim = kg->config.embedding_dimension;

    for (size_t b = 0; b < kg->entity_bucket_count; b++) {
        for (KG_EntityNode *n = kg->entity_buckets[b]; n; n = n->next) {
            float score = 0.0f;
            if (n->entity.name) {
                if (strcmp(n->entity.name, text) == 0) {
                    score += 1.0f;
                } else if (strstr(n->entity.name, text)) {
                    score += 0.5f;
                }
            }
            if (text_embedding && dim > 0 && dimension == dim) {
                const float *emb = kg_embedding_get(kg,
                    n->entity.entity_id, NULL);
                if (emb) {
                    float sim = cosine_similarity(text_embedding, emb, dim);
                    score += sim;
                }
            }
            if (score > 0.0f) {
                if (pair_count >= cap) {
                    /* Grow cap only AFTER a successful realloc: bumping it first
                     * and then break-ing on failure left cap > the real buffer,
                     * so the next entity's pairs[pair_count] write ran OOB. */
                    size_t new_cap = cap * 2;
                    KG_ScorePair *tmp = (KG_ScorePair *)gv_realloc(pairs,
                        new_cap * sizeof(KG_ScorePair));
                    if (!tmp) break;
                    pairs = tmp;
                    cap = new_cap;
                }
                pairs[pair_count].id = n->entity.entity_id;
                pairs[pair_count].score = score;
                pair_count++;
            }
        }
    }

    qsort(pairs, pair_count, sizeof(KG_ScorePair), kg_score_cmp_desc);

    size_t result_count = (k < pair_count) ? k : pair_count;
    for (size_t i = 0; i < result_count; i++) {
        KG_EntityNode *en = kg_find_entity_node(kg, pairs[i].id);
        results[i].entity_id  = pairs[i].id;
        results[i].name       = en ? gv_dup_cstr(en->entity.name) : NULL;
        results[i].type       = en ? gv_dup_cstr(en->entity.type) : NULL;
        results[i].similarity = pairs[i].score;
    }

    gv_free(pairs);
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return (int)result_count;
}

void kg_free_search_results(GV_KGSearchResult *results, size_t count) {
    if (!results) return;
    for (size_t i = 0; i < count; i++) {
        gv_free(results[i].name);
        gv_free(results[i].type);
    }
}

uint64_t kg_resolve_entity_id(GV_KnowledgeGraph *kg, const char *name,
                          const char *type, const float *embedding,
                          size_t dimension) {
    if (!kg || !name || !type) return 0;

    pthread_rwlock_wrlock(&kg->rwlock);

    for (size_t b = 0; b < kg->entity_bucket_count; b++) {
        for (KG_EntityNode *n = kg->entity_buckets[b]; n; n = n->next) {
            if (n->entity.name && n->entity.type &&
                strcmp(n->entity.name, name) == 0 &&
                strcmp(n->entity.type, type) == 0) {
                uint64_t eid = n->entity.entity_id;
                pthread_rwlock_unlock(&kg->rwlock);
                return eid;
            }
        }
    }

    if (embedding && dimension > 0 && kg->config.embedding_dimension > 0 &&
        dimension == kg->config.embedding_dimension) {
        float best_sim = 0.0f;
        uint64_t best_id = 0;
        size_t dim = kg->config.embedding_dimension;

        for (size_t i = 0; i < kg->embedding_count; i++) {
            uint64_t eid = kg->embedding_entity_ids[i];
            KG_EntityNode *en = kg_find_entity_node(kg, eid);
            if (!en || !en->entity.type) continue;
            if (strcmp(en->entity.type, type) != 0) continue;

            float sim = cosine_similarity(embedding,
                kg->all_embeddings + i * dim, dim);
            if (sim > best_sim) {
                best_sim = sim;
                best_id = eid;
            }
        }

        if (best_sim >= kg->config.similarity_threshold && best_id != 0) {
            pthread_rwlock_unlock(&kg->rwlock);
            return best_id;
        }
    }

    pthread_rwlock_unlock(&kg->rwlock);
    uint64_t new_id = kg_add_entity(kg, name, type, embedding, dimension);
    return new_id;
}

/*
 * Legacy int-returning wrapper. Entity IDs are 64-bit and may not fit in an
 * int; rather than silently truncating (which historically corrupted large
 * ids), clamp any value that does not fit a non-negative int to INT_MAX.
 * 0 remains the error/not-found sentinel (entity IDs start at 1).
 */
int kg_resolve_entity(GV_KnowledgeGraph *kg, const char *name,
                          const char *type, const float *embedding,
                          size_t dimension) {
    uint64_t id = kg_resolve_entity_id(kg, name, type, embedding, dimension);
    if (id > (uint64_t)INT_MAX) return INT_MAX;
    return (int)id;
}

int kg_find_duplicates(const GV_KnowledgeGraph *kg, float threshold,
                           GV_KGLinkPrediction *out, size_t max_count) {
    if (!kg || !out || max_count == 0) return -1;
    if (kg->config.embedding_dimension == 0) return 0;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);

    size_t found = 0;
    size_t dim = kg->config.embedding_dimension;
    size_t n = kg->embedding_count;

    for (size_t i = 0; i < n && found < max_count; i++) {
        for (size_t j = i + 1; j < n && found < max_count; j++) {
            float sim = cosine_similarity(
                kg->all_embeddings + i * dim,
                kg->all_embeddings + j * dim, dim);
            if (sim >= threshold) {
                out[found].entity_a = kg->embedding_entity_ids[i];
                out[found].entity_b = kg->embedding_entity_ids[j];
                out[found].predicted_predicate = gv_dup_cstr("duplicate");
                out[found].confidence = sim;
                found++;
            }
        }
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return (int)found;
}

int kg_merge_entities(GV_KnowledgeGraph *kg, uint64_t keep_id,
                          uint64_t merge_id) {
    if (!kg || keep_id == merge_id) return -1;

    pthread_rwlock_wrlock(&kg->rwlock);

    KG_EntityNode *keep_node  = kg_find_entity_node(kg, keep_id);
    KG_EntityNode *merge_node = kg_find_entity_node(kg, merge_id);
    if (!keep_node || !merge_node) {
        pthread_rwlock_unlock(&kg->rwlock);
        return -1;
    }

    for (GV_KGProp *p = merge_node->entity.properties; p; p = p->next) {
        if (!kg_prop_find(keep_node->entity.properties, p->key)) {
            char *pv = gv_prop_to_string(p->value);
            kg_prop_set(&keep_node->entity.properties,
                        &keep_node->entity.prop_count, p->key, pv);
            gv_free(pv);
        }
    }

    uint64_t *rel_ids = NULL;
    size_t rel_count = kg_collect_relations_for_entity(kg, merge_id,
                                                        &rel_ids);
    for (size_t i = 0; i < rel_count; i++) {
        KG_RelationNode *rn = kg_find_relation_node(kg, rel_ids[i]);
        if (!rn) continue;

        uint64_t rid = rn->relation.relation_id;

        if (rn->relation.subject_id == merge_id) {
            kg_index_remove_id(kg->subject_index, kg->spo_bucket_count,
                               merge_id, rid);
            rn->relation.subject_id = keep_id;
            KG_IndexEntry *se = kg_index_get_or_create(
                kg->subject_index, kg->spo_bucket_count, keep_id);
            if (se) kg_idlist_push(&se->list, rid);
        }

        if (rn->relation.object_id == merge_id) {
            kg_index_remove_id(kg->object_index, kg->spo_bucket_count,
                               merge_id, rid);
            rn->relation.object_id = keep_id;
            KG_IndexEntry *oe = kg_index_get_or_create(
                kg->object_index, kg->spo_bucket_count, keep_id);
            if (oe) kg_idlist_push(&oe->list, rid);
        }
    }
    gv_free(rel_ids);

    kg_embedding_remove(kg, merge_id);

    size_t bucket = (size_t)kg_hash_uint64(merge_id,
                                            kg->entity_bucket_count);
    KG_EntityNode *prev = NULL;
    for (KG_EntityNode *n = kg->entity_buckets[bucket]; n; n = n->next) {
        if (n->entity.entity_id == merge_id) {
            if (prev) prev->next = n->next;
            else kg->entity_buckets[bucket] = n->next;
            kg_entity_adjacency_free(n);
            kg_entity_data_free(&n->entity);
            gv_free(n);
            kg->entity_count--;
            break;
        }
        prev = n;
    }

    /* Relations were re-pointed from merge_id to keep_id in place; rebuild the
     * adjacency so every edge references the surviving node. */
    kg_rebuild_adjacency(kg);

    pthread_rwlock_unlock(&kg->rwlock);
    return 0;
}

int kg_predict_links(const GV_KnowledgeGraph *kg, uint64_t entity_id,
                         size_t k, GV_KGLinkPrediction *results) {
    if (!kg || !results || k == 0) return -1;
    if (kg->config.embedding_dimension == 0) return 0;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);

    size_t dim = kg->config.embedding_dimension;
    const float *src_emb = kg_embedding_get(kg, entity_id, NULL);
    if (!src_emb) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return 0;
    }

    size_t cap = 256;
    KG_ScorePair *candidates = (KG_ScorePair *)gv_alloc(
        cap * sizeof(KG_ScorePair));
    if (!candidates) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return -1;
    }
    size_t cand_count = 0;

    for (size_t i = 0; i < kg->embedding_count; i++) {
        uint64_t other_id = kg->embedding_entity_ids[i];
        if (other_id == entity_id) continue;
        if (kg_are_connected(kg, entity_id, other_id)) continue;

        float sim = cosine_similarity(src_emb,
            kg->all_embeddings + i * dim, dim);
        if (sim < kg->config.link_prediction_threshold) continue;

        size_t shared = kg_shared_neighbors(kg, entity_id, other_id);
        float boost = (shared > 0) ? 0.1f * (float)shared : 0.0f;
        if (boost > 0.2f) boost = 0.2f;

        float final_score = sim + boost;
        if (final_score > 1.0f) final_score = 1.0f;

        if (cand_count >= cap) {
            cap *= 2;
            KG_ScorePair *tmp = (KG_ScorePair *)gv_realloc(candidates,
                cap * sizeof(KG_ScorePair));
            if (!tmp) break;
            candidates = tmp;
        }
        candidates[cand_count].id = other_id;
        candidates[cand_count].score = final_score;
        cand_count++;
    }

    qsort(candidates, cand_count, sizeof(KG_ScorePair), kg_score_cmp_desc);

    size_t result_count = (k < cand_count) ? k : cand_count;
    for (size_t i = 0; i < result_count; i++) {
        results[i].entity_a = entity_id;
        results[i].entity_b = candidates[i].id;
        results[i].predicted_predicate = gv_dup_cstr("related_to");
        results[i].confidence = candidates[i].score;
    }

    gv_free(candidates);
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return (int)result_count;
}

int kg_for_each_hop(const GV_KnowledgeGraph *kg, uint64_t entity_id,
                    int dir, const char *predicate,
                    GV_KGHopVisitor visit, void *ctx) {
    if (!kg || !visit) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);

    KG_EntityNode *node = kg_find_entity_node(kg, entity_id);
    if (!node) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return 0;
    }

    int count = 0;
    GV_KGHop hop;

    /* Outgoing: this node is the subject. Pure pointer dereference per edge. */
    if (dir >= 0) {
        for (size_t i = 0; i < node->out_count; i++) {
            const KG_Edge *e = &node->out_edges[i];
            const GV_KGRelation *r = &e->rel->relation;
            if (predicate && strcmp(r->predicate, predicate) != 0) continue;
            hop.relation_id = r->relation_id;
            hop.neighbor_id = e->neighbor->entity.entity_id;
            hop.predicate = r->predicate;
            hop.weight = r->weight;
            hop.outgoing = 1;
            count++;
            if (visit(&hop, ctx)) {
                pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
                return count;
            }
        }
    }

    /* Incoming: this node is the object. */
    if (dir <= 0) {
        for (size_t i = 0; i < node->in_count; i++) {
            const KG_Edge *e = &node->in_edges[i];
            const GV_KGRelation *r = &e->rel->relation;
            if (predicate && strcmp(r->predicate, predicate) != 0) continue;
            hop.relation_id = r->relation_id;
            hop.neighbor_id = e->neighbor->entity.entity_id;
            hop.predicate = r->predicate;
            hop.weight = r->weight;
            hop.outgoing = 0;
            count++;
            if (visit(&hop, ctx)) {
                pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
                return count;
            }
        }
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return count;
}

int kg_get_neighbors(const GV_KnowledgeGraph *kg, uint64_t entity_id,
                         uint64_t *out_ids, size_t max_count) {
    if (!kg || !out_ids || max_count == 0) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);

    KG_EntityNode *node = kg_find_entity_node(kg, entity_id);
    if (!node) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return -1;
    }

    size_t found = 0;

    /* Outgoing neighbours (object side), then incoming (subject side), via the
     * direct adjacency arrays - no id->node hash lookups. */
    for (size_t i = 0; i < node->out_count && found < max_count; i++) {
        uint64_t nbr = node->out_edges[i].neighbor->entity.entity_id;
        int dup = 0;
        for (size_t j = 0; j < found; j++) {
            if (out_ids[j] == nbr) { dup = 1; break; }
        }
        if (!dup) out_ids[found++] = nbr;
    }
    for (size_t i = 0; i < node->in_count && found < max_count; i++) {
        uint64_t nbr = node->in_edges[i].neighbor->entity.entity_id;
        int dup = 0;
        for (size_t j = 0; j < found; j++) {
            if (out_ids[j] == nbr) { dup = 1; break; }
        }
        if (!dup) out_ids[found++] = nbr;
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return (int)found;
}

int kg_traverse(const GV_KnowledgeGraph *kg, uint64_t start,
                    size_t max_depth, uint64_t *out_ids, size_t max_count) {
    if (!kg || !out_ids || max_count == 0) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);

    if (!kg_find_entity_node(kg, start)) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return -1;
    }

    KG_BFSState bfs;
    if (kg_bfs_init(&bfs, max_count) != 0) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return -1;
    }

    kg_bfs_run(kg, start, max_depth, &bfs);

    size_t count = (bfs.count < max_count) ? bfs.count : max_count;
    for (size_t i = 0; i < count; i++) {
        out_ids[i] = bfs.visited[i];
    }

    kg_bfs_free(&bfs);
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return (int)count;
}

static int kg_is_causal_predicate(const char *predicate) {
    if (!predicate) return 0;
    return strcmp(predicate, "causes") == 0 ||
           strcmp(predicate, "caused_by") == 0 ||
           strcmp(predicate, "enables") == 0 ||
           strcmp(predicate, "prevents") == 0;
}

int kg_spreading_activation(const GV_KnowledgeGraph *kg,
                            const uint64_t *seed_entities, size_t seed_count,
                            size_t max_depth, float causal_boost,
                            GV_KGActivation *out, size_t max_out) {
    if (!kg || !seed_entities || seed_count == 0 || !out || max_out == 0) {
        return -1;
    }
    if (causal_boost < 1.0f) causal_boost = 1.0f;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);

    size_t out_count = 0;
    uint64_t *queue_ids = (uint64_t *)gv_alloc(max_out * sizeof(uint64_t));
    float *queue_act = (float *)gv_alloc(max_out * sizeof(float));
    size_t *queue_depth = (size_t *)gv_alloc(max_out * sizeof(size_t));
    if (!queue_ids || !queue_act || !queue_depth) {
        gv_free(queue_ids);
        gv_free(queue_act);
        gv_free(queue_depth);
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return -1;
    }

    size_t q_head = 0;
    size_t q_tail = 0;

    for (size_t s = 0; s < seed_count; s++) {
        if (!kg_find_entity_node(kg, seed_entities[s])) continue;
        size_t exists = 0;
        for (size_t j = 0; j < out_count; j++) {
            if (out[j].entity_id == seed_entities[s]) {
                exists = 1;
                break;
            }
        }
        if (!exists && out_count < max_out) {
            out[out_count].entity_id = seed_entities[s];
            out[out_count].activation = 1.0f;
            out_count++;
        }
        if (q_tail < max_out) {
            queue_ids[q_tail] = seed_entities[s];
            queue_act[q_tail] = 1.0f;
            queue_depth[q_tail] = 0;
            q_tail++;
        }
    }

    if (out_count == 0) {
        gv_free(queue_ids);
        gv_free(queue_act);
        gv_free(queue_depth);
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return 0;
    }

    while (q_head < q_tail) {
        uint64_t cur = queue_ids[q_head];
        float act = queue_act[q_head];
        size_t dep = queue_depth[q_head];
        q_head++;
        if (dep >= max_depth || act < 0.05f) continue;

        KG_IndexEntry *se = kg_index_find(kg->subject_index, kg->spo_bucket_count, cur);
        if (se) {
            for (size_t i = 0; i < se->list.count; i++) {
                KG_RelationNode *rn = kg_find_relation_node(kg, se->list.ids[i]);
                if (!rn) continue;
                uint64_t nbr = rn->relation.object_id;
                float edge = 0.7f * rn->relation.weight;
                if (kg_is_causal_predicate(rn->relation.predicate)) {
                    edge *= causal_boost;
                }
                float nact = act * edge;
                if (nact < 0.05f) continue;

                size_t idx = SIZE_MAX;
                for (size_t j = 0; j < out_count; j++) {
                    if (out[j].entity_id == nbr) {
                        idx = j;
                        break;
                    }
                }
                if (idx != SIZE_MAX) {
                    if (nact > out[idx].activation) out[idx].activation = nact;
                } else if (out_count < max_out) {
                    out[out_count].entity_id = nbr;
                    out[out_count].activation = nact;
                    out_count++;
                }
                if (q_tail < max_out && dep + 1 <= max_depth) {
                    queue_ids[q_tail] = nbr;
                    queue_act[q_tail] = nact;
                    queue_depth[q_tail] = dep + 1;
                    q_tail++;
                }
            }
        }

        KG_IndexEntry *oe = kg_index_find(kg->object_index, kg->spo_bucket_count, cur);
        if (oe) {
            for (size_t i = 0; i < oe->list.count; i++) {
                KG_RelationNode *rn = kg_find_relation_node(kg, oe->list.ids[i]);
                if (!rn) continue;
                uint64_t nbr = rn->relation.subject_id;
                float edge = 0.7f * rn->relation.weight;
                if (kg_is_causal_predicate(rn->relation.predicate)) {
                    edge *= causal_boost;
                }
                float nact = act * edge;
                if (nact < 0.05f) continue;

                size_t idx = SIZE_MAX;
                for (size_t j = 0; j < out_count; j++) {
                    if (out[j].entity_id == nbr) {
                        idx = j;
                        break;
                    }
                }
                if (idx != SIZE_MAX) {
                    if (nact > out[idx].activation) out[idx].activation = nact;
                } else if (out_count < max_out) {
                    out[out_count].entity_id = nbr;
                    out[out_count].activation = nact;
                    out_count++;
                }
                if (q_tail < max_out && dep + 1 <= max_depth) {
                    queue_ids[q_tail] = nbr;
                    queue_act[q_tail] = nact;
                    queue_depth[q_tail] = dep + 1;
                    q_tail++;
                }
            }
        }
    }

    gv_free(queue_ids);
    gv_free(queue_act);
    gv_free(queue_depth);
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return (int)out_count;
}

int kg_shortest_path(const GV_KnowledgeGraph *kg, uint64_t from,
                         uint64_t to, uint64_t *path_ids, size_t max_len) {
    if (!kg || !path_ids || max_len == 0) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);

    if (!kg_find_entity_node(kg, from) || !kg_find_entity_node(kg, to)) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return -1;
    }

    if (from == to) {
        path_ids[0] = from;
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return 1;
    }

    KG_BFSState bfs;
    size_t bfs_cap = kg->entity_count > 0 ? kg->entity_count + 1 : 256;
    if (kg_bfs_init(&bfs, bfs_cap) != 0) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return -1;
    }

    kg_bfs_push(&bfs, from, 0, 0);
    size_t front = 0;
    int found = 0;

    while (front < bfs.count && !found) {
        uint64_t cur = bfs.visited[front];
        size_t dep   = bfs.depths[front];
        front++;

        if (dep >= max_len) continue;

        KG_IndexEntry *se = kg_index_find(kg->subject_index,
                                           kg->spo_bucket_count, cur);
        if (se) {
            for (size_t i = 0; i < se->list.count && !found; i++) {
                KG_RelationNode *rn = kg_find_relation_node(kg,
                                                             se->list.ids[i]);
                if (!rn) continue;
                uint64_t nbr = rn->relation.object_id;
                if (!kg_bfs_seen(&bfs, nbr)) {
                    kg_bfs_push(&bfs, nbr, dep + 1, cur);
                    if (nbr == to) found = 1;
                }
            }
        }

        if (!found) {
            KG_IndexEntry *oe = kg_index_find(kg->object_index,
                                               kg->spo_bucket_count, cur);
            if (oe) {
                for (size_t i = 0; i < oe->list.count && !found; i++) {
                    KG_RelationNode *rn = kg_find_relation_node(kg,
                                                                 oe->list.ids[i]);
                    if (!rn) continue;
                    uint64_t nbr = rn->relation.subject_id;
                    if (!kg_bfs_seen(&bfs, nbr)) {
                        kg_bfs_push(&bfs, nbr, dep + 1, cur);
                        if (nbr == to) found = 1;
                    }
                }
            }
        }
    }

    if (!found) {
        kg_bfs_free(&bfs);
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return -1;
    }

    size_t path_len = 0;
    uint64_t *rev = (uint64_t *)gv_alloc(bfs.count * sizeof(uint64_t));
    if (!rev) {
        kg_bfs_free(&bfs);
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return -1;
    }

    uint64_t cur = to;
    while (cur != 0 && cur != from) {
        rev[path_len++] = cur;
        uint64_t par = 0;
        for (size_t i = 0; i < bfs.count; i++) {
            if (bfs.visited[i] == cur) {
                par = bfs.parent[i];
                break;
            }
        }
        cur = par;
    }
    rev[path_len++] = from;

    size_t out_len = (path_len < max_len) ? path_len : max_len;
    for (size_t i = 0; i < out_len; i++) {
        path_ids[i] = rev[path_len - 1 - i];
    }

    gv_free(rev);
    kg_bfs_free(&bfs);
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return (int)out_len;
}

int kg_extract_subgraph(const GV_KnowledgeGraph *kg, uint64_t center,
                            size_t radius, GV_KGSubgraph *subgraph) {
    if (!kg || !subgraph) return -1;

    memset(subgraph, 0, sizeof(GV_KGSubgraph));

    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);

    if (!kg_find_entity_node(kg, center)) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return -1;
    }

    KG_BFSState bfs;
    size_t cap = kg->entity_count > 0 ? kg->entity_count + 1 : 256;
    if (kg_bfs_init(&bfs, cap) != 0) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return -1;
    }

    kg_bfs_run(kg, center, radius, &bfs);

    subgraph->entity_ids = (uint64_t *)gv_alloc(bfs.count * sizeof(uint64_t));
    if (!subgraph->entity_ids) {
        kg_bfs_free(&bfs);
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return -1;
    }
    memcpy(subgraph->entity_ids, bfs.visited, bfs.count * sizeof(uint64_t));
    subgraph->entity_count = bfs.count;

    size_t rel_cap = 256;
    size_t rel_count = 0;
    uint64_t *rel_ids = (uint64_t *)gv_alloc(rel_cap * sizeof(uint64_t));
    if (!rel_ids) {
        gv_free(subgraph->entity_ids);
        subgraph->entity_ids = NULL;
        subgraph->entity_count = 0;
        kg_bfs_free(&bfs);
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return -1;
    }

    for (size_t i = 0; i < bfs.count; i++) {
        uint64_t eid = bfs.visited[i];
        KG_IndexEntry *se = kg_index_find(kg->subject_index,
                                           kg->spo_bucket_count, eid);
        if (!se) continue;
        for (size_t r = 0; r < se->list.count; r++) {
            uint64_t rid = se->list.ids[r];
            KG_RelationNode *rn = kg_find_relation_node(kg, rid);
            if (!rn) continue;
            if (!kg_bfs_seen(&bfs, rn->relation.object_id)) continue;
            int dup = 0;
            for (size_t x = 0; x < rel_count; x++) {
                if (rel_ids[x] == rid) { dup = 1; break; }
            }
            if (dup) continue;
            if (rel_count >= rel_cap) {
                rel_cap *= 2;
                uint64_t *tmp = (uint64_t *)gv_realloc(rel_ids,
                    rel_cap * sizeof(uint64_t));
                if (!tmp) break;
                rel_ids = tmp;
            }
            rel_ids[rel_count++] = rid;
        }
    }

    subgraph->relation_ids = rel_ids;
    subgraph->relation_count = rel_count;

    kg_bfs_free(&bfs);
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return 0;
}

void kg_free_subgraph(GV_KGSubgraph *subgraph) {
    if (!subgraph) return;
    gv_free(subgraph->entity_ids);
    gv_free(subgraph->relation_ids);
    subgraph->entity_ids = NULL;
    subgraph->relation_ids = NULL;
    subgraph->entity_count = 0;
    subgraph->relation_count = 0;
}

int kg_hybrid_search(const GV_KnowledgeGraph *kg,
                         const float *query_embedding, size_t dimension,
                         const char *entity_type, const char *predicate_filter,
                         size_t k, GV_KGSearchResult *results) {
    if (!kg || !query_embedding || !results || k == 0) return -1;
    if (kg->config.embedding_dimension == 0) return 0;
    if (dimension != kg->config.embedding_dimension) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);

    size_t dim = kg->config.embedding_dimension;
    size_t n = kg->embedding_count;
    if (n == 0) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return 0;
    }

    size_t cap = 256;
    KG_ScorePair *pairs = (KG_ScorePair *)gv_alloc(cap * sizeof(KG_ScorePair));
    if (!pairs) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return -1;
    }
    size_t pair_count = 0;

    for (size_t i = 0; i < n; i++) {
        uint64_t eid = kg->embedding_entity_ids[i];
        KG_EntityNode *en = kg_find_entity_node(kg, eid);
        if (!en) continue;

        if (entity_type && en->entity.type &&
            strcmp(en->entity.type, entity_type) != 0) continue;

        if (predicate_filter &&
            !kg_entity_has_predicate(kg, eid, predicate_filter)) continue;

        float sim = cosine_similarity(query_embedding,
            kg->all_embeddings + i * dim, dim);

        if (pair_count >= cap) {
            cap *= 2;
            KG_ScorePair *tmp = (KG_ScorePair *)gv_realloc(pairs,
                cap * sizeof(KG_ScorePair));
            if (!tmp) break;
            pairs = tmp;
        }
        pairs[pair_count].id = eid;
        pairs[pair_count].score = sim;
        pair_count++;
    }

    qsort(pairs, pair_count, sizeof(KG_ScorePair), kg_score_cmp_desc);

    size_t result_count = (k < pair_count) ? k : pair_count;
    for (size_t i = 0; i < result_count; i++) {
        KG_EntityNode *en = kg_find_entity_node(kg, pairs[i].id);
        results[i].entity_id  = pairs[i].id;
        results[i].name       = en ? gv_dup_cstr(en->entity.name) : NULL;
        results[i].type       = en ? gv_dup_cstr(en->entity.type) : NULL;
        results[i].similarity = pairs[i].score;
    }

    gv_free(pairs);
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return (int)result_count;
}

int kg_get_stats(const GV_KnowledgeGraph *kg, GV_KGStats *stats) {
    if (!kg || !stats) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);

    memset(stats, 0, sizeof(GV_KGStats));
    stats->entity_count   = kg->entity_count;
    stats->relation_count = kg->relation_count;
    stats->triple_count   = kg->relation_count;
    stats->embedding_count = kg->embedding_count;

    size_t type_cap = 128;
    char **types = (char **)gv_calloc(type_cap, sizeof(char *));
    size_t type_count = 0;
    if (types) {
        for (size_t b = 0; b < kg->entity_bucket_count; b++) {
            for (KG_EntityNode *n = kg->entity_buckets[b]; n; n = n->next) {
                if (!n->entity.type) continue;
                int found = 0;
                for (size_t t = 0; t < type_count; t++) {
                    if (strcmp(types[t], n->entity.type) == 0) {
                        found = 1;
                        break;
                    }
                }
                if (!found) {
                    if (type_count >= type_cap) {
                        type_cap *= 2;
                        char **tmp = (char **)gv_realloc(types,
                            type_cap * sizeof(char *));
                        if (!tmp) break;
                        types = tmp;
                    }
                    types[type_count++] = n->entity.type;
                }
            }
        }
        stats->type_count = type_count;
        gv_free(types);
    }

    size_t pred_cap = 128;
    char **preds = (char **)gv_calloc(pred_cap, sizeof(char *));
    size_t pred_count = 0;
    if (preds) {
        for (size_t b = 0; b < kg->relation_bucket_count; b++) {
            for (KG_RelationNode *n = kg->relation_buckets[b]; n;
                 n = n->next) {
                if (!n->relation.predicate) continue;
                int found = 0;
                for (size_t p = 0; p < pred_count; p++) {
                    if (strcmp(preds[p], n->relation.predicate) == 0) {
                        found = 1;
                        break;
                    }
                }
                if (!found) {
                    if (pred_count >= pred_cap) {
                        pred_cap *= 2;
                        char **tmp = (char **)gv_realloc(preds,
                            pred_cap * sizeof(char *));
                        if (!tmp) break;
                        preds = tmp;
                    }
                    preds[pred_count++] = n->relation.predicate;
                }
            }
        }
        stats->predicate_count = pred_count;
        gv_free(preds);
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return 0;
}

float kg_entity_centrality(const GV_KnowledgeGraph *kg,
                               uint64_t entity_id) {
    if (!kg) return -1.0f;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);

    if (!kg_find_entity_node(kg, entity_id)) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return -1.0f;
    }

    size_t total = kg->entity_count;
    if (total <= 1) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return 0.0f;
    }

    size_t out_degree = 0;
    KG_IndexEntry *se = kg_index_find(kg->subject_index,
                                       kg->spo_bucket_count, entity_id);
    if (se) out_degree = se->list.count;

    size_t in_degree = 0;
    KG_IndexEntry *oe = kg_index_find(kg->object_index,
                                       kg->spo_bucket_count, entity_id);
    if (oe) in_degree = oe->list.count;

    float centrality = (float)(in_degree + out_degree) / (float)(total - 1);

    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return centrality;
}

int kg_get_entity_types(const GV_KnowledgeGraph *kg, char **out_types,
                            size_t max_count) {
    if (!kg || !out_types || max_count == 0) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);

    size_t found = 0;
    for (size_t b = 0; b < kg->entity_bucket_count && found < max_count; b++) {
        for (KG_EntityNode *n = kg->entity_buckets[b];
             n && found < max_count; n = n->next) {
            if (!n->entity.type) continue;
            int dup = 0;
            for (size_t i = 0; i < found; i++) {
                if (strcmp(out_types[i], n->entity.type) == 0) {
                    dup = 1;
                    break;
                }
            }
            if (!dup) {
                out_types[found++] = gv_dup_cstr(n->entity.type);
            }
        }
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return (int)found;
}

int kg_get_predicates(const GV_KnowledgeGraph *kg, char **out_predicates,
                          size_t max_count) {
    if (!kg || !out_predicates || max_count == 0) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);

    size_t found = 0;
    for (size_t b = 0; b < kg->relation_bucket_count &&
         found < max_count; b++) {
        for (KG_RelationNode *n = kg->relation_buckets[b];
             n && found < max_count; n = n->next) {
            if (!n->relation.predicate) continue;
            int dup = 0;
            for (size_t i = 0; i < found; i++) {
                if (strcmp(out_predicates[i], n->relation.predicate) == 0) {
                    dup = 1;
                    break;
                }
            }
            if (!dup) {
                out_predicates[found++] = gv_dup_cstr(n->relation.predicate);
            }
        }
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return (int)found;
}


static int kg_write_props(FILE *fp, const GV_KGProp *props, size_t count) {
    if (write_u32(fp, (uint32_t)count) != 0) return -1;
    for (const GV_KGProp *p = props; p; p = p->next) {
        if (write_string(fp, p->key) != 0) return -1;
        /* write_prop_value: u8 type byte + varint length + bytes */
        char *s = gv_prop_to_string(p->value);
        if (s == NULL) {
            /* fallback: write as string */
            if (write_string(fp, "<null>") != 0) return -1;
        } else {
            if (write_string(fp, s) != 0) { gv_free(s); return -1; }
            gv_free(s);
        }
    }
    return 0;
}

int kg_wal_attach(GV_KnowledgeGraph *kg, const char *snapshot_path) {
    if (!kg || !snapshot_path) return -1;

    pthread_rwlock_wrlock(&kg->rwlock);
    size_t need = strlen(snapshot_path) + 5;
    char *path = (char *)gv_alloc(need);
    if (!path) {
        pthread_rwlock_unlock(&kg->rwlock);
        return -1;
    }
    snprintf(path, need, "%s.wal", snapshot_path);

    FILE *f = fopen(path, "ab");
    if (!f) {
        gv_free(path);
        pthread_rwlock_unlock(&kg->rwlock);
        return -1;
    }
    kg_wal_close(kg);
    kg->wal_file = f;
    kg->wal_path = path;
    pthread_rwlock_unlock(&kg->rwlock);
    return 0;
}

int kg_wal_checkpoint(GV_KnowledgeGraph *kg) {
    if (!kg) return -1;

    pthread_rwlock_wrlock(&kg->rwlock);
    if (!kg->wal_file) {
        pthread_rwlock_unlock(&kg->rwlock);
        return -1;
    }
    /* cppcheck-suppress leakReturnValNotUsed */
    if (freopen(kg->wal_path, "wb", kg->wal_file) == NULL) {
        kg->wal_file = NULL;
        pthread_rwlock_unlock(&kg->rwlock);
        return -1;
    }
#ifndef _WIN32
    int bad = fflush(kg->wal_file) != 0 || fsync(fileno(kg->wal_file)) != 0;
#else
    int bad = fflush(kg->wal_file) != 0 || _commit(_fileno(kg->wal_file)) != 0;
#endif
    pthread_rwlock_unlock(&kg->rwlock);
    return bad ? -1 : 0;
}

/* Apply one decoded record. Called with the write lock held and wal_replaying
 * set; uses the lock-free internal insert/remove paths. */
static void kg_wal_apply_record(GV_KnowledgeGraph *kg, uint8_t op,
                                const uint8_t *p, const uint8_t *end) {
    char *s1 = NULL, *s2 = NULL;
    const float *emb = NULL;
    uint64_t id, a, b2;
    float w;
    uint8_t has_emb = 0;
    uint32_t dim = 0;

    switch (op) {
    case KG_WAL_OP_ADD_ENTITY:
        if ((p = gv_wal_get_u64(p, end, &id)) != NULL &&
            (p = gv_wal_get_str(p, end, &s1)) != NULL &&
            (p = gv_wal_get_str(p, end, &s2)) != NULL &&
            (p = gv_wal_get_u8(p, end, &has_emb)) != NULL) {
            if (has_emb &&
                (p = gv_wal_get_u32(p, end, &dim)) != NULL) {
                /* floats are stored inline right after dim (no extra length
                 * prefix - dim already bounds them) */
                if ((size_t)(end - p) < (size_t)dim * sizeof(float)) {
                    gv_free(s1);
                    gv_free(s2);
                    return;
                }
                /* cppcheck-suppress invalidPointerCast */
                emb = (const float *)p;
                p += (size_t)dim * sizeof(float);
            }
            if (p && (!has_emb || p))
                kg_insert_entity_with_id(kg, id, s1, s2,
                                         has_emb ? emb : NULL,
                                         has_emb ? dim : 0);
        }
        break;
    case KG_WAL_OP_REMOVE_ENTITY:
        if ((p = gv_wal_get_u64(p, end, &id)) != NULL)
            kg_remove_entity_internal(kg, id);   /* lock already held */
        break;
    case KG_WAL_OP_ADD_RELATION:
        if ((p = gv_wal_get_u64(p, end, &id)) != NULL &&
            (p = gv_wal_get_u64(p, end, &a)) != NULL &&
            (p = gv_wal_get_str(p, end, &s1)) != NULL &&
            (p = gv_wal_get_u64(p, end, &b2)) != NULL &&
            (p = gv_wal_get_f32(p, end, &w)) != NULL)
            kg_insert_relation_with_id(kg, id, a, s1, b2, w);
        break;
    case KG_WAL_OP_REMOVE_RELATION:
        if ((p = gv_wal_get_u64(p, end, &id)) != NULL)
            kg_remove_relation_internal(kg, id);
        break;
    case KG_WAL_OP_SET_ENTITY_PROP: {
        if ((p = gv_wal_get_u64(p, end, &id)) != NULL &&
            (p = gv_wal_get_str(p, end, &s1)) != NULL &&
            (p = gv_wal_get_str(p, end, &s2)) != NULL) {
            KG_EntityNode *n = kg_find_entity_node(kg, id);
            if (n)
                kg_prop_set(&n->entity.properties, &n->entity.prop_count,
                            s1, s2);
        }
        break;
    }
    case KG_WAL_OP_SET_RELATION_PROP: {
        /* Route through kg_set_relation_prop semantics minus logging by
         * updating the property directly; chunk index kept in sync here. */
        if ((p = gv_wal_get_u64(p, end, &id)) != NULL &&
            (p = gv_wal_get_str(p, end, &s1)) != NULL &&
            (p = gv_wal_get_str(p, end, &s2)) != NULL) {
            KG_RelationNode *n = kg_find_relation_node(kg, id);
            if (n) {
                size_t pc = 0;
                for (GV_KGProp *q = n->relation.properties; q; q = q->next) pc++;
                kg_prop_set(&n->relation.properties, &pc, s1, s2);
                if (strcmp(s1, "chunk_id") == 0) {
                    KG_IndexEntry *ce = kg_index_get_or_create(
                        kg->chunk_index, kg->spo_bucket_count,
                        kg_hash_string(s2));
                    if (ce) kg_idlist_push(&ce->list, id);
                }
            }
        }
        break;
    }
    case KG_WAL_OP_REMOVE_ENTITY_PROP:
        if ((p = gv_wal_get_u64(p, end, &id)) != NULL &&
            (p = gv_wal_get_str(p, end, &s1)) != NULL) {
            KG_EntityNode *n = kg_find_entity_node(kg, id);
            if (n)
                kg_prop_remove(&n->entity.properties, &n->entity.prop_count, s1);
        }
        break;
    case KG_WAL_OP_REMOVE_RELATION_PROP:
        if ((p = gv_wal_get_u64(p, end, &id)) != NULL &&
            (p = gv_wal_get_str(p, end, &s1)) != NULL) {
            KG_RelationNode *n = kg_find_relation_node(kg, id);
            if (n) {
                int is_chunk_key = strcmp(s1, "chunk_id") == 0;
GV_KGProp *oldp = is_chunk_key
                    ? kg_prop_find(n->relation.properties, "chunk_id") : NULL;
if (oldp && oldp->value.type != GV_PROP_NULL) {
    kg_index_remove_id(kg->chunk_index, kg->spo_bucket_count,
                       kg_prop_hash(oldp->value), id);
}
                size_t pc = 0;
                for (GV_KGProp *q = n->relation.properties; q; q = q->next) pc++;
                kg_prop_remove(&n->relation.properties, &pc, s1);
            }
        }
        break;
    default:
        break;  /* unknown op: ignore (forward compatibility) */
    }
    gv_free(s1);
    gv_free(s2);
}

/* Decode one framed record from buf[0..len): [1B op][4B plen][payload].
 * Returns bytes consumed, or 0 if more data is needed. */
static size_t kg_wal_parse_record(GV_KnowledgeGraph *kg, const uint8_t *buf,
                                  size_t len) {
    if (len < 5) return 0;
    uint32_t plen = 0;
    for (int i = 0; i < 4; i++) plen |= (uint32_t)buf[1 + i] << (8 * i);
    if (len < 5u + (size_t)plen) return 0;
    kg_wal_apply_record(kg, buf[0], buf + 5, buf + 5 + plen);
    return 5u + (size_t)plen;
}

/* Replay "<path>.wal" over a freshly loaded graph and keep it attached.
 * Returns 0 on success (including no log present). */
static int kg_wal_replay_and_attach(GV_KnowledgeGraph *kg,
                                    const char *snapshot_path) {
    size_t need = strlen(snapshot_path) + 5;
    char *path = (char *)gv_alloc(need);
    if (!path) return -1;
    snprintf(path, need, "%s.wal", snapshot_path);

    FILE *f = fopen(path, "rb");
    if (!f) {
        gv_free(path);
        return 0;
    }

    uint8_t rec[65536];
    size_t cap = sizeof(rec), len = 0, nread;

    pthread_rwlock_wrlock(&kg->rwlock);
    kg->wal_replaying = 1;

    int ok = 1;
    while (ok && (nread = fread(rec + len, 1, cap - len, f)) > 0) {
        len += nread;
        size_t off = 0;
        while (ok) {
            size_t used = kg_wal_parse_record(kg, rec + off, len - off);
            if (used == 0) break;
            off += used;
            if (off >= len) break;
        }
        if (off > 0) {
            memmove(rec, rec + off, len - off);
            len -= off;
        }
    }

    kg->wal_replaying = 0;
    pthread_rwlock_unlock(&kg->rwlock);
    fclose(f);

    FILE *af = fopen(path, "ab");
    if (!af) {
        gv_free(path);
        return -1;
    }
    kg->wal_file = af;
    kg->wal_path = path;
    return 0;
}

int kg_save(const GV_KnowledgeGraph *kg, const char *path) {
    if (!kg || !path) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return -1;
    }

    if (write_bytes(fp, KG_MAGIC, KG_MAGIC_LEN) != 0) goto fail;
    if (write_u32(fp, KG_VERSION) != 0) goto fail;

    if (write_u32(fp, (uint32_t)kg->config.embedding_dimension) != 0)
        goto fail;
    if (write_f32(fp, kg->config.similarity_threshold) != 0) goto fail;
    if (write_f32(fp, kg->config.link_prediction_threshold) != 0)
        goto fail;

    if (write_u64(fp, (uint64_t)kg->entity_count) != 0) goto fail;
    if (write_u64(fp, (uint64_t)kg->relation_count) != 0) goto fail;
    if (write_u64(fp, kg->next_entity_id) != 0) goto fail;
    if (write_u64(fp, kg->next_relation_id) != 0) goto fail;

    for (size_t b = 0; b < kg->entity_bucket_count; b++) {
        for (KG_EntityNode *n = kg->entity_buckets[b]; n; n = n->next) {
            const GV_KGEntity *e = &n->entity;
            if (write_u64(fp, e->entity_id) != 0) goto fail;
            if (write_string(fp, e->name) != 0) goto fail;
            if (write_string(fp, e->type) != 0) goto fail;

            uint8_t has_emb = (e->embedding && e->dimension > 0) ? 1 : 0;
            if (write_u8(fp, has_emb) != 0) goto fail;
            if (has_emb) {
                if (write_bytes(fp, e->embedding,
                    e->dimension * sizeof(float)) != 0) goto fail;
            }

            if (write_f32(fp, e->confidence) != 0) goto fail;
            if (write_u64(fp, e->created_at) != 0) goto fail;
            if (kg_write_props(fp, e->properties, e->prop_count) != 0)
                goto fail;
        }
    }

    for (size_t b = 0; b < kg->relation_bucket_count; b++) {
        for (KG_RelationNode *n = kg->relation_buckets[b]; n; n = n->next) {
            const GV_KGRelation *r = &n->relation;
            if (write_u64(fp, r->relation_id) != 0) goto fail;
            if (write_u64(fp, r->subject_id) != 0) goto fail;
            if (write_u64(fp, r->object_id) != 0) goto fail;
            if (write_string(fp, r->predicate) != 0) goto fail;
            if (write_f32(fp, r->weight) != 0) goto fail;
            if (write_u64(fp, r->created_at) != 0) goto fail;

            size_t pc = 0;
            for (GV_KGProp *p = r->properties; p; p = p->next) pc++;
            if (kg_write_props(fp, r->properties, pc) != 0) goto fail;
        }
    }

    int is_wal_base = 0;
    if (kg->wal_path) {
        size_t wl = strlen(kg->wal_path);
        size_t pl = strlen(path);
        is_wal_base = wl == pl + 4 &&
                      strncmp(kg->wal_path, path, pl) == 0 &&
                      strcmp(kg->wal_path + pl, ".wal") == 0;
    }

    if (fflush(fp) != 0 || fsync(fileno(fp)) != 0) goto fail;

    fclose(fp);
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);

    /* Successful save of the attached snapshot makes the log redundant. */
    if (is_wal_base) {
        pthread_rwlock_wrlock((pthread_rwlock_t *)&kg->rwlock);
        if (kg->wal_file) {
            /* Truncate the WAL by reopening it; if reopen fails the stream is
             * closed by freopen, so drop our dangling handle rather than use it. */
            FILE *reopened = freopen(kg->wal_path, "wb", kg->wal_file);
            ((GV_KnowledgeGraph *)kg)->wal_file = reopened;
            if (kg->wal_file) {
                fflush(kg->wal_file);
                fsync(fileno(kg->wal_file));
            }
        }
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    }
    return 0;

fail:
    fclose(fp);
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return -1;
}


static GV_KGProp *kg_read_props(FILE *fp, uint32_t count) {
    GV_KGProp *head = NULL;
    GV_KGProp *tail = NULL;
    for (uint32_t i = 0; i < count; i++) {
        GV_KGProp *p = (GV_KGProp *)gv_calloc(1, sizeof(GV_KGProp));
        if (!p) return head;
        p->key = read_string(fp);
        char *val_str = read_string(fp);
        p->value = gv_prop_parse(val_str, GV_PROP_STRING);
        gv_free(val_str);
        p->next = NULL;
        if (!p->key || p->value.type == GV_PROP_NULL) {
            gv_free(p->key);
            gv_prop_free(&p->value);
            gv_free(p);
            return head;
        }
        if (tail) tail->next = p;
        else head = p;
        tail = p;
    }
    return head;
}

GV_KnowledgeGraph *kg_load(const char *path) {
    if (!path) return NULL;

    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;

    char magic[KG_MAGIC_LEN];
    if (read_bytes(fp, magic, KG_MAGIC_LEN) != 0 ||
        memcmp(magic, KG_MAGIC, KG_MAGIC_LEN) != 0) {
        fclose(fp);
        return NULL;
    }

    uint32_t version;
    if (read_u32(fp, &version) != 0 || version != KG_VERSION) {
        fclose(fp);
        return NULL;
    }

    uint32_t emb_dim;
    float sim_thresh, lp_thresh;
    if (read_u32(fp, &emb_dim) != 0) { fclose(fp); return NULL; }
    if (read_f32(fp, &sim_thresh) != 0) { fclose(fp); return NULL; }
    if (read_f32(fp, &lp_thresh) != 0) { fclose(fp); return NULL; }

    GV_KGConfig cfg;
    kg_config_init(&cfg);
    cfg.embedding_dimension = (size_t)emb_dim;
    cfg.similarity_threshold = sim_thresh;
    cfg.link_prediction_threshold = lp_thresh;

    uint64_t entity_count, relation_count, next_eid, next_rid;
    if (read_u64(fp, &entity_count) != 0) { fclose(fp); return NULL; }
    if (read_u64(fp, &relation_count) != 0) { fclose(fp); return NULL; }
    if (read_u64(fp, &next_eid) != 0) { fclose(fp); return NULL; }
    if (read_u64(fp, &next_rid) != 0) { fclose(fp); return NULL; }

    GV_KnowledgeGraph *kg = kg_create(&cfg);
    if (!kg) { fclose(fp); return NULL; }
    kg->next_entity_id = next_eid;
    kg->next_relation_id = next_rid;

    for (uint64_t i = 0; i < entity_count; i++) {
        uint64_t eid;
        if (read_u64(fp, &eid) != 0) goto load_fail;

        char *name = read_string(fp);
        char *type = read_string(fp);
        if (!name || !type) {
            gv_free(name);
            gv_free(type);
            goto load_fail;
        }

        uint8_t has_emb;
        if (read_u8(fp, &has_emb) != 0) {
            gv_free(name); gv_free(type);
            goto load_fail;
        }

        float *emb_data = NULL;
        if (has_emb && emb_dim > 0) {
            emb_data = (float *)gv_alloc(emb_dim * sizeof(float));
            if (!emb_data || read_bytes(fp, emb_data,
                emb_dim * sizeof(float)) != 0) {
                gv_free(emb_data); gv_free(name); gv_free(type);
                goto load_fail;
            }
        }

        float confidence;
        uint64_t created_at;
        if (read_f32(fp, &confidence) != 0 ||
            read_u64(fp, &created_at) != 0) {
            gv_free(emb_data); gv_free(name); gv_free(type);
            goto load_fail;
        }

        uint32_t prop_count;
        if (read_u32(fp, &prop_count) != 0) {
            gv_free(emb_data); gv_free(name); gv_free(type);
            goto load_fail;
        }
        GV_KGProp *props = kg_read_props(fp, prop_count);

        KG_EntityNode *node = (KG_EntityNode *)gv_calloc(1,
                                  sizeof(KG_EntityNode));
        if (!node) {
            gv_free(emb_data); gv_free(name); gv_free(type);
            kg_prop_free_list(props);
            goto load_fail;
        }

        GV_KGEntity *e = &node->entity;
        e->entity_id  = eid;
        e->name       = name;
        e->type       = type;
        e->embedding  = emb_data;
        e->dimension  = has_emb ? (size_t)emb_dim : 0;
        e->confidence = confidence;
        e->created_at = created_at;
        e->properties = props;
        e->prop_count = (size_t)prop_count;

        size_t bucket = (size_t)kg_hash_uint64(eid, kg->entity_bucket_count);
        node->next = kg->entity_buckets[bucket];
        kg->entity_buckets[bucket] = node;
        kg->entity_count++;

        kg_str_index_add(kg, kg->name_index, kg->entity_bucket_count, e->name, eid);
        kg_str_index_add(kg, kg->type_index, kg->entity_bucket_count, e->type, eid);

        if (has_emb && emb_data && emb_dim > 0) {
            kg_embedding_add(kg, eid, emb_data, (size_t)emb_dim);
        }
    }

    for (uint64_t i = 0; i < relation_count; i++) {
        uint64_t rid, sid, oid;
        if (read_u64(fp, &rid) != 0 ||
            read_u64(fp, &sid) != 0 ||
            read_u64(fp, &oid) != 0) goto load_fail;

        char *predicate = read_string(fp);
        if (!predicate) goto load_fail;

        float weight;
        uint64_t created_at;
        if (read_f32(fp, &weight) != 0 ||
            read_u64(fp, &created_at) != 0) {
            gv_free(predicate);
            goto load_fail;
        }

        uint32_t prop_count;
        if (read_u32(fp, &prop_count) != 0) {
            gv_free(predicate);
            goto load_fail;
        }
        GV_KGProp *props = kg_read_props(fp, prop_count);

        KG_RelationNode *node = (KG_RelationNode *)gv_calloc(1,
                                    sizeof(KG_RelationNode));
        if (!node) {
            gv_free(predicate);
            kg_prop_free_list(props);
            goto load_fail;
        }

        GV_KGRelation *r = &node->relation;
        r->relation_id = rid;
        r->subject_id  = sid;
        r->object_id   = oid;
        r->predicate   = predicate;
        r->weight      = weight;
        r->created_at  = created_at;
        r->properties  = props;

        size_t bucket = (size_t)kg_hash_uint64(rid, kg->relation_bucket_count);
        node->next = kg->relation_buckets[bucket];
        kg->relation_buckets[bucket] = node;
        kg->relation_count++;

        KG_IndexEntry *se = kg_index_get_or_create(kg->subject_index,
                                                    kg->spo_bucket_count, sid);
        if (se) kg_idlist_push(&se->list, rid);

        KG_IndexEntry *oe = kg_index_get_or_create(kg->object_index,
                                                    kg->spo_bucket_count, oid);
        if (oe) kg_idlist_push(&oe->list, rid);

        uint64_t pred_hash = kg_hash_string(predicate);
        KG_IndexEntry *pe = kg_index_get_or_create(kg->predicate_index,
                                                    kg->spo_bucket_count,
                                                    pred_hash);
        if (pe) kg_idlist_push(&pe->list, rid);

        GV_KGProp *cp = kg_prop_find(r->properties, "chunk_id");
        if (cp && cp->value.type == GV_PROP_STRING && cp->value.as.s) {
            KG_IndexEntry *ce = kg_index_get_or_create(kg->chunk_index,
                                                       kg->spo_bucket_count,
                                                       kg_hash_string(cp->value.as.s));
            if (ce) kg_idlist_push(&ce->list, rid);
        }
    }

    /* Entities and relations are fully loaded; wire up direct adjacency. */
    kg_rebuild_adjacency(kg);

    fclose(fp);

    /* Crash recovery: replay any WAL left over from mutations after the last
     * save, then keep it attached for future appends. */
    if (kg_wal_replay_and_attach(kg, path) != 0) {
        kg_destroy(kg);
        return NULL;
    }
    return kg;

load_fail:
    fclose(fp);
    kg_destroy(kg);
    return NULL;
}
