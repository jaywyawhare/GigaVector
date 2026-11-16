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
#include "features/knowledge_graph_internal.h"

static int kg_idlist_init(KG_IdList *list) {
    list->ids = (uint64_t *)gv_alloc(KG_INITIAL_IDX * sizeof(uint64_t));
    if (!list->ids) return -1;
    list->count = 0;
    list->capacity = KG_INITIAL_IDX;
    return 0;
}

int kg_idlist_push(KG_IdList *list, uint64_t id) {
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

void kg_wal_close(GV_KnowledgeGraph *kg)
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

uint64_t kg_hash_uint64(uint64_t key, size_t buckets) {
    /* Splitmix-style finaliser */
    key ^= key >> 30;
    key *= 0xbf58476d1ce4e5b9ULL;
    key ^= key >> 27;
    key *= 0x94d049bb133111ebULL;
    key ^= key >> 31;
    return buckets ? key % buckets : 0;
}

uint64_t kg_hash_string(const char *str) {
    uint64_t h = 14695981039346656037ULL; /* FNV-1a offset basis */
    while (*str) {
        h ^= (uint64_t)(unsigned char)(*str++);
        h *= 1099511628211ULL; /* FNV-1a prime */
    }
    return h;
}

/** @brief Hash of a property value's string form (used for chunk-index keying). */
uint64_t kg_prop_hash(GV_PropValue v) {
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

int kg_prop_remove(GV_KGProp **head, size_t *count, const char *key) {
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

void kg_prop_free_list(GV_KGProp *head) {
    while (head) {
        GV_KGProp *next = head->next;
        gv_free(head->key);
        gv_prop_free(&head->value);
        gv_free(head);
        head = next;
    }
}

GV_KGProp *kg_prop_find(GV_KGProp *head, const char *key) {
    for (GV_KGProp *p = head; p; p = p->next) {
        if (p->key && strcmp(p->key, key) == 0) return p;
    }
    return NULL;
}

int kg_prop_set(GV_KGProp **head, size_t *count,
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



KG_EntityNode *kg_find_entity_node(const GV_KnowledgeGraph *kg,
                                           uint64_t entity_id) {
    size_t idx = (size_t)kg_hash_uint64(entity_id, kg->entity_bucket_count);
    for (KG_EntityNode *n = kg->entity_buckets[idx]; n; n = n->next) {
        if (n->entity.entity_id == entity_id) return n;
    }
    return NULL;
}

KG_RelationNode *kg_find_relation_node(const GV_KnowledgeGraph *kg,
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

void kg_entity_adjacency_free(KG_EntityNode *n) {
    gv_free(n->out_edges);
    gv_free(n->in_edges);
    n->out_edges = n->in_edges = NULL;
    n->out_count = n->out_cap = 0;
    n->in_count = n->in_cap = 0;
}

/* Rebuild every node's adjacency from the relation table. O(entities + edges).
 * Used after bulk load and after structural rewrites (merge) where incremental
 * maintenance would be error-prone. */
void kg_rebuild_adjacency(GV_KnowledgeGraph *kg) {
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

KG_IndexEntry *kg_index_find(KG_IndexEntry **table,
                                     size_t buckets, uint64_t key) {
    size_t idx = (size_t)(key % buckets);
    for (KG_IndexEntry *e = table[idx]; e; e = e->next) {
        if (e->key == key) return e;
    }
    return NULL;
}

KG_IndexEntry *kg_index_get_or_create(KG_IndexEntry **table,
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

void kg_index_remove_id(KG_IndexEntry **table, size_t buckets,
                                uint64_t key, uint64_t relation_id) {
    KG_IndexEntry *e = kg_index_find(table, buckets, key);
    if (e) kg_idlist_remove(&e->list, relation_id);
}

/* hash(string)-keyed index maintenance shared by the name/type/chunk indexes */
void kg_str_index_add(GV_KnowledgeGraph *kg, KG_IndexEntry **table,
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

int kg_embedding_add(GV_KnowledgeGraph *kg, uint64_t entity_id,
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

void kg_embedding_remove(GV_KnowledgeGraph *kg, uint64_t entity_id) {
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

const float *kg_embedding_get(const GV_KnowledgeGraph *kg,
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

void kg_entity_data_free(GV_KGEntity *e) {
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

size_t kg_collect_relations_for_entity(const GV_KnowledgeGraph *kg,
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

int kg_remove_relation_internal(GV_KnowledgeGraph *kg,
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
int kg_insert_entity_with_id(GV_KnowledgeGraph *kg, uint64_t eid,
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

int kg_remove_entity_internal(GV_KnowledgeGraph *kg, uint64_t entity_id) {
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
int kg_insert_relation_with_id(GV_KnowledgeGraph *kg, uint64_t rid,
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
void kg_fill_triple(const GV_KnowledgeGraph *kg,
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
