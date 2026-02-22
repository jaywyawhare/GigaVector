#define _POSIX_C_SOURCE 200112L

/**
 * @file gv_knowledge_graph.c
 * @brief Knowledge graph implementation: entity/relation storage, SPO triple
 *        queries, cosine-similarity search, entity resolution, link prediction,
 *        BFS traversal, subgraph extraction, hybrid search, and persistence.
 */

#include "gigavector/gv_knowledge_graph.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

/*
 * Internal Constants
 */

#define KG_MAGIC         "GVKG"
#define KG_MAGIC_LEN     4
#define KG_VERSION       1
#define KG_INITIAL_IDX   64   /* initial capacity for index lists */

/*
 * Internal: Index List (dynamic array of relation IDs)
 */

typedef struct {
    uint64_t *ids;
    size_t    count;
    size_t    capacity;
} KG_IdList;

static int kg_idlist_init(KG_IdList *list) {
    list->ids = (uint64_t *)malloc(KG_INITIAL_IDX * sizeof(uint64_t));
    if (!list->ids) return -1;
    list->count = 0;
    list->capacity = KG_INITIAL_IDX;
    return 0;
}

static int kg_idlist_push(KG_IdList *list, uint64_t id) {
    if (list->count >= list->capacity) {
        size_t new_cap = list->capacity * 2;
        uint64_t *tmp = (uint64_t *)realloc(list->ids,
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
    free(list->ids);
    list->ids = NULL;
    list->count = 0;
    list->capacity = 0;
}

/*
 * Internal: Entity Hash Table Node
 */

typedef struct KG_EntityNode {
    GV_KGEntity             entity;
    struct KG_EntityNode   *next;
} KG_EntityNode;

/*
 * Internal: Relation Hash Table Node
 */

typedef struct KG_RelationNode {
    GV_KGRelation             relation;
    struct KG_RelationNode   *next;
} KG_RelationNode;

/*
 * Internal: SPO Index Entry (hash on key -> list of relation IDs)
 */

typedef struct KG_IndexEntry {
    uint64_t               key;     /* entity_id or hash(predicate) */
    KG_IdList              list;
    struct KG_IndexEntry  *next;
} KG_IndexEntry;

/*
 * Internal: Knowledge Graph Structure
 */

struct GV_KnowledgeGraph {
    /* Configuration */
    GV_KGConfig config;

    /* Entity storage */
    KG_EntityNode  **entity_buckets;
    size_t           entity_bucket_count;
    size_t           entity_count;
    uint64_t         next_entity_id;

    /* Relation storage */
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
    size_t          spo_bucket_count;

    /* Embedding flat array for fast similarity scan */
    float     *all_embeddings;       /* dim * embedding_cap floats */
    uint64_t  *embedding_entity_ids;
    size_t     embedding_count;
    size_t     embedding_cap;

    /* Thread safety */
    pthread_rwlock_t rwlock;
};

/*
 * Internal: Utility Helpers
 */

static char *kg_strdup(const char *s) {
    if (!s) return NULL;
    size_t len = strlen(s);
    char *dup = (char *)malloc(len + 1);
    if (dup) memcpy(dup, s, len + 1);
    return dup;
}

static uint64_t kg_hash_uint64(uint64_t key, size_t buckets) {
    /* Splitmix-style finaliser */
    key ^= key >> 30;
    key *= 0xbf58476d1ce4e5b9ULL;
    key ^= key >> 27;
    key *= 0x94d049bb133111ebULL;
    key ^= key >> 31;
    return key % buckets;
}

static uint64_t kg_hash_string(const char *str) {
    uint64_t h = 14695981039346656037ULL; /* FNV-1a offset basis */
    while (*str) {
        h ^= (uint64_t)(unsigned char)(*str++);
        h *= 1099511628211ULL; /* FNV-1a prime */
    }
    return h;
}

static uint64_t kg_now_epoch(void) {
    return (uint64_t)time(NULL);
}

/*
 * Internal: Property Helpers
 */

static GV_KGProp *kg_prop_clone_list(const GV_KGProp *src) {
    GV_KGProp *head = NULL;
    GV_KGProp *tail = NULL;
    for (const GV_KGProp *p = src; p; p = p->next) {
        GV_KGProp *node = (GV_KGProp *)calloc(1, sizeof(GV_KGProp));
        if (!node) return head; /* partial clone on OOM */
        node->key = kg_strdup(p->key);
        node->value = kg_strdup(p->value);
        node->next = NULL;
        if (tail) tail->next = node;
        else head = node;
        tail = node;
    }
    return head;
}

static void kg_prop_free_list(GV_KGProp *head) {
    while (head) {
        GV_KGProp *next = head->next;
        free(head->key);
        free(head->value);
        free(head);
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
        char *dup = kg_strdup(value);
        if (!dup) return -1;
        free(existing->value);
        existing->value = dup;
        return 0;
    }
    GV_KGProp *node = (GV_KGProp *)calloc(1, sizeof(GV_KGProp));
    if (!node) return -1;
    node->key = kg_strdup(key);
    node->value = kg_strdup(value);
    if (!node->key || !node->value) {
        free(node->key);
        free(node->value);
        free(node);
        return -1;
    }
    node->next = *head;
    *head = node;
    (*count)++;
    return 0;
}

/*
 * Internal: Cosine Similarity
 */

static float kg_cosine_similarity(const float *a, const float *b, size_t dim) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    if (na < 1e-12f || nb < 1e-12f) return 0.0f;
    return dot / (sqrtf(na) * sqrtf(nb));
}

/*
 * Internal: Entity Lookup
 */

static KG_EntityNode *kg_find_entity_node(const GV_KnowledgeGraph *kg,
                                           uint64_t entity_id) {
    size_t idx = (size_t)kg_hash_uint64(entity_id, kg->entity_bucket_count);
    for (KG_EntityNode *n = kg->entity_buckets[idx]; n; n = n->next) {
        if (n->entity.entity_id == entity_id) return n;
    }
    return NULL;
}

/*
 * Internal: Relation Lookup
 */

static KG_RelationNode *kg_find_relation_node(const GV_KnowledgeGraph *kg,
                                               uint64_t relation_id) {
    size_t idx = (size_t)kg_hash_uint64(relation_id,
                                         kg->relation_bucket_count);
    for (KG_RelationNode *n = kg->relation_buckets[idx]; n; n = n->next) {
        if (n->relation.relation_id == relation_id) return n;
    }
    return NULL;
}

/*
 * Internal: SPO Index Helpers
 */

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
    KG_IndexEntry *entry = (KG_IndexEntry *)calloc(1, sizeof(KG_IndexEntry));
    if (!entry) return NULL;
    entry->key = key;
    if (kg_idlist_init(&entry->list) != 0) {
        free(entry);
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

static void kg_index_free_table(KG_IndexEntry **table, size_t buckets) {
    if (!table) return;
    for (size_t i = 0; i < buckets; i++) {
        KG_IndexEntry *e = table[i];
        while (e) {
            KG_IndexEntry *next = e->next;
            kg_idlist_free(&e->list);
            free(e);
            e = next;
        }
    }
    free(table);
}

/*
 * Internal: Embedding Store Helpers
 */

static int kg_embedding_add(GV_KnowledgeGraph *kg, uint64_t entity_id,
                             const float *emb, size_t dim) {
    if (dim != kg->config.embedding_dimension) return -1;
    if (kg->embedding_count >= kg->embedding_cap) {
        size_t new_cap = kg->embedding_cap == 0 ? 256 : kg->embedding_cap * 2;
        float *new_emb = (float *)realloc(kg->all_embeddings,
                                           new_cap * dim * sizeof(float));
        uint64_t *new_ids = (uint64_t *)realloc(kg->embedding_entity_ids,
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
            /* Swap with last */
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

/*
 * Internal: Free a single entity node's heap data (NOT the node itself)
 */

static void kg_entity_data_free(GV_KGEntity *e) {
    free(e->name);
    free(e->type);
    free(e->embedding);
    kg_prop_free_list(e->properties);
    e->name = NULL;
    e->type = NULL;
    e->embedding = NULL;
    e->properties = NULL;
}

/*
 * Internal: Free a single relation node's heap data
 */

static void kg_relation_data_free(GV_KGRelation *r) {
    free(r->predicate);
    kg_prop_free_list(r->properties);
    r->predicate = NULL;
    r->properties = NULL;
}

/*
 * Internal: Collect all relation IDs for an entity (subject or object)
 */

static size_t kg_collect_relations_for_entity(const GV_KnowledgeGraph *kg,
                                               uint64_t entity_id,
                                               uint64_t **out_ids) {
    size_t total = 0;
    size_t cap = 64;
    uint64_t *ids = (uint64_t *)malloc(cap * sizeof(uint64_t));
    if (!ids) { *out_ids = NULL; return 0; }

    /* From subject index */
    KG_IndexEntry *se = kg_index_find(kg->subject_index,
                                       kg->spo_bucket_count, entity_id);
    if (se) {
        for (size_t i = 0; i < se->list.count; i++) {
            if (total >= cap) {
                cap *= 2;
                uint64_t *tmp = (uint64_t *)realloc(ids,
                                                      cap * sizeof(uint64_t));
                if (!tmp) break;
                ids = tmp;
            }
            ids[total++] = se->list.ids[i];
        }
    }

    /* From object index */
    KG_IndexEntry *oe = kg_index_find(kg->object_index,
                                       kg->spo_bucket_count, entity_id);
    if (oe) {
        for (size_t i = 0; i < oe->list.count; i++) {
            /* Avoid duplicates (self-loops) */
            int dup = 0;
            for (size_t j = 0; j < total; j++) {
                if (ids[j] == oe->list.ids[i]) { dup = 1; break; }
            }
            if (dup) continue;
            if (total >= cap) {
                cap *= 2;
                uint64_t *tmp = (uint64_t *)realloc(ids,
                                                      cap * sizeof(uint64_t));
                if (!tmp) break;
                ids = tmp;
            }
            ids[total++] = oe->list.ids[i];
        }
    }

    *out_ids = ids;
    return total;
}

/*
 * Internal: Remove a single relation (no lock)
 */

static int kg_remove_relation_internal(GV_KnowledgeGraph *kg,
                                        uint64_t relation_id) {
    size_t idx = (size_t)kg_hash_uint64(relation_id,
                                         kg->relation_bucket_count);
    KG_RelationNode *prev = NULL;
    for (KG_RelationNode *n = kg->relation_buckets[idx]; n; n = n->next) {
        if (n->relation.relation_id == relation_id) {
            /* Remove from SPO indexes */
            kg_index_remove_id(kg->subject_index, kg->spo_bucket_count,
                               n->relation.subject_id, relation_id);
            kg_index_remove_id(kg->object_index, kg->spo_bucket_count,
                               n->relation.object_id, relation_id);
            uint64_t pred_hash = kg_hash_string(n->relation.predicate);
            kg_index_remove_id(kg->predicate_index, kg->spo_bucket_count,
                               pred_hash, relation_id);

            /* Unlink and free */
            if (prev) prev->next = n->next;
            else kg->relation_buckets[idx] = n->next;
            kg_relation_data_free(&n->relation);
            free(n);
            kg->relation_count--;
            return 0;
        }
        prev = n;
    }
    return -1;
}

/*
 * Internal: Check if two entities are directly connected
 */

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

/*
 * Internal: Count shared neighbours between two entities
 */

static size_t kg_shared_neighbors(const GV_KnowledgeGraph *kg,
                                   uint64_t a, uint64_t b) {
    uint64_t *na = NULL, *nb = NULL;
    size_t ca = kg_collect_relations_for_entity(kg, a, &na);
    size_t cb = kg_collect_relations_for_entity(kg, b, &nb);

    /* Build neighbor sets from relations */
    size_t na_cap = 64, nb_cap = 64;
    uint64_t *neigh_a = (uint64_t *)malloc(na_cap * sizeof(uint64_t));
    uint64_t *neigh_b = (uint64_t *)malloc(nb_cap * sizeof(uint64_t));
    size_t neigh_a_count = 0, neigh_b_count = 0;

    if (!neigh_a || !neigh_b) {
        free(na); free(nb); free(neigh_a); free(neigh_b);
        return 0;
    }

    for (size_t i = 0; i < ca; i++) {
        KG_RelationNode *rn = kg_find_relation_node(kg, na[i]);
        if (!rn) continue;
        uint64_t other = (rn->relation.subject_id == a) ?
                          rn->relation.object_id : rn->relation.subject_id;
        if (neigh_a_count >= na_cap) {
            na_cap *= 2;
            uint64_t *tmp = (uint64_t *)realloc(neigh_a,
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
            uint64_t *tmp = (uint64_t *)realloc(neigh_b,
                                                  nb_cap * sizeof(uint64_t));
            if (!tmp) break;
            neigh_b = tmp;
        }
        neigh_b[neigh_b_count++] = other;
    }

    /* Count intersection */
    size_t shared = 0;
    for (size_t i = 0; i < neigh_a_count; i++) {
        for (size_t j = 0; j < neigh_b_count; j++) {
            if (neigh_a[i] == neigh_b[j]) { shared++; break; }
        }
    }

    free(na); free(nb);
    free(neigh_a); free(neigh_b);
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
    bfs->visited = (uint64_t *)malloc(cap * sizeof(uint64_t));
    bfs->depths  = (size_t *)malloc(cap * sizeof(size_t));
    bfs->parent  = (uint64_t *)calloc(cap, sizeof(uint64_t));
    if (!bfs->visited || !bfs->depths || !bfs->parent) {
        free(bfs->visited); free(bfs->depths); free(bfs->parent);
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
        uint64_t *v = (uint64_t *)realloc(bfs->visited,
                                            new_cap * sizeof(uint64_t));
        size_t *d = (size_t *)realloc(bfs->depths,
                                       new_cap * sizeof(size_t));
        uint64_t *p = (uint64_t *)realloc(bfs->parent,
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
    free(bfs->visited);
    free(bfs->depths);
    free(bfs->parent);
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

        /* Collect neighbour entity IDs from subject index */
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

        /* From object index (incoming edges) */
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

/*
 * Internal: helper for sorting search results by similarity descending
 */

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

/*
 * Lifecycle
 */

void gv_kg_config_init(GV_KGConfig *config) {
    if (!config) return;
    config->entity_bucket_count     = 4096;
    config->relation_bucket_count   = 8192;
    config->embedding_dimension     = 128;
    config->similarity_threshold    = 0.7f;
    config->link_prediction_threshold = 0.8f;
    config->max_entities            = 1000000;
}

GV_KnowledgeGraph *gv_kg_create(const GV_KGConfig *config) {
    GV_KGConfig cfg;
    if (config) {
        cfg = *config;
    } else {
        gv_kg_config_init(&cfg);
    }

    GV_KnowledgeGraph *kg = (GV_KnowledgeGraph *)calloc(1,
                                sizeof(GV_KnowledgeGraph));
    if (!kg) return NULL;

    kg->config = cfg;
    kg->entity_bucket_count  = cfg.entity_bucket_count;
    kg->relation_bucket_count = cfg.relation_bucket_count;
    kg->spo_bucket_count = cfg.relation_bucket_count;
    kg->next_entity_id = 1;
    kg->next_relation_id = 1;

    /* Allocate entity hash table */
    kg->entity_buckets = (KG_EntityNode **)calloc(kg->entity_bucket_count,
                                                   sizeof(KG_EntityNode *));
    if (!kg->entity_buckets) goto fail;

    /* Allocate relation hash table */
    kg->relation_buckets = (KG_RelationNode **)calloc(
        kg->relation_bucket_count, sizeof(KG_RelationNode *));
    if (!kg->relation_buckets) goto fail;

    /* Allocate SPO indexes */
    kg->subject_index = (KG_IndexEntry **)calloc(kg->spo_bucket_count,
                                                  sizeof(KG_IndexEntry *));
    kg->object_index = (KG_IndexEntry **)calloc(kg->spo_bucket_count,
                                                 sizeof(KG_IndexEntry *));
    kg->predicate_index = (KG_IndexEntry **)calloc(kg->spo_bucket_count,
                                                    sizeof(KG_IndexEntry *));
    if (!kg->subject_index || !kg->object_index || !kg->predicate_index)
        goto fail;

    /* Init rwlock */
    if (pthread_rwlock_init(&kg->rwlock, NULL) != 0) goto fail;

    return kg;

fail:
    free(kg->entity_buckets);
    free(kg->relation_buckets);
    free(kg->subject_index);
    free(kg->object_index);
    free(kg->predicate_index);
    free(kg);
    return NULL;
}

void gv_kg_destroy(GV_KnowledgeGraph *kg) {
    if (!kg) return;

    /* Free entity nodes */
    for (size_t i = 0; i < kg->entity_bucket_count; i++) {
        KG_EntityNode *n = kg->entity_buckets[i];
        while (n) {
            KG_EntityNode *next = n->next;
            kg_entity_data_free(&n->entity);
            free(n);
            n = next;
        }
    }
    free(kg->entity_buckets);

    /* Free relation nodes */
    for (size_t i = 0; i < kg->relation_bucket_count; i++) {
        KG_RelationNode *n = kg->relation_buckets[i];
        while (n) {
            KG_RelationNode *next = n->next;
            kg_relation_data_free(&n->relation);
            free(n);
            n = next;
        }
    }
    free(kg->relation_buckets);

    /* Free SPO indexes */
    kg_index_free_table(kg->subject_index, kg->spo_bucket_count);
    kg_index_free_table(kg->object_index, kg->spo_bucket_count);
    kg_index_free_table(kg->predicate_index, kg->spo_bucket_count);

    /* Free embedding storage */
    free(kg->all_embeddings);
    free(kg->embedding_entity_ids);

    pthread_rwlock_destroy(&kg->rwlock);
    free(kg);
}

/*
 * Entity Operations
 */

uint64_t gv_kg_add_entity(GV_KnowledgeGraph *kg, const char *name,
                           const char *type, const float *embedding,
                           size_t dimension) {
    if (!kg || !name || !type) return 0;

    pthread_rwlock_wrlock(&kg->rwlock);

    if (kg->entity_count >= kg->config.max_entities) {
        pthread_rwlock_unlock(&kg->rwlock);
        return 0;
    }

    KG_EntityNode *node = (KG_EntityNode *)calloc(1, sizeof(KG_EntityNode));
    if (!node) {
        pthread_rwlock_unlock(&kg->rwlock);
        return 0;
    }

    uint64_t eid = kg->next_entity_id++;
    GV_KGEntity *e = &node->entity;
    e->entity_id  = eid;
    e->name       = kg_strdup(name);
    e->type       = kg_strdup(type);
    e->properties = NULL;
    e->prop_count = 0;
    e->created_at = kg_now_epoch();
    e->confidence = 1.0f;

    if (!e->name || !e->type) {
        free(e->name);
        free(e->type);
        free(node);
        pthread_rwlock_unlock(&kg->rwlock);
        return 0;
    }

    /* Copy embedding into entity and into flat array */
    if (embedding && dimension > 0 && kg->config.embedding_dimension > 0) {
        e->embedding = (float *)malloc(dimension * sizeof(float));
        if (e->embedding) {
            memcpy(e->embedding, embedding, dimension * sizeof(float));
            e->dimension = dimension;
            kg_embedding_add(kg, eid, embedding, dimension);
        }
    }

    /* Insert into hash table */
    size_t bucket = (size_t)kg_hash_uint64(eid, kg->entity_bucket_count);
    node->next = kg->entity_buckets[bucket];
    kg->entity_buckets[bucket] = node;
    kg->entity_count++;

    pthread_rwlock_unlock(&kg->rwlock);
    return eid;
}

int gv_kg_remove_entity(GV_KnowledgeGraph *kg, uint64_t entity_id) {
    if (!kg) return -1;

    pthread_rwlock_wrlock(&kg->rwlock);

    /* Cascade-delete all relations involving this entity */
    uint64_t *rel_ids = NULL;
    size_t rel_count = kg_collect_relations_for_entity(kg, entity_id,
                                                        &rel_ids);
    for (size_t i = 0; i < rel_count; i++) {
        kg_remove_relation_internal(kg, rel_ids[i]);
    }
    free(rel_ids);

    /* Remove embedding */
    kg_embedding_remove(kg, entity_id);

    /* Remove entity from hash table */
    size_t bucket = (size_t)kg_hash_uint64(entity_id,
                                            kg->entity_bucket_count);
    KG_EntityNode *prev = NULL;
    for (KG_EntityNode *n = kg->entity_buckets[bucket]; n; n = n->next) {
        if (n->entity.entity_id == entity_id) {
            if (prev) prev->next = n->next;
            else kg->entity_buckets[bucket] = n->next;
            kg_entity_data_free(&n->entity);
            free(n);
            kg->entity_count--;
            pthread_rwlock_unlock(&kg->rwlock);
            return 0;
        }
        prev = n;
    }

    pthread_rwlock_unlock(&kg->rwlock);
    return -1;
}

const GV_KGEntity *gv_kg_get_entity(const GV_KnowledgeGraph *kg,
                                     uint64_t entity_id) {
    if (!kg) return NULL;
    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);
    KG_EntityNode *n = kg_find_entity_node(kg, entity_id);
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return n ? &n->entity : NULL;
}

int gv_kg_set_entity_prop(GV_KnowledgeGraph *kg, uint64_t entity_id,
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
    pthread_rwlock_unlock(&kg->rwlock);
    return rc;
}

const char *gv_kg_get_entity_prop(const GV_KnowledgeGraph *kg,
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
    return p ? p->value : NULL;
}

int gv_kg_find_entities_by_type(const GV_KnowledgeGraph *kg, const char *type,
                                 uint64_t *out_ids, size_t max_count) {
    if (!kg || !type || !out_ids || max_count == 0) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);
    size_t found = 0;
    for (size_t i = 0; i < kg->entity_bucket_count && found < max_count; i++) {
        for (KG_EntityNode *n = kg->entity_buckets[i];
             n && found < max_count; n = n->next) {
            if (n->entity.type && strcmp(n->entity.type, type) == 0) {
                out_ids[found++] = n->entity.entity_id;
            }
        }
    }
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return (int)found;
}

int gv_kg_find_entities_by_name(const GV_KnowledgeGraph *kg, const char *name,
                                 uint64_t *out_ids, size_t max_count) {
    if (!kg || !name || !out_ids || max_count == 0) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);
    size_t found = 0;
    for (size_t i = 0; i < kg->entity_bucket_count && found < max_count; i++) {
        for (KG_EntityNode *n = kg->entity_buckets[i];
             n && found < max_count; n = n->next) {
            if (n->entity.name && strcmp(n->entity.name, name) == 0) {
                out_ids[found++] = n->entity.entity_id;
            }
        }
    }
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return (int)found;
}

/*
 * Relation (Triple) Operations
 */

uint64_t gv_kg_add_relation(GV_KnowledgeGraph *kg, uint64_t subject,
                             const char *predicate, uint64_t object,
                             float weight) {
    if (!kg || !predicate) return 0;

    pthread_rwlock_wrlock(&kg->rwlock);

    /* Validate that both entities exist */
    if (!kg_find_entity_node(kg, subject) ||
        !kg_find_entity_node(kg, object)) {
        pthread_rwlock_unlock(&kg->rwlock);
        return 0;
    }

    KG_RelationNode *node = (KG_RelationNode *)calloc(1,
                                sizeof(KG_RelationNode));
    if (!node) {
        pthread_rwlock_unlock(&kg->rwlock);
        return 0;
    }

    uint64_t rid = kg->next_relation_id++;
    GV_KGRelation *r = &node->relation;
    r->relation_id = rid;
    r->subject_id  = subject;
    r->object_id   = object;
    r->predicate   = kg_strdup(predicate);
    r->weight      = weight;
    r->properties  = NULL;
    r->created_at  = kg_now_epoch();

    if (!r->predicate) {
        free(node);
        pthread_rwlock_unlock(&kg->rwlock);
        return 0;
    }

    /* Insert into relation hash table */
    size_t bucket = (size_t)kg_hash_uint64(rid, kg->relation_bucket_count);
    node->next = kg->relation_buckets[bucket];
    kg->relation_buckets[bucket] = node;
    kg->relation_count++;

    /* Update SPO indexes */
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

    pthread_rwlock_unlock(&kg->rwlock);
    return rid;
}

int gv_kg_remove_relation(GV_KnowledgeGraph *kg, uint64_t relation_id) {
    if (!kg) return -1;
    pthread_rwlock_wrlock(&kg->rwlock);
    int rc = kg_remove_relation_internal(kg, relation_id);
    pthread_rwlock_unlock(&kg->rwlock);
    return rc;
}

const GV_KGRelation *gv_kg_get_relation(const GV_KnowledgeGraph *kg,
                                         uint64_t relation_id) {
    if (!kg) return NULL;
    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);
    KG_RelationNode *n = kg_find_relation_node(kg, relation_id);
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return n ? &n->relation : NULL;
}

int gv_kg_set_relation_prop(GV_KnowledgeGraph *kg, uint64_t relation_id,
                             const char *key, const char *value) {
    if (!kg || !key || !value) return -1;

    pthread_rwlock_wrlock(&kg->rwlock);
    KG_RelationNode *n = kg_find_relation_node(kg, relation_id);
    if (!n) {
        pthread_rwlock_unlock(&kg->rwlock);
        return -1;
    }
    /* Relation does not track prop_count in the public struct, use a local */
    size_t dummy_count = 0;
    GV_KGProp *p = n->relation.properties;
    while (p) { dummy_count++; p = p->next; }
    int rc = kg_prop_set(&n->relation.properties, &dummy_count, key, value);
    pthread_rwlock_unlock(&kg->rwlock);
    return rc;
}

/*
 * Triple Store Queries (SPO Pattern Matching)
 */

/**
 * @brief Internal helper: fill a GV_KGTriple from a relation.
 */
static void kg_fill_triple(const GV_KnowledgeGraph *kg,
                            const GV_KGRelation *rel, GV_KGTriple *t) {
    t->subject_id = rel->subject_id;
    t->object_id  = rel->object_id;
    t->predicate  = kg_strdup(rel->predicate);
    t->score      = rel->weight;

    KG_EntityNode *sn = kg_find_entity_node(kg, rel->subject_id);
    t->subject_name = sn ? kg_strdup(sn->entity.name) : kg_strdup("?");

    KG_EntityNode *on = kg_find_entity_node(kg, rel->object_id);
    t->object_name = on ? kg_strdup(on->entity.name) : kg_strdup("?");
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

int gv_kg_query_triples(const GV_KnowledgeGraph *kg, const uint64_t *subject,
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

void gv_kg_free_triples(GV_KGTriple *triples, size_t count) {
    if (!triples) return;
    for (size_t i = 0; i < count; i++) {
        free(triples[i].subject_name);
        free(triples[i].predicate);
        free(triples[i].object_name);
    }
}

/*
 * Semantic Search (Vector-Based)
 */

int gv_kg_search_similar(const GV_KnowledgeGraph *kg,
                          const float *query_embedding, size_t dimension,
                          size_t k, GV_KGSearchResult *results) {
    if (!kg || !query_embedding || !results || k == 0) return -1;
    if (kg->config.embedding_dimension == 0) return 0;
    if (dimension != kg->config.embedding_dimension) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);

    size_t n = kg->embedding_count;
    if (n == 0) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return 0;
    }

    /* Compute similarities */
    KG_ScorePair *pairs = (KG_ScorePair *)malloc(n * sizeof(KG_ScorePair));
    if (!pairs) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return -1;
    }

    size_t dim = kg->config.embedding_dimension;
    for (size_t i = 0; i < n; i++) {
        pairs[i].id = kg->embedding_entity_ids[i];
        pairs[i].score = kg_cosine_similarity(query_embedding,
                                               kg->all_embeddings + i * dim,
                                               dim);
    }

    qsort(pairs, n, sizeof(KG_ScorePair), kg_score_cmp_desc);

    size_t result_count = (k < n) ? k : n;
    for (size_t i = 0; i < result_count; i++) {
        KG_EntityNode *en = kg_find_entity_node(kg, pairs[i].id);
        results[i].entity_id  = pairs[i].id;
        results[i].name       = en ? kg_strdup(en->entity.name) : NULL;
        results[i].type       = en ? kg_strdup(en->entity.type) : NULL;
        results[i].similarity = pairs[i].score;
    }

    free(pairs);
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return (int)result_count;
}

int gv_kg_search_by_text(const GV_KnowledgeGraph *kg, const char *text,
                          const float *text_embedding, size_t dimension,
                          size_t k, GV_KGSearchResult *results) {
    if (!kg || !text || !results || k == 0) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);

    /* Collect candidates: name substring match + embedding similarity */
    size_t cap = 256;
    KG_ScorePair *pairs = (KG_ScorePair *)malloc(cap * sizeof(KG_ScorePair));
    if (!pairs) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return -1;
    }
    size_t pair_count = 0;

    size_t dim = kg->config.embedding_dimension;

    for (size_t b = 0; b < kg->entity_bucket_count; b++) {
        for (KG_EntityNode *n = kg->entity_buckets[b]; n; n = n->next) {
            float score = 0.0f;
            /* Name match component (exact match gets high boost) */
            if (n->entity.name) {
                if (strcmp(n->entity.name, text) == 0) {
                    score += 1.0f;
                } else if (strstr(n->entity.name, text)) {
                    score += 0.5f;
                }
            }
            /* Embedding similarity component */
            if (text_embedding && dim > 0 && dimension == dim) {
                const float *emb = kg_embedding_get(kg,
                    n->entity.entity_id, NULL);
                if (emb) {
                    float sim = kg_cosine_similarity(text_embedding, emb, dim);
                    score += sim;
                }
            }
            if (score > 0.0f) {
                if (pair_count >= cap) {
                    cap *= 2;
                    KG_ScorePair *tmp = (KG_ScorePair *)realloc(pairs,
                        cap * sizeof(KG_ScorePair));
                    if (!tmp) break;
                    pairs = tmp;
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
        results[i].name       = en ? kg_strdup(en->entity.name) : NULL;
        results[i].type       = en ? kg_strdup(en->entity.type) : NULL;
        results[i].similarity = pairs[i].score;
    }

    free(pairs);
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return (int)result_count;
}

void gv_kg_free_search_results(GV_KGSearchResult *results, size_t count) {
    if (!results) return;
    for (size_t i = 0; i < count; i++) {
        free(results[i].name);
        free(results[i].type);
    }
}

/*
 * Entity Resolution / Deduplication
 */

int gv_kg_resolve_entity(GV_KnowledgeGraph *kg, const char *name,
                          const char *type, const float *embedding,
                          size_t dimension) {
    if (!kg || !name || !type) return 0;

    pthread_rwlock_wrlock(&kg->rwlock);

    /* Step 1: exact name match among entities of same type */
    for (size_t b = 0; b < kg->entity_bucket_count; b++) {
        for (KG_EntityNode *n = kg->entity_buckets[b]; n; n = n->next) {
            if (n->entity.name && n->entity.type &&
                strcmp(n->entity.name, name) == 0 &&
                strcmp(n->entity.type, type) == 0) {
                uint64_t eid = n->entity.entity_id;
                pthread_rwlock_unlock(&kg->rwlock);
                return (int)eid;
            }
        }
    }

    /* Step 2: embedding similarity with entities of same type */
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

            float sim = kg_cosine_similarity(embedding,
                kg->all_embeddings + i * dim, dim);
            if (sim > best_sim) {
                best_sim = sim;
                best_id = eid;
            }
        }

        if (best_sim >= kg->config.similarity_threshold && best_id != 0) {
            pthread_rwlock_unlock(&kg->rwlock);
            return (int)best_id;
        }
    }

    /* Step 3: create new entity (unlock first, re-acquire via add) */
    pthread_rwlock_unlock(&kg->rwlock);
    uint64_t new_id = gv_kg_add_entity(kg, name, type, embedding, dimension);
    return (int)new_id;
}

int gv_kg_find_duplicates(const GV_KnowledgeGraph *kg, float threshold,
                           GV_KGLinkPrediction *out, size_t max_count) {
    if (!kg || !out || max_count == 0) return -1;
    if (kg->config.embedding_dimension == 0) return 0;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);

    size_t found = 0;
    size_t dim = kg->config.embedding_dimension;
    size_t n = kg->embedding_count;

    for (size_t i = 0; i < n && found < max_count; i++) {
        for (size_t j = i + 1; j < n && found < max_count; j++) {
            float sim = kg_cosine_similarity(
                kg->all_embeddings + i * dim,
                kg->all_embeddings + j * dim, dim);
            if (sim >= threshold) {
                out[found].entity_a = kg->embedding_entity_ids[i];
                out[found].entity_b = kg->embedding_entity_ids[j];
                out[found].predicted_predicate = kg_strdup("duplicate");
                out[found].confidence = sim;
                found++;
            }
        }
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return (int)found;
}

int gv_kg_merge_entities(GV_KnowledgeGraph *kg, uint64_t keep_id,
                          uint64_t merge_id) {
    if (!kg || keep_id == merge_id) return -1;

    pthread_rwlock_wrlock(&kg->rwlock);

    KG_EntityNode *keep_node  = kg_find_entity_node(kg, keep_id);
    KG_EntityNode *merge_node = kg_find_entity_node(kg, merge_id);
    if (!keep_node || !merge_node) {
        pthread_rwlock_unlock(&kg->rwlock);
        return -1;
    }

    /* Copy properties from merge to keep (don't overwrite existing) */
    for (GV_KGProp *p = merge_node->entity.properties; p; p = p->next) {
        if (!kg_prop_find(keep_node->entity.properties, p->key)) {
            kg_prop_set(&keep_node->entity.properties,
                        &keep_node->entity.prop_count, p->key, p->value);
        }
    }

    /* Re-point all relations from merge_id to keep_id */
    uint64_t *rel_ids = NULL;
    size_t rel_count = kg_collect_relations_for_entity(kg, merge_id,
                                                        &rel_ids);
    for (size_t i = 0; i < rel_count; i++) {
        KG_RelationNode *rn = kg_find_relation_node(kg, rel_ids[i]);
        if (!rn) continue;

        uint64_t rid = rn->relation.relation_id;

        /* Update subject index */
        if (rn->relation.subject_id == merge_id) {
            kg_index_remove_id(kg->subject_index, kg->spo_bucket_count,
                               merge_id, rid);
            rn->relation.subject_id = keep_id;
            KG_IndexEntry *se = kg_index_get_or_create(
                kg->subject_index, kg->spo_bucket_count, keep_id);
            if (se) kg_idlist_push(&se->list, rid);
        }

        /* Update object index */
        if (rn->relation.object_id == merge_id) {
            kg_index_remove_id(kg->object_index, kg->spo_bucket_count,
                               merge_id, rid);
            rn->relation.object_id = keep_id;
            KG_IndexEntry *oe = kg_index_get_or_create(
                kg->object_index, kg->spo_bucket_count, keep_id);
            if (oe) kg_idlist_push(&oe->list, rid);
        }
    }
    free(rel_ids);

    /* Remove embedding for merge entity */
    kg_embedding_remove(kg, merge_id);

    /* Remove merge entity from hash table */
    size_t bucket = (size_t)kg_hash_uint64(merge_id,
                                            kg->entity_bucket_count);
    KG_EntityNode *prev = NULL;
    for (KG_EntityNode *n = kg->entity_buckets[bucket]; n; n = n->next) {
        if (n->entity.entity_id == merge_id) {
            if (prev) prev->next = n->next;
            else kg->entity_buckets[bucket] = n->next;
            kg_entity_data_free(&n->entity);
            free(n);
            kg->entity_count--;
            break;
        }
        prev = n;
    }

    pthread_rwlock_unlock(&kg->rwlock);
    return 0;
}

/*
 * Link Prediction
 */

int gv_kg_predict_links(const GV_KnowledgeGraph *kg, uint64_t entity_id,
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

    /* Collect candidates: entities not directly connected, with embeddings */
    size_t cap = 256;
    KG_ScorePair *candidates = (KG_ScorePair *)malloc(
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

        float sim = kg_cosine_similarity(src_emb,
            kg->all_embeddings + i * dim, dim);
        if (sim < kg->config.link_prediction_threshold) continue;

        /* Structural boost: shared neighbors */
        size_t shared = kg_shared_neighbors(kg, entity_id, other_id);
        float boost = (shared > 0) ? 0.1f * (float)shared : 0.0f;
        if (boost > 0.2f) boost = 0.2f;

        float final_score = sim + boost;
        if (final_score > 1.0f) final_score = 1.0f;

        if (cand_count >= cap) {
            cap *= 2;
            KG_ScorePair *tmp = (KG_ScorePair *)realloc(candidates,
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
        results[i].predicted_predicate = kg_strdup("related_to");
        results[i].confidence = candidates[i].score;
    }

    free(candidates);
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return (int)result_count;
}

/*
 * Graph Traversal
 */

int gv_kg_get_neighbors(const GV_KnowledgeGraph *kg, uint64_t entity_id,
                         uint64_t *out_ids, size_t max_count) {
    if (!kg || !out_ids || max_count == 0) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);

    if (!kg_find_entity_node(kg, entity_id)) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return -1;
    }

    size_t found = 0;

    /* Outgoing (subject_index) */
    KG_IndexEntry *se = kg_index_find(kg->subject_index,
                                       kg->spo_bucket_count, entity_id);
    if (se) {
        for (size_t i = 0; i < se->list.count && found < max_count; i++) {
            KG_RelationNode *rn = kg_find_relation_node(kg, se->list.ids[i]);
            if (!rn) continue;
            uint64_t nbr = rn->relation.object_id;
            /* Dedup */
            int dup = 0;
            for (size_t j = 0; j < found; j++) {
                if (out_ids[j] == nbr) { dup = 1; break; }
            }
            if (!dup) out_ids[found++] = nbr;
        }
    }

    /* Incoming (object_index) */
    KG_IndexEntry *oe = kg_index_find(kg->object_index,
                                       kg->spo_bucket_count, entity_id);
    if (oe) {
        for (size_t i = 0; i < oe->list.count && found < max_count; i++) {
            KG_RelationNode *rn = kg_find_relation_node(kg, oe->list.ids[i]);
            if (!rn) continue;
            uint64_t nbr = rn->relation.subject_id;
            int dup = 0;
            for (size_t j = 0; j < found; j++) {
                if (out_ids[j] == nbr) { dup = 1; break; }
            }
            if (!dup) out_ids[found++] = nbr;
        }
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return (int)found;
}

int gv_kg_traverse(const GV_KnowledgeGraph *kg, uint64_t start,
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

int gv_kg_shortest_path(const GV_KnowledgeGraph *kg, uint64_t from,
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

    /* BFS from 'from' looking for 'to' */
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

        /* Outgoing */
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

        /* Incoming */
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

    /* Reconstruct path from 'to' back to 'from' using parent[] */
    size_t path_len = 0;
    uint64_t *rev = (uint64_t *)malloc(bfs.count * sizeof(uint64_t));
    if (!rev) {
        kg_bfs_free(&bfs);
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return -1;
    }

    uint64_t cur = to;
    while (cur != 0 && cur != from) {
        rev[path_len++] = cur;
        /* Find parent of cur */
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

    /* Reverse into output */
    size_t out_len = (path_len < max_len) ? path_len : max_len;
    for (size_t i = 0; i < out_len; i++) {
        path_ids[i] = rev[path_len - 1 - i];
    }

    free(rev);
    kg_bfs_free(&bfs);
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return (int)out_len;
}

/*
 * Subgraph Extraction
 */

int gv_kg_extract_subgraph(const GV_KnowledgeGraph *kg, uint64_t center,
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

    /* Entity IDs = all BFS-visited nodes */
    subgraph->entity_ids = (uint64_t *)malloc(bfs.count * sizeof(uint64_t));
    if (!subgraph->entity_ids) {
        kg_bfs_free(&bfs);
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return -1;
    }
    memcpy(subgraph->entity_ids, bfs.visited, bfs.count * sizeof(uint64_t));
    subgraph->entity_count = bfs.count;

    /* Collect inter-relations among subgraph entities */
    size_t rel_cap = 256;
    size_t rel_count = 0;
    uint64_t *rel_ids = (uint64_t *)malloc(rel_cap * sizeof(uint64_t));
    if (!rel_ids) {
        free(subgraph->entity_ids);
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
            /* Check if object is also in subgraph */
            if (!kg_bfs_seen(&bfs, rn->relation.object_id)) continue;
            /* Avoid duplicates */
            int dup = 0;
            for (size_t x = 0; x < rel_count; x++) {
                if (rel_ids[x] == rid) { dup = 1; break; }
            }
            if (dup) continue;
            if (rel_count >= rel_cap) {
                rel_cap *= 2;
                uint64_t *tmp = (uint64_t *)realloc(rel_ids,
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

void gv_kg_free_subgraph(GV_KGSubgraph *subgraph) {
    if (!subgraph) return;
    free(subgraph->entity_ids);
    free(subgraph->relation_ids);
    subgraph->entity_ids = NULL;
    subgraph->relation_ids = NULL;
    subgraph->entity_count = 0;
    subgraph->relation_count = 0;
}

/*
 * Hybrid Queries (Vector + Graph)
 */

int gv_kg_hybrid_search(const GV_KnowledgeGraph *kg,
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
    KG_ScorePair *pairs = (KG_ScorePair *)malloc(cap * sizeof(KG_ScorePair));
    if (!pairs) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return -1;
    }
    size_t pair_count = 0;

    for (size_t i = 0; i < n; i++) {
        uint64_t eid = kg->embedding_entity_ids[i];
        KG_EntityNode *en = kg_find_entity_node(kg, eid);
        if (!en) continue;

        /* Type filter */
        if (entity_type && en->entity.type &&
            strcmp(en->entity.type, entity_type) != 0) continue;

        /* Predicate filter */
        if (predicate_filter &&
            !kg_entity_has_predicate(kg, eid, predicate_filter)) continue;

        float sim = kg_cosine_similarity(query_embedding,
            kg->all_embeddings + i * dim, dim);

        if (pair_count >= cap) {
            cap *= 2;
            KG_ScorePair *tmp = (KG_ScorePair *)realloc(pairs,
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
        results[i].name       = en ? kg_strdup(en->entity.name) : NULL;
        results[i].type       = en ? kg_strdup(en->entity.type) : NULL;
        results[i].similarity = pairs[i].score;
    }

    free(pairs);
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return (int)result_count;
}

/*
 * Analytics
 */

int gv_kg_get_stats(const GV_KnowledgeGraph *kg, GV_KGStats *stats) {
    if (!kg || !stats) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);

    memset(stats, 0, sizeof(GV_KGStats));
    stats->entity_count   = kg->entity_count;
    stats->relation_count = kg->relation_count;
    stats->triple_count   = kg->relation_count;
    stats->embedding_count = kg->embedding_count;

    /* Count distinct types */
    size_t type_cap = 128;
    char **types = (char **)calloc(type_cap, sizeof(char *));
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
                        char **tmp = (char **)realloc(types,
                            type_cap * sizeof(char *));
                        if (!tmp) break;
                        types = tmp;
                    }
                    types[type_count++] = n->entity.type;
                }
            }
        }
        stats->type_count = type_count;
        free(types);
    }

    /* Count distinct predicates */
    size_t pred_cap = 128;
    char **preds = (char **)calloc(pred_cap, sizeof(char *));
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
                        char **tmp = (char **)realloc(preds,
                            pred_cap * sizeof(char *));
                        if (!tmp) break;
                        preds = tmp;
                    }
                    preds[pred_count++] = n->relation.predicate;
                }
            }
        }
        stats->predicate_count = pred_count;
        free(preds);
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return 0;
}

float gv_kg_entity_centrality(const GV_KnowledgeGraph *kg,
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

    /* out_degree from subject_index */
    size_t out_degree = 0;
    KG_IndexEntry *se = kg_index_find(kg->subject_index,
                                       kg->spo_bucket_count, entity_id);
    if (se) out_degree = se->list.count;

    /* in_degree from object_index */
    size_t in_degree = 0;
    KG_IndexEntry *oe = kg_index_find(kg->object_index,
                                       kg->spo_bucket_count, entity_id);
    if (oe) in_degree = oe->list.count;

    float centrality = (float)(in_degree + out_degree) / (float)(total - 1);

    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return centrality;
}

int gv_kg_get_entity_types(const GV_KnowledgeGraph *kg, char **out_types,
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
                out_types[found++] = kg_strdup(n->entity.type);
            }
        }
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return (int)found;
}

int gv_kg_get_predicates(const GV_KnowledgeGraph *kg, char **out_predicates,
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
                out_predicates[found++] = kg_strdup(n->relation.predicate);
            }
        }
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return (int)found;
}

/*
 * Persistence: Save
 */

/**
 * @brief Internal: write raw bytes to file.
 */
static int kg_write_bytes(FILE *fp, const void *data, size_t len) {
    return (fwrite(data, 1, len, fp) == len) ? 0 : -1;
}

static int kg_write_u32(FILE *fp, uint32_t v) {
    return kg_write_bytes(fp, &v, sizeof(v));
}

static int kg_write_u64(FILE *fp, uint64_t v) {
    return kg_write_bytes(fp, &v, sizeof(v));
}

static int kg_write_f32(FILE *fp, float v) {
    return kg_write_bytes(fp, &v, sizeof(v));
}

static int kg_write_u8(FILE *fp, uint8_t v) {
    return kg_write_bytes(fp, &v, sizeof(v));
}

static int kg_write_string(FILE *fp, const char *s) {
    uint32_t len = s ? (uint32_t)strlen(s) : 0;
    if (kg_write_u32(fp, len) != 0) return -1;
    if (len > 0 && kg_write_bytes(fp, s, len) != 0) return -1;
    return 0;
}

static int kg_write_props(FILE *fp, const GV_KGProp *props, size_t count) {
    if (kg_write_u32(fp, (uint32_t)count) != 0) return -1;
    for (const GV_KGProp *p = props; p; p = p->next) {
        if (kg_write_string(fp, p->key) != 0) return -1;
        if (kg_write_string(fp, p->value) != 0) return -1;
    }
    return 0;
}

int gv_kg_save(const GV_KnowledgeGraph *kg, const char *path) {
    if (!kg || !path) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return -1;
    }

    /* Header */
    if (kg_write_bytes(fp, KG_MAGIC, KG_MAGIC_LEN) != 0) goto fail;
    if (kg_write_u32(fp, KG_VERSION) != 0) goto fail;

    /* Config subset */
    if (kg_write_u32(fp, (uint32_t)kg->config.embedding_dimension) != 0)
        goto fail;
    if (kg_write_f32(fp, kg->config.similarity_threshold) != 0) goto fail;
    if (kg_write_f32(fp, kg->config.link_prediction_threshold) != 0)
        goto fail;

    /* Counts */
    if (kg_write_u64(fp, (uint64_t)kg->entity_count) != 0) goto fail;
    if (kg_write_u64(fp, (uint64_t)kg->relation_count) != 0) goto fail;
    if (kg_write_u64(fp, kg->next_entity_id) != 0) goto fail;
    if (kg_write_u64(fp, kg->next_relation_id) != 0) goto fail;

    /* Entities */
    for (size_t b = 0; b < kg->entity_bucket_count; b++) {
        for (KG_EntityNode *n = kg->entity_buckets[b]; n; n = n->next) {
            const GV_KGEntity *e = &n->entity;
            if (kg_write_u64(fp, e->entity_id) != 0) goto fail;
            if (kg_write_string(fp, e->name) != 0) goto fail;
            if (kg_write_string(fp, e->type) != 0) goto fail;

            uint8_t has_emb = (e->embedding && e->dimension > 0) ? 1 : 0;
            if (kg_write_u8(fp, has_emb) != 0) goto fail;
            if (has_emb) {
                if (kg_write_bytes(fp, e->embedding,
                    e->dimension * sizeof(float)) != 0) goto fail;
            }

            if (kg_write_f32(fp, e->confidence) != 0) goto fail;
            if (kg_write_u64(fp, e->created_at) != 0) goto fail;
            if (kg_write_props(fp, e->properties, e->prop_count) != 0)
                goto fail;
        }
    }

    /* Relations */
    for (size_t b = 0; b < kg->relation_bucket_count; b++) {
        for (KG_RelationNode *n = kg->relation_buckets[b]; n; n = n->next) {
            const GV_KGRelation *r = &n->relation;
            if (kg_write_u64(fp, r->relation_id) != 0) goto fail;
            if (kg_write_u64(fp, r->subject_id) != 0) goto fail;
            if (kg_write_u64(fp, r->object_id) != 0) goto fail;
            if (kg_write_string(fp, r->predicate) != 0) goto fail;
            if (kg_write_f32(fp, r->weight) != 0) goto fail;
            if (kg_write_u64(fp, r->created_at) != 0) goto fail;

            /* Count properties */
            size_t pc = 0;
            for (GV_KGProp *p = r->properties; p; p = p->next) pc++;
            if (kg_write_props(fp, r->properties, pc) != 0) goto fail;
        }
    }

    fclose(fp);
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return 0;

fail:
    fclose(fp);
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return -1;
}

/*
 * Persistence: Load
 */

static int kg_read_bytes(FILE *fp, void *buf, size_t len) {
    return (fread(buf, 1, len, fp) == len) ? 0 : -1;
}

static int kg_read_u32(FILE *fp, uint32_t *v) {
    return kg_read_bytes(fp, v, sizeof(*v));
}

static int kg_read_u64(FILE *fp, uint64_t *v) {
    return kg_read_bytes(fp, v, sizeof(*v));
}

static int kg_read_f32(FILE *fp, float *v) {
    return kg_read_bytes(fp, v, sizeof(*v));
}

static int kg_read_u8(FILE *fp, uint8_t *v) {
    return kg_read_bytes(fp, v, sizeof(*v));
}

static char *kg_read_string(FILE *fp) {
    uint32_t len;
    if (kg_read_u32(fp, &len) != 0) return NULL;
    if (len == 0) return kg_strdup("");
    char *s = (char *)malloc(len + 1);
    if (!s) return NULL;
    if (kg_read_bytes(fp, s, len) != 0) { free(s); return NULL; }
    s[len] = '\0';
    return s;
}

static GV_KGProp *kg_read_props(FILE *fp, uint32_t count) {
    GV_KGProp *head = NULL;
    GV_KGProp *tail = NULL;
    for (uint32_t i = 0; i < count; i++) {
        GV_KGProp *p = (GV_KGProp *)calloc(1, sizeof(GV_KGProp));
        if (!p) return head;
        p->key = kg_read_string(fp);
        p->value = kg_read_string(fp);
        p->next = NULL;
        if (!p->key || !p->value) {
            free(p->key);
            free(p->value);
            free(p);
            return head;
        }
        if (tail) tail->next = p;
        else head = p;
        tail = p;
    }
    return head;
}

GV_KnowledgeGraph *gv_kg_load(const char *path) {
    if (!path) return NULL;

    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;

    /* Read and verify magic */
    char magic[KG_MAGIC_LEN];
    if (kg_read_bytes(fp, magic, KG_MAGIC_LEN) != 0 ||
        memcmp(magic, KG_MAGIC, KG_MAGIC_LEN) != 0) {
        fclose(fp);
        return NULL;
    }

    /* Version */
    uint32_t version;
    if (kg_read_u32(fp, &version) != 0 || version != KG_VERSION) {
        fclose(fp);
        return NULL;
    }

    /* Config */
    uint32_t emb_dim;
    float sim_thresh, lp_thresh;
    if (kg_read_u32(fp, &emb_dim) != 0) { fclose(fp); return NULL; }
    if (kg_read_f32(fp, &sim_thresh) != 0) { fclose(fp); return NULL; }
    if (kg_read_f32(fp, &lp_thresh) != 0) { fclose(fp); return NULL; }

    GV_KGConfig cfg;
    gv_kg_config_init(&cfg);
    cfg.embedding_dimension = (size_t)emb_dim;
    cfg.similarity_threshold = sim_thresh;
    cfg.link_prediction_threshold = lp_thresh;

    /* Counts */
    uint64_t entity_count, relation_count, next_eid, next_rid;
    if (kg_read_u64(fp, &entity_count) != 0) { fclose(fp); return NULL; }
    if (kg_read_u64(fp, &relation_count) != 0) { fclose(fp); return NULL; }
    if (kg_read_u64(fp, &next_eid) != 0) { fclose(fp); return NULL; }
    if (kg_read_u64(fp, &next_rid) != 0) { fclose(fp); return NULL; }

    /* Create graph */
    GV_KnowledgeGraph *kg = gv_kg_create(&cfg);
    if (!kg) { fclose(fp); return NULL; }
    kg->next_entity_id = next_eid;
    kg->next_relation_id = next_rid;

    /* Read entities */
    for (uint64_t i = 0; i < entity_count; i++) {
        uint64_t eid;
        if (kg_read_u64(fp, &eid) != 0) goto load_fail;

        char *name = kg_read_string(fp);
        char *type = kg_read_string(fp);
        if (!name || !type) {
            free(name);
            free(type);
            goto load_fail;
        }

        uint8_t has_emb;
        if (kg_read_u8(fp, &has_emb) != 0) {
            free(name); free(type);
            goto load_fail;
        }

        float *emb_data = NULL;
        if (has_emb && emb_dim > 0) {
            emb_data = (float *)malloc(emb_dim * sizeof(float));
            if (!emb_data || kg_read_bytes(fp, emb_data,
                emb_dim * sizeof(float)) != 0) {
                free(emb_data); free(name); free(type);
                goto load_fail;
            }
        }

        float confidence;
        uint64_t created_at;
        if (kg_read_f32(fp, &confidence) != 0 ||
            kg_read_u64(fp, &created_at) != 0) {
            free(emb_data); free(name); free(type);
            goto load_fail;
        }

        uint32_t prop_count;
        if (kg_read_u32(fp, &prop_count) != 0) {
            free(emb_data); free(name); free(type);
            goto load_fail;
        }
        GV_KGProp *props = kg_read_props(fp, prop_count);

        /* Insert entity node directly */
        KG_EntityNode *node = (KG_EntityNode *)calloc(1,
                                  sizeof(KG_EntityNode));
        if (!node) {
            free(emb_data); free(name); free(type);
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

        if (has_emb && emb_data && emb_dim > 0) {
            kg_embedding_add(kg, eid, emb_data, (size_t)emb_dim);
        }
    }

    /* Read relations */
    for (uint64_t i = 0; i < relation_count; i++) {
        uint64_t rid, sid, oid;
        if (kg_read_u64(fp, &rid) != 0 ||
            kg_read_u64(fp, &sid) != 0 ||
            kg_read_u64(fp, &oid) != 0) goto load_fail;

        char *predicate = kg_read_string(fp);
        if (!predicate) goto load_fail;

        float weight;
        uint64_t created_at;
        if (kg_read_f32(fp, &weight) != 0 ||
            kg_read_u64(fp, &created_at) != 0) {
            free(predicate);
            goto load_fail;
        }

        uint32_t prop_count;
        if (kg_read_u32(fp, &prop_count) != 0) {
            free(predicate);
            goto load_fail;
        }
        GV_KGProp *props = kg_read_props(fp, prop_count);

        /* Insert relation node directly */
        KG_RelationNode *node = (KG_RelationNode *)calloc(1,
                                    sizeof(KG_RelationNode));
        if (!node) {
            free(predicate);
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

        /* Update SPO indexes */
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
    }

    fclose(fp);
    return kg;

load_fail:
    fclose(fp);
    gv_kg_destroy(kg);
    return NULL;
}
