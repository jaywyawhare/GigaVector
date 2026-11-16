/**
  * @file knowledge_graph_query.c
  * @brief KG search, reasoning and analytics: similarity/text/hybrid search,
  *        entity resolution, link prediction, traversal, spreading activation,
  *        shortest path, subgraph extraction, stats and centrality
  *        (split from knowledge_graph.c).
  */
#define _POSIX_C_SOURCE 200112L

#include "features/knowledge_graph.h"
#include "features/knowledge_graph_internal.h"
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

