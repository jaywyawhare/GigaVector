/**
 * @file document_search.c
 * @brief Unified cross-layer search (Phase 2): fan out to embedding/graph/memory,
 *        join on chunk_id, fuse ranks with RRF, return enriched results.
 */
#include "storage/document_search.h"
#include "core/memory.h"
#include "core/utils.h"
#include "core/types.h"
#include "search/distance.h"
#include "schema/vector.h"

#include <string.h>
#include <stdlib.h>

#define RRF_K 60.0
#define MAX_TRIPLETS_PER_CHUNK 16
#define MAX_FACTS_PER_CHUNK 8

typedef struct {
    char   chunk_id[64];
    int    vrank;        /* -1 = not surfaced by vector layer */
    float  vscore;
    int    mrank;        /* -1 = not surfaced by memory layer */
    char **facts;
    size_t fact_count;
    size_t fact_cap;
} Cand;

static double rrf(int rank) { return rank >= 0 ? 1.0 / (RRF_K + (double)rank) : 0.0; }

static Cand *cand_find_or_add(Cand *arr, size_t *n, size_t cap, const char *id) {
    for (size_t i = 0; i < *n; i++) {
        if (strcmp(arr[i].chunk_id, id) == 0) return &arr[i];
    }
    if (*n >= cap) return NULL;
    Cand *c = &arr[*n];
    memset(c, 0, sizeof(*c));
    snprintf(c->chunk_id, sizeof(c->chunk_id), "%s", id);
    c->vrank = -1;
    c->mrank = -1;
    (*n)++;
    return c;
}

static void cand_add_fact(Cand *c, const char *fact) {
    if (!fact) return;
    if (c->fact_count == c->fact_cap) {
        size_t nc = c->fact_cap ? c->fact_cap * 2 : 4;
        char **nf = (char **)gv_realloc(c->facts, nc * sizeof(char *));
        if (!nf) return;
        c->facts = nf;
        c->fact_cap = nc;
    }
    c->facts[c->fact_count] = gv_dup_cstr(fact);
    if (c->facts[c->fact_count]) c->fact_count++;
}

void gv_enriched_results_free(GV_EnrichedResult *r, size_t n) {
    if (!r) return;
    for (size_t i = 0; i < n; i++) {
        gv_free(r[i].chunk_text);
        if (r[i].triplets) kg_free_triples(r[i].triplets, r[i].triplet_count);
        gv_free(r[i].triplets);
        if (r[i].facts) {
            for (size_t j = 0; j < r[i].fact_count; j++) gv_free(r[i].facts[j]);
            gv_free(r[i].facts);
        }
    }
    gv_free(r);
}

int gv_search_document(GV_Database *db, GV_KnowledgeGraph *kg, GV_MemoryLayer *mem,
                       const char *query_text, const float *query_embedding, size_t k,
                       GV_EnrichedResult **out, size_t *out_count) {
    (void)query_text;
    if (!db || !out || !out_count || k == 0) return -1;
    *out = NULL;
    *out_count = 0;

    size_t fan = k * 4;
    if (fan < 20) fan = 20;

    size_t cap = fan * 2;
    Cand *cands = (Cand *)gv_alloc(cap * sizeof(Cand));
    if (!cands) return -1;
    size_t ncand = 0;

    /* Embedding layer */
    if (query_embedding) {
        GV_SearchResult *sr = (GV_SearchResult *)gv_alloc(fan * sizeof(GV_SearchResult));
        if (sr) {
            int n = db_search(db, query_embedding, fan, sr, GV_DISTANCE_COSINE);
            for (int i = 0; i < n; i++) {
                const char *id = db_get_id_by_index(db, sr[i].id);
                if (!id || strncmp(id, "kgent:", 6) == 0) continue; /* skip entity vecs */
                Cand *c = cand_find_or_add(cands, &ncand, cap, id);
                if (!c) continue;
                if (c->vrank < 0) { c->vrank = i; c->vscore = 1.0f - sr[i].distance; }
            }
            /* db_search hands back owned result vectors — free them. */
            for (int i = 0; i < n; i++)
                if (sr[i].vector) vector_destroy((GV_Vector *)sr[i].vector);
            gv_free(sr);
        }
    }

    /* Memory layer */
    if (mem && query_embedding) {
        GV_MemoryResult *mr = (GV_MemoryResult *)gv_alloc(fan * sizeof(GV_MemoryResult));
        if (mr) {
            int n = memory_search(mem, query_embedding, fan, mr, GV_DISTANCE_COSINE);
            for (int i = 0; i < n; i++) {
                const char *src = (mr[i].metadata && mr[i].metadata->source)
                                      ? mr[i].metadata->source : NULL;
                if (src) {
                    Cand *c = cand_find_or_add(cands, &ncand, cap, src);
                    if (c) {
                        if (c->mrank < 0) c->mrank = i;
                        if (c->fact_count < MAX_FACTS_PER_CHUNK)
                            cand_add_fact(c, mr[i].content);
                    }
                }
                memory_result_free(&mr[i]);
            }
            gv_free(mr);
        }
    }

    if (ncand == 0) { gv_free(cands); return 0; }

    /* Build enriched results: attach triplets, fuse scores */
    GV_EnrichedResult *res = (GV_EnrichedResult *)gv_alloc(ncand * sizeof(GV_EnrichedResult));
    if (!res) {
        for (size_t i = 0; i < ncand; i++)
            for (size_t j = 0; j < cands[i].fact_count; j++) gv_free(cands[i].facts[j]);
        for (size_t i = 0; i < ncand; i++) gv_free(cands[i].facts);
        gv_free(cands);
        return -1;
    }

    for (size_t i = 0; i < ncand; i++) {
        Cand *c = &cands[i];
        GV_EnrichedResult *e = &res[i];
        memset(e, 0, sizeof(*e));
        snprintf(e->chunk_id, sizeof(e->chunk_id), "%s", c->chunk_id);
        e->vector_score = c->vscore;

        /* chunk text from the embedding metadata */
        size_t vidx = 0;
        if (db_get_index_by_id(db, c->chunk_id, &vidx) == 0) {
            const char *txt = db_get_metadata_value(db, vidx, "text");
            if (txt) e->chunk_text = gv_dup_cstr(txt);
        }

        /* graph enrichment: triplets sharing this chunk_id */
        int triplet_hits = 0;
        if (kg) {
            GV_KGTriple *tr = (GV_KGTriple *)gv_alloc(MAX_TRIPLETS_PER_CHUNK * sizeof(GV_KGTriple));
            if (tr) {
                int tn = kg_query_triples_by_chunk(kg, c->chunk_id, tr, MAX_TRIPLETS_PER_CHUNK);
                if (tn > 0) {
                    e->triplets = tr;
                    e->triplet_count = (size_t)tn;
                    triplet_hits = tn;
                } else {
                    gv_free(tr);
                }
            }
        }
        e->triplet_hit_count = triplet_hits;

        /* memory facts (moved from the candidate) */
        e->facts = c->facts;
        e->fact_count = c->fact_count;
        e->memory_hit_count = (int)c->fact_count;
        c->facts = NULL; /* ownership transferred */

        /* RRF fusion across the three layers. The graph baseline for merely
         * having triples is deliberately small (rrf(4)); the strong,
         * query-dependent graph signal is added by the neighbour-expansion
         * pass below. */
        double graph_rank = triplet_hits > 0 ? rrf(4) : 0.0;
        e->score = (float)(rrf(c->vrank) + rrf(c->mrank) + graph_rank);
        e->graph_neighbor_hits = 0;
    }

    /* ── GraphRAG entity-neighbour expansion ─────────────────────────────
     * Seed from the entities of the top-scoring chunks, build the union of
     * those entities plus their 1-hop KG neighbourhoods, and boost every
     * chunk whose own entities fall inside that union. This makes the graph
     * contribution query-dependent: a chunk is promoted when it discusses
     * entities directly connected to (or identical with) the best-matching
     * context — even if it shares no literal triple with them. */
    if (kg && ncand > 1) {
        enum { MAX_SEED_ENTITIES = 32 };
        uint64_t seeds[MAX_SEED_ENTITIES];
        size_t nseeds = 0;
        /* Seed strictly from the best provisional match: expansion should
         * reflect what the query actually retrieved, not the whole pool
         * (seeding from every candidate would make the boost universal and
         * therefore meaningless). */
        for (size_t t = 0; t < res[0].triplet_count &&
                           nseeds < MAX_SEED_ENTITIES; t++) {
            uint64_t sid = res[0].triplets[t].subject_id;
            uint64_t oid = res[0].triplets[t].object_id;
            int seen_s = 0, seen_o = 0;
            for (size_t s2 = 0; s2 < nseeds; s2++) {
                if (seeds[s2] == sid) seen_s = 1;
                if (seeds[s2] == oid) seen_o = 1;
            }
            if (!seen_s && sid != 0) seeds[nseeds++] = sid;
            if (!seen_o && oid != 0 && nseeds < MAX_SEED_ENTITIES)
                seeds[nseeds++] = oid;
        }

        if (nseeds > 0) {
            GV_IdBitmap *ctx_set = gv_id_bitmap_create();
            if (ctx_set) {
                for (size_t s3 = 0; s3 < nseeds; s3++) {
                    gv_id_bitmap_add(ctx_set, seeds[s3]);
                    GV_IdBitmap *nb =
                        kg_entity_neighbor_set(kg, seeds[s3], 2 /* both */);
                    if (!nb) continue;
                    gv_id_bitmap_or_into(ctx_set, nb);
                    gv_id_bitmap_free(nb);
                }
                if (gv_id_bitmap_cardinality(ctx_set) > nseeds ||
                    gv_id_bitmap_cardinality(ctx_set) > 0) {
                    for (size_t i = 0; i < ncand; i++) {
                        GV_EnrichedResult *e = &res[i];
                        int hits = 0;
                        for (size_t t = 0; t < e->triplet_count; t++) {
                            if (gv_id_bitmap_contains(ctx_set,
                                                      e->triplets[t].subject_id))
                                hits++;
                            if (gv_id_bitmap_contains(ctx_set,
                                                      e->triplets[t].object_id))
                                hits++;
                        }
                        e->graph_neighbor_hits = hits;
                        if (hits > 0)
                            e->score = (float)(e->score + rrf(0));
                    }
                }
                gv_id_bitmap_free(ctx_set);
            }
        }
    }

    /* sort by fused score desc (simple insertion sort; ncand is small) */
    for (size_t i = 1; i < ncand; i++) {
        GV_EnrichedResult tmp = res[i];
        size_t j = i;
        while (j > 0 && res[j - 1].score < tmp.score) { res[j] = res[j - 1]; j--; }
        res[j] = tmp;
    }

    gv_free(cands);

    size_t keep = ncand < k ? ncand : k;
    /* free the tail we won't return */
    for (size_t i = keep; i < ncand; i++) {
        gv_free(res[i].chunk_text);
        if (res[i].triplets) kg_free_triples(res[i].triplets, res[i].triplet_count);
        gv_free(res[i].triplets);
        if (res[i].facts) {
            for (size_t j = 0; j < res[i].fact_count; j++) gv_free(res[i].facts[j]);
            gv_free(res[i].facts);
        }
    }

    *out = res;
    *out_count = keep;
    return 0;
}
