/*
 * End-to-end integration test.
 *
 * Unlike the per-subsystem unit tests, this walks one realistic multi-model
 * workflow end to end and checks the subsystems work together:
 *
 *   1. Vector DB   — add/search/filter/range/update/delete/persist (save+reopen)
 *   2. Graph+Cypher— CREATE, MATCH, aggregation, variable-length path + vector
 *                    distance predicate (the multi-model join)
 *   3. SPLADE      — text -> learned-sparse -> relevance search
 *   4. Hybrid      — dense (vector) + sparse (BM25) fusion over one corpus
 *
 * Any failure returns non-zero and names the phase, so this doubles as a smoke
 * test that the whole stack is wired correctly.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gigavector.h"
#include "features/cypher.h"
#include "features/knowledge_graph.h"
#include "multimodal/splade.h"
#include "multimodal/learned_sparse.h"
#include "multimodal/bm25.h"
#include "search/hybrid_search.h"
#include "storage/database.h"
#include "schema/metadata.h"
#include "test_tmp.h"

static int g_fail = 0;
#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (cond) {                                                   \
            printf("  ok   %s\n", msg);                               \
        } else {                                                      \
            printf("  FAIL %s  (%s:%d)\n", msg, __FILE__, __LINE__);  \
            g_fail = 1;                                               \
        }                                                             \
    } while (0)

#define PHASE(name) printf("\n== %s ==\n", name)

/* ---------------------------------------------------------------- phase 1 */

static void phase_vector_db(void) {
    PHASE("1. Vector DB lifecycle (add/search/filter/range/update/delete/persist)");

    char path[512];
    gv_test_make_temp_path(path, sizeof(path), "e2e_vec", ".gvdb");

    GV_Database *db = db_open(path, 4, GV_INDEX_TYPE_FLAT);
    CHECK(db != NULL, "open FLAT database");
    if (!db) return;

    /* Four labelled vectors. */
    const float v_red[4]   = {1.0f, 0.0f, 0.0f, 0.0f};
    const float v_blue[4]  = {0.0f, 1.0f, 0.0f, 0.0f};
    const float v_green[4] = {0.0f, 0.0f, 1.0f, 0.0f};
    const float v_red2[4]  = {0.9f, 0.1f, 0.0f, 0.0f};
    CHECK(db_add_vector_with_metadata(db, v_red, 4, "color", "red") == 0, "add red");
    CHECK(db_add_vector_with_metadata(db, v_blue, 4, "color", "blue") == 0, "add blue");
    CHECK(db_add_vector_with_metadata(db, v_green, 4, "color", "green") == 0, "add green");
    CHECK(db_add_vector_with_metadata(db, v_red2, 4, "color", "red") == 0, "add red2");

    /* kNN: query near red -> red (id 0) is the exact nearest. */
    GV_SearchResult res[4];
    int n = db_search(db, v_red, 3, res, GV_DISTANCE_EUCLIDEAN);
    CHECK(n == 3, "search returns k=3");
    CHECK(n > 0 && res[0].id == 0, "nearest neighbour of red is red (id 0)");
    CHECK(n > 0 && res[0].distance <= res[1].distance, "results ordered by distance");
    gv_search_results_free(res, (size_t)n);

    /* Payload filter: only the two reds. */
    int nf = db_search_with_filter_expr(db, v_red, 4, res, GV_DISTANCE_EUCLIDEAN, "color == \"red\"");
    CHECK(nf == 2, "filtered search (color==red) -> 2 hits");
    int reds_ok = 1;
    for (int i = 0; i < nf; i++) {
        const char *c = vector_get_metadata(res[i].vector, "color");
        if (!c || strcmp(c, "red") != 0) reds_ok = 0;
    }
    CHECK(reds_ok, "every filtered hit has color=red");
    gv_search_results_free(res, (size_t)nf);

    /* Range search: within 0.5 of red -> red (0) and red2 (~0.141). */
    int nr = db_range_search(db, v_red, 0.5f, res, 4, GV_DISTANCE_EUCLIDEAN);
    CHECK(nr == 2, "range search radius 0.5 -> 2 in-radius");
    gv_search_results_free(res, (size_t)nr);

    /* Update id 1 (blue) to near-red, then it should match the red filter query top-2. */
    CHECK(db_update_vector(db, 1, v_red, 4) == 0, "update id 1 to red vector");
    int nu = db_search(db, v_red, 1, res, GV_DISTANCE_EUCLIDEAN);
    CHECK(nu == 1 && (res[0].id == 0 || res[0].id == 1), "post-update nearest is a red-valued id");
    gv_search_results_free(res, (size_t)nu);

    /* Delete id 2 (green); a search for green should no longer return id 2 first with dist 0. */
    CHECK(db_delete_vector_by_index(db, 2) == 0, "delete id 2 (green)");
    int ng = db_search(db, v_green, 3, res, GV_DISTANCE_EUCLIDEAN);
    int found_green = 0;
    for (int i = 0; i < ng; i++) if (res[i].id == 2) found_green = 1;
    CHECK(!found_green, "deleted id 2 absent from results");
    gv_search_results_free(res, (size_t)ng);

    /* Persist and reopen. */
    CHECK(db_save(db, path) == 0, "save database");
    db_close(db);

    GV_Database *db2 = db_open(path, 4, GV_INDEX_TYPE_FLAT);
    CHECK(db2 != NULL, "reopen database");
    if (db2) {
        int n2 = db_search(db2, v_red, 1, res, GV_DISTANCE_EUCLIDEAN);
        CHECK(n2 == 1, "search after reopen works (data persisted)");
        if (n2 > 0) {
            const char *c = vector_get_metadata(res[0].vector, "color");
            CHECK(c && strcmp(c, "red") == 0, "persisted metadata intact (red)");
        }
        gv_search_results_free(res, (size_t)n2);
        db_close(db2);
    }
    remove(path);
}

/* ---------------------------------------------------------------- phase 2 */

static const char *cell(const GV_CypherResult *r, size_t row, size_t col) {
    return r->column_values[row * r->column_count + col];
}

static void phase_graph_cypher(void) {
    PHASE("2. Graph + Cypher (CREATE / MATCH / aggregate / varlen path + vector predicate)");

    GV_KnowledgeGraph *kg = kg_create(NULL);
    CHECK(kg != NULL, "create knowledge graph");
    if (!kg) return;
    GV_CypherEngine *cy = cypher_create(kg);
    CHECK(cy != NULL, "create cypher engine");
    if (!cy) { kg_destroy(kg); return; }

    GV_CypherResult r;
    /* A chain of documents carrying embedding properties. */
    int rc = cypher_execute(cy,
        "CREATE (a:Doc {name:'a', emb:'[1,0,0]'})-[:CITES]->(b:Doc {name:'b', emb:'[0,1,0]'})"
        "-[:CITES]->(c:Doc {name:'c', emb:'[0.95,0.05,0]'})", &r);
    CHECK(rc == 0, "CREATE citation chain with embeddings");
    if (rc == 0) {
        CHECK(r.nodes_created == 3 && r.relationships_created == 2, "3 nodes + 2 rels");
        cypher_free_result(&r);
    }

    CHECK(cypher_execute(cy, "MATCH (d:Doc) RETURN count(*)", &r) == 0
          && r.row_count == 1 && strcmp(cell(&r, 0, 0), "3") == 0, "aggregate count(*) = 3");
    cypher_free_result(&r);

    CHECK(cypher_execute(cy, "MATCH (a:Doc {name:'a'})-[:CITES*1..2]->(x) RETURN x.name ORDER BY x.name", &r) == 0
          && r.row_count == 2 && strcmp(cell(&r, 0, 0), "b") == 0 && strcmp(cell(&r, 1, 0), "c") == 0,
          "variable-length path a-[*1..2]->{b,c}");
    cypher_free_result(&r);

    /* The multi-model join: follow citations, then rank by vector distance. */
    rc = cypher_execute(cy,
        "MATCH (a:Doc {name:'a'})-[:CITES*1..2]->(x) "
        "WHERE vector_distance(x.emb, '[1,0,0]') < 0.5 RETURN x.name ORDER BY x.name", &r);
    CHECK(rc == 0 && r.row_count == 1 && strcmp(cell(&r, 0, 0), "c") == 0,
          "varlen path + vector predicate -> only near doc c");
    cypher_free_result(&r);

    cypher_destroy(cy);
    kg_destroy(kg);
}

/* ---------------------------------------------------------------- phase 3 */

static void phase_splade(void) {
    PHASE("3. SPLADE learned-sparse (text -> encode -> index -> relevance search)");

    GV_SpladeConfig scfg;
    splade_config_init(&scfg);
    GV_SpladeEncoder *enc = splade_create(&scfg);
    CHECK(enc != NULL, "create SPLADE encoder");

    GV_LearnedSparseConfig icfg;
    ls_config_init(&icfg);
    icfg.vocab_size = scfg.vocab_size;
    GV_LearnedSparseIndex *idx = ls_create(&icfg);
    CHECK(idx != NULL, "create learned-sparse index");
    if (!enc || !idx) { if (enc) splade_destroy(enc); if (idx) ls_destroy(idx); return; }

    int d0 = splade_index_add(enc, idx, "distributed consensus and raft replication");
    int d1 = splade_index_add(enc, idx, "neural network embeddings for semantic search");
    int d2 = splade_index_add(enc, idx, "baking sourdough bread at home");
    CHECK(d0 >= 0 && d1 >= 0 && d2 >= 0, "index three documents");

    GV_LearnedSparseResult res[3];
    int n = splade_index_search(enc, idx, "semantic search with neural embeddings", 3, res);
    CHECK(n >= 1, "search returns a hit");
    CHECK(n >= 1 && res[0].doc_index == (size_t)d1, "neural/semantic doc ranks first");
    for (int i = 0; i < n; i++) CHECK(res[i].doc_index != (size_t)d2, "unrelated baking doc excluded");

    ls_destroy(idx);
    splade_destroy(enc);
}

/* ---------------------------------------------------------------- phase 4 */

static void phase_hybrid(void) {
    PHASE("4. Hybrid dense+sparse fusion (vector DB + BM25 -> combined ranking)");

    GV_Database *db = db_open(NULL, 3, GV_INDEX_TYPE_FLAT);
    CHECK(db != NULL, "open in-memory vector DB");

    GV_BM25Config bcfg;
    bm25_config_init(&bcfg);
    GV_BM25Index *bm = bm25_create(&bcfg);
    CHECK(bm != NULL, "create BM25 index");
    if (!db || !bm) { if (db) db_close(db); if (bm) bm25_destroy(bm); return; }

    /* Doc i's vector and text are aligned by insertion order. */
    const float vecs[3][3] = {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.9f, 0.1f, 0.0f}};
    const char *texts[3] = {
        "quick brown fox jumps",
        "lazy dog sleeps",
        "quick fox runs fast",
    };
    for (int i = 0; i < 3; i++) {
        CHECK(db_add_vector(db, vecs[i], 3) == 0, "add vector");
        CHECK(bm25_add_document(bm, (size_t)i, texts[i]) == 0, "add bm25 doc");
    }

    GV_HybridConfig hcfg;
    hybrid_config_init(&hcfg);
    GV_HybridSearcher *hs = hybrid_create(db, bm, &hcfg);
    CHECK(hs != NULL, "create hybrid searcher");
    if (hs) {
        const float q[3] = {1.0f, 0.0f, 0.0f};
        GV_HybridResult res[3];
        int n = hybrid_search(hs, q, "quick fox", 3, res);
        CHECK(n >= 1, "hybrid search returns hits");
        /* Doc 0 matches both the query vector (==[1,0,0]) and the text best. */
        CHECK(n >= 1 && res[0].vector_index == 0, "doc 0 (best dense+sparse) ranks first");
        CHECK(n >= 2 && res[0].combined_score >= res[1].combined_score, "fusion scores ordered");
        hybrid_destroy(hs);
    }

    bm25_destroy(bm);
    db_close(db);
}

int main(void) {
    printf("GigaVector end-to-end integration test\n");
    phase_vector_db();
    phase_graph_cypher();
    phase_splade();
    phase_hybrid();

    printf("\n%s\n", g_fail ? "E2E: SOME CHECKS FAILED" : "E2E: ALL CHECKS PASSED");
    return g_fail ? 1 : 0;
}
