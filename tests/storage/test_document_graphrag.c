/* GraphRAG entity-neighbour expansion: a chunk with NO lexical overlap with
 * the query must still surface (and outrank an unrelated chunk) when its
 * entities are KG-adjacent to the entities of the top-matching chunk. */
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <stdlib.h>
#include "storage/database.h"
#include "features/knowledge_graph.h"
#include "storage/memory_layer.h"
#include "storage/document_ingest.h"
#include "storage/document_search.h"

static int failures = 0;
#define ASSERT(c, m) do { if (!(c)) { printf("FAIL: %s\n", (m)); failures++; } \
                          else { printf("ok: %s\n", (m)); } } while (0)

static const char *KW[8] = {"turing","invent","machine","bletchley","ww2","park","work","the"};
static void embed_text(const char *t, float *v) {
    for (int i = 0; i < 8; i++) v[i] = 0.0f;
    for (int i = 0; i < 8; i++) {
        const char *p = t;
        size_t kl = strlen(KW[i]);
        while ((p = strcasestr(p, KW[i])) != NULL) { v[i] += 1.0f; p += kl; }
    }
}
static int embed_fn(void *ctx, const char **texts, size_t n, size_t dim, float *out) {
    (void)ctx; (void)dim;
    for (size_t j = 0; j < n; j++) embed_text(texts[j], out + j * 8);
    return 0;
}

/* Three documents, three disjoint keyword sets, one connected entity graph. */
static int triples_fn(void *ctx, const char *chunk, GV_ExtractedTriple *out, size_t max) {
    (void)ctx; (void)max;
    int n = 0;
    if (strcasestr(chunk, "invent"))
        out[n++] = (GV_ExtractedTriple){"Alan Turing","Person","invented","Turing Machine","Concept"};
    if (strcasestr(chunk, "codebreak")) {
        /* links docB's entities to the seed entity — no shared keywords */
        out[n++] = (GV_ExtractedTriple){"Alan Turing","Person","worked_at","Bletchley Park","Place"};
    }
    if (strcasestr(chunk, "weather"))
        out[n++] = (GV_ExtractedTriple){"Sunny Weather","Condition","improves","Farming Yield","Outcome"};
    return n;
}
static int facts_fn(void *ctx, const char *chunk, char **out, size_t max) {
    (void)ctx; (void)max; (void)chunk; (void)out;
    return 0;
}

static const char *DOC_A = "Alan Turing did invent the machine";   /* matches query   */
static const char *DOC_B = "codebreaking at bletchley park ww2";   /* KG-connected    */
static const char *DOC_C = "sunny weather helps the harvest";      /* unrelated       */

int main(void) {
    const size_t D = 8;
    GV_Database *db = db_open(NULL, D, GV_INDEX_TYPE_HNSW);
    GV_KGConfig kcfg; kg_config_init(&kcfg); kcfg.embedding_dimension = 0;
    GV_KnowledgeGraph *kg = kg_create(&kcfg);
    ASSERT(db && kg, "layers created");

    GV_IngestConfig icfg;
    memset(&icfg, 0, sizeof(icfg));
    icfg.embed = embed_fn;
    icfg.extract_triples = triples_fn;
    icfg.extract_facts = facts_fn;
    icfg.chunk_tokens = 16;
    icfg.overlap_tokens = 2;

    GV_IngestStats st;
    ASSERT(gv_ingest_document(db, kg, NULL, DOC_A, "docA", &icfg, &st) == 0, "ingest A");
    ASSERT(gv_ingest_document(db, kg, NULL, DOC_B, "docB", &icfg, &st) == 0, "ingest B");
    ASSERT(gv_ingest_document(db, kg, NULL, DOC_C, "docC", &icfg, &st) == 0, "ingest C");

    float q[8]; embed_text("what did turing invent", q);
    GV_EnrichedResult *res = NULL; size_t nres = 0;
    int rc = gv_search_document(db, kg, NULL, "turing invent", q, 3, &res, &nres);
    ASSERT(rc == 0 && nres >= 2, "search returns results");

    /* sanity: A is the direct match and carries its triple */
    ASSERT(res[0].chunk_id && strstr(res[0].chunk_id, "docA") != NULL,
           "docA ranks first (direct match)");
    ASSERT(res[0].graph_neighbor_hits > 0 || res[0].triplet_hit_count > 0,
           "docA has graph signal");

    /* the point of the test: B must outrank C purely on KG adjacency */
    const GV_EnrichedResult *eb = NULL, *ec = NULL;
    for (size_t i = 0; i < nres; i++) {
        if (strstr(res[i].chunk_id, "docB")) eb = &res[i];
        if (strstr(res[i].chunk_id, "docC")) ec = &res[i];
    }
    ASSERT(eb && ec, "both remaining chunks surfaced");
    printf("   docB score=%.4f neighbor_hits=%d | docC score=%.4f neighbor_hits=%d\n",
           eb->score, eb->graph_neighbor_hits, ec->score, ec->graph_neighbor_hits);
    ASSERT(eb->graph_neighbor_hits > 0,
           "docB boosted via entity neighbourhood");
    ASSERT(ec->graph_neighbor_hits == 0,
           "unrelated docC gets no neighbour signal");
    ASSERT(eb->score > ec->score,
           "KG-connected chunk outranks lexically-equally-weak chunk");

    gv_enriched_results_free(res, nres);
    kg_destroy(kg);
    db_close(db);

    printf(failures ? "\nSOME GRAPHRAG TESTS FAILED (%d)\n" : "\nALL GRAPHRAG TESTS PASSED\n",
           failures);
    return failures ? 1 : 0;
}
