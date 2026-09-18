/* Phase 2 — vertical slice: one document in -> chunk_id-joined enriched result out. */
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <stdlib.h>
#include "core/compat.h"   /* strcasestr shim on MinGW/MSVC */
#include "storage/database.h"
#include "features/knowledge_graph.h"
#include "storage/memory_layer.h"
#include "storage/document_ingest.h"
#include "storage/document_search.h"

static int failures = 0;
#define ASSERT(c, m) do { if (!(c)) { printf("FAIL: %s\n", (m)); failures++; } \
                          else { printf("ok: %s\n", (m)); } } while (0)

/* Deterministic keyword embedding: dim=8 bag-of-keywords. */
static const char *KW[8] = {"turing","invent","machine","bletchley","ww2","park","work","the"};
static void embed_text(const char *t, float *v) {
    for (int i = 0; i < 8; i++) v[i] = 0.0f;
    for (int i = 0; i < 8; i++) {
        const char *p = t;
        /* case-insensitive substring count */
        size_t kl = strlen(KW[i]);
        while ((p = strcasestr(p, KW[i])) != NULL) { v[i] += 1.0f; p += kl; }
    }
}
static int embed_fn(void *ctx, const char **texts, size_t n, size_t dim, float *out) {
    (void)ctx; (void)dim;
    for (size_t j = 0; j < n; j++) embed_text(texts[j], out + j * 8);
    return 0;
}

static int triples_fn(void *ctx, const char *chunk, GV_ExtractedTriple *out, size_t max) {
    (void)ctx; (void)max;
    int n = 0;
    if (strcasestr(chunk, "invent")) {
        out[n++] = (GV_ExtractedTriple){"Alan Turing","Person","invented","Turing Machine","Concept"};
    }
    if (strcasestr(chunk, "bletchley")) {
        out[n++] = (GV_ExtractedTriple){"Alan Turing","Person","worked_at","Bletchley Park","Place"};
    }
    return n;
}
static int facts_fn(void *ctx, const char *chunk, char **out, size_t max) {
    (void)ctx; (void)max;
    if (strcasestr(chunk, "invent")) { out[0] = strdup("Turing invented the Turing machine"); return 1; }
    return 0;
}

int main(void) {
    const size_t D = 8;
    GV_Database *db = db_open(NULL, D, GV_INDEX_TYPE_HNSW);
    GV_Database *memdb = db_open(NULL, D, GV_INDEX_TYPE_HNSW);
    GV_KGConfig kcfg; kg_config_init(&kcfg); kcfg.embedding_dimension = 0;
    GV_KnowledgeGraph *kg = kg_create(&kcfg);
    GV_MemoryLayer *mem = memory_layer_create(memdb, NULL);
    ASSERT(db && memdb && kg && mem, "layers created");

    GV_IngestConfig icfg;
    memset(&icfg, 0, sizeof(icfg));
    icfg.embed = embed_fn;
    icfg.extract_triples = triples_fn;
    icfg.extract_facts = facts_fn;
    icfg.chunk_tokens = 8;
    icfg.overlap_tokens = 2;

    const char *doc = "Alan Turing worked at Bletchley Park during WW2. "
                      "He invented the Turing machine.";
    GV_IngestStats st;
    int rc = gv_ingest_document(db, kg, mem, doc, "docT", &icfg, &st);
    ASSERT(rc == 0, "gv_ingest_document ok");
    printf("   stats: chunks=%zu vectors=%zu triples=%zu facts=%zu\n",
           st.chunks, st.vectors, st.triples, st.facts);
    ASSERT(st.chunks == 2 && st.vectors == 2, "2 chunks embedded");
    ASSERT(st.triples == 2, "2 triples added to graph");
    ASSERT(st.facts == 1, "1 fact added to memory");

    /* Query: "what did Turing invent?" -> should rank the 'invented' chunk top,
       enriched with its triplet + fact via the chunk_id join. */
    float q[8]; embed_text("what did turing invent", q);
    GV_EnrichedResult *res = NULL; size_t nres = 0;
    rc = gv_search_document(db, kg, mem, "what did turing invent", q, 3, &res, &nres);
    ASSERT(rc == 0 && nres >= 1, "gv_search_document returns results");

    ASSERT(res[0].chunk_text && strcasestr(res[0].chunk_text, "invent"),
           "top result is the 'invented' chunk");
    ASSERT(res[0].triplet_hit_count >= 1, "top result enriched with a graph triplet");
    ASSERT(res[0].fact_count >= 1, "top result enriched with a memory fact");
    if (res[0].triplet_count > 0)
        printf("   triplet: (%s %s %s)\n", res[0].triplets[0].subject_name,
               res[0].triplets[0].predicate, res[0].triplets[0].object_name);
    if (res[0].fact_count > 0) printf("   fact: %s\n", res[0].facts[0]);
    printf("   fused score=%.4f vector_score=%.4f\n", res[0].score, res[0].vector_score);

    gv_enriched_results_free(res, nres);

    /* Cascading delete across the embedding layer. */
    int deleted = db_delete_by_doc(db, "docT");
    ASSERT(deleted == 2, "delete_by_doc removes both chunk vectors");
    size_t idx;
    ASSERT(db_get_index_by_id(db, "docT:0001", &idx) != 0, "chunk gone after delete");

    memory_layer_destroy(mem);
    kg_destroy(kg);
    db_close(db);
    db_close(memdb);

    printf(failures ? "\nSOME TESTS FAILED (%d)\n" : "\nALL DOCUMENT-PIPELINE TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
