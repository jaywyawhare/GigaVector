/* Phase 1 — KG combined-DB integration: chunk facet, reverse edges, db-delegated similarity. */
#include <stdio.h>
#include <string.h>
#include "features/knowledge_graph.h"
#include "storage/database.h"

static int failures = 0;
#define ASSERT(c, m) do { if (!(c)) { printf("FAIL: %s\n", (m)); failures++; } \
                          else { printf("ok: %s\n", (m)); } } while (0)

int main(void) {
    GV_KGConfig cfg;
    kg_config_init(&cfg);
    cfg.embedding_dimension = 4;
    GV_KnowledgeGraph *kg = kg_create(&cfg);
    ASSERT(kg != NULL, "kg_create");

    uint64_t turing = kg_add_entity(kg, "Alan Turing", "Person", NULL, 0);
    uint64_t machine = kg_add_entity(kg, "Turing Machine", "Concept", NULL, 0);
    uint64_t bletchley = kg_add_entity(kg, "Bletchley Park", "Place", NULL, 0);
    ASSERT(turing && machine && bletchley, "add 3 entities");

    /* 1.3 chunk facet */
    uint64_t r1 = kg_add_relation_with_chunk(kg, turing, "invented", machine, 1.0f, "doc1:0001");
    kg_add_relation_with_chunk(kg, turing, "worked_at", bletchley, 1.0f, "doc1:0000");
    ASSERT(r1 != 0, "add_relation_with_chunk");

    GV_KGTriple tr[8];
    int n = kg_query_triples_by_chunk(kg, "doc1:0001", tr, 8);
    ASSERT(n == 1 && tr[0].subject_id == turing && tr[0].object_id == machine,
           "query_triples_by_chunk returns the invented triple");
    kg_free_triples(tr, n);

    n = kg_query_triples_by_chunk(kg, "doc1:0000", tr, 8);
    ASSERT(n == 1 && strcmp(tr[0].predicate, "worked_at") == 0, "chunk doc1:0000 -> worked_at");
    kg_free_triples(tr, n);

    /* 1.2 reverse edges: who "invented" the machine? */
    GV_KGTriple rv[8];
    int rn = kg_query_reverse(kg, machine, "invented", rv, 8);
    ASSERT(rn == 1 && rv[0].subject_id == turing, "reverse: machine <-invented- Turing");
    kg_free_triples(rv, rn);

    kg_destroy(kg);

    /* 1.1 db-delegated similarity: entities mirrored into an attached DB */
    GV_KGConfig cfg2; kg_config_init(&cfg2); cfg2.embedding_dimension = 4;
    GV_KnowledgeGraph *kg2 = kg_create(&cfg2);
    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_HNSW);
    kg_attach_vector_db(kg2, db);

    float ea[4] = {1, 0, 0, 0};
    float eb[4] = {0, 1, 0, 0};
    uint64_t a = kg_add_entity(kg2, "A", "T", ea, 4);
    uint64_t b = kg_add_entity(kg2, "B", "T", eb, 4);
    ASSERT(a && b, "add 2 entities with embeddings (mirrored to db)");

    GV_KGSearchResult res[4];
    float q[4] = {0.9f, 0.1f, 0, 0}; /* closest to A */
    int sn = kg_search_similar(kg2, q, 4, 2, res);
    ASSERT(sn >= 1, "kg_search_similar via db returns results");
    ASSERT(res[0].entity_id == a, "top entity is A (db-delegated cosine)");
    kg_free_search_results(res, sn);

    kg_destroy(kg2);
    db_close(db);

    printf(failures ? "\nSOME TESTS FAILED (%d)\n" : "\nALL KG-COMBINED TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
