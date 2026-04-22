#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#include "features/knowledge_graph.h"

#define ASSERT(cond, msg)         \
    do {                          \
        if (!(cond)) {            \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1;            \
        }                         \
    } while (0)

#define DIM 4

static void make_embedding(float *out, float base) {
    for (int i = 0; i < DIM; i++)
        out[i] = base + (float)i * 0.1f;
}

static int test_create_destroy(void) {
    GV_KnowledgeGraph *kg = kg_create(NULL);
    ASSERT(kg != NULL, "create with NULL config");
    kg_destroy(kg);

    GV_KGConfig cfg;
    kg_config_init(&cfg);
    ASSERT(cfg.entity_bucket_count == 4096, "default entity buckets");
    ASSERT(cfg.embedding_dimension == 128, "default embedding dim");

    cfg.embedding_dimension = DIM;
    cfg.entity_bucket_count = 64;
    kg = kg_create(&cfg);
    ASSERT(kg != NULL, "create with custom config");
    kg_destroy(kg);

    kg_destroy(NULL);
    return 0;
}

static int test_add_get_entities(void) {
    GV_KGConfig cfg;
    kg_config_init(&cfg);
    cfg.embedding_dimension = DIM;
    GV_KnowledgeGraph *kg = kg_create(&cfg);

    float emb1[DIM]; make_embedding(emb1, 1.0f);
    float emb2[DIM]; make_embedding(emb2, 2.0f);

    uint64_t e1 = kg_add_entity(kg, "Alice", "Person", emb1, DIM);
    ASSERT(e1 > 0, "add entity Alice");
    uint64_t e2 = kg_add_entity(kg, "Bob", "Person", emb2, DIM);
    ASSERT(e2 > 0, "add entity Bob");
    uint64_t e3 = kg_add_entity(kg, "Anthropic", "Company", NULL, 0);
    ASSERT(e3 > 0, "add entity without embedding");

    const GV_KGEntity *ent = kg_get_entity(kg, e1);
    ASSERT(ent != NULL, "get entity Alice");
    ASSERT(strcmp(ent->name, "Alice") == 0, "entity name");
    ASSERT(strcmp(ent->type, "Person") == 0, "entity type");
    ASSERT(ent->dimension == DIM, "entity dimension");

    ASSERT(kg_get_entity(kg, 99999) == NULL, "get nonexistent");

    kg_destroy(kg);
    return 0;
}

static int test_entity_properties(void) {
    GV_KnowledgeGraph *kg = kg_create(NULL);
    uint64_t e = kg_add_entity(kg, "Alice", "Person", NULL, 0);

    ASSERT(kg_set_entity_prop(kg, e, "email", "alice@test.com") == 0, "set prop");
    const char *val = kg_get_entity_prop(kg, e, "email");
    ASSERT(val != NULL && strcmp(val, "alice@test.com") == 0, "get prop");
    ASSERT(kg_get_entity_prop(kg, e, "missing") == NULL, "missing prop");

    kg_destroy(kg);
    return 0;
}

static int test_find_entities(void) {
    GV_KnowledgeGraph *kg = kg_create(NULL);
    kg_add_entity(kg, "Alice", "Person", NULL, 0);
    kg_add_entity(kg, "Bob", "Person", NULL, 0);
    kg_add_entity(kg, "Anthropic", "Company", NULL, 0);
    kg_add_entity(kg, "Alice", "Duplicate", NULL, 0);

    uint64_t ids[10];
    int n = kg_find_entities_by_type(kg, "Person", ids, 10);
    ASSERT(n == 2, "2 Person entities");

    n = kg_find_entities_by_name(kg, "Alice", ids, 10);
    ASSERT(n == 2, "2 entities named Alice");

    n = kg_find_entities_by_type(kg, "Unknown", ids, 10);
    ASSERT(n == 0, "0 Unknown type");

    kg_destroy(kg);
    return 0;
}

static int test_remove_entity(void) {
    GV_KnowledgeGraph *kg = kg_create(NULL);
    uint64_t e1 = kg_add_entity(kg, "Alice", "Person", NULL, 0);
    uint64_t e2 = kg_add_entity(kg, "Bob", "Person", NULL, 0);
    kg_add_relation(kg, e1, "knows", e2, 1.0f);

    GV_KGStats stats;
    kg_get_stats(kg, &stats);
    ASSERT(stats.entity_count == 2, "2 entities before remove");
    ASSERT(stats.relation_count == 1, "1 relation before remove");

    ASSERT(kg_remove_entity(kg, e1) == 0, "remove entity");
    kg_get_stats(kg, &stats);
    ASSERT(stats.entity_count == 1, "1 entity after remove");
    ASSERT(stats.relation_count == 0, "0 relations after cascade");

    kg_destroy(kg);
    return 0;
}

static int test_relations(void) {
    GV_KnowledgeGraph *kg = kg_create(NULL);
    uint64_t e1 = kg_add_entity(kg, "Alice", "Person", NULL, 0);
    uint64_t e2 = kg_add_entity(kg, "Bob", "Person", NULL, 0);
    uint64_t e3 = kg_add_entity(kg, "Anthropic", "Company", NULL, 0);

    uint64_t r1 = kg_add_relation(kg, e1, "works_at", e3, 1.0f);
    ASSERT(r1 > 0, "add relation 1");
    uint64_t r2 = kg_add_relation(kg, e2, "works_at", e3, 0.9f);
    ASSERT(r2 > 0, "add relation 2");
    uint64_t r3 = kg_add_relation(kg, e1, "knows", e2, 0.8f);
    ASSERT(r3 > 0, "add relation 3");

    const GV_KGRelation *rel = kg_get_relation(kg, r1);
    ASSERT(rel != NULL, "get relation");
    ASSERT(rel->subject_id == e1, "relation subject");
    ASSERT(rel->object_id == e3, "relation object");
    ASSERT(strcmp(rel->predicate, "works_at") == 0, "relation predicate");

    ASSERT(kg_remove_relation(kg, r3) == 0, "remove relation");
    ASSERT(kg_get_relation(kg, r3) == NULL, "removed relation gone");

    kg_destroy(kg);
    return 0;
}

static int test_triple_queries(void) {
    GV_KnowledgeGraph *kg = kg_create(NULL);
    uint64_t alice = kg_add_entity(kg, "Alice", "Person", NULL, 0);
    uint64_t bob = kg_add_entity(kg, "Bob", "Person", NULL, 0);
    uint64_t company = kg_add_entity(kg, "Anthropic", "Company", NULL, 0);

    kg_add_relation(kg, alice, "works_at", company, 1.0f);
    kg_add_relation(kg, bob, "works_at", company, 0.9f);
    kg_add_relation(kg, alice, "knows", bob, 0.8f);

    GV_KGTriple triples[10];

    int n = kg_query_triples(kg, NULL, "works_at", NULL, triples, 10);
    ASSERT(n == 2, "2 works_at triples");
    kg_free_triples(triples, n);

    n = kg_query_triples(kg, &alice, NULL, NULL, triples, 10);
    ASSERT(n == 2, "Alice has 2 outgoing triples");
    kg_free_triples(triples, n);

    n = kg_query_triples(kg, NULL, NULL, &company, triples, 10);
    ASSERT(n == 2, "Company has 2 incoming triples");
    kg_free_triples(triples, n);

    n = kg_query_triples(kg, &alice, "knows", NULL, triples, 10);
    ASSERT(n == 1, "Alice knows 1 entity");
    ASSERT(strcmp(triples[0].object_name, "Bob") == 0, "Alice knows Bob");
    kg_free_triples(triples, n);

    n = kg_query_triples(kg, NULL, NULL, NULL, triples, 10);
    ASSERT(n == 3, "3 total triples");
    kg_free_triples(triples, n);

    kg_destroy(kg);
    return 0;
}

static int test_semantic_search(void) {
    GV_KGConfig cfg;
    kg_config_init(&cfg);
    cfg.embedding_dimension = DIM;
    GV_KnowledgeGraph *kg = kg_create(&cfg);

    float emb1[DIM] = {1.0f, 0.0f, 0.0f, 0.0f};
    float emb2[DIM] = {0.9f, 0.1f, 0.0f, 0.0f};
    float emb3[DIM] = {0.0f, 0.0f, 1.0f, 0.0f};

    kg_add_entity(kg, "Alice", "Person", emb1, DIM);
    kg_add_entity(kg, "Bob", "Person", emb2, DIM);
    kg_add_entity(kg, "Anthropic", "Company", emb3, DIM);

    float query[DIM] = {0.95f, 0.05f, 0.0f, 0.0f};
    GV_KGSearchResult results[3];
    int n = kg_search_similar(kg, query, DIM, 3, results);
    ASSERT(n >= 2, "search returns at least 2 results");
    ASSERT(results[0].similarity > 0.9f, "top result high similarity");
    kg_free_search_results(results, n);

    kg_destroy(kg);
    return 0;
}

static int test_hybrid_search(void) {
    GV_KGConfig cfg;
    kg_config_init(&cfg);
    cfg.embedding_dimension = DIM;
    GV_KnowledgeGraph *kg = kg_create(&cfg);

    float emb1[DIM] = {1.0f, 0.0f, 0.0f, 0.0f};
    float emb2[DIM] = {0.9f, 0.1f, 0.0f, 0.0f};
    float emb3[DIM] = {0.8f, 0.2f, 0.0f, 0.0f};

    uint64_t alice = kg_add_entity(kg, "Alice", "Person", emb1, DIM);
    uint64_t bob = kg_add_entity(kg, "Bob", "Person", emb2, DIM);
    uint64_t company = kg_add_entity(kg, "Acme", "Company", emb3, DIM);

    kg_add_relation(kg, alice, "works_at", company, 1.0f);
    kg_add_relation(kg, bob, "works_at", company, 1.0f);

    float query[DIM] = {0.95f, 0.05f, 0.0f, 0.0f};
    GV_KGSearchResult results[5];

    int n = kg_hybrid_search(kg, query, DIM, "Person", NULL, 5, results);
    ASSERT(n == 2, "hybrid: 2 Person results");
    for (int i = 0; i < n; i++) {
        ASSERT(strcmp(results[i].type, "Person") == 0, "hybrid result is Person");
    }
    kg_free_search_results(results, n);

    n = kg_hybrid_search(kg, query, DIM, NULL, "works_at", 5, results);
    ASSERT(n >= 2, "hybrid: entities with works_at predicate");
    kg_free_search_results(results, n);

    kg_destroy(kg);
    return 0;
}

static int test_entity_resolution(void) {
    GV_KGConfig cfg;
    kg_config_init(&cfg);
    cfg.embedding_dimension = DIM;
    cfg.similarity_threshold = 0.9f;
    GV_KnowledgeGraph *kg = kg_create(&cfg);

    float emb[DIM] = {1.0f, 0.0f, 0.0f, 0.0f};
    uint64_t alice = kg_add_entity(kg, "Alice", "Person", emb, DIM);

    int resolved = kg_resolve_entity(kg, "Alice", "Person", emb, DIM);
    ASSERT((uint64_t)resolved == alice, "resolved to existing Alice");

    float emb2[DIM] = {0.0f, 1.0f, 0.0f, 0.0f};
    int resolved2 = kg_resolve_entity(kg, "Bob", "Person", emb2, DIM);
    ASSERT(resolved2 > 0 && (uint64_t)resolved2 != alice, "resolved to new entity");

    kg_destroy(kg);
    return 0;
}

static int test_merge_entities(void) {
    GV_KnowledgeGraph *kg = kg_create(NULL);
    uint64_t e1 = kg_add_entity(kg, "Alice", "Person", NULL, 0);
    uint64_t e2 = kg_add_entity(kg, "Alice Smith", "Person", NULL, 0);
    uint64_t e3 = kg_add_entity(kg, "Bob", "Person", NULL, 0);

    kg_add_relation(kg, e2, "knows", e3, 1.0f);
    kg_set_entity_prop(kg, e2, "email", "alice@test.com");

    ASSERT(kg_merge_entities(kg, e1, e2) == 0, "merge entities");

    ASSERT(kg_get_entity(kg, e2) == NULL, "merged entity removed");

    const char *email = kg_get_entity_prop(kg, e1, "email");
    ASSERT(email != NULL && strcmp(email, "alice@test.com") == 0, "merged prop transferred");

    kg_destroy(kg);
    return 0;
}

static int test_link_prediction(void) {
    GV_KGConfig cfg;
    kg_config_init(&cfg);
    cfg.embedding_dimension = DIM;
    GV_KnowledgeGraph *kg = kg_create(&cfg);

    float emb1[DIM] = {1.0f, 0.0f, 0.0f, 0.0f};
    float emb2[DIM] = {0.9f, 0.1f, 0.0f, 0.0f};
    float emb3[DIM] = {0.0f, 0.0f, 1.0f, 0.0f};

    uint64_t e1 = kg_add_entity(kg, "Alice", "Person", emb1, DIM);
    uint64_t e2 = kg_add_entity(kg, "Bob", "Person", emb2, DIM);
    kg_add_entity(kg, "Charlie", "Person", emb3, DIM);

    kg_add_relation(kg, e1, "knows", e2, 1.0f);

    GV_KGLinkPrediction preds[5];
    int n = kg_predict_links(kg, e1, 5, preds);
    ASSERT(n >= 0, "link prediction returns >= 0");

    kg_destroy(kg);
    return 0;
}

static int test_traversal(void) {
    GV_KnowledgeGraph *kg = kg_create(NULL);
    uint64_t a = kg_add_entity(kg, "A", "Node", NULL, 0);
    uint64_t b = kg_add_entity(kg, "B", "Node", NULL, 0);
    uint64_t c = kg_add_entity(kg, "C", "Node", NULL, 0);
    uint64_t d = kg_add_entity(kg, "D", "Node", NULL, 0);

    kg_add_relation(kg, a, "link", b, 1.0f);
    kg_add_relation(kg, b, "link", c, 1.0f);
    kg_add_relation(kg, c, "link", d, 1.0f);

    uint64_t nbrs[10];
    int n = kg_get_neighbors(kg, b, nbrs, 10);
    ASSERT(n >= 2, "b has >= 2 neighbors (a and c)");

    uint64_t visited[10];
    n = kg_traverse(kg, a, 10, visited, 10);
    ASSERT(n == 4, "traverse reaches all 4 entities");

    uint64_t path[10];
    n = kg_shortest_path(kg, a, d, path, 10);
    ASSERT(n >= 3, "path a->b->c->d has >= 3 nodes");
    ASSERT(path[0] == a, "path starts at a");
    ASSERT(path[n - 1] == d, "path ends at d");

    kg_destroy(kg);
    return 0;
}

static int test_subgraph(void) {
    GV_KnowledgeGraph *kg = kg_create(NULL);
    uint64_t a = kg_add_entity(kg, "A", "N", NULL, 0);
    uint64_t b = kg_add_entity(kg, "B", "N", NULL, 0);
    uint64_t c = kg_add_entity(kg, "C", "N", NULL, 0);
    uint64_t d = kg_add_entity(kg, "D", "N", NULL, 0);

    kg_add_relation(kg, a, "r", b, 1.0f);
    kg_add_relation(kg, b, "r", c, 1.0f);
    kg_add_relation(kg, c, "r", d, 1.0f);

    GV_KGSubgraph sg;
    ASSERT(kg_extract_subgraph(kg, a, 1, &sg) == 0, "extract subgraph radius 1");
    ASSERT(sg.entity_count == 2, "subgraph has 2 entities (a,b)");
    ASSERT(sg.relation_count >= 1, "subgraph has >= 1 relation");
    kg_free_subgraph(&sg);

    ASSERT(kg_extract_subgraph(kg, a, 3, &sg) == 0, "extract subgraph radius 3");
    ASSERT(sg.entity_count == 4, "subgraph has all 4 entities");
    kg_free_subgraph(&sg);

    kg_destroy(kg);
    return 0;
}

static int test_analytics(void) {
    GV_KGConfig cfg;
    kg_config_init(&cfg);
    cfg.embedding_dimension = DIM;
    GV_KnowledgeGraph *kg = kg_create(&cfg);

    float emb[DIM] = {1.0f, 0.0f, 0.0f, 0.0f};
    uint64_t a = kg_add_entity(kg, "A", "TypeA", emb, DIM);
    uint64_t b = kg_add_entity(kg, "B", "TypeA", NULL, 0);
    uint64_t c = kg_add_entity(kg, "C", "TypeB", NULL, 0);

    kg_add_relation(kg, a, "pred1", b, 1.0f);
    kg_add_relation(kg, a, "pred2", c, 1.0f);
    kg_add_relation(kg, b, "pred1", c, 1.0f);

    GV_KGStats stats;
    ASSERT(kg_get_stats(kg, &stats) == 0, "get stats");
    ASSERT(stats.entity_count == 3, "3 entities");
    ASSERT(stats.relation_count == 3, "3 relations");
    ASSERT(stats.type_count == 2, "2 distinct types");
    ASSERT(stats.predicate_count == 2, "2 distinct predicates");
    ASSERT(stats.embedding_count == 1, "1 entity with embedding");

    float cent = kg_entity_centrality(kg, a);
    ASSERT(cent > 0.0f, "a has positive centrality");

    char *types[5];
    int nt = kg_get_entity_types(kg, types, 5);
    ASSERT(nt == 2, "2 entity types");
    for (int i = 0; i < nt; i++) free(types[i]);

    char *preds[5];
    int np = kg_get_predicates(kg, preds, 5);
    ASSERT(np == 2, "2 predicates");
    for (int i = 0; i < np; i++) free(preds[i]);

    kg_destroy(kg);
    return 0;
}

static int test_save_load(void) {
    const char *path = "/tmp/test_gv_kg.gvkg";

    GV_KGConfig cfg;
    kg_config_init(&cfg);
    cfg.embedding_dimension = DIM;
    GV_KnowledgeGraph *kg = kg_create(&cfg);

    float emb[DIM] = {1.0f, 0.5f, 0.0f, 0.0f};
    uint64_t e1 = kg_add_entity(kg, "Alice", "Person", emb, DIM);
    uint64_t e2 = kg_add_entity(kg, "Bob", "Person", NULL, 0);
    kg_set_entity_prop(kg, e1, "email", "alice@test.com");
    kg_add_relation(kg, e1, "knows", e2, 0.8f);

    ASSERT(kg_save(kg, path) == 0, "save KG");
    kg_destroy(kg);

    GV_KnowledgeGraph *kg2 = kg_load(path);
    ASSERT(kg2 != NULL, "load KG");

    GV_KGStats stats;
    kg_get_stats(kg2, &stats);
    ASSERT(stats.entity_count == 2, "loaded entity count");
    ASSERT(stats.relation_count == 1, "loaded relation count");

    const GV_KGEntity *ent = kg_get_entity(kg2, e1);
    ASSERT(ent != NULL, "loaded entity exists");
    ASSERT(strcmp(ent->name, "Alice") == 0, "loaded entity name");
    ASSERT(ent->dimension == DIM, "loaded entity embedding dim");

    const char *email = kg_get_entity_prop(kg2, e1, "email");
    ASSERT(email != NULL && strcmp(email, "alice@test.com") == 0, "loaded entity prop");

    GV_KGTriple triples[5];
    int n = kg_query_triples(kg2, NULL, "knows", NULL, triples, 5);
    ASSERT(n == 1, "loaded triple query works");
    kg_free_triples(triples, n);

    kg_destroy(kg2);
    unlink(path);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestEntry;

int main(void) {
    TestEntry tests[] = {
        {"create/destroy",          test_create_destroy},
        {"add/get entities",        test_add_get_entities},
        {"entity properties",       test_entity_properties},
        {"find entities",           test_find_entities},
        {"remove entity (cascade)", test_remove_entity},
        {"relations",               test_relations},
        {"triple queries (SPO)",    test_triple_queries},
        {"semantic search",         test_semantic_search},
        {"hybrid search",           test_hybrid_search},
        {"entity resolution",       test_entity_resolution},
        {"merge entities",          test_merge_entities},
        {"link prediction",         test_link_prediction},
        {"traversal",               test_traversal},
        {"subgraph extraction",     test_subgraph},
        {"analytics",               test_analytics},
        {"save/load",               test_save_load},
    };

    int total = (int)(sizeof(tests) / sizeof(tests[0]));
    int passed = 0;

    for (int i = 0; i < total; i++) {
        if (tests[i].fn() == 0) {
            passed++;
        }
    }

    return (passed == total) ? 0 : 1;
}
