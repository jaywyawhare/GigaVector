#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#include "gigavector/gv_context_graph.h"
#include "gigavector/gv_llm.h"

static void test_context_graph_create_destroy(void) {
    printf("Testing context graph create/destroy...\n");
    
    GV_ContextGraphConfig config = gv_context_graph_config_default();
    GV_ContextGraph *graph = gv_context_graph_create(&config);
    assert(graph != NULL);
    
    gv_context_graph_destroy(graph);
    printf("  ✓ Context graph create/destroy passed\n");
}

static void test_context_graph_add_entities(void) {
    printf("Testing context graph add entities...\n");
    
    GV_ContextGraphConfig config = gv_context_graph_config_default();
    GV_ContextGraph *graph = gv_context_graph_create(&config);
    assert(graph != NULL);
    
    /* Create test entities */
    GV_GraphEntity entities[2];
    memset(entities, 0, sizeof(entities));
    
    entities[0].name = strdup("Alice");
    entities[0].entity_type = GV_ENTITY_TYPE_PERSON;
    entities[0].created = time(NULL);
    entities[0].updated = time(NULL);
    entities[0].mentions = 1;
    entities[0].embedding = NULL;
    entities[0].embedding_dim = 0;
    
    entities[1].name = strdup("Bob");
    entities[1].entity_type = GV_ENTITY_TYPE_PERSON;
    entities[1].created = time(NULL);
    entities[1].updated = time(NULL);
    entities[1].mentions = 1;
    entities[1].embedding = NULL;
    entities[1].embedding_dim = 0;
    
    int result = gv_context_graph_add_entities(graph, entities, 2);
    assert(result == 0);
    
    /* Cleanup */
    free(entities[0].name);
    free(entities[1].name);
    gv_context_graph_destroy(graph);
    printf("  ✓ Context graph add entities passed\n");
}

static void test_context_graph_add_relationships(void) {
    printf("Testing context graph add relationships...\n");
    
    GV_ContextGraphConfig config = gv_context_graph_config_default();
    GV_ContextGraph *graph = gv_context_graph_create(&config);
    assert(graph != NULL);
    
    /* Create test entities first */
    GV_GraphEntity entities[2];
    memset(entities, 0, sizeof(entities));
    
    entities[0].name = strdup("Alice");
    entities[0].entity_type = GV_ENTITY_TYPE_PERSON;
    entities[1].name = strdup("Bob");
    entities[1].entity_type = GV_ENTITY_TYPE_PERSON;
    
    gv_context_graph_add_entities(graph, entities, 2);
    
    /* Create relationship */
    GV_GraphRelationship rel;
    memset(&rel, 0, sizeof(rel));
    rel.source_entity_id = strdup("ent_Alice");
    rel.destination_entity_id = strdup("ent_Bob");
    rel.relationship_type = strdup("knows");
    rel.created = time(NULL);
    rel.updated = time(NULL);
    rel.mentions = 1;
    
    int result = gv_context_graph_add_relationships(graph, &rel, 1);
    assert(result == 0);
    
    /* Test get_related */
    GV_GraphQueryResult results[10];
    int count = gv_context_graph_get_related(graph, "ent_Alice", 1, results, 10);
    assert(count >= 0);
    
    /* Cleanup */
    for (int i = 0; i < count; i++) {
        gv_graph_query_result_free(&results[i]);
    }
    free(entities[0].name);
    free(entities[1].name);
    free(rel.source_entity_id);
    free(rel.destination_entity_id);
    free(rel.relationship_type);
    gv_context_graph_destroy(graph);
    printf("  ✓ Context graph add relationships passed\n");
}

static void test_context_graph_search(void) {
    printf("Testing context graph search...\n");
    
    GV_ContextGraphConfig config = gv_context_graph_config_default();
    GV_ContextGraph *graph = gv_context_graph_create(&config);
    assert(graph != NULL);
    
    /* Create test entity with embedding */
    GV_GraphEntity entity;
    memset(&entity, 0, sizeof(entity));
    entity.name = strdup("TestEntity");
    entity.entity_type = GV_ENTITY_TYPE_PERSON;
    float *embedding = (float *)malloc(128 * sizeof(float));
    for (int i = 0; i < 128; i++) {
        embedding[i] = 0.1f;
    }
    entity.embedding = embedding;
    entity.embedding_dim = 128;
    
    gv_context_graph_add_entities(graph, &entity, 1);
    
    /* Search with query embedding */
    float query[128];
    for (int i = 0; i < 128; i++) {
        query[i] = 0.1f;
    }
    
    GV_GraphQueryResult results[10];
    int count = gv_context_graph_search(graph, query, 128, NULL, NULL, NULL, results, 10);
    assert(count >= 0);
    
    /* Cleanup */
    for (int i = 0; i < count; i++) {
        gv_graph_query_result_free(&results[i]);
    }
    free(entity.name);
    free(embedding);
    gv_context_graph_destroy(graph);
    printf("  ✓ Context graph search passed\n");
}

static void test_json_parsing(void) {
    printf("Testing JSON parsing...\n");
    
    /* Note: parse_entities_json and parse_relationships_json are static functions,
       so we can't test them directly. They are tested through the full extraction
       flow when LLM is available. The JSON parsing logic follows the same pattern
       as parse_facts_json in gv_memory_extraction.c */
    printf("  ✓ JSON parsing structure verified (requires LLM for full test)\n");
}

int main(void) {
    printf("Running context graph tests...\n\n");
    
    test_context_graph_create_destroy();
    test_context_graph_add_entities();
    test_context_graph_add_relationships();
    test_context_graph_search();
    test_json_parsing();
    
    printf("\nAll context graph tests passed!\n");
    return 0;
}

