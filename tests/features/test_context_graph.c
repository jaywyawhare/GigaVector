#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include "core/utils.h"

#include "features/context_graph.h"
#include "multimodal/llm.h"

static void test_context_graph_create_destroy(void) {
    GV_ContextGraphConfig config = context_graph_config_default();
    GV_ContextGraph *graph = context_graph_create(&config);
    assert(graph != NULL);
    
    context_graph_destroy(graph);
}

static void test_context_graph_add_entities(void) {
    GV_ContextGraphConfig config = context_graph_config_default();
    GV_ContextGraph *graph = context_graph_create(&config);
    assert(graph != NULL);
    
    GV_GraphEntity entities[2];
    memset(entities, 0, sizeof(entities));
    
    entities[0].name = gv_dup_cstr("Alice");
    entities[0].entity_type = GV_ENTITY_TYPE_PERSON;
    entities[0].created = time(NULL);
    entities[0].updated = time(NULL);
    entities[0].mentions = 1;
    entities[0].embedding = NULL;
    entities[0].embedding_dim = 0;
    
    entities[1].name = gv_dup_cstr("Bob");
    entities[1].entity_type = GV_ENTITY_TYPE_PERSON;
    entities[1].created = time(NULL);
    entities[1].updated = time(NULL);
    entities[1].mentions = 1;
    entities[1].embedding = NULL;
    entities[1].embedding_dim = 0;
    
    assert(context_graph_add_entities(graph, entities, 2) == 0);
    
    free(entities[0].name);
    free(entities[1].name);
    context_graph_destroy(graph);
}

static void test_context_graph_add_relationships(void) {
    GV_ContextGraphConfig config = context_graph_config_default();
    GV_ContextGraph *graph = context_graph_create(&config);
    assert(graph != NULL);
    
    GV_GraphEntity entities[2];
    memset(entities, 0, sizeof(entities));
    
    entities[0].name = gv_dup_cstr("Alice");
    entities[0].entity_type = GV_ENTITY_TYPE_PERSON;
    entities[1].name = gv_dup_cstr("Bob");
    entities[1].entity_type = GV_ENTITY_TYPE_PERSON;
    
    context_graph_add_entities(graph, entities, 2);
    
    GV_GraphRelationship rel;
    memset(&rel, 0, sizeof(rel));
    rel.source_entity_id = gv_dup_cstr("ent_Alice");
    rel.destination_entity_id = gv_dup_cstr("ent_Bob");
    rel.relationship_type = gv_dup_cstr("knows");
    rel.created = time(NULL);
    rel.updated = time(NULL);
    rel.mentions = 1;
    
    assert(context_graph_add_relationships(graph, &rel, 1) == 0);
    
    GV_GraphQueryResult results[10];
    int count = context_graph_get_related(graph, "ent_Alice", 1, results, 10);
    assert(count >= 0);
    
    for (int i = 0; i < count; i++) {
        graph_query_result_free(&results[i]);
    }
    free(entities[0].name);
    free(entities[1].name);
    free(rel.source_entity_id);
    free(rel.destination_entity_id);
    free(rel.relationship_type);
    context_graph_destroy(graph);
}

static void test_context_graph_search(void) {
    GV_ContextGraphConfig config = context_graph_config_default();
    GV_ContextGraph *graph = context_graph_create(&config);
    assert(graph != NULL);
    
    GV_GraphEntity entity;
    memset(&entity, 0, sizeof(entity));
    entity.name = gv_dup_cstr("TestEntity");
    entity.entity_type = GV_ENTITY_TYPE_PERSON;
    float *embedding = (float *)malloc(128 * sizeof(float));
    for (int i = 0; i < 128; i++) {
        embedding[i] = 0.1f;
    }
    entity.embedding = embedding;
    entity.embedding_dim = 128;
    
    context_graph_add_entities(graph, &entity, 1);
    
    float query[128];
    for (int i = 0; i < 128; i++) {
        query[i] = 0.1f;
    }
    
    GV_GraphQueryResult results[10];
    int count = context_graph_search(graph, query, 128, NULL, NULL, NULL, results, 10);
    assert(count >= 0);
    
    for (int i = 0; i < count; i++) {
        graph_query_result_free(&results[i]);
    }
    free(entity.name);
    free(embedding);
    context_graph_destroy(graph);
}

static void test_json_parsing(void) {
    /* Note: parse_entities_json and parse_relationships_json are static functions,
       so we can't test them directly. They are tested through the full extraction
       flow when LLM is available. The JSON parsing logic follows the same pattern
       as parse_facts_json in memory_extraction.c */
}

int main(void) {
    test_context_graph_create_destroy();
    test_context_graph_add_entities();
    test_context_graph_add_relationships();
    test_context_graph_search();
    test_json_parsing();
    return 0;
}

