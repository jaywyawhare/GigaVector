#ifndef GIGAVECTOR_GV_GRAPH_PROP_INDEX_H
#define GIGAVECTOR_GV_GRAPH_PROP_INDEX_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "core/prop_value.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GV_GraphPropIndex GV_GraphPropIndex;

GV_GraphPropIndex *graph_prop_index_create(void);
void graph_prop_index_destroy(GV_GraphPropIndex *idx);

int graph_prop_index_add_node(GV_GraphPropIndex *idx, const char *key,
                              GV_PropValue value, uint64_t node_id);
int graph_prop_index_remove_node(GV_GraphPropIndex *idx, const char *key,
                                 GV_PropValue value, uint64_t node_id);

int graph_prop_index_find_exact(const GV_GraphPropIndex *idx, const char *key,
                                GV_PropValue value, uint64_t *out_ids,
                                size_t max_count);

int graph_prop_index_find_range(const GV_GraphPropIndex *idx, const char *key,
                                GV_PropValue min_val, GV_PropValue max_val,
                                uint64_t *out_ids, size_t max_count);

int graph_prop_index_add_edge(GV_GraphPropIndex *idx, const char *key,
                              GV_PropValue value, uint64_t edge_id);
int graph_prop_index_remove_edge(GV_GraphPropIndex *idx, const char *key,
                                 GV_PropValue value, uint64_t edge_id);
int graph_prop_index_find_edges_exact(const GV_GraphPropIndex *idx, const char *key,
                                      GV_PropValue value, uint64_t *out_ids,
                                      size_t max_count);
int graph_prop_index_find_edges_range(const GV_GraphPropIndex *idx, const char *key,
                                      GV_PropValue min_val, GV_PropValue max_val,
                                      uint64_t *out_ids, size_t max_count);

int graph_prop_index_save(const GV_GraphPropIndex *idx, FILE *out);
int graph_prop_index_load(GV_GraphPropIndex **idx, FILE *in);

size_t graph_prop_index_node_count(const GV_GraphPropIndex *idx);
size_t graph_prop_index_edge_count(const GV_GraphPropIndex *idx);

#ifdef __cplusplus
}
#endif
#endif
