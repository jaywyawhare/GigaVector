#ifndef GIGAVECTOR_GV_GRAPH_SCHEMA_H
#define GIGAVECTOR_GV_GRAPH_SCHEMA_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "core/prop_value.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GV_GraphSchema GV_GraphSchema;

typedef enum {
    GV_SCHEMA_CONSTRAINT_REQUIRED,
    GV_SCHEMA_CONSTRAINT_UNIQUE,
    GV_SCHEMA_CONSTRAINT_TYPE,
    GV_SCHEMA_CONSTRAINT_RANGE_INT,
    GV_SCHEMA_CONSTRAINT_RANGE_FLOAT
} GV_SchemaConstraintType;

typedef struct {
    GV_SchemaConstraintType type;
    char *label;
    char *property_key;
    int is_node;
    GV_PropType expected_type;
    int64_t int_min, int_max;
    double float_min, float_max;
} GV_SchemaConstraint;

GV_GraphSchema *graph_schema_create(void);
void graph_schema_destroy(GV_GraphSchema *schema);

int graph_schema_add_required(GV_GraphSchema *schema, const char *label,
                              const char *prop_key, int is_node);
int graph_schema_add_unique(GV_GraphSchema *schema, const char *label,
                            const char *prop_key, int is_node);
int graph_schema_add_type(GV_GraphSchema *schema, const char *label,
                          const char *prop_key, int is_node, GV_PropType type);
int graph_schema_add_range_int(GV_GraphSchema *schema, const char *label,
                               const char *prop_key, int is_node,
                               int64_t min_val, int64_t max_val);
int graph_schema_add_range_float(GV_GraphSchema *schema, const char *label,
                                  const char *prop_key, int is_node,
                                  double min_val, double max_val);

int graph_schema_remove_constraint(GV_GraphSchema *schema, const char *label,
                                   const char *prop_key, int is_node);

int graph_schema_validate_node(const GV_GraphSchema *schema,
                               const char *label,
                               const char **prop_keys,
                               const GV_PropValue *prop_values,
                               size_t prop_count,
                               char *error_buf, size_t error_buf_len);

int graph_schema_validate_edge(const GV_GraphSchema *schema,
                               const char *label,
                               const char **prop_keys,
                               const GV_PropValue *prop_values,
                               size_t prop_count,
                               char *error_buf, size_t error_buf_len);

int graph_schema_register_unique(GV_GraphSchema *schema, const char *label,
                                 const char *prop_key, int is_node,
                                 GV_PropValue value, uint64_t entity_id);
int graph_schema_unregister_unique(GV_GraphSchema *schema, const char *label,
                                   const char *prop_key, int is_node,
                                   GV_PropValue value);
int graph_schema_check_unique(const GV_GraphSchema *schema, const char *label,
                              const char *prop_key, int is_node,
                              GV_PropValue value);

int graph_schema_get_constraints(const GV_GraphSchema *schema,
                                 const char *label, int is_node,
                                 GV_SchemaConstraint *out, size_t max_count);

int graph_schema_save(const GV_GraphSchema *schema, FILE *out);
int graph_schema_load(GV_GraphSchema **schema, FILE *in);

size_t graph_schema_constraint_count(const GV_GraphSchema *schema);

#ifdef __cplusplus
}
#endif
#endif
