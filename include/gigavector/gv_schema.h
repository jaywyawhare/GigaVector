#ifndef GIGAVECTOR_GV_SCHEMA_H
#define GIGAVECTOR_GV_SCHEMA_H
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GV_SCHEMA_STRING = 0,
    GV_SCHEMA_INT = 1,
    GV_SCHEMA_FLOAT = 2,
    GV_SCHEMA_BOOL = 3
} GV_SchemaFieldType;

typedef struct {
    char name[64];
    GV_SchemaFieldType type;
    int required;
    char default_value[256];  /* string representation of default */
} GV_SchemaField;

typedef struct {
    uint32_t version;
    GV_SchemaField *fields;
    size_t field_count;
    size_t field_capacity;
} GV_Schema;

/* Create/destroy */
GV_Schema *gv_schema_create(uint32_t version);
void gv_schema_destroy(GV_Schema *schema);
GV_Schema *gv_schema_copy(const GV_Schema *schema);

/* Field management */
int gv_schema_add_field(GV_Schema *schema, const char *name, GV_SchemaFieldType type,
                         int required, const char *default_value);
int gv_schema_remove_field(GV_Schema *schema, const char *name);
int gv_schema_has_field(const GV_Schema *schema, const char *name);
const GV_SchemaField *gv_schema_get_field(const GV_Schema *schema, const char *name);
size_t gv_schema_field_count(const GV_Schema *schema);

/* Validation */
int gv_schema_validate(const GV_Schema *schema, const char *const *keys,
                        const char *const *values, size_t count);

/* Migration - compute diff between schemas */
typedef struct {
    char name[64];
    int added;      /* 1 if field was added in new schema */
    int removed;    /* 1 if field was removed in new schema */
    int type_changed; /* 1 if field type changed */
    GV_SchemaFieldType old_type;
    GV_SchemaFieldType new_type;
} GV_SchemaDiff;

int gv_schema_diff(const GV_Schema *old_schema, const GV_Schema *new_schema,
                    GV_SchemaDiff *diffs, size_t max_diffs);

/* Check if migration from old to new is safe (no required fields removed, no type conflicts) */
int gv_schema_is_compatible(const GV_Schema *old_schema, const GV_Schema *new_schema);

/* Persistence */
int gv_schema_save(const GV_Schema *schema, FILE *out);
GV_Schema *gv_schema_load(FILE *in);
char *gv_schema_to_json(const GV_Schema *schema);

#ifdef __cplusplus
}
#endif
#endif
