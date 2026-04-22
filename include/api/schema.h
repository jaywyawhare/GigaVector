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

GV_Schema *schema_create(uint32_t version);
/**
 * @brief Destroy an instance and free associated resources.
 *
 * @param schema Schema instance.
 */
void schema_destroy(GV_Schema *schema);
GV_Schema *schema_copy(const GV_Schema *schema);

int schema_add_field(GV_Schema *schema, const char *name, GV_SchemaFieldType type,
                         int required, const char *default_value);
/**
 * @brief Perform the operation.
 *
 * @param schema Schema instance.
 * @param name Name string.
 * @return 0 on success, -1 on error.
 */
int schema_remove_field(GV_Schema *schema, const char *name);
/**
 * @brief Perform the operation.
 *
 * @param schema Schema instance.
 * @param name Name string.
 * @return 0 on success, -1 on error.
 */
int schema_has_field(const GV_Schema *schema, const char *name);
const GV_SchemaField *schema_get_field(const GV_Schema *schema, const char *name);
/**
 * @brief Return the number of stored items.
 *
 * @param schema Schema instance.
 * @return Count value.
 */
size_t schema_field_count(const GV_Schema *schema);

int schema_validate(const GV_Schema *schema, const char *const *keys,
                        const char *const *values, size_t count);

typedef struct {
    char name[64];
    int added;      /* 1 if field was added in new schema */
    int removed;    /* 1 if field was removed in new schema */
    int type_changed; /* 1 if field type changed */
    GV_SchemaFieldType old_type;
    GV_SchemaFieldType new_type;
} GV_SchemaDiff;

int schema_diff(const GV_Schema *old_schema, const GV_Schema *new_schema,
                    GV_SchemaDiff *diffs, size_t max_diffs);

/**
 * @brief Query a boolean condition.
 *
 * @param old_schema old_schema.
 * @param new_schema new_schema.
 * @return 1 if true, 0 if false, -1 on error.
 */
int schema_is_compatible(const GV_Schema *old_schema, const GV_Schema *new_schema);

/**
 * @brief Save state to a file.
 *
 * @param schema Schema instance.
 * @param out Output buffer.
 * @return 0 on success, -1 on error.
 */
int schema_save(const GV_Schema *schema, FILE *out);
GV_Schema *schema_load(FILE *in);
char *schema_to_json(const GV_Schema *schema);

#ifdef __cplusplus
}
#endif
#endif
