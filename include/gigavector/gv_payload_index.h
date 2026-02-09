#ifndef GIGAVECTOR_GV_PAYLOAD_INDEX_H
#define GIGAVECTOR_GV_PAYLOAD_INDEX_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GV_FIELD_INT = 0,
    GV_FIELD_FLOAT = 1,
    GV_FIELD_STRING = 2,
    GV_FIELD_BOOL = 3
} GV_FieldType;

typedef enum {
    GV_PAYLOAD_OP_EQ = 0,
    GV_PAYLOAD_OP_NE = 1,
    GV_PAYLOAD_OP_GT = 2,
    GV_PAYLOAD_OP_GE = 3,
    GV_PAYLOAD_OP_LT = 4,
    GV_PAYLOAD_OP_LE = 5,
    GV_PAYLOAD_OP_CONTAINS = 6,
    GV_PAYLOAD_OP_PREFIX = 7
} GV_PayloadOp;

typedef struct {
    const char *field_name;
    GV_PayloadOp op;
    union {
        int64_t int_val;
        double float_val;
        const char *string_val;
        int bool_val;
    } value;
    GV_FieldType field_type;
} GV_PayloadQuery;

typedef struct GV_PayloadIndex GV_PayloadIndex;

GV_PayloadIndex *gv_payload_index_create(void);
void gv_payload_index_destroy(GV_PayloadIndex *idx);

/* Schema management */
int gv_payload_index_add_field(GV_PayloadIndex *idx, const char *name, GV_FieldType type);
int gv_payload_index_remove_field(GV_PayloadIndex *idx, const char *name);
int gv_payload_index_field_count(const GV_PayloadIndex *idx);

/* Indexing */
int gv_payload_index_insert_int(GV_PayloadIndex *idx, size_t vector_id, const char *field, int64_t value);
int gv_payload_index_insert_float(GV_PayloadIndex *idx, size_t vector_id, const char *field, double value);
int gv_payload_index_insert_string(GV_PayloadIndex *idx, size_t vector_id, const char *field, const char *value);
int gv_payload_index_insert_bool(GV_PayloadIndex *idx, size_t vector_id, const char *field, int value);
int gv_payload_index_remove(GV_PayloadIndex *idx, size_t vector_id);

/* Querying - returns matching vector IDs */
int gv_payload_index_query(const GV_PayloadIndex *idx, const GV_PayloadQuery *query,
                            size_t *result_ids, size_t max_results);

/* Multi-condition query (AND of all conditions) */
int gv_payload_index_query_multi(const GV_PayloadIndex *idx, const GV_PayloadQuery *queries,
                                  size_t query_count, size_t *result_ids, size_t max_results);

/* Stats */
size_t gv_payload_index_total_entries(const GV_PayloadIndex *idx);

#ifdef __cplusplus
}
#endif
#endif
