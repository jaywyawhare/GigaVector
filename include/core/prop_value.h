/**
 * @file prop_value.h
 * @brief Tagged-union property value shared by graph_db and knowledge_graph.
 */

#ifndef GIGAVECTOR_GV_PROP_VALUE_H
#define GIGAVECTOR_GV_PROP_VALUE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GV_PROP_NULL = 0,
    GV_PROP_STRING,
    GV_PROP_INT64,
    GV_PROP_FLOAT64,
    GV_PROP_BOOL,
    GV_PROP_BLOB       /* arbitrary byte array */
} GV_PropType;

typedef struct {
    GV_PropType type;
    union {
        char   *s;          /* heap-allocated string (GV_PROP_STRING) */
        int64_t i;          /* integer (GV_PROP_INT64) */
        double  f;          /* float (GV_PROP_FLOAT64) */
        int     b;          /* boolean (GV_PROP_BOOL) */
        struct { uint8_t *data; size_t len; } blob; /* GV_PROP_BLOB */
    } as;
} GV_PropValue;

/* Construct helpers */
GV_PropValue gv_prop_string(const char *s);
GV_PropValue gv_prop_int64(int64_t v);
GV_PropValue gv_prop_float64(double v);
GV_PropValue gv_prop_bool(int v);
GV_PropValue gv_prop_blob(const uint8_t *data, size_t len);
GV_PropValue gv_prop_null(void);

/* Deep copy */
GV_PropValue gv_prop_clone(GV_PropValue v);

/* Free heap members (does NOT free the GV_PropValue itself) */
void gv_prop_free(GV_PropValue *v);

/* Compare: returns <0, 0, >0 */
int gv_prop_compare(GV_PropValue a, GV_PropValue b);

/* String conversion (caller frees result) */
char *gv_prop_to_string(GV_PropValue v);

/* Parse string into typed value using type hint */
GV_PropValue gv_prop_parse(const char *raw, GV_PropType hint);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_PROP_VALUE_H */
