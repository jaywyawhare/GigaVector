#ifndef GIGAVECTOR_GV_FILTER_H
#define GIGAVECTOR_GV_FILTER_H

#include <stddef.h>

#include "gv_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque handle for a compiled filter expression.
 */
typedef struct GV_Filter GV_Filter;

/**
 * @brief Parse a filter expression into an executable filter object.
 *
 * The expression language supports:
 * - Logical operators: AND, OR, NOT (case-insensitive).
 * - Parentheses for grouping: ( and ).
 * - Comparison operators: ==, !=, >, >=, <, <=.
 * - String matching: CONTAINS, PREFIX.
 *
 * Examples:
 * - category == \"A\" AND score >= 0.5
 * - (country == \"US\" OR country == \"CA\") AND NOT status == \"deleted\"
 * - tag CONTAINS \"news\"
 * - prefix PREFIX \"user:\"
 *
 * @param expr Null-terminated filter expression string; must be non-NULL.
 * @return Pointer to compiled filter on success, or NULL on parse/allocation error.
 */
GV_Filter *gv_filter_parse(const char *expr);

/**
 * @brief Evaluate a compiled filter expression against a vector's metadata.
 *
 * Numeric comparisons attempt to parse metadata values as double-precision
 * numbers; if parsing fails, numeric comparisons evaluate to false. String
 * comparisons operate on the raw metadata value.
 *
 * @param filter Compiled filter expression; must be non-NULL.
 * @param vector Vector whose metadata will be evaluated; must be non-NULL.
 * @return 1 if the vector matches the filter, 0 if it does not match, or -1 on error.
 */
int gv_filter_eval(const GV_Filter *filter, const GV_Vector *vector);

/**
 * @brief Destroy a compiled filter expression and free all associated memory.
 *
 * Safe to call with NULL.
 *
 * @param filter Filter handle to destroy; may be NULL.
 */
void gv_filter_destroy(GV_Filter *filter);

#ifdef __cplusplus
}
#endif

#endif


