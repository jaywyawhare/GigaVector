#ifndef GIGAVECTOR_GV_CYPHER_VECTOR_H
#define GIGAVECTOR_GV_CYPHER_VECTOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations */
struct GV_CypherEngine;

/*
 * Vector distance function types supported in Cypher expressions.
 */
typedef enum {
    GV_VECDIST_L2,         /* Euclidean distance */
    GV_VECDIST_COSINE,     /* Cosine distance (1 - cosine_similarity) */
    GV_VECDIST_DOT,        /* Negative dot product (lower = more similar) */
    GV_VECDIST_HAMMING     /* Hamming distance (for binary vectors) */
} GV_CypherVecDistType;

/*
 * Evaluate a vector distance for use inside cypher.c OPD_FUNC handler.
 *
 * @param stored_str  Comma-separated stored vector string
 * @param query_str   Comma-separated query vector string
 * @param metric      Metric name ("l2", "cosine", "dot", "hamming") or NULL for L2
 * @return Heap-allocated string of the distance, or empty string on error
 */
char *cypher_eval_vector_distance(const char *stored_str, const char *query_str,
                                  const char *metric);

/*
 * Register vector distance functions in a Cypher context.
 *
 * After calling this, the Cypher engine supports:
 * - vector_distance(prop_ref, $query_param) — L2 by default
 * - vector_distance(prop_ref, $query_param, 'cosine') — specified metric
 * - vector_distance_l2(prop_ref, $query_param)
 * - vector_distance_cosine(prop_ref, $query_param)
 *
 * @param ctx    Cypher context to extend
 * @param dimension  Vector dimension for distance computation
 * @return 0 on success, -1 on error
 */
int cypher_register_vector_functions(struct GV_CypherEngine *ctx, size_t dimension);

/*
 * Evaluate a vector distance expression between a stored property value
 * and a query vector.
 *
 * @param stored    Stored vector (from entity/node property)
 * @param stored_dim Dimension of stored vector
 * @param query     Query vector
 * @param query_dim Dimension of query vector
 * @param type      Distance metric
 * @return Distance value, or -1.0f on error
 */
float cypher_vector_distance(const float *stored, size_t stored_dim,
                             const float *query, size_t query_dim,
                             GV_CypherVecDistType type);

/*
 * Parse a distance type string: "l2", "euclidean", "cosine", "dot", "hamming"
 * @return Parsed type, or GV_VECDIST_L2 as default
 */
GV_CypherVecDistType cypher_parse_vecdist_type(const char *s);

#ifdef __cplusplus
}
#endif
#endif
