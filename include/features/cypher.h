/**
 * @file cypher.h
 * @brief Cypher query language over the GigaVector knowledge graph.
 *
 * A pragmatic subset of openCypher mapped onto the SPO triple store:
 *
 *   CREATE (a:Person {name:'Alice'})
 *   CREATE (a:Person {name:'Alice'})-[:KNOWS]->(b:Person {name:'Bob'})
 *   MATCH (a:Person)-[r:KNOWS]->(b) RETURN a.name, type(r), b.name
 *   MATCH (a {name:'Alice'})-[:KNOWS]->(b) WHERE b.name = 'Bob' RETURN b.name
 *   MATCH (n:Person) RETURN n.name
 *
 * Supported:
 *  - Patterns: multi-hop (a)-[:X]->(b)-[:Y]->(c), all directions & rel types,
 *    and variable-length paths (a)-[:X*1..3]->(b) (also *, *n, *n.., *..m).
 *  - Node patterns: variable, :Label, {prop:'val', ...} maps.
 *  - Clauses: MATCH / OPTIONAL MATCH (multiple), UNWIND, WITH (pipelining +
 *    WHERE), CREATE, MERGE (+ ON CREATE SET / ON MATCH SET), SET, REMOVE,
 *    DELETE / DETACH DELETE, RETURN.
 *  - WHERE: AND/OR/NOT/parentheses; = <> < > <= >= ; CONTAINS / STARTS WITH /
 *    ENDS WITH; IN [list]; IS NULL / IS NOT NULL; numeric-aware comparisons.
 *  - Expressions (in RETURN/WHERE/WITH/ORDER): arithmetic + - * / % with
 *    precedence and string concat; CASE (simple + generic); scalar functions
 *    toUpper, toLower, trim, size, length, substring, coalesce, toString,
 *    toInteger, toFloat, abs, ceil, floor, round, sqrt, sign, id, labels, type.
 *  - RETURN / WITH: variables, `var.property`, `type(relVar)`, expressions;
 *    AS aliases; DISTINCT; RETURN *; aggregations count(*), count(x),
 *    collect(x), sum/avg/min/max(x) with grouping (incl. across a WITH
 *    boundary); ORDER BY ... [ASC|DESC]; SKIP n; LIMIT n.
 *  - First-class lists: literals [a,b,c], range(a,b[,step]), indexing l[i],
 *    size/head/last, IN over a list, UNWIND over any list expression, and
 *    list comprehensions [x IN list WHERE c | expr].
 *  - First-class maps: literals {k: v}, m[key] / m.key access, keys(m).
 *  - Pattern comprehensions [(a)-[:X]->(b) WHERE c | expr] and EXISTS(pattern).
 *  - Path variables: MATCH p = (...) with length(p), nodes(p), relationships(p).
 *  - FOREACH (x IN list | CREATE ... | SET ...), CALL db.labels() /
 *    db.relationshipTypes(), and query parameters ($name via cypher_set_parameter).
 *
 * (REMOVE clears a property; the KG has no property-delete primitive. FOREACH
 * bodies support CREATE/SET. Values are string-typed internally.)
 */
#ifndef GIGAVECTOR_GV_CYPHER_H
#define GIGAVECTOR_GV_CYPHER_H

#include <stddef.h>
#include "features/knowledge_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char  **column_names;          /**< column_count strings. */
    size_t  column_count;
    char  **column_values;         /**< Row-major: row_count * column_count cells. */
    size_t  row_count;
    size_t  nodes_created;         /**< Nodes created by a CREATE statement. */
    size_t  relationships_created; /**< Relationships created by a CREATE statement. */
} GV_CypherResult;

typedef struct GV_CypherEngine GV_CypherEngine;

/**
 * @brief Create a Cypher engine bound to a knowledge graph.
 * @param kg Knowledge graph; must outlive the engine.
 * @return Engine, or NULL on error.
 */
GV_CypherEngine *cypher_create(GV_KnowledgeGraph *kg);

/** @brief Destroy a Cypher engine (safe with NULL). */
void cypher_destroy(GV_CypherEngine *eng);

/**
 * @brief Set a query parameter ($name) as a string value (upsert).
 * Referenced in queries as `$name`. Persists across executes until changed.
 * @return 0 on success, -1 on error.
 */
int cypher_set_parameter(GV_CypherEngine *eng, const char *name, const char *value);

/**
 * @brief Execute a Cypher statement.
 *
 * On success the caller owns @p result and frees it with cypher_free_result().
 * On error, -1 is returned and cypher_last_error() describes the problem.
 *
 * @return 0 on success, -1 on error.
 */
int cypher_execute(GV_CypherEngine *eng, const char *query, GV_CypherResult *result);

/** @brief Free a result set (safe with NULL); zeroes the struct. */
void cypher_free_result(GV_CypherResult *result);

/** @brief Last error message for the engine (never NULL). */
const char *cypher_last_error(const GV_CypherEngine *eng);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_CYPHER_H */
