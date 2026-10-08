#ifndef GIGAVECTOR_GV_REST_HANDLERS_H
#define GIGAVECTOR_GV_REST_HANDLERS_H

#include "api/server.h"
#include "storage/database.h"
#include "features/json.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file rest_handlers.h
 * @brief REST API endpoint handlers for GigaVector HTTP server.
 *
 * This module provides the implementation of REST endpoints
 * for vector database operations.
 */

/**
 * @brief Context passed to all handlers.
 */
typedef struct {
    GV_Database *db;                   /**< Database instance. */
    const GV_ServerConfig *config;     /**< Server configuration. */
    struct GV_KnowledgeGraph *kg;      /**< Optional knowledge graph (may be NULL). */
    struct GV_GraphDB *graph;          /**< Optional property graph (may be NULL). */
    struct GV_TxnRegistry *txn_registry; /**< Optional client-transaction registry (may be NULL). */
} GV_HandlerContext;

/**
 * @brief Create a JSON success response.
 *
 * @param data JSON data to include in response (takes ownership).
 * @return Allocated response structure.
 */
GV_HttpResponse *rest_response_json(GV_JsonValue *data);

/**
 * @brief Create a JSON error response.
 *
 * @param status HTTP status code.
 * @param error_code Error code string.
 * @param message Error message.
 * @return Allocated response structure.
 */
GV_HttpResponse *rest_response_error(GV_HttpStatus status,
                                         const char *error_code,
                                         const char *message);

/**
 * @brief Create a success response with message.
 *
 * @param message Success message.
 * @return Allocated response structure.
 */
GV_HttpResponse *rest_response_success(const char *message);

/**
 * @brief Free a response structure.
 *
 * @param response Response to free.
 */
void rest_response_free(GV_HttpResponse *response);

/**
 * @brief Parse URL path parameter (e.g., extract "123" from "/vectors/123").
 *
 * @param url Request URL.
 * @param prefix URL prefix (e.g., "/vectors/").
 * @param param_out Output buffer for parameter.
 * @param param_size Size of output buffer.
 * @return 0 on success, -1 if no parameter found.
 */
int rest_parse_path_param(const char *url, const char *prefix,
                              char *param_out, size_t param_size);

/**
 * @brief Parse query string parameter.
 *
 * @param query_string Query string (after '?').
 * @param key Parameter key to find.
 * @param value_out Output buffer for value.
 * @param value_size Size of output buffer.
 * @return 0 on success, -1 if not found.
 */
int rest_parse_query_param(const char *query_string, const char *key,
                               char *value_out, size_t value_size);

/**
 * @brief Parse JSON request body.
 *
 * @param request HTTP request.
 * @param error Output JSON error code.
 * @return Parsed JSON value, or NULL on error.
 */
GV_JsonValue *rest_parse_body(const GV_HttpRequest *request, GV_JsonError *error);

/**
 * @brief Handle GET /health endpoint.
 *
 * Returns server health status.
 *
 * Response:
 * {
 *   "status": "healthy"|"degraded"|"unhealthy",
 *   "vector_count": <count>,
 *   "uptime_seconds": <seconds>
 * }
 */
GV_HttpResponse *rest_handle_health(const GV_HandlerContext *ctx,
                                        const GV_HttpRequest *request);

/**
 * @brief Handle GET /stats endpoint.
 *
 * Returns database statistics.
 *
 * Response:
 * {
 *   "total_vectors": <count>,
 *   "total_inserts": <count>,
 *   "total_queries": <count>,
 *   "memory_bytes": <bytes>,
 *   "queries_per_second": <qps>
 * }
 */
GV_HttpResponse *rest_handle_stats(const GV_HandlerContext *ctx,
                                       const GV_HttpRequest *request);

/**
 * @brief Handle GET /metrics — Prometheus text exposition of DB counters.
 */
GV_HttpResponse *rest_handle_metrics(const GV_HandlerContext *ctx,
                                         const GV_HttpRequest *request);

/** @brief POST /txn/begin — start a transaction; returns {"txn_id": "..."}. */
GV_HttpResponse *rest_handle_txn_begin(const GV_HandlerContext *ctx,
                                           const GV_HttpRequest *request);
/** @brief POST /txn/commit — body {"txn_id":"..."}; 200/404/409/503. */
GV_HttpResponse *rest_handle_txn_commit(const GV_HandlerContext *ctx,
                                            const GV_HttpRequest *request);
/** @brief POST /txn/rollback — body {"txn_id":"..."}; 200/404/503. */
GV_HttpResponse *rest_handle_txn_rollback(const GV_HandlerContext *ctx,
                                              const GV_HttpRequest *request);

/**
 * @brief Handle POST /vectors endpoint.
 *
 * Add one or more vectors to the database.
 *
 * Request body:
 * {
 *   "vectors": [
 *     {
 *       "data": [0.1, 0.2, ...],
 *       "metadata": {"key": "value", ...}
 *     },
 *     ...
 *   ]
 * }
 *
 * Or single vector:
 * {
 *   "data": [0.1, 0.2, ...],
 *   "metadata": {"key": "value", ...}
 * }
 *
 * Response:
 * {
 *   "success": true,
 *   "inserted": <count>,
 *   "indices": [<index>, ...]
 * }
 */
GV_HttpResponse *rest_handle_vectors_post(const GV_HandlerContext *ctx,
                                              const GV_HttpRequest *request);

/**
 * @brief Handle GET /vectors/{id} endpoint.
 *
 * Get a vector by its index.
 *
 * Response:
 * {
 *   "index": <index>,
 *   "data": [0.1, 0.2, ...],
 *   "metadata": {"key": "value", ...}
 * }
 */
GV_HttpResponse *rest_handle_vectors_get(const GV_HandlerContext *ctx,
                                             const GV_HttpRequest *request,
                                             size_t vector_index);

/**
 * @brief Handle PUT /vectors/{id} endpoint.
 *
 * Update a vector by its index.
 *
 * Request body:
 * {
 *   "data": [0.1, 0.2, ...],
 *   "metadata": {"key": "value", ...}
 * }
 */
GV_HttpResponse *rest_handle_vectors_put(const GV_HandlerContext *ctx,
                                             const GV_HttpRequest *request,
                                             size_t vector_index);

/**
 * @brief Handle DELETE /vectors/{id} endpoint.
 *
 * Delete a vector by its index.
 *
 * Response:
 * {
 *   "success": true,
 *   "deleted_index": <index>
 * }
 */
GV_HttpResponse *rest_handle_vectors_delete(const GV_HandlerContext *ctx,
                                                const GV_HttpRequest *request,
                                                size_t vector_index);

/**
 * @brief Handle POST /search endpoint.
 *
 * Perform k-NN search.
 *
 * Request body:
 * {
 *   "query": [0.1, 0.2, ...],
 *   "k": 10,
 *   "distance": "euclidean"|"cosine"|"dot_product"|"manhattan",
 *   "filter": {
 *     "key": "value"
 *   }
 * }
 *
 * Response:
 * {
 *   "results": [
 *     {
 *       "index": <index>,
 *       "distance": <distance>,
 *       "data": [0.1, 0.2, ...],
 *       "metadata": {"key": "value", ...}
 *     },
 *     ...
 *   ],
 *   "count": <count>,
 *   "latency_ms": <ms>
 * }
 */
GV_HttpResponse *rest_handle_search(const GV_HandlerContext *ctx,
                                        const GV_HttpRequest *request);

/**
 * @brief Handle POST /search/range endpoint.
 *
 * Perform range search.
 *
 * Request body:
 * {
 *   "query": [0.1, 0.2, ...],
 *   "radius": 0.5,
 *   "max_results": 100,
 *   "distance": "euclidean"|"cosine"|"dot_product"|"manhattan",
 *   "filter": {
 *     "key": "value"
 *   }
 * }
 */
GV_HttpResponse *rest_handle_search_range(const GV_HandlerContext *ctx,
                                              const GV_HttpRequest *request);

/**
 * @brief Handle POST /search/batch endpoint.
 *
 * Perform batch search.
 *
 * Request body:
 * {
 *   "queries": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...],
 *   "k": 10,
 *   "distance": "euclidean"|"cosine"|"dot_product"|"manhattan"
 * }
 */
GV_HttpResponse *rest_handle_search_batch(const GV_HandlerContext *ctx,
                                              const GV_HttpRequest *request);

/**
 * @brief Handle POST /compact endpoint.
 *
 * Trigger database compaction.
 *
 * Response:
 * {
 *   "success": true,
 *   "message": "Compaction completed"
 * }
 */
GV_HttpResponse *rest_handle_compact(const GV_HandlerContext *ctx,
                                         const GV_HttpRequest *request);

/**
 * @brief Handle POST /save endpoint.
 *
 * Save database to disk.
 *
 * Request body (optional):
 * {
 *   "filepath": "/path/to/save"
 * }
 *
 * Response:
 * {
 *   "success": true,
 *   "filepath": "/path/saved"
 * }
 */
GV_HttpResponse *rest_handle_save(const GV_HandlerContext *ctx,
                                      const GV_HttpRequest *request);

/**
 * @brief Route a request to the appropriate handler.
 *
 * @param ctx Handler context.
 * @param request HTTP request.
 * @return HTTP response (caller must free).
 */
GV_HttpResponse *rest_route(const GV_HandlerContext *ctx,
                                const GV_HttpRequest *request);

/**
 * @brief Classify a request as a mutation (write/admin) vs a read.
 *
 * Mutations: any PUT/DELETE, and any POST that is not one of the search
 * endpoints (/search, /search/range, /search/batch). GET/OPTIONS are reads.
 * Used to enforce read-only server mode.
 *
 * @param url Request path (query string already stripped).
 * @param method HTTP method.
 * @return 1 if the request mutates state, 0 otherwise.
 */
int rest_request_is_mutation(const char *url, GV_HttpMethod method);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_REST_HANDLERS_H */


/* ── Graph & knowledge-graph endpoints (require server_set_graphs) ────── */

/** GET /graph/stats — node/edge counts and mutation version. */
GV_HttpResponse *rest_handle_graph_stats(const GV_HandlerContext *ctx,
                                         const GV_HttpRequest *request);

/** POST /kg/expand  body: {"seeds": [id, ...], "radius": N} → triples. */
GV_HttpResponse *rest_handle_kg_expand(const GV_HandlerContext *ctx,
                                       const GV_HttpRequest *request);

/** POST /graph/wal/attach and POST /kg/wal/attach body: {"path": "..."}.
 *  Paths are confined to the server data_dir like /save. */
GV_HttpResponse *rest_handle_graph_wal_attach(const GV_HandlerContext *ctx,
                                              const GV_HttpRequest *request);
GV_HttpResponse *rest_handle_kg_wal_attach(const GV_HandlerContext *ctx,
                                           const GV_HttpRequest *request);

/** POST /graph/wal/checkpoint and POST /kg/wal/checkpoint. */
GV_HttpResponse *rest_handle_graph_wal_checkpoint(const GV_HandlerContext *ctx,
                                                  const GV_HttpRequest *request);
GV_HttpResponse *rest_handle_kg_wal_checkpoint(const GV_HandlerContext *ctx,
                                               const GV_HttpRequest *request);
