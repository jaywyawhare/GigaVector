#ifndef GIGAVECTOR_GV_SERVER_H
#define GIGAVECTOR_GV_SERVER_H

#include <stddef.h>
#include <stdint.h>

#include "gv_database.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_server.h
 * @brief HTTP REST API server for GigaVector.
 *
 * This module provides an embedded HTTP server using libmicrohttpd
 * for exposing GigaVector database operations via a REST API.
 */

/**
 * @brief Server error codes.
 */
typedef enum {
    GV_SERVER_OK = 0,
    GV_SERVER_ERROR_NULL_POINTER = -1,
    GV_SERVER_ERROR_INVALID_CONFIG = -2,
    GV_SERVER_ERROR_ALREADY_RUNNING = -3,
    GV_SERVER_ERROR_NOT_RUNNING = -4,
    GV_SERVER_ERROR_START_FAILED = -5,
    GV_SERVER_ERROR_MEMORY = -6,
    GV_SERVER_ERROR_BIND_FAILED = -7
} GV_ServerError;

/**
 * @brief HTTP method types.
 */
typedef enum {
    GV_HTTP_GET = 0,
    GV_HTTP_POST = 1,
    GV_HTTP_PUT = 2,
    GV_HTTP_DELETE = 3,
    GV_HTTP_OPTIONS = 4,
    GV_HTTP_HEAD = 5
} GV_HttpMethod;

/**
 * @brief HTTP status codes.
 */
typedef enum {
    GV_HTTP_200_OK = 200,
    GV_HTTP_201_CREATED = 201,
    GV_HTTP_204_NO_CONTENT = 204,
    GV_HTTP_400_BAD_REQUEST = 400,
    GV_HTTP_401_UNAUTHORIZED = 401,
    GV_HTTP_403_FORBIDDEN = 403,
    GV_HTTP_404_NOT_FOUND = 404,
    GV_HTTP_405_METHOD_NOT_ALLOWED = 405,
    GV_HTTP_413_PAYLOAD_TOO_LARGE = 413,
    GV_HTTP_429_TOO_MANY_REQUESTS = 429,
    GV_HTTP_500_INTERNAL_ERROR = 500,
    GV_HTTP_503_SERVICE_UNAVAILABLE = 503
} GV_HttpStatus;

/**
 * @brief Server configuration.
 */
typedef struct {
    uint16_t port;                     /**< Port to listen on (default: 8080). */
    const char *bind_address;          /**< Address to bind to (default: "0.0.0.0"). */
    size_t thread_pool_size;           /**< Number of worker threads (default: 4). */
    size_t max_connections;            /**< Maximum concurrent connections (default: 100). */
    size_t request_timeout_ms;         /**< Request timeout in milliseconds (default: 30000). */
    size_t max_request_body_bytes;     /**< Maximum request body size (default: 10MB). */
    int enable_cors;                   /**< Enable CORS headers (default: 0). */
    const char *cors_origins;          /**< Allowed CORS origins (default: "*"). */
    int enable_logging;                /**< Enable request logging (default: 1). */
    const char *api_key;               /**< Optional API key for authentication (default: NULL). */
    double max_requests_per_second;    /**< Rate limit: max requests/sec per client IP (0 = unlimited, default: 0). */
    size_t rate_limit_burst;           /**< Rate limit burst size (default: 10). */
} GV_ServerConfig;

/**
 * @brief HTTP request context passed to handlers.
 */
typedef struct {
    GV_HttpMethod method;              /**< HTTP method. */
    const char *url;                   /**< Request URL path. */
    const char *query_string;          /**< Query string (after '?'). */
    const char *body;                  /**< Request body (NULL if none). */
    size_t body_length;                /**< Length of request body. */
    const char *content_type;          /**< Content-Type header. */
    const char *authorization;         /**< Authorization header. */
} GV_HttpRequest;

/**
 * @brief HTTP response structure.
 */
typedef struct {
    GV_HttpStatus status;              /**< HTTP status code. */
    char *body;                        /**< Response body (will be freed by server). */
    size_t body_length;                /**< Length of response body. */
    const char *content_type;          /**< Content-Type header (default: "application/json"). */
} GV_HttpResponse;

/**
 * @brief Server statistics.
 */
typedef struct {
    uint64_t total_requests;           /**< Total requests handled. */
    uint64_t active_connections;       /**< Current active connections. */
    uint64_t requests_per_second;      /**< Current requests per second. */
    uint64_t total_bytes_sent;         /**< Total bytes sent. */
    uint64_t total_bytes_received;     /**< Total bytes received. */
    uint64_t error_count;              /**< Total error responses (4xx, 5xx). */
} GV_ServerStats;

/**
 * @brief Opaque server handle.
 */
typedef struct GV_Server GV_Server;

/* ============================================================================
 * Server Lifecycle
 * ============================================================================ */

/**
 * @brief Create a new HTTP server instance.
 *
 * The server is created but not started. Call gv_server_start() to begin
 * accepting connections.
 *
 * @param db Database instance to serve (must remain valid while server runs).
 * @param config Server configuration; NULL to use defaults.
 * @return Server instance, or NULL on error.
 */
GV_Server *gv_server_create(GV_Database *db, const GV_ServerConfig *config);

/**
 * @brief Start the HTTP server.
 *
 * Begins accepting connections on the configured port. This function
 * returns immediately; the server runs in background threads.
 *
 * @param server Server instance.
 * @return GV_SERVER_OK on success, error code on failure.
 */
int gv_server_start(GV_Server *server);

/**
 * @brief Stop the HTTP server.
 *
 * Gracefully shuts down the server, waiting for active requests to complete.
 *
 * @param server Server instance.
 * @return GV_SERVER_OK on success, error code on failure.
 */
int gv_server_stop(GV_Server *server);

/**
 * @brief Destroy the server and free all resources.
 *
 * If the server is running, it will be stopped first.
 *
 * @param server Server instance (safe to call with NULL).
 */
void gv_server_destroy(GV_Server *server);

/**
 * @brief Check if the server is currently running.
 *
 * @param server Server instance.
 * @return 1 if running, 0 if stopped, -1 on error.
 */
int gv_server_is_running(const GV_Server *server);

/* ============================================================================
 * Server Information
 * ============================================================================ */

/**
 * @brief Get server statistics.
 *
 * @param server Server instance.
 * @param stats Output statistics structure.
 * @return GV_SERVER_OK on success, error code on failure.
 */
int gv_server_get_stats(const GV_Server *server, GV_ServerStats *stats);

/**
 * @brief Get the port the server is listening on.
 *
 * @param server Server instance.
 * @return Port number, or 0 if not running.
 */
uint16_t gv_server_get_port(const GV_Server *server);

/**
 * @brief Get error description string.
 *
 * @param error Error code.
 * @return Human-readable error description.
 */
const char *gv_server_error_string(int error);

/* ============================================================================
 * Configuration Helpers
 * ============================================================================ */

/**
 * @brief Initialize a server configuration with default values.
 *
 * Default values:
 * - port: 8080
 * - bind_address: "0.0.0.0"
 * - thread_pool_size: 4
 * - max_connections: 100
 * - request_timeout_ms: 30000
 * - max_request_body_bytes: 10485760 (10MB)
 * - enable_cors: 0
 * - cors_origins: "*"
 * - enable_logging: 1
 * - api_key: NULL
 *
 * @param config Configuration structure to initialize.
 */
void gv_server_config_init(GV_ServerConfig *config);

/* ============================================================================
 * REST API Endpoints (handled internally)
 *
 * GET    /health              - Health check
 * GET    /stats               - Database statistics
 * POST   /vectors             - Add vector(s)
 * GET    /vectors/{id}        - Get vector by index
 * PUT    /vectors/{id}        - Update vector
 * DELETE /vectors/{id}        - Delete vector
 * POST   /search              - k-NN search
 * POST   /search/range        - Range search
 * POST   /search/batch        - Batch search
 * POST   /compact             - Trigger compaction
 * POST   /save                - Save database to disk
 * ============================================================================ */

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_SERVER_H */
