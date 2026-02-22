/**
 * @file gv_server.c
 * @brief HTTP REST API server implementation for GigaVector.
 *
 * Uses libmicrohttpd for HTTP handling.
 */

#include "gigavector/gv_server.h"
#include "gigavector/gv_rest_handlers.h"
#include "gigavector/gv_json.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <pthread.h>

#ifdef HAVE_MICROHTTPD
#include <microhttpd.h>
#endif

/*  Rate Limiter  */

/**
 * @brief Token-bucket rate limiter state.
 */
typedef struct {
    double tokens;
    double max_tokens;
    double refill_rate;      /* tokens per second */
    uint64_t last_refill_us; /* microseconds (CLOCK_MONOTONIC) */
    pthread_mutex_t mutex;
} GV_RateLimiter;

static uint64_t gv_now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

static void gv_rate_limiter_init(GV_RateLimiter *rl, double rate, size_t burst) {
    rl->max_tokens = (double)burst;
    rl->tokens = rl->max_tokens;
    rl->refill_rate = rate;
    rl->last_refill_us = gv_now_us();
    pthread_mutex_init(&rl->mutex, NULL);
}

static void gv_rate_limiter_destroy(GV_RateLimiter *rl) {
    pthread_mutex_destroy(&rl->mutex);
}

/**
 * @brief Try to consume one token. Returns 1 if allowed, 0 if rate-limited.
 */
static int gv_rate_limiter_allow(GV_RateLimiter *rl) {
    pthread_mutex_lock(&rl->mutex);

    uint64_t now = gv_now_us();
    double elapsed_sec = (double)(now - rl->last_refill_us) / 1000000.0;
    rl->last_refill_us = now;

    /* Refill tokens based on elapsed time */
    rl->tokens += elapsed_sec * rl->refill_rate;
    if (rl->tokens > rl->max_tokens) {
        rl->tokens = rl->max_tokens;
    }

    /* Try to consume one token */
    if (rl->tokens >= 1.0) {
        rl->tokens -= 1.0;
        pthread_mutex_unlock(&rl->mutex);
        return 1;
    }

    pthread_mutex_unlock(&rl->mutex);
    return 0;
}

/*  Internal Structures  */

/**
 * @brief Connection info for POST data accumulation.
 */
typedef struct {
    char *data;
    size_t size;
    size_t capacity;
} GV_ConnectionInfo;

/**
 * @brief Server internal state.
 */
struct GV_Server {
    GV_Database *db;
    GV_ServerConfig config;

#ifdef HAVE_MICROHTTPD
    struct MHD_Daemon *daemon;
#else
    void *daemon;
#endif

    int running;
    time_t start_time;

    /* Statistics */
    uint64_t total_requests;
    uint64_t active_connections;
    uint64_t total_bytes_sent;
    uint64_t total_bytes_received;
    uint64_t error_count;
    pthread_mutex_t stats_mutex;

    /* Handler context */
    GV_HandlerContext handler_ctx;

    /* Rate limiter */
    GV_RateLimiter rate_limiter;
    int rate_limit_enabled;
};

/*  Default Configuration  */

static const GV_ServerConfig DEFAULT_CONFIG = {
    .port = 8080,
    .bind_address = "0.0.0.0",
    .thread_pool_size = 4,
    .max_connections = 100,
    .request_timeout_ms = 30000,
    .max_request_body_bytes = 10485760,  /* 10MB */
    .enable_cors = 0,
    .cors_origins = "*",
    .enable_logging = 1,
    .api_key = NULL,
    .max_requests_per_second = 0,  /* unlimited */
    .rate_limit_burst = 10
};

void gv_server_config_init(GV_ServerConfig *config) {
    if (!config) return;
    *config = DEFAULT_CONFIG;
}

/*  Error Strings  */

const char *gv_server_error_string(int error) {
    switch (error) {
        case GV_SERVER_OK:
            return "Success";
        case GV_SERVER_ERROR_NULL_POINTER:
            return "Null pointer argument";
        case GV_SERVER_ERROR_INVALID_CONFIG:
            return "Invalid configuration";
        case GV_SERVER_ERROR_ALREADY_RUNNING:
            return "Server is already running";
        case GV_SERVER_ERROR_NOT_RUNNING:
            return "Server is not running";
        case GV_SERVER_ERROR_START_FAILED:
            return "Failed to start server";
        case GV_SERVER_ERROR_MEMORY:
            return "Memory allocation failed";
        case GV_SERVER_ERROR_BIND_FAILED:
            return "Failed to bind to address/port";
        default:
            return "Unknown error";
    }
}

#ifdef HAVE_MICROHTTPD

/*  HTTP Method Parsing  */

static GV_HttpMethod parse_method(const char *method) {
    if (!method) return GV_HTTP_GET;
    if (strcmp(method, "GET") == 0) return GV_HTTP_GET;
    if (strcmp(method, "POST") == 0) return GV_HTTP_POST;
    if (strcmp(method, "PUT") == 0) return GV_HTTP_PUT;
    if (strcmp(method, "DELETE") == 0) return GV_HTTP_DELETE;
    if (strcmp(method, "OPTIONS") == 0) return GV_HTTP_OPTIONS;
    if (strcmp(method, "HEAD") == 0) return GV_HTTP_HEAD;
    return GV_HTTP_GET;
}

/*  libmicrohttpd Callbacks  */

/**
 * @brief Callback to free connection info.
 */
static void request_completed_callback(void *cls,
                                        struct MHD_Connection *connection,
                                        void **con_cls,
                                        enum MHD_RequestTerminationCode toe) {
    (void)cls;
    (void)connection;
    (void)toe;

    GV_ConnectionInfo *con_info = *con_cls;
    if (con_info) {
        free(con_info->data);
        free(con_info);
    }
    *con_cls = NULL;
}

/**
 * @brief Add CORS headers to response.
 */
static void add_cors_headers(struct MHD_Response *response, const GV_Server *server) {
    if (server->config.enable_cors) {
        MHD_add_response_header(response, "Access-Control-Allow-Origin",
                                server->config.cors_origins ? server->config.cors_origins : "*");
        MHD_add_response_header(response, "Access-Control-Allow-Methods",
                                "GET, POST, PUT, DELETE, OPTIONS");
        MHD_add_response_header(response, "Access-Control-Allow-Headers",
                                "Content-Type, Authorization, X-API-Key");
        MHD_add_response_header(response, "Access-Control-Max-Age", "86400");
    }
}

/**
 * @brief Check API key authentication.
 */
static int check_auth(const GV_Server *server, struct MHD_Connection *connection) {
    if (!server->config.api_key) {
        return 1;  /* No auth required */
    }

    const char *auth = MHD_lookup_connection_value(connection, MHD_HEADER_KIND, "X-API-Key");
    if (!auth) {
        auth = MHD_lookup_connection_value(connection, MHD_HEADER_KIND, "Authorization");
        if (auth && strncmp(auth, "Bearer ", 7) == 0) {
            auth += 7;
        }
    }

    if (!auth || strcmp(auth, server->config.api_key) != 0) {
        return 0;
    }
    return 1;
}

/**
 * @brief Main request handler callback.
 */
static enum MHD_Result answer_to_connection(void *cls,
                                             struct MHD_Connection *connection,
                                             const char *url,
                                             const char *method,
                                             const char *version,
                                             const char *upload_data,
                                             size_t *upload_data_size,
                                             void **con_cls) {
    (void)version;

    GV_Server *server = (GV_Server *)cls;
    GV_ConnectionInfo *con_info;

    /* First call: set up connection info */
    if (*con_cls == NULL) {
        con_info = calloc(1, sizeof(GV_ConnectionInfo));
        if (!con_info) {
            return MHD_NO;
        }
        *con_cls = con_info;
        return MHD_YES;
    }

    con_info = *con_cls;

    /* Accumulate POST data */
    if (*upload_data_size != 0) {
        size_t new_size = con_info->size + *upload_data_size;

        /* Check size limit */
        if (new_size > server->config.max_request_body_bytes) {
            *upload_data_size = 0;
            return MHD_YES;
        }

        /* Grow buffer if needed */
        if (new_size > con_info->capacity) {
            size_t new_cap = con_info->capacity == 0 ? 4096 : con_info->capacity * 2;
            while (new_cap < new_size) new_cap *= 2;
            char *new_data = realloc(con_info->data, new_cap + 1);
            if (!new_data) {
                return MHD_NO;
            }
            con_info->data = new_data;
            con_info->capacity = new_cap;
        }

        memcpy(con_info->data + con_info->size, upload_data, *upload_data_size);
        con_info->size = new_size;
        con_info->data[con_info->size] = '\0';

        pthread_mutex_lock(&server->stats_mutex);
        server->total_bytes_received += *upload_data_size;
        pthread_mutex_unlock(&server->stats_mutex);

        *upload_data_size = 0;
        return MHD_YES;
    }

    /* Handle OPTIONS for CORS preflight */
    if (strcmp(method, "OPTIONS") == 0) {
        struct MHD_Response *response = MHD_create_response_from_buffer(0, "", MHD_RESPMEM_PERSISTENT);
        add_cors_headers(response, server);
        enum MHD_Result ret = MHD_queue_response(connection, MHD_HTTP_NO_CONTENT, response);
        MHD_destroy_response(response);
        return ret;
    }

    /* Check rate limit */
    if (server->rate_limit_enabled && !gv_rate_limiter_allow(&server->rate_limiter)) {
        const char *rate_json = "{\"error\":\"rate limit exceeded\"}";
        struct MHD_Response *response = MHD_create_response_from_buffer(
            strlen(rate_json), (void *)rate_json, MHD_RESPMEM_PERSISTENT);
        MHD_add_response_header(response, "Content-Type", "application/json");
        add_cors_headers(response, server);
        enum MHD_Result ret = MHD_queue_response(connection, GV_HTTP_429_TOO_MANY_REQUESTS, response);
        MHD_destroy_response(response);

        pthread_mutex_lock(&server->stats_mutex);
        server->total_requests++;
        server->error_count++;
        pthread_mutex_unlock(&server->stats_mutex);

        return ret;
    }

    /* Check authentication */
    if (!check_auth(server, connection)) {
        const char *error_json = "{\"error\":\"Unauthorized\",\"message\":\"Invalid or missing API key\"}";
        struct MHD_Response *response = MHD_create_response_from_buffer(
            strlen(error_json), (void *)error_json, MHD_RESPMEM_PERSISTENT);
        MHD_add_response_header(response, "Content-Type", "application/json");
        add_cors_headers(response, server);
        enum MHD_Result ret = MHD_queue_response(connection, MHD_HTTP_UNAUTHORIZED, response);
        MHD_destroy_response(response);

        pthread_mutex_lock(&server->stats_mutex);
        server->error_count++;
        pthread_mutex_unlock(&server->stats_mutex);

        return ret;
    }

    /* Build request structure */
    GV_HttpRequest request = {
        .method = parse_method(method),
        .url = url,
        .query_string = strchr(url, '?'),
        .body = con_info->data,
        .body_length = con_info->size,
        .content_type = MHD_lookup_connection_value(connection, MHD_HEADER_KIND, "Content-Type"),
        .authorization = MHD_lookup_connection_value(connection, MHD_HEADER_KIND, "Authorization")
    };

    /* Skip '?' in query string */
    if (request.query_string) {
        request.query_string++;
    }

    /* Log request */
    if (server->config.enable_logging) {
        fprintf(stderr, "[GV_Server] %s %s\n", method, url);
    }

    /* Route and handle request */
    GV_HttpResponse *http_response = gv_rest_route(&server->handler_ctx, &request);

    /* Update stats */
    pthread_mutex_lock(&server->stats_mutex);
    server->total_requests++;
    if (http_response && http_response->status >= 400) {
        server->error_count++;
    }
    pthread_mutex_unlock(&server->stats_mutex);

    /* Build HTTP response */
    struct MHD_Response *mhd_response;
    unsigned int status_code = GV_HTTP_500_INTERNAL_ERROR;

    if (http_response && http_response->body) {
        mhd_response = MHD_create_response_from_buffer(
            http_response->body_length,
            http_response->body,
            MHD_RESPMEM_MUST_COPY);
        status_code = http_response->status;

        pthread_mutex_lock(&server->stats_mutex);
        server->total_bytes_sent += http_response->body_length;
        pthread_mutex_unlock(&server->stats_mutex);
    } else {
        const char *error = "{\"error\":\"Internal server error\"}";
        mhd_response = MHD_create_response_from_buffer(
            strlen(error), (void *)error, MHD_RESPMEM_PERSISTENT);
    }

    /* Set headers */
    MHD_add_response_header(mhd_response, "Content-Type",
                            http_response && http_response->content_type ?
                            http_response->content_type : "application/json");
    add_cors_headers(mhd_response, server);

    /* Queue response */
    enum MHD_Result ret = MHD_queue_response(connection, status_code, mhd_response);
    MHD_destroy_response(mhd_response);

    /* Free response */
    gv_rest_response_free(http_response);

    return ret;
}

#endif /* HAVE_MICROHTTPD */

/*  Server Lifecycle  */

GV_Server *gv_server_create(GV_Database *db, const GV_ServerConfig *config) {
    if (!db) {
        return NULL;
    }

    GV_Server *server = calloc(1, sizeof(GV_Server));
    if (!server) {
        return NULL;
    }

    server->db = db;
    server->config = config ? *config : DEFAULT_CONFIG;
    server->running = 0;
    server->daemon = NULL;

    /* Initialize stats mutex */
    if (pthread_mutex_init(&server->stats_mutex, NULL) != 0) {
        free(server);
        return NULL;
    }

    /* Set up handler context */
    server->handler_ctx.db = db;
    server->handler_ctx.config = &server->config;

    return server;
}

int gv_server_start(GV_Server *server) {
    if (!server) {
        return GV_SERVER_ERROR_NULL_POINTER;
    }

    if (server->running) {
        return GV_SERVER_ERROR_ALREADY_RUNNING;
    }

    /* Initialize rate limiter if configured */
    if (server->config.max_requests_per_second > 0) {
        size_t burst = server->config.rate_limit_burst > 0 ? server->config.rate_limit_burst : 10;
        gv_rate_limiter_init(&server->rate_limiter,
                             server->config.max_requests_per_second, burst);
        server->rate_limit_enabled = 1;
    } else {
        server->rate_limit_enabled = 0;
    }

#ifdef HAVE_MICROHTTPD
    /* Use internal select with thread pool for handling connections.
     * Note: MHD_USE_THREAD_PER_CONNECTION and MHD_OPTION_THREAD_POOL_SIZE
     * are mutually exclusive - we use thread pool for better resource control. */
    unsigned int flags = MHD_USE_INTERNAL_POLLING_THREAD;

    server->daemon = MHD_start_daemon(
        flags,
        server->config.port,
        NULL, NULL,  /* Accept policy */
        &answer_to_connection, server,
        MHD_OPTION_NOTIFY_COMPLETED, request_completed_callback, NULL,
        MHD_OPTION_CONNECTION_TIMEOUT, (unsigned int)(server->config.request_timeout_ms / 1000),
        MHD_OPTION_CONNECTION_LIMIT, (unsigned int)server->config.max_connections,
        MHD_OPTION_THREAD_POOL_SIZE, (unsigned int)server->config.thread_pool_size,
        MHD_OPTION_END);

    if (!server->daemon) {
        return GV_SERVER_ERROR_START_FAILED;
    }

    server->running = 1;
    server->start_time = time(NULL);

    if (server->config.enable_logging) {
        fprintf(stderr, "[GV_Server] Started on port %u\n", server->config.port);
    }

    return GV_SERVER_OK;
#else
    (void)server;
    fprintf(stderr, "[GV_Server] Error: libmicrohttpd not available\n");
    return GV_SERVER_ERROR_START_FAILED;
#endif
}

int gv_server_stop(GV_Server *server) {
    if (!server) {
        return GV_SERVER_ERROR_NULL_POINTER;
    }

    if (!server->running) {
        return GV_SERVER_ERROR_NOT_RUNNING;
    }

#ifdef HAVE_MICROHTTPD
    if (server->daemon) {
        MHD_stop_daemon(server->daemon);
        server->daemon = NULL;
    }
#endif

    server->running = 0;

    /* Destroy rate limiter if it was enabled */
    if (server->rate_limit_enabled) {
        gv_rate_limiter_destroy(&server->rate_limiter);
        server->rate_limit_enabled = 0;
    }

    if (server->config.enable_logging) {
        fprintf(stderr, "[GV_Server] Stopped\n");
    }

    return GV_SERVER_OK;
}

void gv_server_destroy(GV_Server *server) {
    if (!server) {
        return;
    }

    if (server->running) {
        gv_server_stop(server);
    }

    pthread_mutex_destroy(&server->stats_mutex);
    free(server);
}

int gv_server_is_running(const GV_Server *server) {
    if (!server) {
        return -1;
    }
    return server->running;
}

/*  Server Information  */

int gv_server_get_stats(const GV_Server *server, GV_ServerStats *stats) {
    if (!server || !stats) {
        return GV_SERVER_ERROR_NULL_POINTER;
    }

    pthread_mutex_lock((pthread_mutex_t *)&server->stats_mutex);

    stats->total_requests = server->total_requests;
    stats->active_connections = server->active_connections;
    stats->total_bytes_sent = server->total_bytes_sent;
    stats->total_bytes_received = server->total_bytes_received;
    stats->error_count = server->error_count;

    /* Calculate requests per second */
    if (server->running && server->start_time > 0) {
        time_t uptime = time(NULL) - server->start_time;
        if (uptime > 0) {
            stats->requests_per_second = server->total_requests / uptime;
        } else {
            stats->requests_per_second = 0;
        }
    } else {
        stats->requests_per_second = 0;
    }

    pthread_mutex_unlock((pthread_mutex_t *)&server->stats_mutex);

    return GV_SERVER_OK;
}

uint16_t gv_server_get_port(const GV_Server *server) {
    if (!server || !server->running) {
        return 0;
    }
    return server->config.port;
}
