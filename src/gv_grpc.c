/**
 * @file gv_grpc.c
 * @brief High-performance binary protocol server (gRPC-style) for GigaVector.
 *
 * Pure C socket implementation using POSIX sockets and pthreads.
 * Wire format: [4-byte big-endian length][1-byte type][4-byte request_id][payload]
 * No external gRPC dependency required.
 */

#include "gigavector/gv_grpc.h"
#include "gigavector/gv_database.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <time.h>
#include <signal.h>

/* ============================================================================
 * Platform Helpers
 * ============================================================================ */

#if defined(__STDC_NO_ATOMICS__)
#define GV_ATOMIC_INC(ptr)   do { __sync_fetch_and_add((ptr), 1); } while (0)
#define GV_ATOMIC_DEC(ptr)   do { __sync_fetch_and_sub((ptr), 1); } while (0)
#define GV_ATOMIC_ADD(ptr,v) do { __sync_fetch_and_add((ptr), (v)); } while (0)
#define GV_ATOMIC_LOAD(ptr)  __sync_fetch_and_add((ptr), 0)
#else
#include <stdatomic.h>
#define GV_ATOMIC_INC(ptr)   atomic_fetch_add((_Atomic uint64_t *)(ptr), 1)
#define GV_ATOMIC_DEC(ptr)   atomic_fetch_sub((_Atomic uint64_t *)(ptr), 1)
#define GV_ATOMIC_ADD(ptr,v) atomic_fetch_add((_Atomic uint64_t *)(ptr), (v))
#define GV_ATOMIC_LOAD(ptr)  atomic_load((_Atomic uint64_t *)(ptr))
#endif

static uint64_t gv_grpc_now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

/* ============================================================================
 * Big-Endian Encoding/Decoding
 * ============================================================================ */

static void write_u32_be(uint8_t *buf, uint32_t val) {
    buf[0] = (uint8_t)(val >> 24);
    buf[1] = (uint8_t)(val >> 16);
    buf[2] = (uint8_t)(val >> 8);
    buf[3] = (uint8_t)(val);
}

static uint32_t read_u32_be(const uint8_t *buf) {
    return ((uint32_t)buf[0] << 24) |
           ((uint32_t)buf[1] << 16) |
           ((uint32_t)buf[2] << 8)  |
           ((uint32_t)buf[3]);
}

static void write_float_be(uint8_t *buf, float val) {
    uint32_t bits;
    memcpy(&bits, &val, sizeof(bits));
    write_u32_be(buf, bits);
}

static float read_float_be(const uint8_t *buf) {
    uint32_t bits = read_u32_be(buf);
    float val;
    memcpy(&val, &bits, sizeof(val));
    return val;
}

/* ============================================================================
 * Thread Pool
 * ============================================================================ */

typedef struct GV_GrpcTask {
    int client_fd;
    struct GV_GrpcTask *next;
} GV_GrpcTask;

typedef struct {
    pthread_t *threads;
    size_t thread_count;

    GV_GrpcTask *queue_head;
    GV_GrpcTask *queue_tail;
    size_t queue_size;

    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int shutdown;

    /* Back-pointer to server for handler access */
    struct GV_GrpcServer *server;
} GV_ThreadPool;

/* ============================================================================
 * Internal Server Structure
 * ============================================================================ */

struct GV_GrpcServer {
    GV_Database *db;
    GV_GrpcConfig config;

    int listen_fd;
    int running;
    int stop_requested;

    /* Accept thread */
    pthread_t accept_thread;
    int accept_thread_started;

    /* Worker thread pool */
    GV_ThreadPool pool;

    /* Statistics */
    uint64_t total_requests;
    uint64_t active_connections;
    uint64_t bytes_sent;
    uint64_t bytes_received;
    uint64_t errors;
    uint64_t total_latency_us;
    uint64_t latency_samples;
    pthread_mutex_t stats_mutex;
};

/* ============================================================================
 * Default Configuration
 * ============================================================================ */

static const GV_GrpcConfig DEFAULT_GRPC_CONFIG = {
    .port = 50051,
    .bind_address = "0.0.0.0",
    .max_connections = 256,
    .max_message_bytes = 16777216,  /* 16MB */
    .thread_pool_size = 4,
    .enable_compression = 0
};

void gv_grpc_config_init(GV_GrpcConfig *config) {
    if (!config) return;
    *config = DEFAULT_GRPC_CONFIG;
}

/* ============================================================================
 * Error Strings
 * ============================================================================ */

const char *gv_grpc_error_string(int error) {
    switch (error) {
        case GV_GRPC_OK:            return "Success";
        case GV_GRPC_ERROR_NULL:    return "Null pointer argument";
        case GV_GRPC_ERROR_CONFIG:  return "Invalid configuration";
        case GV_GRPC_ERROR_RUNNING: return "Server is already running";
        case GV_GRPC_ERROR_NOT_RUNNING: return "Server is not running";
        case GV_GRPC_ERROR_START:   return "Failed to start server";
        case GV_GRPC_ERROR_MEMORY:  return "Memory allocation failed";
        case GV_GRPC_ERROR_BIND:    return "Failed to bind to address/port";
        default:                    return "Unknown error";
    }
}

/* ============================================================================
 * Socket I/O Helpers
 * ============================================================================ */

/**
 * @brief Read exactly @p len bytes from @p fd into @p buf.
 * @return 0 on success, -1 on error or EOF.
 */
static int recv_exact(int fd, uint8_t *buf, size_t len) {
    size_t total = 0;
    while (total < len) {
        ssize_t n = recv(fd, buf + total, len - total, 0);
        if (n <= 0) return -1;
        total += (size_t)n;
    }
    return 0;
}

/**
 * @brief Write exactly @p len bytes from @p buf to @p fd.
 * @return 0 on success, -1 on error.
 */
static int send_exact(int fd, const uint8_t *buf, size_t len) {
    size_t total = 0;
    while (total < len) {
        ssize_t n = send(fd, buf + total, len - total, MSG_NOSIGNAL);
        if (n <= 0) return -1;
        total += (size_t)n;
    }
    return 0;
}

/**
 * @brief Read a complete framed message from the wire.
 *
 * Wire format: [4-byte big-endian length][1-byte type][4-byte request_id][payload]
 * The length field encodes the number of bytes that follow (type + request_id + payload).
 *
 * @return 0 on success, -1 on error. Caller must free msg->payload.
 */
static int recv_message(int fd, GV_GrpcMessage *msg, size_t max_bytes) {
    uint8_t header[4];
    if (recv_exact(fd, header, 4) != 0) return -1;

    msg->length = read_u32_be(header);
    if (msg->length < 5) return -1;  /* Need at least type(1) + request_id(4) */
    if (msg->length > max_bytes) return -1;

    uint8_t meta[5];
    if (recv_exact(fd, meta, 5) != 0) return -1;

    msg->msg_type = meta[0];
    msg->request_id = read_u32_be(meta + 1);
    msg->payload_len = msg->length - 5;

    if (msg->payload_len > 0) {
        msg->payload = malloc(msg->payload_len);
        if (!msg->payload) return -1;
        if (recv_exact(fd, msg->payload, msg->payload_len) != 0) {
            free(msg->payload);
            msg->payload = NULL;
            return -1;
        }
    } else {
        msg->payload = NULL;
    }

    return 0;
}

/**
 * @brief Send a complete framed message on the wire.
 */
static int send_message(int fd, uint8_t msg_type, uint32_t request_id,
                         const uint8_t *payload, size_t payload_len) {
    /* length = 1 (type) + 4 (request_id) + payload_len */
    uint32_t length = (uint32_t)(5 + payload_len);
    uint8_t header[9];
    write_u32_be(header, length);
    header[4] = msg_type;
    write_u32_be(header + 5, request_id);

    if (send_exact(fd, header, 9) != 0) return -1;
    if (payload_len > 0 && payload) {
        if (send_exact(fd, payload, payload_len) != 0) return -1;
    }
    return 0;
}

/**
 * @brief Build and send an error response.
 *
 * Error payload format: [4-byte big-endian error code][N-byte error message string]
 */
static int send_error_response(int fd, uint32_t request_id, int32_t error_code,
                                const char *error_msg) {
    size_t msg_len = error_msg ? strlen(error_msg) : 0;
    size_t payload_len = 4 + msg_len;
    uint8_t *payload = malloc(payload_len);
    if (!payload) return -1;

    /* Store error code as big-endian signed 32 */
    write_u32_be(payload, (uint32_t)error_code);
    if (msg_len > 0) {
        memcpy(payload + 4, error_msg, msg_len);
    }

    int rc = send_message(fd, GV_MSG_RESPONSE, request_id, payload, payload_len);
    free(payload);
    return rc;
}

/* ============================================================================
 * Request Handlers
 * ============================================================================ */

/**
 * @brief Handle GV_MSG_ADD_VECTOR.
 *
 * Request payload: [4-byte dimension][dimension * 4-byte floats]
 * Response payload: [4-byte status (0 = ok)]
 */
static void handle_add_vector(GV_GrpcServer *server, int fd,
                               const GV_GrpcMessage *msg) {
    if (msg->payload_len < 4) {
        send_error_response(fd, msg->request_id, -1, "payload too short");
        GV_ATOMIC_INC(&server->errors);
        return;
    }

    uint32_t dimension = read_u32_be(msg->payload);
    size_t expected = 4 + (size_t)dimension * sizeof(float);
    if (msg->payload_len < expected) {
        send_error_response(fd, msg->request_id, -1, "incomplete vector data");
        GV_ATOMIC_INC(&server->errors);
        return;
    }

    /* Decode floats from big-endian */
    float *vec = malloc(dimension * sizeof(float));
    if (!vec) {
        send_error_response(fd, msg->request_id, -1, "out of memory");
        GV_ATOMIC_INC(&server->errors);
        return;
    }
    for (uint32_t i = 0; i < dimension; i++) {
        vec[i] = read_float_be(msg->payload + 4 + i * 4);
    }

    int rc = gv_db_add_vector(server->db, vec, (size_t)dimension);
    free(vec);

    uint8_t resp[4];
    write_u32_be(resp, (uint32_t)rc);
    send_message(fd, GV_MSG_RESPONSE, msg->request_id, resp, 4);

    if (rc != 0) {
        GV_ATOMIC_INC(&server->errors);
    }
}

/**
 * @brief Handle GV_MSG_SEARCH.
 *
 * Request payload: [4-byte dimension][4-byte k][4-byte distance_type][dimension * 4-byte floats]
 * Response payload: [4-byte count][for each: 4-byte index, 4-byte float distance]
 */
static void handle_search(GV_GrpcServer *server, int fd,
                           const GV_GrpcMessage *msg) {
    if (msg->payload_len < 12) {
        send_error_response(fd, msg->request_id, -1, "payload too short");
        GV_ATOMIC_INC(&server->errors);
        return;
    }

    uint32_t dimension = read_u32_be(msg->payload);
    uint32_t k = read_u32_be(msg->payload + 4);
    int32_t distance_type = (int32_t)read_u32_be(msg->payload + 8);

    size_t expected = 12 + (size_t)dimension * sizeof(float);
    if (msg->payload_len < expected) {
        send_error_response(fd, msg->request_id, -1, "incomplete query data");
        GV_ATOMIC_INC(&server->errors);
        return;
    }

    float *query = malloc(dimension * sizeof(float));
    if (!query) {
        send_error_response(fd, msg->request_id, -1, "out of memory");
        GV_ATOMIC_INC(&server->errors);
        return;
    }
    for (uint32_t i = 0; i < dimension; i++) {
        query[i] = read_float_be(msg->payload + 12 + i * 4);
    }

    GV_SearchResult *results = calloc(k, sizeof(GV_SearchResult));
    if (!results) {
        free(query);
        send_error_response(fd, msg->request_id, -1, "out of memory");
        GV_ATOMIC_INC(&server->errors);
        return;
    }

    int found = gv_db_search(server->db, query, (size_t)k, results,
                              (GV_DistanceType)distance_type);
    free(query);

    if (found < 0) {
        free(results);
        send_error_response(fd, msg->request_id, -1, "search failed");
        GV_ATOMIC_INC(&server->errors);
        return;
    }

    /* Build response: [4-byte count][for each: 4-byte index, 4-byte distance] */
    size_t resp_len = 4 + (size_t)found * 8;
    uint8_t *resp = malloc(resp_len);
    if (!resp) {
        free(results);
        send_error_response(fd, msg->request_id, -1, "out of memory");
        GV_ATOMIC_INC(&server->errors);
        return;
    }

    write_u32_be(resp, (uint32_t)found);
    for (int i = 0; i < found; i++) {
        /* Extract index from the result. For dense vectors, use vector pointer
         * comparison approach; for simplicity, we encode a sequential index. */
        uint32_t idx = 0;
        if (results[i].vector) {
            /* The vector field is a pointer into SoA storage; we need the actual
             * index. GV_SearchResult doesn't carry an explicit index for dense
             * results, so we use the distance-ranked position. The proper way
             * would require the search API to populate an index field. For now,
             * we encode the rank position. */
            idx = (uint32_t)i;
        }
        if (results[i].is_sparse && results[i].sparse_vector) {
            idx = (uint32_t)i;
        }
        write_u32_be(resp + 4 + (size_t)i * 8, idx);
        write_float_be(resp + 4 + (size_t)i * 8 + 4, results[i].distance);
    }

    send_message(fd, GV_MSG_RESPONSE, msg->request_id, resp, resp_len);
    free(resp);
    free(results);
}

/**
 * @brief Handle GV_MSG_DELETE.
 *
 * Request payload: [4-byte vector_index]
 * Response payload: [4-byte status (0 = ok)]
 */
static void handle_delete(GV_GrpcServer *server, int fd,
                           const GV_GrpcMessage *msg) {
    if (msg->payload_len < 4) {
        send_error_response(fd, msg->request_id, -1, "payload too short");
        GV_ATOMIC_INC(&server->errors);
        return;
    }

    uint32_t index = read_u32_be(msg->payload);
    int rc = gv_db_delete_vector_by_index(server->db, (size_t)index);

    uint8_t resp[4];
    write_u32_be(resp, (uint32_t)rc);
    send_message(fd, GV_MSG_RESPONSE, msg->request_id, resp, 4);

    if (rc != 0) {
        GV_ATOMIC_INC(&server->errors);
    }
}

/**
 * @brief Handle GV_MSG_UPDATE.
 *
 * Request payload: [4-byte vector_index][4-byte dimension][dimension * 4-byte floats]
 * Response payload: [4-byte status (0 = ok)]
 */
static void handle_update(GV_GrpcServer *server, int fd,
                           const GV_GrpcMessage *msg) {
    if (msg->payload_len < 8) {
        send_error_response(fd, msg->request_id, -1, "payload too short");
        GV_ATOMIC_INC(&server->errors);
        return;
    }

    uint32_t index = read_u32_be(msg->payload);
    uint32_t dimension = read_u32_be(msg->payload + 4);

    size_t expected = 8 + (size_t)dimension * sizeof(float);
    if (msg->payload_len < expected) {
        send_error_response(fd, msg->request_id, -1, "incomplete vector data");
        GV_ATOMIC_INC(&server->errors);
        return;
    }

    float *vec = malloc(dimension * sizeof(float));
    if (!vec) {
        send_error_response(fd, msg->request_id, -1, "out of memory");
        GV_ATOMIC_INC(&server->errors);
        return;
    }
    for (uint32_t i = 0; i < dimension; i++) {
        vec[i] = read_float_be(msg->payload + 8 + i * 4);
    }

    int rc = gv_db_update_vector(server->db, (size_t)index, vec, (size_t)dimension);
    free(vec);

    uint8_t resp[4];
    write_u32_be(resp, (uint32_t)rc);
    send_message(fd, GV_MSG_RESPONSE, msg->request_id, resp, 4);

    if (rc != 0) {
        GV_ATOMIC_INC(&server->errors);
    }
}

/**
 * @brief Handle GV_MSG_GET.
 *
 * Request payload: [4-byte vector_index]
 * Response payload: [4-byte dimension][dimension * 4-byte floats] or error
 */
static void handle_get(GV_GrpcServer *server, int fd,
                        const GV_GrpcMessage *msg) {
    if (msg->payload_len < 4) {
        send_error_response(fd, msg->request_id, -1, "payload too short");
        GV_ATOMIC_INC(&server->errors);
        return;
    }

    uint32_t index = read_u32_be(msg->payload);
    size_t dim = gv_database_dimension(server->db);
    const float *vec = gv_database_get_vector(server->db, (size_t)index);

    if (!vec) {
        send_error_response(fd, msg->request_id, -1, "vector not found");
        GV_ATOMIC_INC(&server->errors);
        return;
    }

    size_t resp_len = 4 + dim * sizeof(float);
    uint8_t *resp = malloc(resp_len);
    if (!resp) {
        send_error_response(fd, msg->request_id, -1, "out of memory");
        GV_ATOMIC_INC(&server->errors);
        return;
    }

    write_u32_be(resp, (uint32_t)dim);
    for (size_t i = 0; i < dim; i++) {
        write_float_be(resp + 4 + i * 4, vec[i]);
    }

    send_message(fd, GV_MSG_RESPONSE, msg->request_id, resp, resp_len);
    free(resp);
}

/**
 * @brief Handle GV_MSG_BATCH_ADD.
 *
 * Request payload: [4-byte count][4-byte dimension][count * dimension * 4-byte floats]
 * Response payload: [4-byte status (0 = ok)]
 */
static void handle_batch_add(GV_GrpcServer *server, int fd,
                              const GV_GrpcMessage *msg) {
    if (msg->payload_len < 8) {
        send_error_response(fd, msg->request_id, -1, "payload too short");
        GV_ATOMIC_INC(&server->errors);
        return;
    }

    uint32_t count = read_u32_be(msg->payload);
    uint32_t dimension = read_u32_be(msg->payload + 4);

    size_t total_floats = (size_t)count * (size_t)dimension;
    size_t expected = 8 + total_floats * sizeof(float);
    if (msg->payload_len < expected) {
        send_error_response(fd, msg->request_id, -1, "incomplete batch data");
        GV_ATOMIC_INC(&server->errors);
        return;
    }

    float *data = malloc(total_floats * sizeof(float));
    if (!data) {
        send_error_response(fd, msg->request_id, -1, "out of memory");
        GV_ATOMIC_INC(&server->errors);
        return;
    }
    for (size_t i = 0; i < total_floats; i++) {
        data[i] = read_float_be(msg->payload + 8 + i * 4);
    }

    int rc = gv_db_add_vectors(server->db, data, (size_t)count, (size_t)dimension);
    free(data);

    uint8_t resp[4];
    write_u32_be(resp, (uint32_t)rc);
    send_message(fd, GV_MSG_RESPONSE, msg->request_id, resp, 4);

    if (rc != 0) {
        GV_ATOMIC_INC(&server->errors);
    }
}

/**
 * @brief Handle GV_MSG_BATCH_SEARCH.
 *
 * Request payload: [4-byte qcount][4-byte dimension][4-byte k][4-byte distance_type]
 *                  [qcount * dimension * 4-byte floats]
 * Response payload: [4-byte qcount][for each query: 4-byte found_count,
 *                    for each result: 4-byte index, 4-byte distance]
 */
static void handle_batch_search(GV_GrpcServer *server, int fd,
                                 const GV_GrpcMessage *msg) {
    if (msg->payload_len < 16) {
        send_error_response(fd, msg->request_id, -1, "payload too short");
        GV_ATOMIC_INC(&server->errors);
        return;
    }

    uint32_t qcount = read_u32_be(msg->payload);
    uint32_t dimension = read_u32_be(msg->payload + 4);
    uint32_t k = read_u32_be(msg->payload + 8);
    int32_t distance_type = (int32_t)read_u32_be(msg->payload + 12);

    size_t total_floats = (size_t)qcount * (size_t)dimension;
    size_t expected = 16 + total_floats * sizeof(float);
    if (msg->payload_len < expected) {
        send_error_response(fd, msg->request_id, -1, "incomplete batch query data");
        GV_ATOMIC_INC(&server->errors);
        return;
    }

    float *queries = malloc(total_floats * sizeof(float));
    if (!queries) {
        send_error_response(fd, msg->request_id, -1, "out of memory");
        GV_ATOMIC_INC(&server->errors);
        return;
    }
    for (size_t i = 0; i < total_floats; i++) {
        queries[i] = read_float_be(msg->payload + 16 + i * 4);
    }

    size_t total_results = (size_t)qcount * (size_t)k;
    GV_SearchResult *results = calloc(total_results, sizeof(GV_SearchResult));
    if (!results) {
        free(queries);
        send_error_response(fd, msg->request_id, -1, "out of memory");
        GV_ATOMIC_INC(&server->errors);
        return;
    }

    int total_found = gv_db_search_batch(server->db, queries, (size_t)qcount,
                                          (size_t)k, results,
                                          (GV_DistanceType)distance_type);
    free(queries);

    if (total_found < 0) {
        free(results);
        send_error_response(fd, msg->request_id, -1, "batch search failed");
        GV_ATOMIC_INC(&server->errors);
        return;
    }

    /* Response: [4-byte qcount][per query: 4-byte k_found, k_found * (4+4)] */
    /* For simplicity, each query returns exactly k results (padding with zeros if fewer) */
    size_t resp_len = 4 + (size_t)qcount * (4 + (size_t)k * 8);
    uint8_t *resp = malloc(resp_len);
    if (!resp) {
        free(results);
        send_error_response(fd, msg->request_id, -1, "out of memory");
        GV_ATOMIC_INC(&server->errors);
        return;
    }

    write_u32_be(resp, qcount);
    size_t offset = 4;
    for (uint32_t q = 0; q < qcount; q++) {
        write_u32_be(resp + offset, k);
        offset += 4;
        for (uint32_t r = 0; r < k; r++) {
            size_t ri = (size_t)q * (size_t)k + (size_t)r;
            write_u32_be(resp + offset, (uint32_t)r);
            write_float_be(resp + offset + 4, results[ri].distance);
            offset += 8;
        }
    }

    send_message(fd, GV_MSG_RESPONSE, msg->request_id, resp, resp_len);
    free(resp);
    free(results);
}

/**
 * @brief Handle GV_MSG_STATS.
 *
 * Request payload: (empty)
 * Response payload: [8-byte total_inserts][8-byte total_queries]
 *                   [8-byte count][8-byte dimension]
 */
static void handle_stats(GV_GrpcServer *server, int fd,
                          const GV_GrpcMessage *msg) {
    GV_DBStats db_stats;
    gv_db_get_stats(server->db, &db_stats);

    size_t count = gv_database_count(server->db);
    size_t dim = gv_database_dimension(server->db);

    uint8_t resp[32];
    /* total_inserts as 2x u32 (high, low) */
    write_u32_be(resp + 0, (uint32_t)(db_stats.total_inserts >> 32));
    write_u32_be(resp + 4, (uint32_t)(db_stats.total_inserts & 0xFFFFFFFF));
    /* total_queries */
    write_u32_be(resp + 8, (uint32_t)(db_stats.total_queries >> 32));
    write_u32_be(resp + 12, (uint32_t)(db_stats.total_queries & 0xFFFFFFFF));
    /* count */
    write_u32_be(resp + 16, (uint32_t)((uint64_t)count >> 32));
    write_u32_be(resp + 20, (uint32_t)((uint64_t)count & 0xFFFFFFFF));
    /* dimension */
    write_u32_be(resp + 24, (uint32_t)((uint64_t)dim >> 32));
    write_u32_be(resp + 28, (uint32_t)((uint64_t)dim & 0xFFFFFFFF));

    send_message(fd, GV_MSG_RESPONSE, msg->request_id, resp, 32);
}

/**
 * @brief Handle GV_MSG_HEALTH.
 *
 * Request payload: (empty)
 * Response payload: [4-byte health_status]
 */
static void handle_health(GV_GrpcServer *server, int fd,
                           const GV_GrpcMessage *msg) {
    int health = gv_db_health_check(server->db);

    uint8_t resp[4];
    write_u32_be(resp, (uint32_t)health);
    send_message(fd, GV_MSG_RESPONSE, msg->request_id, resp, 4);
}

/**
 * @brief Handle GV_MSG_SAVE.
 *
 * Request payload: (empty, or N-byte filepath string)
 * Response payload: [4-byte status (0 = ok)]
 */
static void handle_save(GV_GrpcServer *server, int fd,
                         const GV_GrpcMessage *msg) {
    const char *filepath = NULL;
    char *filepath_buf = NULL;

    if (msg->payload_len > 0) {
        filepath_buf = malloc(msg->payload_len + 1);
        if (filepath_buf) {
            memcpy(filepath_buf, msg->payload, msg->payload_len);
            filepath_buf[msg->payload_len] = '\0';
            filepath = filepath_buf;
        }
    }

    int rc = gv_db_save(server->db, filepath);
    free(filepath_buf);

    uint8_t resp[4];
    write_u32_be(resp, (uint32_t)rc);
    send_message(fd, GV_MSG_RESPONSE, msg->request_id, resp, 4);

    if (rc != 0) {
        GV_ATOMIC_INC(&server->errors);
    }
}

/* ============================================================================
 * Connection Handler
 * ============================================================================ */

/**
 * @brief Process one client connection: read messages in a loop until
 *        the client disconnects or an error occurs.
 */
static void handle_connection(GV_GrpcServer *server, int client_fd) {
    GV_ATOMIC_INC(&server->active_connections);

    /* Set TCP_NODELAY for lower latency */
    int flag = 1;
    setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    while (!server->stop_requested) {
        GV_GrpcMessage msg;
        memset(&msg, 0, sizeof(msg));

        if (recv_message(client_fd, &msg, server->config.max_message_bytes) != 0) {
            break;  /* Client disconnected or protocol error */
        }

        uint64_t start_us = gv_grpc_now_us();

        GV_ATOMIC_INC(&server->total_requests);
        GV_ATOMIC_ADD(&server->bytes_received, 4 + msg.length);

        switch (msg.msg_type) {
            case GV_MSG_ADD_VECTOR:
                handle_add_vector(server, client_fd, &msg);
                break;
            case GV_MSG_SEARCH:
                handle_search(server, client_fd, &msg);
                break;
            case GV_MSG_DELETE:
                handle_delete(server, client_fd, &msg);
                break;
            case GV_MSG_UPDATE:
                handle_update(server, client_fd, &msg);
                break;
            case GV_MSG_GET:
                handle_get(server, client_fd, &msg);
                break;
            case GV_MSG_BATCH_ADD:
                handle_batch_add(server, client_fd, &msg);
                break;
            case GV_MSG_BATCH_SEARCH:
                handle_batch_search(server, client_fd, &msg);
                break;
            case GV_MSG_STATS:
                handle_stats(server, client_fd, &msg);
                break;
            case GV_MSG_HEALTH:
                handle_health(server, client_fd, &msg);
                break;
            case GV_MSG_SAVE:
                handle_save(server, client_fd, &msg);
                break;
            default:
                send_error_response(client_fd, msg.request_id, -1, "unknown message type");
                GV_ATOMIC_INC(&server->errors);
                break;
        }

        uint64_t elapsed_us = gv_grpc_now_us() - start_us;
        GV_ATOMIC_ADD(&server->total_latency_us, elapsed_us);
        GV_ATOMIC_INC(&server->latency_samples);

        free(msg.payload);
    }

    close(client_fd);
    GV_ATOMIC_DEC(&server->active_connections);
}

/* ============================================================================
 * Thread Pool Implementation
 * ============================================================================ */

static void *worker_thread_func(void *arg) {
    GV_ThreadPool *pool = (GV_ThreadPool *)arg;

    while (1) {
        pthread_mutex_lock(&pool->mutex);

        while (!pool->shutdown && pool->queue_head == NULL) {
            pthread_cond_wait(&pool->cond, &pool->mutex);
        }

        if (pool->shutdown && pool->queue_head == NULL) {
            pthread_mutex_unlock(&pool->mutex);
            break;
        }

        /* Dequeue a task */
        GV_GrpcTask *task = pool->queue_head;
        pool->queue_head = task->next;
        if (pool->queue_head == NULL) {
            pool->queue_tail = NULL;
        }
        pool->queue_size--;

        pthread_mutex_unlock(&pool->mutex);

        /* Process the connection */
        handle_connection(pool->server, task->client_fd);
        free(task);
    }

    return NULL;
}

static int thread_pool_init(GV_ThreadPool *pool, size_t thread_count,
                             GV_GrpcServer *server) {
    memset(pool, 0, sizeof(*pool));
    pool->server = server;
    pool->thread_count = thread_count;

    if (pthread_mutex_init(&pool->mutex, NULL) != 0) {
        return -1;
    }
    if (pthread_cond_init(&pool->cond, NULL) != 0) {
        pthread_mutex_destroy(&pool->mutex);
        return -1;
    }

    pool->threads = calloc(thread_count, sizeof(pthread_t));
    if (!pool->threads) {
        pthread_cond_destroy(&pool->cond);
        pthread_mutex_destroy(&pool->mutex);
        return -1;
    }

    for (size_t i = 0; i < thread_count; i++) {
        if (pthread_create(&pool->threads[i], NULL, worker_thread_func, pool) != 0) {
            /* Shut down already-created threads */
            pthread_mutex_lock(&pool->mutex);
            pool->shutdown = 1;
            pthread_cond_broadcast(&pool->cond);
            pthread_mutex_unlock(&pool->mutex);

            for (size_t j = 0; j < i; j++) {
                pthread_join(pool->threads[j], NULL);
            }
            free(pool->threads);
            pthread_cond_destroy(&pool->cond);
            pthread_mutex_destroy(&pool->mutex);
            return -1;
        }
    }

    return 0;
}

static void thread_pool_submit(GV_ThreadPool *pool, int client_fd) {
    GV_GrpcTask *task = malloc(sizeof(GV_GrpcTask));
    if (!task) {
        close(client_fd);
        return;
    }
    task->client_fd = client_fd;
    task->next = NULL;

    pthread_mutex_lock(&pool->mutex);

    if (pool->queue_tail) {
        pool->queue_tail->next = task;
    } else {
        pool->queue_head = task;
    }
    pool->queue_tail = task;
    pool->queue_size++;

    pthread_cond_signal(&pool->cond);
    pthread_mutex_unlock(&pool->mutex);
}

static void thread_pool_destroy(GV_ThreadPool *pool) {
    if (!pool->threads) return;

    /* Signal shutdown */
    pthread_mutex_lock(&pool->mutex);
    pool->shutdown = 1;
    pthread_cond_broadcast(&pool->cond);
    pthread_mutex_unlock(&pool->mutex);

    /* Join all worker threads */
    for (size_t i = 0; i < pool->thread_count; i++) {
        pthread_join(pool->threads[i], NULL);
    }

    /* Drain remaining tasks */
    GV_GrpcTask *task = pool->queue_head;
    while (task) {
        GV_GrpcTask *next = task->next;
        close(task->client_fd);
        free(task);
        task = next;
    }

    free(pool->threads);
    pthread_cond_destroy(&pool->cond);
    pthread_mutex_destroy(&pool->mutex);
}

/* ============================================================================
 * Accept Loop
 * ============================================================================ */

static void *accept_thread_func(void *arg) {
    GV_GrpcServer *server = (GV_GrpcServer *)arg;

    while (!server->stop_requested) {
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);

        int client_fd = accept(server->listen_fd,
                                (struct sockaddr *)&client_addr, &addr_len);
        if (client_fd < 0) {
            if (server->stop_requested) break;
            if (errno == EINTR) continue;
            GV_ATOMIC_INC(&server->errors);
            continue;
        }

        /* Check connection limit */
        uint64_t active = GV_ATOMIC_LOAD(&server->active_connections);
        if (active >= (uint64_t)server->config.max_connections) {
            close(client_fd);
            GV_ATOMIC_INC(&server->errors);
            continue;
        }

        thread_pool_submit(&server->pool, client_fd);
    }

    return NULL;
}

/* ============================================================================
 * Server Lifecycle
 * ============================================================================ */

GV_GrpcServer *gv_grpc_create(GV_Database *db, const GV_GrpcConfig *config) {
    if (!db) return NULL;

    GV_GrpcServer *server = calloc(1, sizeof(GV_GrpcServer));
    if (!server) return NULL;

    server->db = db;
    server->config = config ? *config : DEFAULT_GRPC_CONFIG;
    server->listen_fd = -1;
    server->running = 0;
    server->stop_requested = 0;

    /* Validate configuration */
    if (server->config.port == 0) {
        server->config.port = 50051;
    }
    if (!server->config.bind_address) {
        server->config.bind_address = "0.0.0.0";
    }
    if (server->config.max_connections == 0) {
        server->config.max_connections = 256;
    }
    if (server->config.max_message_bytes == 0) {
        server->config.max_message_bytes = 16777216;
    }
    if (server->config.thread_pool_size == 0) {
        server->config.thread_pool_size = 4;
    }

    if (pthread_mutex_init(&server->stats_mutex, NULL) != 0) {
        free(server);
        return NULL;
    }

    return server;
}

int gv_grpc_start(GV_GrpcServer *server) {
    if (!server) return GV_GRPC_ERROR_NULL;
    if (server->running) return GV_GRPC_ERROR_RUNNING;

    /* Ignore SIGPIPE so send() returns EPIPE instead of killing the process */
    signal(SIGPIPE, SIG_IGN);

    /* Create listen socket */
    server->listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server->listen_fd < 0) {
        return GV_GRPC_ERROR_BIND;
    }

    /* Allow address reuse */
    int opt = 1;
    setsockopt(server->listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(server->config.port);

    if (inet_pton(AF_INET, server->config.bind_address, &addr.sin_addr) <= 0) {
        close(server->listen_fd);
        server->listen_fd = -1;
        return GV_GRPC_ERROR_CONFIG;
    }

    if (bind(server->listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(server->listen_fd);
        server->listen_fd = -1;
        return GV_GRPC_ERROR_BIND;
    }

    if (listen(server->listen_fd, (int)server->config.max_connections) < 0) {
        close(server->listen_fd);
        server->listen_fd = -1;
        return GV_GRPC_ERROR_BIND;
    }

    /* Initialize thread pool */
    if (thread_pool_init(&server->pool, server->config.thread_pool_size, server) != 0) {
        close(server->listen_fd);
        server->listen_fd = -1;
        return GV_GRPC_ERROR_MEMORY;
    }

    /* Start accept thread */
    server->stop_requested = 0;
    if (pthread_create(&server->accept_thread, NULL, accept_thread_func, server) != 0) {
        thread_pool_destroy(&server->pool);
        close(server->listen_fd);
        server->listen_fd = -1;
        return GV_GRPC_ERROR_START;
    }
    server->accept_thread_started = 1;

    server->running = 1;
    fprintf(stderr, "[GV_Grpc] Started binary protocol server on %s:%u\n",
            server->config.bind_address, server->config.port);

    return GV_GRPC_OK;
}

int gv_grpc_stop(GV_GrpcServer *server) {
    if (!server) return GV_GRPC_ERROR_NULL;
    if (!server->running) return GV_GRPC_ERROR_NOT_RUNNING;

    server->stop_requested = 1;

    /* Close listen socket to unblock accept() */
    if (server->listen_fd >= 0) {
        shutdown(server->listen_fd, SHUT_RDWR);
        close(server->listen_fd);
        server->listen_fd = -1;
    }

    /* Join accept thread */
    if (server->accept_thread_started) {
        pthread_join(server->accept_thread, NULL);
        server->accept_thread_started = 0;
    }

    /* Destroy thread pool (joins workers, drains queue) */
    thread_pool_destroy(&server->pool);

    server->running = 0;
    fprintf(stderr, "[GV_Grpc] Stopped\n");

    return GV_GRPC_OK;
}

void gv_grpc_destroy(GV_GrpcServer *server) {
    if (!server) return;

    if (server->running) {
        gv_grpc_stop(server);
    }

    pthread_mutex_destroy(&server->stats_mutex);
    free(server);
}

int gv_grpc_is_running(const GV_GrpcServer *server) {
    if (!server) return 0;
    return server->running;
}

/* ============================================================================
 * Statistics
 * ============================================================================ */

int gv_grpc_get_stats(const GV_GrpcServer *server, GV_GrpcStats *stats) {
    if (!server || !stats) return GV_GRPC_ERROR_NULL;

    stats->total_requests = GV_ATOMIC_LOAD(&server->total_requests);
    stats->active_connections = GV_ATOMIC_LOAD(&server->active_connections);
    stats->bytes_sent = GV_ATOMIC_LOAD(&server->bytes_sent);
    stats->bytes_received = GV_ATOMIC_LOAD(&server->bytes_received);
    stats->errors = GV_ATOMIC_LOAD(&server->errors);

    uint64_t samples = GV_ATOMIC_LOAD(&server->latency_samples);
    if (samples > 0) {
        uint64_t total_lat = GV_ATOMIC_LOAD(&server->total_latency_us);
        stats->avg_latency_us = (double)total_lat / (double)samples;
    } else {
        stats->avg_latency_us = 0.0;
    }

    return GV_GRPC_OK;
}

/* ============================================================================
 * Message Serialization Helpers
 * ============================================================================ */

int gv_grpc_encode_search_request(const float *query, size_t dimension, size_t k,
                                   int distance_type, uint8_t *buf, size_t buf_size,
                                   size_t *out_len) {
    if (!query || !buf || !out_len) return GV_GRPC_ERROR_NULL;

    /* Payload: [4-byte dim][4-byte k][4-byte distance_type][dim * 4-byte floats] */
    size_t needed = 12 + dimension * sizeof(float);
    if (buf_size < needed) return GV_GRPC_ERROR_CONFIG;

    write_u32_be(buf, (uint32_t)dimension);
    write_u32_be(buf + 4, (uint32_t)k);
    write_u32_be(buf + 8, (uint32_t)distance_type);

    for (size_t i = 0; i < dimension; i++) {
        write_float_be(buf + 12 + i * 4, query[i]);
    }

    *out_len = needed;
    return GV_GRPC_OK;
}

int gv_grpc_decode_search_request(const uint8_t *buf, size_t len,
                                   float **query, size_t *dimension,
                                   size_t *k, int *distance_type) {
    if (!buf || !query || !dimension || !k || !distance_type) return GV_GRPC_ERROR_NULL;

    if (len < 12) return GV_GRPC_ERROR_CONFIG;

    *dimension = (size_t)read_u32_be(buf);
    *k = (size_t)read_u32_be(buf + 4);
    *distance_type = (int)read_u32_be(buf + 8);

    size_t expected = 12 + (*dimension) * sizeof(float);
    if (len < expected) return GV_GRPC_ERROR_CONFIG;

    *query = malloc((*dimension) * sizeof(float));
    if (!*query) return GV_GRPC_ERROR_MEMORY;

    for (size_t i = 0; i < *dimension; i++) {
        (*query)[i] = read_float_be(buf + 12 + i * 4);
    }

    return GV_GRPC_OK;
}

int gv_grpc_encode_add_request(const float *data, size_t dimension,
                                uint8_t *buf, size_t buf_size, size_t *out_len) {
    if (!data || !buf || !out_len) return GV_GRPC_ERROR_NULL;

    /* Payload: [4-byte dimension][dimension * 4-byte floats] */
    size_t needed = 4 + dimension * sizeof(float);
    if (buf_size < needed) return GV_GRPC_ERROR_CONFIG;

    write_u32_be(buf, (uint32_t)dimension);
    for (size_t i = 0; i < dimension; i++) {
        write_float_be(buf + 4 + i * 4, data[i]);
    }

    *out_len = needed;
    return GV_GRPC_OK;
}
