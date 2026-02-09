#ifndef GIGAVECTOR_GV_GRPC_H
#define GIGAVECTOR_GV_GRPC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declaration */
typedef struct GV_Database GV_Database;

typedef enum {
    GV_GRPC_OK = 0,
    GV_GRPC_ERROR_NULL = -1,
    GV_GRPC_ERROR_CONFIG = -2,
    GV_GRPC_ERROR_RUNNING = -3,
    GV_GRPC_ERROR_NOT_RUNNING = -4,
    GV_GRPC_ERROR_START = -5,
    GV_GRPC_ERROR_MEMORY = -6,
    GV_GRPC_ERROR_BIND = -7
} GV_GrpcError;

/* Message types for the binary protocol */
typedef enum {
    GV_MSG_ADD_VECTOR = 1,
    GV_MSG_SEARCH = 2,
    GV_MSG_DELETE = 3,
    GV_MSG_UPDATE = 4,
    GV_MSG_GET = 5,
    GV_MSG_BATCH_ADD = 6,
    GV_MSG_BATCH_SEARCH = 7,
    GV_MSG_STATS = 8,
    GV_MSG_HEALTH = 9,
    GV_MSG_SAVE = 10,
    GV_MSG_RESPONSE = 128
} GV_GrpcMsgType;

typedef struct {
    uint16_t port;                  /* Port to listen on (default: 50051) */
    const char *bind_address;       /* Bind address (default: "0.0.0.0") */
    size_t max_connections;         /* Max concurrent connections */
    size_t max_message_bytes;       /* Max message size (default: 16MB) */
    size_t thread_pool_size;        /* Worker threads (default: 4) */
    int enable_compression;         /* Enable message compression */
} GV_GrpcConfig;

/* Wire format: [4-byte length][1-byte type][payload] */
typedef struct {
    uint32_t length;                /* Total message length (excluding this field) */
    uint8_t msg_type;               /* GV_GrpcMsgType */
    uint32_t request_id;            /* For request-response matching */
    uint8_t *payload;               /* Serialized payload */
    size_t payload_len;
} GV_GrpcMessage;

typedef struct {
    uint64_t total_requests;
    uint64_t active_connections;
    uint64_t bytes_sent;
    uint64_t bytes_received;
    uint64_t errors;
    double avg_latency_us;
} GV_GrpcStats;

typedef struct GV_GrpcServer GV_GrpcServer;

/* Lifecycle */
void gv_grpc_config_init(GV_GrpcConfig *config);
GV_GrpcServer *gv_grpc_create(GV_Database *db, const GV_GrpcConfig *config);
int gv_grpc_start(GV_GrpcServer *server);
int gv_grpc_stop(GV_GrpcServer *server);
void gv_grpc_destroy(GV_GrpcServer *server);
int gv_grpc_is_running(const GV_GrpcServer *server);

/* Stats */
int gv_grpc_get_stats(const GV_GrpcServer *server, GV_GrpcStats *stats);
const char *gv_grpc_error_string(int error);

/* Message serialization helpers */
int gv_grpc_encode_search_request(const float *query, size_t dimension, size_t k,
                                   int distance_type, uint8_t *buf, size_t buf_size, size_t *out_len);
int gv_grpc_decode_search_request(const uint8_t *buf, size_t len,
                                   float **query, size_t *dimension, size_t *k, int *distance_type);
int gv_grpc_encode_add_request(const float *data, size_t dimension,
                                uint8_t *buf, size_t buf_size, size_t *out_len);

#ifdef __cplusplus
}
#endif
#endif
