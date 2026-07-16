#ifndef GIGAVECTOR_GV_GRPC_H
#define GIGAVECTOR_GV_GRPC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

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
    GV_MSG_IVFDISK_TRAIN = 11,
    GV_MSG_AUTH = 12,   /* Auth handshake. Payload = raw shared-secret token bytes. */
    GV_MSG_RESPONSE = 128
} GV_GrpcMsgType;

typedef struct {
    uint16_t port;                  /* Port to listen on (default: 50051) */
    const char *bind_address;       /* Bind address (default: "0.0.0.0") */
    size_t max_connections;         /* Max concurrent connections */
    size_t max_message_bytes;       /* Max message size (default: 16MB) */
    size_t thread_pool_size;        /* Worker threads (default: 4) */
    int enable_compression;         /* Enable message compression */
    const char *auth_token;         /* Optional shared-secret token (default: NULL).
                                     * When set, clients must send a GV_MSG_AUTH
                                     * handshake with this token before any other
                                     * message type is accepted. */
    const char *data_dir;           /* Base dir that SAVE output is confined to
                                     * (default: GV_GRPC_DEFAULT_DATA_DIR). */
} GV_GrpcConfig;

/* Default confinement base directory for the gRPC SAVE handler. */
#define GV_GRPC_DEFAULT_DATA_DIR "./data"

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

/**
 * @brief Initialize a configuration structure with default values.
 *
 * @param config Configuration to apply/output.
 */
void grpc_config_init(GV_GrpcConfig *config);
GV_GrpcServer *grpc_create(GV_Database *db, const GV_GrpcConfig *config);
/**
 * @brief Start the service.
 *
 * @param server server.
 * @return 0 on success, -1 on error.
 */
int grpc_start(GV_GrpcServer *server);
/**
 * @brief Stop the service.
 *
 * @param server server.
 * @return 0 on success, -1 on error.
 */
int grpc_stop(GV_GrpcServer *server);
/**
 * @brief Destroy an instance and free associated resources.
 *
 * @param server server.
 */
void grpc_destroy(GV_GrpcServer *server);
/**
 * @brief Query a boolean condition.
 *
 * @param server server.
 * @return 1 if true, 0 if false, -1 on error.
 */
int grpc_is_running(const GV_GrpcServer *server);
/**
 * @brief Retrieve statistics.
 *
 * @param server server.
 * @param stats Output statistics structure.
 * @return 0 on success, -1 on error.
 */
int grpc_get_stats(const GV_GrpcServer *server, GV_GrpcStats *stats);
const char *grpc_error_string(int error);
int grpc_encode_search_request(const float *query, size_t dimension, size_t k,
                                   int distance_type, uint8_t *buf, size_t buf_size, size_t *out_len);
int grpc_decode_search_request(const uint8_t *buf, size_t len,
                                   float **query, size_t *dimension, size_t *k, int *distance_type);
int grpc_encode_add_request(const float *data, size_t dimension,
                                uint8_t *buf, size_t buf_size, size_t *out_len);

/** Payload: [4-byte count][4-byte dimension][count * dimension floats] */
int grpc_encode_ivfdisk_train_request(const float *data, size_t count, size_t dimension,
                                      uint8_t *buf, size_t buf_size, size_t *out_len);

/** Train IVFDisk centroids on a remote server. Returns 0 on success, -1 on error. */
int grpc_client_ivfdisk_train(const char *host, uint16_t port,
                              const float *data, size_t count, size_t dimension,
                              uint32_t timeout_ms);

/** Parse a complete wire frame from @p data (for fuzzing / in-memory dispatch). */
int grpc_decode_frame(const uint8_t *data, size_t len, size_t max_bytes, GV_GrpcMessage *msg);
void grpc_message_free(GV_GrpcMessage *msg);

/**
 * Dispatch one decoded message through server handlers (response written to @p response_fd).
 * Used by libFuzzer without starting the network server.
 */
int grpc_fuzz_dispatch_message(GV_GrpcServer *server, int response_fd,
                               const GV_GrpcMessage *msg);

typedef struct {
    size_t count;
    size_t *indices;
    float *distances;
} GV_GrpcSearchResponse;

/**
 * @brief Execute a vector search against a remote gRPC-style server.
 *
 * @return Number of hits on success, 0 when empty, -1 on error.
 */
int grpc_client_search(const char *host, uint16_t port,
                         const float *query, size_t dimension, size_t k,
                         int distance_type, GV_GrpcSearchResponse *out,
                         uint32_t timeout_ms);

void grpc_search_response_free(GV_GrpcSearchResponse *resp);

#ifdef __cplusplus
}
#endif
#endif
