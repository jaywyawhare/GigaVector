#ifndef GIGAVECTOR_GV_STREAMING_H
#define GIGAVECTOR_GV_STREAMING_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_streaming.h
 * @brief Streaming data ingestion for GigaVector.
 *
 * Provides integration with message queues like Kafka for real-time vector ingestion.
 */

/* Forward declarations */
struct GV_Database;
typedef struct GV_Database GV_Database;

/**
 * @brief Stream source type.
 */
typedef enum {
    GV_STREAM_KAFKA = 0,            /**< Apache Kafka. */
    GV_STREAM_PULSAR = 1,           /**< Apache Pulsar. */
    GV_STREAM_REDIS = 2,            /**< Redis Streams. */
    GV_STREAM_CUSTOM = 3            /**< Custom source. */
} GV_StreamSource;

/**
 * @brief Stream consumer state.
 */
typedef enum {
    GV_STREAM_STOPPED = 0,          /**< Consumer stopped. */
    GV_STREAM_RUNNING = 1,          /**< Consumer running. */
    GV_STREAM_PAUSED = 2,           /**< Consumer paused. */
    GV_STREAM_ERROR = 3             /**< Consumer in error state. */
} GV_StreamState;

/**
 * @brief Kafka-specific configuration.
 */
typedef struct {
    const char *brokers;            /**< Comma-separated broker list. */
    const char *topic;              /**< Topic to consume from. */
    const char *consumer_group;     /**< Consumer group ID. */
    int32_t partition;              /**< Partition (-1 for all). */
    int64_t start_offset;           /**< Starting offset (-1 for latest). */
    const char *security_protocol;  /**< Security protocol (e.g., "SASL_SSL"). */
    const char *sasl_mechanism;     /**< SASL mechanism (e.g., "PLAIN"). */
    const char *sasl_username;      /**< SASL username. */
    const char *sasl_password;      /**< SASL password. */
} GV_KafkaConfig;

/**
 * @brief Stream consumer configuration.
 */
typedef struct {
    GV_StreamSource source;         /**< Stream source type. */
    size_t batch_size;              /**< Messages per batch (default: 100). */
    uint32_t batch_timeout_ms;      /**< Batch timeout in ms (default: 1000). */
    size_t max_buffer_size;         /**< Maximum buffer size (default: 10000). */
    int auto_commit;                /**< Auto-commit offsets (default: 1). */
    uint32_t commit_interval_ms;    /**< Commit interval (default: 5000). */
    GV_KafkaConfig kafka;           /**< Kafka-specific config. */
} GV_StreamConfig;

/**
 * @brief Stream message.
 */
typedef struct {
    const void *key;                /**< Message key. */
    size_t key_len;                 /**< Key length. */
    const void *value;              /**< Message value (vector data). */
    size_t value_len;               /**< Value length. */
    int64_t offset;                 /**< Message offset. */
    int64_t timestamp;              /**< Message timestamp. */
    int32_t partition;              /**< Partition. */
} GV_StreamMessage;

/**
 * @brief Message handler callback.
 *
 * @param msg Message to process.
 * @param user_data User data.
 * @return 0 to continue, non-zero to stop.
 */
typedef int (*GV_StreamMessageHandler)(const GV_StreamMessage *msg, void *user_data);

/**
 * @brief Vector extractor callback.
 *
 * Extracts vector data from message.
 *
 * @param msg Message to extract from.
 * @param vector Output vector buffer.
 * @param dimension Expected dimension.
 * @param metadata_keys Output metadata keys (optional).
 * @param metadata_values Output metadata values (optional).
 * @param metadata_count Output metadata count.
 * @param user_data User data.
 * @return 0 on success, -1 on error.
 */
typedef int (*GV_VectorExtractor)(const GV_StreamMessage *msg, float *vector,
                                   size_t dimension, char ***metadata_keys,
                                   char ***metadata_values, size_t *metadata_count,
                                   void *user_data);

/**
 * @brief Stream consumer statistics.
 */
typedef struct {
    uint64_t messages_received;     /**< Total messages received. */
    uint64_t messages_processed;    /**< Messages successfully processed. */
    uint64_t messages_failed;       /**< Messages that failed processing. */
    uint64_t vectors_ingested;      /**< Vectors added to database. */
    uint64_t bytes_received;        /**< Total bytes received. */
    double avg_batch_time_ms;       /**< Average batch processing time. */
    int64_t current_offset;         /**< Current consumer offset. */
    int64_t lag;                    /**< Consumer lag. */
} GV_StreamStats;

/**
 * @brief Opaque stream consumer handle.
 */
typedef struct GV_StreamConsumer GV_StreamConsumer;

/* Configuration */

/**
 * @brief Initialize stream configuration with defaults.
 *
 * @param config Configuration to initialize.
 */
void gv_stream_config_init(GV_StreamConfig *config);

/* Stream Consumer Lifecycle */

/**
 * @brief Create a stream consumer.
 *
 * @param db Database to ingest vectors into.
 * @param config Stream configuration.
 * @return Stream consumer, or NULL on error.
 */
GV_StreamConsumer *gv_stream_create(GV_Database *db, const GV_StreamConfig *config);

/**
 * @brief Destroy a stream consumer.
 *
 * @param consumer Stream consumer (safe to call with NULL).
 */
void gv_stream_destroy(GV_StreamConsumer *consumer);

/**
 * @brief Start consuming.
 *
 * @param consumer Stream consumer.
 * @return 0 on success, -1 on error.
 */
int gv_stream_start(GV_StreamConsumer *consumer);

/**
 * @brief Stop consuming.
 *
 * @param consumer Stream consumer.
 * @return 0 on success, -1 on error.
 */
int gv_stream_stop(GV_StreamConsumer *consumer);

/**
 * @brief Pause consuming.
 *
 * @param consumer Stream consumer.
 * @return 0 on success, -1 on error.
 */
int gv_stream_pause(GV_StreamConsumer *consumer);

/**
 * @brief Resume consuming.
 *
 * @param consumer Stream consumer.
 * @return 0 on success, -1 on error.
 */
int gv_stream_resume(GV_StreamConsumer *consumer);

/* Configuration */

/**
 * @brief Set vector extractor.
 *
 * @param consumer Stream consumer.
 * @param extractor Extractor function.
 * @param user_data User data for extractor.
 * @return 0 on success, -1 on error.
 */
int gv_stream_set_extractor(GV_StreamConsumer *consumer, GV_VectorExtractor extractor,
                             void *user_data);

/**
 * @brief Set message handler.
 *
 * @param consumer Stream consumer.
 * @param handler Handler function.
 * @param user_data User data for handler.
 * @return 0 on success, -1 on error.
 */
int gv_stream_set_handler(GV_StreamConsumer *consumer, GV_StreamMessageHandler handler,
                           void *user_data);

/* Status and Statistics */

/**
 * @brief Get consumer state.
 *
 * @param consumer Stream consumer.
 * @return Current state, or -1 on error.
 */
GV_StreamState gv_stream_get_state(GV_StreamConsumer *consumer);

/**
 * @brief Get consumer statistics.
 *
 * @param consumer Stream consumer.
 * @param stats Output statistics.
 * @return 0 on success, -1 on error.
 */
int gv_stream_get_stats(GV_StreamConsumer *consumer, GV_StreamStats *stats);

/**
 * @brief Reset statistics.
 *
 * @param consumer Stream consumer.
 * @return 0 on success, -1 on error.
 */
int gv_stream_reset_stats(GV_StreamConsumer *consumer);

/* Offset Management */

/**
 * @brief Commit current offsets.
 *
 * @param consumer Stream consumer.
 * @return 0 on success, -1 on error.
 */
int gv_stream_commit(GV_StreamConsumer *consumer);

/**
 * @brief Seek to offset.
 *
 * @param consumer Stream consumer.
 * @param offset Offset to seek to.
 * @return 0 on success, -1 on error.
 */
int gv_stream_seek(GV_StreamConsumer *consumer, int64_t offset);

/**
 * @brief Seek to beginning.
 *
 * @param consumer Stream consumer.
 * @return 0 on success, -1 on error.
 */
int gv_stream_seek_beginning(GV_StreamConsumer *consumer);

/**
 * @brief Seek to end.
 *
 * @param consumer Stream consumer.
 * @return 0 on success, -1 on error.
 */
int gv_stream_seek_end(GV_StreamConsumer *consumer);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_STREAMING_H */
