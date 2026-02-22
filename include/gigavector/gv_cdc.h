#ifndef GIGAVECTOR_GV_CDC_H
#define GIGAVECTOR_GV_CDC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_cdc.h
 * @brief Change Data Capture (CDC) stream for GigaVector.
 *
 * Streams database mutations (insert, update, delete) to external consumers
 * via callbacks or a ring buffer.  Enables cross-instance replication,
 * event-driven architectures, and audit logging.
 */

/* Types and Enumerations */

/**
 * @brief CDC event types (bitmask).
 */
typedef enum {
    GV_CDC_INSERT   = 1,            /**< Vector inserted. */
    GV_CDC_UPDATE   = 2,            /**< Vector updated. */
    GV_CDC_DELETE   = 4,            /**< Vector deleted. */
    GV_CDC_SNAPSHOT = 8,            /**< Full snapshot marker. */
    GV_CDC_ALL      = 15            /**< All event types. */
} GV_CDCEventType;

/**
 * @brief A single CDC event.
 */
typedef struct {
    uint64_t sequence_number;       /**< Monotonically increasing sequence. */
    GV_CDCEventType type;           /**< Event type. */
    size_t vector_index;            /**< Index of the affected vector. */
    uint64_t timestamp;             /**< Unix epoch timestamp (nanoseconds). */
    const float *vector_data;       /**< Vector payload (NULL for delete). */
    size_t dimension;               /**< Number of elements in vector_data. */
    const char *metadata_json;      /**< JSON metadata string (NULL if none). */
} GV_CDCEvent;

/**
 * @brief Callback invoked for each matching CDC event.
 *
 * @param event The CDC event.
 * @param user_data Opaque pointer supplied at subscription time.
 */
typedef void (*GV_CDCCallback)(const GV_CDCEvent *event, void *user_data);

/**
 * @brief CDC stream configuration.
 */
typedef struct {
    size_t ring_buffer_size;        /**< Ring buffer capacity in events (default: 65536). */
    int persist_to_file;            /**< Write events to log file (default: 0). */
    const char *log_path;           /**< Path to persistent log file. */
    size_t max_log_size_mb;         /**< Maximum log file size in MiB (default: 256). */
    int include_vector_data;        /**< Include vector data in events (default: 1). */
} GV_CDCConfig;

/**
 * @brief Opaque CDC stream handle.
 */
typedef struct GV_CDCStream GV_CDCStream;

/**
 * @brief Cursor into the CDC stream.
 */
typedef struct {
    uint64_t sequence_number;       /**< Position in the stream. */
} GV_CDCCursor;

/* Configuration */

/**
 * @brief Initialize a CDC configuration with sensible defaults.
 *
 * @param config Configuration to initialize.
 */
void gv_cdc_config_init(GV_CDCConfig *config);

/* Lifecycle */

/**
 * @brief Create a CDC stream.
 *
 * @param config Stream configuration (NULL for defaults).
 * @return CDC stream handle, or NULL on error.
 */
GV_CDCStream *gv_cdc_create(const GV_CDCConfig *config);

/**
 * @brief Destroy a CDC stream and release all resources.
 *
 * @param stream CDC stream (safe to call with NULL).
 */
void gv_cdc_destroy(GV_CDCStream *stream);

/* Publishing */

/**
 * @brief Publish an event to the CDC stream.
 *
 * Deep-copies vector data and metadata into the ring buffer.  Notifies
 * all matching subscribers.  If the ring buffer is full the oldest event
 * is overwritten.
 *
 * @param stream CDC stream.
 * @param event  Event to publish.
 * @return 0 on success, -1 on error.
 */
int gv_cdc_publish(GV_CDCStream *stream, const GV_CDCEvent *event);

/* Subscription (push interface) */

/**
 * @brief Subscribe to CDC events matching a type mask.
 *
 * Up to 32 concurrent subscribers are supported.
 *
 * @param stream     CDC stream.
 * @param event_mask Bitmask of GV_CDCEventType values.
 * @param callback   Function invoked for each matching event.
 * @param user_data  Opaque pointer forwarded to the callback.
 * @return Non-negative subscriber ID on success, -1 on error.
 */
int gv_cdc_subscribe(GV_CDCStream *stream, uint32_t event_mask,
                     GV_CDCCallback callback, void *user_data);

/**
 * @brief Remove a subscription.
 *
 * @param stream        CDC stream.
 * @param subscriber_id ID returned by gv_cdc_subscribe().
 * @return 0 on success, -1 on error.
 */
int gv_cdc_unsubscribe(GV_CDCStream *stream, int subscriber_id);

/* Polling (pull interface) */

/**
 * @brief Poll for events starting from a cursor position.
 *
 * Copies up to @p max_events events into the caller-supplied buffer and
 * advances @p cursor past the returned events.  The caller must NOT free
 * the vector_data / metadata_json pointers in the returned events; those
 * point into the ring buffer and are valid only until the ring wraps.
 *
 * @param stream     CDC stream.
 * @param cursor     In/out cursor (advanced on return).
 * @param events     Output array of events.
 * @param max_events Maximum number of events to return.
 * @return Number of events written (>= 0), or -1 on error.
 */
int gv_cdc_poll(GV_CDCStream *stream, GV_CDCCursor *cursor,
                GV_CDCEvent *events, size_t max_events);

/**
 * @brief Get a cursor pointing to the latest position in the stream.
 *
 * A subsequent gv_cdc_poll() with this cursor will return only events
 * published after this call.
 *
 * @param stream CDC stream.
 * @return Cursor at the current head.
 */
GV_CDCCursor gv_cdc_get_cursor(const GV_CDCStream *stream);

/**
 * @brief Construct a cursor from an explicit sequence number.
 *
 * @param seq Sequence number.
 * @return Cursor positioned at @p seq.
 */
GV_CDCCursor gv_cdc_cursor_from_sequence(uint64_t seq);

/**
 * @brief Return the number of events available after @p cursor.
 *
 * @param stream CDC stream.
 * @param cursor Current consumer position.
 * @return Number of pending events, or 0 on error.
 */
size_t gv_cdc_pending_count(const GV_CDCStream *stream, const GV_CDCCursor *cursor);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_CDC_H */
