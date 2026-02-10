#define _POSIX_C_SOURCE 200809L

/**
 * @file gv_cdc.c
 * @brief Change Data Capture (CDC) stream implementation.
 *
 * Provides a lock-protected ring buffer of CDC events with both push
 * (callback subscription) and pull (cursor-based polling) interfaces.
 * Optionally persists events to a binary log file for durable replay.
 */

#include "gigavector/gv_cdc.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>
#include <time.h>

/* ============================================================================
 * Internal Constants
 * ============================================================================ */

#define MAX_SUBSCRIBERS         32
#define DEFAULT_RING_SIZE       65536
#define DEFAULT_MAX_LOG_SIZE_MB 256

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

/**
 * @brief Deep-copied event stored in the ring buffer.
 *
 * The ring buffer owns the heap allocations pointed to by @c vector_data
 * and @c metadata_json.  They are freed when the slot is overwritten.
 */
typedef struct {
    uint64_t sequence_number;
    GV_CDCEventType type;
    size_t vector_index;
    uint64_t timestamp;
    float *vector_data;             /* owned; NULL for deletes */
    size_t dimension;
    char *metadata_json;            /* owned; NULL if none */
    int valid;                      /* non-zero when slot contains an event */
} CDCRingEntry;

typedef struct {
    GV_CDCCallback callback;
    void *user_data;
    uint32_t event_mask;
    int in_use;
} CDCSubscriber;

struct GV_CDCStream {
    /* Ring buffer */
    CDCRingEntry *ring;
    size_t ring_size;
    size_t head;                    /* next write position */
    uint64_t next_sequence;         /* monotonically increasing counter */

    /* Subscribers */
    CDCSubscriber subscribers[MAX_SUBSCRIBERS];

    /* Persistence */
    int persist_to_file;
    char *log_path;
    size_t max_log_size_bytes;
    FILE *log_fp;
    size_t log_bytes_written;
    int include_vector_data;

    /* Synchronization */
    pthread_mutex_t mutex;
};

/* ============================================================================
 * Forward Declarations
 * ============================================================================ */

static void  free_ring_entry(CDCRingEntry *entry);
static int   deep_copy_event(CDCRingEntry *dst, const GV_CDCEvent *src,
                             uint64_t seq, int include_vector_data);
static void  persist_event(GV_CDCStream *stream, const CDCRingEntry *entry);

/* ============================================================================
 * Configuration
 * ============================================================================ */

static const GV_CDCConfig DEFAULT_CONFIG = {
    .ring_buffer_size    = DEFAULT_RING_SIZE,
    .persist_to_file     = 0,
    .log_path            = NULL,
    .max_log_size_mb     = DEFAULT_MAX_LOG_SIZE_MB,
    .include_vector_data = 1
};

void gv_cdc_config_init(GV_CDCConfig *config) {
    if (!config) return;
    *config = DEFAULT_CONFIG;
}

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

GV_CDCStream *gv_cdc_create(const GV_CDCConfig *config) {
    GV_CDCConfig cfg = config ? *config : DEFAULT_CONFIG;

    if (cfg.ring_buffer_size == 0) cfg.ring_buffer_size = DEFAULT_RING_SIZE;
    if (cfg.max_log_size_mb == 0)  cfg.max_log_size_mb  = DEFAULT_MAX_LOG_SIZE_MB;

    GV_CDCStream *stream = calloc(1, sizeof(GV_CDCStream));
    if (!stream) return NULL;

    stream->ring = calloc(cfg.ring_buffer_size, sizeof(CDCRingEntry));
    if (!stream->ring) {
        free(stream);
        return NULL;
    }

    stream->ring_size           = cfg.ring_buffer_size;
    stream->head                = 0;
    stream->next_sequence       = 1; /* sequences start at 1 */
    stream->persist_to_file     = cfg.persist_to_file;
    stream->include_vector_data = cfg.include_vector_data;
    stream->max_log_size_bytes  = cfg.max_log_size_mb * 1024ULL * 1024ULL;
    stream->log_fp              = NULL;
    stream->log_bytes_written   = 0;

    if (cfg.log_path) {
        stream->log_path = strdup(cfg.log_path);
        if (!stream->log_path) {
            free(stream->ring);
            free(stream);
            return NULL;
        }
    }

    if (stream->persist_to_file && stream->log_path) {
        stream->log_fp = fopen(stream->log_path, "ab");
        /* Failure to open is non-fatal; persistence is best-effort. */
    }

    if (pthread_mutex_init(&stream->mutex, NULL) != 0) {
        if (stream->log_fp) fclose(stream->log_fp);
        free(stream->log_path);
        free(stream->ring);
        free(stream);
        return NULL;
    }

    return stream;
}

void gv_cdc_destroy(GV_CDCStream *stream) {
    if (!stream) return;

    /* Free all ring buffer entries */
    for (size_t i = 0; i < stream->ring_size; i++) {
        free_ring_entry(&stream->ring[i]);
    }
    free(stream->ring);

    if (stream->log_fp) {
        fflush(stream->log_fp);
        fclose(stream->log_fp);
    }
    free(stream->log_path);

    pthread_mutex_destroy(&stream->mutex);
    free(stream);
}

/* ============================================================================
 * Publishing
 * ============================================================================ */

int gv_cdc_publish(GV_CDCStream *stream, const GV_CDCEvent *event) {
    if (!stream || !event) return -1;

    pthread_mutex_lock(&stream->mutex);

    uint64_t seq = stream->next_sequence++;
    size_t slot = stream->head;

    /* Free the old entry if it is being overwritten */
    free_ring_entry(&stream->ring[slot]);

    /* Deep-copy the event into the ring buffer */
    if (deep_copy_event(&stream->ring[slot], event, seq,
                        stream->include_vector_data) != 0) {
        pthread_mutex_unlock(&stream->mutex);
        return -1;
    }

    /* Advance the head (circular) */
    stream->head = (stream->head + 1) % stream->ring_size;

    /* Persist to file if configured */
    if (stream->persist_to_file && stream->log_fp) {
        persist_event(stream, &stream->ring[slot]);
    }

    /*
     * Snapshot the subscriber list while still under the lock so that we
     * have a consistent view.  We invoke callbacks *outside* the lock to
     * prevent deadlocks in user code that may call back into the CDC API.
     */
    CDCSubscriber subs_copy[MAX_SUBSCRIBERS];
    memcpy(subs_copy, stream->subscribers, sizeof(subs_copy));

    /* Build a const event view pointing into the ring entry for callbacks */
    GV_CDCEvent cb_event;
    cb_event.sequence_number = stream->ring[slot].sequence_number;
    cb_event.type            = stream->ring[slot].type;
    cb_event.vector_index    = stream->ring[slot].vector_index;
    cb_event.timestamp       = stream->ring[slot].timestamp;
    cb_event.vector_data     = stream->ring[slot].vector_data;
    cb_event.dimension       = stream->ring[slot].dimension;
    cb_event.metadata_json   = stream->ring[slot].metadata_json;

    pthread_mutex_unlock(&stream->mutex);

    /* Invoke matching subscriber callbacks outside the lock */
    for (int i = 0; i < MAX_SUBSCRIBERS; i++) {
        if (!subs_copy[i].in_use) continue;
        if (!(subs_copy[i].event_mask & (uint32_t)event->type)) continue;

        subs_copy[i].callback(&cb_event, subs_copy[i].user_data);
    }

    return 0;
}

/* ============================================================================
 * Subscription (push interface)
 * ============================================================================ */

int gv_cdc_subscribe(GV_CDCStream *stream, uint32_t event_mask,
                     GV_CDCCallback callback, void *user_data) {
    if (!stream || !callback) return -1;

    pthread_mutex_lock(&stream->mutex);

    int slot = -1;
    for (int i = 0; i < MAX_SUBSCRIBERS; i++) {
        if (!stream->subscribers[i].in_use) {
            slot = i;
            break;
        }
    }

    if (slot < 0) {
        pthread_mutex_unlock(&stream->mutex);
        return -1; /* no free slots */
    }

    CDCSubscriber *sub = &stream->subscribers[slot];
    sub->callback   = callback;
    sub->user_data  = user_data;
    sub->event_mask = event_mask ? event_mask : (uint32_t)GV_CDC_ALL;
    sub->in_use     = 1;

    pthread_mutex_unlock(&stream->mutex);
    return slot; /* subscriber_id */
}

int gv_cdc_unsubscribe(GV_CDCStream *stream, int subscriber_id) {
    if (!stream || subscriber_id < 0 || subscriber_id >= MAX_SUBSCRIBERS)
        return -1;

    pthread_mutex_lock(&stream->mutex);

    if (!stream->subscribers[subscriber_id].in_use) {
        pthread_mutex_unlock(&stream->mutex);
        return -1;
    }

    memset(&stream->subscribers[subscriber_id], 0, sizeof(CDCSubscriber));

    pthread_mutex_unlock(&stream->mutex);
    return 0;
}

/* ============================================================================
 * Polling (pull interface)
 * ============================================================================ */

int gv_cdc_poll(GV_CDCStream *stream, GV_CDCCursor *cursor,
                GV_CDCEvent *events, size_t max_events) {
    if (!stream || !cursor || !events || max_events == 0) return -1;

    pthread_mutex_lock(&stream->mutex);

    int count = 0;

    /*
     * Walk the ring buffer looking for entries whose sequence number is
     * >= cursor->sequence_number.  Because the buffer is circular and
     * entries are written in order, we scan all slots and collect those
     * in range.  We cap the result at max_events.
     *
     * The oldest reachable sequence in the ring is:
     *   next_sequence - min(next_sequence - 1, ring_size)
     * If the cursor is behind that, we clamp it forward.
     */
    uint64_t newest_seq = stream->next_sequence - 1; /* last written */
    if (newest_seq == 0) {
        /* Nothing published yet */
        pthread_mutex_unlock(&stream->mutex);
        return 0;
    }

    uint64_t total_written = stream->next_sequence - 1;
    uint64_t available = total_written < stream->ring_size
                         ? total_written
                         : stream->ring_size;
    uint64_t oldest_seq = total_written - available + 1;

    /* Clamp cursor to oldest available */
    if (cursor->sequence_number < oldest_seq) {
        cursor->sequence_number = oldest_seq;
    }

    /* Iterate from cursor forward */
    for (uint64_t seq = cursor->sequence_number;
         seq <= newest_seq && (size_t)count < max_events;
         seq++) {
        /*
         * Map sequence number to ring index.  Sequences start at 1 and
         * advance with each publish; the ring slot for a given sequence
         * is (seq - 1) % ring_size.
         */
        size_t idx = (size_t)((seq - 1) % stream->ring_size);
        CDCRingEntry *entry = &stream->ring[idx];

        if (!entry->valid || entry->sequence_number != seq) continue;

        GV_CDCEvent *out = &events[count];
        out->sequence_number = entry->sequence_number;
        out->type            = entry->type;
        out->vector_index    = entry->vector_index;
        out->timestamp       = entry->timestamp;
        out->vector_data     = entry->vector_data;
        out->dimension       = entry->dimension;
        out->metadata_json   = entry->metadata_json;
        count++;
    }

    /* Advance cursor past the last returned event */
    if (count > 0) {
        cursor->sequence_number = events[count - 1].sequence_number + 1;
    }

    pthread_mutex_unlock(&stream->mutex);
    return count;
}

GV_CDCCursor gv_cdc_get_cursor(const GV_CDCStream *stream) {
    GV_CDCCursor cursor = { .sequence_number = 0 };
    if (!stream) return cursor;

    GV_CDCStream *s = (GV_CDCStream *)stream;
    pthread_mutex_lock(&s->mutex);
    cursor.sequence_number = s->next_sequence;
    pthread_mutex_unlock(&s->mutex);

    return cursor;
}

GV_CDCCursor gv_cdc_cursor_from_sequence(uint64_t seq) {
    GV_CDCCursor cursor = { .sequence_number = seq };
    return cursor;
}

size_t gv_cdc_pending_count(const GV_CDCStream *stream, const GV_CDCCursor *cursor) {
    if (!stream || !cursor) return 0;

    GV_CDCStream *s = (GV_CDCStream *)stream;
    pthread_mutex_lock(&s->mutex);

    uint64_t next = s->next_sequence;
    if (next <= 1 || cursor->sequence_number >= next) {
        pthread_mutex_unlock(&s->mutex);
        return 0;
    }

    uint64_t total_written = next - 1;
    uint64_t available = total_written < s->ring_size
                         ? total_written
                         : s->ring_size;
    uint64_t oldest_seq = total_written - available + 1;

    uint64_t effective_start = cursor->sequence_number < oldest_seq
                               ? oldest_seq
                               : cursor->sequence_number;

    size_t pending = (size_t)(next - effective_start);

    pthread_mutex_unlock(&s->mutex);
    return pending;
}

/* ============================================================================
 * Ring Buffer Helpers
 * ============================================================================ */

static void free_ring_entry(CDCRingEntry *entry) {
    if (!entry || !entry->valid) return;

    free(entry->vector_data);
    free(entry->metadata_json);
    memset(entry, 0, sizeof(CDCRingEntry));
}

static int deep_copy_event(CDCRingEntry *dst, const GV_CDCEvent *src,
                           uint64_t seq, int include_vector_data) {
    if (!dst || !src) return -1;

    dst->sequence_number = seq;
    dst->type            = src->type;
    dst->vector_index    = src->vector_index;
    dst->timestamp       = src->timestamp;
    dst->dimension       = src->dimension;
    dst->vector_data     = NULL;
    dst->metadata_json   = NULL;
    dst->valid           = 1;

    /* Deep-copy vector data */
    if (include_vector_data && src->vector_data && src->dimension > 0) {
        size_t nbytes = src->dimension * sizeof(float);
        dst->vector_data = malloc(nbytes);
        if (!dst->vector_data) {
            dst->valid = 0;
            return -1;
        }
        memcpy(dst->vector_data, src->vector_data, nbytes);
    }

    /* Deep-copy metadata JSON */
    if (src->metadata_json) {
        dst->metadata_json = strdup(src->metadata_json);
        if (!dst->metadata_json) {
            free(dst->vector_data);
            dst->vector_data = NULL;
            dst->valid = 0;
            return -1;
        }
    }

    return 0;
}

/* ============================================================================
 * File Persistence
 * ============================================================================ */

/**
 * Binary record format (all fields little-endian / native):
 *   [seq_number  : 8 bytes]
 *   [type        : 4 bytes]
 *   [index       : 8 bytes]
 *   [timestamp   : 8 bytes]
 *   [dim         : 8 bytes]
 *   [vector_data : dim * 4 bytes]  (omitted when dim == 0)
 *   [metadata_len: 4 bytes]
 *   [metadata    : metadata_len bytes]
 */
static void persist_event(GV_CDCStream *stream, const CDCRingEntry *entry) {
    if (!stream->log_fp || !entry || !entry->valid) return;

    /* Respect maximum log size */
    if (stream->log_bytes_written >= stream->max_log_size_bytes) return;

    FILE *fp = stream->log_fp;
    size_t written = 0;

    /* seq_number: 8 */
    uint64_t seq = entry->sequence_number;
    written += fwrite(&seq, 1, 8, fp);

    /* type: 4 */
    uint32_t type = (uint32_t)entry->type;
    written += fwrite(&type, 1, 4, fp);

    /* index: 8 */
    uint64_t idx = (uint64_t)entry->vector_index;
    written += fwrite(&idx, 1, 8, fp);

    /* timestamp: 8 */
    uint64_t ts = entry->timestamp;
    written += fwrite(&ts, 1, 8, fp);

    /* dim: 8 */
    uint64_t dim = (uint64_t)entry->dimension;
    written += fwrite(&dim, 1, 8, fp);

    /* vector_data: dim * 4 */
    if (entry->vector_data && entry->dimension > 0) {
        written += fwrite(entry->vector_data, sizeof(float),
                          entry->dimension, fp) * sizeof(float);
    }

    /* metadata_len: 4, metadata: N */
    uint32_t meta_len = 0;
    if (entry->metadata_json) {
        meta_len = (uint32_t)strlen(entry->metadata_json);
    }
    written += fwrite(&meta_len, 1, 4, fp);

    if (meta_len > 0) {
        written += fwrite(entry->metadata_json, 1, meta_len, fp);
    }

    fflush(fp);
    stream->log_bytes_written += written;
}
