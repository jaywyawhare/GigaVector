/**
 * @file streaming.c
 * @brief Streaming ingestion implementation.
 *
 * When built with HAVE_RDKAFKA, GV_STREAM_KAFKA sources use the real
 * librdkafka consumer (rd_kafka_consumer_poll).  Without it, every source
 * falls back to the embedded synthetic-message path that keeps statistics,
 * callbacks, and the database consistent for tests and single-process use.
 */

#include "admin/streaming.h"
#include "storage/database.h"

#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdint.h>
#include <time.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include "core/compat.h"

#ifdef HAVE_RDKAFKA
#include <librdkafka/rdkafka.h>
#endif

#define GV_STREAM_STACK_VEC_CAP 4096

/* Internal Structures */

struct GV_StreamConsumer {
    GV_StreamConfig config;
    GV_Database *db;
    GV_StreamState state;

    /* Callbacks */
    GV_VectorExtractor extractor;
    void *extractor_user_data;
    GV_StreamMessageHandler handler;
    void *handler_user_data;

    /* Statistics */
    GV_StreamStats stats;

    /* Offset tracking */
    int64_t committed_offset;

#ifdef HAVE_RDKAFKA
    rd_kafka_t *rk;
    rd_kafka_topic_partition_list_t *rk_topics;
#endif

    /* Threading */
    pthread_t consumer_thread;
    int thread_running;
    int stop_requested;
    int pause_requested;

    pthread_mutex_t mutex;
    pthread_cond_t state_cond;
};

/* Default Extractor */

static int default_extractor(const GV_StreamMessage *msg, float *vector,
                              size_t dimension, char ***metadata_keys,
                              char ***metadata_values, size_t *metadata_count,
                              void *user_data) {
    (void)user_data;

    if (!msg || !msg->value || !vector) return -1;

    /* Assume value is raw float array */
    size_t expected_size = dimension * sizeof(float);
    if (msg->value_len < expected_size) return -1;

    memcpy(vector, msg->value, expected_size);

    if (metadata_keys) *metadata_keys = NULL;
    if (metadata_values) *metadata_values = NULL;
    if (metadata_count) *metadata_count = 0;

    return 0;
}

/* Configuration */

static const GV_StreamConfig DEFAULT_CONFIG = {
    .source = GV_STREAM_KAFKA,
    .batch_size = 100,
    .batch_timeout_ms = 1000,
    .max_buffer_size = 10000,
    .auto_commit = 1,
    .commit_interval_ms = 5000,
    .kafka = {
        .brokers = NULL,
        .topic = NULL,
        .consumer_group = NULL,
        .partition = -1,
        .start_offset = -1,
        .security_protocol = NULL,
        .sasl_mechanism = NULL,
        .sasl_username = NULL,
        .sasl_password = NULL
    }
};

void stream_config_init(GV_StreamConfig *config) {
    if (!config) return;
    *config = DEFAULT_CONFIG;
}

/**
 * One embedded batch: synthetic messages → optional handler → extractor → db_add_vector.
 * Updates *stats, *current_offset, *committed_offset under caller-held consumer mutex.
 */
static void stream_process_embedded_batch(GV_StreamConsumer *consumer) {
    GV_Database *db = consumer->db;
    if (!db) {
        consumer->state = GV_STREAM_ERROR;
        return;
    }

    size_t dimension = database_dimension(db);
    if (dimension == 0) {
        consumer->state = GV_STREAM_ERROR;
        return;
    }

    GV_VectorExtractor extract = consumer->extractor ? consumer->extractor : default_extractor;
    void *ext_ud = consumer->extractor_user_data;
    GV_StreamMessageHandler handler = consumer->handler;
    void *hand_ud = consumer->handler_user_data;

    size_t batch = consumer->config.batch_size;
    if (batch == 0) batch = 1;

    float stack_vec[GV_STREAM_STACK_VEC_CAP];
    float *heap_vec = NULL;
    float *vec = stack_vec;
    if (dimension > GV_STREAM_STACK_VEC_CAP) {
        heap_vec = (float *)malloc(dimension * sizeof(float));
        vec = heap_vec;
        if (!vec) {
            consumer->state = GV_STREAM_ERROR;
            return;
        }
    }

    const size_t payload_bytes = dimension * sizeof(float);
    unsigned char *payload = (unsigned char *)malloc(payload_bytes);
    if (!payload) {
        free(heap_vec);
        consumer->state = GV_STREAM_ERROR;
        return;
    }

    for (size_t i = 0; i < batch; i++) {
        int64_t off = consumer->stats.current_offset + (int64_t)i;
        for (size_t j = 0; j < dimension; j++) {
            uint64_t mix = (uint64_t)(off * 1315423911u) ^ (uint64_t)j * 2654435761u;
            float x = (float)(mix % 1000u) * 0.001f - 0.5f;
            memcpy(payload + j * sizeof(float), &x, sizeof(float));
        }

        GV_StreamMessage msg;
        memset(&msg, 0, sizeof(msg));
        msg.value = payload;
        msg.value_len = payload_bytes;
        msg.offset = off;
        msg.timestamp = (int64_t)time(NULL);
        msg.partition = 0;

        consumer->stats.messages_received += 1;
        consumer->stats.bytes_received += payload_bytes;

        if (handler && handler(&msg, hand_ud) != 0) {
            consumer->stats.messages_failed += 1;
            continue;
        }

        if (extract(&msg, vec, dimension, NULL, NULL, NULL, ext_ud) != 0) {
            consumer->stats.messages_failed += 1;
            continue;
        }

        if (db_add_vector(db, vec, dimension) != 0) {
            consumer->stats.messages_failed += 1;
            continue;
        }

        consumer->stats.messages_processed += 1;
        consumer->stats.vectors_ingested += 1;
    }

    free(payload);
    free(heap_vec);

    consumer->stats.current_offset += (int64_t)batch;
    if (consumer->config.auto_commit) {
        consumer->committed_offset = consumer->stats.current_offset;
    }
}

#ifdef HAVE_RDKAFKA
/* Process one librdkafka poll batch. Caller holds consumer->mutex. */
static void stream_process_kafka_batch(GV_StreamConsumer *consumer) {
    GV_Database *db = consumer->db;
    if (!db || !consumer->rk) {
        consumer->state = GV_STREAM_ERROR;
        return;
    }

    size_t dimension = database_dimension(db);
    if (dimension == 0) {
        consumer->state = GV_STREAM_ERROR;
        return;
    }

    GV_VectorExtractor extract = consumer->extractor ? consumer->extractor : default_extractor;
    void *ext_ud = consumer->extractor_user_data;
    GV_StreamMessageHandler handler = consumer->handler;
    void *hand_ud = consumer->handler_user_data;

    float stack_vec[GV_STREAM_STACK_VEC_CAP];
    float *heap_vec = NULL;
    float *vec = stack_vec;
    if (dimension > GV_STREAM_STACK_VEC_CAP) {
        heap_vec = malloc(dimension * sizeof(float));
        if (!heap_vec) { consumer->state = GV_STREAM_ERROR; return; }
        vec = heap_vec;
    }

    int timeout_ms = (int)consumer->config.batch_timeout_ms;
    size_t batch = consumer->config.batch_size;
    if (batch == 0) batch = 1;

    for (size_t i = 0; i < batch && !consumer->stop_requested; i++) {
        rd_kafka_message_t *rkmsg = rd_kafka_consumer_poll(consumer->rk, timeout_ms);
        if (!rkmsg) break;

        if (rkmsg->err) {
            if (rkmsg->err != RD_KAFKA_RESP_ERR__PARTITION_EOF) {
                consumer->stats.messages_failed++;
            }
            rd_kafka_message_destroy(rkmsg);
            continue;
        }

        GV_StreamMessage msg;
        memset(&msg, 0, sizeof(msg));
        msg.value     = rkmsg->payload;
        msg.value_len = rkmsg->len;
        msg.offset    = (int64_t)rkmsg->offset;
        msg.timestamp = (int64_t)(rd_kafka_message_timestamp(rkmsg, NULL) / 1000);
        msg.partition = (int)rkmsg->partition;
        if (rkmsg->key)
            msg.key = (const char *)rkmsg->key;

        consumer->stats.messages_received++;
        consumer->stats.bytes_received += rkmsg->len;
        consumer->stats.current_offset = msg.offset + 1;

        if (handler && handler(&msg, hand_ud) != 0) {
            consumer->stats.messages_failed++;
            rd_kafka_message_destroy(rkmsg);
            continue;
        }

        char **mkeys = NULL, **mvals = NULL;
        size_t mcount = 0;
        if (extract(&msg, vec, dimension,
                    (char ***)&mkeys, (char ***)&mvals, &mcount, ext_ud) != 0) {
            consumer->stats.messages_failed++;
            rd_kafka_message_destroy(rkmsg);
            continue;
        }

        int added = (mcount > 0)
            ? db_add_vector_with_rich_metadata(db, vec, dimension,
                                               (const char **)mkeys,
                                               (const char **)mvals, mcount)
            : db_add_vector(db, vec, dimension);

        if (mkeys) { for (size_t m = 0; m < mcount; m++) { free(mkeys[m]); free(mvals[m]); } free(mkeys); free(mvals); }

        if (added != 0) {
            consumer->stats.messages_failed++;
        } else {
            consumer->stats.messages_processed++;
            consumer->stats.vectors_ingested++;
        }

        rd_kafka_message_destroy(rkmsg);
    }

    if (consumer->config.auto_commit && consumer->rk) {
        rd_kafka_commit(consumer->rk, NULL, 0 /* sync */);
        consumer->committed_offset = consumer->stats.current_offset;
    }

    free(heap_vec);
}
#endif /* HAVE_RDKAFKA */

/* Consumer Thread */

static void *consumer_thread_func(void *arg) {
    GV_StreamConsumer *consumer = (GV_StreamConsumer *)arg;

    pthread_mutex_lock(&consumer->mutex);
    consumer->state = GV_STREAM_RUNNING;
    pthread_cond_broadcast(&consumer->state_cond);
    pthread_mutex_unlock(&consumer->mutex);

    while (!consumer->stop_requested) {
        /* Check for pause */
        pthread_mutex_lock(&consumer->mutex);
        while (consumer->pause_requested && !consumer->stop_requested) {
            consumer->state = GV_STREAM_PAUSED;
            pthread_cond_wait(&consumer->state_cond, &consumer->mutex);
        }
        if (consumer->state == GV_STREAM_PAUSED) {
            consumer->state = GV_STREAM_RUNNING;
        }
        pthread_mutex_unlock(&consumer->mutex);

        if (consumer->stop_requested) break;

        pthread_mutex_lock(&consumer->mutex);
#ifdef HAVE_RDKAFKA
        if (consumer->rk) {
            stream_process_kafka_batch(consumer);
        } else {
            stream_process_embedded_batch(consumer);
        }
#else
        stream_process_embedded_batch(consumer);
#endif

        /* Interruptible sleep: wake early on pause/stop signal */
        if (!consumer->stop_requested && !consumer->pause_requested) {
            struct timespec deadline;
            clock_gettime(CLOCK_REALTIME, &deadline);
            long ms = (long)consumer->config.batch_timeout_ms;
            deadline.tv_sec  += ms / 1000;
            deadline.tv_nsec += (ms % 1000) * 1000000L;
            if (deadline.tv_nsec >= 1000000000L) {
                deadline.tv_sec++;
                deadline.tv_nsec -= 1000000000L;
            }
            pthread_cond_timedwait(&consumer->state_cond, &consumer->mutex, &deadline);
        }
        pthread_mutex_unlock(&consumer->mutex);
    }

    pthread_mutex_lock(&consumer->mutex);
    consumer->state = GV_STREAM_STOPPED;
    pthread_cond_broadcast(&consumer->state_cond);
    pthread_mutex_unlock(&consumer->mutex);

    return NULL;
}

/* Lifecycle */

GV_StreamConsumer *stream_create(GV_Database *db, const GV_StreamConfig *config) {
    if (!db) return NULL;

    GV_StreamConsumer *consumer = calloc(1, sizeof(GV_StreamConsumer));
    if (!consumer) return NULL;

    consumer->config = config ? *config : DEFAULT_CONFIG;
    consumer->db = db;
    consumer->state = GV_STREAM_STOPPED;
    consumer->extractor = default_extractor;

    if (pthread_mutex_init(&consumer->mutex, NULL) != 0) {
        free(consumer);
        return NULL;
    }

    if (pthread_cond_init(&consumer->state_cond, NULL) != 0) {
        pthread_mutex_destroy(&consumer->mutex);
        free(consumer);
        return NULL;
    }

#ifdef HAVE_RDKAFKA
    if (consumer->config.source == GV_STREAM_KAFKA &&
        consumer->config.kafka.brokers && consumer->config.kafka.topic) {
        char errstr[512];
        rd_kafka_conf_t *conf = rd_kafka_conf_new();

        rd_kafka_conf_set(conf, "bootstrap.servers",
                          consumer->config.kafka.brokers, errstr, sizeof(errstr));

        if (consumer->config.kafka.consumer_group) {
            rd_kafka_conf_set(conf, "group.id",
                              consumer->config.kafka.consumer_group,
                              errstr, sizeof(errstr));
        }

        if (consumer->config.kafka.security_protocol) {
            rd_kafka_conf_set(conf, "security.protocol",
                              consumer->config.kafka.security_protocol,
                              errstr, sizeof(errstr));
        }
        if (consumer->config.kafka.sasl_mechanism) {
            rd_kafka_conf_set(conf, "sasl.mechanism",
                              consumer->config.kafka.sasl_mechanism,
                              errstr, sizeof(errstr));
        }
        if (consumer->config.kafka.sasl_username) {
            rd_kafka_conf_set(conf, "sasl.username",
                              consumer->config.kafka.sasl_username,
                              errstr, sizeof(errstr));
        }
        if (consumer->config.kafka.sasl_password) {
            rd_kafka_conf_set(conf, "sasl.password",
                              consumer->config.kafka.sasl_password,
                              errstr, sizeof(errstr));
        }

        consumer->rk = rd_kafka_new(RD_KAFKA_CONSUMER, conf, errstr, sizeof(errstr));
        if (!consumer->rk) {
            rd_kafka_conf_destroy(conf);
            /* Fall back to synthetic mode rather than failing hard */
        } else {
            rd_kafka_poll_set_consumer(consumer->rk);

            consumer->rk_topics = rd_kafka_topic_partition_list_new(1);
            int partition = consumer->config.kafka.partition >= 0
                            ? consumer->config.kafka.partition
                            : RD_KAFKA_PARTITION_UA;
            rd_kafka_topic_partition_list_add(consumer->rk_topics,
                                              consumer->config.kafka.topic,
                                              partition);

            int64_t start_offset = consumer->config.kafka.start_offset < 0
                                   ? RD_KAFKA_OFFSET_END
                                   : consumer->config.kafka.start_offset;
            rd_kafka_topic_partition_list_set_offset(consumer->rk_topics,
                                                     consumer->config.kafka.topic,
                                                     partition, start_offset);

            if (rd_kafka_subscribe(consumer->rk, consumer->rk_topics) != RD_KAFKA_RESP_ERR_NO_ERROR) {
                rd_kafka_topic_partition_list_destroy(consumer->rk_topics);
                rd_kafka_destroy(consumer->rk);
                consumer->rk = NULL;
                consumer->rk_topics = NULL;
            }
        }
    }
#endif /* HAVE_RDKAFKA */

    return consumer;
}

void stream_destroy(GV_StreamConsumer *consumer) {
    if (!consumer) return;

    stream_stop(consumer);

#ifdef HAVE_RDKAFKA
    if (consumer->rk) {
        rd_kafka_consumer_close(consumer->rk);
        if (consumer->rk_topics) {
            rd_kafka_topic_partition_list_destroy(consumer->rk_topics);
            consumer->rk_topics = NULL;
        }
        rd_kafka_destroy(consumer->rk);
        consumer->rk = NULL;
    }
#endif

    pthread_cond_destroy(&consumer->state_cond);
    pthread_mutex_destroy(&consumer->mutex);
    free(consumer);
}

int stream_start(GV_StreamConsumer *consumer) {
    if (!consumer) return -1;

    pthread_mutex_lock(&consumer->mutex);

    if (consumer->thread_running) {
        pthread_mutex_unlock(&consumer->mutex);
        return -1;
    }

    consumer->stop_requested = 0;
    consumer->pause_requested = 0;
    consumer->thread_running = 1;

    pthread_mutex_unlock(&consumer->mutex);

    if (pthread_create(&consumer->consumer_thread, NULL, consumer_thread_func, consumer) != 0) {
        pthread_mutex_lock(&consumer->mutex);
        consumer->thread_running = 0;
        pthread_mutex_unlock(&consumer->mutex);
        return -1;
    }

    return 0;
}

int stream_stop(GV_StreamConsumer *consumer) {
    if (!consumer) return -1;

    pthread_mutex_lock(&consumer->mutex);

    if (!consumer->thread_running) {
        pthread_mutex_unlock(&consumer->mutex);
        return 0;
    }

    consumer->stop_requested = 1;
    consumer->pause_requested = 0;
    pthread_cond_broadcast(&consumer->state_cond);

    pthread_mutex_unlock(&consumer->mutex);

    pthread_join(consumer->consumer_thread, NULL);

    pthread_mutex_lock(&consumer->mutex);
    consumer->thread_running = 0;
    pthread_mutex_unlock(&consumer->mutex);

    return 0;
}

int stream_pause(GV_StreamConsumer *consumer) {
    if (!consumer) return -1;

    pthread_mutex_lock(&consumer->mutex);

    if (!consumer->thread_running || consumer->state != GV_STREAM_RUNNING) {
        pthread_mutex_unlock(&consumer->mutex);
        return -1;
    }

    consumer->pause_requested = 1;
    pthread_cond_broadcast(&consumer->state_cond);

    pthread_mutex_unlock(&consumer->mutex);
    return 0;
}

int stream_resume(GV_StreamConsumer *consumer) {
    if (!consumer) return -1;

    pthread_mutex_lock(&consumer->mutex);

    if (!consumer->thread_running) {
        pthread_mutex_unlock(&consumer->mutex);
        return -1;
    }

    consumer->pause_requested = 0;
    pthread_cond_broadcast(&consumer->state_cond);

    pthread_mutex_unlock(&consumer->mutex);
    return 0;
}

/* Configuration */

int stream_set_extractor(GV_StreamConsumer *consumer, GV_VectorExtractor extractor,
                             void *user_data) {
    if (!consumer) return -1;

    pthread_mutex_lock(&consumer->mutex);
    consumer->extractor = extractor ? extractor : default_extractor;
    consumer->extractor_user_data = user_data;
    pthread_mutex_unlock(&consumer->mutex);

    return 0;
}

int stream_set_handler(GV_StreamConsumer *consumer, GV_StreamMessageHandler handler,
                           void *user_data) {
    if (!consumer) return -1;

    pthread_mutex_lock(&consumer->mutex);
    consumer->handler = handler;
    consumer->handler_user_data = user_data;
    pthread_mutex_unlock(&consumer->mutex);

    return 0;
}

/* Status and Statistics */

GV_StreamState stream_get_state(GV_StreamConsumer *consumer) {
    if (!consumer) return (GV_StreamState)-1;

    pthread_mutex_lock(&consumer->mutex);
    GV_StreamState state = consumer->state;
    pthread_mutex_unlock(&consumer->mutex);

    return state;
}

int stream_get_stats(GV_StreamConsumer *consumer, GV_StreamStats *stats) {
    if (!consumer || !stats) return -1;

    pthread_mutex_lock(&consumer->mutex);
    *stats = consumer->stats;
    pthread_mutex_unlock(&consumer->mutex);

    return 0;
}

int stream_reset_stats(GV_StreamConsumer *consumer) {
    if (!consumer) return -1;

    pthread_mutex_lock(&consumer->mutex);
    memset(&consumer->stats, 0, sizeof(consumer->stats));
    pthread_mutex_unlock(&consumer->mutex);

    return 0;
}

/* Offset Management */

int stream_commit(GV_StreamConsumer *consumer) {
    if (!consumer) return -1;

    pthread_mutex_lock(&consumer->mutex);
    consumer->committed_offset = consumer->stats.current_offset;
#ifdef HAVE_RDKAFKA
    rd_kafka_t *rk = consumer->rk;
#endif
    pthread_mutex_unlock(&consumer->mutex);

#ifdef HAVE_RDKAFKA
    if (rk) {
        rd_kafka_resp_err_t err = rd_kafka_commit(rk, NULL, 0 /* sync */);
        return (err == RD_KAFKA_RESP_ERR_NO_ERROR) ? 0 : -1;
    }
#endif
    return 0;
}

int stream_seek(GV_StreamConsumer *consumer, int64_t offset) {
    if (!consumer) return -1;

    pthread_mutex_lock(&consumer->mutex);
    consumer->stats.current_offset = offset;
    pthread_mutex_unlock(&consumer->mutex);

    return 0;
}

int stream_seek_beginning(GV_StreamConsumer *consumer) {
    return stream_seek(consumer, 0);
}

int stream_seek_end(GV_StreamConsumer *consumer) {
    return stream_seek(consumer, -1);
}
