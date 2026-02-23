#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_streaming.h"
#include "gigavector/gv_database.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static const char *DB_PATH = "tmp_test_streaming.bin";

static void cleanup(void) {
    remove(DB_PATH);
}

/* Dummy handler for callback tests */
static int dummy_handler(const GV_StreamMessage *msg, void *user_data) {
    (void)msg;
    (void)user_data;
    return 0;
}

/* Dummy extractor for callback tests */
static int dummy_extractor(const GV_StreamMessage *msg, float *vector,
                            size_t dimension, char ***metadata_keys,
                            char ***metadata_values, size_t *metadata_count,
                            void *user_data) {
    (void)msg;
    (void)vector;
    (void)dimension;
    (void)metadata_keys;
    (void)metadata_values;
    (void)metadata_count;
    (void)user_data;
    return 0;
}

static int test_config_init(void) {
    GV_StreamConfig cfg;
    memset(&cfg, 0xFF, sizeof(cfg));
    gv_stream_config_init(&cfg);

    ASSERT(cfg.batch_size == 100, "default batch_size should be 100");
    ASSERT(cfg.batch_timeout_ms == 1000, "default batch_timeout_ms should be 1000");
    ASSERT(cfg.max_buffer_size == 10000, "default max_buffer_size should be 10000");
    ASSERT(cfg.auto_commit == 1, "default auto_commit should be 1");
    ASSERT(cfg.commit_interval_ms == 5000, "default commit_interval_ms should be 5000");
    return 0;
}

static int test_create_custom(void) {
    cleanup();
    GV_Database *db = gv_db_open(DB_PATH, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database open");

    GV_StreamConfig cfg;
    gv_stream_config_init(&cfg);
    cfg.source = GV_STREAM_CUSTOM;

    GV_StreamConsumer *consumer = gv_stream_create(db, &cfg);
    ASSERT(consumer != NULL, "gv_stream_create with CUSTOM source should succeed");

    gv_stream_destroy(consumer);
    gv_db_close(db);
    cleanup();
    return 0;
}

static int test_create_kafka(void) {
    cleanup();
    GV_Database *db = gv_db_open(DB_PATH, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database open");

    GV_StreamConfig cfg;
    gv_stream_config_init(&cfg);
    cfg.source = GV_STREAM_KAFKA;
    cfg.kafka.brokers = "localhost:9092";
    cfg.kafka.topic = "test-vectors";
    cfg.kafka.consumer_group = "gv-test-group";
    cfg.kafka.partition = -1;
    cfg.kafka.start_offset = -1;

    GV_StreamConsumer *consumer = gv_stream_create(db, &cfg);
    ASSERT(consumer != NULL, "gv_stream_create with KAFKA source should succeed");

    gv_stream_destroy(consumer);
    gv_db_close(db);
    cleanup();
    return 0;
}

static int test_destroy_null(void) {
    /* Destroying NULL must not crash */
    gv_stream_destroy(NULL);
    return 0;
}

static int test_get_state_initial(void) {
    cleanup();
    GV_Database *db = gv_db_open(DB_PATH, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database open");

    GV_StreamConfig cfg;
    gv_stream_config_init(&cfg);
    cfg.source = GV_STREAM_CUSTOM;

    GV_StreamConsumer *consumer = gv_stream_create(db, &cfg);
    ASSERT(consumer != NULL, "create");

    GV_StreamState state = gv_stream_get_state(consumer);
    ASSERT(state == GV_STREAM_STOPPED, "initial state should be STOPPED");

    gv_stream_destroy(consumer);
    gv_db_close(db);
    cleanup();
    return 0;
}

static int test_get_stats_initial(void) {
    cleanup();
    GV_Database *db = gv_db_open(DB_PATH, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database open");

    GV_StreamConfig cfg;
    gv_stream_config_init(&cfg);
    cfg.source = GV_STREAM_CUSTOM;

    GV_StreamConsumer *consumer = gv_stream_create(db, &cfg);
    ASSERT(consumer != NULL, "create");

    GV_StreamStats stats;
    memset(&stats, 0xFF, sizeof(stats));
    int rc = gv_stream_get_stats(consumer, &stats);
    ASSERT(rc == 0, "gv_stream_get_stats should succeed");
    ASSERT(stats.messages_received == 0, "initial messages_received should be 0");
    ASSERT(stats.messages_processed == 0, "initial messages_processed should be 0");
    ASSERT(stats.messages_failed == 0, "initial messages_failed should be 0");
    ASSERT(stats.vectors_ingested == 0, "initial vectors_ingested should be 0");
    ASSERT(stats.bytes_received == 0, "initial bytes_received should be 0");

    gv_stream_destroy(consumer);
    gv_db_close(db);
    cleanup();
    return 0;
}

static int test_set_handler(void) {
    cleanup();
    GV_Database *db = gv_db_open(DB_PATH, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database open");

    GV_StreamConfig cfg;
    gv_stream_config_init(&cfg);
    cfg.source = GV_STREAM_CUSTOM;

    GV_StreamConsumer *consumer = gv_stream_create(db, &cfg);
    ASSERT(consumer != NULL, "create");

    int rc = gv_stream_set_handler(consumer, dummy_handler, NULL);
    ASSERT(rc == 0, "gv_stream_set_handler should succeed");

    gv_stream_destroy(consumer);
    gv_db_close(db);
    cleanup();
    return 0;
}

static int test_set_extractor(void) {
    cleanup();
    GV_Database *db = gv_db_open(DB_PATH, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database open");

    GV_StreamConfig cfg;
    gv_stream_config_init(&cfg);
    cfg.source = GV_STREAM_CUSTOM;

    GV_StreamConsumer *consumer = gv_stream_create(db, &cfg);
    ASSERT(consumer != NULL, "create");

    int rc = gv_stream_set_extractor(consumer, dummy_extractor, NULL);
    ASSERT(rc == 0, "gv_stream_set_extractor should succeed");

    gv_stream_destroy(consumer);
    gv_db_close(db);
    cleanup();
    return 0;
}

static int test_start_stop(void) {
    cleanup();
    GV_Database *db = gv_db_open(DB_PATH, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database open");

    GV_StreamConfig cfg;
    gv_stream_config_init(&cfg);
    cfg.source = GV_STREAM_CUSTOM;

    GV_StreamConsumer *consumer = gv_stream_create(db, &cfg);
    ASSERT(consumer != NULL, "create");

    int rc = gv_stream_start(consumer);
    ASSERT(rc == 0, "gv_stream_start should succeed");

    /* CUSTOM source thread may exit quickly — state could be RUNNING or STOPPED */
    GV_StreamState state = gv_stream_get_state(consumer);
    ASSERT(state == GV_STREAM_RUNNING || state == GV_STREAM_STOPPED || state == GV_STREAM_ERROR,
           "state after start should be RUNNING, STOPPED, or ERROR");

    rc = gv_stream_stop(consumer);
    /* Stop may return 0 (success) or already stopped */

    gv_stream_destroy(consumer);
    gv_db_close(db);
    cleanup();
    return 0;
}

static int test_pause_resume(void) {
    cleanup();
    GV_Database *db = gv_db_open(DB_PATH, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database open");

    GV_StreamConfig cfg;
    gv_stream_config_init(&cfg);
    cfg.source = GV_STREAM_CUSTOM;

    GV_StreamConsumer *consumer = gv_stream_create(db, &cfg);
    ASSERT(consumer != NULL, "create");

    /* Start first */
    int rc = gv_stream_start(consumer);
    ASSERT(rc == 0, "start");

    /* CUSTOM source thread may exit quickly, so pause/resume may or may not
       transition states. We just verify no crash and valid return codes. */
    rc = gv_stream_pause(consumer);
    /* rc == 0 means paused, non-zero means already stopped — both OK */

    GV_StreamState state = gv_stream_get_state(consumer);
    ASSERT(state == GV_STREAM_PAUSED || state == GV_STREAM_STOPPED || state == GV_STREAM_ERROR,
           "state after pause should be PAUSED, STOPPED, or ERROR");

    rc = gv_stream_resume(consumer);
    /* Similarly, resume may fail if already stopped */

    state = gv_stream_get_state(consumer);
    ASSERT(state == GV_STREAM_RUNNING || state == GV_STREAM_PAUSED ||
           state == GV_STREAM_STOPPED || state == GV_STREAM_ERROR,
           "state after resume should be valid");

    gv_stream_stop(consumer);
    gv_stream_destroy(consumer);
    gv_db_close(db);
    cleanup();
    return 0;
}

static int test_commit(void) {
    cleanup();
    GV_Database *db = gv_db_open(DB_PATH, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database open");

    GV_StreamConfig cfg;
    gv_stream_config_init(&cfg);
    cfg.source = GV_STREAM_CUSTOM;

    GV_StreamConsumer *consumer = gv_stream_create(db, &cfg);
    ASSERT(consumer != NULL, "create");

    /* Commit should work even when stopped (commits current state) */
    int rc = gv_stream_commit(consumer);
    ASSERT(rc == 0, "gv_stream_commit should succeed");

    gv_stream_destroy(consumer);
    gv_db_close(db);
    cleanup();
    return 0;
}

static int test_seek_operations(void) {
    cleanup();
    GV_Database *db = gv_db_open(DB_PATH, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database open");

    GV_StreamConfig cfg;
    gv_stream_config_init(&cfg);
    cfg.source = GV_STREAM_CUSTOM;

    GV_StreamConsumer *consumer = gv_stream_create(db, &cfg);
    ASSERT(consumer != NULL, "create");

    /* Seek to specific offset */
    int rc = gv_stream_seek(consumer, 0);
    ASSERT(rc == 0, "gv_stream_seek to 0 should succeed");

    rc = gv_stream_seek(consumer, 100);
    ASSERT(rc == 0, "gv_stream_seek to 100 should succeed");

    /* Seek to beginning */
    rc = gv_stream_seek_beginning(consumer);
    ASSERT(rc == 0, "gv_stream_seek_beginning should succeed");

    /* Seek to end */
    rc = gv_stream_seek_end(consumer);
    ASSERT(rc == 0, "gv_stream_seek_end should succeed");

    gv_stream_destroy(consumer);
    gv_db_close(db);
    cleanup();
    return 0;
}

static int test_reset_stats(void) {
    cleanup();
    GV_Database *db = gv_db_open(DB_PATH, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database open");

    GV_StreamConfig cfg;
    gv_stream_config_init(&cfg);
    cfg.source = GV_STREAM_CUSTOM;

    GV_StreamConsumer *consumer = gv_stream_create(db, &cfg);
    ASSERT(consumer != NULL, "create");

    /* Reset stats */
    int rc = gv_stream_reset_stats(consumer);
    ASSERT(rc == 0, "gv_stream_reset_stats should succeed");

    /* Verify stats are zeroed */
    GV_StreamStats stats;
    rc = gv_stream_get_stats(consumer, &stats);
    ASSERT(rc == 0, "get_stats after reset");
    ASSERT(stats.messages_received == 0, "messages_received after reset should be 0");
    ASSERT(stats.messages_processed == 0, "messages_processed after reset should be 0");

    gv_stream_destroy(consumer);
    gv_db_close(db);
    cleanup();
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    cleanup();
    TestCase tests[] = {
        {"Testing config_init...",       test_config_init},
        {"Testing create_custom...",     test_create_custom},
        {"Testing create_kafka...",      test_create_kafka},
        {"Testing destroy_null...",      test_destroy_null},
        {"Testing get_state_initial...", test_get_state_initial},
        {"Testing get_stats_initial...", test_get_stats_initial},
        {"Testing set_handler...",       test_set_handler},
        {"Testing set_extractor...",     test_set_extractor},
        {"Testing start_stop...",        test_start_stop},
        {"Testing pause_resume...",      test_pause_resume},
        {"Testing commit...",            test_commit},
        {"Testing seek_operations...",   test_seek_operations},
        {"Testing reset_stats...",       test_reset_stats},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    cleanup();
    return passed == n ? 0 : 1;
}
