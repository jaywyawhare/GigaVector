#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_cdc.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

/* Shared state for callback tests */
static int g_callback_count = 0;
static GV_CDCEventType g_last_event_type = 0;

static void test_callback(const GV_CDCEvent *event, void *user_data) {
    (void)user_data;
    g_callback_count++;
    g_last_event_type = event->type;
}

static int test_config_init(void) {
    GV_CDCConfig cfg;
    memset(&cfg, 0xFF, sizeof(cfg));
    gv_cdc_config_init(&cfg);
    ASSERT(cfg.ring_buffer_size == 65536, "ring_buffer_size default should be 65536");
    ASSERT(cfg.persist_to_file == 0,      "persist_to_file default should be 0");
    ASSERT(cfg.max_log_size_mb == 256,    "max_log_size_mb default should be 256");
    ASSERT(cfg.include_vector_data == 1,  "include_vector_data default should be 1");
    return 0;
}

static int test_create_destroy(void) {
    GV_CDCStream *stream = gv_cdc_create(NULL);
    ASSERT(stream != NULL, "gv_cdc_create(NULL) should succeed");
    gv_cdc_destroy(stream);

    GV_CDCConfig cfg;
    gv_cdc_config_init(&cfg);
    stream = gv_cdc_create(&cfg);
    ASSERT(stream != NULL, "gv_cdc_create with config should succeed");
    gv_cdc_destroy(stream);

    /* Destroy NULL is safe */
    gv_cdc_destroy(NULL);
    return 0;
}

static int test_publish_and_poll(void) {
    GV_CDCStream *stream = gv_cdc_create(NULL);
    ASSERT(stream != NULL, "create");

    GV_CDCCursor cursor = gv_cdc_get_cursor(stream);

    float vec[] = {1.0f, 2.0f, 3.0f};
    GV_CDCEvent event;
    memset(&event, 0, sizeof(event));
    event.type = GV_CDC_INSERT;
    event.vector_index = 42;
    event.vector_data = vec;
    event.dimension = 3;
    event.metadata_json = "{\"tag\":\"test\"}";

    ASSERT(gv_cdc_publish(stream, &event) == 0, "publish insert");

    GV_CDCEvent polled[4];
    memset(polled, 0, sizeof(polled));
    int n = gv_cdc_poll(stream, &cursor, polled, 4);
    ASSERT(n == 1, "poll should return 1 event");
    ASSERT(polled[0].type == GV_CDC_INSERT, "polled event type should be INSERT");
    ASSERT(polled[0].vector_index == 42, "polled event vector_index should be 42");

    gv_cdc_destroy(stream);
    return 0;
}

static int test_subscribe_and_callback(void) {
    GV_CDCStream *stream = gv_cdc_create(NULL);
    ASSERT(stream != NULL, "create");

    g_callback_count = 0;
    g_last_event_type = 0;

    int sub_id = gv_cdc_subscribe(stream, GV_CDC_INSERT | GV_CDC_DELETE,
                                  test_callback, NULL);
    ASSERT(sub_id >= 0, "subscribe should return non-negative ID");

    float vec[] = {1.0f};
    GV_CDCEvent evt;
    memset(&evt, 0, sizeof(evt));
    evt.type = GV_CDC_INSERT;
    evt.vector_index = 0;
    evt.vector_data = vec;
    evt.dimension = 1;
    ASSERT(gv_cdc_publish(stream, &evt) == 0, "publish insert");
    ASSERT(g_callback_count == 1, "callback should have been invoked once");
    ASSERT(g_last_event_type == GV_CDC_INSERT, "last event type should be INSERT");

    /* Unsubscribe */
    ASSERT(gv_cdc_unsubscribe(stream, sub_id) == 0, "unsubscribe");

    /* Publish again; callback should NOT fire */
    evt.type = GV_CDC_DELETE;
    evt.vector_data = NULL;
    ASSERT(gv_cdc_publish(stream, &evt) == 0, "publish delete");
    ASSERT(g_callback_count == 1, "callback count should still be 1 after unsubscribe");

    gv_cdc_destroy(stream);
    return 0;
}

static int test_cursor_from_sequence(void) {
    GV_CDCCursor c = gv_cdc_cursor_from_sequence(100);
    ASSERT(c.sequence_number == 100, "cursor sequence_number should be 100");

    GV_CDCCursor c0 = gv_cdc_cursor_from_sequence(0);
    ASSERT(c0.sequence_number == 0, "cursor at 0");
    return 0;
}

static int test_pending_count(void) {
    GV_CDCStream *stream = gv_cdc_create(NULL);
    ASSERT(stream != NULL, "create");

    GV_CDCCursor cursor = gv_cdc_get_cursor(stream);

    /* No events yet */
    ASSERT(gv_cdc_pending_count(stream, &cursor) == 0, "no pending initially");

    /* Publish 3 events */
    float vec[] = {0.5f};
    GV_CDCEvent evt;
    memset(&evt, 0, sizeof(evt));
    evt.type = GV_CDC_INSERT;
    evt.vector_data = vec;
    evt.dimension = 1;
    for (int i = 0; i < 3; i++) {
        evt.vector_index = (size_t)i;
        ASSERT(gv_cdc_publish(stream, &evt) == 0, "publish");
    }

    size_t pending = gv_cdc_pending_count(stream, &cursor);
    ASSERT(pending == 3, "should have 3 pending events");

    /* Poll 2, then check pending again */
    GV_CDCEvent buf[2];
    int polled = gv_cdc_poll(stream, &cursor, buf, 2);
    ASSERT(polled == 2, "should poll 2 events");

    pending = gv_cdc_pending_count(stream, &cursor);
    ASSERT(pending == 1, "should have 1 pending after polling 2");

    gv_cdc_destroy(stream);
    return 0;
}

/* multiple event types */
static int test_multiple_event_types(void) {
    GV_CDCStream *stream = gv_cdc_create(NULL);
    ASSERT(stream != NULL, "create");

    GV_CDCCursor cursor = gv_cdc_get_cursor(stream);

    float vec[] = {1.0f, 2.0f};
    GV_CDCEvent evt;
    memset(&evt, 0, sizeof(evt));
    evt.vector_data = vec;
    evt.dimension = 2;
    evt.vector_index = 0;

    /* Publish INSERT, UPDATE, DELETE */
    evt.type = GV_CDC_INSERT;
    ASSERT(gv_cdc_publish(stream, &evt) == 0, "publish insert");

    evt.type = GV_CDC_UPDATE;
    ASSERT(gv_cdc_publish(stream, &evt) == 0, "publish update");

    evt.type = GV_CDC_DELETE;
    evt.vector_data = NULL;
    evt.dimension = 0;
    ASSERT(gv_cdc_publish(stream, &evt) == 0, "publish delete");

    GV_CDCEvent polled[8];
    int n = gv_cdc_poll(stream, &cursor, polled, 8);
    ASSERT(n == 3, "should poll 3 events");
    ASSERT(polled[0].type == GV_CDC_INSERT, "first should be INSERT");
    ASSERT(polled[1].type == GV_CDC_UPDATE, "second should be UPDATE");
    ASSERT(polled[2].type == GV_CDC_DELETE, "third should be DELETE");

    gv_cdc_destroy(stream);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing config_init...",           test_config_init},
        {"Testing create_destroy...",        test_create_destroy},
        {"Testing publish_and_poll...",      test_publish_and_poll},
        {"Testing subscribe_and_callback..", test_subscribe_and_callback},
        {"Testing cursor_from_sequence...",  test_cursor_from_sequence},
        {"Testing pending_count...",         test_pending_count},
        {"Testing multiple_event_types...",  test_multiple_event_types},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
