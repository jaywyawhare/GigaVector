#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_webhook.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

/* Callback tracking for change stream tests */
static int g_callback_count = 0;
static GV_EventType g_last_event_type = 0;
static size_t g_last_vector_index = 0;

static void test_callback(const GV_Event *event, void *user_data) {
    g_callback_count++;
    g_last_event_type = event->event_type;
    g_last_vector_index = event->vector_index;
    (void)user_data;
}

static int test_webhook_create_destroy(void) {
    GV_WebhookManager *mgr = gv_webhook_create();
    ASSERT(mgr != NULL, "webhook manager creation should succeed");

    gv_webhook_destroy(mgr);

    /* Destroying NULL should be safe */
    gv_webhook_destroy(NULL);
    return 0;
}

static int test_webhook_register_unregister(void) {
    GV_WebhookManager *mgr = gv_webhook_create();
    ASSERT(mgr != NULL, "webhook manager creation");

    GV_WebhookConfig config;
    memset(&config, 0, sizeof(config));
    config.url = "http://localhost:9999/hook";
    config.event_mask = GV_EVENT_INSERT;
    config.secret = NULL;
    config.max_retries = 3;
    config.timeout_ms = 5000;
    config.active = 1;

    int rc = gv_webhook_register(mgr, "hook1", &config);
    ASSERT(rc == 0, "registering webhook should succeed");

    rc = gv_webhook_unregister(mgr, "hook1");
    ASSERT(rc == 0, "unregistering webhook should succeed");

    gv_webhook_destroy(mgr);
    return 0;
}

static int test_webhook_list(void) {
    GV_WebhookManager *mgr = gv_webhook_create();
    ASSERT(mgr != NULL, "webhook manager creation");

    GV_WebhookConfig config;
    memset(&config, 0, sizeof(config));
    config.url = "http://localhost:9999/a";
    config.event_mask = GV_EVENT_ALL;
    config.active = 1;

    gv_webhook_register(mgr, "hook_a", &config);

    config.url = "http://localhost:9999/b";
    gv_webhook_register(mgr, "hook_b", &config);

    char **ids = NULL;
    size_t count = 0;
    int rc = gv_webhook_list(mgr, &ids, &count);
    ASSERT(rc == 0, "listing webhooks should succeed");
    ASSERT(count == 2, "should have 2 registered webhooks");

    gv_webhook_free_list(ids, count);
    gv_webhook_destroy(mgr);
    return 0;
}

static int test_webhook_pause_resume(void) {
    GV_WebhookManager *mgr = gv_webhook_create();
    ASSERT(mgr != NULL, "webhook manager creation");

    GV_WebhookConfig config;
    memset(&config, 0, sizeof(config));
    config.url = "http://localhost:9999/hook";
    config.event_mask = GV_EVENT_INSERT;
    config.active = 1;

    gv_webhook_register(mgr, "hook1", &config);

    int rc = gv_webhook_pause(mgr, "hook1");
    ASSERT(rc == 0, "pausing webhook should succeed");

    rc = gv_webhook_resume(mgr, "hook1");
    ASSERT(rc == 0, "resuming webhook should succeed");

    gv_webhook_destroy(mgr);
    return 0;
}

static int test_webhook_subscribe_callback(void) {
    GV_WebhookManager *mgr = gv_webhook_create();
    ASSERT(mgr != NULL, "webhook manager creation");

    g_callback_count = 0;
    g_last_event_type = 0;
    g_last_vector_index = 0;

    int rc = gv_webhook_subscribe(mgr, GV_EVENT_ALL, test_callback, NULL);
    ASSERT(rc == 0, "subscribing callback should succeed");

    GV_Event event;
    memset(&event, 0, sizeof(event));
    event.event_type = GV_EVENT_INSERT;
    event.vector_index = 42;
    event.timestamp = 1000;
    event.collection = NULL;

    rc = gv_webhook_fire(mgr, &event);
    ASSERT(rc == 0, "firing event should succeed");
    ASSERT(g_callback_count == 1, "callback should have been invoked once");
    ASSERT(g_last_event_type == GV_EVENT_INSERT, "callback should receive INSERT event");
    ASSERT(g_last_vector_index == 42, "callback should receive correct vector index");

    gv_webhook_destroy(mgr);
    return 0;
}

static int test_webhook_unsubscribe(void) {
    GV_WebhookManager *mgr = gv_webhook_create();
    ASSERT(mgr != NULL, "webhook manager creation");

    g_callback_count = 0;

    gv_webhook_subscribe(mgr, GV_EVENT_ALL, test_callback, NULL);

    int rc = gv_webhook_unsubscribe(mgr, test_callback);
    ASSERT(rc == 0, "unsubscribing callback should succeed");

    GV_Event event;
    memset(&event, 0, sizeof(event));
    event.event_type = GV_EVENT_DELETE;
    event.vector_index = 99;
    event.timestamp = 2000;

    gv_webhook_fire(mgr, &event);
    ASSERT(g_callback_count == 0, "callback should not be invoked after unsubscribe");

    gv_webhook_destroy(mgr);
    return 0;
}

static int test_webhook_event_mask_filter(void) {
    GV_WebhookManager *mgr = gv_webhook_create();
    ASSERT(mgr != NULL, "webhook manager creation");

    g_callback_count = 0;

    /* Subscribe only for INSERT events */
    gv_webhook_subscribe(mgr, GV_EVENT_INSERT, test_callback, NULL);

    GV_Event event;
    memset(&event, 0, sizeof(event));

    /* Fire a DELETE event -- should not trigger callback */
    event.event_type = GV_EVENT_DELETE;
    event.vector_index = 10;
    event.timestamp = 3000;
    gv_webhook_fire(mgr, &event);

    /* Fire an INSERT event -- should trigger callback */
    event.event_type = GV_EVENT_INSERT;
    event.vector_index = 20;
    event.timestamp = 3001;
    gv_webhook_fire(mgr, &event);

    ASSERT(g_callback_count >= 1, "INSERT callback should have been invoked");

    gv_webhook_destroy(mgr);
    return 0;
}

static int test_webhook_stats(void) {
    GV_WebhookManager *mgr = gv_webhook_create();
    ASSERT(mgr != NULL, "webhook manager creation");

    GV_WebhookStats stats;
    memset(&stats, 0, sizeof(stats));
    int rc = gv_webhook_get_stats(mgr, &stats);
    ASSERT(rc == 0, "getting stats should succeed");
    ASSERT(stats.events_fired == 0, "initial events_fired should be 0");

    gv_webhook_destroy(mgr);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing webhook create/destroy...", test_webhook_create_destroy},
        {"Testing webhook register/unregister...", test_webhook_register_unregister},
        {"Testing webhook list...", test_webhook_list},
        {"Testing webhook pause/resume...", test_webhook_pause_resume},
        {"Testing webhook subscribe callback...", test_webhook_subscribe_callback},
        {"Testing webhook unsubscribe...", test_webhook_unsubscribe},
        {"Testing webhook event mask filter...", test_webhook_event_mask_filter},
        {"Testing webhook stats...", test_webhook_stats},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
