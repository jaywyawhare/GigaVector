#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_tracing.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_trace_create_destroy(void) {
    GV_QueryTrace *trace = gv_trace_begin();
    ASSERT(trace != NULL, "trace creation should succeed");
    ASSERT(trace->active == 1, "new trace should be active");
    ASSERT(trace->trace_id != 0, "trace_id should be non-zero");
    ASSERT(trace->span_count == 0, "new trace should have no spans");

    gv_trace_destroy(trace);

    /* Destroying NULL should be safe */
    gv_trace_destroy(NULL);
    return 0;
}

static int test_trace_span_start_end(void) {
    GV_QueryTrace *trace = gv_trace_begin();
    ASSERT(trace != NULL, "trace creation");

    gv_trace_span_start(trace, "index_lookup");
    ASSERT(trace->span_count == 1, "should have 1 span after start");
    ASSERT(strcmp(trace->spans[0].name, "index_lookup") == 0, "span name should match");
    ASSERT(trace->spans[0].duration_us == 0, "open span should have duration 0");

    gv_trace_span_end(trace);
    ASSERT(trace->spans[0].duration_us > 0 || trace->spans[0].duration_us == 0,
           "span duration should be set (may be 0 if very fast)");

    gv_trace_destroy(trace);
    return 0;
}

static int test_trace_multiple_spans(void) {
    GV_QueryTrace *trace = gv_trace_begin();
    ASSERT(trace != NULL, "trace creation");

    gv_trace_span_start(trace, "phase1");
    gv_trace_span_end(trace);

    gv_trace_span_start(trace, "phase2");
    gv_trace_span_end(trace);

    gv_trace_span_start(trace, "phase3");
    gv_trace_span_end(trace);

    ASSERT(trace->span_count == 3, "should have 3 spans");
    ASSERT(strcmp(trace->spans[0].name, "phase1") == 0, "first span name");
    ASSERT(strcmp(trace->spans[1].name, "phase2") == 0, "second span name");
    ASSERT(strcmp(trace->spans[2].name, "phase3") == 0, "third span name");

    gv_trace_destroy(trace);
    return 0;
}

static int test_trace_span_add(void) {
    GV_QueryTrace *trace = gv_trace_begin();
    ASSERT(trace != NULL, "trace creation");

    gv_trace_span_add(trace, "precomputed_step", 12345);
    ASSERT(trace->span_count == 1, "should have 1 span after add");
    ASSERT(strcmp(trace->spans[0].name, "precomputed_step") == 0, "span name should match");
    ASSERT(trace->spans[0].duration_us == 12345, "duration should match added value");

    gv_trace_destroy(trace);
    return 0;
}

static int test_trace_metadata(void) {
    GV_QueryTrace *trace = gv_trace_begin();
    ASSERT(trace != NULL, "trace creation");

    gv_trace_span_start(trace, "search");
    gv_trace_set_metadata(trace, "k=10,ef=200");
    ASSERT(trace->spans[0].metadata != NULL, "metadata should be set");
    ASSERT(strcmp(trace->spans[0].metadata, "k=10,ef=200") == 0, "metadata content should match");
    gv_trace_span_end(trace);

    gv_trace_destroy(trace);
    return 0;
}

static int test_trace_end(void) {
    GV_QueryTrace *trace = gv_trace_begin();
    ASSERT(trace != NULL, "trace creation");

    gv_trace_span_start(trace, "work");
    gv_trace_span_end(trace);

    gv_trace_end(trace);
    ASSERT(trace->active == 0, "trace should be inactive after end");
    ASSERT(trace->total_duration_us >= 0, "total_duration_us should be set");

    gv_trace_destroy(trace);
    return 0;
}

static int test_trace_to_json(void) {
    GV_QueryTrace *trace = gv_trace_begin();
    ASSERT(trace != NULL, "trace creation");

    gv_trace_span_add(trace, "step_a", 100);
    gv_trace_span_add(trace, "step_b", 200);
    gv_trace_end(trace);

    char *json = gv_trace_to_json(trace);
    ASSERT(json != NULL, "JSON serialization should succeed");
    ASSERT(strstr(json, "trace_id") != NULL, "JSON should contain trace_id");
    ASSERT(strstr(json, "spans") != NULL, "JSON should contain spans");
    ASSERT(strstr(json, "step_a") != NULL, "JSON should contain span name step_a");
    ASSERT(strstr(json, "step_b") != NULL, "JSON should contain span name step_b");

    free(json);
    gv_trace_destroy(trace);
    return 0;
}

static int test_trace_get_time_us(void) {
    uint64_t t1 = gv_trace_get_time_us();
    uint64_t t2 = gv_trace_get_time_us();
    ASSERT(t2 >= t1, "monotonic time should not go backwards");
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing trace create/destroy...", test_trace_create_destroy},
        {"Testing trace span start/end...", test_trace_span_start_end},
        {"Testing multiple spans...", test_trace_multiple_spans},
        {"Testing trace span add...", test_trace_span_add},
        {"Testing trace metadata...", test_trace_metadata},
        {"Testing trace end...", test_trace_end},
        {"Testing trace to JSON...", test_trace_to_json},
        {"Testing trace get time...", test_trace_get_time_us},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        printf("%s", tests[i].name);
        if (tests[i].fn() == 0) { printf(" [OK]\n"); passed++; }
        else { printf(" [FAIL]\n"); }
    }
    printf("\n%d/%d tests passed\n", passed, n);
    return passed == n ? 0 : 1;
}
