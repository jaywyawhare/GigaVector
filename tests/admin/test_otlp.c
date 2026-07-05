/**
 * @file test_otlp.c
 * @brief Tests for the OTLP exporter.
 *
 * Verifies that the JSON payload builders produce well-formed OTLP JSON
 * containing the required fields (resourceSpans, scopeSpans, traceId for
 * traces; resourceMetrics, scopeMetrics for metrics).
 *
 * HTTP delivery is exercised but expected to fail (no live collector) so
 * we only check that the function returns a defined code and does not crash.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "admin/otlp.h"
#include "admin/tracing.h"
#include "storage/database.h"

#define ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s  (line %d)\n", (msg), __LINE__); \
            return -1; \
        } \
    } while (0)

/* -------------------------------------------------------------------------
 * Trace JSON tests
 * ---------------------------------------------------------------------- */

static int test_build_trace_json_empty_spans(void) {
    GV_QueryTrace *trace = trace_begin();
    ASSERT(trace != NULL, "trace_begin should succeed");
    trace_end(trace);

    char *json = otlp_build_trace_json(trace);
    ASSERT(json != NULL, "otlp_build_trace_json should return non-NULL");

    /* Required OTLP fields */
    ASSERT(strstr(json, "resourceSpans") != NULL,
           "JSON must contain 'resourceSpans'");
    ASSERT(strstr(json, "scopeSpans") != NULL,
           "JSON must contain 'scopeSpans'");

    free(json);
    trace_destroy(trace);
    return 0;
}

static int test_build_trace_json_with_spans(void) {
    GV_QueryTrace *trace = trace_begin();
    ASSERT(trace != NULL, "trace_begin should succeed");

    trace_span_add(trace, "index_lookup", 1500);
    trace_span_add(trace, "scoring",      800);
    trace_end(trace);

    char *json = otlp_build_trace_json(trace);
    ASSERT(json != NULL, "otlp_build_trace_json should succeed");

    /* Required OTLP structural fields */
    ASSERT(strstr(json, "resourceSpans") != NULL, "must contain resourceSpans");
    ASSERT(strstr(json, "scopeSpans")    != NULL, "must contain scopeSpans");
    ASSERT(strstr(json, "traceId")       != NULL, "must contain traceId");
    ASSERT(strstr(json, "spanId")        != NULL, "must contain spanId");

    /* Span names must appear in the payload */
    ASSERT(strstr(json, "index_lookup")  != NULL, "must contain span name index_lookup");
    ASSERT(strstr(json, "scoring")       != NULL, "must contain span name scoring");

    /* Service name attribute */
    ASSERT(strstr(json, "gigavector")    != NULL, "must contain service name gigavector");

    free(json);
    trace_destroy(trace);
    return 0;
}

static int test_build_trace_json_null(void) {
    char *json = otlp_build_trace_json(NULL);
    ASSERT(json == NULL, "NULL trace should return NULL");
    return 0;
}

/* -------------------------------------------------------------------------
 * Metrics JSON tests
 * ---------------------------------------------------------------------- */

static int test_build_metrics_json_basic(void) {
    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open should succeed");

    /* Insert a couple of vectors so metrics are non-trivial */
    float v1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v2[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    db_add_vector(db, v1, 4);
    db_add_vector(db, v2, 4);

    char *json = otlp_build_metrics_json(db);
    ASSERT(json != NULL, "otlp_build_metrics_json should return non-NULL");

    /* Required OTLP structural fields */
    ASSERT(strstr(json, "resourceMetrics") != NULL, "must contain resourceMetrics");
    ASSERT(strstr(json, "scopeMetrics")    != NULL, "must contain scopeMetrics");
    ASSERT(strstr(json, "metrics")         != NULL, "must contain metrics array");

    /* Known metric names */
    ASSERT(strstr(json, "gigavector.total_inserts") != NULL,
           "must contain total_inserts metric");
    ASSERT(strstr(json, "gigavector.total_queries") != NULL,
           "must contain total_queries metric");
    ASSERT(strstr(json, "gigavector.current_qps") != NULL,
           "must contain current_qps metric");

    free(json);
    db_close(db);
    return 0;
}

static int test_build_metrics_json_null(void) {
    char *json = otlp_build_metrics_json(NULL);
    ASSERT(json == NULL, "NULL db should return NULL");
    return 0;
}

/* -------------------------------------------------------------------------
 * Config / public API tests
 * ---------------------------------------------------------------------- */

static int test_set_otlp_endpoint(void) {
    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open should succeed");

    /* Initially disabled */
    ASSERT(db->otlp_config.enabled == 0, "OTLP should be disabled by default");

    /* Enable */
    gv_db_set_otlp_endpoint(db, "http://localhost:4318/v1/traces");
    ASSERT(db->otlp_config.enabled == 1, "OTLP should be enabled after set");
    ASSERT(strcmp(db->otlp_config.endpoint,
                  "http://localhost:4318/v1/traces") == 0,
           "endpoint should match");

    /* Disable by passing NULL */
    gv_db_set_otlp_endpoint(db, NULL);
    ASSERT(db->otlp_config.enabled == 0, "OTLP should be disabled after NULL set");

    /* Disable by passing empty string */
    gv_db_set_otlp_endpoint(db, "http://localhost:4318/v1/traces");
    ASSERT(db->otlp_config.enabled == 1, "re-enable");
    gv_db_set_otlp_endpoint(db, "");
    ASSERT(db->otlp_config.enabled == 0, "OTLP should be disabled after empty set");

    db_close(db);
    return 0;
}

static int test_otlp_flush_disabled(void) {
    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open should succeed");

    /* Flush when disabled should return -1, not crash */
    int rc = gv_db_otlp_flush(db);
    ASSERT(rc == -1, "flush on disabled OTLP should return -1");

    db_close(db);
    return 0;
}

static int test_otlp_flush_no_server(void) {
    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open should succeed");

    /* Point to a non-existent server — should fail gracefully */
    gv_db_set_otlp_endpoint(db, "http://127.0.0.1:19999/v1/metrics");

    /* Return code will be -1 (no server), but must not crash */
    gv_db_otlp_flush(db);   /* ignore return — may be -1 */

    db_close(db);
    return 0;
}

static int test_otlp_export_trace_no_server(void) {
    GV_OtlpConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    strncpy(cfg.endpoint, "http://127.0.0.1:19999/v1/traces",
            sizeof(cfg.endpoint) - 1);
    cfg.enabled = 1;

    GV_QueryTrace *trace = trace_begin();
    ASSERT(trace != NULL, "trace_begin should succeed");
    trace_span_add(trace, "test_span", 100);
    trace_end(trace);

    /* Should fail gracefully (no live collector) */
    otlp_export_trace(&cfg, trace);  /* ignore return value */

    trace_destroy(trace);
    return 0;
}

/* -------------------------------------------------------------------------
 * Test runner
 * ---------------------------------------------------------------------- */

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"build_trace_json_empty_spans",  test_build_trace_json_empty_spans},
        {"build_trace_json_with_spans",   test_build_trace_json_with_spans},
        {"build_trace_json_null",         test_build_trace_json_null},
        {"build_metrics_json_basic",      test_build_metrics_json_basic},
        {"build_metrics_json_null",       test_build_metrics_json_null},
        {"set_otlp_endpoint",             test_set_otlp_endpoint},
        {"otlp_flush_disabled",           test_otlp_flush_disabled},
        {"otlp_flush_no_server",          test_otlp_flush_no_server},
        {"otlp_export_trace_no_server",   test_otlp_export_trace_no_server},
    };

    int n = (int)(sizeof(tests) / sizeof(tests[0]));
    int passed = 0;

    for (int i = 0; i < n; i++) {
        printf("Testing %s... ", tests[i].name);
        fflush(stdout);
        if (tests[i].fn() == 0) {
            printf("PASS\n");
            passed++;
        } else {
            printf("FAIL\n");
        }
    }

    printf("%d/%d passed\n", passed, n);
    return passed == n ? 0 : 1;
}
