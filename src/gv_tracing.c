/**
 * @file gv_tracing.c
 * @brief Query tracing implementation.
 */

#define _POSIX_C_SOURCE 200809L

#include "gigavector/gv_tracing.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>

/* ============================================================================
 * Constants
 * ============================================================================ */

#define GV_TRACE_INITIAL_CAPACITY 16

/* ============================================================================
 * Internal State
 * ============================================================================ */

/** Global trace ID counter. */
static uint64_t gv_trace_id_counter = 0;

/* ============================================================================
 * Utility
 * ============================================================================ */

uint64_t gv_trace_get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

/* ============================================================================
 * Internal Helpers
 * ============================================================================ */

/**
 * @brief Ensure the spans array has room for at least one more span.
 *
 * @param trace Trace whose span array may need to grow.
 * @return 0 on success, -1 on allocation failure.
 */
static int gv_trace_ensure_capacity(GV_QueryTrace *trace) {
    if (trace->span_count < trace->span_capacity) {
        return 0;
    }

    size_t new_capacity = trace->span_capacity * 2;
    if (new_capacity == 0) {
        new_capacity = GV_TRACE_INITIAL_CAPACITY;
    }

    GV_TraceSpan *new_spans = realloc(trace->spans, new_capacity * sizeof(GV_TraceSpan));
    if (!new_spans) {
        return -1;
    }

    trace->spans = new_spans;
    trace->span_capacity = new_capacity;
    return 0;
}

/**
 * @brief Find the last open span (duration_us == 0), searching backwards.
 *
 * @param trace Trace to search.
 * @return Pointer to the last open span, or NULL if none found.
 */
static GV_TraceSpan *gv_trace_find_last_open_span(GV_QueryTrace *trace) {
    if (!trace || trace->span_count == 0) {
        return NULL;
    }

    for (size_t i = trace->span_count; i > 0; i--) {
        if (trace->spans[i - 1].duration_us == 0) {
            return &trace->spans[i - 1];
        }
    }

    return NULL;
}

/* ============================================================================
 * Trace Lifecycle
 * ============================================================================ */

GV_QueryTrace *gv_trace_begin(void) {
    GV_QueryTrace *trace = calloc(1, sizeof(GV_QueryTrace));
    if (!trace) {
        return NULL;
    }

    trace->spans = malloc(GV_TRACE_INITIAL_CAPACITY * sizeof(GV_TraceSpan));
    if (!trace->spans) {
        free(trace);
        return NULL;
    }

    trace->trace_id = ++gv_trace_id_counter;
    trace->total_duration_us = 0;
    trace->span_count = 0;
    trace->span_capacity = GV_TRACE_INITIAL_CAPACITY;
    trace->active = 1;

    return trace;
}

void gv_trace_end(GV_QueryTrace *trace) {
    if (!trace || !trace->active) {
        return;
    }

    if (trace->span_count > 0) {
        uint64_t now = gv_trace_get_time_us();
        trace->total_duration_us = now - trace->spans[0].start_us;
    } else {
        trace->total_duration_us = 0;
    }

    trace->active = 0;
}

void gv_trace_destroy(GV_QueryTrace *trace) {
    if (!trace) {
        return;
    }

    for (size_t i = 0; i < trace->span_count; i++) {
        free((void *)trace->spans[i].name);
        free((void *)trace->spans[i].metadata);
    }

    free(trace->spans);
    free(trace);
}

/* ============================================================================
 * Span Operations
 * ============================================================================ */

void gv_trace_span_start(GV_QueryTrace *trace, const char *name) {
    if (!trace || !trace->active || !name) {
        return;
    }

    if (gv_trace_ensure_capacity(trace) != 0) {
        return;
    }

    GV_TraceSpan *span = &trace->spans[trace->span_count];
    span->name = strdup(name);
    span->start_us = gv_trace_get_time_us();
    span->duration_us = 0;
    span->metadata = NULL;

    if (!span->name) {
        return; /* allocation failed, do not increment count */
    }

    trace->span_count++;
}

void gv_trace_span_end(GV_QueryTrace *trace) {
    if (!trace || !trace->active) {
        return;
    }

    GV_TraceSpan *span = gv_trace_find_last_open_span(trace);
    if (!span) {
        return;
    }

    uint64_t now = gv_trace_get_time_us();
    span->duration_us = now - span->start_us;
}

void gv_trace_span_add(GV_QueryTrace *trace, const char *name, uint64_t duration_us) {
    if (!trace || !trace->active || !name) {
        return;
    }

    if (gv_trace_ensure_capacity(trace) != 0) {
        return;
    }

    GV_TraceSpan *span = &trace->spans[trace->span_count];
    span->name = strdup(name);
    span->start_us = gv_trace_get_time_us();
    span->duration_us = duration_us;
    span->metadata = NULL;

    if (!span->name) {
        return;
    }

    trace->span_count++;
}

void gv_trace_set_metadata(GV_QueryTrace *trace, const char *metadata) {
    if (!trace || !trace->active) {
        return;
    }

    GV_TraceSpan *span = gv_trace_find_last_open_span(trace);
    if (!span) {
        return;
    }

    /* Free any previously set metadata. */
    free((void *)span->metadata);
    span->metadata = metadata ? strdup(metadata) : NULL;
}

/* ============================================================================
 * Serialization
 * ============================================================================ */

/**
 * @brief Escape a string for JSON output.
 *
 * Handles backslash, double-quote, and common control characters.
 *
 * @param src Input string.
 * @param dst Output buffer.
 * @param dst_size Size of output buffer.
 * @return Number of characters written (excluding null terminator).
 */
static size_t gv_trace_json_escape(const char *src, char *dst, size_t dst_size) {
    size_t written = 0;

    if (!src || !dst || dst_size == 0) {
        return 0;
    }

    for (const char *p = src; *p && written + 1 < dst_size; p++) {
        char c = *p;
        if (c == '\\' || c == '"') {
            if (written + 2 >= dst_size) break;
            dst[written++] = '\\';
            dst[written++] = c;
        } else if (c == '\n') {
            if (written + 2 >= dst_size) break;
            dst[written++] = '\\';
            dst[written++] = 'n';
        } else if (c == '\r') {
            if (written + 2 >= dst_size) break;
            dst[written++] = '\\';
            dst[written++] = 'r';
        } else if (c == '\t') {
            if (written + 2 >= dst_size) break;
            dst[written++] = '\\';
            dst[written++] = 't';
        } else {
            dst[written++] = c;
        }
    }

    dst[written] = '\0';
    return written;
}

char *gv_trace_to_json(const GV_QueryTrace *trace) {
    if (!trace) {
        return NULL;
    }

    /*
     * Estimate required buffer size.
     * Base: {"trace_id":...,"total_us":...,"spans":[]}  ~80 bytes
     * Per span: {"name":"...","start_us":...,"duration_us":...,"metadata":...}  ~200 bytes + name/metadata
     */
    size_t estimate = 128 + trace->span_count * 512;
    char *buf = malloc(estimate);
    if (!buf) {
        return NULL;
    }

    size_t offset = 0;
    int ret;

    ret = snprintf(buf + offset, estimate - offset,
                   "{\"trace_id\":%" PRIu64 ",\"total_us\":%" PRIu64 ",\"spans\":[",
                   trace->trace_id, trace->total_duration_us);
    if (ret < 0) {
        free(buf);
        return NULL;
    }
    offset += (size_t)ret;

    for (size_t i = 0; i < trace->span_count; i++) {
        const GV_TraceSpan *span = &trace->spans[i];

        /* Escape the span name. */
        char escaped_name[512];
        gv_trace_json_escape(span->name ? span->name : "", escaped_name, sizeof(escaped_name));

        /* Start the span object. */
        if (i > 0) {
            if (offset < estimate) {
                buf[offset++] = ',';
            }
        }

        ret = snprintf(buf + offset, estimate - offset,
                       "{\"name\":\"%s\",\"start_us\":%" PRIu64 ",\"duration_us\":%" PRIu64 ",\"metadata\":",
                       escaped_name, span->start_us, span->duration_us);
        if (ret < 0) {
            free(buf);
            return NULL;
        }
        offset += (size_t)ret;

        /* Metadata: null or escaped string. */
        if (span->metadata) {
            char escaped_meta[1024];
            gv_trace_json_escape(span->metadata, escaped_meta, sizeof(escaped_meta));
            ret = snprintf(buf + offset, estimate - offset, "\"%s\"}", escaped_meta);
        } else {
            ret = snprintf(buf + offset, estimate - offset, "null}");
        }
        if (ret < 0) {
            free(buf);
            return NULL;
        }
        offset += (size_t)ret;

        /* Grow buffer if running low. */
        if (offset + 512 > estimate) {
            estimate *= 2;
            char *new_buf = realloc(buf, estimate);
            if (!new_buf) {
                free(buf);
                return NULL;
            }
            buf = new_buf;
        }
    }

    ret = snprintf(buf + offset, estimate - offset, "]}");
    if (ret < 0) {
        free(buf);
        return NULL;
    }
    offset += (size_t)ret;

    /* Shrink to fit. */
    char *result = realloc(buf, offset + 1);
    return result ? result : buf;
}

/* ============================================================================
 * Pretty Print
 * ============================================================================ */

void gv_trace_print(const GV_QueryTrace *trace, FILE *out) {
    if (!trace || !out) {
        return;
    }

    fprintf(out, "Trace %" PRIu64 " [%s] total=%" PRIu64 " us\n",
            trace->trace_id,
            trace->active ? "ACTIVE" : "COMPLETE",
            trace->total_duration_us);

    for (size_t i = 0; i < trace->span_count; i++) {
        const GV_TraceSpan *span = &trace->spans[i];
        fprintf(out, "  [%zu] %-30s  start=%" PRIu64 " us  duration=%" PRIu64 " us",
                i, span->name ? span->name : "(null)", span->start_us, span->duration_us);
        if (span->metadata) {
            fprintf(out, "  meta=\"%s\"", span->metadata);
        }
        fprintf(out, "\n");
    }
}
