#ifndef GIGAVECTOR_GV_TRACING_H
#define GIGAVECTOR_GV_TRACING_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_tracing.h
 * @brief Query tracing for GigaVector.
 *
 * Provides lightweight instrumentation for tracing query execution paths,
 * measuring span durations, and serializing trace data to JSON.
 */

/**
 * @brief A single trace span representing a timed operation.
 */
typedef struct {
    const char *name;           /**< Span name (owned, strdup'd). */
    uint64_t start_us;          /**< Start time in microseconds (CLOCK_MONOTONIC). */
    uint64_t duration_us;       /**< Duration in microseconds (0 if still running). */
    const char *metadata;       /**< Optional metadata string (owned, strdup'd, or NULL). */
} GV_TraceSpan;

/**
 * @brief A complete query trace containing multiple spans.
 */
typedef struct {
    uint64_t trace_id;          /**< Unique trace identifier. */
    uint64_t total_duration_us; /**< Total trace duration in microseconds. */
    GV_TraceSpan *spans;        /**< Dynamic array of spans. */
    size_t span_count;          /**< Number of spans recorded. */
    size_t span_capacity;       /**< Allocated capacity of spans array. */
    int active;                 /**< 1 if trace is active, 0 if finalized. */
} GV_QueryTrace;

/* ============================================================================
 * Trace Lifecycle
 * ============================================================================ */

/**
 * @brief Begin a new query trace.
 *
 * Allocates a new trace with an auto-generated trace_id and marks it active.
 *
 * @return New trace (caller owns, use gv_trace_destroy to free), or NULL on allocation failure.
 */
GV_QueryTrace *gv_trace_begin(void);

/**
 * @brief Finalize a trace and compute total_duration_us.
 *
 * Sets total_duration_us based on the elapsed time since the first span started.
 * Marks the trace as inactive.
 *
 * @param trace Trace to finalize.
 */
void gv_trace_end(GV_QueryTrace *trace);

/**
 * @brief Free a trace and all its spans.
 *
 * @param trace Trace to destroy (NULL is safe).
 */
void gv_trace_destroy(GV_QueryTrace *trace);

/* ============================================================================
 * Span Operations
 * ============================================================================ */

/**
 * @brief Start a new named span within a trace.
 *
 * Records the current time as the span start. The span remains open until
 * gv_trace_span_end() is called.
 *
 * @param trace Active trace.
 * @param name Span name (will be strdup'd).
 */
void gv_trace_span_start(GV_QueryTrace *trace, const char *name);

/**
 * @brief End the most recently started open span.
 *
 * Finds the last span with duration_us == 0 and computes its duration.
 *
 * @param trace Active trace.
 */
void gv_trace_span_end(GV_QueryTrace *trace);

/**
 * @brief Add a completed span with a known duration.
 *
 * @param trace Active trace.
 * @param name Span name (will be strdup'd).
 * @param duration_us Duration in microseconds.
 */
void gv_trace_span_add(GV_QueryTrace *trace, const char *name, uint64_t duration_us);

/**
 * @brief Set metadata on the most recently started open span.
 *
 * @param trace Active trace.
 * @param metadata Metadata string (will be strdup'd).
 */
void gv_trace_set_metadata(GV_QueryTrace *trace, const char *metadata);

/* ============================================================================
 * Serialization and Output
 * ============================================================================ */

/**
 * @brief Serialize a trace to a JSON string.
 *
 * Produces a JSON object with trace_id, total_us, and a spans array.
 *
 * @param trace Trace to serialize.
 * @return JSON string (caller must free), or NULL on error.
 */
char *gv_trace_to_json(const GV_QueryTrace *trace);

/**
 * @brief Pretty-print a trace to a file stream.
 *
 * @param trace Trace to print.
 * @param out Output stream (e.g., stdout, stderr).
 */
void gv_trace_print(const GV_QueryTrace *trace, FILE *out);

/* ============================================================================
 * Utility
 * ============================================================================ */

/**
 * @brief Get the current time in microseconds using CLOCK_MONOTONIC.
 *
 * @return Current monotonic time in microseconds.
 */
uint64_t gv_trace_get_time_us(void);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_TRACING_H */
