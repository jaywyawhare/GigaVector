#ifndef GIGAVECTOR_GV_OTLP_H
#define GIGAVECTOR_GV_OTLP_H

#include <stddef.h>
#include <stdint.h>

#include "admin/tracing.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file otlp.h
 * @brief OpenTelemetry Protocol (OTLP) export for GigaVector.
 *
 * Provides lightweight OTLP/HTTP JSON export for traces and metrics.
 * HTTP delivery requires libcurl (compiled in when HAVE_CURL is defined).
 */

/* Forward declaration for GV_Database to avoid circular include. */
struct GV_Database;

/**
 * @brief OTLP exporter configuration.
 *
 * Embed this in GV_Database (or pass directly to the export functions).
 * @p endpoint is the OTLP/HTTP endpoint accepting JSON payloads
 * (e.g., "http://localhost:4318/v1/traces").
 */
typedef struct {
    char endpoint[256]; /**< OTLP traces endpoint URL. */
    char metrics_endpoint[256]; /**< OTLP metrics endpoint URL. */
    int  enabled;       /**< Non-zero to enable export. */
} GV_OtlpConfig;

/**
 * @brief Build an OTLP/HTTP JSON payload from a completed query trace.
 *
 * Produces a JSON string conforming to the OTLP protobuf-JSON mapping for
 * ExportTraceServiceRequest.  The caller is responsible for freeing the
 * returned string with free().
 *
 * @param trace Completed trace (trace_end() must have been called).
 * @return Heap-allocated JSON string, or NULL on allocation failure.
 */
char *otlp_build_trace_json(const GV_QueryTrace *trace);

/**
 * @brief Build an OTLP/HTTP JSON payload for database metrics.
 *
 * Produces a JSON string conforming to the OTLP protobuf-JSON mapping for
 * ExportMetricsServiceRequest, containing counters and latency histograms.
 *
 * @param db Database to sample metrics from.
 * @return Heap-allocated JSON string, or NULL on allocation failure.
 */
char *otlp_build_metrics_json(const struct GV_Database *db);

/**
 * @brief Export a trace to the configured OTLP endpoint via HTTP POST.
 *
 * When HAVE_CURL is not defined, logs to stderr and returns -1.
 *
 * @param config Exporter configuration.  config->enabled must be non-zero.
 * @param trace  Completed trace to export.
 * @return 0 on success, -1 on error.
 */
int otlp_export_trace(const GV_OtlpConfig *config, const GV_QueryTrace *trace);

/**
 * @brief Export database metrics to the configured OTLP endpoint via HTTP POST.
 *
 * When HAVE_CURL is not defined, logs to stderr and returns -1.
 *
 * @param config Exporter configuration.  config->enabled must be non-zero.
 * @param db     Database to sample metrics from.
 * @return 0 on success, -1 on error.
 */
int otlp_export_metrics(const GV_OtlpConfig *config, const struct GV_Database *db);

/**
 * @brief Configure the OTLP traces endpoint on a database.
 *
 * Sets config.endpoint and config.enabled = 1 when @p endpoint is non-NULL
 * and non-empty; sets config.enabled = 0 when @p endpoint is NULL or empty.
 *
 * @param db       Database to configure.
 * @param endpoint OTLP traces endpoint URL, or NULL/empty to disable.
 */
void gv_db_set_otlp_endpoint(struct GV_Database *db, const char *endpoint);

/**
 * @brief Immediately flush a metrics export to the OTLP endpoint.
 *
 * @param db Database whose metrics to export.
 * @return 0 on success, -1 if OTLP is not configured or export fails.
 */
int gv_db_otlp_flush(struct GV_Database *db);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_OTLP_H */
