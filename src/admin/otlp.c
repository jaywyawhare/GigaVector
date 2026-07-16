#define _POSIX_C_SOURCE 200809L

/**
 * @file otlp.c
 * @brief OpenTelemetry Protocol (OTLP) / HTTP JSON exporter.
 *
 * Serialises GV_QueryTrace spans and GV_Database metrics to OTLP/HTTP JSON
 * and POSTs them to configured endpoints.  Actual HTTP delivery requires
 * libcurl; without it the functions log to stderr and return -1 (matching
 * the fallback pattern in webhook.c).
 */

#include "admin/otlp.h"
#include "storage/database.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>
#include <inttypes.h>

/* -------------------------------------------------------------------------
 * Internal helpers
 * ---------------------------------------------------------------------- */

/**
 * Encode a uint64 value as a 32-character zero-padded hex string (16 bytes
 * represented as hex — the minimum trace-id width for OTLP).
 */
static void u64_to_hex32(uint64_t val, char out[33]) {
    /* Pad to 32 hex chars (16 bytes) by zero-prefixing the 16-char form. */
    snprintf(out, 33, "%016" PRIx64 "%016" PRIx64, (uint64_t)0, val);
}

/**
 * Encode a uint64 value as a 16-character zero-padded hex string (8 bytes
 * — the span-id width for OTLP).
 */
static void u64_to_hex16(uint64_t val, char out[17]) {
    snprintf(out, 17, "%016" PRIx64, val);
}

/**
 * Append @p src to @p *buf growing the allocation as needed.
 * @p *buf and @p *capacity are updated in place.
 * @p *len tracks the current string length.
 * Returns 0 on success, -1 on allocation failure.
 */
static int buf_append(char **buf, size_t *len, size_t *capacity, const char *src) {
    size_t src_len = strlen(src);
    if (*len + src_len + 1 > *capacity) {
        size_t new_cap = (*capacity) * 2;
        if (new_cap < *len + src_len + 128) {
            new_cap = *len + src_len + 128;
        }
        char *nb = realloc(*buf, new_cap);
        if (!nb) return -1;
        *buf = nb;
        *capacity = new_cap;
    }
    memcpy(*buf + *len, src, src_len);
    *len += src_len;
    (*buf)[*len] = '\0';
    return 0;
}

/** Append a printf-formatted string to the dynamic buffer. */
static int buf_appendf(char **buf, size_t *len, size_t *capacity,
                       const char *fmt, ...) {
    char tmp[512];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n < 0 || (size_t)n >= sizeof(tmp)) {
        /* Fall back to a large stack buffer. */
        char *big = malloc((size_t)n + 1);
        if (!big) return -1;
        va_start(ap, fmt);
        vsnprintf(big, (size_t)n + 1, fmt, ap);
        va_end(ap);
        int rc = buf_append(buf, len, capacity, big);
        free(big);
        return rc;
    }
    return buf_append(buf, len, capacity, tmp);
}

/* -------------------------------------------------------------------------
 * JSON builders
 * ---------------------------------------------------------------------- */

char *otlp_build_trace_json(const GV_QueryTrace *trace) {
    if (!trace) return NULL;

    size_t cap = 2048;
    size_t len = 0;
    char *buf = malloc(cap);
    if (!buf) return NULL;
    buf[0] = '\0';

    /* OTLP ExportTraceServiceRequest (JSON mapping) */
    if (buf_append(&buf, &len, &cap,
                   "{\"resourceSpans\":[{\"resource\":{\"attributes\":["
                   "{\"key\":\"service.name\","
                   "\"value\":{\"stringValue\":\"gigavector\"}}]},"
                   "\"scopeSpans\":[{\"scope\":{\"name\":\"gigavector\","
                   "\"version\":\"1.0\"},"
                   "\"spans\":[") != 0)
        goto oom;

    char trace_id_hex[33];
    u64_to_hex32(trace->trace_id, trace_id_hex);

    for (size_t i = 0; i < trace->span_count; i++) {
        const GV_TraceSpan *span = &trace->spans[i];

        char span_id_hex[17];
        u64_to_hex16(trace->trace_id ^ ((uint64_t)(i + 1) * 0x9e3779b97f4a7c15ULL),
                     span_id_hex);

        uint64_t start_ns = span->start_us * 1000ULL;
        uint64_t end_ns   = start_ns + span->duration_us * 1000ULL;

        if (i > 0) {
            if (buf_append(&buf, &len, &cap, ",") != 0) goto oom;
        }

        /* Escape span name */
        const char *name = span->name ? span->name : "";
        char escaped[512];
        size_t eo = 0;
        for (const char *p = name; *p && eo + 2 < sizeof(escaped); p++) {
            if (*p == '"' || *p == '\\') escaped[eo++] = '\\';
            escaped[eo++] = *p;
        }
        escaped[eo] = '\0';

        if (buf_appendf(&buf, &len, &cap,
                        "{\"traceId\":\"%s\","
                        "\"spanId\":\"%s\","
                        "\"name\":\"%s\","
                        "\"startTimeUnixNano\":\"%" PRIu64 "\","
                        "\"endTimeUnixNano\":\"%" PRIu64 "\","
                        "\"kind\":2}",
                        trace_id_hex, span_id_hex, escaped,
                        start_ns, end_ns) != 0)
            goto oom;
    }

    if (buf_append(&buf, &len, &cap, "]}]}]}") != 0) goto oom;

    return buf;

oom:
    free(buf);
    return NULL;
}

char *otlp_build_metrics_json(const struct GV_Database *db) {
    if (!db) return NULL;

    /*
     * Snapshot all observability fields into locals under
     * observability_mutex so we get a consistent, non-torn view even while
     * db_record_latency() concurrently mutates them. The histogram buckets
     * and boundaries are deep-copied because they are read in loops below
     * (outside the lock). Uses the same const-cast-to-lock pattern as
     * db_stats.c.
     */
    pthread_mutex_t *lock = (pthread_mutex_t *)&db->observability_mutex;
    pthread_mutex_lock(lock);

    uint64_t snap_total_inserts  = db->total_inserts;
    uint64_t snap_total_queries  = db->total_queries;
    double   snap_current_qps    = db->current_qps;
    double   snap_current_ips    = db->current_ips;

    size_t   hist_bucket_count   = db->search_latency_hist.bucket_count;
    uint64_t hist_total_samples  = db->search_latency_hist.total_samples;
    uint64_t hist_sum_latency_us = db->search_latency_hist.sum_latency_us;

    uint64_t *hist_buckets    = NULL;
    double   *hist_boundaries = NULL;
    int hist_valid = 0;
    if (hist_bucket_count > 0 &&
        db->search_latency_hist.buckets != NULL &&
        db->search_latency_hist.bucket_boundaries != NULL) {
        hist_buckets    = malloc(hist_bucket_count * sizeof(uint64_t));
        hist_boundaries = malloc(hist_bucket_count * sizeof(double));
        if (hist_buckets && hist_boundaries) {
            memcpy(hist_buckets, db->search_latency_hist.buckets,
                   hist_bucket_count * sizeof(uint64_t));
            memcpy(hist_boundaries, db->search_latency_hist.bucket_boundaries,
                   hist_bucket_count * sizeof(double));
            hist_valid = 1;
        }
    }

    pthread_mutex_unlock(lock);

    if (hist_bucket_count > 0 && !hist_valid) {
        /* Copy allocation failed; free partials and skip histogram export. */
        free(hist_buckets);
        free(hist_boundaries);
        hist_buckets = NULL;
        hist_boundaries = NULL;
    }

    size_t cap = 2048;
    size_t len = 0;
    char *buf = malloc(cap);
    if (!buf) goto oom;
    buf[0] = '\0';

    /* OTLP ExportMetricsServiceRequest (JSON mapping) */
    if (buf_append(&buf, &len, &cap,
                   "{\"resourceMetrics\":[{\"resource\":{\"attributes\":["
                   "{\"key\":\"service.name\","
                   "\"value\":{\"stringValue\":\"gigavector\"}}]},"
                   "\"scopeMetrics\":[{\"scope\":{\"name\":\"gigavector\"},"
                   "\"metrics\":[") != 0)
        goto oom;

    /* total_inserts gauge */
    if (buf_appendf(&buf, &len, &cap,
                    "{\"name\":\"gigavector.total_inserts\","
                    "\"description\":\"Total vectors inserted\","
                    "\"gauge\":{\"dataPoints\":["
                    "{\"asInt\":\"%" PRIu64 "\"}]}}",
                    snap_total_inserts) != 0)
        goto oom;

    /* total_queries gauge */
    if (buf_appendf(&buf, &len, &cap,
                    ",{\"name\":\"gigavector.total_queries\","
                    "\"description\":\"Total search queries\","
                    "\"gauge\":{\"dataPoints\":["
                    "{\"asInt\":\"%" PRIu64 "\"}]}}",
                    snap_total_queries) != 0)
        goto oom;

    /* current_qps gauge */
    if (buf_appendf(&buf, &len, &cap,
                    ",{\"name\":\"gigavector.current_qps\","
                    "\"description\":\"Current queries per second\","
                    "\"gauge\":{\"dataPoints\":["
                    "{\"asDouble\":%.4f}]}}",
                    snap_current_qps) != 0)
        goto oom;

    /* current_ips gauge */
    if (buf_appendf(&buf, &len, &cap,
                    ",{\"name\":\"gigavector.current_ips\","
                    "\"description\":\"Current inserts per second\","
                    "\"gauge\":{\"dataPoints\":["
                    "{\"asDouble\":%.4f}]}}",
                    snap_current_ips) != 0)
        goto oom;

    /*
     * search latency histogram
     *
     * OTLP requires len(bucketCounts) == len(explicitBounds) + 1, where the
     * final bucketCounts entry is the implicit (+Inf) overflow bucket. Our
     * histogram has `hist_bucket_count` non-cumulative bucket counts, where
     * buckets[i] counts samples in (boundaries[i-1], boundaries[i]] and the
     * last bucket also absorbs anything above boundaries[count-2]. We
     * therefore export the first (count-1) boundaries as explicitBounds and
     * all `count` bucket counts as bucketCounts, so the last count serves as
     * the +Inf overflow bucket. This satisfies the length invariant and the
     * non-cumulative bucketCounts semantics OTLP expects.
     */
    if (hist_valid && hist_bucket_count > 0) {

        if (buf_append(&buf, &len, &cap,
                       ",{\"name\":\"gigavector.search_latency_us\","
                       "\"description\":\"Search latency histogram in microseconds\","
                       "\"histogram\":{\"aggregationTemporality\":2,"
                       "\"dataPoints\":[{\"count\":\"") != 0)
            goto oom;

        if (buf_appendf(&buf, &len, &cap,
                        "%" PRIu64 "\",\"sum\":%.4f,\"bucketCounts\":[",
                        hist_total_samples,
                        (double)hist_sum_latency_us) != 0)
            goto oom;

        /* bucketCounts: all `hist_bucket_count` counts (last == +Inf bucket) */
        for (size_t i = 0; i < hist_bucket_count; i++) {
            if (i > 0) {
                if (buf_append(&buf, &len, &cap, ",") != 0) goto oom;
            }
            if (buf_appendf(&buf, &len, &cap,
                            "\"%" PRIu64 "\"",
                            hist_buckets[i]) != 0)
                goto oom;
        }

        /* explicitBounds: first (hist_bucket_count - 1) boundaries */
        if (buf_append(&buf, &len, &cap, "],\"explicitBounds\":[") != 0)
            goto oom;

        for (size_t i = 0; i + 1 < hist_bucket_count; i++) {
            if (i > 0) {
                if (buf_append(&buf, &len, &cap, ",") != 0) goto oom;
            }
            if (buf_appendf(&buf, &len, &cap, "%.4f", hist_boundaries[i]) != 0)
                goto oom;
        }

        if (buf_append(&buf, &len, &cap, "]}]}}") != 0) goto oom;
    }

    if (buf_append(&buf, &len, &cap, "]}]}]}") != 0) goto oom;

    free(hist_buckets);
    free(hist_boundaries);
    return buf;

oom:
    free(hist_buckets);
    free(hist_boundaries);
    free(buf);
    return NULL;
}

/* -------------------------------------------------------------------------
 * HTTP delivery (libcurl or stub)
 * ---------------------------------------------------------------------- */

#ifdef HAVE_CURL
#include <curl/curl.h>

static size_t otlp_discard_cb(void *ptr, size_t size, size_t nmemb, void *ud) {
    (void)ptr; (void)ud;
    return size * nmemb;
}

static int otlp_post_json(const char *url, const char *json, int timeout_ms) {
    if (!url || !url[0] || !json) return -1;

    CURL *curl = curl_easy_init();
    if (!curl) return -1;

    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, (long)(timeout_ms > 0 ? timeout_ms : 5000));
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, otlp_discard_cb);

    CURLcode res = curl_easy_perform(curl);
    int success = 0;
    if (res == CURLE_OK) {
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        success = (http_code >= 200 && http_code < 300);
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    return success ? 0 : -1;
}

#else /* !HAVE_CURL */

static int otlp_post_json(const char *url, const char *json, int timeout_ms) {
    (void)timeout_ms;
    fprintf(stderr, "[otlp] stub: would POST to %s (body=%zu bytes)\n",
            url ? url : "(null)", json ? strlen(json) : 0);
    return -1;
}

#endif /* HAVE_CURL */

/* -------------------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------------- */

int otlp_export_trace(const GV_OtlpConfig *config, const GV_QueryTrace *trace) {
    if (!config || !config->enabled || !config->endpoint[0]) return -1;
    if (!trace) return -1;

    char *json = otlp_build_trace_json(trace);
    if (!json) return -1;

    int rc = otlp_post_json(config->endpoint, json, 5000);
    free(json);
    return rc;
}

int otlp_export_metrics(const GV_OtlpConfig *config, const struct GV_Database *db) {
    if (!config || !config->enabled) return -1;
    /* Use metrics_endpoint if set, fall back to traces endpoint */
    const char *url = config->metrics_endpoint[0] ? config->metrics_endpoint
                                                   : config->endpoint;
    if (!url || !url[0]) return -1;
    if (!db) return -1;

    char *json = otlp_build_metrics_json(db);
    if (!json) return -1;

    int rc = otlp_post_json(url, json, 5000);
    free(json);
    return rc;
}

void gv_db_set_otlp_endpoint(struct GV_Database *db, const char *endpoint) {
    if (!db) return;

    if (!endpoint || !endpoint[0]) {
        db->otlp_config.enabled = 0;
        db->otlp_config.endpoint[0] = '\0';
        return;
    }

    strncpy(db->otlp_config.endpoint, endpoint,
            sizeof(db->otlp_config.endpoint) - 1);
    db->otlp_config.endpoint[sizeof(db->otlp_config.endpoint) - 1] = '\0';
    db->otlp_config.enabled = 1;
}

int gv_db_otlp_flush(struct GV_Database *db) {
    if (!db) return -1;
    return otlp_export_metrics(&db->otlp_config, db);
}
