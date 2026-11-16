#ifndef GIGAVECTOR_CORE_HTTP_CLIENT_H
#define GIGAVECTOR_CORE_HTTP_CLIENT_H

#include <stddef.h>

/*
 * Shared libcurl HTTP client. Replaces the per-module curl-handle, write-
 * callback and response-buffer boilerplate that previously lived in webhook,
 * sso, otlp, llm, embedding and auto_embed. Callers still build their own
 * URLs, bodies, auth headers and response parsers; only the transport is here.
 *
 * Compiled with -DHAVE_CURL the request is performed with libcurl; otherwise
 * every request fails with GV_HTTP_ERR_NO_CURL (the stub path). All response
 * memory is gv_alloc'd and must be released with gv_free.
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GV_HTTP_GET,
    GV_HTTP_POST
} GV_HttpMethod;

typedef enum {
    GV_HTTP_OK = 0,        /**< transport completed; inspect response.status */
    GV_HTTP_ERR_BADARG,    /**< NULL request/response or missing URL */
    GV_HTTP_ERR_INIT,      /**< curl handle could not be created */
    GV_HTTP_ERR_TIMEOUT,   /**< request timed out */
    GV_HTTP_ERR_NETWORK,   /**< any other transport failure */
    GV_HTTP_ERR_MEMORY,    /**< response buffer allocation failed */
    GV_HTTP_ERR_NO_CURL    /**< built without libcurl */
} GV_HttpResult;

typedef struct {
    GV_HttpMethod method;
    const char   *url;
    const char *const *headers; /**< array of "Name: Value" strings */
    size_t        num_headers;
    const char   *body;         /**< request body (form-encoded or raw JSON) */
    size_t        body_len;     /**< body length in bytes */
    long          timeout_ms;   /**< total timeout in milliseconds (0 = none) */
    int           verify_tls;   /**< non-zero verifies peer + host certificates */
    int           follow_redirects; /**< non-zero follows 3xx Location headers */
    int           discard_body;     /**< non-zero drops the response body (no buffering) */
    size_t        max_response_bytes; /**< cap on buffered body (0 = default cap) */
} GV_HttpRequest;

typedef struct {
    long   status;    /**< HTTP status code (0 if transport failed) */
    char  *body;      /**< gv_alloc'd, NUL-terminated; NULL when discarded/failed */
    size_t body_len;  /**< body length in bytes */
} GV_HttpResponse;

/** Perform an HTTP request; returns GV_HTTP_OK when the transport completed. */
GV_HttpResult gv_http_perform(const GV_HttpRequest *request,
                              GV_HttpResponse *response);

/** Return non-zero when the HTTP client was compiled with libcurl support. */
int gv_http_available(void);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_CORE_HTTP_CLIENT_H */
