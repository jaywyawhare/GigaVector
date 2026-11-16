/**
 * @file http_client.c
 * @brief Shared libcurl HTTP client used by webhook, sso, otlp, llm,
 *        embedding and auto_embed.
 */

#include "core/http_client.h"
#include "core/memory.h"

#include <stdio.h>
#include <string.h>

#define GV_HTTP_DEFAULT_MAX_BYTES (16u * 1024u * 1024u)

#ifdef HAVE_CURL
#include <curl/curl.h>

/** Growable response accumulator, capped at max_size (aborts the transfer). */
typedef struct {
    char  *data;
    size_t size;
    size_t capacity;
    size_t max_size;
    int    oom;
} GV_HttpAccum;

/** CURLOPT_WRITEFUNCTION that appends into a gv_alloc'd GV_HttpAccum. */
static size_t http_accum_cb(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    GV_HttpAccum *buf = (GV_HttpAccum *)userp;

    if (buf->size + realsize > buf->max_size) {
        return 0;
    }
    if (buf->size + realsize >= buf->capacity) {
        size_t new_capacity = buf->capacity ? buf->capacity * 2 : 4096;
        if (new_capacity < buf->size + realsize + 1) {
            new_capacity = buf->size + realsize + 1;
        }
        if (new_capacity > buf->max_size) {
            new_capacity = buf->max_size + 1;
        }
        char *new_data = (char *)gv_realloc(buf->data, new_capacity);
        if (new_data == NULL) {
            buf->oom = 1;
            return 0;
        }
        buf->data = new_data;
        buf->capacity = new_capacity;
    }

    memcpy(buf->data + buf->size, contents, realsize);
    buf->size += realsize;
    buf->data[buf->size] = '\0';
    return realsize;
}

/** Discard callback for callers that do not read the response body. */
static size_t http_discard_cb(void *ptr, size_t size, size_t nmemb, void *ud) {
    (void)ptr; (void)ud;
    return size * nmemb;
}

GV_HttpResult gv_http_perform(const GV_HttpRequest *request, GV_HttpResponse *response) {
    if (!request || !request->url || !response) return GV_HTTP_ERR_BADARG;

    memset(response, 0, sizeof(*response));

    CURL *curl = curl_easy_init();
    if (!curl) return GV_HTTP_ERR_INIT;

    struct curl_slist *headers = NULL;
    for (size_t i = 0; i < request->num_headers; i++) {
        headers = curl_slist_append(headers, request->headers[i]);
    }

    GV_HttpAccum accum;
    memset(&accum, 0, sizeof(accum));
    accum.max_size = request->max_response_bytes ? request->max_response_bytes
                                                 : GV_HTTP_DEFAULT_MAX_BYTES;

    curl_easy_setopt(curl, CURLOPT_URL, request->url);
    if (headers) curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    if (request->method == GV_HTTP_POST) {
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)request->body_len);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request->body ? request->body : "");
    }
    if (request->timeout_ms > 0) {
        curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, request->timeout_ms);
    }
    if (request->follow_redirects) {
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    }

    /* Secure default: verify certificates unless the caller opts out. */
    if (request->verify_tls) {
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
    } else {
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
    }

    /* NOSIGNAL keeps timeouts thread-safe; webhook delivers from a thread pool. */
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

    if (request->discard_body) {
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, http_discard_cb);
    } else {
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, http_accum_cb);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &accum);
    }

    CURLcode res = curl_easy_perform(curl);

    curl_slist_free_all(headers);

    if (res != CURLE_OK) {
        gv_free(accum.data);
        curl_easy_cleanup(curl);
        if (accum.oom) return GV_HTTP_ERR_MEMORY;
        return res == CURLE_OPERATION_TIMEDOUT ? GV_HTTP_ERR_TIMEOUT
                                               : GV_HTTP_ERR_NETWORK;
    }

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(curl);

    response->status = http_code;
    if (!request->discard_body) {
        /* Guarantee a valid NUL-terminated buffer even for an empty body so
         * callers can strstr/parse the result without a NULL check. */
        if (!accum.data) {
            accum.data = (char *)gv_alloc(1);
            if (!accum.data) return GV_HTTP_ERR_MEMORY;
            accum.data[0] = '\0';
        }
        response->body = accum.data;
        response->body_len = accum.size;
    }
    return GV_HTTP_OK;
}

int gv_http_available(void) {
    return 1;
}

#else /* !HAVE_CURL -- stub path */

GV_HttpResult gv_http_perform(const GV_HttpRequest *request, GV_HttpResponse *response) {
    if (response) memset(response, 0, sizeof(*response));
    fprintf(stderr, "[http] request to %s unavailable: compiled without libcurl\n",
            (request && request->url) ? request->url : "(null)");
    return GV_HTTP_ERR_NO_CURL;
}

int gv_http_available(void) {
    return 0;
}

#endif /* HAVE_CURL */
