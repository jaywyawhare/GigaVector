#ifndef GIGAVECTOR_MULTIMODAL_HTTP_BUFFER_H
#define GIGAVECTOR_MULTIMODAL_HTTP_BUFFER_H

#include <stddef.h>
#include <string.h>
#include "core/memory.h"

/*
 * Growable buffer that accumulates a libcurl response body, capped at max_size
 * (set per-caller since embedding batch responses and LLM responses use
 * different limits). Shared by the multimodal HTTP clients.
 */
typedef struct {
    char  *data;
    size_t size;
    size_t capacity;
    size_t max_size;
} GV_HttpBuffer;

/* libcurl CURLOPT_WRITEFUNCTION callback; @p userp must be a GV_HttpBuffer*
   with data/capacity/max_size initialised. Returns 0 (aborting the transfer)
   on OOM or when the response would exceed max_size. */
static inline size_t gv_http_write_callback(void *contents, size_t size,
                                            size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    GV_HttpBuffer *buf = (GV_HttpBuffer *)userp;

    if (buf->size + realsize > buf->max_size) {
        return 0;
    }
    if (buf->size + realsize >= buf->capacity) {
        size_t new_capacity = buf->capacity * 2;
        if (new_capacity < buf->size + realsize + 1) {
            new_capacity = buf->size + realsize + 1;
        }
        if (new_capacity > buf->max_size) {
            new_capacity = buf->max_size;
        }
        char *new_data = (char *)gv_realloc(buf->data, new_capacity);
        if (new_data == NULL) {
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

#endif /* GIGAVECTOR_MULTIMODAL_HTTP_BUFFER_H */
