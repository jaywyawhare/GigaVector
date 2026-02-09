#define _POSIX_C_SOURCE 200809L

/**
 * @file gv_webhook.c
 * @brief Webhook and change stream implementation.
 *
 * Provides event notification for insert/update/delete mutations via
 * HTTP webhooks (with HMAC-SHA256 signing and retry logic) and
 * in-process change stream callbacks.
 *
 * Note: Actual HTTP delivery requires libcurl.  This implementation
 * provides the full API with a libcurl-based POST path (compiled in
 * when CURL is available) and a stub fallback.
 */

#include "gigavector/gv_webhook.h"
#include "gigavector/gv_crypto.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

/* ============================================================================
 * Internal Constants
 * ============================================================================ */

#define MAX_WEBHOOKS       64
#define MAX_SUBSCRIBERS    32
#define WORK_QUEUE_SIZE    256
#define DELIVERY_THREADS   2
#define DEFAULT_MAX_RETRIES 3
#define DEFAULT_TIMEOUT_MS  5000

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

typedef struct {
    char *id;
    GV_WebhookConfig config;
    int in_use;
} WebhookEntry;

typedef struct {
    GV_ChangeCallback cb;
    void *user_data;
    GV_EventType mask;
    int in_use;
} SubscriberEntry;

/** A single delivery work item placed on the queue. */
typedef struct {
    char *url;                /* Destination URL (owned, must free) */
    char *json_body;          /* JSON payload  (owned, must free) */
    char *secret;             /* HMAC secret   (owned, may be NULL) */
    int max_retries;
    int timeout_ms;
} DeliveryWork;

struct GV_WebhookManager {
    /* Registered webhooks */
    WebhookEntry webhooks[MAX_WEBHOOKS];
    size_t webhook_count;

    /* Change stream subscribers */
    SubscriberEntry subscribers[MAX_SUBSCRIBERS];
    size_t subscriber_count;

    /* Statistics */
    GV_WebhookStats stats;

    /* Work queue for background delivery */
    DeliveryWork queue[WORK_QUEUE_SIZE];
    size_t queue_head;
    size_t queue_tail;
    size_t queue_size;

    /* Thread pool */
    pthread_t workers[DELIVERY_THREADS];
    int workers_running;
    int stop_requested;

    /* Synchronization */
    pthread_mutex_t mutex;
    pthread_cond_t queue_cond;
};

/* ============================================================================
 * Forward Declarations
 * ============================================================================ */

static void *delivery_thread_func(void *arg);
static int  enqueue_delivery(GV_WebhookManager *mgr, DeliveryWork *work);
static void free_delivery_work(DeliveryWork *work);
static char *build_json_payload(const GV_Event *event);
static int  compute_signature(const char *secret, const char *body,
                               char *out_hex, size_t out_len);
static int  deliver_webhook(const DeliveryWork *work);

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

GV_WebhookManager *gv_webhook_create(void) {
    GV_WebhookManager *mgr = calloc(1, sizeof(GV_WebhookManager));
    if (!mgr) return NULL;

    if (pthread_mutex_init(&mgr->mutex, NULL) != 0) {
        free(mgr);
        return NULL;
    }

    if (pthread_cond_init(&mgr->queue_cond, NULL) != 0) {
        pthread_mutex_destroy(&mgr->mutex);
        free(mgr);
        return NULL;
    }

    /* Start background delivery threads */
    mgr->stop_requested = 0;
    mgr->workers_running = 1;
    for (int i = 0; i < DELIVERY_THREADS; i++) {
        if (pthread_create(&mgr->workers[i], NULL, delivery_thread_func, mgr) != 0) {
            /* If we can't start all threads, stop those already started */
            mgr->stop_requested = 1;
            pthread_cond_broadcast(&mgr->queue_cond);
            for (int j = 0; j < i; j++) {
                pthread_join(mgr->workers[j], NULL);
            }
            mgr->workers_running = 0;
            pthread_cond_destroy(&mgr->queue_cond);
            pthread_mutex_destroy(&mgr->mutex);
            free(mgr);
            return NULL;
        }
    }

    return mgr;
}

void gv_webhook_destroy(GV_WebhookManager *mgr) {
    if (!mgr) return;

    /* Signal workers to stop */
    pthread_mutex_lock(&mgr->mutex);
    mgr->stop_requested = 1;
    pthread_cond_broadcast(&mgr->queue_cond);
    pthread_mutex_unlock(&mgr->mutex);

    /* Join worker threads */
    if (mgr->workers_running) {
        for (int i = 0; i < DELIVERY_THREADS; i++) {
            pthread_join(mgr->workers[i], NULL);
        }
        mgr->workers_running = 0;
    }

    /* Drain remaining work items */
    while (mgr->queue_size > 0) {
        DeliveryWork *w = &mgr->queue[mgr->queue_head];
        free_delivery_work(w);
        mgr->queue_head = (mgr->queue_head + 1) % WORK_QUEUE_SIZE;
        mgr->queue_size--;
    }

    /* Free webhook entries */
    for (size_t i = 0; i < MAX_WEBHOOKS; i++) {
        if (mgr->webhooks[i].in_use) {
            free(mgr->webhooks[i].id);
            free(mgr->webhooks[i].config.url);
            free(mgr->webhooks[i].config.secret);
        }
    }

    pthread_cond_destroy(&mgr->queue_cond);
    pthread_mutex_destroy(&mgr->mutex);
    free(mgr);
}

/* ============================================================================
 * Register / Unregister
 * ============================================================================ */

int gv_webhook_register(GV_WebhookManager *mgr, const char *webhook_id,
                         const GV_WebhookConfig *config) {
    if (!mgr || !webhook_id || !config || !config->url) return -1;

    pthread_mutex_lock(&mgr->mutex);

    /* Check for duplicate id */
    for (size_t i = 0; i < MAX_WEBHOOKS; i++) {
        if (mgr->webhooks[i].in_use &&
            strcmp(mgr->webhooks[i].id, webhook_id) == 0) {
            pthread_mutex_unlock(&mgr->mutex);
            return -1; /* already exists */
        }
    }

    /* Find free slot */
    int slot = -1;
    for (size_t i = 0; i < MAX_WEBHOOKS; i++) {
        if (!mgr->webhooks[i].in_use) {
            slot = (int)i;
            break;
        }
    }

    if (slot < 0) {
        pthread_mutex_unlock(&mgr->mutex);
        return -1; /* full */
    }

    WebhookEntry *entry = &mgr->webhooks[slot];
    entry->id = strdup(webhook_id);
    if (!entry->id) {
        pthread_mutex_unlock(&mgr->mutex);
        return -1;
    }

    entry->config.url = strdup(config->url);
    if (!entry->config.url) {
        free(entry->id);
        entry->id = NULL;
        pthread_mutex_unlock(&mgr->mutex);
        return -1;
    }

    entry->config.event_mask = config->event_mask ? config->event_mask : GV_EVENT_ALL;
    entry->config.secret = config->secret ? strdup(config->secret) : NULL;
    entry->config.max_retries = config->max_retries > 0 ? config->max_retries : DEFAULT_MAX_RETRIES;
    entry->config.timeout_ms = config->timeout_ms > 0 ? config->timeout_ms : DEFAULT_TIMEOUT_MS;
    entry->config.active = config->active;
    entry->in_use = 1;
    mgr->webhook_count++;

    pthread_mutex_unlock(&mgr->mutex);
    return 0;
}

int gv_webhook_unregister(GV_WebhookManager *mgr, const char *webhook_id) {
    if (!mgr || !webhook_id) return -1;

    pthread_mutex_lock(&mgr->mutex);

    for (size_t i = 0; i < MAX_WEBHOOKS; i++) {
        if (mgr->webhooks[i].in_use &&
            strcmp(mgr->webhooks[i].id, webhook_id) == 0) {
            free(mgr->webhooks[i].id);
            free(mgr->webhooks[i].config.url);
            free(mgr->webhooks[i].config.secret);
            memset(&mgr->webhooks[i], 0, sizeof(WebhookEntry));
            mgr->webhook_count--;
            pthread_mutex_unlock(&mgr->mutex);
            return 0;
        }
    }

    pthread_mutex_unlock(&mgr->mutex);
    return -1; /* not found */
}

int gv_webhook_pause(GV_WebhookManager *mgr, const char *webhook_id) {
    if (!mgr || !webhook_id) return -1;

    pthread_mutex_lock(&mgr->mutex);

    for (size_t i = 0; i < MAX_WEBHOOKS; i++) {
        if (mgr->webhooks[i].in_use &&
            strcmp(mgr->webhooks[i].id, webhook_id) == 0) {
            mgr->webhooks[i].config.active = 0;
            pthread_mutex_unlock(&mgr->mutex);
            return 0;
        }
    }

    pthread_mutex_unlock(&mgr->mutex);
    return -1;
}

int gv_webhook_resume(GV_WebhookManager *mgr, const char *webhook_id) {
    if (!mgr || !webhook_id) return -1;

    pthread_mutex_lock(&mgr->mutex);

    for (size_t i = 0; i < MAX_WEBHOOKS; i++) {
        if (mgr->webhooks[i].in_use &&
            strcmp(mgr->webhooks[i].id, webhook_id) == 0) {
            mgr->webhooks[i].config.active = 1;
            pthread_mutex_unlock(&mgr->mutex);
            return 0;
        }
    }

    pthread_mutex_unlock(&mgr->mutex);
    return -1;
}

/* ============================================================================
 * List Webhooks
 * ============================================================================ */

int gv_webhook_list(const GV_WebhookManager *mgr, char ***out_ids, size_t *out_count) {
    if (!mgr || !out_ids || !out_count) return -1;

    /* We cast away const to lock; the logical state is not modified. */
    GV_WebhookManager *m = (GV_WebhookManager *)mgr;

    pthread_mutex_lock(&m->mutex);

    size_t count = m->webhook_count;
    if (count == 0) {
        *out_ids = NULL;
        *out_count = 0;
        pthread_mutex_unlock(&m->mutex);
        return 0;
    }

    char **ids = calloc(count, sizeof(char *));
    if (!ids) {
        pthread_mutex_unlock(&m->mutex);
        return -1;
    }

    size_t idx = 0;
    for (size_t i = 0; i < MAX_WEBHOOKS && idx < count; i++) {
        if (m->webhooks[i].in_use) {
            ids[idx] = strdup(m->webhooks[i].id);
            if (!ids[idx]) {
                /* Roll back */
                for (size_t j = 0; j < idx; j++) free(ids[j]);
                free(ids);
                pthread_mutex_unlock(&m->mutex);
                return -1;
            }
            idx++;
        }
    }

    *out_ids = ids;
    *out_count = count;

    pthread_mutex_unlock(&m->mutex);
    return 0;
}

void gv_webhook_free_list(char **ids, size_t count) {
    if (!ids) return;
    for (size_t i = 0; i < count; i++) {
        free(ids[i]);
    }
    free(ids);
}

/* ============================================================================
 * Change Stream Subscribe / Unsubscribe
 * ============================================================================ */

int gv_webhook_subscribe(GV_WebhookManager *mgr, GV_EventType mask,
                          GV_ChangeCallback cb, void *user_data) {
    if (!mgr || !cb) return -1;

    pthread_mutex_lock(&mgr->mutex);

    int slot = -1;
    for (size_t i = 0; i < MAX_SUBSCRIBERS; i++) {
        if (!mgr->subscribers[i].in_use) {
            slot = (int)i;
            break;
        }
    }

    if (slot < 0) {
        pthread_mutex_unlock(&mgr->mutex);
        return -1; /* full */
    }

    SubscriberEntry *entry = &mgr->subscribers[slot];
    entry->cb = cb;
    entry->user_data = user_data;
    entry->mask = mask ? mask : GV_EVENT_ALL;
    entry->in_use = 1;
    mgr->subscriber_count++;

    pthread_mutex_unlock(&mgr->mutex);
    return 0;
}

int gv_webhook_unsubscribe(GV_WebhookManager *mgr, GV_ChangeCallback cb) {
    if (!mgr || !cb) return -1;

    pthread_mutex_lock(&mgr->mutex);

    for (size_t i = 0; i < MAX_SUBSCRIBERS; i++) {
        if (mgr->subscribers[i].in_use && mgr->subscribers[i].cb == cb) {
            memset(&mgr->subscribers[i], 0, sizeof(SubscriberEntry));
            mgr->subscriber_count--;
            pthread_mutex_unlock(&mgr->mutex);
            return 0;
        }
    }

    pthread_mutex_unlock(&mgr->mutex);
    return -1; /* not found */
}

/* ============================================================================
 * Fire Event
 * ============================================================================ */

int gv_webhook_fire(GV_WebhookManager *mgr, const GV_Event *event) {
    if (!mgr || !event) return -1;

    char *json = build_json_payload(event);
    if (!json) return -1;

    pthread_mutex_lock(&mgr->mutex);
    mgr->stats.events_fired++;

    /* --- Webhooks: enqueue background delivery for each matching hook --- */
    for (size_t i = 0; i < MAX_WEBHOOKS; i++) {
        WebhookEntry *wh = &mgr->webhooks[i];
        if (!wh->in_use || !wh->config.active) continue;
        if (!(wh->config.event_mask & (int)event->event_type)) continue;

        DeliveryWork work;
        memset(&work, 0, sizeof(work));
        work.url = strdup(wh->config.url);
        work.json_body = strdup(json);
        work.secret = wh->config.secret ? strdup(wh->config.secret) : NULL;
        work.max_retries = wh->config.max_retries;
        work.timeout_ms = wh->config.timeout_ms;

        if (!work.url || !work.json_body) {
            free(work.url);
            free(work.json_body);
            free(work.secret);
            continue;
        }

        if (enqueue_delivery(mgr, &work) != 0) {
            free_delivery_work(&work);
        }
    }

    /* --- Change stream: invoke matching callbacks synchronously --- */
    for (size_t i = 0; i < MAX_SUBSCRIBERS; i++) {
        SubscriberEntry *sub = &mgr->subscribers[i];
        if (!sub->in_use) continue;
        if (!(sub->mask & (int)event->event_type)) continue;

        mgr->stats.callbacks_invoked++;

        /* Unlock while calling user code to avoid deadlock */
        GV_ChangeCallback cb = sub->cb;
        void *ud = sub->user_data;
        pthread_mutex_unlock(&mgr->mutex);

        cb(event, ud);

        pthread_mutex_lock(&mgr->mutex);
    }

    pthread_mutex_unlock(&mgr->mutex);
    free(json);
    return 0;
}

/* ============================================================================
 * Stats
 * ============================================================================ */

int gv_webhook_get_stats(const GV_WebhookManager *mgr, GV_WebhookStats *stats) {
    if (!mgr || !stats) return -1;

    GV_WebhookManager *m = (GV_WebhookManager *)mgr;

    pthread_mutex_lock(&m->mutex);
    *stats = m->stats;
    pthread_mutex_unlock(&m->mutex);

    return 0;
}

/* ============================================================================
 * JSON Payload Builder
 * ============================================================================ */

static const char *event_type_string(GV_EventType type) {
    switch (type) {
        case GV_EVENT_INSERT: return "insert";
        case GV_EVENT_UPDATE: return "update";
        case GV_EVENT_DELETE: return "delete";
        default:              return "unknown";
    }
}

static char *build_json_payload(const GV_Event *event) {
    if (!event) return NULL;

    const char *collection = event->collection ? event->collection : "default";
    const char *type_str = event_type_string(event->event_type);

    /* Conservative upper bound for the formatted string */
    size_t needed = strlen(type_str) + strlen(collection) + 128;
    char *buf = malloc(needed);
    if (!buf) return NULL;

    snprintf(buf, needed,
             "{\"event\":\"%s\",\"index\":%zu,\"timestamp\":%llu,\"collection\":\"%s\"}",
             type_str,
             event->vector_index,
             (unsigned long long)event->timestamp,
             collection);

    return buf;
}

/* ============================================================================
 * HMAC Signature
 * ============================================================================ */

static int compute_signature(const char *secret, const char *body,
                              char *out_hex, size_t out_len) {
    if (!secret || !body || !out_hex || out_len < 65) return -1;

    unsigned char hmac[32];
    int rc = gv_crypto_hmac_sha256(
        (const unsigned char *)secret, strlen(secret),
        (const unsigned char *)body, strlen(body),
        hmac);

    if (rc != 0) return -1;

    /* Convert to hex string */
    static const char hex_digits[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        out_hex[i * 2]     = hex_digits[(hmac[i] >> 4) & 0x0f];
        out_hex[i * 2 + 1] = hex_digits[hmac[i] & 0x0f];
    }
    out_hex[64] = '\0';

    return 0;
}

/* ============================================================================
 * HTTP Delivery (libcurl or stub)
 * ============================================================================ */

#ifdef GV_HAVE_CURL
#include <curl/curl.h>

/* Discard response body */
static size_t discard_write_cb(void *ptr, size_t size, size_t nmemb, void *ud) {
    (void)ptr; (void)ud;
    return size * nmemb;
}

static int deliver_webhook(const DeliveryWork *work) {
    if (!work || !work->url || !work->json_body) return -1;

    CURL *curl = curl_easy_init();
    if (!curl) return -1;

    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    /* Compute HMAC signature if a secret is configured */
    char sig_hex[65];
    if (work->secret && work->secret[0]) {
        if (compute_signature(work->secret, work->json_body,
                               sig_hex, sizeof(sig_hex)) == 0) {
            char sig_header[128];
            snprintf(sig_header, sizeof(sig_header),
                     "X-Webhook-Signature: sha256=%s", sig_hex);
            headers = curl_slist_append(headers, sig_header);
        }
    }

    curl_easy_setopt(curl, CURLOPT_URL, work->url);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, work->json_body);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, (long)work->timeout_ms);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, discard_write_cb);

    int success = 0;
    int attempts = work->max_retries > 0 ? work->max_retries : DEFAULT_MAX_RETRIES;

    for (int attempt = 0; attempt < attempts; attempt++) {
        CURLcode res = curl_easy_perform(curl);
        if (res == CURLE_OK) {
            long http_code = 0;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
            if (http_code >= 200 && http_code < 300) {
                success = 1;
                break;
            }
        }

        /* Exponential backoff: 1s, 2s, 4s, ... */
        if (attempt + 1 < attempts) {
            unsigned int delay_sec = 1u << (unsigned)attempt;
            sleep(delay_sec);
        }
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    return success ? 0 : -1;
}

#else /* !GV_HAVE_CURL -- stub implementation */

static int deliver_webhook(const DeliveryWork *work) {
    if (!work || !work->url || !work->json_body) return -1;

    /* Compute signature so the code path is exercised even without curl */
    char sig_hex[65];
    if (work->secret && work->secret[0]) {
        compute_signature(work->secret, work->json_body,
                          sig_hex, sizeof(sig_hex));
    }

    /*
     * Without libcurl we cannot actually POST.  Log for diagnostics
     * and report failure so that stats reflect undelivered hooks.
     */
    fprintf(stderr, "[gv_webhook] stub: would POST to %s (body=%zu bytes)\n",
            work->url, strlen(work->json_body));

    /* Simulate retry delay to honour the API contract */
    int attempts = work->max_retries > 0 ? work->max_retries : DEFAULT_MAX_RETRIES;
    for (int attempt = 0; attempt < attempts; attempt++) {
        /* In a stub we just pretend each attempt fails instantly */
        if (attempt + 1 < attempts) {
            unsigned int delay_sec = 1u << (unsigned)attempt;
            sleep(delay_sec);
        }
    }

    return -1;
}

#endif /* GV_HAVE_CURL */

/* ============================================================================
 * Work Queue Helpers
 * ============================================================================ */

/**
 * Enqueue a delivery work item.  Caller must hold mgr->mutex.
 * Ownership of work's heap pointers transfers to the queue on success.
 */
static int enqueue_delivery(GV_WebhookManager *mgr, DeliveryWork *work) {
    if (mgr->queue_size >= WORK_QUEUE_SIZE) return -1; /* queue full */

    mgr->queue[mgr->queue_tail] = *work; /* shallow copy */
    mgr->queue_tail = (mgr->queue_tail + 1) % WORK_QUEUE_SIZE;
    mgr->queue_size++;

    pthread_cond_signal(&mgr->queue_cond);
    return 0;
}

static void free_delivery_work(DeliveryWork *work) {
    if (!work) return;
    free(work->url);
    free(work->json_body);
    free(work->secret);
    work->url = NULL;
    work->json_body = NULL;
    work->secret = NULL;
}

/* ============================================================================
 * Background Delivery Thread
 * ============================================================================ */

static void *delivery_thread_func(void *arg) {
    GV_WebhookManager *mgr = (GV_WebhookManager *)arg;

    for (;;) {
        pthread_mutex_lock(&mgr->mutex);

        /* Wait for work or shutdown */
        while (mgr->queue_size == 0 && !mgr->stop_requested) {
            pthread_cond_wait(&mgr->queue_cond, &mgr->mutex);
        }

        if (mgr->stop_requested && mgr->queue_size == 0) {
            pthread_mutex_unlock(&mgr->mutex);
            break;
        }

        /* Dequeue one item */
        DeliveryWork work = mgr->queue[mgr->queue_head];
        mgr->queue_head = (mgr->queue_head + 1) % WORK_QUEUE_SIZE;
        mgr->queue_size--;

        pthread_mutex_unlock(&mgr->mutex);

        /* Deliver outside the lock */
        int rc = deliver_webhook(&work);

        /* Update statistics */
        pthread_mutex_lock(&mgr->mutex);
        if (rc == 0) {
            mgr->stats.webhooks_delivered++;
        } else {
            mgr->stats.webhooks_failed++;
        }
        pthread_mutex_unlock(&mgr->mutex);

        free_delivery_work(&work);
    }

    return NULL;
}
