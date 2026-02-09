#ifndef GIGAVECTOR_GV_WEBHOOK_H
#define GIGAVECTOR_GV_WEBHOOK_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GV_EVENT_INSERT = 1,
    GV_EVENT_UPDATE = 2,
    GV_EVENT_DELETE = 4,
    GV_EVENT_ALL = 7
} GV_EventType;

typedef struct {
    GV_EventType event_type;
    size_t vector_index;
    uint64_t timestamp;
    const char *collection;   /* NULL for default collection */
} GV_Event;

typedef struct {
    char *url;                /* Webhook URL to POST to */
    GV_EventType event_mask;  /* Which events to trigger on */
    char *secret;             /* Optional HMAC secret for signing */
    int max_retries;          /* Max retry attempts (default: 3) */
    int timeout_ms;           /* HTTP timeout (default: 5000) */
    int active;               /* 1 = active, 0 = paused */
} GV_WebhookConfig;

/* Change stream callback */
typedef void (*GV_ChangeCallback)(const GV_Event *event, void *user_data);

typedef struct GV_WebhookManager GV_WebhookManager;

/* Lifecycle */
GV_WebhookManager *gv_webhook_create(void);
void gv_webhook_destroy(GV_WebhookManager *mgr);

/* Register/unregister webhooks */
int gv_webhook_register(GV_WebhookManager *mgr, const char *webhook_id, const GV_WebhookConfig *config);
int gv_webhook_unregister(GV_WebhookManager *mgr, const char *webhook_id);
int gv_webhook_pause(GV_WebhookManager *mgr, const char *webhook_id);
int gv_webhook_resume(GV_WebhookManager *mgr, const char *webhook_id);

/* List webhooks */
int gv_webhook_list(const GV_WebhookManager *mgr, char ***out_ids, size_t *out_count);
void gv_webhook_free_list(char **ids, size_t count);

/* Fire event (called internally when mutations happen) */
int gv_webhook_fire(GV_WebhookManager *mgr, const GV_Event *event);

/* Change stream: register callback for real-time events */
int gv_webhook_subscribe(GV_WebhookManager *mgr, GV_EventType mask, GV_ChangeCallback cb, void *user_data);
int gv_webhook_unsubscribe(GV_WebhookManager *mgr, GV_ChangeCallback cb);

/* Stats */
typedef struct {
    uint64_t events_fired;
    uint64_t webhooks_delivered;
    uint64_t webhooks_failed;
    uint64_t callbacks_invoked;
} GV_WebhookStats;

int gv_webhook_get_stats(const GV_WebhookManager *mgr, GV_WebhookStats *stats);

#ifdef __cplusplus
}
#endif
#endif
