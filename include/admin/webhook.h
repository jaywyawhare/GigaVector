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

GV_WebhookManager *webhook_create(void);
/**
 * @brief Destroy an instance and free associated resources.
 *
 * @param mgr Manager instance.
 */
void webhook_destroy(GV_WebhookManager *mgr);

/**
 * @brief Perform the operation.
 *
 * @param mgr Manager instance.
 * @param webhook_id Identifier.
 * @param config Configuration to apply/output.
 * @return 0 on success, -1 on error.
 */
int webhook_register(GV_WebhookManager *mgr, const char *webhook_id, const GV_WebhookConfig *config);
/**
 * @brief Perform the operation.
 *
 * @param mgr Manager instance.
 * @param webhook_id Identifier.
 * @return 0 on success, -1 on error.
 */
int webhook_unregister(GV_WebhookManager *mgr, const char *webhook_id);
/**
 * @brief Perform the operation.
 *
 * @param mgr Manager instance.
 * @param webhook_id Identifier.
 * @return 0 on success, -1 on error.
 */
int webhook_pause(GV_WebhookManager *mgr, const char *webhook_id);
/**
 * @brief Perform the operation.
 *
 * @param mgr Manager instance.
 * @param webhook_id Identifier.
 * @return 0 on success, -1 on error.
 */
int webhook_resume(GV_WebhookManager *mgr, const char *webhook_id);

/**
 * @brief List items.
 *
 * @param mgr Manager instance.
 * @param out_ids Output id list (allocated by callee).
 * @param out_count Output item count.
 * @return 0 on success, -1 on error.
 */
int webhook_list(const GV_WebhookManager *mgr, char ***out_ids, size_t *out_count);
/**
 * @brief List items.
 *
 * @param ids ids.
 * @param count Number of items.
 */
void webhook_free_list(char **ids, size_t count);

/**
 * @brief Perform the operation.
 *
 * @param mgr Manager instance.
 * @param event event.
 * @return 0 on success, -1 on error.
 */
int webhook_fire(GV_WebhookManager *mgr, const GV_Event *event);

/**
 * @brief Perform the operation.
 *
 * @param mgr Manager instance.
 * @param mask Event mask.
 * @param cb Callback function.
 * @param user_data Opaque user pointer passed to callback.
 * @return 0 on success, -1 on error.
 */
int webhook_subscribe(GV_WebhookManager *mgr, GV_EventType mask, GV_ChangeCallback cb, void *user_data);
/**
 * @brief Perform the operation.
 *
 * @param mgr Manager instance.
 * @param cb Callback function.
 * @return 0 on success, -1 on error.
 */
int webhook_unsubscribe(GV_WebhookManager *mgr, GV_ChangeCallback cb);

typedef struct {
    uint64_t events_fired;
    uint64_t webhooks_delivered;
    uint64_t webhooks_failed;
    uint64_t callbacks_invoked;
} GV_WebhookStats;

/**
 * @brief Retrieve statistics.
 *
 * @param mgr Manager instance.
 * @param stats Output statistics structure.
 * @return 0 on success, -1 on error.
 */
int webhook_get_stats(const GV_WebhookManager *mgr, GV_WebhookStats *stats);

#ifdef __cplusplus
}
#endif
#endif
