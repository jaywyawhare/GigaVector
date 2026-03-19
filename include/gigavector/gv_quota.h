#ifndef GIGAVECTOR_GV_QUOTA_H
#define GIGAVECTOR_GV_QUOTA_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t max_vectors;          /* Max vectors per tenant (0 = unlimited) */
    size_t max_memory_bytes;     /* Max memory per tenant (0 = unlimited) */
    double max_qps;              /* Max queries per second (0 = unlimited) */
    double max_ips;              /* Max inserts per second (0 = unlimited) */
    size_t max_storage_bytes;    /* Max disk storage (0 = unlimited) */
    size_t max_collections;      /* Max collections/namespaces (0 = unlimited) */
} GV_QuotaConfig;

typedef struct {
    size_t current_vectors;
    size_t current_memory_bytes;
    double current_qps;
    double current_ips;
    size_t current_storage_bytes;
    size_t current_collections;
    uint64_t total_throttled;     /* Total throttled requests */
    uint64_t total_rejected;      /* Total rejected (over hard limit) */
} GV_QuotaUsage;

typedef enum {
    GV_QUOTA_OK = 0,
    GV_QUOTA_THROTTLED = 1,       /* Soft limit hit, request delayed */
    GV_QUOTA_EXCEEDED = 2,        /* Hard limit hit, request rejected */
    GV_QUOTA_ERROR = -1
} GV_QuotaResult;

typedef struct GV_QuotaManager GV_QuotaManager;

GV_QuotaManager *gv_quota_create(void);
void gv_quota_destroy(GV_QuotaManager *mgr);

int gv_quota_set(GV_QuotaManager *mgr, const char *tenant_id, const GV_QuotaConfig *config);
int gv_quota_get(const GV_QuotaManager *mgr, const char *tenant_id, GV_QuotaConfig *config);
int gv_quota_remove(GV_QuotaManager *mgr, const char *tenant_id);

GV_QuotaResult gv_quota_check_insert(GV_QuotaManager *mgr, const char *tenant_id, size_t vector_count);
GV_QuotaResult gv_quota_check_query(GV_QuotaManager *mgr, const char *tenant_id);

int gv_quota_record_insert(GV_QuotaManager *mgr, const char *tenant_id, size_t count, size_t bytes);
int gv_quota_record_query(GV_QuotaManager *mgr, const char *tenant_id);
int gv_quota_record_delete(GV_QuotaManager *mgr, const char *tenant_id, size_t count, size_t bytes);

int gv_quota_get_usage(const GV_QuotaManager *mgr, const char *tenant_id, GV_QuotaUsage *usage);

int gv_quota_reset_usage(GV_QuotaManager *mgr, const char *tenant_id);

void gv_quota_config_init(GV_QuotaConfig *config);

#ifdef __cplusplus
}
#endif
#endif
