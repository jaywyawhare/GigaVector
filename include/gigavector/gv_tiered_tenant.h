#ifndef GIGAVECTOR_GV_TIERED_TENANT_H
#define GIGAVECTOR_GV_TIERED_TENANT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_tiered_tenant.h
 * @brief Tiered multitenancy for GigaVector.
 *
 * Efficiently handle tenants with vastly different sizes.  Small tenants
 * share infrastructure (shared tier), medium tenants get isolated indexes
 * (dedicated tier), large tenants get dedicated resources (premium tier).
 * Auto-promotes tenants between tiers based on usage.
 */

/* ============================================================================
 * Enums
 * ============================================================================ */

/**
 * @brief Tenant tier classification.
 */
typedef enum {
    GV_TIER_SHARED    = 0,   /**< Shared infrastructure (small tenants). */
    GV_TIER_DEDICATED = 1,   /**< Isolated indexes (medium tenants). */
    GV_TIER_PREMIUM   = 2    /**< Dedicated resources (large tenants). */
} GV_TenantTier;

/* ============================================================================
 * Configuration Structures
 * ============================================================================ */

/**
 * @brief Thresholds governing automatic tier promotion/demotion.
 */
typedef struct {
    size_t shared_max_vectors;      /**< Max vectors for shared tier (default 10000). */
    size_t dedicated_max_vectors;   /**< Max vectors for dedicated tier (default 1000000). */
    size_t shared_max_memory_mb;    /**< Max memory MB for shared tier (default 64). */
    size_t dedicated_max_memory_mb; /**< Max memory MB for dedicated tier (default 1024). */
} GV_TierThresholds;

/**
 * @brief Tiered tenant manager configuration.
 */
typedef struct {
    GV_TierThresholds thresholds;   /**< Tier promotion/demotion thresholds. */
    int auto_promote;               /**< Enable auto-promotion (default 1). */
    int auto_demote;                /**< Enable auto-demotion (default 0). */
    size_t max_shared_tenants;      /**< Max tenants in shared tier (default 1000). */
    size_t max_total_tenants;       /**< Max total tenants across all tiers (default 10000). */
} GV_TieredTenantConfig;

/* ============================================================================
 * Tenant Information
 * ============================================================================ */

/**
 * @brief Per-tenant information snapshot.
 */
typedef struct {
    const char   *tenant_id;   /**< Tenant identifier (points into manager storage). */
    GV_TenantTier tier;        /**< Current tier. */
    size_t        vector_count; /**< Current vector count. */
    size_t        memory_bytes; /**< Current memory usage in bytes. */
    uint64_t      created_at;  /**< Creation timestamp (epoch seconds). */
    uint64_t      last_active; /**< Last activity timestamp (epoch seconds). */
    double        qps_avg;     /**< Average queries per second (sliding window). */
} GV_TenantInfo;

/* ============================================================================
 * Opaque Manager Handle
 * ============================================================================ */

/**
 * @brief Opaque tiered tenant manager handle.
 */
typedef struct GV_TieredManager GV_TieredManager;

/* ============================================================================
 * Configuration
 * ============================================================================ */

/**
 * @brief Initialize tiered tenant configuration with defaults.
 *
 * Default values:
 * - thresholds.shared_max_vectors:      10000
 * - thresholds.dedicated_max_vectors:   1000000
 * - thresholds.shared_max_memory_mb:    64
 * - thresholds.dedicated_max_memory_mb: 1024
 * - auto_promote:                       1
 * - auto_demote:                        0
 * - max_shared_tenants:                 1000
 * - max_total_tenants:                  10000
 *
 * @param config Configuration to initialize.
 */
void gv_tiered_config_init(GV_TieredTenantConfig *config);

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

/**
 * @brief Create a tiered tenant manager.
 *
 * @param config Configuration (NULL for defaults).
 * @return Manager instance, or NULL on error.
 */
GV_TieredManager *gv_tiered_create(const GV_TieredTenantConfig *config);

/**
 * @brief Destroy a tiered tenant manager and free all resources.
 *
 * @param mgr Manager instance (safe to call with NULL).
 */
void gv_tiered_destroy(GV_TieredManager *mgr);

/* ============================================================================
 * Tenant Operations
 * ============================================================================ */

/**
 * @brief Add a new tenant.
 *
 * @param mgr          Manager instance.
 * @param tenant_id    Unique tenant identifier (max 127 chars).
 * @param initial_tier Starting tier for the tenant.
 * @return 0 on success, -1 on error (duplicate, full, invalid).
 */
int gv_tiered_add_tenant(GV_TieredManager *mgr, const char *tenant_id,
                          GV_TenantTier initial_tier);

/**
 * @brief Remove a tenant and free its resources.
 *
 * @param mgr       Manager instance.
 * @param tenant_id Tenant identifier.
 * @return 0 on success, -1 on error (not found).
 */
int gv_tiered_remove_tenant(GV_TieredManager *mgr, const char *tenant_id);

/**
 * @brief Manually promote or demote a tenant to a new tier.
 *
 * @param mgr       Manager instance.
 * @param tenant_id Tenant identifier.
 * @param new_tier  Target tier.
 * @return 0 on success, -1 on error.
 */
int gv_tiered_promote(GV_TieredManager *mgr, const char *tenant_id,
                       GV_TenantTier new_tier);

/**
 * @brief Get information about a tenant.
 *
 * @param mgr       Manager instance.
 * @param tenant_id Tenant identifier.
 * @param info      Output information structure.
 * @return 0 on success, -1 on error (not found).
 */
int gv_tiered_get_info(const GV_TieredManager *mgr, const char *tenant_id,
                        GV_TenantInfo *info);

/* ============================================================================
 * Usage Tracking
 * ============================================================================ */

/**
 * @brief Record usage delta for a tenant.
 *
 * @param mgr           Manager instance.
 * @param tenant_id     Tenant identifier.
 * @param vectors_delta Change in vector count (added).
 * @param memory_delta  Change in memory bytes (added).
 * @return 0 on success, -1 on error.
 */
int gv_tiered_record_usage(GV_TieredManager *mgr, const char *tenant_id,
                            size_t vectors_delta, size_t memory_delta);

/**
 * @brief Check all tenants for automatic tier promotion/demotion.
 *
 * Iterates all tenants:
 * - Shared tenants exceeding shared thresholds are promoted to dedicated.
 * - Dedicated tenants exceeding dedicated thresholds are promoted to premium.
 * - If auto_demote is enabled, tenants below 50% of the lower tier threshold
 *   for 7+ days are demoted.
 *
 * @param mgr Manager instance.
 * @return Number of tenants promoted/demoted, or -1 on error.
 */
int gv_tiered_check_promote(GV_TieredManager *mgr);

/* ============================================================================
 * Enumeration
 * ============================================================================ */

/**
 * @brief List tenants in a specific tier.
 *
 * @param mgr       Manager instance.
 * @param tier      Tier to filter by.
 * @param out       Output array of tenant info structures.
 * @param max_count Maximum entries to write.
 * @return Number of entries written, or -1 on error.
 */
int gv_tiered_list_tenants(const GV_TieredManager *mgr, GV_TenantTier tier,
                            GV_TenantInfo *out, size_t max_count);

/**
 * @brief Get total tenant count across all tiers.
 *
 * @param mgr Manager instance.
 * @return Total tenant count.
 */
size_t gv_tiered_tenant_count(const GV_TieredManager *mgr);

/* ============================================================================
 * Persistence
 * ============================================================================ */

/**
 * @brief Save manager state to a binary file.
 *
 * @param mgr  Manager instance.
 * @param path Output file path.
 * @return 0 on success, -1 on error.
 */
int gv_tiered_save(const GV_TieredManager *mgr, const char *path);

/**
 * @brief Load manager state from a binary file.
 *
 * @param path Input file path.
 * @return Manager instance, or NULL on error.
 */
GV_TieredManager *gv_tiered_load(const char *path);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_TIERED_TENANT_H */
