#ifndef GIGAVECTOR_GV_SHARD_H
#define GIGAVECTOR_GV_SHARD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_shard.h
 * @brief Shard management for distributed GigaVector.
 *
 * Provides sharding for horizontal scaling of vector data.
 */

/* Forward declarations */
struct GV_Database;
typedef struct GV_Database GV_Database;

/**
 * @brief Shard state.
 */
typedef enum {
    GV_SHARD_ACTIVE = 0,            /**< Shard is active and serving. */
    GV_SHARD_READONLY = 1,          /**< Shard is read-only. */
    GV_SHARD_MIGRATING = 2,         /**< Shard is being migrated. */
    GV_SHARD_OFFLINE = 3            /**< Shard is offline. */
} GV_ShardState;

/**
 * @brief Sharding strategy.
 */
typedef enum {
    GV_SHARD_HASH = 0,              /**< Hash-based partitioning. */
    GV_SHARD_RANGE = 1,             /**< Range-based partitioning. */
    GV_SHARD_CONSISTENT = 2         /**< Consistent hashing. */
} GV_ShardStrategy;

/**
 * @brief Shard information.
 */
typedef struct {
    uint32_t shard_id;              /**< Shard ID. */
    char *node_address;             /**< Node address (host:port). */
    GV_ShardState state;            /**< Current state. */
    uint64_t vector_count;          /**< Number of vectors. */
    uint64_t capacity;              /**< Maximum vectors. */
    uint32_t replica_count;         /**< Number of replicas. */
    uint64_t last_heartbeat;        /**< Last heartbeat timestamp. */
} GV_ShardInfo;

/**
 * @brief Shard configuration.
 */
typedef struct {
    uint32_t shard_count;           /**< Total number of shards. */
    uint32_t virtual_nodes;         /**< Virtual nodes for consistent hashing. */
    GV_ShardStrategy strategy;      /**< Sharding strategy. */
    uint32_t replication_factor;    /**< Number of replicas per shard. */
} GV_ShardConfig;

/**
 * @brief Opaque shard manager handle.
 */
typedef struct GV_ShardManager GV_ShardManager;

/* ============================================================================
 * Configuration
 * ============================================================================ */

/**
 * @brief Initialize shard configuration with defaults.
 *
 * @param config Configuration to initialize.
 */
void gv_shard_config_init(GV_ShardConfig *config);

/* ============================================================================
 * Shard Manager Lifecycle
 * ============================================================================ */

/**
 * @brief Create a shard manager.
 *
 * @param config Shard configuration (NULL for defaults).
 * @return Shard manager instance, or NULL on error.
 */
GV_ShardManager *gv_shard_manager_create(const GV_ShardConfig *config);

/**
 * @brief Destroy a shard manager.
 *
 * @param mgr Shard manager instance (safe to call with NULL).
 */
void gv_shard_manager_destroy(GV_ShardManager *mgr);

/* ============================================================================
 * Shard Operations
 * ============================================================================ */

/**
 * @brief Add a shard.
 *
 * @param mgr Shard manager.
 * @param shard_id Shard ID.
 * @param node_address Node address (host:port).
 * @return 0 on success, -1 on error.
 */
int gv_shard_add(GV_ShardManager *mgr, uint32_t shard_id, const char *node_address);

/**
 * @brief Remove a shard.
 *
 * @param mgr Shard manager.
 * @param shard_id Shard ID.
 * @return 0 on success, -1 on error.
 */
int gv_shard_remove(GV_ShardManager *mgr, uint32_t shard_id);

/**
 * @brief Get shard for a vector ID.
 *
 * @param mgr Shard manager.
 * @param vector_id Vector ID.
 * @return Shard ID, or -1 on error.
 */
int gv_shard_for_vector(GV_ShardManager *mgr, uint64_t vector_id);

/**
 * @brief Get shard for a key (consistent hashing).
 *
 * @param mgr Shard manager.
 * @param key Key data.
 * @param key_len Key length.
 * @return Shard ID, or -1 on error.
 */
int gv_shard_for_key(GV_ShardManager *mgr, const void *key, size_t key_len);

/**
 * @brief Get shard information.
 *
 * @param mgr Shard manager.
 * @param shard_id Shard ID.
 * @param info Output shard info.
 * @return 0 on success, -1 on error.
 */
int gv_shard_get_info(GV_ShardManager *mgr, uint32_t shard_id, GV_ShardInfo *info);

/**
 * @brief List all shards.
 *
 * @param mgr Shard manager.
 * @param shards Output array of shard info.
 * @param count Output count.
 * @return 0 on success, -1 on error.
 */
int gv_shard_list(GV_ShardManager *mgr, GV_ShardInfo **shards, size_t *count);

/**
 * @brief Free shard info list.
 *
 * @param shards Shard info array.
 * @param count Number of shards.
 */
void gv_shard_free_list(GV_ShardInfo *shards, size_t count);

/**
 * @brief Update shard state.
 *
 * @param mgr Shard manager.
 * @param shard_id Shard ID.
 * @param state New state.
 * @return 0 on success, -1 on error.
 */
int gv_shard_set_state(GV_ShardManager *mgr, uint32_t shard_id, GV_ShardState state);

/* ============================================================================
 * Rebalancing
 * ============================================================================ */

/**
 * @brief Start shard rebalancing.
 *
 * @param mgr Shard manager.
 * @return 0 on success, -1 on error.
 */
int gv_shard_rebalance_start(GV_ShardManager *mgr);

/**
 * @brief Check rebalancing status.
 *
 * @param mgr Shard manager.
 * @param progress Output progress (0.0 - 1.0).
 * @return 1 if rebalancing, 0 if not, -1 on error.
 */
int gv_shard_rebalance_status(GV_ShardManager *mgr, double *progress);

/**
 * @brief Cancel rebalancing.
 *
 * @param mgr Shard manager.
 * @return 0 on success, -1 on error.
 */
int gv_shard_rebalance_cancel(GV_ShardManager *mgr);

/* ============================================================================
 * Local Shard
 * ============================================================================ */

/**
 * @brief Attach a local database as a shard.
 *
 * @param mgr Shard manager.
 * @param shard_id Shard ID.
 * @param db Local database.
 * @return 0 on success, -1 on error.
 */
int gv_shard_attach_local(GV_ShardManager *mgr, uint32_t shard_id, GV_Database *db);

/**
 * @brief Get local shard database.
 *
 * @param mgr Shard manager.
 * @param shard_id Shard ID.
 * @return Database pointer, or NULL if not local.
 */
GV_Database *gv_shard_get_local_db(GV_ShardManager *mgr, uint32_t shard_id);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_SHARD_H */
