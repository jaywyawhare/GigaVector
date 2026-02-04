#ifndef GIGAVECTOR_GV_REPLICATION_H
#define GIGAVECTOR_GV_REPLICATION_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_replication.h
 * @brief Replication and high availability for GigaVector.
 *
 * Provides leader-follower replication with automatic failover.
 */

/* Forward declarations */
struct GV_Database;
typedef struct GV_Database GV_Database;

/**
 * @brief Replication role.
 */
typedef enum {
    GV_REPL_LEADER = 0,             /**< Primary/leader node. */
    GV_REPL_FOLLOWER = 1,           /**< Secondary/follower node. */
    GV_REPL_CANDIDATE = 2           /**< Candidate for leader election. */
} GV_ReplicationRole;

/**
 * @brief Replication state.
 */
typedef enum {
    GV_REPL_SYNCING = 0,            /**< Initial sync in progress. */
    GV_REPL_STREAMING = 1,          /**< Streaming replication active. */
    GV_REPL_LAGGING = 2,            /**< Follower is behind. */
    GV_REPL_DISCONNECTED = 3        /**< Follower disconnected. */
} GV_ReplicationState;

/**
 * @brief Replication configuration.
 */
typedef struct {
    const char *node_id;            /**< This node's ID. */
    const char *listen_address;     /**< Replication listen address. */
    const char *leader_address;     /**< Initial leader address (for followers). */
    uint32_t sync_interval_ms;      /**< Sync interval in milliseconds. */
    uint32_t election_timeout_ms;   /**< Election timeout. */
    uint32_t heartbeat_interval_ms; /**< Leader heartbeat interval. */
    size_t max_lag_entries;         /**< Max WAL entries before resync. */
} GV_ReplicationConfig;

/**
 * @brief Replica information.
 */
typedef struct {
    char *node_id;                  /**< Replica node ID. */
    char *address;                  /**< Replica address. */
    GV_ReplicationRole role;        /**< Current role. */
    GV_ReplicationState state;      /**< Replication state. */
    uint64_t last_wal_position;     /**< Last replicated WAL position. */
    uint64_t lag_entries;           /**< Number of entries behind. */
    uint64_t last_heartbeat;        /**< Last heartbeat timestamp. */
} GV_ReplicaInfo;

/**
 * @brief Replication statistics.
 */
typedef struct {
    GV_ReplicationRole role;        /**< Current role. */
    uint64_t term;                  /**< Current election term. */
    char *leader_id;                /**< Current leader ID. */
    size_t follower_count;          /**< Number of followers (if leader). */
    uint64_t wal_position;          /**< Current WAL position. */
    uint64_t commit_position;       /**< Committed WAL position. */
    uint64_t bytes_replicated;      /**< Total bytes replicated. */
} GV_ReplicationStats;

/**
 * @brief Opaque replication manager handle.
 */
typedef struct GV_ReplicationManager GV_ReplicationManager;

/* ============================================================================
 * Configuration
 * ============================================================================ */

/**
 * @brief Initialize replication configuration with defaults.
 *
 * @param config Configuration to initialize.
 */
void gv_replication_config_init(GV_ReplicationConfig *config);

/* ============================================================================
 * Replication Manager Lifecycle
 * ============================================================================ */

/**
 * @brief Create a replication manager.
 *
 * @param db Database to replicate.
 * @param config Replication configuration.
 * @return Replication manager, or NULL on error.
 */
GV_ReplicationManager *gv_replication_create(GV_Database *db, const GV_ReplicationConfig *config);

/**
 * @brief Destroy a replication manager.
 *
 * @param mgr Replication manager (safe to call with NULL).
 */
void gv_replication_destroy(GV_ReplicationManager *mgr);

/**
 * @brief Start replication.
 *
 * @param mgr Replication manager.
 * @return 0 on success, -1 on error.
 */
int gv_replication_start(GV_ReplicationManager *mgr);

/**
 * @brief Stop replication.
 *
 * @param mgr Replication manager.
 * @return 0 on success, -1 on error.
 */
int gv_replication_stop(GV_ReplicationManager *mgr);

/* ============================================================================
 * Role Management
 * ============================================================================ */

/**
 * @brief Get current role.
 *
 * @param mgr Replication manager.
 * @return Current role, or -1 on error.
 */
GV_ReplicationRole gv_replication_get_role(GV_ReplicationManager *mgr);

/**
 * @brief Force step down from leader.
 *
 * @param mgr Replication manager.
 * @return 0 on success, -1 on error.
 */
int gv_replication_step_down(GV_ReplicationManager *mgr);

/**
 * @brief Request leadership.
 *
 * @param mgr Replication manager.
 * @return 0 if became leader, -1 on error.
 */
int gv_replication_request_leadership(GV_ReplicationManager *mgr);

/* ============================================================================
 * Replica Management
 * ============================================================================ */

/**
 * @brief Add a follower replica.
 *
 * @param mgr Replication manager.
 * @param node_id Follower node ID.
 * @param address Follower address.
 * @return 0 on success, -1 on error.
 */
int gv_replication_add_follower(GV_ReplicationManager *mgr, const char *node_id,
                                 const char *address);

/**
 * @brief Remove a follower replica.
 *
 * @param mgr Replication manager.
 * @param node_id Follower node ID.
 * @return 0 on success, -1 on error.
 */
int gv_replication_remove_follower(GV_ReplicationManager *mgr, const char *node_id);

/**
 * @brief List all replicas.
 *
 * @param mgr Replication manager.
 * @param replicas Output replica array.
 * @param count Output count.
 * @return 0 on success, -1 on error.
 */
int gv_replication_list_replicas(GV_ReplicationManager *mgr, GV_ReplicaInfo **replicas,
                                  size_t *count);

/**
 * @brief Free replica info list.
 *
 * @param replicas Replica array to free.
 * @param count Number of replicas.
 */
void gv_replication_free_replicas(GV_ReplicaInfo *replicas, size_t count);

/* ============================================================================
 * Synchronization
 * ============================================================================ */

/**
 * @brief Force synchronous commit.
 *
 * Waits for all followers to acknowledge.
 *
 * @param mgr Replication manager.
 * @param timeout_ms Timeout in milliseconds.
 * @return 0 on success, -1 on timeout/error.
 */
int gv_replication_sync_commit(GV_ReplicationManager *mgr, uint32_t timeout_ms);

/**
 * @brief Get replication lag.
 *
 * @param mgr Replication manager.
 * @return Lag in WAL entries, or -1 on error.
 */
int64_t gv_replication_get_lag(GV_ReplicationManager *mgr);

/**
 * @brief Wait for replication to catch up.
 *
 * @param mgr Replication manager.
 * @param max_lag Maximum acceptable lag.
 * @param timeout_ms Timeout in milliseconds.
 * @return 0 on success, -1 on timeout/error.
 */
int gv_replication_wait_sync(GV_ReplicationManager *mgr, size_t max_lag, uint32_t timeout_ms);

/* ============================================================================
 * Statistics
 * ============================================================================ */

/**
 * @brief Get replication statistics.
 *
 * @param mgr Replication manager.
 * @param stats Output statistics.
 * @return 0 on success, -1 on error.
 */
int gv_replication_get_stats(GV_ReplicationManager *mgr, GV_ReplicationStats *stats);

/**
 * @brief Free replication stats.
 *
 * @param stats Stats to free.
 */
void gv_replication_free_stats(GV_ReplicationStats *stats);

/**
 * @brief Check if replication is healthy.
 *
 * @param mgr Replication manager.
 * @return 1 if healthy, 0 if not, -1 on error.
 */
int gv_replication_is_healthy(GV_ReplicationManager *mgr);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_REPLICATION_H */
