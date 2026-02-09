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

/* ============================================================================
 * Read Replica Load Balancing
 * ============================================================================ */

/**
 * @brief Read routing policy for distributing read queries across replicas.
 */
typedef enum {
    GV_READ_LEADER_ONLY = 0,    /**< All reads go to leader (strongest consistency). */
    GV_READ_ROUND_ROBIN = 1,    /**< Distribute reads across replicas in round-robin. */
    GV_READ_LEAST_LAG = 2,      /**< Route reads to replica with smallest replication lag. */
    GV_READ_RANDOM = 3          /**< Route reads to a random connected replica. */
} GV_ReadPolicy;

/**
 * @brief Set read routing policy.
 *
 * @param mgr Replication manager.
 * @param policy Desired read policy.
 * @return 0 on success, -1 on error.
 */
int gv_replication_set_read_policy(GV_ReplicationManager *mgr, GV_ReadPolicy policy);

/**
 * @brief Get current read routing policy.
 *
 * @param mgr Replication manager.
 * @return Current policy, or -1 on error.
 */
GV_ReadPolicy gv_replication_get_read_policy(GV_ReplicationManager *mgr);

/**
 * @brief Route a read query to the appropriate replica.
 *
 * Returns the database instance that should handle the read query,
 * based on the current read policy. When policy is LEADER_ONLY, always
 * returns the leader database. For other policies, may return a follower's
 * database if followers have registered their DB instances.
 *
 * @param mgr Replication manager.
 * @return Database instance for reading, or NULL if no suitable replica found.
 */
GV_Database *gv_replication_route_read(GV_ReplicationManager *mgr);

/**
 * @brief Set the maximum acceptable lag for read replicas.
 *
 * Replicas with lag exceeding this value will not receive read queries
 * (except under LEADER_ONLY policy where this is ignored).
 *
 * @param mgr Replication manager.
 * @param max_lag Maximum acceptable lag in WAL entries.
 * @return 0 on success, -1 on error.
 */
int gv_replication_set_max_read_lag(GV_ReplicationManager *mgr, uint64_t max_lag);

/**
 * @brief Register a follower's database instance for read routing.
 *
 * When a follower connects, its local database instance can be registered
 * so that read queries can be routed to it.
 *
 * @param mgr Replication manager.
 * @param node_id Follower node ID.
 * @param db Follower's database instance.
 * @return 0 on success, -1 on error.
 */
int gv_replication_register_follower_db(GV_ReplicationManager *mgr,
                                         const char *node_id, GV_Database *db);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_REPLICATION_H */
