#ifndef GIGAVECTOR_GV_REPLICATION_H
#define GIGAVECTOR_GV_REPLICATION_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file replication.h
 * @brief Replication and high availability for GigaVector.
 *
 * Leader-side WAL progress is advanced with replication_leader_append_wal()
 * after durable writes. Registered followers are coordinated in-process (no TCP
 * yet); sync_commit(), lag, and read routing use that model consistently.
 */

struct GV_Database;
typedef struct GV_Database GV_Database;

struct GV_MemoryLayer;
typedef struct GV_MemoryLayer GV_MemoryLayer;

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

struct GV_ReplTransport;
typedef struct GV_ReplTransport GV_ReplTransport;

/**
 * @brief Initialize replication configuration with defaults.
 *
 * @param config Configuration to initialize.
 */
void replication_config_init(GV_ReplicationConfig *config);

/**
 * @brief Create a replication manager.
 *
 * @param db Database to replicate.
 * @param config Replication configuration.
 * @return Replication manager, or NULL on error.
 */
GV_ReplicationManager *replication_create(GV_Database *db, const GV_ReplicationConfig *config);

/**
 * @brief Destroy a replication manager.
 *
 * @param mgr Replication manager (safe to call with NULL).
 */
void replication_destroy(GV_ReplicationManager *mgr);

/**
 * @brief Start replication.
 *
 * @param mgr Replication manager.
 * @return 0 on success, -1 on error.
 */
int replication_start(GV_ReplicationManager *mgr);

/**
 * @brief Stop replication.
 *
 * @param mgr Replication manager.
 * @return 0 on success, -1 on error.
 */
int replication_stop(GV_ReplicationManager *mgr);

/**
 * @brief Get current role.
 *
 * @param mgr Replication manager.
 * @return Current role, or -1 on error.
 */
GV_ReplicationRole replication_get_role(GV_ReplicationManager *mgr);

/**
 * @brief Force step down from leader.
 *
 * @param mgr Replication manager.
 * @return 0 on success, -1 on error.
 */
int replication_step_down(GV_ReplicationManager *mgr);

/**
 * @brief Request leadership.
 *
 * @param mgr Replication manager.
 * @return 0 if became leader, -1 on error.
 */
int replication_request_leadership(GV_ReplicationManager *mgr);

/**
 * @brief Add a follower replica.
 *
 * @param mgr Replication manager.
 * @param node_id Follower node ID.
 * @param address Follower address.
 * @return 0 on success, -1 on error.
 */
int replication_add_follower(GV_ReplicationManager *mgr, const char *node_id,
                                 const char *address);

/**
 * @brief Remove a follower replica.
 *
 * @param mgr Replication manager.
 * @param node_id Follower node ID.
 * @return 0 on success, -1 on error.
 */
int replication_remove_follower(GV_ReplicationManager *mgr, const char *node_id);

/**
 * @brief List all replicas.
 *
 * @param mgr Replication manager.
 * @param replicas Output replica array.
 * @param count Output count.
 * @return 0 on success, -1 on error.
 */
int replication_list_replicas(GV_ReplicationManager *mgr, GV_ReplicaInfo **replicas,
                                  size_t *count);

/**
 * @brief Free replica info list.
 *
 * @param replicas Replica array to free.
 * @param count Number of replicas.
 */
void replication_free_replicas(GV_ReplicaInfo *replicas, size_t count);

/**
 * @brief Force synchronous commit.
 *
 * Waits for all followers to acknowledge.
 *
 * @param mgr Replication manager.
 * @param timeout_ms Timeout in milliseconds.
 * @return 0 on success, -1 on timeout/error.
 */
int replication_sync_commit(GV_ReplicationManager *mgr, uint32_t timeout_ms);

/**
 * @brief Get replication lag.
 *
 * @param mgr Replication manager.
 * @return Lag in WAL entries, or -1 on error.
 */
int64_t replication_get_lag(GV_ReplicationManager *mgr);

/**
 * @brief Wait for replication to catch up.
 *
 * @param mgr Replication manager.
 * @param max_lag Maximum acceptable lag.
 * @param timeout_ms Timeout in milliseconds.
 * @return 0 on success, -1 on timeout/error.
 */
int replication_wait_sync(GV_ReplicationManager *mgr, size_t max_lag, uint32_t timeout_ms);

/**
 * @brief Get replication statistics.
 *
 * @param mgr Replication manager.
 * @param stats Output statistics.
 * @return 0 on success, -1 on error.
 */
int replication_get_stats(GV_ReplicationManager *mgr, GV_ReplicationStats *stats);

/**
 * @brief Free replication stats.
 *
 * @param stats Stats to free.
 */
void replication_free_stats(GV_ReplicationStats *stats);

/**
 * @brief Check if replication is healthy.
 *
 * @param mgr Replication manager.
 * @return 1 if healthy, 0 if not, -1 on error.
 */
int replication_is_healthy(GV_ReplicationManager *mgr);

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
int replication_set_read_policy(GV_ReplicationManager *mgr, GV_ReadPolicy policy);

/**
 * @brief Get current read routing policy.
 *
 * @param mgr Replication manager.
 * @return Current policy, or -1 on error.
 */
GV_ReadPolicy replication_get_read_policy(GV_ReplicationManager *mgr);

/**
 * @brief Route a read query to the appropriate replica.
 *
 * Returns the database instance that should handle the read query,
 * based on the current read policy. When policy is LEADER_ONLY, always
 * returns the leader database. For other policies, may return a follower's
 * database if followers have registered their DB instances.
 *
 * LIFETIME CONTRACT (pin/release): when this returns a *follower* database, that
 * follower is pinned (an in-flight read count is incremented) so that
 * replication_remove_follower() cannot free or reuse the slot while the returned
 * handle is in use. The caller MUST pair every non-NULL return with exactly one
 * call to replication_release_read(mgr, returned_db) once done reading. Calling
 * replication_release_read() with the leader database (or NULL) is a harmless
 * no-op, so callers may unconditionally release the handle they received.
 * Failing to release leaks the pin and will make a future
 * replication_remove_follower() for that node block indefinitely.
 *
 * @param mgr Replication manager.
 * @return Database instance for reading, or NULL if no suitable replica found.
 */
GV_Database *replication_route_read(GV_ReplicationManager *mgr);

/**
 * @brief Release a database handle previously returned by replication_route_read().
 *
 * Decrements the in-flight pin taken by replication_route_read() for the follower
 * backing @p db, waking any replication_remove_follower() that is waiting for the
 * slot to drain. Passing the leader database or NULL is a no-op that returns 0.
 *
 * @param mgr Replication manager.
 * @param db  Handle returned by replication_route_read().
 * @return 0 on success (including no-op leader/NULL releases), -1 if @p mgr is
 *         NULL or @p db is not a currently-pinned follower handle.
 */
int replication_release_read(GV_ReplicationManager *mgr, GV_Database *db);

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
int replication_set_max_read_lag(GV_ReplicationManager *mgr, uint64_t max_lag);

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
int replication_register_follower_db(GV_ReplicationManager *mgr,
                                         const char *node_id, GV_Database *db);

/**
 * @brief Register a follower's memory layer for read routing.
 *
 * Pair with replication_register_follower_db() so memory reranking can use the
 * same replica selected by replication_route_read().
 *
 * @param mgr Replication manager.
 * @param node_id Follower node ID.
 * @param layer Follower's memory layer instance.
 * @return 0 on success, -1 on error.
 */
int replication_register_follower_memory(GV_ReplicationManager *mgr,
                                             const char *node_id,
                                             GV_MemoryLayer *layer);

/**
 * @brief Route a memory rerank query to the same replica as vector reads.
 *
 * Returns NULL when reads are routed to the leader or the selected follower has
 * no registered memory layer (caller should use the leader memory layer).
 *
 * @param mgr Replication manager.
 * @return Memory layer for reading, or NULL for leader/fallback.
 */
GV_MemoryLayer *replication_route_read_memory(GV_ReplicationManager *mgr);

/**
 * @brief Advance the leader WAL cursor after durable local writes.
 *
 * Call after appending to the leader WAL (or equivalent committed entries).
 * In the embedded coordinator, connected followers are updated to the new
 * head immediately so sync_commit(), lag, and read routing stay coherent.
 *
 * @param mgr Replication manager.
 * @param entry_delta Number of WAL entries (or logical positions) to add.
 * @param byte_delta Optional byte volume to add to replication stats (may be 0).
 * @return 0 on success, -1 if @p mgr is NULL or this node is not the leader.
 */
int replication_leader_append_wal(GV_ReplicationManager *mgr,
                                      uint64_t entry_delta,
                                      uint64_t byte_delta);

/**
 * Enable DST simulation mode: followers are not auto-caught-up and commit
 * advances only via replication_sync_commit() after explicit acks.
 */
int replication_set_dst_simulation_mode(GV_ReplicationManager *mgr, int enabled);
int replication_get_dst_simulation_mode(const GV_ReplicationManager *mgr);

/** Transport helpers (used by repl_transport.c). */
GV_Database *replication_get_db(GV_ReplicationManager *mgr);
const char *replication_get_node_id(const GV_ReplicationManager *mgr);
const GV_ReplicationConfig *replication_get_config(const GV_ReplicationManager *mgr);
int replication_replica_handshake(GV_ReplicationManager *mgr, const char *node_id,
                                  uint64_t *catchup_from_out);
int replication_replica_ack(GV_ReplicationManager *mgr, const char *node_id,
                            uint64_t entry_index);
int replication_follower_apply_entry(GV_ReplicationManager *mgr, uint64_t entry_index,
                                     const uint8_t *record, size_t len);
void replication_get_positions(const GV_ReplicationManager *mgr,
                               uint64_t *wal_position, uint64_t *commit_position);
void replication_note_leader_heartbeat(GV_ReplicationManager *mgr);
GV_ReplicationRole replication_get_role_for_transport(const GV_ReplicationManager *mgr);

/** Access TCP transport for hook installation (may be NULL before start). */
GV_ReplTransport *replication_get_transport(GV_ReplicationManager *mgr);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_REPLICATION_H */
