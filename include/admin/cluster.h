#ifndef GIGAVECTOR_GV_CLUSTER_H
#define GIGAVECTOR_GV_CLUSTER_H

#include <stddef.h>
#include <stdint.h>

#include "admin/shard.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file cluster.h
 * @brief Cluster management for distributed GigaVector.
 *
 * Provides cluster coordination, node discovery, and health monitoring.
 */

typedef enum {
    GV_NODE_COORDINATOR = 0,        /**< Cluster coordinator. */
    GV_NODE_DATA = 1,               /**< Data node. */
    GV_NODE_QUERY = 2               /**< Query-only node. */
} GV_NodeRole;

typedef enum {
    GV_NODE_JOINING = 0,            /**< Node is joining cluster. */
    GV_NODE_ACTIVE = 1,             /**< Node is active. */
    GV_NODE_LEAVING = 2,            /**< Node is leaving cluster. */
    GV_NODE_DEAD = 3                /**< Node is unreachable. */
} GV_NodeState;

typedef struct {
    char *node_id;                  /**< Unique node identifier. */
    char *address;                  /**< Node address (host:port). */
    GV_NodeRole role;               /**< Node role. */
    GV_NodeState state;             /**< Node state. */
    uint32_t *shard_ids;            /**< Shards on this node. */
    size_t shard_count;             /**< Number of shards. */
    uint64_t last_heartbeat;        /**< Last heartbeat timestamp. */
    double load;                    /**< Current load (0.0 - 1.0). */
} GV_NodeInfo;

typedef struct {
    const char *node_id;            /**< This node's ID. */
    const char *listen_address;     /**< Address to listen on. */
    const char *seed_nodes;         /**< Comma-separated seed nodes. */
    GV_NodeRole role;               /**< This node's role. */
    uint32_t heartbeat_interval_ms; /**< Heartbeat interval. */
    uint32_t failure_timeout_ms;    /**< Node failure timeout. */
    const char *raft_data_dir;      /**< If set, the Raft log/vote are persisted
                                         under this directory (one file per node)
                                         and reloaded on restart for durable HA.
                                         NULL keeps the older in-memory behaviour. */
} GV_ClusterConfig;

typedef struct {
    size_t total_nodes;             /**< Total nodes in cluster. */
    size_t active_nodes;            /**< Active nodes. */
    size_t total_shards;            /**< Total shards. */
    uint64_t total_vectors;         /**< Total vectors across cluster. */
    double avg_load;                /**< Average cluster load. */
} GV_ClusterStats;

typedef struct GV_Cluster GV_Cluster;

/**
 * @brief Initialize cluster configuration with defaults.
 *
 * @param config Configuration to initialize.
 */
void cluster_config_init(GV_ClusterConfig *config);

/**
 * @brief Create a cluster instance.
 *
 * @param config Cluster configuration.
 * @return Cluster instance, or NULL on error.
 */
GV_Cluster *cluster_create(const GV_ClusterConfig *config);

/**
 * @brief Destroy a cluster instance.
 *
 * @param cluster Cluster instance (safe to call with NULL).
 */
void cluster_destroy(GV_Cluster *cluster);

/**
 * @brief Start cluster services.
 *
 * @param cluster Cluster instance.
 * @return 0 on success, -1 on error.
 */
int cluster_start(GV_Cluster *cluster);

/**
 * @brief Stop cluster services.
 *
 * @param cluster Cluster instance.
 * @return 0 on success, -1 on error.
 */
int cluster_stop(GV_Cluster *cluster);

/**
 * @brief Get this node's information.
 *
 * @param cluster Cluster instance.
 * @param info Output node info.
 * @return 0 on success, -1 on error.
 */
int cluster_get_local_node(GV_Cluster *cluster, GV_NodeInfo *info);

/**
 * @brief Get a node's information.
 *
 * @param cluster Cluster instance.
 * @param node_id Node ID.
 * @param info Output node info.
 * @return 0 on success, -1 on error.
 */
int cluster_get_node(GV_Cluster *cluster, const char *node_id, GV_NodeInfo *info);

/**
 * @brief List all nodes.
 *
 * @param cluster Cluster instance.
 * @param nodes Output node array.
 * @param count Output count.
 * @return 0 on success, -1 on error.
 */
int cluster_list_nodes(GV_Cluster *cluster, GV_NodeInfo **nodes, size_t *count);

/**
 * @brief Free node info.
 *
 * @param info Node info to free.
 */
void cluster_free_node_info(GV_NodeInfo *info);

/**
 * @brief Free node list.
 *
 * @param nodes Node array to free.
 * @param count Number of nodes.
 */
void cluster_free_node_list(GV_NodeInfo *nodes, size_t count);

/**
 * @brief Get cluster statistics.
 *
 * @param cluster Cluster instance.
 * @param stats Output statistics.
 * @return 0 on success, -1 on error.
 */
int cluster_get_stats(GV_Cluster *cluster, GV_ClusterStats *stats);

/**
 * @brief Get the shard manager.
 *
 * @param cluster Cluster instance.
 * @return Shard manager, or NULL on error.
 */
GV_ShardManager *cluster_get_shard_manager(GV_Cluster *cluster);

/**
 * @brief Distributed k-NN search across the cluster's local and remote shards.
 *
 * Fans out to local shards (shard_search) and each distinct remote node
 * (shard_rpc_search) via shard_rpc_search_distributed(), then merges a global
 * top-k. Results are owned by the caller (free with gv_search_results_free()).
 *
 * @param cluster       Cluster instance.
 * @param query_data    Query vector.
 * @param dim           Query dimension (required to reach remote nodes).
 * @param k             Number of neighbours to return.
 * @param results       Output array of at least @p k elements.
 * @param distance_type Distance metric to use.
 * @return Number of neighbours written (0..k), or -1 on error.
 */
int cluster_search(GV_Cluster *cluster, const float *query_data, size_t dim,
                   size_t k, GV_SearchResult *results,
                   GV_DistanceType distance_type);

/**
 * @brief Attach this node's local database as shard 0 (the node's own data).
 *
 * Call before cluster_start(). The database is served to peers over the RPC
 * listener and searched locally by cluster_search().
 *
 * @return 0 on success, -1 on error.
 */
int cluster_attach_local_db(GV_Cluster *cluster, GV_Database *db);

/**
 * @brief Port the node's search RPC listener is bound to (0 if not started).
 *
 * Useful after binding an ephemeral port (listen_address host:0) to tell peers
 * where to reach this node.
 */
uint16_t cluster_rpc_port(const GV_Cluster *cluster);

/**
 * @brief Enable raft leader election over the cluster's RPC transport.
 *
 * Call after cluster_start(). @p addrs is the ordered set of participating node
 * RPC endpoints ("host:port"); every node passes the same @p addrs and its own
 * position as @p my_index. Nodes exchange raft messages over the shared listener
 * and elect a single leader.
 *
 * @return 0 on success, -1 on error.
 */
int cluster_enable_raft(GV_Cluster *cluster, const char *const *addrs,
                        size_t n, int my_index);

/** @brief Believed raft leader's index, or -1 if unknown / raft disabled. */
int cluster_raft_leader(GV_Cluster *cluster);

/** @brief Non-zero if this node is currently the raft leader. */
int cluster_is_raft_leader(GV_Cluster *cluster);

/**
 * @brief Replicate a shard-placement assignment through the raft log.
 *
 * Leader-only: proposes "shard @p shard_id is owned by node @p node_index".
 * On commit the assignment is applied on every node (see cluster_shard_owner).
 *
 * @return 0 if accepted (this node is leader), -1 otherwise.
 */
int cluster_assign_shard(GV_Cluster *cluster, uint64_t shard_id, int node_index);

/**
 * @brief Owning node index of @p shard_id in the replicated placement map, or
 *        -1 if unassigned. Converges on all nodes once the assignment commits.
 */
int cluster_shard_owner(GV_Cluster *cluster, uint64_t shard_id);

/**
 * @brief Check if cluster is healthy.
 *
 * @param cluster Cluster instance.
 * @return 1 if healthy, 0 if not, -1 on error.
 */
int cluster_is_healthy(GV_Cluster *cluster);

/**
 * @brief Wait for cluster to be ready.
 *
 * @param cluster Cluster instance.
 * @param timeout_ms Timeout in milliseconds.
 * @return 0 if ready, -1 on timeout/error.
 */
int cluster_wait_ready(GV_Cluster *cluster, uint32_t timeout_ms);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_CLUSTER_H */
