#ifndef GIGAVECTOR_GV_CLUSTER_H
#define GIGAVECTOR_GV_CLUSTER_H

#include <stddef.h>
#include <stdint.h>

#include "gv_shard.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_cluster.h
 * @brief Cluster management for distributed GigaVector.
 *
 * Provides cluster coordination, node discovery, and health monitoring.
 */

/**
 * @brief Node role.
 */
typedef enum {
    GV_NODE_COORDINATOR = 0,        /**< Cluster coordinator. */
    GV_NODE_DATA = 1,               /**< Data node. */
    GV_NODE_QUERY = 2               /**< Query-only node. */
} GV_NodeRole;

/**
 * @brief Node state.
 */
typedef enum {
    GV_NODE_JOINING = 0,            /**< Node is joining cluster. */
    GV_NODE_ACTIVE = 1,             /**< Node is active. */
    GV_NODE_LEAVING = 2,            /**< Node is leaving cluster. */
    GV_NODE_DEAD = 3                /**< Node is unreachable. */
} GV_NodeState;

/**
 * @brief Node information.
 */
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

/**
 * @brief Cluster configuration.
 */
typedef struct {
    const char *node_id;            /**< This node's ID. */
    const char *listen_address;     /**< Address to listen on. */
    const char *seed_nodes;         /**< Comma-separated seed nodes. */
    GV_NodeRole role;               /**< This node's role. */
    uint32_t heartbeat_interval_ms; /**< Heartbeat interval. */
    uint32_t failure_timeout_ms;    /**< Node failure timeout. */
} GV_ClusterConfig;

/**
 * @brief Cluster statistics.
 */
typedef struct {
    size_t total_nodes;             /**< Total nodes in cluster. */
    size_t active_nodes;            /**< Active nodes. */
    size_t total_shards;            /**< Total shards. */
    uint64_t total_vectors;         /**< Total vectors across cluster. */
    double avg_load;                /**< Average cluster load. */
} GV_ClusterStats;

/**
 * @brief Opaque cluster handle.
 */
typedef struct GV_Cluster GV_Cluster;

/* ============================================================================
 * Configuration
 * ============================================================================ */

/**
 * @brief Initialize cluster configuration with defaults.
 *
 * @param config Configuration to initialize.
 */
void gv_cluster_config_init(GV_ClusterConfig *config);

/* ============================================================================
 * Cluster Lifecycle
 * ============================================================================ */

/**
 * @brief Create a cluster instance.
 *
 * @param config Cluster configuration.
 * @return Cluster instance, or NULL on error.
 */
GV_Cluster *gv_cluster_create(const GV_ClusterConfig *config);

/**
 * @brief Destroy a cluster instance.
 *
 * @param cluster Cluster instance (safe to call with NULL).
 */
void gv_cluster_destroy(GV_Cluster *cluster);

/**
 * @brief Start cluster services.
 *
 * @param cluster Cluster instance.
 * @return 0 on success, -1 on error.
 */
int gv_cluster_start(GV_Cluster *cluster);

/**
 * @brief Stop cluster services.
 *
 * @param cluster Cluster instance.
 * @return 0 on success, -1 on error.
 */
int gv_cluster_stop(GV_Cluster *cluster);

/* ============================================================================
 * Node Management
 * ============================================================================ */

/**
 * @brief Get this node's information.
 *
 * @param cluster Cluster instance.
 * @param info Output node info.
 * @return 0 on success, -1 on error.
 */
int gv_cluster_get_local_node(GV_Cluster *cluster, GV_NodeInfo *info);

/**
 * @brief Get a node's information.
 *
 * @param cluster Cluster instance.
 * @param node_id Node ID.
 * @param info Output node info.
 * @return 0 on success, -1 on error.
 */
int gv_cluster_get_node(GV_Cluster *cluster, const char *node_id, GV_NodeInfo *info);

/**
 * @brief List all nodes.
 *
 * @param cluster Cluster instance.
 * @param nodes Output node array.
 * @param count Output count.
 * @return 0 on success, -1 on error.
 */
int gv_cluster_list_nodes(GV_Cluster *cluster, GV_NodeInfo **nodes, size_t *count);

/**
 * @brief Free node info.
 *
 * @param info Node info to free.
 */
void gv_cluster_free_node_info(GV_NodeInfo *info);

/**
 * @brief Free node list.
 *
 * @param nodes Node array to free.
 * @param count Number of nodes.
 */
void gv_cluster_free_node_list(GV_NodeInfo *nodes, size_t count);

/* ============================================================================
 * Cluster Operations
 * ============================================================================ */

/**
 * @brief Get cluster statistics.
 *
 * @param cluster Cluster instance.
 * @param stats Output statistics.
 * @return 0 on success, -1 on error.
 */
int gv_cluster_get_stats(GV_Cluster *cluster, GV_ClusterStats *stats);

/**
 * @brief Get the shard manager.
 *
 * @param cluster Cluster instance.
 * @return Shard manager, or NULL on error.
 */
GV_ShardManager *gv_cluster_get_shard_manager(GV_Cluster *cluster);

/**
 * @brief Check if cluster is healthy.
 *
 * @param cluster Cluster instance.
 * @return 1 if healthy, 0 if not, -1 on error.
 */
int gv_cluster_is_healthy(GV_Cluster *cluster);

/**
 * @brief Wait for cluster to be ready.
 *
 * @param cluster Cluster instance.
 * @param timeout_ms Timeout in milliseconds.
 * @return 0 if ready, -1 on timeout/error.
 */
int gv_cluster_wait_ready(GV_Cluster *cluster, uint32_t timeout_ms);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_CLUSTER_H */
