#ifndef GIGAVECTOR_GV_SHARD_H
#define GIGAVECTOR_GV_SHARD_H

#include <stddef.h>
#include <stdint.h>

#include "core/types.h"       /* GV_SearchResult */
#include "search/distance.h"  /* GV_DistanceType */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file shard.h
 * @brief Shard management for distributed GigaVector.
 *
 * Provides sharding for horizontal scaling of vector data.
 */

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

/**
 * @brief Initialize shard configuration with defaults.
 *
 * @param config Configuration to initialize.
 */
void shard_config_init(GV_ShardConfig *config);

/**
 * @brief Create a shard manager.
 *
 * @param config Shard configuration (NULL for defaults).
 * @return Shard manager instance, or NULL on error.
 */
GV_ShardManager *shard_manager_create(const GV_ShardConfig *config);

/**
 * @brief Destroy a shard manager.
 *
 * @param mgr Shard manager instance (safe to call with NULL).
 */
void shard_manager_destroy(GV_ShardManager *mgr);

/**
 * @brief Add a shard.
 *
 * @param mgr Shard manager.
 * @param shard_id Shard ID.
 * @param node_address Node address (host:port).
 * @return 0 on success, -1 on error.
 */
int shard_add(GV_ShardManager *mgr, uint32_t shard_id, const char *node_address);

/**
 * @brief Remove a shard.
 *
 * @param mgr Shard manager.
 * @param shard_id Shard ID.
 * @return 0 on success, -1 on error.
 */
int shard_remove(GV_ShardManager *mgr, uint32_t shard_id);

/**
 * @brief Get shard for a vector ID.
 *
 * @param mgr Shard manager.
 * @param vector_id Vector ID.
 * @return Shard ID, or -1 on error.
 */
int shard_for_vector(GV_ShardManager *mgr, uint64_t vector_id);

/**
 * @brief Get shard for a key (consistent hashing).
 *
 * @param mgr Shard manager.
 * @param key Key data.
 * @param key_len Key length.
 * @return Shard ID, or -1 on error.
 */
int shard_for_key(GV_ShardManager *mgr, const void *key, size_t key_len);

/**
 * @brief Get shard information.
 *
 * @param mgr Shard manager.
 * @param shard_id Shard ID.
 * @param info Output shard info.
 * @return 0 on success, -1 on error.
 */
int shard_get_info(GV_ShardManager *mgr, uint32_t shard_id, GV_ShardInfo *info);

/**
 * @brief List all shards.
 *
 * @param mgr Shard manager.
 * @param shards Output array of shard info.
 * @param count Output count.
 * @return 0 on success, -1 on error.
 */
int shard_list(GV_ShardManager *mgr, GV_ShardInfo **shards, size_t *count);

/**
 * @brief Free shard info list.
 *
 * @param shards Shard info array.
 * @param count Number of shards.
 */
void shard_free_list(GV_ShardInfo *shards, size_t count);

/**
 * @brief Update shard state.
 *
 * @param mgr Shard manager.
 * @param shard_id Shard ID.
 * @param state New state.
 * @return 0 on success, -1 on error.
 */
int shard_set_state(GV_ShardManager *mgr, uint32_t shard_id, GV_ShardState state);

/**
 * @brief Start rebalancing shards (moves vectors between attached local DBs).
 *
 * Moves raw vectors and metadata only.
 *
 * @param mgr Shard manager.
 * @return 0 on success, -1 on error.
 */
int shard_rebalance_start(GV_ShardManager *mgr);

/**
 * @brief Check rebalancing status.
 *
 * @param mgr Shard manager.
 * @param progress Output progress (0.0 - 1.0).
 * @return 1 if rebalancing, 0 if not, -1 on error.
 */
int shard_rebalance_status(GV_ShardManager *mgr, double *progress);

/**
 * @brief Cancel rebalancing.
 *
 * @param mgr Shard manager.
 * @return 0 on success, -1 on error.
 */
int shard_rebalance_cancel(GV_ShardManager *mgr);

/**
 * @brief Attach a local database as a shard.
 *
 * @param mgr Shard manager.
 * @param shard_id Shard ID.
 * @param db Local database.
 * @return 0 on success, -1 on error.
 */
int shard_attach_local(GV_ShardManager *mgr, uint32_t shard_id, GV_Database *db);

/**
 * @brief Get local shard database.
 *
 * @param mgr Shard manager.
 * @param shard_id Shard ID.
 * @return Database pointer, or NULL if not local.
 */
GV_Database *shard_get_local_db(GV_ShardManager *mgr, uint32_t shard_id);

/**
 * @brief Dimension of this manager's local shard databases (0 if none).
 * Used to validate a network-supplied query dimension before searching.
 */
size_t shard_manager_dimension(GV_ShardManager *mgr);

/**
 * @brief Migrate vectors between attached local shard databases.
 *
 * Copies full vector metadata and inverted-index entries to the destination.
 *
 * @return Number of vectors migrated, or -1 on error.
 */
int shard_migrate_vectors(GV_ShardManager *mgr, uint32_t from_shard, uint32_t to_shard, size_t count);

/**
 * @brief Migrate one vector by index between attached local shard databases.
 *
 * @param mgr Shard manager.
 * @param from_shard Source shard ID.
 * @param to_shard Destination shard ID.
 * @param vector_index Vector index on the source shard database.
 * @param out_new_index Optional output for destination vector index.
 * @return 0 on success, -1 on error.
 */
int shard_migrate_vector_at(GV_ShardManager *mgr, uint32_t from_shard, uint32_t to_shard,
                            size_t vector_index, size_t *out_new_index);

/**
 * @brief Distributed k-NN search: scatter over local shard databases, gather
 *        the global top-k.
 *
 * Runs db_search() on every online shard that has an attached local database,
 * then merges the per-shard results into a single distance-ordered top-k. All
 * distance metrics rank smaller-is-better, so the merge is an ascending sort.
 *
 * On success @p results is filled with owned vectors exactly like db_search();
 * the caller frees them with gv_search_results_free().
 *
 * @param mgr           Shard manager.
 * @param query_data    Query vector (length == shard databases' dimension).
 * @param k             Number of neighbours to return.
 * @param results       Output array of at least @p k elements.
 * @param distance_type Distance metric to use.
 * @return Number of neighbours written (0..k), or -1 on error.
 */
int shard_search(GV_ShardManager *mgr, const float *query_data, size_t k,
                 GV_SearchResult *results, GV_DistanceType distance_type);

/**
 * @brief Merge owned candidate results into a distance-ordered top-k.
 *
 * Sorts @p all ascending by distance, copies the closest min(@p k, @p total)
 * into @p out (transferring each kept result's owned .vector), and frees the
 * discarded tail's vectors. Does NOT free the @p all buffer itself. Shared by
 * the local (shard_search) and distributed (shard_rpc) coordinators so there is
 * one merge implementation.
 *
 * @return Number of results written to @p out.
 */
size_t shard_merge_topk(GV_SearchResult *all, size_t total, size_t k,
                        GV_SearchResult *out);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_SHARD_H */
