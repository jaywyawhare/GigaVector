/**
 * @file graph_distributed.h
 * @brief Cross-partition graph traversal.
 *
 * A large graph is partitioned across several GV_GraphDB instances (one per
 * node/shard); each node's adjacency lives on the partition that owns it, so an
 * edge may cross a partition boundary. graph_dist_khop() runs a breadth-first
 * k-hop expansion that routes every node's neighbour lookup to its owning
 * partition and deduplicates the visited set globally - the distributed
 * traversal primitive underneath multi-hop graph queries.
 */

#ifndef GIGAVECTOR_GV_GRAPH_DISTRIBUTED_H
#define GIGAVECTOR_GV_GRAPH_DISTRIBUTED_H

#include <stddef.h>
#include <stdint.h>

#include "features/graph_db.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Maps a node id to the index of the partition that owns it.
 * @param node_id Node identifier.
 * @param nparts  Number of partitions.
 * @param ctx     Opaque caller context.
 * @return Partition index in [0, nparts).
 */
typedef uint32_t (*GV_GraphPlacementFn)(uint64_t node_id, size_t nparts, void *ctx);

/**
 * @brief Breadth-first k-hop expansion across partitioned graph shards.
 *
 * Starting from @p start, follows neighbours for up to @p k hops, fetching each
 * node's neighbours from its owning partition (via @p place, or node_id %%
 * nparts when @p place is NULL). Every distinct node reached within k hops
 * (including @p start) is written to @p out_ids until it is full; the traversal
 * continues internally so the reachable set is complete even if @p out_ids
 * fills.
 *
 * @param parts       Array of @p nparts partition graphs (entries may be NULL).
 * @param nparts      Number of partitions (> 0).
 * @param place       Placement function, or NULL for modulo placement.
 * @param place_ctx   Context passed to @p place.
 * @param start       Starting node id.
 * @param k           Maximum hop distance.
 * @param predicate   Edge-label filter for each hop, or NULL for any out-edge.
 * @param out_ids     Output array of reached node ids.
 * @param max_out     Capacity of @p out_ids.
 * @return Number of node ids written to @p out_ids, or -1 on error.
 */
int graph_dist_khop(GV_GraphDB *const *parts, size_t nparts,
                    GV_GraphPlacementFn place, void *place_ctx,
                    uint64_t start, size_t k, const char *predicate,
                    uint64_t *out_ids, size_t max_out);

/**
 * @brief Fetch the neighbours of @p node_id into @p out (capacity @p max).
 * @return Number of neighbours written, or -1 on error.
 */
typedef int (*GV_GraphNeighborFn)(uint64_t node_id, uint64_t *out, size_t max,
                                  void *ctx);

/**
 * @brief Transport-agnostic k-hop BFS: the neighbour source is a callback, so
 *        the same traversal drives local partitions or remote (RPC) ones.
 *
 * @param fetch     Neighbour source (routes per node to local DB or remote node).
 * @param fetch_ctx Context passed to @p fetch.
 * @param max_nodes Upper bound on distinct nodes explored (sizes working sets).
 * @param start     Starting node id.
 * @param k         Maximum hop distance.
 * @param min_hops  Minimum shortest-hop distance for a node to be emitted (0
 *                  includes @p start; 1 excludes it, etc.).
 * @param out_ids   Output array of reached node ids.
 * @param max_out   Capacity of @p out_ids.
 * @return Number of node ids written, or -1 on error.
 */
int graph_dist_khop_fetch(GV_GraphNeighborFn fetch, void *fetch_ctx,
                          size_t max_nodes, uint64_t start, size_t k,
                          size_t min_hops, uint64_t *out_ids, size_t max_out);

/**
 * @brief Multi-source variant of graph_dist_khop_fetch: BFS seeded from all of
 *        @p starts at once (used to chain Cypher pattern segments, where one
 *        segment's result set is the next segment's frontier).
 *
 * @return Number of node ids written, or -1 on error.
 */
int graph_dist_khop_multi(GV_GraphNeighborFn fetch, void *fetch_ctx,
                          size_t max_nodes, const uint64_t *starts, size_t n_starts,
                          size_t k, size_t min_hops,
                          uint64_t *out_ids, size_t max_out);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_GRAPH_DISTRIBUTED_H */
