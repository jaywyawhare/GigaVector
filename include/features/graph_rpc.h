/**
 * @file graph_rpc.h
 * @brief TCP transport for cross-partition graph traversal.
 *
 * A node runs graph_rpc_serve() to answer neighbour lookups against its local
 * graph partition; a coordinator fetches neighbours from a remote partition with
 * graph_rpc_neighbors(). graph_dist_khop_net() drives a k-hop traversal over a
 * mix of local and remote partitions, routing each node's expansion to its owner.
 *
 * Wire format is a fixed binary frame; a homogeneous cluster is assumed.
 */

#ifndef GIGAVECTOR_GV_GRAPH_RPC_H
#define GIGAVECTOR_GV_GRAPH_RPC_H

#include <stddef.h>
#include <stdint.h>

#include "features/graph_db.h"
#include "features/graph_distributed.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GV_GraphRpcServer GV_GraphRpcServer;

/** @brief Start a neighbour-serving listener for a local partition (port 0 = ephemeral). */
GV_GraphRpcServer *graph_rpc_serve(GV_GraphDB *partition,
                                   const char *bind_addr, uint16_t port);

/** @brief Actual listening port. */
uint16_t graph_rpc_server_port(const GV_GraphRpcServer *server);

/** @brief Stop the listener, join its thread, and free the handle (NULL-safe). */
void graph_rpc_server_stop(GV_GraphRpcServer *server);

/**
 * @brief Fetch a node's neighbours from a remote partition.
 * @return Number of neighbour ids written to @p out (0..max), or -1 on error.
 */
int graph_rpc_neighbors(const char *host, uint16_t port, uint64_t node_id,
                        const char *predicate, uint64_t *out, size_t max);

/**
 * @brief A partition that is either local (in-process) or remote (host:port).
 */
typedef struct {
    GV_GraphDB *local;        /**< Non-NULL for a local partition. */
    const char *remote_addr;  /**< "host:port" for a remote partition. */
} GV_GraphPartition;

/**
 * @brief k-hop traversal across local and remote partitions.
 *
 * Routes each node's neighbour fetch to its owning partition (@p place, or
 * modulo), calling graph_get_neighbors() locally or graph_rpc_neighbors()
 * remotely, and merges a global visited set. An unreachable remote partition
 * contributes nothing rather than failing the query.
 *
 * @param parts     Array of @p nparts partitions (local or remote).
 * @param nparts    Number of partitions (> 0).
 * @param place     Placement fn (NULL = node_id %% nparts).
 * @param place_ctx Context for @p place.
 * @param start     Starting node id.
 * @param k         Maximum hop distance.
 * @param out_ids   Output array of reached node ids.
 * @param max_out   Capacity of @p out_ids.
 * @param max_nodes Upper bound on distinct nodes explored (sizes working sets).
 * @return Number of node ids written, or -1 on error.
 */
int graph_dist_khop_net(const GV_GraphPartition *parts, size_t nparts,
                        GV_GraphPlacementFn place, void *place_ctx,
                        uint64_t start, size_t k, const char *predicate,
                        uint64_t *out_ids, size_t max_out, size_t max_nodes);

/**
 * @brief Distributed variable-length pattern match:
 *        `MATCH (a)-[:rel_type*min_hops..max_hops]->(b) WHERE id(a)=start_id
 *        RETURN id(b)` executed across local and remote partitions.
 *
 * Returns the ids of nodes @c b whose shortest typed path from @p start_id has
 * length in [min_hops, max_hops]. @p rel_type NULL matches any edge type.
 *
 * @return Number of matched node ids written, or -1 on error.
 */
int cypher_dist_match(const GV_GraphPartition *parts, size_t nparts,
                      GV_GraphPlacementFn place, void *place_ctx,
                      uint64_t start_id, const char *rel_type,
                      size_t min_hops, size_t max_hops,
                      uint64_t *out_ids, size_t max_out, size_t max_nodes);

/** @brief One `-[:rel_type*min_hops..max_hops]->` segment of a chained pattern. */
typedef struct {
    const char *rel_type;   /**< Edge type, or NULL for any. */
    size_t      min_hops;
    size_t      max_hops;
} GV_CypherSegment;

/**
 * @brief Distributed chained multi-segment pattern match:
 *        `MATCH (a)-[seg0]->(x)-[seg1]->(y)...->(z) WHERE id(a)=start_id
 *        RETURN id(z)`.
 *
 * Each segment expands the previous segment's result set (multi-source), so the
 * pattern is evaluated left-to-right across partitions. Returns the final node
 * set (the @c z bindings).
 *
 * @return Number of matched node ids written, or -1 on error.
 */
int cypher_dist_path(const GV_GraphPartition *parts, size_t nparts,
                     GV_GraphPlacementFn place, void *place_ctx,
                     uint64_t start_id, const GV_CypherSegment *segments,
                     size_t n_segments, uint64_t *out_ids, size_t max_out,
                     size_t max_nodes);

/**
 * @brief Execute a constrained Cypher pattern across partitions.
 *
 * Parses `MATCH (a)-[:REL*MIN..MAX]->(b) WHERE id(a) = N RETURN b` (":REL" and
 * the min bound are optional: "*MAX" means exactly MAX, "*MIN.." style ranges
 * supported) and runs cypher_dist_match(). Returns matched @c b ids.
 *
 * @return Number of matched node ids written, or -1 on parse/exec error.
 */
int cypher_dist_query(const GV_GraphPartition *parts, size_t nparts,
                      GV_GraphPlacementFn place, void *place_ctx,
                      const char *query,
                      uint64_t *out_ids, size_t max_out, size_t max_nodes);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_GRAPH_RPC_H */
