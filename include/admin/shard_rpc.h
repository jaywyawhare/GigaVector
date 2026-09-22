/**
 * @file shard_rpc.h
 * @brief Minimal TCP transport for distributed k-NN search across shard nodes.
 *
 * Turns the in-process shard_search() into a networked one: a node runs
 * shard_rpc_serve() to answer search requests against its local shards, and a
 * coordinator calls shard_rpc_search() to query a remote node and merge the
 * results with its own (via the same distance-ordered top-k merge).
 *
 * Wire format is a fixed binary frame; a cluster is assumed homogeneous
 * (matching float/size endianness), as is typical for internal DB RPC.
 */

#ifndef GIGAVECTOR_GV_SHARD_RPC_H
#define GIGAVECTOR_GV_SHARD_RPC_H

#include <stddef.h>
#include <stdint.h>

#include "core/types.h"       /* GV_SearchResult */
#include "search/distance.h"  /* GV_DistanceType */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GV_ShardManager GV_ShardManager;
typedef struct GV_ShardRpcServer GV_ShardRpcServer;

/**
 * @brief Start a search-serving listener bound to a shard manager.
 *
 * Accepts connections on a background thread; each request is answered by
 * shard_search() over @p mgr's local shards. Pass port 0 to bind an ephemeral
 * port (retrieve it with shard_rpc_server_port()).
 *
 * @return Server handle, or NULL on error.
 */
GV_ShardRpcServer *shard_rpc_serve(GV_ShardManager *mgr,
                                   const char *bind_addr, uint16_t port);

/** @brief Actual listening port (useful after binding port 0). */
uint16_t shard_rpc_server_port(const GV_ShardRpcServer *server);

/**
 * @brief Fill @p out with this node's membership list (e.g. newline-separated
 *        "host:port" addresses). @return byte length written, or -1.
 */
typedef int (*GV_MembersProviderFn)(char *out, size_t out_sz, void *ctx);

/**
 * @brief Register a membership provider so the listener also answers peer
 *        membership pulls over the same port (used for gossip discovery).
 */
void shard_rpc_set_members_provider(GV_ShardRpcServer *server,
                                    GV_MembersProviderFn fn, void *ctx);

/**
 * @brief Pull a peer's membership list into @p out (NUL-terminated).
 * @return byte length, or -1 on error (peer unreachable / no provider).
 */
int shard_rpc_fetch_members(const char *host, uint16_t port,
                            char *out, size_t out_sz);

/** @brief Handler for an inbound raft message payload. */
typedef void (*GV_RaftMsgHandler)(const void *bytes, size_t len, void *ctx);

/** @brief Register a handler for inbound raft messages on this listener. */
void shard_rpc_set_raft_handler(GV_ShardRpcServer *server,
                                GV_RaftMsgHandler fn, void *ctx);

/**
 * @brief Fire-and-forget delivery of a serialized raft message to a peer.
 * @return 0 on success, -1 if the peer is unreachable.
 */
int shard_rpc_send_raft(const char *host, uint16_t port,
                        const void *bytes, size_t len);

/** @brief Stop the listener, join its thread, and free the handle (NULL-safe). */
void shard_rpc_server_stop(GV_ShardRpcServer *server);

/**
 * @brief Query a remote shard node for its top-k neighbours.
 *
 * @param host          Remote host.
 * @param port          Remote shard_rpc port.
 * @param query         Query vector.
 * @param dim           Query dimension.
 * @param k             Number of neighbours to request.
 * @param distance_type Distance metric.
 * @param results       Output array of at least @p k elements; returned vectors
 *                      are owned by the caller (free with gv_search_results_free).
 * @return Number of neighbours written (0..k), or -1 on error.
 */
int shard_rpc_search(const char *host, uint16_t port,
                     const float *query, size_t dim, size_t k,
                     GV_DistanceType distance_type, GV_SearchResult *results);

/**
 * @brief Distributed k-NN across a shard manager's local AND remote shards.
 *
 * Runs shard_search() over the manager's local shards, then fans out once to
 * each distinct remote node (a shard with no attached local DB, its
 * node_address being that node's shard_rpc endpoint "host:port") via
 * shard_rpc_search(), and merges everything into a global top-k. A remote node
 * that is unreachable is skipped, yielding partial (best-effort) results rather
 * than failing the whole query.
 *
 * @param mgr           Shard manager (mix of local and remote shards).
 * @param query         Query vector.
 * @param dim           Query dimension (needed to serialize remote requests).
 * @param k             Number of neighbours to return.
 * @param distance_type Distance metric.
 * @param results       Output array of at least @p k elements (caller frees via
 *                      gv_search_results_free).
 * @return Number of neighbours written (0..k), or -1 on error.
 */
int shard_rpc_search_distributed(GV_ShardManager *mgr, const float *query,
                                 size_t dim, size_t k,
                                 GV_DistanceType distance_type,
                                 GV_SearchResult *results);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_SHARD_RPC_H */
