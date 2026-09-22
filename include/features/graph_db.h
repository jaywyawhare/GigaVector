/**
 * @file graph_db.h
 * @brief Full graph database layer for GigaVector.
 *
 * Provides a property-graph model with nodes, directed edges, key-value
 * properties, traversal algorithms (BFS, DFS, Dijkstra, all-paths), analytics
 * (PageRank, clustering coefficient, connected components), and binary
 * persistence.
 */

#ifndef GIGAVECTOR_GV_GRAPH_DB_H
#define GIGAVECTOR_GV_GRAPH_DB_H

#include <stddef.h>
#include <stdint.h>

#include "core/prop_value.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Key-value property stored as a singly-linked list.
 */
typedef struct GV_GraphProp {
    char *key;                      /**< Property key (heap-allocated). */
    GV_PropValue value;             /**< Property value (typed). */
    struct GV_GraphProp *next;      /**< Next property in the list. */
} GV_GraphProp;

/**
 * @brief Lightweight edge reference stored in a node's adjacency list.
 */
typedef struct {
    uint64_t edge_id;               /**< Referenced edge identifier. */
    uint64_t neighbor_id;           /**< Node on the other end of the edge. */
} GV_GraphEdgeRef;

/**
 * @brief Graph node with label, properties, and adjacency lists.
 */
typedef struct {
    uint64_t node_id;               /**< Unique node identifier (>0). */
    char *label;                    /**< Primary type category (e.g. "Person"). Legacy single-label. */
    char **labels;                  /**< Array of label strings (heap-allocated). */
    size_t label_count;             /**< Number of labels. */
    size_t label_cap;               /**< Capacity of labels array. */
    GV_GraphProp *properties;       /**< Linked list of key-value pairs. */
    size_t prop_count;              /**< Number of properties. */
    GV_GraphEdgeRef *out_edges;     /**< Outgoing adjacency array. */
    size_t out_count;               /**< Number of outgoing edges. */
    size_t out_cap;                 /**< Capacity of out_edges array. */
    GV_GraphEdgeRef *in_edges;      /**< Incoming adjacency array. */
    size_t in_count;                /**< Number of incoming edges. */
    size_t in_cap;                  /**< Capacity of in_edges array. */
} GV_GraphNode;

/**
 * @brief Directed, weighted graph edge with label and properties.
 */
typedef struct {
    uint64_t edge_id;               /**< Unique edge identifier (>0). */
    uint64_t source_id;             /**< Source node identifier. */
    uint64_t target_id;             /**< Target node identifier. */
    char *label;                    /**< Primary relationship type (e.g. "KNOWS"). Legacy single-label. */
    char **labels;                  /**< Array of label strings (heap-allocated). */
    size_t label_count;             /**< Number of labels. */
    size_t label_cap;               /**< Capacity of labels array. */
    float weight;                   /**< Edge weight (default 1.0). */
    GV_GraphProp *properties;       /**< Linked list of key-value pairs. */
    size_t prop_count;              /**< Number of properties. */
} GV_GraphEdge;

/**
 * @brief Result of a path query (shortest path, all paths, etc.).
 */
typedef struct {
    uint64_t *node_ids;             /**< Ordered array of node IDs on the path. */
    uint64_t *edge_ids;             /**< Ordered array of edge IDs on the path. */
    size_t length;                  /**< Number of edges in the path. */
    float total_weight;             /**< Sum of edge weights along the path. */
} GV_GraphPath;

/**
 * @brief Configuration for creating a GV_GraphDB instance.
 */
typedef struct {
    size_t node_bucket_count;       /**< Hash table bucket count for nodes (default 4096). */
    size_t edge_bucket_count;       /**< Hash table bucket count for edges (default 8192). */
    int enforce_referential_integrity; /**< Check source/target exist on add_edge (default 1). */
} GV_GraphDBConfig;

/**
 * @brief Opaque graph database handle.
 */
typedef struct GV_GraphDB GV_GraphDB;

/**
 * @brief Opaque repeatable-read read transaction over a graph.
 *
 * A read transaction pins a consistent snapshot: between
 * graph_read_txn_begin() and graph_read_txn_end() every accessor sees the
 * same version of the graph even if other threads mutate concurrently.
 */
typedef struct GV_GraphReadTxn GV_GraphReadTxn;

/**
 * @brief Begin a repeatable-read read transaction.
 *
 * Holds the graph-wide read lock until graph_read_txn_end(). Writers are not
 * blocked from preparing mutations, but they cannot commit while any read
 * transaction is open. Keep transactions short; do not mutate the graph (or
 * call blocking functions) inside one.
 *
 * @param g Graph database; must be non-NULL.
 * @return Transaction handle, or NULL on error. Release with
 *         graph_read_txn_end().
 */
GV_GraphReadTxn *graph_read_txn_begin(GV_GraphDB *g);

/**
 * @brief End a read transaction and release its snapshot.
 *
 * @param txn Transaction handle; safe to call with NULL.
 */
void graph_read_txn_end(GV_GraphReadTxn *txn);

/**
 * @brief Graph mutation counter as of the transaction's snapshot.
 *
 * Compare with graph_version(): if equal, no mutation committed since the
 * transaction began.
 */
uint64_t graph_read_txn_version(const GV_GraphReadTxn *txn);

/**
 * @brief Current graph mutation counter (monotonic, bumps per mutation).
 */
uint64_t graph_version(const GV_GraphDB *g);

/** @brief graph_get_node() within the transaction's snapshot. */
const GV_GraphNode *graph_read_txn_get_node(GV_GraphReadTxn *txn,
                                            uint64_t node_id);

/** @brief graph_get_node_prop() within the transaction's snapshot. */
const char *graph_read_txn_get_node_prop(GV_GraphReadTxn *txn,
                                         uint64_t node_id, const char *key);

/** @brief graph_get_edge() within the transaction's snapshot. */
const GV_GraphEdge *graph_read_txn_get_edge(GV_GraphReadTxn *txn,
                                            uint64_t edge_id);

/** @brief graph_find_nodes_by_label() within the transaction's snapshot. */
int graph_read_txn_find_nodes_by_label(GV_GraphReadTxn *txn, const char *label,
                                       uint64_t *out_ids, size_t max_count);

/**
 * @brief Opaque write transaction over a graph.
 *
 * Mutations are staged locally and become visible to all threads only at
 * graph_write_txn_commit() — abort discards everything. The transaction
 * holds the graph write lock from begin() until commit/abort, so keep it
 * short-lived. On commit the whole batch is appended to the WAL (when
 * attached) with a single fsync before being applied.
 */
typedef struct GV_GraphWriteTxn GV_GraphWriteTxn;

/** @brief Begin an atomic write transaction (acquires the write lock). */
GV_GraphWriteTxn *graph_write_txn_begin(GV_GraphDB *g);

/** @brief Stage a node insert; returns its pre-assigned id (>0), or 0 on error. */
uint64_t graph_write_txn_add_node(GV_GraphWriteTxn *txn, const char *label);

/** @brief Stage a node removal (cascades to incident edges at commit). */
int graph_write_txn_remove_node(GV_GraphWriteTxn *txn, uint64_t node_id);

/** @brief Stage a directed weighted edge; returns its pre-assigned id (>0), or 0. */
uint64_t graph_write_txn_add_edge(GV_GraphWriteTxn *txn, uint64_t source,
                                  uint64_t target, const char *label,
                                  float weight);

/** @brief Stage an edge removal. */
int graph_write_txn_remove_edge(GV_GraphWriteTxn *txn, uint64_t edge_id);

/** @brief Stage a node property set. */
int graph_write_txn_set_node_prop(GV_GraphWriteTxn *txn, uint64_t node_id,
                                  const char *key, const char *value);

/** @brief Stage an edge property set. */
int graph_write_txn_set_edge_prop(GV_GraphWriteTxn *txn, uint64_t edge_id,
                                  const char *key, const char *value);

/**
 * @brief Commit atomically: one WAL batch + fsync, then apply under the lock.
 * @return 0 on success, -1 on error (the transaction is consumed either way).
 */
int graph_write_txn_commit(GV_GraphWriteTxn *txn);

/** @brief Abort and discard every staged mutation. Safe with NULL. */
void graph_write_txn_abort(GV_GraphWriteTxn *txn);

/**
 * @brief Acquire the graph-wide read lock for a consistent multi-call snapshot.
 *
 * Analytics that traverse many nodes/edges across several calls should wrap the
 * whole traversal in graph_read_lock()/graph_read_unlock() so concurrent
 * mutations cannot produce torn reads between calls. Do not mutate the graph
 * while holding the read lock, and never hold it across graph_save().
 *
 * @param g Graph database; must be non-NULL.
 */
void graph_read_lock(const GV_GraphDB *g);

/**
 * @brief Release the read lock acquired by graph_read_lock().
 *
 * @param g Graph database; must be non-NULL.
 */
void graph_read_unlock(const GV_GraphDB *g);

/**
 * @brief Node lookup WITHOUT taking the read lock.
 *
 * Only for callers that already hold graph_read_lock() (the pinned-snapshot
 * analytics layer, GV_GAContext). Using it without the lock is a data race.
 */
const GV_GraphNode *graph_get_node_unlocked(const GV_GraphDB *g, uint64_t node_id);

/** @brief Edge lookup without locking; see graph_get_node_unlocked(). */
const GV_GraphEdge *graph_get_edge_unlocked(const GV_GraphDB *g, uint64_t edge_id);

/**
 * @brief Enumerate all node ids WITHOUT taking the read lock; caller must
 *        already hold graph_read_lock().
 */
int graph_get_all_node_ids_unlocked(const GV_GraphDB *g, uint64_t *out_ids,
                                    size_t max_count);

/**
 * @brief Initialize a configuration struct with default values.
 *
 * @param config Configuration to initialize; must be non-NULL.
 */
void graph_config_init(GV_GraphDBConfig *config);

/**
 * @brief Create a new graph database.
 *
 * @param config Configuration; NULL for defaults.
 * @return Allocated graph database, or NULL on error.
 */
GV_GraphDB *graph_create(const GV_GraphDBConfig *config);

/**
 * @brief Destroy a graph database and free all resources.
 *
 * @param g Graph database to destroy; safe to call with NULL.
 */
void graph_destroy(GV_GraphDB *g);

/**
 * @brief Add a new node with the given label.
 *
 * @param g Graph database; must be non-NULL.
 * @param label Node label (will be copied); must be non-NULL.
 * @return Newly assigned node_id (>0), or 0 on error.
 */
uint64_t graph_add_node(GV_GraphDB *g, const char *label);

/**
 * @brief Add a node with a caller-chosen id (e.g. a global id in a partitioned
 *        graph).
 *
 * @param g Graph database; must be non-NULL.
 * @param node_id Desired node id (>0); must not already exist.
 * @param label Node label (will be copied); must be non-NULL.
 * @return @p node_id on success, or 0 if it already exists or on error.
 */
uint64_t graph_add_node_with_id(GV_GraphDB *g, uint64_t node_id, const char *label);

/**
 * @brief Remove a node and cascade-delete all its incident edges.
 *
 * @param g Graph database; must be non-NULL.
 * @param node_id Node to remove.
 * @return 0 on success, -1 if not found or on error.
 */
int graph_remove_node(GV_GraphDB *g, uint64_t node_id);

/**
 * @brief Look up a node by ID.
 *
 * @param g Graph database; must be non-NULL.
 * @param node_id Node identifier.
 * @return Pointer to the node (valid only while no other thread mutates the
 *         graph — a concurrent remove_node frees it; wrap use in
 *         graph_read_lock()/graph_read_unlock() for safety), or NULL if not found.
 */
const GV_GraphNode *graph_get_node(const GV_GraphDB *g, uint64_t node_id);

/**
 * @brief Set (or overwrite) a property on a node.
 *
 * @param g Graph database; must be non-NULL.
 * @param node_id Target node.
 * @param key Property key (will be copied); must be non-NULL.
 * @param value Property value (will be copied); must be non-NULL.
 * @return 0 on success, -1 on error.
 */
int graph_set_node_prop(GV_GraphDB *g, uint64_t node_id,
                           const char *key, const char *value);

/**
 * @brief Get a property value from a node.
 *
 * @param g Graph database; must be non-NULL.
 * @param node_id Target node.
 * @param key Property key; must be non-NULL.
 * @return Property value string (valid until next mutation), or NULL if not found.
 */
const char *graph_get_node_prop(const GV_GraphDB *g, uint64_t node_id,
                                   const char *key);

/**
 * @brief Find all nodes with a given label.
 *
 * @param g Graph database; must be non-NULL.
 * @param label Label to search for; must be non-NULL.
 * @param out_ids Output array for matching node IDs; must be pre-allocated.
 * @param max_count Capacity of out_ids.
 * @return Number of matching nodes written to out_ids, or -1 on error.
 */
int graph_find_nodes_by_label(const GV_GraphDB *g, const char *label,
                                 uint64_t *out_ids, size_t max_count);

/**
 * @brief Add a directed, weighted edge between two nodes.
 *
 * @param g Graph database; must be non-NULL.
 * @param source Source node ID.
 * @param target Target node ID.
 * @param label Relationship label (will be copied); must be non-NULL.
 * @param weight Edge weight (typically >= 0; use 1.0 as default).
 * @return Newly assigned edge_id (>0), or 0 on error.
 */
uint64_t graph_add_edge(GV_GraphDB *g, uint64_t source, uint64_t target,
                           const char *label, float weight);

/**
 * @brief Remove an edge by ID.
 *
 * @param g Graph database; must be non-NULL.
 * @param edge_id Edge to remove.
 * @return 0 on success, -1 if not found or on error.
 */
int graph_remove_edge(GV_GraphDB *g, uint64_t edge_id);

/** One recorded edge mutation (incremental-CSR support). */
typedef struct {
    uint64_t version;   /**< mutation_count value AFTER the change. */
    uint64_t source;    /**< source node id. */
    uint64_t target;    /**< target node id. */
    int      removed;   /**< non-zero if the edge was removed. */
} GV_GraphEdgeDelta;

/**
 * @brief Fetch edge mutations recorded after `since_version`.
 *
 * Used by the CSR cache to patch matrices incrementally. The journal covers
 * plain edge add/remove only — node changes, write transactions and WAL
 * replay invalidate coverage.
 *
 * @param g Graph database; must be non-NULL.
 * @param since_version caller's last observed graph_version().
 * @param out Optional buffer (may be NULL for a size query).
 * @param cap Capacity of `out`.
 * @param out_len Set to the number of available deltas (never NULL).
 * @return 0 if covered, -1 if the journal cannot serve `since_version`
 *         (caller must do a full rebuild).
 */
int graph_edge_deltas_since(const GV_GraphDB *g, uint64_t since_version,
                            GV_GraphEdgeDelta *out, size_t cap,
                            size_t *out_len);

/**
 * @brief Look up an edge by ID.
 *
 * @param g Graph database; must be non-NULL.
 * @param edge_id Edge identifier.
 * @return Pointer to the edge (valid only while no other thread mutates the
 *         graph — a concurrent remove_edge frees it; wrap use in
 *         graph_read_lock()/graph_read_unlock() for safety), or NULL if not found.
 */
const GV_GraphEdge *graph_get_edge(const GV_GraphDB *g, uint64_t edge_id);

/**
 * @brief Set (or overwrite) a property on an edge.
 *
 * @param g Graph database; must be non-NULL.
 * @param edge_id Target edge.
 * @param key Property key (will be copied); must be non-NULL.
 * @param value Property value (will be copied); must be non-NULL.
 * @return 0 on success, -1 on error.
 */
int graph_set_edge_prop(GV_GraphDB *g, uint64_t edge_id,
                           const char *key, const char *value);

/**
 * @brief Get a property value from an edge.
 *
 * @param g Graph database; must be non-NULL.
 * @param edge_id Target edge.
 * @param key Property key; must be non-NULL.
 * @return Property value string (valid until next mutation), or NULL if not found.
 */
const char *graph_get_edge_prop(const GV_GraphDB *g, uint64_t edge_id,
                                   const char *key);

/**
 * @brief Get outgoing edge IDs from a node.
 *
 * @param g Graph database; must be non-NULL.
 * @param node_id Source node.
 * @param out_ids Output array for edge IDs; must be pre-allocated.
 * @param max_count Capacity of out_ids.
 * @return Number of edge IDs written, or -1 on error.
 */
int graph_get_edges_out(const GV_GraphDB *g, uint64_t node_id,
                           uint64_t *out_ids, size_t max_count);

/**
 * @brief Get incoming edge IDs to a node.
 *
 * @param g Graph database; must be non-NULL.
 * @param node_id Target node.
 * @param out_ids Output array for edge IDs; must be pre-allocated.
 * @param max_count Capacity of out_ids.
 * @return Number of edge IDs written, or -1 on error.
 */
int graph_get_edges_in(const GV_GraphDB *g, uint64_t node_id,
                          uint64_t *out_ids, size_t max_count);

/**
 * @brief Get unique neighbor node IDs (union of out-neighbors and in-neighbors).
 *
 * @param g Graph database; must be non-NULL.
 * @param node_id Node whose neighbors to retrieve.
 * @param out_ids Output array for neighbor node IDs; must be pre-allocated.
 * @param max_count Capacity of out_ids.
 * @return Number of unique neighbor IDs written, or -1 on error.
 */
int graph_get_neighbors(const GV_GraphDB *g, uint64_t node_id,
                           uint64_t *out_ids, size_t max_count);

/**
 * @brief Get outgoing neighbour node IDs, optionally filtered by edge label.
 *
 * Directed (out-edges only). When @p predicate is non-NULL, only neighbours
 * reached by an edge whose label equals @p predicate are returned — the typed
 * traversal step used by graph queries.
 *
 * @param g Graph database; must be non-NULL.
 * @param node_id Source node.
 * @param predicate Edge label to match, or NULL for any out-edge.
 * @param out_ids Output array for neighbour node IDs; must be pre-allocated.
 * @param max_count Capacity of out_ids.
 * @return Number of neighbour IDs written, or -1 on error.
 */
int graph_get_out_neighbors_typed(const GV_GraphDB *g, uint64_t node_id,
                                  const char *predicate,
                                  uint64_t *out_ids, size_t max_count);

/**
 * @brief Breadth-first search from a starting node.
 *
 * @param g Graph database; must be non-NULL.
 * @param start Starting node ID.
 * @param max_depth Maximum BFS depth (0 = start only).
 * @param out_ids Output array for visited node IDs; must be pre-allocated.
 * @param max_count Capacity of out_ids.
 * @return Number of visited nodes written, or -1 on error.
 */
int graph_bfs(const GV_GraphDB *g, uint64_t start, size_t max_depth,
                 uint64_t *out_ids, size_t max_count);

/**
 * @brief Depth-first search from a starting node.
 *
 * @param g Graph database; must be non-NULL.
 * @param start Starting node ID.
 * @param max_depth Maximum DFS depth (0 = start only).
 * @param out_ids Output array for visited node IDs; must be pre-allocated.
 * @param max_count Capacity of out_ids.
 * @return Number of visited nodes written, or -1 on error.
 */
int graph_dfs(const GV_GraphDB *g, uint64_t start, size_t max_depth,
                 uint64_t *out_ids, size_t max_count);

/**
 * @brief Find the weighted shortest path using Dijkstra's algorithm.
 *
 * @param g Graph database; must be non-NULL.
 * @param from Source node ID.
 * @param to Destination node ID.
 * @param path Output path structure (caller must free with graph_free_path).
 * @return 0 on success, -1 if no path exists or on error.
 */
int graph_shortest_path(const GV_GraphDB *g, uint64_t from, uint64_t to,
                           GV_GraphPath *path);

/**
 * @brief Find all simple paths between two nodes up to a maximum depth.
 *
 * @param g Graph database; must be non-NULL.
 * @param from Source node ID.
 * @param to Destination node ID.
 * @param max_depth Maximum path length in edges.
 * @param paths Output array of GV_GraphPath; must be pre-allocated.
 * @param max_paths Capacity of paths array.
 * @return Number of paths found, or -1 on error.
 */
int graph_all_paths(const GV_GraphDB *g, uint64_t from, uint64_t to,
                       size_t max_depth, GV_GraphPath *paths, size_t max_paths);

/**
 * @brief Free memory associated with a path structure.
 *
 * @param path Path to free; safe to call with NULL.
 */
void graph_free_path(GV_GraphPath *path);

/**
 * @brief Compute the PageRank score for a single node.
 *
 * Uses the iterative power method over the entire graph.
 *
 * @param g Graph database; must be non-NULL.
 * @param node_id Node whose PageRank to return.
 * @param iterations Number of power iterations.
 * @param damping Damping factor (typically 0.85).
 * @return PageRank score, or 0.0 on error.
 */
float graph_pagerank(const GV_GraphDB *g, uint64_t node_id,
                        size_t iterations, float damping);

/**
 * @brief Total degree (in + out) of a node.
 *
 * @param g Graph database; must be non-NULL.
 * @param node_id Node identifier.
 * @return Degree, or 0 if node not found.
 */
size_t graph_degree(const GV_GraphDB *g, uint64_t node_id);

/**
 * @brief In-degree of a node.
 *
 * @param g Graph database; must be non-NULL.
 * @param node_id Node identifier.
 * @return In-degree, or 0 if node not found.
 */
size_t graph_in_degree(const GV_GraphDB *g, uint64_t node_id);

/**
 * @brief Out-degree of a node.
 *
 * @param g Graph database; must be non-NULL.
 * @param node_id Node identifier.
 * @return Out-degree, or 0 if node not found.
 */
size_t graph_out_degree(const GV_GraphDB *g, uint64_t node_id);

/**
 * @brief Identify connected components in the graph (treating edges as undirected).
 *
 * Assigns a component ID to each node. The component_ids array is indexed by
 * the order in which nodes are enumerated from the hash table; use in
 * conjunction with a full node scan.
 *
 * @param g Graph database; must be non-NULL.
 * @param component_ids Output array (one entry per node); must be pre-allocated
 *                      to at least graph_node_count(g) entries.
 * @param max_count Capacity of component_ids.
 * @return Number of distinct connected components, or -1 on error.
 */
int graph_connected_components(const GV_GraphDB *g,
                                  uint64_t *component_ids, size_t max_count);

/**
 * @brief Local clustering coefficient of a node.
 *
 * Measures the fraction of a node's neighbor pairs that are themselves
 * connected (treating edges as undirected).
 *
 * @param g Graph database; must be non-NULL.
 * @param node_id Node identifier.
 * @return Clustering coefficient in [0.0, 1.0], or 0.0 if node not found or
 *         has fewer than 2 neighbors.
 */
float graph_clustering_coefficient(const GV_GraphDB *g, uint64_t node_id);

/**
 * @brief Return the number of nodes in the graph.
 *
 * @param g Graph database; must be non-NULL.
 * @return Node count.
 */
size_t graph_node_count(const GV_GraphDB *g);

/**
 * @brief Return the number of edges in the graph.
 *
 * @param g Graph database; must be non-NULL.
 * @return Edge count.
 */
size_t graph_edge_count(const GV_GraphDB *g);

/**
 * @brief Enumerate all node IDs in the graph.
 *
 * Fills out_ids (which must hold at least graph_node_count(g) entries) with
 * every node ID. Order is unspecified but stable within a single call. Used as
 * the enumeration primitive by the graph-algorithms layer (graph_algos.h).
 *
 * @param g Graph database; must be non-NULL.
 * @param out_ids Output array; must be pre-allocated to >= graph_node_count(g).
 * @param max_count Capacity of out_ids.
 * @return Number of node IDs written, or -1 if max_count < node_count or on error.
 */
int graph_get_all_node_ids(const GV_GraphDB *g, uint64_t *out_ids, size_t max_count);

/**
 * @brief Add a label to a node (multi-label support).
 *
 * @param g Graph database; must be non-NULL.
 * @param node_id Target node.
 * @param label Label to add (will be copied); must be non-NULL.
 * @return 0 on success, -1 if not found, already present, or on error.
 */
int graph_node_add_label(GV_GraphDB *g, uint64_t node_id, const char *label);

/**
 * @brief Remove a label from a node.
 *
 * @param g Graph database; must be non-NULL.
 * @param node_id Target node.
 * @param label Label to remove; must be non-NULL.
 * @return 0 on success, -1 if not found, label not present, or on error.
 */
int graph_node_remove_label(GV_GraphDB *g, uint64_t node_id, const char *label);

/**
 * @brief Check whether a node has a specific label.
 *
 * @param node Node to query; must be non-NULL.
 * @param label Label to check; must be non-NULL.
 * @return 1 if the label is present, 0 otherwise.
 */
int graph_node_has_label(const GV_GraphNode *node, const char *label);

/**
 * @brief Add a label to an edge (multi-label support).
 *
 * @param g Graph database; must be non-NULL.
 * @param edge_id Target edge.
 * @param label Label to add (will be copied); must be non-NULL.
 * @return 0 on success, -1 if not found, already present, or on error.
 */
int graph_edge_add_label(GV_GraphDB *g, uint64_t edge_id, const char *label);

/**
 * @brief Remove a label from an edge.
 *
 * @param g Graph database; must be non-NULL.
 * @param edge_id Target edge.
 * @param label Label to remove; must be non-NULL.
 * @return 0 on success, -1 if not found, label not present, or on error.
 */
int graph_edge_remove_label(GV_GraphDB *g, uint64_t edge_id, const char *label);

/**
 * @brief Set a node property as int64.
 *
 * @return 0 on success, -1 on error.
 */
int graph_set_node_prop_int64(GV_GraphDB *g, uint64_t node_id,
                               const char *key, int64_t value);

/**
 * @brief Set a node property as float64.
 *
 * @return 0 on success, -1 on error.
 */
int graph_set_node_prop_float64(GV_GraphDB *g, uint64_t node_id,
                                 const char *key, double value);

/**
 * @brief Set a node property as bool.
 *
 * @return 0 on success, -1 on error.
 */
int graph_set_node_prop_bool(GV_GraphDB *g, uint64_t node_id,
                              const char *key, int value);

/**
 * @brief Set an edge property as int64.
 *
 * @return 0 on success, -1 on error.
 */
int graph_set_edge_prop_int64(GV_GraphDB *g, uint64_t edge_id,
                               const char *key, int64_t value);

/**
 * @brief Set an edge property as float64.
 *
 * @return 0 on success, -1 on error.
 */
int graph_set_edge_prop_float64(GV_GraphDB *g, uint64_t edge_id,
                                 const char *key, double value);

/**
 * @brief Set an edge property as bool.
 *
 * @return 0 on success, -1 on error.
 */
int graph_set_edge_prop_bool(GV_GraphDB *g, uint64_t edge_id,
                              const char *key, int value);

/**
 * @brief Get a node property as int64.
 *
 * @param fallback Value returned when the property is missing or wrong type.
 * @return Property value, or fallback.
 */
int64_t graph_get_node_prop_int64(const GV_GraphDB *g, uint64_t node_id,
                                   const char *key, int64_t fallback);

/**
 * @brief Get a node property as float64.
 *
 * @param fallback Value returned when the property is missing or wrong type.
 * @return Property value, or fallback.
 */
double graph_get_node_prop_float64(const GV_GraphDB *g, uint64_t node_id,
                                    const char *key, double fallback);

/**
 * @brief Get a node property as int (bool).
 *
 * @param fallback Value returned when the property is missing or wrong type.
 * @return Property value, or fallback.
 */
int graph_get_node_prop_bool(const GV_GraphDB *g, uint64_t node_id,
                              const char *key, int fallback);

/**
 * @brief Get an edge property as int64.
 *
 * @param fallback Value returned when the property is missing or wrong type.
 * @return Property value, or fallback.
 */
int64_t graph_get_edge_prop_int64(const GV_GraphDB *g, uint64_t edge_id,
                                   const char *key, int64_t fallback);

/**
 * @brief Get an edge property as float64.
 *
 * @param fallback Value returned when the property is missing or wrong type.
 * @return Property value, or fallback.
 */
double graph_get_edge_prop_float64(const GV_GraphDB *g, uint64_t edge_id,
                                    const char *key, double fallback);

/**
 * @brief Get an edge property as int (bool).
 *
 * @param fallback Value returned when the property is missing or wrong type.
 * @return Property value, or fallback.
 */
int graph_get_edge_prop_bool(const GV_GraphDB *g, uint64_t edge_id,
                              const char *key, int fallback);

/**
 * @brief Find nodes whose property falls within [min_val, max_val].
 *
 * Only compares values of the same type. Returns node IDs of matches.
 *
 * @param g Graph database; must be non-NULL.
 * @param key Property key to filter on; must be non-NULL.
 * @param min_val Minimum value (inclusive, by gv_prop_compare ordering).
 * @param max_val Maximum value (inclusive).
 * @param out_ids Output array for matching node IDs; must be pre-allocated.
 * @param max_count Capacity of out_ids.
 * @return Number of matching nodes written, or -1 on error.
 */
int graph_find_nodes_by_prop(const GV_GraphDB *g, const char *key,
                             GV_PropValue min_val, GV_PropValue max_val,
                             uint64_t *out_ids, size_t max_count);

/**
 * @brief Save the graph to a binary file.
 *
 * Format uses magic bytes "GVGR" followed by version, counts, and serialized
 * nodes/edges with their properties.
 *
 * If @p path is the snapshot this graph's WAL is attached to (see
 * graph_wal_attach), a successful save also truncates the WAL — it acts as a
 * checkpoint. The file is fsync'd before returning.
 *
 * @param g Graph database; must be non-NULL.
 * @param path File path to write; must be non-NULL.
 * @return 0 on success, -1 on error.
 */
int graph_save(const GV_GraphDB *g, const char *path);

/**
 * @brief Load a graph from a binary file previously written by graph_save.
 *
 * If "<path>.wal" exists, it is replayed on top of the snapshot (crash
 * recovery) and stays attached for future appends.
 *
 * @param path File path to read; must be non-NULL.
 * @return Loaded graph database, or NULL on error.
 */
GV_GraphDB *graph_load(const char *path);

/**
 * @brief Attach a write-ahead log for crash-safe durability.
 *
 * Every subsequent mutation is appended to "<snapshot_path>.wal" and fsync'd
 * before the mutating call returns. After a crash, graph_load(snapshot_path)
 * replays the WAL over the last saved snapshot.
 *
 * @param g Graph database; must be non-NULL.
 * @param snapshot_path Path of the snapshot file the log belongs to; must be
 *                      non-NULL.
 * @return 0 on success, -1 on error.
 */
int graph_wal_attach(GV_GraphDB *g, const char *snapshot_path);

/**
 * @brief Truncate and fsync the attached WAL (checkpoint).
 *
 * Call after graph_save() when saving to a different path than the one the
 * WAL was attached to. graph_save() already checkpoints automatically when
 * the paths match.
 *
 * @param g Graph database; must be non-NULL.
 * @return 0 on success, -1 if no WAL is attached or on error.
 */
int graph_wal_checkpoint(GV_GraphDB *g);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_GRAPH_DB_H */
