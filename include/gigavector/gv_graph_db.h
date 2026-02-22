/**
 * @file gv_graph_db.h
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

#ifdef __cplusplus
extern "C" {
#endif

/* Data Structures */

/**
 * @brief Key-value property stored as a singly-linked list.
 */
typedef struct GV_GraphProp {
    char *key;                      /**< Property key (heap-allocated). */
    char *value;                    /**< Property value (heap-allocated). */
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
    char *label;                    /**< Type category (e.g. "Person"). */
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
    char *label;                    /**< Relationship type (e.g. "KNOWS"). */
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

/* Lifecycle */

/**
 * @brief Initialize a configuration struct with default values.
 *
 * @param config Configuration to initialize; must be non-NULL.
 */
void gv_graph_config_init(GV_GraphDBConfig *config);

/**
 * @brief Create a new graph database.
 *
 * @param config Configuration; NULL for defaults.
 * @return Allocated graph database, or NULL on error.
 */
GV_GraphDB *gv_graph_create(const GV_GraphDBConfig *config);

/**
 * @brief Destroy a graph database and free all resources.
 *
 * @param g Graph database to destroy; safe to call with NULL.
 */
void gv_graph_destroy(GV_GraphDB *g);

/* Node Operations */

/**
 * @brief Add a new node with the given label.
 *
 * @param g Graph database; must be non-NULL.
 * @param label Node label (will be copied); must be non-NULL.
 * @return Newly assigned node_id (>0), or 0 on error.
 */
uint64_t gv_graph_add_node(GV_GraphDB *g, const char *label);

/**
 * @brief Remove a node and cascade-delete all its incident edges.
 *
 * @param g Graph database; must be non-NULL.
 * @param node_id Node to remove.
 * @return 0 on success, -1 if not found or on error.
 */
int gv_graph_remove_node(GV_GraphDB *g, uint64_t node_id);

/**
 * @brief Look up a node by ID.
 *
 * @param g Graph database; must be non-NULL.
 * @param node_id Node identifier.
 * @return Pointer to the node (valid until next mutation), or NULL if not found.
 */
const GV_GraphNode *gv_graph_get_node(const GV_GraphDB *g, uint64_t node_id);

/**
 * @brief Set (or overwrite) a property on a node.
 *
 * @param g Graph database; must be non-NULL.
 * @param node_id Target node.
 * @param key Property key (will be copied); must be non-NULL.
 * @param value Property value (will be copied); must be non-NULL.
 * @return 0 on success, -1 on error.
 */
int gv_graph_set_node_prop(GV_GraphDB *g, uint64_t node_id,
                           const char *key, const char *value);

/**
 * @brief Get a property value from a node.
 *
 * @param g Graph database; must be non-NULL.
 * @param node_id Target node.
 * @param key Property key; must be non-NULL.
 * @return Property value string (valid until next mutation), or NULL if not found.
 */
const char *gv_graph_get_node_prop(const GV_GraphDB *g, uint64_t node_id,
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
int gv_graph_find_nodes_by_label(const GV_GraphDB *g, const char *label,
                                 uint64_t *out_ids, size_t max_count);

/* Edge Operations */

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
uint64_t gv_graph_add_edge(GV_GraphDB *g, uint64_t source, uint64_t target,
                           const char *label, float weight);

/**
 * @brief Remove an edge by ID.
 *
 * @param g Graph database; must be non-NULL.
 * @param edge_id Edge to remove.
 * @return 0 on success, -1 if not found or on error.
 */
int gv_graph_remove_edge(GV_GraphDB *g, uint64_t edge_id);

/**
 * @brief Look up an edge by ID.
 *
 * @param g Graph database; must be non-NULL.
 * @param edge_id Edge identifier.
 * @return Pointer to the edge (valid until next mutation), or NULL if not found.
 */
const GV_GraphEdge *gv_graph_get_edge(const GV_GraphDB *g, uint64_t edge_id);

/**
 * @brief Set (or overwrite) a property on an edge.
 *
 * @param g Graph database; must be non-NULL.
 * @param edge_id Target edge.
 * @param key Property key (will be copied); must be non-NULL.
 * @param value Property value (will be copied); must be non-NULL.
 * @return 0 on success, -1 on error.
 */
int gv_graph_set_edge_prop(GV_GraphDB *g, uint64_t edge_id,
                           const char *key, const char *value);

/**
 * @brief Get a property value from an edge.
 *
 * @param g Graph database; must be non-NULL.
 * @param edge_id Target edge.
 * @param key Property key; must be non-NULL.
 * @return Property value string (valid until next mutation), or NULL if not found.
 */
const char *gv_graph_get_edge_prop(const GV_GraphDB *g, uint64_t edge_id,
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
int gv_graph_get_edges_out(const GV_GraphDB *g, uint64_t node_id,
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
int gv_graph_get_edges_in(const GV_GraphDB *g, uint64_t node_id,
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
int gv_graph_get_neighbors(const GV_GraphDB *g, uint64_t node_id,
                           uint64_t *out_ids, size_t max_count);

/* Traversal */

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
int gv_graph_bfs(const GV_GraphDB *g, uint64_t start, size_t max_depth,
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
int gv_graph_dfs(const GV_GraphDB *g, uint64_t start, size_t max_depth,
                 uint64_t *out_ids, size_t max_count);

/**
 * @brief Find the weighted shortest path using Dijkstra's algorithm.
 *
 * @param g Graph database; must be non-NULL.
 * @param from Source node ID.
 * @param to Destination node ID.
 * @param path Output path structure (caller must free with gv_graph_free_path).
 * @return 0 on success, -1 if no path exists or on error.
 */
int gv_graph_shortest_path(const GV_GraphDB *g, uint64_t from, uint64_t to,
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
int gv_graph_all_paths(const GV_GraphDB *g, uint64_t from, uint64_t to,
                       size_t max_depth, GV_GraphPath *paths, size_t max_paths);

/**
 * @brief Free memory associated with a path structure.
 *
 * @param path Path to free; safe to call with NULL.
 */
void gv_graph_free_path(GV_GraphPath *path);

/* Analytics */

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
float gv_graph_pagerank(const GV_GraphDB *g, uint64_t node_id,
                        size_t iterations, float damping);

/**
 * @brief Total degree (in + out) of a node.
 *
 * @param g Graph database; must be non-NULL.
 * @param node_id Node identifier.
 * @return Degree, or 0 if node not found.
 */
size_t gv_graph_degree(const GV_GraphDB *g, uint64_t node_id);

/**
 * @brief In-degree of a node.
 *
 * @param g Graph database; must be non-NULL.
 * @param node_id Node identifier.
 * @return In-degree, or 0 if node not found.
 */
size_t gv_graph_in_degree(const GV_GraphDB *g, uint64_t node_id);

/**
 * @brief Out-degree of a node.
 *
 * @param g Graph database; must be non-NULL.
 * @param node_id Node identifier.
 * @return Out-degree, or 0 if node not found.
 */
size_t gv_graph_out_degree(const GV_GraphDB *g, uint64_t node_id);

/**
 * @brief Identify connected components in the graph (treating edges as undirected).
 *
 * Assigns a component ID to each node. The component_ids array is indexed by
 * the order in which nodes are enumerated from the hash table; use in
 * conjunction with a full node scan.
 *
 * @param g Graph database; must be non-NULL.
 * @param component_ids Output array (one entry per node); must be pre-allocated
 *                      to at least gv_graph_node_count(g) entries.
 * @param max_count Capacity of component_ids.
 * @return Number of distinct connected components, or -1 on error.
 */
int gv_graph_connected_components(const GV_GraphDB *g,
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
float gv_graph_clustering_coefficient(const GV_GraphDB *g, uint64_t node_id);

/* Stats */

/**
 * @brief Return the number of nodes in the graph.
 *
 * @param g Graph database; must be non-NULL.
 * @return Node count.
 */
size_t gv_graph_node_count(const GV_GraphDB *g);

/**
 * @brief Return the number of edges in the graph.
 *
 * @param g Graph database; must be non-NULL.
 * @return Edge count.
 */
size_t gv_graph_edge_count(const GV_GraphDB *g);

/* Persistence */

/**
 * @brief Save the graph to a binary file.
 *
 * Format uses magic bytes "GVGR" followed by version, counts, and serialized
 * nodes/edges with their properties.
 *
 * @param g Graph database; must be non-NULL.
 * @param path File path to write; must be non-NULL.
 * @return 0 on success, -1 on error.
 */
int gv_graph_save(const GV_GraphDB *g, const char *path);

/**
 * @brief Load a graph from a binary file previously written by gv_graph_save.
 *
 * @param path File path to read; must be non-NULL.
 * @return Loaded graph database, or NULL on error.
 */
GV_GraphDB *gv_graph_load(const char *path);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_GRAPH_DB_H */
