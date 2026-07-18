/**
 * @file graph_algos.h
 * @brief Graph data-science algorithm library for the GigaVector property graph.
 *
 * A TigerGraph-GDS-comparable set of classic graph algorithms operating on a
 * GV_GraphDB (see graph_db.h). Grouped into: centrality, community detection,
 * path/traversal, topological similarity & link prediction, classification, and
 * node embeddings. All algorithms treat the graph as directed where meaningful
 * and undirected otherwise (documented per function), and honour edge weights
 * where the algorithm is weighted.
 *
 * Node results are returned in parallel arrays keyed by node_id; the caller owns
 * the result and frees it with the matching *_free helper. Every algorithm is
 * read-only w.r.t. the graph.
 */

#ifndef GIGAVECTOR_GV_GRAPH_ALGOS_H
#define GIGAVECTOR_GV_GRAPH_ALGOS_H

#include <stddef.h>
#include <stdint.h>

#include "features/graph_db.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Shared node-index context ───────────────────────────────────────────────
 * Enumerates the graph's nodes once and builds a dense id <-> index map so
 * algorithms can work over compact arrays [0, N). Build once, reuse, free. */
typedef struct GV_GAContext GV_GAContext;

/** Build an algorithm context (snapshots the node set + id/index map). NULL on error/empty. */
GV_GAContext *gv_ga_build(const GV_GraphDB *g);
/** Number of nodes N in the context. */
size_t   gv_ga_count(const GV_GAContext *ctx);
/** node_id at dense index (idx < N). */
uint64_t gv_ga_id(const GV_GAContext *ctx, size_t idx);
/** Dense index of node_id, or (size_t)-1 if not present. */
size_t   gv_ga_index(const GV_GAContext *ctx, uint64_t node_id);
/** Free the context. */
void     gv_ga_free(GV_GAContext *ctx);

/* ── Result containers ──────────────────────────────────────────────────────*/

/** Per-node real-valued scores (centrality, similarity, triangles, etc.). */
typedef struct {
    uint64_t *node_ids;   /**< N node IDs (owned). */
    double   *scores;     /**< N scores, scores[i] belongs to node_ids[i] (owned). */
    size_t    count;      /**< N. */
} GV_GraphNodeScores;
void graph_node_scores_free(GV_GraphNodeScores *s);

/** Per-node integer labels (community, color, core number, SCC). */
typedef struct {
    uint64_t *node_ids;   /**< N node IDs (owned). */
    int64_t  *labels;     /**< N labels (owned). */
    size_t    count;      /**< N. */
    size_t    num_labels; /**< Number of distinct labels (communities/colors/...). */
} GV_GraphNodeLabels;
void graph_node_labels_free(GV_GraphNodeLabels *l);

/** Dense node embeddings: count x dim row-major float matrix. */
typedef struct {
    uint64_t *node_ids;   /**< N node IDs (owned). */
    float    *vectors;    /**< N*dim floats, row i = embedding of node_ids[i] (owned). */
    size_t    count;      /**< N. */
    size_t    dim;        /**< Embedding dimension. */
} GV_GraphEmbeddings;
void graph_embeddings_free(GV_GraphEmbeddings *e);

/** A set of edges (e.g. a spanning forest). */
typedef struct {
    uint64_t *edge_ids;   /**< Edge IDs (owned). */
    size_t    count;
    double    total_weight;
} GV_GraphEdgeSet;
void graph_edge_set_free(GV_GraphEdgeSet *s);

/* ── Centrality ─────────────────────────────────────────────────────────────*/

/** Betweenness centrality (Brandes). weighted!=0 uses edge weights (Dijkstra),
 *  else unweighted BFS. directed!=0 respects edge direction. O(V*E) / O(V*E+V^2 log V). */
int graph_betweenness_centrality(const GV_GraphDB *g, int weighted, int directed,
                                 GV_GraphNodeScores *out);

/** Closeness centrality = (reachable-1) / sum(dist) per node (Wasserman-Faust
 *  normalized for disconnected graphs). weighted/directed as above. */
int graph_closeness_centrality(const GV_GraphDB *g, int weighted, int directed,
                               GV_GraphNodeScores *out);

/** Harmonic centrality = sum(1/dist) over reachable nodes. */
int graph_harmonic_centrality(const GV_GraphDB *g, int weighted, int directed,
                              GV_GraphNodeScores *out);

/** Eigenvector centrality via power iteration on the (out-)adjacency. */
int graph_eigenvector_centrality(const GV_GraphDB *g, size_t max_iters, double tol,
                                 GV_GraphNodeScores *out);

/** Degree centrality, normalized by (N-1). mode: 0=total, 1=in, 2=out. */
int graph_degree_centrality(const GV_GraphDB *g, int mode, GV_GraphNodeScores *out);

/** Full PageRank vector (power iteration). damping ~0.85. */
int graph_pagerank_all(const GV_GraphDB *g, size_t iters, double damping,
                       GV_GraphNodeScores *out);

/** Personalized PageRank biased toward the given source node IDs (teleport set). */
int graph_personalized_pagerank(const GV_GraphDB *g,
                                const uint64_t *sources, size_t num_sources,
                                size_t iters, double damping,
                                GV_GraphNodeScores *out);

/** ArticleRank — PageRank variant that dampens the influence of low-degree nodes. */
int graph_article_rank(const GV_GraphDB *g, size_t iters, double damping,
                       GV_GraphNodeScores *out);

/** HITS hubs & authorities via power iteration. Fills `hubs` and `authorities`
 *  (either may be NULL to skip). iters default 100 if 0, tol default 1e-8 if <=0. */
int graph_hits(const GV_GraphDB *g, size_t iters, double tol,
               GV_GraphNodeScores *hubs, GV_GraphNodeScores *authorities);

/* ── Matrix-native traversal (GraphBLAS-lite) ───────────────────────────────
 * Reachability and activation expressed as sparse linear algebra over the CSR
 * adjacency matrix (see graph_csr.h) — repeated boolean SpMV for variable-length
 * reachability, weighted SpMV for spreading activation. */

/** k-hop reachability from a seed set, via repeated boolean SpMV (matrix-based
 *  variable-length path expansion). out->scores[i] = 1.0 if node i is reachable
 *  from any of `sources` within `k` hops, else 0.0. directed!=0 follows edge
 *  direction (forward), else undirected. k==0 marks only the sources. */
int graph_khop_reachable(const GV_GraphDB *g, const uint64_t *sources, size_t num_sources,
                         size_t k, int directed, GV_GraphNodeScores *out);

/** Spreading activation via weighted SpMV: seeds start at activation 1.0 and
 *  propagate for `iters` rounds — each round adds decay·(A·a) to the accumulated
 *  activation (A = in-adjacency so activation flows along edges toward targets).
 *  directed!=0 follows edges; weighted!=0 uses edge weights. out->scores = total
 *  accumulated activation per node. This is the linear-algebra form of a
 *  spreading-activation / personalized-diffusion query. */
int graph_spread_activation(const GV_GraphDB *g, const uint64_t *seeds, size_t num_seeds,
                            size_t iters, double decay, int directed, int weighted,
                            GV_GraphNodeScores *out);

/* ── Community detection / clustering ───────────────────────────────────────*/

/** Label Propagation community detection (synchronous-ish, deterministic seed). */
int graph_label_propagation(const GV_GraphDB *g, size_t max_iters,
                            GV_GraphNodeLabels *out);

/** Louvain modularity-maximizing community detection (one level of aggregation
 *  by default; iterates local moves to convergence). Treated as undirected. */
int graph_louvain(const GV_GraphDB *g, size_t max_passes, GV_GraphNodeLabels *out);

/** Per-node triangle count (undirected). Sets *total to the graph triangle count
 *  (each triangle counted once) when non-NULL. */
int graph_triangle_count(const GV_GraphDB *g, GV_GraphNodeScores *out, uint64_t *total);

/** K-core decomposition: core number per node (undirected). */
int graph_kcore(const GV_GraphDB *g, GV_GraphNodeLabels *out);

/** Strongly connected components (Tarjan). labels[i] = component id. */
int graph_strongly_connected_components(const GV_GraphDB *g, GV_GraphNodeLabels *out);

/* ── Path / traversal ───────────────────────────────────────────────────────*/

/** Single-source shortest-path distances (Dijkstra if weighted, BFS otherwise).
 *  out->scores[i] = distance from `source` to node i, or INFINITY if unreachable. */
int graph_single_source_shortest_path(const GV_GraphDB *g, uint64_t source,
                                      int weighted, int directed,
                                      GV_GraphNodeScores *out);

/** A* shortest path. `heuristic(node_id, goal_id, user)` must return an admissible
 *  lower bound on the remaining distance (>=0); pass NULL to fall back to Dijkstra.
 *  Fills `path` (free with graph_free_path). Returns 0 on success, -1 if no path. */
typedef double (*GV_GraphHeuristic)(uint64_t node_id, uint64_t goal_id, void *user);
int graph_astar(const GV_GraphDB *g, uint64_t from, uint64_t to,
                GV_GraphHeuristic heuristic, void *user, GV_GraphPath *path);

/** Directed cycle detection. Sets *is_dag=1 if acyclic. If a cycle exists and
 *  cycle_out!=NULL, fills it with one cycle's node IDs. Returns 0 on success. */
int graph_detect_cycle(const GV_GraphDB *g, int *is_dag, GV_GraphPath *cycle_out);

/** Minimum spanning forest (Kruskal, undirected, by edge weight). */
int graph_minimum_spanning_forest(const GV_GraphDB *g, GV_GraphEdgeSet *out);

/** Maximum flow from source to sink (Edmonds-Karp; edge weight = capacity,
 *  directed). Returns the max-flow value (>=0), or -1.0 on error. */
double graph_max_flow(const GV_GraphDB *g, uint64_t source, uint64_t sink);

/* ── Topological similarity & link prediction (pairwise) ────────────────────*/

/** Neighbor-set measures between two nodes (undirected neighborhoods). */
double graph_common_neighbors(const GV_GraphDB *g, uint64_t a, uint64_t b);
double graph_total_neighbors(const GV_GraphDB *g, uint64_t a, uint64_t b);
double graph_preferential_attachment(const GV_GraphDB *g, uint64_t a, uint64_t b);
double graph_jaccard_similarity(const GV_GraphDB *g, uint64_t a, uint64_t b);
double graph_cosine_neighborhood(const GV_GraphDB *g, uint64_t a, uint64_t b);
double graph_adamic_adar(const GV_GraphDB *g, uint64_t a, uint64_t b);
double graph_resource_allocation(const GV_GraphDB *g, uint64_t a, uint64_t b);

/** Similarity measure selector for top-k. */
typedef enum {
    GV_GSIM_JACCARD = 0,
    GV_GSIM_COSINE,
    GV_GSIM_COMMON_NEIGHBORS,
    GV_GSIM_ADAMIC_ADAR,
    GV_GSIM_RESOURCE_ALLOCATION,
    GV_GSIM_PREFERENTIAL_ATTACHMENT
} GV_GraphSimMeasure;

/** Top-k most similar nodes to `node` by the chosen neighborhood measure,
 *  excluding `node` itself. Fills out with up to k results sorted descending. */
int graph_topk_similar(const GV_GraphDB *g, uint64_t node, GV_GraphSimMeasure measure,
                       size_t k, GV_GraphNodeScores *out);

/* ── Classification ─────────────────────────────────────────────────────────*/

/** Greedy graph coloring (undirected). labels[i] = color index; num_labels = colors used. */
int graph_coloring(const GV_GraphDB *g, GV_GraphNodeLabels *out);

/** Greedy maximal independent set (undirected). Fills out->scores with 1.0 for
 *  nodes in the set, 0.0 otherwise (scores container reused as a 0/1 mask). */
int graph_maximal_independent_set(const GV_GraphDB *g, GV_GraphNodeScores *out);

/* ── Node embeddings ────────────────────────────────────────────────────────*/

/** FastRP structural embeddings. `dim` = embedding dimension; `iters` = number of
 *  neighborhood-propagation hops; `weights[iters]` scale each hop's contribution
 *  (pass NULL for a sensible default); `seed` for the sparse random projection. */
int graph_fastrp(const GV_GraphDB *g, size_t dim, size_t iters,
                 const double *weights, uint64_t seed, GV_GraphEmbeddings *out);

/** node2vec-style biased random walks. Generates `num_walks` walks of length
 *  `walk_len` from every node using return param p and in-out param q. On success
 *  *out_walks is an allocated (N*num_walks) x walk_len array of node IDs (0-padded
 *  for early-terminated walks) and *out_num_walks / *out_walk_len are set. Caller
 *  frees *out_walks with gv_free (or free()). */
int graph_node2vec_walks(const GV_GraphDB *g, size_t num_walks, size_t walk_len,
                         double p, double q, uint64_t seed,
                         uint64_t **out_walks, size_t *out_num_walks, size_t *out_walk_len);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_GRAPH_ALGOS_H */
