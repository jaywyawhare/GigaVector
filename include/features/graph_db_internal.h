/**
  * @file graph_db_internal.h
  * @brief Internal graph-DB storage structs, WAL macros and cross-file prototypes.
  *
  * Shared between graph_db.c (core), graph_db_traverse.c and graph_db_persist.c.
  * Not a public API; do not include outside the graph-DB engine.
  */
#ifndef GV_FEATURES_GRAPH_DB_INTERNAL_H
#define GV_FEATURES_GRAPH_DB_INTERNAL_H

#include "features/graph_db.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <pthread.h>

#define GV_GRAPH_MAGIC       "GVGR"
#define GV_GRAPH_MAGIC_LEN   4
#define GV_GRAPH_VERSION     1

typedef struct NodeEntry {
    GV_GraphNode node;
    struct NodeEntry *next;
    struct NodeEntry *next_free;  /* slab freelist link */
    uint8_t in_use;               /* slab bookkeeping for destroy */
} NodeEntry;

typedef struct EdgeEntry {
    GV_GraphEdge edge;
    struct EdgeEntry *next;
    struct EdgeEntry *next_free;  /* slab freelist link */
    uint8_t in_use;               /* slab bookkeeping for destroy */
} EdgeEntry;

/* label -> node id list, so graph_find_nodes_by_label is O(matching) not O(nodes) */
/* Opaque handle for multi-call repeatable-read transactions (graph_db.h). */
struct GV_GraphReadTxn {
    GV_GraphDB *g;
    uint64_t    version_at_begin;
};

typedef struct LabelEntry {
    char *label;
    uint64_t *ids;
    size_t count;
    size_t cap;
    struct LabelEntry *next;
} LabelEntry;

struct GV_GraphDB {
    NodeEntry **node_buckets;       /**< Node hash table bucket array. */
    size_t node_bucket_count;       /**< Number of node buckets. */
    size_t node_count;              /**< Total number of nodes. */
    uint64_t next_node_id;          /**< Next auto-increment node ID. */

    EdgeEntry **edge_buckets;       /**< Edge hash table bucket array. */
    size_t edge_bucket_count;       /**< Number of edge buckets. */
    size_t edge_count;              /**< Total number of edges. */
    uint64_t next_edge_id;          /**< Next auto-increment edge ID. */

    LabelEntry **label_buckets;     /**< Label index bucket array. */
    size_t label_bucket_count;      /**< Number of label-index buckets. */

    /* Slab record pools (see node_pool_alloc / edge_pool_alloc). */
    struct NodeSlab *node_slabs;
    NodeEntry *node_free;
    struct EdgeSlab *edge_slabs;
    EdgeEntry *edge_free;

    int enforce_referential_integrity;

    /* Write-ahead log: mutations are appended + fsync'd before the mutating
     * call returns; graph_load replays the log over the last snapshot. */
    FILE *wal_file;
    char *wal_path;
    int   wal_replaying;            /* suppress logging during replay */

    uint64_t mutation_count;        /* bumped on every successful mutation */

    /* Edge-change journal for incremental CSR maintenance: records endpoint
     * pairs of edge add/remove since `journal_base_version`. Structural
     * mutations (node add/remove, txn commit, WAL replay) invalidate it. */
    GV_GraphEdgeDelta *edge_journal;
    size_t   edge_journal_len;
    size_t   edge_journal_cap;
    uint64_t journal_base_version;  /* journal covers versions > this */

    pthread_rwlock_t rwlock;        /**< Reader-writer lock for thread safety. */
};

/* Linear-probe hash set for traversal visited tracking */
typedef struct {
    uint64_t *slots;
    int *occupied;
    size_t capacity;
    size_t count;
} VisitedSet;

/* ── WAL record format (little-endian) ──────────────────────────────────
 * [u8 op][payload]
 *   1 ADD_NODE      u64 id, str label
 *   2 REMOVE_NODE   u64 id
 *   3 ADD_EDGE      u64 id, u64 src, u64 dst, str label, f32 weight
 *   4 REMOVE_EDGE   u64 id
 *   5 SET_NODE_PROP u64 id, str key, str value
 *   6 SET_EDGE_PROP u64 id, str key, str value
 * str = u32 length + bytes */
#define GV_GWAL_OP_ADD_NODE      1
#define GV_GWAL_OP_REMOVE_NODE   2
#define GV_GWAL_OP_ADD_EDGE      3
#define GV_GWAL_OP_REMOVE_EDGE   4
#define GV_GWAL_OP_SET_NODE_PROP 5
#define GV_GWAL_OP_SET_EDGE_PROP 6
#define GV_GWAL_BUF_CAP 4096
#define GV_EDGE_JOURNAL_MAX 4096

/* --- Cross-file prototypes (split from graph_db.c) --- */
/* Shared hash-table / pool / WAL / visited-set helpers (graph_db.c) used by the
 * traversal (graph_db_traverse.c) and persistence (graph_db_persist.c) splits. */
NodeEntry *find_node_entry(const GV_GraphDB *g, uint64_t node_id);
EdgeEntry *find_edge_entry(const GV_GraphDB *g, uint64_t edge_id);
size_t idmap_lookup(const uint64_t *keys, const int *occ,
                    const size_t *idx_arr, size_t cap, uint64_t key);
uint64_t *collect_all_node_ids(const GV_GraphDB *g, size_t *out_count);
int  visited_init(VisitedSet *vs, size_t capacity);
void visited_free(VisitedSet *vs);
int  visited_contains(const VisitedSet *vs, uint64_t id);
int  visited_insert(VisitedSet *vs, uint64_t id);
void journal_break_locked(GV_GraphDB *g);
void graph_wal_close(GV_GraphDB *g);
int set_prop(GV_GraphProp **head, size_t *count, const char *key, const char *value);
int label_index_add(GV_GraphDB *g, const char *label, uint64_t node_id);
int adj_add(GV_GraphEdgeRef **arr, size_t *count, size_t *cap,
            uint64_t edge_id, uint64_t neighbor_id);
NodeEntry *node_pool_alloc(GV_GraphDB *g);
void node_pool_free(GV_GraphDB *g, NodeEntry *e);
EdgeEntry *edge_pool_alloc(GV_GraphDB *g);
void edge_pool_free(GV_GraphDB *g, EdgeEntry *e);
void gwal_apply_record(GV_GraphDB *g, uint8_t op, const uint8_t *p, const uint8_t *end);
size_t gwal_parse_record(GV_GraphDB *g, const uint8_t *buf, size_t len);

#endif /* GV_FEATURES_GRAPH_DB_INTERNAL_H */
