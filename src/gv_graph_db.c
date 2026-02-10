/**
 * @file gv_graph_db.c
 * @brief Full graph database layer implementation for GigaVector.
 *
 * Implements a property-graph model backed by hash-table storage with chaining,
 * dynamic adjacency lists, thread-safe access via pthread_rwlock_t, traversal
 * algorithms (BFS, DFS, Dijkstra, all-paths), graph analytics (PageRank,
 * clustering coefficient, connected components), and binary persistence.
 */

#define _POSIX_C_SOURCE 200112L

#include "gigavector/gv_graph_db.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <pthread.h>

/* ============================================================================
 * Constants
 * ============================================================================ */

#define GV_GRAPH_MAGIC       "GVGR"
#define GV_GRAPH_MAGIC_LEN   4
#define GV_GRAPH_VERSION     1

#define DEFAULT_NODE_BUCKETS 4096
#define DEFAULT_EDGE_BUCKETS 8192
#define INITIAL_ADJ_CAP      4

/* ============================================================================
 * Internal Hash Table Entry Types
 * ============================================================================ */

/**
 * @brief Hash table entry wrapping a GV_GraphNode (chaining via next pointer).
 */
typedef struct NodeEntry {
    GV_GraphNode node;
    struct NodeEntry *next;
} NodeEntry;

/**
 * @brief Hash table entry wrapping a GV_GraphEdge (chaining via next pointer).
 */
typedef struct EdgeEntry {
    GV_GraphEdge edge;
    struct EdgeEntry *next;
} EdgeEntry;

/* ============================================================================
 * Opaque GV_GraphDB Definition
 * ============================================================================ */

struct GV_GraphDB {
    NodeEntry **node_buckets;       /**< Node hash table bucket array. */
    size_t node_bucket_count;       /**< Number of node buckets. */
    size_t node_count;              /**< Total number of nodes. */
    uint64_t next_node_id;          /**< Next auto-increment node ID. */

    EdgeEntry **edge_buckets;       /**< Edge hash table bucket array. */
    size_t edge_bucket_count;       /**< Number of edge buckets. */
    size_t edge_count;              /**< Total number of edges. */
    uint64_t next_edge_id;          /**< Next auto-increment edge ID. */

    int enforce_referential_integrity;

    pthread_rwlock_t rwlock;        /**< Reader-writer lock for thread safety. */
};

/* ============================================================================
 * Hash Function (djb2 on uint64_t)
 * ============================================================================ */

static size_t hash_u64(uint64_t id, size_t bucket_count)
{
    /* djb2 applied to the bytes of the id */
    size_t hash = 5381;
    const unsigned char *p = (const unsigned char *)&id;
    for (size_t i = 0; i < sizeof(id); i++) {
        hash = ((hash << 5) + hash) + p[i];
    }
    return hash % bucket_count;
}

/* ============================================================================
 * String Helpers
 * ============================================================================ */

static char *str_dup(const char *s)
{
    if (!s) return NULL;
    size_t len = strlen(s);
    char *copy = (char *)malloc(len + 1);
    if (!copy) return NULL;
    memcpy(copy, s, len + 1);
    return copy;
}

/* ============================================================================
 * Property Helpers
 * ============================================================================ */

static void free_prop_list(GV_GraphProp *head)
{
    while (head) {
        GV_GraphProp *next = head->next;
        free(head->key);
        free(head->value);
        free(head);
        head = next;
    }
}

/**
 * @brief Find a property by key in a linked list.
 */
static GV_GraphProp *find_prop(GV_GraphProp *head, const char *key)
{
    while (head) {
        if (strcmp(head->key, key) == 0) return head;
        head = head->next;
    }
    return NULL;
}

/**
 * @brief Set or overwrite a property in a linked list.
 * @return 0 on success, -1 on allocation failure.
 */
static int set_prop(GV_GraphProp **head, size_t *count,
                    const char *key, const char *value)
{
    GV_GraphProp *existing = find_prop(*head, key);
    if (existing) {
        char *new_val = str_dup(value);
        if (!new_val) return -1;
        free(existing->value);
        existing->value = new_val;
        return 0;
    }
    /* Create new property node */
    GV_GraphProp *prop = (GV_GraphProp *)calloc(1, sizeof(GV_GraphProp));
    if (!prop) return -1;
    prop->key = str_dup(key);
    prop->value = str_dup(value);
    if (!prop->key || !prop->value) {
        free(prop->key);
        free(prop->value);
        free(prop);
        return -1;
    }
    prop->next = *head;
    *head = prop;
    (*count)++;
    return 0;
}

/* ============================================================================
 * Node Lookup (internal, no locking)
 * ============================================================================ */

static NodeEntry *find_node_entry(const GV_GraphDB *g, uint64_t node_id)
{
    size_t idx = hash_u64(node_id, g->node_bucket_count);
    NodeEntry *e = g->node_buckets[idx];
    while (e) {
        if (e->node.node_id == node_id) return e;
        e = e->next;
    }
    return NULL;
}

/* ============================================================================
 * Edge Lookup (internal, no locking)
 * ============================================================================ */

static EdgeEntry *find_edge_entry(const GV_GraphDB *g, uint64_t edge_id)
{
    size_t idx = hash_u64(edge_id, g->edge_bucket_count);
    EdgeEntry *e = g->edge_buckets[idx];
    while (e) {
        if (e->edge.edge_id == edge_id) return e;
        e = e->next;
    }
    return NULL;
}

/* ============================================================================
 * Adjacency List Helpers
 * ============================================================================ */

static int adj_add(GV_GraphEdgeRef **arr, size_t *count, size_t *cap,
                   uint64_t edge_id, uint64_t neighbor_id)
{
    if (*count >= *cap) {
        size_t new_cap = (*cap == 0) ? INITIAL_ADJ_CAP : (*cap * 2);
        GV_GraphEdgeRef *tmp = (GV_GraphEdgeRef *)realloc(
            *arr, new_cap * sizeof(GV_GraphEdgeRef));
        if (!tmp) return -1;
        *arr = tmp;
        *cap = new_cap;
    }
    (*arr)[*count].edge_id = edge_id;
    (*arr)[*count].neighbor_id = neighbor_id;
    (*count)++;
    return 0;
}

static void adj_remove(GV_GraphEdgeRef *arr, size_t *count, uint64_t edge_id)
{
    for (size_t i = 0; i < *count; i++) {
        if (arr[i].edge_id == edge_id) {
            arr[i] = arr[*count - 1];
            (*count)--;
            return;
        }
    }
}

/* ============================================================================
 * Node Cleanup Helper
 * ============================================================================ */

static void free_node_internals(GV_GraphNode *node)
{
    free(node->label);
    free_prop_list(node->properties);
    free(node->out_edges);
    free(node->in_edges);
}

/* ============================================================================
 * Edge Cleanup Helper
 * ============================================================================ */

static void free_edge_internals(GV_GraphEdge *edge)
{
    free(edge->label);
    free_prop_list(edge->properties);
}

/* ============================================================================
 * Internal: Remove a single edge (no lock, updates adjacency)
 * ============================================================================ */

static int remove_edge_internal(GV_GraphDB *g, uint64_t edge_id)
{
    size_t idx = hash_u64(edge_id, g->edge_bucket_count);
    EdgeEntry *prev = NULL;
    EdgeEntry *e = g->edge_buckets[idx];
    while (e) {
        if (e->edge.edge_id == edge_id) break;
        prev = e;
        e = e->next;
    }
    if (!e) return -1;

    /* Remove from source out_edges */
    NodeEntry *src = find_node_entry(g, e->edge.source_id);
    if (src) {
        adj_remove(src->node.out_edges, &src->node.out_count, edge_id);
    }

    /* Remove from target in_edges */
    NodeEntry *tgt = find_node_entry(g, e->edge.target_id);
    if (tgt) {
        adj_remove(tgt->node.in_edges, &tgt->node.in_count, edge_id);
    }

    /* Unlink from bucket chain */
    if (prev) {
        prev->next = e->next;
    } else {
        g->edge_buckets[idx] = e->next;
    }

    free_edge_internals(&e->edge);
    free(e);
    g->edge_count--;
    return 0;
}

/* ============================================================================
 * Visited-Set Helper (simple linear probe hash set for traversal)
 * ============================================================================ */

typedef struct {
    uint64_t *slots;
    int *occupied;
    size_t capacity;
    size_t count;
} VisitedSet;

static int visited_init(VisitedSet *vs, size_t capacity)
{
    if (capacity < 16) capacity = 16;
    vs->slots = (uint64_t *)malloc(capacity * sizeof(uint64_t));
    vs->occupied = (int *)calloc(capacity, sizeof(int));
    if (!vs->slots || !vs->occupied) {
        free(vs->slots);
        free(vs->occupied);
        return -1;
    }
    vs->capacity = capacity;
    vs->count = 0;
    return 0;
}

static void visited_free(VisitedSet *vs)
{
    free(vs->slots);
    free(vs->occupied);
    vs->slots = NULL;
    vs->occupied = NULL;
}

static int visited_contains(const VisitedSet *vs, uint64_t id)
{
    size_t idx = (size_t)(id * 2654435761ULL) % vs->capacity;
    for (size_t i = 0; i < vs->capacity; i++) {
        size_t pos = (idx + i) % vs->capacity;
        if (!vs->occupied[pos]) return 0;
        if (vs->slots[pos] == id) return 1;
    }
    return 0;
}

static int visited_insert(VisitedSet *vs, uint64_t id)
{
    /* Rehash if load factor > 0.7 */
    if (vs->count * 10 > vs->capacity * 7) {
        size_t new_cap = vs->capacity * 2;
        uint64_t *new_slots = (uint64_t *)malloc(new_cap * sizeof(uint64_t));
        int *new_occ = (int *)calloc(new_cap, sizeof(int));
        if (!new_slots || !new_occ) {
            free(new_slots);
            free(new_occ);
            return -1;
        }
        for (size_t i = 0; i < vs->capacity; i++) {
            if (vs->occupied[i]) {
                uint64_t val = vs->slots[i];
                size_t h = (size_t)(val * 2654435761ULL) % new_cap;
                for (size_t j = 0; j < new_cap; j++) {
                    size_t p = (h + j) % new_cap;
                    if (!new_occ[p]) {
                        new_slots[p] = val;
                        new_occ[p] = 1;
                        break;
                    }
                }
            }
        }
        free(vs->slots);
        free(vs->occupied);
        vs->slots = new_slots;
        vs->occupied = new_occ;
        vs->capacity = new_cap;
    }

    size_t idx = (size_t)(id * 2654435761ULL) % vs->capacity;
    for (size_t i = 0; i < vs->capacity; i++) {
        size_t pos = (idx + i) % vs->capacity;
        if (!vs->occupied[pos]) {
            vs->slots[pos] = id;
            vs->occupied[pos] = 1;
            vs->count++;
            return 1; /* inserted */
        }
        if (vs->slots[pos] == id) return 0; /* already present */
    }
    return -1; /* should not happen */
}

/* ============================================================================
 * Distance Map Helper (open-addressing hash map: uint64_t -> float)
 * ============================================================================ */

typedef struct {
    uint64_t *keys;
    float *vals;
    uint64_t *prev_node;    /* predecessor node for path reconstruction */
    uint64_t *prev_edge;    /* predecessor edge for path reconstruction */
    int *occupied;
    size_t capacity;
    size_t count;
} DistMap;

static int distmap_init(DistMap *dm, size_t capacity)
{
    if (capacity < 16) capacity = 16;
    dm->keys = (uint64_t *)malloc(capacity * sizeof(uint64_t));
    dm->vals = (float *)malloc(capacity * sizeof(float));
    dm->prev_node = (uint64_t *)malloc(capacity * sizeof(uint64_t));
    dm->prev_edge = (uint64_t *)malloc(capacity * sizeof(uint64_t));
    dm->occupied = (int *)calloc(capacity, sizeof(int));
    if (!dm->keys || !dm->vals || !dm->prev_node || !dm->prev_edge || !dm->occupied) {
        free(dm->keys);
        free(dm->vals);
        free(dm->prev_node);
        free(dm->prev_edge);
        free(dm->occupied);
        return -1;
    }
    dm->capacity = capacity;
    dm->count = 0;
    return 0;
}

static void distmap_free(DistMap *dm)
{
    free(dm->keys);
    free(dm->vals);
    free(dm->prev_node);
    free(dm->prev_edge);
    free(dm->occupied);
}

static size_t distmap_probe(const DistMap *dm, uint64_t key)
{
    return (size_t)(key * 2654435761ULL) % dm->capacity;
}

static int distmap_rehash(DistMap *dm)
{
    size_t new_cap = dm->capacity * 2;
    uint64_t *nk = (uint64_t *)malloc(new_cap * sizeof(uint64_t));
    float *nv = (float *)malloc(new_cap * sizeof(float));
    uint64_t *npn = (uint64_t *)malloc(new_cap * sizeof(uint64_t));
    uint64_t *npe = (uint64_t *)malloc(new_cap * sizeof(uint64_t));
    int *no = (int *)calloc(new_cap, sizeof(int));
    if (!nk || !nv || !npn || !npe || !no) {
        free(nk); free(nv); free(npn); free(npe); free(no);
        return -1;
    }
    for (size_t i = 0; i < dm->capacity; i++) {
        if (dm->occupied[i]) {
            size_t h = (size_t)(dm->keys[i] * 2654435761ULL) % new_cap;
            for (size_t j = 0; j < new_cap; j++) {
                size_t p = (h + j) % new_cap;
                if (!no[p]) {
                    nk[p] = dm->keys[i];
                    nv[p] = dm->vals[i];
                    npn[p] = dm->prev_node[i];
                    npe[p] = dm->prev_edge[i];
                    no[p] = 1;
                    break;
                }
            }
        }
    }
    free(dm->keys); free(dm->vals); free(dm->prev_node);
    free(dm->prev_edge); free(dm->occupied);
    dm->keys = nk; dm->vals = nv; dm->prev_node = npn;
    dm->prev_edge = npe; dm->occupied = no;
    dm->capacity = new_cap;
    return 0;
}

/**
 * @brief Set distance for a key. Returns the slot index.
 */
static int distmap_set(DistMap *dm, uint64_t key, float dist,
                       uint64_t pnode, uint64_t pedge)
{
    if (dm->count * 10 > dm->capacity * 7) {
        if (distmap_rehash(dm) != 0) return -1;
    }
    size_t h = distmap_probe(dm, key);
    for (size_t i = 0; i < dm->capacity; i++) {
        size_t p = (h + i) % dm->capacity;
        if (!dm->occupied[p]) {
            dm->keys[p] = key;
            dm->vals[p] = dist;
            dm->prev_node[p] = pnode;
            dm->prev_edge[p] = pedge;
            dm->occupied[p] = 1;
            dm->count++;
            return 0;
        }
        if (dm->keys[p] == key) {
            dm->vals[p] = dist;
            dm->prev_node[p] = pnode;
            dm->prev_edge[p] = pedge;
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Get distance for a key. Returns FLT_MAX if not found.
 */
static float distmap_get(const DistMap *dm, uint64_t key)
{
    size_t h = distmap_probe(dm, key);
    for (size_t i = 0; i < dm->capacity; i++) {
        size_t p = (h + i) % dm->capacity;
        if (!dm->occupied[p]) return FLT_MAX;
        if (dm->keys[p] == key) return dm->vals[p];
    }
    return FLT_MAX;
}

/**
 * @brief Get predecessor node for a key. Returns 0 if not found.
 */
static uint64_t distmap_get_prev_node(const DistMap *dm, uint64_t key)
{
    size_t h = distmap_probe(dm, key);
    for (size_t i = 0; i < dm->capacity; i++) {
        size_t p = (h + i) % dm->capacity;
        if (!dm->occupied[p]) return 0;
        if (dm->keys[p] == key) return dm->prev_node[p];
    }
    return 0;
}

/**
 * @brief Get predecessor edge for a key. Returns 0 if not found.
 */
static uint64_t distmap_get_prev_edge(const DistMap *dm, uint64_t key)
{
    size_t h = distmap_probe(dm, key);
    for (size_t i = 0; i < dm->capacity; i++) {
        size_t p = (h + i) % dm->capacity;
        if (!dm->occupied[p]) return 0;
        if (dm->keys[p] == key) return dm->prev_edge[p];
    }
    return 0;
}

/* ============================================================================
 * Min-Heap for Dijkstra (array-based)
 * ============================================================================ */

typedef struct {
    uint64_t node_id;
    float dist;
} HeapEntry;

typedef struct {
    HeapEntry *data;
    size_t size;
    size_t capacity;
} MinHeap;

static int heap_init(MinHeap *h, size_t capacity)
{
    if (capacity < 16) capacity = 16;
    h->data = (HeapEntry *)malloc(capacity * sizeof(HeapEntry));
    if (!h->data) return -1;
    h->size = 0;
    h->capacity = capacity;
    return 0;
}

static void heap_free(MinHeap *h)
{
    free(h->data);
    h->data = NULL;
}

static void heap_swap(HeapEntry *a, HeapEntry *b)
{
    HeapEntry tmp = *a;
    *a = *b;
    *b = tmp;
}

static void heap_sift_up(MinHeap *h, size_t idx)
{
    while (idx > 0) {
        size_t parent = (idx - 1) / 2;
        if (h->data[parent].dist > h->data[idx].dist) {
            heap_swap(&h->data[parent], &h->data[idx]);
            idx = parent;
        } else {
            break;
        }
    }
}

static void heap_sift_down(MinHeap *h, size_t idx)
{
    while (1) {
        size_t smallest = idx;
        size_t left = 2 * idx + 1;
        size_t right = 2 * idx + 2;
        if (left < h->size && h->data[left].dist < h->data[smallest].dist)
            smallest = left;
        if (right < h->size && h->data[right].dist < h->data[smallest].dist)
            smallest = right;
        if (smallest != idx) {
            heap_swap(&h->data[smallest], &h->data[idx]);
            idx = smallest;
        } else {
            break;
        }
    }
}

static int heap_push(MinHeap *h, uint64_t node_id, float dist)
{
    if (h->size >= h->capacity) {
        size_t new_cap = h->capacity * 2;
        HeapEntry *tmp = (HeapEntry *)realloc(h->data,
                                              new_cap * sizeof(HeapEntry));
        if (!tmp) return -1;
        h->data = tmp;
        h->capacity = new_cap;
    }
    h->data[h->size].node_id = node_id;
    h->data[h->size].dist = dist;
    heap_sift_up(h, h->size);
    h->size++;
    return 0;
}

static int heap_pop(MinHeap *h, HeapEntry *out)
{
    if (h->size == 0) return -1;
    *out = h->data[0];
    h->size--;
    if (h->size > 0) {
        h->data[0] = h->data[h->size];
        heap_sift_down(h, 0);
    }
    return 0;
}

/* ============================================================================
 * ID-to-Index Map Lookup Helper
 * ============================================================================ */

/**
 * @brief Look up a node ID in an open-addressing hash map and return its index.
 * @return The index associated with the key, or (size_t)-1 if not found.
 */
static size_t idmap_lookup(const uint64_t *keys, const int *occ,
                           const size_t *idx_arr, size_t cap, uint64_t key)
{
    size_t h = (size_t)(key * 2654435761ULL) % cap;
    for (size_t j = 0; j < cap; j++) {
        size_t p = (h + j) % cap;
        if (!occ[p]) return (size_t)-1;
        if (keys[p] == key) return idx_arr[p];
    }
    return (size_t)-1;
}

/* ============================================================================
 * Collect All Node IDs Helper
 * ============================================================================ */

/**
 * @brief Collect all node IDs from the hash table into an array.
 * @return Allocated array of node IDs (caller must free), or NULL.
 *         Sets *out_count to the number of IDs.
 */
static uint64_t *collect_all_node_ids(const GV_GraphDB *g, size_t *out_count)
{
    if (g->node_count == 0) {
        *out_count = 0;
        return NULL;
    }
    uint64_t *ids = (uint64_t *)malloc(g->node_count * sizeof(uint64_t));
    if (!ids) {
        *out_count = 0;
        return NULL;
    }
    size_t idx = 0;
    for (size_t b = 0; b < g->node_bucket_count; b++) {
        NodeEntry *e = g->node_buckets[b];
        while (e) {
            if (idx < g->node_count) {
                ids[idx++] = e->node.node_id;
            }
            e = e->next;
        }
    }
    *out_count = idx;
    return ids;
}

/* ============================================================================
 * File I/O Helpers
 * ============================================================================ */

static int write_u32(FILE *f, uint32_t v)
{
    return fwrite(&v, sizeof(v), 1, f) == 1 ? 0 : -1;
}

static int write_u64(FILE *f, uint64_t v)
{
    return fwrite(&v, sizeof(v), 1, f) == 1 ? 0 : -1;
}

static int write_float(FILE *f, float v)
{
    return fwrite(&v, sizeof(v), 1, f) == 1 ? 0 : -1;
}

static int write_str(FILE *f, const char *s)
{
    uint32_t len = (uint32_t)strlen(s);
    if (write_u32(f, len) != 0) return -1;
    if (len > 0 && fwrite(s, 1, len, f) != len) return -1;
    return 0;
}

static int read_u32(FILE *f, uint32_t *v)
{
    return fread(v, sizeof(*v), 1, f) == 1 ? 0 : -1;
}

static int read_u64(FILE *f, uint64_t *v)
{
    return fread(v, sizeof(*v), 1, f) == 1 ? 0 : -1;
}

static int read_float(FILE *f, float *v)
{
    return fread(v, sizeof(*v), 1, f) == 1 ? 0 : -1;
}

static char *read_str(FILE *f)
{
    uint32_t len;
    if (read_u32(f, &len) != 0) return NULL;
    char *s = (char *)malloc((size_t)len + 1);
    if (!s) return NULL;
    if (len > 0 && fread(s, 1, len, f) != len) {
        free(s);
        return NULL;
    }
    s[len] = '\0';
    return s;
}

/* ============================================================================
 * Lifecycle Implementation
 * ============================================================================ */

void gv_graph_config_init(GV_GraphDBConfig *config)
{
    if (!config) return;
    config->node_bucket_count = DEFAULT_NODE_BUCKETS;
    config->edge_bucket_count = DEFAULT_EDGE_BUCKETS;
    config->enforce_referential_integrity = 1;
}

GV_GraphDB *gv_graph_create(const GV_GraphDBConfig *config)
{
    GV_GraphDBConfig cfg;
    if (config) {
        cfg = *config;
    } else {
        gv_graph_config_init(&cfg);
    }
    if (cfg.node_bucket_count == 0) cfg.node_bucket_count = DEFAULT_NODE_BUCKETS;
    if (cfg.edge_bucket_count == 0) cfg.edge_bucket_count = DEFAULT_EDGE_BUCKETS;

    GV_GraphDB *g = (GV_GraphDB *)calloc(1, sizeof(GV_GraphDB));
    if (!g) return NULL;

    g->node_bucket_count = cfg.node_bucket_count;
    g->edge_bucket_count = cfg.edge_bucket_count;
    g->enforce_referential_integrity = cfg.enforce_referential_integrity;
    g->next_node_id = 1;
    g->next_edge_id = 1;

    g->node_buckets = (NodeEntry **)calloc(g->node_bucket_count,
                                           sizeof(NodeEntry *));
    g->edge_buckets = (EdgeEntry **)calloc(g->edge_bucket_count,
                                           sizeof(EdgeEntry *));
    if (!g->node_buckets || !g->edge_buckets) {
        free(g->node_buckets);
        free(g->edge_buckets);
        free(g);
        return NULL;
    }

    if (pthread_rwlock_init(&g->rwlock, NULL) != 0) {
        free(g->node_buckets);
        free(g->edge_buckets);
        free(g);
        return NULL;
    }

    return g;
}

void gv_graph_destroy(GV_GraphDB *g)
{
    if (!g) return;

    /* Free all nodes */
    for (size_t b = 0; b < g->node_bucket_count; b++) {
        NodeEntry *e = g->node_buckets[b];
        while (e) {
            NodeEntry *next = e->next;
            free_node_internals(&e->node);
            free(e);
            e = next;
        }
    }
    free(g->node_buckets);

    /* Free all edges */
    for (size_t b = 0; b < g->edge_bucket_count; b++) {
        EdgeEntry *e = g->edge_buckets[b];
        while (e) {
            EdgeEntry *next = e->next;
            free_edge_internals(&e->edge);
            free(e);
            e = next;
        }
    }
    free(g->edge_buckets);

    pthread_rwlock_destroy(&g->rwlock);
    free(g);
}

/* ============================================================================
 * Node Operations
 * ============================================================================ */

uint64_t gv_graph_add_node(GV_GraphDB *g, const char *label)
{
    if (!g || !label) return 0;

    char *lbl = str_dup(label);
    if (!lbl) return 0;

    NodeEntry *entry = (NodeEntry *)calloc(1, sizeof(NodeEntry));
    if (!entry) {
        free(lbl);
        return 0;
    }

    pthread_rwlock_wrlock(&g->rwlock);

    uint64_t id = g->next_node_id++;
    entry->node.node_id = id;
    entry->node.label = lbl;
    entry->node.properties = NULL;
    entry->node.prop_count = 0;
    entry->node.out_edges = NULL;
    entry->node.out_count = 0;
    entry->node.out_cap = 0;
    entry->node.in_edges = NULL;
    entry->node.in_count = 0;
    entry->node.in_cap = 0;

    /* Insert into hash table */
    size_t idx = hash_u64(id, g->node_bucket_count);
    entry->next = g->node_buckets[idx];
    g->node_buckets[idx] = entry;
    g->node_count++;

    pthread_rwlock_unlock(&g->rwlock);
    return id;
}

int gv_graph_remove_node(GV_GraphDB *g, uint64_t node_id)
{
    if (!g) return -1;

    pthread_rwlock_wrlock(&g->rwlock);

    NodeEntry *ne = find_node_entry(g, node_id);
    if (!ne) {
        pthread_rwlock_unlock(&g->rwlock);
        return -1;
    }

    /* Cascade-delete outgoing edges */
    while (ne->node.out_count > 0) {
        uint64_t eid = ne->node.out_edges[0].edge_id;
        remove_edge_internal(g, eid);
    }

    /* Cascade-delete incoming edges */
    while (ne->node.in_count > 0) {
        uint64_t eid = ne->node.in_edges[0].edge_id;
        remove_edge_internal(g, eid);
    }

    /* Remove node from bucket chain */
    size_t idx = hash_u64(node_id, g->node_bucket_count);
    NodeEntry *prev = NULL;
    NodeEntry *cur = g->node_buckets[idx];
    while (cur) {
        if (cur->node.node_id == node_id) break;
        prev = cur;
        cur = cur->next;
    }
    if (cur) {
        if (prev) {
            prev->next = cur->next;
        } else {
            g->node_buckets[idx] = cur->next;
        }
        free_node_internals(&cur->node);
        free(cur);
        g->node_count--;
    }

    pthread_rwlock_unlock(&g->rwlock);
    return 0;
}

const GV_GraphNode *gv_graph_get_node(const GV_GraphDB *g, uint64_t node_id)
{
    if (!g) return NULL;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&g->rwlock);
    NodeEntry *e = find_node_entry(g, node_id);
    const GV_GraphNode *result = e ? &e->node : NULL;
    pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
    return result;
}

int gv_graph_set_node_prop(GV_GraphDB *g, uint64_t node_id,
                           const char *key, const char *value)
{
    if (!g || !key || !value) return -1;

    pthread_rwlock_wrlock(&g->rwlock);
    NodeEntry *e = find_node_entry(g, node_id);
    if (!e) {
        pthread_rwlock_unlock(&g->rwlock);
        return -1;
    }
    int rc = set_prop(&e->node.properties, &e->node.prop_count, key, value);
    pthread_rwlock_unlock(&g->rwlock);
    return rc;
}

const char *gv_graph_get_node_prop(const GV_GraphDB *g, uint64_t node_id,
                                   const char *key)
{
    if (!g || !key) return NULL;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&g->rwlock);
    NodeEntry *e = find_node_entry(g, node_id);
    const char *result = NULL;
    if (e) {
        GV_GraphProp *p = find_prop(e->node.properties, key);
        if (p) result = p->value;
    }
    pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
    return result;
}

int gv_graph_find_nodes_by_label(const GV_GraphDB *g, const char *label,
                                 uint64_t *out_ids, size_t max_count)
{
    if (!g || !label || !out_ids) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&g->rwlock);

    int count = 0;
    for (size_t b = 0; b < g->node_bucket_count; b++) {
        NodeEntry *e = g->node_buckets[b];
        while (e) {
            if (strcmp(e->node.label, label) == 0) {
                if ((size_t)count < max_count) {
                    out_ids[count] = e->node.node_id;
                }
                count++;
            }
            e = e->next;
        }
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
    /* Return actual count written (capped by max_count) */
    return (count > (int)max_count) ? (int)max_count : count;
}

/* ============================================================================
 * Edge Operations
 * ============================================================================ */

uint64_t gv_graph_add_edge(GV_GraphDB *g, uint64_t source, uint64_t target,
                           const char *label, float weight)
{
    if (!g || !label) return 0;

    char *lbl = str_dup(label);
    if (!lbl) return 0;

    EdgeEntry *entry = (EdgeEntry *)calloc(1, sizeof(EdgeEntry));
    if (!entry) {
        free(lbl);
        return 0;
    }

    pthread_rwlock_wrlock(&g->rwlock);

    /* Referential integrity check */
    if (g->enforce_referential_integrity) {
        NodeEntry *src = find_node_entry(g, source);
        NodeEntry *tgt = find_node_entry(g, target);
        if (!src || !tgt) {
            pthread_rwlock_unlock(&g->rwlock);
            free(lbl);
            free(entry);
            return 0;
        }
    }

    uint64_t id = g->next_edge_id++;
    entry->edge.edge_id = id;
    entry->edge.source_id = source;
    entry->edge.target_id = target;
    entry->edge.label = lbl;
    entry->edge.weight = weight;
    entry->edge.properties = NULL;
    entry->edge.prop_count = 0;

    /* Insert into edge hash table */
    size_t idx = hash_u64(id, g->edge_bucket_count);
    entry->next = g->edge_buckets[idx];
    g->edge_buckets[idx] = entry;
    g->edge_count++;

    /* Update adjacency lists */
    NodeEntry *src_node = find_node_entry(g, source);
    NodeEntry *tgt_node = find_node_entry(g, target);
    if (src_node) {
        adj_add(&src_node->node.out_edges, &src_node->node.out_count,
                &src_node->node.out_cap, id, target);
    }
    if (tgt_node) {
        adj_add(&tgt_node->node.in_edges, &tgt_node->node.in_count,
                &tgt_node->node.in_cap, id, source);
    }

    pthread_rwlock_unlock(&g->rwlock);
    return id;
}

int gv_graph_remove_edge(GV_GraphDB *g, uint64_t edge_id)
{
    if (!g) return -1;

    pthread_rwlock_wrlock(&g->rwlock);
    int rc = remove_edge_internal(g, edge_id);
    pthread_rwlock_unlock(&g->rwlock);
    return rc;
}

const GV_GraphEdge *gv_graph_get_edge(const GV_GraphDB *g, uint64_t edge_id)
{
    if (!g) return NULL;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&g->rwlock);
    EdgeEntry *e = find_edge_entry(g, edge_id);
    const GV_GraphEdge *result = e ? &e->edge : NULL;
    pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
    return result;
}

int gv_graph_set_edge_prop(GV_GraphDB *g, uint64_t edge_id,
                           const char *key, const char *value)
{
    if (!g || !key || !value) return -1;

    pthread_rwlock_wrlock(&g->rwlock);
    EdgeEntry *e = find_edge_entry(g, edge_id);
    if (!e) {
        pthread_rwlock_unlock(&g->rwlock);
        return -1;
    }
    int rc = set_prop(&e->edge.properties, &e->edge.prop_count, key, value);
    pthread_rwlock_unlock(&g->rwlock);
    return rc;
}

const char *gv_graph_get_edge_prop(const GV_GraphDB *g, uint64_t edge_id,
                                   const char *key)
{
    if (!g || !key) return NULL;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&g->rwlock);
    EdgeEntry *e = find_edge_entry(g, edge_id);
    const char *result = NULL;
    if (e) {
        GV_GraphProp *p = find_prop(e->edge.properties, key);
        if (p) result = p->value;
    }
    pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
    return result;
}

int gv_graph_get_edges_out(const GV_GraphDB *g, uint64_t node_id,
                           uint64_t *out_ids, size_t max_count)
{
    if (!g || !out_ids) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&g->rwlock);

    NodeEntry *e = find_node_entry(g, node_id);
    if (!e) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return -1;
    }

    size_t n = e->node.out_count;
    if (n > max_count) n = max_count;
    for (size_t i = 0; i < n; i++) {
        out_ids[i] = e->node.out_edges[i].edge_id;
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
    return (int)n;
}

int gv_graph_get_edges_in(const GV_GraphDB *g, uint64_t node_id,
                          uint64_t *out_ids, size_t max_count)
{
    if (!g || !out_ids) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&g->rwlock);

    NodeEntry *e = find_node_entry(g, node_id);
    if (!e) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return -1;
    }

    size_t n = e->node.in_count;
    if (n > max_count) n = max_count;
    for (size_t i = 0; i < n; i++) {
        out_ids[i] = e->node.in_edges[i].edge_id;
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
    return (int)n;
}

int gv_graph_get_neighbors(const GV_GraphDB *g, uint64_t node_id,
                           uint64_t *out_ids, size_t max_count)
{
    if (!g || !out_ids) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&g->rwlock);

    NodeEntry *e = find_node_entry(g, node_id);
    if (!e) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return -1;
    }

    /* Use a small visited set for deduplication */
    size_t total_refs = e->node.out_count + e->node.in_count;
    VisitedSet seen;
    if (visited_init(&seen, total_refs > 16 ? total_refs * 2 : 32) != 0) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return -1;
    }

    int count = 0;
    /* Outgoing neighbors */
    for (size_t i = 0; i < e->node.out_count && (size_t)count < max_count; i++) {
        uint64_t nid = e->node.out_edges[i].neighbor_id;
        if (visited_insert(&seen, nid) == 1) {
            out_ids[count++] = nid;
        }
    }
    /* Incoming neighbors */
    for (size_t i = 0; i < e->node.in_count && (size_t)count < max_count; i++) {
        uint64_t nid = e->node.in_edges[i].neighbor_id;
        if (visited_insert(&seen, nid) == 1) {
            out_ids[count++] = nid;
        }
    }

    visited_free(&seen);
    pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
    return count;
}

/* ============================================================================
 * Traversal: BFS
 * ============================================================================ */

int gv_graph_bfs(const GV_GraphDB *g, uint64_t start, size_t max_depth,
                 uint64_t *out_ids, size_t max_count)
{
    if (!g || !out_ids) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&g->rwlock);

    /* Verify start node exists */
    if (!find_node_entry(g, start)) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return -1;
    }

    /* BFS queue: pairs of (node_id, depth) */
    size_t queue_cap = g->node_count > 64 ? g->node_count : 64;
    uint64_t *q_ids = (uint64_t *)malloc(queue_cap * sizeof(uint64_t));
    size_t *q_depths = (size_t *)malloc(queue_cap * sizeof(size_t));
    if (!q_ids || !q_depths) {
        free(q_ids);
        free(q_depths);
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return -1;
    }

    VisitedSet visited;
    if (visited_init(&visited, queue_cap * 2) != 0) {
        free(q_ids);
        free(q_depths);
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return -1;
    }

    size_t q_head = 0, q_tail = 0;
    int result_count = 0;

    /* Enqueue start */
    q_ids[q_tail] = start;
    q_depths[q_tail] = 0;
    q_tail++;
    visited_insert(&visited, start);

    while (q_head < q_tail) {
        uint64_t cur_id = q_ids[q_head];
        size_t cur_depth = q_depths[q_head];
        q_head++;

        /* Output this node */
        if ((size_t)result_count < max_count) {
            out_ids[result_count] = cur_id;
        }
        result_count++;

        /* Expand neighbors if within depth limit */
        if (cur_depth < max_depth) {
            NodeEntry *ne = find_node_entry(g, cur_id);
            if (ne) {
                /* Outgoing neighbors */
                for (size_t i = 0; i < ne->node.out_count; i++) {
                    uint64_t nid = ne->node.out_edges[i].neighbor_id;
                    if (visited_insert(&visited, nid) == 1) {
                        /* Grow queue if needed */
                        if (q_tail >= queue_cap) {
                            size_t new_cap = queue_cap * 2;
                            uint64_t *nqi = (uint64_t *)realloc(
                                q_ids, new_cap * sizeof(uint64_t));
                            size_t *nqd = (size_t *)realloc(
                                q_depths, new_cap * sizeof(size_t));
                            if (!nqi || !nqd) {
                                free(nqi ? nqi : q_ids);
                                free(nqd ? nqd : q_depths);
                                if (nqi) q_ids = nqi;
                                if (nqd) q_depths = nqd;
                                goto bfs_done;
                            }
                            q_ids = nqi;
                            q_depths = nqd;
                            queue_cap = new_cap;
                        }
                        q_ids[q_tail] = nid;
                        q_depths[q_tail] = cur_depth + 1;
                        q_tail++;
                    }
                }
                /* Incoming neighbors (treat as undirected for BFS) */
                for (size_t i = 0; i < ne->node.in_count; i++) {
                    uint64_t nid = ne->node.in_edges[i].neighbor_id;
                    if (visited_insert(&visited, nid) == 1) {
                        if (q_tail >= queue_cap) {
                            size_t new_cap = queue_cap * 2;
                            uint64_t *nqi = (uint64_t *)realloc(
                                q_ids, new_cap * sizeof(uint64_t));
                            size_t *nqd = (size_t *)realloc(
                                q_depths, new_cap * sizeof(size_t));
                            if (!nqi || !nqd) {
                                free(nqi ? nqi : q_ids);
                                free(nqd ? nqd : q_depths);
                                if (nqi) q_ids = nqi;
                                if (nqd) q_depths = nqd;
                                goto bfs_done;
                            }
                            q_ids = nqi;
                            q_depths = nqd;
                            queue_cap = new_cap;
                        }
                        q_ids[q_tail] = nid;
                        q_depths[q_tail] = cur_depth + 1;
                        q_tail++;
                    }
                }
            }
        }
    }

bfs_done:
    visited_free(&visited);
    free(q_ids);
    free(q_depths);
    pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);

    return (result_count > (int)max_count) ? (int)max_count : result_count;
}

/* ============================================================================
 * Traversal: DFS
 * ============================================================================ */

/**
 * @brief Recursive DFS helper (internal, no locking).
 */
static void dfs_recurse(const GV_GraphDB *g, uint64_t node_id,
                        size_t depth, size_t max_depth,
                        VisitedSet *visited,
                        uint64_t *out_ids, size_t max_count, int *count)
{
    if ((size_t)*count < max_count) {
        out_ids[*count] = node_id;
    }
    (*count)++;

    if (depth >= max_depth) return;

    NodeEntry *ne = find_node_entry(g, node_id);
    if (!ne) return;

    /* Outgoing */
    for (size_t i = 0; i < ne->node.out_count; i++) {
        uint64_t nid = ne->node.out_edges[i].neighbor_id;
        if (visited_insert(visited, nid) == 1) {
            dfs_recurse(g, nid, depth + 1, max_depth,
                        visited, out_ids, max_count, count);
        }
    }
    /* Incoming (treat as undirected) */
    for (size_t i = 0; i < ne->node.in_count; i++) {
        uint64_t nid = ne->node.in_edges[i].neighbor_id;
        if (visited_insert(visited, nid) == 1) {
            dfs_recurse(g, nid, depth + 1, max_depth,
                        visited, out_ids, max_count, count);
        }
    }
}

int gv_graph_dfs(const GV_GraphDB *g, uint64_t start, size_t max_depth,
                 uint64_t *out_ids, size_t max_count)
{
    if (!g || !out_ids) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&g->rwlock);

    if (!find_node_entry(g, start)) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return -1;
    }

    size_t vis_cap = g->node_count > 16 ? g->node_count * 2 : 32;
    VisitedSet visited;
    if (visited_init(&visited, vis_cap) != 0) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return -1;
    }

    visited_insert(&visited, start);
    int count = 0;
    dfs_recurse(g, start, 0, max_depth, &visited, out_ids, max_count, &count);

    visited_free(&visited);
    pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);

    return (count > (int)max_count) ? (int)max_count : count;
}

/* ============================================================================
 * Traversal: Dijkstra Shortest Path
 * ============================================================================ */

int gv_graph_shortest_path(const GV_GraphDB *g, uint64_t from, uint64_t to,
                           GV_GraphPath *path)
{
    if (!g || !path) return -1;

    memset(path, 0, sizeof(GV_GraphPath));

    pthread_rwlock_rdlock((pthread_rwlock_t *)&g->rwlock);

    if (!find_node_entry(g, from) || !find_node_entry(g, to)) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return -1;
    }

    /* Trivial case */
    if (from == to) {
        path->node_ids = (uint64_t *)malloc(sizeof(uint64_t));
        if (path->node_ids) {
            path->node_ids[0] = from;
        }
        path->edge_ids = NULL;
        path->length = 0;
        path->total_weight = 0.0f;
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return 0;
    }

    size_t cap = g->node_count > 16 ? g->node_count * 2 : 32;
    DistMap dm;
    if (distmap_init(&dm, cap) != 0) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return -1;
    }

    MinHeap heap;
    if (heap_init(&heap, cap) != 0) {
        distmap_free(&dm);
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return -1;
    }

    VisitedSet finalized;
    if (visited_init(&finalized, cap) != 0) {
        distmap_free(&dm);
        heap_free(&heap);
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return -1;
    }

    distmap_set(&dm, from, 0.0f, 0, 0);
    heap_push(&heap, from, 0.0f);

    int found = 0;

    while (heap.size > 0) {
        HeapEntry he;
        heap_pop(&heap, &he);

        if (visited_contains(&finalized, he.node_id)) continue;
        visited_insert(&finalized, he.node_id);

        if (he.node_id == to) {
            found = 1;
            break;
        }

        NodeEntry *ne = find_node_entry(g, he.node_id);
        if (!ne) continue;

        /* Relax outgoing edges */
        for (size_t i = 0; i < ne->node.out_count; i++) {
            uint64_t eid = ne->node.out_edges[i].edge_id;
            uint64_t nid = ne->node.out_edges[i].neighbor_id;
            if (visited_contains(&finalized, nid)) continue;

            EdgeEntry *ee = find_edge_entry(g, eid);
            if (!ee) continue;

            float w = ee->edge.weight;
            if (w < 0.0f) w = 0.0f; /* Dijkstra requires non-negative weights */
            float new_dist = he.dist + w;
            float old_dist = distmap_get(&dm, nid);

            if (new_dist < old_dist) {
                distmap_set(&dm, nid, new_dist, he.node_id, eid);
                heap_push(&heap, nid, new_dist);
            }
        }
    }

    if (!found) {
        distmap_free(&dm);
        heap_free(&heap);
        visited_free(&finalized);
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return -1;
    }

    /* Reconstruct path */
    /* Count path length by backtracking from 'to' to 'from' */
    size_t path_len = 0;
    {
        uint64_t cur = to;
        while (cur != from) {
            path_len++;
            cur = distmap_get_prev_node(&dm, cur);
            if (cur == 0 && from != 0) {
                /* Should not happen if found == 1, but guard against it */
                distmap_free(&dm);
                heap_free(&heap);
                visited_free(&finalized);
                pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
                return -1;
            }
        }
    }

    path->length = path_len;
    path->total_weight = distmap_get(&dm, to);
    path->node_ids = (uint64_t *)malloc((path_len + 1) * sizeof(uint64_t));
    path->edge_ids = (uint64_t *)malloc(path_len * sizeof(uint64_t));

    if (!path->node_ids || (path_len > 0 && !path->edge_ids)) {
        free(path->node_ids);
        free(path->edge_ids);
        memset(path, 0, sizeof(GV_GraphPath));
        distmap_free(&dm);
        heap_free(&heap);
        visited_free(&finalized);
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return -1;
    }

    /* Fill in reverse order */
    {
        uint64_t cur = to;
        for (size_t i = path_len; ; ) {
            path->node_ids[i] = cur;
            if (i == 0) break;
            i--;
            path->edge_ids[i] = distmap_get_prev_edge(&dm, cur);
            cur = distmap_get_prev_node(&dm, cur);
        }
    }

    distmap_free(&dm);
    heap_free(&heap);
    visited_free(&finalized);
    pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
    return 0;
}

/* ============================================================================
 * Traversal: All Paths (DFS with backtracking)
 * ============================================================================ */

/**
 * @brief Build a GV_GraphPath from the current path stack.
 */
static int build_path(const GV_GraphDB *g,
                      const uint64_t *node_stack, const uint64_t *edge_stack,
                      size_t depth, GV_GraphPath *out)
{
    (void)g;
    out->length = depth;
    out->node_ids = (uint64_t *)malloc((depth + 1) * sizeof(uint64_t));
    out->edge_ids = depth > 0 ? (uint64_t *)malloc(depth * sizeof(uint64_t)) : NULL;
    if (!out->node_ids || (depth > 0 && !out->edge_ids)) {
        free(out->node_ids);
        free(out->edge_ids);
        memset(out, 0, sizeof(GV_GraphPath));
        return -1;
    }
    memcpy(out->node_ids, node_stack, (depth + 1) * sizeof(uint64_t));
    if (depth > 0) {
        memcpy(out->edge_ids, edge_stack, depth * sizeof(uint64_t));
    }

    /* Compute total weight */
    out->total_weight = 0.0f;
    for (size_t i = 0; i < depth; i++) {
        EdgeEntry *ee = find_edge_entry(g, edge_stack[i]);
        if (ee) out->total_weight += ee->edge.weight;
    }
    return 0;
}

/**
 * @brief Simple on-path check using the node_stack directly.
 */
static int is_on_path(const uint64_t *node_stack, size_t depth, uint64_t id)
{
    for (size_t i = 0; i <= depth; i++) {
        if (node_stack[i] == id) return 1;
    }
    return 0;
}

static void all_paths_dfs(const GV_GraphDB *g,
                          uint64_t cur, uint64_t to,
                          size_t depth, size_t max_depth,
                          uint64_t *node_stack, uint64_t *edge_stack,
                          GV_GraphPath *paths, size_t max_paths,
                          int *path_count)
{
    if (cur == to) {
        if ((size_t)*path_count < max_paths) {
            build_path(g, node_stack, edge_stack, depth, &paths[*path_count]);
        }
        (*path_count)++;
        return;
    }

    if (depth >= max_depth) return;
    if ((size_t)*path_count >= max_paths) return;

    NodeEntry *ne = find_node_entry(g, cur);
    if (!ne) return;

    for (size_t i = 0; i < ne->node.out_count; i++) {
        uint64_t nid = ne->node.out_edges[i].neighbor_id;
        uint64_t eid = ne->node.out_edges[i].edge_id;

        if (is_on_path(node_stack, depth, nid)) continue;

        node_stack[depth + 1] = nid;
        edge_stack[depth] = eid;

        all_paths_dfs(g, nid, to, depth + 1, max_depth,
                      node_stack, edge_stack,
                      paths, max_paths, path_count);

        if ((size_t)*path_count >= max_paths) return;
    }
}

int gv_graph_all_paths(const GV_GraphDB *g, uint64_t from, uint64_t to,
                       size_t max_depth, GV_GraphPath *paths, size_t max_paths)
{
    if (!g || !paths) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&g->rwlock);

    if (!find_node_entry(g, from) || !find_node_entry(g, to)) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return -1;
    }

    /* Allocate stacks for DFS */
    uint64_t *node_stack = (uint64_t *)malloc((max_depth + 2) * sizeof(uint64_t));
    uint64_t *edge_stack = (uint64_t *)malloc((max_depth + 1) * sizeof(uint64_t));
    if (!node_stack || !edge_stack) {
        free(node_stack);
        free(edge_stack);
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return -1;
    }

    node_stack[0] = from;
    int path_count = 0;

    all_paths_dfs(g, from, to, 0, max_depth,
                  node_stack, edge_stack,
                  paths, max_paths, &path_count);

    free(node_stack);
    free(edge_stack);
    pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);

    return (path_count > (int)max_paths) ? (int)max_paths : path_count;
}

void gv_graph_free_path(GV_GraphPath *path)
{
    if (!path) return;
    free(path->node_ids);
    free(path->edge_ids);
    path->node_ids = NULL;
    path->edge_ids = NULL;
    path->length = 0;
    path->total_weight = 0.0f;
}

/* ============================================================================
 * Analytics: PageRank
 * ============================================================================ */

float gv_graph_pagerank(const GV_GraphDB *g, uint64_t node_id,
                        size_t iterations, float damping)
{
    if (!g || iterations == 0) return 0.0f;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&g->rwlock);

    size_t N = g->node_count;
    if (N == 0) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return 0.0f;
    }

    /* Collect all node IDs */
    size_t id_count = 0;
    uint64_t *all_ids = collect_all_node_ids(g, &id_count);
    if (!all_ids || id_count == 0) {
        free(all_ids);
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return 0.0f;
    }

    /* Map node_id -> index for O(1) lookup.
     * Use a hash map (open addressing). */
    size_t map_cap = id_count * 3;
    uint64_t *map_keys = (uint64_t *)malloc(map_cap * sizeof(uint64_t));
    int *map_occ = (int *)calloc(map_cap, sizeof(int));
    size_t *map_idx = (size_t *)malloc(map_cap * sizeof(size_t));
    float *scores = (float *)malloc(id_count * sizeof(float));
    float *new_scores = (float *)malloc(id_count * sizeof(float));

    if (!map_keys || !map_occ || !map_idx || !scores || !new_scores) {
        free(all_ids); free(map_keys); free(map_occ);
        free(map_idx); free(scores); free(new_scores);
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return 0.0f;
    }

    /* Build ID->index map */
    for (size_t i = 0; i < id_count; i++) {
        size_t h = (size_t)(all_ids[i] * 2654435761ULL) % map_cap;
        for (size_t j = 0; j < map_cap; j++) {
            size_t p = (h + j) % map_cap;
            if (!map_occ[p]) {
                map_keys[p] = all_ids[i];
                map_occ[p] = 1;
                map_idx[p] = i;
                break;
            }
        }
        scores[i] = 1.0f / (float)N;
    }

    /* Iterative PageRank */
    float base = (1.0f - damping) / (float)N;
    for (size_t iter = 0; iter < iterations; iter++) {
        for (size_t i = 0; i < id_count; i++) {
            new_scores[i] = base;
        }

        for (size_t i = 0; i < id_count; i++) {
            NodeEntry *ne = find_node_entry(g, all_ids[i]);
            if (!ne) continue;
            size_t out_deg = ne->node.out_count;
            if (out_deg == 0) {
                /* Dangling node: distribute evenly */
                float share = damping * scores[i] / (float)N;
                for (size_t j = 0; j < id_count; j++) {
                    new_scores[j] += share;
                }
            } else {
                float share = damping * scores[i] / (float)out_deg;
                for (size_t e = 0; e < out_deg; e++) {
                    uint64_t tgt = ne->node.out_edges[e].neighbor_id;
                    size_t tgt_idx = idmap_lookup(map_keys, map_occ, map_idx,
                                                  map_cap, tgt);
                    if (tgt_idx != (size_t)-1) {
                        new_scores[tgt_idx] += share;
                    }
                }
            }
        }

        /* Swap scores */
        float *tmp = scores;
        scores = new_scores;
        new_scores = tmp;
    }

    /* Find the target node's score */
    float result = 0.0f;
    size_t target_idx = idmap_lookup(map_keys, map_occ, map_idx,
                                     map_cap, node_id);
    if (target_idx != (size_t)-1) {
        result = scores[target_idx];
    }

    free(all_ids);
    free(map_keys);
    free(map_occ);
    free(map_idx);
    free(scores);
    free(new_scores);
    pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
    return result;
}

/* ============================================================================
 * Analytics: Degree Functions
 * ============================================================================ */

size_t gv_graph_degree(const GV_GraphDB *g, uint64_t node_id)
{
    if (!g) return 0;
    pthread_rwlock_rdlock((pthread_rwlock_t *)&g->rwlock);
    NodeEntry *e = find_node_entry(g, node_id);
    size_t deg = e ? (e->node.in_count + e->node.out_count) : 0;
    pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
    return deg;
}

size_t gv_graph_in_degree(const GV_GraphDB *g, uint64_t node_id)
{
    if (!g) return 0;
    pthread_rwlock_rdlock((pthread_rwlock_t *)&g->rwlock);
    NodeEntry *e = find_node_entry(g, node_id);
    size_t deg = e ? e->node.in_count : 0;
    pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
    return deg;
}

size_t gv_graph_out_degree(const GV_GraphDB *g, uint64_t node_id)
{
    if (!g) return 0;
    pthread_rwlock_rdlock((pthread_rwlock_t *)&g->rwlock);
    NodeEntry *e = find_node_entry(g, node_id);
    size_t deg = e ? e->node.out_count : 0;
    pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
    return deg;
}

/* ============================================================================
 * Analytics: Connected Components
 * ============================================================================ */

int gv_graph_connected_components(const GV_GraphDB *g,
                                  uint64_t *component_ids, size_t max_count)
{
    if (!g || !component_ids) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&g->rwlock);

    size_t N = g->node_count;
    if (N == 0) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return 0;
    }
    if (max_count < N) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return -1;
    }

    /* Collect all node IDs */
    size_t id_count = 0;
    uint64_t *all_ids = collect_all_node_ids(g, &id_count);
    if (!all_ids) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return -1;
    }

    /* Initialize component IDs to 0 (unassigned) */
    for (size_t i = 0; i < id_count; i++) {
        component_ids[i] = 0;
    }

    /* Build ID->index map */
    size_t map_cap = id_count * 3;
    uint64_t *map_keys = (uint64_t *)malloc(map_cap * sizeof(uint64_t));
    int *map_occ = (int *)calloc(map_cap, sizeof(int));
    size_t *map_idx = (size_t *)malloc(map_cap * sizeof(size_t));

    if (!map_keys || !map_occ || !map_idx) {
        free(all_ids); free(map_keys); free(map_occ); free(map_idx);
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return -1;
    }

    for (size_t i = 0; i < id_count; i++) {
        size_t h = (size_t)(all_ids[i] * 2654435761ULL) % map_cap;
        for (size_t j = 0; j < map_cap; j++) {
            size_t p = (h + j) % map_cap;
            if (!map_occ[p]) {
                map_keys[p] = all_ids[i];
                map_occ[p] = 1;
                map_idx[p] = i;
                break;
            }
        }
    }

    /* Inline index lookup */
    /* BFS queue */
    uint64_t *queue = (uint64_t *)malloc(id_count * sizeof(uint64_t));
    if (!queue) {
        free(all_ids); free(map_keys); free(map_occ); free(map_idx);
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return -1;
    }

    uint64_t comp_id = 0;

    for (size_t i = 0; i < id_count; i++) {
        if (component_ids[i] != 0) continue; /* Already assigned */

        comp_id++;
        size_t q_head = 0, q_tail = 0;
        queue[q_tail++] = all_ids[i];
        component_ids[i] = comp_id;

        while (q_head < q_tail) {
            uint64_t cur = queue[q_head++];
            NodeEntry *ne = find_node_entry(g, cur);
            if (!ne) continue;

            /* Outgoing neighbors */
            for (size_t e = 0; e < ne->node.out_count; e++) {
                uint64_t nid = ne->node.out_edges[e].neighbor_id;
                size_t nidx = idmap_lookup(map_keys, map_occ, map_idx,
                                           map_cap, nid);
                if (nidx != (size_t)-1 && component_ids[nidx] == 0) {
                    component_ids[nidx] = comp_id;
                    queue[q_tail++] = nid;
                }
            }
            /* Incoming neighbors (undirected treatment) */
            for (size_t e = 0; e < ne->node.in_count; e++) {
                uint64_t nid = ne->node.in_edges[e].neighbor_id;
                size_t nidx = idmap_lookup(map_keys, map_occ, map_idx,
                                           map_cap, nid);
                if (nidx != (size_t)-1 && component_ids[nidx] == 0) {
                    component_ids[nidx] = comp_id;
                    queue[q_tail++] = nid;
                }
            }
        }
    }

    free(all_ids);
    free(map_keys);
    free(map_occ);
    free(map_idx);
    free(queue);
    pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
    return (int)comp_id;
}

/* ============================================================================
 * Analytics: Clustering Coefficient
 * ============================================================================ */

float gv_graph_clustering_coefficient(const GV_GraphDB *g, uint64_t node_id)
{
    if (!g) return 0.0f;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&g->rwlock);

    NodeEntry *ne = find_node_entry(g, node_id);
    if (!ne) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return 0.0f;
    }

    /* Collect unique neighbors (undirected) */
    size_t total_refs = ne->node.out_count + ne->node.in_count;
    if (total_refs < 2) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return 0.0f;
    }

    /* Gather unique neighbor IDs */
    uint64_t *neighbors = (uint64_t *)malloc(total_refs * sizeof(uint64_t));
    if (!neighbors) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return 0.0f;
    }

    VisitedSet seen;
    if (visited_init(&seen, total_refs * 2 + 16) != 0) {
        free(neighbors);
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return 0.0f;
    }

    size_t k = 0;
    for (size_t i = 0; i < ne->node.out_count; i++) {
        uint64_t nid = ne->node.out_edges[i].neighbor_id;
        if (visited_insert(&seen, nid) == 1) {
            neighbors[k++] = nid;
        }
    }
    for (size_t i = 0; i < ne->node.in_count; i++) {
        uint64_t nid = ne->node.in_edges[i].neighbor_id;
        if (visited_insert(&seen, nid) == 1) {
            neighbors[k++] = nid;
        }
    }
    visited_free(&seen);

    if (k < 2) {
        free(neighbors);
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return 0.0f;
    }

    /* Build a set of neighbor IDs for fast lookup */
    VisitedSet nbr_set;
    if (visited_init(&nbr_set, k * 3 + 16) != 0) {
        free(neighbors);
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return 0.0f;
    }
    for (size_t i = 0; i < k; i++) {
        visited_insert(&nbr_set, neighbors[i]);
    }

    /* Count edges among neighbors (undirected: check if any edge exists
     * between each pair, treating directed edges as undirected connections) */
    size_t edge_count_among = 0;
    for (size_t i = 0; i < k; i++) {
        NodeEntry *nne = find_node_entry(g, neighbors[i]);
        if (!nne) continue;

        /* Check outgoing edges to other neighbors */
        for (size_t e = 0; e < nne->node.out_count; e++) {
            uint64_t tgt = nne->node.out_edges[e].neighbor_id;
            if (tgt != node_id && visited_contains(&nbr_set, tgt)) {
                edge_count_among++;
            }
        }
        /* Check incoming edges from other neighbors */
        for (size_t e = 0; e < nne->node.in_count; e++) {
            uint64_t src = nne->node.in_edges[e].neighbor_id;
            if (src != node_id && visited_contains(&nbr_set, src)) {
                edge_count_among++;
            }
        }
    }

    /* Each undirected edge is counted twice (once from each endpoint) */
    edge_count_among /= 2;

    /* Possible edges = k*(k-1)/2 for undirected */
    size_t possible = k * (k - 1) / 2;
    float cc = (possible > 0) ? (float)edge_count_among / (float)possible : 0.0f;

    visited_free(&nbr_set);
    free(neighbors);
    pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
    return cc;
}

/* ============================================================================
 * Stats
 * ============================================================================ */

size_t gv_graph_node_count(const GV_GraphDB *g)
{
    if (!g) return 0;
    pthread_rwlock_rdlock((pthread_rwlock_t *)&g->rwlock);
    size_t count = g->node_count;
    pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
    return count;
}

size_t gv_graph_edge_count(const GV_GraphDB *g)
{
    if (!g) return 0;
    pthread_rwlock_rdlock((pthread_rwlock_t *)&g->rwlock);
    size_t count = g->edge_count;
    pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
    return count;
}

/* ============================================================================
 * Persistence: Save
 * ============================================================================ */

int gv_graph_save(const GV_GraphDB *g, const char *path)
{
    if (!g || !path) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&g->rwlock);

    FILE *f = fopen(path, "wb");
    if (!f) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
        return -1;
    }

    /* Magic */
    if (fwrite(GV_GRAPH_MAGIC, 1, GV_GRAPH_MAGIC_LEN, f) != GV_GRAPH_MAGIC_LEN)
        goto save_fail;

    /* Version */
    uint32_t version = GV_GRAPH_VERSION;
    if (write_u32(f, version) != 0) goto save_fail;

    /* Counts and ID counters */
    if (write_u64(f, (uint64_t)g->node_count) != 0) goto save_fail;
    if (write_u64(f, (uint64_t)g->edge_count) != 0) goto save_fail;
    if (write_u64(f, g->next_node_id) != 0) goto save_fail;
    if (write_u64(f, g->next_edge_id) != 0) goto save_fail;

    /* Serialize nodes */
    for (size_t b = 0; b < g->node_bucket_count; b++) {
        NodeEntry *e = g->node_buckets[b];
        while (e) {
            if (write_u64(f, e->node.node_id) != 0) goto save_fail;
            if (write_str(f, e->node.label) != 0) goto save_fail;

            /* Properties */
            if (write_u32(f, (uint32_t)e->node.prop_count) != 0) goto save_fail;
            GV_GraphProp *prop = e->node.properties;
            while (prop) {
                if (write_str(f, prop->key) != 0) goto save_fail;
                if (write_str(f, prop->value) != 0) goto save_fail;
                prop = prop->next;
            }

            e = e->next;
        }
    }

    /* Serialize edges */
    for (size_t b = 0; b < g->edge_bucket_count; b++) {
        EdgeEntry *e = g->edge_buckets[b];
        while (e) {
            if (write_u64(f, e->edge.edge_id) != 0) goto save_fail;
            if (write_u64(f, e->edge.source_id) != 0) goto save_fail;
            if (write_u64(f, e->edge.target_id) != 0) goto save_fail;
            if (write_str(f, e->edge.label) != 0) goto save_fail;
            if (write_float(f, e->edge.weight) != 0) goto save_fail;

            /* Properties */
            if (write_u32(f, (uint32_t)e->edge.prop_count) != 0) goto save_fail;
            GV_GraphProp *prop = e->edge.properties;
            while (prop) {
                if (write_str(f, prop->key) != 0) goto save_fail;
                if (write_str(f, prop->value) != 0) goto save_fail;
                prop = prop->next;
            }

            e = e->next;
        }
    }

    fclose(f);
    pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
    return 0;

save_fail:
    fclose(f);
    pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
    return -1;
}

/* ============================================================================
 * Persistence: Load
 * ============================================================================ */

GV_GraphDB *gv_graph_load(const char *path)
{
    if (!path) return NULL;

    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    /* Read and verify magic */
    char magic[GV_GRAPH_MAGIC_LEN];
    if (fread(magic, 1, GV_GRAPH_MAGIC_LEN, f) != GV_GRAPH_MAGIC_LEN) {
        fclose(f);
        return NULL;
    }
    if (memcmp(magic, GV_GRAPH_MAGIC, GV_GRAPH_MAGIC_LEN) != 0) {
        fclose(f);
        return NULL;
    }

    /* Version */
    uint32_t version;
    if (read_u32(f, &version) != 0 || version != GV_GRAPH_VERSION) {
        fclose(f);
        return NULL;
    }

    /* Counts and ID counters */
    uint64_t node_count, edge_count, next_node_id, next_edge_id;
    if (read_u64(f, &node_count) != 0) { fclose(f); return NULL; }
    if (read_u64(f, &edge_count) != 0) { fclose(f); return NULL; }
    if (read_u64(f, &next_node_id) != 0) { fclose(f); return NULL; }
    if (read_u64(f, &next_edge_id) != 0) { fclose(f); return NULL; }

    /* Create graph with default config */
    GV_GraphDBConfig cfg;
    gv_graph_config_init(&cfg);
    /* Disable referential integrity during load since we add nodes first,
     * then edges -- but edges reference nodes already added. Keep it enabled
     * to validate; we add nodes first so it should be fine. However, to be
     * safe during load, we temporarily disable it. */
    cfg.enforce_referential_integrity = 0;

    GV_GraphDB *g = gv_graph_create(&cfg);
    if (!g) {
        fclose(f);
        return NULL;
    }

    /* Load nodes */
    for (uint64_t i = 0; i < node_count; i++) {
        uint64_t nid;
        if (read_u64(f, &nid) != 0) goto load_fail;

        char *label = read_str(f);
        if (!label) goto load_fail;

        uint32_t prop_count;
        if (read_u32(f, &prop_count) != 0) {
            free(label);
            goto load_fail;
        }

        /* Create node entry directly (bypass auto-increment) */
        NodeEntry *entry = (NodeEntry *)calloc(1, sizeof(NodeEntry));
        if (!entry) {
            free(label);
            goto load_fail;
        }
        entry->node.node_id = nid;
        entry->node.label = label;
        entry->node.properties = NULL;
        entry->node.prop_count = 0;
        entry->node.out_edges = NULL;
        entry->node.out_count = 0;
        entry->node.out_cap = 0;
        entry->node.in_edges = NULL;
        entry->node.in_count = 0;
        entry->node.in_cap = 0;

        /* Load properties */
        for (uint32_t p = 0; p < prop_count; p++) {
            char *key = read_str(f);
            char *val = read_str(f);
            if (!key || !val) {
                free(key); free(val);
                free_node_internals(&entry->node);
                free(entry);
                goto load_fail;
            }
            if (set_prop(&entry->node.properties, &entry->node.prop_count,
                         key, val) != 0) {
                free(key); free(val);
                free_node_internals(&entry->node);
                free(entry);
                goto load_fail;
            }
            free(key);
            free(val);
        }

        /* Insert into hash table */
        size_t idx = hash_u64(nid, g->node_bucket_count);
        entry->next = g->node_buckets[idx];
        g->node_buckets[idx] = entry;
        g->node_count++;
    }
    g->next_node_id = next_node_id;

    /* Load edges */
    for (uint64_t i = 0; i < edge_count; i++) {
        uint64_t eid, src_id, tgt_id;
        if (read_u64(f, &eid) != 0) goto load_fail;
        if (read_u64(f, &src_id) != 0) goto load_fail;
        if (read_u64(f, &tgt_id) != 0) goto load_fail;

        char *label = read_str(f);
        if (!label) goto load_fail;

        float weight;
        if (read_float(f, &weight) != 0) {
            free(label);
            goto load_fail;
        }

        uint32_t prop_count;
        if (read_u32(f, &prop_count) != 0) {
            free(label);
            goto load_fail;
        }

        EdgeEntry *entry = (EdgeEntry *)calloc(1, sizeof(EdgeEntry));
        if (!entry) {
            free(label);
            goto load_fail;
        }
        entry->edge.edge_id = eid;
        entry->edge.source_id = src_id;
        entry->edge.target_id = tgt_id;
        entry->edge.label = label;
        entry->edge.weight = weight;
        entry->edge.properties = NULL;
        entry->edge.prop_count = 0;

        /* Load properties */
        for (uint32_t p = 0; p < prop_count; p++) {
            char *key = read_str(f);
            char *val = read_str(f);
            if (!key || !val) {
                free(key); free(val);
                free_edge_internals(&entry->edge);
                free(entry);
                goto load_fail;
            }
            if (set_prop(&entry->edge.properties, &entry->edge.prop_count,
                         key, val) != 0) {
                free(key); free(val);
                free_edge_internals(&entry->edge);
                free(entry);
                goto load_fail;
            }
            free(key);
            free(val);
        }

        /* Insert into edge hash table */
        size_t idx = hash_u64(eid, g->edge_bucket_count);
        entry->next = g->edge_buckets[idx];
        g->edge_buckets[idx] = entry;
        g->edge_count++;

        /* Update adjacency lists */
        NodeEntry *src_node = find_node_entry(g, src_id);
        NodeEntry *tgt_node = find_node_entry(g, tgt_id);
        if (src_node) {
            adj_add(&src_node->node.out_edges, &src_node->node.out_count,
                    &src_node->node.out_cap, eid, tgt_id);
        }
        if (tgt_node) {
            adj_add(&tgt_node->node.in_edges, &tgt_node->node.in_count,
                    &tgt_node->node.in_cap, eid, src_id);
        }
    }
    g->next_edge_id = next_edge_id;

    /* Re-enable referential integrity (default) */
    g->enforce_referential_integrity = 1;

    fclose(f);
    return g;

load_fail:
    fclose(f);
    gv_graph_destroy(g);
    return NULL;
}
