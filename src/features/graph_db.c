/**
 * @file graph_db.c
 * @brief Full graph database layer implementation for GigaVector.
 *
 * Implements a property-graph model backed by hash-table storage with chaining,
 * dynamic adjacency lists, thread-safe access via pthread_rwlock_t, traversal
 * algorithms (BFS, DFS, Dijkstra, all-paths), graph analytics (PageRank,
 * clustering coefficient, connected components), and binary persistence.
 */

#define _POSIX_C_SOURCE 200112L

#include "features/graph_db.h"
#include "core/memory.h"
#include "core/utils.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include <pthread.h>
#ifndef _WIN32
#include <unistd.h>
#else
#include <io.h>
#define fsync(fd) _commit(fd)
#endif

#include "features/graph_db_internal.h"

#define DEFAULT_NODE_BUCKETS 4096
#define DEFAULT_EDGE_BUCKETS 8192
#define INITIAL_ADJ_CAP      4

void journal_break_locked(GV_GraphDB *g)
{
    g->edge_journal_len = 0;
    g->journal_base_version = g->mutation_count;
}

static void journal_edge_locked(GV_GraphDB *g, uint64_t u, uint64_t v,
                                int removed)
{
    if (!g || u == 0 || v == 0) return;
    if (g->edge_journal_len >= GV_EDGE_JOURNAL_MAX) {
        journal_break_locked(g);   /* overflow: coverage restarts from now */
        return;
    }
    if (g->edge_journal_len == g->edge_journal_cap) {
        size_t nc = g->edge_journal_cap ? g->edge_journal_cap * 2 : 64;
        GV_GraphEdgeDelta *t = (GV_GraphEdgeDelta *)gv_realloc(
            g->edge_journal, nc * sizeof(GV_GraphEdgeDelta));
        if (!t) { journal_break_locked(g); return; }
        g->edge_journal = t;
        g->edge_journal_cap = nc;
    }
    GV_GraphEdgeDelta *d = &g->edge_journal[g->edge_journal_len++];
    d->version = g->mutation_count;
    d->source  = u;
    d->target  = v;
    d->removed = (uint8_t)(removed ? 1 : 0);
}

typedef struct {
    uint8_t  buf[GV_GWAL_BUF_CAP];
    size_t   len;
    int      overflow;
} GWalBuf;

static void gwal_init(GWalBuf *b)
{
    b->len = 0;
    b->overflow = 0;
}

/* Returns 0 and flags the record unloggable if it would exceed the buffer. */
static int gwal_reserve(GWalBuf *b, size_t n)
{
    if (b->len + n > GV_GWAL_BUF_CAP) {
        b->overflow = 1;
        return 0;
    }
    return 1;
}

static void gwal_put_u8(GWalBuf *b, uint8_t v)
{
    if (!gwal_reserve(b, 1)) return;
    b->buf[b->len++] = v;
}

static void gwal_put_u32(GWalBuf *b, uint32_t v)
{
    if (!gwal_reserve(b, 4)) return;
    for (int i = 0; i < 4; i++) b->buf[b->len++] = (uint8_t)(v >> (8 * i));
}

static void gwal_put_u64(GWalBuf *b, uint64_t v)
{
    if (!gwal_reserve(b, 8)) return;
    for (int i = 0; i < 8; i++) b->buf[b->len++] = (uint8_t)(v >> (8 * i));
}

static void gwal_put_f32(GWalBuf *b, float v)
{
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    gwal_put_u32(b, bits);
}

static void gwal_put_str(GWalBuf *b, const char *s)
{
    size_t n = s ? strlen(s) : 0;
    if (!gwal_reserve(b, 4)) return;
    if (n > UINT32_MAX || !gwal_reserve(b, n)) return;
    gwal_put_u32(b, (uint32_t)n);
    if (n) {
        memcpy(b->buf + b->len, s, n);
        b->len += n;
    }
}

static const uint8_t *gwal_get_u32(const uint8_t *p, const uint8_t *end, uint32_t *v)
{
    if (end - p < 4) return NULL;
    *v = 0;
    for (int i = 0; i < 4; i++) *v |= (uint32_t)p[i] << (8 * i);
    return p + 4;
}

static const uint8_t *gwal_get_u64(const uint8_t *p, const uint8_t *end, uint64_t *v)
{
    if (end - p < 8) return NULL;
    *v = 0;
    for (int i = 0; i < 8; i++) *v |= (uint64_t)p[i] << (8 * i);
    return p + 8;
}

static const uint8_t *gwal_get_f32(const uint8_t *p, const uint8_t *end, float *v)
{
    uint32_t bits;
    p = gwal_get_u32(p, end, &bits);
    if (p) memcpy(v, &bits, sizeof(*v));
    return p;
}

static const uint8_t *gwal_get_str(const uint8_t *p, const uint8_t *end, char **out)
{
    uint32_t n;
    if ((p = gwal_get_u32(p, end, &n)) == NULL || (size_t)(end - p) < n) return NULL;
    *out = (char *)gv_alloc((size_t)n + 1);
    if (!*out) return NULL;
    if (n) memcpy(*out, p, n);
    (*out)[n] = '\0';
    return p + n;
}

/* Append+fsync one record, framed as [op][u32 payload_len][payload]. Called
 * with the write lock held (mutations) or during single-threaded replay.
 * Returns 0 on success. */
static int gwal_append(GV_GraphDB *g, const GWalBuf *b)
{
    if (!g->wal_file || g->wal_replaying) return 0;
    if (b->overflow) return -1;   /* record too large to frame */
    uint8_t hdr[5];
    hdr[0] = b->buf[0];
    uint32_t plen = (uint32_t)(b->len - 1);
    for (int i = 0; i < 4; i++) hdr[1 + i] = (uint8_t)(plen >> (8 * i));
    if (fwrite(hdr, 1, sizeof(hdr), g->wal_file) != sizeof(hdr)) return -1;
    if (b->len > 1 && fwrite(b->buf + 1, 1, b->len - 1, g->wal_file) != b->len - 1)
        return -1;
    if (fflush(g->wal_file) != 0) return -1;
    if (fsync(fileno(g->wal_file)) != 0) return -1;
    return 0;
}

void graph_wal_close(GV_GraphDB *g)
{
    if (g->wal_file) {
        fclose(g->wal_file);
        g->wal_file = NULL;
    }
    gv_free(g->wal_path);
    g->wal_path = NULL;
}


static void free_prop_list(GV_GraphProp *head)
{
    while (head) {
        GV_GraphProp *next = head->next;
        gv_free(head->key);
        gv_prop_free(&head->value);
        gv_free(head);
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
 * @brief Borrowed string view of a property, valid until it is mutated or freed.
 * @return The internal string for GV_PROP_STRING props, NULL for other types.
 */
static const char *prop_borrow_string(const GV_GraphProp *p)
{
    return (p && p->value.type == GV_PROP_STRING) ? p->value.as.s : NULL;
}

/**
 * @brief Set or overwrite a property in a linked list.
 * @return 0 on success, -1 on allocation failure.
 */
/** @brief Set a property to a typed value, taking ownership of its heap members. */
static int set_prop_typed(GV_GraphProp **head, size_t *count,
                          const char *key, GV_PropValue value)
{
    GV_GraphProp *existing = find_prop(*head, key);
    if (existing) {
        gv_prop_free(&existing->value);
        existing->value = value;
        return 0;
    }
    GV_GraphProp *prop = (GV_GraphProp *)gv_calloc(1, sizeof(GV_GraphProp));
    if (!prop) { gv_prop_free(&value); return -1; }
    prop->key = gv_dup_cstr(key);
    if (!prop->key) {
        gv_free(prop);
        gv_prop_free(&value);
        return -1;
    }
    prop->value = value;
    prop->next = *head;
    *head = prop;
    (*count)++;
    return 0;
}

int set_prop(GV_GraphProp **head, size_t *count,
                    const char *key, const char *value)
{
    GV_PropValue v = gv_prop_string(value);
    if (v.type == GV_PROP_STRING && !v.as.s) return -1; /* strdup failed */
    return set_prop_typed(head, count, key, v);
}

NodeEntry *find_node_entry(const GV_GraphDB *g, uint64_t node_id)
{
    size_t idx = hash_u64(node_id, g->node_bucket_count);
    NodeEntry *e = g->node_buckets[idx];
    while (e) {
        if (e->node.node_id == node_id) return e;
        e = e->next;
    }
    return NULL;
}

EdgeEntry *find_edge_entry(const GV_GraphDB *g, uint64_t edge_id)
{
    size_t idx = hash_u64(edge_id, g->edge_bucket_count);
    EdgeEntry *e = g->edge_buckets[idx];
    while (e) {
        if (e->edge.edge_id == edge_id) return e;
        e = e->next;
    }
    return NULL;
}

static LabelEntry *find_label_entry(const GV_GraphDB *g, const char *label)
{
    size_t idx = hash_str(label) % g->label_bucket_count;
    LabelEntry *e = g->label_buckets[idx];
    while (e) {
        if (strcmp(e->label, label) == 0) return e;
        e = e->next;
    }
    return NULL;
}

int label_index_add(GV_GraphDB *g, const char *label, uint64_t node_id)
{
    LabelEntry *e = find_label_entry(g, label);
    if (!e) {
        e = (LabelEntry *)gv_calloc(1, sizeof(LabelEntry));
        if (!e) return -1;
        e->label = gv_dup_cstr(label);
        if (!e->label) {
            gv_free(e);
            return -1;
        }
        size_t idx = hash_str(label) % g->label_bucket_count;
        e->next = g->label_buckets[idx];
        g->label_buckets[idx] = e;
    }
    if (e->count >= e->cap) {
        size_t new_cap = e->cap ? e->cap * 2 : 4;
        uint64_t *tmp = (uint64_t *)gv_realloc(e->ids, new_cap * sizeof(uint64_t));
        if (!tmp) return -1;
        e->ids = tmp;
        e->cap = new_cap;
    }
    e->ids[e->count++] = node_id;
    return 0;
}

static void label_index_remove(GV_GraphDB *g, const char *label, uint64_t node_id)
{
    LabelEntry *e = find_label_entry(g, label);
    if (!e) return;
    for (size_t i = 0; i < e->count; i++) {
        if (e->ids[i] == node_id) {
            e->ids[i] = e->ids[e->count - 1];
            e->count--;
            break;
        }
    }
}

/* Double a chained table once load factor exceeds 1.0 so lookups stay O(1). */
static int grow_node_table(GV_GraphDB *g)
{
    if (g->node_count <= g->node_bucket_count) return 0;
    size_t new_count = g->node_bucket_count * 2;
    NodeEntry **nb = (NodeEntry **)gv_calloc(new_count, sizeof(NodeEntry *));
    if (!nb) return 0; /* keep serving from the old table on OOM */
    for (size_t b = 0; b < g->node_bucket_count; b++) {
        NodeEntry *e = g->node_buckets[b];
        while (e) {
            NodeEntry *next = e->next;
            size_t idx = hash_u64(e->node.node_id, new_count);
            e->next = nb[idx];
            nb[idx] = e;
            e = next;
        }
    }
    gv_free(g->node_buckets);
    g->node_buckets = nb;
    g->node_bucket_count = new_count;
    return 0;
}

static int grow_edge_table(GV_GraphDB *g)
{
    if (g->edge_count <= g->edge_bucket_count) return 0;
    size_t new_count = g->edge_bucket_count * 2;
    EdgeEntry **eb = (EdgeEntry **)gv_calloc(new_count, sizeof(EdgeEntry *));
    if (!eb) return 0;
    for (size_t b = 0; b < g->edge_bucket_count; b++) {
        EdgeEntry *e = g->edge_buckets[b];
        while (e) {
            EdgeEntry *next = e->next;
            size_t idx = hash_u64(e->edge.edge_id, new_count);
            e->next = eb[idx];
            eb[idx] = e;
            e = next;
        }
    }
    gv_free(g->edge_buckets);
    g->edge_buckets = eb;
    g->edge_bucket_count = new_count;
    return 0;
}

static void free_node_internals(GV_GraphNode *node);
static void free_edge_internals(GV_GraphEdge *edge);
int adj_add(GV_GraphEdgeRef **arr, size_t *count, size_t *cap,
                   uint64_t edge_id, uint64_t neighbor_id);
static void adj_remove(GV_GraphEdgeRef *arr, size_t *count, uint64_t edge_id);
static int remove_edge_internal(GV_GraphDB *g, uint64_t edge_id);
static int graph_remove_node_internal(GV_GraphDB *g, uint64_t node_id);

/* ── Slab record pools (step 1 of the page-store migration) ─────────────
 * Nodes/edges live in fixed-size slabs instead of one heap allocation per
 * record: allocation/free is O(1) with immediate slot reuse, memory is
 * contiguous per slab (locality for bucket chains), and the per-object
 * allocator traffic disappears. Slab addresses are stable forever, so the
 * interior-pointer contract of graph_get_node/graph_get_edge is unchanged.
 * A `in_use` byte per record lets destroy free internals only for live
 * records. Step 2 (page IDs + mmap paging) can replace the slab storage
 * without touching any caller. */
#define GV_SLAB_SLOTS 512

typedef struct NodeSlab {
    struct NodeSlab *next;
    size_t used;
    NodeEntry slots[GV_SLAB_SLOTS];
} NodeSlab;

typedef struct EdgeSlab {
    struct EdgeSlab *next;
    size_t used;
    EdgeEntry slots[GV_SLAB_SLOTS];
} EdgeSlab;

/* Freelist links through the entry's own storage (records are far larger
 * than a pointer). */
NodeEntry *node_pool_alloc(GV_GraphDB *g)
{
    if (g->node_free) {
        NodeEntry *e = g->node_free;
        g->node_free = e->next_free;
        memset(e, 0, sizeof(*e));
        e->in_use = 1;
        return e;
    }
    NodeSlab *head = g->node_slabs;
    if (!head || head->used == GV_SLAB_SLOTS) {
        head = (NodeSlab *)gv_calloc(1, sizeof(NodeSlab));
        if (!head) return NULL;
        head->next = g->node_slabs;
        g->node_slabs = head;
    }
    NodeEntry *e = &head->slots[head->used++];
    memset(e, 0, sizeof(*e));
    e->in_use = 1;
    return e;
}

void node_pool_free(GV_GraphDB *g, NodeEntry *e)
{
    if (!e) return;
    free_node_internals(&e->node);
    e->in_use = 0;
    e->next_free = g->node_free;
    g->node_free = e;
}

EdgeEntry *edge_pool_alloc(GV_GraphDB *g)
{
    if (g->edge_free) {
        EdgeEntry *e = g->edge_free;
        g->edge_free = e->next_free;
        memset(e, 0, sizeof(*e));
        e->in_use = 1;
        return e;
    }
    EdgeSlab *head = g->edge_slabs;
    if (!head || head->used == GV_SLAB_SLOTS) {
        head = (EdgeSlab *)gv_calloc(1, sizeof(EdgeSlab));
        if (!head) return NULL;
        head->next = g->edge_slabs;
        g->edge_slabs = head;
    }
    EdgeEntry *e = &head->slots[head->used++];
    memset(e, 0, sizeof(*e));
    e->in_use = 1;
    return e;
}

void edge_pool_free(GV_GraphDB *g, EdgeEntry *e)
{
    if (!e) return;
    free_edge_internals(&e->edge);
    e->in_use = 0;
    e->next_free = g->edge_free;
    g->edge_free = e;
}

/* Insert a node/edge with an explicit id (WAL replay). Caller holds the write
 * lock. Returns 0 on success. */
static int insert_node_with_id(GV_GraphDB *g, uint64_t id, const char *label)
{
    if (!label) return -1;
    NodeEntry *entry = node_pool_alloc(g);
    if (!entry) return -1;
    entry->node.node_id = id;
    entry->node.label = gv_dup_cstr(label);
    if (!entry->node.label) {
        node_pool_free(g, entry);
        return -1;
    }
    size_t idx = hash_u64(id, g->node_bucket_count);
    entry->next = g->node_buckets[idx];
    g->node_buckets[idx] = entry;
    g->node_count++;
    if (id >= g->next_node_id) g->next_node_id = id + 1;
    if (label_index_add(g, entry->node.label, id) != 0) {
        label_index_remove(g, entry->node.label, id);
        g->node_count--;
        /* unlink from bucket */
        idx = hash_u64(id, g->node_bucket_count);
        NodeEntry *prev = NULL, *cur = g->node_buckets[idx];
        while (cur && cur->node.node_id != id) { prev = cur; cur = cur->next; }
        if (cur) {
            if (prev) prev->next = cur->next;
            else g->node_buckets[idx] = cur->next;
        }
        node_pool_free(g, entry);
        return -1;
    }
    grow_node_table(g);
    return 0;
}

static int insert_edge_with_id(GV_GraphDB *g, uint64_t id, uint64_t source,
                               uint64_t target, const char *label, float weight)
{
    NodeEntry *src_node = find_node_entry(g, source);
    NodeEntry *tgt_node = find_node_entry(g, target);
    if (!src_node || !tgt_node || !label) return -1;

    EdgeEntry *entry = edge_pool_alloc(g);
    if (!entry) return -1;
    entry->edge.edge_id = id;
    entry->edge.source_id = source;
    entry->edge.target_id = target;
    entry->edge.label = gv_dup_cstr(label);
    entry->edge.weight = weight;
    if (!entry->edge.label) {
        edge_pool_free(g, entry);
        return -1;
    }

    size_t idx = hash_u64(id, g->edge_bucket_count);
    entry->next = g->edge_buckets[idx];
    g->edge_buckets[idx] = entry;
    g->edge_count++;

    int out_ok = adj_add(&src_node->node.out_edges, &src_node->node.out_count,
                         &src_node->node.out_cap, id, target) == 0;
    int in_ok = out_ok &&
                adj_add(&tgt_node->node.in_edges, &tgt_node->node.in_count,
                        &tgt_node->node.in_cap, id, source) == 0;
    if (!in_ok) {
        /* Undo the out-edge if it was added (the old code decremented the
         * target's in_count even though the failed in-edge add never
         * incremented it, corrupting a legitimate in-edge and leaving a
         * dangling out-edge ref to the about-to-be-freed entry). */
        if (out_ok) adj_remove(src_node->node.out_edges, &src_node->node.out_count, id);
        g->edge_buckets[idx] = entry->next;
        g->edge_count--;
        edge_pool_free(g, entry);
        return -1;
    }
    if (id >= g->next_edge_id) g->next_edge_id = id + 1;
    grow_edge_table(g);
    return 0;
}

/* Replay one WAL record onto the graph. Called during graph_load with the
 * write lock held and wal_replaying set. Truncated/garbage tail records are
 * treated as end-of-log (a crash mid-append loses only that record). */
void gwal_apply_record(GV_GraphDB *g, uint8_t op,
                              const uint8_t *p, const uint8_t *end)
{
    char *s1 = NULL, *s2 = NULL;
    uint64_t id, a, b;
    float w;

    switch (op) {
    case GV_GWAL_OP_ADD_NODE:
        if ((p = gwal_get_u64(p, end, &id)) != NULL &&
            (p = gwal_get_str(p, end, &s1)) != NULL)
            insert_node_with_id(g, id, s1);
        break;
    case GV_GWAL_OP_REMOVE_NODE:
        if ((p = gwal_get_u64(p, end, &id)) != NULL)
            graph_remove_node_internal(g, id);   /* lock already held by replay */
        break;
    case GV_GWAL_OP_ADD_EDGE:
        if ((p = gwal_get_u64(p, end, &id)) != NULL &&
            (p = gwal_get_u64(p, end, &a)) != NULL &&
            (p = gwal_get_u64(p, end, &b)) != NULL &&
            (p = gwal_get_str(p, end, &s1)) != NULL &&
            (p = gwal_get_f32(p, end, &w)) != NULL)
            insert_edge_with_id(g, id, a, b, s1, w);
        break;
    case GV_GWAL_OP_REMOVE_EDGE:
        if ((p = gwal_get_u64(p, end, &id)) != NULL)
            remove_edge_internal(g, id);
        break;
    case GV_GWAL_OP_SET_NODE_PROP:
        if ((p = gwal_get_u64(p, end, &id)) != NULL &&
            (p = gwal_get_str(p, end, &s1)) != NULL &&
            (p = gwal_get_str(p, end, &s2)) != NULL) {
            NodeEntry *ne = find_node_entry(g, id);
            if (ne) set_prop(&ne->node.properties, &ne->node.prop_count, s1, s2);
        }
        break;
    case GV_GWAL_OP_SET_EDGE_PROP:
        if ((p = gwal_get_u64(p, end, &id)) != NULL &&
            (p = gwal_get_str(p, end, &s1)) != NULL &&
            (p = gwal_get_str(p, end, &s2)) != NULL) {
            EdgeEntry *ee = find_edge_entry(g, id);
            if (ee) set_prop(&ee->edge.properties, &ee->edge.prop_count, s1, s2);
        }
        break;
    default:
        break;  /* unknown op: ignore (forward compatibility) */
    }
    gv_free(s1);
    gv_free(s2);
}

int adj_add(GV_GraphEdgeRef **arr, size_t *count, size_t *cap,
                   uint64_t edge_id, uint64_t neighbor_id)
{
    if (*count >= *cap) {
        size_t new_cap = (*cap == 0) ? INITIAL_ADJ_CAP : (*cap * 2);
        GV_GraphEdgeRef *tmp = (GV_GraphEdgeRef *)gv_realloc(
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

static void free_labels(char **labels, size_t count)
{
    for (size_t i = 0; i < count; i++) gv_free(labels[i]);
    gv_free(labels);
}

static void free_node_internals(GV_GraphNode *node)
{
    gv_free(node->label);
    free_labels(node->labels, node->label_count);
    free_prop_list(node->properties);
    gv_free(node->out_edges);
    gv_free(node->in_edges);
}

static void free_edge_internals(GV_GraphEdge *edge)
{
    gv_free(edge->label);
    free_labels(edge->labels, edge->label_count);
    free_prop_list(edge->properties);
}

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

    NodeEntry *src = find_node_entry(g, e->edge.source_id);
    if (src) {
        adj_remove(src->node.out_edges, &src->node.out_count, edge_id);
    }

    NodeEntry *tgt = find_node_entry(g, e->edge.target_id);
    if (tgt) {
        adj_remove(tgt->node.in_edges, &tgt->node.in_count, edge_id);
    }

    if (prev) {
        prev->next = e->next;
    } else {
        g->edge_buckets[idx] = e->next;
    }

    edge_pool_free(g, e);
    g->edge_count--;
    return 0;
}


int visited_init(VisitedSet *vs, size_t capacity)
{
    if (capacity < 16) capacity = 16;
    vs->slots = (uint64_t *)gv_alloc(capacity * sizeof(uint64_t));
    vs->occupied = (int *)gv_calloc(capacity, sizeof(int));
    if (!vs->slots || !vs->occupied) {
        gv_free(vs->slots);
        gv_free(vs->occupied);
        return -1;
    }
    vs->capacity = capacity;
    vs->count = 0;
    return 0;
}

void visited_free(VisitedSet *vs)
{
    gv_free(vs->slots);
    gv_free(vs->occupied);
    vs->slots = NULL;
    vs->occupied = NULL;
}

int visited_contains(const VisitedSet *vs, uint64_t id)
{
    size_t idx = (size_t)(id * 2654435761ULL) % vs->capacity;
    for (size_t i = 0; i < vs->capacity; i++) {
        size_t pos = (idx + i) % vs->capacity;
        if (!vs->occupied[pos]) return 0;
        if (vs->slots[pos] == id) return 1;
    }
    return 0;
}

int visited_insert(VisitedSet *vs, uint64_t id)
{
    /* Rehash if load factor > 0.7 */
    if (vs->count * 10 > vs->capacity * 7) {
        size_t new_cap = vs->capacity * 2;
        uint64_t *new_slots = (uint64_t *)gv_alloc(new_cap * sizeof(uint64_t));
        int *new_occ = (int *)gv_calloc(new_cap, sizeof(int));
        if (!new_slots || !new_occ) {
            gv_free(new_slots);
            gv_free(new_occ);
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
        gv_free(vs->slots);
        gv_free(vs->occupied);
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


size_t idmap_lookup(const uint64_t *keys, const int *occ,
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

uint64_t *collect_all_node_ids(const GV_GraphDB *g, size_t *out_count)
{
    if (g->node_count == 0) {
        *out_count = 0;
        return NULL;
    }
    uint64_t *ids = (uint64_t *)gv_alloc(g->node_count * sizeof(uint64_t));
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


uint64_t graph_version(const GV_GraphDB *g)
{
    if (!g) return 0;
    graph_read_lock(g);
    uint64_t v = g->mutation_count;
    graph_read_unlock(g);
    return v;
}

int graph_edge_deltas_since(const GV_GraphDB *g, uint64_t since_version,
                            GV_GraphEdgeDelta *out, size_t cap,
                            size_t *out_len)
{
    if (!g || !out_len) return -1;
    *out_len = 0;
    graph_read_lock(g);
    if (g->journal_base_version > since_version) {
        graph_read_unlock(g);
        return -1;   /* coverage broken: full rebuild required */
    }
    size_t avail = 0;
    for (size_t i = 0; i < g->edge_journal_len; i++)
        if (g->edge_journal[i].version > since_version) avail++;
    *out_len = avail;
    if (out) {
        size_t copied = 0;
        for (size_t i = 0; i < g->edge_journal_len && copied < cap; i++) {
            if (g->edge_journal[i].version > since_version)
                out[copied++] = g->edge_journal[i];
        }
    }
    graph_read_unlock(g);
    return 0;
}

GV_GraphReadTxn *graph_read_txn_begin(GV_GraphDB *g)
{
    if (!g) return NULL;
    GV_GraphReadTxn *txn =
        (GV_GraphReadTxn *)gv_calloc(1, sizeof(GV_GraphReadTxn));
    if (!txn) return NULL;
    pthread_rwlock_rdlock(&g->rwlock);
    txn->g = g;
    txn->version_at_begin = g->mutation_count;
    return txn;
}

void graph_read_txn_end(GV_GraphReadTxn *txn)
{
    if (!txn) return;
    if (txn->g) pthread_rwlock_unlock(&txn->g->rwlock);
    gv_free(txn);
}

uint64_t graph_read_txn_version(const GV_GraphReadTxn *txn)
{
    return txn ? txn->version_at_begin : 0;
}

const GV_GraphNode *graph_read_txn_get_node(GV_GraphReadTxn *txn,
                                            uint64_t node_id)
{
    if (!txn || !txn->g) return NULL;
    NodeEntry *e = find_node_entry(txn->g, node_id);
    return e ? &e->node : NULL;
}

const char *graph_read_txn_get_node_prop(GV_GraphReadTxn *txn,
                                         uint64_t node_id, const char *key)
{
    if (!txn || !txn->g || !key) return NULL;
    NodeEntry *e = find_node_entry(txn->g, node_id);
    if (!e) return NULL;
    GV_GraphProp *p = find_prop(e->node.properties, key);
    return prop_borrow_string(p);
}

const GV_GraphEdge *graph_read_txn_get_edge(GV_GraphReadTxn *txn,
                                            uint64_t edge_id)
{
    if (!txn || !txn->g) return NULL;
    EdgeEntry *e = find_edge_entry(txn->g, edge_id);
    return e ? &e->edge : NULL;
}

int graph_read_txn_find_nodes_by_label(GV_GraphReadTxn *txn, const char *label,
                                       uint64_t *out_ids, size_t max_count)
{
    if (!txn || !txn->g || !label || !out_ids) return -1;
    LabelEntry *le = find_label_entry(txn->g, label);
    if (!le) return 0;
    int count = 0;
    for (size_t i = 0; i < le->count; i++) {
        if ((size_t)count < max_count) out_ids[count] = le->ids[i];
        count++;
    }
    return (count > (int)max_count) ? (int)max_count : count;
}


/* ── Write transactions: staged mutations, atomic commit ──────────────── */

struct GV_GraphWriteTxn {
    GV_GraphDB *g;
    uint8_t *buf;          /* concatenated [op][len][payload] frames */
    size_t   len;
    size_t   cap;
    uint64_t *staged_nodes;/* ids of nodes created inside this txn */
    size_t   n_staged;
    size_t   cap_staged;
    int      finished;
};

static int gtxn_has_staged_node(const GV_GraphWriteTxn *t, uint64_t id)
{
    for (size_t i = 0; i < t->n_staged; i++)
        if (t->staged_nodes[i] == id) return 1;
    return 0;
}

static int gtxn_stage(GV_GraphWriteTxn *t, const GWalBuf *b)
{
    if (!t || b->overflow) return -1;
    size_t need = t->len + 5 + (b->len - 1);
    if (need > t->cap) {
        size_t nc = t->cap ? t->cap : 256;
        while (nc < need) nc *= 2;
        uint8_t *nb = (uint8_t *)gv_realloc(t->buf, nc);
        if (!nb) return -1;
        t->buf = nb;
        t->cap = nc;
    }
    t->buf[t->len++] = b->buf[0];
    uint32_t plen = (uint32_t)(b->len - 1);
    for (int i = 0; i < 4; i++) t->buf[t->len++] = (uint8_t)(plen >> (8 * i));
    if (b->len > 1) {
        memcpy(t->buf + t->len, b->buf + 1, b->len - 1);
        t->len += b->len - 1;
    }
    return 0;
}

GV_GraphWriteTxn *graph_write_txn_begin(GV_GraphDB *g)
{
    if (!g) return NULL;
    GV_GraphWriteTxn *t =
        (GV_GraphWriteTxn *)gv_calloc(1, sizeof(GV_GraphWriteTxn));
    if (!t) return NULL;
    pthread_rwlock_wrlock(&g->rwlock);   /* held for the txn's lifetime */
    t->g = g;
    return t;
}

static void gtxn_finish(GV_GraphWriteTxn *t)
{
    if (!t) return;
    if (t->g) pthread_rwlock_unlock(&t->g->rwlock);
    gv_free(t->buf);
    gv_free(t->staged_nodes);
    gv_free(t);
}

uint64_t graph_write_txn_add_node(GV_GraphWriteTxn *t, const char *label)
{
    if (!t || !t->g || t->finished || !label) return 0;
    GV_GraphDB *g = t->g;
    char *lbl = gv_dup_cstr(label);
    if (!lbl) return 0;
    uint64_t id = g->next_node_id++;
    GWalBuf b;
    gwal_init(&b);
    gwal_put_u8(&b, GV_GWAL_OP_ADD_NODE);
    gwal_put_u64(&b, id);
    gwal_put_str(&b, lbl);
    int rc = gtxn_stage(t, &b);
    if (rc == 0) {
        if (t->n_staged >= t->cap_staged) {
            size_t nc = t->cap_staged ? t->cap_staged * 2 : 8;
            uint64_t *ns = (uint64_t *)gv_realloc((void *)t->staged_nodes,
                                                  nc * sizeof(uint64_t));
            if (!ns) { gv_free(lbl); return 0; }
            t->staged_nodes = ns;
            t->cap_staged = nc;
        }
        t->staged_nodes[t->n_staged++] = id;
    }
    gv_free(lbl);
    return rc == 0 ? id : 0;
}

int graph_write_txn_remove_node(GV_GraphWriteTxn *t, uint64_t node_id)
{
    if (!t || !t->g || t->finished) return -1;
    /* Validate against committed state or nodes staged earlier in this txn;
     * staged removes of missing nodes are no-ops at apply time. */
    if (!find_node_entry(t->g, node_id) && !gtxn_has_staged_node(t, node_id))
        return -1;
    GWalBuf b;
    gwal_init(&b);
    gwal_put_u8(&b, GV_GWAL_OP_REMOVE_NODE);
    gwal_put_u64(&b, node_id);
    return gtxn_stage(t, &b);
}

uint64_t graph_write_txn_add_edge(GV_GraphWriteTxn *t, uint64_t source,
                                  uint64_t target, const char *label,
                                  float weight)
{
    if (!t || !t->g || t->finished || !label) return 0;
    GV_GraphDB *g = t->g;
    /* Endpoints may be nodes staged earlier in this transaction. */
    if ((!find_node_entry(g, source) && !gtxn_has_staged_node(t, source)) ||
        (!find_node_entry(g, target) && !gtxn_has_staged_node(t, target)))
        return 0;
    char *lbl = gv_dup_cstr(label);
    if (!lbl) return 0;
    uint64_t id = g->next_edge_id++;
    GWalBuf b;
    gwal_init(&b);
    gwal_put_u8(&b, GV_GWAL_OP_ADD_EDGE);
    gwal_put_u64(&b, id);
    gwal_put_u64(&b, source);
    gwal_put_u64(&b, target);
    gwal_put_str(&b, lbl);
    gwal_put_f32(&b, weight);
    int rc = gtxn_stage(t, &b);
    gv_free(lbl);
    return rc == 0 ? id : 0;
}

int graph_write_txn_remove_edge(GV_GraphWriteTxn *t, uint64_t edge_id)
{
    if (!t || !t->g || t->finished) return -1;
    /* No existence check: the edge may have been staged earlier in this
     * transaction; removing a missing edge applies as a no-op anyway. */
    GWalBuf b;
    gwal_init(&b);
    gwal_put_u8(&b, GV_GWAL_OP_REMOVE_EDGE);
    gwal_put_u64(&b, edge_id);
    return gtxn_stage(t, &b);
}

int graph_write_txn_set_node_prop(GV_GraphWriteTxn *t, uint64_t node_id,
                                  const char *key, const char *value)
{
    if (!t || !t->g || t->finished || !key || !value) return -1;
    if (!find_node_entry(t->g, node_id) && !gtxn_has_staged_node(t, node_id))
        return -1;
    GWalBuf b;
    gwal_init(&b);
    gwal_put_u8(&b, GV_GWAL_OP_SET_NODE_PROP);
    gwal_put_u64(&b, node_id);
    gwal_put_str(&b, key);
    gwal_put_str(&b, value);
    return gtxn_stage(t, &b);
}

int graph_write_txn_set_edge_prop(GV_GraphWriteTxn *t, uint64_t edge_id,
                                  const char *key, const char *value)
{
    if (!t || !t->g || t->finished || !key || !value) return -1;
    if (!find_edge_entry(t->g, edge_id)) return -1;
    GWalBuf b;
    gwal_init(&b);
    gwal_put_u8(&b, GV_GWAL_OP_SET_EDGE_PROP);
    gwal_put_u64(&b, edge_id);
    gwal_put_str(&b, key);
    gwal_put_str(&b, value);
    return gtxn_stage(t, &b);
}

int graph_write_txn_commit(GV_GraphWriteTxn *t)
{
    if (!t || t->finished) { gtxn_finish(t); return -1; }
    GV_GraphDB *g = t->g;

    /* 1. Write-ahead: one batched append + a single fsync for the whole
     *    transaction (group commit). */
    if (g->wal_file && t->len > 0) {
        if (fwrite(t->buf, 1, t->len, g->wal_file) != t->len ||
            fflush(g->wal_file) != 0 || fsync(fileno(g->wal_file)) != 0) {
            t->finished = 1;
            gtxn_finish(t);
            return -1;
        }
    }

    /* 2. Apply in order under the lock we already hold; suppress per-op
     *    logging since the batch was just written ahead. */
    g->wal_replaying = 1;
    size_t off = 0;
    while (off < t->len) {
        size_t used = gwal_parse_record(g, t->buf + off, t->len - off);
        if (used == 0) break;   /* cannot happen with well-formed staging */
        off += used;
    }
    g->wal_replaying = 0;
    g->mutation_count++;
    journal_break_locked(g);   /* batched replay: journal can't track it */

    t->finished = 1;
    gtxn_finish(t);
    return 0;
}

void graph_write_txn_abort(GV_GraphWriteTxn *t)
{
    if (!t || t->finished) { gtxn_finish(t); return; }
    /* IDs consumed by staged adds leave gaps - same semantics as failed
     * non-transactional inserts. Nothing else to undo: nothing was applied. */
    t->finished = 1;
    gtxn_finish(t);
}

/* Per-thread recursion counting for the read lock.
 *
 * The pinned-snapshot analytics layer (gv_ga_build) holds the read lock for the
 * duration of a computation, and some algorithms nest another gv_ga_build (or a
 * locking accessor like graph_node_count) inside that span. A plain recursive
 * pthread_rwlock_rdlock() deadlocks on a writer-preferring implementation
 * (notably macOS/ARM): the second rdlock blocks behind a queued writer that is
 * itself waiting for this thread's first rdlock. Counting recursion per thread
 * so only the outermost acquire touches the OS lock avoids that entirely, and
 * keeps concurrent readers on other threads working as before.
 *
 * Single-graph recursion is tracked (the common case); a thread that nests read
 * locks across two different graphs falls back to a direct acquire. */
static _Thread_local const GV_GraphDB *gv_rd_held_g = NULL;
static _Thread_local unsigned         gv_rd_depth   = 0;

void graph_read_lock(const GV_GraphDB *g)
{
    if (!g) return;
    if (gv_rd_held_g == g) { gv_rd_depth++; return; }
    pthread_rwlock_rdlock((pthread_rwlock_t *)&g->rwlock);
    if (gv_rd_held_g == NULL) { gv_rd_held_g = g; gv_rd_depth = 1; }
}

void graph_read_unlock(const GV_GraphDB *g)
{
    if (!g) return;
    if (gv_rd_held_g == g) {
        if (--gv_rd_depth > 0) return;
        gv_rd_held_g = NULL;
    }
    pthread_rwlock_unlock((pthread_rwlock_t *)&g->rwlock);
}

/* Unlocked accessors for callers that already hold the read lock - the
 * pinned-snapshot analytics layer (GV_GAContext) uses these so a whole
 * algorithm run sees one consistent snapshot without recursive locking. */
const GV_GraphNode *graph_get_node_unlocked(const GV_GraphDB *g, uint64_t node_id)
{
    if (!g) return NULL;
    NodeEntry *e = find_node_entry(g, node_id);
    return e ? &e->node : NULL;
}

const GV_GraphEdge *graph_get_edge_unlocked(const GV_GraphDB *g, uint64_t edge_id)
{
    if (!g) return NULL;
    EdgeEntry *e = find_edge_entry(g, edge_id);
    return e ? &e->edge : NULL;
}

void graph_config_init(GV_GraphDBConfig *config)
{
    if (!config) return;
    config->node_bucket_count = DEFAULT_NODE_BUCKETS;
    config->edge_bucket_count = DEFAULT_EDGE_BUCKETS;
    config->enforce_referential_integrity = 1;
}

GV_GraphDB *graph_create(const GV_GraphDBConfig *config)
{
    GV_GraphDBConfig cfg;
    if (config) {
        cfg = *config;
    } else {
        graph_config_init(&cfg);
    }
    if (cfg.node_bucket_count == 0) cfg.node_bucket_count = DEFAULT_NODE_BUCKETS;
    if (cfg.edge_bucket_count == 0) cfg.edge_bucket_count = DEFAULT_EDGE_BUCKETS;

    GV_GraphDB *g = (GV_GraphDB *)gv_calloc(1, sizeof(GV_GraphDB));
    if (!g) return NULL;

    g->node_bucket_count = cfg.node_bucket_count;
    g->edge_bucket_count = cfg.edge_bucket_count;
    g->enforce_referential_integrity = cfg.enforce_referential_integrity;
    g->next_node_id = 1;
    g->next_edge_id = 1;

    g->node_buckets = (NodeEntry **)gv_calloc(g->node_bucket_count,
                                            sizeof(NodeEntry *));
    g->edge_buckets = (EdgeEntry **)gv_calloc(g->edge_bucket_count,
                                           sizeof(EdgeEntry *));
    g->label_bucket_count = g->node_bucket_count;
    g->label_buckets = (LabelEntry **)gv_calloc(g->label_bucket_count,
                                             sizeof(LabelEntry *));
    if (!g->node_buckets || !g->edge_buckets || !g->label_buckets) {
        gv_free(g->node_buckets);
        gv_free(g->edge_buckets);
        gv_free(g->label_buckets);
        gv_free(g);
        return NULL;
    }

    if (pthread_rwlock_init(&g->rwlock, NULL) != 0) {
        gv_free(g->node_buckets);
        gv_free(g->edge_buckets);
        gv_free(g->label_buckets);
        gv_free(g);
        return NULL;
    }

    return g;
}

void graph_destroy(GV_GraphDB *g)
{
    if (!g) return;

    for (struct NodeSlab *ns = g->node_slabs; ns; ) {
        struct NodeSlab *next = ns->next;
        for (size_t i = 0; i < ns->used; i++)
            if (ns->slots[i].in_use) free_node_internals(&ns->slots[i].node);
        gv_free(ns);
        ns = next;
    }
    gv_free(g->node_buckets);

    for (struct EdgeSlab *es = g->edge_slabs; es; ) {
        struct EdgeSlab *next = es->next;
        for (size_t i = 0; i < es->used; i++)
            if (es->slots[i].in_use) free_edge_internals(&es->slots[i].edge);
        gv_free(es);
        es = next;
    }
    gv_free(g->edge_buckets);

    gv_free(g->edge_journal);
    g->edge_journal = NULL;

    for (size_t b = 0; b < g->label_bucket_count; b++) {
        LabelEntry *e = g->label_buckets[b];
        while (e) {
            LabelEntry *next = e->next;
            gv_free(e->label);
            gv_free(e->ids);
            gv_free(e);
            e = next;
        }
    }
    gv_free(g->label_buckets);

    graph_wal_close(g);

    pthread_rwlock_destroy(&g->rwlock);
    gv_free(g);
}

uint64_t graph_add_node(GV_GraphDB *g, const char *label)
{
    if (!g || !label) return 0;

    char *lbl = gv_dup_cstr(label);
    if (!lbl) return 0;

    pthread_rwlock_wrlock(&g->rwlock);

    /* Pool state is shared between writers: allocate under the write lock. */
    NodeEntry *entry = node_pool_alloc(g);
    if (!entry) {
        pthread_rwlock_unlock(&g->rwlock);
        gv_free(lbl);
        return 0;
    }

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

    size_t idx = hash_u64(id, g->node_bucket_count);
    entry->next = g->node_buckets[idx];
    g->node_buckets[idx] = entry;
    g->node_count++;

    int rc = label_index_add(g, lbl, id);
    grow_node_table(g);

    if (rc == 0) {
        GWalBuf b;
        gwal_init(&b);
        gwal_put_u8(&b, GV_GWAL_OP_ADD_NODE);
        gwal_put_u64(&b, id);
        gwal_put_str(&b, lbl);
        gwal_append(g, &b);
    g->mutation_count++;
    journal_break_locked(g);   /* structural: dense indices shift */
    }

    pthread_rwlock_unlock(&g->rwlock);
    return rc == 0 ? id : 0;
}

uint64_t graph_add_node_with_id(GV_GraphDB *g, uint64_t node_id, const char *label)
{
    if (!g || !label || node_id == 0) return 0;

    pthread_rwlock_wrlock(&g->rwlock);
    if (find_node_entry(g, node_id)) {          /* id already taken */
        pthread_rwlock_unlock(&g->rwlock);
        return 0;
    }
    if (insert_node_with_id(g, node_id, label) != 0) {
        pthread_rwlock_unlock(&g->rwlock);
        return 0;
    }

    GWalBuf b;
    gwal_init(&b);
    gwal_put_u8(&b, GV_GWAL_OP_ADD_NODE);
    gwal_put_u64(&b, node_id);
    gwal_put_str(&b, label);
    gwal_append(g, &b);
    g->mutation_count++;
    journal_break_locked(g);

    pthread_rwlock_unlock(&g->rwlock);
    return node_id;
}

static int graph_remove_node_internal(GV_GraphDB *g, uint64_t node_id)
{
    NodeEntry *ne = find_node_entry(g, node_id);
    if (!ne) {
        return -1;
    }

    while (ne->node.out_count > 0) {
        uint64_t eid = ne->node.out_edges[0].edge_id;
        remove_edge_internal(g, eid);
    }

    while (ne->node.in_count > 0) {
        uint64_t eid = ne->node.in_edges[0].edge_id;
        remove_edge_internal(g, eid);
    }

    size_t idx = hash_u64(node_id, g->node_bucket_count);
    NodeEntry *prev = NULL;
    NodeEntry *cur = g->node_buckets[idx];
    while (cur) {
        if (cur->node.node_id == node_id) break;
        prev = cur;
        cur = cur->next;
    }
    if (cur) {
        label_index_remove(g, cur->node.label, node_id);
        if (prev) {
            prev->next = cur->next;
        } else {
            g->node_buckets[idx] = cur->next;
        }
        node_pool_free(g, cur);
        g->node_count--;

        GWalBuf b;
        gwal_init(&b);
        gwal_put_u8(&b, GV_GWAL_OP_REMOVE_NODE);
        gwal_put_u64(&b, node_id);
        gwal_append(g, &b);
    g->mutation_count++;
    }

    return 0;
}

int graph_remove_node(GV_GraphDB *g, uint64_t node_id)
{
    if (!g) return -1;

    pthread_rwlock_wrlock(&g->rwlock);
    int rc = graph_remove_node_internal(g, node_id);
    if (rc == 0) {
        g->mutation_count++;
        journal_break_locked(g);   /* structural: dense indices shift */
    }
    pthread_rwlock_unlock(&g->rwlock);
    return rc;
}

const GV_GraphNode *graph_get_node(const GV_GraphDB *g, uint64_t node_id)
{
    if (!g) return NULL;

    graph_read_lock(g);
    NodeEntry *e = find_node_entry(g, node_id);
    const GV_GraphNode *result = e ? &e->node : NULL;
    graph_read_unlock(g);
    return result;
}

int graph_set_node_prop(GV_GraphDB *g, uint64_t node_id,
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
    if (rc == 0) {
        GWalBuf b;
        gwal_init(&b);
        gwal_put_u8(&b, GV_GWAL_OP_SET_NODE_PROP);
        gwal_put_u64(&b, node_id);
        gwal_put_str(&b, key);
        gwal_put_str(&b, value);
        gwal_append(g, &b);
    g->mutation_count++;
    }
    pthread_rwlock_unlock(&g->rwlock);
    return rc;
}

const char *graph_get_node_prop(const GV_GraphDB *g, uint64_t node_id,
                                   const char *key)
{
    if (!g || !key) return NULL;

    graph_read_lock(g);
    NodeEntry *e = find_node_entry(g, node_id);
    const char *result = NULL;
    if (e) {
        GV_GraphProp *p = find_prop(e->node.properties, key);
        if (p) result = prop_borrow_string(p);
    }
    graph_read_unlock(g);
    return result;
}

int graph_find_nodes_by_label(const GV_GraphDB *g, const char *label,
                                 uint64_t *out_ids, size_t max_count)
{
    if (!g || !label || !out_ids) return -1;

    graph_read_lock(g);

    int count = 0;
    LabelEntry *le = find_label_entry(g, label);
    if (le) {
        for (size_t i = 0; i < le->count; i++) {
            if ((size_t)count < max_count) {
                out_ids[count] = le->ids[i];
            }
            count++;
        }
    }

    graph_read_unlock(g);
    return (count > (int)max_count) ? (int)max_count : count;
}

uint64_t graph_add_edge(GV_GraphDB *g, uint64_t source, uint64_t target,
                           const char *label, float weight)
{
    if (!g || !label) return 0;

    char *lbl = gv_dup_cstr(label);
    if (!lbl) return 0;

    pthread_rwlock_wrlock(&g->rwlock);

    /* Pool state is shared between writers: allocate under the write lock. */
    EdgeEntry *entry = edge_pool_alloc(g);
    if (!entry) {
        pthread_rwlock_unlock(&g->rwlock);
        gv_free(lbl);
        return 0;
    }

    /* Referential integrity check */
    if (g->enforce_referential_integrity) {
        NodeEntry *src = find_node_entry(g, source);
        NodeEntry *tgt = find_node_entry(g, target);
        if (!src || !tgt) {
            pthread_rwlock_unlock(&g->rwlock);
            gv_free(lbl);
            edge_pool_free(g, entry);
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

    size_t idx = hash_u64(id, g->edge_bucket_count);
    entry->next = g->edge_buckets[idx];
    g->edge_buckets[idx] = entry;
    g->edge_count++;

    NodeEntry *src_node = find_node_entry(g, source);
    NodeEntry *tgt_node = find_node_entry(g, target);
    int adj_ok = 1;
    if (src_node &&
        adj_add(&src_node->node.out_edges, &src_node->node.out_count,
                &src_node->node.out_cap, id, target) != 0) {
        adj_ok = 0;
    }
    if (adj_ok && tgt_node &&
        adj_add(&tgt_node->node.in_edges, &tgt_node->node.in_count,
                &tgt_node->node.in_cap, id, source) != 0) {
        adj_ok = 0;
        /* Undo the out-edge just appended (adj_add appends at the tail). */
        if (src_node && src_node->node.out_count > 0) src_node->node.out_count--;
    }
    if (!adj_ok) {
        /* Adjacency insert failed (OOM). Roll back the edge-table entry so we
         * never leave a phantom edge that exists in the edge table but is
         * unreachable via traversal (and disagrees with edge_count). `entry` was
         * prepended as the head of bucket `idx`. */
        g->edge_buckets[idx] = entry->next;
        g->edge_count--;
        edge_pool_free(g, entry);
        pthread_rwlock_unlock(&g->rwlock);
        gv_free(lbl);
        return 0;  /* 0 = invalid edge id (the consumed id is simply skipped) */
    }

    grow_edge_table(g);

    GWalBuf b;
    gwal_init(&b);
    gwal_put_u8(&b, GV_GWAL_OP_ADD_EDGE);
    gwal_put_u64(&b, id);
    gwal_put_u64(&b, source);
    gwal_put_u64(&b, target);
    gwal_put_str(&b, label);
    gwal_put_f32(&b, weight);
    gwal_append(g, &b);
    g->mutation_count++;
    journal_edge_locked(g, source, target, 0);

    pthread_rwlock_unlock(&g->rwlock);
    return id;
}

int graph_remove_edge(GV_GraphDB *g, uint64_t edge_id)
{
    if (!g) return -1;

    pthread_rwlock_wrlock(&g->rwlock);
    /* Capture endpoints before removal for the CSR change journal. */
    EdgeEntry *je = find_edge_entry(g, edge_id);
    uint64_t jsrc = je ? je->edge.source_id : 0;
    uint64_t jdst = je ? je->edge.target_id : 0;
    int rc = remove_edge_internal(g, edge_id);
    if (rc == 0) {
        GWalBuf b;
        gwal_init(&b);
        gwal_put_u8(&b, GV_GWAL_OP_REMOVE_EDGE);
        gwal_put_u64(&b, edge_id);
        gwal_append(g, &b);
    g->mutation_count++;
    journal_edge_locked(g, jsrc, jdst, 1);
    }
    pthread_rwlock_unlock(&g->rwlock);
    return rc;
}

const GV_GraphEdge *graph_get_edge(const GV_GraphDB *g, uint64_t edge_id)
{
    if (!g) return NULL;

    graph_read_lock(g);
    EdgeEntry *e = find_edge_entry(g, edge_id);
    const GV_GraphEdge *result = e ? &e->edge : NULL;
    graph_read_unlock(g);
    return result;
}

int graph_set_edge_prop(GV_GraphDB *g, uint64_t edge_id,
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
    if (rc == 0) {
        GWalBuf b;
        gwal_init(&b);
        gwal_put_u8(&b, GV_GWAL_OP_SET_EDGE_PROP);
        gwal_put_u64(&b, edge_id);
        gwal_put_str(&b, key);
        gwal_put_str(&b, value);
        gwal_append(g, &b);
    g->mutation_count++;
    }
    pthread_rwlock_unlock(&g->rwlock);
    return rc;
}

const char *graph_get_edge_prop(const GV_GraphDB *g, uint64_t edge_id,
                                   const char *key)
{
    if (!g || !key) return NULL;

    graph_read_lock(g);
    EdgeEntry *e = find_edge_entry(g, edge_id);
    const char *result = NULL;
    if (e) {
        GV_GraphProp *p = find_prop(e->edge.properties, key);
        if (p) result = prop_borrow_string(p);
    }
    graph_read_unlock(g);
    return result;
}

/* ── Multi-label support ─────────────────────────────────────────────── */

/** @brief Append a label (no duplicates). @return 0 added, 1 already present, -1 on OOM. */
static int labels_add(char ***labels, size_t *count, size_t *cap,
                      const char *label)
{
    for (size_t i = 0; i < *count; i++) {
        if (strcmp((*labels)[i], label) == 0) return 1;
    }
    if (*count == *cap) {
        size_t newcap = *cap ? *cap * 2 : 4;
        char **grown = (char **)gv_realloc(*labels, newcap * sizeof(char *));
        if (!grown) return -1;
        *labels = grown;
        *cap = newcap;
    }
    char *dup = gv_dup_cstr(label);
    if (!dup) return -1;
    (*labels)[(*count)++] = dup;
    return 0;
}

/**
 * @brief Sync the legacy single-label field to labels[0] as an independent copy.
 *
 * Kept unaliased on purpose so freeing @p label never touches the labels array.
 */
static void sync_legacy_label(char **label, char **labels, size_t count)
{
    gv_free(*label);
    *label = count ? gv_dup_cstr(labels[0]) : NULL;
}

/** @brief Remove a label. @return 0 on success, -1 if absent. */
static int labels_remove(char **labels, size_t *count, const char *label)
{
    for (size_t i = 0; i < *count; i++) {
        if (strcmp(labels[i], label) == 0) {
            gv_free(labels[i]);
            labels[i] = labels[*count - 1];
            (*count)--;
            return 0;
        }
    }
    return -1;
}

int graph_node_add_label(GV_GraphDB *g, uint64_t node_id, const char *label)
{
    if (!g || !label) return -1;
    pthread_rwlock_wrlock(&g->rwlock);
    NodeEntry *e = find_node_entry(g, node_id);
    if (!e) { pthread_rwlock_unlock(&g->rwlock); return -1; }
    int rc = labels_add(&e->node.labels, &e->node.label_count,
                        &e->node.label_cap, label);
    if (rc == 0) {
        sync_legacy_label(&e->node.label, e->node.labels, e->node.label_count);
        label_index_add(g, label, node_id);
        g->mutation_count++;
    }
    pthread_rwlock_unlock(&g->rwlock);
    return rc == 0 ? 0 : -1;
}

int graph_node_remove_label(GV_GraphDB *g, uint64_t node_id, const char *label)
{
    if (!g || !label) return -1;
    pthread_rwlock_wrlock(&g->rwlock);
    NodeEntry *e = find_node_entry(g, node_id);
    if (!e) { pthread_rwlock_unlock(&g->rwlock); return -1; }
    int rc = labels_remove(e->node.labels, &e->node.label_count, label);
    if (rc == 0) {
        label_index_remove(g, label, node_id);
        sync_legacy_label(&e->node.label, e->node.labels, e->node.label_count);
        g->mutation_count++;
    }
    pthread_rwlock_unlock(&g->rwlock);
    return rc;
}

int graph_node_has_label(const GV_GraphNode *node, const char *label)
{
    if (!node || !label) return 0;
    for (size_t i = 0; i < node->label_count; i++) {
        if (strcmp(node->labels[i], label) == 0) return 1;
    }
    return node->label && strcmp(node->label, label) == 0 ? 1 : 0;
}

int graph_edge_add_label(GV_GraphDB *g, uint64_t edge_id, const char *label)
{
    if (!g || !label) return -1;
    pthread_rwlock_wrlock(&g->rwlock);
    EdgeEntry *e = find_edge_entry(g, edge_id);
    if (!e) { pthread_rwlock_unlock(&g->rwlock); return -1; }
    int rc = labels_add(&e->edge.labels, &e->edge.label_count,
                        &e->edge.label_cap, label);
    if (rc == 0) {
        sync_legacy_label(&e->edge.label, e->edge.labels, e->edge.label_count);
        g->mutation_count++;
    }
    pthread_rwlock_unlock(&g->rwlock);
    return rc == 0 ? 0 : -1;
}

int graph_edge_remove_label(GV_GraphDB *g, uint64_t edge_id, const char *label)
{
    if (!g || !label) return -1;
    pthread_rwlock_wrlock(&g->rwlock);
    EdgeEntry *e = find_edge_entry(g, edge_id);
    if (!e) { pthread_rwlock_unlock(&g->rwlock); return -1; }
    int rc = labels_remove(e->edge.labels, &e->edge.label_count, label);
    if (rc == 0) {
        sync_legacy_label(&e->edge.label, e->edge.labels, e->edge.label_count);
        g->mutation_count++;
    }
    pthread_rwlock_unlock(&g->rwlock);
    return rc;
}

/* ── Typed property setters/getters ──────────────────────────────────── */

static int node_set_prop_typed(GV_GraphDB *g, uint64_t node_id,
                               const char *key, GV_PropValue v)
{
    if (!g || !key) { gv_prop_free(&v); return -1; }
    pthread_rwlock_wrlock(&g->rwlock);
    NodeEntry *e = find_node_entry(g, node_id);
    if (!e) { pthread_rwlock_unlock(&g->rwlock); gv_prop_free(&v); return -1; }
    int rc = set_prop_typed(&e->node.properties, &e->node.prop_count, key, v);
    if (rc == 0) g->mutation_count++;
    pthread_rwlock_unlock(&g->rwlock);
    return rc;
}

static int edge_set_prop_typed(GV_GraphDB *g, uint64_t edge_id,
                               const char *key, GV_PropValue v)
{
    if (!g || !key) { gv_prop_free(&v); return -1; }
    pthread_rwlock_wrlock(&g->rwlock);
    EdgeEntry *e = find_edge_entry(g, edge_id);
    if (!e) { pthread_rwlock_unlock(&g->rwlock); gv_prop_free(&v); return -1; }
    int rc = set_prop_typed(&e->edge.properties, &e->edge.prop_count, key, v);
    if (rc == 0) g->mutation_count++;
    pthread_rwlock_unlock(&g->rwlock);
    return rc;
}

int graph_set_node_prop_int64(GV_GraphDB *g, uint64_t node_id,
                               const char *key, int64_t value)
{ return node_set_prop_typed(g, node_id, key, gv_prop_int64(value)); }

int graph_set_node_prop_float64(GV_GraphDB *g, uint64_t node_id,
                                 const char *key, double value)
{ return node_set_prop_typed(g, node_id, key, gv_prop_float64(value)); }

int graph_set_node_prop_bool(GV_GraphDB *g, uint64_t node_id,
                              const char *key, int value)
{ return node_set_prop_typed(g, node_id, key, gv_prop_bool(value)); }

int graph_set_edge_prop_int64(GV_GraphDB *g, uint64_t edge_id,
                               const char *key, int64_t value)
{ return edge_set_prop_typed(g, edge_id, key, gv_prop_int64(value)); }

int graph_set_edge_prop_float64(GV_GraphDB *g, uint64_t edge_id,
                                 const char *key, double value)
{ return edge_set_prop_typed(g, edge_id, key, gv_prop_float64(value)); }

int graph_set_edge_prop_bool(GV_GraphDB *g, uint64_t edge_id,
                              const char *key, int value)
{ return edge_set_prop_typed(g, edge_id, key, gv_prop_bool(value)); }

static GV_PropValue node_get_prop(const GV_GraphDB *g, uint64_t node_id,
                                  const char *key)
{
    GV_PropValue r = gv_prop_null();
    if (!g || !key) return r;
    graph_read_lock(g);
    NodeEntry *e = find_node_entry(g, node_id);
    if (e) {
        GV_GraphProp *p = find_prop(e->node.properties, key);
        if (p) r = p->value;
    }
    graph_read_unlock(g);
    return r;
}

static GV_PropValue edge_get_prop(const GV_GraphDB *g, uint64_t edge_id,
                                  const char *key)
{
    GV_PropValue r = gv_prop_null();
    if (!g || !key) return r;
    graph_read_lock(g);
    EdgeEntry *e = find_edge_entry(g, edge_id);
    if (e) {
        GV_GraphProp *p = find_prop(e->edge.properties, key);
        if (p) r = p->value;
    }
    graph_read_unlock(g);
    return r;
}

int64_t graph_get_node_prop_int64(const GV_GraphDB *g, uint64_t node_id,
                                   const char *key, int64_t fallback)
{
    GV_PropValue v = node_get_prop(g, node_id, key);
    return v.type == GV_PROP_INT64 ? v.as.i : fallback;
}

double graph_get_node_prop_float64(const GV_GraphDB *g, uint64_t node_id,
                                    const char *key, double fallback)
{
    GV_PropValue v = node_get_prop(g, node_id, key);
    return v.type == GV_PROP_FLOAT64 ? v.as.f : fallback;
}

int graph_get_node_prop_bool(const GV_GraphDB *g, uint64_t node_id,
                              const char *key, int fallback)
{
    GV_PropValue v = node_get_prop(g, node_id, key);
    return v.type == GV_PROP_BOOL ? v.as.b : fallback;
}

int64_t graph_get_edge_prop_int64(const GV_GraphDB *g, uint64_t edge_id,
                                   const char *key, int64_t fallback)
{
    GV_PropValue v = edge_get_prop(g, edge_id, key);
    return v.type == GV_PROP_INT64 ? v.as.i : fallback;
}

double graph_get_edge_prop_float64(const GV_GraphDB *g, uint64_t edge_id,
                                    const char *key, double fallback)
{
    GV_PropValue v = edge_get_prop(g, edge_id, key);
    return v.type == GV_PROP_FLOAT64 ? v.as.f : fallback;
}

int graph_get_edge_prop_bool(const GV_GraphDB *g, uint64_t edge_id,
                              const char *key, int fallback)
{
    GV_PropValue v = edge_get_prop(g, edge_id, key);
    return v.type == GV_PROP_BOOL ? v.as.b : fallback;
}

int graph_find_nodes_by_prop(const GV_GraphDB *g, const char *key,
                             GV_PropValue min_val, GV_PropValue max_val,
                             uint64_t *out_ids, size_t max_count)
{
    if (!g || !key || !out_ids) return -1;
    graph_read_lock(g);
    int count = 0;
    for (size_t b = 0; b < g->node_bucket_count; b++) {
        for (NodeEntry *e = g->node_buckets[b]; e; e = e->next) {
            GV_GraphProp *p = find_prop(e->node.properties, key);
            if (!p || p->value.type != min_val.type) continue;
            if (gv_prop_compare(p->value, min_val) < 0) continue;
            if (gv_prop_compare(p->value, max_val) > 0) continue;
            if ((size_t)count < max_count) out_ids[count] = e->node.node_id;
            count++;
        }
    }
    graph_read_unlock(g);
    return count;
}

int graph_get_edges_out(const GV_GraphDB *g, uint64_t node_id,
                           uint64_t *out_ids, size_t max_count)
{
    if (!g || !out_ids) return -1;

    graph_read_lock(g);

    NodeEntry *e = find_node_entry(g, node_id);
    if (!e) {
        graph_read_unlock(g);
        return -1;
    }

    size_t n = e->node.out_count;
    if (n > max_count) n = max_count;
    for (size_t i = 0; i < n; i++) {
        out_ids[i] = e->node.out_edges[i].edge_id;
    }

    graph_read_unlock(g);
    return (int)n;
}

int graph_get_edges_in(const GV_GraphDB *g, uint64_t node_id,
                          uint64_t *out_ids, size_t max_count)
{
    if (!g || !out_ids) return -1;

    graph_read_lock(g);

    NodeEntry *e = find_node_entry(g, node_id);
    if (!e) {
        graph_read_unlock(g);
        return -1;
    }

    size_t n = e->node.in_count;
    if (n > max_count) n = max_count;
    for (size_t i = 0; i < n; i++) {
        out_ids[i] = e->node.in_edges[i].edge_id;
    }

    graph_read_unlock(g);
    return (int)n;
}

int graph_get_neighbors(const GV_GraphDB *g, uint64_t node_id,
                           uint64_t *out_ids, size_t max_count)
{
    if (!g || !out_ids) return -1;

    graph_read_lock(g);

    NodeEntry *e = find_node_entry(g, node_id);
    if (!e) {
        graph_read_unlock(g);
        return -1;
    }

    size_t total_refs = e->node.out_count + e->node.in_count;
    VisitedSet seen;
    if (visited_init(&seen, total_refs > 16 ? total_refs * 2 : 32) != 0) {
        graph_read_unlock(g);
        return -1;
    }

    int count = 0;
    for (size_t i = 0; i < e->node.out_count && (size_t)count < max_count; i++) {
        uint64_t nid = e->node.out_edges[i].neighbor_id;
        if (visited_insert(&seen, nid) == 1) {
            out_ids[count++] = nid;
        }
    }
    for (size_t i = 0; i < e->node.in_count && (size_t)count < max_count; i++) {
        uint64_t nid = e->node.in_edges[i].neighbor_id;
        if (visited_insert(&seen, nid) == 1) {
            out_ids[count++] = nid;
        }
    }

    visited_free(&seen);
    graph_read_unlock(g);
    return count;
}

int graph_get_out_neighbors_typed(const GV_GraphDB *g, uint64_t node_id,
                                  const char *predicate,
                                  uint64_t *out_ids, size_t max_count)
{
    if (!g || !out_ids) return -1;

    graph_read_lock(g);

    NodeEntry *e = find_node_entry(g, node_id);
    if (!e) {
        graph_read_unlock(g);
        return -1;
    }

    int count = 0;
    for (size_t i = 0; i < e->node.out_count && (size_t)count < max_count; i++) {
        if (predicate) {
            EdgeEntry *ee = find_edge_entry(g, e->node.out_edges[i].edge_id);
            const char *lbl = ee ? ee->edge.label : NULL;
            if (!lbl || strcmp(lbl, predicate) != 0) continue;
        }
        out_ids[count++] = e->node.out_edges[i].neighbor_id;
    }

    graph_read_unlock(g);
    return count;
}


size_t graph_node_count(const GV_GraphDB *g)
{
    if (!g) return 0;
    graph_read_lock(g);
    size_t count = g->node_count;
    graph_read_unlock(g);
    return count;
}

size_t graph_edge_count(const GV_GraphDB *g)
{
    if (!g) return 0;
    graph_read_lock(g);
    size_t count = g->edge_count;
    graph_read_unlock(g);
    return count;
}
