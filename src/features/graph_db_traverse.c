/**
  * @file graph_db_traverse.c
  * @brief Graph traversal and analytics: BFS/DFS, Dijkstra, all-paths,
  *        PageRank, degree, connected components, clustering (split from graph_db.c).
  */
#define _POSIX_C_SOURCE 200112L

#include "features/graph_db_internal.h"
#include "core/memory.h"
#include "core/utils.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <pthread.h>

/* Open-addressing hash map (uint64_t -> float) for Dijkstra distances and path reconstruction */
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
    dm->keys = (uint64_t *)gv_alloc(capacity * sizeof(uint64_t));
    dm->vals = (float *)gv_alloc(capacity * sizeof(float));
    dm->prev_node = (uint64_t *)gv_alloc(capacity * sizeof(uint64_t));
    dm->prev_edge = (uint64_t *)gv_alloc(capacity * sizeof(uint64_t));
    dm->occupied = (int *)gv_calloc(capacity, sizeof(int));
    if (!dm->keys || !dm->vals || !dm->prev_node || !dm->prev_edge || !dm->occupied) {
        gv_free(dm->keys);
        gv_free(dm->vals);
        gv_free(dm->prev_node);
        gv_free(dm->prev_edge);
        gv_free(dm->occupied);
        return -1;
    }
    dm->capacity = capacity;
    dm->count = 0;
    return 0;
}

static void distmap_free(DistMap *dm)
{
    gv_free(dm->keys);
    gv_free(dm->vals);
    gv_free(dm->prev_node);
    gv_free(dm->prev_edge);
    gv_free(dm->occupied);
}

static size_t distmap_probe(const DistMap *dm, uint64_t key)
{
    return (size_t)(key * 2654435761ULL) % dm->capacity;
}

static int distmap_rehash(DistMap *dm)
{
    size_t new_cap = dm->capacity * 2;
    uint64_t *nk = (uint64_t *)gv_alloc(new_cap * sizeof(uint64_t));
    float *nv = (float *)gv_alloc(new_cap * sizeof(float));
    uint64_t *npn = (uint64_t *)gv_alloc(new_cap * sizeof(uint64_t));
    uint64_t *npe = (uint64_t *)gv_alloc(new_cap * sizeof(uint64_t));
    int *no = (int *)gv_calloc(new_cap, sizeof(int));
    if (!nk || !nv || !npn || !npe || !no) {
        gv_free(nk); gv_free(nv); gv_free(npn); gv_free(npe); gv_free(no);
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
    gv_free(dm->keys); gv_free(dm->vals); gv_free(dm->prev_node);
    gv_free(dm->prev_edge); gv_free(dm->occupied);
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
    h->data = (HeapEntry *)gv_alloc(capacity * sizeof(HeapEntry));
    if (!h->data) return -1;
    h->size = 0;
    h->capacity = capacity;
    return 0;
}

static void heap_free(MinHeap *h)
{
    gv_free(h->data);
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
        HeapEntry *tmp = (HeapEntry *)gv_realloc(h->data,
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

int graph_bfs(const GV_GraphDB *g, uint64_t start, size_t max_depth,
                 uint64_t *out_ids, size_t max_count)
{
    if (!g || !out_ids) return -1;

    graph_read_lock(g);

    if (!find_node_entry(g, start)) {
        graph_read_unlock(g);
        return -1;
    }

    size_t queue_cap = g->node_count > 64 ? g->node_count : 64;
    uint64_t *q_ids = (uint64_t *)gv_alloc(queue_cap * sizeof(uint64_t));
    size_t *q_depths = (size_t *)gv_alloc(queue_cap * sizeof(size_t));
    if (!q_ids || !q_depths) {
        gv_free(q_ids);
        gv_free(q_depths);
        graph_read_unlock(g);
        return -1;
    }

    VisitedSet visited;
    if (visited_init(&visited, queue_cap * 2) != 0) {
        gv_free(q_ids);
        gv_free(q_depths);
        graph_read_unlock(g);
        return -1;
    }

    size_t q_head = 0, q_tail = 0;
    int result_count = 0;

    q_ids[q_tail] = start;
    q_depths[q_tail] = 0;
    q_tail++;
    visited_insert(&visited, start);

    while (q_head < q_tail) {
        uint64_t cur_id = q_ids[q_head];
        size_t cur_depth = q_depths[q_head];
        q_head++;

        if ((size_t)result_count < max_count) {
            out_ids[result_count] = cur_id;
        }
        result_count++;

        if (cur_depth < max_depth) {
            NodeEntry *ne = find_node_entry(g, cur_id);
            if (ne) {
                for (size_t i = 0; i < ne->node.out_count; i++) {
                    uint64_t nid = ne->node.out_edges[i].neighbor_id;
                    if (visited_insert(&visited, nid) == 1) {
                        if (q_tail >= queue_cap) {
                            size_t new_cap = queue_cap * 2;
                            uint64_t *nqi = (uint64_t *)gv_realloc(
                                q_ids, new_cap * sizeof(uint64_t));
                            size_t *nqd = (size_t *)gv_realloc(
                                q_depths, new_cap * sizeof(size_t));
                            if (!nqi || !nqd) {
                                gv_free(nqi ? nqi : q_ids);
                                gv_free(nqd ? nqd : q_depths);
                                /* Null both so bfs_done's cleanup can't free the
                                 * same (already-freed) blocks again - the prior
                                 * `q_ids = nqi` left a freed pointer live. */
                                q_ids = NULL;
                                q_depths = NULL;
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
                            uint64_t *nqi = (uint64_t *)gv_realloc(
                                q_ids, new_cap * sizeof(uint64_t));
                            size_t *nqd = (size_t *)gv_realloc(
                                q_depths, new_cap * sizeof(size_t));
                            if (!nqi || !nqd) {
                                gv_free(nqi ? nqi : q_ids);
                                gv_free(nqd ? nqd : q_depths);
                                /* Null both so bfs_done's cleanup can't free the
                                 * same (already-freed) blocks again - the prior
                                 * `q_ids = nqi` left a freed pointer live. */
                                q_ids = NULL;
                                q_depths = NULL;
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
    gv_free(q_ids);
    gv_free(q_depths);
    graph_read_unlock(g);

    return (result_count > (int)max_count) ? (int)max_count : result_count;
}

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

    for (size_t i = 0; i < ne->node.out_count; i++) {
        uint64_t nid = ne->node.out_edges[i].neighbor_id;
        if (visited_insert(visited, nid) == 1) {
            dfs_recurse(g, nid, depth + 1, max_depth,
                        visited, out_ids, max_count, count);
        }
    }
    for (size_t i = 0; i < ne->node.in_count; i++) {
        uint64_t nid = ne->node.in_edges[i].neighbor_id;
        if (visited_insert(visited, nid) == 1) {
            dfs_recurse(g, nid, depth + 1, max_depth,
                        visited, out_ids, max_count, count);
        }
    }
}

int graph_dfs(const GV_GraphDB *g, uint64_t start, size_t max_depth,
                 uint64_t *out_ids, size_t max_count)
{
    if (!g || !out_ids) return -1;

    graph_read_lock(g);

    if (!find_node_entry(g, start)) {
        graph_read_unlock(g);
        return -1;
    }

    size_t vis_cap = g->node_count > 16 ? g->node_count * 2 : 32;
    VisitedSet visited;
    if (visited_init(&visited, vis_cap) != 0) {
        graph_read_unlock(g);
        return -1;
    }

    visited_insert(&visited, start);
    int count = 0;
    dfs_recurse(g, start, 0, max_depth, &visited, out_ids, max_count, &count);

    visited_free(&visited);
    graph_read_unlock(g);

    return (count > (int)max_count) ? (int)max_count : count;
}

int graph_shortest_path(const GV_GraphDB *g, uint64_t from, uint64_t to,
                           GV_GraphPath *path)
{
    if (!g || !path) return -1;

    memset(path, 0, sizeof(GV_GraphPath));

    graph_read_lock(g);

    if (!find_node_entry(g, from) || !find_node_entry(g, to)) {
        graph_read_unlock(g);
        return -1;
    }

    if (from == to) {
        path->node_ids = (uint64_t *)gv_alloc(sizeof(uint64_t));
        if (path->node_ids) {
            path->node_ids[0] = from;
        }
        path->edge_ids = NULL;
        path->length = 0;
        path->total_weight = 0.0f;
        graph_read_unlock(g);
        return 0;
    }

    size_t cap = g->node_count > 16 ? g->node_count * 2 : 32;
    DistMap dm;
    if (distmap_init(&dm, cap) != 0) {
        graph_read_unlock(g);
        return -1;
    }

    MinHeap heap;
    if (heap_init(&heap, cap) != 0) {
        distmap_free(&dm);
        graph_read_unlock(g);
        return -1;
    }

    VisitedSet finalized;
    if (visited_init(&finalized, cap) != 0) {
        distmap_free(&dm);
        heap_free(&heap);
        graph_read_unlock(g);
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
        graph_read_unlock(g);
        return -1;
    }

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
                graph_read_unlock(g);
                return -1;
            }
        }
    }

    path->length = path_len;
    path->total_weight = distmap_get(&dm, to);
    path->node_ids = (uint64_t *)gv_alloc((path_len + 1) * sizeof(uint64_t));
    path->edge_ids = (uint64_t *)gv_alloc(path_len * sizeof(uint64_t));

    if (!path->node_ids || (path_len > 0 && !path->edge_ids)) {
        gv_free(path->node_ids);
        gv_free(path->edge_ids);
        memset(path, 0, sizeof(GV_GraphPath));
        distmap_free(&dm);
        heap_free(&heap);
        visited_free(&finalized);
        graph_read_unlock(g);
        return -1;
    }

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
    graph_read_unlock(g);
    return 0;
}

static int build_path(const GV_GraphDB *g,
                      const uint64_t *node_stack, const uint64_t *edge_stack,
                      size_t depth, GV_GraphPath *out)
{
    (void)g;
    out->length = depth;
    out->node_ids = (uint64_t *)gv_alloc((depth + 1) * sizeof(uint64_t));
    out->edge_ids = depth > 0 ? (uint64_t *)gv_alloc(depth * sizeof(uint64_t)) : NULL;
    if (!out->node_ids || (depth > 0 && !out->edge_ids)) {
        gv_free(out->node_ids);
        gv_free(out->edge_ids);
        memset(out, 0, sizeof(GV_GraphPath));
        return -1;
    }
    memcpy(out->node_ids, node_stack, (depth + 1) * sizeof(uint64_t));
    if (depth > 0) {
        memcpy(out->edge_ids, edge_stack, depth * sizeof(uint64_t));
    }

    out->total_weight = 0.0f;
    for (size_t i = 0; i < depth; i++) {
        EdgeEntry *ee = find_edge_entry(g, edge_stack[i]);
        if (ee) out->total_weight += ee->edge.weight;
    }
    return 0;
}

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

int graph_all_paths(const GV_GraphDB *g, uint64_t from, uint64_t to,
                       size_t max_depth, GV_GraphPath *paths, size_t max_paths)
{
    if (!g || !paths) return -1;

    graph_read_lock(g);

    if (!find_node_entry(g, from) || !find_node_entry(g, to)) {
        graph_read_unlock(g);
        return -1;
    }

    uint64_t *node_stack = (uint64_t *)gv_alloc((max_depth + 2) * sizeof(uint64_t));
    uint64_t *edge_stack = (uint64_t *)gv_alloc((max_depth + 1) * sizeof(uint64_t));
    if (!node_stack || !edge_stack) {
        gv_free(node_stack);
        gv_free(edge_stack);
        graph_read_unlock(g);
        return -1;
    }

    node_stack[0] = from;
    int path_count = 0;

    all_paths_dfs(g, from, to, 0, max_depth,
                  node_stack, edge_stack,
                  paths, max_paths, &path_count);

    gv_free(node_stack);
    gv_free(edge_stack);
    graph_read_unlock(g);

    return (path_count > (int)max_paths) ? (int)max_paths : path_count;
}

void graph_free_path(GV_GraphPath *path)
{
    if (!path) return;
    gv_free(path->node_ids);
    gv_free(path->edge_ids);
    path->node_ids = NULL;
    path->edge_ids = NULL;
    path->length = 0;
    path->total_weight = 0.0f;
}

float graph_pagerank(const GV_GraphDB *g, uint64_t node_id,
                        size_t iterations, float damping)
{
    if (!g || iterations == 0) return 0.0f;

    graph_read_lock(g);

    size_t N = g->node_count;
    if (N == 0) {
        graph_read_unlock(g);
        return 0.0f;
    }

    size_t id_count = 0;
    uint64_t *all_ids = collect_all_node_ids(g, &id_count);
    if (!all_ids || id_count == 0) {
        gv_free(all_ids);
        graph_read_unlock(g);
        return 0.0f;
    }

    size_t map_cap = id_count * 3;
    uint64_t *map_keys = (uint64_t *)gv_alloc(map_cap * sizeof(uint64_t));
    int *map_occ = (int *)gv_calloc(map_cap, sizeof(int));
    size_t *map_idx = (size_t *)gv_alloc(map_cap * sizeof(size_t));
    float *scores = (float *)gv_alloc(id_count * sizeof(float));
    float *new_scores = (float *)gv_alloc(id_count * sizeof(float));

    if (!map_keys || !map_occ || !map_idx || !scores || !new_scores) {
        gv_free(all_ids); gv_free(map_keys); gv_free(map_occ);
        gv_free(map_idx); gv_free(scores); gv_free(new_scores);
        graph_read_unlock(g);
        return 0.0f;
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
        scores[i] = 1.0f / (float)N;
    }

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

        float *tmp = scores;
        scores = new_scores;
        new_scores = tmp;
    }

    float result = 0.0f;
    size_t target_idx = idmap_lookup(map_keys, map_occ, map_idx,
                                     map_cap, node_id);
    if (target_idx != (size_t)-1) {
        result = scores[target_idx];
    }

    gv_free(all_ids);
    gv_free(map_keys);
    gv_free(map_occ);
    gv_free(map_idx);
    gv_free(scores);
    gv_free(new_scores);
    graph_read_unlock(g);
    return result;
}

size_t graph_degree(const GV_GraphDB *g, uint64_t node_id)
{
    if (!g) return 0;
    graph_read_lock(g);
    NodeEntry *e = find_node_entry(g, node_id);
    size_t deg = e ? (e->node.in_count + e->node.out_count) : 0;
    graph_read_unlock(g);
    return deg;
}

size_t graph_in_degree(const GV_GraphDB *g, uint64_t node_id)
{
    if (!g) return 0;
    graph_read_lock(g);
    NodeEntry *e = find_node_entry(g, node_id);
    size_t deg = e ? e->node.in_count : 0;
    graph_read_unlock(g);
    return deg;
}

size_t graph_out_degree(const GV_GraphDB *g, uint64_t node_id)
{
    if (!g) return 0;
    graph_read_lock(g);
    NodeEntry *e = find_node_entry(g, node_id);
    size_t deg = e ? e->node.out_count : 0;
    graph_read_unlock(g);
    return deg;
}

int graph_connected_components(const GV_GraphDB *g,
                                  uint64_t *component_ids, size_t max_count)
{
    if (!g || !component_ids) return -1;

    graph_read_lock(g);

    size_t N = g->node_count;
    if (N == 0) {
        graph_read_unlock(g);
        return 0;
    }
    if (max_count < N) {
        graph_read_unlock(g);
        return -1;
    }

    size_t id_count = 0;
    uint64_t *all_ids = collect_all_node_ids(g, &id_count);
    if (!all_ids) {
        graph_read_unlock(g);
        return -1;
    }

    for (size_t i = 0; i < id_count; i++) {
        component_ids[i] = 0;
    }

    size_t map_cap = id_count * 3;
    uint64_t *map_keys = (uint64_t *)gv_alloc(map_cap * sizeof(uint64_t));
    int *map_occ = (int *)gv_calloc(map_cap, sizeof(int));
    size_t *map_idx = (size_t *)gv_alloc(map_cap * sizeof(size_t));

    if (!map_keys || !map_occ || !map_idx) {
        gv_free(all_ids); gv_free(map_keys); gv_free(map_occ); gv_free(map_idx);
        graph_read_unlock(g);
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

    uint64_t *queue = (uint64_t *)gv_alloc(id_count * sizeof(uint64_t));
    if (!queue) {
        gv_free(all_ids); gv_free(map_keys); gv_free(map_occ); gv_free(map_idx);
        graph_read_unlock(g);
        return -1;
    }

    uint64_t comp_id = 0;

    for (size_t i = 0; i < id_count; i++) {
        if (component_ids[i] != 0) continue;

        comp_id++;
        size_t q_head = 0, q_tail = 0;
        queue[q_tail++] = all_ids[i];
        component_ids[i] = comp_id;

        while (q_head < q_tail) {
            uint64_t cur = queue[q_head++];
            NodeEntry *ne = find_node_entry(g, cur);
            if (!ne) continue;

            for (size_t e = 0; e < ne->node.out_count; e++) {
                uint64_t nid = ne->node.out_edges[e].neighbor_id;
                size_t nidx = idmap_lookup(map_keys, map_occ, map_idx,
                                           map_cap, nid);
                if (nidx != (size_t)-1 && component_ids[nidx] == 0) {
                    component_ids[nidx] = comp_id;
                    queue[q_tail++] = nid;
                }
            }
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

    gv_free(all_ids);
    gv_free(map_keys);
    gv_free(map_occ);
    gv_free(map_idx);
    gv_free(queue);
    graph_read_unlock(g);
    return (int)comp_id;
}

float graph_clustering_coefficient(const GV_GraphDB *g, uint64_t node_id)
{
    if (!g) return 0.0f;

    graph_read_lock(g);

    NodeEntry *ne = find_node_entry(g, node_id);
    if (!ne) {
        graph_read_unlock(g);
        return 0.0f;
    }

    size_t total_refs = ne->node.out_count + ne->node.in_count;
    if (total_refs < 2) {
        graph_read_unlock(g);
        return 0.0f;
    }

    uint64_t *neighbors = (uint64_t *)gv_alloc(total_refs * sizeof(uint64_t));
    if (!neighbors) {
        graph_read_unlock(g);
        return 0.0f;
    }

    VisitedSet seen;
    if (visited_init(&seen, total_refs * 2 + 16) != 0) {
        gv_free(neighbors);
        graph_read_unlock(g);
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
        gv_free(neighbors);
        graph_read_unlock(g);
        return 0.0f;
    }

    VisitedSet nbr_set;
    if (visited_init(&nbr_set, k * 3 + 16) != 0) {
        gv_free(neighbors);
        graph_read_unlock(g);
        return 0.0f;
    }
    for (size_t i = 0; i < k; i++) {
        visited_insert(&nbr_set, neighbors[i]);
    }

    size_t edge_count_among = 0;
    for (size_t i = 0; i < k; i++) {
        NodeEntry *nne = find_node_entry(g, neighbors[i]);
        if (!nne) continue;

        for (size_t e = 0; e < nne->node.out_count; e++) {
            uint64_t tgt = nne->node.out_edges[e].neighbor_id;
            if (tgt != node_id && visited_contains(&nbr_set, tgt)) {
                edge_count_among++;
            }
        }
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
    gv_free(neighbors);
    graph_read_unlock(g);
    return cc;
}

