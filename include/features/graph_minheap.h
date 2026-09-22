#ifndef GIGAVECTOR_FEATURES_GRAPH_MINHEAP_H
#define GIGAVECTOR_FEATURES_GRAPH_MINHEAP_H

#include <stddef.h>
#include "core/memory.h"

/*
 * Indexed binary min-heap over dense node indices, ordered by a double key,
 * with O(log n) decrease/increase-key via the pos[] back-index. Shared by the
 * graph shortest-path and centrality algorithms.
 *
 * The `mh_push` policy differs per algorithm (decrease-key-only for Dijkstra vs.
 * unconditional update elsewhere), so callers define their own mh_push on top of
 * mh_sift_up/mh_sift_down; everything else lives here.
 */
typedef struct {
    size_t *idx;    /* heap array of dense node indices */
    double *key;    /* key[node] = current priority of that node */
    size_t *pos;    /* pos[node] = position in idx[], or (size_t)-1 if absent */
    size_t  size;
    size_t  cap;
} MinHeap;

static inline int mh_init(MinHeap *h, size_t n) {
    h->size = 0;
    h->cap = n;
    h->idx = NULL;
    h->key = NULL;
    h->pos = NULL;
    if (n == 0) return 0;
    h->idx = (size_t *)gv_alloc(n * sizeof(size_t));
    h->key = (double *)gv_alloc(n * sizeof(double));
    h->pos = (size_t *)gv_alloc(n * sizeof(size_t));
    if (!h->idx || !h->key || !h->pos) {
        gv_free(h->idx); gv_free(h->key); gv_free(h->pos);
        h->idx = NULL; h->key = NULL; h->pos = NULL;
        return -1;
    }
    for (size_t i = 0; i < n; i++) h->pos[i] = (size_t)-1;
    return 0;
}

static inline void mh_free(MinHeap *h) {
    if (!h) return;
    gv_free(h->idx); gv_free(h->key); gv_free(h->pos);
    h->idx = NULL; h->key = NULL; h->pos = NULL;
    h->size = 0; h->cap = 0;
}

static inline void mh_swap(MinHeap *h, size_t a, size_t b) {
    size_t ta = h->idx[a], tb = h->idx[b];
    h->idx[a] = tb; h->idx[b] = ta;
    h->pos[tb] = a; h->pos[ta] = b;
}

static inline void mh_sift_up(MinHeap *h, size_t i) {
    while (i > 0) {
        size_t parent = (i - 1) / 2;
        if (h->key[h->idx[parent]] <= h->key[h->idx[i]]) break;
        mh_swap(h, parent, i);
        i = parent;
    }
}

static inline void mh_sift_down(MinHeap *h, size_t i) {
    for (;;) {
        size_t l = 2 * i + 1, r = 2 * i + 2, smallest = i;
        if (l < h->size && h->key[h->idx[l]] < h->key[h->idx[smallest]]) smallest = l;
        if (r < h->size && h->key[h->idx[r]] < h->key[h->idx[smallest]]) smallest = r;
        if (smallest == i) break;
        mh_swap(h, i, smallest);
        i = smallest;
    }
}

/* Remove and return the min-key node index; returns (size_t)-1 when empty. */
static inline size_t mh_pop(MinHeap *h) {
    if (h->size == 0) return (size_t)-1;
    size_t top = h->idx[0];
    h->pos[top] = (size_t)-1;
    h->size--;
    if (h->size > 0) {
        h->idx[0] = h->idx[h->size];
        h->pos[h->idx[0]] = 0;
        mh_sift_down(h, 0);
    }
    return top;
}

#endif /* GIGAVECTOR_FEATURES_GRAPH_MINHEAP_H */
