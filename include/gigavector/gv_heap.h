#ifndef GV_HEAP_H
#define GV_HEAP_H

#include <stddef.h>

/*
 * Macro-generated heaps for top-k selection. Item type must have a `dist` field.
 *
 * GV_HEAP_DEFINE (max-heap): root = largest dist. Use for distance metrics
 *   where lower is better — push evicts the worst (largest) candidate.
 *
 * GV_MIN_HEAP_DEFINE (min-heap): root = smallest dist. Use for similarity
 *   scores where higher is better — push evicts the worst (smallest) candidate.
 *
 * Both generate:
 *   static void prefix_sift_down(type *heap, size_t size, size_t i);
 *   static void prefix_push(type *heap, size_t *size, size_t cap, type item);
 */
#define GV_HEAP_DEFINE(prefix, type)                                           \
static void prefix##_sift_down(type *heap, size_t size, size_t i) {            \
    while (1) {                                                                \
        size_t l = 2 * i + 1, r = l + 1, largest = i;                         \
        if (l < size && heap[l].dist > heap[largest].dist) largest = l;        \
        if (r < size && heap[r].dist > heap[largest].dist) largest = r;        \
        if (largest == i) break;                                               \
        type tmp = heap[i]; heap[i] = heap[largest]; heap[largest] = tmp;      \
        i = largest;                                                           \
    }                                                                          \
}                                                                              \
                                                                               \
static void prefix##_push(type *heap, size_t *size, size_t cap, type item) {   \
    if (*size < cap) {                                                         \
        heap[*size] = item;                                                    \
        (*size)++;                                                             \
        size_t i = *size - 1;                                                  \
        while (i > 0) {                                                        \
            size_t parent = (i - 1) / 2;                                       \
            if (heap[i].dist > heap[parent].dist) {                            \
                type tmp = heap[i]; heap[i] = heap[parent];                    \
                heap[parent] = tmp; i = parent;                                \
            } else break;                                                      \
        }                                                                      \
    } else if (item.dist < heap[0].dist) {                                     \
        heap[0] = item;                                                        \
        prefix##_sift_down(heap, *size, 0);                                    \
    }                                                                          \
}

/* Min-heap: root = smallest dist. Evicts smallest when full, keeping top-k largest. */
#define GV_MIN_HEAP_DEFINE(prefix, type)                                       \
static void prefix##_sift_down(type *heap, size_t size, size_t i) {            \
    while (1) {                                                                \
        size_t l = 2 * i + 1, r = l + 1, smallest = i;                        \
        if (l < size && heap[l].dist < heap[smallest].dist) smallest = l;      \
        if (r < size && heap[r].dist < heap[smallest].dist) smallest = r;      \
        if (smallest == i) break;                                              \
        type tmp = heap[i]; heap[i] = heap[smallest]; heap[smallest] = tmp;    \
        i = smallest;                                                          \
    }                                                                          \
}                                                                              \
                                                                               \
static void prefix##_push(type *heap, size_t *size, size_t cap, type item) {   \
    if (*size < cap) {                                                         \
        heap[*size] = item;                                                    \
        (*size)++;                                                             \
        size_t i = *size - 1;                                                  \
        while (i > 0) {                                                        \
            size_t parent = (i - 1) / 2;                                       \
            if (heap[i].dist < heap[parent].dist) {                            \
                type tmp = heap[i]; heap[i] = heap[parent];                    \
                heap[parent] = tmp; i = parent;                                \
            } else break;                                                      \
        }                                                                      \
    } else if (item.dist > heap[0].dist) {                                     \
        heap[0] = item;                                                        \
        prefix##_sift_down(heap, *size, 0);                                    \
    }                                                                          \
}

#endif /* GV_HEAP_H */
