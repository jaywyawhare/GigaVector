#ifndef GV_HEAP_H
#define GV_HEAP_H

#include <stddef.h>

/*
 * Macro-generated max-heap for top-k nearest neighbor selection.
 *
 * Usage:
 *   // Define a heap item type with at least a `dist` field:
 *   typedef struct { float dist; size_t idx; } MyHeapItem;
 *
 *   // Generate heap functions with a given prefix:
 *   GV_HEAP_DEFINE(my_heap, MyHeapItem)
 *
 *   // This generates:
 *   //   static void my_heap_sift_down(MyHeapItem *heap, size_t size, size_t i);
 *   //   static void my_heap_push(MyHeapItem *heap, size_t *size, size_t cap,
 *   //                            MyHeapItem item);
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

#endif /* GV_HEAP_H */
