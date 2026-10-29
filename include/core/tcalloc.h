#ifndef GV_TCALLOC_H
#define GV_TCALLOC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * gv_tcalloc: a thread-caching slab allocator used behind gv_alloc/gv_free.
 *
 * Small requests (<= GV_TC_MAX_SMALL) are served from per-thread free lists of
 * fixed-size objects carved out of slabs, so the hot path is a lock-free pop/push
 * with no libc malloc call. Each thread refills/flushes in batches from a
 * per-size-class central list (one sharded mutex), so a parallel HNSW build does
 * not serialize on a single global heap lock the way glibc's main arena can.
 * Larger requests go straight to mmap with a slab-aligned header. Every slab is
 * aligned to GV_TC_SLAB_SIZE, so free() recovers an object's size class by
 * masking the pointer to its slab header, no per-pointer bookkeeping table.
 *
 * Under AddressSanitizer / ThreadSanitizer this layer compiles to thin libc
 * forwarders (see GV_TC_FORCE_LIBC) so those tools keep intercepting allocations.
 * At runtime, GV_ALLOC=libc forces the libc path for the whole process (used by
 * the valgrind gates so leak tracking stays per-object).
 */

/** Allocate @p size bytes (16-byte aligned); returns NULL on size 0 or OOM. */
void *gv_tc_malloc(size_t size);

/** Allocate @p nmemb*@p size zeroed bytes; returns NULL on overflow or OOM. */
void *gv_tc_calloc(size_t nmemb, size_t size);

/** Resize @p ptr to @p size bytes, preserving contents; NULL ptr acts as malloc. */
void *gv_tc_realloc(void *ptr, size_t size);

/** Free a pointer returned by gv_tc_malloc/calloc/realloc; NULL is a no-op. */
void gv_tc_free(void *ptr);

/** Usable bytes backing @p ptr (its size class or large payload), 0 for NULL. */
size_t gv_tc_usable_size(void *ptr);

/** 1 if the custom allocator is active, 0 if routing to libc (sanitizer/env). */
int gv_tc_active(void);

/**
 * Enable (@p on != 0) or disable the custom allocator at runtime, overriding the
 * GV_ALLOC default. Safe to flip after allocations: foreign pointers are routed
 * back to libc by the ownership registry. No-op under sanitizer builds.
 */
void gv_tc_set_enabled(int on);

#ifdef __cplusplus
}
#endif

#endif /* GV_TCALLOC_H */
