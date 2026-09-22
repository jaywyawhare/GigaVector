/*
 * Allocator stubs for CBMC harnesses.
 *
 * Linking src/core/memory.c pulls the whole memory-accounting subsystem --
 * pools, tracking tables, locks, resource limits -- into every model. None of
 * that is relevant to the properties these harnesses check, and it is what
 * made most of them intractable: CBMC was symbolically executing
 * gv_memory_untrack_locked and friends instead of the code under test.
 *
 * These stubs keep the ALLOCATION CONTRACT that the harnesses depend on --
 * a request either returns a distinct, correctly-sized, writable block or
 * NULL, and freeing is safe -- while removing the bookkeeping. CBMC models
 * malloc/free natively and precisely, including the failure path, so the
 * failure-handling paths in the code under test are still explored.
 *
 * Link this INSTEAD of src/core/memory.c.
 */
#include <stdlib.h>
#include <string.h>

void *gv_alloc(size_t size)                 { return malloc(size); }
void *gv_calloc(size_t n, size_t size)      { return calloc(n, size); }
void *gv_realloc(void *p, size_t size)      { return realloc(p, size); }
void  gv_free(void *p)                      { free(p); }
void *gv_db_alloc(void *db, size_t size)    { (void)db; return malloc(size); }
void *gv_db_calloc(void *db, size_t n, size_t s) { (void)db; return calloc(n, s); }
void *gv_db_realloc(void *db, void *p, size_t s) { (void)db; return realloc(p, s); }
void  gv_db_free(void *db, void *p)         { (void)db; free(p); }
