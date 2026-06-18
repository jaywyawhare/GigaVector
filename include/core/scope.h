#ifndef GV_SCOPE_H
#define GV_SCOPE_H

#include <stddef.h>
#include <stdio.h>

#include "core/arena.h"
#include "storage/database.h"
#include "storage/wal.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GV_OWN_BORROW = 0,
    GV_OWN_ARENA  = 1,
    GV_OWN_HEAP   = 2
} GV_OwnKind;

typedef struct GV_Bytes {
    void      *data;
    size_t     len;
    GV_OwnKind kind;
    GV_Arena  *arena;
} GV_Bytes;

void     gv_bytes_release(GV_Bytes *bytes);
GV_Bytes gv_bytes_from_arena(GV_Arena *arena, void *data, size_t len);
GV_Bytes gv_bytes_copy_heap(const void *data, size_t len);

#ifndef GV_TLS_ARENA_BYTES
#define GV_TLS_ARENA_BYTES (256u * 1024u)
#endif

GV_Arena *gv_tls_arena(void);
void      gv_tls_arena_reset(void);

#define GV_WITH_ARENA(arena, cap)                                              \
    for (GV_Arena arena, *_gv_scope_##arena =                                   \
            (GV_Arena *)((gv_arena_init(&(arena), (cap)), (void *)1));         \
         _gv_scope_##arena;                                                    \
         gv_arena_fini(&(arena)), _gv_scope_##arena = NULL)

#define GV_WITH_ARENA_STATIC(arena, backing, cap)                              \
    for (GV_Arena arena, *_gv_scope_##arena =                                   \
            (GV_Arena *)((gv_arena_init_static(&(arena), (backing), (cap)),     \
                          (void *)1));                                         \
         _gv_scope_##arena;                                                    \
         gv_arena_reset(&(arena)), _gv_scope_##arena = NULL)

#define GV_WITH_DB(db, path, dim, idx)                                         \
    for (GV_Database *db = db_open((path), (dim), (idx)),                      \
              *_gv_db_done = (GV_Database *)1;                                 \
         _gv_db_done;                                                          \
         (db != NULL ? (void)db_close(db) : (void)0), db = NULL,             \
         _gv_db_done = NULL)

#define GV_WITH_WAL(wal, path, dim, idx)                                       \
    for (GV_WAL *wal = wal_open((path), (dim), (uint32_t)(idx)),              \
              *_gv_wal_done = (GV_WAL *)1;                                     \
         _gv_wal_done;                                                         \
         (wal != NULL ? (void)wal_close(wal) : (void)0), wal = NULL,         \
         _gv_wal_done = NULL)

#define GV_WITH_FILE(fp, path, mode)                                           \
    for (FILE *fp = fopen((path), (mode)), *_gv_file_done = (FILE *)1;         \
         _gv_file_done;                                                        \
         (fp != NULL ? (void)fclose(fp) : (void)0), fp = NULL,                \
         _gv_file_done = NULL)

#ifdef __cplusplus
}
#endif

#endif /* GV_SCOPE_H */
