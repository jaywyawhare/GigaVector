#ifndef GIGAVECTOR_OBJECT_STORE_H
#define GIGAVECTOR_OBJECT_STORE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file object_store.h
 * @brief Pluggable blob/object store (key -> bytes).
 *
 * A small put/get/delete/exists abstraction that decouples GigaVector from any
 * particular remote store. The built-in filesystem backend treats a local
 * directory as the store (atomic writes via tmp+rename), which is also the
 * natural stand-in for a networked backend (S3/GCS/Azure): implement the five
 * GV_ObjectStoreOps hooks and wrap them with object_store_wrap().
 *
 * Used for native backups (backup_to_object_store) instead of shelling out.
 */

typedef struct GV_ObjectStore GV_ObjectStore;

typedef struct {
    int  (*put)(void *ctx, const char *key, const void *data, size_t len);
    /* On success sets *data_out (malloc'd, caller frees) and *len_out. */
    int  (*get)(void *ctx, const char *key, void **data_out, size_t *len_out);
    int  (*del)(void *ctx, const char *key);
    int  (*exists)(void *ctx, const char *key); /* 1 = present, 0 = absent, -1 = error */
    void (*destroy)(void *ctx);
    void *ctx;
} GV_ObjectStoreOps;

/** Wrap a custom backend (e.g. an S3 client). Takes ownership of ops.ctx via
 *  ops.destroy. Returns NULL on error. */
GV_ObjectStore *object_store_wrap(const GV_ObjectStoreOps *ops);

/** Filesystem backend: @p root_dir is created if absent and used as the store. */
GV_ObjectStore *object_store_open_fs(const char *root_dir);

void object_store_destroy(GV_ObjectStore *os);

int object_store_put(GV_ObjectStore *os, const char *key, const void *data, size_t len);
/** On success sets *data_out (caller frees with free()) and *len_out. */
int object_store_get(GV_ObjectStore *os, const char *key, void **data_out, size_t *len_out);
int object_store_delete(GV_ObjectStore *os, const char *key);
int object_store_exists(GV_ObjectStore *os, const char *key);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_OBJECT_STORE_H */
