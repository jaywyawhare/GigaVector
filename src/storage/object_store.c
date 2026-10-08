/*
 * Object store abstraction + filesystem backend (see object_store.h).
 */

#include "storage/object_store.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "core/memory.h"
#include "core/utils.h"

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#define gv_mkdir(p) _mkdir(p)
#define fsync(fd) _commit(fd)
#else
#include <unistd.h>
#include <fcntl.h>
#define gv_mkdir(p) mkdir((p), 0700)
#endif

struct GV_ObjectStore {
    GV_ObjectStoreOps ops;
};

GV_ObjectStore *object_store_wrap(const GV_ObjectStoreOps *ops) {
    if (!ops || !ops->put || !ops->get || !ops->del || !ops->exists) return NULL;
    GV_ObjectStore *os = (GV_ObjectStore *)gv_calloc(1, sizeof(*os));
    if (!os) return NULL;
    os->ops = *ops;
    return os;
}

void object_store_destroy(GV_ObjectStore *os) {
    if (!os) return;
    if (os->ops.destroy) os->ops.destroy(os->ops.ctx);
    gv_free(os);
}

int object_store_put(GV_ObjectStore *os, const char *key, const void *data, size_t len) {
    if (!os || !key) return -1;
    return os->ops.put(os->ops.ctx, key, data, len);
}
int object_store_get(GV_ObjectStore *os, const char *key, void **data_out, size_t *len_out) {
    if (!os || !key || !data_out || !len_out) return -1;
    return os->ops.get(os->ops.ctx, key, data_out, len_out);
}
int object_store_delete(GV_ObjectStore *os, const char *key) {
    if (!os || !key) return -1;
    return os->ops.del(os->ops.ctx, key);
}
int object_store_exists(GV_ObjectStore *os, const char *key) {
    if (!os || !key) return -1;
    return os->ops.exists(os->ops.ctx, key);
}


typedef struct { char *root; } FsCtx;

/* Encode a key into a single safe flat filename: keep [A-Za-z0-9._-], percent-
 * encode everything else (so '/', spaces, etc. cannot escape the root dir). */
static char *fs_encode_key(const char *root, const char *key) {
    size_t rl = strlen(root), kl = strlen(key);
    char *path = (char *)gv_alloc(rl + 1 + kl * 3 + 1);
    if (!path) return NULL;
    size_t p = 0;
    memcpy(path, root, rl); p = rl;
    path[p++] = '/';
    static const char hex[] = "0123456789ABCDEF";
    for (size_t i = 0; i < kl; i++) {
        unsigned char c = (unsigned char)key[i];
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
            (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-') {
            path[p++] = (char)c;
        } else {
            path[p++] = '%'; path[p++] = hex[c >> 4]; path[p++] = hex[c & 0xF];
        }
    }
    path[p] = '\0';
    return path;
}

static int fs_put(void *ctx, const char *key, const void *data, size_t len) {
    FsCtx *fs = (FsCtx *)ctx;
    char *path = fs_encode_key(fs->root, key);
    if (!path) return -1;
    char *tmp = (char *)gv_alloc(strlen(path) + 16);
    if (!tmp) { gv_free(path); return -1; }
    snprintf(tmp, strlen(path) + 16, "%s.tmp%d", path, (int)getpid());

    int rc = -1;
    FILE *f = fopen(tmp, "wb");
    if (f) {
        int ok = (len == 0) || (fwrite(data, 1, len, f) == len);
        if (ok) ok = (fflush(f) == 0 && fsync(fileno(f)) == 0);
        fclose(f);
        if (ok && rename(tmp, path) == 0) {
            rc = 0;
#ifndef _WIN32
            /* fsync the containing directory so the rename itself survives a
             * crash (the file bytes were fsync'd above, the link was not). */
            char *slash = strrchr(path, '/');
            if (slash) {
                *slash = '\0';
                int dfd = open(path[0] ? path : "/", O_RDONLY | O_DIRECTORY);
                *slash = '/';
                if (dfd >= 0) { fsync(dfd); close(dfd); }
            }
#endif
        } else {
            remove(tmp);
        }
    }
    gv_free(tmp);
    gv_free(path);
    return rc;
}

static int fs_get(void *ctx, const char *key, void **data_out, size_t *len_out) {
    FsCtx *fs = (FsCtx *)ctx;
    char *path = fs_encode_key(fs->root, key);
    if (!path) return -1;
    int rc = -1;
    FILE *f = fopen(path, "rb");
    if (f) {
        if (fseek(f, 0, SEEK_END) == 0) {
            long sz = ftell(f);
            if (sz >= 0 && fseek(f, 0, SEEK_SET) == 0) {
                /* Returned buffer is plain malloc'd so the caller frees with free()
                 * (see header), independent of the internal gv_alloc arena. */
                void *buf = malloc((size_t)sz ? (size_t)sz : 1);
                if (buf && ((size_t)sz == 0 || fread(buf, 1, (size_t)sz, f) == (size_t)sz)) {
                    *data_out = buf; *len_out = (size_t)sz; rc = 0;
                } else {
                    free(buf);
                }
            }
        }
        fclose(f);
    }
    gv_free(path);
    return rc;
}

static int fs_del(void *ctx, const char *key) {
    FsCtx *fs = (FsCtx *)ctx;
    char *path = fs_encode_key(fs->root, key);
    if (!path) return -1;
    int rc = remove(path) == 0 ? 0 : -1;
    gv_free(path);
    return rc;
}

static int fs_exists(void *ctx, const char *key) {
    FsCtx *fs = (FsCtx *)ctx;
    char *path = fs_encode_key(fs->root, key);
    if (!path) return -1;
    struct stat st;
    int rc = (stat(path, &st) == 0) ? 1 : 0;
    gv_free(path);
    return rc;
}

static void fs_destroy(void *ctx) {
    FsCtx *fs = (FsCtx *)ctx;
    if (!fs) return;
    gv_free(fs->root);
    gv_free(fs);
}

GV_ObjectStore *object_store_open_fs(const char *root_dir) {
    if (!root_dir) return NULL;
    gv_mkdir(root_dir); /* ok if it already exists */
    struct stat st;
    if (stat(root_dir, &st) != 0) return NULL;
    FsCtx *fs = (FsCtx *)gv_calloc(1, sizeof(*fs));
    if (!fs) return NULL;
    fs->root = gv_dup_cstr(root_dir);
    if (!fs->root) { gv_free(fs); return NULL; }
    GV_ObjectStoreOps ops = {
        .put = fs_put, .get = fs_get, .del = fs_del,
        .exists = fs_exists, .destroy = fs_destroy, .ctx = fs,
    };
    GV_ObjectStore *os = object_store_wrap(&ops);
    if (!os) { fs_destroy(fs); return NULL; }
    return os;
}
