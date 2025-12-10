#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gigavector/gv_wal.h"

#define GV_WAL_MAGIC "GVW1"
#define GV_WAL_VERSION 1u
#define GV_WAL_TYPE_INSERT 1u

struct GV_WAL {
    FILE *file;
    char *path;
    size_t dimension;
};

static int gv_wal_write_u32(FILE *f, uint32_t v) {
    return fwrite(&v, sizeof(uint32_t), 1, f) == 1 ? 0 : -1;
}

static int gv_wal_write_u8(FILE *f, uint8_t v) {
    return fwrite(&v, sizeof(uint8_t), 1, f) == 1 ? 0 : -1;
}

static int gv_wal_write_floats(FILE *f, const float *data, size_t n) {
    return fwrite(data, sizeof(float), n, f) == n ? 0 : -1;
}

static int gv_wal_write_string(FILE *f, const char *s) {
    if (s == NULL) {
        if (gv_wal_write_u32(f, 0) != 0) return -1;
        return 0;
    }
    size_t len = strlen(s);
    if (len > UINT32_MAX) return -1;
    if (gv_wal_write_u32(f, (uint32_t)len) != 0) return -1;
    return fwrite(s, 1, len, f) == len ? 0 : -1;
}

static int gv_wal_read_u32(FILE *f, uint32_t *out) {
    return (out != NULL && fread(out, sizeof(uint32_t), 1, f) == 1) ? 0 : -1;
}

static int gv_wal_read_u8(FILE *f, uint8_t *out) {
    return (out != NULL && fread(out, sizeof(uint8_t), 1, f) == 1) ? 0 : -1;
}

static int gv_wal_read_floats(FILE *f, float *data, size_t n) {
    return fread(data, sizeof(float), n, f) == n ? 0 : -1;
}

static int gv_wal_read_string(FILE *f, char **out) {
    uint32_t len = 0;
    if (gv_wal_read_u32(f, &len) != 0) return -1;
    if (len == 0) {
        *out = NULL;
        return 0;
    }
    char *buf = (char *)malloc(len + 1);
    if (buf == NULL) return -1;
    if (fread(buf, 1, len, f) != len) {
        free(buf);
        return -1;
    }
    buf[len] = '\0';
    *out = buf;
    return 0;
}

GV_WAL *gv_wal_open(const char *path, size_t dimension) {
    if (path == NULL || dimension == 0) {
        return NULL;
    }

    FILE *f = fopen(path, "ab+");
    if (f == NULL) {
        return NULL;
    }

    /* Ensure we read from start for header validation */
    rewind(f);

    char magic[4] = {0};
    if (fread(magic, 1, 4, f) != 4) {
        /* New file: write header */
        rewind(f);
        if (fwrite(GV_WAL_MAGIC, 1, 4, f) != 4) {
            fclose(f);
            return NULL;
        }
        if (gv_wal_write_u32(f, GV_WAL_VERSION) != 0 ||
            gv_wal_write_u32(f, (uint32_t)dimension) != 0) {
            fclose(f);
            return NULL;
        }
        fflush(f);
    } else {
        uint32_t version = 0;
        uint32_t file_dim = 0;
        if (memcmp(magic, GV_WAL_MAGIC, 4) != 0 ||
            gv_wal_read_u32(f, &version) != 0 ||
            gv_wal_read_u32(f, &file_dim) != 0) {
            fclose(f);
            return NULL;
        }
        if (version != GV_WAL_VERSION || file_dim != (uint32_t)dimension) {
            fclose(f);
            errno = EINVAL;
            return NULL;
        }
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }

    GV_WAL *wal = (GV_WAL *)malloc(sizeof(GV_WAL));
    if (wal == NULL) {
        fclose(f);
        return NULL;
    }
    wal->file = f;
    wal->dimension = dimension;
    wal->path = strdup(path);
    if (wal->path == NULL) {
        fclose(f);
        free(wal);
        return NULL;
    }
    return wal;
}

int gv_wal_append_insert(GV_WAL *wal, const float *data, size_t dimension,
                         const char *metadata_key, const char *metadata_value) {
    if (wal == NULL || wal->file == NULL || data == NULL || dimension == 0) {
        return -1;
    }
    if (dimension != wal->dimension) {
        return -1;
    }

    if (gv_wal_write_u8(wal->file, GV_WAL_TYPE_INSERT) != 0) return -1;
    if (gv_wal_write_u32(wal->file, (uint32_t)dimension) != 0) return -1;
    if (gv_wal_write_floats(wal->file, data, dimension) != 0) return -1;

    /* Allow single metadata pair; count=0 or 1 */
    uint32_t meta_count = (metadata_key != NULL && metadata_value != NULL) ? 1u : 0u;
    if (gv_wal_write_u32(wal->file, meta_count) != 0) return -1;
    if (meta_count == 1u) {
        if (gv_wal_write_string(wal->file, metadata_key) != 0) return -1;
        if (gv_wal_write_string(wal->file, metadata_value) != 0) return -1;
    }

    if (fflush(wal->file) != 0) {
        return -1;
    }
    return 0;
}

int gv_wal_replay(const char *path, size_t expected_dimension,
                  int (*on_insert)(void *ctx, const float *data, size_t dimension,
                                   const char *metadata_key, const char *metadata_value),
                  void *ctx) {
    if (path == NULL || expected_dimension == 0 || on_insert == NULL) {
        return -1;
    }

    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        return (errno == ENOENT) ? 0 : -1;
    }

    char magic[4] = {0};
    uint32_t version = 0;
    uint32_t file_dim = 0;
    if (fread(magic, 1, 4, f) != 4 ||
        memcmp(magic, GV_WAL_MAGIC, 4) != 0 ||
        gv_wal_read_u32(f, &version) != 0 ||
        gv_wal_read_u32(f, &file_dim) != 0 ||
        version != GV_WAL_VERSION ||
        file_dim != (uint32_t)expected_dimension) {
        fclose(f);
        return -1;
    }

    while (1) {
        uint8_t type = 0;
        if (gv_wal_read_u8(f, &type) != 0) {
            if (feof(f)) break;
            fclose(f);
            return -1;
        }

        if (type == GV_WAL_TYPE_INSERT) {
            uint32_t dim = 0;
            if (gv_wal_read_u32(f, &dim) != 0 || dim != (uint32_t)expected_dimension) {
                fclose(f);
                return -1;
            }
            float *buf = (float *)malloc(sizeof(float) * dim);
            if (buf == NULL) {
                fclose(f);
                return -1;
            }
            if (gv_wal_read_floats(f, buf, dim) != 0) {
                free(buf);
                fclose(f);
                return -1;
            }
            uint32_t meta_count = 0;
            if (gv_wal_read_u32(f, &meta_count) != 0 || meta_count > 1) {
                free(buf);
                fclose(f);
                return -1;
            }
            char *k = NULL;
            char *v = NULL;
            if (meta_count == 1) {
                if (gv_wal_read_string(f, &k) != 0 || gv_wal_read_string(f, &v) != 0) {
                    free(buf);
                    free(k);
                    free(v);
                    fclose(f);
                    return -1;
                }
            }

            int cb_res = on_insert(ctx, buf, dim, k, v);
            free(buf);
            free(k);
            free(v);
            if (cb_res != 0) {
                fclose(f);
                return -1;
            }
        } else {
            fclose(f);
            return -1;
        }
    }

    fclose(f);
    return 0;
}

int gv_wal_dump(const char *path, size_t expected_dimension, FILE *out) {
    if (path == NULL || expected_dimension == 0 || out == NULL) {
        return -1;
    }

    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        return (errno == ENOENT) ? -1 : -1;
    }

    char magic[4] = {0};
    uint32_t version = 0;
    uint32_t file_dim = 0;
    if (fread(magic, 1, 4, f) != 4 ||
        memcmp(magic, GV_WAL_MAGIC, 4) != 0 ||
        gv_wal_read_u32(f, &version) != 0 ||
        gv_wal_read_u32(f, &file_dim) != 0 ||
        version != GV_WAL_VERSION ||
        file_dim != (uint32_t)expected_dimension) {
        fclose(f);
        return -1;
    }

    fprintf(out, "WAL %s: version=%u dimension=%u\n", path, version, file_dim);
    size_t record_index = 0;
    while (1) {
        uint8_t type = 0;
        if (gv_wal_read_u8(f, &type) != 0) {
            if (feof(f)) break;
            fclose(f);
            return -1;
        }

        if (type == GV_WAL_TYPE_INSERT) {
            uint32_t dim = 0;
            if (gv_wal_read_u32(f, &dim) != 0 || dim != (uint32_t)expected_dimension) {
                fclose(f);
                return -1;
            }
            float *buf = (float *)malloc(sizeof(float) * dim);
            if (buf == NULL) {
                fclose(f);
                return -1;
            }
            if (gv_wal_read_floats(f, buf, dim) != 0) {
                free(buf);
                fclose(f);
                return -1;
            }
            uint32_t meta_count = 0;
            if (gv_wal_read_u32(f, &meta_count) != 0 || meta_count > 1) {
                free(buf);
                fclose(f);
                return -1;
            }
            char *k = NULL;
            char *v = NULL;
            if (meta_count == 1) {
                if (gv_wal_read_string(f, &k) != 0 || gv_wal_read_string(f, &v) != 0) {
                    free(buf);
                    free(k);
                    free(v);
                    fclose(f);
                    return -1;
                }
            }

            fprintf(out, "#%zu INSERT dim=%u first=%.6f", record_index, dim, buf[0]);
            if (dim > 1) {
                fprintf(out, " second=%.6f", buf[1]);
            }
            if (meta_count == 1 && k && v) {
                fprintf(out, " meta[%s]=%s", k, v);
            }
            fprintf(out, "\n");

            free(buf);
            free(k);
            free(v);
            record_index++;
        } else {
            fclose(f);
            return -1;
        }
    }

    fclose(f);
    return 0;
}

void gv_wal_close(GV_WAL *wal) {
    if (wal == NULL) {
        return;
    }
    if (wal->file) {
        fflush(wal->file);
        fclose(wal->file);
    }
    free(wal->path);
    free(wal);
}

int gv_wal_reset(const char *path) {
    if (path == NULL) {
        return -1;
    }
    FILE *f = fopen(path, "wb");
    if (f == NULL) {
        return (errno == ENOENT) ? 0 : -1;
    }
    fclose(f);
    return 0;
}


