#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gigavector/gv_wal.h"

#define GV_WAL_MAGIC "GVW1"
#define GV_WAL_VERSION 3u
#define GV_WAL_TYPE_INSERT 1u

struct GV_WAL {
    FILE *file;
    char *path;
    size_t dimension;
    uint32_t index_type;
    uint32_t version;
};

/* Simple CRC32 (polynomial 0xEDB88320), tableless for portability */
static uint32_t gv_crc32_init(void) {
    return 0xFFFFFFFFu;
}

static uint32_t gv_crc32_update(uint32_t crc, const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    for (size_t i = 0; i < len; ++i) {
        crc ^= p[i];
        for (int k = 0; k < 8; ++k) {
            crc = (crc >> 1) ^ (0xEDB88320u & -(int)(crc & 1u));
        }
    }
    return crc;
}

static uint32_t gv_crc32_finish(uint32_t crc) {
    return crc ^ 0xFFFFFFFFu;
}

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

GV_WAL *gv_wal_open(const char *path, size_t dimension, uint32_t index_type) {
    if (path == NULL || dimension == 0) {
        return NULL;
    }

    uint32_t file_version = GV_WAL_VERSION;
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
            gv_wal_write_u32(f, (uint32_t)dimension) != 0 ||
            gv_wal_write_u32(f, index_type) != 0) {
            fclose(f);
            return NULL;
        }
        fflush(f);
    } else {
        uint32_t version = 0;
        uint32_t file_dim = 0;
        uint32_t file_index = 0;
        if (memcmp(magic, GV_WAL_MAGIC, 4) != 0 ||
            gv_wal_read_u32(f, &version) != 0 ||
            gv_wal_read_u32(f, &file_dim) != 0) {
            fclose(f);
            return NULL;
        }
        if (version >= 3) {
            if (gv_wal_read_u32(f, &file_index) != 0) {
                fclose(f);
                return NULL;
            }
        }
        if ((version != 1 && version != 2 && version != GV_WAL_VERSION) || file_dim != (uint32_t)dimension) {
            fprintf(stderr, "WAL open failed: version/dimension mismatch (got v%u dim=%u expected dim=%zu)\n",
                    version, file_dim, dimension);
            fclose(f);
            errno = EINVAL;
            return NULL;
        }
        if (index_type != 0 && file_index != 0 && file_index != index_type) {
            fprintf(stderr, "WAL open failed: index type mismatch (got %u expected %u)\n",
                    file_index, index_type);
            fclose(f);
            errno = EINVAL;
            return NULL;
        }
        file_version = version;
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
    wal->index_type = index_type;
    wal->version = file_version;
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

    uint32_t crc = gv_crc32_init();

    if (gv_wal_write_u8(wal->file, GV_WAL_TYPE_INSERT) != 0) return -1;
    crc = gv_crc32_update(crc, &(uint8_t){GV_WAL_TYPE_INSERT}, sizeof(uint8_t));
    if (gv_wal_write_u32(wal->file, (uint32_t)dimension) != 0) return -1;
    uint32_t dim_u32 = (uint32_t)dimension;
    crc = gv_crc32_update(crc, &dim_u32, sizeof(uint32_t));
    if (gv_wal_write_floats(wal->file, data, dimension) != 0) return -1;
    crc = gv_crc32_update(crc, data, dimension * sizeof(float));

    /* Allow single metadata pair; count=0 or 1 */
    uint32_t meta_count = (metadata_key != NULL && metadata_value != NULL) ? 1u : 0u;
    if (gv_wal_write_u32(wal->file, meta_count) != 0) return -1;
    crc = gv_crc32_update(crc, &meta_count, sizeof(uint32_t));
    if (meta_count == 1u) {
        if (gv_wal_write_string(wal->file, metadata_key) != 0) return -1;
        uint32_t klen = (uint32_t)strlen(metadata_key);
        crc = gv_crc32_update(crc, &klen, sizeof(uint32_t));
        crc = gv_crc32_update(crc, metadata_key, klen);
        if (gv_wal_write_string(wal->file, metadata_value) != 0) return -1;
        uint32_t vlen = (uint32_t)strlen(metadata_value);
        crc = gv_crc32_update(crc, &vlen, sizeof(uint32_t));
        crc = gv_crc32_update(crc, metadata_value, vlen);
    }

    if (wal->version >= 2) {
        crc = gv_crc32_finish(crc);
        if (gv_wal_write_u32(wal->file, crc) != 0) return -1;
    }

    if (fflush(wal->file) != 0) {
        return -1;
    }
    return 0;
}

int gv_wal_replay(const char *path, size_t expected_dimension,
                  int (*on_insert)(void *ctx, const float *data, size_t dimension,
                                   const char *metadata_key, const char *metadata_value),
                  void *ctx, uint32_t expected_index_type) {
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
    uint32_t file_index = 0;
    if (fread(magic, 1, 4, f) != 4 ||
        memcmp(magic, GV_WAL_MAGIC, 4) != 0 ||
        gv_wal_read_u32(f, &version) != 0 ||
        gv_wal_read_u32(f, &file_dim) != 0 ||
        (version != 1 && version != 2 && version != GV_WAL_VERSION) ||
        file_dim != (uint32_t)expected_dimension) {
        fclose(f);
        return -1;
    }
    if (version >= 3) {
        if (gv_wal_read_u32(f, &file_index) != 0) {
            fclose(f);
            return -1;
        }
        if (expected_index_type != 0 && file_index != 0 && file_index != expected_index_type) {
            fprintf(stderr, "WAL replay failed: index type mismatch (got %u expected %u)\n",
                    file_index, expected_index_type);
            fclose(f);
            return -1;
        }
    }
    int has_crc = (version >= 2);

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

            if (has_crc) {
                uint32_t crc = gv_crc32_init();
                uint8_t type_byte = GV_WAL_TYPE_INSERT;
                crc = gv_crc32_update(crc, &type_byte, sizeof(uint8_t));
                crc = gv_crc32_update(crc, &dim, sizeof(uint32_t));
                crc = gv_crc32_update(crc, buf, dim * sizeof(float));
                crc = gv_crc32_update(crc, &meta_count, sizeof(uint32_t));
                if (meta_count == 1 && k && v) {
                    uint32_t klen = (uint32_t)strlen(k);
                    uint32_t vlen = (uint32_t)strlen(v);
                    crc = gv_crc32_update(crc, &klen, sizeof(uint32_t));
                    crc = gv_crc32_update(crc, k, klen);
                    crc = gv_crc32_update(crc, &vlen, sizeof(uint32_t));
                    crc = gv_crc32_update(crc, v, vlen);
                }
                crc = gv_crc32_finish(crc);
                uint32_t stored_crc = 0;
                if (gv_wal_read_u32(f, &stored_crc) != 0 || stored_crc != crc) {
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

int gv_wal_dump(const char *path, size_t expected_dimension, uint32_t expected_index_type, FILE *out) {
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
    uint32_t file_index = 0;
    if (fread(magic, 1, 4, f) != 4 ||
        memcmp(magic, GV_WAL_MAGIC, 4) != 0 ||
        gv_wal_read_u32(f, &version) != 0 ||
        gv_wal_read_u32(f, &file_dim) != 0 ||
        (version != 1 && version != 2 && version != GV_WAL_VERSION) ||
        file_dim != (uint32_t)expected_dimension) {
        fclose(f);
        return -1;
    }
    if (version >= 3) {
        if (gv_wal_read_u32(f, &file_index) != 0) {
            fclose(f);
            return -1;
        }
        if (expected_index_type != 0 && file_index != 0 && file_index != expected_index_type) {
            fprintf(stderr, "WAL dump failed: index type mismatch (got %u expected %u)\n",
                    file_index, expected_index_type);
            fclose(f);
            return -1;
        }
    }

    fprintf(out, "WAL %s: version=%u dimension=%u index_type=%u\n", path, version, file_dim, file_index);
    int has_crc = (version >= 2);
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

            if (has_crc) {
                uint32_t crc = gv_crc32_init();
                uint8_t type_byte = GV_WAL_TYPE_INSERT;
                crc = gv_crc32_update(crc, &type_byte, sizeof(uint8_t));
                crc = gv_crc32_update(crc, &dim, sizeof(uint32_t));
                crc = gv_crc32_update(crc, buf, dim * sizeof(float));
                crc = gv_crc32_update(crc, &meta_count, sizeof(uint32_t));
                if (meta_count == 1 && k && v) {
                    uint32_t klen = (uint32_t)strlen(k);
                    uint32_t vlen = (uint32_t)strlen(v);
                    crc = gv_crc32_update(crc, &klen, sizeof(uint32_t));
                    crc = gv_crc32_update(crc, k, klen);
                    crc = gv_crc32_update(crc, &vlen, sizeof(uint32_t));
                    crc = gv_crc32_update(crc, v, vlen);
                }
                crc = gv_crc32_finish(crc);
                uint32_t stored_crc = 0;
                if (gv_wal_read_u32(f, &stored_crc) != 0 || stored_crc != crc) {
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


