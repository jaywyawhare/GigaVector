#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <unistd.h>
#else
#include <io.h>
#define fsync(fd) _commit(fd)
#endif

#include "storage/wal.h"
#include "core/utils.h"

#define GV_WAL_MAGIC "GVW1"
#define GV_WAL_VERSION 3u
#define GV_WAL_TYPE_INSERT 1u
#define GV_WAL_TYPE_DELETE 2u
#define GV_WAL_TYPE_UPDATE 3u
#define GV_WAL_TYPE_IVFDISK_APPEND 4u

struct GV_WAL {
    FILE *file;
    char *path;
    size_t dimension;
    uint32_t index_type;
    uint32_t version;
};


static int wal_sync(FILE *f) {
    if (fflush(f) != 0) return -1;
    if (fsync(fileno(f)) != 0) return -1;
    return 0;
}


GV_WAL *wal_open(const char *path, size_t dimension, uint32_t index_type) {
    if (path == NULL || dimension == 0) {
        return NULL;
    }

    uint32_t file_version = GV_WAL_VERSION;
    FILE *f = fopen(path, "ab+");
    if (f == NULL) {
        return NULL;
    }

    rewind(f);

    char magic[4] = {0};
    if (fread(magic, 1, 4, f) != 4) {
        rewind(f);
        if (fwrite(GV_WAL_MAGIC, 1, 4, f) != 4) {
            fclose(f);
            return NULL;
        }
        if (write_u32(f, GV_WAL_VERSION) != 0 ||
            write_u32(f, (uint32_t)dimension) != 0 ||
            write_u32(f, index_type) != 0) {
            fclose(f);
            return NULL;
        }
        wal_sync(f);
    } else {
        uint32_t version = 0;
        uint32_t file_dim = 0;
        uint32_t file_index = 0;
        if (memcmp(magic, GV_WAL_MAGIC, 4) != 0 ||
            read_u32(f, &version) != 0 ||
            read_u32(f, &file_dim) != 0) {
            fclose(f);
            return NULL;
        }
        if (version >= 3) {
            if (read_u32(f, &file_index) != 0) {
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
    wal->path = gv_dup_cstr(path);
    wal->index_type = index_type;
    wal->version = file_version;
    if (wal->path == NULL) {
        fclose(f);
        free(wal);
        return NULL;
    }
    return wal;
}

int wal_append_insert(GV_WAL *wal, const float *data, size_t dimension,
                         const char *metadata_key, const char *metadata_value) {
    if (wal == NULL || wal->file == NULL || data == NULL || dimension == 0) {
        return -1;
    }
    if (dimension != wal->dimension) {
        return -1;
    }

    uint32_t crc = gv_crc32_init();

    if (write_u8(wal->file, GV_WAL_TYPE_INSERT) != 0) return -1;
    crc = gv_crc32_update(crc, &(uint8_t){GV_WAL_TYPE_INSERT}, sizeof(uint8_t));
    if (write_u32(wal->file, (uint32_t)dimension) != 0) return -1;
    uint32_t dim_u32 = (uint32_t)dimension;
    crc = gv_crc32_update(crc, &dim_u32, sizeof(uint32_t));
    if (write_floats(wal->file, data, dimension) != 0) return -1;
    crc = gv_crc32_update(crc, data, dimension * sizeof(float));

    uint32_t meta_count = (metadata_key != NULL && metadata_value != NULL) ? 1u : 0u;
    if (write_u32(wal->file, meta_count) != 0) return -1;
    crc = gv_crc32_update(crc, &meta_count, sizeof(uint32_t));
    if (meta_count == 1u) {
        if (write_string(wal->file, metadata_key) != 0) return -1;
        uint32_t klen = (uint32_t)strlen(metadata_key);
        crc = gv_crc32_update(crc, &klen, sizeof(uint32_t));
        crc = gv_crc32_update(crc, metadata_key, klen);
        if (write_string(wal->file, metadata_value) != 0) return -1;
        uint32_t vlen = (uint32_t)strlen(metadata_value);
        crc = gv_crc32_update(crc, &vlen, sizeof(uint32_t));
        crc = gv_crc32_update(crc, metadata_value, vlen);
    }

    if (wal->version >= 2) {
        crc = gv_crc32_finish(crc);
        if (write_u32(wal->file, crc) != 0) return -1;
    }

    if (wal_sync(wal->file) != 0) {
        return -1;
    }
    return 0;
}

int wal_append_insert_rich(GV_WAL *wal, const float *data, size_t dimension,
                               const char *const *metadata_keys, const char *const *metadata_values,
                               size_t metadata_count) {
    if (wal == NULL || wal->file == NULL || data == NULL || dimension == 0) {
        return -1;
    }
    if (dimension != wal->dimension) {
        return -1;
    }
    if (metadata_count > 0 && (metadata_keys == NULL || metadata_values == NULL)) {
        return -1;
    }

    uint32_t crc = gv_crc32_init();

    if (write_u8(wal->file, GV_WAL_TYPE_INSERT) != 0) return -1;
    crc = gv_crc32_update(crc, &(uint8_t){GV_WAL_TYPE_INSERT}, sizeof(uint8_t));
    if (write_u32(wal->file, (uint32_t)dimension) != 0) return -1;
    uint32_t dim_u32 = (uint32_t)dimension;
    crc = gv_crc32_update(crc, &dim_u32, sizeof(uint32_t));
    if (write_floats(wal->file, data, dimension) != 0) return -1;
    crc = gv_crc32_update(crc, data, dimension * sizeof(float));

    uint32_t meta_count_u32 = (uint32_t)metadata_count;
    if (write_u32(wal->file, meta_count_u32) != 0) return -1;
    crc = gv_crc32_update(crc, &meta_count_u32, sizeof(uint32_t));
    
    for (size_t i = 0; i < metadata_count; i++) {
        if (metadata_keys[i] == NULL || metadata_values[i] == NULL) {
            return -1;
        }
        if (write_string(wal->file, metadata_keys[i]) != 0) return -1;
        uint32_t klen = (uint32_t)strlen(metadata_keys[i]);
        crc = gv_crc32_update(crc, &klen, sizeof(uint32_t));
        crc = gv_crc32_update(crc, metadata_keys[i], klen);
        if (write_string(wal->file, metadata_values[i]) != 0) return -1;
        uint32_t vlen = (uint32_t)strlen(metadata_values[i]);
        crc = gv_crc32_update(crc, &vlen, sizeof(uint32_t));
        crc = gv_crc32_update(crc, metadata_values[i], vlen);
    }

    if (wal->version >= 2) {
        crc = gv_crc32_finish(crc);
        if (write_u32(wal->file, crc) != 0) return -1;
    }

    if (wal_sync(wal->file) != 0) {
        return -1;
    }
    return 0;
}

int wal_append_delete(GV_WAL *wal, size_t vector_index) {
    if (wal == NULL || wal->file == NULL) {
        return -1;
    }

    uint32_t crc = gv_crc32_init();

    if (write_u8(wal->file, GV_WAL_TYPE_DELETE) != 0) return -1;
    crc = gv_crc32_update(crc, &(uint8_t){GV_WAL_TYPE_DELETE}, sizeof(uint8_t));
    
    uint64_t index_u64 = (uint64_t)vector_index;
    if (fwrite(&index_u64, sizeof(uint64_t), 1, wal->file) != 1) return -1;
    crc = gv_crc32_update(crc, &index_u64, sizeof(uint64_t));

    if (wal->version >= 2) {
        crc = gv_crc32_finish(crc);
        if (write_u32(wal->file, crc) != 0) return -1;
    }

    if (wal_sync(wal->file) != 0) return -1;
    return 0;
}

int wal_append_update(GV_WAL *wal, size_t vector_index, const float *data, size_t dimension,
                         const char *const *metadata_keys, const char *const *metadata_values,
                         size_t metadata_count) {
    if (wal == NULL || wal->file == NULL || data == NULL || dimension != wal->dimension) {
        return -1;
    }

    uint32_t crc = gv_crc32_init();

    if (write_u8(wal->file, GV_WAL_TYPE_UPDATE) != 0) return -1;
    crc = gv_crc32_update(crc, &(uint8_t){GV_WAL_TYPE_UPDATE}, sizeof(uint8_t));
    
    uint64_t index_u64 = (uint64_t)vector_index;
    if (fwrite(&index_u64, sizeof(uint64_t), 1, wal->file) != 1) return -1;
    crc = gv_crc32_update(crc, &index_u64, sizeof(uint64_t));
    
    if (write_u32(wal->file, (uint32_t)dimension) != 0) return -1;
    uint32_t dim_u32 = (uint32_t)dimension;
    crc = gv_crc32_update(crc, &dim_u32, sizeof(uint32_t));
    if (write_floats(wal->file, data, dimension) != 0) return -1;
    crc = gv_crc32_update(crc, data, dimension * sizeof(float));

    uint32_t meta_count_u32 = (uint32_t)metadata_count;
    if (write_u32(wal->file, meta_count_u32) != 0) return -1;
    crc = gv_crc32_update(crc, &meta_count_u32, sizeof(uint32_t));
    
    for (size_t i = 0; i < metadata_count; i++) {
        if (metadata_keys[i] == NULL || metadata_values[i] == NULL) {
            return -1;
        }
        if (write_string(wal->file, metadata_keys[i]) != 0) return -1;
        uint32_t klen = (uint32_t)strlen(metadata_keys[i]);
        crc = gv_crc32_update(crc, &klen, sizeof(uint32_t));
        crc = gv_crc32_update(crc, metadata_keys[i], klen);
        if (write_string(wal->file, metadata_values[i]) != 0) return -1;
        uint32_t vlen = (uint32_t)strlen(metadata_values[i]);
        crc = gv_crc32_update(crc, &vlen, sizeof(uint32_t));
        crc = gv_crc32_update(crc, metadata_values[i], vlen);
    }

    if (wal->version >= 2) {
        crc = gv_crc32_finish(crc);
        if (write_u32(wal->file, crc) != 0) return -1;
    }

    if (wal_sync(wal->file) != 0) return -1;
    return 0;
}

int wal_append_ivfdisk_append(GV_WAL *wal, uint64_t head_id, uint64_t vector_id,
                              const float *data, size_t dimension)
{
    if (!wal || !wal->file || !data || dimension == 0) return -1;
    if (dimension != wal->dimension) return -1;

    uint32_t crc = gv_crc32_init();
    if (write_u8(wal->file, GV_WAL_TYPE_IVFDISK_APPEND) != 0) return -1;
    crc = gv_crc32_update(crc, &(uint8_t){GV_WAL_TYPE_IVFDISK_APPEND}, sizeof(uint8_t));
    if (write_u64(wal->file, head_id) != 0) return -1;
    crc = gv_crc32_update(crc, &head_id, sizeof(uint64_t));
    if (write_u64(wal->file, vector_id) != 0) return -1;
    crc = gv_crc32_update(crc, &vector_id, sizeof(uint64_t));
    if (write_u32(wal->file, (uint32_t)dimension) != 0) return -1;
    uint32_t dim_u32 = (uint32_t)dimension;
    crc = gv_crc32_update(crc, &dim_u32, sizeof(uint32_t));
    if (write_floats(wal->file, data, dimension) != 0) return -1;
    crc = gv_crc32_update(crc, data, dimension * sizeof(float));
    if (wal->version >= 2) {
        crc = gv_crc32_finish(crc);
        if (write_u32(wal->file, crc) != 0) return -1;
    }
    return wal_sync(wal->file);
}

static int wal_skip_ivfdisk_append_record(FILE *f, int has_crc)
{
    uint64_t head_id = 0, vector_id = 0;
    uint32_t dim = 0;
    if (fread(&head_id, sizeof(uint64_t), 1, f) != 1) return -1;
    if (fread(&vector_id, sizeof(uint64_t), 1, f) != 1) return -1;
    if (read_u32(f, &dim) != 0) return -1;
    if (fseek(f, (long)(dim * sizeof(float)), SEEK_CUR) != 0) return -1;
    if (has_crc) {
        uint32_t stored_crc = 0;
        if (read_u32(f, &stored_crc) != 0) return -1;
    }
    (void)head_id;
    (void)vector_id;
    return 0;
}

int wal_replay(const char *path, size_t expected_dimension,
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
        read_u32(f, &version) != 0 ||
        read_u32(f, &file_dim) != 0 ||
        (version != 1 && version != 2 && version != GV_WAL_VERSION) ||
        file_dim != (uint32_t)expected_dimension) {
        fclose(f);
        return -1;
    }
    if (version >= 3) {
        if (read_u32(f, &file_index) != 0) {
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
        if (read_u8(f, &type) != 0) {
            if (feof(f)) break;
            fclose(f);
            return -1;
        }

        if (type == GV_WAL_TYPE_DELETE) {
            uint64_t index_u64 = 0;
            if (fread(&index_u64, sizeof(uint64_t), 1, f) != 1) {
                fclose(f);
                return -1;
            }
            if (has_crc) {
                uint32_t crc = gv_crc32_init();
                uint8_t type_byte = GV_WAL_TYPE_DELETE;
                crc = gv_crc32_update(crc, &type_byte, sizeof(uint8_t));
                crc = gv_crc32_update(crc, &index_u64, sizeof(uint64_t));
                crc = gv_crc32_finish(crc);
                uint32_t stored_crc = 0;
                if (read_u32(f, &stored_crc) != 0 || stored_crc != crc) {
                    fclose(f);
                    return -1;
                }
            }
            /* Skip delete records in replay - they are handled during load */
            continue;
        }

        if (type == GV_WAL_TYPE_IVFDISK_APPEND) {
            if (wal_skip_ivfdisk_append_record(f, has_crc) != 0) {
                fclose(f);
                return -1;
            }
            continue;
        }

        if (type == GV_WAL_TYPE_UPDATE) {
            uint64_t index_u64 = 0;
            if (fread(&index_u64, sizeof(uint64_t), 1, f) != 1) {
                fclose(f);
                return -1;
            }
            uint32_t dim = 0;
            if (read_u32(f, &dim) != 0 || dim != (uint32_t)expected_dimension) {
                fclose(f);
                return -1;
            }
            float *buf = (float *)malloc(sizeof(float) * dim);
            if (buf == NULL) {
                fclose(f);
                return -1;
            }
            if (read_floats(f, buf, dim) != 0) {
                free(buf);
                fclose(f);
                return -1;
            }
            uint32_t meta_count = 0;
            if (read_u32(f, &meta_count) != 0) {
                free(buf);
                fclose(f);
                return -1;
            }
            
            char **keys = NULL;
            char **values = NULL;
            if (meta_count > 65536) {
                free(buf);
                fclose(f);
                return -1;
            }
            if (meta_count > 0) {
                keys = (char **)malloc(sizeof(char *) * meta_count);
                values = (char **)malloc(sizeof(char *) * meta_count);
                if (keys == NULL || values == NULL) {
                    free(buf);
                    free(keys);
                    free(values);
                    fclose(f);
                    return -1;
                }
                for (uint32_t i = 0; i < meta_count; ++i) {
                    keys[i] = read_string(f);
                    values[i] = read_string(f);
                    if (keys[i] == NULL || values[i] == NULL) {
                        for (uint32_t j = 0; j <= i; ++j) {
                            free(keys[j]);
                            free(values[j]);
                        }
                        free(buf);
                        free(keys);
                        free(values);
                        fclose(f);
                        return -1;
                    }
                }
            }

            if (has_crc) {
                uint32_t crc = gv_crc32_init();
                uint8_t type_byte = GV_WAL_TYPE_UPDATE;
                crc = gv_crc32_update(crc, &type_byte, sizeof(uint8_t));
                crc = gv_crc32_update(crc, &index_u64, sizeof(uint64_t));
                crc = gv_crc32_update(crc, &dim, sizeof(uint32_t));
                crc = gv_crc32_update(crc, buf, dim * sizeof(float));
                crc = gv_crc32_update(crc, &meta_count, sizeof(uint32_t));
                for (uint32_t i = 0; i < meta_count; ++i) {
                    if (keys[i] && values[i]) {
                        uint32_t klen = (uint32_t)strlen(keys[i]);
                        uint32_t vlen = (uint32_t)strlen(values[i]);
                        crc = gv_crc32_update(crc, &klen, sizeof(uint32_t));
                        crc = gv_crc32_update(crc, keys[i], klen);
                        crc = gv_crc32_update(crc, &vlen, sizeof(uint32_t));
                        crc = gv_crc32_update(crc, values[i], vlen);
                    }
                }
                crc = gv_crc32_finish(crc);
                uint32_t stored_crc = 0;
                if (read_u32(f, &stored_crc) != 0 || stored_crc != crc) {
                    for (uint32_t i = 0; i < meta_count; ++i) {
                        free(keys[i]);
                        free(values[i]);
                    }
                    free(buf);
                    free(keys);
                    free(values);
                    fclose(f);
                    return -1;
                }
            }
            
            for (uint32_t i = 0; i < meta_count; ++i) {
                free(keys[i]);
                free(values[i]);
            }
            free(keys);
            free(values);

            free(buf);
            /* Skip update records in replay - they modify already-inserted vectors */
            continue;
        }

        if (type == GV_WAL_TYPE_INSERT) {
            uint32_t dim = 0;
            if (read_u32(f, &dim) != 0 || dim != (uint32_t)expected_dimension) {
                fclose(f);
                return -1;
            }
            float *buf = (float *)malloc(sizeof(float) * dim);
            if (buf == NULL) {
                fclose(f);
                return -1;
            }
            if (read_floats(f, buf, dim) != 0) {
                free(buf);
                fclose(f);
                return -1;
            }
            uint32_t meta_count = 0;
            if (read_u32(f, &meta_count) != 0) {
                free(buf);
                fclose(f);
                return -1;
            }
            
            char **keys = NULL;
            char **values = NULL;
            if (meta_count > 65536) {
                free(buf);
                fclose(f);
                return -1;
            }
            if (meta_count > 0) {
                keys = (char **)malloc(sizeof(char *) * meta_count);
                values = (char **)malloc(sizeof(char *) * meta_count);
                if (keys == NULL || values == NULL) {
                    free(buf);
                    free(keys);
                    free(values);
                    fclose(f);
                    return -1;
                }
                for (uint32_t i = 0; i < meta_count; i++) {
                    keys[i] = read_string(f);
                    values[i] = read_string(f);
                    if (keys[i] == NULL || values[i] == NULL) {
                        for (uint32_t j = 0; j <= i; j++) {
                            free(keys[j]);
                            free(values[j]);
                        }
                        free(buf);
                        free(keys);
                        free(values);
                        fclose(f);
                        return -1;
                    }
                }
            }

            if (has_crc) {
                uint32_t crc = gv_crc32_init();
                uint8_t type_byte = GV_WAL_TYPE_INSERT;
                crc = gv_crc32_update(crc, &type_byte, sizeof(uint8_t));
                crc = gv_crc32_update(crc, &dim, sizeof(uint32_t));
                crc = gv_crc32_update(crc, buf, dim * sizeof(float));
                crc = gv_crc32_update(crc, &meta_count, sizeof(uint32_t));
                for (uint32_t i = 0; i < meta_count; i++) {
                    if (keys[i] && values[i]) {
                        uint32_t klen = (uint32_t)strlen(keys[i]);
                        uint32_t vlen = (uint32_t)strlen(values[i]);
                        crc = gv_crc32_update(crc, &klen, sizeof(uint32_t));
                        crc = gv_crc32_update(crc, keys[i], klen);
                        crc = gv_crc32_update(crc, &vlen, sizeof(uint32_t));
                        crc = gv_crc32_update(crc, values[i], vlen);
                    }
                }
                crc = gv_crc32_finish(crc);
                uint32_t stored_crc = 0;
                if (read_u32(f, &stored_crc) != 0 || stored_crc != crc) {
                    for (uint32_t i = 0; i < meta_count; i++) {
                        free(keys[i]);
                        free(values[i]);
                    }
                    free(buf);
                    free(keys);
                    free(values);
                    fclose(f);
                    return -1;
                }
            }

            /* For backward compatibility: if single entry, use standard callback */
            /* For multiple entries, we need to handle it specially */
            int cb_res = 0;
            if (meta_count == 0) {
                cb_res = on_insert(ctx, buf, dim, NULL, NULL);
            } else if (meta_count == 1) {
                cb_res = on_insert(ctx, buf, dim, keys[0], values[0]);
            } else {
                /* Multiple metadata entries: call callback with first entry */
                /* The callback implementation should handle adding remaining entries */
                /* For now, we'll call it once per entry - the database code needs to */
                /* accumulate metadata for the same vector data */
                cb_res = on_insert(ctx, buf, dim, keys[0], values[0]);
                /* Note: This is a limitation - the callback signature only supports */
                /* one metadata entry. For full rich metadata support, the callback */
                /* would need to be updated or we need a different replay mechanism. */
                /* For now, only the first metadata entry will be replayed. */
            }
            
            for (uint32_t i = 0; i < meta_count; i++) {
                free(keys[i]);
                free(values[i]);
            }
            free(buf);
            free(keys);
            free(values);
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

int wal_replay_rich(const char *path, size_t expected_dimension,
                       int (*on_insert)(void *ctx, const float *data, size_t dimension,
                                        const char *const *metadata_keys, const char *const *metadata_values,
                                        size_t metadata_count),
                       int (*on_delete)(void *ctx, size_t vector_index),
                       int (*on_update)(void *ctx, size_t vector_index, const float *data,
                                        size_t dimension, const char *const *metadata_keys,
                                        const char *const *metadata_values, size_t metadata_count),
                       int (*on_ivfdisk_append)(void *ctx, uint64_t head_id, uint64_t vector_id,
                                                const float *data, size_t dimension),
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
        read_u32(f, &version) != 0 ||
        read_u32(f, &file_dim) != 0 ||
        (version != 1 && version != 2 && version != GV_WAL_VERSION) ||
        file_dim != (uint32_t)expected_dimension) {
        fclose(f);
        return -1;
    }
    if (version >= 3) {
        if (read_u32(f, &file_index) != 0) {
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
        if (read_u8(f, &type) != 0) {
            if (feof(f)) break;
            fclose(f);
            return -1;
        }

        if (type == GV_WAL_TYPE_DELETE) {
            uint64_t index_u64 = 0;
            if (fread(&index_u64, sizeof(uint64_t), 1, f) != 1) {
                fclose(f);
                return -1;
            }
            if (has_crc) {
                uint32_t crc = gv_crc32_init();
                uint8_t type_byte = GV_WAL_TYPE_DELETE;
                crc = gv_crc32_update(crc, &type_byte, sizeof(uint8_t));
                crc = gv_crc32_update(crc, &index_u64, sizeof(uint64_t));
                crc = gv_crc32_finish(crc);
                uint32_t stored_crc = 0;
                if (read_u32(f, &stored_crc) != 0 || stored_crc != crc) {
                    fclose(f);
                    return -1;
                }
            }
            if (on_delete != NULL) {
                int cb_res = on_delete(ctx, (size_t)index_u64);
                if (cb_res != 0) { fclose(f); return -1; }
            }
            continue;
        }

        if (type == GV_WAL_TYPE_IVFDISK_APPEND) {
            uint64_t head_id = 0, vector_id = 0;
            uint32_t dim = 0;
            if (fread(&head_id, sizeof(uint64_t), 1, f) != 1 ||
                fread(&vector_id, sizeof(uint64_t), 1, f) != 1 ||
                read_u32(f, &dim) != 0 ||
                dim != (uint32_t)expected_dimension) {
                fclose(f);
                return -1;
            }
            float *buf = (float *)malloc((size_t)dim * sizeof(float));
            if (!buf) { fclose(f); return -1; }
            if (read_floats(f, buf, dim) != 0) {
                free(buf);
                fclose(f);
                return -1;
            }
            if (has_crc) {
                uint32_t crc = gv_crc32_init();
                uint8_t type_byte = GV_WAL_TYPE_IVFDISK_APPEND;
                crc = gv_crc32_update(crc, &type_byte, sizeof(uint8_t));
                crc = gv_crc32_update(crc, &head_id, sizeof(uint64_t));
                crc = gv_crc32_update(crc, &vector_id, sizeof(uint64_t));
                crc = gv_crc32_update(crc, &dim, sizeof(uint32_t));
                crc = gv_crc32_update(crc, buf, dim * sizeof(float));
                crc = gv_crc32_finish(crc);
                uint32_t stored_crc = 0;
                if (read_u32(f, &stored_crc) != 0 || stored_crc != crc) {
                    free(buf);
                    fclose(f);
                    return -1;
                }
            }
            if (on_ivfdisk_append != NULL) {
                int cb_res = on_ivfdisk_append(ctx, head_id, vector_id, buf, dim);
                if (cb_res != 0) {
                    free(buf);
                    fclose(f);
                    return -1;
                }
            }
            free(buf);
            continue;
        }

        if (type == GV_WAL_TYPE_UPDATE) {
            uint64_t index_u64 = 0;
            if (fread(&index_u64, sizeof(uint64_t), 1, f) != 1) {
                fclose(f);
                return -1;
            }
            uint32_t dim = 0;
            if (read_u32(f, &dim) != 0 || dim != (uint32_t)expected_dimension) {
                fclose(f);
                return -1;
            }
            float *buf = (float *)malloc(sizeof(float) * dim);
            if (buf == NULL) {
                fclose(f);
                return -1;
            }
            if (read_floats(f, buf, dim) != 0) {
                free(buf);
                fclose(f);
                return -1;
            }
            uint32_t meta_count = 0;
            if (read_u32(f, &meta_count) != 0) {
                free(buf);
                fclose(f);
                return -1;
            }
            
            char **keys = NULL;
            char **values = NULL;
            if (meta_count > 65536) {
                free(buf);
                fclose(f);
                return -1;
            }
            if (meta_count > 0) {
                keys = (char **)malloc(sizeof(char *) * meta_count);
                values = (char **)malloc(sizeof(char *) * meta_count);
                if (keys == NULL || values == NULL) {
                    free(buf);
                    free(keys);
                    free(values);
                    fclose(f);
                    return -1;
                }
                for (uint32_t i = 0; i < meta_count; ++i) {
                    keys[i] = read_string(f);
                    values[i] = read_string(f);
                    if (keys[i] == NULL || values[i] == NULL) {
                        for (uint32_t j = 0; j <= i; ++j) {
                            free(keys[j]);
                            free(values[j]);
                        }
                        free(buf);
                        free(keys);
                        free(values);
                        fclose(f);
                        return -1;
                    }
                }
            }

            if (has_crc) {
                uint32_t crc = gv_crc32_init();
                uint8_t type_byte = GV_WAL_TYPE_UPDATE;
                crc = gv_crc32_update(crc, &type_byte, sizeof(uint8_t));
                crc = gv_crc32_update(crc, &index_u64, sizeof(uint64_t));
                crc = gv_crc32_update(crc, &dim, sizeof(uint32_t));
                crc = gv_crc32_update(crc, buf, dim * sizeof(float));
                crc = gv_crc32_update(crc, &meta_count, sizeof(uint32_t));
                for (uint32_t i = 0; i < meta_count; ++i) {
                    if (keys[i] && values[i]) {
                        uint32_t klen = (uint32_t)strlen(keys[i]);
                        uint32_t vlen = (uint32_t)strlen(values[i]);
                        crc = gv_crc32_update(crc, &klen, sizeof(uint32_t));
                        crc = gv_crc32_update(crc, keys[i], klen);
                        crc = gv_crc32_update(crc, &vlen, sizeof(uint32_t));
                        crc = gv_crc32_update(crc, values[i], vlen);
                    }
                }
                crc = gv_crc32_finish(crc);
                uint32_t stored_crc = 0;
                if (read_u32(f, &stored_crc) != 0 || stored_crc != crc) {
                    for (uint32_t i = 0; i < meta_count; ++i) {
                        free(keys[i]);
                        free(values[i]);
                    }
                    free(buf);
                    free(keys);
                    free(values);
                    fclose(f);
                    return -1;
                }
            }
            
            if (on_update != NULL) {
                int cb_res = on_update(ctx, (size_t)index_u64, buf, dim,
                    (const char *const *)keys, (const char *const *)values, meta_count);
                for (uint32_t i = 0; i < meta_count; ++i) {
                    free(keys[i]);
                    free(values[i]);
                }
                free(keys);
                free(values);
                free(buf);
                if (cb_res != 0) { fclose(f); return -1; }
            } else {
                for (uint32_t i = 0; i < meta_count; ++i) {
                    free(keys[i]);
                    free(values[i]);
                }
                free(keys);
                free(values);
                free(buf);
            }
            continue;
        }

        if (type == GV_WAL_TYPE_INSERT) {
            uint32_t dim = 0;
            if (read_u32(f, &dim) != 0 || dim != (uint32_t)expected_dimension) {
                fclose(f);
                return -1;
            }
            float *buf = (float *)malloc(sizeof(float) * dim);
            if (buf == NULL) {
                fclose(f);
                return -1;
            }
            if (read_floats(f, buf, dim) != 0) {
                free(buf);
                fclose(f);
                return -1;
            }
            uint32_t meta_count = 0;
            if (read_u32(f, &meta_count) != 0) {
                free(buf);
                fclose(f);
                return -1;
            }

            char **keys = NULL;
            char **values = NULL;
            if (meta_count > 65536) {
                free(buf);
                fclose(f);
                return -1;
            }
            if (meta_count > 0) {
                keys = (char **)malloc(sizeof(char *) * meta_count);
                values = (char **)malloc(sizeof(char *) * meta_count);
                if (keys == NULL || values == NULL) {
                    free(buf);
                    free(keys);
                    free(values);
                    fclose(f);
                    return -1;
                }
                for (uint32_t i = 0; i < meta_count; i++) {
                    keys[i] = read_string(f);
                    values[i] = read_string(f);
                    if (keys[i] == NULL || values[i] == NULL) {
                        for (uint32_t j = 0; j <= i; j++) {
                            free(keys[j]);
                            free(values[j]);
                        }
                        free(buf);
                        free(keys);
                        free(values);
                        fclose(f);
                        return -1;
                    }
                }
            }

            if (has_crc) {
                uint32_t crc = gv_crc32_init();
                uint8_t type_byte = GV_WAL_TYPE_INSERT;
                crc = gv_crc32_update(crc, &type_byte, sizeof(uint8_t));
                crc = gv_crc32_update(crc, &dim, sizeof(uint32_t));
                crc = gv_crc32_update(crc, buf, dim * sizeof(float));
                crc = gv_crc32_update(crc, &meta_count, sizeof(uint32_t));
                for (uint32_t i = 0; i < meta_count; i++) {
                    if (keys[i] && values[i]) {
                        uint32_t klen = (uint32_t)strlen(keys[i]);
                        uint32_t vlen = (uint32_t)strlen(values[i]);
                        crc = gv_crc32_update(crc, &klen, sizeof(uint32_t));
                        crc = gv_crc32_update(crc, keys[i], klen);
                        crc = gv_crc32_update(crc, &vlen, sizeof(uint32_t));
                        crc = gv_crc32_update(crc, values[i], vlen);
                    }
                }
                crc = gv_crc32_finish(crc);
                uint32_t stored_crc = 0;
                if (read_u32(f, &stored_crc) != 0 || stored_crc != crc) {
                    for (uint32_t i = 0; i < meta_count; i++) {
                        free(keys[i]);
                        free(values[i]);
                    }
                    free(buf);
                    free(keys);
                    free(values);
                    fclose(f);
                    return -1;
                }
            }

            int cb_res = on_insert(ctx, buf, dim, (const char *const *)keys, (const char *const *)values, meta_count);

            for (uint32_t i = 0; i < meta_count; i++) {
                free(keys[i]);
                free(values[i]);
            }
            free(keys);
            free(values);
            free(buf);
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

int wal_dump(const char *path, size_t expected_dimension, uint32_t expected_index_type, FILE *out) {
    if (path == NULL || expected_dimension == 0 || out == NULL) {
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
        read_u32(f, &version) != 0 ||
        read_u32(f, &file_dim) != 0 ||
        (version != 1 && version != 2 && version != GV_WAL_VERSION) ||
        file_dim != (uint32_t)expected_dimension) {
        fclose(f);
        return -1;
    }
    if (version >= 3) {
        if (read_u32(f, &file_index) != 0) {
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
        if (read_u8(f, &type) != 0) {
            if (feof(f)) break;
            fclose(f);
            return -1;
        }

        if (type == GV_WAL_TYPE_INSERT) {
            uint32_t dim = 0;
            if (read_u32(f, &dim) != 0 || dim != (uint32_t)expected_dimension) {
                fclose(f);
                return -1;
            }
            float *buf = (float *)malloc(sizeof(float) * dim);
            if (buf == NULL) {
                fclose(f);
                return -1;
            }
            if (read_floats(f, buf, dim) != 0) {
                free(buf);
                fclose(f);
                return -1;
            }
            uint32_t meta_count = 0;
            if (read_u32(f, &meta_count) != 0) {
                free(buf);
                fclose(f);
                return -1;
            }
            
            char **keys = NULL;
            char **values = NULL;
            if (meta_count > 65536) {
                free(buf);
                fclose(f);
                return -1;
            }
            if (meta_count > 0) {
                keys = (char **)malloc(sizeof(char *) * meta_count);
                values = (char **)malloc(sizeof(char *) * meta_count);
                if (keys == NULL || values == NULL) {
                    free(buf);
                    free(keys);
                    free(values);
                    fclose(f);
                    return -1;
                }
                for (uint32_t i = 0; i < meta_count; i++) {
                    keys[i] = read_string(f);
                    values[i] = read_string(f);
                    if (keys[i] == NULL || values[i] == NULL) {
                        for (uint32_t j = 0; j <= i; j++) {
                            free(keys[j]);
                            free(values[j]);
                        }
                        free(buf);
                        free(keys);
                        free(values);
                        fclose(f);
                        return -1;
                    }
                }
            }

            if (has_crc) {
                uint32_t crc = gv_crc32_init();
                uint8_t type_byte = GV_WAL_TYPE_INSERT;
                crc = gv_crc32_update(crc, &type_byte, sizeof(uint8_t));
                crc = gv_crc32_update(crc, &dim, sizeof(uint32_t));
                crc = gv_crc32_update(crc, buf, dim * sizeof(float));
                crc = gv_crc32_update(crc, &meta_count, sizeof(uint32_t));
                for (uint32_t i = 0; i < meta_count; i++) {
                    if (keys[i] && values[i]) {
                        uint32_t klen = (uint32_t)strlen(keys[i]);
                        uint32_t vlen = (uint32_t)strlen(values[i]);
                        crc = gv_crc32_update(crc, &klen, sizeof(uint32_t));
                        crc = gv_crc32_update(crc, keys[i], klen);
                        crc = gv_crc32_update(crc, &vlen, sizeof(uint32_t));
                        crc = gv_crc32_update(crc, values[i], vlen);
                    }
                }
                crc = gv_crc32_finish(crc);
                uint32_t stored_crc = 0;
                if (read_u32(f, &stored_crc) != 0 || stored_crc != crc) {
                    for (uint32_t i = 0; i < meta_count; i++) {
                        free(keys[i]);
                        free(values[i]);
                    }
                    free(buf);
                    free(keys);
                    free(values);
                    fclose(f);
                    return -1;
                }
            }

            fprintf(out, "#%zu INSERT dim=%u first=%.6f", record_index, dim, buf[0]);
            if (dim > 1) {
                fprintf(out, " second=%.6f", buf[1]);
            }
            for (uint32_t i = 0; i < meta_count; i++) {
                if (keys[i] && values[i]) {
                    fprintf(out, " meta[%s]=%s", keys[i], values[i]);
                }
            }
            fprintf(out, "\n");

            for (uint32_t i = 0; i < meta_count; i++) {
                free(keys[i]);
                free(values[i]);
            }
            free(buf);
            free(keys);
            free(values);
            record_index++;
        } else {
            fclose(f);
            return -1;
        }
    }

    fclose(f);
    return 0;
}

void wal_close(GV_WAL *wal) {
    if (wal == NULL) {
        return;
    }
    if (wal->file) {
        wal_sync(wal->file);
        fclose(wal->file);
    }
    free(wal->path);
    free(wal);
}

int wal_reset(const char *path) {
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

int wal_truncate(GV_WAL *wal) {
    if (wal == NULL || wal->path == NULL) {
        return -1;
    }

    if (wal->file != NULL) {
        wal_sync(wal->file);
        fclose(wal->file);
        wal->file = NULL;
    }

    FILE *f = fopen(wal->path, "wb");
    if (f == NULL) {
        return -1;
    }

    if (fwrite(GV_WAL_MAGIC, 1, 4, f) != 4) {
        fclose(f);
        return -1;
    }
    if (write_u32(f, wal->version) != 0 ||
        write_u32(f, (uint32_t)wal->dimension) != 0 ||
        write_u32(f, wal->index_type) != 0) {
        fclose(f);
        return -1;
    }

    if (wal_sync(f) != 0 || fclose(f) != 0) {
        return -1;
    }

    wal->file = fopen(wal->path, "ab+");
    if (wal->file == NULL) {
        return -1;
    }

    fseek(wal->file, 0, SEEK_END);

    return 0;
}

static int wal_record_has_crc(uint32_t version) {
    return version >= 2;
}

static int wal_skip_record_from_file(FILE *f, int has_crc, long *record_start, size_t *record_len) {
    if (record_start) *record_start = ftell(f);
    uint8_t type = 0;
    if (read_u8(f, &type) != 0) return -1;

    if (type == GV_WAL_TYPE_DELETE) {
        if (fseek(f, (long)sizeof(uint64_t), SEEK_CUR) != 0) return -1;
        if (has_crc) {
            if (fseek(f, (long)sizeof(uint32_t), SEEK_CUR) != 0) return -1;
        }
    } else if (type == GV_WAL_TYPE_UPDATE || type == GV_WAL_TYPE_INSERT) {
        if (type == GV_WAL_TYPE_UPDATE) {
            if (fseek(f, (long)sizeof(uint64_t), SEEK_CUR) != 0) return -1;
        }
        uint32_t dim = 0;
        if (read_u32(f, &dim) != 0) return -1;
        if (fseek(f, (long)(dim * sizeof(float)), SEEK_CUR) != 0) return -1;
        uint32_t meta_count = 0;
        if (read_u32(f, &meta_count) != 0) return -1;
        for (uint32_t i = 0; i < meta_count; i++) {
            char *k = read_string(f);
            char *v = read_string(f);
            free(k);
            free(v);
            if (k == NULL || v == NULL) return -1;
        }
        if (has_crc) {
            if (fseek(f, (long)sizeof(uint32_t), SEEK_CUR) != 0) return -1;
        }
    } else if (type == GV_WAL_TYPE_IVFDISK_APPEND) {
        if (fseek(f, (long)(2 * sizeof(uint64_t)), SEEK_CUR) != 0) return -1;
        uint32_t dim = 0;
        if (read_u32(f, &dim) != 0) return -1;
        if (fseek(f, (long)(dim * sizeof(float)), SEEK_CUR) != 0) return -1;
        if (has_crc) {
            if (fseek(f, (long)sizeof(uint32_t), SEEK_CUR) != 0) return -1;
        }
    } else {
        return -1;
    }

    if (record_len) {
        long end = ftell(f);
        if (record_start && end >= *record_start) {
            *record_len = (size_t)(end - *record_start);
        } else {
            *record_len = 0;
        }
    }
    return 0;
}

uint64_t wal_count_entries(const char *path) {
    if (!path) return 0;

    FILE *f = fopen(path, "rb");
    if (!f) return (errno == ENOENT) ? 0 : 0;

    char magic[4];
    uint32_t version = 0, file_dim = 0;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, GV_WAL_MAGIC, 4) != 0 ||
        read_u32(f, &version) != 0 || read_u32(f, &file_dim) != 0) {
        fclose(f);
        return 0;
    }
    if (version >= 3) {
        uint32_t file_index = 0;
        if (read_u32(f, &file_index) != 0) {
            fclose(f);
            return 0;
        }
        (void)file_index;
    }

    int has_crc = wal_record_has_crc(version);
    uint64_t count = 0;
    while (1) {
        long start = ftell(f);
        if (start < 0) break;
        uint8_t type = 0;
        if (fread(&type, 1, 1, f) != 1) break;
        fseek(f, start, SEEK_SET);
        if (wal_skip_record_from_file(f, has_crc, NULL, NULL) != 0) break;
        count++;
    }
    fclose(f);
    return count;
}

int wal_read_entry_at(const char *path, uint64_t entry_index, uint8_t *out_type,
                      uint8_t **out_buf, size_t *out_len) {
    if (!path || !out_buf || !out_len) return -1;
    *out_buf = NULL;
    *out_len = 0;
    if (out_type) *out_type = 0;

    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    char magic[4];
    uint32_t version = 0, file_dim = 0;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, GV_WAL_MAGIC, 4) != 0 ||
        read_u32(f, &version) != 0 || read_u32(f, &file_dim) != 0) {
        fclose(f);
        return -1;
    }
    if (version >= 3) {
        uint32_t file_index = 0;
        if (read_u32(f, &file_index) != 0) {
            fclose(f);
            return -1;
        }
        (void)file_index;
    }

    int has_crc = wal_record_has_crc(version);
    uint64_t idx = 0;
    while (1) {
        long start = 0;
        size_t len = 0;
        if (wal_skip_record_from_file(f, has_crc, &start, &len) != 0) {
            fclose(f);
            return -1;
        }
        if (idx == entry_index) {
            if (fseek(f, start, SEEK_SET) != 0) {
                fclose(f);
                return -1;
            }
            uint8_t *buf = (uint8_t *)malloc(len);
            if (!buf) {
                fclose(f);
                return -1;
            }
            if (fread(buf, 1, len, f) != len) {
                free(buf);
                fclose(f);
                return -1;
            }
            *out_buf = buf;
            *out_len = len;
            if (out_type) *out_type = buf[0];
            fclose(f);
            return 0;
        }
        idx++;
    }
    fclose(f);
    return -1;
}

int wal_append_raw(GV_WAL *wal, const uint8_t *record, size_t len) {
    if (!wal || !wal->file || !record || len == 0) return -1;
    if (fwrite(record, 1, len, wal->file) != len) return -1;
    return wal_sync(wal->file);
}

int wal_apply_record_buffer(const uint8_t *record, size_t len, int has_crc,
                            size_t expected_dimension,
                            int (*on_insert)(void *ctx, const float *data, size_t dimension,
                                             const char *const *metadata_keys,
                                             const char *const *metadata_values,
                                             size_t metadata_count),
                            int (*on_delete)(void *ctx, size_t vector_index),
                            int (*on_update)(void *ctx, size_t vector_index,
                                             const float *data, size_t dimension,
                                             const char *const *metadata_keys,
                                             const char *const *metadata_values,
                                             size_t metadata_count),
                            int (*on_ivfdisk_append)(void *ctx, uint64_t head_id, uint64_t vector_id,
                                                     const float *data, size_t dimension),
                            void *ctx) {
    if (!record || len == 0) return -1;
    size_t pos = 0;
    uint8_t type = record[pos++];

    if (type == GV_WAL_TYPE_DELETE) {
        if (len < 1 + sizeof(uint64_t) + (has_crc ? sizeof(uint32_t) : 0)) return -1;
        uint64_t index_u64 = 0;
        memcpy(&index_u64, record + pos, sizeof(uint64_t));
        if (on_delete) return on_delete(ctx, (size_t)index_u64);
        return 0;
    }

    if (type == GV_WAL_TYPE_IVFDISK_APPEND) {
        if (len < 1 + 2 * sizeof(uint64_t) + sizeof(uint32_t)) return -1;
        uint64_t head_id = 0, vector_id = 0;
        memcpy(&head_id, record + pos, sizeof(uint64_t));
        pos += sizeof(uint64_t);
        memcpy(&vector_id, record + pos, sizeof(uint64_t));
        pos += sizeof(uint64_t);
        uint32_t dim = 0;
        memcpy(&dim, record + pos, sizeof(uint32_t));
        pos += sizeof(uint32_t);
        if (dim != (uint32_t)expected_dimension) return -1;
        if (len < pos + dim * sizeof(float) + (has_crc ? sizeof(uint32_t) : 0)) return -1;
        float *vec = (float *)malloc((size_t)dim * sizeof(float));
        if (!vec) return -1;
        memcpy(vec, record + pos, (size_t)dim * sizeof(float));
        int rc = 0;
        if (on_ivfdisk_append) {
            rc = on_ivfdisk_append(ctx, head_id, vector_id, vec, dim);
        }
        free(vec);
        return rc;
    }

    if (type != GV_WAL_TYPE_INSERT && type != GV_WAL_TYPE_UPDATE) return -1;

    uint64_t vector_index = 0;
    if (type == GV_WAL_TYPE_UPDATE) {
        if (len < 1 + sizeof(uint64_t) + sizeof(uint32_t)) return -1;
        memcpy(&vector_index, record + pos, sizeof(uint64_t));
        pos += sizeof(uint64_t);
    }

    if (len < pos + sizeof(uint32_t)) return -1;
    uint32_t dim = 0;
    memcpy(&dim, record + pos, sizeof(uint32_t));
    pos += sizeof(uint32_t);
    if (dim != (uint32_t)expected_dimension) return -1;
    if (len < pos + dim * sizeof(float) + sizeof(uint32_t)) return -1;

    float *vec = (float *)malloc((size_t)dim * sizeof(float));
    if (!vec) return -1;
    memcpy(vec, record + pos, (size_t)dim * sizeof(float));
    pos += dim * sizeof(float);

    uint32_t meta_count = 0;
    memcpy(&meta_count, record + pos, sizeof(uint32_t));
    pos += sizeof(uint32_t);
    if (meta_count > 65536) {
        free(vec);
        return -1;
    }

    char **keys = NULL;
    char **values = NULL;
    if (meta_count > 0) {
        keys = (char **)calloc(meta_count, sizeof(char *));
        values = (char **)calloc(meta_count, sizeof(char *));
        if (!keys || !values) {
            free(keys);
            free(values);
            free(vec);
            return -1;
        }
        for (uint32_t i = 0; i < meta_count; i++) {
            if (pos + sizeof(uint32_t) > len) goto fail;
            uint32_t klen = 0;
            memcpy(&klen, record + pos, sizeof(uint32_t));
            pos += sizeof(uint32_t);
            if (klen > 1024 * 1024) goto fail;
            if (pos + klen > len) goto fail;
            keys[i] = (char *)malloc(klen + 1);
            if (!keys[i]) goto fail;
            memcpy(keys[i], record + pos, klen);
            keys[i][klen] = '\0';
            pos += klen;

            if (pos + sizeof(uint32_t) > len) goto fail;
            uint32_t vlen = 0;
            memcpy(&vlen, record + pos, sizeof(uint32_t));
            pos += sizeof(uint32_t);
            if (vlen > 1024 * 1024) goto fail;
            if (pos + vlen > len) goto fail;
            values[i] = (char *)malloc(vlen + 1);
            if (!values[i]) goto fail;
            memcpy(values[i], record + pos, vlen);
            values[i][vlen] = '\0';
            pos += vlen;
        }
    }
    if (has_crc) pos += sizeof(uint32_t);
    (void)pos;

    int rc = -1;
    if (type == GV_WAL_TYPE_INSERT) {
        rc = on_insert ? on_insert(ctx, vec, dim,
                                   (const char *const *)keys,
                                   (const char *const *)values,
                                   meta_count) : 0;
    } else if (on_update) {
        rc = on_update(ctx, (size_t)vector_index, vec, dim,
                       (const char *const *)keys,
                       (const char *const *)values,
                       meta_count);
    }

    for (uint32_t i = 0; i < meta_count; i++) {
        free(keys[i]);
        free(values[i]);
    }
    free(keys);
    free(values);
    free(vec);
    return rc;

fail:
    if (keys && values) {
        for (uint32_t i = 0; i < meta_count; i++) {
            free(keys[i]);
            free(values[i]);
        }
    }
    free(keys);
    free(values);
    free(vec);
    return -1;
}

