#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gigavector/gv_database.h"
#include "gigavector/gv_distance.h"
#include "gigavector/gv_hnsw.h"
#include "gigavector/gv_kdtree.h"
#include "gigavector/gv_metadata.h"
#include "gigavector/gv_vector.h"
#include "gigavector/gv_wal.h"

static char *gv_db_strdup(const char *src) {
    if (src == NULL) {
        return NULL;
    }
    size_t len = strlen(src) + 1;
    char *copy = (char *)malloc(len);
    if (copy == NULL) {
        return NULL;
    }
    memcpy(copy, src, len);
    return copy;
}

static int gv_db_write_header(FILE *out, uint32_t dimension, uint64_t count, uint32_t version) {
    const uint32_t magic = 0x47564442; /* "GVDB" in hex */
    if (fwrite(&magic, sizeof(uint32_t), 1, out) != 1) {
        return -1;
    }
    if (fwrite(&version, sizeof(uint32_t), 1, out) != 1) {
        return -1;
    }
    if (fwrite(&dimension, sizeof(uint32_t), 1, out) != 1) {
        return -1;
    }
    if (fwrite(&count, sizeof(uint64_t), 1, out) != 1) {
        return -1;
    }
    return 0;
}

static int gv_db_read_header(FILE *in, uint32_t *dimension_out, uint64_t *count_out, uint32_t *version_out) {
    uint32_t magic = 0;
    uint32_t version = 0;
    if (fread(&magic, sizeof(uint32_t), 1, in) != 1) {
        return -1;
    }
    if (fread(&version, sizeof(uint32_t), 1, in) != 1) {
        return -1;
    }
    if (magic != 0x47564442 /* "GVDB" */) {
        return -1;
    }
    if (fread(dimension_out, sizeof(uint32_t), 1, in) != 1) {
        return -1;
    }
    if (fread(count_out, sizeof(uint64_t), 1, in) != 1) {
        return -1;
    }
    if (version_out != NULL) {
        *version_out = version;
    }
    return 0;
}

static int gv_write_uint32(FILE *out, uint32_t value) {
    return (fwrite(&value, sizeof(uint32_t), 1, out) == 1) ? 0 : -1;
}

static int gv_read_uint32(FILE *in, uint32_t *value) {
    return (value != NULL && fread(value, sizeof(uint32_t), 1, in) == 1) ? 0 : -1;
}

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

static char *gv_db_build_wal_path(const char *filepath) {
    if (filepath == NULL) {
        return NULL;
    }

    const char *dir_override = getenv("GV_WAL_DIR");
    const char *basename = strrchr(filepath, '/');
    basename = (basename == NULL) ? filepath : basename + 1;

    char buf[1024];
    if (dir_override != NULL && dir_override[0] != '\0') {
        snprintf(buf, sizeof(buf), "%s/%s.wal", dir_override, basename);
    } else {
        snprintf(buf, sizeof(buf), "%s.wal", filepath);
    }

    return gv_db_strdup(buf);
}

static int gv_db_wal_apply(void *ctx, const float *data, size_t dimension,
                           const char *metadata_key, const char *metadata_value) {
    GV_Database *db = (GV_Database *)ctx;
    (void)dimension;
    if (gv_db_add_vector_with_metadata(db, data, db->dimension, metadata_key, metadata_value) != 0) {
        return -1;
    }
    return 0;
}

GV_Database *gv_db_open(const char *filepath, size_t dimension, GV_IndexType index_type) {
    if (dimension == 0 && filepath == NULL) {
        return NULL;
    }

    GV_Database *db = (GV_Database *)malloc(sizeof(GV_Database));
    if (db == NULL) {
        return NULL;
    }

    db->dimension = dimension;
    db->index_type = index_type;
    db->root = NULL;
    db->hnsw_index = NULL;
    db->filepath = NULL;
    db->wal_path = NULL;
    db->wal = NULL;
    db->wal_replaying = 0;
    db->count = 0;

    if (index_type == GV_INDEX_TYPE_HNSW && filepath == NULL) {
        db->hnsw_index = gv_hnsw_create(dimension, NULL);
        if (db->hnsw_index == NULL) {
            free(db);
            return NULL;
        }
    }

    if (filepath != NULL) {
        db->filepath = gv_db_strdup(filepath);
        if (db->filepath == NULL) {
            free(db);
            return NULL;
        }

        db->wal_path = gv_db_build_wal_path(filepath);
        if (db->wal_path == NULL) {
            free(db->filepath);
            free(db);
            return NULL;
        }
    }

    if (filepath == NULL) {
        if (db->wal_path != NULL) {
            db->wal = gv_wal_open(db->wal_path, db->dimension);
        }
        return db;
    }

    FILE *in = fopen(filepath, "rb");
    if (in == NULL) {
        if (errno == ENOENT) {
            if (index_type == GV_INDEX_TYPE_HNSW) {
                db->hnsw_index = gv_hnsw_create(dimension, NULL);
                if (db->hnsw_index == NULL) {
                    free(db->filepath);
                    free(db->wal_path);
                    free(db);
                    return NULL;
                }
            }
            if (db->wal_path != NULL) {
                db->wal = gv_wal_open(db->wal_path, db->dimension);
                if (db->wal == NULL) {
                    if (db->hnsw_index) gv_hnsw_destroy(db->hnsw_index);
                    free(db->filepath);
                    free(db->wal_path);
                    free(db);
                    return NULL;
                }
            }
            return db;
        }
        free(db->filepath);
        free(db->wal_path);
        free(db);
        return NULL;
    }

    uint32_t file_dim = 0;
    uint64_t file_count = 0;
    uint32_t file_version = 0;
    if (gv_db_read_header(in, &file_dim, &file_count, &file_version) != 0) {
        fclose(in);
        free(db->filepath);
        free(db->wal_path);
        free(db);
        return NULL;
    }

    if (dimension != 0 && dimension != (size_t)file_dim) {
        fclose(in);
        free(db->filepath);
        free(db->wal_path);
        free(db);
        return NULL;
    }

    db->dimension = (size_t)file_dim;

    if (file_version != 1 && file_version != 2 && file_version != 3) {
        fclose(in);
        if (db->hnsw_index) gv_hnsw_destroy(db->hnsw_index);
        free(db->filepath);
        free(db->wal_path);
        free(db);
        return NULL;
    }

    uint32_t file_index_type = GV_INDEX_TYPE_KDTREE;
    if (file_version >= 2) {
        if (gv_read_uint32(in, &file_index_type) != 0) {
            fclose(in);
            if (db->hnsw_index) gv_hnsw_destroy(db->hnsw_index);
            free(db->filepath);
            free(db->wal_path);
            free(db);
            return NULL;
        }
    }

    if (file_index_type != db->index_type) {
        fclose(in);
        if (db->hnsw_index) gv_hnsw_destroy(db->hnsw_index);
        free(db->filepath);
        free(db->wal_path);
        free(db);
        return NULL;
    }

    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        if (gv_kdtree_load_recursive(&(db->root), in, db->dimension, file_version) != 0) {
            fclose(in);
            gv_kdtree_destroy_recursive(db->root);
            free(db->filepath);
            free(db->wal_path);
            free(db);
            return NULL;
        }
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        void *loaded_index = NULL;
        if (gv_hnsw_load(&loaded_index, in, db->dimension, file_version) != 0) {
            fclose(in);
            if (loaded_index) gv_hnsw_destroy(loaded_index);
            if (db->hnsw_index) gv_hnsw_destroy(db->hnsw_index);
            free(db->filepath);
            free(db->wal_path);
            free(db);
            return NULL;
        }
        if (db->hnsw_index != NULL) {
            gv_hnsw_destroy(db->hnsw_index);
        }
        db->hnsw_index = loaded_index;
        db->count = gv_hnsw_count(db->hnsw_index);
    } else {
        fclose(in);
        if (db->hnsw_index) gv_hnsw_destroy(db->hnsw_index);
        free(db->filepath);
        free(db->wal_path);
        free(db);
        return NULL;
    }

    if (file_version >= 3) {
        if (fseek(in, 0, SEEK_END) != 0) {
            goto load_fail;
        }
        long end_pos = ftell(in);
        if (end_pos < 4) {
            goto load_fail;
        }
        if (fseek(in, end_pos - (long)sizeof(uint32_t), SEEK_SET) != 0) {
            goto load_fail;
        }
        uint32_t stored_crc = 0;
        if (gv_read_uint32(in, &stored_crc) != 0) {
            goto load_fail;
        }
        if (fseek(in, 0, SEEK_SET) != 0) {
            goto load_fail;
        }
        uint32_t crc = gv_crc32_init();
        char buf[65536];
        long remaining = end_pos - (long)sizeof(uint32_t);
        while (remaining > 0) {
            size_t chunk = (remaining > (long)sizeof(buf)) ? sizeof(buf) : (size_t)remaining;
            if (fread(buf, 1, chunk, in) != chunk) {
                goto load_fail;
            }
            crc = gv_crc32_update(crc, buf, chunk);
            remaining -= (long)chunk;
        }
        crc = gv_crc32_finish(crc);
        if (crc != stored_crc) {
            goto load_fail;
        }
    }

    if (fclose(in) != 0) {
        goto load_fail;
    }

    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        db->count = file_count;
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        db->count = gv_hnsw_count(db->hnsw_index);
    }

    if (db->wal_path != NULL) {
        db->wal = gv_wal_open(db->wal_path, db->dimension);
        if (db->wal == NULL) {
            if (db->index_type == GV_INDEX_TYPE_KDTREE) {
                gv_kdtree_destroy_recursive(db->root);
            } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
                if (db->hnsw_index) gv_hnsw_destroy(db->hnsw_index);
            }
            free(db->filepath);
            free(db->wal_path);
            free(db);
            return NULL;
        }

        db->wal_replaying = 1;
        if (gv_wal_replay(db->wal_path, db->dimension, gv_db_wal_apply, db) != 0) {
            db->wal_replaying = 0;
            gv_wal_close(db->wal);
            db->wal = NULL;
            if (db->index_type == GV_INDEX_TYPE_KDTREE) {
                gv_kdtree_destroy_recursive(db->root);
            } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
                if (db->hnsw_index) gv_hnsw_destroy(db->hnsw_index);
            }
            free(db->filepath);
            free(db->wal_path);
            free(db);
            return NULL;
        }
        db->wal_replaying = 0;
    }
    return db;

load_fail:
    if (in != NULL) {
        fclose(in);
    }
    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        gv_kdtree_destroy_recursive(db->root);
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        if (db->hnsw_index) gv_hnsw_destroy(db->hnsw_index);
    }
    free(db->filepath);
    free(db->wal_path);
    free(db);
    return NULL;
}

void gv_db_close(GV_Database *db) {
    if (db == NULL) {
        return;
    }

    if (db->wal) {
        gv_wal_close(db->wal);
    }

    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        gv_kdtree_destroy_recursive(db->root);
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        gv_hnsw_destroy(db->hnsw_index);
    }
    free(db->filepath);
    free(db->wal_path);
    free(db);
}

int gv_db_set_wal(GV_Database *db, const char *wal_path) {
    if (db == NULL) {
        return -1;
    }

    if (db->wal) {
        gv_wal_close(db->wal);
        db->wal = NULL;
    }
    free(db->wal_path);
    db->wal_path = NULL;

    if (wal_path == NULL) {
        return 0;
    }

    db->wal_path = gv_db_strdup(wal_path);
    if (db->wal_path == NULL) {
        return -1;
    }
    db->wal = gv_wal_open(db->wal_path, db->dimension);
    if (db->wal == NULL) {
        free(db->wal_path);
        db->wal_path = NULL;
        return -1;
    }
    return 0;
}

void gv_db_disable_wal(GV_Database *db) {
    if (db == NULL) {
        return;
    }
    if (db->wal) {
        gv_wal_close(db->wal);
        db->wal = NULL;
    }
    free(db->wal_path);
    db->wal_path = NULL;
}

int gv_db_wal_dump(const GV_Database *db, FILE *out) {
    if (db == NULL || out == NULL || db->wal_path == NULL) {
        return -1;
    }
    return gv_wal_dump(db->wal_path, db->dimension, out);
}

int gv_db_add_vector(GV_Database *db, const float *data, size_t dimension) {
    if (db == NULL || data == NULL || dimension == 0 || dimension != db->dimension) {
        return -1;
    }

    if (db->wal != NULL && db->wal_replaying == 0) {
        if (gv_wal_append_insert(db->wal, data, dimension, NULL, NULL) != 0) {
            return -1;
        }
    }

    GV_Vector *vector = gv_vector_create_from_data(dimension, data);
    if (vector == NULL) {
        return -1;
    }

    int status = -1;
    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        status = gv_kdtree_insert(&(db->root), vector, 0);
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        status = gv_hnsw_insert(db->hnsw_index, vector);
    }

    if (status != 0) {
        gv_vector_destroy(vector);
        return -1;
    }

    db->count += 1;
    return 0;
}

int gv_db_add_vector_with_metadata(GV_Database *db, const float *data, size_t dimension,
                                    const char *metadata_key, const char *metadata_value) {
    if (db == NULL || data == NULL || dimension == 0 || dimension != db->dimension) {
        return -1;
    }

    if (db->wal != NULL && db->wal_replaying == 0) {
        if (gv_wal_append_insert(db->wal, data, dimension, metadata_key, metadata_value) != 0) {
            return -1;
        }
    }

    GV_Vector *vector = gv_vector_create_from_data(dimension, data);
    if (vector == NULL) {
        return -1;
    }

    if (metadata_key != NULL && metadata_value != NULL) {
        if (gv_vector_set_metadata(vector, metadata_key, metadata_value) != 0) {
            gv_vector_destroy(vector);
            return -1;
        }
    }

    int status = -1;
    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        status = gv_kdtree_insert(&(db->root), vector, 0);
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        status = gv_hnsw_insert(db->hnsw_index, vector);
    }

    if (status != 0) {
        gv_vector_destroy(vector);
        return -1;
    }

    db->count += 1;
    return 0;
}

int gv_db_save(const GV_Database *db, const char *filepath) {
    if (db == NULL) {
        return -1;
    }

    const char *out_path = filepath != NULL ? filepath : db->filepath;
    if (out_path == NULL) {
        return -1;
    }

    if (db->dimension == 0 || db->dimension > UINT32_MAX) {
        return -1;
    }

    FILE *out = fopen(out_path, "wb");
    if (out == NULL) {
        return -1;
    }

    const uint32_t version = 3;
    int status = gv_db_write_header(out, (uint32_t)db->dimension, db->count, version);
    if (status == 0) {
        uint32_t index_type_u32 = (uint32_t)db->index_type;
        if (gv_write_uint32(out, index_type_u32) != 0) {
            status = -1;
        } else if (db->index_type == GV_INDEX_TYPE_KDTREE) {
            status = gv_kdtree_save_recursive(db->root, out, version);
        } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
            status = gv_hnsw_save(db->hnsw_index, out, version);
        } else {
            status = -1;
        }
    }

    /* Append checksum (version >=3) */
    if (status == 0) {
        long end_pos = ftell(out);
        if (end_pos < 0) {
            status = -1;
        } else {
            uint32_t crc = gv_crc32_init();
            if (fflush(out) != 0 || fseek(out, 0, SEEK_SET) != 0) {
                status = -1;
            } else {
                char buf[65536];
                long remaining = end_pos;
                while (status == 0 && remaining > 0) {
                    size_t chunk = (remaining > (long)sizeof(buf)) ? sizeof(buf) : (size_t)remaining;
                    if (fread(buf, 1, chunk, out) != chunk) {
                        status = -1;
                        break;
                    }
                    crc = gv_crc32_update(crc, buf, chunk);
                    remaining -= (long)chunk;
                }
                if (status == 0) {
                    crc = gv_crc32_finish(crc);
                    if (fseek(out, 0, SEEK_END) != 0 || gv_write_uint32(out, crc) != 0) {
                        status = -1;
                    }
                }
            }
        }
    }

    if (fclose(out) != 0) {
        status = -1;
    }

    if (db->wal_path != NULL) {
        gv_wal_reset(db->wal_path);
    }

    return 0;
}

int gv_db_search(const GV_Database *db, const float *query_data, size_t k,
                 GV_SearchResult *results, GV_DistanceType distance_type) {
    if (db == NULL || query_data == NULL || results == NULL || k == 0) {
        return -1;
    }

    if (db->index_type == GV_INDEX_TYPE_KDTREE && db->root == NULL) {
        return 0;
    }
    if (db->index_type == GV_INDEX_TYPE_HNSW && db->hnsw_index == NULL) {
        return 0;
    }

    GV_Vector query_vec;
    query_vec.dimension = db->dimension;
    query_vec.data = (float *)query_data;
    query_vec.metadata = NULL;

    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        return gv_kdtree_knn_search(db->root, &query_vec, k, results, distance_type);
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        return gv_hnsw_search(db->hnsw_index, &query_vec, k, results, distance_type, NULL, NULL);
    }
    return -1;
}

int gv_db_search_filtered(const GV_Database *db, const float *query_data, size_t k,
                          GV_SearchResult *results, GV_DistanceType distance_type,
                          const char *filter_key, const char *filter_value) {
    if (db == NULL || query_data == NULL || results == NULL || k == 0) {
        return -1;
    }

    if (db->index_type == GV_INDEX_TYPE_KDTREE && db->root == NULL) {
        return 0;
    }
    if (db->index_type == GV_INDEX_TYPE_HNSW && db->hnsw_index == NULL) {
        return 0;
    }

    GV_Vector query_vec;
    query_vec.dimension = db->dimension;
    query_vec.data = (float *)query_data;
    query_vec.metadata = NULL;

    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        return gv_kdtree_knn_search_filtered(db->root, &query_vec, k, results, distance_type,
                                            filter_key, filter_value);
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        return gv_hnsw_search(db->hnsw_index, &query_vec, k, results, distance_type,
                            filter_key, filter_value);
    }
    return -1;
}

