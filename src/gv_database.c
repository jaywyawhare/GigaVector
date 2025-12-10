#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include "gigavector/gv_database.h"
#include "gigavector/gv_distance.h"
#include "gigavector/gv_hnsw.h"
#include "gigavector/gv_ivfpq.h"
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
    if (db == NULL || data == NULL) {
        return -1;
    }
    if (dimension != db->dimension) {
        return -1;
    }
    /* IVF-PQ requires training before inserts can be replayed */
    if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        if (gv_ivfpq_is_trained(db->hnsw_index) == 0) {
            return -1;
        }
    }
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
    pthread_rwlock_init(&db->rwlock, NULL);
    pthread_mutex_init(&db->wal_mutex, NULL);
    db->count = 0;

    if (index_type == GV_INDEX_TYPE_HNSW && filepath == NULL) {
        db->hnsw_index = gv_hnsw_create(dimension, NULL);
        if (db->hnsw_index == NULL) {
            pthread_rwlock_destroy(&db->rwlock);
            pthread_mutex_destroy(&db->wal_mutex);
            free(db);
            return NULL;
        }
    } else if (index_type == GV_INDEX_TYPE_IVFPQ && filepath == NULL) {
        db->hnsw_index = NULL;
        db->root = NULL;
        GV_IVFPQConfig cfg = {.nlist = 64, .m = 8, .nbits = 8, .nprobe = 4, .train_iters = 15};
        db->hnsw_index = gv_ivfpq_create(dimension, &cfg);
        if (db->hnsw_index == NULL) {
            pthread_rwlock_destroy(&db->rwlock);
            pthread_mutex_destroy(&db->wal_mutex);
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
            db->wal = gv_wal_open(db->wal_path, db->dimension, (uint32_t)db->index_type);
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
                db->wal = gv_wal_open(db->wal_path, db->dimension, (uint32_t)db->index_type);
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

    if (file_version != 1 && file_version != 2 && file_version != 3 && file_version != 4) {
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
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        void *loaded_index = NULL;
        if (gv_ivfpq_load(&loaded_index, in, db->dimension, file_version) != 0) {
            fclose(in);
            if (loaded_index) gv_ivfpq_destroy(loaded_index);
            free(db->filepath);
            free(db->wal_path);
            free(db);
            return NULL;
        }
        db->hnsw_index = loaded_index;
        db->count = gv_ivfpq_count(db->hnsw_index);
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
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        db->count = gv_ivfpq_count(db->hnsw_index);
    }

    if (db->wal_path != NULL) {
        db->wal = gv_wal_open(db->wal_path, db->dimension, (uint32_t)db->index_type);
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
        if (gv_wal_replay(db->wal_path, db->dimension, gv_db_wal_apply, db, (uint32_t)db->index_type) != 0) {
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

    pthread_rwlock_destroy(&db->rwlock);
    pthread_mutex_destroy(&db->wal_mutex);
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
    db->wal = gv_wal_open(db->wal_path, db->dimension, (uint32_t)db->index_type);
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
    return gv_wal_dump(db->wal_path, db->dimension, (uint32_t)db->index_type, out);
}

int gv_db_add_vector(GV_Database *db, const float *data, size_t dimension) {
    if (db == NULL || data == NULL || dimension == 0 || dimension != db->dimension) {
        return -1;
    }

    if (db->wal != NULL && db->wal_replaying == 0) {
        pthread_mutex_lock(&db->wal_mutex);
        int wal_res = gv_wal_append_insert(db->wal, data, dimension, NULL, NULL);
        pthread_mutex_unlock(&db->wal_mutex);
        if (wal_res != 0) {
            return -1;
        }
    }

    pthread_rwlock_wrlock(&db->rwlock);
    GV_Vector *vector = gv_vector_create_from_data(dimension, data);
    if (vector == NULL) {
        pthread_rwlock_unlock(&db->rwlock);
        return -1;
    }

    int status = -1;
    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        status = gv_kdtree_insert(&(db->root), vector, 0);
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        status = gv_hnsw_insert(db->hnsw_index, vector);
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        status = gv_ivfpq_insert(db->hnsw_index, vector);
    }

    if (status != 0) {
        gv_vector_destroy(vector);
        pthread_rwlock_unlock(&db->rwlock);
        return -1;
    }

    db->count += 1;
    pthread_rwlock_unlock(&db->rwlock);
    return 0;
}

int gv_db_add_vector_with_metadata(GV_Database *db, const float *data, size_t dimension,
                                    const char *metadata_key, const char *metadata_value) {
    if (db == NULL || data == NULL || dimension == 0 || dimension != db->dimension) {
        return -1;
    }

    if (db->wal != NULL && db->wal_replaying == 0) {
        pthread_mutex_lock(&db->wal_mutex);
        int wal_res = gv_wal_append_insert(db->wal, data, dimension, metadata_key, metadata_value);
        pthread_mutex_unlock(&db->wal_mutex);
        if (wal_res != 0) {
            return -1;
        }
    }

    pthread_rwlock_wrlock(&db->rwlock);
    GV_Vector *vector = gv_vector_create_from_data(dimension, data);
    if (vector == NULL) {
        pthread_rwlock_unlock(&db->rwlock);
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
        pthread_rwlock_unlock(&db->rwlock);
        return -1;
    }

    db->count += 1;
    pthread_rwlock_unlock(&db->rwlock);
    return 0;
}

int gv_db_add_vectors(GV_Database *db, const float *data, size_t count, size_t dimension) {
    if (db == NULL || data == NULL || count == 0 || dimension != db->dimension) {
        return -1;
    }
    for (size_t i = 0; i < count; ++i) {
        const float *vec = data + i * dimension;
        if (gv_db_add_vector(db, vec, dimension) != 0) {
            return -1;
        }
    }
    return 0;
}

int gv_db_add_vectors_with_metadata(GV_Database *db, const float *data,
                                    const char *const *keys, const char *const *values,
                                    size_t count, size_t dimension) {
    if (db == NULL || data == NULL || count == 0 || dimension != db->dimension) {
        return -1;
    }
    for (size_t i = 0; i < count; ++i) {
        const float *vec = data + i * dimension;
        const char *k = (keys != NULL) ? keys[i] : NULL;
        const char *v = (values != NULL) ? values[i] : NULL;
        if (gv_db_add_vector_with_metadata(db, vec, dimension, k, v) != 0) {
            return -1;
        }
    }
    return 0;
}

int gv_db_save(const GV_Database *db, const char *filepath) {
    if (db == NULL) {
        return -1;
    }

    /* Serialize writers while saving */
    pthread_rwlock_rdlock((pthread_rwlock_t *)&db->rwlock);
    const char *out_path = filepath != NULL ? filepath : db->filepath;
    if (out_path == NULL) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return -1;
    }

    if (db->dimension == 0 || db->dimension > UINT32_MAX) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return -1;
    }

    FILE *out = fopen(out_path, "wb");
    if (out == NULL) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return -1;
    }

    const uint32_t version = 4;
    int status = gv_db_write_header(out, (uint32_t)db->dimension, db->count, version);
    if (status == 0) {
        uint32_t index_type_u32 = (uint32_t)db->index_type;
        if (gv_write_uint32(out, index_type_u32) != 0) {
            status = -1;
        } else if (db->index_type == GV_INDEX_TYPE_KDTREE) {
            status = gv_kdtree_save_recursive(db->root, out, version);
        } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
            status = gv_hnsw_save(db->hnsw_index, out, version);
        } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
            status = gv_ivfpq_save(db->hnsw_index, out, version);
        } else {
            status = -1;
        }
    }

    /* Append checksum (version >=3) */
    if (fclose(out) != 0) {
        status = -1;
    }

    if (status == 0) {
        /* Compute checksum over file and append */
        FILE *rf = fopen(out_path, "rb");
        if (rf == NULL) {
            status = -1;
        } else {
            uint32_t crc = gv_crc32_init();
            char buf[65536];
            size_t nread = 0;
            while ((nread = fread(buf, 1, sizeof(buf), rf)) > 0) {
                crc = gv_crc32_update(crc, buf, nread);
            }
            if (ferror(rf)) {
                status = -1;
            }
            fclose(rf);
            if (status == 0) {
                crc = gv_crc32_finish(crc);
                FILE *af = fopen(out_path, "ab");
                if (af == NULL || gv_write_uint32(af, crc) != 0 || fclose(af) != 0) {
                    status = -1;
                }
            }
        }
    }

    if (db->wal_path != NULL && status == 0) {
        gv_wal_reset(db->wal_path);
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
    return status == 0 ? 0 : -1;
}

int gv_db_search(const GV_Database *db, const float *query_data, size_t k,
                 GV_SearchResult *results, GV_DistanceType distance_type) {
    if (db == NULL || query_data == NULL || results == NULL || k == 0) {
        return -1;
    }

    pthread_rwlock_rdlock((pthread_rwlock_t *)&db->rwlock);

    if (db->index_type == GV_INDEX_TYPE_KDTREE && db->root == NULL) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return 0;
    }
    if (db->index_type == GV_INDEX_TYPE_HNSW && db->hnsw_index == NULL) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return 0;
    }
    if (db->index_type == GV_INDEX_TYPE_IVFPQ && db->hnsw_index == NULL) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return 0;
    }
    if (db->index_type == GV_INDEX_TYPE_IVFPQ && db->hnsw_index == NULL) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return 0;
    }

    GV_Vector query_vec;
    query_vec.dimension = db->dimension;
    query_vec.data = (float *)query_data;
    query_vec.metadata = NULL;

    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        int r = gv_kdtree_knn_search(db->root, &query_vec, k, results, distance_type);
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return r;
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        int r = gv_hnsw_search(db->hnsw_index, &query_vec, k, results, distance_type, NULL, NULL);
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return r;
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        int r = gv_ivfpq_search(db->hnsw_index, &query_vec, k, results, distance_type, 0, 0);
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return r;
    }
    pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
    return -1;
}

int gv_db_search_ivfpq_opts(const GV_Database *db, const float *query_data, size_t k,
                            GV_SearchResult *results, GV_DistanceType distance_type,
                            size_t nprobe_override, size_t rerank_top) {
    if (db == NULL || query_data == NULL || results == NULL || k == 0) {
        return -1;
    }
    if (db->index_type != GV_INDEX_TYPE_IVFPQ) {
        return -1;
    }

    pthread_rwlock_rdlock((pthread_rwlock_t *)&db->rwlock);

    if (db->hnsw_index == NULL) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return 0;
    }

    GV_Vector query_vec;
    query_vec.dimension = db->dimension;
    query_vec.data = (float *)query_data;
    query_vec.metadata = NULL;

    int r = gv_ivfpq_search(db->hnsw_index, &query_vec, k, results, distance_type,
                            nprobe_override, rerank_top);
    pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
    return r;
}

int gv_db_search_batch(const GV_Database *db, const float *queries, size_t qcount, size_t k,
                       GV_SearchResult *results, GV_DistanceType distance_type) {
    if (db == NULL || queries == NULL || results == NULL || qcount == 0 || k == 0) {
        return -1;
    }
    pthread_rwlock_rdlock((pthread_rwlock_t *)&db->rwlock);
    if (db->index_type == GV_INDEX_TYPE_KDTREE && db->root == NULL) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return 0;
    }
    if (db->index_type == GV_INDEX_TYPE_HNSW && db->hnsw_index == NULL) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return 0;
    }

    GV_Vector qv;
    qv.dimension = db->dimension;
    qv.metadata = NULL;

    for (size_t i = 0; i < qcount; ++i) {
        qv.data = (float *)(queries + i * db->dimension);
        GV_SearchResult *slot = results + i * k;
        int r = -1;
        if (db->index_type == GV_INDEX_TYPE_KDTREE) {
            r = gv_kdtree_knn_search(db->root, &qv, k, slot, distance_type);
        } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
            r = gv_hnsw_search(db->hnsw_index, &qv, k, slot, distance_type, NULL, NULL);
        } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
            r = gv_ivfpq_search(db->hnsw_index, &qv, k, slot, distance_type, 0, 0);
        }
        if (r < 0) {
            pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
            return -1;
        }
        /* If fewer than k, remaining results are unspecified; caller can check r. */
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
    return (int)(qcount * k);
}

int gv_db_search_filtered(const GV_Database *db, const float *query_data, size_t k,
                          GV_SearchResult *results, GV_DistanceType distance_type,
                          const char *filter_key, const char *filter_value) {
    if (db == NULL || query_data == NULL || results == NULL || k == 0) {
        return -1;
    }

    pthread_rwlock_rdlock((pthread_rwlock_t *)&db->rwlock);

    if (db->index_type == GV_INDEX_TYPE_KDTREE && db->root == NULL) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return 0;
    }
    if (db->index_type == GV_INDEX_TYPE_HNSW && db->hnsw_index == NULL) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return 0;
    }

    GV_Vector query_vec;
    query_vec.dimension = db->dimension;
    query_vec.data = (float *)query_data;
    query_vec.metadata = NULL;

    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        int r = gv_kdtree_knn_search_filtered(db->root, &query_vec, k, results, distance_type,
                                            filter_key, filter_value);
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return r;
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        int r = gv_hnsw_search(db->hnsw_index, &query_vec, k, results, distance_type,
                            filter_key, filter_value);
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return r;
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        /* No native filter; apply post-filter on results */
        GV_SearchResult *tmp = (GV_SearchResult *)malloc(sizeof(GV_SearchResult) * k);
        if (!tmp) {
            pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
            return -1;
        }
        int r = gv_ivfpq_search(db->hnsw_index, &query_vec, k, tmp, distance_type, 0, 0);
        if (r <= 0) {
            free(tmp);
            pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
            return r;
        }
        int out = 0;
        for (int i = 0; i < r && out < (int)k; ++i) {
            if (filter_key == NULL || filter_value == NULL) {
                results[out++] = tmp[i];
            } else {
                const char *val = gv_vector_get_metadata(tmp[i].vector, filter_key);
                if (val && strcmp(val, filter_value) == 0) {
                    results[out++] = tmp[i];
                }
            }
        }
        free(tmp);
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return out;
    }
    pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
    return -1;
}

