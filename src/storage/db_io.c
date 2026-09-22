#include <errno.h>
#include <stdint.h>
#include "core/compat.h"   /* gv_rename_replace (Windows-safe atomic publish) */
#ifndef _WIN32
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#ifndef _WIN32
#include <unistd.h>
#include <fcntl.h>
#endif
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#ifndef ssize_t
typedef SSIZE_T ssize_t;
#endif
/* getline() shim provided by core/compat.h (included above) on Windows/MinGW. */
#endif

#include "core/types.h"
#include "core/memory.h"
#include "core/log.h"
#include "core/utils.h"
#include "storage/database.h"
#include "storage/db_internal.h"
#include "storage/wal.h"
#include "storage/soa_storage.h"
#include "storage/value_store.h"
#include "search/distance.h"
#include "schema/metadata.h"
#include "schema/vector.h"
#include "multimodal/metadata_index.h"
#include "index/hnsw.h"
#include "index/ivfpq.h"
#include "index/kdtree.h"
#include "index/sparse_index.h"
#include "index/flat.h"
#include "index/ivfflat.h"
#include "index/ivfdisk.h"
#include "index/ivfsq8.h"
#include "index/ivfturboquant.h"
#include "index/pq.h"
#include "index/lsh.h"
#include "index/rabitq.h"
#include "admin/cdc.h"
#include "admin/webhook.h"
#include "features/json.h"
#include "core/scope.h"
#include "specialized/point_id.h"

static int write_uint32(FILE *out, uint32_t value) {
    return write_u32(out, value);
}


int db_save_locked(const GV_Database *db, const char *filepath) {
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

    char temp_path[1024];
    if (snprintf(temp_path, sizeof(temp_path), "%s.tmp", out_path) >= (int)sizeof(temp_path)) {
        return -1;
    }

    FILE *out = fopen(temp_path, "wb");
    if (out == NULL) {
        GV_LOG_ERROR("db_save: fopen('%s', \"wb\") failed (errno=%d)", temp_path, errno);
        return -1;
    }

    /* v5 adds the per-vector deleted flag to the sparse-index payload. */
    const uint32_t version = 6;
    int status = db_write_header(out, (uint32_t)db->dimension, db->count, version);
    if (status == 0) {
        uint32_t index_type_u32 = (uint32_t)db->index_type;
        if (write_uint32(out, index_type_u32) != 0) {
            status = -1;
        } else if (db->index_type == GV_INDEX_TYPE_KDTREE) {
            if (db->soa_storage == NULL) {
                status = -1;
            } else {
                status = kdtree_save_recursive(db->root, db->soa_storage, out, version);
            }
        } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
            status = gv_hnsw_save(db->hnsw_index, out, version);
        } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
            status = gv_ivfpq_save(db->hnsw_index, out, version);
        } else if (db->index_type == GV_INDEX_TYPE_SPARSE) {
            status = sparse_index_save(db->sparse_index, out, version);
        } else if (db->index_type == GV_INDEX_TYPE_FLAT) {
            status = flat_save(db->hnsw_index, out, version);
        } else if (db->index_type == GV_INDEX_TYPE_IVFFLAT) {
            status = ivfflat_save(db->hnsw_index, out, version);
        } else if (db->index_type == GV_INDEX_TYPE_IVFSQ8) {
            status = ivfsq8_save(db->hnsw_index, out, version);
        } else if (db->index_type == GV_INDEX_TYPE_IVFTURBOQUANT) {
            status = ivfturboquant_save(db->hnsw_index, out, version);
        } else if (db->index_type == GV_INDEX_TYPE_PQ) {
            status = pq_save(db->hnsw_index, out, version);
        } else if (db->index_type == GV_INDEX_TYPE_LSH) {
            status = lsh_save(db->hnsw_index, out, version);
        } else if (db->index_type == GV_INDEX_TYPE_RABITQ) {
            status = rabitq_save(db->hnsw_index, out, version);
        } else if (db->index_type == GV_INDEX_TYPE_IVFDISK) {
            if (db->hnsw_index == NULL || db->soa_storage == NULL) {
                status = -1;
            } else {
                status = ivfdisk_save((const GV_IVFDiskIndex *)db->hnsw_index, out, version);
                if (status == 0) {
                    status = soa_storage_save(db->soa_storage, out, version);
                }
            }
        } else {
            status = -1;
        }
    }

    if (status != 0) {
        GV_LOG_ERROR("db_save: serialization of index (type %d) to '%s' failed",
                     (int)db->index_type, temp_path);
    }
    if (fclose(out) != 0) {
        GV_LOG_ERROR("db_save: fclose of '%s' failed (errno=%d) - data may not be flushed",
                     temp_path, errno);
        status = -1;
    }

    if (status == 0) {
        FILE *rf = fopen(temp_path, "rb");
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
                FILE *af = fopen(temp_path, "ab");
                if (af == NULL || write_uint32(af, crc) != 0 || fclose(af) != 0) {
                    status = -1;
                }
            }
        }
    }

    /* Durably flush the temp file's data to disk BEFORE the rename so that a
     * crash after the rename cannot leave a renamed-but-empty file while the
     * WAL (the only other copy) has already been truncated. We must re-open
     * the temp file because the CRC append above already closed it. */
    if (status == 0) {
        FILE *sf = fopen(temp_path, "rb+");
        if (sf == NULL) {
            status = -1;
        } else {
            if (fflush(sf) != 0) {
                status = -1;
            } else {
#ifndef _WIN32
                if (fsync(fileno(sf)) != 0) {
                    GV_LOG_ERROR("db_save: fsync of '%s' failed (errno=%d) - snapshot not durable",
                                 temp_path, errno);
                    status = -1;
                }
#else
                if (_commit(_fileno(sf)) != 0) {
                    status = -1;
                }
#endif
            }
            if (fclose(sf) != 0) {
                status = -1;
            }
        }
    }

    /* Atomically publish the new snapshot; on any error leave the original
     * file untouched and remove the temp file. */
    if (status == 0) {
        if (gv_rename_replace(temp_path, out_path) != 0) {
            GV_LOG_ERROR("db_save: rename('%s' -> '%s') failed (errno=%d) - snapshot not published",
                         temp_path, out_path, errno);
            status = -1;
        }
    }
    if (status != 0) {
        unlink(temp_path);
        return -1;
    }

    /* fsync the containing directory so the rename (a directory metadata
     * change) is durable. Without this, a crash could lose the rename even
     * though the file data reached disk. POSIX only; on _WIN32 directory
     * fsync semantics differ and rename durability is handled differently, so
     * we omit it there. If the directory fsync fails we must NOT truncate the
     * WAL, since the snapshot may not be durably published. */
#ifndef _WIN32
    {
        char dir_path[1024];
        size_t out_len = strlen(out_path);
        const char *slash = NULL;
        for (size_t i = out_len; i > 0; --i) {
            if (out_path[i - 1] == '/') {
                slash = &out_path[i - 1];
                break;
            }
        }
        if (slash == NULL) {
            dir_path[0] = '.';
            dir_path[1] = '\0';
        } else if (slash == out_path) {
            dir_path[0] = '/';
            dir_path[1] = '\0';
        } else {
            size_t dlen = (size_t)(slash - out_path);
            if (dlen >= sizeof(dir_path)) {
                dlen = sizeof(dir_path) - 1;
            }
            memcpy(dir_path, out_path, dlen);
            dir_path[dlen] = '\0';
        }
        int dfd = open(dir_path, O_RDONLY | O_DIRECTORY);
        if (dfd < 0) {
            GV_LOG_ERROR("db_save: open('%s', O_DIRECTORY) failed (errno=%d) - cannot fsync dir for rename durability",
                         dir_path, errno);
            status = -1;
        } else {
            if (fsync(dfd) != 0) {
                GV_LOG_ERROR("db_save: fsync of directory '%s' failed (errno=%d) - rename may not be durable; WAL retained",
                             dir_path, errno);
                status = -1;
            }
            close(dfd);
        }
    }
    if (status != 0) {
        /* Snapshot rename may not be durable: keep the WAL as the recovery
         * source rather than truncating it. The renamed file is left in place
         * (it is at least as good as the previous snapshot on the next open). */
        return -1;
    }
#endif

    if (db->wal != NULL) {
        pthread_mutex_lock((pthread_mutex_t *)&db->wal_mutex);
        int truncate_status = wal_truncate(db->wal);
        if (truncate_status == 0) {
            ((GV_Database *)db)->total_wal_records = 0;
        }
        pthread_mutex_unlock((pthread_mutex_t *)&db->wal_mutex);
    } else if (db->wal_path != NULL) {
        /* Fallback: if WAL handle is NULL but path exists, use reset */
        wal_reset(db->wal_path);
    }

    /* Persist the chunk_id -> index map to a "{filepath}.ids" sidecar.
     * point_id_save is crash-atomic (temp + fsync + rename), so a failure here
     * means the sidecar wasn't durably written at all (not torn) — and a missing/
     * stale sidecar corrupts chunk_id -> index lookups on reload, so we must
     * surface it. The main snapshot is already durable above, so ordering is safe. */
    if (status == 0 && filepath != NULL && db->id_map != NULL &&
        point_id_count(db->id_map) > 0) {
        char ids_path[1024];
        int w = snprintf(ids_path, sizeof(ids_path), "%s.ids", filepath);
        if (w <= 0 || (size_t)w >= sizeof(ids_path)) {
            status = -1;
        } else if (point_id_save(db->id_map, ids_path) != 0) {
            GV_LOG_ERROR("db_save: point_id_save to '%s' failed - chunk_id->index map not persisted",
                         ids_path);
            status = -1;
        }
    }

    return status;
}

int db_save(const GV_Database *db, const char *filepath) {
    if (db == NULL) {
        return -1;
    }

    /* Materialise committed MVCC tombstones as real hard deletes before serialising,
     * so they persist across every index's save format. Checkpoint semantics: a
     * reloaded database has no older snapshots to serve, so the tombstone becomes
     * a permanent delete. (No-op unless transactions have run.) */
    if (db->commit_version != 0 && db->soa_storage != NULL) {
        GV_Database *mdb = (GV_Database *)db;
        /* Snapshot the tombstoned indices UNDER the read lock so the scan doesn't
         * race a concurrent db_add_vector resizing delete_version[]/deleted[].
         * The hard-deletes run afterwards because db_delete_vector_by_index takes
         * the write lock itself (and it soft-deletes, so indices stay stable). */
        size_t *to_delete = NULL, ndel = 0, cap = 0;
        pthread_rwlock_rdlock((pthread_rwlock_t *)&db->rwlock);
        size_t n = db->soa_storage->count;
        for (size_t i = 0; i < n; i++) {
            if (db->soa_storage->delete_version[i] != 0 && db->soa_storage->deleted[i] == 0) {
                if (ndel == cap) {
                    size_t ncap = cap ? cap * 2 : 64;
                    size_t *tmp = (size_t *)gv_realloc(to_delete, ncap * sizeof(size_t));
                    if (!tmp) break;  /* best-effort: materialize what we captured */
                    to_delete = tmp; cap = ncap;
                }
                to_delete[ndel++] = i;
            }
        }
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        for (size_t k = 0; k < ndel; k++) db_delete_vector_by_index(mdb, to_delete[k]);
        gv_free(to_delete);
    }

    pthread_rwlock_rdlock((pthread_rwlock_t *)&db->rwlock);
    int status = db_save_locked(db, filepath);
    pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
    return status;
}

int db_export_json(const GV_Database *db, const char *filepath) {
    if (db == NULL || filepath == NULL) {
        return -1;
    }

    FILE *fp = fopen(filepath, "w");
    if (!fp) {
        return -1;
    }

    pthread_rwlock_rdlock((pthread_rwlock_t *)&db->rwlock);

    int exported = 0;

    if (db->soa_storage != NULL) {
        size_t total = db->soa_storage->count;
        for (size_t i = 0; i < total; i++) {
            if (soa_storage_is_deleted(db->soa_storage, i) == 1) {
                continue;
            }
            const float *data = soa_storage_get_data(db->soa_storage, i);
            if (!data) continue;

            GV_JsonValue *obj = json_object();
            if (!obj) continue;

            json_object_set(obj, "index", json_number((double)i));

            GV_JsonValue *vec_arr = json_array();
            if (vec_arr) {
                for (size_t d = 0; d < db->dimension; d++) {
                    json_array_push(vec_arr, json_number((double)data[d]));
                }
                json_object_set(obj, "vector", vec_arr);
            }

            GV_Metadata *meta = soa_storage_get_metadata(db->soa_storage, i);
            if (meta) {
                GV_JsonValue *meta_obj = json_object();
                if (meta_obj) {
                    GV_Metadata *m = meta;
                    while (m) {
                        json_object_set(meta_obj, m->key, json_string(m->value));
                        m = m->next;
                    }
                    json_object_set(obj, "metadata", meta_obj);
                }
            }

            char *line = json_stringify(obj, false);
            json_free(obj);
            if (line) {
                fprintf(fp, "%s\n", line);
                gv_free(line);
                exported++;
            }
        }
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
    fclose(fp);
    return exported;
}

int db_import_json(GV_Database *db, const char *filepath) {
    if (db == NULL || filepath == NULL) {
        return -1;
    }

    FILE *fp = fopen(filepath, "r");
    if (!fp) {
        return -1;
    }

    int imported = 0;
    char *line = NULL;
    size_t line_cap = 0;
    ssize_t line_len;

    while ((line_len = getline(&line, &line_cap, fp)) != -1) {
        if (line_len <= 1) continue;

        GV_JsonError err;
        GV_JsonValue *obj = json_parse(line, &err);
        if (!obj || obj->type != GV_JSON_OBJECT) {
            json_free(obj);
            continue;
        }

        GV_JsonValue *vec_arr = json_object_get(obj, "vector");
        if (!vec_arr || vec_arr->type != GV_JSON_ARRAY) {
            json_free(obj);
            continue;
        }

        size_t dim = json_array_length(vec_arr);
        if (dim != db->dimension) {
            json_free(obj);
            continue;
        }

        int data_on_heap = 0;
        float *data = (float *)gv_tls_alloc_or_heap(
            dim * sizeof(float), sizeof(float), &data_on_heap);
        if (!data) {
            json_free(obj);
            continue;
        }

        int valid = 1;
        for (size_t d = 0; d < dim; d++) {
            GV_JsonValue *elem = json_array_get(vec_arr, d);
            double val;
            if (!elem || json_get_number(elem, &val) != GV_JSON_OK) {
                valid = 0;
                break;
            }
            data[d] = (float)val;
        }

        if (!valid) {
            gv_tls_free_or_heap(data, data_on_heap);
            json_free(obj);
            continue;
        }

        GV_JsonValue *meta_obj = json_object_get(obj, "metadata");
        int insert_ok = -1;

        if (meta_obj && meta_obj->type == GV_JSON_OBJECT && json_object_length(meta_obj) > 0) {
            size_t meta_count = json_object_length(meta_obj);
            int keys_on_heap = 0;
            int vals_on_heap = 0;
            const char **keys = (const char **)gv_tls_alloc_or_heap(
                meta_count * sizeof(const char *), sizeof(const char *), &keys_on_heap);
            const char **vals = (const char **)gv_tls_alloc_or_heap(
                meta_count * sizeof(const char *), sizeof(const char *), &vals_on_heap);

            if (keys && vals) {
                for (size_t m = 0; m < meta_count; m++) {
                    keys[m] = meta_obj->data.object.entries[m].key;
                    const char *sv = json_get_string(meta_obj->data.object.entries[m].value);
                    vals[m] = sv ? sv : "";
                }
                insert_ok = db_add_vector_with_rich_metadata(db, data, dim,
                                                                 keys, vals, meta_count);
            }
            gv_tls_free_or_heap((void *)keys, keys_on_heap);
            gv_tls_free_or_heap((void *)vals, vals_on_heap);
        } else {
            insert_ok = db_add_vector(db, data, dim);
        }

        gv_tls_free_or_heap(data, data_on_heap);
        json_free(obj);

        if (insert_ok == 0) {
            imported++;
        }
    }

    gv_free(line);
    fclose(fp);
    return imported;
}

size_t database_count(const GV_Database *db) {
    if (!db) return 0;
    return db->count;
}

size_t database_dimension(const GV_Database *db) {
    if (!db) return 0;
    return db->dimension;
}

const float *database_get_vector(const GV_Database *db, size_t index) {
    if (!db || !db->soa_storage) return NULL;
    if (index >= db->count) return NULL;
    return soa_storage_get_data(db->soa_storage, index);
}
