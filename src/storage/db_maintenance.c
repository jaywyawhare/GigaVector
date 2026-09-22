#include <errno.h>
#include <stdint.h>
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
#include "specialized/point_id.h"
#include "core/scope.h"
#include "index/index_maintenance.h"

char *db_build_wal_path(const char *filepath) {
    if (filepath == NULL) {
        return NULL;
    }

    const char *dir_override = getenv("GV_WAL_DIR");
    const char *basename = strrchr(filepath, '/');
    basename = (basename == NULL) ? filepath : basename + 1;

    char buf[1024];
    int written;
    if (dir_override != NULL && dir_override[0] != '\0') {
        written = snprintf(buf, sizeof(buf), "%s/%s.wal", dir_override, basename);
    } else {
        written = snprintf(buf, sizeof(buf), "%s.wal", filepath);
    }
    if (written < 0 || (size_t)written >= sizeof(buf)) {
        return NULL;
    }

    return gv_dup_cstr(buf);
}

static int db_wal_apply_delete(void *ctx, size_t vector_index) {
    GV_Database *db = (GV_Database *)ctx;
    if (db == NULL) return -1;
    return db_delete_vector_by_index(db, vector_index);
}

static int db_wal_apply_update(void *ctx, size_t vector_index, const float *data,
                                size_t dimension,
                                const char *const *metadata_keys, const char *const *metadata_values,
                                size_t metadata_count) {
    GV_Database *db = (GV_Database *)ctx;
    if (db == NULL || data == NULL || dimension != db->dimension) return -1;
    (void)metadata_keys; (void)metadata_values; (void)metadata_count;
    return db_update_vector(db, vector_index, data, dimension);
}

static int db_wal_apply_rich(void *ctx, const float *data, size_t dimension,
                                const char *const *metadata_keys, const char *const *metadata_values,
                                size_t metadata_count) {
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
    if (db->index_type == GV_INDEX_TYPE_IVFFLAT) {
        if (ivfflat_is_trained(db->hnsw_index) == 0) {
            return -1;
        }
    }
    if (db->index_type == GV_INDEX_TYPE_IVFSQ8) {
        if (ivfsq8_is_trained(db->hnsw_index) == 0) {
            return -1;
        }
    }
    if (db->index_type == GV_INDEX_TYPE_IVFTURBOQUANT) {
        if (ivfturboquant_is_trained(db->hnsw_index) == 0) {
            return -1;
        }
    }
    if (db->index_type == GV_INDEX_TYPE_IVFDISK) {
        if (ivfdisk_is_trained((GV_IVFDiskIndex *)db->hnsw_index) == 0) {
            return -1;
        }
    }
    if (db->index_type == GV_INDEX_TYPE_PQ) {
        if (pq_is_trained(db->hnsw_index) == 0) {
            return -1;
        }
    }
    if (db_add_vector_with_rich_metadata(db, data, db->dimension, metadata_keys, metadata_values, metadata_count) != 0) {
        return -1;
    }
    return 0;
}

static int db_wal_apply_ivfdisk_append(void *ctx, uint64_t head_id, uint64_t vector_id,
                                       const float *data, size_t dimension)
{
    GV_Database *db = (GV_Database *)ctx;
    if (!db || db->index_type != GV_INDEX_TYPE_IVFDISK || !db->hnsw_index || !data) {
        return -1;
    }
    if (dimension != db->dimension) return -1;
    if (ivfdisk_is_trained((GV_IVFDiskIndex *)db->hnsw_index) == 0) return -1;
    return ivfdisk_insert_to_head((GV_IVFDiskIndex *)db->hnsw_index, head_id, data,
                                  dimension, (size_t)vector_id);
}

GV_IndexType index_suggest(size_t dimension, size_t expected_count) {
    return index_suggest_with_budget(dimension, expected_count, 0, 0);
}

size_t index_suggest_bytes_per_vector(size_t dimension, size_t metadata_bytes_per_vector) {
    size_t meta = metadata_bytes_per_vector ? metadata_bytes_per_vector
                                            : GV_INDEX_SUGGEST_METADATA_OVERHEAD;
    return dimension * sizeof(float) + meta;
}

int db_replay_wal(GV_Database *db) {
    if (db == NULL || db->wal_path == NULL) {
        return 0;
    }
    if (access(db->wal_path, F_OK) != 0) {
        return 0;
    }
    if (db->wal == NULL) {
        db->wal = wal_open(db->wal_path, db->dimension, (uint32_t)db->index_type);
        if (db->wal == NULL) {
            GV_LOG_ERROR("db_replay_wal: wal_open failed for '%s' (errno=%d)",
                         db->wal_path, errno);
            return -1;
        }
    }

    db->wal_replaying = 1;
    int rc = wal_replay_rich(db->wal_path, db->dimension, db_wal_apply_rich,
                             db_wal_apply_delete, db_wal_apply_update,
                             db_wal_apply_ivfdisk_append,
                             db, (uint32_t)db->index_type);
    db->wal_replaying = 0;
    if (rc != 0) {
        GV_LOG_ERROR("db_replay_wal: WAL replay of '%s' failed (rc=%d) - recovery incomplete",
                     db->wal_path, rc);
        return -1;
    }
    db_refresh_count(db);
    return 0;
}


int db_set_wal(GV_Database *db, const char *wal_path) {
    if (db == NULL) {
        return -1;
    }

    if (db->wal) {
        wal_close(db->wal);
        db->wal = NULL;
    }
    gv_free(db->wal_path);
    db->wal_path = NULL;

    if (wal_path == NULL) {
        return 0;
    }

    db->wal_path = gv_dup_cstr(wal_path);
    if (db->wal_path == NULL) {
        return -1;
    }
    db->wal = wal_open(db->wal_path, db->dimension, (uint32_t)db->index_type);
    if (db->wal == NULL) {
        GV_LOG_ERROR("db_set_wal: wal_open failed for '%s' (errno=%d)", wal_path, errno);
        gv_free(db->wal_path);
        db->wal_path = NULL;
        return -1;
    }
    return 0;
}

void db_disable_wal(GV_Database *db) {
    if (db == NULL) {
        return;
    }
    if (db->wal) {
        wal_close(db->wal);
        db->wal = NULL;
    }
    gv_free(db->wal_path);
    db->wal_path = NULL;
}

int db_wal_dump(const GV_Database *db, FILE *out) {
    if (db == NULL || out == NULL || db->wal_path == NULL) {
        return -1;
    }
    return wal_dump(db->wal_path, db->dimension, (uint32_t)db->index_type, out);
}

const char *db_wal_path(const GV_Database *db) {
    if (!db) return NULL;
    return db->wal_path;
}

int db_apply_wal_record(GV_Database *db, const uint8_t *record, size_t len) {
    if (!db || !record || len == 0) return -1;

    int has_crc = 1;
    (void)has_crc;

    db->wal_replaying = 1;
    int rc = wal_apply_record_buffer(record, len, has_crc, db->dimension,
                                     db_wal_apply_rich, db_wal_apply_delete,
                                     db_wal_apply_update, db_wal_apply_ivfdisk_append,
                                     db);
    db->wal_replaying = 0;
    if (rc != 0) return rc;

    if (db->wal != NULL) {
        pthread_mutex_lock(&db->wal_mutex);
        rc = wal_append_raw(db->wal, record, len);
        if (rc == 0) db->total_wal_records += 1;
        pthread_mutex_unlock(&db->wal_mutex);
    }
    return rc;
}

/* Thread-local MVCC create_version stamp applied by db_add_vector to the slot it
 * inserts. A committing transaction sets this (for its own thread only) so its
 * staged inserts are tagged atomically under the insert's write lock; concurrent
 * non-transactional inserts on other threads see 0 and are left unstamped. */

struct db_compact_id_remap_ctx {
    GV_PointIDMap *dst;
    const size_t  *index_map;
    size_t         old_count;
};

static int db_compact_remap_id_cb(const char *id, size_t old_index, void *vctx) {
    struct db_compact_id_remap_ctx *ctx = (struct db_compact_id_remap_ctx *)vctx;
    if (old_index < ctx->old_count) {
        size_t new_index = ctx->index_map[old_index];
        if (new_index != (size_t)-1) {
            point_id_set(ctx->dst, id, new_index);
        }
    }
    return 0; /* continue iteration */
}

int db_compact_soa_storage_locked(GV_Database *db) {
    if (db == NULL || db->soa_storage == NULL) {
        return -1;
    }

    GV_SoAStorage *storage = db->soa_storage;
    size_t dimension = storage->dimension;
    
    size_t deleted_count = 0;
    for (size_t i = 0; i < storage->count; ++i) {
        if (storage->deleted[i] != 0) {
            deleted_count++;
        }
    }

    if (deleted_count == 0) {
        return 0; /* Nothing to compact */
    }

    size_t new_count = storage->count - deleted_count;
    if (dimension == 0 || new_count > SIZE_MAX / dimension / sizeof(float)) return -1;

    float *new_data = NULL;
    GV_Metadata **new_metadata = NULL;
    int *new_deleted = NULL;
    new_data = (float *)gv_db_alloc(db, new_count * dimension * sizeof(float));
    new_metadata = (GV_Metadata **)gv_db_calloc(db, new_count, sizeof(GV_Metadata *));
    new_deleted = (int *)gv_db_calloc(db, new_count, sizeof(int));
    
    if (new_data == NULL || new_metadata == NULL || new_deleted == NULL) {
        if (new_data != NULL) gv_db_free(db, new_data);
        if (new_metadata != NULL) gv_db_free(db, new_metadata);
        if (new_deleted != NULL) gv_db_free(db, new_deleted);
        return -1;
    }

    int map_on_heap = 0;
    size_t *index_map = (size_t *)gv_tls_alloc_or_heap(
        storage->count * sizeof(size_t), sizeof(size_t), &map_on_heap);
    if (index_map == NULL) {
        gv_db_free(db, new_data);
        gv_db_free(db, new_metadata);
        gv_db_free(db, new_deleted);
        return -1;
    }

    size_t new_idx = 0;
    for (size_t old_idx = 0; old_idx < storage->count; ++old_idx) {
        if (storage->deleted[old_idx] == 0) {
            memcpy(new_data + (new_idx * dimension),
                   storage->data + (old_idx * dimension),
                   dimension * sizeof(float));
            new_metadata[new_idx] = storage->metadata[old_idx];
            storage->metadata[old_idx] = NULL; /* Transfer ownership */
            new_deleted[new_idx] = 0;
            index_map[old_idx] = new_idx;
            new_idx++;
        } else {
            if (storage->metadata[old_idx] != NULL) {
                GV_Vector temp_vec = {
                    .dimension = dimension,
                    .data = NULL,
                    .metadata = storage->metadata[old_idx]
                };
                vector_clear_metadata(&temp_vec);
            }
            index_map[old_idx] = (size_t)-1; /* Mark as deleted */
        }
    }

    gv_db_free(db, storage->data);
    gv_db_free(db, storage->metadata);
    gv_db_free(db, storage->deleted);

    storage->data = new_data;
    storage->metadata = new_metadata;
    storage->deleted = new_deleted;
    storage->count = new_count;
    storage->capacity = new_count; /* Shrink to fit */

    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        GV_KDNode *old_root = db->root;
        db->root = NULL;
        for (size_t i = 0; i < new_count; ++i) {
            kdtree_insert(&(db->root), storage, i, 0);
        }
        kdtree_destroy_recursive(old_root);
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        if (db->hnsw_index != NULL) {
            /* Forward declaration for accessing HNSW index internals */
            typedef struct {
                size_t dimension;
                size_t M;
                size_t efConstruction;
                size_t efSearch;
                size_t maxLevel;
                int use_binary_quant;
                size_t quant_rerank;
                int use_acorn;
                size_t acorn_hops;
                void *entryPoint;
                size_t count;
                void **nodes;
                size_t nodes_capacity;
                GV_SoAStorage *soa_storage;
                int soa_storage_owned;
            } GV_HNSWIndex_Internal;
            
            GV_HNSWIndex_Internal *old_index = (GV_HNSWIndex_Internal *)db->hnsw_index;
            GV_HNSWConfig config = {
                .M = old_index->M,
                .efConstruction = old_index->efConstruction,
                .efSearch = old_index->efSearch,
                .maxLevel = old_index->maxLevel,
                .use_binary_quant = old_index->use_binary_quant,
                .quant_rerank = old_index->quant_rerank,
                .use_acorn = old_index->use_acorn,
                .acorn_hops = old_index->acorn_hops
            };
            
            gv_hnsw_destroy(db->hnsw_index);
            db->hnsw_index = NULL;
            
            db->hnsw_index = gv_hnsw_create(dimension, &config, storage);
            if (db->hnsw_index == NULL) {
                /* Failed to create new index. Free the compaction map before
                 * bailing so we do not leak it; the storage is already
                 * compacted but the index is now NULL (degraded). */
                gv_tls_free_or_heap(index_map, map_on_heap);
                return -1;
            }

            for (size_t i = 0; i < new_count; ++i) {
                GV_Vector temp_vec = {
                    .dimension = dimension,
                    .data = storage->data + (i * dimension),
                    .metadata = storage->metadata[i]
                };

                if (gv_hnsw_insert(db->hnsw_index, &temp_vec) != 0) {
                    /* On failure, clean up. Free the compaction map to avoid a
                     * leak (it is heap-allocated when it overflowed the TLS
                     * arena). */
                    gv_hnsw_destroy(db->hnsw_index);
                    db->hnsw_index = NULL;
                    gv_tls_free_or_heap(index_map, map_on_heap);
                    return -1;
                }

                temp_vec.metadata = NULL;
            }
        }
    }

    if (db->metadata_index != NULL) {
        GV_MetadataIndex *old_index = db->metadata_index;
        db->metadata_index = metadata_index_create();
        if (db->metadata_index != NULL) {
            for (size_t i = 0; i < new_count; ++i) {
                GV_Metadata *meta = storage->metadata[i];
                if (meta != NULL) {
                    GV_Metadata *current = meta;
                    while (current != NULL) {
                        metadata_index_add(db->metadata_index, current->key, current->value, i);
                        current = current->next;
                    }
                }
            }
            metadata_index_destroy(old_index);
        }
    }

    /* Remap the external chunk_id -> internal-index map (id_map) to the new
     * indices, dropping entries whose vector was compacted away.  Without this,
     * every chunk_id would resolve to the wrong vector after compaction. */
    if (db->id_map != NULL && point_id_count(db->id_map) > 0) {
        GV_PointIDMap *remapped = point_id_create(point_id_count(db->id_map));
        if (remapped != NULL) {
            struct db_compact_id_remap_ctx ctx = {
                remapped, index_map, new_count + deleted_count
            };
            point_id_iterate(db->id_map, db_compact_remap_id_cb, &ctx);
            point_id_destroy(db->id_map);
            db->id_map = remapped;
        }
        /* On OOM keep the old (now-stale) map rather than dropping all ids. */
    }

    gv_tls_free_or_heap(index_map, map_on_heap);
    return 0;
}

static int db_compact_wal(GV_Database *db) {
    if (db == NULL || db->wal == NULL || db->filepath == NULL) {
        return 0; /* No WAL to compact */
    }

    FILE *wal_file = fopen(db->wal_path, "rb");
    if (wal_file == NULL) {
        return 0; /* WAL doesn't exist or can't be opened */
    }

    if (fseek(wal_file, 0, SEEK_END) != 0) {
        fclose(wal_file);
        return 0;
    }

    long wal_size = ftell(wal_file);
    fclose(wal_file);

    if (wal_size < 0 || (size_t)wal_size < db->wal_compaction_threshold) {
        return 0; /* WAL is below threshold */
    }

    /* db_compact (our only caller) already holds the write lock, so use the
     * locked variant to avoid re-acquiring the non-recursive rwlock.
     * db_save_locked performs its own temp-file + atomic rename and truncates
     * the WAL on success. */
    db_save_locked(db, db->filepath);

    return 0;
}

int db_value_store_enable(GV_Database *db, const char *path) {
    if (!db || !path) return -1;
    if (db->value_store) return 0; /* already enabled */
    db->value_store = value_store_open(path);
    return db->value_store ? 0 : -1;
}

int db_value_store_put(GV_Database *db, uint64_t key, const void *value, size_t len) {
    if (!db || !db->value_store) return -1;
    return value_store_put(db->value_store, key, value, len);
}

int db_value_store_get(GV_Database *db, uint64_t key, void **value_out, size_t *len_out) {
    if (!db || !db->value_store) return -1;
    return value_store_get(db->value_store, key, value_out, len_out);
}

int db_value_store_delete(GV_Database *db, uint64_t key) {
    if (!db || !db->value_store) return -1;
    return value_store_delete(db->value_store, key);
}

int db_value_store_gc(GV_Database *db) {
    if (!db || !db->value_store) return -1;
    return value_store_gc(db->value_store);
}

int db_compact(GV_Database *db) {
    if (db == NULL) {
        return -1;
    }

    gv_tls_arena_reset();

    pthread_rwlock_wrlock(&db->rwlock);

    if (db->soa_storage != NULL) {
        size_t deleted_count = 0;
        for (size_t i = 0; i < db->soa_storage->count; ++i) {
            if (db->soa_storage->deleted[i] != 0) {
                deleted_count++;
            }
        }
        
        double deleted_ratio = (db->soa_storage->count > 0) ?
            (double)deleted_count / (double)db->soa_storage->count : 0.0;

        if (deleted_ratio >= db->deleted_ratio_threshold) {
            db_compact_soa_storage_locked(db);
        }
    }

    db_compact_wal(db);

    if (db->index_type == GV_INDEX_TYPE_IVFDISK && db->hnsw_index != NULL) {
        ivfdisk_head_checkpoint_if_needed((GV_IVFDiskIndex *)db->hnsw_index);
        GV_IVFDiskMaintenanceConfig mcfg;
        ivfdisk_maintenance_config_init(&mcfg);
        ivfdisk_maintenance_run((GV_IVFDiskIndex *)db->hnsw_index, &mcfg, NULL);
    }

    pthread_rwlock_unlock(&db->rwlock);
    return 0;
}

static void *db_compaction_thread(void *arg) {
    GV_Database *db = (GV_Database *)arg;
    if (db == NULL) {
        return NULL;
    }

    pthread_mutex_lock(&db->compaction_mutex);

    while (db->compaction_running) {
        struct timespec timeout;
        clock_gettime(CLOCK_REALTIME, &timeout);
        timeout.tv_sec += db->compaction_interval_sec;

        int wait_result = pthread_cond_timedwait(&db->compaction_cond,
                                                  &db->compaction_mutex,
                                                  &timeout);

        if (wait_result == ETIMEDOUT || wait_result == 0) {
            db_compact(db);
        }
    }

    pthread_mutex_unlock(&db->compaction_mutex);
    return NULL;
}

int db_start_background_compaction(GV_Database *db) {
    if (db == NULL) {
        return -1;
    }

    pthread_mutex_lock(&db->compaction_mutex);

    if (db->compaction_running) {
        pthread_mutex_unlock(&db->compaction_mutex);
        return 0; /* Already running */
    }

    db->compaction_running = 1;
    int result = pthread_create(&db->compaction_thread, NULL,
                                db_compaction_thread, db);

    pthread_mutex_unlock(&db->compaction_mutex);

    if (result != 0) {
        db->compaction_running = 0;
        return -1;
    }

    return 0;
}

void db_stop_background_compaction(GV_Database *db) {
    if (db == NULL) {
        return;
    }

    pthread_mutex_lock(&db->compaction_mutex);

    if (!db->compaction_running) {
        pthread_mutex_unlock(&db->compaction_mutex);
        return;
    }

    db->compaction_running = 0;
    pthread_cond_signal(&db->compaction_cond);
    pthread_mutex_unlock(&db->compaction_mutex);

    pthread_join(db->compaction_thread, NULL);
}

void db_set_compaction_interval(GV_Database *db, size_t interval_sec) {
    if (db == NULL) {
        return;
    }
    pthread_mutex_lock(&db->compaction_mutex);
    db->compaction_interval_sec = interval_sec;
    pthread_cond_signal(&db->compaction_cond); /* Wake up thread to check new interval */
    pthread_mutex_unlock(&db->compaction_mutex);
}

void db_set_wal_compaction_threshold(GV_Database *db, size_t threshold_bytes) {
    if (db == NULL) {
        return;
    }
    db->wal_compaction_threshold = threshold_bytes;
}

void db_set_deleted_ratio_threshold(GV_Database *db, double ratio) {
    if (db == NULL) {
        return;
    }
    if (ratio < 0.0) {
        ratio = 0.0;
    }
    if (ratio > 1.0) {
        ratio = 1.0;
    }
    db->deleted_ratio_threshold = ratio;
}

