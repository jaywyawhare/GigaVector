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
#endif
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#endif

#include "core/types.h"
#include "core/scope.h"
#include "core/memory.h"
#include "core/log.h"
#include "core/utils.h"
#include "core/sim_time.h"

#include "storage/database.h"
#include "storage/db_internal.h"
#include "storage/wal.h"
#include "storage/soa_storage.h"
#include "storage/tiered_storage.h"
#include "search/distance.h"
#include "search/filter.h"
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
#include "index/index_maintenance.h"
#include "index/ivfsq8.h"
#include "index/ivfturboquant.h"
#include "index/pq.h"
#include "index/lsh.h"
#include "index/rabitq.h"
#include "index/exact_search.h"
#include "admin/cdc.h"
#include "admin/webhook.h"
#include "specialized/point_id.h"
#include "specialized/quantization.h"
#include "index/ivf_retrain.h"

static _Thread_local uint64_t g_txn_commit_stamp = 0;
void db_set_commit_stamp(uint64_t stamp) { g_txn_commit_stamp = stamp; }

/* When set on a thread, db_add_vector applies to the index but does NOT write
 * its own per-insert WAL record - the transaction commit path writes one atomic
 * TXN record for the whole batch instead. See transaction.c:db_commit. */
static _Thread_local int g_txn_wal_suppress = 0;
void db_set_wal_suppress(int on) { g_txn_wal_suppress = on; }

int db_add_vector(GV_Database *db, const float *data, size_t dimension) {
    if (db == NULL || data == NULL || dimension == 0 || dimension != db->dimension) {
        return -1;
    }

    uint64_t start_time_us = db_get_time_us();

    size_t vector_memory = db_estimate_vector_memory(dimension);
    if (db_check_resource_limits(db, 1, vector_memory) != 0) {
        return -1;
    }

    db_increment_concurrent_ops(db);

    pthread_rwlock_wrlock(&db->rwlock);

    int status = -1;
    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        if (db->soa_storage == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            db_decrement_concurrent_ops(db);
            return -1;
        }
        int normalized_on_heap = 0;
        float *normalized_data = (float *)gv_tls_alloc_or_heap(
            dimension * sizeof(float), sizeof(float), &normalized_on_heap);
        if (normalized_data == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            db_decrement_concurrent_ops(db);
            return -1;
        }
        memcpy(normalized_data, data, dimension * sizeof(float));
        if (db->cosine_normalized) {
            float norm_sq = 0.0f;
            for (size_t i = 0; i < dimension; ++i) {
                float v = normalized_data[i];
                norm_sq += v * v;
            }
            if (norm_sq > 0.0f) {
                float inv = 1.0f / sqrtf(norm_sq);
                for (size_t i = 0; i < dimension; ++i) {
                    normalized_data[i] *= inv;
                }
            }
        }
        size_t vector_index = soa_storage_add(db->soa_storage, normalized_data, NULL);
        gv_tls_free_or_heap(normalized_data, normalized_on_heap);
        if (vector_index == (size_t)-1) {
            pthread_rwlock_unlock(&db->rwlock);
            db_decrement_concurrent_ops(db);
            return -1;
        }
        status = kdtree_insert(&(db->root), db->soa_storage, vector_index, 0);
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        GV_Vector *vector = vector_create_from_data(dimension, data);
        if (vector == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        if (db->cosine_normalized) {
            db_normalize_vector(vector);
        }
        status = gv_hnsw_insert(db->hnsw_index, vector);
        /* HNSW copies data into SoA and takes ownership of metadata (NULLing it on
         * success, restoring it on failure); the GV_Vector shell is never stored, so
         * free it either way to avoid orphaning the struct + data copy. */
        vector_destroy(vector);
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        GV_Vector *vector = vector_create_from_data(dimension, data);
        if (vector == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        if (db->cosine_normalized) {
            db_normalize_vector(vector);
        }
        status = gv_ivfpq_insert(db->hnsw_index, vector);
        if (status != 0) {
            vector_destroy(vector);
        }
    } else if (db->index_type == GV_INDEX_TYPE_FLAT) {
        GV_Vector *vector = vector_create_from_data(dimension, data);
        if (vector == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        if (db->cosine_normalized) {
            db_normalize_vector(vector);
        }
        status = flat_insert(db->hnsw_index, vector);
        if (status != 0) {
            vector_destroy(vector);
        }
    } else if (db->index_type == GV_INDEX_TYPE_IVFFLAT) {
        GV_Vector *vector = vector_create_from_data(dimension, data);
        if (vector == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        if (db->cosine_normalized) {
            db_normalize_vector(vector);
        }
        status = ivfflat_insert(db->hnsw_index, vector);
        if (status != 0) {
            vector_destroy(vector);
        }
    } else if (db->index_type == GV_INDEX_TYPE_IVFDISK) {
        if (ivfdisk_is_trained((GV_IVFDiskIndex *)db->hnsw_index) == 0) {
            pthread_rwlock_unlock(&db->rwlock);
            db_decrement_concurrent_ops(db);
            return -1;
        }
        int normalized_on_heap = 0;
        float *normalized_data = (float *)gv_tls_alloc_or_heap(
            dimension * sizeof(float), sizeof(float), &normalized_on_heap);
        if (normalized_data == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            db_decrement_concurrent_ops(db);
            return -1;
        }
        memcpy(normalized_data, data, dimension * sizeof(float));
        if (db->cosine_normalized) {
            float norm_sq = 0.0f;
            for (size_t i = 0; i < dimension; ++i) {
                float v = normalized_data[i];
                norm_sq += v * v;
            }
            if (norm_sq > 0.0f) {
                float inv = 1.0f / sqrtf(norm_sq);
                for (size_t i = 0; i < dimension; ++i) {
                    normalized_data[i] *= inv;
                }
            }
        }
        size_t vector_index = soa_storage_add(db->soa_storage, normalized_data, NULL);
        gv_tls_free_or_heap(normalized_data, normalized_on_heap);
        if (vector_index == (size_t)-1) {
            pthread_rwlock_unlock(&db->rwlock);
            db_decrement_concurrent_ops(db);
            return -1;
        }
        const float *stored = soa_storage_get_data(db->soa_storage, vector_index);
        if (db->wal_replaying) {
            status = 0;
        } else {
            uint64_t heads[2];
            size_t nh = 0;
            status = ivfdisk_insert_routed((GV_IVFDiskIndex *)db->hnsw_index, stored,
                                           dimension, vector_index, heads, &nh, 2);
            if (status == 0 && db->wal != NULL && !g_txn_wal_suppress) {
                pthread_mutex_lock(&db->wal_mutex);
                for (size_t hi = 0; hi < nh; ++hi) {
                    if (wal_append_ivfdisk_append(db->wal, heads[hi], (uint64_t)vector_index,
                                                  stored, dimension) != 0) {
                        status = -1;
                        break;
                    }
                }
                pthread_mutex_unlock(&db->wal_mutex);
            }
        }
    } else if (db->index_type == GV_INDEX_TYPE_IVFSQ8) {
        GV_Vector *vector = vector_create_from_data(dimension, data);
        if (vector == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        if (db->cosine_normalized) {
            db_normalize_vector(vector);
        }
        status = ivfsq8_insert(db->hnsw_index, vector);
        if (status != 0) {
            vector_destroy(vector);
        }
    } else if (db->index_type == GV_INDEX_TYPE_IVFTURBOQUANT) {
        GV_Vector *vector = vector_create_from_data(dimension, data);
        if (vector == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        if (db->cosine_normalized) {
            db_normalize_vector(vector);
        }
        status = ivfturboquant_insert(db->hnsw_index, vector);
        if (status != 0) {
            vector_destroy(vector);
        }
    } else if (db->index_type == GV_INDEX_TYPE_PQ) {
        GV_Vector *vector = vector_create_from_data(dimension, data);
        if (vector == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        if (db->cosine_normalized) {
            db_normalize_vector(vector);
        }
        status = pq_insert(db->hnsw_index, vector);
        if (status != 0) {
            vector_destroy(vector);
        }
    } else if (db->index_type == GV_INDEX_TYPE_LSH) {
        GV_Vector *vector = vector_create_from_data(dimension, data);
        if (vector == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        if (db->cosine_normalized) {
            db_normalize_vector(vector);
        }
        status = lsh_insert(db->hnsw_index, vector);
        if (status != 0) {
            vector_destroy(vector);
        }
    } else if (db->index_type == GV_INDEX_TYPE_RABITQ) {
        GV_Vector *vector = vector_create_from_data(dimension, data);
        if (vector == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        if (db->cosine_normalized) {
            db_normalize_vector(vector);
        }
        status = rabitq_insert(db->hnsw_index, vector);
        if (status != 0) {
            vector_destroy(vector);
        }
    }

    if (status != 0) {
        pthread_rwlock_unlock(&db->rwlock);
        db_decrement_concurrent_ops(db);
        return -1;
    }

    /* Append-after-apply: the in-memory insert succeeded, so now record it in
     * the WAL. The record bytes are written+fflush'd UNDER the write lock (so
     * WAL order matches the in-memory positional order, and no phantom insert is
     * recorded for a failed apply), but the fsync durability barrier is DEFERRED
     * until after the write lock is released - so a slow fsync no longer stalls
     * concurrent readers. wal_fsync_deferred() is serialized against WAL
     * truncation, and an un-fsync'd record that a checkpoint truncates is still
     * durable via the snapshot (the checkpoint ran under the rwlock after this
     * insert was applied). */
    GV_WAL *deferred_wal = NULL;
    if (db->wal != NULL && db->wal_replaying == 0 && !g_txn_wal_suppress) {
        pthread_mutex_lock(&db->wal_mutex);
        int wal_res = wal_append_insert_deferred(db->wal, data, dimension, NULL, NULL);
        if (wal_res == 0) {
            db->total_wal_records += 1;
            deferred_wal = db->wal;
        }
        pthread_mutex_unlock(&db->wal_mutex);
        if (wal_res != 0) {
            GV_LOG_ERROR("db_add_vector: wal_append_insert failed (rc=%d) - insert not durable",
                         wal_res);
            pthread_rwlock_unlock(&db->rwlock);
            db_decrement_concurrent_ops(db);
            return -1;
        }
    }

    size_t ts_slot_0 = db->count; /* 0-based slot for this vector */
    db->count += 1;
    if (g_txn_commit_stamp != 0 && db->soa_storage != NULL) {
        /* Atomic with the insert (under the write lock): tag this vector's MVCC
         * create version so a committing transaction can't stamp the wrong slot. */
        soa_storage_set_create_version(db->soa_storage, ts_slot_0, g_txn_commit_stamp);
        soa_storage_set_delete_version(db->soa_storage, ts_slot_0, 0);
    }
    db->total_inserts += 1;
    db->generation += 1;
    db_update_memory_usage(db);
    if (db->tiering_enabled && db->tiered_storage) {
        tiered_storage_record_insert(db->tiered_storage, ts_slot_0, start_time_us);
    }
    size_t emit_index = db->count - 1;
    pthread_rwlock_unlock(&db->rwlock);

    /* Durability barrier OUTSIDE the write lock: concurrent readers are not
     * blocked during the fsync. Non-fatal on failure (bytes are already written
     * and fflush'd; this forces them to stable storage). */
    if (deferred_wal != NULL) {
        if (wal_fsync_deferred(deferred_wal) != 0) {
            GV_LOG_ERROR("db_add_vector: deferred WAL fsync failed - insert may not be durable");
        }
    }

    db_emit_change(db, GV_CDC_INSERT, GV_EVENT_INSERT, emit_index, data, dimension);

    /* IVF incremental retrain: check drift after threshold is reached */
    if (db->retrain_enabled &&
        (db->index_type == GV_INDEX_TYPE_IVFFLAT ||
         db->index_type == GV_INDEX_TYPE_IVFPQ   ||
         db->index_type == GV_INDEX_TYPE_IVFSQ8  ||
         db->index_type == GV_INDEX_TYPE_IVFTURBOQUANT)) {
        pthread_mutex_lock(&db->retrain_mutex);
        db->inserts_since_retrain++;
        size_t ins = db->inserts_since_retrain;
        size_t min_vecs = db->retrain_min_new_vectors;
        int already_running = db->retrain_running;
        pthread_mutex_unlock(&db->retrain_mutex);

        if (ins >= min_vecs && !already_running) {
            /* Read the index inertia under the read lock so we do not touch
             * posting lists while the background retrain thread mutates them
             * under the write lock (which would be a data race / UAF). The
             * rdlock blocks until any in-flight retrain releases its wrlock,
             * so the index is quiescent while we compute drift. */
            pthread_rwlock_rdlock(&db->rwlock);
            float drift = ivf_retrain_check_drift(db);
            pthread_rwlock_unlock(&db->rwlock);

            /* Trigger the retrain WITHOUT holding rwlock, since the retrain
             * thread needs the write lock. ivf_retrain_trigger re-checks
             * retrain_running under retrain_mutex, so a concurrently started
             * retrain will not be double-triggered. */
            if (drift > 0.0f && drift > 1.0f + db->retrain_drift_threshold) {
                ivf_retrain_trigger(db);
            }
        }
    }

    uint64_t end_time_us = db_get_time_us();
    uint64_t latency_us = end_time_us - start_time_us;
    db_record_latency(db, latency_us, 1);

    db_decrement_concurrent_ops(db);
    return 0;
}

int db_add_vector_with_metadata(GV_Database *db, const float *data, size_t dimension,
                                     const char *metadata_key, const char *metadata_value) {
    if (db == NULL || data == NULL || dimension == 0 || dimension != db->dimension ||
        metadata_key == NULL || metadata_value == NULL) {
        return -1;
    }

    uint64_t start_time_us = db_get_time_us();

    size_t vector_memory = db_estimate_vector_memory(dimension);
    if (db_check_resource_limits(db, 1, vector_memory) != 0) {
        return -1;
    }

    db_increment_concurrent_ops(db);
    if (db == NULL || data == NULL || dimension == 0 || dimension != db->dimension) {
        db_decrement_concurrent_ops(db);
        return -1;
    }

    pthread_rwlock_wrlock(&db->rwlock);

    int status = -1;
    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        if (db->soa_storage == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            db_decrement_concurrent_ops(db);
            return -1;
        }
        int normalized_on_heap = 0;
        float *normalized_data = (float *)gv_tls_alloc_or_heap(
            dimension * sizeof(float), sizeof(float), &normalized_on_heap);
        if (normalized_data == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            db_decrement_concurrent_ops(db);
            return -1;
        }
        memcpy(normalized_data, data, dimension * sizeof(float));
        if (db->cosine_normalized) {
            float norm_sq = 0.0f;
            for (size_t i = 0; i < dimension; ++i) {
                float v = normalized_data[i];
                norm_sq += v * v;
            }
            if (norm_sq > 0.0f) {
                float inv = 1.0f / sqrtf(norm_sq);
                for (size_t i = 0; i < dimension; ++i) {
                    normalized_data[i] *= inv;
                }
            }
        }
        GV_Metadata *metadata = NULL;
        if (metadata_key != NULL && metadata_value != NULL) {
            GV_Vector temp_vec;
            temp_vec.dimension = dimension;
            temp_vec.data = NULL;
            temp_vec.metadata = NULL;
            if (vector_set_metadata(&temp_vec, metadata_key, metadata_value) == 0) {
                metadata = temp_vec.metadata;
            }
        }
        size_t vector_index = soa_storage_add(db->soa_storage, normalized_data, metadata);
        gv_tls_free_or_heap(normalized_data, normalized_on_heap);
        if (vector_index == (size_t)-1) {
            if (metadata != NULL) {
                GV_Vector temp_vec;
                temp_vec.dimension = dimension;
                temp_vec.data = NULL;
                temp_vec.metadata = metadata;
                vector_clear_metadata(&temp_vec);
            }
            pthread_rwlock_unlock(&db->rwlock);
            db_decrement_concurrent_ops(db);
            return -1;
        }
        if (metadata != NULL && db->metadata_index != NULL) {
            GV_Metadata *current = metadata;
            while (current != NULL) {
                metadata_index_add(db->metadata_index, current->key, current->value, vector_index);
                current = current->next;
            }
        }
        status = kdtree_insert(&(db->root), db->soa_storage, vector_index, 0);
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        GV_Vector *vector = vector_create_from_data(dimension, data);
        if (vector == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            db_decrement_concurrent_ops(db);
            return -1;
        }
        if (db->cosine_normalized) {
            db_normalize_vector(vector);
        }
        if (metadata_key != NULL && metadata_value != NULL) {
            if (vector_set_metadata(vector, metadata_key, metadata_value) != 0) {
                vector_destroy(vector);
                pthread_rwlock_unlock(&db->rwlock);
                db_decrement_concurrent_ops(db);
                return -1;
            }
        }
        /* Save metadata pointer before insert - hnsw_insert transfers ownership and NULLs it */
        GV_Metadata *saved_meta = vector->metadata;
        status = gv_hnsw_insert(db->hnsw_index, vector);
        if (status == 0 && saved_meta != NULL && db->metadata_index != NULL) {
            /* Update metadata index - use db->count as vector index */
            size_t vector_index = db->count;
            /* Walk the metadata via SoA storage since insert transferred ownership */
            GV_Metadata *current = saved_meta;
            while (current != NULL) {
                metadata_index_add(db->metadata_index, current->key, current->value, vector_index);
                current = current->next;
            }
        }
        /* HNSW does not store the shell; free it on success (metadata already
         * transferred to SoA, so vector->metadata is NULL) and on failure. */
        vector_destroy(vector);
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        GV_Vector *vector = vector_create_from_data(dimension, data);
        if (vector == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            db_decrement_concurrent_ops(db);
            return -1;
        }
        if (db->cosine_normalized) {
            db_normalize_vector(vector);
        }
        if (metadata_key != NULL && metadata_value != NULL) {
            if (vector_set_metadata(vector, metadata_key, metadata_value) != 0) {
                vector_destroy(vector);
                pthread_rwlock_unlock(&db->rwlock);
                db_decrement_concurrent_ops(db);
                return -1;
            }
        }
        GV_Metadata *saved_meta_for_ivfpq = vector->metadata;
        status = gv_ivfpq_insert(db->hnsw_index, vector);
        if (status == 0 && saved_meta_for_ivfpq != NULL && db->metadata_index != NULL) {
            size_t vector_index = db->count;
            GV_Metadata *current = saved_meta_for_ivfpq;
            while (current != NULL) {
                metadata_index_add(db->metadata_index, current->key, current->value, vector_index);
                current = current->next;
            }
        }
        if (status != 0) {
            vector_destroy(vector);
        }
    } else if (db->index_type == GV_INDEX_TYPE_FLAT ||
               db->index_type == GV_INDEX_TYPE_IVFFLAT ||
               db->index_type == GV_INDEX_TYPE_IVFSQ8 ||
               db->index_type == GV_INDEX_TYPE_IVFTURBOQUANT ||
               db->index_type == GV_INDEX_TYPE_PQ ||
               db->index_type == GV_INDEX_TYPE_LSH ||
       db->index_type == GV_INDEX_TYPE_RABITQ) {
        GV_Vector *vector = vector_create_from_data(dimension, data);
        if (vector == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            db_decrement_concurrent_ops(db);
            return -1;
        }
        if (db->cosine_normalized) {
            db_normalize_vector(vector);
        }
        if (metadata_key != NULL && metadata_value != NULL) {
            if (vector_set_metadata(vector, metadata_key, metadata_value) != 0) {
                vector_destroy(vector);
                pthread_rwlock_unlock(&db->rwlock);
                db_decrement_concurrent_ops(db);
                return -1;
            }
        }
        /* flat/pq/lsh inserts may destroy the vector and clear metadata; ivfflat keeps it */
        GV_Metadata *saved_meta_for_index = vector->metadata;

        if (db->index_type == GV_INDEX_TYPE_FLAT) {
            status = flat_insert(db->hnsw_index, vector);
        } else if (db->index_type == GV_INDEX_TYPE_IVFFLAT) {
            status = ivfflat_insert(db->hnsw_index, vector);
        } else if (db->index_type == GV_INDEX_TYPE_IVFSQ8) {
            status = ivfsq8_insert(db->hnsw_index, vector);
        } else if (db->index_type == GV_INDEX_TYPE_IVFTURBOQUANT) {
            status = ivfturboquant_insert(db->hnsw_index, vector);
        } else if (db->index_type == GV_INDEX_TYPE_PQ) {
            status = pq_insert(db->hnsw_index, vector);
        } else if (db->index_type == GV_INDEX_TYPE_RABITQ) {
            status = rabitq_insert(db->hnsw_index, vector);
        } else {
            status = lsh_insert(db->hnsw_index, vector);
        }
        if (status == 0 && saved_meta_for_index != NULL && db->metadata_index != NULL) {
            size_t vector_index = db->count;
            GV_Metadata *current = saved_meta_for_index;
            while (current != NULL) {
                metadata_index_add(db->metadata_index, current->key, current->value, vector_index);
                current = current->next;
            }
        }
        if (status != 0) {
            vector_destroy(vector);
        }
    }

    if (status != 0) {
        pthread_rwlock_unlock(&db->rwlock);
        db_decrement_concurrent_ops(db);
        return -1;
    }

    /* Append-after-apply under the write lock: WAL order matches in-memory
     * order, and no phantom insert is durably recorded for a failed apply. */
    if (db->wal != NULL && db->wal_replaying == 0) {
        pthread_mutex_lock(&db->wal_mutex);
        int wal_res = wal_append_insert(db->wal, data, dimension, metadata_key, metadata_value);
        if (wal_res == 0) {
            db->total_wal_records += 1;
        }
        pthread_mutex_unlock(&db->wal_mutex);
        if (wal_res != 0) {
            GV_LOG_ERROR("db_add_vector_with_metadata: wal_append_insert failed (rc=%d) - insert not durable",
                         wal_res);
            pthread_rwlock_unlock(&db->rwlock);
            db_decrement_concurrent_ops(db);
            return -1;
        }
    }

    size_t ts_slot_1 = db->count;
    db->count += 1;
    db->total_inserts += 1;
    if (db->tiering_enabled && db->tiered_storage) {
        tiered_storage_record_insert(db->tiered_storage, ts_slot_1, start_time_us);
    }
    db->generation += 1;
    size_t emit_index = db->count - 1;
    pthread_rwlock_unlock(&db->rwlock);

    db_emit_change(db, GV_CDC_INSERT, GV_EVENT_INSERT, emit_index, data, dimension);

    uint64_t end_time_us = db_get_time_us();
    uint64_t latency_us = end_time_us - start_time_us;
    db_record_latency(db, latency_us, 1);

    db_decrement_concurrent_ops(db);
    return 0;
}

int db_add_sparse_vector(GV_Database *db, const uint32_t *indices, const float *values,
                            size_t nnz, size_t dimension,
                            const char *metadata_key, const char *metadata_value) {
    if (db == NULL || db->index_type != GV_INDEX_TYPE_SPARSE || dimension != db->dimension) {
        return -1;
    }
    if ((indices == NULL || values == NULL) && nnz > 0) {
        return -1;
    }

    pthread_rwlock_wrlock(&db->rwlock);
    GV_SparseVector *sv = sparse_vector_create(dimension, indices, values, nnz);
    if (sv == NULL) {
        pthread_rwlock_unlock(&db->rwlock);
        return -1;
    }
    if (metadata_key && metadata_value) {
        /* GV_SparseVector and GV_Vector have different layouts: sparse->metadata
         * lives at offset 24 while GV_Vector::metadata is at offset 16 (the sparse
         * ::entries slot). Set the real sparse metadata field via a scratch
         * GV_Vector to avoid clobbering the entries pointer. */
        GV_Vector meta_holder = { 0, NULL, NULL };
        if (vector_set_metadata(&meta_holder, metadata_key, metadata_value) != 0) {
            sparse_vector_destroy(sv);
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        sv->metadata = meta_holder.metadata;
    }

    int status = sparse_index_add(db->sparse_index, sv);
    if (status != 0) {
        sparse_vector_destroy(sv);
        pthread_rwlock_unlock(&db->rwlock);
        return -1;
    }
    db->count += 1;
    db->total_inserts += 1;
    db->generation += 1;
    size_t emit_index = db->count - 1;
    pthread_rwlock_unlock(&db->rwlock);

    /* Sparse insert: no dense payload to attach (event still fires). */
    db_emit_change(db, GV_CDC_INSERT, GV_EVENT_INSERT, emit_index, NULL, 0);
    return 0;
}

int db_add_vector_with_rich_metadata(GV_Database *db, const float *data, size_t dimension,
                                        const char *const *metadata_keys, const char *const *metadata_values,
                                        size_t metadata_count) {
    if (db == NULL || data == NULL || dimension == 0 || dimension != db->dimension) {
        return -1;
    }
    if (metadata_count > 0 && (metadata_keys == NULL || metadata_values == NULL)) {
        return -1;
    }
    
    uint64_t start_time_us = db_get_time_us();

    pthread_rwlock_wrlock(&db->rwlock);

    int status = -1;
    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        if (db->soa_storage == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        int normalized_on_heap = 0;
        float *normalized_data = (float *)gv_tls_alloc_or_heap(
            dimension * sizeof(float), sizeof(float), &normalized_on_heap);
        if (normalized_data == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        memcpy(normalized_data, data, dimension * sizeof(float));
        if (db->cosine_normalized) {
            float norm_sq = 0.0f;
            for (size_t i = 0; i < dimension; ++i) {
                float v = normalized_data[i];
                norm_sq += v * v;
            }
            if (norm_sq > 0.0f) {
                float inv = 1.0f / sqrtf(norm_sq);
                for (size_t i = 0; i < dimension; ++i) {
                    normalized_data[i] *= inv;
                }
            }
        }
        GV_Metadata *metadata = NULL;
        if (metadata_count > 0) {
            GV_Vector temp_vec;
            temp_vec.dimension = dimension;
            temp_vec.data = NULL;
            temp_vec.metadata = NULL;
            for (size_t i = 0; i < metadata_count; i++) {
                if (metadata_keys[i] != NULL && metadata_values[i] != NULL) {
                    if (vector_set_metadata(&temp_vec, metadata_keys[i], metadata_values[i]) != 0) {
                        vector_clear_metadata(&temp_vec);
                        gv_tls_free_or_heap(normalized_data, normalized_on_heap);
                        pthread_rwlock_unlock(&db->rwlock);
                        return -1;
                    }
                }
            }
            metadata = temp_vec.metadata;
        }
        size_t vector_index = soa_storage_add(db->soa_storage, normalized_data, metadata);
        gv_tls_free_or_heap(normalized_data, normalized_on_heap);
        if (vector_index == (size_t)-1) {
            if (metadata != NULL) {
                GV_Vector temp_vec;
                temp_vec.dimension = dimension;
                temp_vec.data = NULL;
                temp_vec.metadata = metadata;
                vector_clear_metadata(&temp_vec);
            }
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        status = kdtree_insert(&(db->root), db->soa_storage, vector_index, 0);
        if (status == 0 && metadata_count > 0 && db->metadata_index != NULL) {
            for (size_t i = 0; i < metadata_count; i++) {
                if (metadata_keys[i] != NULL && metadata_values[i] != NULL) {
                    metadata_index_add(db->metadata_index, metadata_keys[i],
                                       metadata_values[i], vector_index);
                }
            }
        }
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        GV_Vector *vector = vector_create_from_data(dimension, data);
        if (vector == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        if (db->cosine_normalized) {
            db_normalize_vector(vector);
        }
        for (size_t i = 0; i < metadata_count; i++) {
            if (metadata_keys[i] != NULL && metadata_values[i] != NULL) {
                if (vector_set_metadata(vector, metadata_keys[i], metadata_values[i]) != 0) {
                    vector_destroy(vector);
                    pthread_rwlock_unlock(&db->rwlock);
                    return -1;
                }
            }
        }
        status = gv_hnsw_insert(db->hnsw_index, vector);
        if (status == 0 && metadata_count > 0 && db->metadata_index != NULL) {
            size_t vector_index = db->count;
            for (size_t i = 0; i < metadata_count; i++) {
                if (metadata_keys[i] != NULL && metadata_values[i] != NULL) {
                    metadata_index_add(db->metadata_index, metadata_keys[i],
                                       metadata_values[i], vector_index);
                }
            }
        }
        /* HNSW does not store the shell; free it on success (metadata already
         * transferred to SoA, so vector->metadata is NULL) and on failure. */
        vector_destroy(vector);
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        GV_Vector *vector = vector_create_from_data(dimension, data);
        if (vector == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        if (db->cosine_normalized) {
            db_normalize_vector(vector);
        }
        for (size_t i = 0; i < metadata_count; i++) {
            if (metadata_keys[i] != NULL && metadata_values[i] != NULL) {
                if (vector_set_metadata(vector, metadata_keys[i], metadata_values[i]) != 0) {
                    vector_destroy(vector);
                    pthread_rwlock_unlock(&db->rwlock);
                    return -1;
                }
            }
        }
        status = gv_ivfpq_insert(db->hnsw_index, vector);
        if (status == 0 && metadata_count > 0 && db->metadata_index != NULL) {
            size_t vector_index = db->count;
            for (size_t i = 0; i < metadata_count; i++) {
                if (metadata_keys[i] != NULL && metadata_values[i] != NULL) {
                    metadata_index_add(db->metadata_index, metadata_keys[i],
                                       metadata_values[i], vector_index);
                }
            }
        }
        if (status != 0) {
            vector_destroy(vector);
        }
    } else if (db->index_type == GV_INDEX_TYPE_IVFDISK) {
        if (ivfdisk_is_trained((GV_IVFDiskIndex *)db->hnsw_index) == 0) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        int normalized_on_heap = 0;
        float *normalized_data = (float *)gv_tls_alloc_or_heap(
            dimension * sizeof(float), sizeof(float), &normalized_on_heap);
        if (normalized_data == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        memcpy(normalized_data, data, dimension * sizeof(float));
        if (db->cosine_normalized) {
            float norm_sq = 0.0f;
            for (size_t i = 0; i < dimension; ++i) {
                float v = normalized_data[i];
                norm_sq += v * v;
            }
            if (norm_sq > 0.0f) {
                float inv = 1.0f / sqrtf(norm_sq);
                for (size_t i = 0; i < dimension; ++i) {
                    normalized_data[i] *= inv;
                }
            }
        }
        GV_Metadata *metadata = NULL;
        if (metadata_count > 0) {
            GV_Vector temp_vec;
            temp_vec.dimension = dimension;
            temp_vec.data = NULL;
            temp_vec.metadata = NULL;
            for (size_t i = 0; i < metadata_count; i++) {
                if (metadata_keys[i] != NULL && metadata_values[i] != NULL) {
                    if (vector_set_metadata(&temp_vec, metadata_keys[i], metadata_values[i]) != 0) {
                        vector_clear_metadata(&temp_vec);
                        gv_tls_free_or_heap(normalized_data, normalized_on_heap);
                        pthread_rwlock_unlock(&db->rwlock);
                        return -1;
                    }
                }
            }
            metadata = temp_vec.metadata;
        }
        size_t vector_index = soa_storage_add(db->soa_storage, normalized_data, metadata);
        gv_tls_free_or_heap(normalized_data, normalized_on_heap);
        if (vector_index == (size_t)-1) {
            if (metadata != NULL) {
                GV_Vector temp_vec;
                temp_vec.dimension = dimension;
                temp_vec.data = NULL;
                temp_vec.metadata = metadata;
                vector_clear_metadata(&temp_vec);
            }
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        if (metadata_count > 0 && db->metadata_index != NULL) {
            for (size_t i = 0; i < metadata_count; i++) {
                if (metadata_keys[i] != NULL && metadata_values[i] != NULL) {
                    metadata_index_add(db->metadata_index, metadata_keys[i],
                                       metadata_values[i], vector_index);
                }
            }
        }
        const float *stored = soa_storage_get_data(db->soa_storage, vector_index);
        if (db->wal_replaying) {
            status = 0;
        } else {
            uint64_t heads[2];
            size_t nh = 0;
            status = ivfdisk_insert_routed((GV_IVFDiskIndex *)db->hnsw_index, stored,
                                           dimension, vector_index, heads, &nh, 2);
            if (status == 0 && db->wal != NULL && !g_txn_wal_suppress) {
                pthread_mutex_lock(&db->wal_mutex);
                for (size_t hi = 0; hi < nh; ++hi) {
                    if (wal_append_ivfdisk_append(db->wal, heads[hi], (uint64_t)vector_index,
                                                  stored, dimension) != 0) {
                        status = -1;
                        break;
                    }
                }
                pthread_mutex_unlock(&db->wal_mutex);
            }
        }
    } else if (db->index_type == GV_INDEX_TYPE_FLAT ||
               db->index_type == GV_INDEX_TYPE_IVFFLAT ||
         db->index_type == GV_INDEX_TYPE_IVFSQ8 ||
         db->index_type == GV_INDEX_TYPE_IVFTURBOQUANT ||
               db->index_type == GV_INDEX_TYPE_PQ ||
               db->index_type == GV_INDEX_TYPE_LSH ||
       db->index_type == GV_INDEX_TYPE_RABITQ) {
        GV_Vector *vector = vector_create_from_data(dimension, data);
        if (vector == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        if (db->cosine_normalized) {
            db_normalize_vector(vector);
        }
        for (size_t i = 0; i < metadata_count; i++) {
            if (metadata_keys[i] != NULL && metadata_values[i] != NULL) {
                if (vector_set_metadata(vector, metadata_keys[i], metadata_values[i]) != 0) {
                    vector_destroy(vector);
                    pthread_rwlock_unlock(&db->rwlock);
                    return -1;
                }
            }
        }
        if (db->index_type == GV_INDEX_TYPE_FLAT) {
            status = flat_insert(db->hnsw_index, vector);
        } else if (db->index_type == GV_INDEX_TYPE_IVFFLAT) {
            status = ivfflat_insert(db->hnsw_index, vector);
        } else if (db->index_type == GV_INDEX_TYPE_IVFSQ8) {
            status = ivfsq8_insert(db->hnsw_index, vector);
        } else if (db->index_type == GV_INDEX_TYPE_IVFTURBOQUANT) {
            status = ivfturboquant_insert(db->hnsw_index, vector);
        } else if (db->index_type == GV_INDEX_TYPE_PQ) {
            status = pq_insert(db->hnsw_index, vector);
        } else if (db->index_type == GV_INDEX_TYPE_RABITQ) {
            status = rabitq_insert(db->hnsw_index, vector);
        } else {
            status = lsh_insert(db->hnsw_index, vector);
        }
        if (status == 0 && metadata_count > 0 && db->metadata_index != NULL) {
            size_t vector_index = db->count;
            for (size_t i = 0; i < metadata_count; i++) {
                if (metadata_keys[i] != NULL && metadata_values[i] != NULL) {
                    metadata_index_add(db->metadata_index, metadata_keys[i],
                                       metadata_values[i], vector_index);
                }
            }
        }
        if (status != 0) {
            vector_destroy(vector);
        }
    }

    if (status != 0) {
        pthread_rwlock_unlock(&db->rwlock);
        db_decrement_concurrent_ops(db);
        return -1;
    }

    /* Append-after-apply under the write lock: the generic INSERT record is
     * written only after the in-memory insert (and, for IVFDISK, its routed
     * append records) succeeded. This keeps WAL order consistent with the
     * positional in-memory order and prevents durably recording an insert
     * that never applied. */
    if (db->wal != NULL && db->wal_replaying == 0) {
        pthread_mutex_lock(&db->wal_mutex);
        int wal_res = wal_append_insert_rich(db->wal, data, dimension, metadata_keys, metadata_values, metadata_count);
        if (wal_res == 0) {
            db->total_wal_records += 1;
        }
        pthread_mutex_unlock(&db->wal_mutex);
        if (wal_res != 0) {
            GV_LOG_ERROR("db_add_vector_with_rich_metadata: wal_append_insert_rich failed (rc=%d) - insert not durable",
                         wal_res);
            pthread_rwlock_unlock(&db->rwlock);
            db_decrement_concurrent_ops(db);
            return -1;
        }
    }

    size_t ts_slot_2 = db->count;
    db->count += 1;
    db->total_inserts += 1;
    db->generation += 1;
    db_update_memory_usage(db);
    if (db->tiering_enabled && db->tiered_storage) {
        tiered_storage_record_insert(db->tiered_storage, ts_slot_2, start_time_us);
    }
    size_t emit_index = db->count - 1;
    pthread_rwlock_unlock(&db->rwlock);

    db_emit_change(db, GV_CDC_INSERT, GV_EVENT_INSERT, emit_index, data, dimension);

    uint64_t end_time_us = db_get_time_us();
    uint64_t latency_us = end_time_us - start_time_us;
    db_record_latency(db, latency_us, 1);

    db_decrement_concurrent_ops(db);
    return 0;
}

int db_ivfpq_train(GV_Database *db, const float *data, size_t count, size_t dimension) {
    if (db == NULL || data == NULL || count == 0 || dimension != db->dimension) {
        return -1;
    }
    if (db->index_type != GV_INDEX_TYPE_IVFPQ || db->hnsw_index == NULL) {
        return -1;
    }
    return gv_ivfpq_train(db->hnsw_index, data, count);
}

int db_ivfflat_train(GV_Database *db, const float *data, size_t count, size_t dimension) {
    if (db == NULL || data == NULL || count == 0 || dimension != db->dimension) {
        return -1;
    }
    if (db->index_type != GV_INDEX_TYPE_IVFFLAT || db->hnsw_index == NULL) {
        return -1;
    }
    return ivfflat_train(db->hnsw_index, data, count);
}

int db_ivfdisk_train(GV_Database *db, const float *data, size_t count, size_t dimension) {
    if (db == NULL || data == NULL || count == 0 || dimension != db->dimension) {
        return -1;
    }
    if (db->index_type != GV_INDEX_TYPE_IVFDISK || db->hnsw_index == NULL) {
        return -1;
    }
    return ivfdisk_train((GV_IVFDiskIndex *)db->hnsw_index, data, count);
}

int db_ivfsq8_train(GV_Database *db, const float *data, size_t count, size_t dimension) {
    if (db == NULL || data == NULL || count == 0 || dimension != db->dimension) {
        return -1;
    }
    if (db->index_type != GV_INDEX_TYPE_IVFSQ8 || db->hnsw_index == NULL) {
        return -1;
    }
    return ivfsq8_train(db->hnsw_index, data, count);
}

int db_ivfturboquant_train(GV_Database *db, const float *data, size_t count, size_t dimension) {
    if (db == NULL || data == NULL || count == 0 || dimension != db->dimension) {
        return -1;
    }
    if (db->index_type != GV_INDEX_TYPE_IVFTURBOQUANT || db->hnsw_index == NULL) {
        return -1;
    }
    return ivfturboquant_train(db->hnsw_index, data, count);
}

int db_pq_train(GV_Database *db, const float *data, size_t count, size_t dimension) {
    if (db == NULL || data == NULL || count == 0 || dimension != db->dimension) {
        return -1;
    }
    if (db->index_type != GV_INDEX_TYPE_PQ || db->hnsw_index == NULL) {
        return -1;
    }
    return pq_train(db->hnsw_index, data, count);
}

int db_add_vectors(GV_Database *db, const float *data, size_t count, size_t dimension) {
    if (db == NULL || data == NULL || count == 0 || dimension != db->dimension) {
        return -1;
    }

    if (db->index_type == GV_INDEX_TYPE_HNSW && db->hnsw_index != NULL &&
        db->wal == NULL && !db->cosine_normalized) {
        pthread_rwlock_wrlock(&db->rwlock);
        gv_hnsw_reserve(db->hnsw_index, count);

        size_t emit_start = db->count;
        size_t inserted = 0;
        for (size_t i = 0; i < count; ++i) {
            const float *vec = data + i * dimension;
            int status = gv_hnsw_insert_raw(db->hnsw_index, vec, dimension);
            if (status != 0) {
                break;
            }
            db->count += 1;
            db->total_inserts += 1;
            inserted += 1;
        }

        db_update_memory_usage(db);
        pthread_rwlock_unlock(&db->rwlock);

        /* Emit change events only for the vectors that were actually committed
         * (mirrors the partial-progress return contract below). */
        for (size_t i = 0; i < inserted; ++i) {
            db_emit_change(db, GV_CDC_INSERT, GV_EVENT_INSERT, emit_start + i,
                           data + i * dimension, dimension);
        }
        /* 0 on full success, -1 if any insert failed (partial commit may remain;
         * callers that need the committed count should insert one at a time). */
        return (inserted == count) ? 0 : -1;
    }

    size_t inserted = 0;
    for (size_t i = 0; i < count; ++i) {
        const float *vec = data + i * dimension;
        if (db_add_vector(db, vec, dimension) != 0) {
            break;
        }
        inserted += 1;
    }
    return (inserted == count) ? 0 : -1;
}

int db_add_vectors_with_metadata(GV_Database *db, const float *data,
                                    const char *const *keys, const char *const *values,
                                    size_t count, size_t dimension) {
    if (db == NULL || data == NULL || count == 0 || dimension != db->dimension) {
        return -1;
    }
    for (size_t i = 0; i < count; ++i) {
        const float *vec = data + i * dimension;
        const char *k = (keys != NULL) ? keys[i] : NULL;
        const char *v = (values != NULL) ? values[i] : NULL;
        if (db_add_vector_with_metadata(db, vec, dimension, k, v) != 0) {
            return -1;
        }
    }
    return 0;
}

/* Internal save routine. Caller MUST hold db->rwlock (read or write). Writes to
 * a temporary file and atomically renames it over out_path on success so a
 * crash mid-save cannot destroy the existing snapshot. */

int db_delete_vector_by_index(GV_Database *db, size_t vector_index) {
    if (db == NULL) {
        return -1;
    }

    pthread_rwlock_wrlock(&db->rwlock);

    /* Validate the target index for the active index type before mutating any
     * state. This lets us append to the WAL first (durability) and only then
     * apply the in-memory mutation. */
    switch (db->index_type) {
        case GV_INDEX_TYPE_KDTREE:
            if (db->soa_storage == NULL || vector_index >= db->soa_storage->count) {
                pthread_rwlock_unlock(&db->rwlock);
                return -1;
            }
            break;
        case GV_INDEX_TYPE_SPARSE:
            if (db->sparse_index == NULL) {
                pthread_rwlock_unlock(&db->rwlock);
                return -1;
            }
            break;
        case GV_INDEX_TYPE_HNSW:
        case GV_INDEX_TYPE_IVFPQ:
        case GV_INDEX_TYPE_FLAT:
        case GV_INDEX_TYPE_IVFFLAT:
        case GV_INDEX_TYPE_IVFSQ8:
        case GV_INDEX_TYPE_IVFTURBOQUANT:
        case GV_INDEX_TYPE_PQ:
        case GV_INDEX_TYPE_LSH:
        case GV_INDEX_TYPE_RABITQ:
            if (db->hnsw_index == NULL) {
                pthread_rwlock_unlock(&db->rwlock);
                return -1;
            }
            break;
        case GV_INDEX_TYPE_IVFDISK:
            if (db->hnsw_index == NULL || db->soa_storage == NULL) {
                pthread_rwlock_unlock(&db->rwlock);
                return -1;
            }
            if (vector_index >= db->soa_storage->count ||
                soa_storage_is_deleted(db->soa_storage, vector_index) == 1) {
                pthread_rwlock_unlock(&db->rwlock);
                return -1;
            }
            break;
        default:
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
    }

    /* WAL-first: append the delete record before mutating in-memory state so a
     * WAL failure leaves the database unchanged. The WAL append is serialized
     * with wal_mutex (acquired after rwlock, matching the insert path's
     * ordering) so records from concurrent operations cannot interleave. */
    if (db->wal != NULL) {
        pthread_mutex_lock(&db->wal_mutex);
        int wal_res = wal_append_delete(db->wal, vector_index);
        pthread_mutex_unlock(&db->wal_mutex);
        if (wal_res != 0) {
            GV_LOG_ERROR("db_delete_vector_by_index: wal_append_delete(index=%zu) failed (rc=%d) - delete not durable",
                         vector_index, wal_res);
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        db->total_wal_records += 1;
    }

    int status = -1;
    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        status = kdtree_delete(&(db->root), db->soa_storage, vector_index);
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        status = gv_hnsw_delete_by_vector_index(db->hnsw_index, vector_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        status = gv_ivfpq_delete(db->hnsw_index, vector_index);
    } else if (db->index_type == GV_INDEX_TYPE_SPARSE) {
        status = sparse_index_delete(db->sparse_index, vector_index);
    } else if (db->index_type == GV_INDEX_TYPE_FLAT) {
        status = flat_delete(db->hnsw_index, vector_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFFLAT) {
        status = ivfflat_delete(db->hnsw_index, vector_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFSQ8) {
        status = ivfsq8_delete(db->hnsw_index, vector_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFTURBOQUANT) {
        status = ivfturboquant_delete(db->hnsw_index, vector_index);
    } else if (db->index_type == GV_INDEX_TYPE_PQ) {
        status = pq_delete(db->hnsw_index, vector_index);
    } else if (db->index_type == GV_INDEX_TYPE_LSH) {
        status = lsh_delete(db->hnsw_index, vector_index);
    } else if (db->index_type == GV_INDEX_TYPE_RABITQ) {
        status = rabitq_delete(db->hnsw_index, vector_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFDISK) {
        const float *stored = soa_storage_get_data(db->soa_storage, vector_index);
        status = ivfdisk_delete((GV_IVFDiskIndex *)db->hnsw_index, vector_index, stored);
        if (status == 0) {
            status = soa_storage_mark_deleted(db->soa_storage, vector_index);
        }
    }

    if (status == 0) {
        if (db->metadata_index != NULL) {
            metadata_index_remove_vector(db->metadata_index, vector_index);
        }
        db->generation += 1;
    }

    pthread_rwlock_unlock(&db->rwlock);

    if (status == 0) {
        db_emit_change(db, GV_CDC_DELETE, GV_EVENT_DELETE, vector_index, NULL, 0);
    }
    return status;
}

int db_update_vector(GV_Database *db, size_t vector_index, const float *new_data, size_t dimension) {
    if (db == NULL || new_data == NULL || dimension != db->dimension) {
        return -1;
    }

    pthread_rwlock_wrlock(&db->rwlock);

    /* Validate the target index for the active index type before mutating any
     * state so we can append to the WAL first (durability). */
    switch (db->index_type) {
        case GV_INDEX_TYPE_KDTREE:
            if (db->soa_storage == NULL || vector_index >= db->soa_storage->count) {
                pthread_rwlock_unlock(&db->rwlock);
                return -1;
            }
            if (soa_storage_is_deleted(db->soa_storage, vector_index) == 1) {
                pthread_rwlock_unlock(&db->rwlock);
                return -1;
            }
            break;
        case GV_INDEX_TYPE_HNSW:
        case GV_INDEX_TYPE_IVFPQ:
        case GV_INDEX_TYPE_FLAT:
        case GV_INDEX_TYPE_IVFFLAT:
        case GV_INDEX_TYPE_IVFSQ8:
        case GV_INDEX_TYPE_IVFTURBOQUANT:
        case GV_INDEX_TYPE_PQ:
        case GV_INDEX_TYPE_LSH:
        case GV_INDEX_TYPE_RABITQ:
            if (db->hnsw_index == NULL) {
                pthread_rwlock_unlock(&db->rwlock);
                return -1;
            }
            break;
        case GV_INDEX_TYPE_IVFDISK:
            if (db->hnsw_index == NULL || db->soa_storage == NULL) {
                pthread_rwlock_unlock(&db->rwlock);
                return -1;
            }
            if (vector_index >= db->soa_storage->count ||
                soa_storage_is_deleted(db->soa_storage, vector_index) == 1) {
                pthread_rwlock_unlock(&db->rwlock);
                return -1;
            }
            break;
        case GV_INDEX_TYPE_SPARSE:
        default:
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
    }

    /* WAL-first: append the update record before mutating in-memory state so a
     * WAL failure leaves the database unchanged. Serialized with wal_mutex
     * (acquired after rwlock, matching the insert path's ordering). */
    if (db->wal != NULL) {
        pthread_mutex_lock(&db->wal_mutex);
        int wal_res = wal_append_update(db->wal, vector_index, new_data, dimension, NULL, NULL, 0);
        pthread_mutex_unlock(&db->wal_mutex);
        if (wal_res != 0) {
            GV_LOG_ERROR("db_update_vector: wal_append_update(index=%zu) failed (rc=%d) - update not durable",
                         vector_index, wal_res);
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        db->total_wal_records += 1;
    }

    int status = -1;
    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        if (db->filepath == NULL) {
            // In-memory KDTREE: update only SOA storage
            status = soa_storage_update_data(db->soa_storage, vector_index, new_data);
        } else {
            // File-based KDTREE: update tree and SOA
            status = kdtree_update(&(db->root), db->soa_storage, vector_index, new_data);
        }
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        status = gv_hnsw_update(db->hnsw_index, vector_index, new_data, dimension);
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        status = gv_ivfpq_update(db->hnsw_index, vector_index, new_data, dimension);
    } else if (db->index_type == GV_INDEX_TYPE_FLAT) {
        status = flat_update(db->hnsw_index, vector_index, new_data, dimension);
    } else if (db->index_type == GV_INDEX_TYPE_IVFFLAT) {
        status = ivfflat_update(db->hnsw_index, vector_index, new_data, dimension);
    } else if (db->index_type == GV_INDEX_TYPE_IVFSQ8) {
        status = ivfsq8_update(db->hnsw_index, vector_index, new_data, dimension);
    } else if (db->index_type == GV_INDEX_TYPE_IVFTURBOQUANT) {
        status = ivfturboquant_update(db->hnsw_index, vector_index, new_data, dimension);
    } else if (db->index_type == GV_INDEX_TYPE_PQ) {
        status = pq_update(db->hnsw_index, vector_index, new_data, dimension);
    } else if (db->index_type == GV_INDEX_TYPE_LSH) {
        status = lsh_update(db->hnsw_index, vector_index, new_data, dimension);
    } else if (db->index_type == GV_INDEX_TYPE_RABITQ) {
        status = rabitq_update(db->hnsw_index, vector_index, new_data, dimension);
    } else if (db->index_type == GV_INDEX_TYPE_IVFDISK) {
        status = soa_storage_update_data(db->soa_storage, vector_index, new_data);
        if (status == 0) {
            status = ivfdisk_update((GV_IVFDiskIndex *)db->hnsw_index, vector_index,
                                    new_data, dimension);
        }
    }

    /* Refresh insertion timestamp on successful update. */
    if (status == 0 && db->soa_storage != NULL &&
        vector_index < db->soa_storage->count) {
        soa_storage_set_timestamp(db->soa_storage, vector_index, gv_time_now_ms());
    }

    /* WAL append already performed above (WAL-first). */
    if (status == 0) {
        db->generation += 1;
    }

    pthread_rwlock_unlock(&db->rwlock);

    if (status == 0) {
        db_emit_change(db, GV_CDC_UPDATE, GV_EVENT_UPDATE, vector_index, new_data, dimension);
    }
    return status;
}

int db_update_vector_metadata(GV_Database *db, size_t vector_index,
                                  const char *const *metadata_keys, const char *const *metadata_values,
                                  size_t metadata_count) {
    if (db == NULL || vector_index >= db->count) {
        return -1;
    }

    pthread_rwlock_wrlock(&db->rwlock);

    int status = -1;
    const float *vector_data = NULL;

    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        if (db->soa_storage == NULL || vector_index >= db->soa_storage->count) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        if (soa_storage_is_deleted(db->soa_storage, vector_index) == 1) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        vector_data = soa_storage_get_data(db->soa_storage, vector_index);
        if (vector_data == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        
        /* Merge: start from existing metadata, overlay updated keys */
        GV_Vector temp_vec;
        temp_vec.dimension = db->dimension;
        temp_vec.data = NULL;
        temp_vec.metadata = NULL;

        GV_Metadata *old_metadata = soa_storage_get_metadata(db->soa_storage, vector_index);
        for (GV_Metadata *cur = old_metadata; cur != NULL; cur = cur->next) {
            if (cur->key != NULL && cur->value != NULL) {
                if (vector_set_metadata(&temp_vec, cur->key, cur->value) != 0) {
                    vector_clear_metadata(&temp_vec);
                    pthread_rwlock_unlock(&db->rwlock);
                    return -1;
                }
            }
        }
        
        for (size_t i = 0; i < metadata_count; ++i) {
            if (metadata_keys[i] != NULL && metadata_values[i] != NULL) {
                if (vector_set_metadata(&temp_vec, metadata_keys[i], metadata_values[i]) != 0) {
                    vector_clear_metadata(&temp_vec);
                    pthread_rwlock_unlock(&db->rwlock);
                    return -1;
                }
            }
        }
        
        /* Copy old metadata for inverted-index diff before update */
        GV_Metadata *old_metadata_copy = NULL;
        if (old_metadata != NULL && db->metadata_index != NULL) {
            /* Copy old metadata chain to avoid use-after-gv_free */
            GV_Metadata *current = old_metadata;
            GV_Metadata *prev = NULL;
            while (current != NULL) {
                GV_Metadata *copy = (GV_Metadata *)gv_db_alloc(db, sizeof(GV_Metadata));
                if (copy == NULL) {
                    /* Free what we've copied so far */
                    while (old_metadata_copy != NULL) {
                        GV_Metadata *next = old_metadata_copy->next;
                        gv_free(old_metadata_copy->key);
                        gv_free(old_metadata_copy->value);
                        gv_db_free(db, old_metadata_copy);
                        old_metadata_copy = next;
                    }
                    vector_clear_metadata(&temp_vec);
                    pthread_rwlock_unlock(&db->rwlock);
                    return -1;
                }
                copy->key = current->key ? gv_dup_cstr(current->key) : NULL;
                copy->value = current->value ? gv_dup_cstr(current->value) : NULL;
                copy->next = NULL;
                if (prev == NULL) {
                    old_metadata_copy = copy;
                } else {
                    prev->next = copy;
                }
                prev = copy;
                current = current->next;
            }
        }
        
        status = soa_storage_update_metadata(db->soa_storage, vector_index, temp_vec.metadata);
        // Ownership transferred to SOA, prevent double-gv_free
        temp_vec.metadata = NULL;

        if (status == 0 && db->metadata_index != NULL) {
            GV_Metadata *new_metadata = soa_storage_get_metadata(db->soa_storage, vector_index);
            metadata_index_update(db->metadata_index, vector_index, old_metadata_copy, new_metadata);
        }
        
        while (old_metadata_copy != NULL) {
            GV_Metadata *next = old_metadata_copy->next;
            gv_free(old_metadata_copy->key);
            gv_free(old_metadata_copy->value);
            gv_db_free(db, old_metadata_copy);
            old_metadata_copy = next;
        }
    } else if (db->index_type == GV_INDEX_TYPE_HNSW ||
               db->index_type == GV_INDEX_TYPE_IVFPQ ||
               db->index_type == GV_INDEX_TYPE_FLAT ||
               db->index_type == GV_INDEX_TYPE_LSH ||
               db->index_type == GV_INDEX_TYPE_RABITQ) {
        /* All these index types use SoA storage for metadata */
        if (db->soa_storage == NULL || vector_index >= db->soa_storage->count) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        if (soa_storage_is_deleted(db->soa_storage, vector_index) == 1) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        vector_data = soa_storage_get_data(db->soa_storage, vector_index);
        if (vector_data == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }

        GV_Vector temp_vec;
        temp_vec.dimension = db->dimension;
        temp_vec.data = NULL;
        temp_vec.metadata = NULL;

        GV_Metadata *old_metadata = soa_storage_get_metadata(db->soa_storage, vector_index);
        for (GV_Metadata *cur = old_metadata; cur != NULL; cur = cur->next) {
            if (cur->key != NULL && cur->value != NULL) {
                if (vector_set_metadata(&temp_vec, cur->key, cur->value) != 0) {
                    vector_clear_metadata(&temp_vec);
                    pthread_rwlock_unlock(&db->rwlock);
                    return -1;
                }
            }
        }

        for (size_t i = 0; i < metadata_count; ++i) {
            if (metadata_keys[i] != NULL && metadata_values[i] != NULL) {
                if (vector_set_metadata(&temp_vec, metadata_keys[i], metadata_values[i]) != 0) {
                    vector_clear_metadata(&temp_vec);
                    pthread_rwlock_unlock(&db->rwlock);
                    return -1;
                }
            }
        }

        GV_Metadata *old_metadata_copy = NULL;
        if (old_metadata != NULL && db->metadata_index != NULL) {
            GV_Metadata *current = old_metadata;
            GV_Metadata *prev = NULL;
            while (current != NULL) {
                GV_Metadata *copy = (GV_Metadata *)gv_db_alloc(db, sizeof(GV_Metadata));
                if (copy == NULL) {
                    while (old_metadata_copy != NULL) {
                        GV_Metadata *next = old_metadata_copy->next;
                        gv_free(old_metadata_copy->key);
                        gv_free(old_metadata_copy->value);
                        gv_db_free(db, old_metadata_copy);
                        old_metadata_copy = next;
                    }
                    vector_clear_metadata(&temp_vec);
                    pthread_rwlock_unlock(&db->rwlock);
                    return -1;
                }
                copy->key = current->key ? gv_dup_cstr(current->key) : NULL;
                copy->value = current->value ? gv_dup_cstr(current->value) : NULL;
                copy->next = NULL;
                if (prev == NULL) {
                    old_metadata_copy = copy;
                } else {
                    prev->next = copy;
                }
                prev = copy;
                current = current->next;
            }
        }

        status = soa_storage_update_metadata(db->soa_storage, vector_index, temp_vec.metadata);
        temp_vec.metadata = NULL;

        if (status == 0 && db->metadata_index != NULL) {
            GV_Metadata *new_metadata = soa_storage_get_metadata(db->soa_storage, vector_index);
            metadata_index_update(db->metadata_index, vector_index, old_metadata_copy, new_metadata);
        }

        while (old_metadata_copy != NULL) {
            GV_Metadata *next = old_metadata_copy->next;
            gv_free(old_metadata_copy->key);
            gv_free(old_metadata_copy->value);
            gv_db_free(db, old_metadata_copy);
            old_metadata_copy = next;
        }
    } else if (db->index_type == GV_INDEX_TYPE_SPARSE) {
        /* Update per-vector metadata in place via the sparse-index setter
         * (sparse_index_set_metadata attaches onto GV_SparseVector::metadata). */
        if (db->sparse_index == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        status = 0;
        for (size_t i = 0; i < metadata_count; i++) {
            if (metadata_keys[i] == NULL || metadata_values[i] == NULL) {
                continue;
            }
            if (sparse_index_set_metadata(db->sparse_index, vector_index,
                                          metadata_keys[i], metadata_values[i]) != 0) {
                status = -1;
                break;
            }
            if (db->metadata_index != NULL) {
                metadata_index_add(db->metadata_index, metadata_keys[i],
                                   metadata_values[i], vector_index);
            }
        }
    } else {
        pthread_rwlock_unlock(&db->rwlock);
        return -1;
    }

    if (status == 0 && db->wal != NULL && vector_data != NULL) {
        /* Serialize the WAL append with wal_mutex (acquired after rwlock) so it
         * cannot interleave with concurrent inserts/deletes/updates. */
        pthread_mutex_lock(&db->wal_mutex);
        int wal_res = wal_append_update(db->wal, vector_index, vector_data, db->dimension,
                                        metadata_keys, metadata_values, metadata_count);
        pthread_mutex_unlock(&db->wal_mutex);
        if (wal_res != 0) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        db->total_wal_records += 1;
    }

    pthread_rwlock_unlock(&db->rwlock);
    return status;
}

int db_upsert(GV_Database *db, size_t vector_index, const float *data, size_t dimension) {
    if (db == NULL || data == NULL || dimension != db->dimension) {
        return -1;
    }

    if (vector_index < db->count) {
        return db_update_vector(db, vector_index, data, dimension);
    } else if (vector_index == db->count) {
        return db_add_vector(db, data, dimension);
    }

    return -1; /* Index out of range */
}

int db_upsert_with_metadata(GV_Database *db, size_t vector_index,
                                const float *data, size_t dimension,
                                const char *const *metadata_keys,
                                const char *const *metadata_values,
                                size_t metadata_count) {
    if (db == NULL || data == NULL || dimension != db->dimension) {
        return -1;
    }

    if (vector_index < db->count) {
        int status = db_update_vector(db, vector_index, data, dimension);
        if (status != 0) return status;
        if (metadata_keys && metadata_values && metadata_count > 0) {
            return db_update_vector_metadata(db, vector_index,
                                                 metadata_keys, metadata_values,
                                                 metadata_count);
        }
        return 0;
    } else if (vector_index == db->count) {
        return db_add_vector_with_rich_metadata(db, data, dimension,
                                                    metadata_keys, metadata_values,
                                                    metadata_count);
    }

    return -1;
}

int db_delete_vectors(GV_Database *db, const size_t *indices, size_t count) {
    if (db == NULL || indices == NULL || count == 0) {
        return -1;
    }

    int deleted = 0;
    for (size_t i = 0; i < count; i++) {
        if (db_delete_vector_by_index(db, indices[i]) == 0) {
            deleted++;
        }
    }
    return deleted;
}

int db_scroll(const GV_Database *db, size_t offset, size_t limit,
                 GV_ScrollResult *results) {
    if (db == NULL || results == NULL || limit == 0) {
        return -1;
    }

    pthread_rwlock_rdlock((pthread_rwlock_t *)&db->rwlock);

    if (db->soa_storage != NULL) {
        size_t found = 0;
        size_t skipped = 0;
        size_t total = db->soa_storage->count;

        for (size_t i = 0; i < total && found < limit; i++) {
            if (soa_storage_is_deleted(db->soa_storage, i) == 1) {
                continue;
            }
            if (skipped < offset) {
                skipped++;
                continue;
            }
            results[found].index = i;
            results[found].data = soa_storage_get_data(db->soa_storage, i);
            results[found].dimension = db->dimension;
            results[found].metadata = soa_storage_get_metadata(db->soa_storage, i);
            found++;
        }

        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return (int)found;
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
    return 0;
}
