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
static FILE *fmemopen(void *buf, size_t size, const char *mode) {
    (void)mode;
    char dir[MAX_PATH];
    char path[MAX_PATH];
    if (GetTempPathA((DWORD)sizeof(dir), dir) == 0) return NULL;
    if (GetTempFileNameA(dir, "gvm", 0, path) == 0) return NULL;
    FILE *tmp = fopen(path, "wb+");
    if (!tmp) {
        DeleteFileA(path);
        return NULL;
    }
    if (fwrite(buf, 1, size, tmp) != size) {
        fclose(tmp);
        DeleteFileA(path);
        return NULL;
    }
    rewind(tmp);
    return tmp;
}
/* getline() shim provided by core/compat.h (included below) on Windows/MinGW. */
#endif

#include "core/types.h"
#include "core/scope.h"
#include "core/memory.h"
#include "core/log.h"

#include "storage/database.h"
#include "storage/db_internal.h"
#include "search/distance.h"
#include "index/exact_search.h"
#include "index/hnsw.h"
#include "index/ivfpq.h"
#include "index/kdtree.h"
#include "schema/metadata.h"
#include "index/sparse_index.h"
#include "schema/vector.h"
#include "storage/wal.h"
#include "storage/mmap.h"
#include "storage/soa_storage.h"
#include "multimodal/metadata_index.h"
#include "index/flat.h"
#include "index/ivfflat.h"
#include "index/ivfdisk.h"
#include "index/index_maintenance.h"
#include "index/ivfsq8.h"
#include "index/ivfturboquant.h"
#include "index/pq.h"
#include "index/lsh.h"
#include "index/rabitq.h"
#include "core/utils.h"
#include "core/sim_time.h"
#include "search/filter.h"
#include "specialized/optimizer.h"
#include "index/ivf_retrain.h"
#include "admin/ab_test.h"
#include "storage/tiered_storage.h"
#include "storage/value_store.h"
#include "admin/cdc.h"
#include "admin/webhook.h"
#include "specialized/point_id.h"

#include <math.h>
#ifndef _WIN32
#include <sys/time.h>
#endif
#include "core/compat.h"

/* helpers now defined in db_stats.c and db_resource.c */

void db_fill_ivfdisk_search_vectors(GV_Database *db, GV_SearchResult *results, int n)
{
    if (!db || !results || n <= 0 || db->soa_storage == NULL) return;
    for (int i = 0; i < n; ++i) {
        GV_Vector view;
        memset(&view, 0, sizeof(view));
        if (soa_storage_get_vector_view(db->soa_storage, results[i].id, &view) != 0 ||
            view.data == NULL) {
            continue;
        }
        results[i].vector = vector_create_from_data(view.dimension, view.data);
    }
}

static int db_ivfdisk_create_index(GV_Database *db, size_t dimension, const char *filepath,
                                   const GV_IVFDiskConfig *config)
{
    char data_dir[1024];
    GV_IVFDiskConfig cfg;
    ivfdisk_config_init(&cfg);
    if (config) cfg = *config;
    if (!cfg.data_dir || !*cfg.data_dir) {
        if (!filepath || !*filepath) return -1;
        if (snprintf(data_dir, sizeof(data_dir), "%s.ivfdisk", filepath) >= (int)sizeof(data_dir)) {
            return -1;
        }
        cfg.data_dir = data_dir;
    }
    db->hnsw_index = ivfdisk_create(dimension, &cfg);
    return db->hnsw_index ? 0 : -1;
}

static void db_attach_soa_storage(GV_Database *db) {
    if (db != NULL && db->soa_storage != NULL) {
        soa_storage_bind_database(db->soa_storage, db);
    }
}

static void db_init_common_fields(GV_Database *db) {
    db->generation = 0;
    db->compaction_running = 0;
    pthread_mutex_init(&db->compaction_mutex, NULL);
    pthread_cond_init(&db->compaction_cond, NULL);
    db->compaction_interval_sec = 300;  /* Default: 5 minutes */
    db->wal_compaction_threshold = 10 * 1024 * 1024;  /* Default: 10MB */
    db->deleted_ratio_threshold = 0.1;  /* Default: 10% */
    db->resource_limits.max_memory_bytes = 0;  /* Unlimited by default */
    db->resource_limits.max_vectors = 0;  /* Unlimited by default */
    db->resource_limits.max_concurrent_operations = 0;  /* Unlimited by default */
    db->current_memory_bytes = 0;
    db->current_concurrent_ops = 0;
    db->commit_version = 0;
    pthread_mutex_init(&db->resource_mutex, NULL);
    pthread_mutex_init(&db->txn_mutex, NULL);
    memset(&db->insert_latency_hist, 0, sizeof(GV_LatencyHistogram));
    memset(&db->search_latency_hist, 0, sizeof(GV_LatencyHistogram));
    db->last_qps_update_time_us = 0;
    db->last_ips_update_time_us = 0;
    db->first_insert_time_us = 0;
    db->query_count_since_update = 0;
    db->insert_count_since_update = 0;
    db->current_qps = 0.0;
    db->current_ips = 0.0;
    memset(&db->recall_metrics, 0, sizeof(GV_RecallMetrics));
    pthread_mutex_init(&db->observability_mutex, NULL);
    gv_memory_init(&db->memory_pool);

    /* IVF incremental retraining defaults */
    db->retrain_enabled = 0;
    db->retrain_drift_threshold = 0.15f;
    db->retrain_min_new_vectors = 50000;
    db->inserts_since_retrain = 0;
    db->last_retrain_drift = 1.0f;
    db->initial_inertia = 0.0f;
    db->retrain_running = 0;
    db->retrain_thread_joinable = 0;
    pthread_mutex_init(&db->retrain_mutex, NULL);
    /* Tiered storage — disabled by default */
    db->tiering_enabled      = 0;
    db->hot_max_age_seconds  = 86400;    /* 1 day */
    db->warm_max_age_seconds = 604800;   /* 7 days */
    db->hot_max_vectors      = 100000;
    db->tiered_storage       = NULL;
    db->value_store          = NULL;
    db->ab_test = NULL;
    pthread_mutex_init(&db->ab_mutex, NULL);
    db->cdc_stream = NULL;
    db->webhook_mgr = NULL;
    db->id_map = point_id_create(1024);  /* NULL-safe: all id-map ops tolerate NULL */
}

/*
 * Emit an insert/update/delete change notification to the optionally-attached
 * CDC stream and/or webhook manager. No-op during WAL replay or when no sink is
 * attached. Must be called AFTER releasing db->rwlock: cdc_publish/webhook_fire
 * notify subscriber callbacks synchronously, which may re-enter the database.
 * vector_data is NULL for deletes; it is deep-copied by cdc_publish.
 */
void db_emit_change(GV_Database *db, GV_CDCEventType cdc_type,
                           GV_EventType wh_type, size_t vector_index,
                           const float *vector_data, size_t dimension) {
    if (db->wal_replaying) return;
    if (db->cdc_stream == NULL && db->webhook_mgr == NULL) return;

    uint64_t ts_ns = db_get_time_us() * 1000ULL;

    if (db->cdc_stream != NULL) {
        GV_CDCEvent ev;
        memset(&ev, 0, sizeof(ev));
        ev.type = cdc_type;
        ev.vector_index = vector_index;
        ev.timestamp = ts_ns;
        ev.vector_data = vector_data;
        ev.dimension = (vector_data != NULL) ? dimension : 0;
        ev.metadata_json = NULL;
        cdc_publish(db->cdc_stream, &ev);
    }
    if (db->webhook_mgr != NULL) {
        GV_Event ev;
        memset(&ev, 0, sizeof(ev));
        ev.event_type = wh_type;
        ev.vector_index = vector_index;
        ev.timestamp = ts_ns;
        ev.collection = NULL;
        webhook_fire(db->webhook_mgr, &ev);
    }
}

int db_set_ef_construction(GV_Database *db, size_t ef) {
    if (db == NULL || db->index_type != GV_INDEX_TYPE_HNSW || db->hnsw_index == NULL) return -1;
    gv_hnsw_set_ef_construction(db->hnsw_index, ef);
    return 0;
}

int db_hnsw_build_parallel(GV_Database *db, size_t num_threads) {
    if (db == NULL || db->index_type != GV_INDEX_TYPE_HNSW || db->hnsw_index == NULL) return -1;
    return gv_hnsw_build_parallel(db->hnsw_index, num_threads);
}

int db_add_vectors_parallel(GV_Database *db, const float *data, size_t count,
                            size_t dimension, size_t num_threads) {
    if (db == NULL || data == NULL || count == 0) return -1;
    if (db->index_type != GV_INDEX_TYPE_HNSW || db->hnsw_index == NULL) return -1;
    if (dimension != db->dimension || db->count != 0) return -1; /* fresh HNSW index only */

    pthread_rwlock_wrlock(&db->rwlock);
    /* Stage all vector data into SoA (data copy only — no graph links yet). */
    for (size_t i = 0; i < count; ++i) {
        if (soa_storage_add(db->soa_storage, data + i * dimension, NULL) == (size_t)-1) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
    }
    /* Durability: append WAL records under group-commit (fast) if a WAL exists. */
    if (db->wal != NULL) {
        pthread_mutex_lock(&db->wal_mutex);
        wal_set_sync_interval(db->wal, count + 1);
        for (size_t i = 0; i < count; ++i) {
            if (wal_append_insert(db->wal, data + i * dimension, dimension, NULL, NULL) == 0)
                db->total_wal_records += 1;
        }
        wal_set_sync_interval(db->wal, 1); /* forces a durable flush + restores per-record */
        pthread_mutex_unlock(&db->wal_mutex);
    }
    /* Build the HNSW graph in parallel over the staged vectors. */
    int rc = gv_hnsw_build_parallel(db->hnsw_index, num_threads);
    if (rc == 0) {
        db->count = count;
        db->total_inserts += count;
        db_update_memory_usage(db);
    } else {
        /* Build failed: discard the staged SoA rows so soa_storage->count stays
         * consistent with db->count (0). Otherwise the next db_add_vector would
         * use db->count==0 as its SoA slot and overwrite/misindex these rows —
         * persistent corruption. The WAL records remain durable and are recovered
         * by incremental replay on the next reopen (metadata was NULL, so no
         * per-row heap is orphaned by the reset). */
        if (db->soa_storage != NULL) db->soa_storage->count = 0;
    }
    pthread_rwlock_unlock(&db->rwlock);
    return rc;
}

void db_set_bulk_load(GV_Database *db, int on) {
    if (db == NULL || db->wal == NULL) return;
    /* Group-commit the WAL during bulk load: fsync every 8192 records instead of
     * every insert. Turning it off forces a durable flush and restores per-record
     * fsync. db_save/db_close also flush durably, so a clean shutdown loses nothing. */
    wal_set_sync_interval(db->wal, on ? 8192 : 1);
}

void db_set_cdc_stream(GV_Database *db, GV_CDCStream *stream) {
    if (db != NULL) db->cdc_stream = stream;
}

GV_CDCStream *db_get_cdc_stream(const GV_Database *db) {
    return db != NULL ? db->cdc_stream : NULL;
}

void db_set_webhook_manager(GV_Database *db, GV_WebhookManager *mgr) {
    if (db != NULL) db->webhook_mgr = mgr;
}

GV_WebhookManager *db_get_webhook_manager(const GV_Database *db) {
    return db != NULL ? db->webhook_mgr : NULL;
}

/* Identity seam (Phase 0): string primary key (chunk_id) -> internal index.
 * The chunk_id string is stored both in the vector's metadata (key "chunk_id")
 * and in db->id_map, which is persisted to a "{filepath}.ids" sidecar. */

int db_add_vector_with_id_meta(GV_Database *db, const char *string_id,
                               const float *data, size_t dimension,
                               const char *const *keys, const char *const *vals,
                               size_t n) {
    if (db == NULL || string_id == NULL || data == NULL) {
        return -1;
    }
    /* Prepend {chunk_id: string_id} to the caller's metadata pairs. */
    size_t total = n + 1;
    const char **k = (const char **)gv_alloc(total * sizeof(char *));
    const char **v = (const char **)gv_alloc(total * sizeof(char *));
    if (k == NULL || v == NULL) {
        gv_free((void *)k);
        gv_free((void *)v);
        return -1;
    }
    k[0] = "chunk_id";
    v[0] = string_id;
    for (size_t i = 0; i < n; ++i) {
        k[i + 1] = keys[i];
        v[i + 1] = vals[i];
    }
    int rc = db_add_vector_with_rich_metadata(db, data, dimension, k, v, total);
    gv_free((void *)k);
    gv_free((void *)v);
    if (rc != 0) {
        return rc;
    }
    /* Vector was appended at logical index count-1 (same convention as CDC
     * events). The document-ingest path is sequential per document, so this is
     * race-free there; concurrent multi-writer id mapping is a Phase-3 concern. */
    pthread_rwlock_rdlock(&db->rwlock);
    size_t idx = db->count - 1;
    pthread_rwlock_unlock(&db->rwlock);
    if (db->id_map != NULL) {
        point_id_set(db->id_map, string_id, idx);
    }
    return 0;
}

int db_add_vector_with_id(GV_Database *db, const char *string_id,
                          const float *data, size_t dimension) {
    return db_add_vector_with_id_meta(db, string_id, data, dimension, NULL, NULL, 0);
}

int db_get_index_by_id(const GV_Database *db, const char *string_id, size_t *out_index) {
    if (db == NULL || db->id_map == NULL || string_id == NULL || out_index == NULL) {
        return -1;
    }
    return point_id_get(db->id_map, string_id, out_index);
}

const char *db_get_id_by_index(const GV_Database *db, size_t index) {
    if (db == NULL || db->id_map == NULL) {
        return NULL;
    }
    return point_id_reverse_lookup(db->id_map, index);
}

const char *db_get_metadata_value(const GV_Database *db, size_t index,
                                  const char *key) {
    if (db == NULL || db->soa_storage == NULL || key == NULL) {
        return NULL;
    }
    GV_Metadata *m = soa_storage_get_metadata(db->soa_storage, index);
    for (; m != NULL; m = m->next) {
        if (m->key != NULL && strcmp(m->key, key) == 0) {
            return m->value;
        }
    }
    return NULL;
}

int db_delete_by_id(GV_Database *db, const char *string_id) {
    if (db == NULL || db->id_map == NULL || string_id == NULL) {
        return -1;
    }
    size_t idx = 0;
    if (point_id_get(db->id_map, string_id, &idx) != 0) {
        return -1;
    }
    int rc = db_delete_vector_by_index(db, idx);
    if (rc == 0) {
        point_id_remove(db->id_map, string_id);
    }
    return rc;
}

typedef struct {
    const char *prefix;
    size_t prefix_len;
    char **ids;
    size_t count;
    size_t cap;
} DbDocScanCtx;

static int db_doc_scan_cb(const char *id, size_t index, void *ctx) {
    (void)index;
    DbDocScanCtx *c = (DbDocScanCtx *)ctx;
    if (strncmp(id, c->prefix, c->prefix_len) != 0) {
        return 0;
    }
    if (c->count == c->cap) {
        size_t ncap = c->cap ? c->cap * 2 : 8;
        char **ni = (char **)gv_realloc(c->ids, ncap * sizeof(char *));
        if (ni == NULL) {
            return -1;
        }
        c->ids = ni;
        c->cap = ncap;
    }
    c->ids[c->count] = gv_dup_cstr(id);
    if (c->ids[c->count] == NULL) {
        return -1;
    }
    c->count++;
    return 0;
}

int db_delete_by_doc(GV_Database *db, const char *doc_id) {
    if (db == NULL || db->id_map == NULL || doc_id == NULL) {
        return -1;
    }
    char prefix[64];
    int pn = snprintf(prefix, sizeof(prefix), "%s:", doc_id);
    if (pn < 0 || (size_t)pn >= (int)sizeof(prefix)) {
        return -1;
    }
    /* Collect matching ids first (can't mutate the map mid-iteration). */
    DbDocScanCtx ctx = { prefix, (size_t)pn, NULL, 0, 0 };
    point_id_iterate(db->id_map, db_doc_scan_cb, &ctx);
    int deleted = 0;
    for (size_t i = 0; i < ctx.count; ++i) {
        if (db_delete_by_id(db, ctx.ids[i]) == 0) {
            deleted++;
        }
        gv_free(ctx.ids[i]);
    }
    gv_free(ctx.ids);
    return deleted;
}

int db_write_header(FILE *out, uint32_t dimension, uint64_t count, uint32_t version) {
    const uint32_t magic = 0x47564442; /* "GVDB" in hex */
    if (write_u32(out, magic) != 0) {
        GV_LOG_ERROR("db_write_header: failed to write magic (errno=%d)", errno);
        return -1;
    }
    if (write_u32(out, version) != 0) {
        return -1;
    }
    if (write_u32(out, dimension) != 0) {
        return -1;
    }
    if (write_u64(out, count) != 0) {
        return -1;
    }
    return 0;
}

static int db_read_header(FILE *in, uint32_t *dimension_out, uint64_t *count_out, uint32_t *version_out) {
    uint32_t magic = 0;
    uint32_t version = 0;
    if (read_u32(in, &magic) != 0) {
        return -1;
    }
    if (read_u32(in, &version) != 0) {
        return -1;
    }
    if (magic != 0x47564442 /* "GVDB" */) {
        GV_LOG_ERROR("db_read_header: bad magic 0x%08x (expected 0x47564442) - not a GigaVector snapshot or corrupt file",
                     magic);
        return -1;
    }
    if (read_u32(in, dimension_out) != 0) {
        return -1;
    }
    if (read_u64(in, count_out) != 0) {
        return -1;
    }

    /* Corrupt-snapshot allocation guard. The 64-bit count is attacker/
     * corruption-controlled and is later used to size allocations of the form
     * count * dimension * sizeof(float). Validate it here, at the single
     * header-read site, so every loader benefits. */
    {
        uint32_t dim = *dimension_out;
        uint64_t count = *count_out;

        /* (a) Reject if count * dimension * sizeof(float) would overflow a
         * size_t. Do the multiply in the widest available integer and check
         * each step with __builtin_mul_overflow. */
        size_t elems = 0;
        size_t bytes = 0;
        if (__builtin_mul_overflow((size_t)count, (size_t)dim, &elems)) {
            GV_LOG_ERROR("db_read_header: corrupt snapshot - count=%llu * dim=%u overflows size_t",
                         (unsigned long long)count, dim);
            return -1;
        }
        if (__builtin_mul_overflow(elems, sizeof(float), &bytes)) {
            GV_LOG_ERROR("db_read_header: corrupt snapshot - element count %zu * sizeof(float) overflows size_t",
                         elems);
            return -1;
        }

        /* (b) Sanity-check the count against the actual remaining file size
         * when the stream is seekable. A count implying far more data than the
         * file physically contains is corrupt. We use a conservative lower
         * bound of one byte per vector (quantized/sparse formats store fewer
         * than dimension*sizeof(float) bytes each, so a tighter bound would
         * risk false rejections). This still catches the pathological case of
         * a multi-billion count in a tiny file. */
        long cur = ftell(in);
        if (cur >= 0) {
            if (fseek(in, 0, SEEK_END) == 0) {
                long end = ftell(in);
                /* Restore the stream position for the caller regardless. */
                (void)fseek(in, cur, SEEK_SET);
                if (end >= 0 && (uint64_t)end >= (uint64_t)cur) {
                    uint64_t remaining = (uint64_t)end - (uint64_t)cur;
                    if (count > remaining) {
                        GV_LOG_ERROR("db_read_header: corrupt snapshot - header count=%llu exceeds %llu remaining file bytes",
                                     (unsigned long long)count, (unsigned long long)remaining);
                        return -1;
                    }
                }
            } else {
                /* Best-effort: try to restore position if the SEEK_END failed
                 * after moving; harmless if it did not move. */
                (void)fseek(in, cur, SEEK_SET);
            }
        }
    }

    if (version_out != NULL) {
        *version_out = version;
    }
    return 0;
}

static int read_uint32(FILE *in, uint32_t *value) {
    return (value != NULL && read_u32(in, value) == 0) ? 0 : -1;
}



static GV_IndexType index_suggest_heuristic(size_t dimension, size_t expected_count) {
    if (expected_count <= 500) {
        return GV_INDEX_TYPE_FLAT;
    }
    if (expected_count <= 20000 && dimension <= 64) {
        return GV_INDEX_TYPE_KDTREE;
    }
    if (expected_count >= 500000 && dimension >= 128) {
        return GV_INDEX_TYPE_IVFPQ;
    }
    return GV_INDEX_TYPE_HNSW;
}

GV_IndexType index_suggest_with_budget(size_t dimension, size_t expected_count,
                                       size_t max_memory_bytes, size_t bytes_per_vector) {
    if (max_memory_bytes > 0 && expected_count > 0 && dimension > 0) {
        size_t bpv = bytes_per_vector ? bytes_per_vector
                                      : index_suggest_bytes_per_vector(dimension, 0);
        size_t estimated = expected_count * bpv;
        size_t threshold = (size_t)((double)max_memory_bytes * GV_INDEX_SUGGEST_RAM_THRESHOLD_RATIO);
        if (estimated > threshold) {
            /* Both large-scale cases map to IVFDISK: it is fully wired through
             * db_open/save/load/add/search. DISKANN is a standalone API not
             * openable via db_open, so suggesting it here would hand callers an
             * index type db_open cannot build (silently-dead database). */
            return GV_INDEX_TYPE_IVFDISK;
        }
    }
    return index_suggest_heuristic(dimension, expected_count);
}

void db_normalize_vector(GV_Vector *vector) {
    if (vector == NULL || vector->data == NULL || vector->dimension == 0) {
        return;
    }
    float norm_sq = 0.0f;
    for (size_t i = 0; i < vector->dimension; ++i) {
        float v = vector->data[i];
        norm_sq += v * v;
    }
    if (norm_sq <= 0.0f) {
        return;
    }
    float inv = 1.0f / sqrtf(norm_sq);
    for (size_t i = 0; i < vector->dimension; ++i) {
        vector->data[i] *= inv;
    }
}

static void db_rebuild_metadata_index_from_soa(GV_Database *db) {
    if (db == NULL || db->soa_storage == NULL) {
        return;
    }

    GV_MetadataIndex *fresh = metadata_index_create();
    if (fresh == NULL) {
        return;
    }

    size_t total = db->soa_storage->count;
    for (size_t i = 0; i < total; ++i) {
        if (soa_storage_is_deleted(db->soa_storage, i) == 1) {
            continue;
        }
        GV_Metadata *meta = db->soa_storage->metadata[i];
        for (GV_Metadata *current = meta; current != NULL; current = current->next) {
            if (current->key != NULL && current->value != NULL) {
                metadata_index_add(fresh, current->key, current->value, i);
            }
        }
    }

    metadata_index_destroy(db->metadata_index);
    db->metadata_index = fresh;
}

void db_refresh_count(GV_Database *db) {
    if (db == NULL) {
        return;
    }
    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        db->count = db->soa_storage ? soa_storage_count(db->soa_storage) : 0;
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        db->count = db->hnsw_index ? gv_hnsw_count(db->hnsw_index) : 0;
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        db->count = db->hnsw_index ? gv_ivfpq_count(db->hnsw_index) : 0;
    } else if (db->index_type == GV_INDEX_TYPE_FLAT) {
        db->count = db->hnsw_index ? flat_count(db->hnsw_index) : 0;
    } else if (db->index_type == GV_INDEX_TYPE_IVFFLAT) {
        db->count = db->hnsw_index ? ivfflat_count(db->hnsw_index) : 0;
    } else if (db->index_type == GV_INDEX_TYPE_IVFSQ8) {
        db->count = db->hnsw_index ? ivfsq8_count(db->hnsw_index) : 0;
    } else if (db->index_type == GV_INDEX_TYPE_IVFTURBOQUANT) {
        db->count = db->hnsw_index ? ivfturboquant_count(db->hnsw_index) : 0;
    } else if (db->index_type == GV_INDEX_TYPE_PQ) {
        db->count = db->hnsw_index ? pq_count(db->hnsw_index) : 0;
    } else if (db->index_type == GV_INDEX_TYPE_LSH) {
        db->count = db->hnsw_index ? lsh_count(db->hnsw_index) : 0;
    } else if (db->index_type == GV_INDEX_TYPE_RABITQ) {
        db->count = db->hnsw_index ? rabitq_count(db->hnsw_index) : 0;
    } else if (db->index_type == GV_INDEX_TYPE_IVFDISK) {
        db->count = db->soa_storage ? soa_storage_count(db->soa_storage)
                                    : (db->hnsw_index ? ivfdisk_count((GV_IVFDiskIndex *)db->hnsw_index) : 0);
    }
}

static void db_destroy_indexes(GV_Database *db) {
    if (db == NULL) {
        return;
    }
    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        kdtree_destroy_recursive(db->root);
        db->root = NULL;
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        if (db->hnsw_index) {
            gv_hnsw_destroy(db->hnsw_index);
            db->hnsw_index = NULL;
        }
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        if (db->hnsw_index) {
            gv_ivfpq_destroy(db->hnsw_index);
            db->hnsw_index = NULL;
        }
    } else if (db->index_type == GV_INDEX_TYPE_SPARSE) {
        if (db->sparse_index) {
            sparse_index_destroy(db->sparse_index);
            db->sparse_index = NULL;
        }
    } else if (db->index_type == GV_INDEX_TYPE_FLAT) {
        if (db->hnsw_index) {
            flat_destroy(db->hnsw_index);
            db->hnsw_index = NULL;
        }
    } else if (db->index_type == GV_INDEX_TYPE_IVFFLAT) {
        if (db->hnsw_index) {
            ivfflat_destroy(db->hnsw_index);
            db->hnsw_index = NULL;
        }
    } else if (db->index_type == GV_INDEX_TYPE_IVFSQ8) {
        if (db->hnsw_index) {
            ivfsq8_destroy(db->hnsw_index);
            db->hnsw_index = NULL;
        }
    } else if (db->index_type == GV_INDEX_TYPE_IVFTURBOQUANT) {
        if (db->hnsw_index) {
            ivfturboquant_destroy(db->hnsw_index);
            db->hnsw_index = NULL;
        }
    } else if (db->index_type == GV_INDEX_TYPE_PQ) {
        if (db->hnsw_index) {
            pq_destroy(db->hnsw_index);
            db->hnsw_index = NULL;
        }
    } else if (db->index_type == GV_INDEX_TYPE_LSH) {
        if (db->hnsw_index) {
            lsh_destroy(db->hnsw_index);
            db->hnsw_index = NULL;
        }
    } else if (db->index_type == GV_INDEX_TYPE_RABITQ) {
        if (db->hnsw_index) {
            rabitq_destroy(db->hnsw_index);
            db->hnsw_index = NULL;
        }
    } else if (db->index_type == GV_INDEX_TYPE_IVFDISK) {
        if (db->hnsw_index) {
            ivfdisk_destroy((GV_IVFDiskIndex *)db->hnsw_index);
            db->hnsw_index = NULL;
        }
    }
}

static void db_free_open_failure(GV_Database *db) {
    if (db == NULL) {
        return;
    }
    db_destroy_indexes(db);
    if (db->metadata_index) {
        metadata_index_destroy(db->metadata_index);
        db->metadata_index = NULL;
    }
    if (db->soa_storage) {
        soa_storage_destroy(db->soa_storage);
        db->soa_storage = NULL;
    }
    if (db->id_map) {
        point_id_destroy(db->id_map);
        db->id_map = NULL;
    }
    pthread_rwlock_destroy(&db->rwlock);
    pthread_mutex_destroy(&db->wal_mutex);
    pthread_mutex_destroy(&db->compaction_mutex);
    pthread_cond_destroy(&db->compaction_cond);
    pthread_mutex_destroy(&db->resource_mutex);
    pthread_mutex_destroy(&db->txn_mutex);
    pthread_mutex_destroy(&db->observability_mutex);
    pthread_mutex_destroy(&db->retrain_mutex);
    pthread_mutex_destroy(&db->ab_mutex);
    gv_free(db->filepath);
    gv_free(db->wal_path);
    gv_free(db);
}

GV_Database *db_open(const char *filepath, size_t dimension, GV_IndexType index_type) {
    if (dimension == 0 && filepath == NULL) {
        return NULL;
    }

    /* Reject an out-of-range index type up front. An unrecognised value would
     * otherwise be stored verbatim and silently no-op through every dispatch
     * chain (create/save/load/search), leaving a hollow, unusable database.
     * GV_INDEX_TYPE_KDTREE (0) is the minimum and GV_INDEX_TYPE_RABITQ the
     * maximum valid enumerator. */
    if ((int)index_type < (int)GV_INDEX_TYPE_KDTREE ||
        (int)index_type > (int)GV_INDEX_TYPE_RABITQ) {
        GV_LOG_ERROR("db_open: index_type %d out of valid range [%d,%d]",
                     (int)index_type, (int)GV_INDEX_TYPE_KDTREE, (int)GV_INDEX_TYPE_RABITQ);
        return NULL;
    }

    /* Zero-initialize so every field (e.g. otlp_config) has a defined default;
     * db_init_common_fields sets the non-zero ones afterward. */
    GV_Database *db = (GV_Database *)gv_calloc(1, sizeof(GV_Database));
    if (db == NULL) {
        GV_LOG_ERROR("db_open: allocation of GV_Database (%zu bytes) failed for '%s'",
                     sizeof(GV_Database), filepath ? filepath : "(in-memory)");
        return NULL;
    }

    db->dimension = dimension;
    db->index_type = index_type;
    db->root = NULL;
    db->hnsw_index = NULL;
    db->sparse_index = NULL;
    db->soa_storage = NULL;
    db->filepath = NULL;
    db->wal_path = NULL;
    db->wal = NULL;
    db->wal_replaying = 0;
    pthread_rwlock_init(&db->rwlock, NULL);
    pthread_mutex_init(&db->wal_mutex, NULL);
    db->count = 0;
    db->exact_search_threshold = 1000;
    db->force_exact_search = 0;
    db->total_inserts = 0;
    db->total_queries = 0;
    db->total_range_queries = 0;
    db->total_wal_records = 0;
    db->cosine_normalized = 0;
    db->metadata_index = metadata_index_create();
    if (db->metadata_index == NULL) {
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        gv_free(db);
        return NULL;
    }
    db_init_common_fields(db);

    if (index_type == GV_INDEX_TYPE_KDTREE || index_type == GV_INDEX_TYPE_HNSW ||
        index_type == GV_INDEX_TYPE_FLAT || index_type == GV_INDEX_TYPE_LSH ||
        index_type == GV_INDEX_TYPE_RABITQ || index_type == GV_INDEX_TYPE_IVFDISK) {
        db->soa_storage = soa_storage_create(dimension, 0);
        if (db->soa_storage == NULL) {
            db_free_open_failure(db);
            return NULL;
        }
        db_attach_soa_storage(db);
    }

    if (index_type == GV_INDEX_TYPE_HNSW && filepath == NULL) {
        db->hnsw_index = gv_hnsw_create(dimension, NULL, db->soa_storage);
        if (db->hnsw_index == NULL) {
            /* assign-then-check: db->hnsw_index is NULL here, so
             * db_free_open_failure's db_destroy_indexes is a no-op (no
             * double-free). The helper frees soa_storage. */
            db_free_open_failure(db);
            return NULL;
        }
    } else if (index_type == GV_INDEX_TYPE_IVFPQ && filepath == NULL) {
        db->hnsw_index = NULL;
        db->root = NULL;
        GV_IVFPQConfig cfg = {.nlist = 64, .m = 8, .nbits = 8, .nprobe = 4, .train_iters = 15};
        db->hnsw_index = gv_ivfpq_create(dimension, &cfg);
        if (db->hnsw_index == NULL) {
            db_free_open_failure(db);
            return NULL;
        }
    } else if (index_type == GV_INDEX_TYPE_SPARSE && filepath == NULL) {
        db->sparse_index = sparse_index_create(dimension);
        if (db->sparse_index == NULL) {
            db_free_open_failure(db);
            return NULL;
        }
    } else if (index_type == GV_INDEX_TYPE_FLAT && filepath == NULL) {
        db->hnsw_index = flat_create(dimension, NULL, db->soa_storage);
        if (db->hnsw_index == NULL) {
            db_free_open_failure(db);
            return NULL;
        }
    } else if (index_type == GV_INDEX_TYPE_IVFFLAT && filepath == NULL) {
        GV_IVFFlatConfig cfg = {.nlist = 64, .nprobe = 4, .train_iters = 15, .use_cosine = 0};
        db->hnsw_index = ivfflat_create(dimension, &cfg);
        if (db->hnsw_index == NULL) {
            db_free_open_failure(db);
            return NULL;
        }
    } else if (index_type == GV_INDEX_TYPE_IVFSQ8 && filepath == NULL) {
        GV_IVFSQ8Config cfg = {
            .nlist = 64, .nprobe = 4, .train_iters = 15, .use_cosine = 0,
            .per_dimension = 0, .default_rerank = 200
        };
        db->hnsw_index = ivfsq8_create(dimension, &cfg);
        if (db->hnsw_index == NULL) {
            db_free_open_failure(db);
            return NULL;
        }
    } else if (index_type == GV_INDEX_TYPE_IVFTURBOQUANT && filepath == NULL) {
        GV_IVFTurboQuantConfig cfg = {
            .nlist = 64, .nprobe = 4, .train_iters = 15, .use_cosine = 0,
            .default_rerank = 200,
            .turbo = {.bits = 8, .projections = dimension / 4, .seed = 42,
                      .use_qjl = 1, .rotation = GV_TURBOQUANT_ROTATION_AUTO}
        };
        if (cfg.turbo.projections == 0) cfg.turbo.projections = 2;
        db->hnsw_index = ivfturboquant_create(dimension, &cfg);
        if (db->hnsw_index == NULL) {
            db_free_open_failure(db);
            return NULL;
        }
    } else if (index_type == GV_INDEX_TYPE_PQ && filepath == NULL) {
        GV_PQConfig cfg = {.m = 8, .nbits = 8, .train_iters = 15};
        db->hnsw_index = pq_create(dimension, &cfg);
        if (db->hnsw_index == NULL) {
            db_free_open_failure(db);
            return NULL;
        }
    } else if (index_type == GV_INDEX_TYPE_LSH && filepath == NULL) {
        GV_LSHConfig cfg = {.num_tables = 8, .num_hash_bits = 16, .seed = 42};
        db->hnsw_index = lsh_create(dimension, &cfg, db->soa_storage);
        if (db->hnsw_index == NULL) {
            db_free_open_failure(db);
            return NULL;
        }
    } else if (index_type == GV_INDEX_TYPE_RABITQ && filepath == NULL) {
        GV_RaBitQConfig cfg = {.seed = 42, .rerank_factor = 4};
        db->hnsw_index = rabitq_create(dimension, &cfg, db->soa_storage);
        if (db->hnsw_index == NULL) {
            db_free_open_failure(db);
            return NULL;
        }
    }

    if (filepath != NULL) {
        db->filepath = gv_dup_cstr(filepath);
        if (db->filepath == NULL) {
            db_free_open_failure(db);
            return NULL;
        }

        db->wal_path = db_build_wal_path(filepath);
        if (db->wal_path == NULL) {
            db_free_open_failure(db);
            return NULL;
        }
    }

    if (filepath == NULL) {
        if (index_type == GV_INDEX_TYPE_IVFDISK) {
            /* IVFDISK has no in-memory (filepath==NULL) form. db->hnsw_index is
             * still NULL here (never created for IVFDISK in the chain above), so
             * db_free_open_failure's db_destroy_indexes is a no-op. */
            db_free_open_failure(db);
            return NULL;
        }
        if (db->wal_path != NULL) {
            db->wal = wal_open(db->wal_path, db->dimension, (uint32_t)db->index_type);
        }
        return db;
    }

    FILE *in = fopen(filepath, "rb");
    if (in == NULL) {
        if (errno == ENOENT) {
            if (index_type == GV_INDEX_TYPE_HNSW) {
                db->hnsw_index = gv_hnsw_create(dimension, NULL, db->soa_storage);
                if (db->hnsw_index == NULL) {
                    db_free_open_failure(db);
                    return NULL;
                }
            } else if (index_type == GV_INDEX_TYPE_FLAT) {
                db->hnsw_index = flat_create(dimension, NULL, db->soa_storage);
                if (db->hnsw_index == NULL) {
                    db_free_open_failure(db);
                    return NULL;
                }
            } else if (index_type == GV_INDEX_TYPE_IVFFLAT) {
                GV_IVFFlatConfig cfg = {.nlist = 64, .nprobe = 4, .train_iters = 15, .use_cosine = 0};
                db->hnsw_index = ivfflat_create(dimension, &cfg);
                if (db->hnsw_index == NULL) {
                    db_free_open_failure(db);
                    return NULL;
                }
            } else if (index_type == GV_INDEX_TYPE_IVFSQ8) {
                GV_IVFSQ8Config cfg = {
                    .nlist = 64, .nprobe = 4, .train_iters = 15, .use_cosine = 0,
                    .per_dimension = 0, .default_rerank = 200
                };
                db->hnsw_index = ivfsq8_create(dimension, &cfg);
                if (db->hnsw_index == NULL) {
                    db_free_open_failure(db);
                    return NULL;
                }
            } else if (index_type == GV_INDEX_TYPE_IVFTURBOQUANT) {
                GV_IVFTurboQuantConfig cfg = {
                    .nlist = 64, .nprobe = 4, .train_iters = 15, .use_cosine = 0,
                    .default_rerank = 200,
                    .turbo = {.bits = 8, .projections = dimension / 4, .seed = 42,
                              .use_qjl = 1, .rotation = GV_TURBOQUANT_ROTATION_AUTO}
                };
                if (cfg.turbo.projections == 0) cfg.turbo.projections = 2;
                db->hnsw_index = ivfturboquant_create(dimension, &cfg);
                if (db->hnsw_index == NULL) {
                    db_free_open_failure(db);
                    return NULL;
                }
            } else if (index_type == GV_INDEX_TYPE_PQ) {
                GV_PQConfig cfg = {.m = 8, .nbits = 8, .train_iters = 15};
                db->hnsw_index = pq_create(dimension, &cfg);
                if (db->hnsw_index == NULL) {
                    db_free_open_failure(db);
                    return NULL;
                }
            } else if (index_type == GV_INDEX_TYPE_LSH) {
                GV_LSHConfig cfg = {.num_tables = 8, .num_hash_bits = 16, .seed = 42};
                db->hnsw_index = lsh_create(dimension, &cfg, db->soa_storage);
                if (db->hnsw_index == NULL) {
                    db_free_open_failure(db);
                    return NULL;
                }
            } else if (index_type == GV_INDEX_TYPE_RABITQ) {
                GV_RaBitQConfig cfg = {.seed = 42, .rerank_factor = 4};
                db->hnsw_index = rabitq_create(dimension, &cfg, db->soa_storage);
                if (db->hnsw_index == NULL) {
                    db_free_open_failure(db);
                    return NULL;
                }
            } else if (index_type == GV_INDEX_TYPE_IVFPQ) {
                GV_IVFPQConfig cfg = {.nlist = 64, .m = 8, .nbits = 8, .nprobe = 4, .train_iters = 15};
                db->hnsw_index = gv_ivfpq_create(dimension, &cfg);
                if (db->hnsw_index == NULL) {
                    db_free_open_failure(db);
                    return NULL;
                }
            } else if (index_type == GV_INDEX_TYPE_IVFDISK) {
                if (db_ivfdisk_create_index(db, dimension, filepath, NULL) != 0) {
                    /* On failure db->hnsw_index is NULL (ivfdisk_create returned
                     * NULL or an early return before assignment), so the
                     * helper's index destroy is a safe no-op. */
                    db_free_open_failure(db);
                    return NULL;
                }
            }
            if (db->wal_path != NULL) {
                db->wal = wal_open(db->wal_path, db->dimension, (uint32_t)db->index_type);
                if (db->wal == NULL) {
                    GV_LOG_ERROR("db_open: wal_open failed for '%s' (errno=%d)",
                                 db->wal_path, errno);
                    db_free_open_failure(db);
                    return NULL;
                }
                if (db_replay_wal(db) != 0) {
                    GV_LOG_ERROR("db_open: WAL replay failed for new db from '%s'", filepath);
                    wal_close(db->wal);
                    db->wal = NULL;
                    db_free_open_failure(db);
                    return NULL;
                }
            }
            return db;
        }
        /* fopen failed with errno != ENOENT. db->hnsw_index is NULL here; route
         * through the helper so soa_storage (created above for some index
         * types) is not leaked. */
        GV_LOG_ERROR("db_open: fopen('%s', \"rb\") failed (errno=%d)", filepath, errno);
        db_free_open_failure(db);
        return NULL;
    }

    uint32_t file_dim = 0;
    uint64_t file_count = 0;
    uint32_t file_version = 0;
    if (db_read_header(in, &file_dim, &file_count, &file_version) != 0) {
        /* db->hnsw_index is NULL here (all in-memory create branches are
         * guarded by filepath==NULL and did not run on this load path), so
         * db_free_open_failure's db_destroy_indexes is a no-op; the helper also
         * frees soa_storage which the old manual cleanup leaked. */
        GV_LOG_ERROR("db_open: failed to read/validate header of '%s' (truncated or corrupt)",
                     filepath);
        fclose(in);
        db_free_open_failure(db);
        return NULL;
    }

    if (dimension != 0 && dimension != (size_t)file_dim) {
        GV_LOG_ERROR("db_open: dimension mismatch for '%s' - requested %zu but file has %u",
                     filepath, dimension, file_dim);
        fclose(in);
        db_free_open_failure(db);
        return NULL;
    }

    db->dimension = (size_t)file_dim;

    if (file_version < 1 || file_version > 6) {
        /* db->hnsw_index is NULL on the load path (create branches guarded by
         * filepath==NULL); db_free_open_failure is a safe no-op for indexes and
         * frees soa_storage that the old manual cleanup leaked. */
        GV_LOG_ERROR("db_open: unsupported snapshot version %u in '%s' (supported: 1-6)",
                     file_version, filepath);
        fclose(in);
        db_free_open_failure(db);
        return NULL;
    }

    uint32_t file_index_type = GV_INDEX_TYPE_KDTREE;
    if (file_version >= 2) {
        if (read_uint32(in, &file_index_type) != 0) {
            fclose(in);
            db_free_open_failure(db);
            return NULL;
        }
    }

    /* Verify the trailing whole-file CRC BEFORE parsing the (untrusted) index
     * payload, so a corrupt file is rejected without allocating/parsing based
     * on attacker-controlled bytes. Versions < 3 predate the CRC and are
     * skipped for backward compatibility. */
    if (file_version >= 3) {
        long resume_pos = ftell(in);
        if (resume_pos < 0) {
            fclose(in);
            db_free_open_failure(db);
            return NULL;
        }
        if (fseek(in, 0, SEEK_END) != 0) {
            fclose(in);
            db_free_open_failure(db);
            return NULL;
        }
        long end_pos = ftell(in);
        if (end_pos < 4 || fseek(in, end_pos - (long)sizeof(uint32_t), SEEK_SET) != 0) {
            fclose(in);
            db_free_open_failure(db);
            return NULL;
        }
        uint32_t stored_crc = 0;
        if (read_uint32(in, &stored_crc) != 0 || fseek(in, 0, SEEK_SET) != 0) {
            fclose(in);
            db_free_open_failure(db);
            return NULL;
        }
        uint32_t crc = gv_crc32_init();
        char crcbuf[65536];
        long remaining = end_pos - (long)sizeof(uint32_t);
        while (remaining > 0) {
            size_t chunk = (remaining > (long)sizeof(crcbuf)) ? sizeof(crcbuf) : (size_t)remaining;
            if (fread(crcbuf, 1, chunk, in) != chunk) {
                fclose(in);
                db_free_open_failure(db);
                return NULL;
            }
            crc = gv_crc32_update(crc, crcbuf, chunk);
            remaining -= (long)chunk;
        }
        crc = gv_crc32_finish(crc);
        if (crc != stored_crc) {
            GV_LOG_ERROR("db_open: CRC mismatch in '%s' - computed 0x%08x, stored 0x%08x (corrupt snapshot)",
                         filepath, crc, stored_crc);
            fclose(in);
            db_free_open_failure(db);
            return NULL;
        }
        /* CRC valid: rewind to where index parsing should resume. */
        if (fseek(in, resume_pos, SEEK_SET) != 0) {
            fclose(in);
            db_free_open_failure(db);
            return NULL;
        }
    }

    if (file_index_type != db->index_type) {
        /* db->hnsw_index is NULL on the load path; helper is a safe no-op for
         * indexes and frees soa_storage. */
        GV_LOG_ERROR("db_open: index-type mismatch for '%s' - file has %u, requested %d",
                     filepath, file_index_type, (int)db->index_type);
        fclose(in);
        db_free_open_failure(db);
        return NULL;
    }

    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        if (db->soa_storage == NULL) {
            fclose(in);
            db_free_open_failure(db);
            return NULL;
        }
        if (kdtree_load_recursive(&(db->root), db->soa_storage, in, db->dimension, file_version) != 0) {
            /* Partially-built tree lives in db->root; db_destroy_indexes frees
             * it (KDTREE branch) so drop the manual kdtree_destroy_recursive. */
            fclose(in);
            db_free_open_failure(db);
            return NULL;
        }
        db_rebuild_metadata_index_from_soa(db);
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        void *loaded_index = NULL;
        if (gv_hnsw_load(&loaded_index, in, db->dimension, file_version,
                         db->soa_storage) != 0) {
            /* Keep the LOCAL loaded_index destroy (the helper doesn't know about
             * it). db->hnsw_index is NULL here, so the helper's index destroy is
             * a no-op. */
            fclose(in);
            if (loaded_index) gv_hnsw_destroy(loaded_index);
            db_free_open_failure(db);
            return NULL;
        }
        if (db->hnsw_index != NULL) {
            gv_hnsw_destroy(db->hnsw_index);
        }
        db->hnsw_index = loaded_index;
        db->count = gv_hnsw_count(db->hnsw_index);
        db_rebuild_metadata_index_from_soa(db);
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        void *loaded_index = NULL;
        if (gv_ivfpq_load(&loaded_index, in, db->dimension, file_version) != 0) {
            fclose(in);
            if (loaded_index) gv_ivfpq_destroy(loaded_index);
            db_free_open_failure(db);
            return NULL;
        }
        db->hnsw_index = loaded_index;
        db->count = gv_ivfpq_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_SPARSE) {
        GV_SparseIndex *loaded_index = NULL;
        if (sparse_index_load(&loaded_index, in, db->dimension, (size_t)file_count, file_version) != 0) {
            fclose(in);
            db_free_open_failure(db);
            return NULL;
        }
        db->sparse_index = loaded_index;
        db->count = (size_t)file_count;
    } else if (db->index_type == GV_INDEX_TYPE_FLAT) {
        void *loaded_index = NULL;
        if (flat_load(&loaded_index, in, db->dimension, file_version) != 0) {
            fclose(in);
            if (loaded_index) flat_destroy(loaded_index);
            db_free_open_failure(db);
            return NULL;
        }
        db->hnsw_index = loaded_index;
        db->count = flat_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFFLAT) {
        void *loaded_index = NULL;
        if (ivfflat_load(&loaded_index, in, db->dimension, file_version) != 0) {
            fclose(in);
            if (loaded_index) ivfflat_destroy(loaded_index);
            db_free_open_failure(db);
            return NULL;
        }
        db->hnsw_index = loaded_index;
        db->count = ivfflat_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFSQ8) {
        void *loaded_index = NULL;
        if (ivfsq8_load(&loaded_index, in, db->dimension, file_version) != 0) {
            fclose(in);
            if (loaded_index) ivfsq8_destroy(loaded_index);
            db_free_open_failure(db);
            return NULL;
        }
        db->hnsw_index = loaded_index;
        db->count = ivfsq8_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFTURBOQUANT) {
        void *loaded_index = NULL;
        if (ivfturboquant_load(&loaded_index, in, db->dimension, file_version) != 0) {
            fclose(in);
            if (loaded_index) ivfturboquant_destroy(loaded_index);
            db_free_open_failure(db);
            return NULL;
        }
        db->hnsw_index = loaded_index;
        db->count = ivfturboquant_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_PQ) {
        void *loaded_index = NULL;
        if (pq_load(&loaded_index, in, db->dimension, file_version) != 0) {
            fclose(in);
            if (loaded_index) pq_destroy(loaded_index);
            db_free_open_failure(db);
            return NULL;
        }
        db->hnsw_index = loaded_index;
        db->count = pq_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_LSH) {
        void *loaded_index = NULL;
        if (lsh_load(&loaded_index, in, db->dimension, file_version) != 0) {
            fclose(in);
            if (loaded_index) lsh_destroy(loaded_index);
            db_free_open_failure(db);
            return NULL;
        }
        db->hnsw_index = loaded_index;
        db->count = lsh_count(db->hnsw_index);

    } else if (db->index_type == GV_INDEX_TYPE_RABITQ) {
        void *loaded_index = NULL;
        if (rabitq_load(&loaded_index, in, db->dimension, file_version) != 0) {
            fclose(in);
            if (loaded_index) rabitq_destroy(loaded_index);
            db_free_open_failure(db);
            return NULL;
        }
        db->hnsw_index = loaded_index;
        db->count = rabitq_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFDISK) {
        if (db->soa_storage == NULL) {
            fclose(in);
            db_free_open_failure(db);
            return NULL;
        }
        char data_dir[1024];
        if (db->filepath == NULL ||
            snprintf(data_dir, sizeof(data_dir), "%s.ivfdisk", db->filepath) >= (int)sizeof(data_dir)) {
            fclose(in);
            db_free_open_failure(db);
            return NULL;
        }
        GV_IVFDiskIndex *loaded_index = NULL;
        if (ivfdisk_load(&loaded_index, in, db->dimension, data_dir, file_version) != 0 ||
            soa_storage_load(db->soa_storage, in, file_version) != 0) {
            fclose(in);
            if (loaded_index) ivfdisk_destroy(loaded_index);
            db_free_open_failure(db);
            return NULL;
        }
        db->hnsw_index = loaded_index;
        db->count = soa_storage_count(db->soa_storage);
    } else {
        fclose(in);
        db_free_open_failure(db);
        return NULL;
    }

    /* Whole-file CRC was already verified before index parsing (see above). */

    if (fclose(in) != 0) {
        in = NULL;  /* stream already closed; avoid double fclose in load_fail */
        goto load_fail;
    }
    in = NULL;

    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        db->count = file_count;
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        db->count = gv_hnsw_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        db->count = gv_ivfpq_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_FLAT) {
        db->count = flat_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFFLAT) {
        db->count = ivfflat_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFSQ8) {
        db->count = ivfsq8_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFTURBOQUANT) {
        db->count = ivfturboquant_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_PQ) {
        db->count = pq_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_LSH) {
        db->count = lsh_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_RABITQ) {
        db->count = rabitq_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFDISK) {
        db->count = db->soa_storage ? soa_storage_count(db->soa_storage) : ivfdisk_count((GV_IVFDiskIndex *)db->hnsw_index);
    }

    if (db->wal_path != NULL) {
        db->wal = wal_open(db->wal_path, db->dimension, (uint32_t)db->index_type);
        if (db->wal == NULL) {
            db_free_open_failure(db);
            return NULL;
        }

        if (db_replay_wal(db) != 0) {
            wal_close(db->wal);
            db->wal = NULL;
            db_free_open_failure(db);
            return NULL;
        }
    }

    /* Load the chunk_id -> index map sidecar if present ("{filepath}.ids"). */
    if (filepath != NULL) {
        char ids_path[1024];
        int w = snprintf(ids_path, sizeof(ids_path), "%s.ids", filepath);
        if (w > 0 && (size_t)w < (int)sizeof(ids_path)) {
            GV_PointIDMap *loaded = point_id_load(ids_path);
            if (loaded != NULL) {
                if (db->id_map != NULL) {
                    point_id_destroy(db->id_map);
                }
                db->id_map = loaded;
            }
        }
    }
    return db;

load_fail:
    if (in != NULL) {
        fclose(in);
    }
    db_free_open_failure(db);
    return NULL;
}

int db_warmup(GV_Database *db) {
    if (db == NULL) return -1;
    /* Prefetch the backing data/snapshot file (and WAL) into the page cache so the
     * first queries against a freshly-opened or mmap'd/disk index don't pay cold
     * page-fault latency. In-memory databases (no filepath) are a no-op. */
    if (db->filepath) gv_file_warmup(db->filepath);
    if (db->wal_path)  gv_file_warmup(db->wal_path);
    return 0;
}

void db_close(GV_Database *db) {
    if (db == NULL) {
        return;
    }

    /* Stop background threads FIRST: both the compaction and IVF-retrain workers
     * take db->rwlock and dereference the index/SoA storage, so they must be
     * joined before any of that state (or the locks) is torn down below. */
    if (db->compaction_running) {
        db_stop_background_compaction(db);
    }
    ivf_retrain_stop(db);
    /* Tear down any running A/B test BEFORE the rwlock/index are destroyed below:
     * the A/B worker mirrors/compares against the PRIMARY db and takes db->rwlock,
     * so an in-flight A/B search would otherwise touch a destroyed lock/index.
     * gv_db_ab_test_stop detaches, drains in-flight searches, and destroys the
     * shadow db; it locks ab_mutex itself and is a no-op when no test is active. */
    gv_db_ab_test_stop(db);

    if (db->wal) {
        wal_close(db->wal);
    }

    /* If opened via db_open_mmap, wal_path holds an opaque GV_MMap* handle. */
    if (db->filepath == NULL && db->wal_path != NULL && db->wal == NULL) {
        GV_MMap *mm = (GV_MMap *)db->wal_path;
        mmap_close(mm);
        db->wal_path = NULL;
    }

    pthread_rwlock_destroy(&db->rwlock);
    pthread_mutex_destroy(&db->wal_mutex);
    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        kdtree_destroy_recursive(db->root);
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        gv_hnsw_destroy(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        gv_ivfpq_destroy(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_SPARSE) {
        sparse_index_destroy(db->sparse_index);
    } else if (db->index_type == GV_INDEX_TYPE_FLAT) {
        flat_destroy(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFFLAT) {
        ivfflat_destroy(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFSQ8) {
        ivfsq8_destroy(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFTURBOQUANT) {
        ivfturboquant_destroy(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_PQ) {
        pq_destroy(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_LSH) {
        lsh_destroy(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_RABITQ) {
        rabitq_destroy(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFDISK) {
        ivfdisk_destroy((GV_IVFDiskIndex *)db->hnsw_index);
    }
    if (db->soa_storage != NULL) {
        soa_storage_destroy(db->soa_storage);
    }
    if (db->metadata_index != NULL) {
        metadata_index_destroy(db->metadata_index);
    }
    if (db->id_map != NULL) {
        point_id_destroy(db->id_map);
    }
    /* Background threads were already joined at the top of db_close. */
    pthread_mutex_destroy(&db->compaction_mutex);
    pthread_cond_destroy(&db->compaction_cond);
    pthread_mutex_destroy(&db->resource_mutex);
    pthread_mutex_destroy(&db->txn_mutex);
    pthread_mutex_destroy(&db->retrain_mutex);
    if (db->insert_latency_hist.buckets != NULL) {
        gv_db_free(db, db->insert_latency_hist.buckets);
        gv_db_free(db, db->insert_latency_hist.bucket_boundaries);
        db->insert_latency_hist.buckets = NULL;
        db->insert_latency_hist.bucket_boundaries = NULL;
    }
    if (db->search_latency_hist.buckets != NULL) {
        gv_db_free(db, db->search_latency_hist.buckets);
        gv_db_free(db, db->search_latency_hist.bucket_boundaries);
        db->search_latency_hist.buckets = NULL;
        db->search_latency_hist.bucket_boundaries = NULL;
    }
    pthread_mutex_destroy(&db->observability_mutex);
    if (db->tiered_storage != NULL) {
        tiered_storage_destroy(db->tiered_storage);
        db->tiered_storage = NULL;
    }
    if (db->value_store != NULL) {
        value_store_close(db->value_store);
        db->value_store = NULL;
    }
    /* The A/B test was already stopped/drained at the top of db_close (before the
     * rwlock/index teardown it depends on); nothing left to detach here. */
    pthread_mutex_destroy(&db->ab_mutex);
    gv_memory_fini(&db->memory_pool);
    gv_free(db->filepath);
    gv_free(db->wal_path);
    gv_free(db);
}

/* Fully tear down a partially-constructed db from db_open_from_memory_impl's
 * error paths. Must be called only AFTER db_init_common_fields(db) has run (so
 * the compaction/resource/observability mutexes and memory pool exist). Frees
 * everything that may have been allocated so far — soa_storage (bound via
 * gv_db_alloc), metadata_index, id_map, any loaded index — without touching the
 * success path. */
static void db_open_from_memory_cleanup(GV_Database *db) {
    if (db == NULL) {
        return;
    }
    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        kdtree_destroy_recursive(db->root);
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        gv_hnsw_destroy(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        gv_ivfpq_destroy(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_SPARSE) {
        sparse_index_destroy(db->sparse_index);
    } else if (db->index_type == GV_INDEX_TYPE_FLAT) {
        flat_destroy(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFFLAT) {
        ivfflat_destroy(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFSQ8) {
        ivfsq8_destroy(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFTURBOQUANT) {
        ivfturboquant_destroy(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_PQ) {
        pq_destroy(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_LSH) {
        lsh_destroy(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_RABITQ) {
        rabitq_destroy(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFDISK) {
        ivfdisk_destroy((GV_IVFDiskIndex *)db->hnsw_index);
    }
    if (db->soa_storage != NULL) {
        soa_storage_destroy(db->soa_storage);
    }
    if (db->metadata_index != NULL) {
        metadata_index_destroy(db->metadata_index);
    }
    if (db->id_map != NULL) {
        point_id_destroy(db->id_map);
    }
    pthread_mutex_destroy(&db->compaction_mutex);
    pthread_cond_destroy(&db->compaction_cond);
    pthread_mutex_destroy(&db->resource_mutex);
    pthread_mutex_destroy(&db->txn_mutex);
    pthread_mutex_destroy(&db->retrain_mutex);
    pthread_mutex_destroy(&db->observability_mutex);
    pthread_mutex_destroy(&db->ab_mutex);
    gv_memory_fini(&db->memory_pool);
    pthread_rwlock_destroy(&db->rwlock);
    pthread_mutex_destroy(&db->wal_mutex);
    gv_free(db);
}

static GV_Database *db_open_from_memory_impl(const void *data, size_t size,
                                                size_t dimension, GV_IndexType index_type,
                                                const char *ivfdisk_data_dir) {
    if (data == NULL || size == 0) {
        return NULL;
    }
    if (dimension == 0) {
        return NULL;
    }
    /* Reject an out-of-range index type (see db_open for rationale). */
    if ((int)index_type < (int)GV_INDEX_TYPE_KDTREE ||
        (int)index_type > (int)GV_INDEX_TYPE_RABITQ) {
        return NULL;
    }
    if (index_type == GV_INDEX_TYPE_IVFDISK &&
        (ivfdisk_data_dir == NULL || ivfdisk_data_dir[0] == '\0')) {
        return NULL;
    }

    /* Zero-initialize so every field (e.g. otlp_config) has a defined default;
     * db_init_common_fields sets the non-zero ones afterward. */
    GV_Database *db = (GV_Database *)gv_calloc(1, sizeof(GV_Database));
    if (db == NULL) {
        return NULL;
    }

    db->dimension = dimension;
    db->index_type = index_type;
    db->root = NULL;
    db->hnsw_index = NULL;
    db->sparse_index = NULL;
    db->soa_storage = NULL;
    db->filepath = NULL;
    db->wal_path = NULL;
    db->wal = NULL;
    db->wal_replaying = 0;
    pthread_rwlock_init(&db->rwlock, NULL);
    pthread_mutex_init(&db->wal_mutex, NULL);
    db->count = 0;
    db->exact_search_threshold = 1000;
    db->force_exact_search = 0;
    db->total_inserts = 0;
    db->total_queries = 0;
    db->total_range_queries = 0;
    db->total_wal_records = 0;
    db->cosine_normalized = 0;
    db->metadata_index = metadata_index_create();
    if (db->metadata_index == NULL) {
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        gv_free(db);
        return NULL;
    }
    db_init_common_fields(db);

    if (index_type == GV_INDEX_TYPE_KDTREE || index_type == GV_INDEX_TYPE_HNSW ||
        index_type == GV_INDEX_TYPE_FLAT || index_type == GV_INDEX_TYPE_LSH ||
        index_type == GV_INDEX_TYPE_RABITQ || index_type == GV_INDEX_TYPE_IVFDISK) {
        db->soa_storage = soa_storage_create(dimension, 0);
        if (db->soa_storage == NULL) {
            db_open_from_memory_cleanup(db);
            return NULL;
        }
        db_attach_soa_storage(db);
    }

    FILE *in = fmemopen((void *)data, size, "rb");
    if (in == NULL) {
        db_open_from_memory_cleanup(db);
        return NULL;
    }

    uint32_t file_dim = 0;
    uint64_t file_count = 0;
    uint32_t file_version = 0;
    if (db_read_header(in, &file_dim, &file_count, &file_version) != 0) {
        fclose(in);
        db_open_from_memory_cleanup(db);
        return NULL;
    }

    if (dimension != 0 && dimension != (size_t)file_dim) {
        fclose(in);
        db_open_from_memory_cleanup(db);
        return NULL;
    }
    db->dimension = (size_t)file_dim;

    if (file_version < 1 || file_version > 6) {
        fclose(in);
        db_open_from_memory_cleanup(db);
        return NULL;
    }

    uint32_t file_index_type = GV_INDEX_TYPE_KDTREE;
    if (file_version >= 2) {
        if (read_uint32(in, &file_index_type) != 0) {
            fclose(in);
            db_open_from_memory_cleanup(db);
            return NULL;
        }
    }

    /* Verify the trailing whole-file CRC BEFORE parsing the (untrusted) index
     * payload. Versions < 3 predate the CRC and are skipped. */
    if (file_version >= 3) {
        long resume_pos = ftell(in);
        long end_pos = -1;
        uint32_t stored_crc = 0;
        int crc_ok = (resume_pos >= 0) &&
                     (fseek(in, 0, SEEK_END) == 0) &&
                     ((end_pos = ftell(in)) >= 4) &&
                     (fseek(in, end_pos - (long)sizeof(uint32_t), SEEK_SET) == 0) &&
                     (read_uint32(in, &stored_crc) == 0) &&
                     (fseek(in, 0, SEEK_SET) == 0);
        if (crc_ok) {
            uint32_t crc = gv_crc32_init();
            char crcbuf[65536];
            long remaining = end_pos - (long)sizeof(uint32_t);
            while (remaining > 0) {
                size_t chunk = (remaining > (long)sizeof(crcbuf)) ? sizeof(crcbuf) : (size_t)remaining;
                if (fread(crcbuf, 1, chunk, in) != chunk) {
                    crc_ok = 0;
                    break;
                }
                crc = gv_crc32_update(crc, crcbuf, chunk);
                remaining -= (long)chunk;
            }
            if (crc_ok) {
                crc = gv_crc32_finish(crc);
                if (crc != stored_crc || fseek(in, resume_pos, SEEK_SET) != 0) {
                    crc_ok = 0;
                }
            }
        }
        if (!crc_ok) {
            fclose(in);
            db_open_from_memory_cleanup(db);
            return NULL;
        }
    }

    if (file_index_type != (uint32_t)db->index_type) {
        fclose(in);
        db_open_from_memory_cleanup(db);
        return NULL;
    }

    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        if (db->soa_storage == NULL) {
            fclose(in);
            db_open_from_memory_cleanup(db);
            return NULL;
        }
        if (kdtree_load_recursive(&(db->root), db->soa_storage, in, db->dimension, file_version) != 0) {
            fclose(in);
            db_open_from_memory_cleanup(db);
            return NULL;
        }
        db_rebuild_metadata_index_from_soa(db);
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        void *loaded_index = NULL;
        if (gv_hnsw_load(&loaded_index, in, db->dimension, file_version,
                         db->soa_storage) != 0) {
            fclose(in);
            if (loaded_index) gv_hnsw_destroy(loaded_index);
            db_open_from_memory_cleanup(db);
            return NULL;
        }
        db->hnsw_index = loaded_index;
        db->count = gv_hnsw_count(db->hnsw_index);
        db_rebuild_metadata_index_from_soa(db);
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        void *loaded_index = NULL;
        if (gv_ivfpq_load(&loaded_index, in, db->dimension, file_version) != 0) {
            fclose(in);
            if (loaded_index) gv_ivfpq_destroy(loaded_index);
            db_open_from_memory_cleanup(db);
            return NULL;
        }
        db->hnsw_index = loaded_index;
        db->count = gv_ivfpq_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_SPARSE) {
        GV_SparseIndex *loaded_index = NULL;
        if (sparse_index_load(&loaded_index, in, db->dimension, (size_t)file_count, file_version) != 0) {
            fclose(in);
            db_open_from_memory_cleanup(db);
            return NULL;
        }
        db->sparse_index = loaded_index;
        db->count = (size_t)file_count;
    } else if (db->index_type == GV_INDEX_TYPE_FLAT) {
        void *loaded_index = NULL;
        if (flat_load(&loaded_index, in, db->dimension, file_version) != 0) {
            fclose(in);
            if (loaded_index) flat_destroy(loaded_index);
            db_open_from_memory_cleanup(db);
            return NULL;
        }
        db->hnsw_index = loaded_index;
        db->count = flat_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFFLAT) {
        void *loaded_index = NULL;
        if (ivfflat_load(&loaded_index, in, db->dimension, file_version) != 0) {
            fclose(in);
            if (loaded_index) ivfflat_destroy(loaded_index);
            db_open_from_memory_cleanup(db);
            return NULL;
        }
        db->hnsw_index = loaded_index;
        db->count = ivfflat_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFSQ8) {
        void *loaded_index = NULL;
        if (ivfsq8_load(&loaded_index, in, db->dimension, file_version) != 0) {
            fclose(in);
            if (loaded_index) ivfsq8_destroy(loaded_index);
            db_open_from_memory_cleanup(db);
            return NULL;
        }
        db->hnsw_index = loaded_index;
        db->count = ivfsq8_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFTURBOQUANT) {
        void *loaded_index = NULL;
        if (ivfturboquant_load(&loaded_index, in, db->dimension, file_version) != 0) {
            fclose(in);
            if (loaded_index) ivfturboquant_destroy(loaded_index);
            db_open_from_memory_cleanup(db);
            return NULL;
        }
        db->hnsw_index = loaded_index;
        db->count = ivfturboquant_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_PQ) {
        void *loaded_index = NULL;
        if (pq_load(&loaded_index, in, db->dimension, file_version) != 0) {
            fclose(in);
            if (loaded_index) pq_destroy(loaded_index);
            db_open_from_memory_cleanup(db);
            return NULL;
        }
        db->hnsw_index = loaded_index;
        db->count = pq_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_LSH) {
        void *loaded_index = NULL;
        if (lsh_load(&loaded_index, in, db->dimension, file_version) != 0) {
            fclose(in);
            if (loaded_index) lsh_destroy(loaded_index);
            db_open_from_memory_cleanup(db);
            return NULL;
        }
        db->hnsw_index = loaded_index;
        db->count = lsh_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFDISK) {
        if (db->soa_storage == NULL) {
            fclose(in);
            db_open_from_memory_cleanup(db);
            return NULL;
        }
        GV_IVFDiskIndex *loaded_index = NULL;
        if (ivfdisk_load(&loaded_index, in, db->dimension, ivfdisk_data_dir, file_version) != 0 ||
            soa_storage_load(db->soa_storage, in, file_version) != 0) {
            fclose(in);
            if (loaded_index) ivfdisk_destroy(loaded_index);
            db_open_from_memory_cleanup(db);
            return NULL;
        }
        db->hnsw_index = loaded_index;
        db->count = soa_storage_count(db->soa_storage);
    } else {
        fclose(in);
        db_open_from_memory_cleanup(db);
        return NULL;
    }

    /* Whole-file CRC was already verified before index parsing (see above). */

    fclose(in);

    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        db->count = file_count;
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        db->count = gv_hnsw_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        db->count = gv_ivfpq_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_FLAT) {
        db->count = flat_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFFLAT) {
        db->count = ivfflat_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFSQ8) {
        db->count = ivfsq8_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFTURBOQUANT) {
        db->count = ivfturboquant_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_PQ) {
        db->count = pq_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_LSH) {
        db->count = lsh_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_RABITQ) {
        db->count = rabitq_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFDISK) {
        db->count = soa_storage_count(db->soa_storage);
    }

    /* WAL is intentionally disabled for memory-backed snapshots. */
    return db;
}

GV_Database *db_open_from_memory(const void *data, size_t size,
                                    size_t dimension, GV_IndexType index_type) {
    if (index_type == GV_INDEX_TYPE_IVFDISK) {
        return NULL;
    }
    return db_open_from_memory_impl(data, size, dimension, index_type, NULL);
}

GV_Database *db_open_from_memory_ivfdisk(const void *data, size_t size,
                                            size_t dimension,
                                            const char *ivfdisk_data_dir) {
    return db_open_from_memory_impl(data, size, dimension,
                                    GV_INDEX_TYPE_IVFDISK, ivfdisk_data_dir);
}

GV_Database *db_open_mmap(const char *filepath, size_t dimension, GV_IndexType index_type) {
    if (filepath == NULL) {
        return NULL;
    }
    char ivfdisk_dir[1024];
    const char *ivfdisk_data_dir = NULL;
    if (index_type == GV_INDEX_TYPE_IVFDISK) {
        if (snprintf(ivfdisk_dir, sizeof(ivfdisk_dir), "%s.ivfdisk", filepath) >= (int)sizeof(ivfdisk_dir)) {
            return NULL;
        }
        ivfdisk_data_dir = ivfdisk_dir;
    }
    GV_MMap *mm = mmap_open_readonly(filepath);
    if (mm == NULL) {
        return NULL;
    }
    const void *data = mmap_data(mm);
    size_t size = mmap_size(mm);
    if (data == NULL || size == 0) {
        mmap_close(mm);
        return NULL;
    }

    GV_Database *db = NULL;
    if (index_type == GV_INDEX_TYPE_IVFDISK) {
        db = db_open_from_memory_ivfdisk(data, size, dimension, ivfdisk_data_dir);
    } else {
        db = db_open_from_memory(data, size, dimension, index_type);
    }
    if (db == NULL) {
        mmap_close(mm);
        return NULL;
    }

    /* Attach mapping to db->filepath so user can still see origin; store pointer via wal_path. */
    /* We reuse wal_path as an opaque holder for the mapping pointer in this special mode. */
    db->wal = NULL;
    db->wal_replaying = 0;
    db->wal_path = (char *)mm; /* opaque handle; freed in close via mmap_close */
    return db;
}

GV_Database *db_open_with_hnsw_config(const char *filepath, size_t dimension, 
                                         GV_IndexType index_type, const GV_HNSWConfig *hnsw_config) {
    if (index_type != GV_INDEX_TYPE_HNSW) {
        return db_open(filepath, dimension, index_type);
    }

    if (dimension == 0) {
        return NULL;
    }

    /* Zero-initialize so every field (e.g. otlp_config) has a defined default;
     * db_init_common_fields sets the non-zero ones afterward. */
    GV_Database *db = (GV_Database *)gv_calloc(1, sizeof(GV_Database));
    if (db == NULL) {
        return NULL;
    }

    db->dimension = dimension;
    db->index_type = index_type;
    db->root = NULL;
    db->hnsw_index = NULL;
    db->sparse_index = NULL;
    db->soa_storage = NULL;
    db->filepath = NULL;
    db->wal_path = NULL;
    if (filepath != NULL) {
        db->filepath = gv_dup_cstr(filepath);
        if (db->filepath == NULL) {
            free(db);
            return NULL;
        }
        db->wal_path = db_build_wal_path(filepath);
        if (db->wal_path == NULL) {
            free(db->filepath);
            free(db);
            return NULL;
        }
    }
    db->wal = NULL;
    db->wal_replaying = 0;
    pthread_rwlock_init(&db->rwlock, NULL);
    pthread_mutex_init(&db->wal_mutex, NULL);
    db->count = 0;
    db->exact_search_threshold = 1000;
    db->force_exact_search = 0;
    db->total_inserts = 0;
    db->total_queries = 0;
    db->total_range_queries = 0;
    db->total_wal_records = 0;
    db->cosine_normalized = 0;
    db->metadata_index = metadata_index_create();
    if (db->metadata_index == NULL) {
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        gv_free(db->filepath);
        gv_free(db->wal_path);
        gv_free(db);
        return NULL;
    }
    db_init_common_fields(db);

    if (index_type == GV_INDEX_TYPE_KDTREE || index_type == GV_INDEX_TYPE_HNSW) {
        db->soa_storage = soa_storage_create(dimension, 0);
        if (db->soa_storage == NULL) {
            metadata_index_destroy(db->metadata_index);
            pthread_rwlock_destroy(&db->rwlock);
            pthread_mutex_destroy(&db->wal_mutex);
            gv_free(db);
            return NULL;
        }
        db_attach_soa_storage(db);
    }

    db->hnsw_index = gv_hnsw_create(dimension, hnsw_config, db->soa_storage);
    if (db->hnsw_index == NULL) {
        if (db->soa_storage != NULL) {
            soa_storage_destroy(db->soa_storage);
        }
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        gv_free(db);
        return NULL;
    }

    if (db->wal_path != NULL) {
        db->wal = wal_open(db->wal_path, db->dimension, (uint32_t)db->index_type);
    }

    return db;
}

GV_Database *db_open_with_ivfpq_config(const char *filepath, size_t dimension, 
                                          GV_IndexType index_type, const GV_IVFPQConfig *ivfpq_config) {
    if (index_type != GV_INDEX_TYPE_IVFPQ) {
        return db_open(filepath, dimension, index_type);
    }

    if (dimension == 0) {
        return NULL;
    }

    /* Zero-initialize so every field (e.g. otlp_config) has a defined default;
     * db_init_common_fields sets the non-zero ones afterward. */
    GV_Database *db = (GV_Database *)gv_calloc(1, sizeof(GV_Database));
    if (db == NULL) {
        return NULL;
    }

    db->dimension = dimension;
    db->index_type = index_type;
    db->root = NULL;
    db->hnsw_index = NULL;
    db->sparse_index = NULL;
    db->filepath = NULL;
    db->wal_path = NULL;

    if (filepath != NULL) {
        db->filepath = gv_dup_cstr(filepath);
        if (db->filepath == NULL) {
            gv_free(db);
            return NULL;
        }
        db->wal_path = db_build_wal_path(filepath);
        if (db->wal_path == NULL) {
            gv_free(db->filepath);
            gv_free(db);
            return NULL;
        }
    }
    db->wal = NULL;
    db->wal_replaying = 0;
    pthread_rwlock_init(&db->rwlock, NULL);
    pthread_mutex_init(&db->wal_mutex, NULL);
    db->count = 0;
    db->exact_search_threshold = 1000;
    db->force_exact_search = 0;
    db->total_inserts = 0;
    db->total_queries = 0;
    db->total_range_queries = 0;
    db->total_wal_records = 0;
    db->cosine_normalized = 0;
    db->soa_storage = NULL;
    db->metadata_index = metadata_index_create();
    if (db->metadata_index == NULL) {
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        gv_free(db);
        return NULL;
    }
    db_init_common_fields(db);

    if (ivfpq_config != NULL) {
        db->hnsw_index = gv_ivfpq_create(dimension, ivfpq_config);
    } else {
        GV_IVFPQConfig default_cfg = {.nlist = 64, .m = 8, .nbits = 8, .nprobe = 4, .train_iters = 15};
        db->hnsw_index = gv_ivfpq_create(dimension, &default_cfg);
    }
    
    if (db->hnsw_index == NULL) {
        metadata_index_destroy(db->metadata_index);
        pthread_mutex_destroy(&db->resource_mutex);
    pthread_mutex_destroy(&db->txn_mutex);
        pthread_mutex_destroy(&db->observability_mutex);
    pthread_mutex_destroy(&db->ab_mutex);
        pthread_cond_destroy(&db->compaction_cond);
        pthread_mutex_destroy(&db->compaction_mutex);
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        gv_free(db);
        return NULL;
    }

    if (db->wal_path != NULL) {
        db->wal = wal_open(db->wal_path, db->dimension, (uint32_t)db->index_type);
    }

    return db;
}

GV_Database *db_open_with_ivfflat_config(const char *filepath, size_t dimension,
                                             GV_IndexType index_type, const GV_IVFFlatConfig *config) {
    if (index_type != GV_INDEX_TYPE_IVFFLAT) {
        return db_open(filepath, dimension, index_type);
    }
    if (dimension == 0) {
        return NULL;
    }

    /* Zero-initialize so every field (e.g. otlp_config) has a defined default;
     * db_init_common_fields sets the non-zero ones afterward. */
    GV_Database *db = (GV_Database *)gv_calloc(1, sizeof(GV_Database));
    if (db == NULL) {
        return NULL;
    }

    db->dimension = dimension;
    db->index_type = index_type;
    db->root = NULL;
    db->hnsw_index = NULL;
    db->sparse_index = NULL;
    db->soa_storage = NULL;
    db->filepath = NULL;
    db->wal_path = NULL;
    if (filepath != NULL) {
        db->filepath = gv_dup_cstr(filepath);
        if (db->filepath == NULL) {
            gv_free(db);
            return NULL;
        }
        db->wal_path = db_build_wal_path(filepath);
        if (db->wal_path == NULL) {
            gv_free(db->filepath);
            gv_free(db);
            return NULL;
        }
    }
    db->wal = NULL;
    db->wal_replaying = 0;
    pthread_rwlock_init(&db->rwlock, NULL);
    pthread_mutex_init(&db->wal_mutex, NULL);
    db->count = 0;
    db->exact_search_threshold = 1000;
    db->force_exact_search = 0;
    db->total_inserts = 0;
    db->total_queries = 0;
    db->total_range_queries = 0;
    db->total_wal_records = 0;
    db->cosine_normalized = 0;
    db->metadata_index = metadata_index_create();
    if (db->metadata_index == NULL) {
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        gv_free(db);
        return NULL;
    }
    db_init_common_fields(db);

    if (config != NULL) {
        db->hnsw_index = ivfflat_create(dimension, config);
    } else {
        GV_IVFFlatConfig default_cfg = {.nlist = 64, .nprobe = 4, .train_iters = 15, .use_cosine = 0};
        db->hnsw_index = ivfflat_create(dimension, &default_cfg);
    }

    if (db->hnsw_index == NULL) {
        metadata_index_destroy(db->metadata_index);
        pthread_mutex_destroy(&db->resource_mutex);
    pthread_mutex_destroy(&db->txn_mutex);
        pthread_mutex_destroy(&db->observability_mutex);
    pthread_mutex_destroy(&db->ab_mutex);
        pthread_cond_destroy(&db->compaction_cond);
        pthread_mutex_destroy(&db->compaction_mutex);
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        gv_free(db);
        return NULL;
    }

    return db;
}

GV_Database *db_open_with_ivfdisk_config(const char *filepath, size_t dimension,
                                             GV_IndexType index_type, const GV_IVFDiskConfig *config) {
    if (index_type != GV_INDEX_TYPE_IVFDISK) {
        return db_open(filepath, dimension, index_type);
    }
    if (dimension == 0 || filepath == NULL || !*filepath) {
        return NULL;
    }

    /* Zero-initialize so every field (e.g. otlp_config) has a defined default;
     * db_init_common_fields sets the non-zero ones afterward. */
    GV_Database *db = (GV_Database *)gv_calloc(1, sizeof(GV_Database));
    if (db == NULL) {
        return NULL;
    }

    db->dimension = dimension;
    db->index_type = index_type;
    db->root = NULL;
    db->hnsw_index = NULL;
    db->sparse_index = NULL;
    db->soa_storage = soa_storage_create(dimension, 0);
    db->filepath = gv_dup_cstr(filepath);
    db->wal_path = NULL;
    db->wal = NULL;
    db->wal_replaying = 0;
    pthread_rwlock_init(&db->rwlock, NULL);
    pthread_mutex_init(&db->wal_mutex, NULL);
    db->count = 0;
    db->exact_search_threshold = 1000;
    db->force_exact_search = 0;
    db->total_inserts = 0;
    db->total_queries = 0;
    db->total_range_queries = 0;
    db->total_wal_records = 0;
    db->cosine_normalized = 0;
    db->metadata_index = metadata_index_create();
    if (db->metadata_index == NULL || db->soa_storage == NULL || db->filepath == NULL) {
        if (db->soa_storage) soa_storage_destroy(db->soa_storage);
        if (db->metadata_index) metadata_index_destroy(db->metadata_index);
        gv_free(db->filepath);
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        gv_free(db);
        return NULL;
    }
    db_init_common_fields(db);
    db_attach_soa_storage(db);

    db->wal_path = db_build_wal_path(filepath);
    if (db->wal_path == NULL ||
        db_ivfdisk_create_index(db, dimension, filepath, config) != 0) {
        metadata_index_destroy(db->metadata_index);
        soa_storage_destroy(db->soa_storage);
        gv_free(db->filepath);
        gv_free(db->wal_path);
        pthread_mutex_destroy(&db->resource_mutex);
    pthread_mutex_destroy(&db->txn_mutex);
        pthread_mutex_destroy(&db->observability_mutex);
    pthread_mutex_destroy(&db->ab_mutex);
        pthread_cond_destroy(&db->compaction_cond);
        pthread_mutex_destroy(&db->compaction_mutex);
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        gv_free(db);
        return NULL;
    }

    db->wal = wal_open(db->wal_path, db->dimension, (uint32_t)db->index_type);
    return db;
}

GV_Database *db_open_with_ivfsq8_config(const char *filepath, size_t dimension,
                                        GV_IndexType index_type, const GV_IVFSQ8Config *config) {
    if (index_type != GV_INDEX_TYPE_IVFSQ8) {
        return db_open(filepath, dimension, index_type);
    }
    if (dimension == 0) {
        return NULL;
    }

    /* Zero-initialize so every field (e.g. otlp_config) has a defined default;
     * db_init_common_fields sets the non-zero ones afterward. */
    GV_Database *db = (GV_Database *)gv_calloc(1, sizeof(GV_Database));
    if (db == NULL) {
        return NULL;
    }

    db->dimension = dimension;
    db->index_type = index_type;
    db->root = NULL;
    db->hnsw_index = NULL;
    db->sparse_index = NULL;
    db->soa_storage = NULL;
    db->filepath = NULL;
    db->wal_path = NULL;
    if (filepath != NULL) {
        db->filepath = gv_dup_cstr(filepath);
        if (db->filepath == NULL) {
            gv_free(db);
            return NULL;
        }
        db->wal_path = db_build_wal_path(filepath);
        if (db->wal_path == NULL) {
            gv_free(db->filepath);
            gv_free(db);
            return NULL;
        }
    }
    db->wal = NULL;
    db->wal_replaying = 0;
    pthread_rwlock_init(&db->rwlock, NULL);
    pthread_mutex_init(&db->wal_mutex, NULL);
    db->count = 0;
    db->exact_search_threshold = 1000;
    db->force_exact_search = 0;
    db->total_inserts = 0;
    db->total_queries = 0;
    db->total_range_queries = 0;
    db->total_wal_records = 0;
    db->cosine_normalized = 0;
    db->metadata_index = metadata_index_create();
    if (db->metadata_index == NULL) {
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        gv_free(db);
        return NULL;
    }
    db_init_common_fields(db);

    if (config != NULL) {
        db->hnsw_index = ivfsq8_create(dimension, config);
    } else {
        GV_IVFSQ8Config default_cfg = {
            .nlist = 64, .nprobe = 4, .train_iters = 15, .use_cosine = 0,
            .per_dimension = 0, .default_rerank = 200
        };
        db->hnsw_index = ivfsq8_create(dimension, &default_cfg);
    }

    if (db->hnsw_index == NULL) {
        metadata_index_destroy(db->metadata_index);
        pthread_mutex_destroy(&db->resource_mutex);
    pthread_mutex_destroy(&db->txn_mutex);
        pthread_mutex_destroy(&db->observability_mutex);
    pthread_mutex_destroy(&db->ab_mutex);
        pthread_cond_destroy(&db->compaction_cond);
        pthread_mutex_destroy(&db->compaction_mutex);
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        gv_free(db);
        return NULL;
    }

    return db;
}


GV_Database *db_open_with_ivfturboquant_config(const char *filepath, size_t dimension,
                                               GV_IndexType index_type,
                                               const GV_IVFTurboQuantConfig *config) {
    if (index_type != GV_INDEX_TYPE_IVFTURBOQUANT) {
        return db_open(filepath, dimension, index_type);
    }
    if (dimension == 0 || dimension % 2 != 0) {
        return NULL;
    }

    /* Zero-initialize so every field (e.g. otlp_config) has a defined default;
     * db_init_common_fields sets the non-zero ones afterward. */
    GV_Database *db = (GV_Database *)gv_calloc(1, sizeof(GV_Database));
    if (db == NULL) {
        return NULL;
    }

    db->dimension = dimension;
    db->index_type = index_type;
    db->root = NULL;
    db->hnsw_index = NULL;
    db->sparse_index = NULL;
    db->soa_storage = NULL;
    db->filepath = NULL;
    db->wal_path = NULL;
    if (filepath != NULL) {
        db->filepath = gv_dup_cstr(filepath);
        if (db->filepath == NULL) {
            gv_free(db);
            return NULL;
        }
        db->wal_path = db_build_wal_path(filepath);
        if (db->wal_path == NULL) {
            gv_free(db->filepath);
            gv_free(db);
            return NULL;
        }
    }
    db->wal = NULL;
    db->wal_replaying = 0;
    pthread_rwlock_init(&db->rwlock, NULL);
    pthread_mutex_init(&db->wal_mutex, NULL);
    db->count = 0;
    db->exact_search_threshold = 1000;
    db->force_exact_search = 0;
    db->total_inserts = 0;
    db->total_queries = 0;
    db->total_range_queries = 0;
    db->total_wal_records = 0;
    db->cosine_normalized = 0;
    db->metadata_index = metadata_index_create();
    if (db->metadata_index == NULL) {
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        gv_free(db);
        return NULL;
    }
    db_init_common_fields(db);

    if (config != NULL) {
        db->hnsw_index = ivfturboquant_create(dimension, config);
    } else {
        GV_IVFTurboQuantConfig default_cfg;
        default_cfg.nlist = 64;
        default_cfg.nprobe = 4;
        default_cfg.train_iters = 15;
        default_cfg.use_cosine = 0;
        default_cfg.default_rerank = 200;
        default_cfg.turbo.bits = 8;
        default_cfg.turbo.projections = dimension / 4;
        if (default_cfg.turbo.projections == 0) default_cfg.turbo.projections = 2;
        default_cfg.turbo.seed = 42;
        default_cfg.turbo.use_qjl = 1;
        default_cfg.turbo.rotation = GV_TURBOQUANT_ROTATION_AUTO;
        db->hnsw_index = ivfturboquant_create(dimension, &default_cfg);
    }

    if (db->hnsw_index == NULL) {
        metadata_index_destroy(db->metadata_index);
        pthread_mutex_destroy(&db->resource_mutex);
    pthread_mutex_destroy(&db->txn_mutex);
        pthread_mutex_destroy(&db->observability_mutex);
    pthread_mutex_destroy(&db->ab_mutex);
        pthread_mutex_destroy(&db->compaction_mutex);
        pthread_cond_destroy(&db->compaction_cond);
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        gv_free(db->filepath);
        gv_free(db->wal_path);
        gv_free(db);
        return NULL;
    }

    return db;
}


GV_Database *db_open_with_pq_config(const char *filepath, size_t dimension,
                                        GV_IndexType index_type, const GV_PQConfig *config) {
    if (index_type != GV_INDEX_TYPE_PQ) {
        return db_open(filepath, dimension, index_type);
    }
    if (dimension == 0) {
        return NULL;
    }

    /* Zero-initialize so every field (e.g. otlp_config) has a defined default;
     * db_init_common_fields sets the non-zero ones afterward. */
    GV_Database *db = (GV_Database *)gv_calloc(1, sizeof(GV_Database));
    if (db == NULL) {
        return NULL;
    }

    db->dimension = dimension;
    db->index_type = index_type;
    db->root = NULL;
    db->hnsw_index = NULL;
    db->sparse_index = NULL;
    db->soa_storage = NULL;
    db->filepath = NULL;
    db->wal_path = NULL;
    if (filepath != NULL) {
        db->filepath = gv_dup_cstr(filepath);
        if (db->filepath == NULL) {
            gv_free(db);
            return NULL;
        }
        db->wal_path = db_build_wal_path(filepath);
        if (db->wal_path == NULL) {
            gv_free(db->filepath);
            gv_free(db);
            return NULL;
        }
    }
    db->wal = NULL;
    db->wal_replaying = 0;
    pthread_rwlock_init(&db->rwlock, NULL);
    pthread_mutex_init(&db->wal_mutex, NULL);
    db->count = 0;
    db->exact_search_threshold = 1000;
    db->force_exact_search = 0;
    db->total_inserts = 0;
    db->total_queries = 0;
    db->total_range_queries = 0;
    db->total_wal_records = 0;
    db->cosine_normalized = 0;
    db->metadata_index = metadata_index_create();
    if (db->metadata_index == NULL) {
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        gv_free(db);
        return NULL;
    }
    db_init_common_fields(db);

    if (config != NULL) {
        db->hnsw_index = pq_create(dimension, config);
    } else {
        GV_PQConfig default_cfg = {.m = 8, .nbits = 8, .train_iters = 15};
        db->hnsw_index = pq_create(dimension, &default_cfg);
    }

    if (db->hnsw_index == NULL) {
        metadata_index_destroy(db->metadata_index);
        pthread_mutex_destroy(&db->resource_mutex);
    pthread_mutex_destroy(&db->txn_mutex);
        pthread_mutex_destroy(&db->observability_mutex);
    pthread_mutex_destroy(&db->ab_mutex);
        pthread_cond_destroy(&db->compaction_cond);
        pthread_mutex_destroy(&db->compaction_mutex);
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        gv_free(db);
        return NULL;
    }

    return db;
}

GV_Database *db_open_with_lsh_config(const char *filepath, size_t dimension,
                                         GV_IndexType index_type, const GV_LSHConfig *config) {
    if (index_type != GV_INDEX_TYPE_LSH && index_type != GV_INDEX_TYPE_RABITQ) {
        return db_open(filepath, dimension, index_type);
    }
    if (dimension == 0) {
        return NULL;
    }

    /* Zero-initialize so every field (e.g. otlp_config) has a defined default;
     * db_init_common_fields sets the non-zero ones afterward. */
    GV_Database *db = (GV_Database *)gv_calloc(1, sizeof(GV_Database));
    if (db == NULL) {
        return NULL;
    }

    db->dimension = dimension;
    db->index_type = index_type;
    db->root = NULL;
    db->hnsw_index = NULL;
    db->sparse_index = NULL;
    db->soa_storage = NULL;
    db->filepath = NULL;
    db->wal_path = NULL;
    if (filepath != NULL) {
        db->filepath = gv_dup_cstr(filepath);
        if (db->filepath == NULL) {
            gv_free(db);
            return NULL;
        }
        db->wal_path = db_build_wal_path(filepath);
        if (db->wal_path == NULL) {
            gv_free(db->filepath);
            gv_free(db);
            return NULL;
        }
    }
    db->wal = NULL;
    db->wal_replaying = 0;
    pthread_rwlock_init(&db->rwlock, NULL);
    pthread_mutex_init(&db->wal_mutex, NULL);
    db->count = 0;
    db->exact_search_threshold = 1000;
    db->force_exact_search = 0;
    db->total_inserts = 0;
    db->total_queries = 0;
    db->total_range_queries = 0;
    db->total_wal_records = 0;
    db->cosine_normalized = 0;
    db->metadata_index = metadata_index_create();
    if (db->metadata_index == NULL) {
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        gv_free(db);
        return NULL;
    }
    db_init_common_fields(db);

    db->soa_storage = soa_storage_create(dimension, 0);
    if (db->soa_storage == NULL) {
        metadata_index_destroy(db->metadata_index);
        pthread_mutex_destroy(&db->resource_mutex);
    pthread_mutex_destroy(&db->txn_mutex);
        pthread_mutex_destroy(&db->observability_mutex);
    pthread_mutex_destroy(&db->ab_mutex);
        pthread_cond_destroy(&db->compaction_cond);
        pthread_mutex_destroy(&db->compaction_mutex);
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        gv_free(db);
        return NULL;
    }
    db_attach_soa_storage(db);

    if (index_type == GV_INDEX_TYPE_RABITQ) {
        GV_RaBitQConfig rq_cfg = {.seed = 42, .rerank_factor = 4};
        db->hnsw_index = rabitq_create(dimension, &rq_cfg, db->soa_storage);
    } else if (config != NULL) {
        db->hnsw_index = lsh_create(dimension, config, db->soa_storage);
    } else {
        GV_LSHConfig default_cfg = {.num_tables = 8, .num_hash_bits = 16, .seed = 42};
        db->hnsw_index = lsh_create(dimension, &default_cfg, db->soa_storage);
    }

    if (db->hnsw_index == NULL) {
        soa_storage_destroy(db->soa_storage);
        metadata_index_destroy(db->metadata_index);
        pthread_mutex_destroy(&db->resource_mutex);
    pthread_mutex_destroy(&db->txn_mutex);
        pthread_mutex_destroy(&db->observability_mutex);
    pthread_mutex_destroy(&db->ab_mutex);
        pthread_cond_destroy(&db->compaction_cond);
        pthread_mutex_destroy(&db->compaction_mutex);
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        gv_free(db);
        return NULL;
    }

    return db;
}

typedef struct {
    size_t dimension;
    size_t M;
    size_t efConstruction;
    size_t efSearch;
    /* ... rest not needed */
} GV_HNSWIndexPartial;

typedef struct {
    size_t dimension;
    GV_IVFFlatConfig config;
    /* ... rest not needed */
} GV_IVFFlatIndexPartial;

typedef struct {
    size_t dimension;
    GV_IVFSQ8Config config;
    /* ... rest not needed */
} GV_IVFSQ8IndexPartial;

typedef struct {
    size_t dimension;
    GV_IVFTurboQuantConfig config;
    /* ... rest not needed */
} GV_IVFTurboQuantIndexPartial;

/*
 * Serializes the save/set/search/restore sequence that temporarily overrides
 * shared index params (HNSW efSearch, IVF nprobe) for one query. We can't wrap
 * it in wrlock(&db->rwlock) because db_search() rdlocks the same non-recursive
 * rwlock internally -> deadlock; there's no per-DB mutex available (header out
 * of scope). This file-scoped mutex makes mutate+search+restore mutually
 * exclusive, at the cost of serializing all param-override searches process-wide.
 */
static pthread_mutex_t g_search_params_mutex = PTHREAD_MUTEX_INITIALIZER;

int db_search_with_params(const GV_Database *db, const float *query_data, size_t k,
                              GV_SearchResult *results, GV_DistanceType distance_type,
                              const GV_SearchParams *params) {
    if (params == NULL) {
        return db_search(db, query_data, k, results, distance_type);
    }

    if (db == NULL || query_data == NULL || results == NULL || k == 0) {
        return -1;
    }

    if (db->index_type == GV_INDEX_TYPE_HNSW && params->ef_search > 0 && db->hnsw_index != NULL) {
        GV_HNSWIndexPartial *idx = (GV_HNSWIndexPartial *)db->hnsw_index;
        pthread_mutex_lock(&g_search_params_mutex);
        size_t saved_ef = idx->efSearch;
        idx->efSearch = params->ef_search;
        int r = db_search(db, query_data, k, results, distance_type);
        idx->efSearch = saved_ef;
        pthread_mutex_unlock(&g_search_params_mutex);
        return r;
    }

    if (db->index_type == GV_INDEX_TYPE_IVFFLAT && params->nprobe > 0 && db->hnsw_index != NULL) {
        GV_IVFFlatIndexPartial *idx = (GV_IVFFlatIndexPartial *)db->hnsw_index;
        pthread_mutex_lock(&g_search_params_mutex);
        size_t saved_nprobe = idx->config.nprobe;
        idx->config.nprobe = params->nprobe;
        int r = db_search(db, query_data, k, results, distance_type);
        idx->config.nprobe = saved_nprobe;
        pthread_mutex_unlock(&g_search_params_mutex);
        return r;
    }

    if (db->index_type == GV_INDEX_TYPE_IVFDISK && params->nprobe > 0 && db->hnsw_index != NULL) {
        GV_IVFDiskIndex *idx = (GV_IVFDiskIndex *)db->hnsw_index;
        pthread_mutex_lock(&g_search_params_mutex);
        size_t saved_nprobe = ivfdisk_get_nprobe(idx);
        ivfdisk_set_nprobe(idx, params->nprobe);
        int r = db_search(db, query_data, k, results, distance_type);
        ivfdisk_set_nprobe(idx, saved_nprobe);
        pthread_mutex_unlock(&g_search_params_mutex);
        return r;
    }

    if (db->index_type == GV_INDEX_TYPE_IVFSQ8 && params->nprobe > 0 && db->hnsw_index != NULL) {
        GV_IVFSQ8IndexPartial *idx = (GV_IVFSQ8IndexPartial *)db->hnsw_index;
        pthread_mutex_lock(&g_search_params_mutex);
        size_t saved_nprobe = idx->config.nprobe;
        idx->config.nprobe = params->nprobe;
        int r = db_search(db, query_data, k, results, distance_type);
        idx->config.nprobe = saved_nprobe;
        pthread_mutex_unlock(&g_search_params_mutex);
        return r;
    }

    if (db->index_type == GV_INDEX_TYPE_IVFTURBOQUANT && params->nprobe > 0 && db->hnsw_index != NULL) {
        GV_IVFTurboQuantIndexPartial *idx = (GV_IVFTurboQuantIndexPartial *)db->hnsw_index;
        pthread_mutex_lock(&g_search_params_mutex);
        size_t saved_nprobe = idx->config.nprobe;
        idx->config.nprobe = params->nprobe;
        int r = db_search(db, query_data, k, results, distance_type);
        idx->config.nprobe = saved_nprobe;
        pthread_mutex_unlock(&g_search_params_mutex);
        return r;
    }

    if (db->index_type == GV_INDEX_TYPE_IVFPQ && db->hnsw_index != NULL) {
        memset(results, 0, k * sizeof(GV_SearchResult));
        pthread_rwlock_rdlock((pthread_rwlock_t *)&db->rwlock);
        ((GV_Database *)db)->total_queries += 1;

        GV_Vector query_vec;
        query_vec.dimension = db->dimension;
        query_vec.data = (float *)query_data;
        query_vec.metadata = NULL;

        int r = gv_ivfpq_search(db->hnsw_index, &query_vec, k, results, distance_type,
                                 params->nprobe, params->rerank_top);
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return r;
    }

    return db_search(db, query_data, k, results, distance_type);
}

#include "features/json.h"

