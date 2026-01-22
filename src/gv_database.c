#include <errno.h>
#include <stdint.h>
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

#include "gigavector/gv_types.h"

// Helper function to create metadata


#include "gigavector/gv_database.h"
#include "gigavector/gv_distance.h"
#include "gigavector/gv_exact_search.h"
#include "gigavector/gv_hnsw.h"
#include "gigavector/gv_ivfpq.h"
#include "gigavector/gv_kdtree.h"
#include "gigavector/gv_metadata.h"
#include "gigavector/gv_sparse_index.h"
#include "gigavector/gv_vector.h"
#include "gigavector/gv_wal.h"
#include "gigavector/gv_mmap.h"
#include "gigavector/gv_soa_storage.h"
#include "gigavector/gv_metadata_index.h"

#include <math.h>
#include <sys/time.h>

/* Forward declarations for resource limits */
static size_t gv_db_estimate_vector_memory(size_t dimension);
static void gv_db_update_memory_usage(GV_Database *db);
static int gv_db_check_resource_limits(GV_Database *db, size_t additional_vectors, size_t additional_memory);
static void gv_db_increment_concurrent_ops(GV_Database *db);
static void gv_db_decrement_concurrent_ops(GV_Database *db);
static uint64_t gv_db_get_time_us(void);

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


static int gv_db_wal_apply_rich(void *ctx, const float *data, size_t dimension,
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
    if (gv_db_add_vector_with_rich_metadata(db, data, db->dimension, metadata_keys, metadata_values, metadata_count) != 0) {
        return -1;
    }
    return 0;
}

GV_IndexType gv_index_suggest(size_t dimension, size_t expected_count) {
    if (expected_count <= 20000 && dimension <= 64) {
        return GV_INDEX_TYPE_KDTREE;
    }
    if (expected_count >= 500000 && dimension >= 128) {
        return GV_INDEX_TYPE_IVFPQ;
    }
    return GV_INDEX_TYPE_HNSW;
}

static void gv_db_normalize_vector(GV_Vector *vector) {
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

void gv_db_set_cosine_normalized(GV_Database *db, int enabled) {
    if (db == NULL) {
        return;
    }
    db->cosine_normalized = enabled ? 1 : 0;
}

void gv_db_get_stats(const GV_Database *db, GV_DBStats *out) {
    if (db == NULL || out == NULL) {
        return;
    }
    pthread_rwlock_rdlock((pthread_rwlock_t *)&db->rwlock);
    out->total_inserts = db->total_inserts;
    out->total_queries = db->total_queries;
    out->total_range_queries = db->total_range_queries;
    out->total_wal_records = db->total_wal_records;
    pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
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
    db->metadata_index = gv_metadata_index_create();
    if (db->metadata_index == NULL) {
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        free(db);
        return NULL;
    }
    /* Initialize compaction fields */
    db->compaction_running = 0;
    pthread_mutex_init(&db->compaction_mutex, NULL);
    pthread_cond_init(&db->compaction_cond, NULL);
    db->compaction_interval_sec = 300;  /* Default: 5 minutes */
    db->wal_compaction_threshold = 10 * 1024 * 1024;  /* Default: 10MB */
    db->deleted_ratio_threshold = 0.1;  /* Default: 10% */
    /* Initialize resource limits */
    db->resource_limits.max_memory_bytes = 0;  /* Unlimited by default */
    db->resource_limits.max_vectors = 0;  /* Unlimited by default */
    db->resource_limits.max_concurrent_operations = 0;  /* Unlimited by default */
    db->current_memory_bytes = 0;
    db->current_concurrent_ops = 0;
    pthread_mutex_init(&db->resource_mutex, NULL);
    /* Initialize observability fields */
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

    if (index_type == GV_INDEX_TYPE_KDTREE || index_type == GV_INDEX_TYPE_HNSW) {
        db->soa_storage = gv_soa_storage_create(dimension, 0);
        if (db->soa_storage == NULL) {
            gv_metadata_index_destroy(db->metadata_index);
            pthread_rwlock_destroy(&db->rwlock);
            pthread_mutex_destroy(&db->wal_mutex);
            free(db);
            return NULL;
        }
    }

    if (index_type == GV_INDEX_TYPE_HNSW && filepath == NULL) {
        db->hnsw_index = gv_hnsw_create(dimension, NULL, db->soa_storage);
        if (db->hnsw_index == NULL) {
            if (db->soa_storage != NULL) {
                gv_soa_storage_destroy(db->soa_storage);
            }
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
    } else if (index_type == GV_INDEX_TYPE_SPARSE && filepath == NULL) {
        db->sparse_index = gv_sparse_index_create(dimension);
        if (db->sparse_index == NULL) {
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
                db->hnsw_index = gv_hnsw_create(dimension, NULL, db->soa_storage);
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
        if (db->soa_storage == NULL) {
            fclose(in);
            free(db->filepath);
            free(db->wal_path);
            free(db);
            return NULL;
        }
        if (gv_kdtree_load_recursive(&(db->root), db->soa_storage, in, db->dimension, file_version) != 0) {
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
    } else if (db->index_type == GV_INDEX_TYPE_SPARSE) {
        GV_SparseIndex *loaded_index = NULL;
        if (gv_sparse_index_load(&loaded_index, in, db->dimension, (size_t)file_count, file_version) != 0) {
            fclose(in);
            free(db->filepath);
            free(db->wal_path);
            free(db);
            return NULL;
        }
        db->sparse_index = loaded_index;
        db->count = (size_t)file_count;
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
        if (gv_wal_replay_rich(db->wal_path, db->dimension, gv_db_wal_apply_rich, db, (uint32_t)db->index_type) != 0) {
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

    /* If opened via gv_db_open_mmap, wal_path holds an opaque GV_MMap* handle. */
    if (db->filepath == NULL && db->wal_path != NULL && db->wal == NULL) {
        GV_MMap *mm = (GV_MMap *)db->wal_path;
        gv_mmap_close(mm);
        db->wal_path = NULL;
    }

    pthread_rwlock_destroy(&db->rwlock);
    pthread_mutex_destroy(&db->wal_mutex);
    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        gv_kdtree_destroy_recursive(db->root);
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        gv_hnsw_destroy(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        gv_ivfpq_destroy(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_SPARSE) {
        gv_sparse_index_destroy(db->sparse_index);
    }
    if (db->soa_storage != NULL) {
        gv_soa_storage_destroy(db->soa_storage);
    }
    if (db->metadata_index != NULL) {
        gv_metadata_index_destroy(db->metadata_index);
    }
    /* Stop background compaction if running */
    if (db->compaction_running) {
        gv_db_stop_background_compaction(db);
    }
    pthread_mutex_destroy(&db->compaction_mutex);
    pthread_cond_destroy(&db->compaction_cond);
    pthread_mutex_destroy(&db->resource_mutex);
    /* Free observability resources */
    if (db->insert_latency_hist.buckets != NULL) {
        free(db->insert_latency_hist.buckets);
        free(db->insert_latency_hist.bucket_boundaries);
    }
    if (db->search_latency_hist.buckets != NULL) {
        free(db->search_latency_hist.buckets);
        free(db->search_latency_hist.bucket_boundaries);
    }
    pthread_mutex_destroy(&db->observability_mutex);
    free(db->filepath);
    free(db->wal_path);
    free(db);
}

GV_Database *gv_db_open_from_memory(const void *data, size_t size,
                                    size_t dimension, GV_IndexType index_type) {
    if (data == NULL || size == 0) {
        return NULL;
    }
    if (dimension == 0) {
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
    db->metadata_index = gv_metadata_index_create();
    if (db->metadata_index == NULL) {
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        free(db);
        return NULL;
    }
    /* Initialize compaction fields */
    db->compaction_running = 0;
    pthread_mutex_init(&db->compaction_mutex, NULL);
    pthread_cond_init(&db->compaction_cond, NULL);
    db->compaction_interval_sec = 300;  /* Default: 5 minutes */
    db->wal_compaction_threshold = 10 * 1024 * 1024;  /* Default: 10MB */
    db->deleted_ratio_threshold = 0.1;  /* Default: 10% */
    /* Initialize resource limits */
    db->resource_limits.max_memory_bytes = 0;  /* Unlimited by default */
    db->resource_limits.max_vectors = 0;  /* Unlimited by default */
    db->resource_limits.max_concurrent_operations = 0;  /* Unlimited by default */
    db->current_memory_bytes = 0;
    db->current_concurrent_ops = 0;
    pthread_mutex_init(&db->resource_mutex, NULL);
    /* Initialize observability fields */
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

    if (index_type == GV_INDEX_TYPE_KDTREE || index_type == GV_INDEX_TYPE_HNSW) {
        db->soa_storage = gv_soa_storage_create(dimension, 0);
        if (db->soa_storage == NULL) {
            gv_metadata_index_destroy(db->metadata_index);
            pthread_rwlock_destroy(&db->rwlock);
            pthread_mutex_destroy(&db->wal_mutex);
            free(db);
            return NULL;
        }
    }

    FILE *in = fmemopen((void *)data, size, "rb");
    if (in == NULL) {
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        free(db);
        return NULL;
    }

    uint32_t file_dim = 0;
    uint64_t file_count = 0;
    uint32_t file_version = 0;
    if (gv_db_read_header(in, &file_dim, &file_count, &file_version) != 0) {
        fclose(in);
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        free(db);
        return NULL;
    }

    if (dimension != 0 && dimension != (size_t)file_dim) {
        fclose(in);
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        free(db);
        return NULL;
    }
    db->dimension = (size_t)file_dim;

    if (file_version != 1 && file_version != 2 && file_version != 3 && file_version != 4) {
        fclose(in);
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        free(db);
        return NULL;
    }

    uint32_t file_index_type = GV_INDEX_TYPE_KDTREE;
    if (file_version >= 2) {
        if (gv_read_uint32(in, &file_index_type) != 0) {
            fclose(in);
            pthread_rwlock_destroy(&db->rwlock);
            pthread_mutex_destroy(&db->wal_mutex);
            free(db);
            return NULL;
        }
    }

    if (file_index_type != (uint32_t)db->index_type) {
        fclose(in);
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        free(db);
        return NULL;
    }

    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        if (db->soa_storage == NULL) {
            fclose(in);
            pthread_rwlock_destroy(&db->rwlock);
            pthread_mutex_destroy(&db->wal_mutex);
            free(db);
            return NULL;
        }
        if (gv_kdtree_load_recursive(&(db->root), db->soa_storage, in, db->dimension, file_version) != 0) {
            fclose(in);
            pthread_rwlock_destroy(&db->rwlock);
            pthread_mutex_destroy(&db->wal_mutex);
            free(db);
            return NULL;
        }
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        void *loaded_index = NULL;
        if (gv_hnsw_load(&loaded_index, in, db->dimension, file_version) != 0) {
            fclose(in);
            if (loaded_index) gv_hnsw_destroy(loaded_index);
            pthread_rwlock_destroy(&db->rwlock);
            pthread_mutex_destroy(&db->wal_mutex);
            free(db);
            return NULL;
        }
        db->hnsw_index = loaded_index;
        db->count = gv_hnsw_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        void *loaded_index = NULL;
        if (gv_ivfpq_load(&loaded_index, in, db->dimension, file_version) != 0) {
            fclose(in);
            if (loaded_index) gv_ivfpq_destroy(loaded_index);
            pthread_rwlock_destroy(&db->rwlock);
            pthread_mutex_destroy(&db->wal_mutex);
            free(db);
            return NULL;
        }
        db->hnsw_index = loaded_index;
        db->count = gv_ivfpq_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_SPARSE) {
        GV_SparseIndex *loaded_index = NULL;
        if (gv_sparse_index_load(&loaded_index, in, db->dimension, (size_t)file_count, file_version) != 0) {
            fclose(in);
            pthread_rwlock_destroy(&db->rwlock);
            pthread_mutex_destroy(&db->wal_mutex);
            free(db);
            return NULL;
        }
        db->sparse_index = loaded_index;
        db->count = (size_t)file_count;
    } else {
        fclose(in);
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        free(db);
        return NULL;
    }

    if (file_version >= 3) {
        if (fseek(in, 0, SEEK_END) != 0) {
            fclose(in);
            if (db->index_type == GV_INDEX_TYPE_KDTREE) {
                gv_kdtree_destroy_recursive(db->root);
            } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
                if (db->hnsw_index) gv_hnsw_destroy(db->hnsw_index);
            } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
                if (db->hnsw_index) gv_ivfpq_destroy(db->hnsw_index);
            }
            pthread_rwlock_destroy(&db->rwlock);
            pthread_mutex_destroy(&db->wal_mutex);
            free(db);
            return NULL;
        }
        long end_pos = ftell(in);
        if (end_pos < 4) {
            fclose(in);
            if (db->index_type == GV_INDEX_TYPE_KDTREE) {
                gv_kdtree_destroy_recursive(db->root);
            } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
                if (db->hnsw_index) gv_hnsw_destroy(db->hnsw_index);
            } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
                if (db->hnsw_index) gv_ivfpq_destroy(db->hnsw_index);
            }
            pthread_rwlock_destroy(&db->rwlock);
            pthread_mutex_destroy(&db->wal_mutex);
            free(db);
            return NULL;
        }
        if (fseek(in, end_pos - (long)sizeof(uint32_t), SEEK_SET) != 0) {
            fclose(in);
            if (db->index_type == GV_INDEX_TYPE_KDTREE) {
                gv_kdtree_destroy_recursive(db->root);
            } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
                if (db->hnsw_index) gv_hnsw_destroy(db->hnsw_index);
            } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
                if (db->hnsw_index) gv_ivfpq_destroy(db->hnsw_index);
            }
            pthread_rwlock_destroy(&db->rwlock);
            pthread_mutex_destroy(&db->wal_mutex);
            free(db);
            return NULL;
        }
        uint32_t stored_crc = 0;
        if (gv_read_uint32(in, &stored_crc) != 0) {
            fclose(in);
            if (db->index_type == GV_INDEX_TYPE_KDTREE) {
                gv_kdtree_destroy_recursive(db->root);
            } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
                if (db->hnsw_index) gv_hnsw_destroy(db->hnsw_index);
            } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
                if (db->hnsw_index) gv_ivfpq_destroy(db->hnsw_index);
            }
            pthread_rwlock_destroy(&db->rwlock);
            pthread_mutex_destroy(&db->wal_mutex);
            free(db);
            return NULL;
        }
        if (fseek(in, 0, SEEK_SET) != 0) {
            fclose(in);
            if (db->index_type == GV_INDEX_TYPE_KDTREE) {
                gv_kdtree_destroy_recursive(db->root);
            } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
                if (db->hnsw_index) gv_hnsw_destroy(db->hnsw_index);
            } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
                if (db->hnsw_index) gv_ivfpq_destroy(db->hnsw_index);
            }
            pthread_rwlock_destroy(&db->rwlock);
            pthread_mutex_destroy(&db->wal_mutex);
            free(db);
            return NULL;
        }
        uint32_t crc = gv_crc32_init();
        char buf[65536];
        long remaining = end_pos - (long)sizeof(uint32_t);
        while (remaining > 0) {
            size_t chunk = (remaining > (long)sizeof(buf)) ? sizeof(buf) : (size_t)remaining;
            if (fread(buf, 1, chunk, in) != chunk) {
                fclose(in);
                if (db->index_type == GV_INDEX_TYPE_KDTREE) {
                    gv_kdtree_destroy_recursive(db->root);
                } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
                    if (db->hnsw_index) gv_hnsw_destroy(db->hnsw_index);
                } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
                    if (db->hnsw_index) gv_ivfpq_destroy(db->hnsw_index);
                }
                pthread_rwlock_destroy(&db->rwlock);
                pthread_mutex_destroy(&db->wal_mutex);
                free(db);
                return NULL;
            }
            crc = gv_crc32_update(crc, buf, chunk);
            remaining -= (long)chunk;
        }
        crc = gv_crc32_finish(crc);
        if (crc != stored_crc) {
            fclose(in);
            if (db->index_type == GV_INDEX_TYPE_KDTREE) {
                gv_kdtree_destroy_recursive(db->root);
            } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
                if (db->hnsw_index) gv_hnsw_destroy(db->hnsw_index);
            } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
                if (db->hnsw_index) gv_ivfpq_destroy(db->hnsw_index);
            }
            pthread_rwlock_destroy(&db->rwlock);
            pthread_mutex_destroy(&db->wal_mutex);
            free(db);
            return NULL;
        }
    }

    fclose(in);

    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        db->count = file_count;
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        db->count = gv_hnsw_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        db->count = gv_ivfpq_count(db->hnsw_index);
    }

    /* WAL is intentionally disabled for memory-backed snapshots. */
    return db;
}

GV_Database *gv_db_open_mmap(const char *filepath, size_t dimension, GV_IndexType index_type) {
    if (filepath == NULL) {
        return NULL;
    }
    GV_MMap *mm = gv_mmap_open_readonly(filepath);
    if (mm == NULL) {
        return NULL;
    }
    const void *data = gv_mmap_data(mm);
    size_t size = gv_mmap_size(mm);
    if (data == NULL || size == 0) {
        gv_mmap_close(mm);
        return NULL;
    }

    GV_Database *db = gv_db_open_from_memory(data, size, dimension, index_type);
    if (db == NULL) {
        gv_mmap_close(mm);
        return NULL;
    }

    /* Attach mapping to db->filepath so user can still see origin; store pointer via wal_path. */
    /* We reuse wal_path as an opaque holder for the mapping pointer in this special mode. */
    db->wal = NULL;
    db->wal_replaying = 0;
    db->wal_path = (char *)mm; /* opaque handle; freed in close via gv_mmap_close */
    return db;
}

GV_Database *gv_db_open_with_hnsw_config(const char *filepath, size_t dimension, 
                                         GV_IndexType index_type, const GV_HNSWConfig *hnsw_config) {
    if (index_type != GV_INDEX_TYPE_HNSW || filepath != NULL) {
        return gv_db_open(filepath, dimension, index_type);
    }

    if (dimension == 0) {
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
    db->metadata_index = gv_metadata_index_create();
    if (db->metadata_index == NULL) {
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        free(db);
        return NULL;
    }
    /* Initialize compaction fields */
    db->compaction_running = 0;
    pthread_mutex_init(&db->compaction_mutex, NULL);
    pthread_cond_init(&db->compaction_cond, NULL);
    db->compaction_interval_sec = 300;  /* Default: 5 minutes */
    db->wal_compaction_threshold = 10 * 1024 * 1024;  /* Default: 10MB */
    db->deleted_ratio_threshold = 0.1;  /* Default: 10% */
    /* Initialize resource limits */
    db->resource_limits.max_memory_bytes = 0;  /* Unlimited by default */
    db->resource_limits.max_vectors = 0;  /* Unlimited by default */
    db->resource_limits.max_concurrent_operations = 0;  /* Unlimited by default */
    db->current_memory_bytes = 0;
    db->current_concurrent_ops = 0;
    pthread_mutex_init(&db->resource_mutex, NULL);
    /* Initialize observability fields */
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

    if (index_type == GV_INDEX_TYPE_KDTREE || index_type == GV_INDEX_TYPE_HNSW) {
        db->soa_storage = gv_soa_storage_create(dimension, 0);
        if (db->soa_storage == NULL) {
            gv_metadata_index_destroy(db->metadata_index);
            pthread_rwlock_destroy(&db->rwlock);
            pthread_mutex_destroy(&db->wal_mutex);
            free(db);
            return NULL;
        }
    }

    db->hnsw_index = gv_hnsw_create(dimension, hnsw_config, db->soa_storage);
    if (db->hnsw_index == NULL) {
        if (db->soa_storage != NULL) {
            gv_soa_storage_destroy(db->soa_storage);
        }
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        free(db);
        return NULL;
    }

    if (db->wal_path != NULL) {
        db->wal = gv_wal_open(db->wal_path, db->dimension, (uint32_t)db->index_type);
    }

    return db;
}

GV_Database *gv_db_open_with_ivfpq_config(const char *filepath, size_t dimension, 
                                          GV_IndexType index_type, const GV_IVFPQConfig *ivfpq_config) {
    if (index_type != GV_INDEX_TYPE_IVFPQ || filepath != NULL) {
        return gv_db_open(filepath, dimension, index_type);
    }

    if (dimension == 0) {
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
    db->sparse_index = NULL;
    db->sparse_index = NULL;
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
    db->soa_storage = NULL;
    db->metadata_index = gv_metadata_index_create();
    if (db->metadata_index == NULL) {
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        free(db);
        return NULL;
    }
    /* Initialize compaction and resource fields like in gv_db_open */
    db->compaction_running = 0;
    pthread_mutex_init(&db->compaction_mutex, NULL);
    pthread_cond_init(&db->compaction_cond, NULL);
    db->compaction_interval_sec = 300;
    db->wal_compaction_threshold = 10 * 1024 * 1024;
    db->deleted_ratio_threshold = 0.1;
    db->resource_limits.max_memory_bytes = 0;
    db->resource_limits.max_vectors = 0;
    db->resource_limits.max_concurrent_operations = 0;
    db->current_memory_bytes = 0;
    db->current_concurrent_ops = 0;
    pthread_mutex_init(&db->resource_mutex, NULL);
    memset(&db->insert_latency_hist, 0, sizeof(db->insert_latency_hist));
    memset(&db->search_latency_hist, 0, sizeof(db->search_latency_hist));
    pthread_mutex_init(&db->observability_mutex, NULL);

    if (ivfpq_config != NULL) {
        db->hnsw_index = gv_ivfpq_create(dimension, ivfpq_config);
    } else {
        GV_IVFPQConfig default_cfg = {.nlist = 64, .m = 8, .nbits = 8, .nprobe = 4, .train_iters = 15};
        db->hnsw_index = gv_ivfpq_create(dimension, &default_cfg);
    }
    
    if (db->hnsw_index == NULL) {
        gv_metadata_index_destroy(db->metadata_index);
        pthread_mutex_destroy(&db->resource_mutex);
        pthread_mutex_destroy(&db->observability_mutex);
        pthread_cond_destroy(&db->compaction_cond);
        pthread_mutex_destroy(&db->compaction_mutex);
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        free(db);
        return NULL;
    }

    if (db->wal_path != NULL) {
        db->wal = gv_wal_open(db->wal_path, db->dimension, (uint32_t)db->index_type);
    }

    return db;
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

    /* Start timing */
    uint64_t start_time_us = gv_db_get_time_us();

    /* Check resource limits */
    size_t vector_memory = gv_db_estimate_vector_memory(dimension);
    if (gv_db_check_resource_limits(db, 1, vector_memory) != 0) {
        return -1; /* Resource limit exceeded */
    }

    /* Increment concurrent operations */
    gv_db_increment_concurrent_ops(db);

    if (db->wal != NULL && db->wal_replaying == 0) {
        pthread_mutex_lock(&db->wal_mutex);
        int wal_res = gv_wal_append_insert(db->wal, data, dimension, NULL, NULL);
        pthread_mutex_unlock(&db->wal_mutex);
        if (wal_res != 0) {
            gv_db_decrement_concurrent_ops(db);
            return -1;
        }
        db->total_wal_records += 1;
    }

    pthread_rwlock_wrlock(&db->rwlock);
    
    int status = -1;
    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        if (db->soa_storage == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        float *normalized_data = (float *)malloc(dimension * sizeof(float));
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
        size_t vector_index = gv_soa_storage_add(db->soa_storage, normalized_data, NULL);
        free(normalized_data);
        if (vector_index == (size_t)-1) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        status = gv_kdtree_insert(&(db->root), db->soa_storage, vector_index, 0);
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        GV_Vector *vector = gv_vector_create_from_data(dimension, data);
        if (vector == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        if (db->cosine_normalized) {
            gv_db_normalize_vector(vector);
        }
        status = gv_hnsw_insert(db->hnsw_index, vector);
        if (status != 0) {
            gv_vector_destroy(vector);
        }
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        GV_Vector *vector = gv_vector_create_from_data(dimension, data);
        if (vector == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        if (db->cosine_normalized) {
            gv_db_normalize_vector(vector);
        }
        status = gv_ivfpq_insert(db->hnsw_index, vector);
        if (status != 0) {
            gv_vector_destroy(vector);
        }
    }

    if (status != 0) {
        pthread_rwlock_unlock(&db->rwlock);
        gv_db_decrement_concurrent_ops(db);
        return -1;
    }

    db->count += 1;
    db->total_inserts += 1;
    gv_db_update_memory_usage(db);
    pthread_rwlock_unlock(&db->rwlock);
    
    /* Record latency for insert operation */
    uint64_t end_time_us = gv_db_get_time_us();
    uint64_t latency_us = end_time_us - start_time_us;
    gv_db_record_latency(db, latency_us, 1);
    
    gv_db_decrement_concurrent_ops(db);
    return 0;
}

int gv_db_add_vector_with_metadata(GV_Database *db, const float *data, size_t dimension,
                                     const char *metadata_key, const char *metadata_value) {
    if (db == NULL || data == NULL || dimension == 0 || dimension != db->dimension ||
        metadata_key == NULL || metadata_value == NULL) {
        return -1;
    }

    /* Start timing */
    uint64_t start_time_us = gv_db_get_time_us();

    /* Check resource limits */
    size_t vector_memory = gv_db_estimate_vector_memory(dimension);
    if (gv_db_check_resource_limits(db, 1, vector_memory) != 0) {
        return -1; /* Resource limit exceeded */
    }

    /* Increment concurrent operations */
    gv_db_increment_concurrent_ops(db);
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
        db->total_wal_records += 1;
    }

    pthread_rwlock_wrlock(&db->rwlock);
    
    int status = -1;
    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        if (db->soa_storage == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        float *normalized_data = (float *)malloc(dimension * sizeof(float));
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
        if (metadata_key != NULL && metadata_value != NULL) {
            GV_Vector temp_vec;
            temp_vec.dimension = dimension;
            temp_vec.data = NULL;
            temp_vec.metadata = NULL;
            if (gv_vector_set_metadata(&temp_vec, metadata_key, metadata_value) == 0) {
                metadata = temp_vec.metadata;
            }
        }
        size_t vector_index = gv_soa_storage_add(db->soa_storage, normalized_data, metadata);
        free(normalized_data);
        if (vector_index == (size_t)-1) {
            if (metadata != NULL) {
                GV_Vector temp_vec;
                temp_vec.dimension = dimension;
                temp_vec.data = NULL;
                temp_vec.metadata = metadata;
                gv_vector_clear_metadata(&temp_vec);
            }
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        /* Update metadata index */
        if (metadata != NULL && db->metadata_index != NULL) {
            GV_Metadata *current = metadata;
            while (current != NULL) {
                gv_metadata_index_add(db->metadata_index, current->key, current->value, vector_index);
                current = current->next;
            }
        }
        status = gv_kdtree_insert(&(db->root), db->soa_storage, vector_index, 0);
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        GV_Vector *vector = gv_vector_create_from_data(dimension, data);
        if (vector == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        if (db->cosine_normalized) {
            gv_db_normalize_vector(vector);
        }
        if (metadata_key != NULL && metadata_value != NULL) {
            if (gv_vector_set_metadata(vector, metadata_key, metadata_value) != 0) {
                gv_vector_destroy(vector);
                pthread_rwlock_unlock(&db->rwlock);
                return -1;
            }
        }
        status = gv_hnsw_insert(db->hnsw_index, vector);
        if (status == 0 && vector->metadata != NULL && db->metadata_index != NULL) {
            /* Update metadata index - use db->count as vector index */
            size_t vector_index = db->count;
            GV_Metadata *current = vector->metadata;
            while (current != NULL) {
                gv_metadata_index_add(db->metadata_index, current->key, current->value, vector_index);
                current = current->next;
            }
        }
        if (status != 0) {
            gv_vector_destroy(vector);
        }
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        GV_Vector *vector = gv_vector_create_from_data(dimension, data);
        if (vector == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        if (db->cosine_normalized) {
            gv_db_normalize_vector(vector);
        }
        if (metadata_key != NULL && metadata_value != NULL) {
            if (gv_vector_set_metadata(vector, metadata_key, metadata_value) != 0) {
                gv_vector_destroy(vector);
                pthread_rwlock_unlock(&db->rwlock);
                return -1;
            }
        }
        status = gv_ivfpq_insert(db->hnsw_index, vector);
        if (status == 0 && vector->metadata != NULL && db->metadata_index != NULL) {
            /* Update metadata index - use db->count as vector index */
            size_t vector_index = db->count;
            GV_Metadata *current = vector->metadata;
            while (current != NULL) {
                gv_metadata_index_add(db->metadata_index, current->key, current->value, vector_index);
                current = current->next;
            }
        }
        if (status != 0) {
            gv_vector_destroy(vector);
        }
    }

    if (status != 0) {
        pthread_rwlock_unlock(&db->rwlock);
        return -1;
    }

    db->count += 1;
    db->total_inserts += 1;
    pthread_rwlock_unlock(&db->rwlock);
    
    /* Record latency for insert operation */
    uint64_t end_time_us = gv_db_get_time_us();
    uint64_t latency_us = end_time_us - start_time_us;
    gv_db_record_latency(db, latency_us, 1);
    
    gv_db_decrement_concurrent_ops(db);
    return 0;
}

int gv_db_add_sparse_vector(GV_Database *db, const uint32_t *indices, const float *values,
                            size_t nnz, size_t dimension,
                            const char *metadata_key, const char *metadata_value) {
    if (db == NULL || db->index_type != GV_INDEX_TYPE_SPARSE || dimension != db->dimension) {
        return -1;
    }
    if ((indices == NULL || values == NULL) && nnz > 0) {
        return -1;
    }

    pthread_rwlock_wrlock(&db->rwlock);
    GV_SparseVector *sv = gv_sparse_vector_create(dimension, indices, values, nnz);
    if (sv == NULL) {
        pthread_rwlock_unlock(&db->rwlock);
        return -1;
    }
    if (metadata_key && metadata_value) {
        /* Cast sparse vector to vector for metadata operations */
        GV_Vector *vec = (GV_Vector *)sv;
        /* Ensure metadata is NULL (should be from calloc, but be safe) */
        vec->metadata = NULL;
        if (gv_vector_set_metadata(vec, metadata_key, metadata_value) != 0) {
            gv_sparse_vector_destroy(sv);
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
    }

    int status = gv_sparse_index_add(db->sparse_index, sv);
    if (status != 0) {
        gv_sparse_vector_destroy(sv);
        pthread_rwlock_unlock(&db->rwlock);
        return -1;
    }
    db->count += 1;
    db->total_inserts += 1;
    pthread_rwlock_unlock(&db->rwlock);
    return 0;
}

int gv_db_add_vector_with_rich_metadata(GV_Database *db, const float *data, size_t dimension,
                                        const char *const *metadata_keys, const char *const *metadata_values,
                                        size_t metadata_count) {
    if (db == NULL || data == NULL || dimension == 0 || dimension != db->dimension) {
        return -1;
    }
    if (metadata_count > 0 && (metadata_keys == NULL || metadata_values == NULL)) {
        return -1;
    }
    
    /* Record start time for latency measurement */
    uint64_t start_time_us = gv_db_get_time_us();

    if (db->wal != NULL && db->wal_replaying == 0) {
        pthread_mutex_lock(&db->wal_mutex);
        int wal_res = gv_wal_append_insert_rich(db->wal, data, dimension, metadata_keys, metadata_values, metadata_count);
        pthread_mutex_unlock(&db->wal_mutex);
        if (wal_res != 0) {
            return -1;
        }
        db->total_wal_records += 1;
    }

    pthread_rwlock_wrlock(&db->rwlock);
    
    int status = -1;
    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        if (db->soa_storage == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        float *normalized_data = (float *)malloc(dimension * sizeof(float));
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
                    if (gv_vector_set_metadata(&temp_vec, metadata_keys[i], metadata_values[i]) != 0) {
                        gv_vector_clear_metadata(&temp_vec);
                        free(normalized_data);
                        pthread_rwlock_unlock(&db->rwlock);
                        return -1;
                    }
                }
            }
            metadata = temp_vec.metadata;
        }
        size_t vector_index = gv_soa_storage_add(db->soa_storage, normalized_data, metadata);
        free(normalized_data);
        if (vector_index == (size_t)-1) {
            if (metadata != NULL) {
                GV_Vector temp_vec;
                temp_vec.dimension = dimension;
                temp_vec.data = NULL;
                temp_vec.metadata = metadata;
                gv_vector_clear_metadata(&temp_vec);
            }
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        status = gv_kdtree_insert(&(db->root), db->soa_storage, vector_index, 0);
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        GV_Vector *vector = gv_vector_create_from_data(dimension, data);
        if (vector == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        if (db->cosine_normalized) {
            gv_db_normalize_vector(vector);
        }
        for (size_t i = 0; i < metadata_count; i++) {
            if (metadata_keys[i] != NULL && metadata_values[i] != NULL) {
                if (gv_vector_set_metadata(vector, metadata_keys[i], metadata_values[i]) != 0) {
                    gv_vector_destroy(vector);
                    pthread_rwlock_unlock(&db->rwlock);
                    return -1;
                }
            }
        }
        status = gv_hnsw_insert(db->hnsw_index, vector);
        if (status != 0) {
            gv_vector_destroy(vector);
        }
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        GV_Vector *vector = gv_vector_create_from_data(dimension, data);
        if (vector == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        if (db->cosine_normalized) {
            gv_db_normalize_vector(vector);
        }
        for (size_t i = 0; i < metadata_count; i++) {
            if (metadata_keys[i] != NULL && metadata_values[i] != NULL) {
                if (gv_vector_set_metadata(vector, metadata_keys[i], metadata_values[i]) != 0) {
                    gv_vector_destroy(vector);
                    pthread_rwlock_unlock(&db->rwlock);
                    return -1;
                }
            }
        }
        status = gv_ivfpq_insert(db->hnsw_index, vector);
        if (status != 0) {
            gv_vector_destroy(vector);
        }
    }

    if (status != 0) {
        pthread_rwlock_unlock(&db->rwlock);
        gv_db_decrement_concurrent_ops(db);
        return -1;
    }

    db->count += 1;
    db->total_inserts += 1;
    gv_db_update_memory_usage(db);
    pthread_rwlock_unlock(&db->rwlock);
    
    /* Record latency for insert operation */
    uint64_t end_time_us = gv_db_get_time_us();
    uint64_t latency_us = end_time_us - start_time_us;
    gv_db_record_latency(db, latency_us, 1);
    
    gv_db_decrement_concurrent_ops(db);
    return 0;
}

int gv_db_ivfpq_train(GV_Database *db, const float *data, size_t count, size_t dimension) {
    if (db == NULL || data == NULL || count == 0 || dimension != db->dimension) {
        return -1;
    }
    if (db->index_type != GV_INDEX_TYPE_IVFPQ || db->hnsw_index == NULL) {
        return -1;
    }
    return gv_ivfpq_train(db->hnsw_index, data, count);
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
            if (db->soa_storage == NULL) {
                status = -1;
            } else {
                status = gv_kdtree_save_recursive(db->root, db->soa_storage, out, version);
            }
        } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
            status = gv_hnsw_save(db->hnsw_index, out, version);
        } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
            status = gv_ivfpq_save(db->hnsw_index, out, version);
        } else if (db->index_type == GV_INDEX_TYPE_SPARSE) {
            status = gv_sparse_index_save(db->sparse_index, out, version);
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

    if (db->wal != NULL && status == 0) {
        pthread_mutex_lock((pthread_mutex_t *)&db->wal_mutex);
        int truncate_status = gv_wal_truncate(db->wal);
        if (truncate_status == 0) {
            /* Reset WAL record count after successful truncation */
            ((GV_Database *)db)->total_wal_records = 0;
        }
        pthread_mutex_unlock((pthread_mutex_t *)&db->wal_mutex);
    } else if (db->wal_path != NULL && status == 0) {
        /* Fallback: if WAL handle is NULL but path exists, use reset */
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

    /* Start timing */
    uint64_t start_time_us = gv_db_get_time_us();

    /* ensure result fields start clean */
    memset(results, 0, k * sizeof(GV_SearchResult));

    pthread_rwlock_rdlock((pthread_rwlock_t *)&db->rwlock);
    ((GV_Database *)db)->total_queries += 1;

    if (db->index_type == GV_INDEX_TYPE_KDTREE && db->root == NULL) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        uint64_t end_time_us = gv_db_get_time_us();
        uint64_t latency_us = end_time_us - start_time_us;
        gv_db_record_latency((GV_Database *)db, latency_us, 0);
        return 0;
    }
    if (db->index_type == GV_INDEX_TYPE_HNSW && db->hnsw_index == NULL) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        uint64_t end_time_us = gv_db_get_time_us();
        uint64_t latency_us = end_time_us - start_time_us;
        gv_db_record_latency((GV_Database *)db, latency_us, 0);
        return 0;
    }
    if (db->index_type == GV_INDEX_TYPE_IVFPQ && db->hnsw_index == NULL) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        uint64_t end_time_us = gv_db_get_time_us();
        uint64_t latency_us = end_time_us - start_time_us;
        gv_db_record_latency((GV_Database *)db, latency_us, 0);
        return 0;
    }

    GV_Vector query_vec;
    query_vec.dimension = db->dimension;
    query_vec.data = (float *)query_data;
    query_vec.metadata = NULL;

    int use_exact = 0;
    if (db->exact_search_threshold > 0 && db->count <= db->exact_search_threshold) {
        use_exact = 1;
    }
    if (db->force_exact_search) {
        use_exact = 1;
    }

    if (db->index_type == GV_INDEX_TYPE_KDTREE && use_exact) {
        if (db->soa_storage == NULL) {
            pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
            uint64_t end_time_us = gv_db_get_time_us();
            uint64_t latency_us = end_time_us - start_time_us;
            gv_db_record_latency((GV_Database *)db, latency_us, 0);
            return -1;
        }
        int r = gv_exact_knn_search_kdtree(db->root, db->soa_storage, db->count, &query_vec, k, results, distance_type);
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        uint64_t end_time_us = gv_db_get_time_us();
        uint64_t latency_us = end_time_us - start_time_us;
        gv_db_record_latency((GV_Database *)db, latency_us, 0);
        return r;
    }

    int r = -1;
    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        if (db->soa_storage == NULL) {
            pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
            uint64_t end_time_us = gv_db_get_time_us();
            uint64_t latency_us = end_time_us - start_time_us;
            gv_db_record_latency((GV_Database *)db, latency_us, 0);
            return -1;
        }
        r = gv_kdtree_knn_search(db->root, db->soa_storage, &query_vec, k, results, distance_type);
        /* Note: KD-tree search now copies vectors internally, so results are valid after return */
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        r = gv_hnsw_search(db->hnsw_index, &query_vec, k, results, distance_type, NULL, NULL);
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        r = gv_ivfpq_search(db->hnsw_index, &query_vec, k, results, distance_type, 0, 0);
    }
    pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
    
    /* Record latency */
    uint64_t end_time_us = gv_db_get_time_us();
    uint64_t latency_us = end_time_us - start_time_us;
    gv_db_record_latency((GV_Database *)db, latency_us, 0);
    
    return r;
}

int gv_db_search_ivfpq_opts(const GV_Database *db, const float *query_data, size_t k,
                            GV_SearchResult *results, GV_DistanceType distance_type,
                            size_t nprobe_override, size_t rerank_top) {
    (void)nprobe_override;
    (void)rerank_top;
    // For now, delegate to regular search
    return gv_db_search(db, query_data, k, results, distance_type);
}

int gv_db_search_batch(const GV_Database *db, const float *queries, size_t qcount, size_t k,
                       GV_SearchResult *results, GV_DistanceType distance_type) {
    if (db == NULL || queries == NULL || results == NULL || qcount == 0 || k == 0) {
        return -1;
    }
    pthread_rwlock_rdlock((pthread_rwlock_t *)&db->rwlock);
    ((GV_Database *)db)->total_queries += 1;
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
            if (db->soa_storage == NULL) {
                r = -1;
            } else {
                r = gv_kdtree_knn_search(db->root, db->soa_storage, &qv, k, slot, distance_type);
            }
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

    memset(results, 0, k * sizeof(GV_SearchResult));

    pthread_rwlock_rdlock((pthread_rwlock_t *)&db->rwlock);
    ((GV_Database *)db)->total_queries += 1;

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
        if (db->soa_storage == NULL) {
            pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
            return -1;
        }
        int r = gv_kdtree_knn_search_filtered(db->root, db->soa_storage, &query_vec, k, results, distance_type,
                                            filter_key, filter_value);
        /* Note: KD-tree filtered search now copies vectors internally */
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

int gv_db_search_with_filter_expr(const GV_Database *db, const float *query_data, size_t k,
                                  GV_SearchResult *results, GV_DistanceType distance_type,
                                  const char *filter_expr) {
    if (db == NULL || query_data == NULL || results == NULL || k == 0 || filter_expr == NULL) {
        return -1;
    }

    GV_Filter *filter = gv_filter_parse(filter_expr);
    if (filter == NULL) {
        return -1;
    }

    pthread_rwlock_rdlock((pthread_rwlock_t *)&db->rwlock);
    ((GV_Database *)db)->total_queries += 1;

    if (db->index_type == GV_INDEX_TYPE_KDTREE && db->root == NULL) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        gv_filter_destroy(filter);
        return 0;
    }
    if (db->index_type == GV_INDEX_TYPE_HNSW && db->hnsw_index == NULL) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        gv_filter_destroy(filter);
        return 0;
    }
    if (db->index_type == GV_INDEX_TYPE_IVFPQ && db->hnsw_index == NULL) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        gv_filter_destroy(filter);
        return 0;
    }

    size_t max_candidates = k * 4;
    if (max_candidates < k) {
        max_candidates = k;
    }
    if (max_candidates > db->count) {
        max_candidates = db->count;
    }
    if (max_candidates == 0) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        gv_filter_destroy(filter);
        return 0;
    }

    GV_SearchResult *tmp = (GV_SearchResult *)malloc(max_candidates * sizeof(GV_SearchResult));
    if (!tmp) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        gv_filter_destroy(filter);
        return -1;
    }

    GV_Vector query_vec;
    query_vec.dimension = db->dimension;
    query_vec.data = (float *)query_data;
    query_vec.metadata = NULL;

    int n = 0;
    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        if (db->soa_storage == NULL) {
            n = -1;
        } else {
            n = gv_kdtree_knn_search(db->root, db->soa_storage, &query_vec, max_candidates, tmp, distance_type);
        }
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        n = gv_hnsw_search(db->hnsw_index, &query_vec, max_candidates, tmp, distance_type, NULL, NULL);
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        n = gv_ivfpq_search(db->hnsw_index, &query_vec, max_candidates, tmp, distance_type, 0, 0);
    } else {
        free(tmp);
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        gv_filter_destroy(filter);
        return -1;
    }

    if (n <= 0) {
        free(tmp);
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        gv_filter_destroy(filter);
        return n;
    }

    size_t out = 0;
    for (int i = 0; i < n && out < k; ++i) {
        int match = gv_filter_eval(filter, tmp[i].vector);
        if (match < 0) {
            free(tmp);
            pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
            gv_filter_destroy(filter);
            return -1;
        }
        if (match == 1) {
            results[out++] = tmp[i];
        }
    }

    free(tmp);
    pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
    gv_filter_destroy(filter);
    return (int)out;
}

void gv_db_set_exact_search_threshold(GV_Database *db, size_t threshold) {
    if (db == NULL) {
        return;
    }
    db->exact_search_threshold = threshold;
}

void gv_db_set_force_exact_search(GV_Database *db, int enabled) {
    if (db == NULL) {
        return;
    }
    db->force_exact_search = enabled ? 1 : 0;
}

int gv_db_search_sparse(const GV_Database *db, const uint32_t *indices, const float *values,
                        size_t nnz, size_t k, GV_SearchResult *results, GV_DistanceType distance_type) {
    if (db == NULL || db->index_type != GV_INDEX_TYPE_SPARSE || results == NULL || k == 0) {
        return -1;
    }
    if ((indices == NULL || values == NULL) && nnz > 0) {
        return -1;
    }
    ((GV_Database *)db)->total_queries += 1;
    GV_SparseVector *query = gv_sparse_vector_create(db->dimension, indices, values, nnz);
    if (query == NULL) {
        return -1;
    }
    int r = gv_sparse_index_search(db->sparse_index, query, k, results, distance_type);
    gv_sparse_vector_destroy(query);
    return r;
}


int gv_db_range_search(const GV_Database *db, const float *query_data, float radius,
                       GV_SearchResult *results, size_t max_results, GV_DistanceType distance_type) {
    if (db == NULL || query_data == NULL || results == NULL || max_results == 0 || radius < 0.0f) {
        return -1;
    }

    memset(results, 0, max_results * sizeof(GV_SearchResult));

    pthread_rwlock_rdlock((pthread_rwlock_t *)&db->rwlock);
    ((GV_Database *)db)->total_range_queries += 1;

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

    GV_Vector query_vec;
    query_vec.dimension = db->dimension;
    query_vec.data = (float *)query_data;
    query_vec.metadata = NULL;

    int r;
    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        if (db->soa_storage == NULL) {
            pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
            return -1;
        }
        r = gv_kdtree_range_search(db->root, db->soa_storage, &query_vec, radius, results, max_results, distance_type);
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        r = gv_hnsw_range_search(db->hnsw_index, &query_vec, radius, results, max_results, distance_type, NULL, NULL);
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        r = gv_ivfpq_range_search(db->hnsw_index, &query_vec, radius, results, max_results, distance_type);
    } else {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return -1;
    }
    
    pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
    return r;
}

int gv_db_range_search_filtered(const GV_Database *db, const float *query_data, float radius,
                                 GV_SearchResult *results, size_t max_results,
                                 GV_DistanceType distance_type,
                                 const char *filter_key, const char *filter_value) {
    if (db == NULL || query_data == NULL || results == NULL || max_results == 0 || radius < 0.0f) {
        return -1;
    }

    memset(results, 0, max_results * sizeof(GV_SearchResult));

    pthread_rwlock_rdlock((pthread_rwlock_t *)&db->rwlock);
    ((GV_Database *)db)->total_range_queries += 1;

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

    GV_Vector query_vec;
    query_vec.dimension = db->dimension;
    query_vec.data = (float *)query_data;
    query_vec.metadata = NULL;

    int r;
    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        if (db->soa_storage == NULL) {
            pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
            return -1;
        }
        r = gv_kdtree_range_search_filtered(db->root, db->soa_storage, &query_vec, radius, results, max_results,
                                            distance_type, filter_key, filter_value);
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        r = gv_hnsw_range_search(db->hnsw_index, &query_vec, radius, results, max_results,
                                distance_type, filter_key, filter_value);
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        r = gv_ivfpq_range_search(db->hnsw_index, &query_vec, radius, results, max_results, distance_type);
        if (r > 0 && filter_key != NULL) {
            int out = 0;
            for (int i = 0; i < r && out < (int)max_results; ++i) {
                const char *val = gv_vector_get_metadata(results[i].vector, filter_key);
                if (val && strcmp(val, filter_value) == 0) {
                    if (out != i) {
                        results[out] = results[i];
                    }
                    out++;
                }
            }
            r = out;
        }
    } else {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return -1;
    }
    
    pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
    return r;
}

int gv_db_delete_vector_by_index(GV_Database *db, size_t vector_index) {
    if (db == NULL) {
        return -1;
    }

    pthread_rwlock_wrlock(&db->rwlock);

    int status = -1;
    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        if (db->soa_storage == NULL || vector_index >= db->soa_storage->count) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        status = gv_kdtree_delete(&(db->root), db->soa_storage, vector_index);
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        if (db->hnsw_index == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        status = gv_hnsw_delete(db->hnsw_index, vector_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        if (db->hnsw_index == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        status = gv_ivfpq_delete(db->hnsw_index, vector_index);
    } else if (db->index_type == GV_INDEX_TYPE_SPARSE) {
        if (db->sparse_index == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        status = gv_sparse_index_delete(db->sparse_index, vector_index);
    } else {
        pthread_rwlock_unlock(&db->rwlock);
        return -1;
    }

    if (status == 0) {
        /* Remove from metadata index */
        if (db->metadata_index != NULL) {
            gv_metadata_index_remove_vector(db->metadata_index, vector_index);
        }
        /* Update WAL */
        if (db->wal != NULL) {
            if (gv_wal_append_delete(db->wal, vector_index) != 0) {
                pthread_rwlock_unlock(&db->rwlock);
                return -1;
            }
            db->total_wal_records += 1;
        }
    }

    pthread_rwlock_unlock(&db->rwlock);
    return status;
}

int gv_db_update_vector(GV_Database *db, size_t vector_index, const float *new_data, size_t dimension) {
    if (db == NULL || new_data == NULL || dimension != db->dimension) {
        return -1;
    }

    pthread_rwlock_wrlock(&db->rwlock);

    int status = -1;
    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        if (db->soa_storage == NULL || vector_index >= db->soa_storage->count) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        if (gv_soa_storage_is_deleted(db->soa_storage, vector_index) == 1) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        if (db->filepath == NULL) {
            // In-memory KDTREE: update only SOA storage
            status = gv_soa_storage_update_data(db->soa_storage, vector_index, new_data);
        } else {
            // File-based KDTREE: update tree and SOA
            status = gv_kdtree_update(&(db->root), db->soa_storage, vector_index, new_data);
        }
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        if (db->hnsw_index == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        status = gv_hnsw_update(db->hnsw_index, vector_index, new_data, dimension);
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        if (db->hnsw_index == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        status = gv_ivfpq_update(db->hnsw_index, vector_index, new_data, dimension);
    } else if (db->index_type == GV_INDEX_TYPE_SPARSE) {
        pthread_rwlock_unlock(&db->rwlock);
        return -1; /* Sparse vectors require special handling - use gv_db_update_sparse_vector */
    } else {
        pthread_rwlock_unlock(&db->rwlock);
        return -1;
    }

    if (status == 0 && db->wal != NULL) {
        /* Write update record to WAL with empty metadata (data-only update) */
        if (gv_wal_append_update(db->wal, vector_index, new_data, dimension, NULL, NULL, 0) != 0) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        db->total_wal_records += 1;
    }

    pthread_rwlock_unlock(&db->rwlock);
    return status;
}

int gv_db_update_vector_metadata(GV_Database *db, size_t vector_index,
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
        if (gv_soa_storage_is_deleted(db->soa_storage, vector_index) == 1) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        vector_data = gv_soa_storage_get_data(db->soa_storage, vector_index);
        if (vector_data == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        
        /* Create new metadata chain by building a temporary vector */
        GV_Vector temp_vec;
        temp_vec.dimension = db->dimension;
        temp_vec.data = NULL;
        temp_vec.metadata = NULL;
        
        for (size_t i = 0; i < metadata_count; ++i) {
            if (metadata_keys[i] != NULL && metadata_values[i] != NULL) {
                if (gv_vector_set_metadata(&temp_vec, metadata_keys[i], metadata_values[i]) != 0) {
                    gv_vector_clear_metadata(&temp_vec);
                    pthread_rwlock_unlock(&db->rwlock);
                    return -1;
                }
            }
        }
        
        /* Get old metadata and copy keys/values before updating (update will free old metadata) */
        GV_Metadata *old_metadata = gv_soa_storage_get_metadata(db->soa_storage, vector_index);
        GV_Metadata *old_metadata_copy = NULL;
        if (old_metadata != NULL && db->metadata_index != NULL) {
            /* Copy old metadata chain to avoid use-after-free */
            GV_Metadata *current = old_metadata;
            GV_Metadata *prev = NULL;
            while (current != NULL) {
                GV_Metadata *copy = (GV_Metadata *)malloc(sizeof(GV_Metadata));
                if (copy == NULL) {
                    /* Free what we've copied so far */
                    while (old_metadata_copy != NULL) {
                        GV_Metadata *next = old_metadata_copy->next;
                        free(old_metadata_copy->key);
                        free(old_metadata_copy->value);
                        free(old_metadata_copy);
                        old_metadata_copy = next;
                    }
                    gv_vector_clear_metadata(&temp_vec);
                    pthread_rwlock_unlock(&db->rwlock);
                    return -1;
                }
                copy->key = current->key ? strdup(current->key) : NULL;
                copy->value = current->value ? strdup(current->value) : NULL;
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
        
        status = gv_soa_storage_update_metadata(db->soa_storage, vector_index, temp_vec.metadata);
        // Ownership transferred to SOA, prevent double-free
        temp_vec.metadata = NULL;

        /* Update metadata index */
        if (status == 0 && db->metadata_index != NULL) {
            gv_metadata_index_update(db->metadata_index, vector_index, old_metadata_copy, temp_vec.metadata);
        }
        
        /* Free the copied old metadata */
        while (old_metadata_copy != NULL) {
            GV_Metadata *next = old_metadata_copy->next;
            free(old_metadata_copy->key);
            free(old_metadata_copy->value);
            free(old_metadata_copy);
            old_metadata_copy = next;
        }
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        if (db->hnsw_index == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        /* For HNSW, we need to get the vector and update its metadata */
        /* This is a simplified approach - in practice, HNSW should also use SoA */
        pthread_rwlock_unlock(&db->rwlock);
        return -1; /* Not fully implemented for HNSW without SoA */
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        if (db->hnsw_index == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        /* Similar to HNSW - would need vector access */
        pthread_rwlock_unlock(&db->rwlock);
        return -1; /* Not fully implemented for IVFPQ */
    } else if (db->index_type == GV_INDEX_TYPE_SPARSE) {
        pthread_rwlock_unlock(&db->rwlock);
        return -1; /* Sparse vectors require special handling */
    } else {
        pthread_rwlock_unlock(&db->rwlock);
        return -1;
    }

    if (status == 0 && db->wal != NULL && vector_data != NULL) {
        /* Write update record to WAL with metadata */
        if (gv_wal_append_update(db->wal, vector_index, vector_data, db->dimension, 
                                 metadata_keys, metadata_values, metadata_count) != 0) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        db->total_wal_records += 1;
    }

    pthread_rwlock_unlock(&db->rwlock);
    return status;
}

/* Background compaction implementation */

/**
 * @brief Compact SoA storage by removing deleted vectors.
 *
 * This function compacts the SoA storage arrays by removing all deleted vectors
 * and updating vector indices in the indexes.
 */
static int gv_db_compact_soa_storage(GV_Database *db) {
    if (db == NULL || db->soa_storage == NULL) {
        return -1;
    }

    GV_SoAStorage *storage = db->soa_storage;
    size_t dimension = storage->dimension;
    
    /* Count deleted vectors */
    size_t deleted_count = 0;
    for (size_t i = 0; i < storage->count; ++i) {
        if (storage->deleted[i] != 0) {
            deleted_count++;
        }
    }

    if (deleted_count == 0) {
        return 0; /* Nothing to compact */
    }

    /* Create new compacted arrays */
    size_t new_count = storage->count - deleted_count;
    float *new_data = (float *)malloc(new_count * dimension * sizeof(float));
    GV_Metadata **new_metadata = (GV_Metadata **)calloc(new_count, sizeof(GV_Metadata *));
    int *new_deleted = (int *)calloc(new_count, sizeof(int));
    
    if (new_data == NULL || new_metadata == NULL || new_deleted == NULL) {
        free(new_data);
        free(new_metadata);
        free(new_deleted);
        return -1;
    }

    /* Build mapping from old index to new index */
    size_t *index_map = (size_t *)malloc(storage->count * sizeof(size_t));
    if (index_map == NULL) {
        free(new_data);
        free(new_metadata);
        free(new_deleted);
        return -1;
    }

    size_t new_idx = 0;
    for (size_t old_idx = 0; old_idx < storage->count; ++old_idx) {
        if (storage->deleted[old_idx] == 0) {
            /* Copy vector data */
            memcpy(new_data + (new_idx * dimension),
                   storage->data + (old_idx * dimension),
                   dimension * sizeof(float));
            new_metadata[new_idx] = storage->metadata[old_idx];
            storage->metadata[old_idx] = NULL; /* Transfer ownership */
            new_deleted[new_idx] = 0;
            index_map[old_idx] = new_idx;
            new_idx++;
        } else {
            /* Free metadata for deleted vectors */
            if (storage->metadata[old_idx] != NULL) {
                GV_Vector temp_vec = {
                    .dimension = dimension,
                    .data = NULL,
                    .metadata = storage->metadata[old_idx]
                };
                gv_vector_clear_metadata(&temp_vec);
            }
            index_map[old_idx] = (size_t)-1; /* Mark as deleted */
        }
    }

    /* Free old arrays */
    free(storage->data);
    free(storage->metadata);
    free(storage->deleted);

    /* Update storage */
    storage->data = new_data;
    storage->metadata = new_metadata;
    storage->deleted = new_deleted;
    storage->count = new_count;
    storage->capacity = new_count; /* Shrink to fit */

    /* Rebuild indexes with new indices */
    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        /* Rebuild KD-tree */
        GV_KDNode *old_root = db->root;
        db->root = NULL;
        for (size_t i = 0; i < new_count; ++i) {
            gv_kdtree_insert(&(db->root), storage, i, 0);
        }
        gv_kdtree_destroy_recursive(old_root);
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        /* Full HNSW rebuild: destroy old index and rebuild from compacted storage */
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
            
            /* Extract config from existing index before destroying */
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
            
            /* Destroy old index (but SoA storage is shared, so it won't be destroyed) */
            gv_hnsw_destroy(db->hnsw_index);
            db->hnsw_index = NULL;
            
            /* Create new HNSW index with same config and existing SoA storage */
            db->hnsw_index = gv_hnsw_create(dimension, &config, storage);
            if (db->hnsw_index == NULL) {
                return -1; /* Failed to create new index */
            }
            
            /* Re-insert all vectors from compacted storage */
            for (size_t i = 0; i < new_count; ++i) {
                GV_Vector temp_vec = {
                    .dimension = dimension,
                    .data = storage->data + (i * dimension),
                    .metadata = storage->metadata[i]
                };
                
                /* Insert vector (metadata ownership transferred) */
                if (gv_hnsw_insert(db->hnsw_index, &temp_vec) != 0) {
                    /* On failure, clean up */
                    gv_hnsw_destroy(db->hnsw_index);
                    db->hnsw_index = NULL;
                    return -1;
                }
                
                /* Clear metadata pointer since ownership was transferred */
                temp_vec.metadata = NULL;
            }
        }
    }

    /* Update metadata index */
    if (db->metadata_index != NULL) {
        /* Rebuild metadata index with new indices */
        GV_MetadataIndex *old_index = db->metadata_index;
        db->metadata_index = gv_metadata_index_create();
        if (db->metadata_index != NULL) {
            for (size_t i = 0; i < new_count; ++i) {
                GV_Metadata *meta = storage->metadata[i];
                if (meta != NULL) {
                    GV_Metadata *current = meta;
                    while (current != NULL) {
                        gv_metadata_index_add(db->metadata_index, current->key, current->value, i);
                        current = current->next;
                    }
                }
            }
            gv_metadata_index_destroy(old_index);
        }
    }

    free(index_map);
    return 0;
}

/**
 * @brief Compact WAL if it exceeds threshold.
 */
static int gv_db_compact_wal(GV_Database *db) {
    if (db == NULL || db->wal == NULL || db->filepath == NULL) {
        return 0; /* No WAL to compact */
    }

    /* Check WAL file size */
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

    /* Save database to trigger WAL truncation */
    /* This will create a new snapshot and truncate the WAL */
    char *temp_path = (char *)malloc(strlen(db->filepath) + 10);
    if (temp_path == NULL) {
        return -1;
    }
    snprintf(temp_path, strlen(db->filepath) + 10, "%s.tmp", db->filepath);

    int save_result = gv_db_save(db, temp_path);
    if (save_result == 0) {
        /* Replace original file */
        if (rename(temp_path, db->filepath) == 0) {
            /* Truncate WAL */
            if (db->wal != NULL) {
                gv_wal_truncate(db->wal);
            }
        } else {
            unlink(temp_path);
        }
    } else {
        unlink(temp_path);
    }

    free(temp_path);
    return 0;
}

/**
 * @brief Main compaction function.
 */
int gv_db_compact(GV_Database *db) {
    if (db == NULL) {
        return -1;
    }

    pthread_rwlock_wrlock(&db->rwlock);

    /* Check if compaction is needed */
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
            gv_db_compact_soa_storage(db);
        }
    }

    /* Compact WAL if needed */
    gv_db_compact_wal(db);

    pthread_rwlock_unlock(&db->rwlock);
    return 0;
}

/**
 * @brief Background compaction thread function.
 */
static void *gv_db_compaction_thread(void *arg) {
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
            /* Time to run compaction */
            gv_db_compact(db);
        }
    }

    pthread_mutex_unlock(&db->compaction_mutex);
    return NULL;
}

/**
 * @brief Start background compaction thread.
 */
int gv_db_start_background_compaction(GV_Database *db) {
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
                                gv_db_compaction_thread, db);

    pthread_mutex_unlock(&db->compaction_mutex);

    if (result != 0) {
        db->compaction_running = 0;
        return -1;
    }

    return 0;
}

/**
 * @brief Stop background compaction thread.
 */
void gv_db_stop_background_compaction(GV_Database *db) {
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

    /* Wait for thread to finish */
    pthread_join(db->compaction_thread, NULL);
}

/**
 * @brief Set compaction interval in seconds.
 */
void gv_db_set_compaction_interval(GV_Database *db, size_t interval_sec) {
    if (db == NULL) {
        return;
    }
    pthread_mutex_lock(&db->compaction_mutex);
    db->compaction_interval_sec = interval_sec;
    pthread_cond_signal(&db->compaction_cond); /* Wake up thread to check new interval */
    pthread_mutex_unlock(&db->compaction_mutex);
}

/**
 * @brief Set WAL compaction threshold in bytes.
 */
void gv_db_set_wal_compaction_threshold(GV_Database *db, size_t threshold_bytes) {
    if (db == NULL) {
        return;
    }
    db->wal_compaction_threshold = threshold_bytes;
}

/**
 * @brief Set deleted vector ratio threshold for triggering compaction.
 */
void gv_db_set_deleted_ratio_threshold(GV_Database *db, double ratio) {
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

/* Resource limits implementation */

/**
 * @brief Estimate memory usage for a vector.
 */
static size_t gv_db_estimate_vector_memory(size_t dimension) {
    /* Vector data: dimension * sizeof(float) */
    size_t data_size = dimension * sizeof(float);
    /* Metadata overhead: assume average 64 bytes per vector */
    size_t metadata_overhead = 64;
    /* SoA storage overhead: deleted flag, metadata pointer */
    size_t storage_overhead = sizeof(int) + sizeof(void *);
    return data_size + metadata_overhead + storage_overhead;
}

/**
 * @brief Update memory usage estimate.
 */
static void gv_db_update_memory_usage(GV_Database *db) {
    if (db == NULL) {
        return;
    }

    pthread_mutex_lock(&db->resource_mutex);

    size_t total_memory = 0;

    /* Base database structure */
    total_memory += sizeof(GV_Database);

    /* SoA storage */
    if (db->soa_storage != NULL) {
        size_t vector_memory = gv_db_estimate_vector_memory(db->soa_storage->dimension);
        total_memory += sizeof(GV_SoAStorage);
        total_memory += db->soa_storage->count * vector_memory;
        total_memory += db->soa_storage->capacity * (sizeof(GV_Metadata *) + sizeof(int));
    }

    /* Index memory (rough estimate) */
    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        /* KD-tree: roughly 3 pointers per node */
        total_memory += db->count * (sizeof(GV_KDNode) + sizeof(size_t));
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        /* HNSW: estimate based on node count and connections */
        total_memory += db->count * (sizeof(void *) * 2); /* Rough estimate */
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        /* IVFPQ: estimate based on entries */
        total_memory += db->count * (sizeof(void *) * 2); /* Rough estimate */
    }

    /* Metadata index */
    if (db->metadata_index != NULL) {
        /* Rough estimate: 100 bytes per metadata entry */
        total_memory += db->count * 100;
    }

    /* WAL (if exists) */
    if (db->wal != NULL && db->wal_path != NULL) {
        FILE *wal_file = fopen(db->wal_path, "rb");
        if (wal_file != NULL) {
            if (fseek(wal_file, 0, SEEK_END) == 0) {
                long wal_size = ftell(wal_file);
                if (wal_size > 0) {
                    total_memory += (size_t)wal_size;
                }
            }
            fclose(wal_file);
        }
    }

    db->current_memory_bytes = total_memory;

    pthread_mutex_unlock(&db->resource_mutex);
}

/**
 * @brief Check if resource limits would be exceeded.
 */
static int gv_db_check_resource_limits(GV_Database *db, size_t additional_vectors, size_t additional_memory) {
    if (db == NULL) {
        return -1;
    }

    pthread_mutex_lock(&db->resource_mutex);

    /* Check vector count limit */
    if (db->resource_limits.max_vectors > 0) {
        if (db->count + additional_vectors > db->resource_limits.max_vectors) {
            pthread_mutex_unlock(&db->resource_mutex);
            return -1; /* Limit exceeded */
        }
    }

    /* Check memory limit */
    if (db->resource_limits.max_memory_bytes > 0) {
        size_t estimated_new_memory = db->current_memory_bytes + additional_memory;
        if (estimated_new_memory > db->resource_limits.max_memory_bytes) {
            pthread_mutex_unlock(&db->resource_mutex);
            return -1; /* Limit exceeded */
        }
    }

    /* Check concurrent operations limit */
    if (db->resource_limits.max_concurrent_operations > 0) {
        if (db->current_concurrent_ops >= db->resource_limits.max_concurrent_operations) {
            pthread_mutex_unlock(&db->resource_mutex);
            return -1; /* Limit exceeded */
        }
    }

    pthread_mutex_unlock(&db->resource_mutex);
    return 0;
}

/**
 * @brief Increment concurrent operations counter.
 */
static void gv_db_increment_concurrent_ops(GV_Database *db) {
    if (db == NULL) {
        return;
    }
    pthread_mutex_lock(&db->resource_mutex);
    db->current_concurrent_ops++;
    pthread_mutex_unlock(&db->resource_mutex);
}

/**
 * @brief Decrement concurrent operations counter.
 */
static void gv_db_decrement_concurrent_ops(GV_Database *db) {
    if (db == NULL) {
        return;
    }
    pthread_mutex_lock(&db->resource_mutex);
    if (db->current_concurrent_ops > 0) {
        db->current_concurrent_ops--;
    }
    pthread_mutex_unlock(&db->resource_mutex);
}

/**
 * @brief Set resource limits for the database.
 */
int gv_db_set_resource_limits(GV_Database *db, const GV_ResourceLimits *limits) {
    if (db == NULL || limits == NULL) {
        return -1;
    }

    pthread_mutex_lock(&db->resource_mutex);
    db->resource_limits.max_memory_bytes = limits->max_memory_bytes;
    db->resource_limits.max_vectors = limits->max_vectors;
    db->resource_limits.max_concurrent_operations = limits->max_concurrent_operations;
    pthread_mutex_unlock(&db->resource_mutex);

    /* Update memory usage estimate */
    gv_db_update_memory_usage(db);

    return 0;
}

/**
 * @brief Get current resource limits.
 */
void gv_db_get_resource_limits(const GV_Database *db, GV_ResourceLimits *limits) {
    if (db == NULL || limits == NULL) {
        return;
    }

    pthread_mutex_lock((pthread_mutex_t *)&db->resource_mutex);
    limits->max_memory_bytes = db->resource_limits.max_memory_bytes;
    limits->max_vectors = db->resource_limits.max_vectors;
    limits->max_concurrent_operations = db->resource_limits.max_concurrent_operations;
    pthread_mutex_unlock((pthread_mutex_t *)&db->resource_mutex);
}

/**
 * @brief Get current estimated memory usage in bytes.
 */
size_t gv_db_get_memory_usage(const GV_Database *db) {
    if (db == NULL) {
        return 0;
    }

    gv_db_update_memory_usage((GV_Database *)db);

    pthread_mutex_lock((pthread_mutex_t *)&db->resource_mutex);
    size_t usage = db->current_memory_bytes;
    pthread_mutex_unlock((pthread_mutex_t *)&db->resource_mutex);

    return usage;
}

/**
 * @brief Get current number of concurrent operations.
 */
size_t gv_db_get_concurrent_operations(const GV_Database *db) {
    if (db == NULL) {
        return 0;
    }

    pthread_mutex_lock((pthread_mutex_t *)&db->resource_mutex);
    size_t ops = db->current_concurrent_ops;
    pthread_mutex_unlock((pthread_mutex_t *)&db->resource_mutex);

    return ops;
}

/* Observability implementation */

/**
 * @brief Get current time in microseconds.
 */
static uint64_t gv_db_get_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
}

/**
 * @brief Initialize latency histogram.
 */
static int gv_db_init_latency_histogram(GV_LatencyHistogram *hist, size_t bucket_count) {
    if (hist == NULL || bucket_count == 0) {
        return -1;
    }

    hist->bucket_count = bucket_count;
    hist->buckets = (uint64_t *)calloc(bucket_count, sizeof(uint64_t));
    hist->bucket_boundaries = (double *)malloc(bucket_count * sizeof(double));
    
    if (hist->buckets == NULL || hist->bucket_boundaries == NULL) {
        free(hist->buckets);
        free(hist->bucket_boundaries);
        return -1;
    }

    /* Create logarithmic buckets: 1us, 10us, 100us, 1ms, 10ms, 100ms, 1s, 10s */
    for (size_t i = 0; i < bucket_count; ++i) {
        hist->bucket_boundaries[i] = pow(10.0, (double)(i + 1));
    }

    hist->total_samples = 0;
    hist->sum_latency_us = 0;
    return 0;
}

/**
 * @brief Add sample to latency histogram.
 */
static void gv_db_add_latency_sample(GV_LatencyHistogram *hist, uint64_t latency_us) {
    if (hist == NULL || hist->buckets == NULL) {
        return;
    }

    hist->total_samples++;
    hist->sum_latency_us += latency_us;

    /* Find appropriate bucket */
    for (size_t i = 0; i < hist->bucket_count; ++i) {
        if (latency_us <= (uint64_t)hist->bucket_boundaries[i]) {
            hist->buckets[i]++;
            return;
        }
    }
    /* If exceeds all buckets, add to last bucket */
    if (hist->bucket_count > 0) {
        hist->buckets[hist->bucket_count - 1]++;
    }
}

/**
 * @brief Calculate percentile from histogram.
 */




/**
 * @brief Record latency for an operation.
 */
void gv_db_record_latency(GV_Database *db, uint64_t latency_us, int is_insert) {
    if (db == NULL) {
        return;
    }

    pthread_mutex_lock(&db->observability_mutex);

    /* Initialize histogram if not already initialized */
    if (is_insert && db->insert_latency_hist.buckets == NULL) {
        gv_db_init_latency_histogram(&db->insert_latency_hist, 8);
    } else if (!is_insert && db->search_latency_hist.buckets == NULL) {
        gv_db_init_latency_histogram(&db->search_latency_hist, 8);
    }

    if (is_insert) {
        gv_db_add_latency_sample(&db->insert_latency_hist, latency_us);
        db->insert_count_since_update++;
        
        /* Calculate IPS immediately for inserts */
        uint64_t now_us = gv_db_get_time_us();
        if (db->first_insert_time_us == 0) {
            /* Track first insert time for precise calculation - set once and never reset */
            db->first_insert_time_us = now_us;
        }
        if (db->last_ips_update_time_us == 0) {
            db->last_ips_update_time_us = now_us;
        }
        
        /* Always calculate IPS when we have inserts - calculate on every insert */
        /* This ensures current_ips is always up-to-date with the actual high rate */
        if (db->insert_count_since_update > 0) {
            uint64_t elapsed_us = now_us - db->last_ips_update_time_us;
            /* Use actual elapsed time, but ensure it's at least 100 microseconds to avoid division issues */
            uint64_t effective_elapsed_us = (elapsed_us > 100) ? elapsed_us : 100;
            double elapsed_sec = (double)effective_elapsed_us / 1000000.0;
            double calculated_ips = (double)db->insert_count_since_update / elapsed_sec;
            /* Always update current_ips with the actual calculated rate - preserve the high rate */
            /* This ensures we keep the precise rate even after counts reset */
            db->current_ips = calculated_ips;
        }
        
        /* Reset insert counts if more than 1 second has elapsed */
        uint64_t elapsed_us = now_us - db->last_ips_update_time_us;
        if (elapsed_us >= 1000000) {
            /* Preserve current_ips - it represents the last calculated high rate */
            /* Only reset counts, preserve last_ips_update_time_us for precise fallback calculation */
            /* CRITICAL: Never reset first_insert_time_us - it's needed for precise IPS calculation */
            db->insert_count_since_update = 0;
            /* Don't reset last_ips_update_time_us - keep it for precise IPS calculation from total_inserts */
            /* This allows us to calculate precise IPS using actual elapsed time */
        }
    } else {
        gv_db_add_latency_sample(&db->search_latency_hist, latency_us);
        db->query_count_since_update++;
        
        /* Update QPS periodically for queries */
        uint64_t now_us = gv_db_get_time_us();
        if (db->last_qps_update_time_us == 0) {
            db->last_qps_update_time_us = now_us;
        } else {
            uint64_t elapsed_us = now_us - db->last_qps_update_time_us;
            if (elapsed_us >= 1000000) { /* Update every second */
                double elapsed_sec = (double)elapsed_us / 1000000.0;
                if (db->query_count_since_update > 0) {
                    db->current_qps = (double)db->query_count_since_update / elapsed_sec;
                } else if (db->current_qps > 0.0) {
                    /* Decay QPS gradually if no new queries */
                    db->current_qps *= 0.5;
                }
                db->query_count_since_update = 0;
                db->last_qps_update_time_us = now_us;
                /* Don't reset insert counts or IPS timestamp - they're tracked separately */
            }
        }
    }

    pthread_mutex_unlock(&db->observability_mutex);
}

/**
 * @brief Record recall for a search operation.
 */
void gv_db_record_recall(GV_Database *db, double recall) {
    if (db == NULL || recall < 0.0 || recall > 1.0) {
        return;
    }

    pthread_mutex_lock(&db->observability_mutex);

    db->recall_metrics.total_queries++;
    
    /* Update average recall */
    double total = db->recall_metrics.avg_recall * (double)(db->recall_metrics.total_queries - 1) + recall;
    db->recall_metrics.avg_recall = total / (double)db->recall_metrics.total_queries;

    /* Update min/max */
    if (db->recall_metrics.total_queries == 1) {
        db->recall_metrics.min_recall = recall;
        db->recall_metrics.max_recall = recall;
    } else {
        if (recall < db->recall_metrics.min_recall) {
            db->recall_metrics.min_recall = recall;
        }
        if (recall > db->recall_metrics.max_recall) {
            db->recall_metrics.max_recall = recall;
        }
    }

    pthread_mutex_unlock(&db->observability_mutex);
}

/**
 * @brief Get detailed statistics for the database.
 */
int gv_db_get_detailed_stats(const GV_Database *db, GV_DetailedStats *out) {
    if (db == NULL || out == NULL) {
        return -1;
    }

    memset(out, 0, sizeof(GV_DetailedStats));

    pthread_rwlock_rdlock((pthread_rwlock_t *)&db->rwlock);
    pthread_mutex_lock((pthread_mutex_t *)&db->observability_mutex);

    /* Basic stats */
    out->basic_stats.total_inserts = db->total_inserts;
    out->basic_stats.total_queries = db->total_queries;
    out->basic_stats.total_range_queries = db->total_range_queries;
    out->basic_stats.total_wal_records = db->total_wal_records;

    /* Copy latency histograms */
    GV_Database *db_nonconst = (GV_Database *)db;
    if (db_nonconst->insert_latency_hist.buckets != NULL && db_nonconst->insert_latency_hist.bucket_count > 0) {
        size_t count = db_nonconst->insert_latency_hist.bucket_count;
        out->insert_latency.bucket_count = count;
        out->insert_latency.buckets = (uint64_t *)malloc(count * sizeof(uint64_t));
        out->insert_latency.bucket_boundaries = (double *)malloc(count * sizeof(double));
        if (out->insert_latency.buckets != NULL && out->insert_latency.bucket_boundaries != NULL) {
            memcpy(out->insert_latency.buckets, db_nonconst->insert_latency_hist.buckets,
                   count * sizeof(uint64_t));
            memcpy(out->insert_latency.bucket_boundaries, db_nonconst->insert_latency_hist.bucket_boundaries,
                   count * sizeof(double));
            out->insert_latency.total_samples = db_nonconst->insert_latency_hist.total_samples;
            out->insert_latency.sum_latency_us = db_nonconst->insert_latency_hist.sum_latency_us;
        }
    }

    if (db_nonconst->search_latency_hist.buckets != NULL && db_nonconst->search_latency_hist.bucket_count > 0) {
        size_t count = db_nonconst->search_latency_hist.bucket_count;
        out->search_latency.bucket_count = count;
        out->search_latency.buckets = (uint64_t *)malloc(count * sizeof(uint64_t));
        out->search_latency.bucket_boundaries = (double *)malloc(count * sizeof(double));
        if (out->search_latency.buckets != NULL && out->search_latency.bucket_boundaries != NULL) {
            memcpy(out->search_latency.buckets, db_nonconst->search_latency_hist.buckets,
                   count * sizeof(uint64_t));
            memcpy(out->search_latency.bucket_boundaries, db_nonconst->search_latency_hist.bucket_boundaries,
                   count * sizeof(double));
            out->search_latency.total_samples = db_nonconst->search_latency_hist.total_samples;
            out->search_latency.sum_latency_us = db_nonconst->search_latency_hist.sum_latency_us;
        }
    }

    /* QPS and IPS - calculate on-demand based on current counts and elapsed time */
    uint64_t now_us = gv_db_get_time_us();
    
    /* Calculate QPS */
    if (db->last_qps_update_time_us == 0) {
        out->queries_per_second = 0.0;
    } else {
        uint64_t elapsed_us = now_us - db->last_qps_update_time_us;
        if (elapsed_us == 0) elapsed_us = 1; /* Avoid division by zero */
        double elapsed_sec = (double)elapsed_us / 1000000.0;
        
        /* Always try to calculate from current counts first if available */
        if (db->query_count_since_update > 0 && elapsed_us < 5000000) {
            /* Calculate instantaneous rate from current counts (within 5 seconds) */
            out->queries_per_second = (double)db->query_count_since_update / elapsed_sec;
        } else if (elapsed_us < 2000000 && db->current_qps > 0.0) {
            /* Within 2 seconds, use stored QPS (still recent) */
            out->queries_per_second = db->current_qps;
        } else {
            /* Use stored QPS (may decay to 0 if no recent operations) */
            out->queries_per_second = db->current_qps;
        }
    }
    
    /* Calculate IPS - always use actual data for precise calculation */
    /* This is independent of QPS, so calculate it separately */
    {
        /* Priority 1: Use first_insert_time_us if available (most accurate) */
        if (db->total_inserts > 0 && db->first_insert_time_us > 0) {
            uint64_t total_elapsed_us = now_us - db->first_insert_time_us;
            if (total_elapsed_us > 0) {
                double total_elapsed_sec = (double)total_elapsed_us / 1000000.0;
                /* Use actual elapsed time for precise calculation - no minimum threshold */
                /* Even if very small, use actual time for accuracy */
                if (total_elapsed_sec > 0.0) {
                    double precise_ips = (double)db->total_inserts / total_elapsed_sec;
                    out->inserts_per_second = precise_ips;
                    ((GV_Database *)db)->current_ips = precise_ips;
                } else {
                    /* Should not happen, but use 1ms minimum */
                    double precise_ips = (double)db->total_inserts / 0.001;
                    out->inserts_per_second = precise_ips;
                    ((GV_Database *)db)->current_ips = precise_ips;
                }
            } else {
                /* Should not happen, but use 1ms minimum */
                double precise_ips = (double)db->total_inserts / 0.001;
                out->inserts_per_second = precise_ips;
                ((GV_Database *)db)->current_ips = precise_ips;
            }
        }
        /* Priority 2: Use current counts if available */
        else if (db->last_ips_update_time_us > 0 && db->insert_count_since_update > 0) {
            uint64_t insert_elapsed_us = now_us - db->last_ips_update_time_us;
            if (insert_elapsed_us == 0) insert_elapsed_us = 1;
            uint64_t effective_elapsed_us = insert_elapsed_us > 100 ? insert_elapsed_us : 100;
            double effective_elapsed_sec = (double)effective_elapsed_us / 1000000.0;
            double calculated_ips = (double)db->insert_count_since_update / effective_elapsed_sec;
            out->inserts_per_second = calculated_ips;
            ((GV_Database *)db)->current_ips = calculated_ips;
        }
        /* Priority 3: Use stored current_ips */
        else if (db->current_ips > 0.0) {
            out->inserts_per_second = db->current_ips;
        }
        /* Priority 4: Fallback - calculate from total_inserts using last_ips_update_time_us */
        /* This handles cases where first_insert_time_us wasn't set but last_ips_update_time_us was */
        else if (db->total_inserts > 0 && db->last_ips_update_time_us > 0) {
            uint64_t total_elapsed_us = now_us - db->last_ips_update_time_us;
            if (total_elapsed_us > 0) {
                double total_elapsed_sec = (double)total_elapsed_us / 1000000.0;
                /* Use actual elapsed time for precise calculation */
                if (total_elapsed_sec > 0.0) {
                    double precise_ips = (double)db->total_inserts / total_elapsed_sec;
                    out->inserts_per_second = precise_ips;
                    ((GV_Database *)db)->current_ips = precise_ips;
                    /* Also set first_insert_time_us for future use if not already set */
                    if (db->first_insert_time_us == 0) {
                        ((GV_Database *)db)->first_insert_time_us = db->last_ips_update_time_us;
                    }
                } else {
                    double precise_ips = (double)db->total_inserts / 0.001;
                    out->inserts_per_second = precise_ips;
                    ((GV_Database *)db)->current_ips = precise_ips;
                }
            } else {
                out->inserts_per_second = 0.0;
            }
        }
        /* Priority 5: For reopened databases, if we have inserts but no timing, show 0 */
        /* (Historical inserts can't be timed accurately after database reopen) */
        else if (db->total_inserts > 0) {
            /* Historical inserts - cannot calculate precise rate without timing data */
            /* Show 0.00 to indicate no recent insert activity */
            out->inserts_per_second = 0.0;
        }
        /* No data available */
        else {
            out->inserts_per_second = 0.0;
        }
    }
    out->last_qps_update_time = db->last_qps_update_time_us;

    /* Memory breakdown */
    gv_db_update_memory_usage((GV_Database *)db);
    out->memory.total_bytes = db->current_memory_bytes;
    
    if (db->soa_storage != NULL) {
        size_t vector_memory = gv_db_estimate_vector_memory(db->soa_storage->dimension);
        out->memory.soa_storage_bytes = sizeof(GV_SoAStorage) +
            db->soa_storage->count * vector_memory +
            db->soa_storage->capacity * (sizeof(GV_Metadata *) + sizeof(int));
    }

    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        out->memory.index_bytes = db->count * (sizeof(void *) * 3); /* Rough estimate */
    } else if (db->index_type == GV_INDEX_TYPE_HNSW || db->index_type == GV_INDEX_TYPE_IVFPQ) {
        out->memory.index_bytes = db->count * (sizeof(void *) * 2); /* Rough estimate */
    }

    if (db->metadata_index != NULL) {
        out->memory.metadata_index_bytes = db->count * 100; /* Rough estimate */
    }

    if (db->wal_path != NULL) {
        FILE *wal_file = fopen(db->wal_path, "rb");
        if (wal_file != NULL) {
            if (fseek(wal_file, 0, SEEK_END) == 0) {
                long wal_size = ftell(wal_file);
                if (wal_size > 0) {
                    out->memory.wal_bytes = (size_t)wal_size;
                }
            }
            fclose(wal_file);
        }
    }

    /* Recall metrics */
    out->recall.total_queries = db_nonconst->recall_metrics.total_queries;
    out->recall.avg_recall = db_nonconst->recall_metrics.avg_recall;
    out->recall.min_recall = db_nonconst->recall_metrics.min_recall;
    out->recall.max_recall = db_nonconst->recall_metrics.max_recall;

    /* Health indicators */
    if (db->soa_storage != NULL) {
        size_t deleted_count = 0;
        for (size_t i = 0; i < db->soa_storage->count; ++i) {
            if (db->soa_storage->deleted[i] != 0) {
                deleted_count++;
            }
        }
        out->deleted_vector_count = deleted_count;
        out->deleted_ratio = (db->soa_storage->count > 0) ?
            (double)deleted_count / (double)db->soa_storage->count : 0.0;
    }

    /* Determine health status */
    if (out->deleted_ratio > 0.5) {
        out->health_status = -2; /* Unhealthy */
    } else if (out->deleted_ratio > 0.2) {
        out->health_status = -1; /* Degraded */
    } else {
        out->health_status = 0; /* Healthy */
    }

    pthread_mutex_unlock((pthread_mutex_t *)&db->observability_mutex);
    pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);

    return 0;
}

/**
 * @brief Free resources allocated by gv_db_get_detailed_stats().
 */
void gv_db_free_detailed_stats(GV_DetailedStats *stats) {
    if (stats == NULL) {
        return;
    }

    if (stats->insert_latency.buckets != NULL) {
        free(stats->insert_latency.buckets);
        stats->insert_latency.buckets = NULL;
    }
    if (stats->insert_latency.bucket_boundaries != NULL) {
        free(stats->insert_latency.bucket_boundaries);
        stats->insert_latency.bucket_boundaries = NULL;
    }
    if (stats->search_latency.buckets != NULL) {
        free(stats->search_latency.buckets);
        stats->search_latency.buckets = NULL;
    }
    if (stats->search_latency.bucket_boundaries != NULL) {
        free(stats->search_latency.bucket_boundaries);
        stats->search_latency.bucket_boundaries = NULL;
    }
}

/**
 * @brief Perform health check on the database.
 */
int gv_db_health_check(const GV_Database *db) {
    if (db == NULL) {
        return -2; /* Unhealthy */
    }

    pthread_rwlock_rdlock((pthread_rwlock_t *)&db->rwlock);

    int health = 0; /* Healthy by default */

    /* Check deleted vector ratio */
    if (db->soa_storage != NULL) {
        size_t deleted_count = 0;
        for (size_t i = 0; i < db->soa_storage->count; ++i) {
            if (db->soa_storage->deleted[i] != 0) {
                deleted_count++;
            }
        }
        double deleted_ratio = (db->soa_storage->count > 0) ?
            (double)deleted_count / (double)db->soa_storage->count : 0.0;

        if (deleted_ratio > 0.5) {
            health = -2; /* Unhealthy */
        } else if (deleted_ratio > 0.2) {
            health = -1; /* Degraded */
        }
    }

    /* Check resource limits */
    if (db->resource_limits.max_memory_bytes > 0) {
        gv_db_update_memory_usage((GV_Database *)db);
        if (db->current_memory_bytes > db->resource_limits.max_memory_bytes) {
            health = -1; /* Degraded */
        }
    }

    if (db->resource_limits.max_vectors > 0) {
        if (db->count > db->resource_limits.max_vectors) {
            health = -1; /* Degraded */
        }
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);

    return health;
}
