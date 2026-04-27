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
#ifndef ssize_t
typedef SSIZE_T ssize_t;
#endif
static FILE *fmemopen(void *buf, size_t size, const char *mode) {
    (void)mode;
    FILE *tmp = tmpfile();
    if (!tmp) return NULL;
    if (fwrite(buf, 1, size, tmp) != size) { fclose(tmp); return NULL; }
    rewind(tmp);
    return tmp;
}
static ssize_t getline(char **lineptr, size_t *n, FILE *stream) {
    if (!lineptr || !n || !stream) return -1;
    int c;
    size_t len = 0;
    if (!*lineptr || *n == 0) {
        *n = 128;
        *lineptr = (char *)malloc(*n);
        if (!*lineptr) return -1;
    }
    while ((c = fgetc(stream)) != EOF) {
        if (len + 2 > *n) {
            size_t newn = *n * 2;
            char *tmp = (char *)realloc(*lineptr, newn);
            if (!tmp) return -1;
            *lineptr = tmp;
            *n = newn;
        }
        (*lineptr)[len++] = (char)c;
        if (c == '\n') break;
    }
    if (len == 0) return -1;
    (*lineptr)[len] = '\0';
    return (ssize_t)len;
}
#endif

#include "core/types.h"

#include "storage/database.h"
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
#include "index/pq.h"
#include "index/lsh.h"
#include "core/utils.h"

#include <math.h>
#ifndef _WIN32
#include <sys/time.h>
#endif
#include "core/compat.h"

static size_t db_estimate_vector_memory(size_t dimension);
static void db_update_memory_usage(GV_Database *db);
static int db_check_resource_limits(GV_Database *db, size_t additional_vectors, size_t additional_memory);
static void db_increment_concurrent_ops(GV_Database *db);
static void db_decrement_concurrent_ops(GV_Database *db);
static uint64_t db_get_time_us(void);

static void db_init_common_fields(GV_Database *db) {
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
    pthread_mutex_init(&db->resource_mutex, NULL);
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
}

static int db_write_header(FILE *out, uint32_t dimension, uint64_t count, uint32_t version) {
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

static int db_read_header(FILE *in, uint32_t *dimension_out, uint64_t *count_out, uint32_t *version_out) {
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

static int write_uint32(FILE *out, uint32_t value) {
    return (fwrite(&value, sizeof(uint32_t), 1, out) == 1) ? 0 : -1;
}

static int read_uint32(FILE *in, uint32_t *value) {
    return (value != NULL && fread(value, sizeof(uint32_t), 1, in) == 1) ? 0 : -1;
}


static char *db_build_wal_path(const char *filepath) {
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

GV_IndexType index_suggest(size_t dimension, size_t expected_count) {
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

static void db_normalize_vector(GV_Vector *vector) {
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

void db_set_cosine_normalized(GV_Database *db, int enabled) {
    if (db == NULL) {
        return;
    }
    db->cosine_normalized = enabled ? 1 : 0;
}

void db_get_stats(const GV_Database *db, GV_DBStats *out) {
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

GV_Database *db_open(const char *filepath, size_t dimension, GV_IndexType index_type) {
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
    db->metadata_index = metadata_index_create();
    if (db->metadata_index == NULL) {
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        free(db);
        return NULL;
    }
    db_init_common_fields(db);

    if (index_type == GV_INDEX_TYPE_KDTREE || index_type == GV_INDEX_TYPE_HNSW ||
        index_type == GV_INDEX_TYPE_FLAT || index_type == GV_INDEX_TYPE_LSH) {
        db->soa_storage = soa_storage_create(dimension, 0);
        if (db->soa_storage == NULL) {
            metadata_index_destroy(db->metadata_index);
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
                soa_storage_destroy(db->soa_storage);
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
        db->sparse_index = sparse_index_create(dimension);
        if (db->sparse_index == NULL) {
            pthread_rwlock_destroy(&db->rwlock);
            pthread_mutex_destroy(&db->wal_mutex);
            free(db);
            return NULL;
        }
    } else if (index_type == GV_INDEX_TYPE_FLAT && filepath == NULL) {
        db->hnsw_index = flat_create(dimension, NULL, db->soa_storage);
        if (db->hnsw_index == NULL) {
            if (db->soa_storage != NULL) {
                soa_storage_destroy(db->soa_storage);
            }
            pthread_rwlock_destroy(&db->rwlock);
            pthread_mutex_destroy(&db->wal_mutex);
            free(db);
            return NULL;
        }
    } else if (index_type == GV_INDEX_TYPE_IVFFLAT && filepath == NULL) {
        GV_IVFFlatConfig cfg = {.nlist = 64, .nprobe = 4, .train_iters = 15, .use_cosine = 0};
        db->hnsw_index = ivfflat_create(dimension, &cfg);
        if (db->hnsw_index == NULL) {
            pthread_rwlock_destroy(&db->rwlock);
            pthread_mutex_destroy(&db->wal_mutex);
            free(db);
            return NULL;
        }
    } else if (index_type == GV_INDEX_TYPE_PQ && filepath == NULL) {
        GV_PQConfig cfg = {.m = 8, .nbits = 8, .train_iters = 15};
        db->hnsw_index = pq_create(dimension, &cfg);
        if (db->hnsw_index == NULL) {
            pthread_rwlock_destroy(&db->rwlock);
            pthread_mutex_destroy(&db->wal_mutex);
            free(db);
            return NULL;
        }
    } else if (index_type == GV_INDEX_TYPE_LSH && filepath == NULL) {
        GV_LSHConfig cfg = {.num_tables = 8, .num_hash_bits = 16, .seed = 42};
        db->hnsw_index = lsh_create(dimension, &cfg, db->soa_storage);
        if (db->hnsw_index == NULL) {
            if (db->soa_storage != NULL) {
                soa_storage_destroy(db->soa_storage);
            }
            pthread_rwlock_destroy(&db->rwlock);
            pthread_mutex_destroy(&db->wal_mutex);
            free(db);
            return NULL;
        }
    }

    if (filepath != NULL) {
        db->filepath = gv_dup_cstr(filepath);
        if (db->filepath == NULL) {
            metadata_index_destroy(db->metadata_index);
            pthread_mutex_destroy(&db->compaction_mutex);
            pthread_cond_destroy(&db->compaction_cond);
            pthread_mutex_destroy(&db->resource_mutex);
            pthread_mutex_destroy(&db->observability_mutex);
            pthread_rwlock_destroy(&db->rwlock);
            pthread_mutex_destroy(&db->wal_mutex);
            if (db->soa_storage) soa_storage_destroy(db->soa_storage);
            free(db);
            return NULL;
        }

        db->wal_path = db_build_wal_path(filepath);
        if (db->wal_path == NULL) {
            metadata_index_destroy(db->metadata_index);
            pthread_mutex_destroy(&db->compaction_mutex);
            pthread_cond_destroy(&db->compaction_cond);
            pthread_mutex_destroy(&db->resource_mutex);
            pthread_mutex_destroy(&db->observability_mutex);
            pthread_rwlock_destroy(&db->rwlock);
            pthread_mutex_destroy(&db->wal_mutex);
            if (db->soa_storage) soa_storage_destroy(db->soa_storage);
            free(db->filepath);
            free(db);
            return NULL;
        }
    }

    if (filepath == NULL) {
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
                    free(db->filepath);
                    free(db->wal_path);
                    free(db);
                    return NULL;
                }
            } else if (index_type == GV_INDEX_TYPE_FLAT) {
                db->hnsw_index = flat_create(dimension, NULL, db->soa_storage);
                if (db->hnsw_index == NULL) {
                    free(db->filepath);
                    free(db->wal_path);
                    free(db);
                    return NULL;
                }
            } else if (index_type == GV_INDEX_TYPE_IVFFLAT) {
                GV_IVFFlatConfig cfg = {.nlist = 64, .nprobe = 4, .train_iters = 15, .use_cosine = 0};
                db->hnsw_index = ivfflat_create(dimension, &cfg);
                if (db->hnsw_index == NULL) {
                    free(db->filepath);
                    free(db->wal_path);
                    free(db);
                    return NULL;
                }
            } else if (index_type == GV_INDEX_TYPE_PQ) {
                GV_PQConfig cfg = {.m = 8, .nbits = 8, .train_iters = 15};
                db->hnsw_index = pq_create(dimension, &cfg);
                if (db->hnsw_index == NULL) {
                    free(db->filepath);
                    free(db->wal_path);
                    free(db);
                    return NULL;
                }
            } else if (index_type == GV_INDEX_TYPE_LSH) {
                GV_LSHConfig cfg = {.num_tables = 8, .num_hash_bits = 16, .seed = 42};
                db->hnsw_index = lsh_create(dimension, &cfg, db->soa_storage);
                if (db->hnsw_index == NULL) {
                    free(db->filepath);
                    free(db->wal_path);
                    free(db);
                    return NULL;
                }
            } else if (index_type == GV_INDEX_TYPE_IVFPQ) {
                GV_IVFPQConfig cfg = {.nlist = 64, .m = 8, .nbits = 8, .nprobe = 4, .train_iters = 15};
                db->hnsw_index = gv_ivfpq_create(dimension, &cfg);
                if (db->hnsw_index == NULL) {
                    free(db->filepath);
                    free(db->wal_path);
                    free(db);
                    return NULL;
                }
            }
            if (db->wal_path != NULL) {
                db->wal = wal_open(db->wal_path, db->dimension, (uint32_t)db->index_type);
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
    if (db_read_header(in, &file_dim, &file_count, &file_version) != 0) {
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
        if (read_uint32(in, &file_index_type) != 0) {
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
        if (kdtree_load_recursive(&(db->root), db->soa_storage, in, db->dimension, file_version) != 0) {
            fclose(in);
            kdtree_destroy_recursive(db->root);
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
        if (sparse_index_load(&loaded_index, in, db->dimension, (size_t)file_count, file_version) != 0) {
            fclose(in);
            free(db->filepath);
            free(db->wal_path);
            free(db);
            return NULL;
        }
        db->sparse_index = loaded_index;
        db->count = (size_t)file_count;
    } else if (db->index_type == GV_INDEX_TYPE_FLAT) {
        void *loaded_index = NULL;
        if (flat_load(&loaded_index, in, db->dimension, file_version) != 0) {
            fclose(in);
            if (loaded_index) flat_destroy(loaded_index);
            free(db->filepath);
            free(db->wal_path);
            free(db);
            return NULL;
        }
        db->hnsw_index = loaded_index;
        db->count = flat_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFFLAT) {
        void *loaded_index = NULL;
        if (ivfflat_load(&loaded_index, in, db->dimension, file_version) != 0) {
            fclose(in);
            if (loaded_index) ivfflat_destroy(loaded_index);
            free(db->filepath);
            free(db->wal_path);
            free(db);
            return NULL;
        }
        db->hnsw_index = loaded_index;
        db->count = ivfflat_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_PQ) {
        void *loaded_index = NULL;
        if (pq_load(&loaded_index, in, db->dimension, file_version) != 0) {
            fclose(in);
            if (loaded_index) pq_destroy(loaded_index);
            free(db->filepath);
            free(db->wal_path);
            free(db);
            return NULL;
        }
        db->hnsw_index = loaded_index;
        db->count = pq_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_LSH) {
        void *loaded_index = NULL;
        if (lsh_load(&loaded_index, in, db->dimension, file_version) != 0) {
            fclose(in);
            if (loaded_index) lsh_destroy(loaded_index);
            free(db->filepath);
            free(db->wal_path);
            free(db);
            return NULL;
        }
        db->hnsw_index = loaded_index;
        db->count = lsh_count(db->hnsw_index);
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
        if (read_uint32(in, &stored_crc) != 0) {
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
    } else if (db->index_type == GV_INDEX_TYPE_FLAT) {
        db->count = flat_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFFLAT) {
        db->count = ivfflat_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_PQ) {
        db->count = pq_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_LSH) {
        db->count = lsh_count(db->hnsw_index);
    }

    if (db->wal_path != NULL) {
        db->wal = wal_open(db->wal_path, db->dimension, (uint32_t)db->index_type);
        if (db->wal == NULL) {
            if (db->index_type == GV_INDEX_TYPE_KDTREE) {
                kdtree_destroy_recursive(db->root);
            } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
                if (db->hnsw_index) gv_hnsw_destroy(db->hnsw_index);
            }
            free(db->filepath);
            free(db->wal_path);
            free(db);
            return NULL;
        }

        db->wal_replaying = 1;
        if (wal_replay_rich(db->wal_path, db->dimension, db_wal_apply_rich,
                            db_wal_apply_delete, db_wal_apply_update,
                            db, (uint32_t)db->index_type) != 0) {
            db->wal_replaying = 0;
            wal_close(db->wal);
            db->wal = NULL;
            if (db->index_type == GV_INDEX_TYPE_KDTREE) {
                kdtree_destroy_recursive(db->root);
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
        kdtree_destroy_recursive(db->root);
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        if (db->hnsw_index) gv_hnsw_destroy(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        if (db->hnsw_index) gv_ivfpq_destroy(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_FLAT) {
        if (db->hnsw_index) flat_destroy(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFFLAT) {
        if (db->hnsw_index) ivfflat_destroy(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_PQ) {
        if (db->hnsw_index) pq_destroy(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_LSH) {
        if (db->hnsw_index) lsh_destroy(db->hnsw_index);
    }
    if (db->metadata_index) metadata_index_destroy(db->metadata_index);
    if (db->soa_storage) soa_storage_destroy(db->soa_storage);
    pthread_rwlock_destroy(&db->rwlock);
    pthread_mutex_destroy(&db->wal_mutex);
    pthread_mutex_destroy(&db->compaction_mutex);
    pthread_cond_destroy(&db->compaction_cond);
    pthread_mutex_destroy(&db->resource_mutex);
    pthread_mutex_destroy(&db->observability_mutex);
    free(db->filepath);
    free(db->wal_path);
    free(db);
    return NULL;
}

void db_close(GV_Database *db) {
    if (db == NULL) {
        return;
    }

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
    } else if (db->index_type == GV_INDEX_TYPE_PQ) {
        pq_destroy(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_LSH) {
        lsh_destroy(db->hnsw_index);
    }
    if (db->soa_storage != NULL) {
        soa_storage_destroy(db->soa_storage);
    }
    if (db->metadata_index != NULL) {
        metadata_index_destroy(db->metadata_index);
    }
    if (db->compaction_running) {
        db_stop_background_compaction(db);
    }
    pthread_mutex_destroy(&db->compaction_mutex);
    pthread_cond_destroy(&db->compaction_cond);
    pthread_mutex_destroy(&db->resource_mutex);
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

GV_Database *db_open_from_memory(const void *data, size_t size,
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
    db->metadata_index = metadata_index_create();
    if (db->metadata_index == NULL) {
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        free(db);
        return NULL;
    }
    db_init_common_fields(db);

    if (index_type == GV_INDEX_TYPE_KDTREE || index_type == GV_INDEX_TYPE_HNSW ||
        index_type == GV_INDEX_TYPE_FLAT || index_type == GV_INDEX_TYPE_LSH) {
        db->soa_storage = soa_storage_create(dimension, 0);
        if (db->soa_storage == NULL) {
            metadata_index_destroy(db->metadata_index);
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
    if (db_read_header(in, &file_dim, &file_count, &file_version) != 0) {
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
        if (read_uint32(in, &file_index_type) != 0) {
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
        if (kdtree_load_recursive(&(db->root), db->soa_storage, in, db->dimension, file_version) != 0) {
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
        if (sparse_index_load(&loaded_index, in, db->dimension, (size_t)file_count, file_version) != 0) {
            fclose(in);
            pthread_rwlock_destroy(&db->rwlock);
            pthread_mutex_destroy(&db->wal_mutex);
            free(db);
            return NULL;
        }
        db->sparse_index = loaded_index;
        db->count = (size_t)file_count;
    } else if (db->index_type == GV_INDEX_TYPE_FLAT) {
        void *loaded_index = NULL;
        if (flat_load(&loaded_index, in, db->dimension, file_version) != 0) {
            fclose(in);
            if (loaded_index) flat_destroy(loaded_index);
            pthread_rwlock_destroy(&db->rwlock);
            pthread_mutex_destroy(&db->wal_mutex);
            free(db);
            return NULL;
        }
        db->hnsw_index = loaded_index;
        db->count = flat_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFFLAT) {
        void *loaded_index = NULL;
        if (ivfflat_load(&loaded_index, in, db->dimension, file_version) != 0) {
            fclose(in);
            if (loaded_index) ivfflat_destroy(loaded_index);
            pthread_rwlock_destroy(&db->rwlock);
            pthread_mutex_destroy(&db->wal_mutex);
            free(db);
            return NULL;
        }
        db->hnsw_index = loaded_index;
        db->count = ivfflat_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_PQ) {
        void *loaded_index = NULL;
        if (pq_load(&loaded_index, in, db->dimension, file_version) != 0) {
            fclose(in);
            if (loaded_index) pq_destroy(loaded_index);
            pthread_rwlock_destroy(&db->rwlock);
            pthread_mutex_destroy(&db->wal_mutex);
            free(db);
            return NULL;
        }
        db->hnsw_index = loaded_index;
        db->count = pq_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_LSH) {
        void *loaded_index = NULL;
        if (lsh_load(&loaded_index, in, db->dimension, file_version) != 0) {
            fclose(in);
            if (loaded_index) lsh_destroy(loaded_index);
            pthread_rwlock_destroy(&db->rwlock);
            pthread_mutex_destroy(&db->wal_mutex);
            free(db);
            return NULL;
        }
        db->hnsw_index = loaded_index;
        db->count = lsh_count(db->hnsw_index);
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
                kdtree_destroy_recursive(db->root);
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
                kdtree_destroy_recursive(db->root);
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
                kdtree_destroy_recursive(db->root);
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
        if (read_uint32(in, &stored_crc) != 0) {
            fclose(in);
            if (db->index_type == GV_INDEX_TYPE_KDTREE) {
                kdtree_destroy_recursive(db->root);
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
                kdtree_destroy_recursive(db->root);
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
                    kdtree_destroy_recursive(db->root);
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
                kdtree_destroy_recursive(db->root);
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
    } else if (db->index_type == GV_INDEX_TYPE_FLAT) {
        db->count = flat_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFFLAT) {
        db->count = ivfflat_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_PQ) {
        db->count = pq_count(db->hnsw_index);
    } else if (db->index_type == GV_INDEX_TYPE_LSH) {
        db->count = lsh_count(db->hnsw_index);
    }

    /* WAL is intentionally disabled for memory-backed snapshots. */
    return db;
}

GV_Database *db_open_mmap(const char *filepath, size_t dimension, GV_IndexType index_type) {
    if (filepath == NULL) {
        return NULL;
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

    GV_Database *db = db_open_from_memory(data, size, dimension, index_type);
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
    if (index_type != GV_INDEX_TYPE_HNSW || filepath != NULL) {
        return db_open(filepath, dimension, index_type);
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
    db->metadata_index = metadata_index_create();
    if (db->metadata_index == NULL) {
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        free(db);
        return NULL;
    }
    db_init_common_fields(db);

    if (index_type == GV_INDEX_TYPE_KDTREE || index_type == GV_INDEX_TYPE_HNSW) {
        db->soa_storage = soa_storage_create(dimension, 0);
        if (db->soa_storage == NULL) {
            metadata_index_destroy(db->metadata_index);
            pthread_rwlock_destroy(&db->rwlock);
            pthread_mutex_destroy(&db->wal_mutex);
            free(db);
            return NULL;
        }
    }

    db->hnsw_index = gv_hnsw_create(dimension, hnsw_config, db->soa_storage);
    if (db->hnsw_index == NULL) {
        if (db->soa_storage != NULL) {
            soa_storage_destroy(db->soa_storage);
        }
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        free(db);
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
    db->soa_storage = NULL;
    db->metadata_index = metadata_index_create();
    if (db->metadata_index == NULL) {
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        free(db);
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
        pthread_mutex_destroy(&db->observability_mutex);
        pthread_cond_destroy(&db->compaction_cond);
        pthread_mutex_destroy(&db->compaction_mutex);
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        free(db);
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
        free(db);
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
        pthread_mutex_destroy(&db->observability_mutex);
        pthread_cond_destroy(&db->compaction_cond);
        pthread_mutex_destroy(&db->compaction_mutex);
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        free(db);
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
        free(db);
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
        pthread_mutex_destroy(&db->observability_mutex);
        pthread_cond_destroy(&db->compaction_cond);
        pthread_mutex_destroy(&db->compaction_mutex);
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        free(db);
        return NULL;
    }

    return db;
}

GV_Database *db_open_with_lsh_config(const char *filepath, size_t dimension,
                                         GV_IndexType index_type, const GV_LSHConfig *config) {
    if (index_type != GV_INDEX_TYPE_LSH) {
        return db_open(filepath, dimension, index_type);
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
        free(db);
        return NULL;
    }
    db_init_common_fields(db);

    db->soa_storage = soa_storage_create(dimension, 0);
    if (db->soa_storage == NULL) {
        metadata_index_destroy(db->metadata_index);
        pthread_mutex_destroy(&db->resource_mutex);
        pthread_mutex_destroy(&db->observability_mutex);
        pthread_cond_destroy(&db->compaction_cond);
        pthread_mutex_destroy(&db->compaction_mutex);
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        free(db);
        return NULL;
    }

    if (config != NULL) {
        db->hnsw_index = lsh_create(dimension, config, db->soa_storage);
    } else {
        GV_LSHConfig default_cfg = {.num_tables = 8, .num_hash_bits = 16, .seed = 42};
        db->hnsw_index = lsh_create(dimension, &default_cfg, db->soa_storage);
    }

    if (db->hnsw_index == NULL) {
        soa_storage_destroy(db->soa_storage);
        metadata_index_destroy(db->metadata_index);
        pthread_mutex_destroy(&db->resource_mutex);
        pthread_mutex_destroy(&db->observability_mutex);
        pthread_cond_destroy(&db->compaction_cond);
        pthread_mutex_destroy(&db->compaction_mutex);
        pthread_rwlock_destroy(&db->rwlock);
        pthread_mutex_destroy(&db->wal_mutex);
        free(db);
        return NULL;
    }

    return db;
}

int db_set_wal(GV_Database *db, const char *wal_path) {
    if (db == NULL) {
        return -1;
    }

    if (db->wal) {
        wal_close(db->wal);
        db->wal = NULL;
    }
    free(db->wal_path);
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
        free(db->wal_path);
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
    free(db->wal_path);
    db->wal_path = NULL;
}

int db_wal_dump(const GV_Database *db, FILE *out) {
    if (db == NULL || out == NULL || db->wal_path == NULL) {
        return -1;
    }
    return wal_dump(db->wal_path, db->dimension, (uint32_t)db->index_type, out);
}

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

    if (db->wal != NULL && db->wal_replaying == 0) {
        pthread_mutex_lock(&db->wal_mutex);
        int wal_res = wal_append_insert(db->wal, data, dimension, NULL, NULL);
        pthread_mutex_unlock(&db->wal_mutex);
        if (wal_res != 0) {
            db_decrement_concurrent_ops(db);
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
        size_t vector_index = soa_storage_add(db->soa_storage, normalized_data, NULL);
        free(normalized_data);
        if (vector_index == (size_t)-1) {
            pthread_rwlock_unlock(&db->rwlock);
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
        if (status != 0) {
            vector_destroy(vector);
        }
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
    }

    if (status != 0) {
        pthread_rwlock_unlock(&db->rwlock);
        db_decrement_concurrent_ops(db);
        return -1;
    }

    db->count += 1;
    db->total_inserts += 1;
    db_update_memory_usage(db);
    pthread_rwlock_unlock(&db->rwlock);

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

    if (db->wal != NULL && db->wal_replaying == 0) {
        pthread_mutex_lock(&db->wal_mutex);
        int wal_res = wal_append_insert(db->wal, data, dimension, metadata_key, metadata_value);
        pthread_mutex_unlock(&db->wal_mutex);
        if (wal_res != 0) {
            db_decrement_concurrent_ops(db);
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
            if (vector_set_metadata(&temp_vec, metadata_key, metadata_value) == 0) {
                metadata = temp_vec.metadata;
            }
        }
        size_t vector_index = soa_storage_add(db->soa_storage, normalized_data, metadata);
        free(normalized_data);
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
            return -1;
        }
        if (db->cosine_normalized) {
            db_normalize_vector(vector);
        }
        if (metadata_key != NULL && metadata_value != NULL) {
            if (vector_set_metadata(vector, metadata_key, metadata_value) != 0) {
                vector_destroy(vector);
                pthread_rwlock_unlock(&db->rwlock);
                return -1;
            }
        }
        /* Save metadata pointer before insert — hnsw_insert transfers ownership and NULLs it */
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
        if (status != 0) {
            vector_destroy(vector);
        }
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        GV_Vector *vector = vector_create_from_data(dimension, data);
        if (vector == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        if (db->cosine_normalized) {
            db_normalize_vector(vector);
        }
        if (metadata_key != NULL && metadata_value != NULL) {
            if (vector_set_metadata(vector, metadata_key, metadata_value) != 0) {
                vector_destroy(vector);
                pthread_rwlock_unlock(&db->rwlock);
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
               db->index_type == GV_INDEX_TYPE_PQ ||
               db->index_type == GV_INDEX_TYPE_LSH) {
        GV_Vector *vector = vector_create_from_data(dimension, data);
        if (vector == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        if (db->cosine_normalized) {
            db_normalize_vector(vector);
        }
        if (metadata_key != NULL && metadata_value != NULL) {
            if (vector_set_metadata(vector, metadata_key, metadata_value) != 0) {
                vector_destroy(vector);
                pthread_rwlock_unlock(&db->rwlock);
                return -1;
            }
        }
        /* flat/pq/lsh inserts may destroy the vector and clear metadata; ivfflat keeps it */
        GV_Metadata *saved_meta_for_index = vector->metadata;

        if (db->index_type == GV_INDEX_TYPE_FLAT) {
            status = flat_insert(db->hnsw_index, vector);
        } else if (db->index_type == GV_INDEX_TYPE_IVFFLAT) {
            status = ivfflat_insert(db->hnsw_index, vector);
        } else if (db->index_type == GV_INDEX_TYPE_PQ) {
            status = pq_insert(db->hnsw_index, vector);
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
        return -1;
    }

    db->count += 1;
    db->total_inserts += 1;
    pthread_rwlock_unlock(&db->rwlock);

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
        /* Cast sparse vector to vector for metadata operations */
        GV_Vector *vec = (GV_Vector *)sv;
        /* Ensure metadata is NULL (should be from calloc, but be safe) */
        vec->metadata = NULL;
        if (vector_set_metadata(vec, metadata_key, metadata_value) != 0) {
            sparse_vector_destroy(sv);
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
    }

    int status = sparse_index_add(db->sparse_index, sv);
    if (status != 0) {
        sparse_vector_destroy(sv);
        pthread_rwlock_unlock(&db->rwlock);
        return -1;
    }
    db->count += 1;
    db->total_inserts += 1;
    pthread_rwlock_unlock(&db->rwlock);
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

    if (db->wal != NULL && db->wal_replaying == 0) {
        pthread_mutex_lock(&db->wal_mutex);
        int wal_res = wal_append_insert_rich(db->wal, data, dimension, metadata_keys, metadata_values, metadata_count);
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
                    if (vector_set_metadata(&temp_vec, metadata_keys[i], metadata_values[i]) != 0) {
                        vector_clear_metadata(&temp_vec);
                        free(normalized_data);
                        pthread_rwlock_unlock(&db->rwlock);
                        return -1;
                    }
                }
            }
            metadata = temp_vec.metadata;
        }
        size_t vector_index = soa_storage_add(db->soa_storage, normalized_data, metadata);
        free(normalized_data);
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
        if (status != 0) {
            vector_destroy(vector);
        }
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
        if (status != 0) {
            vector_destroy(vector);
        }
    } else if (db->index_type == GV_INDEX_TYPE_FLAT ||
               db->index_type == GV_INDEX_TYPE_IVFFLAT ||
               db->index_type == GV_INDEX_TYPE_PQ ||
               db->index_type == GV_INDEX_TYPE_LSH) {
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
        } else if (db->index_type == GV_INDEX_TYPE_PQ) {
            status = pq_insert(db->hnsw_index, vector);
        } else {
            status = lsh_insert(db->hnsw_index, vector);
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

    db->count += 1;
    db->total_inserts += 1;
    db_update_memory_usage(db);
    pthread_rwlock_unlock(&db->rwlock);

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

        for (size_t i = 0; i < count; ++i) {
            const float *vec = data + i * dimension;
            int status = gv_hnsw_insert_raw(db->hnsw_index, vec, dimension);
            if (status != 0) {
                pthread_rwlock_unlock(&db->rwlock);
                return -1;
            }
            db->count += 1;
            db->total_inserts += 1;
        }

        db_update_memory_usage(db);
        pthread_rwlock_unlock(&db->rwlock);
        return 0;
    }

    for (size_t i = 0; i < count; ++i) {
        const float *vec = data + i * dimension;
        if (db_add_vector(db, vec, dimension) != 0) {
            return -1;
        }
    }
    return 0;
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

int db_save(const GV_Database *db, const char *filepath) {
    if (db == NULL) {
        return -1;
    }

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
        } else if (db->index_type == GV_INDEX_TYPE_PQ) {
            status = pq_save(db->hnsw_index, out, version);
        } else if (db->index_type == GV_INDEX_TYPE_LSH) {
            status = lsh_save(db->hnsw_index, out, version);
        } else {
            status = -1;
        }
    }

    if (fclose(out) != 0) {
        status = -1;
    }

    if (status == 0) {
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
                if (af == NULL || write_uint32(af, crc) != 0 || fclose(af) != 0) {
                    status = -1;
                }
            }
        }
    }

    if (db->wal != NULL && status == 0) {
        pthread_mutex_lock((pthread_mutex_t *)&db->wal_mutex);
        int truncate_status = wal_truncate(db->wal);
        if (truncate_status == 0) {
            ((GV_Database *)db)->total_wal_records = 0;
        }
        pthread_mutex_unlock((pthread_mutex_t *)&db->wal_mutex);
    } else if (db->wal_path != NULL && status == 0) {
        /* Fallback: if WAL handle is NULL but path exists, use reset */
        wal_reset(db->wal_path);
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
    return status == 0 ? 0 : -1;
}

int db_search(const GV_Database *db, const float *query_data, size_t k,
                 GV_SearchResult *results, GV_DistanceType distance_type) {
    if (db == NULL || query_data == NULL || results == NULL || k == 0) {
        return -1;
    }

    uint64_t start_time_us = db_get_time_us();

    memset(results, 0, k * sizeof(GV_SearchResult));

    pthread_rwlock_rdlock((pthread_rwlock_t *)&db->rwlock);
    ((GV_Database *)db)->total_queries += 1;

    if (db->index_type == GV_INDEX_TYPE_KDTREE && db->root == NULL) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        uint64_t end_time_us = db_get_time_us();
        uint64_t latency_us = end_time_us - start_time_us;
        db_record_latency((GV_Database *)db, latency_us, 0);
        return 0;
    }
    if (db->index_type == GV_INDEX_TYPE_HNSW && db->hnsw_index == NULL) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        uint64_t end_time_us = db_get_time_us();
        uint64_t latency_us = end_time_us - start_time_us;
        db_record_latency((GV_Database *)db, latency_us, 0);
        return 0;
    }
    if (db->index_type == GV_INDEX_TYPE_IVFPQ && db->hnsw_index == NULL) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        uint64_t end_time_us = db_get_time_us();
        uint64_t latency_us = end_time_us - start_time_us;
        db_record_latency((GV_Database *)db, latency_us, 0);
        return 0;
    }
    if ((db->index_type == GV_INDEX_TYPE_FLAT ||
         db->index_type == GV_INDEX_TYPE_IVFFLAT ||
         db->index_type == GV_INDEX_TYPE_PQ ||
         db->index_type == GV_INDEX_TYPE_LSH) && db->hnsw_index == NULL) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        uint64_t end_time_us = db_get_time_us();
        uint64_t latency_us = end_time_us - start_time_us;
        db_record_latency((GV_Database *)db, latency_us, 0);
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
            uint64_t end_time_us = db_get_time_us();
            uint64_t latency_us = end_time_us - start_time_us;
            db_record_latency((GV_Database *)db, latency_us, 0);
            return -1;
        }
        int r = exact_knn_search_kdtree(db->root, db->soa_storage, db->count, &query_vec, k, results, distance_type);
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        uint64_t end_time_us = db_get_time_us();
        uint64_t latency_us = end_time_us - start_time_us;
        db_record_latency((GV_Database *)db, latency_us, 0);
        return r;
    }

    int r = -1;
    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        if (db->soa_storage == NULL) {
            pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
            uint64_t end_time_us = db_get_time_us();
            uint64_t latency_us = end_time_us - start_time_us;
            db_record_latency((GV_Database *)db, latency_us, 0);
            return -1;
        }
        r = kdtree_knn_search(db->root, db->soa_storage, &query_vec, k, results, distance_type);
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        r = gv_hnsw_search(db->hnsw_index, &query_vec, k, results, distance_type, NULL, NULL);
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        r = gv_ivfpq_search(db->hnsw_index, &query_vec, k, results, distance_type, 0, 0);
    } else if (db->index_type == GV_INDEX_TYPE_FLAT) {
        r = flat_search(db->hnsw_index, &query_vec, k, results, distance_type, NULL, NULL);
    } else if (db->index_type == GV_INDEX_TYPE_IVFFLAT) {
        r = ivfflat_search(db->hnsw_index, &query_vec, k, results, distance_type, NULL, NULL);
    } else if (db->index_type == GV_INDEX_TYPE_PQ) {
        r = pq_search(db->hnsw_index, &query_vec, k, results, distance_type, NULL, NULL);
    } else if (db->index_type == GV_INDEX_TYPE_LSH) {
        r = lsh_search(db->hnsw_index, &query_vec, k, results, distance_type, NULL, NULL);
    }
    pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);

    uint64_t end_time_us = db_get_time_us();
    uint64_t latency_us = end_time_us - start_time_us;
    db_record_latency((GV_Database *)db, latency_us, 0);

    return r;
}

int db_search_ivfpq_opts(const GV_Database *db, const float *query_data, size_t k,
                            GV_SearchResult *results, GV_DistanceType distance_type,
                            size_t nprobe_override, size_t rerank_top) {
    if (db == NULL || query_data == NULL || results == NULL || k == 0) return -1;
    if (db->index_type != GV_INDEX_TYPE_IVFPQ || db->hnsw_index == NULL) {
        return db_search(db, query_data, k, results, distance_type);
    }
    pthread_rwlock_rdlock((pthread_rwlock_t *)&db->rwlock);
    ((GV_Database *)db)->total_queries += 1;
    GV_Vector query_vec;
    query_vec.data = (float *)query_data;
    query_vec.dimension = db->dimension;
    int r = gv_ivfpq_search(db->hnsw_index, &query_vec, k, results, distance_type,
                            nprobe_override, rerank_top);
    pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
    return r;
}

int db_search_batch(const GV_Database *db, const float *queries, size_t qcount, size_t k,
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
    if ((db->index_type == GV_INDEX_TYPE_FLAT ||
         db->index_type == GV_INDEX_TYPE_IVFFLAT ||
         db->index_type == GV_INDEX_TYPE_PQ ||
         db->index_type == GV_INDEX_TYPE_LSH) && db->hnsw_index == NULL) {
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
                r = kdtree_knn_search(db->root, db->soa_storage, &qv, k, slot, distance_type);
            }
        } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
            r = gv_hnsw_search(db->hnsw_index, &qv, k, slot, distance_type, NULL, NULL);
        } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
            r = gv_ivfpq_search(db->hnsw_index, &qv, k, slot, distance_type, 0, 0);
        } else if (db->index_type == GV_INDEX_TYPE_FLAT) {
            r = flat_search(db->hnsw_index, &qv, k, slot, distance_type, NULL, NULL);
        } else if (db->index_type == GV_INDEX_TYPE_IVFFLAT) {
            r = ivfflat_search(db->hnsw_index, &qv, k, slot, distance_type, NULL, NULL);
        } else if (db->index_type == GV_INDEX_TYPE_PQ) {
            r = pq_search(db->hnsw_index, &qv, k, slot, distance_type, NULL, NULL);
        } else if (db->index_type == GV_INDEX_TYPE_LSH) {
            r = lsh_search(db->hnsw_index, &qv, k, slot, distance_type, NULL, NULL);
        }
        if (r < 0) {
            pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
            return -1;
        }
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
    return (int)(qcount * k);
}

void gv_search_results_free(GV_SearchResult *results, size_t count) {
    if (!results) return;
    for (size_t i = 0; i < count; i++) {
        if (results[i].vector) {
            vector_destroy((GV_Vector *)results[i].vector);
            results[i].vector = NULL;
        }
    }
}

int db_search_filtered(const GV_Database *db, const float *query_data, size_t k,
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
    if ((db->index_type == GV_INDEX_TYPE_FLAT ||
         db->index_type == GV_INDEX_TYPE_IVFFLAT ||
         db->index_type == GV_INDEX_TYPE_PQ ||
         db->index_type == GV_INDEX_TYPE_LSH) && db->hnsw_index == NULL) {
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
        int r = kdtree_knn_search_filtered(db->root, db->soa_storage, &query_vec, k, results, distance_type,
                                            filter_key, filter_value);
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return r;
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        int r = gv_hnsw_search(db->hnsw_index, &query_vec, k, results, distance_type,
                            filter_key, filter_value);
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return r;
    } else if (db->index_type == GV_INDEX_TYPE_FLAT) {
        int r = flat_search(db->hnsw_index, &query_vec, k, results, distance_type,
                            filter_key, filter_value);
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return r;
    } else if (db->index_type == GV_INDEX_TYPE_IVFFLAT) {
        int r = ivfflat_search(db->hnsw_index, &query_vec, k, results, distance_type,
                            filter_key, filter_value);
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return r;
    } else if (db->index_type == GV_INDEX_TYPE_PQ) {
        int r = pq_search(db->hnsw_index, &query_vec, k, results, distance_type,
                            filter_key, filter_value);
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return r;
    } else if (db->index_type == GV_INDEX_TYPE_LSH) {
        int r = lsh_search(db->hnsw_index, &query_vec, k, results, distance_type,
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
                const char *val = vector_get_metadata(tmp[i].vector, filter_key);
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

int db_search_with_filter_expr(const GV_Database *db, const float *query_data, size_t k,
                                  GV_SearchResult *results, GV_DistanceType distance_type,
                                  const char *filter_expr) {
    if (db == NULL || query_data == NULL || results == NULL || k == 0 || filter_expr == NULL) {
        return -1;
    }

    GV_Filter *filter = filter_parse(filter_expr);
    if (filter == NULL) {
        return -1;
    }

    pthread_rwlock_rdlock((pthread_rwlock_t *)&db->rwlock);
    ((GV_Database *)db)->total_queries += 1;

    if (db->index_type == GV_INDEX_TYPE_KDTREE && db->root == NULL) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        filter_destroy(filter);
        return 0;
    }
    if (db->index_type == GV_INDEX_TYPE_HNSW && db->hnsw_index == NULL) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        filter_destroy(filter);
        return 0;
    }
    if (db->index_type == GV_INDEX_TYPE_IVFPQ && db->hnsw_index == NULL) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        filter_destroy(filter);
        return 0;
    }
    if ((db->index_type == GV_INDEX_TYPE_FLAT ||
         db->index_type == GV_INDEX_TYPE_IVFFLAT ||
         db->index_type == GV_INDEX_TYPE_PQ ||
         db->index_type == GV_INDEX_TYPE_LSH) && db->hnsw_index == NULL) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        filter_destroy(filter);
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
        filter_destroy(filter);
        return 0;
    }

    GV_SearchResult *tmp = (GV_SearchResult *)malloc(max_candidates * sizeof(GV_SearchResult));
    if (!tmp) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        filter_destroy(filter);
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
            n = kdtree_knn_search(db->root, db->soa_storage, &query_vec, max_candidates, tmp, distance_type);
        }
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        n = gv_hnsw_search(db->hnsw_index, &query_vec, max_candidates, tmp, distance_type, NULL, NULL);
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        n = gv_ivfpq_search(db->hnsw_index, &query_vec, max_candidates, tmp, distance_type, 0, 0);
    } else if (db->index_type == GV_INDEX_TYPE_FLAT) {
        n = flat_search(db->hnsw_index, &query_vec, max_candidates, tmp, distance_type, NULL, NULL);
    } else if (db->index_type == GV_INDEX_TYPE_IVFFLAT) {
        n = ivfflat_search(db->hnsw_index, &query_vec, max_candidates, tmp, distance_type, NULL, NULL);
    } else if (db->index_type == GV_INDEX_TYPE_PQ) {
        n = pq_search(db->hnsw_index, &query_vec, max_candidates, tmp, distance_type, NULL, NULL);
    } else if (db->index_type == GV_INDEX_TYPE_LSH) {
        n = lsh_search(db->hnsw_index, &query_vec, max_candidates, tmp, distance_type, NULL, NULL);
    } else {
        free(tmp);
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        filter_destroy(filter);
        return -1;
    }

    if (n <= 0) {
        free(tmp);
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        filter_destroy(filter);
        return n;
    }

    size_t out = 0;
    for (int i = 0; i < n && out < k; ++i) {
        int match = filter_eval(filter, tmp[i].vector);
        if (match < 0) {
            free(tmp);
            pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
            filter_destroy(filter);
            return -1;
        }
        if (match == 1) {
            results[out++] = tmp[i];
        }
    }

    free(tmp);
    pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
    filter_destroy(filter);
    return (int)out;
}

void db_set_exact_search_threshold(GV_Database *db, size_t threshold) {
    if (db == NULL) {
        return;
    }
    db->exact_search_threshold = threshold;
}

void db_set_force_exact_search(GV_Database *db, int enabled) {
    if (db == NULL) {
        return;
    }
    db->force_exact_search = enabled ? 1 : 0;
}

int db_search_sparse(const GV_Database *db, const uint32_t *indices, const float *values,
                        size_t nnz, size_t k, GV_SearchResult *results, GV_DistanceType distance_type) {
    if (db == NULL || db->index_type != GV_INDEX_TYPE_SPARSE || results == NULL || k == 0) {
        return -1;
    }
    if ((indices == NULL || values == NULL) && nnz > 0) {
        return -1;
    }
    ((GV_Database *)db)->total_queries += 1;
    GV_SparseVector *query = sparse_vector_create(db->dimension, indices, values, nnz);
    if (query == NULL) {
        return -1;
    }
    int r = sparse_index_search(db->sparse_index, query, k, results, distance_type);
    sparse_vector_destroy(query);
    return r;
}

int db_range_search(const GV_Database *db, const float *query_data, float radius,
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
    if ((db->index_type == GV_INDEX_TYPE_FLAT ||
         db->index_type == GV_INDEX_TYPE_IVFFLAT ||
         db->index_type == GV_INDEX_TYPE_PQ ||
         db->index_type == GV_INDEX_TYPE_LSH) && db->hnsw_index == NULL) {
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
        r = kdtree_range_search(db->root, db->soa_storage, &query_vec, radius, results, max_results, distance_type);
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        r = gv_hnsw_range_search(db->hnsw_index, &query_vec, radius, results, max_results, distance_type, NULL, NULL);
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        r = gv_ivfpq_range_search(db->hnsw_index, &query_vec, radius, results, max_results, distance_type);
    } else if (db->index_type == GV_INDEX_TYPE_FLAT) {
        r = flat_range_search(db->hnsw_index, &query_vec, radius, results, max_results, distance_type, NULL, NULL);
    } else if (db->index_type == GV_INDEX_TYPE_IVFFLAT) {
        r = ivfflat_range_search(db->hnsw_index, &query_vec, radius, results, max_results, distance_type, NULL, NULL);
    } else if (db->index_type == GV_INDEX_TYPE_PQ) {
        r = pq_range_search(db->hnsw_index, &query_vec, radius, results, max_results, distance_type, NULL, NULL);
    } else if (db->index_type == GV_INDEX_TYPE_LSH) {
        r = lsh_range_search(db->hnsw_index, &query_vec, radius, results, max_results, distance_type, NULL, NULL);
    } else {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return -1;
    }
    
    pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
    return r;
}

int db_range_search_filtered(const GV_Database *db, const float *query_data, float radius,
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
    if ((db->index_type == GV_INDEX_TYPE_FLAT ||
         db->index_type == GV_INDEX_TYPE_IVFFLAT ||
         db->index_type == GV_INDEX_TYPE_PQ ||
         db->index_type == GV_INDEX_TYPE_LSH) && db->hnsw_index == NULL) {
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
        r = kdtree_range_search_filtered(db->root, db->soa_storage, &query_vec, radius, results, max_results,
                                            distance_type, filter_key, filter_value);
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        r = gv_hnsw_range_search(db->hnsw_index, &query_vec, radius, results, max_results,
                                distance_type, filter_key, filter_value);
    } else if (db->index_type == GV_INDEX_TYPE_FLAT) {
        r = flat_range_search(db->hnsw_index, &query_vec, radius, results, max_results,
                                distance_type, filter_key, filter_value);
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return r;
    } else if (db->index_type == GV_INDEX_TYPE_IVFFLAT) {
        r = ivfflat_range_search(db->hnsw_index, &query_vec, radius, results, max_results,
                                    distance_type, filter_key, filter_value);
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return r;
    } else if (db->index_type == GV_INDEX_TYPE_PQ) {
        r = pq_range_search(db->hnsw_index, &query_vec, radius, results, max_results,
                               distance_type, filter_key, filter_value);
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return r;
    } else if (db->index_type == GV_INDEX_TYPE_LSH) {
        r = lsh_range_search(db->hnsw_index, &query_vec, radius, results, max_results,
                                distance_type, filter_key, filter_value);
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return r;
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        r = gv_ivfpq_range_search(db->hnsw_index, &query_vec, radius, results, max_results, distance_type);
        if (r > 0 && filter_key != NULL) {
            int out = 0;
            for (int i = 0; i < r && out < (int)max_results; ++i) {
                const char *val = vector_get_metadata(results[i].vector, filter_key);
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

int db_delete_vector_by_index(GV_Database *db, size_t vector_index) {
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
        status = kdtree_delete(&(db->root), db->soa_storage, vector_index);
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
        status = sparse_index_delete(db->sparse_index, vector_index);
    } else if (db->index_type == GV_INDEX_TYPE_FLAT) {
        if (db->hnsw_index == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        status = flat_delete(db->hnsw_index, vector_index);
    } else if (db->index_type == GV_INDEX_TYPE_IVFFLAT) {
        if (db->hnsw_index == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        status = ivfflat_delete(db->hnsw_index, vector_index);
    } else if (db->index_type == GV_INDEX_TYPE_PQ) {
        if (db->hnsw_index == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        status = pq_delete(db->hnsw_index, vector_index);
    } else if (db->index_type == GV_INDEX_TYPE_LSH) {
        if (db->hnsw_index == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        status = lsh_delete(db->hnsw_index, vector_index);
    } else {
        pthread_rwlock_unlock(&db->rwlock);
        return -1;
    }

    if (status == 0) {
        if (db->metadata_index != NULL) {
            metadata_index_remove_vector(db->metadata_index, vector_index);
        }
        if (db->wal != NULL) {
            if (wal_append_delete(db->wal, vector_index) != 0) {
                pthread_rwlock_unlock(&db->rwlock);
                return -1;
            }
            db->total_wal_records += 1;
        }
    }

    pthread_rwlock_unlock(&db->rwlock);
    return status;
}

int db_update_vector(GV_Database *db, size_t vector_index, const float *new_data, size_t dimension) {
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
        if (soa_storage_is_deleted(db->soa_storage, vector_index) == 1) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        if (db->filepath == NULL) {
            // In-memory KDTREE: update only SOA storage
            status = soa_storage_update_data(db->soa_storage, vector_index, new_data);
        } else {
            // File-based KDTREE: update tree and SOA
            status = kdtree_update(&(db->root), db->soa_storage, vector_index, new_data);
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
    } else if (db->index_type == GV_INDEX_TYPE_FLAT) {
        if (db->hnsw_index == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        status = flat_update(db->hnsw_index, vector_index, new_data, dimension);
    } else if (db->index_type == GV_INDEX_TYPE_IVFFLAT) {
        if (db->hnsw_index == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        status = ivfflat_update(db->hnsw_index, vector_index, new_data, dimension);
    } else if (db->index_type == GV_INDEX_TYPE_PQ) {
        if (db->hnsw_index == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        status = pq_update(db->hnsw_index, vector_index, new_data, dimension);
    } else if (db->index_type == GV_INDEX_TYPE_LSH) {
        if (db->hnsw_index == NULL) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        status = lsh_update(db->hnsw_index, vector_index, new_data, dimension);
    } else if (db->index_type == GV_INDEX_TYPE_SPARSE) {
        pthread_rwlock_unlock(&db->rwlock);
        return -1;
    } else {
        pthread_rwlock_unlock(&db->rwlock);
        return -1;
    }

    if (status == 0 && db->wal != NULL) {
        if (wal_append_update(db->wal, vector_index, new_data, dimension, NULL, NULL, 0) != 0) {
            pthread_rwlock_unlock(&db->rwlock);
            return -1;
        }
        db->total_wal_records += 1;
    }

    pthread_rwlock_unlock(&db->rwlock);
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
        
        /* Create new metadata chain by building a temporary vector */
        GV_Vector temp_vec;
        temp_vec.dimension = db->dimension;
        temp_vec.data = NULL;
        temp_vec.metadata = NULL;
        
        for (size_t i = 0; i < metadata_count; ++i) {
            if (metadata_keys[i] != NULL && metadata_values[i] != NULL) {
                if (vector_set_metadata(&temp_vec, metadata_keys[i], metadata_values[i]) != 0) {
                    vector_clear_metadata(&temp_vec);
                    pthread_rwlock_unlock(&db->rwlock);
                    return -1;
                }
            }
        }
        
        /* Get old metadata and copy keys/values before updating (update will free old metadata) */
        GV_Metadata *old_metadata = soa_storage_get_metadata(db->soa_storage, vector_index);
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
        // Ownership transferred to SOA, prevent double-free
        temp_vec.metadata = NULL;

        if (status == 0 && db->metadata_index != NULL) {
            metadata_index_update(db->metadata_index, vector_index, old_metadata_copy, temp_vec.metadata);
        }
        
        while (old_metadata_copy != NULL) {
            GV_Metadata *next = old_metadata_copy->next;
            free(old_metadata_copy->key);
            free(old_metadata_copy->value);
            free(old_metadata_copy);
            old_metadata_copy = next;
        }
    } else if (db->index_type == GV_INDEX_TYPE_HNSW ||
               db->index_type == GV_INDEX_TYPE_IVFPQ ||
               db->index_type == GV_INDEX_TYPE_FLAT ||
               db->index_type == GV_INDEX_TYPE_LSH) {
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

        for (size_t i = 0; i < metadata_count; ++i) {
            if (metadata_keys[i] != NULL && metadata_values[i] != NULL) {
                if (vector_set_metadata(&temp_vec, metadata_keys[i], metadata_values[i]) != 0) {
                    vector_clear_metadata(&temp_vec);
                    pthread_rwlock_unlock(&db->rwlock);
                    return -1;
                }
            }
        }

        GV_Metadata *old_metadata = soa_storage_get_metadata(db->soa_storage, vector_index);
        GV_Metadata *old_metadata_copy = NULL;
        if (old_metadata != NULL && db->metadata_index != NULL) {
            GV_Metadata *current = old_metadata;
            GV_Metadata *prev = NULL;
            while (current != NULL) {
                GV_Metadata *copy = (GV_Metadata *)malloc(sizeof(GV_Metadata));
                if (copy == NULL) {
                    while (old_metadata_copy != NULL) {
                        GV_Metadata *next = old_metadata_copy->next;
                        free(old_metadata_copy->key);
                        free(old_metadata_copy->value);
                        free(old_metadata_copy);
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
            metadata_index_update(db->metadata_index, vector_index, old_metadata_copy, temp_vec.metadata);
        }

        while (old_metadata_copy != NULL) {
            GV_Metadata *next = old_metadata_copy->next;
            free(old_metadata_copy->key);
            free(old_metadata_copy->value);
            free(old_metadata_copy);
            old_metadata_copy = next;
        }
    } else if (db->index_type == GV_INDEX_TYPE_SPARSE) {
        /* Sparse index doesn't use SoA storage — metadata not supported */
        pthread_rwlock_unlock(&db->rwlock);
        return -1;
    } else {
        pthread_rwlock_unlock(&db->rwlock);
        return -1;
    }

    if (status == 0 && db->wal != NULL && vector_data != NULL) {
        if (wal_append_update(db->wal, vector_index, vector_data, db->dimension,
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
static int db_compact_soa_storage(GV_Database *db) {
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
    float *new_data = (float *)malloc(new_count * dimension * sizeof(float));
    GV_Metadata **new_metadata = (GV_Metadata **)calloc(new_count, sizeof(GV_Metadata *));
    int *new_deleted = (int *)calloc(new_count, sizeof(int));
    
    if (new_data == NULL || new_metadata == NULL || new_deleted == NULL) {
        free(new_data);
        free(new_metadata);
        free(new_deleted);
        return -1;
    }

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

    free(storage->data);
    free(storage->metadata);
    free(storage->deleted);

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
                return -1; /* Failed to create new index */
            }
            
            for (size_t i = 0; i < new_count; ++i) {
                GV_Vector temp_vec = {
                    .dimension = dimension,
                    .data = storage->data + (i * dimension),
                    .metadata = storage->metadata[i]
                };
                
                if (gv_hnsw_insert(db->hnsw_index, &temp_vec) != 0) {
                    /* On failure, clean up */
                    gv_hnsw_destroy(db->hnsw_index);
                    db->hnsw_index = NULL;
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

    free(index_map);
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

    char *temp_path = (char *)malloc(strlen(db->filepath) + 10);
    if (temp_path == NULL) {
        return -1;
    }
    snprintf(temp_path, strlen(db->filepath) + 10, "%s.tmp", db->filepath);

    int save_result = db_save(db, temp_path);
    if (save_result == 0) {
        if (rename(temp_path, db->filepath) == 0) {
            if (db->wal != NULL) {
                wal_truncate(db->wal);
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

int db_compact(GV_Database *db) {
    if (db == NULL) {
        return -1;
    }

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
            db_compact_soa_storage(db);
        }
    }

    db_compact_wal(db);

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

static size_t db_estimate_vector_memory(size_t dimension) {
    size_t data_size = dimension * sizeof(float);
    size_t metadata_overhead = 64; /* average metadata bytes per vector */
    size_t storage_overhead = sizeof(int) + sizeof(void *);
    return data_size + metadata_overhead + storage_overhead;
}

static void db_update_memory_usage(GV_Database *db) {
    if (db == NULL) {
        return;
    }

    pthread_mutex_lock(&db->resource_mutex);

    size_t total_memory = 0;

    total_memory += sizeof(GV_Database);

    if (db->soa_storage != NULL) {
        size_t vector_memory = db_estimate_vector_memory(db->soa_storage->dimension);
        total_memory += sizeof(GV_SoAStorage);
        total_memory += db->soa_storage->count * vector_memory;
        total_memory += db->soa_storage->capacity * (sizeof(GV_Metadata *) + sizeof(int));
    }

    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        total_memory += db->count * (sizeof(GV_KDNode) + sizeof(size_t));
    } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
        total_memory += db->count * (sizeof(void *) * 2); /* rough estimate */
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        total_memory += db->count * (sizeof(void *) * 2); /* rough estimate */
    }

    if (db->metadata_index != NULL) {
        total_memory += db->count * 100; /* rough estimate: 100 bytes per entry */
    }

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

static int db_check_resource_limits(GV_Database *db, size_t additional_vectors, size_t additional_memory) {
    if (db == NULL) {
        return -1;
    }

    pthread_mutex_lock(&db->resource_mutex);

    if (db->resource_limits.max_vectors > 0) {
        if (db->count + additional_vectors > db->resource_limits.max_vectors) {
            pthread_mutex_unlock(&db->resource_mutex);
            return -1;
        }
    }

    if (db->resource_limits.max_memory_bytes > 0) {
        size_t estimated_new_memory = db->current_memory_bytes + additional_memory;
        if (estimated_new_memory > db->resource_limits.max_memory_bytes) {
            pthread_mutex_unlock(&db->resource_mutex);
            return -1;
        }
    }

    if (db->resource_limits.max_concurrent_operations > 0) {
        if (db->current_concurrent_ops >= db->resource_limits.max_concurrent_operations) {
            pthread_mutex_unlock(&db->resource_mutex);
            return -1;
        }
    }

    pthread_mutex_unlock(&db->resource_mutex);
    return 0;
}

static void db_increment_concurrent_ops(GV_Database *db) {
    if (db == NULL) {
        return;
    }
    pthread_mutex_lock(&db->resource_mutex);
    db->current_concurrent_ops++;
    pthread_mutex_unlock(&db->resource_mutex);
}

static void db_decrement_concurrent_ops(GV_Database *db) {
    if (db == NULL) {
        return;
    }
    pthread_mutex_lock(&db->resource_mutex);
    if (db->current_concurrent_ops > 0) {
        db->current_concurrent_ops--;
    }
    pthread_mutex_unlock(&db->resource_mutex);
}

int db_set_resource_limits(GV_Database *db, const GV_ResourceLimits *limits) {
    if (db == NULL || limits == NULL) {
        return -1;
    }

    pthread_mutex_lock(&db->resource_mutex);
    db->resource_limits.max_memory_bytes = limits->max_memory_bytes;
    db->resource_limits.max_vectors = limits->max_vectors;
    db->resource_limits.max_concurrent_operations = limits->max_concurrent_operations;
    pthread_mutex_unlock(&db->resource_mutex);

    db_update_memory_usage(db);

    return 0;
}

void db_get_resource_limits(const GV_Database *db, GV_ResourceLimits *limits) {
    if (db == NULL || limits == NULL) {
        return;
    }

    pthread_mutex_lock((pthread_mutex_t *)&db->resource_mutex);
    limits->max_memory_bytes = db->resource_limits.max_memory_bytes;
    limits->max_vectors = db->resource_limits.max_vectors;
    limits->max_concurrent_operations = db->resource_limits.max_concurrent_operations;
    pthread_mutex_unlock((pthread_mutex_t *)&db->resource_mutex);
}

size_t db_get_memory_usage(const GV_Database *db) {
    if (db == NULL) {
        return 0;
    }

    db_update_memory_usage((GV_Database *)db);

    pthread_mutex_lock((pthread_mutex_t *)&db->resource_mutex);
    size_t usage = db->current_memory_bytes;
    pthread_mutex_unlock((pthread_mutex_t *)&db->resource_mutex);

    return usage;
}

size_t db_get_concurrent_operations(const GV_Database *db) {
    if (db == NULL) {
        return 0;
    }

    pthread_mutex_lock((pthread_mutex_t *)&db->resource_mutex);
    size_t ops = db->current_concurrent_ops;
    pthread_mutex_unlock((pthread_mutex_t *)&db->resource_mutex);

    return ops;
}

static uint64_t db_get_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
}

static int db_init_latency_histogram(GV_LatencyHistogram *hist, size_t bucket_count) {
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

    /* Logarithmic buckets: 1us, 10us, 100us, 1ms, 10ms, 100ms, 1s, 10s */
    for (size_t i = 0; i < bucket_count; ++i) {
        hist->bucket_boundaries[i] = pow(10.0, (double)(i + 1));
    }

    hist->total_samples = 0;
    hist->sum_latency_us = 0;
    return 0;
}

static void db_add_latency_sample(GV_LatencyHistogram *hist, uint64_t latency_us) {
    if (hist == NULL || hist->buckets == NULL) {
        return;
    }

    hist->total_samples++;
    hist->sum_latency_us += latency_us;

    for (size_t i = 0; i < hist->bucket_count; ++i) {
        if (latency_us <= (uint64_t)hist->bucket_boundaries[i]) {
            hist->buckets[i]++;
            return;
        }
    }
    if (hist->bucket_count > 0) {
        hist->buckets[hist->bucket_count - 1]++;
    }
}

void db_record_latency(GV_Database *db, uint64_t latency_us, int is_insert) {
    if (db == NULL) {
        return;
    }

    pthread_mutex_lock(&db->observability_mutex);

    if (is_insert && db->insert_latency_hist.buckets == NULL) {
        db_init_latency_histogram(&db->insert_latency_hist, 8);
    } else if (!is_insert && db->search_latency_hist.buckets == NULL) {
        db_init_latency_histogram(&db->search_latency_hist, 8);
    }

    if (is_insert) {
        db_add_latency_sample(&db->insert_latency_hist, latency_us);
        db->insert_count_since_update++;
        
        uint64_t now_us = db_get_time_us();
        if (db->first_insert_time_us == 0) {
            /* Set once and never reset — needed for precise lifetime IPS */
            db->first_insert_time_us = now_us;
        }
        if (db->last_ips_update_time_us == 0) {
            db->last_ips_update_time_us = now_us;
        }
        
        if (db->insert_count_since_update > 0) {
            uint64_t elapsed_us = now_us - db->last_ips_update_time_us;
            /* Clamp to 100us minimum to avoid division issues */
            uint64_t effective_elapsed_us = (elapsed_us > 100) ? elapsed_us : 100;
            double elapsed_sec = (double)effective_elapsed_us / 1000000.0;
            double calculated_ips = (double)db->insert_count_since_update / elapsed_sec;
            db->current_ips = calculated_ips;
        }
        
        uint64_t elapsed_us = now_us - db->last_ips_update_time_us;
        if (elapsed_us >= 1000000) {
            /* Reset window counts but never reset first_insert_time_us or last_ips_update_time_us —
               they are needed for precise lifetime IPS fallback calculation. */
            db->insert_count_since_update = 0;
        }
    } else {
        db_add_latency_sample(&db->search_latency_hist, latency_us);
        db->query_count_since_update++;
        
        uint64_t now_us = db_get_time_us();
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
            }
        }
    }

    pthread_mutex_unlock(&db->observability_mutex);
}

void db_record_recall(GV_Database *db, double recall) {
    if (db == NULL || recall < 0.0 || recall > 1.0) {
        return;
    }

    pthread_mutex_lock(&db->observability_mutex);

    db->recall_metrics.total_queries++;
    
    double total = db->recall_metrics.avg_recall * (double)(db->recall_metrics.total_queries - 1) + recall;
    db->recall_metrics.avg_recall = total / (double)db->recall_metrics.total_queries;

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

int db_get_detailed_stats(const GV_Database *db, GV_DetailedStats *out) {
    if (db == NULL || out == NULL) {
        return -1;
    }

    memset(out, 0, sizeof(GV_DetailedStats));

    pthread_rwlock_rdlock((pthread_rwlock_t *)&db->rwlock);
    pthread_mutex_lock((pthread_mutex_t *)&db->observability_mutex);

    out->basic_stats.total_inserts = db->total_inserts;
    out->basic_stats.total_queries = db->total_queries;
    out->basic_stats.total_range_queries = db->total_range_queries;
    out->basic_stats.total_wal_records = db->total_wal_records;

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

    uint64_t now_us = db_get_time_us();
    
    if (db->last_qps_update_time_us == 0) {
        out->queries_per_second = 0.0;
    } else {
        uint64_t elapsed_us = now_us - db->last_qps_update_time_us;
        if (elapsed_us == 0) elapsed_us = 1; /* Avoid division by zero */
        double elapsed_sec = (double)elapsed_us / 1000000.0;
        
        if (db->query_count_since_update > 0 && elapsed_us < 5000000) {
            out->queries_per_second = (double)db->query_count_since_update / elapsed_sec;
        } else {
            out->queries_per_second = db->current_qps;
        }
    }
    
    {
        /* Priority 1: Use first_insert_time_us if available (most accurate) */
        if (db->total_inserts > 0 && db->first_insert_time_us > 0) {
            uint64_t total_elapsed_us = now_us - db->first_insert_time_us;
            if (total_elapsed_us > 0) {
                double total_elapsed_sec = (double)total_elapsed_us / 1000000.0;
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
        /* Priority 4: first_insert_time_us not set; fall back to last_ips_update_time_us */
        else if (db->total_inserts > 0 && db->last_ips_update_time_us > 0) {
            uint64_t total_elapsed_us = now_us - db->last_ips_update_time_us;
            if (total_elapsed_us > 0) {
                double total_elapsed_sec = (double)total_elapsed_us / 1000000.0;
                if (total_elapsed_sec > 0.0) {
                    double precise_ips = (double)db->total_inserts / total_elapsed_sec;
                    out->inserts_per_second = precise_ips;
                    ((GV_Database *)db)->current_ips = precise_ips;
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
        /* Priority 5: inserts exist but no timing data (e.g. after database reopen) — rate unknowable */
        else if (db->total_inserts > 0) {
            out->inserts_per_second = 0.0;
        } else {
            out->inserts_per_second = 0.0;
        }
    }
    out->last_qps_update_time = db->last_qps_update_time_us;

    db_update_memory_usage((GV_Database *)db);
    out->memory.total_bytes = db->current_memory_bytes;
    
    if (db->soa_storage != NULL) {
        size_t vector_memory = db_estimate_vector_memory(db->soa_storage->dimension);
        out->memory.soa_storage_bytes = sizeof(GV_SoAStorage) +
            db->soa_storage->count * vector_memory +
            db->soa_storage->capacity * (sizeof(GV_Metadata *) + sizeof(int));
    }

    if (db->index_type == GV_INDEX_TYPE_KDTREE) {
        out->memory.index_bytes = db->count * (sizeof(void *) * 3); /* rough estimate */
    } else if (db->index_type == GV_INDEX_TYPE_HNSW || db->index_type == GV_INDEX_TYPE_IVFPQ) {
        out->memory.index_bytes = db->count * (sizeof(void *) * 2); /* rough estimate */
    }

    if (db->metadata_index != NULL) {
        out->memory.metadata_index_bytes = db->count * 100; /* rough estimate */
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

    out->recall.total_queries = db_nonconst->recall_metrics.total_queries;
    out->recall.avg_recall = db_nonconst->recall_metrics.avg_recall;
    out->recall.min_recall = db_nonconst->recall_metrics.min_recall;
    out->recall.max_recall = db_nonconst->recall_metrics.max_recall;

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

void db_free_detailed_stats(GV_DetailedStats *stats) {
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

int db_health_check(const GV_Database *db) {
    if (db == NULL) {
        return -2; /* Unhealthy */
    }

    pthread_rwlock_rdlock((pthread_rwlock_t *)&db->rwlock);

    int health = 0; /* Healthy by default */

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

    if (db->resource_limits.max_memory_bytes > 0) {
        db_update_memory_usage((GV_Database *)db);
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

/* Internal struct layouts for accessing ef_search / nprobe */
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
        size_t saved_ef = idx->efSearch;
        idx->efSearch = params->ef_search;
        int r = db_search(db, query_data, k, results, distance_type);
        idx->efSearch = saved_ef;
        return r;
    }

    if (db->index_type == GV_INDEX_TYPE_IVFFLAT && params->nprobe > 0 && db->hnsw_index != NULL) {
        GV_IVFFlatIndexPartial *idx = (GV_IVFFlatIndexPartial *)db->hnsw_index;
        size_t saved_nprobe = idx->config.nprobe;
        idx->config.nprobe = params->nprobe;
        int r = db_search(db, query_data, k, results, distance_type);
        idx->config.nprobe = saved_nprobe;
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
                free(line);
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

        float *data = (float *)malloc(dim * sizeof(float));
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
            free(data);
            json_free(obj);
            continue;
        }

        GV_JsonValue *meta_obj = json_object_get(obj, "metadata");
        int insert_ok = -1;

        if (meta_obj && meta_obj->type == GV_JSON_OBJECT && json_object_length(meta_obj) > 0) {
            size_t meta_count = json_object_length(meta_obj);
            const char **keys = (const char **)malloc(meta_count * sizeof(const char *));
            const char **vals = (const char **)malloc(meta_count * sizeof(const char *));

            if (keys && vals) {
                for (size_t m = 0; m < meta_count; m++) {
                    keys[m] = meta_obj->data.object.entries[m].key;
                    const char *sv = json_get_string(meta_obj->data.object.entries[m].value);
                    vals[m] = sv ? sv : "";
                }
                insert_ok = db_add_vector_with_rich_metadata(db, data, dim,
                                                                 keys, vals, meta_count);
            }
            free(keys);
            free(vals);
        } else {
            insert_ok = db_add_vector(db, data, dim);
        }

        free(data);
        json_free(obj);

        if (insert_ok == 0) {
            imported++;
        }
    }

    free(line);
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
