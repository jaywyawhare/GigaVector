#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#ifndef _WIN32
#include <sys/time.h>
#endif

#include "storage/db_internal.h"
#include "storage/soa_storage.h"
#include "index/kdtree.h"
#include "core/compat.h"

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

size_t db_estimate_vector_memory(size_t dimension) {
    size_t data_size = dimension * sizeof(float);
    size_t metadata_overhead = 64; /* average metadata bytes per vector */
    size_t storage_overhead = sizeof(int) + sizeof(void *);
    return data_size + metadata_overhead + storage_overhead;
}

void db_update_memory_usage(GV_Database *db) {
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

uint64_t db_get_time_us(void) {
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
                    double precise_ips = (double)db->total_inserts / 0.001;
                    out->inserts_per_second = precise_ips;
                    ((GV_Database *)db)->current_ips = precise_ips;
                }
            } else {
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
