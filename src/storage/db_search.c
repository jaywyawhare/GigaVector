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
#include "core/memory.h"
#include "core/log.h"
#include "core/utils.h"
#include "core/scope.h"
#include "storage/database.h"
#include "storage/db_internal.h"
#include "search/distance.h"
#include "search/filter.h"
#include "index/exact_search.h"
#include "index/hnsw.h"
#include "index/ivfpq.h"
#include "index/kdtree.h"
#include "schema/metadata.h"
#include "index/sparse_index.h"
#include "schema/vector.h"
#include "storage/wal.h"
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
#include "specialized/optimizer.h"
#include "admin/cdc.h"

static int db_search_raw(const GV_Database *db, const float *query_data, size_t k,
                 GV_SearchResult *results, GV_DistanceType distance_type) {
    if (db == NULL || query_data == NULL || results == NULL || k == 0) {
        return -1;
    }

    gv_tls_arena_reset();

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
         db->index_type == GV_INDEX_TYPE_IVFSQ8 ||
         db->index_type == GV_INDEX_TYPE_IVFTURBOQUANT ||
         db->index_type == GV_INDEX_TYPE_PQ ||
         db->index_type == GV_INDEX_TYPE_LSH ||
         db->index_type == GV_INDEX_TYPE_RABITQ) && db->hnsw_index == NULL) {
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
    } else if (db->index_type == GV_INDEX_TYPE_IVFDISK) {
        r = ivfdisk_search((GV_IVFDiskIndex *)db->hnsw_index, query_data, k, results, distance_type);
        if (r > 0) db_fill_ivfdisk_search_vectors((GV_Database *)db, results, r);
    } else if (db->index_type == GV_INDEX_TYPE_IVFSQ8) {
        r = ivfsq8_search(db->hnsw_index, &query_vec, k, results, distance_type, NULL, NULL);
    } else if (db->index_type == GV_INDEX_TYPE_IVFTURBOQUANT) {
        r = ivfturboquant_search(db->hnsw_index, &query_vec, k, results, distance_type, NULL, NULL);
    } else if (db->index_type == GV_INDEX_TYPE_PQ) {
        r = pq_search(db->hnsw_index, &query_vec, k, results, distance_type, NULL, NULL);
    } else if (db->index_type == GV_INDEX_TYPE_LSH) {
        r = lsh_search(db->hnsw_index, &query_vec, k, results, distance_type, NULL, NULL);
    } else if (db->index_type == GV_INDEX_TYPE_RABITQ) {
        r = rabitq_search(db->hnsw_index, &query_vec, k, results, distance_type, NULL, NULL);
    }
    pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);

    uint64_t end_time_us = db_get_time_us();
    uint64_t latency_us = end_time_us - start_time_us;
    db_record_latency((GV_Database *)db, latency_us, 0);

    return r;
}

/* MVCC visibility: a vector at SoA index `idx` is visible to a reader whose
 * snapshot is `snapshot` when it was created at or before the snapshot and not
 * yet deleted as of the snapshot. Pre-transaction vectors (version 0) are always
 * visible. */
static int db_vector_visible(const GV_Database *db, size_t idx, uint64_t snapshot) {
    const GV_SoAStorage *s = db->soa_storage;
    if (!s || idx >= s->count) return 1;   /* non-SoA indexes: no MVCC stamps, always visible */
    uint64_t cv = s->create_version[idx];
    uint64_t dv = s->delete_version[idx];
    if (cv > snapshot) return 0;                       /* created after our snapshot */
    if (dv != 0 && dv <= snapshot) return 0;           /* deleted at/before our snapshot */
    return 1;
}

/* Snapshot read: search the committed index, then drop results not visible at
 * `snapshot`. When the database has never run a transaction (commit_version == 0)
 * every version stamp is 0, so this is a pure pass-through with zero overhead. */
int db_search_at_version(const GV_Database *db, const float *query_data, size_t k,
                         GV_SearchResult *results, GV_DistanceType distance_type,
                         uint64_t snapshot) {
    if (db == NULL || query_data == NULL || results == NULL || k == 0) return -1;
    if (db->commit_version == 0) return db_search_raw(db, query_data, k, results, distance_type);

    /* Over-fetch so filtering still returns up to k visible neighbours. */
    size_t fetch = k * 2 + 16;
    GV_SearchResult *tmp = (GV_SearchResult *)gv_alloc(fetch * sizeof(GV_SearchResult));
    if (!tmp) return db_search_raw(db, query_data, k, results, distance_type);

    int n = db_search_raw(db, query_data, fetch, tmp, distance_type);
    if (n < 0) { gv_free(tmp); return n; }

    /* The visibility filter reads the per-vector create/delete-version arrays,
     * which a concurrent insert can realloc under the write lock. Hold the read
     * lock across the filter so those reads are stable. (db_search_raw manages
     * the lock itself, so it must run outside this critical section.) */
    pthread_rwlock_rdlock((pthread_rwlock_t *)&db->rwlock);
    size_t kept = 0;
    for (int i = 0; i < n; i++) {
        if (kept < k && db_vector_visible(db, tmp[i].id, snapshot)) {
            results[kept++] = tmp[i];          /* move (incl. owned .vector) */
        } else if (tmp[i].vector) {
            vector_destroy((GV_Vector *)tmp[i].vector);   /* drop invisible/extra */
        }
    }
    pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
    gv_free(tmp);
    return (int)kept;
}

int db_search(const GV_Database *db, const float *query_data, size_t k,
              GV_SearchResult *results, GV_DistanceType distance_type) {
    /* Non-transactional reads see the latest committed snapshot. */
    uint64_t snap = db ? db->commit_version : 0;
    int n = db_search_at_version(db, query_data, k, results, distance_type, snap);
    /* Phase 4: record accesses so the hot working set drives tier promotion. */
    if (n > 0 && db && db->tiering_enabled && db->tiered_storage) {
        for (int i = 0; i < n; i++)
            gv_db_record_vector_access((GV_Database *)db, results[i].id);
    }
    return n;
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

typedef struct {
    const GV_Database *db;
    const float *queries;
    GV_SearchResult *results;
    size_t start;
    size_t count;
    size_t k;
    GV_DistanceType distance_type;
    int error;
} BatchSearchJob;

static void *db_batch_search_worker(void *arg) {
    BatchSearchJob *job = (BatchSearchJob *)arg;
    const GV_Database *db = job->db;
    size_t dim = db->dimension;

    for (size_t i = 0; i < job->count; i++) {
        GV_Vector qv;
        qv.dimension = dim;
        qv.metadata = NULL;
        qv.data = (float *)(job->queries + (job->start + i) * dim);
        GV_SearchResult *slot = job->results + (job->start + i) * job->k;
        int r = -1;

        if (db->index_type == GV_INDEX_TYPE_KDTREE) {
            r = db->soa_storage
                    ? kdtree_knn_search(db->root, db->soa_storage, &qv, job->k, slot, job->distance_type)
                    : -1;
        } else if (db->index_type == GV_INDEX_TYPE_HNSW) {
            r = gv_hnsw_search(db->hnsw_index, &qv, job->k, slot, job->distance_type, NULL, NULL);
        } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
            r = gv_ivfpq_search(db->hnsw_index, &qv, job->k, slot, job->distance_type, 0, 0);
        } else if (db->index_type == GV_INDEX_TYPE_FLAT) {
            r = flat_search(db->hnsw_index, &qv, job->k, slot, job->distance_type, NULL, NULL);
        } else if (db->index_type == GV_INDEX_TYPE_IVFFLAT) {
            r = ivfflat_search(db->hnsw_index, &qv, job->k, slot, job->distance_type, NULL, NULL);
        } else if (db->index_type == GV_INDEX_TYPE_IVFSQ8) {
            r = ivfsq8_search(db->hnsw_index, &qv, job->k, slot, job->distance_type, NULL, NULL);
        } else if (db->index_type == GV_INDEX_TYPE_IVFTURBOQUANT) {
            r = ivfturboquant_search(db->hnsw_index, &qv, job->k, slot, job->distance_type, NULL, NULL);
        } else if (db->index_type == GV_INDEX_TYPE_PQ) {
            r = pq_search(db->hnsw_index, &qv, job->k, slot, job->distance_type, NULL, NULL);
        } else if (db->index_type == GV_INDEX_TYPE_LSH) {
            r = lsh_search(db->hnsw_index, &qv, job->k, slot, job->distance_type, NULL, NULL);
        }

        if (r < 0) {
            job->error = 1;
            return NULL;
        }
    }
    return NULL;
}

int db_search_batch(const GV_Database *db, const float *queries, size_t qcount, size_t k,
                       GV_SearchResult *results, GV_DistanceType distance_type) {
    if (db == NULL || queries == NULL || results == NULL || qcount == 0 || k == 0) {
        return -1;
    }
    gv_tls_arena_reset();
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
         db->index_type == GV_INDEX_TYPE_IVFSQ8 ||
         db->index_type == GV_INDEX_TYPE_IVFTURBOQUANT ||
         db->index_type == GV_INDEX_TYPE_PQ ||
         db->index_type == GV_INDEX_TYPE_LSH ||
         db->index_type == GV_INDEX_TYPE_RABITQ) && db->hnsw_index == NULL) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return 0;
    }

#ifdef _WIN32
    long ncpu = 1;
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    ncpu = (long)si.dwNumberOfProcessors;
#else
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
#endif
    if (ncpu < 1) ncpu = 1;

    size_t nthreads = (size_t)ncpu < qcount ? (size_t)ncpu : qcount;

    pthread_t *tids = (pthread_t *)malloc(nthreads * sizeof(pthread_t));
    BatchSearchJob *jobs = (BatchSearchJob *)malloc(nthreads * sizeof(BatchSearchJob));
    if (!tids || !jobs) {
        free(tids);
        free(jobs);
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return -1;
    }

    size_t base = qcount / nthreads;
    size_t rem  = qcount % nthreads;
    size_t offset = 0;
    /* Compact array of successfully-created thread handles to join. Using tids
     * as a dense array (indexed by thread number, not chunk index) means we
     * never join uninitialized slots and never leak a real thread on partial
     * pthread_create failure. */
    size_t njoin = 0;

    for (size_t t = 0; t < nthreads; t++) {
        size_t chunk = base + (t < rem ? 1 : 0);
        jobs[t].db            = db;
        jobs[t].queries       = queries;
        jobs[t].results       = results;
        jobs[t].start         = offset;
        jobs[t].count         = chunk;
        jobs[t].k             = k;
        jobs[t].distance_type = distance_type;
        jobs[t].error         = 0;
        offset += chunk;

        if (chunk == 0) continue;
        if (pthread_create(&tids[njoin], NULL, db_batch_search_worker, &jobs[t]) != 0) {
            jobs[t].error = 1;
            /* run this chunk on caller thread; do NOT record a thread handle */
            db_batch_search_worker(&jobs[t]);
        } else {
            njoin++;
        }
    }

    /* Join exactly the threads we created. jobs/results stay live until every
     * real thread has been joined below. */
    for (size_t t = 0; t < njoin; t++) {
        pthread_join(tids[t], NULL);
    }

    int had_error = 0;
    for (size_t t = 0; t < nthreads; t++) {
        if (jobs[t].error) had_error = 1;
    }

    free(tids);
    free(jobs);
    pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);

    return had_error ? -1 : (int)(qcount * k);
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
         db->index_type == GV_INDEX_TYPE_IVFSQ8 ||
         db->index_type == GV_INDEX_TYPE_IVFTURBOQUANT ||
         db->index_type == GV_INDEX_TYPE_PQ ||
         db->index_type == GV_INDEX_TYPE_LSH ||
         db->index_type == GV_INDEX_TYPE_RABITQ) && db->hnsw_index == NULL) {
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
    } else if (db->index_type == GV_INDEX_TYPE_IVFSQ8) {
        int r = ivfsq8_search(db->hnsw_index, &query_vec, k, results, distance_type,
                              filter_key, filter_value);
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return r;
    } else if (db->index_type == GV_INDEX_TYPE_IVFTURBOQUANT) {
        int r = ivfturboquant_search(db->hnsw_index, &query_vec, k, results, distance_type,
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
    } else if (db->index_type == GV_INDEX_TYPE_RABITQ) {
        int r = rabitq_search(db->hnsw_index, &query_vec, k, results, distance_type,
                            filter_key, filter_value);
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return r;
    } else if (db->index_type == GV_INDEX_TYPE_IVFPQ) {
        /* No native filter; apply post-filter on results */
        int tmp_on_heap = 0;
        GV_SearchResult *tmp = (GV_SearchResult *)gv_tls_calloc_or_heap(
            k, sizeof(GV_SearchResult), &tmp_on_heap);
        if (!tmp) {
            pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
            return -1;
        }
        int r = gv_ivfpq_search(db->hnsw_index, &query_vec, k, tmp, distance_type, 0, 0);
        if (r <= 0) {
            gv_tls_free_or_heap(tmp, tmp_on_heap);
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
        gv_tls_free_or_heap(tmp, tmp_on_heap);
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return out;
    }
    pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
    return -1;
}

static double db_estimate_filter_selectivity(const GV_Database *db, const char *filter_expr) {
    if (!db || !filter_expr || db->count == 0 || !db->metadata_index) {
        return 1.0;
    }

    const char *p = filter_expr;
    while (*p == ' ' || *p == '\t') {
        ++p;
    }

    char key[128];
    char val[256];
    if (sscanf(p, "%127[^ =] == \"%255[^\"]\"", key, val) == 2) {
        size_t matches = metadata_index_count(db->metadata_index, key, val);
        if (matches > 0) {
            double sel = (double)matches / (double)db->count;
            return sel < 1.0 ? sel : 1.0;
        }
        return 0.01;
    }

    return 0.05;
}

static size_t db_filter_search_candidates(const GV_Database *db, size_t k,
                                          const char *filter_expr) {
    if (!db || k == 0) {
        return k;
    }

    double selectivity = db_estimate_filter_selectivity(db, filter_expr);
    size_t max_candidates = k * 4;
    GV_QueryOptimizer *opt = optimizer_create();
    if (opt) {
        GV_CollectionStats stats;
        memset(&stats, 0, sizeof(stats));
        stats.total_vectors = db->count;
        stats.dimension = db->dimension;
        stats.index_type = (int)db->index_type;
        if (selectivity > 0.0) {
            stats.avg_vectors_per_filter_match = db->count * selectivity;
        }
        optimizer_update_stats(opt, &stats);

        GV_QueryPlan plan;
        if (optimizer_plan(opt, k, 1, selectivity, &plan) == 0) {
            if (plan.strategy == GV_PLAN_EXACT_SCAN) {
                max_candidates = db->count;
            } else if (plan.strategy == GV_PLAN_OVERSAMPLE_FILTER && plan.oversample_k > 0) {
                max_candidates = plan.oversample_k;
            } else if (plan.oversample_k > 0) {
                max_candidates = plan.oversample_k;
            }
        }
        optimizer_destroy(opt);
    }

    if (max_candidates < k) {
        max_candidates = k;
    }
    if (max_candidates > db->count) {
        max_candidates = db->count;
    }
    return max_candidates;
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
         db->index_type == GV_INDEX_TYPE_IVFSQ8 ||
         db->index_type == GV_INDEX_TYPE_IVFTURBOQUANT ||
         db->index_type == GV_INDEX_TYPE_PQ ||
         db->index_type == GV_INDEX_TYPE_LSH ||
         db->index_type == GV_INDEX_TYPE_RABITQ) && db->hnsw_index == NULL) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        filter_destroy(filter);
        return 0;
    }

    size_t max_candidates = db_filter_search_candidates(db, k, filter_expr);
    if (max_candidates == 0) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        filter_destroy(filter);
        return 0;
    }

    int tmp_on_heap = 0;
    GV_SearchResult *tmp = (GV_SearchResult *)gv_tls_calloc_or_heap(
        max_candidates, sizeof(GV_SearchResult), &tmp_on_heap);
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
    } else if (db->index_type == GV_INDEX_TYPE_IVFSQ8) {
        n = ivfsq8_search(db->hnsw_index, &query_vec, max_candidates, tmp, distance_type, NULL, NULL);
    } else if (db->index_type == GV_INDEX_TYPE_IVFTURBOQUANT) {
        n = ivfturboquant_search(db->hnsw_index, &query_vec, max_candidates, tmp, distance_type, NULL, NULL);
    } else if (db->index_type == GV_INDEX_TYPE_PQ) {
        n = pq_search(db->hnsw_index, &query_vec, max_candidates, tmp, distance_type, NULL, NULL);
    } else if (db->index_type == GV_INDEX_TYPE_LSH) {
        n = lsh_search(db->hnsw_index, &query_vec, max_candidates, tmp, distance_type, NULL, NULL);
    } else {
        gv_tls_free_or_heap(tmp, tmp_on_heap);
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        filter_destroy(filter);
        return -1;
    }

    if (n <= 0) {
        gv_tls_free_or_heap(tmp, tmp_on_heap);
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        filter_destroy(filter);
        return n;
    }

    size_t out = 0;
    for (int i = 0; i < n && out < k; ++i) {
        int match = filter_eval(filter, tmp[i].vector);
        if (match < 0) {
            /* Every candidate's .vector is a fresh owned copy; free them all. */
            for (int j = 0; j < n; ++j) {
                vector_destroy((GV_Vector *)tmp[j].vector);
                tmp[j].vector = NULL;
            }
            gv_tls_free_or_heap(tmp, tmp_on_heap);
            pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
            filter_destroy(filter);
            return -1;
        }
        if (match == 1) {
            results[out++] = tmp[i];
            /* Ownership transferred to caller; don't free below. */
            tmp[i].vector = NULL;
        }
    }

    /* Free the owned vectors of every candidate NOT transferred into results
     * (non-matching, plus any left unexamined when out reached k). */
    for (int i = 0; i < n; ++i) {
        vector_destroy((GV_Vector *)tmp[i].vector);
        tmp[i].vector = NULL;
    }

    gv_tls_free_or_heap(tmp, tmp_on_heap);
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
         db->index_type == GV_INDEX_TYPE_IVFSQ8 ||
         db->index_type == GV_INDEX_TYPE_IVFTURBOQUANT ||
         db->index_type == GV_INDEX_TYPE_PQ ||
         db->index_type == GV_INDEX_TYPE_LSH ||
         db->index_type == GV_INDEX_TYPE_RABITQ) && db->hnsw_index == NULL) {
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
    } else if (db->index_type == GV_INDEX_TYPE_IVFSQ8) {
        r = ivfsq8_range_search(db->hnsw_index, &query_vec, radius, results, max_results, distance_type, NULL, NULL);
    } else if (db->index_type == GV_INDEX_TYPE_IVFTURBOQUANT) {
        r = ivfturboquant_range_search(db->hnsw_index, &query_vec, radius, results, max_results, distance_type, NULL, NULL);
    } else if (db->index_type == GV_INDEX_TYPE_PQ) {
        r = pq_range_search(db->hnsw_index, &query_vec, radius, results, max_results, distance_type, NULL, NULL);
    } else if (db->index_type == GV_INDEX_TYPE_LSH) {
        r = lsh_range_search(db->hnsw_index, &query_vec, radius, results, max_results, distance_type, NULL, NULL);
    } else if (db->index_type == GV_INDEX_TYPE_RABITQ) {
        r = rabitq_range_search(db->hnsw_index, &query_vec, radius, results, max_results, distance_type, NULL, NULL);
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
         db->index_type == GV_INDEX_TYPE_IVFSQ8 ||
         db->index_type == GV_INDEX_TYPE_IVFTURBOQUANT ||
         db->index_type == GV_INDEX_TYPE_PQ ||
         db->index_type == GV_INDEX_TYPE_LSH ||
         db->index_type == GV_INDEX_TYPE_RABITQ) && db->hnsw_index == NULL) {
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
    } else if (db->index_type == GV_INDEX_TYPE_IVFSQ8) {
        r = ivfsq8_range_search(db->hnsw_index, &query_vec, radius, results, max_results,
                                distance_type, filter_key, filter_value);
        pthread_rwlock_unlock((pthread_rwlock_t *)&db->rwlock);
        return r;
    } else if (db->index_type == GV_INDEX_TYPE_IVFTURBOQUANT) {
        r = ivfturboquant_range_search(db->hnsw_index, &query_vec, radius, results, max_results,
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
    } else if (db->index_type == GV_INDEX_TYPE_RABITQ) {
        r = rabitq_range_search(db->hnsw_index, &query_vec, radius, results, max_results,
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

