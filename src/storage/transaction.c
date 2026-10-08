#include "storage/transaction.h"
#include "storage/soa_storage.h"
#include "storage/wal.h"
#include "storage/db_internal.h"   /* db_estimate_vector_memory */
#include "schema/vector.h"
#include "core/memory.h"
#include "core/log.h"

#include <string.h>
#include <pthread.h>

struct GV_DBTxn {
    GV_Database *db;
    uint64_t read_version;
    GV_TxnState state;
    size_t dimension;
    float **ins_data;  size_t ins_n, ins_cap;   /* staged inserts (owned copies) */
    size_t *del_idx;   size_t del_n, del_cap;    /* staged deletes (committed SoA indices) */
};

GV_DBTxn *db_begin(GV_Database *db) {
    if (!db) return NULL;
    GV_DBTxn *t = (GV_DBTxn *)gv_calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->db = db;
    t->dimension = db->dimension;
    t->state = GV_TXN_ACTIVE;
    /* Snapshot under the commit lock so we pair cleanly with in-flight commits,
     * and register this snapshot's read_version so GC never reclaims a version
     * still visible to us. */
    pthread_mutex_lock(&db->txn_mutex);
    t->read_version = __atomic_load_n(&db->commit_version, __ATOMIC_SEQ_CST);
    if (db->txn_active_count == db->txn_active_cap) {
        size_t nc = db->txn_active_cap ? db->txn_active_cap * 2 : 8;
        uint64_t *na = (uint64_t *)gv_realloc(db->txn_active_versions, nc * sizeof(uint64_t));
        if (!na) { pthread_mutex_unlock(&db->txn_mutex); gv_free(t); return NULL; }
        db->txn_active_versions = na;
        db->txn_active_cap = nc;
    }
    db->txn_active_versions[db->txn_active_count++] = t->read_version;
    pthread_mutex_unlock(&db->txn_mutex);
    return t;
}

GV_TxnState db_txn_state(const GV_DBTxn *t) { return t ? t->state : GV_TXN_ABORTED; }
uint64_t db_txn_read_version(const GV_DBTxn *t) { return t ? t->read_version : 0; }

int db_txn_add_vector(GV_DBTxn *t, const float *data, size_t dimension) {
    if (!t || !data || t->state != GV_TXN_ACTIVE || dimension != t->dimension) return -1;
    if (t->ins_n == t->ins_cap) {
        size_t nc = t->ins_cap ? t->ins_cap * 2 : 8;
        float **na = (float **)gv_realloc(t->ins_data, nc * sizeof(float *));
        if (!na) return -1;
        t->ins_data = na; t->ins_cap = nc;
    }
    float *copy = (float *)gv_alloc(dimension * sizeof(float));
    if (!copy) return -1;
    memcpy(copy, data, dimension * sizeof(float));
    t->ins_data[t->ins_n++] = copy;
    return 0;
}

int db_txn_delete(GV_DBTxn *t, size_t vector_index) {
    if (!t || t->state != GV_TXN_ACTIVE) return -1;
    if (t->del_n == t->del_cap) {
        size_t nc = t->del_cap ? t->del_cap * 2 : 8;
        size_t *nd = (size_t *)gv_realloc(t->del_idx, nc * sizeof(size_t));
        if (!nd) return -1;
        t->del_idx = nd; t->del_cap = nc;
    }
    t->del_idx[t->del_n++] = vector_index;
    return 0;
}

static int txn_is_deleted_locally(const GV_DBTxn *t, size_t idx) {
    for (size_t i = 0; i < t->del_n; i++) if (t->del_idx[i] == idx) return 1;
    return 0;
}

/* Remove one occurrence of this txn's read_version from the active set. Called
 * from txn_free on every terminal path (commit/rollback/conflict/abort), always
 * without txn_mutex held. */
static void txn_deregister(GV_DBTxn *t) {
    GV_Database *db = t->db;
    if (!db) return;
    pthread_mutex_lock(&db->txn_mutex);
    for (size_t i = 0; i < db->txn_active_count; i++) {
        if (db->txn_active_versions[i] == t->read_version) {
            db->txn_active_versions[i] = db->txn_active_versions[db->txn_active_count - 1];
            db->txn_active_count--;
            break;
        }
    }
    pthread_mutex_unlock(&db->txn_mutex);
}

static void txn_free(GV_DBTxn *t) {
    txn_deregister(t);
    for (size_t i = 0; i < t->ins_n; i++) gv_free(t->ins_data[i]);
    gv_free(t->ins_data);
    gv_free(t->del_idx);
    gv_free(t);
}

int db_txn_search(GV_DBTxn *t, const float *query, size_t k,
                  GV_SearchResult *results, GV_DistanceType metric) {
    if (!t || !query || !results || k == 0 || t->state != GV_TXN_ACTIVE) return -1;
    size_t dim = t->dimension;

    /* Committed candidates visible at our snapshot, over-fetched to survive the
     * local-delete filter. */
    size_t fetch = k + t->del_n + t->ins_n + 8;
    GV_SearchResult *cand = (GV_SearchResult *)gv_calloc(fetch, sizeof(GV_SearchResult));
    if (!cand) return -1;
    int n = db_search_at_version(t->db, query, fetch, cand, metric, t->read_version);
    if (n < 0) { gv_free(cand); return -1; }

    /* Merge: committed (minus our staged deletes) + our staged inserts, by distance. */
    GV_Vector q; q.dimension = dim; q.data = (float *)query; q.metadata = NULL;
    size_t out = 0;
    /* Simple insertion into results[] kept ascending by distance. */
    #define PUSH(_dist, _id, _vec) do { \
        float d_ = (_dist); \
        if (out < k) { \
            size_t p = out; \
            while (p > 0 && results[p-1].distance > d_) { results[p] = results[p-1]; p--; } \
            results[p].distance = d_; results[p].id = (_id); results[p].vector = (_vec); \
            results[p].sparse_vector = NULL; results[p].is_sparse = 0; out++; \
        } else if (d_ < results[k-1].distance) { \
            if (results[k-1].vector) vector_destroy((GV_Vector *)results[k-1].vector); \
            size_t p = k-1; \
            while (p > 0 && results[p-1].distance > d_) { results[p] = results[p-1]; p--; } \
            results[p].distance = d_; results[p].id = (_id); results[p].vector = (_vec); \
            results[p].sparse_vector = NULL; results[p].is_sparse = 0; \
        } else if (_vec) { vector_destroy((GV_Vector *)(_vec)); } \
    } while (0)

    for (int i = 0; i < n; i++) {
        if (txn_is_deleted_locally(t, cand[i].id)) {
            if (cand[i].vector) vector_destroy((GV_Vector *)cand[i].vector);
            continue;
        }
        PUSH(cand[i].distance, cand[i].id, (GV_Vector *)cand[i].vector);
    }
    /* Staged inserts get a synthetic id (top of the range) so callers can spot them. */
    for (size_t j = 0; j < t->ins_n; j++) {
        GV_Vector v; v.dimension = dim; v.data = t->ins_data[j]; v.metadata = NULL;
        float d = distance(&v, &q, metric);
        if (d < 0.0f) continue;
        GV_Vector *copy = vector_create_from_data(dim, t->ins_data[j]);
        PUSH(d, (size_t)-1 - j, copy);
    }
    #undef PUSH
    gv_free(cand);
    return (int)out;
}

int db_commit(GV_DBTxn *t) {
    if (!t) return -1;
    if (t->state != GV_TXN_ACTIVE) { int s = t->state; (void)s; return -1; }
    GV_Database *db = t->db;

    pthread_mutex_lock(&db->txn_mutex);

    /* Write-write conflict check: any staged delete whose target was deleted by a
     * transaction that committed after our snapshot is a conflict. */
    for (size_t i = 0; i < t->del_n; i++) {
        uint64_t dv = soa_storage_delete_version(db->soa_storage, t->del_idx[i]);
        if (dv != 0 && dv > t->read_version) {
            pthread_mutex_unlock(&db->txn_mutex);
            t->state = GV_TXN_ABORTED;
            txn_free(t);
            return GV_TXN_CONFLICT;
        }
    }

    /* Read-only transaction: nothing to apply. */
    if (t->ins_n == 0 && t->del_n == 0) {
        pthread_mutex_unlock(&db->txn_mutex);
        t->state = GV_TXN_COMMITTED;
        txn_free(t);
        return GV_TXN_OK;
    }

    /* Atomic admission control: reject the whole commit up front if applying
     * every staged insert would breach a resource limit (max_vectors /
     * max_memory). This makes the common, deterministic failure all-or-nothing
     * instead of leaving a partially-applied transaction. (A rare mid-loop OOM
     * is still reported as a hard failure below.) */
    if (t->ins_n > 0) {
        size_t per_vec = db_estimate_vector_memory(t->dimension);
        if (db_check_resource_limits(db, t->ins_n, per_vec * t->ins_n) != 0) {
            pthread_mutex_unlock(&db->txn_mutex);
            t->state = GV_TXN_ABORTED;
            txn_free(t);
            return -1;
        }
    }

    /* Allocate the delete-index buffer BEFORE applying anything: if this fails
     * we must abort before any in-memory change, otherwise deletes would be
     * applied but omitted from the atomic TXN record and lost on replay. */
    uint64_t *txn_dels = NULL;
    if (t->del_n > 0) {
        txn_dels = (uint64_t *)gv_alloc(t->del_n * sizeof(uint64_t));
        if (!txn_dels) {
            pthread_mutex_unlock(&db->txn_mutex);
            t->state = GV_TXN_ABORTED;
            txn_free(t);
            return -1;
        }
    }

    uint64_t cv = __atomic_add_fetch(&db->commit_version, 1, __ATOMIC_SEQ_CST);

    /* Apply staged inserts to the index ONLY: db_set_wal_suppress stops
     * db_add_vector from writing its own per-insert WAL record, so the entire
     * transaction is logged as a single atomic TXN record below (crash-atomic).
     * db_add_vector still stamps each slot's MVCC create_version with `cv`. */
    db_set_commit_stamp(cv);
    db_set_wal_suppress(1);
    int apply_ok = 1;
    for (size_t i = 0; i < t->ins_n; i++) {
        if (db_add_vector(db, t->ins_data[i], t->dimension) != 0) { apply_ok = 0; break; }
    }
    db_set_wal_suppress(0);
    db_set_commit_stamp(0);
    if (!apply_ok) {
        gv_free(txn_dels);
        pthread_mutex_unlock(&db->txn_mutex);
        t->state = GV_TXN_ABORTED;
        txn_free(t);
        return -1;
    }

    /* Apply staged deletes as MVCC tombstones (older snapshots still see them),
     * collecting the indices we actually tombstoned so they go into the atomic
     * TXN record. No per-delete WAL record is written here. */
    size_t txn_del_n = 0;
    pthread_rwlock_wrlock(&db->rwlock);
    for (size_t i = 0; i < t->del_n; i++) {
        uint64_t dv = soa_storage_delete_version(db->soa_storage, t->del_idx[i]);
        if (dv == 0) {
            soa_storage_set_delete_version(db->soa_storage, t->del_idx[i], cv);
            if (txn_dels) txn_dels[txn_del_n++] = (uint64_t)t->del_idx[i];
        }
    }
    pthread_rwlock_unlock(&db->rwlock);

    /* One atomic WAL record for the whole transaction. A crash before this fsync
     * loses the entire transaction (none replayed); a crash after replays it in
     * full — crash-atomic. In-memory databases (no WAL) skip logging. */
    if (db->wal != NULL) {
        pthread_mutex_lock(&db->wal_mutex);
        if (wal_append_txn(db->wal, (const float *const *)t->ins_data, t->dimension,
                           t->ins_n, txn_dels, txn_del_n) != 0) {
            /* Durability failure: the in-memory commit stands but is not logged
             * (mirrors db_add_vector's non-fatal WAL-error semantics). Surface it. */
            GV_LOG_ERROR("db_commit: wal_append_txn failed - transaction not durable");
        }
        pthread_mutex_unlock(&db->wal_mutex);
    }
    gv_free(txn_dels);

    pthread_mutex_unlock(&db->txn_mutex);
    t->state = GV_TXN_COMMITTED;
    txn_free(t);
    return GV_TXN_OK;
}

int db_rollback(GV_DBTxn *t) {
    if (!t) return -1;
    t->state = GV_TXN_ABORTED;
    txn_free(t);
    return 0;
}

uint64_t db_txn_min_active_version(GV_Database *db) {
    if (!db) return 0;
    pthread_mutex_lock(&db->txn_mutex);
    uint64_t min = 0; int have = 0;
    for (size_t i = 0; i < db->txn_active_count; i++) {
        uint64_t v = db->txn_active_versions[i];
        if (!have || v < min) { min = v; have = 1; }
    }
    pthread_mutex_unlock(&db->txn_mutex);
    return have ? min : 0;   /* 0 = no active transactions */
}

size_t db_txn_gc_auto(GV_Database *db) {
    if (!db || !db->soa_storage) return 0;
    /* Compute the safe point atomically: the smallest read_version any live
     * snapshot holds (so tombstones deleted at <= that version are invisible to
     * all of them), or commit_version+1 when no transaction is active. */
    pthread_mutex_lock(&db->txn_mutex);
    uint64_t cv = __atomic_load_n(&db->commit_version, __ATOMIC_SEQ_CST);
    uint64_t min = 0; int have = 0;
    for (size_t i = 0; i < db->txn_active_count; i++) {
        uint64_t v = db->txn_active_versions[i];
        if (!have || v < min) { min = v; have = 1; }
    }
    pthread_mutex_unlock(&db->txn_mutex);
    uint64_t safe_below = have ? (min + 1) : (cv + 1);
    return db_txn_gc(db, safe_below);
}

size_t db_txn_gc(GV_Database *db, uint64_t safe_below) {
    if (!db || !db->soa_storage) return 0;
    size_t reclaimed = 0;
    size_t n = db->soa_storage->count;
    for (size_t i = 0; i < n; i++) {
        uint64_t dv = soa_storage_delete_version(db->soa_storage, i);
        if (dv != 0 && dv < safe_below && soa_storage_is_deleted(db->soa_storage, i) == 0) {
            if (db_delete_vector_by_index(db, i) == 0) reclaimed++;
        }
    }
    return reclaimed;
}
