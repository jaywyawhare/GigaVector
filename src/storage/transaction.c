#include "storage/transaction.h"
#include "storage/soa_storage.h"
#include "storage/wal.h"
#include "schema/vector.h"
#include "core/memory.h"

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
    /* Snapshot under the commit lock so we pair cleanly with in-flight commits. */
    pthread_mutex_lock(&db->txn_mutex);
    t->read_version = db->commit_version;
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

static void txn_free(GV_DBTxn *t) {
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

    uint64_t cv = ++db->commit_version;

    /* Apply staged inserts. db_add_vector stamps each inserted slot with `cv`
     * atomically under its own write lock via the thread-local commit stamp, so
     * the right slot is tagged even if a non-transactional insert interleaves.
     * If any insert fails the commit is aborted (best-effort: earlier inserts of
     * this transaction may remain — a hard failure is reported, not GV_TXN_OK). */
    db_set_commit_stamp(cv);
    int apply_ok = 1;
    for (size_t i = 0; i < t->ins_n; i++) {
        if (db_add_vector(db, t->ins_data[i], t->dimension) != 0) { apply_ok = 0; break; }
    }
    db_set_commit_stamp(0);
    if (!apply_ok) {
        pthread_mutex_unlock(&db->txn_mutex);
        t->state = GV_TXN_ABORTED;
        txn_free(t);
        return -1;
    }

    /* Apply staged deletes as MVCC tombstones (older snapshots still see them).
     * Also WAL-log each as a delete so a crash before the next snapshot replays it:
     * post-recovery there are no older snapshots, so a hard delete is the correct
     * materialisation of the tombstone. */
    pthread_rwlock_wrlock(&db->rwlock);
    for (size_t i = 0; i < t->del_n; i++) {
        uint64_t dv = soa_storage_delete_version(db->soa_storage, t->del_idx[i]);
        if (dv == 0) {
            soa_storage_set_delete_version(db->soa_storage, t->del_idx[i], cv);
            if (db->wal) {
                pthread_mutex_lock(&db->wal_mutex);
                wal_append_delete(db->wal, t->del_idx[i]);
                pthread_mutex_unlock(&db->wal_mutex);
            }
        }
    }
    pthread_rwlock_unlock(&db->rwlock);

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
