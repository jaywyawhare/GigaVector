/**
 * @file gv_mvcc.c
 * @brief Multi-Version Concurrency Control (MVCC) implementation.
 */

#include "gigavector/gv_mvcc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdint.h>

/*  Internal Structures  */

/**
 * @brief Linked-list node tracking active transaction IDs.
 */
typedef struct GV_TxnEntry {
    uint64_t txn_id;
    struct GV_TxnEntry *next;
} GV_TxnEntry;

/**
 * @brief Initial capacity for dynamic arrays.
 */
#define MVCC_INIT_CAP 64

/**
 * @brief MVCC Manager internal structure.
 */
struct GV_MVCCManager {
    size_t dimension;

    /* Version store (dynamic array) */
    GV_MVCCVersion *versions;
    size_t ver_count;
    size_t ver_capacity;

    /* Transaction ID generator (monotonically increasing) */
    uint64_t next_txn_id;

    /* Linked list of active transaction IDs */
    GV_TxnEntry *active_txns;

    /* Mutex protecting all shared state */
    pthread_mutex_t mutex;
};

/**
 * @brief Transaction internal structure.
 */
struct GV_Transaction {
    GV_MVCCManager *mgr;
    uint64_t txn_id;
    GV_TxnStatus status;

    /* Indices into mgr->versions[] that this txn created */
    size_t *added_indices;
    size_t added_count;
    size_t added_capacity;

    /* Indices into mgr->versions[] that this txn marked for deletion */
    size_t *deleted_indices;
    size_t deleted_count;
    size_t deleted_capacity;

    /* Snapshot: highest txn_id that was committed when this txn began.
     * Used for visibility decisions. */
    uint64_t snapshot_txn_id;
};

/*  Internal Helpers  */

/**
 * @brief Check whether a given txn_id is present in the active transaction list.
 *
 * Must be called while mgr->mutex is held.
 */
static int mvcc_is_active(const GV_MVCCManager *mgr, uint64_t txn_id)
{
    const GV_TxnEntry *e = mgr->active_txns;
    while (e) {
        if (e->txn_id == txn_id)
            return 1;
        e = e->next;
    }
    return 0;
}

/**
 * @brief Remove a txn_id from the active transaction list.
 *
 * Must be called while mgr->mutex is held.
 */
static void mvcc_remove_active(GV_MVCCManager *mgr, uint64_t txn_id)
{
    GV_TxnEntry **pp = &mgr->active_txns;
    while (*pp) {
        if ((*pp)->txn_id == txn_id) {
            GV_TxnEntry *tmp = *pp;
            *pp = tmp->next;
            free(tmp);
            return;
        }
        pp = &(*pp)->next;
    }
}

/**
 * @brief Determine the minimum snapshot_txn_id across all active transactions.
 *
 * This is the "low-water mark": any version deleted by a committed txn whose
 * txn_id is below this value is invisible to every active transaction, and can
 * be garbage-collected.
 *
 * Must be called while mgr->mutex is held.
 * Returns UINT64_MAX if there are no active transactions.
 */
static uint64_t mvcc_min_active_snapshot(const GV_MVCCManager *mgr)
{
    (void)mgr;
    /* We don't store snapshot_txn_id in the active list directly; however the
     * snapshot_txn_id for a given txn is always txn_id - 1.  So the minimum
     * snapshot across active txns equals (min_active_txn_id - 1).  To keep
     * the GC logic straightforward we return the minimum active txn_id itself
     * and let the caller treat versions deleted by txns < this value as
     * reclaimable. */
    uint64_t min_id = UINT64_MAX;
    const GV_TxnEntry *e = mgr->active_txns;
    while (e) {
        if (e->txn_id < min_id)
            min_id = e->txn_id;
        e = e->next;
    }
    return min_id;
}

/**
 * @brief Ensure a version array has room for one more entry.
 */
static int mvcc_grow_versions(GV_MVCCManager *mgr)
{
    if (mgr->ver_count < mgr->ver_capacity)
        return 0;

    size_t new_cap = mgr->ver_capacity == 0 ? MVCC_INIT_CAP : mgr->ver_capacity * 2;
    GV_MVCCVersion *tmp = realloc(mgr->versions, new_cap * sizeof(GV_MVCCVersion));
    if (!tmp)
        return -1;
    mgr->versions = tmp;
    mgr->ver_capacity = new_cap;
    return 0;
}

/**
 * @brief Append an index to a dynamic size_t array (used for added/deleted tracking).
 */
static int idx_array_push(size_t **arr, size_t *count, size_t *capacity, size_t value)
{
    if (*count >= *capacity) {
        size_t new_cap = *capacity == 0 ? MVCC_INIT_CAP : (*capacity) * 2;
        size_t *tmp = realloc(*arr, new_cap * sizeof(size_t));
        if (!tmp)
            return -1;
        *arr = tmp;
        *capacity = new_cap;
    }
    (*arr)[(*count)++] = value;
    return 0;
}

/**
 * @brief Internal visibility check (does not lock; caller must hold mutex or
 *        ensure consistency by other means).
 *
 * A version is visible to a reader whose snapshot is @p snapshot_id when:
 *   1. create_txn <= snapshot_id  (version was created at or before the snapshot)
 *   2. create_txn is committed    (not active, not aborted)
 *   3. delete_txn == 0            (never deleted)
 *      OR delete_txn > snapshot_id  (deleted after the snapshot)
 *      OR delete_txn is NOT committed (pending delete from another txn)
 *
 * Additionally, a transaction can always see its own uncommitted writes:
 *   - If create_txn == reader_txn_id, it is visible (own insert).
 *   - If delete_txn == reader_txn_id, it is NOT visible (own pending delete).
 *
 * @param mgr        The MVCC manager.
 * @param ver        The version to test.
 * @param reader_txn The transaction ID of the reader.
 * @param snapshot   The snapshot_txn_id of the reader.
 * @return 1 if visible, 0 otherwise.
 */
static int mvcc_version_visible(const GV_MVCCManager *mgr,
                                const GV_MVCCVersion *ver,
                                uint64_t reader_txn,
                                uint64_t snapshot)
{
    /* ---------- creation visibility ---------- */
    if (ver->create_txn == reader_txn) {
        /* Own write -- visible unless we also deleted it */
        if (ver->delete_txn == reader_txn)
            return 0;
        return 1;
    }

    /* Created after the snapshot -- not visible */
    if (ver->create_txn > snapshot)
        return 0;

    /* Created by an active (uncommitted) txn that is not ours -- not visible */
    if (mvcc_is_active(mgr, ver->create_txn))
        return 0;

    /* ---------- deletion visibility ---------- */
    if (ver->delete_txn == 0)
        return 1; /* not deleted */

    if (ver->delete_txn == reader_txn)
        return 0; /* we deleted it ourselves */

    if (ver->delete_txn > snapshot)
        return 1; /* deleted after our snapshot */

    /* Deleted by an active (uncommitted) txn -- treat as still alive */
    if (mvcc_is_active(mgr, ver->delete_txn))
        return 1;

    /* Deleted by a committed txn within our snapshot range -- not visible */
    return 0;
}

/*  Manager Lifecycle  */

GV_MVCCManager *gv_mvcc_create(size_t dimension)
{
    GV_MVCCManager *mgr = calloc(1, sizeof(GV_MVCCManager));
    if (!mgr)
        return NULL;

    mgr->dimension = dimension;
    mgr->versions = NULL;
    mgr->ver_count = 0;
    mgr->ver_capacity = 0;
    mgr->next_txn_id = 1; /* txn IDs start at 1 */
    mgr->active_txns = NULL;

    if (pthread_mutex_init(&mgr->mutex, NULL) != 0) {
        free(mgr);
        return NULL;
    }

    return mgr;
}

void gv_mvcc_destroy(GV_MVCCManager *mgr)
{
    if (!mgr)
        return;

    /* Free version data */
    for (size_t i = 0; i < mgr->ver_count; i++) {
        free(mgr->versions[i].data);
    }
    free(mgr->versions);

    /* Free active txn list */
    GV_TxnEntry *e = mgr->active_txns;
    while (e) {
        GV_TxnEntry *next = e->next;
        free(e);
        e = next;
    }

    pthread_mutex_destroy(&mgr->mutex);
    free(mgr);
}

/*  Transaction Lifecycle  */

GV_Transaction *gv_txn_begin(GV_MVCCManager *mgr)
{
    if (!mgr)
        return NULL;

    GV_Transaction *txn = calloc(1, sizeof(GV_Transaction));
    if (!txn)
        return NULL;

    txn->mgr = mgr;
    txn->status = GV_TXN_ACTIVE;
    txn->added_indices = NULL;
    txn->added_count = 0;
    txn->added_capacity = 0;
    txn->deleted_indices = NULL;
    txn->deleted_count = 0;
    txn->deleted_capacity = 0;

    pthread_mutex_lock(&mgr->mutex);

    txn->txn_id = mgr->next_txn_id++;
    /* Snapshot is everything committed before this txn */
    txn->snapshot_txn_id = txn->txn_id - 1;

    /* Add to active list */
    GV_TxnEntry *entry = malloc(sizeof(GV_TxnEntry));
    if (!entry) {
        pthread_mutex_unlock(&mgr->mutex);
        free(txn);
        return NULL;
    }
    entry->txn_id = txn->txn_id;
    entry->next = mgr->active_txns;
    mgr->active_txns = entry;

    pthread_mutex_unlock(&mgr->mutex);
    return txn;
}

int gv_txn_commit(GV_Transaction *txn)
{
    if (!txn || !txn->mgr)
        return -1;
    if (txn->status != GV_TXN_ACTIVE)
        return -1;

    GV_MVCCManager *mgr = txn->mgr;

    pthread_mutex_lock(&mgr->mutex);

    /* Pending deletes are already recorded in the version store (delete_txn
     * set to our txn_id).  When we commit, those become permanent -- no
     * further action needed on the versions themselves. */

    txn->status = GV_TXN_COMMITTED;
    mvcc_remove_active(mgr, txn->txn_id);

    pthread_mutex_unlock(&mgr->mutex);
    return 0;
}

int gv_txn_rollback(GV_Transaction *txn)
{
    if (!txn || !txn->mgr)
        return -1;
    if (txn->status != GV_TXN_ACTIVE)
        return -1;

    GV_MVCCManager *mgr = txn->mgr;

    pthread_mutex_lock(&mgr->mutex);

    /* Undo inserts: mark every version we created as immediately deleted so
     * that no future transaction will ever see them. */
    for (size_t i = 0; i < txn->added_count; i++) {
        size_t idx = txn->added_indices[i];
        if (idx < mgr->ver_count) {
            mgr->versions[idx].delete_txn = txn->txn_id;
        }
    }

    /* Undo pending deletes: clear the delete_txn we stamped */
    for (size_t i = 0; i < txn->deleted_count; i++) {
        size_t idx = txn->deleted_indices[i];
        if (idx < mgr->ver_count) {
            /* Only undo if we are still the one who marked it */
            if (mgr->versions[idx].delete_txn == txn->txn_id)
                mgr->versions[idx].delete_txn = 0;
        }
    }

    txn->status = GV_TXN_ABORTED;
    mvcc_remove_active(mgr, txn->txn_id);

    pthread_mutex_unlock(&mgr->mutex);
    return 0;
}

uint64_t gv_txn_id(const GV_Transaction *txn)
{
    if (!txn)
        return 0;
    return txn->txn_id;
}

GV_TxnStatus gv_txn_status(const GV_Transaction *txn)
{
    if (!txn)
        return GV_TXN_ABORTED;
    return txn->status;
}

/*  Transaction Operations  */

int gv_txn_add_vector(GV_Transaction *txn, const float *data, size_t dimension)
{
    if (!txn || !txn->mgr || !data)
        return -1;
    if (txn->status != GV_TXN_ACTIVE)
        return -1;
    if (dimension == 0)
        return -1;

    GV_MVCCManager *mgr = txn->mgr;

    /* Allocate a copy of the vector data */
    float *copy = malloc(dimension * sizeof(float));
    if (!copy)
        return -1;
    memcpy(copy, data, dimension * sizeof(float));

    pthread_mutex_lock(&mgr->mutex);

    if (mvcc_grow_versions(mgr) != 0) {
        pthread_mutex_unlock(&mgr->mutex);
        free(copy);
        return -1;
    }

    size_t ver_idx = mgr->ver_count;
    GV_MVCCVersion *ver = &mgr->versions[ver_idx];
    ver->vector_index = ver_idx;
    ver->create_txn = txn->txn_id;
    ver->delete_txn = 0;
    ver->data = copy;
    ver->dimension = dimension;
    mgr->ver_count++;

    pthread_mutex_unlock(&mgr->mutex);

    /* Track in the transaction's local added list (no lock needed -- only
     * this txn touches its own bookkeeping arrays). */
    if (idx_array_push(&txn->added_indices, &txn->added_count,
                       &txn->added_capacity, ver_idx) != 0) {
        /* Best-effort: version is already in the store; mark it deleted to
         * prevent it from becoming visible. */
        pthread_mutex_lock(&mgr->mutex);
        mgr->versions[ver_idx].delete_txn = txn->txn_id;
        pthread_mutex_unlock(&mgr->mutex);
        return -1;
    }

    return 0;
}

int gv_txn_delete_vector(GV_Transaction *txn, size_t vector_index)
{
    if (!txn || !txn->mgr)
        return -1;
    if (txn->status != GV_TXN_ACTIVE)
        return -1;

    GV_MVCCManager *mgr = txn->mgr;

    pthread_mutex_lock(&mgr->mutex);

    /* Scan for a visible version at the requested vector_index */
    int found = 0;
    for (size_t i = 0; i < mgr->ver_count; i++) {
        GV_MVCCVersion *ver = &mgr->versions[i];
        if (ver->vector_index != vector_index)
            continue;

        if (!mvcc_version_visible(mgr, ver, txn->txn_id, txn->snapshot_txn_id))
            continue;

        /* Check for write-write conflict: if another active txn already
         * stamped a pending delete on this version, we must not overwrite it. */
        if (ver->delete_txn != 0 && ver->delete_txn != txn->txn_id) {
            pthread_mutex_unlock(&mgr->mutex);
            return -1; /* conflict */
        }

        ver->delete_txn = txn->txn_id;
        found = 1;

        /* Track the version index so we can undo on rollback */
        pthread_mutex_unlock(&mgr->mutex);
        if (idx_array_push(&txn->deleted_indices, &txn->deleted_count,
                           &txn->deleted_capacity, i) != 0) {
            /* Undo the stamp */
            pthread_mutex_lock(&mgr->mutex);
            ver->delete_txn = 0;
            pthread_mutex_unlock(&mgr->mutex);
            return -1;
        }
        return 0;
    }

    pthread_mutex_unlock(&mgr->mutex);

    if (!found)
        return -1; /* no visible version to delete */

    return 0;
}

int gv_txn_get_vector(const GV_Transaction *txn, size_t vector_index, float *out)
{
    if (!txn || !txn->mgr || !out)
        return -1;

    GV_MVCCManager *mgr = txn->mgr;

    pthread_mutex_lock(&mgr->mutex);

    for (size_t i = 0; i < mgr->ver_count; i++) {
        const GV_MVCCVersion *ver = &mgr->versions[i];
        if (ver->vector_index != vector_index)
            continue;

        if (!mvcc_version_visible(mgr, ver, txn->txn_id, txn->snapshot_txn_id))
            continue;

        memcpy(out, ver->data, ver->dimension * sizeof(float));
        pthread_mutex_unlock(&mgr->mutex);
        return 0;
    }

    pthread_mutex_unlock(&mgr->mutex);
    return -1; /* not found */
}

size_t gv_txn_count(const GV_Transaction *txn)
{
    if (!txn || !txn->mgr)
        return 0;

    GV_MVCCManager *mgr = txn->mgr;
    size_t count = 0;

    pthread_mutex_lock(&mgr->mutex);

    for (size_t i = 0; i < mgr->ver_count; i++) {
        if (mvcc_version_visible(mgr, &mgr->versions[i],
                                 txn->txn_id, txn->snapshot_txn_id))
            count++;
    }

    pthread_mutex_unlock(&mgr->mutex);
    return count;
}

/*  Visibility (public API)  */

int gv_mvcc_is_visible(const GV_MVCCManager *mgr, const GV_MVCCVersion *ver,
                       uint64_t txn_id)
{
    if (!mgr || !ver)
        return 0;

    /* The public API takes only a txn_id.  We derive the snapshot as
     * txn_id - 1, which matches the convention in gv_txn_begin. */
    uint64_t snapshot = (txn_id > 0) ? txn_id - 1 : 0;

    /* We need the lock to walk the active list safely */
    pthread_mutex_lock((pthread_mutex_t *)&((GV_MVCCManager *)mgr)->mutex);
    int result = mvcc_version_visible(mgr, ver, txn_id, snapshot);
    pthread_mutex_unlock((pthread_mutex_t *)&((GV_MVCCManager *)mgr)->mutex);

    return result;
}

/*  Garbage Collection  */

int gv_mvcc_gc(GV_MVCCManager *mgr)
{
    if (!mgr)
        return -1;

    pthread_mutex_lock(&mgr->mutex);

    uint64_t min_active = mvcc_min_active_snapshot(mgr);

    /* A version can be reclaimed if:
     *   - It has been deleted (delete_txn != 0)
     *   - The deleting txn is committed (not in the active list)
     *   - The deleting txn_id < min_active (all active txns started after
     *     the delete was committed, so none can ever see this version)
     *
     * Additionally, aborted inserts (create_txn is not active AND
     * delete_txn == create_txn, meaning rollback) can also be reclaimed
     * when create_txn < min_active.
     */

    size_t dst = 0;
    for (size_t src = 0; src < mgr->ver_count; src++) {
        GV_MVCCVersion *ver = &mgr->versions[src];
        int reclaimable = 0;

        if (ver->delete_txn != 0 &&
            !mvcc_is_active(mgr, ver->delete_txn) &&
            ver->delete_txn < min_active) {
            reclaimable = 1;
        }

        if (reclaimable) {
            free(ver->data);
            ver->data = NULL;
        } else {
            if (dst != src)
                mgr->versions[dst] = mgr->versions[src];
            dst++;
        }
    }

    size_t removed = mgr->ver_count - dst;
    mgr->ver_count = dst;

    pthread_mutex_unlock(&mgr->mutex);

    return (int)removed;
}

size_t gv_mvcc_version_count(const GV_MVCCManager *mgr)
{
    if (!mgr)
        return 0;

    pthread_mutex_lock((pthread_mutex_t *)&((GV_MVCCManager *)mgr)->mutex);
    size_t count = mgr->ver_count;
    pthread_mutex_unlock((pthread_mutex_t *)&((GV_MVCCManager *)mgr)->mutex);

    return count;
}

size_t gv_mvcc_active_txn_count(const GV_MVCCManager *mgr)
{
    if (!mgr)
        return 0;

    pthread_mutex_lock((pthread_mutex_t *)&((GV_MVCCManager *)mgr)->mutex);
    size_t count = 0;
    const GV_TxnEntry *e = mgr->active_txns;
    while (e) {
        count++;
        e = e->next;
    }
    pthread_mutex_unlock((pthread_mutex_t *)&((GV_MVCCManager *)mgr)->mutex);

    return count;
}
