#ifndef GIGAVECTOR_GV_TRANSACTION_H
#define GIGAVECTOR_GV_TRANSACTION_H

#include <stddef.h>
#include <stdint.h>

#include "storage/database.h"
#include "search/distance.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file transaction.h
 * @brief Multi-statement transactions with MVCC snapshot isolation.
 *
 * A transaction captures a read snapshot at begin. All reads through it see a
 * consistent view of the database as of that snapshot — inserts and deletes
 * committed by other transactions afterward are invisible, and reads never block
 * writers. Writes are staged and applied atomically at commit; conflicting
 * concurrent deletes are rejected (first-committer-wins → serializable for the
 * write set). Targets the dense SoA-backed indexes (KD-tree / HNSW / FLAT / IVF).
 */

typedef enum {
    GV_TXN_ACTIVE = 0,
    GV_TXN_COMMITTED = 1,
    GV_TXN_ABORTED = 2
} GV_TxnState;

/* db_commit return codes. */
#define GV_TXN_OK        0
#define GV_TXN_CONFLICT  1   /* write-write conflict; the transaction is aborted */

typedef struct GV_DBTxn GV_DBTxn;

/** @brief Begin a transaction, capturing the current committed snapshot. */
GV_DBTxn *db_begin(GV_Database *db);

/** @brief Stage an insert (data is copied). Returns 0, or -1 on error. */
int db_txn_add_vector(GV_DBTxn *txn, const float *data, size_t dimension);

/** @brief Stage a delete of a committed vector by its SoA index. Returns 0/-1. */
int db_txn_delete(GV_DBTxn *txn, size_t vector_index);

/**
 * @brief Snapshot-consistent k-NN search within the transaction.
 *
 * Sees the transaction's snapshot plus its own staged inserts (read-your-writes),
 * and hides its own staged deletes. Returns the number of results, or -1 on error.
 */
int db_txn_search(GV_DBTxn *txn, const float *query, size_t k,
                  GV_SearchResult *results, GV_DistanceType metric);

/**
 * @brief Commit the transaction: apply staged writes atomically.
 * @return GV_TXN_OK on success, GV_TXN_CONFLICT if a staged delete conflicts with
 *         a concurrent commit (transaction is aborted), -1 on error. The handle
 *         is freed on any terminal outcome.
 */
int db_commit(GV_DBTxn *txn);

/** @brief Abort the transaction, discarding staged writes. Frees the handle. */
int db_rollback(GV_DBTxn *txn);

GV_TxnState db_txn_state(const GV_DBTxn *txn);
uint64_t db_txn_read_version(const GV_DBTxn *txn);

/**
 * @brief Reclaim MVCC tombstones no live snapshot can observe.
 *
 * Hard-deletes vectors whose delete version is < @p safe_below — they are
 * invisible to every snapshot at or after @p safe_below. Pass the minimum read
 * version among still-active transactions (or the current commit version + 1 when
 * none are active) to reclaim everything safely.
 *
 * @return Number of tombstones reclaimed.
 */
size_t db_txn_gc(GV_Database *db, uint64_t safe_below);

/**
 * @brief Minimum read version among live transactions, or 0 if none are active.
 *
 * This is the GC safe-point: a tombstone with delete version <= this value is
 * invisible to every live snapshot and can be reclaimed.
 */
uint64_t db_txn_min_active_version(GV_Database *db);

/**
 * @brief Reclaim all tombstones no live snapshot can observe, choosing the safe
 *        point automatically from the live-transaction set.
 *
 * Equivalent to db_txn_gc() with safe_below derived from the minimum active read
 * version (or commit_version + 1 when no transactions are active). Safe to call
 * periodically to keep tombstones from accumulating.
 *
 * @return Number of tombstones reclaimed.
 */
size_t db_txn_gc_auto(GV_Database *db);

#ifdef __cplusplus
}
#endif

#endif
