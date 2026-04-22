#ifndef GIGAVECTOR_GV_MVCC_H
#define GIGAVECTOR_GV_MVCC_H
#include <stddef.h>
#include <stdint.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GV_TXN_ACTIVE = 0,
    GV_TXN_COMMITTED = 1,
    GV_TXN_ABORTED = 2
} GV_TxnStatus;

typedef struct GV_MVCCManager GV_MVCCManager;
typedef struct GV_Transaction GV_Transaction;

typedef struct {
    size_t vector_index;
    uint64_t create_txn;    /* txn that created this version */
    uint64_t delete_txn;    /* txn that deleted it (0 = alive) */
    float *data;            /* vector data copy */
    size_t dimension;
} GV_MVCCVersion;

/**
 * @brief Create an MVCC manager for versioned vector storage.
 *
 * @param dimension Vector dimensionality managed by this MVCC instance.
 * @return Newly allocated manager, or NULL on allocation failure.
 */
GV_MVCCManager *gv_mvcc_create(size_t dimension);

/**
 * @brief Destroy an MVCC manager and free all associated resources.
 *
 * Safe to call with NULL.
 *
 * @param mgr MVCC manager to destroy.
 */
void gv_mvcc_destroy(GV_MVCCManager *mgr);

/**
 * @brief Begin a new transaction.
 *
 * @param mgr MVCC manager; must be non-NULL.
 * @return New transaction handle, or NULL on error.
 */
GV_Transaction *gv_txn_begin(GV_MVCCManager *mgr);

/**
 * @brief Commit a transaction, making its writes visible.
 *
 * @param txn Transaction handle; must be non-NULL.
 * @return 0 on success, -1 on error.
 */
int gv_txn_commit(GV_Transaction *txn);

/**
 * @brief Roll back a transaction, discarding its writes.
 *
 * @param txn Transaction handle; must be non-NULL.
 * @return 0 on success, -1 on error.
 */
int gv_txn_rollback(GV_Transaction *txn);

/**
 * @brief Get the transaction id.
 *
 * @param txn Transaction handle; must be non-NULL.
 * @return Transaction id (monotonic within a manager), or 0 on error.
 */
uint64_t gv_txn_id(const GV_Transaction *txn);

/**
 * @brief Get the current status of a transaction.
 *
 * @param txn Transaction handle; must be non-NULL.
 * @return Transaction status.
 */
GV_TxnStatus gv_txn_status(const GV_Transaction *txn);

/**
 * @brief Add a vector version within a transaction.
 *
 * @param txn Transaction handle; must be non-NULL.
 * @param data Vector data; must be non-NULL.
 * @param dimension Vector dimensionality; must match MVCC manager dimension.
 * @return 0 on success, -1 on error.
 */
int gv_txn_add_vector(GV_Transaction *txn, const float *data, size_t dimension);

/**
 * @brief Mark a vector as deleted within a transaction.
 *
 * @param txn Transaction handle; must be non-NULL.
 * @param vector_index Vector id to delete.
 * @return 0 on success, -1 on error.
 */
int gv_txn_delete_vector(GV_Transaction *txn, size_t vector_index);

/**
 * @brief Read a vector as visible to the transaction.
 *
 * @param txn Transaction handle; must be non-NULL.
 * @param vector_index Vector id to read.
 * @param out Output buffer for vector data; must have capacity for dimension floats.
 * @return 0 on success, -1 if not visible/not found or on error.
 */
int gv_txn_get_vector(const GV_Transaction *txn, size_t vector_index, float *out);

/**
 * @brief Count vectors visible to a transaction.
 *
 * @param txn Transaction handle; must be non-NULL.
 * @return Count of visible vectors.
 */
size_t gv_txn_count(const GV_Transaction *txn);

/**
 * @brief Check whether a version is visible to a transaction id.
 *
 * @param mgr MVCC manager; must be non-NULL.
 * @param ver Version to test; must be non-NULL.
 * @param txn_id Transaction id performing the read.
 * @return 1 if visible, 0 if not visible, -1 on error.
 */
int gv_mvcc_is_visible(const GV_MVCCManager *mgr, const GV_MVCCVersion *ver, uint64_t txn_id);

/**
 * @brief Run garbage collection to reclaim versions not visible to any active transaction.
 *
 * @param mgr MVCC manager; must be non-NULL.
 * @return 0 on success, -1 on error.
 */
int gv_mvcc_gc(GV_MVCCManager *mgr);

/**
 * @brief Get the total number of stored versions.
 *
 * @param mgr MVCC manager; must be non-NULL.
 * @return Version count.
 */
size_t gv_mvcc_version_count(const GV_MVCCManager *mgr);

/**
 * @brief Get the number of active transactions.
 *
 * @param mgr MVCC manager; must be non-NULL.
 * @return Active transaction count.
 */
size_t gv_mvcc_active_txn_count(const GV_MVCCManager *mgr);

#ifdef __cplusplus
}
#endif
#endif
