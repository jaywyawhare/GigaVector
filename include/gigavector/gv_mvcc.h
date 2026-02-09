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

/* Manager */
GV_MVCCManager *gv_mvcc_create(size_t dimension);
void gv_mvcc_destroy(GV_MVCCManager *mgr);

/* Transaction lifecycle */
GV_Transaction *gv_txn_begin(GV_MVCCManager *mgr);
int gv_txn_commit(GV_Transaction *txn);
int gv_txn_rollback(GV_Transaction *txn);
uint64_t gv_txn_id(const GV_Transaction *txn);
GV_TxnStatus gv_txn_status(const GV_Transaction *txn);

/* Transaction operations */
int gv_txn_add_vector(GV_Transaction *txn, const float *data, size_t dimension);
int gv_txn_delete_vector(GV_Transaction *txn, size_t vector_index);
int gv_txn_get_vector(const GV_Transaction *txn, size_t vector_index, float *out);
size_t gv_txn_count(const GV_Transaction *txn);

/* Visibility check */
int gv_mvcc_is_visible(const GV_MVCCManager *mgr, const GV_MVCCVersion *ver, uint64_t txn_id);

/* GC old versions */
int gv_mvcc_gc(GV_MVCCManager *mgr);
size_t gv_mvcc_version_count(const GV_MVCCManager *mgr);
size_t gv_mvcc_active_txn_count(const GV_MVCCManager *mgr);

#ifdef __cplusplus
}
#endif
#endif
