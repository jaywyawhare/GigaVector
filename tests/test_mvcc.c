#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_mvcc.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

/* ---------- test functions ---------- */

static int test_manager_create_destroy(void) {
    GV_MVCCManager *mgr = gv_mvcc_create(4);
    ASSERT(mgr != NULL, "gv_mvcc_create returned NULL");
    ASSERT(gv_mvcc_version_count(mgr) == 0, "initial version count should be 0");
    ASSERT(gv_mvcc_active_txn_count(mgr) == 0, "initial active txn count should be 0");
    gv_mvcc_destroy(mgr);
    return 0;
}

static int test_txn_begin_commit(void) {
    GV_MVCCManager *mgr = gv_mvcc_create(4);
    ASSERT(mgr != NULL, "create manager");

    GV_Transaction *txn = gv_txn_begin(mgr);
    ASSERT(txn != NULL, "gv_txn_begin returned NULL");
    ASSERT(gv_txn_status(txn) == GV_TXN_ACTIVE, "txn should be ACTIVE after begin");
    ASSERT(gv_txn_id(txn) > 0, "txn id should be > 0");
    ASSERT(gv_mvcc_active_txn_count(mgr) >= 1, "active txn count should be >= 1");

    int rc = gv_txn_commit(txn);
    ASSERT(rc == 0, "gv_txn_commit should return 0");

    gv_mvcc_destroy(mgr);
    return 0;
}

static int test_txn_begin_rollback(void) {
    GV_MVCCManager *mgr = gv_mvcc_create(4);
    ASSERT(mgr != NULL, "create manager");

    GV_Transaction *txn = gv_txn_begin(mgr);
    ASSERT(txn != NULL, "begin txn");
    ASSERT(gv_txn_status(txn) == GV_TXN_ACTIVE, "txn should be active");

    int rc = gv_txn_rollback(txn);
    ASSERT(rc == 0, "gv_txn_rollback should return 0");

    gv_mvcc_destroy(mgr);
    return 0;
}

static int test_txn_add_and_get_vector(void) {
    GV_MVCCManager *mgr = gv_mvcc_create(4);
    ASSERT(mgr != NULL, "create manager");

    GV_Transaction *txn = gv_txn_begin(mgr);
    ASSERT(txn != NULL, "begin txn");

    float vec[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    int rc = gv_txn_add_vector(txn, vec, 4);
    ASSERT(rc == 0, "add vector should succeed");
    ASSERT(gv_txn_count(txn) == 1, "count should be 1 after add");

    float out[4] = {0};
    rc = gv_txn_get_vector(txn, 0, out);
    ASSERT(rc == 0, "get vector should succeed");
    ASSERT(out[0] == 1.0f && out[1] == 2.0f && out[2] == 3.0f && out[3] == 4.0f,
           "retrieved vector data should match");

    rc = gv_txn_commit(txn);
    ASSERT(rc == 0, "commit should succeed");

    gv_mvcc_destroy(mgr);
    return 0;
}

static int test_txn_delete_vector(void) {
    GV_MVCCManager *mgr = gv_mvcc_create(4);
    ASSERT(mgr != NULL, "create manager");

    /* Add a vector and commit */
    GV_Transaction *txn1 = gv_txn_begin(mgr);
    float vec[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    gv_txn_add_vector(txn1, vec, 4);
    gv_txn_commit(txn1);

    /* Delete that vector in a new txn */
    GV_Transaction *txn2 = gv_txn_begin(mgr);
    ASSERT(txn2 != NULL, "begin txn2");
    int rc = gv_txn_delete_vector(txn2, 0);
    ASSERT(rc == 0, "delete vector should succeed");
    gv_txn_commit(txn2);

    gv_mvcc_destroy(mgr);
    return 0;
}

static int test_multiple_txns(void) {
    GV_MVCCManager *mgr = gv_mvcc_create(4);
    ASSERT(mgr != NULL, "create manager");

    GV_Transaction *t1 = gv_txn_begin(mgr);
    GV_Transaction *t2 = gv_txn_begin(mgr);
    ASSERT(t1 != NULL && t2 != NULL, "begin two txns");
    ASSERT(gv_txn_id(t1) != gv_txn_id(t2), "txn ids should differ");
    ASSERT(gv_mvcc_active_txn_count(mgr) >= 2, "active count >= 2");

    float v1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v2[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    gv_txn_add_vector(t1, v1, 4);
    gv_txn_add_vector(t2, v2, 4);

    gv_txn_commit(t1);
    gv_txn_rollback(t2);

    gv_mvcc_destroy(mgr);
    return 0;
}

static int test_gc(void) {
    GV_MVCCManager *mgr = gv_mvcc_create(4);
    ASSERT(mgr != NULL, "create manager");

    /* Add and commit a vector, then delete and commit */
    GV_Transaction *t1 = gv_txn_begin(mgr);
    float vec[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    gv_txn_add_vector(t1, vec, 4);
    gv_txn_commit(t1);

    GV_Transaction *t2 = gv_txn_begin(mgr);
    gv_txn_delete_vector(t2, 0);
    gv_txn_commit(t2);

    /* Run GC - should clean up old versions */
    int rc = gv_mvcc_gc(mgr);
    ASSERT(rc == 0, "gc should succeed");

    gv_mvcc_destroy(mgr);
    return 0;
}

static int test_null_safety(void) {
    /* gv_mvcc_create with 0 dimension may return NULL or handle gracefully */
    GV_MVCCManager *mgr = gv_mvcc_create(0);
    if (mgr != NULL) {
        gv_mvcc_destroy(mgr);
    }

    /* destroy NULL should be safe */
    gv_mvcc_destroy(NULL);
    return 0;
}

/* ---------- harness ---------- */

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing manager create/destroy...", test_manager_create_destroy},
        {"Testing txn begin/commit...", test_txn_begin_commit},
        {"Testing txn begin/rollback...", test_txn_begin_rollback},
        {"Testing txn add and get vector...", test_txn_add_and_get_vector},
        {"Testing txn delete vector...", test_txn_delete_vector},
        {"Testing multiple txns...", test_multiple_txns},
        {"Testing GC...", test_gc},
        {"Testing null safety...", test_null_safety},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
