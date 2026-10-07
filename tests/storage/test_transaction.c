/* MVCC transactions: snapshot isolation, read-your-writes, tombstones,
 * write-write conflict, rollback. FLAT index for deterministic exact search. */
#include <stdio.h>
#include <string.h>
#include "storage/database.h"
#include "storage/transaction.h"

static int failures = 0;
#define ASSERT(c, m) do { if (!(c)) { printf("FAIL: %s\n", (m)); failures++; } \
                          else { printf("ok: %s\n", (m)); } } while (0)

static int has_id(const GV_SearchResult *r, int n, size_t id) {
    for (int i = 0; i < n; i++) if (r[i].id == id) return 1;
    return 0;
}
static void freeres(GV_SearchResult *r, int n) {
    for (int i = 0; i < n; i++) if (r[i].vector) gv_search_results_free(&r[i], 1);
}

int main(void) {
    const size_t D = 4;
    GV_Database *db = db_open(NULL, D, GV_INDEX_TYPE_FLAT);
    float v0[] = {1, 0, 0, 0}, v1[] = {0, 1, 0, 0}, v2[] = {0, 0, 1, 0}, v3[] = {0, 0, 0, 1};
    { int _r = db_add_vector(db, v0, D); (void)_r; }  /* index 0, version 0 -> always visible */
    { int _r = db_add_vector(db, v1, D); (void)_r; }  /* index 1 */
    { int _r = db_add_vector(db, v2, D); (void)_r; }  /* index 2 */

    GV_SearchResult res[16];

    /* ---- snapshot isolation ---- */
    GV_DBTxn *A = db_begin(db);                 /* read_version = 0 */
    GV_DBTxn *B = db_begin(db);
    ASSERT(db_txn_add_vector(B, v3, D) == 0, "stage insert in B");
    ASSERT(db_commit(B) == GV_TXN_OK, "commit B (adds v3, version 1)");

    /* plain reader sees the committed insert */
    int n = db_search(db, v3, 1, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n == 1 && res[0].id == 3, "committed insert visible to new plain read");
    freeres(res, n);

    /* A began before B committed -> must NOT see v3 (snapshot isolation) */
    int na = db_txn_search(A, v3, 4, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(!has_id(res, na, 3), "older snapshot does not see later insert");
    freeres(res, na);
    db_rollback(A);   /* read-only txn: release the handle */

    /* a fresh snapshot does see it */
    GV_DBTxn *C = db_begin(db);
    int nc = db_txn_search(C, v3, 4, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(has_id(res, nc, 3), "new snapshot sees the insert");
    freeres(res, nc);
    db_rollback(C);

    /* ---- read-your-writes ---- */
    GV_DBTxn *Dt = db_begin(db);
    float v4[] = {0.9f, 0, 0, 0};
    db_txn_add_vector(Dt, v4, D);
    int nd = db_txn_search(Dt, v0, 3, res, GV_DISTANCE_EUCLIDEAN);
    /* staged insert has synthetic id (size_t)-1; must appear alongside v0 */
    ASSERT(has_id(res, nd, (size_t)-1), "txn sees its own staged insert (read-your-writes)");
    ASSERT(has_id(res, nd, 0), "and still sees committed v0");
    freeres(res, nd);
    db_rollback(Dt);   /* discard staged insert */
    n = db_search(db, v4, 1, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(!(n == 1 && res[0].id >= 4), "rolled-back insert never committed");
    freeres(res, n);

    /* ---- MVCC delete tombstone + snapshot ---- */
    GV_DBTxn *E = db_begin(db);                 /* snapshot before the delete */
    GV_DBTxn *F = db_begin(db);
    ASSERT(db_txn_delete(F, 0) == 0, "stage delete of index 0 in F");
    ASSERT(db_commit(F) == GV_TXN_OK, "commit F (tombstones v0)");

    /* new plain read no longer sees v0 */
    n = db_search(db, v0, 3, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(!has_id(res, n, 0), "committed delete hidden from new reads");
    freeres(res, n);
    /* E's older snapshot still sees v0 */
    int ne = db_txn_search(E, v0, 3, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(has_id(res, ne, 0), "older snapshot still sees the deleted vector");
    freeres(res, ne);
    db_rollback(E);

    /* ---- write-write conflict (first-committer-wins) ---- */
    GV_DBTxn *G = db_begin(db);
    GV_DBTxn *H = db_begin(db);
    db_txn_delete(G, 2);
    db_txn_delete(H, 2);
    ASSERT(db_commit(G) == GV_TXN_OK, "G deletes index 2, commits");
    ASSERT(db_commit(H) == GV_TXN_CONFLICT, "H's conflicting delete is rejected");

    /* ---- atomic admission: a resource-limit breach rejects the WHOLE commit
     * with no partial application ---- */
    {
        GV_Database *ldb = db_open(NULL, D, GV_INDEX_TYPE_FLAT);
        ASSERT(ldb != NULL, "open limited db");
        { int _r = db_add_vector(ldb, v0, D); (void)_r; }   /* count = 1 */
        GV_ResourceLimits lim;
        memset(&lim, 0, sizeof(lim));
        lim.max_vectors = 2;                                /* room for exactly 1 more */
        ASSERT(db_set_resource_limits(ldb, &lim) == 0, "set max_vectors=2");

        GV_DBTxn *T = db_begin(ldb);
        ASSERT(db_txn_add_vector(T, v1, D) == 0, "stage insert 1");
        ASSERT(db_txn_add_vector(T, v2, D) == 0, "stage insert 2");
        ASSERT(db_txn_add_vector(T, v3, D) == 0, "stage insert 3 (would exceed limit)");
        /* 1 existing + 3 staged = 4 > max_vectors(2): reject atomically. */
        ASSERT(db_commit(T) == -1, "over-limit commit rejected");

        GV_SearchResult lr[8];
        int ln = db_search(ldb, v1, 8, lr, GV_DISTANCE_EUCLIDEAN);
        ASSERT(ln == 1, "no staged inserts applied (count still 1)");
        freeres(lr, ln);
        db_close(ldb);
    }

    db_close(db);
    printf(failures ? "\nSOME TESTS FAILED (%d)\n" : "\nALL TRANSACTION TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
