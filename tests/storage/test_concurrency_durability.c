/* Concurrency + durability: many writer threads (plain inserts AND multi-op
 * transactions) racing with reader threads against a WAL-backed database, then
 * a reopen that must replay every committed write exactly. Exercises the global
 * lock, the atomic TXN WAL record, and crash-free durability together. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

/* Include the specific storage headers (not gigavector.h): the umbrella header
 * pulls in specialized/mvcc.h, whose GV_TXN_* enum collides with the identical
 * names in storage/transaction.h, so the two cannot share a translation unit. */
#include "storage/database.h"
#include "storage/transaction.h"
#include "../test_tmp.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while (0)

#define DIM 4u
#define N_PLAIN_THREADS 4
#define PLAIN_PER_THREAD 50
#define N_TXN_THREADS 2
#define TXN_PER_THREAD 10      /* each transaction stages 2 inserts */
#define N_READ_THREADS 4
#define READS_PER_THREAD 64
#define N_SEED 4

typedef struct { GV_Database *db; int tid; int ok; } Arg;

static void mkvec(float *v, int tid, int i) {
    v[0] = (float)(tid * 1000 + i);
    v[1] = (float)tid;
    v[2] = (float)i;
    v[3] = 1.0f;
}

static void *plain_writer(void *p) {
    Arg *a = (Arg *)p;
    a->ok = 0;
    for (int i = 0; i < PLAIN_PER_THREAD; i++) {
        float v[DIM];
        mkvec(v, a->tid, i);
        if (db_add_vector(a->db, v, DIM) != 0) { a->ok = -1; return NULL; }
    }
    return NULL;
}

static void *txn_writer(void *p) {
    Arg *a = (Arg *)p;
    a->ok = 0;
    for (int t = 0; t < TXN_PER_THREAD; t++) {
        GV_DBTxn *tx = db_begin(a->db);
        if (!tx) { a->ok = -1; return NULL; }
        float v0[DIM], v1[DIM];
        mkvec(v0, a->tid, t * 2);
        mkvec(v1, a->tid, t * 2 + 1);
        if (db_txn_add_vector(tx, v0, DIM) != 0 || db_txn_add_vector(tx, v1, DIM) != 0) {
            db_rollback(tx); a->ok = -1; return NULL;
        }
        if (db_commit(tx) != GV_TXN_OK) { a->ok = -1; return NULL; }
    }
    return NULL;
}

static void *reader(void *p) {
    Arg *a = (Arg *)p;
    a->ok = 0;
    for (int i = 0; i < READS_PER_THREAD; i++) {
        float q[DIM] = {(float)i, 0, 0, 1.0f};
        GV_SearchResult res[5];
        int n = db_search(a->db, q, 5, res, GV_DISTANCE_EUCLIDEAN);
        for (int j = 0; j < n; j++) {
            if (!(res[j].distance >= 0.0f) || res[j].distance != res[j].distance) a->ok = -1;
        }
        gv_search_results_free(res, (size_t)(n > 0 ? n : 0));
    }
    return NULL;
}

static int test_concurrent_durability(void) {
    char db_path[256], wal_path[512];
    ASSERT(gv_test_make_temp_path(db_path, sizeof(db_path), "gv_conc_dur", ".bin") == 0, "db path");
    snprintf(wal_path, sizeof(wal_path), "%s.wal", db_path);
    remove(db_path); remove(wal_path);

    GV_Database *db = db_open(db_path, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "open db");
    ASSERT(db_set_wal(db, wal_path) == 0, "enable wal");

    for (int i = 0; i < N_SEED; i++) {
        float v[DIM]; mkvec(v, 999, i);
        { int _r = db_add_vector(db, v, DIM); (void)_r; }
    }

    pthread_t pt[N_PLAIN_THREADS], tt[N_TXN_THREADS], rt[N_READ_THREADS];
    Arg pa[N_PLAIN_THREADS], ta[N_TXN_THREADS], ra[N_READ_THREADS];

    for (int i = 0; i < N_PLAIN_THREADS; i++) { pa[i].db = db; pa[i].tid = i; pthread_create(&pt[i], NULL, plain_writer, &pa[i]); }
    for (int i = 0; i < N_TXN_THREADS; i++)   { ta[i].db = db; ta[i].tid = 100 + i; pthread_create(&tt[i], NULL, txn_writer, &ta[i]); }
    for (int i = 0; i < N_READ_THREADS; i++)  { ra[i].db = db; ra[i].tid = i; pthread_create(&rt[i], NULL, reader, &ra[i]); }

    for (int i = 0; i < N_PLAIN_THREADS; i++) pthread_join(pt[i], NULL);
    for (int i = 0; i < N_TXN_THREADS; i++)   pthread_join(tt[i], NULL);
    for (int i = 0; i < N_READ_THREADS; i++)  pthread_join(rt[i], NULL);

    for (int i = 0; i < N_PLAIN_THREADS; i++) ASSERT(pa[i].ok == 0, "plain writer ok");
    for (int i = 0; i < N_TXN_THREADS; i++)   ASSERT(ta[i].ok == 0, "txn writer ok");
    for (int i = 0; i < N_READ_THREADS; i++)  ASSERT(ra[i].ok == 0, "reader saw only valid results");

    size_t expected = (size_t)N_SEED
                    + (size_t)N_PLAIN_THREADS * PLAIN_PER_THREAD
                    + (size_t)N_TXN_THREADS * TXN_PER_THREAD * 2;
    ASSERT(database_count(db) == expected, "all concurrent writes accounted for");

    /* No db_save: a reopen MUST replay the WAL (plain INSERT + atomic TXN
     * records written concurrently) and reconstruct the exact same state. */
    db_close(db);
    GV_Database *db2 = db_open(db_path, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db2 != NULL, "reopen db");
    ASSERT(database_count(db2) == expected, "WAL replay reconstructs every committed write");

    /* Spot-check that a specific committed vector is findable after recovery. */
    float probe[DIM]; mkvec(probe, 0, 0);
    GV_SearchResult r[1];
    int n = db_search(db2, probe, 1, r, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n == 1 && r[0].distance == 0.0f, "a known vector survives recovery exactly");
    gv_search_results_free(r, (size_t)(n > 0 ? n : 0));

    db_close(db2);
    remove(db_path); remove(wal_path);
    return 0;
}

int main(void) {
    if (test_concurrent_durability() != 0) {
        printf("\nCONCURRENCY/DURABILITY TEST FAILED\n");
        return 1;
    }
    printf("\nCONCURRENCY/DURABILITY TEST PASSED\n");
    return 0;
}
