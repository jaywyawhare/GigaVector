/**
 * @file test_snapshot_concurrency.c
 * @brief TSAN regression: the snapshot manager is FFI-exposed with no external
 *        lock, so concurrent snapshot_create (which gv_realloc-grows entries[])
 *        must not race snapshot_open/_get/_count/_list/_delete reading it.
 */

#include "storage/snapshot.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define N_ITERS 2000
#define DIM 8

static GV_SnapshotManager *g_mgr;

static void *writer_fn(void *arg)
{
    (void)arg;
    float vec[DIM] = {0};
    for (int i = 0; i < N_ITERS; i++) {
        uint64_t id = snapshot_create(g_mgr, 1, vec, DIM, "w");
        if (id != 0 && (i % 3) == 0) {
            snapshot_delete(g_mgr, id);
        }
    }
    return NULL;
}

static void *reader_fn(void *arg)
{
    (void)arg;
    GV_SnapshotInfo infos[64];
    for (int i = 0; i < N_ITERS; i++) {
        int n = snapshot_list(g_mgr, infos, 64);
        for (int j = 0; j < n; j++) {
            GV_Snapshot *s = snapshot_open(g_mgr, infos[j].snapshot_id);
            if (s) {
                (void)snapshot_count(s);
                (void)snapshot_dimension(s);
                (void)snapshot_get_vector(s, 0);
                snapshot_close(s);
            }
        }
    }
    return NULL;
}

int main(void)
{
    /* Generous cap so writers keep growing entries[] during the run. */
    g_mgr = snapshot_manager_create(100000);
    assert(g_mgr);

    pthread_t w1, w2, r1, r2;
    pthread_create(&w1, NULL, writer_fn, NULL);
    pthread_create(&w2, NULL, writer_fn, NULL);
    pthread_create(&r1, NULL, reader_fn, NULL);
    pthread_create(&r2, NULL, reader_fn, NULL);

    pthread_join(w1, NULL);
    pthread_join(w2, NULL);
    pthread_join(r1, NULL);
    pthread_join(r2, NULL);

    snapshot_manager_destroy(g_mgr);
    printf("test_snapshot_concurrency: OK\n");
    return 0;
}
