#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include "gigavector.h"

#define ASSERT(cond, msg)         \
    do {                          \
        if (!(cond)) {            \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1;            \
        }                         \
    } while (0)

#define N_INSERT_THREADS 4
#define N_SEARCH_THREADS 4
#define INSERTS_PER_THREAD 64
#define SEARCHES_PER_THREAD 32

typedef struct {
    GV_Database *db;
    int thread_id;
    int result;
} ThreadArg;

static void *insert_worker(void *arg) {
    ThreadArg *a = (ThreadArg *)arg;
    for (int i = 0; i < INSERTS_PER_THREAD; i++) {
        float v[4];
        for (int j = 0; j < 4; j++)
            v[j] = (float)(a->thread_id * INSERTS_PER_THREAD + i + j) * 0.01f;
        if (db_add_vector(a->db, v, 4) != 0) {
            a->result = -1;
            return NULL;
        }
    }
    a->result = 0;
    return NULL;
}

static void *search_worker(void *arg) {
    ThreadArg *a = (ThreadArg *)arg;
    for (int i = 0; i < SEARCHES_PER_THREAD; i++) {
        float q[4] = {(float)i * 0.01f, 0.0f, 0.0f, 0.0f};
        GV_SearchResult res[3];
        int n = db_search(a->db, q, 3, res, GV_DISTANCE_EUCLIDEAN);
        (void)n;
        gv_search_results_free(res, (size_t)(n > 0 ? n : 0));
    }
    a->result = 0;
    return NULL;
}

static int test_concurrent_insert_search(void) {
    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_HNSW);
    if (db == NULL) return 0;

    /* Seed with some initial data so searches have something to find */
    for (int i = 0; i < 32; i++) {
        float v[4] = {(float)i * 0.01f, (float)i * 0.02f,
                      (float)i * 0.03f, (float)i * 0.04f};
        db_add_vector(db, v, 4);
    }

    pthread_t insert_threads[N_INSERT_THREADS];
    pthread_t search_threads[N_SEARCH_THREADS];
    ThreadArg insert_args[N_INSERT_THREADS];
    ThreadArg search_args[N_SEARCH_THREADS];

    for (int i = 0; i < N_INSERT_THREADS; i++) {
        insert_args[i].db = db;
        insert_args[i].thread_id = i;
        insert_args[i].result = 0;
        pthread_create(&insert_threads[i], NULL, insert_worker, &insert_args[i]);
    }
    for (int i = 0; i < N_SEARCH_THREADS; i++) {
        search_args[i].db = db;
        search_args[i].thread_id = i;
        search_args[i].result = 0;
        pthread_create(&search_threads[i], NULL, search_worker, &search_args[i]);
    }

    for (int i = 0; i < N_INSERT_THREADS; i++)
        pthread_join(insert_threads[i], NULL);
    for (int i = 0; i < N_SEARCH_THREADS; i++)
        pthread_join(search_threads[i], NULL);

    for (int i = 0; i < N_INSERT_THREADS; i++)
        ASSERT(insert_args[i].result == 0, "insert thread succeeded");

    db_close(db);
    return 0;
}

int main(void) {
    return test_concurrent_insert_search();
}
