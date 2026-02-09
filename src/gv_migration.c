#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <stdint.h>

#include "gigavector/gv_migration.h"
#include "gigavector/gv_hnsw.h"
#include "gigavector/gv_flat.h"
#include "gigavector/gv_ivfflat.h"
#include "gigavector/gv_pq.h"
#include "gigavector/gv_lsh.h"
#include "gigavector/gv_vector.h"
#include "gigavector/gv_soa_storage.h"

/* Index type constants matching GV_IndexType enum used across the codebase */
#define MIG_INDEX_KDTREE   0
#define MIG_INDEX_HNSW     1
#define MIG_INDEX_FLAT      4
#define MIG_INDEX_IVFFLAT   5
#define MIG_INDEX_PQ        6
#define MIG_INDEX_LSH       7

/* Batch size for progress updates and cancel checks */
#define MIGRATION_BATCH_SIZE 100

struct GV_Migration {
    pthread_t thread;
    GV_MigrationStatus status;
    double progress;
    size_t vectors_migrated;
    size_t total_vectors;
    uint64_t start_time_us;
    size_t dimension;
    int new_index_type;
    const float *source_data;       /* not owned */
    const void *new_index_config;   /* not owned */
    void *new_index;                /* the built index, owned until taken */
    int cancel_requested;
    pthread_mutex_t mutex;
    char error_message[256];
};

/* ---------- helper: monotonic time in microseconds ---------- */

static uint64_t now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

/* ---------- helper: set error message under lock ---------- */

static void migration_set_error(GV_Migration *mig, const char *msg)
{
    pthread_mutex_lock(&mig->mutex);
    mig->status = GV_MIGRATION_FAILED;
    strncpy(mig->error_message, msg, sizeof(mig->error_message) - 1);
    mig->error_message[sizeof(mig->error_message) - 1] = '\0';
    pthread_mutex_unlock(&mig->mutex);
}

/* ---------- helper: check cancel under lock ---------- */

static int migration_is_cancelled(GV_Migration *mig)
{
    int cancelled;
    pthread_mutex_lock(&mig->mutex);
    cancelled = mig->cancel_requested;
    pthread_mutex_unlock(&mig->mutex);
    return cancelled;
}

/* ---------- helper: update progress under lock ---------- */

static void migration_update_progress(GV_Migration *mig, size_t migrated)
{
    pthread_mutex_lock(&mig->mutex);
    mig->vectors_migrated = migrated;
    if (mig->total_vectors > 0) {
        mig->progress = (double)migrated / (double)mig->total_vectors;
    }
    pthread_mutex_unlock(&mig->mutex);
}

/* ---------- index creation helpers ---------- */

static void *create_kdtree_index(GV_Migration *mig)
{
    /* For KDTREE, the "index" is a GV_SoAStorage* since the KD-tree is
       built by inserting vector indices into a GV_KDNode tree on top of
       SoA storage.  We create a new SoA storage and populate it.  The
       caller can use gv_kdtree_insert to build the tree from the storage
       afterwards. */
    GV_SoAStorage *storage = gv_soa_storage_create(mig->dimension, mig->total_vectors);
    if (!storage) {
        migration_set_error(mig, "Failed to create SoA storage for KDTREE migration");
        return NULL;
    }

    for (size_t i = 0; i < mig->total_vectors; i++) {
        if (migration_is_cancelled(mig)) {
            gv_soa_storage_destroy(storage);
            return NULL;
        }

        const float *vec_data = mig->source_data + i * mig->dimension;
        size_t idx = gv_soa_storage_add(storage, vec_data, NULL);
        if (idx == (size_t)-1) {
            gv_soa_storage_destroy(storage);
            migration_set_error(mig, "Failed to add vector to SoA storage during KDTREE migration");
            return NULL;
        }

        if ((i + 1) % MIGRATION_BATCH_SIZE == 0 || i + 1 == mig->total_vectors) {
            migration_update_progress(mig, i + 1);
        }
    }

    return storage;
}

static void *create_hnsw_index(GV_Migration *mig)
{
    const GV_HNSWConfig *cfg = (const GV_HNSWConfig *)mig->new_index_config;
    void *index = gv_hnsw_create(mig->dimension, cfg, NULL);
    if (!index) {
        migration_set_error(mig, "Failed to create HNSW index");
        return NULL;
    }

    for (size_t i = 0; i < mig->total_vectors; i++) {
        if (migration_is_cancelled(mig)) {
            gv_hnsw_destroy(index);
            return NULL;
        }

        const float *vec_data = mig->source_data + i * mig->dimension;
        GV_Vector *vec = gv_vector_create_from_data(mig->dimension, vec_data);
        if (!vec) {
            gv_hnsw_destroy(index);
            migration_set_error(mig, "Failed to create vector during HNSW migration");
            return NULL;
        }

        if (gv_hnsw_insert(index, vec) != 0) {
            gv_vector_destroy(vec);
            gv_hnsw_destroy(index);
            migration_set_error(mig, "Failed to insert vector into HNSW index");
            return NULL;
        }

        if ((i + 1) % MIGRATION_BATCH_SIZE == 0 || i + 1 == mig->total_vectors) {
            migration_update_progress(mig, i + 1);
        }
    }

    return index;
}

static void *create_flat_index(GV_Migration *mig)
{
    const GV_FlatConfig *cfg = (const GV_FlatConfig *)mig->new_index_config;
    void *index = gv_flat_create(mig->dimension, cfg, NULL);
    if (!index) {
        migration_set_error(mig, "Failed to create Flat index");
        return NULL;
    }

    for (size_t i = 0; i < mig->total_vectors; i++) {
        if (migration_is_cancelled(mig)) {
            gv_flat_destroy(index);
            return NULL;
        }

        const float *vec_data = mig->source_data + i * mig->dimension;
        GV_Vector *vec = gv_vector_create_from_data(mig->dimension, vec_data);
        if (!vec) {
            gv_flat_destroy(index);
            migration_set_error(mig, "Failed to create vector during Flat migration");
            return NULL;
        }

        if (gv_flat_insert(index, vec) != 0) {
            gv_vector_destroy(vec);
            gv_flat_destroy(index);
            migration_set_error(mig, "Failed to insert vector into Flat index");
            return NULL;
        }

        if ((i + 1) % MIGRATION_BATCH_SIZE == 0 || i + 1 == mig->total_vectors) {
            migration_update_progress(mig, i + 1);
        }
    }

    return index;
}

static void *create_ivfflat_index(GV_Migration *mig)
{
    const GV_IVFFlatConfig *cfg = (const GV_IVFFlatConfig *)mig->new_index_config;
    void *index = gv_ivfflat_create(mig->dimension, cfg);
    if (!index) {
        migration_set_error(mig, "Failed to create IVF-Flat index");
        return NULL;
    }

    /* IVF-Flat requires training before insertion */
    if (gv_ivfflat_train(index, mig->source_data, mig->total_vectors) != 0) {
        gv_ivfflat_destroy(index);
        migration_set_error(mig, "Failed to train IVF-Flat index");
        return NULL;
    }

    if (migration_is_cancelled(mig)) {
        gv_ivfflat_destroy(index);
        return NULL;
    }

    for (size_t i = 0; i < mig->total_vectors; i++) {
        if (migration_is_cancelled(mig)) {
            gv_ivfflat_destroy(index);
            return NULL;
        }

        const float *vec_data = mig->source_data + i * mig->dimension;
        GV_Vector *vec = gv_vector_create_from_data(mig->dimension, vec_data);
        if (!vec) {
            gv_ivfflat_destroy(index);
            migration_set_error(mig, "Failed to create vector during IVF-Flat migration");
            return NULL;
        }

        if (gv_ivfflat_insert(index, vec) != 0) {
            gv_vector_destroy(vec);
            gv_ivfflat_destroy(index);
            migration_set_error(mig, "Failed to insert vector into IVF-Flat index");
            return NULL;
        }

        if ((i + 1) % MIGRATION_BATCH_SIZE == 0 || i + 1 == mig->total_vectors) {
            migration_update_progress(mig, i + 1);
        }
    }

    return index;
}

static void *create_pq_index(GV_Migration *mig)
{
    const GV_PQConfig *cfg = (const GV_PQConfig *)mig->new_index_config;
    void *index = gv_pq_create(mig->dimension, cfg);
    if (!index) {
        migration_set_error(mig, "Failed to create PQ index");
        return NULL;
    }

    /* PQ requires training before insertion */
    if (gv_pq_train(index, mig->source_data, mig->total_vectors) != 0) {
        gv_pq_destroy(index);
        migration_set_error(mig, "Failed to train PQ index");
        return NULL;
    }

    if (migration_is_cancelled(mig)) {
        gv_pq_destroy(index);
        return NULL;
    }

    for (size_t i = 0; i < mig->total_vectors; i++) {
        if (migration_is_cancelled(mig)) {
            gv_pq_destroy(index);
            return NULL;
        }

        const float *vec_data = mig->source_data + i * mig->dimension;
        GV_Vector *vec = gv_vector_create_from_data(mig->dimension, vec_data);
        if (!vec) {
            gv_pq_destroy(index);
            migration_set_error(mig, "Failed to create vector during PQ migration");
            return NULL;
        }

        if (gv_pq_insert(index, vec) != 0) {
            gv_vector_destroy(vec);
            gv_pq_destroy(index);
            migration_set_error(mig, "Failed to insert vector into PQ index");
            return NULL;
        }

        if ((i + 1) % MIGRATION_BATCH_SIZE == 0 || i + 1 == mig->total_vectors) {
            migration_update_progress(mig, i + 1);
        }
    }

    return index;
}

static void *create_lsh_index(GV_Migration *mig)
{
    const GV_LSHConfig *cfg = (const GV_LSHConfig *)mig->new_index_config;
    void *index = gv_lsh_create(mig->dimension, cfg, NULL);
    if (!index) {
        migration_set_error(mig, "Failed to create LSH index");
        return NULL;
    }

    for (size_t i = 0; i < mig->total_vectors; i++) {
        if (migration_is_cancelled(mig)) {
            gv_lsh_destroy(index);
            return NULL;
        }

        const float *vec_data = mig->source_data + i * mig->dimension;
        GV_Vector *vec = gv_vector_create_from_data(mig->dimension, vec_data);
        if (!vec) {
            gv_lsh_destroy(index);
            migration_set_error(mig, "Failed to create vector during LSH migration");
            return NULL;
        }

        if (gv_lsh_insert(index, vec) != 0) {
            gv_vector_destroy(vec);
            gv_lsh_destroy(index);
            migration_set_error(mig, "Failed to insert vector into LSH index");
            return NULL;
        }

        if ((i + 1) % MIGRATION_BATCH_SIZE == 0 || i + 1 == mig->total_vectors) {
            migration_update_progress(mig, i + 1);
        }
    }

    return index;
}

/* ---------- migration thread entry point ---------- */

static void *migration_thread_func(void *arg)
{
    GV_Migration *mig = (GV_Migration *)arg;

    pthread_mutex_lock(&mig->mutex);
    mig->status = GV_MIGRATION_RUNNING;
    mig->start_time_us = now_us();
    pthread_mutex_unlock(&mig->mutex);

    void *new_index = NULL;

    switch (mig->new_index_type) {
    case MIG_INDEX_KDTREE:
        new_index = create_kdtree_index(mig);
        break;
    case MIG_INDEX_HNSW:
        new_index = create_hnsw_index(mig);
        break;
    case MIG_INDEX_FLAT:
        new_index = create_flat_index(mig);
        break;
    case MIG_INDEX_IVFFLAT:
        new_index = create_ivfflat_index(mig);
        break;
    case MIG_INDEX_PQ:
        new_index = create_pq_index(mig);
        break;
    case MIG_INDEX_LSH:
        new_index = create_lsh_index(mig);
        break;
    default: {
        char buf[256];
        snprintf(buf, sizeof(buf), "Unsupported index type: %d", mig->new_index_type);
        migration_set_error(mig, buf);
        return NULL;
    }
    }

    pthread_mutex_lock(&mig->mutex);

    if (mig->cancel_requested) {
        /* The create_*_index helper already cleaned up the index on cancel */
        mig->status = GV_MIGRATION_CANCELLED;
        mig->error_message[0] = '\0';
    } else if (new_index != NULL) {
        mig->new_index = new_index;
        mig->status = GV_MIGRATION_COMPLETED;
        mig->progress = 1.0;
        mig->vectors_migrated = mig->total_vectors;
    }
    /* If new_index is NULL and not cancelled, status was already set to FAILED
       by the create helper via migration_set_error. */

    pthread_mutex_unlock(&mig->mutex);
    return NULL;
}

/* ================================================================
 *  Public API
 * ================================================================ */

GV_Migration *gv_migration_start(const float *source_data, size_t count,
                                  size_t dimension, int new_index_type,
                                  const void *new_index_config)
{
    if (!source_data || count == 0 || dimension == 0) {
        return NULL;
    }

    GV_Migration *mig = (GV_Migration *)calloc(1, sizeof(GV_Migration));
    if (!mig) {
        return NULL;
    }

    mig->status = GV_MIGRATION_PENDING;
    mig->progress = 0.0;
    mig->vectors_migrated = 0;
    mig->total_vectors = count;
    mig->start_time_us = 0;
    mig->dimension = dimension;
    mig->new_index_type = new_index_type;
    mig->source_data = source_data;
    mig->new_index_config = new_index_config;
    mig->new_index = NULL;
    mig->cancel_requested = 0;
    mig->error_message[0] = '\0';

    if (pthread_mutex_init(&mig->mutex, NULL) != 0) {
        free(mig);
        return NULL;
    }

    if (pthread_create(&mig->thread, NULL, migration_thread_func, mig) != 0) {
        pthread_mutex_destroy(&mig->mutex);
        free(mig);
        return NULL;
    }

    return mig;
}

int gv_migration_get_info(const GV_Migration *mig, GV_MigrationInfo *info)
{
    if (!mig || !info) {
        return -1;
    }

    /* Cast away const for mutex lock -- the mutex is logically mutable */
    GV_Migration *m = (GV_Migration *)(uintptr_t)mig;

    pthread_mutex_lock(&m->mutex);

    info->status = m->status;
    info->progress = m->progress;
    info->vectors_migrated = m->vectors_migrated;
    info->total_vectors = m->total_vectors;
    info->start_time_us = m->start_time_us;

    if (m->start_time_us > 0) {
        info->elapsed_us = now_us() - m->start_time_us;
    } else {
        info->elapsed_us = 0;
    }

    strncpy(info->error_message, m->error_message, sizeof(info->error_message) - 1);
    info->error_message[sizeof(info->error_message) - 1] = '\0';

    pthread_mutex_unlock(&m->mutex);
    return 0;
}

int gv_migration_wait(GV_Migration *mig)
{
    if (!mig) {
        return -1;
    }

    if (pthread_join(mig->thread, NULL) != 0) {
        return -1;
    }

    return (mig->status == GV_MIGRATION_COMPLETED) ? 0 : -1;
}

int gv_migration_cancel(GV_Migration *mig)
{
    if (!mig) {
        return -1;
    }

    pthread_mutex_lock(&mig->mutex);

    if (mig->status != GV_MIGRATION_PENDING && mig->status != GV_MIGRATION_RUNNING) {
        /* Migration already finished; cannot cancel */
        pthread_mutex_unlock(&mig->mutex);
        return -1;
    }

    mig->cancel_requested = 1;
    pthread_mutex_unlock(&mig->mutex);
    return 0;
}

void *gv_migration_take_index(GV_Migration *mig)
{
    if (!mig) {
        return NULL;
    }

    pthread_mutex_lock(&mig->mutex);

    if (mig->status != GV_MIGRATION_COMPLETED || mig->new_index == NULL) {
        pthread_mutex_unlock(&mig->mutex);
        return NULL;
    }

    void *index = mig->new_index;
    mig->new_index = NULL;
    pthread_mutex_unlock(&mig->mutex);

    return index;
}

void gv_migration_destroy(GV_Migration *mig)
{
    if (!mig) {
        return;
    }

    /* Ensure the thread has finished before destroying.
       If the caller did not call wait or cancel, request cancel and join. */
    pthread_mutex_lock(&mig->mutex);
    int still_running = (mig->status == GV_MIGRATION_PENDING ||
                         mig->status == GV_MIGRATION_RUNNING);
    if (still_running) {
        mig->cancel_requested = 1;
    }
    pthread_mutex_unlock(&mig->mutex);

    pthread_join(mig->thread, NULL);

    /* If the index was built but never taken, destroy it based on type */
    if (mig->new_index != NULL) {
        switch (mig->new_index_type) {
        case MIG_INDEX_KDTREE:
            gv_soa_storage_destroy((GV_SoAStorage *)mig->new_index);
            break;
        case MIG_INDEX_HNSW:
            gv_hnsw_destroy(mig->new_index);
            break;
        case MIG_INDEX_FLAT:
            gv_flat_destroy(mig->new_index);
            break;
        case MIG_INDEX_IVFFLAT:
            gv_ivfflat_destroy(mig->new_index);
            break;
        case MIG_INDEX_PQ:
            gv_pq_destroy(mig->new_index);
            break;
        case MIG_INDEX_LSH:
            gv_lsh_destroy(mig->new_index);
            break;
        default:
            /* Unknown type -- best effort: do nothing to avoid double free */
            break;
        }
        mig->new_index = NULL;
    }

    pthread_mutex_destroy(&mig->mutex);
    free(mig);
}
