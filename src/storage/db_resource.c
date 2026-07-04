#include <pthread.h>

#include "storage/db_internal.h"

int db_check_resource_limits(GV_Database *db, size_t additional_vectors, size_t additional_memory) {
    if (db == NULL) {
        return -1;
    }

    pthread_mutex_lock(&db->resource_mutex);

    if (db->resource_limits.max_vectors > 0) {
        if (db->count + additional_vectors > db->resource_limits.max_vectors) {
            pthread_mutex_unlock(&db->resource_mutex);
            return -1;
        }
    }

    if (db->resource_limits.max_memory_bytes > 0) {
        size_t estimated_new_memory = db->current_memory_bytes + additional_memory;
        if (estimated_new_memory > db->resource_limits.max_memory_bytes) {
            pthread_mutex_unlock(&db->resource_mutex);
            return -1;
        }
    }

    if (db->resource_limits.max_concurrent_operations > 0) {
        if (db->current_concurrent_ops >= db->resource_limits.max_concurrent_operations) {
            pthread_mutex_unlock(&db->resource_mutex);
            return -1;
        }
    }

    pthread_mutex_unlock(&db->resource_mutex);
    return 0;
}

void db_increment_concurrent_ops(GV_Database *db) {
    if (db == NULL) {
        return;
    }
    pthread_mutex_lock(&db->resource_mutex);
    db->current_concurrent_ops++;
    pthread_mutex_unlock(&db->resource_mutex);
}

void db_decrement_concurrent_ops(GV_Database *db) {
    if (db == NULL) {
        return;
    }
    pthread_mutex_lock(&db->resource_mutex);
    if (db->current_concurrent_ops > 0) {
        db->current_concurrent_ops--;
    }
    pthread_mutex_unlock(&db->resource_mutex);
}

int db_set_resource_limits(GV_Database *db, const GV_ResourceLimits *limits) {
    if (db == NULL || limits == NULL) {
        return -1;
    }

    pthread_mutex_lock(&db->resource_mutex);
    db->resource_limits.max_memory_bytes = limits->max_memory_bytes;
    db->resource_limits.max_vectors = limits->max_vectors;
    db->resource_limits.max_concurrent_operations = limits->max_concurrent_operations;
    pthread_mutex_unlock(&db->resource_mutex);

    db_update_memory_usage(db);

    return 0;
}

void db_get_resource_limits(const GV_Database *db, GV_ResourceLimits *limits) {
    if (db == NULL || limits == NULL) {
        return;
    }

    pthread_mutex_lock((pthread_mutex_t *)&db->resource_mutex);
    limits->max_memory_bytes = db->resource_limits.max_memory_bytes;
    limits->max_vectors = db->resource_limits.max_vectors;
    limits->max_concurrent_operations = db->resource_limits.max_concurrent_operations;
    pthread_mutex_unlock((pthread_mutex_t *)&db->resource_mutex);
}

size_t db_get_memory_usage(const GV_Database *db) {
    if (db == NULL) {
        return 0;
    }

    db_update_memory_usage((GV_Database *)db);

    pthread_mutex_lock((pthread_mutex_t *)&db->resource_mutex);
    size_t usage = db->current_memory_bytes;
    pthread_mutex_unlock((pthread_mutex_t *)&db->resource_mutex);

    return usage;
}

size_t db_get_concurrent_operations(const GV_Database *db) {
    if (db == NULL) {
        return 0;
    }

    pthread_mutex_lock((pthread_mutex_t *)&db->resource_mutex);
    size_t ops = db->current_concurrent_ops;
    pthread_mutex_unlock((pthread_mutex_t *)&db->resource_mutex);

    return ops;
}
