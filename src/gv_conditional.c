/**
 * @file gv_conditional.c
 * @brief CAS-style conditional mutation implementation.
 *
 * Provides compare-and-swap semantics for vector and metadata updates
 * with optimistic concurrency control via per-vector version tracking.
 */

#define _POSIX_C_SOURCE 200112L

#include "gigavector/gv_conditional.h"
#include "gigavector/gv_database.h"
#include "gigavector/gv_metadata.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

/**
 * @brief Per-vector version tracking entry.
 */
typedef struct {
    uint64_t version;              /**< Monotonically increasing version counter. */
    uint64_t updated_at;           /**< Last update timestamp (microseconds since epoch). */
} GV_VersionSlot;

/**
 * @brief Initial capacity for the version tracking array.
 */
#define COND_INITIAL_CAPACITY 1024

/**
 * @brief Conditional-update manager internal structure.
 */
struct GV_CondManager {
    GV_Database *db;               /**< Bound database handle. */
    GV_VersionSlot *slots;         /**< Dynamic array of version slots indexed by vector index. */
    size_t slot_count;             /**< Number of allocated slots. */
};

/* ============================================================================
 * Helpers
 * ============================================================================ */

/**
 * @brief Get current time in microseconds since epoch.
 */
static uint64_t now_microseconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

/**
 * @brief Ensure the version slots array can hold at least (index + 1) entries.
 *
 * Newly allocated slots are zero-initialized (version = 0, updated_at = 0).
 *
 * @return 0 on success, -1 on allocation failure.
 */
static int ensure_slot_capacity(GV_CondManager *mgr, size_t index)
{
    if (index < mgr->slot_count) return 0;

    size_t new_count = mgr->slot_count == 0 ? COND_INITIAL_CAPACITY : mgr->slot_count;
    while (new_count <= index) {
        new_count *= 2;
    }

    GV_VersionSlot *tmp = realloc(mgr->slots, new_count * sizeof(GV_VersionSlot));
    if (!tmp) return -1;

    /* Zero-initialize newly allocated region */
    memset(tmp + mgr->slot_count, 0,
           (new_count - mgr->slot_count) * sizeof(GV_VersionSlot));

    mgr->slots = tmp;
    mgr->slot_count = new_count;
    return 0;
}

/**
 * @brief Find a metadata value by key in a linked list.
 *
 * @return Pointer to the value string, or NULL if not found.
 */
static const char *find_metadata_value(const GV_Metadata *meta, const char *key)
{
    const GV_Metadata *cur = meta;
    while (cur) {
        if (cur->key && strcmp(cur->key, key) == 0) {
            return cur->value;
        }
        cur = cur->next;
    }
    return NULL;
}

/**
 * @brief Check whether a metadata field exists in a linked list.
 *
 * @return 1 if found, 0 if not found.
 */
static int metadata_field_exists(const GV_Metadata *meta, const char *key)
{
    return find_metadata_value(meta, key) != NULL ? 1 : 0;
}

/**
 * @brief Evaluate a single condition against the current state of a vector.
 *
 * Caller must hold the database write lock.
 *
 * @return GV_COND_OK if the condition holds, GV_COND_FAILED otherwise.
 */
static GV_ConditionalResult evaluate_condition(const GV_CondManager *mgr,
                                                size_t index,
                                                const GV_Condition *cond)
{
    GV_Database *db = mgr->db;
    GV_SoAStorage *soa = db->soa_storage;

    switch (cond->type) {

    case GV_COND_VERSION_EQ: {
        uint64_t cur_ver = 0;
        if (index < mgr->slot_count) {
            cur_ver = mgr->slots[index].version;
        }
        return (cur_ver == cond->version) ? GV_COND_OK : GV_COND_CONFLICT;
    }

    case GV_COND_VERSION_LT: {
        uint64_t cur_ver = 0;
        if (index < mgr->slot_count) {
            cur_ver = mgr->slots[index].version;
        }
        return (cur_ver < cond->version) ? GV_COND_OK : GV_COND_CONFLICT;
    }

    case GV_COND_METADATA_EQ: {
        if (!cond->field_name || !cond->field_value) return GV_COND_FAILED;
        GV_Metadata *meta = gv_soa_storage_get_metadata(soa, index);
        const char *val = find_metadata_value(meta, cond->field_name);
        if (!val || strcmp(val, cond->field_value) != 0) {
            return GV_COND_FAILED;
        }
        return GV_COND_OK;
    }

    case GV_COND_METADATA_EXISTS: {
        if (!cond->field_name) return GV_COND_FAILED;
        GV_Metadata *meta = gv_soa_storage_get_metadata(soa, index);
        return metadata_field_exists(meta, cond->field_name) ? GV_COND_OK : GV_COND_FAILED;
    }

    case GV_COND_METADATA_NOT_EXISTS: {
        if (!cond->field_name) return GV_COND_FAILED;
        GV_Metadata *meta = gv_soa_storage_get_metadata(soa, index);
        return metadata_field_exists(meta, cond->field_name) ? GV_COND_FAILED : GV_COND_OK;
    }

    case GV_COND_NOT_DELETED: {
        int deleted = gv_soa_storage_is_deleted(soa, index);
        return (deleted == 0) ? GV_COND_OK : GV_COND_FAILED;
    }

    default:
        return GV_COND_FAILED;
    }
}

/**
 * @brief Evaluate all conditions for a given vector index.
 *
 * Returns the first non-OK result, or GV_COND_OK if all conditions pass.
 * Caller must hold the database write lock.
 */
static GV_ConditionalResult evaluate_all(const GV_CondManager *mgr,
                                          size_t index,
                                          const GV_Condition *conditions,
                                          size_t condition_count)
{
    for (size_t i = 0; i < condition_count; i++) {
        GV_ConditionalResult r = evaluate_condition(mgr, index, &conditions[i]);
        if (r != GV_COND_OK) return r;
    }
    return GV_COND_OK;
}

/**
 * @brief Bump the version for a vector index and record the update timestamp.
 *
 * Caller must hold the database write lock and must have already ensured
 * slot capacity via ensure_slot_capacity().
 */
static void bump_version(GV_CondManager *mgr, size_t index)
{
    mgr->slots[index].version++;
    mgr->slots[index].updated_at = now_microseconds();
}

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

GV_CondManager *gv_cond_create(void *db)
{
    if (!db) return NULL;

    GV_CondManager *mgr = calloc(1, sizeof(GV_CondManager));
    if (!mgr) return NULL;

    mgr->db = (GV_Database *)db;
    mgr->slots = NULL;
    mgr->slot_count = 0;

    return mgr;
}

void gv_cond_destroy(GV_CondManager *mgr)
{
    if (!mgr) return;
    free(mgr->slots);
    free(mgr);
}

/* ============================================================================
 * Conditional Vector Update
 * ============================================================================ */

GV_ConditionalResult gv_cond_update_vector(GV_CondManager *mgr, size_t index,
                                            const float *new_data, size_t dimension,
                                            const GV_Condition *conditions,
                                            size_t condition_count)
{
    if (!mgr || !new_data) return GV_COND_FAILED;

    GV_Database *db = mgr->db;

    /* Validate dimension */
    if (dimension != db->dimension) return GV_COND_FAILED;

    /* Acquire write lock */
    pthread_rwlock_wrlock(&db->rwlock);

    /* Validate index bounds */
    if (!db->soa_storage || index >= db->soa_storage->count) {
        pthread_rwlock_unlock(&db->rwlock);
        return GV_COND_NOT_FOUND;
    }

    /* Ensure version tracking capacity */
    if (ensure_slot_capacity(mgr, index) != 0) {
        pthread_rwlock_unlock(&db->rwlock);
        return GV_COND_FAILED;
    }

    /* Evaluate conditions */
    GV_ConditionalResult result = evaluate_all(mgr, index, conditions, condition_count);
    if (result != GV_COND_OK) {
        pthread_rwlock_unlock(&db->rwlock);
        return result;
    }

    /* Apply mutation */
    if (gv_soa_storage_update_data(db->soa_storage, index, new_data) != 0) {
        pthread_rwlock_unlock(&db->rwlock);
        return GV_COND_FAILED;
    }

    /* Bump version */
    bump_version(mgr, index);

    pthread_rwlock_unlock(&db->rwlock);
    return GV_COND_OK;
}

/* ============================================================================
 * Conditional Metadata Update
 * ============================================================================ */

GV_ConditionalResult gv_cond_update_metadata(GV_CondManager *mgr, size_t index,
                                              const char *key, const char *value,
                                              const GV_Condition *conditions,
                                              size_t condition_count)
{
    if (!mgr || !key || !value) return GV_COND_FAILED;

    GV_Database *db = mgr->db;

    /* Acquire write lock */
    pthread_rwlock_wrlock(&db->rwlock);

    /* Validate index bounds */
    if (!db->soa_storage || index >= db->soa_storage->count) {
        pthread_rwlock_unlock(&db->rwlock);
        return GV_COND_NOT_FOUND;
    }

    /* Ensure version tracking capacity */
    if (ensure_slot_capacity(mgr, index) != 0) {
        pthread_rwlock_unlock(&db->rwlock);
        return GV_COND_FAILED;
    }

    /* Evaluate conditions */
    GV_ConditionalResult result = evaluate_all(mgr, index, conditions, condition_count);
    if (result != GV_COND_OK) {
        pthread_rwlock_unlock(&db->rwlock);
        return result;
    }

    /* Build new metadata: clone existing list, then set/update the key */
    GV_Metadata *existing = gv_soa_storage_get_metadata(db->soa_storage, index);

    /* Clone existing metadata into a temporary GV_Vector for set_metadata */
    GV_Metadata *new_list = NULL;
    GV_Metadata **tail = &new_list;
    for (const GV_Metadata *cur = existing; cur; cur = cur->next) {
        GV_Metadata *node = malloc(sizeof(GV_Metadata));
        if (!node) {
            gv_metadata_free(new_list);
            pthread_rwlock_unlock(&db->rwlock);
            return GV_COND_FAILED;
        }
        node->key = strdup(cur->key);
        node->value = strdup(cur->value);
        node->next = NULL;
        if (!node->key || !node->value) {
            free(node->key);
            free(node->value);
            free(node);
            gv_metadata_free(new_list);
            pthread_rwlock_unlock(&db->rwlock);
            return GV_COND_FAILED;
        }
        *tail = node;
        tail = &node->next;
    }

    /* Update or insert the target key */
    int found = 0;
    for (GV_Metadata *cur = new_list; cur; cur = cur->next) {
        if (strcmp(cur->key, key) == 0) {
            char *dup = strdup(value);
            if (!dup) {
                gv_metadata_free(new_list);
                pthread_rwlock_unlock(&db->rwlock);
                return GV_COND_FAILED;
            }
            free(cur->value);
            cur->value = dup;
            found = 1;
            break;
        }
    }

    if (!found) {
        GV_Metadata *node = malloc(sizeof(GV_Metadata));
        if (!node) {
            gv_metadata_free(new_list);
            pthread_rwlock_unlock(&db->rwlock);
            return GV_COND_FAILED;
        }
        node->key = strdup(key);
        node->value = strdup(value);
        node->next = new_list;
        if (!node->key || !node->value) {
            free(node->key);
            free(node->value);
            free(node);
            gv_metadata_free(new_list);
            pthread_rwlock_unlock(&db->rwlock);
            return GV_COND_FAILED;
        }
        new_list = node;
    }

    /* Apply: gv_soa_storage_update_metadata transfers ownership of new_list */
    if (gv_soa_storage_update_metadata(db->soa_storage, index, new_list) != 0) {
        gv_metadata_free(new_list);
        pthread_rwlock_unlock(&db->rwlock);
        return GV_COND_FAILED;
    }

    /* Bump version */
    bump_version(mgr, index);

    pthread_rwlock_unlock(&db->rwlock);
    return GV_COND_OK;
}

/* ============================================================================
 * Conditional Delete
 * ============================================================================ */

GV_ConditionalResult gv_cond_delete(GV_CondManager *mgr, size_t index,
                                     const GV_Condition *conditions,
                                     size_t condition_count)
{
    if (!mgr) return GV_COND_FAILED;

    GV_Database *db = mgr->db;

    /* Acquire write lock */
    pthread_rwlock_wrlock(&db->rwlock);

    /* Validate index bounds */
    if (!db->soa_storage || index >= db->soa_storage->count) {
        pthread_rwlock_unlock(&db->rwlock);
        return GV_COND_NOT_FOUND;
    }

    /* Ensure version tracking capacity */
    if (ensure_slot_capacity(mgr, index) != 0) {
        pthread_rwlock_unlock(&db->rwlock);
        return GV_COND_FAILED;
    }

    /* Evaluate conditions */
    GV_ConditionalResult result = evaluate_all(mgr, index, conditions, condition_count);
    if (result != GV_COND_OK) {
        pthread_rwlock_unlock(&db->rwlock);
        return result;
    }

    /* Apply deletion */
    if (gv_soa_storage_mark_deleted(db->soa_storage, index) != 0) {
        pthread_rwlock_unlock(&db->rwlock);
        return GV_COND_FAILED;
    }

    /* Bump version */
    bump_version(mgr, index);

    pthread_rwlock_unlock(&db->rwlock);
    return GV_COND_OK;
}

/* ============================================================================
 * Version Queries
 * ============================================================================ */

uint64_t gv_cond_get_version(const GV_CondManager *mgr, size_t index)
{
    if (!mgr) return 0;

    /* Read lock is sufficient for a version query */
    pthread_rwlock_rdlock(&mgr->db->rwlock);

    uint64_t ver = 0;
    if (index < mgr->slot_count) {
        ver = mgr->slots[index].version;
    }

    pthread_rwlock_unlock(&mgr->db->rwlock);
    return ver;
}

/* ============================================================================
 * Batch Operations
 * ============================================================================ */

int gv_cond_batch_update(GV_CondManager *mgr,
                          const size_t *indices,
                          const float **vectors,
                          const GV_Condition **conditions,
                          const size_t *condition_counts,
                          size_t batch_size,
                          GV_ConditionalResult *results)
{
    if (!mgr || !indices || !vectors || !conditions || !condition_counts
        || batch_size == 0 || !results) {
        return -1;
    }

    GV_Database *db = mgr->db;
    size_t dimension = db->dimension;
    int success_count = 0;

    /* Acquire write lock once for the entire batch */
    pthread_rwlock_wrlock(&db->rwlock);

    for (size_t i = 0; i < batch_size; i++) {
        size_t idx = indices[i];

        /* Bounds check */
        if (!db->soa_storage || idx >= db->soa_storage->count) {
            results[i] = GV_COND_NOT_FOUND;
            continue;
        }

        /* Ensure slot capacity */
        if (ensure_slot_capacity(mgr, idx) != 0) {
            results[i] = GV_COND_FAILED;
            continue;
        }

        /* Evaluate conditions */
        GV_ConditionalResult r = evaluate_all(mgr, idx,
                                               conditions[i],
                                               condition_counts[i]);
        if (r != GV_COND_OK) {
            results[i] = r;
            continue;
        }

        /* Apply mutation */
        if (gv_soa_storage_update_data(db->soa_storage, idx, vectors[i]) != 0) {
            results[i] = GV_COND_FAILED;
            continue;
        }

        /* Bump version */
        bump_version(mgr, idx);
        results[i] = GV_COND_OK;
        success_count++;
    }

    pthread_rwlock_unlock(&db->rwlock);

    /* Suppress unused-variable warning for dimension; it is implicitly
     * validated by gv_soa_storage_update_data against the SoA storage. */
    (void)dimension;

    return success_count;
}

/* ============================================================================
 * Convenience Wrappers
 * ============================================================================ */

GV_ConditionalResult gv_cond_migrate_embedding(GV_CondManager *mgr, size_t index,
                                                const float *new_embedding,
                                                size_t dimension,
                                                uint64_t expected_version)
{
    if (!mgr || !new_embedding) return GV_COND_FAILED;

    /* Build a VERSION_EQ condition for the expected version */
    GV_Condition cond;
    memset(&cond, 0, sizeof(cond));
    cond.type = GV_COND_VERSION_EQ;
    cond.version = expected_version;

    return gv_cond_update_vector(mgr, index, new_embedding, dimension, &cond, 1);
}
