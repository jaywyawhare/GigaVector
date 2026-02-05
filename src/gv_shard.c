/**
 * @file gv_shard.c
 * @brief Shard management implementation.
 */

#include "gigavector/gv_shard.h"
#include "gigavector/gv_database.h"
#include "gigavector/gv_auth.h"  /* For hashing */

#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

#define MAX_SHARDS 256
#define VIRTUAL_NODES_DEFAULT 150

typedef struct {
    uint32_t shard_id;
    char *node_address;
    GV_ShardState state;
    uint64_t vector_count;
    uint64_t capacity;
    uint32_t replica_count;
    uint64_t last_heartbeat;
    GV_Database *local_db;
} ShardEntry;

typedef struct {
    uint32_t hash;
    uint32_t shard_id;
} VirtualNode;

struct GV_ShardManager {
    GV_ShardConfig config;
    ShardEntry shards[MAX_SHARDS];
    size_t shard_count;

    /* Consistent hashing ring */
    VirtualNode *ring;
    size_t ring_size;

    /* Rebalancing state */
    int rebalancing;
    double rebalance_progress;

    pthread_rwlock_t rwlock;
};

/* ============================================================================
 * Configuration
 * ============================================================================ */

static const GV_ShardConfig DEFAULT_CONFIG = {
    .shard_count = 8,
    .virtual_nodes = VIRTUAL_NODES_DEFAULT,
    .strategy = GV_SHARD_CONSISTENT,
    .replication_factor = 1
};

void gv_shard_config_init(GV_ShardConfig *config) {
    if (!config) return;
    *config = DEFAULT_CONFIG;
}

/* ============================================================================
 * Hash Functions
 * ============================================================================ */

static uint32_t hash_key(const void *key, size_t len) {
    /* FNV-1a hash */
    uint32_t hash = 2166136261u;
    const unsigned char *data = (const unsigned char *)key;
    for (size_t i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= 16777619u;
    }
    return hash;
}

static uint32_t hash_shard_vnode(uint32_t shard_id, uint32_t vnode) {
    uint64_t val = ((uint64_t)shard_id << 32) | vnode;
    return hash_key(&val, sizeof(val));
}

/* ============================================================================
 * Ring Management
 * ============================================================================ */

static int compare_vnodes(const void *a, const void *b) {
    const VirtualNode *va = (const VirtualNode *)a;
    const VirtualNode *vb = (const VirtualNode *)b;
    if (va->hash < vb->hash) return -1;
    if (va->hash > vb->hash) return 1;
    return 0;
}

static void rebuild_ring(GV_ShardManager *mgr) {
    free(mgr->ring);
    mgr->ring = NULL;
    mgr->ring_size = 0;

    if (mgr->shard_count == 0) return;

    size_t total_vnodes = mgr->shard_count * mgr->config.virtual_nodes;
    mgr->ring = malloc(total_vnodes * sizeof(VirtualNode));
    if (!mgr->ring) return;

    size_t idx = 0;
    for (size_t i = 0; i < mgr->shard_count; i++) {
        if (mgr->shards[i].state == GV_SHARD_OFFLINE) continue;

        for (uint32_t v = 0; v < mgr->config.virtual_nodes; v++) {
            mgr->ring[idx].hash = hash_shard_vnode(mgr->shards[i].shard_id, v);
            mgr->ring[idx].shard_id = mgr->shards[i].shard_id;
            idx++;
        }
    }

    mgr->ring_size = idx;
    qsort(mgr->ring, mgr->ring_size, sizeof(VirtualNode), compare_vnodes);
}

static uint32_t find_shard_in_ring(GV_ShardManager *mgr, uint32_t hash) {
    if (mgr->ring_size == 0) return 0;

    /* Binary search for first node >= hash */
    size_t lo = 0, hi = mgr->ring_size;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (mgr->ring[mid].hash < hash) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    /* Wrap around if past end */
    if (lo >= mgr->ring_size) lo = 0;

    return mgr->ring[lo].shard_id;
}

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

GV_ShardManager *gv_shard_manager_create(const GV_ShardConfig *config) {
    GV_ShardManager *mgr = calloc(1, sizeof(GV_ShardManager));
    if (!mgr) return NULL;

    mgr->config = config ? *config : DEFAULT_CONFIG;

    if (pthread_rwlock_init(&mgr->rwlock, NULL) != 0) {
        free(mgr);
        return NULL;
    }

    return mgr;
}

void gv_shard_manager_destroy(GV_ShardManager *mgr) {
    if (!mgr) return;

    for (size_t i = 0; i < mgr->shard_count; i++) {
        free(mgr->shards[i].node_address);
    }

    free(mgr->ring);
    pthread_rwlock_destroy(&mgr->rwlock);
    free(mgr);
}

/* ============================================================================
 * Shard Operations
 * ============================================================================ */

int gv_shard_add(GV_ShardManager *mgr, uint32_t shard_id, const char *node_address) {
    if (!mgr || !node_address) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    /* Check if already exists */
    for (size_t i = 0; i < mgr->shard_count; i++) {
        if (mgr->shards[i].shard_id == shard_id) {
            pthread_rwlock_unlock(&mgr->rwlock);
            return -1;
        }
    }

    if (mgr->shard_count >= MAX_SHARDS) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    ShardEntry *entry = &mgr->shards[mgr->shard_count];
    entry->shard_id = shard_id;
    entry->node_address = strdup(node_address);
    entry->state = GV_SHARD_ACTIVE;
    entry->vector_count = 0;
    entry->capacity = 0;
    entry->replica_count = mgr->config.replication_factor;
    entry->last_heartbeat = (uint64_t)time(NULL);
    entry->local_db = NULL;

    mgr->shard_count++;

    rebuild_ring(mgr);

    pthread_rwlock_unlock(&mgr->rwlock);
    return 0;
}

int gv_shard_remove(GV_ShardManager *mgr, uint32_t shard_id) {
    if (!mgr) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    for (size_t i = 0; i < mgr->shard_count; i++) {
        if (mgr->shards[i].shard_id == shard_id) {
            free(mgr->shards[i].node_address);

            /* Shift remaining */
            for (size_t j = i; j < mgr->shard_count - 1; j++) {
                mgr->shards[j] = mgr->shards[j + 1];
            }
            mgr->shard_count--;

            rebuild_ring(mgr);

            pthread_rwlock_unlock(&mgr->rwlock);
            return 0;
        }
    }

    pthread_rwlock_unlock(&mgr->rwlock);
    return -1;
}

int gv_shard_for_vector(GV_ShardManager *mgr, uint64_t vector_id) {
    if (!mgr) return -1;

    pthread_rwlock_rdlock(&mgr->rwlock);

    int shard_id;
    if (mgr->config.strategy == GV_SHARD_HASH) {
        /* Simple modulo */
        shard_id = vector_id % mgr->config.shard_count;
    } else {
        /* Consistent hashing */
        uint32_t hash = hash_key(&vector_id, sizeof(vector_id));
        shard_id = find_shard_in_ring(mgr, hash);
    }

    pthread_rwlock_unlock(&mgr->rwlock);
    return shard_id;
}

int gv_shard_for_key(GV_ShardManager *mgr, const void *key, size_t key_len) {
    if (!mgr || !key) return -1;

    pthread_rwlock_rdlock(&mgr->rwlock);

    uint32_t hash = hash_key(key, key_len);
    int shard_id = find_shard_in_ring(mgr, hash);

    pthread_rwlock_unlock(&mgr->rwlock);
    return shard_id;
}

int gv_shard_get_info(GV_ShardManager *mgr, uint32_t shard_id, GV_ShardInfo *info) {
    if (!mgr || !info) return -1;

    pthread_rwlock_rdlock(&mgr->rwlock);

    for (size_t i = 0; i < mgr->shard_count; i++) {
        if (mgr->shards[i].shard_id == shard_id) {
            info->shard_id = mgr->shards[i].shard_id;
            info->node_address = mgr->shards[i].node_address ? strdup(mgr->shards[i].node_address) : NULL;
            info->state = mgr->shards[i].state;
            info->vector_count = mgr->shards[i].vector_count;
            info->capacity = mgr->shards[i].capacity;
            info->replica_count = mgr->shards[i].replica_count;
            info->last_heartbeat = mgr->shards[i].last_heartbeat;

            pthread_rwlock_unlock(&mgr->rwlock);
            return 0;
        }
    }

    pthread_rwlock_unlock(&mgr->rwlock);
    return -1;
}

int gv_shard_list(GV_ShardManager *mgr, GV_ShardInfo **shards, size_t *count) {
    if (!mgr || !shards || !count) return -1;

    pthread_rwlock_rdlock(&mgr->rwlock);

    *count = mgr->shard_count;
    if (*count == 0) {
        *shards = NULL;
        pthread_rwlock_unlock(&mgr->rwlock);
        return 0;
    }

    *shards = malloc(*count * sizeof(GV_ShardInfo));
    if (!*shards) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    for (size_t i = 0; i < *count; i++) {
        (*shards)[i].shard_id = mgr->shards[i].shard_id;
        (*shards)[i].node_address = mgr->shards[i].node_address ? strdup(mgr->shards[i].node_address) : NULL;
        (*shards)[i].state = mgr->shards[i].state;
        (*shards)[i].vector_count = mgr->shards[i].vector_count;
        (*shards)[i].capacity = mgr->shards[i].capacity;
        (*shards)[i].replica_count = mgr->shards[i].replica_count;
        (*shards)[i].last_heartbeat = mgr->shards[i].last_heartbeat;
    }

    pthread_rwlock_unlock(&mgr->rwlock);
    return 0;
}

void gv_shard_free_list(GV_ShardInfo *shards, size_t count) {
    if (!shards) return;
    for (size_t i = 0; i < count; i++) {
        free(shards[i].node_address);
    }
    free(shards);
}

int gv_shard_set_state(GV_ShardManager *mgr, uint32_t shard_id, GV_ShardState state) {
    if (!mgr) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    for (size_t i = 0; i < mgr->shard_count; i++) {
        if (mgr->shards[i].shard_id == shard_id) {
            mgr->shards[i].state = state;

            /* Rebuild ring if state changed to/from offline */
            rebuild_ring(mgr);

            pthread_rwlock_unlock(&mgr->rwlock);
            return 0;
        }
    }

    pthread_rwlock_unlock(&mgr->rwlock);
    return -1;
}

/* ============================================================================
 * Rebalancing
 * ============================================================================ */

/**
 * @brief Internal structure for tracking rebalance moves.
 */
typedef struct {
    uint32_t from_shard;
    uint32_t to_shard;
    size_t count;
} RebalanceMove;

/**
 * @brief Perform actual vector migration between local shards.
 */
static int migrate_vectors(GV_ShardManager *mgr, uint32_t from_shard, uint32_t to_shard, size_t count) {
    ShardEntry *from = NULL;
    ShardEntry *to = NULL;

    for (size_t i = 0; i < mgr->shard_count; i++) {
        if (mgr->shards[i].shard_id == from_shard) from = &mgr->shards[i];
        if (mgr->shards[i].shard_id == to_shard) to = &mgr->shards[i];
    }

    if (!from || !to || !from->local_db || !to->local_db) {
        return -1;  /* Cannot migrate without local databases */
    }

    size_t dimension = from->local_db->dimension;
    if (dimension != to->local_db->dimension) {
        return -1;  /* Dimension mismatch */
    }

    /* Migrate vectors one at a time (could be batched for efficiency) */
    size_t migrated = 0;
    size_t from_count = gv_database_count(from->local_db);

    for (size_t i = 0; i < count && from_count > 0; i++) {
        /* Get last vector from source (simpler than random) */
        size_t idx = from_count - 1;

        /* Get vector data */
        const float *vec = gv_database_get_vector(from->local_db, idx);
        if (!vec) break;

        /* Add to destination */
        if (gv_db_add_vector(to->local_db, vec, dimension) != 0) {
            break;
        }

        /* Mark as deleted in source */
        gv_db_delete_vector_by_index(from->local_db, idx);

        migrated++;
        from_count--;

        /* Update counts */
        from->vector_count = from_count;
        to->vector_count = gv_database_count(to->local_db);
    }

    return (int)migrated;
}

int gv_shard_rebalance_start(GV_ShardManager *mgr) {
    if (!mgr) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    if (mgr->rebalancing) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    mgr->rebalancing = 1;
    mgr->rebalance_progress = 0.0;

    /* Calculate total vectors and target per shard */
    uint64_t total_vectors = 0;
    size_t active_shards = 0;

    for (size_t i = 0; i < mgr->shard_count; i++) {
        if (mgr->shards[i].state == GV_SHARD_ACTIVE) {
            total_vectors += mgr->shards[i].vector_count;
            active_shards++;
        }
    }

    if (active_shards == 0) {
        mgr->rebalancing = 0;
        pthread_rwlock_unlock(&mgr->rwlock);
        return 0;  /* Nothing to rebalance */
    }

    uint64_t target_per_shard = total_vectors / active_shards;
    uint64_t tolerance = target_per_shard / 10;  /* 10% tolerance */
    if (tolerance < 10) tolerance = 10;

    /* Identify overloaded and underloaded shards */
    size_t *overloaded = calloc(mgr->shard_count, sizeof(size_t));
    size_t *underloaded = calloc(mgr->shard_count, sizeof(size_t));
    int64_t *excess = calloc(mgr->shard_count, sizeof(int64_t));

    if (!overloaded || !underloaded || !excess) {
        free(overloaded);
        free(underloaded);
        free(excess);
        mgr->rebalancing = 0;
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    size_t over_count = 0, under_count = 0;

    for (size_t i = 0; i < mgr->shard_count; i++) {
        if (mgr->shards[i].state != GV_SHARD_ACTIVE) continue;

        int64_t diff = (int64_t)mgr->shards[i].vector_count - (int64_t)target_per_shard;
        excess[i] = diff;

        if (diff > (int64_t)tolerance) {
            overloaded[over_count++] = i;
        } else if (diff < -(int64_t)tolerance) {
            underloaded[under_count++] = i;
        }
    }

    /* Perform migrations */
    size_t total_moves = 0;
    size_t completed_moves = 0;

    /* Calculate total moves needed */
    for (size_t i = 0; i < over_count; i++) {
        if (excess[overloaded[i]] > (int64_t)tolerance) {
            total_moves += (size_t)(excess[overloaded[i]] - (int64_t)tolerance);
        }
    }

    pthread_rwlock_unlock(&mgr->rwlock);

    /* Perform actual migrations (outside lock for long operations) */
    for (size_t o = 0; o < over_count && under_count > 0; o++) {
        size_t over_idx = overloaded[o];
        int64_t to_move = excess[over_idx] - (int64_t)tolerance;

        for (size_t u = 0; u < under_count && to_move > 0; u++) {
            size_t under_idx = underloaded[u];
            int64_t can_receive = (int64_t)tolerance - excess[under_idx];

            if (can_receive <= 0) continue;

            size_t move_count = (size_t)(to_move < can_receive ? to_move : can_receive);

            int migrated = migrate_vectors(mgr,
                mgr->shards[over_idx].shard_id,
                mgr->shards[under_idx].shard_id,
                move_count);

            if (migrated > 0) {
                excess[over_idx] -= migrated;
                excess[under_idx] += migrated;
                to_move -= migrated;
                completed_moves += migrated;

                /* Update progress */
                pthread_rwlock_wrlock(&mgr->rwlock);
                if (total_moves > 0) {
                    mgr->rebalance_progress = (double)completed_moves / (double)total_moves;
                }
                pthread_rwlock_unlock(&mgr->rwlock);
            }
        }
    }

    /* Mark rebalancing complete */
    pthread_rwlock_wrlock(&mgr->rwlock);
    mgr->rebalancing = 0;
    mgr->rebalance_progress = 1.0;
    pthread_rwlock_unlock(&mgr->rwlock);

    free(overloaded);
    free(underloaded);
    free(excess);

    return 0;
}

int gv_shard_rebalance_status(GV_ShardManager *mgr, double *progress) {
    if (!mgr || !progress) return -1;

    pthread_rwlock_rdlock(&mgr->rwlock);
    *progress = mgr->rebalance_progress;
    int running = mgr->rebalancing;
    pthread_rwlock_unlock(&mgr->rwlock);

    return running;
}

int gv_shard_rebalance_cancel(GV_ShardManager *mgr) {
    if (!mgr) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);
    mgr->rebalancing = 0;
    pthread_rwlock_unlock(&mgr->rwlock);

    return 0;
}

/* ============================================================================
 * Local Shard
 * ============================================================================ */

int gv_shard_attach_local(GV_ShardManager *mgr, uint32_t shard_id, GV_Database *db) {
    if (!mgr || !db) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    for (size_t i = 0; i < mgr->shard_count; i++) {
        if (mgr->shards[i].shard_id == shard_id) {
            mgr->shards[i].local_db = db;
            mgr->shards[i].vector_count = gv_database_count(db);
            pthread_rwlock_unlock(&mgr->rwlock);
            return 0;
        }
    }

    pthread_rwlock_unlock(&mgr->rwlock);
    return -1;
}

GV_Database *gv_shard_get_local_db(GV_ShardManager *mgr, uint32_t shard_id) {
    if (!mgr) return NULL;

    pthread_rwlock_rdlock(&mgr->rwlock);

    for (size_t i = 0; i < mgr->shard_count; i++) {
        if (mgr->shards[i].shard_id == shard_id) {
            GV_Database *db = mgr->shards[i].local_db;
            pthread_rwlock_unlock(&mgr->rwlock);
            return db;
        }
    }

    pthread_rwlock_unlock(&mgr->rwlock);
    return NULL;
}
