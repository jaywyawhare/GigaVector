/**
 * @file gv_cluster.c
 * @brief Cluster management implementation.
 */

#include "gigavector/gv_cluster.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

#define MAX_NODES 64

typedef struct {
    char *node_id;
    char *address;
    GV_NodeRole role;
    GV_NodeState state;
    uint32_t *shard_ids;
    size_t shard_count;
    uint64_t last_heartbeat;
    double load;
} NodeEntry;

struct GV_Cluster {
    GV_ClusterConfig config;
    char *local_node_id;
    GV_ShardManager *shard_mgr;

    NodeEntry nodes[MAX_NODES];
    size_t node_count;

    /* Heartbeat thread */
    pthread_t heartbeat_thread;
    int heartbeat_running;
    int stop_requested;

    pthread_rwlock_t rwlock;
    pthread_mutex_t state_mutex;
    pthread_cond_t ready_cond;
    int is_ready;
};

/* ============================================================================
 * Configuration
 * ============================================================================ */

static const GV_ClusterConfig DEFAULT_CONFIG = {
    .node_id = NULL,
    .listen_address = "0.0.0.0:7000",
    .seed_nodes = NULL,
    .role = GV_NODE_DATA,
    .heartbeat_interval_ms = 1000,
    .failure_timeout_ms = 5000
};

void gv_cluster_config_init(GV_ClusterConfig *config) {
    if (!config) return;
    *config = DEFAULT_CONFIG;
}

/* ============================================================================
 * Internal Helpers
 * ============================================================================ */

static char *generate_node_id(void) {
    char id[64];
    snprintf(id, sizeof(id), "node-%lx-%d",
             (unsigned long)time(NULL), (int)getpid());
    return strdup(id);
}

static NodeEntry *find_node(GV_Cluster *cluster, const char *node_id) {
    for (size_t i = 0; i < cluster->node_count; i++) {
        if (strcmp(cluster->nodes[i].node_id, node_id) == 0) {
            return &cluster->nodes[i];
        }
    }
    return NULL;
}

static void *heartbeat_thread_func(void *arg) {
    GV_Cluster *cluster = (GV_Cluster *)arg;

    while (!cluster->stop_requested) {
        /* Sleep for heartbeat interval */
        usleep(cluster->config.heartbeat_interval_ms * 1000);

        if (cluster->stop_requested) break;

        pthread_rwlock_wrlock(&cluster->rwlock);

        uint64_t now = (uint64_t)time(NULL);
        uint64_t timeout = cluster->config.failure_timeout_ms / 1000;

        /* Check node health */
        for (size_t i = 0; i < cluster->node_count; i++) {
            if (cluster->nodes[i].state == GV_NODE_ACTIVE) {
                if (now - cluster->nodes[i].last_heartbeat > timeout) {
                    cluster->nodes[i].state = GV_NODE_DEAD;
                }
            }
        }

        /* Update local node heartbeat */
        NodeEntry *local = find_node(cluster, cluster->local_node_id);
        if (local) {
            local->last_heartbeat = now;
        }

        pthread_rwlock_unlock(&cluster->rwlock);

        /* TODO: Send heartbeats to other nodes */
    }

    return NULL;
}

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

GV_Cluster *gv_cluster_create(const GV_ClusterConfig *config) {
    GV_Cluster *cluster = calloc(1, sizeof(GV_Cluster));
    if (!cluster) return NULL;

    cluster->config = config ? *config : DEFAULT_CONFIG;

    /* Generate node ID if not provided */
    if (cluster->config.node_id) {
        cluster->local_node_id = strdup(cluster->config.node_id);
    } else {
        cluster->local_node_id = generate_node_id();
    }

    /* Create shard manager */
    cluster->shard_mgr = gv_shard_manager_create(NULL);
    if (!cluster->shard_mgr) {
        free(cluster->local_node_id);
        free(cluster);
        return NULL;
    }

    if (pthread_rwlock_init(&cluster->rwlock, NULL) != 0) {
        gv_shard_manager_destroy(cluster->shard_mgr);
        free(cluster->local_node_id);
        free(cluster);
        return NULL;
    }

    if (pthread_mutex_init(&cluster->state_mutex, NULL) != 0) {
        pthread_rwlock_destroy(&cluster->rwlock);
        gv_shard_manager_destroy(cluster->shard_mgr);
        free(cluster->local_node_id);
        free(cluster);
        return NULL;
    }

    if (pthread_cond_init(&cluster->ready_cond, NULL) != 0) {
        pthread_mutex_destroy(&cluster->state_mutex);
        pthread_rwlock_destroy(&cluster->rwlock);
        gv_shard_manager_destroy(cluster->shard_mgr);
        free(cluster->local_node_id);
        free(cluster);
        return NULL;
    }

    /* Add local node */
    NodeEntry *local = &cluster->nodes[0];
    local->node_id = strdup(cluster->local_node_id);
    local->address = cluster->config.listen_address ? strdup(cluster->config.listen_address) : NULL;
    local->role = cluster->config.role;
    local->state = GV_NODE_JOINING;
    local->shard_ids = NULL;
    local->shard_count = 0;
    local->last_heartbeat = (uint64_t)time(NULL);
    local->load = 0.0;
    cluster->node_count = 1;

    return cluster;
}

void gv_cluster_destroy(GV_Cluster *cluster) {
    if (!cluster) return;

    gv_cluster_stop(cluster);

    for (size_t i = 0; i < cluster->node_count; i++) {
        free(cluster->nodes[i].node_id);
        free(cluster->nodes[i].address);
        free(cluster->nodes[i].shard_ids);
    }

    pthread_cond_destroy(&cluster->ready_cond);
    pthread_mutex_destroy(&cluster->state_mutex);
    pthread_rwlock_destroy(&cluster->rwlock);

    gv_shard_manager_destroy(cluster->shard_mgr);
    free(cluster->local_node_id);
    free(cluster);
}

int gv_cluster_start(GV_Cluster *cluster) {
    if (!cluster) return -1;

    pthread_rwlock_wrlock(&cluster->rwlock);

    if (cluster->heartbeat_running) {
        pthread_rwlock_unlock(&cluster->rwlock);
        return -1;
    }

    cluster->stop_requested = 0;
    cluster->heartbeat_running = 1;

    /* Set local node to active */
    NodeEntry *local = find_node(cluster, cluster->local_node_id);
    if (local) {
        local->state = GV_NODE_ACTIVE;
    }

    pthread_rwlock_unlock(&cluster->rwlock);

    /* Start heartbeat thread */
    if (pthread_create(&cluster->heartbeat_thread, NULL, heartbeat_thread_func, cluster) != 0) {
        pthread_rwlock_wrlock(&cluster->rwlock);
        cluster->heartbeat_running = 0;
        pthread_rwlock_unlock(&cluster->rwlock);
        return -1;
    }

    /* TODO: Connect to seed nodes */
    /* TODO: Start RPC server */

    /* Mark cluster ready */
    pthread_mutex_lock(&cluster->state_mutex);
    cluster->is_ready = 1;
    pthread_cond_broadcast(&cluster->ready_cond);
    pthread_mutex_unlock(&cluster->state_mutex);

    return 0;
}

int gv_cluster_stop(GV_Cluster *cluster) {
    if (!cluster) return -1;

    pthread_rwlock_wrlock(&cluster->rwlock);

    if (!cluster->heartbeat_running) {
        pthread_rwlock_unlock(&cluster->rwlock);
        return 0;
    }

    cluster->stop_requested = 1;
    pthread_rwlock_unlock(&cluster->rwlock);

    pthread_join(cluster->heartbeat_thread, NULL);

    pthread_rwlock_wrlock(&cluster->rwlock);
    cluster->heartbeat_running = 0;

    /* Set local node to leaving */
    NodeEntry *local = find_node(cluster, cluster->local_node_id);
    if (local) {
        local->state = GV_NODE_LEAVING;
    }

    pthread_rwlock_unlock(&cluster->rwlock);

    return 0;
}

/* ============================================================================
 * Node Management
 * ============================================================================ */

static void copy_node_info(const NodeEntry *entry, GV_NodeInfo *info) {
    info->node_id = entry->node_id ? strdup(entry->node_id) : NULL;
    info->address = entry->address ? strdup(entry->address) : NULL;
    info->role = entry->role;
    info->state = entry->state;
    if (entry->shard_count > 0 && entry->shard_ids) {
        info->shard_ids = malloc(entry->shard_count * sizeof(uint32_t));
        memcpy(info->shard_ids, entry->shard_ids, entry->shard_count * sizeof(uint32_t));
    } else {
        info->shard_ids = NULL;
    }
    info->shard_count = entry->shard_count;
    info->last_heartbeat = entry->last_heartbeat;
    info->load = entry->load;
}

int gv_cluster_get_local_node(GV_Cluster *cluster, GV_NodeInfo *info) {
    if (!cluster || !info) return -1;
    return gv_cluster_get_node(cluster, cluster->local_node_id, info);
}

int gv_cluster_get_node(GV_Cluster *cluster, const char *node_id, GV_NodeInfo *info) {
    if (!cluster || !node_id || !info) return -1;

    pthread_rwlock_rdlock(&cluster->rwlock);

    NodeEntry *entry = find_node(cluster, node_id);
    if (!entry) {
        pthread_rwlock_unlock(&cluster->rwlock);
        return -1;
    }

    copy_node_info(entry, info);

    pthread_rwlock_unlock(&cluster->rwlock);
    return 0;
}

int gv_cluster_list_nodes(GV_Cluster *cluster, GV_NodeInfo **nodes, size_t *count) {
    if (!cluster || !nodes || !count) return -1;

    pthread_rwlock_rdlock(&cluster->rwlock);

    *count = cluster->node_count;
    if (*count == 0) {
        *nodes = NULL;
        pthread_rwlock_unlock(&cluster->rwlock);
        return 0;
    }

    *nodes = malloc(*count * sizeof(GV_NodeInfo));
    if (!*nodes) {
        pthread_rwlock_unlock(&cluster->rwlock);
        return -1;
    }

    for (size_t i = 0; i < *count; i++) {
        copy_node_info(&cluster->nodes[i], &(*nodes)[i]);
    }

    pthread_rwlock_unlock(&cluster->rwlock);
    return 0;
}

void gv_cluster_free_node_info(GV_NodeInfo *info) {
    if (!info) return;
    free(info->node_id);
    free(info->address);
    free(info->shard_ids);
    memset(info, 0, sizeof(*info));
}

void gv_cluster_free_node_list(GV_NodeInfo *nodes, size_t count) {
    if (!nodes) return;
    for (size_t i = 0; i < count; i++) {
        gv_cluster_free_node_info(&nodes[i]);
    }
    free(nodes);
}

/* ============================================================================
 * Cluster Operations
 * ============================================================================ */

int gv_cluster_get_stats(GV_Cluster *cluster, GV_ClusterStats *stats) {
    if (!cluster || !stats) return -1;

    pthread_rwlock_rdlock(&cluster->rwlock);

    memset(stats, 0, sizeof(*stats));
    stats->total_nodes = cluster->node_count;

    double total_load = 0.0;
    for (size_t i = 0; i < cluster->node_count; i++) {
        if (cluster->nodes[i].state == GV_NODE_ACTIVE) {
            stats->active_nodes++;
        }
        total_load += cluster->nodes[i].load;
    }

    stats->avg_load = cluster->node_count > 0 ? total_load / cluster->node_count : 0.0;

    /* Get shard stats */
    GV_ShardInfo *shards;
    size_t shard_count;
    if (gv_shard_list(cluster->shard_mgr, &shards, &shard_count) == 0) {
        stats->total_shards = shard_count;
        for (size_t i = 0; i < shard_count; i++) {
            stats->total_vectors += shards[i].vector_count;
        }
        gv_shard_free_list(shards, shard_count);
    }

    pthread_rwlock_unlock(&cluster->rwlock);
    return 0;
}

GV_ShardManager *gv_cluster_get_shard_manager(GV_Cluster *cluster) {
    if (!cluster) return NULL;
    return cluster->shard_mgr;
}

int gv_cluster_is_healthy(GV_Cluster *cluster) {
    if (!cluster) return -1;

    pthread_rwlock_rdlock(&cluster->rwlock);

    int healthy = 1;
    size_t active = 0;
    for (size_t i = 0; i < cluster->node_count; i++) {
        if (cluster->nodes[i].state == GV_NODE_ACTIVE) {
            active++;
        }
    }

    /* Healthy if at least half of nodes are active */
    if (cluster->node_count > 0 && active < (cluster->node_count + 1) / 2) {
        healthy = 0;
    }

    pthread_rwlock_unlock(&cluster->rwlock);
    return healthy;
}

int gv_cluster_wait_ready(GV_Cluster *cluster, uint32_t timeout_ms) {
    if (!cluster) return -1;

    pthread_mutex_lock(&cluster->state_mutex);

    if (cluster->is_ready) {
        pthread_mutex_unlock(&cluster->state_mutex);
        return 0;
    }

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += timeout_ms / 1000;
    ts.tv_nsec += (timeout_ms % 1000) * 1000000;
    if (ts.tv_nsec >= 1000000000) {
        ts.tv_sec++;
        ts.tv_nsec -= 1000000000;
    }

    while (!cluster->is_ready) {
        int rc = pthread_cond_timedwait(&cluster->ready_cond, &cluster->state_mutex, &ts);
        if (rc != 0) {
            pthread_mutex_unlock(&cluster->state_mutex);
            return -1;
        }
    }

    pthread_mutex_unlock(&cluster->state_mutex);
    return 0;
}
