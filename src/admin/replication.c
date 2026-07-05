/**
 * @file replication.c
 * @brief Replication and high availability implementation.
 *
 * Provides an in-process leader/follower coordinator: registered followers are
 * treated as connected and caught up to the leader WAL head until a wire
 * protocol is added. Use replication_leader_append_wal() after local WAL
 * writes to advance the logical replication position.
 */

#include "admin/replication.h"
#include "core/memory.h"
#include "admin/repl_transport.h"
#include "storage/database.h"
#include "storage/memory_layer.h"
#include "core/utils.h"
#include "core/sim_time.h"

#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include "core/compat.h"

#define MAX_REPLICAS 16

typedef struct {
    char *node_id;
    char *address;
    GV_ReplicationState state;
    uint64_t last_wal_position;
    uint64_t last_heartbeat;
    int connected;
} ReplicaEntry;

struct GV_ReplicationManager {
    GV_ReplicationConfig config;
    GV_Database *db;
    char *node_id;

    /* Role and election */
    GV_ReplicationRole role;
    uint64_t term;
    char *leader_id;
    char *voted_for;

    /* Replicas */
    ReplicaEntry replicas[MAX_REPLICAS];
    size_t replica_count;

    /* WAL positions */
    uint64_t wal_position;
    uint64_t commit_position;
    uint64_t bytes_replicated;

    /* Read replica load balancing */
    GV_ReadPolicy read_policy;
    uint64_t max_read_lag;
    GV_Database *follower_dbs[MAX_REPLICAS];
    GV_MemoryLayer *follower_memories[MAX_REPLICAS];
    size_t round_robin_next;

    /* Threads */
    pthread_t replication_thread;
    int running;
    int stop_requested;

    pthread_rwlock_t rwlock;
    pthread_mutex_t election_mutex;
    pthread_cond_t sync_cond;

    GV_ReplTransport *transport;
    int dst_simulation_mode;
};

static const GV_ReplicationConfig DEFAULT_CONFIG = {
    .node_id = NULL,
    .listen_address = "0.0.0.0:7001",
    .leader_address = NULL,
    .sync_interval_ms = 100,
    .election_timeout_ms = 3000,
    .heartbeat_interval_ms = 500,
    .max_lag_entries = 10000
};

void replication_config_init(GV_ReplicationConfig *config) {
    if (!config) return;
    *config = DEFAULT_CONFIG;
}

static char *generate_node_id(void) {
    char id[64];
    snprintf(id, sizeof(id), "repl-%lx-%d",
             (unsigned long)gv_time_now_sec(), (int)getpid());
    return gv_dup_cstr(id);
}

static ReplicaEntry *find_replica(GV_ReplicationManager *mgr, const char *node_id) {
    for (size_t i = 0; i < mgr->replica_count; i++) {
        if (strcmp(mgr->replicas[i].node_id, node_id) == 0) {
            return &mgr->replicas[i];
        }
    }
    return NULL;
}

/**
 * Embedded coordinator: registered followers are modelled as fully caught up to
 * the leader WAL head (no network transport). Keeps sync_commit, wait_sync,
 * read routing, and is_healthy consistent in-process.
 */
static void replication_embedded_followers_catch_up_locked(GV_ReplicationManager *mgr) {
    if (mgr->role != GV_REPL_LEADER) return;
    for (size_t i = 0; i < mgr->replica_count; i++) {
        if (!mgr->replicas[i].connected) continue;
        /* In-process followers with registered DBs catch up instantly. */
        if (mgr->follower_dbs[i] != NULL) {
            mgr->replicas[i].last_wal_position = mgr->wal_position;
            if (mgr->replicas[i].state == GV_REPL_SYNCING) {
                mgr->replicas[i].state = GV_REPL_STREAMING;
            }
        }
    }
}

static void *replication_thread_func(void *arg) {
    GV_ReplicationManager *mgr = (GV_ReplicationManager *)arg;

    while (!mgr->stop_requested) {
        gv_time_sleep_ms(mgr->config.sync_interval_ms);
        if (mgr->stop_requested) break;

        pthread_rwlock_wrlock(&mgr->rwlock);

        uint64_t now = gv_time_now_sec();

        if (mgr->role == GV_REPL_LEADER) {
            if (!mgr->dst_simulation_mode) {
                replication_embedded_followers_catch_up_locked(mgr);
            }
            for (size_t i = 0; i < mgr->replica_count; i++) {
                if (mgr->replicas[i].connected) {
                    /* Without a wire protocol, heartbeats only advance time;
                     * WAL catch-up for connected replicas is done above. */
                    mgr->replicas[i].last_heartbeat = now;
                }
            }
        } else if (mgr->role == GV_REPL_FOLLOWER) {
            uint64_t timeout_sec = mgr->config.election_timeout_ms / 1000;
            if (timeout_sec == 0) timeout_sec = 3;  /* Default 3 seconds */

            if (mgr->leader_id) {
                uint64_t leader_last_heartbeat = 0;
                for (size_t i = 0; i < mgr->replica_count; i++) {
                    if (mgr->replicas[i].node_id &&
                        strcmp(mgr->replicas[i].node_id, mgr->leader_id) == 0) {
                        leader_last_heartbeat = mgr->replicas[i].last_heartbeat;
                        break;
                    }
                }

                if (leader_last_heartbeat > 0 && now > leader_last_heartbeat) {
                    uint64_t elapsed = now - leader_last_heartbeat;
                    if (elapsed > timeout_sec) {
                        mgr->role = GV_REPL_CANDIDATE;
                        mgr->term++;
                        gv_free(mgr->leader_id);
                        mgr->leader_id = NULL;
                        gv_free(mgr->voted_for);
                        mgr->voted_for = mgr->node_id ? gv_dup_cstr(mgr->node_id) : NULL;
                    }
                }
            }
        } else if (mgr->role == GV_REPL_CANDIDATE) {
            /* Count in-process registered follower DBs as automatic votes: they
             * are co-located and can trivially accept this node as leader.
             * Remote-only replicas (follower_dbs[i] == NULL) cannot be asked
             * without a wire protocol, so they are not counted. */
            size_t votes = 1; /* self-vote */
            for (size_t i = 0; i < mgr->replica_count; i++) {
                if (mgr->follower_dbs[i] != NULL) {
                    votes++;
                }
            }
            size_t quorum = (mgr->replica_count + 1) / 2 + 1;
            if (votes >= quorum) {
                mgr->role = GV_REPL_LEADER;
                gv_free(mgr->leader_id);
                mgr->leader_id = mgr->node_id ? gv_dup_cstr(mgr->node_id) : NULL;
                replication_embedded_followers_catch_up_locked(mgr);
            }
        }

        pthread_rwlock_unlock(&mgr->rwlock);
    }

    return NULL;
}

GV_ReplicationManager *replication_create(GV_Database *db, const GV_ReplicationConfig *config) {
    if (!db) return NULL;

    GV_ReplicationManager *mgr = gv_calloc(1, sizeof(GV_ReplicationManager));
    if (!mgr) return NULL;

    mgr->config = config ? *config : DEFAULT_CONFIG;
    mgr->db = db;

    if (mgr->config.node_id) {
        mgr->node_id = gv_dup_cstr(mgr->config.node_id);
    } else {
        mgr->node_id = generate_node_id();
    }

    if (mgr->config.leader_address) {
        mgr->role = GV_REPL_FOLLOWER;
    } else {
        mgr->role = GV_REPL_LEADER;
        mgr->leader_id = gv_dup_cstr(mgr->node_id);
    }

    mgr->term = 1;

    if (pthread_rwlock_init(&mgr->rwlock, NULL) != 0) {
        gv_free(mgr->node_id);
        gv_free(mgr->leader_id);
        gv_free(mgr);
        return NULL;
    }

    if (pthread_mutex_init(&mgr->election_mutex, NULL) != 0) {
        pthread_rwlock_destroy(&mgr->rwlock);
        gv_free(mgr->node_id);
        gv_free(mgr->leader_id);
        gv_free(mgr);
        return NULL;
    }

    if (pthread_cond_init(&mgr->sync_cond, NULL) != 0) {
        pthread_mutex_destroy(&mgr->election_mutex);
        pthread_rwlock_destroy(&mgr->rwlock);
        gv_free(mgr->node_id);
        gv_free(mgr->leader_id);
        gv_free(mgr);
        return NULL;
    }

    return mgr;
}

void replication_destroy(GV_ReplicationManager *mgr) {
    if (!mgr) return;

    replication_stop(mgr);

    if (mgr->transport) {
        repl_transport_destroy(mgr->transport);
        mgr->transport = NULL;
    }

    for (size_t i = 0; i < mgr->replica_count; i++) {
        gv_free(mgr->replicas[i].node_id);
        gv_free(mgr->replicas[i].address);
    }

    pthread_cond_destroy(&mgr->sync_cond);
    pthread_mutex_destroy(&mgr->election_mutex);
    pthread_rwlock_destroy(&mgr->rwlock);

    gv_free(mgr->node_id);
    gv_free(mgr->leader_id);
    gv_free(mgr->voted_for);
    gv_free(mgr);
}

int replication_start(GV_ReplicationManager *mgr) {
    if (!mgr) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    if (mgr->running) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    mgr->stop_requested = 0;
    mgr->running = 1;

    if (!mgr->transport) {
        mgr->transport = repl_transport_create(mgr);
    }
    GV_ReplTransport *transport = mgr->transport;
    pthread_rwlock_unlock(&mgr->rwlock);

    if (transport) {
        repl_transport_start(transport);
    }

    if (pthread_create(&mgr->replication_thread, NULL, replication_thread_func, mgr) != 0) {
        pthread_rwlock_wrlock(&mgr->rwlock);
        mgr->running = 0;
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    return 0;
}

int replication_stop(GV_ReplicationManager *mgr) {
    if (!mgr) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    if (!mgr->running) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return 0;
    }

    mgr->stop_requested = 1;
    GV_ReplTransport *transport = mgr->transport;
    pthread_rwlock_unlock(&mgr->rwlock);

    if (transport) {
        repl_transport_stop(transport);
    }

    pthread_join(mgr->replication_thread, NULL);

    pthread_rwlock_wrlock(&mgr->rwlock);
    mgr->running = 0;
    pthread_rwlock_unlock(&mgr->rwlock);

    return 0;
}

GV_ReplicationRole replication_get_role(GV_ReplicationManager *mgr) {
    if (!mgr) return (GV_ReplicationRole)-1;

    pthread_rwlock_rdlock(&mgr->rwlock);
    GV_ReplicationRole role = mgr->role;
    pthread_rwlock_unlock(&mgr->rwlock);

    return role;
}

int replication_step_down(GV_ReplicationManager *mgr) {
    if (!mgr) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    if (mgr->role != GV_REPL_LEADER) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    mgr->role = GV_REPL_FOLLOWER;
    gv_free(mgr->leader_id);
    mgr->leader_id = NULL;

    pthread_rwlock_unlock(&mgr->rwlock);
    return 0;
}

int replication_request_leadership(GV_ReplicationManager *mgr) {
    if (!mgr) return -1;

    pthread_mutex_lock(&mgr->election_mutex);
    pthread_rwlock_wrlock(&mgr->rwlock);

    mgr->role = GV_REPL_CANDIDATE;
    mgr->term++;
    gv_free(mgr->voted_for);
    mgr->voted_for = gv_dup_cstr(mgr->node_id);

    size_t votes = 1;
    for (size_t i = 0; i < mgr->replica_count; i++) {
        if (mgr->follower_dbs[i] != NULL) {
            votes++;
        }
    }
    size_t quorum = (mgr->replica_count + 1) / 2 + 1;
    if (votes >= quorum) {
        mgr->role = GV_REPL_LEADER;
        gv_free(mgr->leader_id);
        mgr->leader_id = gv_dup_cstr(mgr->node_id);
        replication_embedded_followers_catch_up_locked(mgr);
    }

    GV_ReplicationRole result_role = mgr->role;
    pthread_rwlock_unlock(&mgr->rwlock);
    pthread_mutex_unlock(&mgr->election_mutex);

    return result_role == GV_REPL_LEADER ? 0 : -1;
}

int replication_add_follower(GV_ReplicationManager *mgr, const char *node_id,
                                 const char *address) {
    if (!mgr || !node_id || !address) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    if (mgr->replica_count >= MAX_REPLICAS) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    if (find_replica(mgr, node_id)) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    ReplicaEntry *entry = &mgr->replicas[mgr->replica_count];
    entry->node_id = gv_dup_cstr(node_id);
    entry->address = gv_dup_cstr(address);
    entry->state = GV_REPL_STREAMING;
    entry->last_wal_position = mgr->wal_position;
    entry->last_heartbeat = gv_time_now_sec();
    entry->connected = 1;

    mgr->follower_dbs[mgr->replica_count] = NULL;
    mgr->follower_memories[mgr->replica_count] = NULL;
    mgr->replica_count++;

    pthread_rwlock_unlock(&mgr->rwlock);
    return 0;
}

int replication_remove_follower(GV_ReplicationManager *mgr, const char *node_id) {
    if (!mgr || !node_id) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    for (size_t i = 0; i < mgr->replica_count; i++) {
        if (strcmp(mgr->replicas[i].node_id, node_id) == 0) {
            gv_free(mgr->replicas[i].node_id);
            gv_free(mgr->replicas[i].address);

            for (size_t j = i; j < mgr->replica_count - 1; j++) {
                mgr->replicas[j] = mgr->replicas[j + 1];
                mgr->follower_dbs[j] = mgr->follower_dbs[j + 1];
                mgr->follower_memories[j] = mgr->follower_memories[j + 1];
            }
            mgr->replica_count--;
            mgr->follower_dbs[mgr->replica_count] = NULL;
            mgr->follower_memories[mgr->replica_count] = NULL;

            pthread_rwlock_unlock(&mgr->rwlock);
            return 0;
        }
    }

    pthread_rwlock_unlock(&mgr->rwlock);
    return -1;
}

int replication_list_replicas(GV_ReplicationManager *mgr, GV_ReplicaInfo **replicas,
                                  size_t *count) {
    if (!mgr || !replicas || !count) return -1;

    pthread_rwlock_rdlock(&mgr->rwlock);

    *count = mgr->replica_count;
    if (*count == 0) {
        *replicas = NULL;
        pthread_rwlock_unlock(&mgr->rwlock);
        return 0;
    }

    *replicas = gv_alloc(*count * sizeof(GV_ReplicaInfo));
    if (!*replicas) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    for (size_t i = 0; i < *count; i++) {
        (*replicas)[i].node_id = gv_dup_cstr(mgr->replicas[i].node_id);
        (*replicas)[i].address = gv_dup_cstr(mgr->replicas[i].address);
        (*replicas)[i].role = GV_REPL_FOLLOWER;
        (*replicas)[i].state = mgr->replicas[i].state;
        (*replicas)[i].last_wal_position = mgr->replicas[i].last_wal_position;
        (*replicas)[i].lag_entries = mgr->wal_position - mgr->replicas[i].last_wal_position;
        (*replicas)[i].last_heartbeat = mgr->replicas[i].last_heartbeat;
    }

    pthread_rwlock_unlock(&mgr->rwlock);
    return 0;
}

void replication_free_replicas(GV_ReplicaInfo *replicas, size_t count) {
    if (!replicas) return;
    for (size_t i = 0; i < count; i++) {
        gv_free(replicas[i].node_id);
        gv_free(replicas[i].address);
    }
    gv_free(replicas);
}

int replication_sync_commit(GV_ReplicationManager *mgr, uint32_t timeout_ms) {
    if (!mgr) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    if (mgr->role != GV_REPL_LEADER) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    if (!mgr->dst_simulation_mode) {
        replication_embedded_followers_catch_up_locked(mgr);
    }

    if (mgr->replica_count == 0) {
        if (mgr->wal_position > mgr->commit_position) {
            mgr->commit_position = mgr->wal_position;
        }
        pthread_rwlock_unlock(&mgr->rwlock);
        return 0;
    }

    size_t required_acks = (mgr->replica_count / 2) + 1;
    if (required_acks > mgr->replica_count) {
        required_acks = mgr->replica_count;
    }

    uint64_t current_wal = mgr->wal_position;
    pthread_rwlock_unlock(&mgr->rwlock);

    uint64_t start_time = gv_time_now_ms();
    uint64_t deadline = start_time + timeout_ms;

    while (gv_time_now_ms() < deadline) {
        pthread_rwlock_rdlock(&mgr->rwlock);

        size_t acks = 0;
        for (size_t i = 0; i < mgr->replica_count; i++) {
            if (mgr->replicas[i].connected &&
                mgr->replicas[i].last_wal_position >= current_wal) {
                acks++;
            }
        }

        pthread_rwlock_unlock(&mgr->rwlock);

        if (acks >= required_acks) {
            pthread_rwlock_wrlock(&mgr->rwlock);
            if (current_wal > mgr->commit_position) {
                mgr->commit_position = current_wal;
            }
            pthread_rwlock_unlock(&mgr->rwlock);
            return 0;
        }

        gv_time_sleep_ms(10);
    }

    return -1;  /* Timeout */
}

int64_t replication_get_lag(GV_ReplicationManager *mgr) {
    if (!mgr) return -1;

    pthread_rwlock_rdlock(&mgr->rwlock);

    int64_t max_lag = 0;
    for (size_t i = 0; i < mgr->replica_count; i++) {
        int64_t lag = mgr->wal_position - mgr->replicas[i].last_wal_position;
        if (lag > max_lag) max_lag = lag;
    }

    pthread_rwlock_unlock(&mgr->rwlock);
    return max_lag;
}

int replication_wait_sync(GV_ReplicationManager *mgr, size_t max_lag, uint32_t timeout_ms) {
    if (!mgr) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);
    if (!mgr->dst_simulation_mode) {
        replication_embedded_followers_catch_up_locked(mgr);
    }
    pthread_rwlock_unlock(&mgr->rwlock);

    uint64_t start_time = gv_time_now_ms();
    uint64_t deadline = start_time + timeout_ms;

    while (gv_time_now_ms() < deadline) {
        pthread_rwlock_rdlock(&mgr->rwlock);

        int all_synced = 1;
        for (size_t i = 0; i < mgr->replica_count; i++) {
            if (!mgr->replicas[i].connected) continue;

            uint64_t lag = 0;
            if (mgr->wal_position > mgr->replicas[i].last_wal_position) {
                lag = mgr->wal_position - mgr->replicas[i].last_wal_position;
            }

            if (lag > max_lag) {
                all_synced = 0;
                break;
            }
        }

        pthread_rwlock_unlock(&mgr->rwlock);

        if (all_synced) {
            return 0;
        }

        usleep(10000);  /* 10ms */
    }

    return -1;  /* Timeout - not all replicas synced */
}

int replication_get_stats(GV_ReplicationManager *mgr, GV_ReplicationStats *stats) {
    if (!mgr || !stats) return -1;

    pthread_rwlock_rdlock(&mgr->rwlock);

    memset(stats, 0, sizeof(*stats));
    stats->role = mgr->role;
    stats->term = mgr->term;
    stats->leader_id = mgr->leader_id ? gv_dup_cstr(mgr->leader_id) : NULL;
    stats->follower_count = mgr->replica_count;
    stats->wal_position = mgr->wal_position;
    stats->commit_position = mgr->commit_position;
    stats->bytes_replicated = mgr->bytes_replicated;

    pthread_rwlock_unlock(&mgr->rwlock);
    return 0;
}

void replication_free_stats(GV_ReplicationStats *stats) {
    if (!stats) return;
    gv_free(stats->leader_id);
    memset(stats, 0, sizeof(*stats));
}

int replication_is_healthy(GV_ReplicationManager *mgr) {
    if (!mgr) return -1;

    pthread_rwlock_rdlock(&mgr->rwlock);

    int healthy = 1;

    if (mgr->role == GV_REPL_LEADER) {
        size_t connected = 0;
        for (size_t i = 0; i < mgr->replica_count; i++) {
            if (mgr->replicas[i].connected &&
                mgr->replicas[i].state == GV_REPL_STREAMING) {
                connected++;
            }
        }
        if (mgr->replica_count > 0 && connected < (mgr->replica_count + 1) / 2) {
            healthy = 0;
        }
    } else if (mgr->role == GV_REPL_FOLLOWER) {
        if (!mgr->leader_id) {
            healthy = 0;
        }
    }

    pthread_rwlock_unlock(&mgr->rwlock);
    return healthy;
}

int replication_set_read_policy(GV_ReplicationManager *mgr, GV_ReadPolicy policy) {
    if (!mgr) return -1;
    if (policy < GV_READ_LEADER_ONLY || policy > GV_READ_RANDOM) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);
    mgr->read_policy = policy;
    mgr->round_robin_next = 0;
    pthread_rwlock_unlock(&mgr->rwlock);
    return 0;
}

GV_ReadPolicy replication_get_read_policy(GV_ReplicationManager *mgr) {
    if (!mgr) return (GV_ReadPolicy)-1;

    pthread_rwlock_rdlock(&mgr->rwlock);
    GV_ReadPolicy policy = mgr->read_policy;
    pthread_rwlock_unlock(&mgr->rwlock);
    return policy;
}

int replication_set_max_read_lag(GV_ReplicationManager *mgr, uint64_t max_lag) {
    if (!mgr) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);
    mgr->max_read_lag = max_lag;
    pthread_rwlock_unlock(&mgr->rwlock);
    return 0;
}

int replication_register_follower_db(GV_ReplicationManager *mgr,
                                         const char *node_id, GV_Database *db) {
    if (!mgr || !node_id || !db) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    for (size_t i = 0; i < mgr->replica_count; i++) {
        if (strcmp(mgr->replicas[i].node_id, node_id) == 0) {
            mgr->follower_dbs[i] = db;
            pthread_rwlock_unlock(&mgr->rwlock);
            return 0;
        }
    }

    pthread_rwlock_unlock(&mgr->rwlock);
    return -1;  /* Replica not found */
}

int replication_register_follower_memory(GV_ReplicationManager *mgr,
                                             const char *node_id,
                                             GV_MemoryLayer *layer) {
    if (!mgr || !node_id || !layer) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    for (size_t i = 0; i < mgr->replica_count; i++) {
        if (strcmp(mgr->replicas[i].node_id, node_id) == 0) {
            mgr->follower_memories[i] = layer;
            pthread_rwlock_unlock(&mgr->rwlock);
            return 0;
        }
    }

    pthread_rwlock_unlock(&mgr->rwlock);
    return -1;
}

/* Check if a replica is eligible for reads (connected, streaming, within lag) */
static int replica_eligible_for_read(const GV_ReplicationManager *mgr, size_t idx) {
    if (!mgr->replicas[idx].connected) return 0;
    if (mgr->replicas[idx].state != GV_REPL_STREAMING) return 0;
    if (!mgr->follower_dbs[idx]) return 0;

    if (mgr->max_read_lag > 0) {
        uint64_t lag = 0;
        if (mgr->wal_position > mgr->replicas[idx].last_wal_position) {
            lag = mgr->wal_position - mgr->replicas[idx].last_wal_position;
        }
        if (lag > mgr->max_read_lag) return 0;
    }

    return 1;
}

typedef struct {
    GV_Database *db;
    GV_MemoryLayer *memory;
} GV_ReadRouteTarget;

static void replication_route_read_locked(GV_ReplicationManager *mgr,
                                              GV_ReadRouteTarget *target) {
    target->db = mgr->db;
    target->memory = NULL;

    if (mgr->read_policy == GV_READ_LEADER_ONLY) {
        return;
    }

    size_t eligible[MAX_REPLICAS];
    size_t eligible_count = 0;

    for (size_t i = 0; i < mgr->replica_count; i++) {
        if (replica_eligible_for_read(mgr, i)) {
            eligible[eligible_count++] = i;
        }
    }

    if (eligible_count == 0) {
        return;
    }

    size_t chosen = 0;

    switch (mgr->read_policy) {
        case GV_READ_ROUND_ROBIN:
            chosen = mgr->round_robin_next % eligible_count;
            mgr->round_robin_next++;
            break;

        case GV_READ_LEAST_LAG: {
            uint64_t min_lag = UINT64_MAX;
            for (size_t i = 0; i < eligible_count; i++) {
                size_t ri = eligible[i];
                uint64_t lag = 0;
                if (mgr->wal_position > mgr->replicas[ri].last_wal_position) {
                    lag = mgr->wal_position - mgr->replicas[ri].last_wal_position;
                }
                if (lag < min_lag) {
                    min_lag = lag;
                    chosen = i;
                }
            }
            break;
        }

        case GV_READ_RANDOM:
            chosen = (size_t)(gv_time_now_sec() * 2654435761UL) % eligible_count;
            break;

        default:
            return;
    }

    size_t slot = eligible[chosen];
    target->db = mgr->follower_dbs[slot];
    target->memory = mgr->follower_memories[slot];
}

GV_Database *replication_route_read(GV_ReplicationManager *mgr) {
    if (!mgr) return NULL;

    GV_ReadRouteTarget target;
    pthread_rwlock_wrlock(&mgr->rwlock);
    replication_route_read_locked(mgr, &target);
    pthread_rwlock_unlock(&mgr->rwlock);
    return target.db;
}

GV_MemoryLayer *replication_route_read_memory(GV_ReplicationManager *mgr) {
    if (!mgr) return NULL;

    GV_ReadRouteTarget target;
    pthread_rwlock_wrlock(&mgr->rwlock);
    replication_route_read_locked(mgr, &target);
    pthread_rwlock_unlock(&mgr->rwlock);
    return target.memory;
}

int replication_leader_append_wal(GV_ReplicationManager *mgr,
                                     uint64_t entry_delta,
                                     uint64_t byte_delta) {
    if (!mgr) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    if (mgr->role != GV_REPL_LEADER) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    mgr->wal_position += entry_delta;
    mgr->bytes_replicated += byte_delta;
    if (!mgr->dst_simulation_mode) {
        mgr->commit_position = mgr->wal_position;
        replication_embedded_followers_catch_up_locked(mgr);
    }

    uint64_t entry_index = mgr->wal_position - entry_delta;
    GV_ReplTransport *transport = mgr->transport;
    GV_Database *db = mgr->db;

    pthread_rwlock_unlock(&mgr->rwlock);

    if (transport && db && entry_delta > 0) {
        for (uint64_t i = 0; i < entry_delta; i++) {
            repl_transport_broadcast_entry(transport, db, entry_index + i);
        }
    }
    return 0;
}

GV_Database *replication_get_db(GV_ReplicationManager *mgr) {
    return mgr ? mgr->db : NULL;
}

const char *replication_get_node_id(const GV_ReplicationManager *mgr) {
    return (mgr && mgr->node_id) ? mgr->node_id : NULL;
}

const GV_ReplicationConfig *replication_get_config(const GV_ReplicationManager *mgr) {
    return mgr ? &mgr->config : NULL;
}

int replication_replica_handshake(GV_ReplicationManager *mgr, const char *node_id,
                                  uint64_t *catchup_from_out) {
    if (!mgr || !node_id) return -1;
    pthread_rwlock_wrlock(&mgr->rwlock);
    uint64_t catchup = 0;
    for (size_t i = 0; i < mgr->replica_count; i++) {
        if (mgr->replicas[i].node_id && strcmp(mgr->replicas[i].node_id, node_id) == 0) {
            mgr->replicas[i].connected = 1;
            mgr->replicas[i].state = GV_REPL_SYNCING;
            mgr->replicas[i].last_heartbeat = gv_time_now_sec();
            catchup = mgr->replicas[i].last_wal_position;
            break;
        }
    }
    pthread_rwlock_unlock(&mgr->rwlock);
    if (catchup_from_out) *catchup_from_out = catchup;
    return 0;
}

int replication_replica_ack(GV_ReplicationManager *mgr, const char *node_id,
                            uint64_t entry_index) {
    if (!mgr || !node_id) return -1;
    pthread_rwlock_wrlock(&mgr->rwlock);
    for (size_t i = 0; i < mgr->replica_count; i++) {
        if (mgr->replicas[i].node_id && strcmp(mgr->replicas[i].node_id, node_id) == 0) {
            if (entry_index + 1 > mgr->replicas[i].last_wal_position) {
                mgr->replicas[i].last_wal_position = entry_index + 1;
            }
            mgr->replicas[i].last_heartbeat = (uint64_t)time(NULL);
            mgr->replicas[i].connected = 1;
            mgr->replicas[i].state = GV_REPL_STREAMING;
            break;
        }
    }
    pthread_rwlock_unlock(&mgr->rwlock);
    return 0;
}

int replication_follower_apply_entry(GV_ReplicationManager *mgr, uint64_t entry_index,
                                     const uint8_t *record, size_t len) {
    if (!mgr || !mgr->db || !record || len == 0) return -1;
    if (db_apply_wal_record(mgr->db, record, len) != 0) return -1;
    pthread_rwlock_wrlock(&mgr->rwlock);
    if (entry_index + 1 > mgr->wal_position) {
        mgr->wal_position = entry_index + 1;
    }
    mgr->commit_position = mgr->wal_position;
    pthread_rwlock_unlock(&mgr->rwlock);
    return 0;
}

void replication_get_positions(const GV_ReplicationManager *mgr,
                               uint64_t *wal_position, uint64_t *commit_position) {
    if (!mgr) return;
    pthread_rwlock_rdlock((pthread_rwlock_t *)&mgr->rwlock);
    if (wal_position) *wal_position = mgr->wal_position;
    if (commit_position) *commit_position = mgr->commit_position;
    pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
}

void replication_note_leader_heartbeat(GV_ReplicationManager *mgr) {
    if (!mgr) return;
    pthread_rwlock_wrlock(&mgr->rwlock);
    mgr->role = GV_REPL_FOLLOWER;
    pthread_rwlock_unlock(&mgr->rwlock);
}

GV_ReplicationRole replication_get_role_for_transport(const GV_ReplicationManager *mgr) {
    if (!mgr) return (GV_ReplicationRole)-1;
    pthread_rwlock_rdlock((pthread_rwlock_t *)&mgr->rwlock);
    GV_ReplicationRole role = mgr->role;
    pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
    return role;
}

int replication_set_dst_simulation_mode(GV_ReplicationManager *mgr, int enabled) {
    if (!mgr) return -1;
    pthread_rwlock_wrlock(&mgr->rwlock);
    mgr->dst_simulation_mode = enabled ? 1 : 0;
    pthread_rwlock_unlock(&mgr->rwlock);
    return 0;
}

int replication_get_dst_simulation_mode(const GV_ReplicationManager *mgr) {
    if (!mgr) return 0;
    pthread_rwlock_rdlock((pthread_rwlock_t *)&mgr->rwlock);
    int mode = mgr->dst_simulation_mode;
    pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
    return mode;
}

GV_ReplTransport *replication_get_transport(GV_ReplicationManager *mgr) {
    return mgr ? mgr->transport : NULL;
}
