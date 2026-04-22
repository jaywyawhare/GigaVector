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
#include "storage/database.h"
#include "core/utils.h"

#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

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
    size_t round_robin_next;

    /* Threads */
    pthread_t replication_thread;
    int running;
    int stop_requested;

    pthread_rwlock_t rwlock;
    pthread_mutex_t election_mutex;
    pthread_cond_t sync_cond;
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
             (unsigned long)time(NULL), (int)getpid());
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
        mgr->replicas[i].last_wal_position = mgr->wal_position;
        if (mgr->replicas[i].state == GV_REPL_SYNCING) {
            mgr->replicas[i].state = GV_REPL_STREAMING;
        }
    }
}

static void *replication_thread_func(void *arg) {
    GV_ReplicationManager *mgr = (GV_ReplicationManager *)arg;

    while (!mgr->stop_requested) {
        usleep(mgr->config.sync_interval_ms * 1000);
        if (mgr->stop_requested) break;

        pthread_rwlock_wrlock(&mgr->rwlock);

        uint64_t now = (uint64_t)time(NULL);

        if (mgr->role == GV_REPL_LEADER) {
            replication_embedded_followers_catch_up_locked(mgr);
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
                        free(mgr->leader_id);
                        mgr->leader_id = NULL;
                        free(mgr->voted_for);
                        mgr->voted_for = mgr->node_id ? gv_dup_cstr(mgr->node_id) : NULL;

                        /* Note: In a real implementation, this would:
                         * 1. Request votes from all known replicas
                         * 2. Wait for majority response
                         * 3. If majority votes received, become leader
                         * 4. If another leader is discovered, become follower
                         * 5. If election times out, restart election with new term */
                    }
                }
            }
        }

        pthread_rwlock_unlock(&mgr->rwlock);
    }

    return NULL;
}

GV_ReplicationManager *replication_create(GV_Database *db, const GV_ReplicationConfig *config) {
    if (!db) return NULL;

    GV_ReplicationManager *mgr = calloc(1, sizeof(GV_ReplicationManager));
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
        free(mgr->node_id);
        free(mgr->leader_id);
        free(mgr);
        return NULL;
    }

    if (pthread_mutex_init(&mgr->election_mutex, NULL) != 0) {
        pthread_rwlock_destroy(&mgr->rwlock);
        free(mgr->node_id);
        free(mgr->leader_id);
        free(mgr);
        return NULL;
    }

    if (pthread_cond_init(&mgr->sync_cond, NULL) != 0) {
        pthread_mutex_destroy(&mgr->election_mutex);
        pthread_rwlock_destroy(&mgr->rwlock);
        free(mgr->node_id);
        free(mgr->leader_id);
        free(mgr);
        return NULL;
    }

    return mgr;
}

void replication_destroy(GV_ReplicationManager *mgr) {
    if (!mgr) return;

    replication_stop(mgr);

    for (size_t i = 0; i < mgr->replica_count; i++) {
        free(mgr->replicas[i].node_id);
        free(mgr->replicas[i].address);
    }

    pthread_cond_destroy(&mgr->sync_cond);
    pthread_mutex_destroy(&mgr->election_mutex);
    pthread_rwlock_destroy(&mgr->rwlock);

    free(mgr->node_id);
    free(mgr->leader_id);
    free(mgr->voted_for);
    free(mgr);
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

    pthread_rwlock_unlock(&mgr->rwlock);

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
    pthread_rwlock_unlock(&mgr->rwlock);

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
    free(mgr->leader_id);
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
    free(mgr->voted_for);
    mgr->voted_for = gv_dup_cstr(mgr->node_id);

    /* In a real implementation, we would:
     * 1. Send RequestVote to all replicas
     * 2. Wait for majority
     * 3. Become leader if we win
     */

    if (mgr->replica_count == 0) {
        mgr->role = GV_REPL_LEADER;
        free(mgr->leader_id);
        mgr->leader_id = gv_dup_cstr(mgr->node_id);
    }

    pthread_rwlock_unlock(&mgr->rwlock);
    pthread_mutex_unlock(&mgr->election_mutex);

    return mgr->role == GV_REPL_LEADER ? 0 : -1;
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
    entry->last_heartbeat = (uint64_t)time(NULL);
    entry->connected = 1;

    mgr->follower_dbs[mgr->replica_count] = NULL;
    mgr->replica_count++;

    pthread_rwlock_unlock(&mgr->rwlock);
    return 0;
}

int replication_remove_follower(GV_ReplicationManager *mgr, const char *node_id) {
    if (!mgr || !node_id) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    for (size_t i = 0; i < mgr->replica_count; i++) {
        if (strcmp(mgr->replicas[i].node_id, node_id) == 0) {
            free(mgr->replicas[i].node_id);
            free(mgr->replicas[i].address);

            for (size_t j = i; j < mgr->replica_count - 1; j++) {
                mgr->replicas[j] = mgr->replicas[j + 1];
                mgr->follower_dbs[j] = mgr->follower_dbs[j + 1];
            }
            mgr->replica_count--;
            mgr->follower_dbs[mgr->replica_count] = NULL;

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

    *replicas = malloc(*count * sizeof(GV_ReplicaInfo));
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
        free(replicas[i].node_id);
        free(replicas[i].address);
    }
    free(replicas);
}

int replication_sync_commit(GV_ReplicationManager *mgr, uint32_t timeout_ms) {
    if (!mgr) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    if (mgr->role != GV_REPL_LEADER) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    replication_embedded_followers_catch_up_locked(mgr);

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

    uint64_t start_time = (uint64_t)time(NULL) * 1000;
    uint64_t deadline = start_time + timeout_ms;

    while ((uint64_t)time(NULL) * 1000 < deadline) {
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

        usleep(10000);  /* 10ms */
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
    replication_embedded_followers_catch_up_locked(mgr);
    pthread_rwlock_unlock(&mgr->rwlock);

    uint64_t start_time = (uint64_t)time(NULL) * 1000;
    uint64_t deadline = start_time + timeout_ms;

    while ((uint64_t)time(NULL) * 1000 < deadline) {
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
    free(stats->leader_id);
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

GV_Database *replication_route_read(GV_ReplicationManager *mgr) {
    if (!mgr) return NULL;

    pthread_rwlock_wrlock(&mgr->rwlock);

    if (mgr->read_policy == GV_READ_LEADER_ONLY) {
        GV_Database *db = mgr->db;
        pthread_rwlock_unlock(&mgr->rwlock);
        return db;
    }

    size_t eligible[MAX_REPLICAS];
    size_t eligible_count = 0;

    for (size_t i = 0; i < mgr->replica_count; i++) {
        if (replica_eligible_for_read(mgr, i)) {
            eligible[eligible_count++] = i;
        }
    }

    if (eligible_count == 0) {
        GV_Database *db = mgr->db;
        pthread_rwlock_unlock(&mgr->rwlock);
        return db;
    }

    GV_Database *result = NULL;

    switch (mgr->read_policy) {
        case GV_READ_ROUND_ROBIN: {
            size_t idx = mgr->round_robin_next % eligible_count;
            result = mgr->follower_dbs[eligible[idx]];
            mgr->round_robin_next++;
            break;
        }

        case GV_READ_LEAST_LAG: {
            uint64_t min_lag = UINT64_MAX;
            size_t best = 0;
            for (size_t i = 0; i < eligible_count; i++) {
                size_t ri = eligible[i];
                uint64_t lag = 0;
                if (mgr->wal_position > mgr->replicas[ri].last_wal_position) {
                    lag = mgr->wal_position - mgr->replicas[ri].last_wal_position;
                }
                if (lag < min_lag) {
                    min_lag = lag;
                    best = i;
                }
            }
            result = mgr->follower_dbs[eligible[best]];
            break;
        }

        case GV_READ_RANDOM: {
            size_t idx = (size_t)(time(NULL) * 2654435761UL) % eligible_count;
            result = mgr->follower_dbs[eligible[idx]];
            break;
        }

        default:
            result = mgr->db;
            break;
    }

    pthread_rwlock_unlock(&mgr->rwlock);
    return result;
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
    replication_embedded_followers_catch_up_locked(mgr);

    pthread_rwlock_unlock(&mgr->rwlock);
    return 0;
}
