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
#include "admin/raft.h"
#include "core/memory.h"
#include "admin/repl_transport.h"
#include "storage/database.h"
#include "storage/memory_layer.h"
#include "core/utils.h"
#include "core/sim_time.h"

#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
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
    uint64_t last_leader_contact;   /* last time a leader heartbeat/WAL was received (sec) */

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

    /*
     * Per-follower in-flight read pin count. Incremented under rwlock by
     * replication_route_read_locked() just before a follower's GV_Database* is
     * handed out; decremented by replication_release_read() when the caller is
     * done. replication_remove_follower() blocks (on pin_cond) until the target
     * slot's pin reaches 0 before freeing/reusing it, preventing use-after-free.
     */
    int follower_pin[MAX_REPLICAS];

    /* Threads */
    pthread_t replication_thread;
    int running;
    _Atomic int stop_requested;

    pthread_rwlock_t rwlock;
    pthread_mutex_t election_mutex;
    pthread_cond_t sync_cond;

    /* Guards follower_pin[] transitions to zero and wakes remove_follower(). */
    pthread_mutex_t pin_mutex;
    pthread_cond_t pin_cond;

    GV_ReplTransport *transport;
    int dst_simulation_mode;

    /* Optional Raft consensus driver (opt-in via replication_enable_raft). When
     * set, leadership/term follow the raft core rather than the ad-hoc election.
     * Messages between peers are delivered through per-manager inboxes drained on
     * replication_raft_tick, so a send never re-enters a peer's raft under lock. */
    GV_Raft *raft;
    int raft_self_id;
    GV_ReplicationManager *raft_peers[MAX_REPLICAS];
    int raft_peer_ids[MAX_REPLICAS];
    size_t raft_peer_count;
    struct RaftInboxMsg *raft_inbox_head;
    struct RaftInboxMsg *raft_inbox_tail;
    pthread_mutex_t raft_mutex;   /* guards the raft core */
    pthread_mutex_t raft_inbox_mutex;  /* leaf lock guarding the inbox queue only */
};

/* A raft message queued for in-process delivery to a peer manager. */
typedef struct RaftInboxMsg {
    int from;
    GV_RaftMsg msg;
    GV_RaftEntry *entries;   /* owned deep copy for AppendEntries */
    struct RaftInboxMsg *next;
} RaftInboxMsg;

static void repl_raft_free_inbox_msg(RaftInboxMsg *qm);

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

/*
 * Raft election restriction (safety §5.4.1): a voter grants its vote to a
 * candidate only if the candidate's log is at least as up-to-date as the
 * voter's own. Here the candidate's position is mgr->wal_position and each
 * in-process follower's replicated position is replicas[i].last_wal_position;
 * a follower whose log is ahead of the candidate withholds its vote, which
 * prevents electing a leader that would truncate already-replicated entries.
 * Remote-only replicas (no registered DB) cannot be polled in-process and are
 * not counted. Returns 1 if a quorum of the cluster has granted the vote.
 * Caller must hold mgr->rwlock.
 */
static int replication_candidate_has_quorum_locked(const GV_ReplicationManager *mgr) {
    /* A node configured as a remote follower (leader_address set) with no
     * registered in-process voting followers cannot solicit real votes over
     * this ad-hoc path - the "quorum" would be a lone self-vote, so it would
     * promote itself to leader while the real leader is still up, producing
     * split-brain and divergent WALs. Refuse: safe automatic failover for a
     * networked topology requires the real consensus path
     * (replication_enable_raft), which solicits and persists real votes. A
     * standalone node (no leader_address) and an in-process leader with
     * registered follower DBs (replica_count > 0) are unaffected. */
    if (mgr->config.leader_address != NULL && mgr->replica_count == 0 && mgr->raft == NULL) {
        return 0;
    }
    size_t votes = 1; /* self-vote */
    for (size_t i = 0; i < mgr->replica_count; i++) {
        if (mgr->follower_dbs[i] == NULL) continue;
        if (mgr->wal_position >= mgr->replicas[i].last_wal_position) {
            votes++; /* follower's log is not ahead - it grants the vote */
        }
    }
    size_t quorum = (mgr->replica_count + 1) / 2 + 1;
    return votes >= quorum;
}

static void *replication_thread_func(void *arg) {
    GV_ReplicationManager *mgr = (GV_ReplicationManager *)arg;

    while (!atomic_load(&mgr->stop_requested)) {
        gv_time_sleep_ms(mgr->config.sync_interval_ms);
        if (atomic_load(&mgr->stop_requested)) break;

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
        } else if (!mgr->raft && mgr->role == GV_REPL_FOLLOWER) {
            /* Ad-hoc election is the default only when Raft is not driving this
             * manager. With Raft enabled, role/term transitions come solely from
             * replication_raft_tick(); running this timeout here would fight it. */
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
            } else if (mgr->config.leader_address && mgr->last_leader_contact > 0) {
                /* TCP follower (connected to a leader_address, no in-process leader
                 * replica): detect leader failure from the last heartbeat/WAL frame
                 * received over the wire, then stand for election. */
                if (now > mgr->last_leader_contact &&
                    (now - mgr->last_leader_contact) > timeout_sec) {
                    mgr->role = GV_REPL_CANDIDATE;
                    mgr->term++;
                    gv_free(mgr->voted_for);
                    mgr->voted_for = mgr->node_id ? gv_dup_cstr(mgr->node_id) : NULL;
                }
            }
        } else if (!mgr->raft && mgr->role == GV_REPL_CANDIDATE) {
            /* Tally votes under the Raft election restriction (see helper). */
            if (replication_candidate_has_quorum_locked(mgr)) {
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

    if (pthread_mutex_init(&mgr->pin_mutex, NULL) != 0) {
        pthread_cond_destroy(&mgr->sync_cond);
        pthread_mutex_destroy(&mgr->election_mutex);
        pthread_rwlock_destroy(&mgr->rwlock);
        gv_free(mgr->node_id);
        gv_free(mgr->leader_id);
        gv_free(mgr);
        return NULL;
    }

    if (pthread_cond_init(&mgr->pin_cond, NULL) != 0) {
        pthread_mutex_destroy(&mgr->pin_mutex);
        pthread_cond_destroy(&mgr->sync_cond);
        pthread_mutex_destroy(&mgr->election_mutex);
        pthread_rwlock_destroy(&mgr->rwlock);
        gv_free(mgr->node_id);
        gv_free(mgr->leader_id);
        gv_free(mgr);
        return NULL;
    }

    pthread_mutex_init(&mgr->raft_mutex, NULL);
    pthread_mutex_init(&mgr->raft_inbox_mutex, NULL);

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

    /* Tear down the raft driver and drain any undelivered inbox messages. */
    if (mgr->raft) { raft_destroy(mgr->raft); mgr->raft = NULL; }
    {
        RaftInboxMsg *p = mgr->raft_inbox_head;
        while (p) { RaftInboxMsg *n = p->next; repl_raft_free_inbox_msg(p); p = n; }
        mgr->raft_inbox_head = mgr->raft_inbox_tail = NULL;
    }
    pthread_mutex_destroy(&mgr->raft_mutex);
    pthread_mutex_destroy(&mgr->raft_inbox_mutex);

    pthread_cond_destroy(&mgr->pin_cond);
    pthread_mutex_destroy(&mgr->pin_mutex);
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

    atomic_store(&mgr->stop_requested, 0);

    if (!mgr->transport) {
        mgr->transport = repl_transport_create(mgr);
    }

    /*
     * Create the worker thread and publish `running` while still holding the
     * wrlock. This guarantees any thread that observes running==1 also
     * observes a fully-initialized replication_thread handle, so a concurrent
     * replication_stop() can never pthread_join() a garbage handle. On create
     * failure we leave running==0 and bail out.
     */
    if (pthread_create(&mgr->replication_thread, NULL, replication_thread_func, mgr) != 0) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }
    mgr->running = 1;

    GV_ReplTransport *transport = mgr->transport;
    pthread_rwlock_unlock(&mgr->rwlock);

    if (transport) {
        repl_transport_start(transport);
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

    atomic_store(&mgr->stop_requested, 1);
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

    if (replication_candidate_has_quorum_locked(mgr)) {
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
    mgr->follower_pin[mgr->replica_count] = 0;
    mgr->replica_count++;

    pthread_rwlock_unlock(&mgr->rwlock);
    return 0;
}

int replication_remove_follower(GV_ReplicationManager *mgr, const char *node_id) {
    if (!mgr || !node_id) return -1;

    /*
     * Removal must not free/reuse a slot while a route_read() caller still holds
     * a borrowed follower handle. We do this in two phases so we never block on
     * the pin-drain condvar while holding the rwlock (which would deadlock the
     * readers that need the rwlock to route/finish):
     *
     *   Phase 1 (rwlock): locate the slot and mark it disconnected so
     *       replica_eligible_for_read() rejects it -> no NEW pins can be taken.
     *   Phase 2 (pin_mutex): wait until the slot's in-flight pin count drains to
     *       0, signalled by replication_release_read(). Then re-take the rwlock
     *       to compact the arrays.
     *
     * We re-locate the node by id after the wait because the arrays could have
     * been compacted by a concurrent removal in the interim.
     */
    for (;;) {
        pthread_rwlock_wrlock(&mgr->rwlock);

        size_t idx = mgr->replica_count;
        for (size_t i = 0; i < mgr->replica_count; i++) {
            if (mgr->replicas[i].node_id &&
                strcmp(mgr->replicas[i].node_id, node_id) == 0) {
                idx = i;
                break;
            }
        }

        if (idx == mgr->replica_count) {
            pthread_rwlock_unlock(&mgr->rwlock);
            return -1;
        }

        /* Prevent any new pins from being taken on this slot. */
        mgr->replicas[idx].connected = 0;

        if (mgr->follower_pin[idx] > 0) {
            /*
             * In-flight readers exist. Drop the rwlock (so release_read can run)
             * and wait for the drain. release_read signals pin_cond under
             * pin_mutex; we take pin_mutex, re-check under it, then wait.
             */
            pthread_rwlock_unlock(&mgr->rwlock);

            pthread_mutex_lock(&mgr->pin_mutex);
            /*
             * Re-take the rwlock briefly to read the current pin count for this
             * node under the pin_mutex, so we don't miss a wakeup. Order is
             * always pin_mutex -> rwlock here; release_read never holds the
             * rwlock while taking pin_mutex, so there is no lock-ordering cycle.
             */
            int draining = 1;
            while (draining) {
                pthread_rwlock_rdlock(&mgr->rwlock);
                size_t cur = mgr->replica_count;
                for (size_t i = 0; i < mgr->replica_count; i++) {
                    if (mgr->replicas[i].node_id &&
                        strcmp(mgr->replicas[i].node_id, node_id) == 0) {
                        cur = i;
                        break;
                    }
                }
                int pinned = (cur < mgr->replica_count) ? mgr->follower_pin[cur] : 0;
                pthread_rwlock_unlock(&mgr->rwlock);

                if (pinned <= 0) {
                    draining = 0;
                } else {
                    pthread_cond_wait(&mgr->pin_cond, &mgr->pin_mutex);
                }
            }
            pthread_mutex_unlock(&mgr->pin_mutex);

            /* Pins drained; loop to re-acquire the rwlock and re-locate/free. */
            continue;
        }

        /* Pin count is 0 and no new pins possible: safe to free and compact. */
        gv_free(mgr->replicas[idx].node_id);
        gv_free(mgr->replicas[idx].address);

        for (size_t j = idx; j < mgr->replica_count - 1; j++) {
            mgr->replicas[j] = mgr->replicas[j + 1];
            mgr->follower_dbs[j] = mgr->follower_dbs[j + 1];
            mgr->follower_memories[j] = mgr->follower_memories[j + 1];
            mgr->follower_pin[j] = mgr->follower_pin[j + 1];
        }
        mgr->replica_count--;
        mgr->follower_dbs[mgr->replica_count] = NULL;
        mgr->follower_memories[mgr->replica_count] = NULL;
        mgr->follower_pin[mgr->replica_count] = 0;

        pthread_rwlock_unlock(&mgr->rwlock);
        return 0;
    }
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

    /* Cluster-majority durability. Cluster size is replica_count + 1 (the leader
     * already holds the write), so the follower acks needed are majority(N+1) - 1
     * = (replica_count + 1) / 2 - not a majority of followers alone (which would
     * require ALL of 2 followers and hang if one is down). */
    size_t required_acks = (mgr->replica_count + 1) / 2;
    if (required_acks == 0) required_acks = 1;
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

        gv_time_sleep_ms(10);   /* sim-time aware (matches the deadline clock) */
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
    int pinned_slot;   /* Follower slot pinned for this route, or -1 (leader/none). */
} GV_ReadRouteTarget;

/*
 * Select a target under mgr->rwlock (write lock). If a follower slot is chosen,
 * its in-flight pin count is incremented before returning; the caller
 * (replication_route_read) is then responsible for pairing that with
 * replication_release_read() once done with the returned handle. When the leader
 * is selected, no pin is taken (pinned_slot == -1).
 */
static void replication_route_read_locked(GV_ReplicationManager *mgr,
                                              GV_ReadRouteTarget *target) {
    target->db = mgr->db;
    target->memory = NULL;
    target->pinned_slot = -1;

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
    /* Pin the follower so remove_follower() cannot free it out from under the
     * caller until replication_release_read() decrements this. */
    mgr->follower_pin[slot]++;
    target->pinned_slot = (int)slot;
}

/*
 * LIFETIME CONTRACT:
 *
 * When a follower slot is selected, this function PINS it (follower_pin[slot]++
 * under mgr->rwlock) before returning, and the pin is deliberately held past the
 * unlock so the follower stays alive for the caller. The caller MUST balance a
 * non-NULL follower return with replication_release_read(mgr, returned_db) when
 * done; replication_remove_follower() blocks on pin_cond until all pins drain,
 * so it cannot free a follower with a read in flight. Leader targets are not
 * pinned and need no release. (replication_remove_follower() does not itself
 * db_close the follower - the ultimate free is owned externally - but it will
 * not return while pinned, so the drain ordering is enforced here, not by the
 * caller.)
 */
GV_Database *replication_route_read(GV_ReplicationManager *mgr) {
    if (!mgr) return NULL;

    GV_ReadRouteTarget target;
    pthread_rwlock_wrlock(&mgr->rwlock);
    replication_route_read_locked(mgr, &target);
    pthread_rwlock_unlock(&mgr->rwlock);
    /*
     * If a follower slot was pinned, the pin is intentionally held past the
     * unlock: it keeps that follower alive for the caller, who MUST call
     * replication_release_read(mgr, returned_db) when done. Leader targets take
     * no pin and require no release.
     */
    return target.db;
}

int replication_release_read(GV_ReplicationManager *mgr, GV_Database *db) {
    if (!mgr || !db) return -1;

    /* The leader DB is never pinned by route_read; releasing it is a no-op. */
    if (db == mgr->db) return 0;

    pthread_rwlock_wrlock(&mgr->rwlock);

    int slot = -1;
    for (size_t i = 0; i < mgr->replica_count; i++) {
        if (mgr->follower_dbs[i] == db) {
            slot = (int)i;
            break;
        }
    }

    if (slot < 0 || mgr->follower_pin[slot] <= 0) {
        /* Unknown handle or nothing pinned: nothing to release. */
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    int now = --mgr->follower_pin[slot];
    pthread_rwlock_unlock(&mgr->rwlock);

    /*
     * If this was the last in-flight pin, wake any remove_follower() waiting for
     * the slot to drain. We take pin_mutex (never while holding the rwlock) so a
     * concurrent waiter that is between its pin re-check and pthread_cond_wait()
     * cannot miss this signal. Signalling on every zero-transition is sufficient.
     */
    if (now == 0) {
        pthread_mutex_lock(&mgr->pin_mutex);
        pthread_cond_broadcast(&mgr->pin_cond);
        pthread_mutex_unlock(&mgr->pin_mutex);
    }
    return 0;
}

GV_MemoryLayer *replication_route_read_memory(GV_ReplicationManager *mgr) {
    if (!mgr) return NULL;

    GV_ReadRouteTarget target;
    pthread_rwlock_wrlock(&mgr->rwlock);
    replication_route_read_locked(mgr, &target);
    /*
     * This entry point returns a borrowed memory-layer pointer and has no
     * paired release in its contract, so we must not leak the pin taken by
     * route_read_locked(). Drop it here while still holding the rwlock: no
     * remove_follower() drain can be in progress observing this slot (removal
     * marks the slot disconnected under the rwlock, which makes it ineligible
     * for selection above), so undoing the increment under the same lock leaves
     * the pin count exactly as it was. (The pre-existing lifetime caveat for the
     * returned memory pointer is unchanged by this fix.)
     */
    if (target.pinned_slot >= 0) {
        mgr->follower_pin[target.pinned_slot]--;
    }
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
            mgr->replicas[i].last_heartbeat = gv_time_now_sec();
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

/* ---- Raft consensus integration (opt-in) ---------------------------------- */

static GV_ReplicationManager *repl_raft_find_peer(GV_ReplicationManager *mgr, int id) {
    for (size_t i = 0; i < mgr->raft_peer_count; i++)
        if (mgr->raft_peer_ids[i] == id) return mgr->raft_peers[i];
    return NULL;
}

/* raft send callback (ctx = sending manager): enqueue a deep copy to the peer's
 * inbox under the peer's leaf inbox lock - never the peer's raft lock - so a
 * send can't re-enter or deadlock a peer that is mid-tick. */
static void repl_raft_send(void *ctx, int to, const GV_RaftMsg *msg) {
    GV_ReplicationManager *mgr = (GV_ReplicationManager *)ctx;
    GV_ReplicationManager *peer = repl_raft_find_peer(mgr, to);
    if (!peer) return;
    RaftInboxMsg *qm = (RaftInboxMsg *)gv_calloc(1, sizeof(RaftInboxMsg));
    if (!qm) return;
    qm->from = mgr->raft_self_id;
    qm->msg = *msg;
    qm->next = NULL;
    if (msg->type == GV_RAFT_MSG_APPEND_ENTRIES && msg->n_entries > 0) {
        qm->entries = (GV_RaftEntry *)gv_calloc(msg->n_entries, sizeof(GV_RaftEntry));
        if (!qm->entries) { gv_free(qm); return; }
        for (size_t i = 0; i < msg->n_entries; i++) {
            qm->entries[i].term = msg->entries[i].term;
            qm->entries[i].len = msg->entries[i].len;
            qm->entries[i].data = NULL;
            if (msg->entries[i].len) {
                qm->entries[i].data = gv_alloc(msg->entries[i].len);
                if (!qm->entries[i].data) {
                    /* Drop the whole message on OOM rather than enqueue an entry
                     * with len>0 and data==NULL (which raft_step would memcpy from). */
                    qm->msg.n_entries = i;   /* only [0,i) were fully copied */
                    repl_raft_free_inbox_msg(qm);
                    return;
                }
                memcpy(qm->entries[i].data, msg->entries[i].data, msg->entries[i].len);
            }
        }
        qm->msg.entries = qm->entries;
    }
    pthread_mutex_lock(&peer->raft_inbox_mutex);
    if (peer->raft_inbox_tail) peer->raft_inbox_tail->next = qm; else peer->raft_inbox_head = qm;
    peer->raft_inbox_tail = qm;
    pthread_mutex_unlock(&peer->raft_inbox_mutex);
}

/* raft apply callback: a committed entry advances the replication commit position. */
static void repl_raft_apply(void *ctx, uint64_t index, const void *data, size_t len) {
    (void)data; (void)len;
    GV_ReplicationManager *mgr = (GV_ReplicationManager *)ctx;
    pthread_rwlock_wrlock(&mgr->rwlock);
    if (index > mgr->commit_position) mgr->commit_position = index;
    pthread_rwlock_unlock(&mgr->rwlock);
}

static void repl_raft_free_inbox_msg(RaftInboxMsg *qm) {
    if (qm->entries) {
        for (size_t i = 0; i < qm->msg.n_entries; i++) gv_free(qm->entries[i].data);
        gv_free(qm->entries);
    }
    gv_free(qm);
}

/* Reflect raft leadership/term into the replication role. Caller holds raft_mutex. */
static void repl_raft_sync_role(GV_ReplicationManager *mgr) {
    GV_RaftRole rr = raft_role(mgr->raft);
    uint64_t rterm = raft_current_term(mgr->raft);
    pthread_rwlock_wrlock(&mgr->rwlock);
    mgr->role = (rr == GV_RAFT_LEADER) ? GV_REPL_LEADER
              : (rr == GV_RAFT_CANDIDATE) ? GV_REPL_CANDIDATE : GV_REPL_FOLLOWER;
    mgr->term = rterm;
    if (rr == GV_RAFT_LEADER) {
        gv_free(mgr->leader_id);
        mgr->leader_id = mgr->node_id ? gv_dup_cstr(mgr->node_id) : NULL;
    }
    pthread_rwlock_unlock(&mgr->rwlock);
}

int replication_enable_raft(GV_ReplicationManager *mgr, int self_id,
                            const int *peer_ids, size_t n_peers) {
    if (!mgr || (n_peers > 0 && !peer_ids)) return -1;
    pthread_mutex_lock(&mgr->raft_mutex);
    if (mgr->raft) { pthread_mutex_unlock(&mgr->raft_mutex); return -1; }
    GV_RaftConfig cfg; raft_config_init(&cfg);
    GV_RaftCallbacks cb; memset(&cb, 0, sizeof(cb));
    cb.send = repl_raft_send; cb.apply = repl_raft_apply; cb.ctx = mgr;
    GV_Raft *r = raft_create(self_id, peer_ids, n_peers, &cfg, &cb);
    /* Publish mgr->raft under rwlock: the replication worker reads it (the
     * !mgr->raft election gating) under rwlock, so writing it under raft_mutex
     * alone would be a torn/racy read. Order raft_mutex -> rwlock matches the
     * global lock hierarchy. */
    pthread_rwlock_wrlock(&mgr->rwlock);
    mgr->raft = r;
    mgr->raft_self_id = self_id;
    pthread_rwlock_unlock(&mgr->rwlock);
    int ok = r != NULL;
    pthread_mutex_unlock(&mgr->raft_mutex);
    return ok ? 0 : -1;
}

int replication_register_raft_peer(GV_ReplicationManager *mgr, int peer_id,
                                   GV_ReplicationManager *peer) {
    if (!mgr || !peer) return -1;
    /* The send callback reads the peer arrays while holding raft_mutex (during a
     * tick/step); take the same lock here so a late registration can't race a
     * concurrent send. Registration is still expected to be setup-only. */
    pthread_mutex_lock(&mgr->raft_mutex);
    if (mgr->raft_peer_count >= MAX_REPLICAS) { pthread_mutex_unlock(&mgr->raft_mutex); return -1; }
    mgr->raft_peer_ids[mgr->raft_peer_count] = peer_id;
    mgr->raft_peers[mgr->raft_peer_count] = peer;
    mgr->raft_peer_count++;
    pthread_mutex_unlock(&mgr->raft_mutex);
    return 0;
}

int replication_raft_tick(GV_ReplicationManager *mgr, uint32_t ms) {
    if (!mgr) return -1;
    pthread_mutex_lock(&mgr->raft_mutex);
    if (!mgr->raft) { pthread_mutex_unlock(&mgr->raft_mutex); return -1; }
    raft_tick(mgr->raft, ms);
    /* Drain the inbox (snapshot under the leaf lock) then step each message. */
    pthread_mutex_lock(&mgr->raft_inbox_mutex);
    RaftInboxMsg *head = mgr->raft_inbox_head;
    mgr->raft_inbox_head = NULL;
    mgr->raft_inbox_tail = NULL;
    pthread_mutex_unlock(&mgr->raft_inbox_mutex);
    while (head) {
        RaftInboxMsg *next = head->next;
        raft_step(mgr->raft, head->from, &head->msg);
        repl_raft_free_inbox_msg(head);
        head = next;
    }
    repl_raft_sync_role(mgr);
    pthread_mutex_unlock(&mgr->raft_mutex);
    return 0;
}

int replication_raft_submit(GV_ReplicationManager *mgr, const void *data, size_t len) {
    if (!mgr) return -1;
    pthread_mutex_lock(&mgr->raft_mutex);
    if (!mgr->raft) { pthread_mutex_unlock(&mgr->raft_mutex); return -1; }
    int rc = raft_submit(mgr->raft, data, len, NULL);
    pthread_mutex_unlock(&mgr->raft_mutex);
    return rc;
}

void replication_note_leader_heartbeat(GV_ReplicationManager *mgr) {
    if (!mgr) return;
    pthread_rwlock_wrlock(&mgr->rwlock);
    mgr->role = GV_REPL_FOLLOWER;
    mgr->last_leader_contact = gv_time_now_sec();  /* drives TCP-follower failure detection */
    /* Refresh the leader replica's heartbeat so the follower election-timeout path
     * (which keys off leader_id -> replica.last_heartbeat) sees the node as alive.
     * NOTE: a node started as a follower has leader_id == NULL until a handshake
     * establishes it; full TCP-follower failure detection additionally needs the
     * leader's node id carried in the heartbeat (or the Raft core wired in). */
    if (mgr->leader_id) {
        uint64_t now = gv_time_now_sec();
        for (size_t i = 0; i < mgr->replica_count; i++) {
            if (mgr->replicas[i].node_id &&
                strcmp(mgr->replicas[i].node_id, mgr->leader_id) == 0) {
                mgr->replicas[i].last_heartbeat = now;
                break;
            }
        }
    }
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
