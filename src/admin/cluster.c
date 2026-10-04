/**
 * @file cluster.c
 * @brief Cluster management implementation.
 */

#include <stdatomic.h>
#include "admin/cluster.h"
#include "admin/shard_rpc.h"
#include "admin/raft.h"
#include "core/memory.h"
#include "core/utils.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>
#include <time.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include "core/compat.h"

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

#define RAFT_OUTBOX_CAP 256

typedef struct { int to; uint8_t *bytes; size_t len; } RaftOut;

struct GV_Cluster {
    GV_ClusterConfig config;
    char *local_node_id;
    GV_ShardManager *shard_mgr;
    GV_ShardRpcServer *rpc_server;   /* serves distributed search to peers */

    /* Optional raft leader election over the RPC transport. */
    GV_Raft         *raft;
    pthread_mutex_t  raft_lock;
    pthread_t        raft_thread;
    _Atomic int      raft_running;
    char           **raft_addrs;     /* index -> "host:port" */
    size_t           raft_n;
    int              raft_my_index;
    RaftOut          outbox[RAFT_OUTBOX_CAP];
    size_t           outbox_n;

    /* Replicated state machine: shard->owner placement applied from the raft log. */
    struct { uint64_t shard; int node; } placement[256];
    size_t           placement_n;

    NodeEntry nodes[MAX_NODES];
    size_t node_count;

    /* Heartbeat thread */
    pthread_t heartbeat_thread;
    int heartbeat_running;
    _Atomic int stop_requested;

    pthread_rwlock_t rwlock;
    pthread_mutex_t state_mutex;
    pthread_cond_t ready_cond;
    int is_ready;
};

static const GV_ClusterConfig DEFAULT_CONFIG = {
    .node_id = NULL,
    .listen_address = "0.0.0.0:7000",
    .seed_nodes = NULL,
    .role = GV_NODE_DATA,
    .heartbeat_interval_ms = 1000,
    .failure_timeout_ms = 5000
};

void cluster_config_init(GV_ClusterConfig *config) {
    if (!config) return;
    *config = DEFAULT_CONFIG;
}

static char *generate_node_id(void) {
    char id[64];
    snprintf(id, sizeof(id), "node-%lx-%d",
             (unsigned long)time(NULL), (int)getpid());
    return gv_dup_cstr(id);
}

static void cluster_gossip_pull(GV_Cluster *cluster);
static void cluster_raft_shutdown(GV_Cluster *cluster);

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
                /* Guard against unsigned underflow on backward clock movement
                 * (would otherwise mark a live node DEAD). */
                if (now > cluster->nodes[i].last_heartbeat &&
                    now - cluster->nodes[i].last_heartbeat > timeout) {
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

        /* Anti-entropy gossip: pull each known peer's member list and learn any
         * peers we did not have, so membership converges from partial seeds. */
        cluster_gossip_pull(cluster);
    }

    return NULL;
}

GV_Cluster *cluster_create(const GV_ClusterConfig *config) {
    GV_Cluster *cluster = gv_calloc(1, sizeof(GV_Cluster));
    if (!cluster) return NULL;

    cluster->config = config ? *config : DEFAULT_CONFIG;

    /* Generate node ID if not provided */
    if (cluster->config.node_id) {
        cluster->local_node_id = gv_dup_cstr(cluster->config.node_id);
    } else {
        cluster->local_node_id = generate_node_id();
    }

    /* Create shard manager */
    cluster->shard_mgr = shard_manager_create(NULL);
    if (!cluster->shard_mgr) {
        gv_free(cluster->local_node_id);
        gv_free(cluster);
        return NULL;
    }

    if (pthread_rwlock_init(&cluster->rwlock, NULL) != 0) {
        shard_manager_destroy(cluster->shard_mgr);
        gv_free(cluster->local_node_id);
        gv_free(cluster);
        return NULL;
    }

    if (pthread_mutex_init(&cluster->state_mutex, NULL) != 0) {
        pthread_rwlock_destroy(&cluster->rwlock);
        shard_manager_destroy(cluster->shard_mgr);
        gv_free(cluster->local_node_id);
        gv_free(cluster);
        return NULL;
    }

    if (pthread_cond_init(&cluster->ready_cond, NULL) != 0) {
        pthread_mutex_destroy(&cluster->state_mutex);
        pthread_rwlock_destroy(&cluster->rwlock);
        shard_manager_destroy(cluster->shard_mgr);
        gv_free(cluster->local_node_id);
        gv_free(cluster);
        return NULL;
    }

    /* Add local node */
    NodeEntry *local = &cluster->nodes[0];
    local->node_id = gv_dup_cstr(cluster->local_node_id);
    local->address = cluster->config.listen_address ? gv_dup_cstr(cluster->config.listen_address) : NULL;
    local->role = cluster->config.role;
    local->state = GV_NODE_JOINING;
    local->shard_ids = NULL;
    local->shard_count = 0;
    local->last_heartbeat = (uint64_t)time(NULL);
    local->load = 0.0;
    cluster->node_count = 1;

    return cluster;
}

void cluster_destroy(GV_Cluster *cluster) {
    if (!cluster) return;

    cluster_stop(cluster);

    /* Guard against a start that bound the RPC server but never ran heartbeat. */
    if (cluster->rpc_server) {
        shard_rpc_server_stop(cluster->rpc_server);
        cluster->rpc_server = NULL;
    }
    cluster_raft_shutdown(cluster);

    for (size_t i = 0; i < cluster->node_count; i++) {
        gv_free(cluster->nodes[i].node_id);
        gv_free(cluster->nodes[i].address);
        gv_free(cluster->nodes[i].shard_ids);
    }

    pthread_cond_destroy(&cluster->ready_cond);
    pthread_mutex_destroy(&cluster->state_mutex);
    pthread_rwlock_destroy(&cluster->rwlock);

    shard_manager_destroy(cluster->shard_mgr);
    gv_free(cluster->local_node_id);
    gv_free(cluster);
}

/** @brief Split "host:port" into host + port. @return 0 on success, -1 on error. */
static int cluster_split_addr(const char *addr, char *host, size_t hostsz,
                              uint16_t *port) {
    if (!addr) return -1;
    const char *colon = strrchr(addr, ':');
    if (!colon || colon == addr) return -1;
    size_t hlen = (size_t)(colon - addr);
    if (hlen >= hostsz) return -1;
    memcpy(host, addr, hlen);
    host[hlen] = '\0';
    int p = (int)strtol(colon + 1, NULL, 10);
    if (p < 0 || p > 65535) return -1;
    *port = (uint16_t)p;
    return 0;
}

int cluster_attach_local_db(GV_Cluster *cluster, GV_Database *db) {
    if (!cluster || !db) return -1;
    /* The local node owns shard 0 (node index 0); peers occupy 1.. */
    shard_add(cluster->shard_mgr, 0, cluster->config.listen_address);
    return shard_attach_local(cluster->shard_mgr, 0, db);
}

uint16_t cluster_rpc_port(const GV_Cluster *cluster) {
    return (cluster && cluster->rpc_server)
        ? shard_rpc_server_port(cluster->rpc_server) : 0;
}

/* This node's advertised address: listen host + the actual bound RPC port. */
static void cluster_self_addr(GV_Cluster *c, char *out, size_t sz) {
    char host[256];
    uint16_t port;
    out[0] = '\0';
    if (!c->config.listen_address) return;
    if (cluster_split_addr(c->config.listen_address, host, sizeof(host), &port) != 0) return;
    uint16_t rp = cluster_rpc_port(c);
    snprintf(out, sz, "%s:%u", host, (unsigned)(rp ? rp : port));
}

/* Membership provider: newline-separated addresses of self + live peers. */
static int cluster_members_cb(char *out, size_t sz, void *ctx) {
    GV_Cluster *c = (GV_Cluster *)ctx;
    char self[288];
    cluster_self_addr(c, self, sizeof(self));

    size_t off = 0;
    pthread_rwlock_rdlock(&c->rwlock);
    if (self[0] && off + 1 < sz)
        off += (size_t)snprintf(out + off, sz - off, "%s\n", self);
    for (size_t i = 0; i < c->node_count && off + 1 < sz; i++) {
        const NodeEntry *n = &c->nodes[i];
        if (!n->address || n->state == GV_NODE_DEAD) continue;
        if (strcmp(n->address, self) == 0) continue;
        off += (size_t)snprintf(out + off, sz - off, "%s\n", n->address);
    }
    pthread_rwlock_unlock(&c->rwlock);
    return (int)off;
}

/* Learn a peer address as a node + remote shard if unknown. Caller holds wrlock. */
static void cluster_learn_locked(GV_Cluster *c, const char *addr, const char *self) {
    if (!addr || !*addr) return;
    if (self && strcmp(addr, self) == 0) return;
    for (size_t i = 0; i < c->node_count; i++) {
        if (c->nodes[i].address && strcmp(c->nodes[i].address, addr) == 0) return;
    }
    if (c->node_count >= MAX_NODES) return;

    char nid[300];
    snprintf(nid, sizeof(nid), "gossip-%.290s", addr);
    NodeEntry *e = &c->nodes[c->node_count];
    e->node_id = gv_dup_cstr(nid);
    e->address = gv_dup_cstr(addr);
    if (!e->node_id || !e->address) {
        gv_free(e->node_id);
        gv_free(e->address);
        e->node_id = e->address = NULL;
        return;
    }
    e->role = GV_NODE_DATA;
    e->state = GV_NODE_ACTIVE;
    e->shard_ids = NULL;
    e->shard_count = 0;
    e->last_heartbeat = (uint64_t)time(NULL);
    e->load = 0.0;
    c->node_count++;
    /* Register as a remote shard so cluster_search fans out to it. */
    shard_add(c->shard_mgr, (uint32_t)(c->node_count - 1), addr);
}

static void cluster_gossip_pull(GV_Cluster *cluster) {
    char self[288];
    cluster_self_addr(cluster, self, sizeof(self));

    /* Snapshot peer addresses so network I/O happens outside the lock. */
    char peers[MAX_NODES][288];
    size_t npeers = 0;
    pthread_rwlock_rdlock(&cluster->rwlock);
    for (size_t i = 0; i < cluster->node_count && npeers < MAX_NODES; i++) {
        const NodeEntry *n = &cluster->nodes[i];
        if (n->address && n->state != GV_NODE_DEAD && strcmp(n->address, self) != 0)
            snprintf(peers[npeers++], sizeof(peers[0]), "%s", n->address);
    }
    pthread_rwlock_unlock(&cluster->rwlock);

    for (size_t i = 0; i < npeers; i++) {
        char host[256];
        uint16_t port;
        if (cluster_split_addr(peers[i], host, sizeof(host), &port) != 0) continue;
        char buf[1 << 16];
        if (shard_rpc_fetch_members(host, port, buf, sizeof(buf)) <= 0) continue;

        pthread_rwlock_wrlock(&cluster->rwlock);
        char *save = NULL;
        for (char *line = strtok_r(buf, "\n", &save); line;
             line = strtok_r(NULL, "\n", &save)) {
            cluster_learn_locked(cluster, line, self);
        }
        pthread_rwlock_unlock(&cluster->rwlock);
    }
}

/* ── Raft leader election ─────────────────────────────────────────────── */

static size_t raft_put32(uint8_t *b, size_t o, uint32_t v) { memcpy(b + o, &v, 4); return o + 4; }
static size_t raft_put64(uint8_t *b, size_t o, uint64_t v) { memcpy(b + o, &v, 8); return o + 8; }
static size_t raft_get32(const uint8_t *b, size_t o, uint32_t *v) { memcpy(v, b + o, 4); return o + 4; }
static size_t raft_get64(const uint8_t *b, size_t o, uint64_t *v) { memcpy(v, b + o, 8); return o + 8; }

/* Fixed 68-byte scalar header + n_entries; entries carry election traffic (0). */
static size_t raft_msg_serialize(const GV_RaftMsg *m, uint8_t *b, size_t cap) {
    size_t need = 4 + 8 + 4 + 8 + 8 + 4 + 8 + 8 + 8 + 4 + 8 + 4;
    for (size_t i = 0; i < m->n_entries; i++) need += 8 + 4 + m->entries[i].len;
    if (need > cap) return 0;
    size_t o = 0;
    o = raft_put32(b, o, (uint32_t)m->type);
    o = raft_put64(b, o, m->term);
    o = raft_put32(b, o, (uint32_t)m->from);
    o = raft_put64(b, o, m->last_log_index);
    o = raft_put64(b, o, m->last_log_term);
    o = raft_put32(b, o, (uint32_t)m->vote_granted);
    o = raft_put64(b, o, m->prev_log_index);
    o = raft_put64(b, o, m->prev_log_term);
    o = raft_put64(b, o, m->leader_commit);
    o = raft_put32(b, o, (uint32_t)m->success);
    o = raft_put64(b, o, m->match_index);
    o = raft_put32(b, o, (uint32_t)m->n_entries);
    for (size_t i = 0; i < m->n_entries; i++) {
        o = raft_put64(b, o, m->entries[i].term);
        o = raft_put32(b, o, (uint32_t)m->entries[i].len);
        memcpy(b + o, m->entries[i].data, m->entries[i].len);
        o += m->entries[i].len;
    }
    return o;
}

/* Deserialize into @p m; entry payloads point into @p b (borrowed during step). */
static int raft_msg_deserialize(const uint8_t *b, size_t len, GV_RaftMsg *m,
                                GV_RaftEntry *ebuf, size_t ecap) {
    if (len < 68) return -1;
    size_t o = 0;
    uint32_t t32;
    o = raft_get32(b, o, &t32); m->type = (GV_RaftMsgType)t32;
    o = raft_get64(b, o, &m->term);
    o = raft_get32(b, o, &t32); m->from = (int)t32;
    o = raft_get64(b, o, &m->last_log_index);
    o = raft_get64(b, o, &m->last_log_term);
    o = raft_get32(b, o, &t32); m->vote_granted = (int)t32;
    o = raft_get64(b, o, &m->prev_log_index);
    o = raft_get64(b, o, &m->prev_log_term);
    o = raft_get64(b, o, &m->leader_commit);
    o = raft_get32(b, o, &t32); m->success = (int)t32;
    o = raft_get64(b, o, &m->match_index);
    uint32_t ne;
    o = raft_get32(b, o, &ne);
    if (ne > ecap) return -1;
    for (uint32_t i = 0; i < ne; i++) {
        if (o + 12 > len) return -1;
        o = raft_get64(b, o, &ebuf[i].term);
        uint32_t el;
        o = raft_get32(b, o, &el);
        if (o + el > len) return -1;
        ebuf[i].data = (void *)(b + o);
        ebuf[i].len = el;
        o += el;
    }
    m->entries = ne ? ebuf : NULL;
    m->n_entries = ne;
    return 0;
}

/* Raft send callback: queue the message; flushed after the raft lock drops. */
static void raft_send_cb(void *ctx, int to, const GV_RaftMsg *msg) {
    GV_Cluster *c = (GV_Cluster *)ctx;
    if (c->outbox_n >= RAFT_OUTBOX_CAP) return; /* raft tolerates dropped messages */
    uint8_t tmp[4096];
    size_t len = raft_msg_serialize(msg, tmp, sizeof(tmp));
    if (len == 0) return;
    uint8_t *copy = (uint8_t *)gv_alloc(len);
    if (!copy) return;
    memcpy(copy, tmp, len);
    c->outbox[c->outbox_n].to = to;
    c->outbox[c->outbox_n].bytes = copy;
    c->outbox[c->outbox_n].len = len;
    c->outbox_n++;
}

#define CLUSTER_OP_ASSIGN_SHARD 1  /* [u8 op][u64 shard][u32 node] */

/* Apply a committed raft log entry to the replicated state machine. Runs on
 * every node under the raft lock (invoked from within raft_step/raft_tick). */
static void cluster_raft_apply(void *ctx, uint64_t index, const void *data, size_t len) {
    (void)index;
    GV_Cluster *c = (GV_Cluster *)ctx;
    const uint8_t *b = (const uint8_t *)data;
    if (len < 1 + 8 + 4 || b[0] != CLUSTER_OP_ASSIGN_SHARD) return;

    uint64_t shard;
    uint32_t node;
    memcpy(&shard, b + 1, 8);
    memcpy(&node, b + 9, 4);

    for (size_t i = 0; i < c->placement_n; i++) {
        if (c->placement[i].shard == shard) { c->placement[i].node = (int)node; return; }
    }
    if (c->placement_n < 256) {
        c->placement[c->placement_n].shard = shard;
        c->placement[c->placement_n].node = (int)node;
        c->placement_n++;
    }
}

/* Deliver queued messages over the network (no locks held). */
static void raft_flush(GV_Cluster *c, RaftOut *items, size_t n) {
    for (size_t i = 0; i < n; i++) {
        int to = items[i].to;
        if (to >= 0 && (size_t)to < c->raft_n && c->raft_addrs[to]) {
            char host[256];
            uint16_t port;
            if (cluster_split_addr(c->raft_addrs[to], host, sizeof(host), &port) == 0)
                shard_rpc_send_raft(host, port, items[i].bytes, items[i].len);
        }
        gv_free(items[i].bytes);
    }
}

static void cluster_raft_recv(const void *bytes, size_t len, void *ctx) {
    GV_Cluster *c = (GV_Cluster *)ctx;
    GV_RaftMsg m;
    memset(&m, 0, sizeof(m));
    GV_RaftEntry entries[64];
    if (raft_msg_deserialize((const uint8_t *)bytes, len, &m, entries, 64) != 0) return;

    RaftOut local[RAFT_OUTBOX_CAP];
    size_t n = 0;
    pthread_mutex_lock(&c->raft_lock);
    if (c->raft) {
        raft_step(c->raft, m.from, &m);
        n = c->outbox_n;
        memcpy(local, c->outbox, n * sizeof(RaftOut));
        c->outbox_n = 0;
    }
    pthread_mutex_unlock(&c->raft_lock);
    raft_flush(c, local, n);
}

static void *raft_thread_func(void *arg) {
    GV_Cluster *c = (GV_Cluster *)arg;
    const uint32_t tick_ms = 10;
    while (atomic_load(&c->raft_running)) {
        RaftOut local[RAFT_OUTBOX_CAP];
        size_t n;
        pthread_mutex_lock(&c->raft_lock);
        raft_tick(c->raft, tick_ms);
        n = c->outbox_n;
        memcpy(local, c->outbox, n * sizeof(RaftOut));
        c->outbox_n = 0;
        pthread_mutex_unlock(&c->raft_lock);
        raft_flush(c, local, n);
        usleep(tick_ms * 1000);
    }
    return NULL;
}

static void cluster_raft_shutdown(GV_Cluster *c) {
    if (!c->raft) return;
    atomic_store(&c->raft_running, 0);
    pthread_join(c->raft_thread, NULL);
    if (c->rpc_server) shard_rpc_set_raft_handler(c->rpc_server, NULL, NULL);

    pthread_mutex_lock(&c->raft_lock);
    raft_destroy(c->raft);
    c->raft = NULL;
    for (size_t i = 0; i < c->outbox_n; i++) gv_free(c->outbox[i].bytes);
    c->outbox_n = 0;
    pthread_mutex_unlock(&c->raft_lock);
    pthread_mutex_destroy(&c->raft_lock);

    for (size_t i = 0; i < c->raft_n; i++) gv_free(c->raft_addrs[i]);
    gv_free(c->raft_addrs);
    c->raft_addrs = NULL;
    c->raft_n = 0;
}

int cluster_enable_raft(GV_Cluster *cluster, const char *const *addrs,
                        size_t n, int my_index) {
    if (!cluster || !addrs || n < 1 || my_index < 0 || (size_t)my_index >= n) return -1;
    if (!cluster->rpc_server || cluster->raft) return -1;

    cluster->raft_addrs = (char **)gv_calloc(n, sizeof(char *));
    if (!cluster->raft_addrs) return -1;
    for (size_t i = 0; i < n; i++) {
        cluster->raft_addrs[i] = gv_dup_cstr(addrs[i]);
        if (!cluster->raft_addrs[i]) {
            for (size_t j = 0; j < i; j++) gv_free(cluster->raft_addrs[j]);
            gv_free(cluster->raft_addrs);
            cluster->raft_addrs = NULL;
            return -1;
        }
    }
    cluster->raft_n = n;
    cluster->raft_my_index = my_index;

    int *peers = (int *)gv_alloc((n > 1 ? n - 1 : 1) * sizeof(int));
    if (!peers) { cluster_raft_shutdown(cluster); return -1; }
    size_t np = 0;
    for (int i = 0; i < (int)n; i++) if (i != my_index) peers[np++] = i;

    GV_RaftConfig cfg;
    raft_config_init(&cfg);
    GV_RaftCallbacks cb = {
        .send = raft_send_cb,
        .apply = cluster_raft_apply,
        .persist = NULL,
        .persist_log = NULL,
        .truncate_log = NULL,
        .ctx = cluster,
    };
    pthread_mutex_init(&cluster->raft_lock, NULL);
    cluster->outbox_n = 0;
    cluster->raft = raft_create(my_index, peers, np, &cfg, &cb);
    gv_free(peers);
    if (!cluster->raft) {
        pthread_mutex_destroy(&cluster->raft_lock);
        for (size_t i = 0; i < n; i++) gv_free(cluster->raft_addrs[i]);
        gv_free(cluster->raft_addrs);
        cluster->raft_addrs = NULL;
        cluster->raft_n = 0;
        return -1;
    }

    shard_rpc_set_raft_handler(cluster->rpc_server, cluster_raft_recv, cluster);
    atomic_store(&cluster->raft_running, 1);
    if (pthread_create(&cluster->raft_thread, NULL, raft_thread_func, cluster) != 0) {
        atomic_store(&cluster->raft_running, 0);
        cluster_raft_shutdown(cluster);
        return -1;
    }
    return 0;
}

int cluster_raft_leader(GV_Cluster *cluster) {
    if (!cluster || !cluster->raft) return -1;
    pthread_mutex_lock(&cluster->raft_lock);
    int leader = raft_leader_id(cluster->raft);
    pthread_mutex_unlock(&cluster->raft_lock);
    return leader;
}

int cluster_is_raft_leader(GV_Cluster *cluster) {
    if (!cluster || !cluster->raft) return 0;
    pthread_mutex_lock(&cluster->raft_lock);
    int is_leader = raft_role(cluster->raft) == GV_RAFT_LEADER;
    pthread_mutex_unlock(&cluster->raft_lock);
    return is_leader;
}

int cluster_assign_shard(GV_Cluster *cluster, uint64_t shard_id, int node_index) {
    if (!cluster || !cluster->raft) return -1;

    uint8_t cmd[1 + 8 + 4];
    cmd[0] = CLUSTER_OP_ASSIGN_SHARD;
    memcpy(cmd + 1, &shard_id, 8);
    uint32_t node = (uint32_t)node_index;
    memcpy(cmd + 9, &node, 4);

    RaftOut local[RAFT_OUTBOX_CAP];
    size_t n = 0;
    pthread_mutex_lock(&cluster->raft_lock);
    int rc = raft_submit(cluster->raft, cmd, sizeof(cmd), NULL); /* leader-only */
    n = cluster->outbox_n;
    memcpy(local, cluster->outbox, n * sizeof(RaftOut));
    cluster->outbox_n = 0;
    pthread_mutex_unlock(&cluster->raft_lock);
    raft_flush(cluster, local, n);
    return rc;
}

int cluster_shard_owner(GV_Cluster *cluster, uint64_t shard_id) {
    if (!cluster || !cluster->raft) return -1;
    int owner = -1;
    pthread_mutex_lock(&cluster->raft_lock);
    for (size_t i = 0; i < cluster->placement_n; i++) {
        if (cluster->placement[i].shard == shard_id) { owner = cluster->placement[i].node; break; }
    }
    pthread_mutex_unlock(&cluster->raft_lock);
    return owner;
}

int cluster_start(GV_Cluster *cluster) {
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

    if (cluster->config.seed_nodes && cluster->config.seed_nodes[0] != '\0') {
        char *seeds = gv_dup_cstr(cluster->config.seed_nodes);
        if (seeds) {
            char *saveptr = NULL;
            char *tok = strtok_r(seeds, ",", &saveptr);
            while (tok) {
                while (*tok == ' ' || *tok == '\t') tok++;
                char *end = tok + strlen(tok) - 1;
                while (end > tok && (*end == ' ' || *end == '\t')) {
                    *end-- = '\0';
                }
                if (*tok != '\0' && cluster->node_count < MAX_NODES) {
                    char seed_id[128];
                    snprintf(seed_id, sizeof(seed_id), "seed-%.122s", tok);
                    for (char *p = seed_id + 5; *p; p++) {
                        if (*p == ':' || *p == '.') *p = '_';
                    }
                    if (!find_node(cluster, seed_id)) {
                        char *nid = gv_dup_cstr(seed_id);
                        char *addr = gv_dup_cstr(tok);
                        if (nid && addr) {
                            NodeEntry *entry = &cluster->nodes[cluster->node_count];
                            entry->node_id = nid;
                            entry->address = addr;
                            entry->role = GV_NODE_DATA;
                            entry->state = GV_NODE_JOINING;
                            entry->shard_ids = NULL;
                            entry->shard_count = 0;
                            entry->last_heartbeat = (uint64_t)time(NULL);
                            entry->load = 0.0;
                            cluster->node_count++;
                            /* Register the peer as a remote shard (node index ==
                             * shard id) so cluster_search fans out to it. */
                            shard_add(cluster->shard_mgr,
                                      (uint32_t)(cluster->node_count - 1), tok);
                        } else {
                            gv_free(nid);
                            gv_free(addr);
                        }
                    }
                }
                tok = strtok_r(NULL, ",", &saveptr);
            }
            gv_free(seeds);
        }
    }

    pthread_rwlock_unlock(&cluster->rwlock);

    /* Auto-start the search RPC listener so peers can query this node's shards. */
    if (cluster->config.listen_address && !cluster->rpc_server) {
        char host[256];
        uint16_t port;
        if (cluster_split_addr(cluster->config.listen_address, host,
                               sizeof(host), &port) == 0) {
            cluster->rpc_server = shard_rpc_serve(cluster->shard_mgr, host, port);
            if (cluster->rpc_server) {
                shard_rpc_set_members_provider(cluster->rpc_server,
                                               cluster_members_cb, cluster);
            }
        }
    }

    /* Start heartbeat thread */
    if (pthread_create(&cluster->heartbeat_thread, NULL, heartbeat_thread_func, cluster) != 0) {
        pthread_rwlock_wrlock(&cluster->rwlock);
        cluster->heartbeat_running = 0;
        pthread_rwlock_unlock(&cluster->rwlock);
        return -1;
    }

    /* Mark cluster ready */
    pthread_mutex_lock(&cluster->state_mutex);
    cluster->is_ready = 1;
    pthread_cond_broadcast(&cluster->ready_cond);
    pthread_mutex_unlock(&cluster->state_mutex);

    return 0;
}

int cluster_stop(GV_Cluster *cluster) {
    if (!cluster) return -1;

    pthread_rwlock_wrlock(&cluster->rwlock);

    if (!cluster->heartbeat_running) {
        pthread_rwlock_unlock(&cluster->rwlock);
        return 0;
    }

    cluster->stop_requested = 1;
    pthread_rwlock_unlock(&cluster->rwlock);

    pthread_join(cluster->heartbeat_thread, NULL);

    /* Stop the RPC listener first (joins its accept thread) so no inbound raft
     * message can touch the raft state/mutex we are about to destroy. */
    if (cluster->rpc_server) {
        shard_rpc_server_stop(cluster->rpc_server);
        cluster->rpc_server = NULL;
    }

    /* Then stop the raft driver thread and free raft state. */
    cluster_raft_shutdown(cluster);

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

static void copy_node_info(const NodeEntry *entry, GV_NodeInfo *info) {
    info->node_id = entry->node_id ? gv_dup_cstr(entry->node_id) : NULL;
    info->address = entry->address ? gv_dup_cstr(entry->address) : NULL;
    info->role = entry->role;
    info->state = entry->state;
    if (entry->shard_count > 0 && entry->shard_ids) {
        info->shard_ids = gv_alloc(entry->shard_count * sizeof(uint32_t));
        memcpy(info->shard_ids, entry->shard_ids, entry->shard_count * sizeof(uint32_t));
    } else {
        info->shard_ids = NULL;
    }
    info->shard_count = entry->shard_count;
    info->last_heartbeat = entry->last_heartbeat;
    info->load = entry->load;
}

int cluster_get_local_node(GV_Cluster *cluster, GV_NodeInfo *info) {
    if (!cluster || !info) return -1;
    return cluster_get_node(cluster, cluster->local_node_id, info);
}

int cluster_get_node(GV_Cluster *cluster, const char *node_id, GV_NodeInfo *info) {
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

int cluster_list_nodes(GV_Cluster *cluster, GV_NodeInfo **nodes, size_t *count) {
    if (!cluster || !nodes || !count) return -1;

    pthread_rwlock_rdlock(&cluster->rwlock);

    *count = cluster->node_count;
    if (*count == 0) {
        *nodes = NULL;
        pthread_rwlock_unlock(&cluster->rwlock);
        return 0;
    }

    *nodes = gv_alloc(*count * sizeof(GV_NodeInfo));
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

void cluster_free_node_info(GV_NodeInfo *info) {
    if (!info) return;
    gv_free(info->node_id);
    gv_free(info->address);
    gv_free(info->shard_ids);
    memset(info, 0, sizeof(*info));
}

void cluster_free_node_list(GV_NodeInfo *nodes, size_t count) {
    if (!nodes) return;
    for (size_t i = 0; i < count; i++) {
        cluster_free_node_info(&nodes[i]);
    }
    gv_free(nodes);
}

int cluster_get_stats(GV_Cluster *cluster, GV_ClusterStats *stats) {
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
    if (shard_list(cluster->shard_mgr, &shards, &shard_count) == 0) {
        stats->total_shards = shard_count;
        for (size_t i = 0; i < shard_count; i++) {
            stats->total_vectors += shards[i].vector_count;
        }
        shard_free_list(shards, shard_count);
    }

    pthread_rwlock_unlock(&cluster->rwlock);
    return 0;
}

GV_ShardManager *cluster_get_shard_manager(GV_Cluster *cluster) {
    if (!cluster) return NULL;
    return cluster->shard_mgr;
}

int cluster_is_healthy(GV_Cluster *cluster) {
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

int cluster_wait_ready(GV_Cluster *cluster, uint32_t timeout_ms) {
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

int cluster_search(GV_Cluster *cluster, const float *query_data, size_t dim,
                   size_t k, GV_SearchResult *results,
                   GV_DistanceType distance_type) {
    GV_ShardManager *mgr = cluster_get_shard_manager(cluster);
    if (!mgr) return -1;
    return shard_rpc_search_distributed(mgr, query_data, dim, k, distance_type,
                                        results);
}
