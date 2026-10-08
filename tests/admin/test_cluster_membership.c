#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "admin/cluster.h"
#include "storage/database.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int addr_in_nodes(GV_Cluster *c, const char *addr) {
    GV_NodeInfo *nodes = NULL;
    size_t n = 0;
    if (cluster_list_nodes(c, &nodes, &n) != 0) return 0;
    int found = 0;
    for (size_t i = 0; i < n; i++)
        if (nodes[i].address && strcmp(nodes[i].address, addr) == 0) found = 1;
    cluster_free_node_list(nodes, n);
    return found;
}

/* Two nodes form a cluster: the coordinator discovers the peer via a seed
 * address, auto-registers it as a remote shard, and cluster_search transparently
 * merges the peer's data with its own - no manual shard_add by the caller. */
static int test_two_node_discovery(void) {
    /* --- Node A: holds the global nearest, serves on an ephemeral port. --- */
    GV_Database *dbA = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(dbA != NULL, "open A db");
    float a0[4] = {0, 0, 0, 0.5f};   /* dist 0.25 -> global nearest */
    float a1[4] = {0, 3, 0, 0};      /* dist 9 */
    ASSERT(db_add_vector(dbA, a0, 4) == 0 && db_add_vector(dbA, a1, 4) == 0, "A adds");

    GV_ClusterConfig cfgA;
    cluster_config_init(&cfgA);
    cfgA.node_id = "A";
    cfgA.listen_address = "127.0.0.1:0";   /* ephemeral */
    GV_Cluster *A = cluster_create(&cfgA);
    ASSERT(A != NULL, "create A");
    ASSERT(cluster_attach_local_db(A, dbA) == 0, "attach A db");
    ASSERT(cluster_start(A) == 0, "start A");
    uint16_t portA = cluster_rpc_port(A);
    ASSERT(portA != 0, "A auto-started an RPC listener");

    /* --- Node B (coordinator): discovers A via seed. --- */
    GV_Database *dbB = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(dbB != NULL, "open B db");
    float b0[4] = {1, 0, 0, 0};      /* dist 1  */
    float b1[4] = {5, 0, 0, 0};      /* dist 25 */
    ASSERT(db_add_vector(dbB, b0, 4) == 0 && db_add_vector(dbB, b1, 4) == 0, "B adds");

    char seed[64];
    snprintf(seed, sizeof(seed), "127.0.0.1:%u", portA);
    GV_ClusterConfig cfgB;
    cluster_config_init(&cfgB);
    cfgB.node_id = "B";
    cfgB.listen_address = "127.0.0.1:0";
    cfgB.seed_nodes = seed;
    GV_Cluster *B = cluster_create(&cfgB);
    ASSERT(B != NULL, "create B");
    ASSERT(cluster_attach_local_db(B, dbB) == 0, "attach B db");
    ASSERT(cluster_start(B) == 0, "start B");

    /* B knows two nodes now (itself + discovered seed A). */
    GV_NodeInfo *nodes = NULL;
    size_t ncount = 0;
    ASSERT(cluster_list_nodes(B, &nodes, &ncount) == 0, "list B nodes");
    ASSERT(ncount == 2, "coordinator discovered the peer via seed");
    cluster_free_node_list(nodes, ncount);

    /* Distributed search over B transparently spans B (local) + A (remote). */
    float q[4] = {0, 0, 0, 0};
    GV_SearchResult res[3];
    int n = cluster_search(B, q, 4, 3, res, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n == 3, "cluster_search returns global top-3 across both nodes");
    ASSERT(res[0].distance <= res[1].distance && res[1].distance <= res[2].distance,
           "globally ordered");
    ASSERT(res[0].vector && fabsf(res[0].vector->data[3] - 0.5f) < 1e-6f,
           "nearest came from discovered peer A");
    ASSERT(res[1].vector && fabsf(res[1].vector->data[0] - 1.0f) < 1e-6f,
           "second came from local node B");
    gv_search_results_free(res, (size_t)n);

    cluster_stop(B);
    cluster_stop(A);
    cluster_destroy(B);
    cluster_destroy(A);
    db_close(dbB);
    db_close(dbA);
    return 0;
}

/* Anti-entropy gossip: C seeds only B, B seeds only A. C must learn A
 * transitively via B's membership list, register it as a shard, and then reach
 * A's data through cluster_search - convergence from partial seeds. */
static int test_gossip_discovery(void) {
    const float target[4] = {9, 0, 0, 0};

    GV_ClusterConfig ca;
    cluster_config_init(&ca);
    ca.node_id = "A";
    ca.listen_address = "127.0.0.1:0";
    ca.heartbeat_interval_ms = 50;
    ca.failure_timeout_ms = 600000;   /* don't reap peers during the test */
    GV_Cluster *A = cluster_create(&ca);
    ASSERT(A != NULL, "create A");
    GV_Database *dbA = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(cluster_attach_local_db(A, dbA) == 0, "attach A db");
    ASSERT(db_add_vector(dbA, target, 4) == 0, "A holds the target vector");
    ASSERT(cluster_start(A) == 0, "start A");
    char addrA[64];
    snprintf(addrA, sizeof(addrA), "127.0.0.1:%u", cluster_rpc_port(A));

    GV_ClusterConfig cb;
    cluster_config_init(&cb);
    cb.node_id = "B";
    cb.listen_address = "127.0.0.1:0";
    cb.seed_nodes = addrA;            /* B knows only A */
    cb.heartbeat_interval_ms = 50;
    cb.failure_timeout_ms = 600000;
    GV_Cluster *B = cluster_create(&cb);
    ASSERT(B != NULL, "create B");
    GV_Database *dbB = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    cluster_attach_local_db(B, dbB);
    ASSERT(cluster_start(B) == 0, "start B");
    char addrB[64];
    snprintf(addrB, sizeof(addrB), "127.0.0.1:%u", cluster_rpc_port(B));

    GV_ClusterConfig cc;
    cluster_config_init(&cc);
    cc.node_id = "C";
    cc.listen_address = "127.0.0.1:0";
    cc.seed_nodes = addrB;            /* C knows only B, NOT A */
    cc.heartbeat_interval_ms = 50;
    cc.failure_timeout_ms = 600000;
    GV_Cluster *C = cluster_create(&cc);
    ASSERT(C != NULL, "create C");
    GV_Database *dbC = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    cluster_attach_local_db(C, dbC);
    ASSERT(cluster_start(C) == 0, "start C");

    /* C should learn A transitively (via B) within a few gossip rounds. */
    int learned = 0;
    for (int i = 0; i < 120 && !learned; i++) {
        usleep(50000);
        learned = addr_in_nodes(C, addrA);
    }
    ASSERT(learned, "C discovered A transitively via B's gossip");

    /* End-to-end: cluster_search on C reaches A's data via the learned shard. */
    int found = 0;
    for (int i = 0; i < 120 && found < 1; i++) {
        GV_SearchResult res[4];
        found = cluster_search(C, target, 4, 1, res, GV_DISTANCE_EUCLIDEAN);
        if (found > 0) {
            gv_search_results_free(res, (size_t)found);
            break;
        }
        usleep(50000);
    }
    ASSERT(found >= 1, "cluster_search(C) reaches A's data via gossip-discovered shard");

    cluster_stop(C);
    cluster_stop(B);
    cluster_stop(A);
    cluster_destroy(C);
    cluster_destroy(B);
    cluster_destroy(A);
    db_close(dbC);
    db_close(dbB);
    db_close(dbA);
    return 0;
}

/* Three nodes run raft over the cluster RPC transport and must converge on
 * exactly one leader that all three agree on. */
static int test_raft_leader_election(void) {
    GV_Cluster *cl[3];
    char addr[3][64];
    const char *ids[3] = { "R0", "R1", "R2" };

    for (int i = 0; i < 3; i++) {
        GV_ClusterConfig cfg;
        cluster_config_init(&cfg);
        cfg.node_id = ids[i];
        cfg.listen_address = "127.0.0.1:0";
        cfg.heartbeat_interval_ms = 1000;
        cfg.failure_timeout_ms = 600000;
        cl[i] = cluster_create(&cfg);
        ASSERT(cl[i] != NULL, "create raft node");
        ASSERT(cluster_start(cl[i]) == 0, "start raft node");
        snprintf(addr[i], sizeof(addr[i]), "127.0.0.1:%u", cluster_rpc_port(cl[i]));
    }

    const char *addrs[3] = { addr[0], addr[1], addr[2] };
    for (int i = 0; i < 3; i++)
        ASSERT(cluster_enable_raft(cl[i], addrs, 3, i) == 0, "enable raft");

    /* Wait for a stable election: exactly one leader, all three in agreement. */
    int leader = -1, converged = 0;
    for (int t = 0; t < 200 && !converged; t++) {
        usleep(50000);
        int l0 = cluster_raft_leader(cl[0]);
        int n_leaders = 0;
        for (int i = 0; i < 3; i++) if (cluster_is_raft_leader(cl[i])) n_leaders++;
        int agree = l0 >= 0 && cluster_raft_leader(cl[1]) == l0 &&
                    cluster_raft_leader(cl[2]) == l0;
        if (n_leaders == 1 && agree) { leader = l0; converged = 1; }
    }
    ASSERT(converged, "cluster converged on exactly one raft leader");
    ASSERT(leader >= 0 && leader < 3, "leader index is valid");
    ASSERT(cluster_is_raft_leader(cl[leader]), "elected node reports leadership");

    for (int i = 0; i < 3; i++) cluster_destroy(cl[i]);
    return 0;
}

/* Raft-replicated cluster op: the leader proposes a shard placement through the
 * raft log; it must commit and apply on every node's state machine. */
static int test_raft_replicated_placement(void) {
    GV_Cluster *cl[3];
    char addr[3][64];
    const char *ids[3] = { "P0", "P1", "P2" };

    for (int i = 0; i < 3; i++) {
        GV_ClusterConfig cfg;
        cluster_config_init(&cfg);
        cfg.node_id = ids[i];
        cfg.listen_address = "127.0.0.1:0";
        cfg.heartbeat_interval_ms = 1000;
        cfg.failure_timeout_ms = 600000;
        cl[i] = cluster_create(&cfg);
        ASSERT(cl[i] != NULL, "create node");
        ASSERT(cluster_start(cl[i]) == 0, "start node");
        snprintf(addr[i], sizeof(addr[i]), "127.0.0.1:%u", cluster_rpc_port(cl[i]));
    }
    const char *addrs[3] = { addr[0], addr[1], addr[2] };
    for (int i = 0; i < 3; i++)
        ASSERT(cluster_enable_raft(cl[i], addrs, 3, i) == 0, "enable raft");

    int leader = -1;
    for (int t = 0; t < 200 && leader < 0; t++) {
        usleep(50000);
        int nl = 0, li = -1;
        for (int i = 0; i < 3; i++) if (cluster_is_raft_leader(cl[i])) { nl++; li = i; }
        if (nl == 1) leader = li;
    }
    ASSERT(leader >= 0, "single leader elected");

    /* Only the leader may propose. */
    ASSERT(cluster_assign_shard(cl[(leader + 1) % 3], 42, 1) == -1,
           "non-leader proposal rejected");
    ASSERT(cluster_assign_shard(cl[leader], 42, 1) == 0, "leader proposal accepted");

    /* The committed placement must apply on all three nodes. */
    int converged = 0;
    for (int t = 0; t < 200 && !converged; t++) {
        usleep(50000);
        converged = cluster_shard_owner(cl[0], 42) == 1 &&
                    cluster_shard_owner(cl[1], 42) == 1 &&
                    cluster_shard_owner(cl[2], 42) == 1;
    }
    ASSERT(converged, "placement replicated + applied on all nodes");

    /* A re-assignment (log upsert) also converges everywhere. */
    ASSERT(cluster_assign_shard(cl[leader], 42, 2) == 0, "reassign accepted");
    int reconverged = 0;
    for (int t = 0; t < 200 && !reconverged; t++) {
        usleep(50000);
        reconverged = cluster_shard_owner(cl[0], 42) == 2 &&
                      cluster_shard_owner(cl[1], 42) == 2 &&
                      cluster_shard_owner(cl[2], 42) == 2;
    }
    ASSERT(reconverged, "reassignment replicated to all nodes");

    for (int i = 0; i < 3; i++) cluster_destroy(cl[i]);
    return 0;
}

/* Failover: kill the elected leader; the surviving majority (2 of 3) must
 * elect a new leader from among themselves. */
static int test_raft_failover(void) {
    GV_Cluster *cl[3];
    char addr[3][64];
    const char *ids[3] = { "F0", "F1", "F2" };

    for (int i = 0; i < 3; i++) {
        GV_ClusterConfig cfg;
        cluster_config_init(&cfg);
        cfg.node_id = ids[i];
        cfg.listen_address = "127.0.0.1:0";
        cfg.heartbeat_interval_ms = 1000;
        cfg.failure_timeout_ms = 600000;
        cl[i] = cluster_create(&cfg);
        ASSERT(cl[i] != NULL, "create node");
        ASSERT(cluster_start(cl[i]) == 0, "start node");
        snprintf(addr[i], sizeof(addr[i]), "127.0.0.1:%u", cluster_rpc_port(cl[i]));
    }
    const char *addrs[3] = { addr[0], addr[1], addr[2] };
    for (int i = 0; i < 3; i++)
        ASSERT(cluster_enable_raft(cl[i], addrs, 3, i) == 0, "enable raft");

    int leader = -1;
    for (int t = 0; t < 200 && leader < 0; t++) {
        usleep(50000);
        int nl = 0, li = -1;
        for (int i = 0; i < 3; i++) if (cluster_is_raft_leader(cl[i])) { nl++; li = i; }
        if (nl == 1) leader = li;
    }
    ASSERT(leader >= 0, "initial leader elected");

    /* Take the leader down. */
    cluster_destroy(cl[leader]);
    cl[leader] = NULL;

    /* The two survivors must converge on a new leader that is NOT the dead one. */
    int new_leader = -1, converged = 0;
    for (int t = 0; t < 300 && !converged; t++) {
        usleep(50000);
        int nl = 0, li = -1;
        for (int i = 0; i < 3; i++) {
            if (cl[i] && cluster_is_raft_leader(cl[i])) { nl++; li = i; }
        }
        /* Both survivors must agree on the leader index. */
        int agree = 1, seen = 0, agreed_idx = -1;
        for (int i = 0; i < 3; i++) {
            if (!cl[i]) continue;
            int l = cluster_raft_leader(cl[i]);
            if (!seen) { agreed_idx = l; seen = 1; }
            else if (l != agreed_idx) agree = 0;
        }
        if (nl == 1 && agree && agreed_idx >= 0) { new_leader = li; converged = 1; }
    }
    ASSERT(converged, "survivors elected a new leader after failover");
    ASSERT(new_leader != leader, "new leader differs from the failed node");
    ASSERT(new_leader >= 0 && cl[new_leader] != NULL, "new leader is a surviving node");

    for (int i = 0; i < 3; i++) if (cl[i]) cluster_destroy(cl[i]);
    return 0;
}

/* Online resharding: the leader rebalances N shards round-robin across the
 * node set and the new placement converges on every node. */
static int test_online_resharding(void) {
    GV_Cluster *cl[3];
    char addr[3][64];
    const char *ids[3] = { "S0", "S1", "S2" };

    for (int i = 0; i < 3; i++) {
        GV_ClusterConfig cfg;
        cluster_config_init(&cfg);
        cfg.node_id = ids[i];
        cfg.listen_address = "127.0.0.1:0";
        cfg.heartbeat_interval_ms = 1000;
        cfg.failure_timeout_ms = 600000;
        cl[i] = cluster_create(&cfg);
        ASSERT(cl[i] != NULL, "create node");
        ASSERT(cluster_start(cl[i]) == 0, "start node");
        snprintf(addr[i], sizeof(addr[i]), "127.0.0.1:%u", cluster_rpc_port(cl[i]));
    }
    const char *addrs[3] = { addr[0], addr[1], addr[2] };
    for (int i = 0; i < 3; i++)
        ASSERT(cluster_enable_raft(cl[i], addrs, 3, i) == 0, "enable raft");

    int leader = -1;
    for (int t = 0; t < 200 && leader < 0; t++) {
        usleep(50000);
        int nl = 0, li = -1;
        for (int i = 0; i < 3; i++) if (cluster_is_raft_leader(cl[i])) { nl++; li = i; }
        if (nl == 1) leader = li;
    }
    ASSERT(leader >= 0, "leader elected");

    /* Non-leader cannot rebalance. */
    ASSERT(cluster_rebalance_shards(cl[(leader + 1) % 3], 6) == -1, "non-leader rebalance rejected");
    /* Leader rebalances 6 shards across 3 nodes -> round-robin 0,1,2,0,1,2. */
    ASSERT(cluster_rebalance_shards(cl[leader], 6) == 0, "leader rebalance accepted");

    int converged = 0;
    for (int t = 0; t < 300 && !converged; t++) {
        usleep(50000);
        int ok = 1;
        for (uint64_t s = 0; s < 6 && ok; s++) {
            int want = (int)(s % 3);
            for (int i = 0; i < 3; i++)
                if (cluster_shard_owner(cl[i], s) != want) { ok = 0; break; }
        }
        converged = ok;
    }
    ASSERT(converged, "round-robin shard placement converged on all nodes");

    for (int i = 0; i < 3; i++) cluster_destroy(cl[i]);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing two_node_discovery...", test_two_node_discovery},
        {"Testing gossip_discovery...", test_gossip_discovery},
        {"Testing raft_leader_election...", test_raft_leader_election},
        {"Testing raft_replicated_placement...", test_raft_replicated_placement},
        {"Testing raft_failover...", test_raft_failover},
        {"Testing online_resharding...", test_online_resharding},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        printf("  %s ", tests[i].name);
        if (tests[i].fn() == 0) { printf("OK\n"); passed++; }
        else printf("FAILED\n");
    }
    printf("\n%d/%d tests passed\n", passed, n);
    return passed == n ? 0 : 1;
}
