/*
 * test_raft.c — Deterministic multi-node simulation of the Raft core.
 *
 * A virtual network delivers deep-copied messages between nodes with a one-step
 * latency and supports partitions (nodes only talk within their group). Drives
 * raft_tick()/raft_step() to exercise leader election, log replication, and
 * safety + liveness across partitions with an applied-log agreement oracle.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "admin/raft.h"

static int failures = 0;
#define ASSERT(c, m) do { if (!(c)) { printf("FAIL: %s\n", (m)); failures++; } \
                          else { printf("ok: %s\n", (m)); } } while (0)

#define MAXN 7
#define APPLY_CAP 256

/* One queued network message (owns a deep copy of entries). */
typedef struct {
    int to, from;
    GV_RaftMsg msg;
    GV_RaftEntry *entries; /* owned copy for AppendEntries */
} QMsg;

typedef struct {
    GV_Raft *nodes[MAXN];
    int      n;
    int      group[MAXN];      /* partition group id; same group => can communicate */
    QMsg     q[8192];
    size_t   qhead, qtail;
    /* applied log per node: list of first 8 bytes of each applied entry as uint64 */
    uint64_t applied[MAXN][APPLY_CAP];
    size_t   applied_n[MAXN];
} Sim;

static Sim g_sim;

static int node_index_by_id(int id) { return id; } /* ids are 0..n-1 */

static void sim_send(void *ctx, int to, const GV_RaftMsg *msg) {
    (void)ctx;
    if ((g_sim.qtail + 1) % 8192 == g_sim.qhead) return; /* queue full: drop */
    QMsg *qm = &g_sim.q[g_sim.qtail];
    qm->to = to;
    qm->from = msg->from;
    qm->msg = *msg;
    qm->entries = NULL;
    if (msg->type == GV_RAFT_MSG_APPEND_ENTRIES && msg->n_entries > 0) {
        qm->entries = (GV_RaftEntry *)malloc(msg->n_entries * sizeof(GV_RaftEntry));
        for (size_t i = 0; i < msg->n_entries; i++) {
            qm->entries[i].term = msg->entries[i].term;
            qm->entries[i].len  = msg->entries[i].len;
            qm->entries[i].data = NULL;
            if (msg->entries[i].len) {
                qm->entries[i].data = malloc(msg->entries[i].len);
                memcpy(qm->entries[i].data, msg->entries[i].data, msg->entries[i].len);
            }
        }
        qm->msg.entries = qm->entries;
    }
    g_sim.qtail = (g_sim.qtail + 1) % 8192;
}

static void sim_apply(void *ctx, uint64_t index, const void *data, size_t len) {
    (void)index;
    int id = *(int *)ctx;
    if (g_sim.applied_n[id] >= APPLY_CAP) return;
    uint64_t v = 0;
    if (len >= sizeof(uint64_t)) memcpy(&v, data, sizeof(uint64_t));
    g_sim.applied[id][g_sim.applied_n[id]++] = v;
}

static int g_ids[MAXN];

static void qmsg_free(QMsg *qm) {
    if (qm->entries) {
        for (size_t i = 0; i < qm->msg.n_entries; i++) free(qm->entries[i].data);
        free(qm->entries);
        qm->entries = NULL;
    }
}

/* Deliver every message queued at the start of this step (new sends wait). */
static void sim_deliver_round(void) {
    size_t count = (g_sim.qtail + 8192 - g_sim.qhead) % 8192;
    for (size_t k = 0; k < count; k++) {
        QMsg *qm = &g_sim.q[g_sim.qhead];
        g_sim.qhead = (g_sim.qhead + 1) % 8192;
        int fi = node_index_by_id(qm->from), ti = node_index_by_id(qm->to);
        /* Drop across partition boundaries. */
        if (g_sim.group[fi] == g_sim.group[ti]) {
            raft_step(g_sim.nodes[ti], qm->from, &qm->msg);
        }
        qmsg_free(qm);
    }
}

static void sim_run(int steps, uint32_t step_ms) {
    for (int s = 0; s < steps; s++) {
        for (int i = 0; i < g_sim.n; i++) raft_tick(g_sim.nodes[i], step_ms);
        sim_deliver_round();
    }
}

static int count_leaders(void) {
    int c = 0;
    for (int i = 0; i < g_sim.n; i++) if (raft_role(g_sim.nodes[i]) == GV_RAFT_LEADER) c++;
    return c;
}
static int find_leader_in_group(int grp) {
    for (int i = 0; i < g_sim.n; i++)
        if (g_sim.group[i] == grp && raft_role(g_sim.nodes[i]) == GV_RAFT_LEADER) return i;
    return -1;
}

static void sim_init(int n) {
    memset(&g_sim, 0, sizeof(g_sim));
    g_sim.n = n;
    GV_RaftConfig cfg; raft_config_init(&cfg);
    cfg.election_timeout_min_ms = 150; cfg.election_timeout_max_ms = 300; cfg.heartbeat_ms = 50;
    for (int i = 0; i < n; i++) {
        g_ids[i] = i;
        int peers[MAXN]; size_t np = 0;
        for (int j = 0; j < n; j++) if (j != i) peers[np++] = j;
        GV_RaftCallbacks cb; memset(&cb, 0, sizeof(cb));
        cb.send = sim_send; cb.apply = sim_apply; cb.ctx = &g_ids[i];
        g_sim.nodes[i] = raft_create(i, peers, np, &cfg, &cb);
        g_sim.group[i] = 0;
    }
}
static void sim_free(void) { for (int i = 0; i < g_sim.n; i++) raft_destroy(g_sim.nodes[i]); }

static void submit_to_leader(int leader, uint64_t val) {
    raft_submit(g_sim.nodes[leader], &val, sizeof(val), NULL);
}

int main(void) {
    /* ---- 1. Leader election (3 nodes, no partition) ---- */
    sim_init(3);
    sim_run(60, 10);
    ASSERT(count_leaders() == 1, "exactly one leader elected");
    int leader = find_leader_in_group(0);
    ASSERT(leader >= 0, "leader found");
    uint64_t term = raft_current_term(g_sim.nodes[leader]);
    int agree = 1;
    for (int i = 0; i < 3; i++) if (raft_current_term(g_sim.nodes[i]) != term) agree = 0;
    ASSERT(agree, "all nodes on the same term");

    /* ---- 2. Log replication + commit ---- */
    for (uint64_t v = 1; v <= 5; v++) submit_to_leader(leader, v);
    sim_run(60, 10);
    ASSERT(raft_commit_index(g_sim.nodes[leader]) == 5, "leader committed 5 entries");
    int all_committed = 1, logs_match = 1;
    for (int i = 0; i < 3; i++) {
        if (raft_commit_index(g_sim.nodes[i]) != 5) all_committed = 0;
        if (g_sim.applied_n[i] != 5) logs_match = 0;
        else for (size_t k = 0; k < 5; k++) if (g_sim.applied[i][k] != k + 1) logs_match = 0;
    }
    ASSERT(all_committed, "all nodes committed to index 5");
    ASSERT(logs_match, "all nodes applied 1..5 in order (agreement oracle)");
    sim_free();

    /* ---- 3. Partition safety + liveness (5 nodes) ---- */
    sim_init(5);
    sim_run(80, 10);
    ASSERT(count_leaders() == 1, "5-node cluster elects one leader");
    leader = find_leader_in_group(0);
    for (uint64_t v = 1; v <= 3; v++) submit_to_leader(leader, v);
    sim_run(60, 10);
    ASSERT(raft_commit_index(g_sim.nodes[leader]) == 3, "committed 3 before partition");

    /* Partition the leader + one follower into a minority group (2 vs 3). */
    int minority_a = leader;
    int minority_b = (leader + 1) % 5;
    g_sim.group[minority_a] = 1;
    g_sim.group[minority_b] = 1;   /* group 1 has 2 nodes (minority) */
    /* group 0 keeps the other 3 (majority). */

    /* Old leader tries to commit into the minority — must NOT commit. */
    uint64_t before = raft_commit_index(g_sim.nodes[minority_a]);
    submit_to_leader(minority_a, 100);
    sim_run(80, 10);
    ASSERT(raft_commit_index(g_sim.nodes[minority_a]) == before,
           "minority leader cannot advance commit");

    /* Majority elects a new leader at a higher term and commits new writes. */
    int new_leader = -1;
    for (int i = 0; i < 5; i++)
        if (g_sim.group[i] == 0 && raft_role(g_sim.nodes[i]) == GV_RAFT_LEADER) new_leader = i;
    ASSERT(new_leader >= 0, "majority elected a new leader");
    ASSERT(raft_current_term(g_sim.nodes[new_leader]) > term, "new leader has a higher term");
    for (uint64_t v = 10; v <= 12; v++) submit_to_leader(new_leader, v);
    sim_run(80, 10);
    ASSERT(raft_commit_index(g_sim.nodes[new_leader]) > before, "majority leader commits new entries");

    /* ---- 4. Heal: old leader steps down, logs reconcile ---- */
    for (int i = 0; i < 5; i++) g_sim.group[i] = 0;
    sim_run(150, 10);
    ASSERT(count_leaders() == 1, "single leader after heal");
    int final_leader = find_leader_in_group(0);
    uint64_t ci = raft_commit_index(g_sim.nodes[final_leader]);
    int converged = 1;
    for (int i = 0; i < 5; i++) {
        if (raft_commit_index(g_sim.nodes[i]) != ci) converged = 0;
        /* every node's committed prefix must have identical entry terms */
        for (uint64_t idx = 1; idx <= ci; idx++)
            if (raft_log_term_at(g_sim.nodes[i], idx) != raft_log_term_at(g_sim.nodes[final_leader], idx))
                converged = 0;
    }
    ASSERT(converged, "all nodes converge to identical committed log after heal");
    /* The stale write 100 from the deposed leader must never have been committed. */
    int stale_committed = 0;
    for (int i = 0; i < 5; i++)
        for (size_t k = 0; k < g_sim.applied_n[i]; k++)
            if (g_sim.applied[i][k] == 100) stale_committed = 1;
    ASSERT(!stale_committed, "uncommitted minority write never applied anywhere");
    sim_free();

    /* ---- 5. Single-node cluster commits immediately ---- */
    sim_init(1);
    sim_run(30, 10);
    ASSERT(raft_role(g_sim.nodes[0]) == GV_RAFT_LEADER, "single node becomes leader");
    submit_to_leader(0, 7);
    sim_run(2, 10);
    ASSERT(raft_commit_index(g_sim.nodes[0]) == 1, "single node commits immediately");
    sim_free();

    /* drain any remaining queued messages' allocations */
    sim_deliver_round();

    printf(failures ? "\nSOME RAFT TESTS FAILED (%d)\n" : "\nALL RAFT TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
