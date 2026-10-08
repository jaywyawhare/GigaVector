/*
 * test_repl_raft.c - the Raft consensus core driving the replication layer.
 *
 * Three replication managers each own a raft node; messages are delivered
 * in-process and applied on replication_raft_tick(). Verifies a single leader
 * is elected and a submitted command commits across the cluster.
 */
#include <stdio.h>
#include <string.h>
#include "admin/replication.h"
#include "storage/database.h"

static int failures = 0;
#define ASSERT(c, m) do { if (!(c)) { printf("FAIL: %s\n", (m)); failures++; } \
                          else { printf("ok: %s\n", (m)); } } while (0)

#define N 3

static void run(GV_ReplicationManager **mgr, int steps, unsigned ms) {
    for (int s = 0; s < steps; s++)
        for (int i = 0; i < N; i++)
            replication_raft_tick(mgr[i], ms);
}

static int count_leaders(GV_ReplicationManager **mgr) {
    int c = 0;
    for (int i = 0; i < N; i++)
        if (replication_get_role(mgr[i]) == GV_REPL_LEADER) c++;
    return c;
}

int main(void) {
    GV_Database *db[N];
    GV_ReplicationManager *mgr[N];
    const char *ids[N] = { "n0", "n1", "n2" };

    for (int i = 0; i < N; i++) {
        db[i] = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
        GV_ReplicationConfig cfg; replication_config_init(&cfg);
        cfg.node_id = ids[i];
        cfg.listen_address = "127.0.0.1:0";
        mgr[i] = replication_create(db[i], &cfg);
        ASSERT(mgr[i] != NULL, "create manager");
    }

    for (int i = 0; i < N; i++) {
        int peers[N - 1]; size_t np = 0;
        for (int j = 0; j < N; j++) if (j != i) peers[np++] = j;
        ASSERT(replication_enable_raft(mgr[i], i, peers, np) == 0, "enable raft");
    }
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (j != i) replication_register_raft_peer(mgr[i], j, mgr[j]);

    /* Election. */
    run(mgr, 60, 10);
    ASSERT(count_leaders(mgr) == 1, "exactly one raft leader elected");

    int leader = -1;
    for (int i = 0; i < N; i++)
        if (replication_get_role(mgr[i]) == GV_REPL_LEADER) leader = i;
    ASSERT(leader >= 0, "leader found");

    /* Replicate a command and let it commit across the cluster. */
    uint64_t payload = 0x1234;
    ASSERT(replication_raft_submit(mgr[leader], &payload, sizeof(payload)) == 0, "submit on leader");
    ASSERT(replication_raft_submit(mgr[(leader + 1) % N], &payload, sizeof(payload)) == -1,
           "submit on follower rejected");
    run(mgr, 40, 10);

    int all_committed = 1;
    for (int i = 0; i < N; i++) {
        uint64_t wal = 0, commit = 0;
        replication_get_positions(mgr[i], &wal, &commit);
        if (commit < 1) all_committed = 0;
    }
    ASSERT(all_committed, "committed entry replicated to all nodes");

    for (int i = 0; i < N; i++) { replication_destroy(mgr[i]); db_close(db[i]); }

    printf(failures ? "\nSOME REPL-RAFT TESTS FAILED (%d)\n" : "\nALL REPL-RAFT TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
