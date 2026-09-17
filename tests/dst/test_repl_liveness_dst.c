/**
 * @file test_repl_liveness_dst.c
 * @brief DST liveness: partition blocks progress; heal restores quorum and commit.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "admin/replication.h"
#include "admin/repl_sim.h"
#include "core/sim_time.h"
#include "storage/database.h"
#include "../test_tmp.h"
#include "dst_harness.h"
#include "dst_repl_helpers.h"

#define ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1; \
        } \
    } while (0)

#define DIM 4
#define FOLLOWER_ID "liveness-follower"

static int test_repl_liveness_seeded(void) {
    uint64_t seed = gv_dst_seed_from_env();
    size_t rounds = gv_dst_iters_from_env(20);
    GV_DstRng rng = gv_dst_rng_seed(seed);

    gv_sim_time_set_mode(GV_TIME_SIM);
    gv_sim_time_reset(1700001000ULL);

    fprintf(stderr, "DST repl liveness: seed=%llu rounds=%zu\n",
            (unsigned long long)seed, rounds);

    char leader_path[256];
    char follower_path[256];
    if (gv_test_make_temp_path(leader_path, sizeof(leader_path), "gv_live_leader", ".gv") != 0) return 0;
    if (gv_test_make_temp_path(follower_path, sizeof(follower_path), "gv_live_follower", ".gv") != 0) return 0;
    remove(leader_path);
    remove(follower_path);

    GV_Database *leader = db_open(leader_path, DIM, GV_INDEX_TYPE_FLAT);
    GV_Database *follower = db_open(follower_path, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(leader && follower, "open dbs");

    GV_ReplicationConfig cfg;
    replication_config_init(&cfg);
    cfg.node_id = "live-leader";

    GV_ReplicationManager *mgr = replication_create(leader, &cfg);
    ASSERT(mgr != NULL, "replication_create");
    ASSERT(replication_set_dst_simulation_mode(mgr, 1) == 0, "sim mode");
    ASSERT(replication_add_follower(mgr, FOLLOWER_ID, "127.0.0.1:1") == 0, "add_follower");
    ASSERT(replication_register_follower_db(mgr, FOLLOWER_ID, follower) == 0, "register db");

    GV_ReplSim *sim = repl_sim_create(seed ^ 0xDEADBEEFULL);
    ASSERT(sim != NULL, "repl_sim_create");

    GV_ReplSimFaultConfig partition = { .drop_permille = 0, .partitioned = 1, .reorder = 0 };
    repl_sim_set_faults(sim, &partition);

    for (size_t r = 0; r < rounds; r++) {
        float vec[DIM];
        for (size_t d = 0; d < DIM; d++) {
            vec[d] = gv_dst_rng_float(&rng);
        }

        ASSERT(db_add_vector(leader, vec, DIM) == 0, "add_vector");
        ASSERT(replication_leader_append_wal(mgr, 1, 0) == 0, "append_wal");
        ASSERT(gv_dst_sim_enqueue_latest(sim, leader, FOLLOWER_ID) == 0, "enqueue while partitioned");

        ASSERT(replication_sync_commit(mgr, 50) != 0, "sync blocked during partition");
        ASSERT(repl_sim_pending_count(sim, FOLLOWER_ID) >= 1, "message still pending");

        repl_sim_heal(sim);
        ASSERT(gv_dst_sim_deliver_latest(sim, mgr, FOLLOWER_ID, follower) == 0, "deliver after heal");
        ASSERT(replication_sync_commit(mgr, 5000) == 0, "sync succeeds after heal");

        GV_SearchResult out[4];
        int lh = db_search(leader, vec, 4, out, GV_DISTANCE_EUCLIDEAN);
        if (lh > 0) gv_search_results_free(out, lh);
        int fh = db_search(follower, vec, 4, out, GV_DISTANCE_EUCLIDEAN);
        if (fh > 0) gv_search_results_free(out, fh);
        ASSERT(lh > 0 && fh == lh, "search agreement after liveness heal");

        repl_sim_set_faults(sim, &partition);
    }

    repl_sim_destroy(sim);
    replication_destroy(mgr);
    db_close(leader);
    db_close(follower);
    remove(leader_path);
    remove(follower_path);
    gv_sim_time_set_mode(GV_TIME_WALL);
    return 0;
}

int main(void) {
    return test_repl_liveness_seeded();
}
