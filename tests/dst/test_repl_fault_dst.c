/**
 * @file test_repl_fault_dst.c
 * @brief DST: replication with simulated packet drop/reorder on WAL delivery.
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
#define FOLLOWER_ID "dst-follower"

static int leader_search_count(GV_Database *db, const float *query) {
    GV_SearchResult out[8];
    int n = db_search(db, query, 8, out, GV_DISTANCE_EUCLIDEAN);
    if (n > 0) gv_search_results_free(out, n);
    return n;
}

static int test_repl_fault_seeded(void) {
    uint64_t seed = gv_dst_seed_from_env();
    size_t iters = gv_dst_iters_from_env(60);
    GV_DstRng rng = gv_dst_rng_seed(seed);

    gv_sim_time_set_mode(GV_TIME_SIM);
    gv_sim_time_reset(1700000000ULL);

    fprintf(stderr, "DST repl fault: seed=%llu iters=%zu\n",
            (unsigned long long)seed, iters);

    char leader_path[256];
    char follower_path[256];
    if (gv_test_make_temp_path(leader_path, sizeof(leader_path), "gv_fault_leader", ".gv") != 0) return 0;
    if (gv_test_make_temp_path(follower_path, sizeof(follower_path), "gv_fault_follower", ".gv") != 0) return 0;
    remove(leader_path);
    remove(follower_path);

    GV_Database *leader = db_open(leader_path, DIM, GV_INDEX_TYPE_FLAT);
    GV_Database *follower = db_open(follower_path, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(leader && follower, "open dbs");

    GV_ReplicationConfig cfg;
    replication_config_init(&cfg);
    cfg.node_id = "dst-leader";

    GV_ReplicationManager *mgr = replication_create(leader, &cfg);
    ASSERT(mgr != NULL, "replication_create");
    ASSERT(replication_set_dst_simulation_mode(mgr, 1) == 0, "sim mode");
    ASSERT(replication_add_follower(mgr, FOLLOWER_ID, "127.0.0.1:1") == 0, "add_follower");
    ASSERT(replication_register_follower_db(mgr, FOLLOWER_ID, follower) == 0, "register db");

    GV_ReplSim *sim = repl_sim_create(seed);
    ASSERT(sim != NULL, "repl_sim_create");

    GV_ReplSimFaultConfig faults = {
        .drop_permille = (uint32_t)(gv_dst_rng_u32(&rng, 300) + 50),
        .partitioned = 0,
        .reorder = (int)(gv_dst_rng_u32(&rng, 2))
    };
    repl_sim_set_faults(sim, &faults);

    for (size_t i = 0; i < iters; i++) {
        float vec[DIM];
        for (size_t d = 0; d < DIM; d++) {
            vec[d] = gv_dst_rng_float(&rng);
        }

        ASSERT(db_add_vector(leader, vec, DIM) == 0, "add_vector");
        ASSERT(replication_leader_append_wal(mgr, 1, 0) == 0, "append_wal");

        if (gv_dst_rng_u32(&rng, 4) == 0) {
            repl_sim_heal(sim);
        }

        (void)gv_dst_sim_enqueue_latest(sim, leader, FOLLOWER_ID);

        while (repl_sim_pending_count(sim, FOLLOWER_ID) > 0) {
            if (gv_dst_sim_deliver_latest(sim, mgr, FOLLOWER_ID, follower) != 0) {
                break;
            }
        }

        repl_sim_flush_follower(sim, FOLLOWER_ID);
        while (gv_dst_sim_deliver_latest(sim, mgr, FOLLOWER_ID, follower) == 0) {
        }

        if (replication_sync_commit(mgr, 5000) != 0) {
            ASSERT(gv_dst_catch_up_follower(mgr, leader, follower, FOLLOWER_ID) == 0,
                   "catch-up heal after faults");
            ASSERT(replication_sync_commit(mgr, 5000) == 0, "sync_commit after heal");
        }

        int leader_hits = leader_search_count(leader, vec);
        int follower_hits = leader_search_count(follower, vec);
        ASSERT(leader_hits > 0, "leader search");
        ASSERT(follower_hits == leader_hits, "follower matches leader");
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
    return test_repl_fault_seeded();
}
