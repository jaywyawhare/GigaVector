/**
 * @file test_repl_oracle.c
 * @brief Deterministic simulation test: leader WAL -> follower apply oracle.
 *
 * Simulates replication without TCP: after each leader write we read the WAL
 * record and apply it to the follower DB, then verify search agreement.
 *
 * Replay a failure: GV_DST_SEED=<seed> make c-test-single TEST=dst/test_repl_oracle
 */

#include <stdio.h>
#include "core/memory.h"
#include <stdlib.h>
#include <string.h>

#include "admin/replication.h"
#include "storage/database.h"
#include "storage/wal.h"
#include "../test_tmp.h"
#include "dst_harness.h"

#define ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL [seed workflow]: %s\n", msg); \
            return -1; \
        } \
    } while (0)

#define DIM 4

static int leader_search_count(GV_Database *db, const float *query) {
    GV_SearchResult out[8];
    return db_search(db, query, 8, out, GV_DISTANCE_EUCLIDEAN);
}

static int apply_latest_wal_to_follower(GV_Database *leader, GV_Database *follower) {
    const char *path = db_wal_path(leader);
    if (!path) return -1;

    uint64_t total = wal_count_entries(path);
    if (total == 0) return -1;

    uint8_t type = 0;
    uint8_t *record = NULL;
    size_t record_len = 0;
    if (wal_read_entry_at(path, total - 1, &type, &record, &record_len) != 0) {
        gv_free(record);
        return -1;
    }

    int rc = db_apply_wal_record(follower, record, record_len);
    gv_free(record);
    return rc;
}

static int test_repl_oracle_seeded(void) {
    char leader_path[256];
    char follower_path[256];
    if (gv_test_make_temp_path(leader_path, sizeof(leader_path), "gv_dst_leader", ".gv") != 0) return 0;
    if (gv_test_make_temp_path(follower_path, sizeof(follower_path), "gv_dst_follower", ".gv") != 0) return 0;
    remove(leader_path);
    remove(follower_path);

    char leader_wal[512];
    char follower_wal[512];
    snprintf(leader_wal, sizeof(leader_wal), "%s.wal", leader_path);
    snprintf(follower_wal, sizeof(follower_wal), "%s.wal", follower_path);
    remove(leader_wal);
    remove(follower_wal);

    uint64_t seed = gv_dst_seed_from_env();
    size_t iters = gv_dst_iters_from_env(80);
    GV_DstRng rng = gv_dst_rng_seed(seed);

    fprintf(stderr, "DST repl oracle: seed=%llu iters=%zu\n",
            (unsigned long long)seed, iters);

    GV_Database *leader = db_open(leader_path, DIM, GV_INDEX_TYPE_FLAT);
    GV_Database *follower = db_open(follower_path, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(leader != NULL && follower != NULL, "open leader/follower file dbs");

    GV_ReplicationConfig cfg;
    replication_config_init(&cfg);
    cfg.node_id = "dst-leader";

    GV_ReplicationManager *mgr = replication_create(leader, &cfg);
    ASSERT(mgr != NULL, "replication_create");
    ASSERT(replication_add_follower(mgr, "dst-follower", "127.0.0.1:1") == 0, "add_follower");
    ASSERT(replication_register_follower_db(mgr, "dst-follower", follower) == 0, "register_follower_db");

    for (size_t i = 0; i < iters; i++) {
        float vec[DIM];
        for (size_t d = 0; d < DIM; d++) {
            vec[d] = gv_dst_rng_float(&rng);
        }

        ASSERT(db_add_vector(leader, vec, DIM) == 0, "leader add_vector");
        ASSERT(replication_leader_append_wal(mgr, 1, 0) == 0, "leader_append_wal");
        ASSERT(apply_latest_wal_to_follower(leader, follower) == 0, "apply wal to follower");
        ASSERT(replication_sync_commit(mgr, 500) == 0, "sync_commit");

        int leader_hits = leader_search_count(leader, vec);
        int follower_hits = leader_search_count(follower, vec);
        ASSERT(leader_hits > 0, "leader search finds inserted vector");
        ASSERT(follower_hits == leader_hits, "follower matches leader search count");
    }

    replication_destroy(mgr);
    db_close(leader);
    db_close(follower);
    remove(leader_path);
    remove(follower_path);
    remove(leader_wal);
    remove(follower_wal);
    return 0;
}

int main(void) {
    return test_repl_oracle_seeded();
}
