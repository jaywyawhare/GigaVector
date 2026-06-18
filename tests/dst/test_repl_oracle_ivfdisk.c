/**
 * @file test_repl_oracle_ivfdisk.c
 * @brief Replication oracle for IVFDisk: leader WAL -> follower apply.
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
            fprintf(stderr, "FAIL [ivfdisk repl oracle]: %s\n", msg); \
            return -1; \
        } \
    } while (0)

#define DIM 4
#define TRAIN 32

static int leader_search_count(GV_Database *db, const float *query) {
    GV_SearchResult out[8];
    return db_search(db, query, 8, out, GV_DISTANCE_EUCLIDEAN);
}

static int apply_new_wal_records(GV_Database *leader, GV_Database *follower,
                                 uint64_t *applied_count) {
    const char *path = db_wal_path(leader);
    if (!path) return -1;

    uint64_t total = wal_count_entries(path);
    for (uint64_t i = *applied_count; i < total; ++i) {
        uint8_t type = 0;
        uint8_t *record = NULL;
        size_t record_len = 0;
        if (wal_read_entry_at(path, i, &type, &record, &record_len) != 0) {
            gv_free(record);
            return -1;
        }
        if (db_apply_wal_record(follower, record, record_len) != 0) {
            gv_free(record);
            return -1;
        }
        gv_free(record);
    }
    *applied_count = total;
    return 0;
}

static void fill_train(float *train, size_t count, size_t dim) {
    for (size_t i = 0; i < count; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            train[i * dim + d] = (float)(i + d) / (float)count;
        }
    }
}

static int test_repl_oracle_ivfdisk_seeded(void) {
    char leader_path[256];
    char follower_path[256];
    if (gv_test_make_temp_path(leader_path, sizeof(leader_path), "gv_dst_ivf_leader", ".gv") != 0) return 0;
    if (gv_test_make_temp_path(follower_path, sizeof(follower_path), "gv_dst_ivf_follower", ".gv") != 0) return 0;
    remove(leader_path);
    remove(follower_path);

    char leader_wal[512];
    char follower_wal[512];
    snprintf(leader_wal, sizeof(leader_wal), "%s.wal", leader_path);
    snprintf(follower_wal, sizeof(follower_wal), "%s.wal", follower_path);
    remove(leader_wal);
    remove(follower_wal);

    GV_IVFDiskConfig cfg;
    ivfdisk_config_init(&cfg);
    cfg.nlist = 4;
    cfg.nprobe = 2;

    GV_Database *leader = db_open_with_ivfdisk_config(leader_path, DIM, GV_INDEX_TYPE_IVFDISK, &cfg);
    GV_Database *follower = db_open_with_ivfdisk_config(follower_path, DIM, GV_INDEX_TYPE_IVFDISK, &cfg);
    ASSERT(leader != NULL && follower != NULL, "open leader/follower ivfdisk dbs");

    float train[TRAIN * DIM];
    fill_train(train, TRAIN, DIM);
    ASSERT(db_ivfdisk_train(leader, train, TRAIN, DIM) == 0, "leader train");
    ASSERT(db_ivfdisk_train(follower, train, TRAIN, DIM) == 0, "follower train");
    ASSERT(db_set_wal(leader, leader_wal) == 0, "leader wal");

    uint64_t seed = gv_dst_seed_from_env();
    size_t iters = gv_dst_iters_from_env(40);
    GV_DstRng rng = gv_dst_rng_seed(seed);

    GV_ReplicationConfig rcfg;
    replication_config_init(&rcfg);
    rcfg.node_id = "dst-ivf-leader";

    GV_ReplicationManager *mgr = replication_create(leader, &rcfg);
    ASSERT(mgr != NULL, "replication_create");
    ASSERT(replication_add_follower(mgr, "dst-ivf-follower", "127.0.0.1:1") == 0, "add_follower");
    ASSERT(replication_register_follower_db(mgr, "dst-ivf-follower", follower) == 0, "register_follower_db");

    uint64_t applied = 0;
    for (size_t i = 0; i < iters; i++) {
        float vec[DIM];
        for (size_t d = 0; d < DIM; d++) {
            vec[d] = gv_dst_rng_float(&rng);
        }

        ASSERT(db_add_vector(leader, vec, DIM) == 0, "leader add_vector");
        uint64_t total = wal_count_entries(db_wal_path(leader));
        uint64_t delta = total - applied;
        ASSERT(replication_leader_append_wal(mgr, delta, 0) == 0, "leader_append_wal");
        ASSERT(apply_new_wal_records(leader, follower, &applied) == 0, "apply wal to follower");
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
    return test_repl_oracle_ivfdisk_seeded();
}
