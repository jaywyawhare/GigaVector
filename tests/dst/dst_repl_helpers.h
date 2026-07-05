#ifndef GV_DST_REPL_HELPERS_H
#define GV_DST_REPL_HELPERS_H

#include <stddef.h>
#include "core/memory.h"
#include <stdint.h>

#include "admin/replication.h"
#include "admin/repl_sim.h"
#include "storage/database.h"
#include "storage/wal.h"

static inline int gv_dst_read_latest_wal(const GV_Database *leader,
                                         uint64_t *entry_index,
                                         uint8_t **record, size_t *record_len) {
    const char *path = db_wal_path(leader);
    if (!path) return -1;

    uint64_t total = wal_count_entries(path);
    if (total == 0) return -1;

    uint8_t type = 0;
    if (wal_read_entry_at(path, total - 1, &type, record, record_len) != 0) {
        gv_free(*record);
        *record = NULL;
        return -1;
    }
    if (entry_index) *entry_index = total - 1;
    return 0;
}

static inline int gv_dst_sim_deliver_latest(GV_ReplSim *sim, GV_ReplicationManager *mgr,
                                            const char *follower_id,
                                            GV_Database *follower_db) {
    uint64_t entry_index = 0;
    uint8_t *record = NULL;
    size_t record_len = 0;
    if (repl_sim_deliver_wal(sim, follower_id, &entry_index, &record, &record_len) != 0) {
        return -1;
    }
    int rc = db_apply_wal_record(follower_db, record, record_len);
    gv_free(record);
    if (rc != 0) return -1;
    return replication_replica_ack(mgr, follower_id, entry_index);
}

static inline int gv_dst_sim_enqueue_latest(GV_ReplSim *sim, const GV_Database *leader,
                                            const char *follower_id) {
    uint64_t entry_index = 0;
    uint8_t *record = NULL;
    size_t record_len = 0;
    if (gv_dst_read_latest_wal(leader, &entry_index, &record, &record_len) != 0) {
        return -1;
    }
    int rc = repl_sim_enqueue_wal(sim, follower_id, entry_index, record, record_len);
    gv_free(record);
    return rc;
}

static inline int gv_dst_catch_up_follower(GV_ReplicationManager *mgr,
                                            const GV_Database *leader,
                                            GV_Database *follower,
                                            const char *follower_id) {
    const char *leader_path = db_wal_path(leader);
    const char *follower_path = db_wal_path(follower);
    if (!leader_path || !follower_path || !mgr || !follower || !follower_id) return -1;

    uint64_t leader_total = wal_count_entries(leader_path);
    uint64_t follower_total = wal_count_entries(follower_path);

    for (uint64_t idx = follower_total; idx < leader_total; idx++) {
        uint8_t type = 0;
        uint8_t *record = NULL;
        size_t record_len = 0;
        if (wal_read_entry_at(leader_path, idx, &type, &record, &record_len) != 0) {
            gv_free(record);
            return -1;
        }
        if (db_apply_wal_record(follower, record, record_len) != 0) {
            gv_free(record);
            return -1;
        }
        gv_free(record);
        if (replication_replica_ack(mgr, follower_id, idx) != 0) {
            return -1;
        }
    }
    return 0;
}

#endif
