#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "admin/replication.h"
#include "storage/database.h"
#include "storage/memory_layer.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_replication_config_init(void) {
    GV_ReplicationConfig config;
    memset(&config, 0xFF, sizeof(config));
    replication_config_init(&config);

    ASSERT(config.sync_interval_ms > 0, "default sync_interval_ms should be positive");
    ASSERT(config.election_timeout_ms > 0, "default election_timeout_ms should be positive");
    ASSERT(config.heartbeat_interval_ms > 0, "default heartbeat_interval_ms should be positive");
    ASSERT(config.max_lag_entries > 0, "default max_lag_entries should be positive");
    return 0;
}

static int test_replication_config_init_idempotent(void) {
    GV_ReplicationConfig c1, c2;
    memset(&c1, 0, sizeof(c1));
    memset(&c2, 0, sizeof(c2));
    replication_config_init(&c1);
    replication_config_init(&c2);
    ASSERT(c1.sync_interval_ms == c2.sync_interval_ms, "sync_interval_ms should match");
    ASSERT(c1.election_timeout_ms == c2.election_timeout_ms, "election_timeout_ms should match");
    ASSERT(c1.heartbeat_interval_ms == c2.heartbeat_interval_ms, "heartbeat_interval_ms should match");
    return 0;
}

static int test_replication_create_destroy(void) {
    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "create test database");

    GV_ReplicationConfig config;
    replication_config_init(&config);
    config.node_id = "leader-1";
    config.listen_address = "127.0.0.1:9000";

    GV_ReplicationManager *mgr = replication_create(db, &config);
    ASSERT(mgr != NULL, "replication_create should succeed");

    replication_destroy(mgr);
    db_close(db);

    replication_destroy(NULL);
    return 0;
}

static int test_replication_initial_role(void) {
    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "create test database");

    GV_ReplicationConfig config;
    replication_config_init(&config);
    config.node_id = "node-1";
    config.listen_address = "127.0.0.1:9001";

    GV_ReplicationManager *mgr = replication_create(db, &config);
    ASSERT(mgr != NULL, "replication_create should succeed");

    GV_ReplicationRole role = replication_get_role(mgr);
    ASSERT(role == GV_REPL_LEADER, "initial role should be LEADER (no peers)");

    replication_destroy(mgr);
    db_close(db);
    return 0;
}

static int test_replication_get_stats(void) {
    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "create test database");

    GV_ReplicationConfig config;
    replication_config_init(&config);
    config.node_id = "stats-node";
    config.listen_address = "127.0.0.1:9002";

    GV_ReplicationManager *mgr = replication_create(db, &config);
    ASSERT(mgr != NULL, "replication_create should succeed");

    GV_ReplicationStats stats;
    memset(&stats, 0xFF, sizeof(stats));
    int rc = replication_get_stats(mgr, &stats);
    ASSERT(rc == 0, "get_stats should succeed");
    ASSERT(stats.role == GV_REPL_LEADER, "stats role should be LEADER");
    ASSERT(stats.follower_count == 0, "initial follower_count should be 0");

    replication_free_stats(&stats);
    replication_destroy(mgr);
    db_close(db);
    return 0;
}

static int test_replication_get_stats_null(void) {
    GV_ReplicationStats stats;
    int rc = replication_get_stats(NULL, &stats);
    ASSERT(rc == -1, "get_stats(NULL) should return -1");
    return 0;
}

static int test_replication_add_follower(void) {
    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "create test database");

    GV_ReplicationConfig config;
    replication_config_init(&config);
    config.node_id = "leader-add";
    config.listen_address = "127.0.0.1:9003";

    GV_ReplicationManager *mgr = replication_create(db, &config);
    ASSERT(mgr != NULL, "replication_create should succeed");

    int rc = replication_add_follower(mgr, "follower-1", "192.168.1.10:9000");
    ASSERT(rc == 0, "add_follower should succeed");

    rc = replication_add_follower(mgr, "follower-2", "192.168.1.11:9000");
    ASSERT(rc == 0, "add second follower should succeed");

    GV_ReplicaInfo *replicas = NULL;
    size_t count = 0;
    rc = replication_list_replicas(mgr, &replicas, &count);
    ASSERT(rc == 0, "list_replicas should succeed");
    ASSERT(count >= 2, "should have at least 2 replicas listed");

    replication_free_replicas(replicas, count);
    replication_destroy(mgr);
    db_close(db);
    return 0;
}

static int test_replication_remove_follower(void) {
    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "create test database");

    GV_ReplicationConfig config;
    replication_config_init(&config);
    config.node_id = "leader-rm";
    config.listen_address = "127.0.0.1:9004";

    GV_ReplicationManager *mgr = replication_create(db, &config);
    ASSERT(mgr != NULL, "replication_create should succeed");

    replication_add_follower(mgr, "follower-x", "192.168.1.20:9000");

    int rc = replication_remove_follower(mgr, "follower-x");
    ASSERT(rc == 0, "remove_follower should succeed");

    rc = replication_remove_follower(mgr, "no-such-follower");
    ASSERT(rc == -1, "removing non-existent follower should return -1");

    replication_destroy(mgr);
    db_close(db);
    return 0;
}

static int test_replication_is_healthy(void) {
    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "create test database");

    GV_ReplicationConfig config;
    replication_config_init(&config);
    config.node_id = "health-node";
    config.listen_address = "127.0.0.1:9005";

    GV_ReplicationManager *mgr = replication_create(db, &config);
    ASSERT(mgr != NULL, "replication_create should succeed");

    int healthy = replication_is_healthy(mgr);
    ASSERT(healthy == 1 || healthy == 0, "is_healthy should return 0 or 1");

    ASSERT(replication_is_healthy(NULL) == -1, "is_healthy(NULL) should return -1");

    replication_destroy(mgr);
    db_close(db);
    return 0;
}

static int test_replication_read_policy(void) {
    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "create test database");

    GV_ReplicationConfig config;
    replication_config_init(&config);
    config.node_id = "policy-node";
    config.listen_address = "127.0.0.1:9006";

    GV_ReplicationManager *mgr = replication_create(db, &config);
    ASSERT(mgr != NULL, "replication_create should succeed");

    GV_ReadPolicy policy = replication_get_read_policy(mgr);
    ASSERT(policy == GV_READ_LEADER_ONLY, "default read policy should be LEADER_ONLY");

    int rc = replication_set_read_policy(mgr, GV_READ_ROUND_ROBIN);
    ASSERT(rc == 0, "set_read_policy to ROUND_ROBIN should succeed");
    policy = replication_get_read_policy(mgr);
    ASSERT(policy == GV_READ_ROUND_ROBIN, "policy should be ROUND_ROBIN after set");

    rc = replication_set_read_policy(mgr, GV_READ_LEAST_LAG);
    ASSERT(rc == 0, "set_read_policy to LEAST_LAG should succeed");
    policy = replication_get_read_policy(mgr);
    ASSERT(policy == GV_READ_LEAST_LAG, "policy should be LEAST_LAG after set");

    rc = replication_set_read_policy(mgr, GV_READ_RANDOM);
    ASSERT(rc == 0, "set_read_policy to RANDOM should succeed");
    policy = replication_get_read_policy(mgr);
    ASSERT(policy == GV_READ_RANDOM, "policy should be RANDOM after set");

    replication_destroy(mgr);
    db_close(db);
    return 0;
}

static int test_replication_route_read(void) {
    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "create test database");

    GV_ReplicationConfig config;
    replication_config_init(&config);
    config.node_id = "route-node";
    config.listen_address = "127.0.0.1:9007";

    GV_ReplicationManager *mgr = replication_create(db, &config);
    ASSERT(mgr != NULL, "replication_create should succeed");

    replication_set_read_policy(mgr, GV_READ_LEADER_ONLY);
    GV_Database *routed = replication_route_read(mgr);
    ASSERT(routed != NULL, "route_read should return a valid database");
    ASSERT(routed == db, "route_read with LEADER_ONLY should return leader db");

    ASSERT(replication_route_read(NULL) == NULL, "route_read(NULL) should return NULL");

    replication_destroy(mgr);
    db_close(db);
    return 0;
}

static int test_replication_get_lag(void) {
    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "create test database");

    GV_ReplicationConfig config;
    replication_config_init(&config);
    config.node_id = "lag-node";
    config.listen_address = "127.0.0.1:9008";

    GV_ReplicationManager *mgr = replication_create(db, &config);
    ASSERT(mgr != NULL, "replication_create should succeed");

    int64_t lag = replication_get_lag(mgr);
    ASSERT(lag >= 0, "initial lag should be >= 0 (leader has no lag)");

    ASSERT(replication_get_lag(NULL) == -1, "get_lag(NULL) should return -1");

    replication_destroy(mgr);
    db_close(db);
    return 0;
}

static int test_replication_step_down_and_request(void) {
    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "create test database");

    GV_ReplicationConfig config;
    replication_config_init(&config);
    config.node_id = "stepdown-node";
    config.listen_address = "127.0.0.1:9009";

    GV_ReplicationManager *mgr = replication_create(db, &config);
    ASSERT(mgr != NULL, "replication_create should succeed");

    int rc = replication_step_down(mgr);
    ASSERT(rc == 0, "step_down should succeed");

    rc = replication_request_leadership(mgr);
    ASSERT(rc == 0, "request_leadership should succeed (single node)");

    GV_ReplicationRole role = replication_get_role(mgr);
    ASSERT(role == GV_REPL_LEADER, "should be LEADER after requesting leadership");

    ASSERT(replication_step_down(NULL) == -1, "step_down(NULL) should return -1");
    ASSERT(replication_request_leadership(NULL) == -1,
           "request_leadership(NULL) should return -1");

    replication_destroy(mgr);
    db_close(db);
    return 0;
}

static int test_replication_register_follower_db(void) {
    GV_Database *leader_db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(leader_db != NULL, "create leader database");

    GV_Database *follower_db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(follower_db != NULL, "create follower database");

    GV_ReplicationConfig config;
    replication_config_init(&config);
    config.node_id = "register-leader";
    config.listen_address = "127.0.0.1:9010";

    GV_ReplicationManager *mgr = replication_create(leader_db, &config);
    ASSERT(mgr != NULL, "replication_create should succeed");

    replication_add_follower(mgr, "follower-reg", "192.168.1.30:9000");

    int rc = replication_register_follower_db(mgr, "follower-reg", follower_db);
    ASSERT(rc == 0, "register_follower_db should succeed");

    rc = replication_register_follower_db(mgr, "no-such-follower", follower_db);
    ASSERT(rc == -1, "register_follower_db for unknown follower should fail");

    rc = replication_register_follower_db(NULL, "follower-reg", follower_db);
    ASSERT(rc == -1, "register_follower_db(NULL mgr) should return -1");

    replication_destroy(mgr);
    db_close(leader_db);
    db_close(follower_db);
    return 0;
}

/*
 * Raft election restriction: a candidate whose log is behind a registered
 * follower must NOT win leadership (the follower withholds its vote), whereas an
 * up-to-date candidate with the same follower does win. Guards against electing
 * a stale leader that would truncate already-replicated entries.
 */
static int test_replication_election_restriction(void) {
    /* --- stale candidate is denied --- */
    GV_Database *ldb = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    GV_Database *fdb = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(ldb && fdb, "create databases");

    GV_ReplicationConfig config;
    replication_config_init(&config);
    config.node_id = "stale-cand";
    config.listen_address = "127.0.0.1:9110";

    GV_ReplicationManager *mgr = replication_create(ldb, &config);
    ASSERT(mgr != NULL, "create manager");
    ASSERT(replication_add_follower(mgr, "f1", "192.168.1.40:9000") == 0, "add follower");
    ASSERT(replication_register_follower_db(mgr, "f1", fdb) == 0, "register follower db");

    /* Follower has replicated up to index 10; the candidate's log is empty (0). */
    replication_replica_ack(mgr, "f1", 9); /* sets last_wal_position = 10 */
    int rc = replication_request_leadership(mgr);
    ASSERT(rc == -1, "stale candidate (log behind follower) is denied leadership");
    ASSERT(replication_get_role(mgr) != GV_REPL_LEADER, "stale candidate did not become leader");

    replication_destroy(mgr);
    db_close(ldb);
    db_close(fdb);

    /* --- up-to-date candidate wins with the same follower set --- */
    ldb = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    fdb = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(ldb && fdb, "create databases 2");
    config.node_id = "fresh-cand";
    config.listen_address = "127.0.0.1:9111";
    mgr = replication_create(ldb, &config);
    ASSERT(mgr != NULL, "create manager 2");
    ASSERT(replication_add_follower(mgr, "f1", "192.168.1.41:9000") == 0, "add follower 2");
    ASSERT(replication_register_follower_db(mgr, "f1", fdb) == 0, "register follower db 2");

    /* Follower is not ahead (position 0); candidate log (0) is up-to-date. */
    rc = replication_request_leadership(mgr);
    ASSERT(rc == 0, "up-to-date candidate wins leadership");
    ASSERT(replication_get_role(mgr) == GV_REPL_LEADER, "up-to-date candidate became leader");

    replication_destroy(mgr);
    db_close(ldb);
    db_close(fdb);
    return 0;
}

/*
 * Split-brain guard: a node configured as a remote follower (leader_address
 * set) with no registered in-process voting followers must NOT auto-promote
 * itself to leader via the ad-hoc (non-Raft) election - that was a self-vote
 * "quorum" of 1 and produced split-brain with the real leader. Safe failover
 * for this topology requires replication_enable_raft.
 */
static int test_replication_follower_no_split_brain(void) {
    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "create database");

    GV_ReplicationConfig config;
    replication_config_init(&config);
    config.node_id = "backup-1";
    config.listen_address = "127.0.0.1:9120";
    config.leader_address = "127.0.0.1:9121"; /* this node follows a remote leader */

    GV_ReplicationManager *mgr = replication_create(db, &config);
    ASSERT(mgr != NULL, "create manager");

    /* No registered follower DBs: a lone remote follower must not seize leadership. */
    int rc = replication_request_leadership(mgr);
    ASSERT(rc != 0, "remote follower with no voting peers is denied ad-hoc leadership");
    ASSERT(replication_get_role(mgr) != GV_REPL_LEADER,
           "remote follower did not self-promote (no split-brain)");

    replication_destroy(mgr);
    db_close(db);
    return 0;
}

static int test_replication_route_read_memory(void) {
    GV_Database *leader_db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(leader_db != NULL, "create leader database");

    GV_Database *follower_db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(follower_db != NULL, "create follower database");

    GV_MemoryLayer *leader_mem = memory_layer_create(leader_db, NULL);
    ASSERT(leader_mem != NULL, "create leader memory layer");

    GV_MemoryLayer *follower_mem = memory_layer_create(follower_db, NULL);
    ASSERT(follower_mem != NULL, "create follower memory layer");

    GV_ReplicationConfig config;
    replication_config_init(&config);
    config.node_id = "mem-leader";
    config.listen_address = "127.0.0.1:9012";

    GV_ReplicationManager *mgr = replication_create(leader_db, &config);
    ASSERT(mgr != NULL, "replication_create should succeed");

    replication_add_follower(mgr, "follower-mem", "127.0.0.1:9101");
    ASSERT(replication_register_follower_db(mgr, "follower-mem", follower_db) == 0,
           "register follower db");
    ASSERT(replication_register_follower_memory(mgr, "follower-mem", follower_mem) == 0,
           "register follower memory");

    replication_set_read_policy(mgr, GV_READ_ROUND_ROBIN);
    ASSERT(replication_route_read(mgr) == follower_db,
           "round robin routes db to follower");
    ASSERT(replication_route_read_memory(mgr) == follower_mem,
           "round robin routes memory to follower");

    replication_set_read_policy(mgr, GV_READ_LEADER_ONLY);
    ASSERT(replication_route_read_memory(mgr) == NULL,
           "leader-only returns NULL memory route");

    memory_layer_destroy(follower_mem);
    memory_layer_destroy(leader_mem);
    replication_destroy(mgr);
    db_close(follower_db);
    db_close(leader_db);
    return 0;
}

static int test_replication_set_max_read_lag(void) {
    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "create test database");

    GV_ReplicationConfig config;
    replication_config_init(&config);
    config.node_id = "maxlag-node";
    config.listen_address = "127.0.0.1:9011";

    GV_ReplicationManager *mgr = replication_create(db, &config);
    ASSERT(mgr != NULL, "replication_create should succeed");

    int rc = replication_set_max_read_lag(mgr, 1000);
    ASSERT(rc == 0, "set_max_read_lag should succeed");

    rc = replication_set_max_read_lag(mgr, 0);
    ASSERT(rc == 0, "set_max_read_lag to 0 should succeed");

    ASSERT(replication_set_max_read_lag(NULL, 100) == -1,
           "set_max_read_lag(NULL) should return -1");

    replication_destroy(mgr);
    db_close(db);
    return 0;
}

static int test_replication_leader_append_and_sync(void) {
    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "create db for wal test");

    GV_Database *follower_db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(follower_db != NULL, "create follower db for wal test");

    GV_ReplicationConfig config;
    replication_config_init(&config);
    config.node_id = "wal-node";

    GV_ReplicationManager *mgr = replication_create(db, &config);
    ASSERT(mgr != NULL, "replication_create for wal test");

    ASSERT(replication_leader_append_wal(NULL, 1, 0) == -1,
           "leader_append_wal(NULL) should fail");

    ASSERT(replication_add_follower(mgr, "f-wal", "127.0.0.1:9100") == 0,
           "add_follower for wal test");
    ASSERT(replication_register_follower_db(mgr, "f-wal", follower_db) == 0,
           "register_follower_db for wal test");

    ASSERT(replication_leader_append_wal(mgr, 3, 64) == 0, "leader_append_wal");
    ASSERT(replication_sync_commit(mgr, 2000) == 0, "sync_commit after append");

    ASSERT(replication_step_down(mgr) == 0, "step_down for negative append test");
    ASSERT(replication_leader_append_wal(mgr, 1, 0) == -1,
           "leader_append_wal as non-leader should fail");

    replication_destroy(mgr);
    db_close(db);
    db_close(follower_db);
    return 0;
}

static int test_replication_free_replicas_null(void) {
    replication_free_replicas(NULL, 0);
    return 0;
}

static int test_replication_role_enum_values(void) {
    ASSERT(GV_REPL_LEADER == 0, "GV_REPL_LEADER should be 0");
    ASSERT(GV_REPL_FOLLOWER == 1, "GV_REPL_FOLLOWER should be 1");
    ASSERT(GV_REPL_CANDIDATE == 2, "GV_REPL_CANDIDATE should be 2");
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing replication_config_init...", test_replication_config_init},
        {"Testing replication_config_init_idempotent...", test_replication_config_init_idempotent},
        {"Testing replication_create_destroy...", test_replication_create_destroy},
        {"Testing replication_initial_role...", test_replication_initial_role},
        {"Testing replication_get_stats...", test_replication_get_stats},
        {"Testing replication_get_stats_null...", test_replication_get_stats_null},
        {"Testing replication_add_follower...", test_replication_add_follower},
        {"Testing replication_remove_follower...", test_replication_remove_follower},
        {"Testing replication_is_healthy...", test_replication_is_healthy},
        {"Testing replication_read_policy...", test_replication_read_policy},
        {"Testing replication_route_read...", test_replication_route_read},
        {"Testing replication_get_lag...", test_replication_get_lag},
        {"Testing replication_step_down_and_request...", test_replication_step_down_and_request},
        {"Testing replication_register_follower_db...", test_replication_register_follower_db},
        {"Testing replication_election_restriction...", test_replication_election_restriction},
        {"Testing replication_follower_no_split_brain...", test_replication_follower_no_split_brain},
        {"Testing replication_route_read_memory...", test_replication_route_read_memory},
        {"Testing replication_set_max_read_lag...", test_replication_set_max_read_lag},
        {"Testing replication_leader_append_and_sync...", test_replication_leader_append_and_sync},
        {"Testing replication_free_replicas_null...", test_replication_free_replicas_null},
        {"Testing replication_role_enum_values...", test_replication_role_enum_values},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        printf("  %s ", tests[i].name);
        if (tests[i].fn() == 0) {
            printf("OK\n");
            passed++;
        } else {
            printf("FAILED\n");
        }
    }
    printf("\n%d/%d tests passed\n", passed, n);
    return passed == n ? 0 : 1;
}
