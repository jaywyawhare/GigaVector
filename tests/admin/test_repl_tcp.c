#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include "admin/replication.h"
#include "storage/database.h"
#include "../test_tmp.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int pick_port(void) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    bind(fd, (struct sockaddr *)&addr, sizeof(addr));
    socklen_t len = sizeof(addr);
    getsockname(fd, (struct sockaddr *)&addr, &len);
    close(fd);
    return ntohs(addr.sin_port);
}

static int test_tcp_wal_replication(void) {
    char leader_path[512];
    char follower_path[512];
    ASSERT(gv_test_make_temp_path(leader_path, sizeof(leader_path), "gv_tcp_leader", ".gv") == 0,
           "leader temp path");
    ASSERT(gv_test_make_temp_path(follower_path, sizeof(follower_path), "gv_tcp_follower", ".gv") == 0,
           "follower temp path");
    gv_test_remove_db(leader_path);
    gv_test_remove_db(follower_path);

    int port = pick_port();
    char addr[64];
    snprintf(addr, sizeof(addr), "127.0.0.1:%d", port);

    GV_Database *leader_db = db_open(leader_path, 4, GV_INDEX_TYPE_FLAT);
    GV_Database *follower_db = db_open(follower_path, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(leader_db != NULL && follower_db != NULL, "open dbs");

    GV_ReplicationConfig leader_cfg;
    replication_config_init(&leader_cfg);
    leader_cfg.node_id = "leader";
    leader_cfg.listen_address = addr;

    GV_ReplicationConfig follower_cfg;
    replication_config_init(&follower_cfg);
    follower_cfg.node_id = "follower-tcp";
    follower_cfg.leader_address = addr;

    GV_ReplicationManager *leader = replication_create(leader_db, &leader_cfg);
    GV_ReplicationManager *follower = replication_create(follower_db, &follower_cfg);
    ASSERT(leader != NULL && follower != NULL, "create replication");

    replication_add_follower(leader, "follower-tcp", addr);
    replication_start(leader);
    replication_start(follower);
    usleep(500000);

    float vec[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    ASSERT(db_add_vector(leader_db, vec, 4) == 0, "add vector on leader");
    fprintf(stderr, "calling leader_append_wal\n");
    ASSERT(replication_leader_append_wal(leader, 1, 0) == 0, "leader append wal");
    fprintf(stderr, "leader_append_wal returned\n");

    int synced = 0;
    for (int i = 0; i < 50; i++) {
        float q[4] = {1.0f, 0.0f, 0.0f, 0.0f};
        GV_SearchResult out[1];
        if (db_search(follower_db, q, 1, out, GV_DISTANCE_EUCLIDEAN) > 0) {
            synced = 1;
            break;
        }
        usleep(100000);
    }
    ASSERT(synced, "follower received WAL over TCP");

    replication_stop(follower);
    replication_stop(leader);
    replication_destroy(follower);
    replication_destroy(leader);
    db_close(follower_db);
    db_close(leader_db);
    gv_test_remove_db(leader_path);
    gv_test_remove_db(follower_path);
    return 0;
}

int main(void) {
    return test_tcp_wal_replication();
}
