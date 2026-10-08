#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "admin/shard.h"
#include "admin/shard_rpc.h"
#include "storage/database.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

/* Build a 3-shard manager with six vectors so the true nearest neighbours are
 * spread across shards, exactly as in the in-process shard_search test. */
static GV_ShardManager *make_cluster(GV_Database *db[3]) {
    GV_ShardManager *mgr = shard_manager_create(NULL);
    if (!mgr) return NULL;
    float v[6][4] = {
        {1, 0, 0, 0}, {5, 0, 0, 0},      /* shard 0 */
        {0, 2, 0, 0}, {0, 6, 0, 0},      /* shard 1 */
        {0, 0, 3, 0}, {0, 0, 0, 0.5f},   /* shard 2, last is global nearest */
    };
    for (int i = 0; i < 3; i++) {
        shard_add(mgr, (uint32_t)i, "node:6000");
        db[i] = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
        shard_attach_local(mgr, (uint32_t)i, db[i]);
        int r0 = db_add_vector(db[i], v[i * 2], 4);
        int r1 = db_add_vector(db[i], v[i * 2 + 1], 4);
        (void)r0; (void)r1;
    }
    return mgr;
}

/* A remote search over TCP must return the same global top-k as an in-process
 * shard_search against the same shards. */
static int test_rpc_matches_local(void) {
    GV_Database *db[3];
    GV_ShardManager *mgr = make_cluster(db);
    ASSERT(mgr != NULL, "create cluster");

    GV_ShardRpcServer *srv = shard_rpc_serve(mgr, "127.0.0.1", 0);
    ASSERT(srv != NULL, "start rpc server");
    uint16_t port = shard_rpc_server_port(srv);
    ASSERT(port != 0, "server bound a port");

    float q[4] = {0, 0, 0, 0};

    GV_SearchResult local[3];
    int ln = shard_search(mgr, q, 3, local, GV_DISTANCE_EUCLIDEAN);
    ASSERT(ln == 3, "local search returns 3");

    GV_SearchResult remote[3];
    int rn = shard_rpc_search("127.0.0.1", port, q, 4, 3, GV_DISTANCE_EUCLIDEAN, remote);
    ASSERT(rn == 3, "remote search returns 3");

    for (int i = 0; i < 3; i++) {
        ASSERT(remote[i].id == local[i].id, "remote id matches local");
        ASSERT(fabsf(remote[i].distance - local[i].distance) < 1e-6f,
               "remote distance matches local");
        ASSERT(remote[i].vector != NULL, "remote result carries a vector");
    }
    /* Global nearest is v2b (component 3 == 0.5) - proves cross-shard merge. */
    ASSERT(fabsf(remote[0].vector->data[3] - 0.5f) < 1e-6f, "remote nearest is v2b");

    gv_search_results_free(local, (size_t)ln);
    gv_search_results_free(remote, (size_t)rn);

    /* k larger than available returns all six. */
    GV_SearchResult all[16];
    int an = shard_rpc_search("127.0.0.1", port, q, 4, 16, GV_DISTANCE_EUCLIDEAN, all);
    ASSERT(an == 6, "remote k>total returns all six");
    gv_search_results_free(all, (size_t)an);

    shard_rpc_server_stop(srv);
    shard_manager_destroy(mgr);
    for (int i = 0; i < 3; i++) db_close(db[i]);
    return 0;
}

static int test_rpc_errors(void) {
    float q[4] = {0, 0, 0, 0};
    GV_SearchResult out[1];
    ASSERT(shard_rpc_search("127.0.0.1", 1, q, 4, 1, GV_DISTANCE_EUCLIDEAN, out) == -1,
           "connect to dead port fails");
    ASSERT(shard_rpc_search(NULL, 8080, q, 4, 1, GV_DISTANCE_EUCLIDEAN, out) == -1,
           "NULL host rejected");
    ASSERT(shard_rpc_search("127.0.0.1", 8080, q, 4, 0, GV_DISTANCE_EUCLIDEAN, out) == -1,
           "k=0 rejected");
    shard_rpc_server_stop(NULL);
    return 0;
}

/* End-to-end distributed query: a coordinator with one local shard plus one
 * remote shard (a second node reached over TCP) must merge both sides, with the
 * global nearest coming from the remote node - proving the fan-out contributes. */
static int test_distributed_local_plus_remote(void) {
    /* Remote node: holds the global nearest (component 3 == 0.5, dist 0.25). */
    GV_Database *rdb = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    GV_ShardManager *remote = shard_manager_create(NULL);
    ASSERT(rdb && remote, "remote setup");
    shard_add(remote, 0, "local:0");
    shard_attach_local(remote, 0, rdb);
    float rn0[4] = {0, 0, 0, 0.5f};   /* dist 0.25 -> global nearest */
    float rn1[4] = {0, 2, 0, 0};      /* dist 4 */
    ASSERT(db_add_vector(rdb, rn0, 4) == 0 && db_add_vector(rdb, rn1, 4) == 0, "remote adds");

    GV_ShardRpcServer *srv = shard_rpc_serve(remote, "127.0.0.1", 0);
    ASSERT(srv != NULL, "serve remote");
    char addr[64];
    snprintf(addr, sizeof(addr), "127.0.0.1:%u", shard_rpc_server_port(srv));

    /* Coordinator: one local shard + one remote shard pointing at the node above. */
    GV_Database *ldb = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    GV_ShardManager *coord = shard_manager_create(NULL);
    ASSERT(ldb && coord, "coord setup");
    shard_add(coord, 0, "local:0");
    shard_attach_local(coord, 0, ldb);          /* local shard */
    shard_add(coord, 1, addr);                  /* remote shard (no local db) */
    float ln0[4] = {1, 0, 0, 0};      /* dist 1 */
    float ln1[4] = {3, 0, 0, 0};      /* dist 9 */
    ASSERT(db_add_vector(ldb, ln0, 4) == 0 && db_add_vector(ldb, ln1, 4) == 0, "local adds");

    float q[4] = {0, 0, 0, 0};
    GV_SearchResult res[3];
    int n = shard_rpc_search_distributed(coord, q, 4, 3, GV_DISTANCE_EUCLIDEAN, res);
    ASSERT(n == 3, "distributed returns global top-3 (local + remote)");
    ASSERT(res[0].distance <= res[1].distance && res[1].distance <= res[2].distance,
           "globally ordered");
    /* Nearest is the remote 0.5 vector; second is the local {1,0,0,0}. */
    ASSERT(res[0].vector && fabsf(res[0].vector->data[3] - 0.5f) < 1e-6f,
           "global nearest came from the remote node");
    ASSERT(res[1].vector && fabsf(res[1].vector->data[0] - 1.0f) < 1e-6f,
           "second nearest came from the local shard");
    gv_search_results_free(res, (size_t)n);

    /* Coordinator with the remote node down still returns local results. */
    shard_rpc_server_stop(srv);
    GV_SearchResult res2[3];
    int n2 = shard_rpc_search_distributed(coord, q, 4, 3, GV_DISTANCE_EUCLIDEAN, res2);
    ASSERT(n2 == 2, "unreachable remote -> local-only partial results");
    gv_search_results_free(res2, (size_t)n2);

    shard_manager_destroy(coord);
    shard_manager_destroy(remote);
    db_close(ldb);
    db_close(rdb);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing rpc_matches_local...", test_rpc_matches_local},
        {"Testing distributed_local_plus_remote...", test_distributed_local_plus_remote},
        {"Testing rpc_errors...",        test_rpc_errors},
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
