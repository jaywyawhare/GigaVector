#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "api/grpc.h"
#include "core/arena.h"
#include "core/scope.h"
#include "storage/database.h"

#define ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1; \
        } \
    } while (0)

static int test_search_scratch_arena(void) {
    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db open");

    float v[4] = {1, 0, 0, 0};
    ASSERT(db_add_vector(db, v, 4) == 0, "insert");

    for (int i = 0; i < 256; ++i) {
        GV_WITH_ARENA(scratch, 64 * 1024) {
            GV_SearchResult results[4];
            float q[4] = {1, 0, 0, 0};
            int n = db_search(db, q, 1, results, GV_DISTANCE_EUCLIDEAN);
            ASSERT(n == 1, "search hit");
            (void)gv_arena_used(&scratch);
        }
        gv_tls_arena_reset();
    }

    db_close(db);
    return 0;
}

int main(void) {
    if (test_search_scratch_arena() != 0) {
        return 1;
    }
    fprintf(stderr, "OK: grpc/search arena dst\n");
    return 0;
}
