/**
 * @file test_ivfdisk_maintenance_dst.c
 * @brief DST oracle: insert → compact/split → search recall preserved.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gigavector.h"
#include "index/index_maintenance.h"
#include "../test_tmp.h"

#define ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s (%s)\n", msg, #cond); \
            return -1; \
        } \
    } while (0)

#define DIM 4

static int recall_at_k(const GV_SearchResult *hits, int n, size_t target)
{
    for (int i = 0; i < n; ++i) {
        if (hits[i].id == target) return 1;
    }
    return 0;
}

static int test_insert_compact_search_oracle(void)
{
    char db_path[512];
    ASSERT(gv_test_mkstemp(db_path, sizeof(db_path), "gv_dst_ivfdisk_maint") >= 0, "mkstemp db");
    unlink(db_path);

    GV_IVFDiskConfig cfg;
    ivfdisk_config_init(&cfg);
    cfg.nlist = 4;
    cfg.nprobe = 4;
    cfg.max_list_bytes = 8192;

    GV_Database *db = db_open_with_ivfdisk_config(db_path, DIM, GV_INDEX_TYPE_IVFDISK, &cfg);
    ASSERT(db != NULL, "open db");

    float train[64 * DIM];
    for (size_t i = 0; i < 64; ++i) {
        for (size_t d = 0; d < DIM; ++d) {
            train[i * DIM + d] = (float)(i + d) / 32.f;
        }
    }
    ASSERT(db_ivfdisk_train(db, train, 64, DIM) == 0, "train");

    for (size_t i = 0; i < 24; ++i) {
        ASSERT(db_add_vector(db, train + i * DIM, DIM) == 0, "insert");
    }

    float current[24 * DIM];
    memcpy(current, train, 24 * DIM * sizeof(float));
    for (size_t i = 0; i < 8; ++i) {
        for (size_t d = 0; d < DIM; ++d) {
            current[i * DIM + d] = train[i * DIM + d] + 0.02f;
        }
        ASSERT(db_update_vector(db, i, current + i * DIM, DIM) == 0, "update");
    }

    ASSERT(db_compact(db) == 0, "compact triggers maintenance");

    for (size_t q = 0; q < 4; ++q) {
        size_t k = db->count > 0 ? db->count : 1;
        GV_SearchResult *hits = (GV_SearchResult *)calloc(k, sizeof(GV_SearchResult));
        ASSERT(hits != NULL, "alloc hits");
        int found = db_search(db, current + q * DIM, k, hits, GV_DISTANCE_EUCLIDEAN);
        ASSERT(found > 0, "search after maintenance");
        ASSERT(recall_at_k(hits, found, q), "query finds itself with full oversample");
        for (int i = 0; i < found; ++i) {
            if (hits[i].vector) vector_destroy((GV_Vector *)hits[i].vector);
        }
        free(hits);
    }

    db_close(db);
    unlink(db_path);
    return 0;
}

int main(void)
{
    if (test_insert_compact_search_oracle() != 0) {
        fprintf(stderr, "DST IVFDisk maintenance oracle failed\n");
        return 1;
    }
    printf("DST IVFDisk maintenance oracle passed\n");
    return 0;
}
