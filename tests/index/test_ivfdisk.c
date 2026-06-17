#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "gigavector.h"
#include "core/sim_time.h"
#include "index/flat.h"
#include "index/ivfdisk.h"
#include "index/index_maintenance.h"
#include "storage/posting_list.h"
#include "storage/soa_storage.h"
#include "../test_tmp.h"

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s:%d: %s (%s)\n", __FILE__, __LINE__, msg, #cond); \
        return -1; \
    } \
} while (0)

static int test_ivfdisk_create_train_search(void)
{
    char dir[512];
    ASSERT(gv_test_mkdtemp(dir, sizeof(dir), "gv_ivfdisk_test") == 0, "mkdtemp");

    GV_IVFDiskConfig cfg;
    ivfdisk_config_init(&cfg);
    cfg.nlist = 8;
    cfg.nprobe = 2;
    cfg.train_iters = 8;
    cfg.data_dir = dir;

    const size_t dim = 8;
    GV_IVFDiskIndex *idx = ivfdisk_create(dim, &cfg);
    ASSERT(idx != NULL, "create");

    float train[80 * 8];
    for (size_t i = 0; i < 80; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            train[i * dim + d] = (float)(i + d) * 0.01f;
        }
    }
    ASSERT(ivfdisk_train(idx, train, 80) == 0, "train");
    ASSERT(ivfdisk_is_trained(idx) == 1, "trained");

    for (size_t i = 0; i < 40; ++i) {
        ASSERT(ivfdisk_insert(idx, train + i * dim, dim, i) == 0, "insert");
    }
    ASSERT(ivfdisk_count(idx) == 40, "count");

    float query[8];
    memcpy(query, train, dim * sizeof(float));
    GV_SearchResult results[5];
    memset(results, 0, sizeof(results));
    int found = ivfdisk_search(idx, query, 5, results, GV_DISTANCE_EUCLIDEAN);
    ASSERT(found > 0, "search found");
    ASSERT(results[0].id == 0, "nearest self");

    ivfdisk_destroy(idx);
    return 0;
}

static int test_ivfdisk_save_load_roundtrip(void)
{
    char dir[512];
    ASSERT(gv_test_mkdtemp(dir, sizeof(dir), "gv_ivfdisk_save") == 0, "mkdtemp save dir");

    GV_IVFDiskConfig cfg;
    ivfdisk_config_init(&cfg);
    cfg.nlist = 4;
    cfg.nprobe = 2;
    cfg.data_dir = dir;

    const size_t dim = 4;
    GV_IVFDiskIndex *idx = ivfdisk_create(dim, &cfg);
    ASSERT(idx != NULL, "create");

    float train[20 * 4];
    for (size_t i = 0; i < 20; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            train[i * dim + d] = (float)i * 0.1f + (float)d * 0.01f;
        }
    }
    ASSERT(ivfdisk_train(idx, train, 20) == 0, "train");
    for (size_t i = 0; i < 10; ++i) {
        ASSERT(ivfdisk_insert(idx, train + i * dim, dim, i) == 0, "insert");
    }

    char snap_path[512];
    int fd = gv_test_mkstemp(snap_path, sizeof(snap_path), "gv_ivfdisk_snap");
    ASSERT(fd >= 0, "mkstemp snap");
    FILE *out = fdopen(fd, "wb");
    ASSERT(out != NULL, "fdopen out");
    ASSERT(ivfdisk_save(idx, out, 4) == 0, "save head");
    ASSERT(fclose(out) == 0, "close out");

    FILE *in = fopen(snap_path, "rb");
    ASSERT(in != NULL, "fopen in");
    GV_IVFDiskIndex *loaded = NULL;
    ASSERT(ivfdisk_load(&loaded, in, dim, dir, 4) == 0, "load head");
    ASSERT(fclose(in) == 0, "close in");
    unlink(snap_path);

    ASSERT(loaded != NULL, "loaded");
    ASSERT(ivfdisk_is_trained(loaded) == 1, "loaded trained");
    ASSERT(ivfdisk_count(loaded) == 10, "loaded count");

    float query[4];
    memcpy(query, train, dim * sizeof(float));
    GV_SearchResult results[3];
    memset(results, 0, sizeof(results));
    int found = ivfdisk_search(loaded, query, 3, results, GV_DISTANCE_EUCLIDEAN);
    ASSERT(found > 0, "loaded search");

    ivfdisk_destroy(loaded);
    ivfdisk_destroy(idx);
    return 0;
}

static int test_db_ivfdisk_save_load(void)
{
    char db_path[512];
    ASSERT(gv_test_mkstemp(db_path, sizeof(db_path), "gv_db_ivfdisk") >= 0, "mkstemp db");
    unlink(db_path);

    const size_t dim = 4;
    GV_IVFDiskConfig cfg;
    ivfdisk_config_init(&cfg);
    cfg.nlist = 8;
    cfg.nprobe = 2;
    GV_Database *db = db_open_with_ivfdisk_config(db_path, dim, GV_INDEX_TYPE_IVFDISK, &cfg);
    ASSERT(db != NULL, "db open");

    float train[64 * 4];
    for (size_t i = 0; i < 64; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            train[i * dim + d] = (float)(i + d) / 32.f;
        }
    }
    ASSERT(db_ivfdisk_train(db, train, 64, dim) == 0, "db train");

    for (size_t i = 0; i < 16; ++i) {
        ASSERT(db_add_vector(db, train + i * dim, dim) == 0, "db add");
    }

    ASSERT(db_save(db, NULL) == 0, "db save");
    db_close(db);

    db = db_open(db_path, dim, GV_INDEX_TYPE_IVFDISK);
    ASSERT(db != NULL, "db reopen");
    ASSERT(db->count == 16, "db count");

    float query[4];
    memcpy(query, train, dim * sizeof(float));
    GV_SearchResult results[5];
    memset(results, 0, sizeof(results));
    int found = db_search(db, query, 5, results, GV_DISTANCE_EUCLIDEAN);
    ASSERT(found > 0, "db search after load");

    for (int i = 0; i < found; ++i) {
        if (results[i].vector) vector_destroy((GV_Vector *)results[i].vector);
    }

    db_close(db);
    unlink(db_path);
    char data_dir[512];
    snprintf(data_dir, sizeof(data_dir), "%s.ivfdisk", db_path);
    /* posting catalog persists under data_dir; leave for /tmp cleanup */
    (void)data_dir;
    return 0;
}

static int test_ivfdisk_delete_update(void)
{
    char dir[512];
    ASSERT(gv_test_mkdtemp(dir, sizeof(dir), "gv_ivfdisk_del") == 0, "mkdtemp");

    const size_t dim = 8;
    GV_IVFDiskConfig cfg;
    ivfdisk_config_init(&cfg);
    cfg.nlist = 4;
    cfg.nprobe = 4;
    cfg.data_dir = dir;

    GV_IVFDiskIndex *idx = ivfdisk_create(dim, &cfg);
    ASSERT(idx != NULL, "create");

    float train[40 * 8];
    for (size_t i = 0; i < 40; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            train[i * dim + d] = (float)(i + d) * 0.05f;
        }
    }
    ASSERT(ivfdisk_train(idx, train, 40) == 0, "train");
    for (size_t i = 0; i < 20; ++i) {
        ASSERT(ivfdisk_insert(idx, train + i * dim, dim, i) == 0, "insert");
    }

    ASSERT(ivfdisk_delete(idx, 0, train) == 0, "delete");
    float updated[8];
    for (size_t d = 0; d < dim; ++d) updated[d] = 99.f + (float)d;
    ASSERT(ivfdisk_update(idx, 1, updated, dim) == 0, "update");

    GV_SearchResult results[10];
    memset(results, 0, sizeof(results));
    int found = ivfdisk_search(idx, train, 10, results, GV_DISTANCE_EUCLIDEAN);
    ASSERT(found > 0, "search after delete/update");
    for (int i = 0; i < found; ++i) {
        ASSERT(results[i].id != 0, "deleted id absent");
    }

    ivfdisk_destroy(idx);
    return 0;
}

static int test_ivfdisk_sq8_and_hnsw(void)
{
    char dir[512];
    ASSERT(gv_test_mkdtemp(dir, sizeof(dir), "gv_ivfdisk_sq8") == 0, "mkdtemp");

    const size_t dim = 16;
    const size_t n = 128;
    float *data = (float *)malloc(n * dim * sizeof(float));
    ASSERT(data != NULL, "alloc");
    for (size_t i = 0; i < n * dim; ++i) {
        data[i] = (float)((i * 13u) % 997u) / 997.f;
    }

    GV_IVFDiskConfig cfg;
    ivfdisk_config_init(&cfg);
    cfg.nlist = 16;
    cfg.nprobe = 8;
    cfg.use_sq8 = 1;
    cfg.use_hnsw_head = 1;
    cfg.data_dir = dir;

    GV_IVFDiskIndex *idx = ivfdisk_create(dim, &cfg);
    ASSERT(idx != NULL, "create sq8/hnsw");
    ASSERT(ivfdisk_train(idx, data, n) == 0, "train");
    for (size_t i = 0; i < n; ++i) {
        ASSERT(ivfdisk_insert(idx, data + i * dim, dim, i) == 0, "insert");
    }

    uint64_t linear = ivfdisk_nearest_head(idx, data);
    (void)linear;

    GV_SearchResult results[5];
    memset(results, 0, sizeof(results));
    int found = ivfdisk_search(idx, data, 5, results, GV_DISTANCE_EUCLIDEAN);
    ASSERT(found > 0, "sq8 search");
    ASSERT(results[0].id == 0, "self nearest");

    ivfdisk_destroy(idx);
    free(data);
    return 0;
}

static int test_ivfdisk_recall_vs_flat(void)
{
    char dir[512];
    ASSERT(gv_test_mkdtemp(dir, sizeof(dir), "gv_ivfdisk_recall") == 0, "mkdtemp");

    const size_t dim = 32;
    const size_t n = 2000;
    const size_t k = 10;

    float *data = (float *)malloc(n * dim * sizeof(float));
    ASSERT(data != NULL, "alloc data");
    for (size_t i = 0; i < n * dim; ++i) {
        data[i] = (float)((i * 17u) % 1000u) / 1000.f;
    }

    GV_IVFDiskConfig cfg;
    ivfdisk_config_init(&cfg);
    cfg.nlist = 64;
    cfg.nprobe = 16;
    cfg.use_sq8 = 1;
    cfg.data_dir = dir;
    GV_IVFDiskIndex *idx = ivfdisk_create(dim, &cfg);
    ASSERT(idx != NULL, "create");
    ASSERT(ivfdisk_train(idx, data, n) == 0, "train");
    for (size_t i = 0; i < n; ++i) {
        ASSERT(ivfdisk_insert(idx, data + i * dim, dim, i) == 0, "insert");
    }

    void *flat = flat_create(dim, NULL, NULL);
    ASSERT(flat != NULL, "flat create");
    for (size_t i = 0; i < n; ++i) {
        GV_Vector *v = vector_create_from_data(dim, data + i * dim);
        ASSERT(v != NULL, "vec");
        ASSERT(flat_insert(flat, v) == 0, "flat insert");
    }

    size_t hits = 0;
    for (size_t qi = 0; qi < 50; ++qi) {
        const float *query = data + (qi * 7 % n) * dim;
        GV_SearchResult ivf_res[10], flat_res[10];
        memset(ivf_res, 0, sizeof(ivf_res));
        memset(flat_res, 0, sizeof(flat_res));

        GV_Vector qv;
        qv.dimension = dim;
        qv.data = (float *)query;
        qv.metadata = NULL;

        int nf = ivfdisk_search(idx, query, k, ivf_res, GV_DISTANCE_EUCLIDEAN);
        int nb = flat_search(flat, &qv, k, flat_res, GV_DISTANCE_EUCLIDEAN, NULL, NULL);
        ASSERT(nf > 0 && nb > 0, "searches");

        for (int i = 0; i < nf; ++i) {
            for (int j = 0; j < nb; ++j) {
                if (ivf_res[i].id == flat_res[j].id) {
                    hits++;
                    break;
                }
            }
        }
        for (int i = 0; i < nb; ++i) {
            if (flat_res[i].vector) vector_destroy((GV_Vector *)flat_res[i].vector);
        }
    }

    double recall = (double)hits / (50.0 * (double)k);
    ASSERT(recall >= 0.9, "recall >= 90% on benchmark suite");

    flat_destroy(flat);
    ivfdisk_destroy(idx);
    free(data);
    return 0;
}

static int test_ivfdisk_border_replication(void)
{
    char dir[512];
    ASSERT(gv_test_mkdtemp(dir, sizeof(dir), "gv_ivfdisk_border") == 0, "mkdtemp");

    const size_t dim = 2;
    GV_IVFDiskConfig cfg;
    ivfdisk_config_init(&cfg);
    cfg.nlist = 2;
    cfg.nprobe = 2;
    cfg.border_ratio = 10.f;
    cfg.data_dir = dir;

    GV_IVFDiskIndex *idx = ivfdisk_create(dim, &cfg);
    ASSERT(idx != NULL, "create");

    float train[100 * 2];
    for (size_t i = 0; i < 50; ++i) {
        train[i * 2] = 0.f;
        train[i * 2 + 1] = (float)i * 0.01f;
        train[(50 + i) * 2] = 10.f;
        train[(50 + i) * 2 + 1] = (float)i * 0.01f;
    }
    ASSERT(ivfdisk_train(idx, train, 100) == 0, "train");

    float border[2] = {5.f, 0.f};
    uint64_t heads[2];
    size_t nh = 0;
    ASSERT(ivfdisk_insert_routed(idx, border, dim, 999, heads, &nh, 2) == 0, "insert routed");
    ASSERT(nh == 2, "border vector replicated to two heads");
    ASSERT(ivfdisk_head_live_count(idx, heads[0]) >= 1, "primary head live");
    ASSERT(ivfdisk_head_live_count(idx, heads[1]) >= 1, "secondary head live");

    ivfdisk_destroy(idx);
    return 0;
}

static int test_ivfdisk_head_ratio_enforced(void)
{
    char dir[512];
    ASSERT(gv_test_mkdtemp(dir, sizeof(dir), "gv_ivfdisk_ratio") == 0, "mkdtemp");

    const size_t dim = 32;
    GV_IVFDiskConfig cfg;
    ivfdisk_config_init(&cfg);
    cfg.nlist = 512;
    cfg.cache_size_mb = 1;
    cfg.head_ratio = 0.001f;
    cfg.data_dir = dir;

    GV_IVFDiskIndex *idx = ivfdisk_create(dim, &cfg);
    ASSERT(idx != NULL, "create");

    float *train = (float *)malloc(cfg.nlist * dim * sizeof(float));
    ASSERT(train != NULL, "alloc train");
    for (size_t i = 0; i < cfg.nlist * dim; ++i) {
        train[i] = (float)i * 0.001f;
    }
    ASSERT(ivfdisk_train(idx, train, cfg.nlist) != 0, "train rejects centroid RAM over head_ratio");
    free(train);
    ivfdisk_destroy(idx);
    return 0;
}

static int test_db_ivfdisk_wal_replay(void)
{
    char db_path[512];
    ASSERT(gv_test_mkstemp(db_path, sizeof(db_path), "gv_ivfdisk_wal") >= 0, "mkstemp db");
    unlink(db_path);

    char wal_path[512];
    snprintf(wal_path, sizeof(wal_path), "%s.wal", db_path);
    remove(wal_path);

    const size_t dim = 4;
    GV_IVFDiskConfig cfg;
    ivfdisk_config_init(&cfg);
    cfg.nlist = 4;
    cfg.nprobe = 2;

    GV_Database *db = db_open_with_ivfdisk_config(db_path, dim, GV_INDEX_TYPE_IVFDISK, &cfg);
    ASSERT(db != NULL, "db open");
    ASSERT(db_set_wal(db, wal_path) == 0, "set wal");

    float train[32 * 4];
    for (size_t i = 0; i < 32; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            train[i * dim + d] = (float)(i + d) / 16.f;
        }
    }
    ASSERT(db_ivfdisk_train(db, train, 32, dim) == 0, "train");
    ASSERT(db_save(db, NULL) == 0, "save after train");

    for (size_t i = 0; i < 8; ++i) {
        ASSERT(db_add_vector(db, train + i * dim, dim) == 0, "add");
    }
    db_close(db);

    db = db_open(db_path, dim, GV_INDEX_TYPE_IVFDISK);
    ASSERT(db != NULL, "reopen");
    ASSERT(db->count == 8, "wal replay count");

    float query[4];
    memcpy(query, train, dim * sizeof(float));
    GV_SearchResult results[4];
    memset(results, 0, sizeof(results));
    int found = db_search(db, query, 4, results, GV_DISTANCE_EUCLIDEAN);
    ASSERT(found > 0, "search after wal replay");
    for (int i = 0; i < found; ++i) {
        if (results[i].vector) vector_destroy((GV_Vector *)results[i].vector);
    }

    db_close(db);
    unlink(db_path);
    remove(wal_path);
    return 0;
}

static int test_db_ivfdisk_mmap_open(void)
{
    char db_path[512];
    ASSERT(gv_test_mkstemp(db_path, sizeof(db_path), "gv_ivfdisk_mmap") >= 0, "mkstemp db");
    unlink(db_path);

    const size_t dim = 4;
    GV_IVFDiskConfig cfg;
    ivfdisk_config_init(&cfg);
    cfg.nlist = 4;
    cfg.nprobe = 2;

    GV_Database *db = db_open_with_ivfdisk_config(db_path, dim, GV_INDEX_TYPE_IVFDISK, &cfg);
    ASSERT(db != NULL, "db open");

    float train[24 * 4];
    for (size_t i = 0; i < 24; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            train[i * dim + d] = (float)(i + d) / 12.f;
        }
    }
    ASSERT(db_ivfdisk_train(db, train, 24, dim) == 0, "train");
    for (size_t i = 0; i < 6; ++i) {
        ASSERT(db_add_vector(db, train + i * dim, dim) == 0, "add");
    }
    ASSERT(db_save(db, NULL) == 0, "save");
    db_close(db);

    db = db_open_mmap(db_path, dim, GV_INDEX_TYPE_IVFDISK);
    ASSERT(db != NULL, "mmap open");
    ASSERT(db->count == 6, "mmap count");

    float query[4];
    memcpy(query, train, dim * sizeof(float));
    GV_SearchResult results[3];
    memset(results, 0, sizeof(results));
    int found = db_search(db, query, 3, results, GV_DISTANCE_EUCLIDEAN);
    ASSERT(found > 0, "mmap search");
    for (int i = 0; i < found; ++i) {
        if (results[i].vector) vector_destroy((GV_Vector *)results[i].vector);
    }

    db_close(db);
    unlink(db_path);
    return 0;
}

static int test_ivfdisk_maintenance_merge(void)
{
    char dir[512];
    ASSERT(gv_test_mkdtemp(dir, sizeof(dir), "gv_ivfdisk_maint_m") == 0, "mkdtemp");

    GV_IVFDiskConfig cfg;
    ivfdisk_config_init(&cfg);
    cfg.nlist = 4;
    cfg.nprobe = 2;
    cfg.data_dir = dir;

    const size_t dim = 4;
    GV_IVFDiskIndex *idx = ivfdisk_create(dim, &cfg);
    ASSERT(idx != NULL, "create");

    float train[32 * 4];
    for (size_t i = 0; i < 32; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            train[i * dim + d] = (float)(i + d) * 0.05f;
        }
    }
    ASSERT(ivfdisk_train(idx, train, 32) == 0, "train");

    for (size_t i = 0; i < 8; ++i) {
        ASSERT(ivfdisk_insert_to_head(idx, 0, train, dim, i) == 0, "insert to head 0");
    }
    for (size_t r = 0; r < 2; ++r) {
        for (size_t i = 0; i < 8; ++i) {
            float updated[4];
            for (size_t d = 0; d < dim; ++d) {
                updated[d] = train[d] + (float)(i + r) * 0.01f;
            }
            ASSERT(ivfdisk_update(idx, i, updated, dim) == 0, "update");
        }
    }

    struct GV_PostingCatalog *cat = ivfdisk_catalog(idx);
    ASSERT(cat != NULL, "catalog");
    size_t segs_before = posting_catalog_segment_count_for_head(cat, 0);
    ASSERT(segs_before > 1, "updates create multiple segments");

    GV_IVFDiskMaintenanceConfig mcfg;
    ivfdisk_maintenance_config_init(&mcfg);
    mcfg.live_ratio_threshold = 0.6f;
    GV_IVFDiskMaintenanceStats mstats;
    ASSERT(ivfdisk_maintenance_run(idx, &mcfg, &mstats) == 0, "maintenance");
    ASSERT(mstats.merges > 0 || mstats.defrags > 0, "merge or defrag ran");

    size_t segs_after = posting_catalog_segment_count_for_head(cat, 0);
    ASSERT(segs_after <= segs_before, "segment count reduced");

    float query[4];
    memcpy(query, train, dim * sizeof(float));
    GV_SearchResult results[3];
    memset(results, 0, sizeof(results));
    ASSERT(ivfdisk_search(idx, query, 3, results, GV_DISTANCE_EUCLIDEAN) > 0, "search after merge");

    ivfdisk_destroy(idx);
    return 0;
}

static int test_ivfdisk_maintenance_split(void)
{
    char dir[512];
    ASSERT(gv_test_mkdtemp(dir, sizeof(dir), "gv_ivfdisk_maint_s") == 0, "mkdtemp");

    GV_IVFDiskConfig cfg;
    ivfdisk_config_init(&cfg);
    cfg.nlist = 2;
    cfg.nprobe = 2;
    cfg.max_list_bytes = 4096;
    cfg.data_dir = dir;

    const size_t dim = 2;
    GV_IVFDiskIndex *idx = ivfdisk_create(dim, &cfg);
    ASSERT(idx != NULL, "create");

    float train[40 * 2];
    for (size_t i = 0; i < 20; ++i) {
        train[i * 2] = 0.f;
        train[i * 2 + 1] = (float)i * 0.01f;
    }
    for (size_t i = 0; i < 20; ++i) {
        train[(20 + i) * 2] = 10.f;
        train[(20 + i) * 2 + 1] = (float)i * 0.01f;
    }
    ASSERT(ivfdisk_train(idx, train, 40) == 0, "train");
    size_t nlist_before = ivfdisk_nlist(idx);

    for (size_t i = 0; i < 30; ++i) {
        ASSERT(ivfdisk_insert_to_head(idx, 0, train + (i % 20) * dim, dim, i) == 0,
               "insert to head 0");
    }

    GV_IVFDiskMaintenanceStats mstats;
    ASSERT(ivfdisk_maintenance_run(idx, NULL, &mstats) == 0, "maintenance split");
    ASSERT(mstats.splits > 0, "split executed");
    ASSERT(ivfdisk_nlist(idx) > nlist_before, "nlist grew after split");

    float query[2] = {0.f, 0.05f};
    GV_SearchResult results[5];
    memset(results, 0, sizeof(results));
    ASSERT(ivfdisk_search(idx, query, 5, results, GV_DISTANCE_EUCLIDEAN) > 0, "search after split");

    ivfdisk_destroy(idx);
    return 0;
}

static int test_ivfdisk_head_checkpoint_replay(void)
{
    char dir[512];
    ASSERT(gv_test_mkdtemp(dir, sizeof(dir), "gv_ivfdisk_ckpt") == 0, "mkdtemp");

    char snap_path[512];
    int snap_fd = gv_test_mkstemp(snap_path, sizeof(snap_path), "gv_ivfdisk_ckpt_snap");
    ASSERT(snap_fd >= 0, "mkstemp snap");

    GV_IVFDiskConfig cfg;
    ivfdisk_config_init(&cfg);
    cfg.nlist = 4;
    cfg.data_dir = dir;

    const size_t dim = 4;
    GV_IVFDiskIndex *idx = ivfdisk_create(dim, &cfg);
    ASSERT(idx != NULL, "create");

    float train[16 * 4];
    for (size_t i = 0; i < 16; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            train[i * dim + d] = (float)i * 0.1f;
        }
    }
    ASSERT(ivfdisk_train(idx, train, 16) == 0, "train");
    for (size_t i = 0; i < 4; ++i) {
        ASSERT(ivfdisk_insert(idx, train + i * dim, dim, i) == 0, "insert");
    }

    FILE *out = fdopen(snap_fd, "wb");
    ASSERT(out != NULL, "fdopen snap");
    ASSERT(ivfdisk_save(idx, out, 4) == 0, "save snap");
    ASSERT(fclose(out) == 0, "close snap");

    float extra[4] = {9.f, 9.1f, 9.2f, 9.3f};
    uint64_t new_head = 0;
    ASSERT(ivfdisk_add_centroid(idx, extra, &new_head) == 0, "add centroid");
    ASSERT(new_head == 4, "new head id");
    ASSERT(ivfdisk_head_checkpoint(idx) == 0, "checkpoint");
    ivfdisk_destroy(idx);

    FILE *in = fopen(snap_path, "rb");
    ASSERT(in != NULL, "fopen snap");
    GV_IVFDiskIndex *loaded = NULL;
    ASSERT(ivfdisk_load(&loaded, in, dim, dir, 4) == 0, "reload with checkpoint");
    ASSERT(fclose(in) == 0, "close snap in");
    unlink(snap_path);

    ASSERT(ivfdisk_nlist(loaded) == 5, "nlist from checkpoint replay");
    ASSERT(ivfdisk_is_trained(loaded) == 1, "still trained");

    float query[4];
    memcpy(query, extra, dim * sizeof(float));
    GV_SearchResult results[3];
    memset(results, 0, sizeof(results));
    ASSERT(ivfdisk_search(loaded, query, 3, results, GV_DISTANCE_EUCLIDEAN) >= 0, "search ok");

    ivfdisk_destroy(loaded);
    return 0;
}

static int test_ivfdisk_lazy_merge_on_search(void)
{
    char dir[512];
    ASSERT(gv_test_mkdtemp(dir, sizeof(dir), "gv_ivfdisk_lazy") == 0, "mkdtemp");

    GV_IVFDiskConfig cfg;
    ivfdisk_config_init(&cfg);
    cfg.nlist = 2;
    cfg.nprobe = 2;
    cfg.data_dir = dir;

    const size_t dim = 4;
    GV_IVFDiskIndex *idx = ivfdisk_create(dim, &cfg);
    ASSERT(idx != NULL, "create");

    float train[8 * 4];
    for (size_t i = 0; i < 8; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            train[i * dim + d] = (float)i + (float)d * 0.01f;
        }
    }
    ASSERT(ivfdisk_train(idx, train, 8) == 0, "train");
    for (size_t i = 0; i < 4; ++i) {
        ASSERT(ivfdisk_insert(idx, train + i * dim, dim, i) == 0, "insert");
    }
    for (size_t i = 0; i < 4; ++i) {
        float updated[4];
        for (size_t d = 0; d < dim; ++d) {
            updated[d] = train[i * dim + d] + 100.f;
        }
        ASSERT(ivfdisk_update(idx, i, updated, dim) == 0, "update");
    }

    struct GV_PostingCatalog *cat = ivfdisk_catalog(idx);
    size_t segs_before = posting_catalog_segment_count_for_head(cat, 0);
    ASSERT(segs_before > 1, "updates create multiple segments");

    float query[4];
    memcpy(query, train, dim * sizeof(float));
    GV_SearchResult results[2];
    memset(results, 0, sizeof(results));
    ASSERT(ivfdisk_search(idx, query, 2, results, GV_DISTANCE_EUCLIDEAN) > 0, "search");

    size_t segs_after = posting_catalog_segment_count_for_head(cat, 0);
    ASSERT(segs_after < segs_before, "lazy merge reduced segments on search");

    ivfdisk_destroy(idx);
    return 0;
}

static int test_ivfdisk_maintenance_multi_split(void)
{
    char dir[512];
    ASSERT(gv_test_mkdtemp(dir, sizeof(dir), "gv_ivfdisk_msplit") == 0, "mkdtemp");

    GV_IVFDiskConfig cfg;
    ivfdisk_config_init(&cfg);
    cfg.nlist = 2;
    cfg.nprobe = 2;
    cfg.max_list_bytes = 512;
    cfg.data_dir = dir;

    const size_t dim = 2;
    GV_IVFDiskIndex *idx = ivfdisk_create(dim, &cfg);
    ASSERT(idx != NULL, "create");

    float train[40 * 2];
    for (size_t i = 0; i < 20; ++i) {
        train[i * 2] = 0.f;
        train[i * 2 + 1] = (float)i * 0.01f;
    }
    for (size_t i = 0; i < 20; ++i) {
        train[(20 + i) * 2] = 10.f;
        train[(20 + i) * 2 + 1] = (float)i * 0.01f;
    }
    ASSERT(ivfdisk_train(idx, train, 40) == 0, "train");
    size_t nlist_before = ivfdisk_nlist(idx);

    for (size_t i = 0; i < 50; ++i) {
        ASSERT(ivfdisk_insert_to_head(idx, 0, train + (i % 20) * dim, dim, i) == 0,
               "insert to head 0");
    }

    GV_IVFDiskMaintenanceStats mstats;
    ASSERT(ivfdisk_maintenance_run(idx, NULL, &mstats) == 0, "maintenance split");
    ASSERT(mstats.splits > 0, "split executed");
    ASSERT(ivfdisk_nlist(idx) >= nlist_before + 2, "K=ceil split created >=2 new heads");

    ivfdisk_destroy(idx);
    return 0;
}

static int test_ivfdisk_head_checkpoint_timer(void)
{
    char dir[512];
    ASSERT(gv_test_mkdtemp(dir, sizeof(dir), "gv_ivfdisk_ckpt_t") == 0, "mkdtemp");

    gv_sim_time_set_mode(GV_TIME_SIM);
    gv_sim_time_reset(1700000000ULL);

    GV_IVFDiskConfig cfg;
    ivfdisk_config_init(&cfg);
    cfg.nlist = 4;
    cfg.head_checkpoint_interval_sec = 60;
    cfg.head_wal_checkpoint_bytes = 1024u * 1024u;
    cfg.data_dir = dir;

    const size_t dim = 4;
    GV_IVFDiskIndex *idx = ivfdisk_create(dim, &cfg);
    ASSERT(idx != NULL, "create");

    float train[8 * 4];
    for (size_t i = 0; i < 8; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            train[i * dim + d] = (float)i;
        }
    }
    ASSERT(ivfdisk_train(idx, train, 8) == 0, "train");

    float extra[4] = {9.f, 9.1f, 9.2f, 9.3f};
    uint64_t new_head = 0;
    ASSERT(ivfdisk_add_centroid(idx, extra, &new_head) == 0, "add centroid");

    char wal_path[1024];
    snprintf(wal_path, sizeof(wal_path), "%s/head_wal.bin", dir);
    ASSERT(access(wal_path, F_OK) == 0, "head wal exists before timer checkpoint");

    gv_sim_time_advance_sec(61);
    ASSERT(ivfdisk_head_checkpoint_if_needed(idx) == 0, "timer checkpoint");
    ASSERT(access(wal_path, F_OK) != 0, "head wal truncated after timer checkpoint");
    ASSERT(ivfdisk_nlist(idx) == 5, "centroid count preserved");

    gv_sim_time_set_mode(GV_TIME_WALL);
    ivfdisk_destroy(idx);
    return 0;
}

int main(void)
{
    struct { const char *name; int (*fn)(void); } tests[] = {
        { "create/train/search", test_ivfdisk_create_train_search },
        { "save/load roundtrip", test_ivfdisk_save_load_roundtrip },
        { "db save/load", test_db_ivfdisk_save_load },
        { "delete/update", test_ivfdisk_delete_update },
        { "sq8/hnsw head", test_ivfdisk_sq8_and_hnsw },
        { "recall vs flat", test_ivfdisk_recall_vs_flat },
        { "border replication", test_ivfdisk_border_replication },
        { "head_ratio enforced", test_ivfdisk_head_ratio_enforced },
        { "wal replay", test_db_ivfdisk_wal_replay },
        { "mmap open", test_db_ivfdisk_mmap_open },
        { "maintenance merge", test_ivfdisk_maintenance_merge },
        { "maintenance split", test_ivfdisk_maintenance_split },
        { "maintenance multi split", test_ivfdisk_maintenance_multi_split },
        { "lazy merge on search", test_ivfdisk_lazy_merge_on_search },
        { "head checkpoint replay", test_ivfdisk_head_checkpoint_replay },
        { "head checkpoint timer", test_ivfdisk_head_checkpoint_timer },
    };
    int failed = 0;
    for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); ++i) {
        fprintf(stderr, "Running %s...\n", tests[i].name);
        if (tests[i].fn() != 0) failed++;
    }
    if (failed) {
        fprintf(stderr, "%d test(s) failed\n", failed);
        return 1;
    }
    fprintf(stderr, "All IVFDisk tests passed\n");
    return 0;
}
