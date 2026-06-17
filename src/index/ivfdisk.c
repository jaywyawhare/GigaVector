/**
 * @file ivfdisk.c
 * @brief IVF head in RAM + on-disk posting lists per centroid (SPANN-style hybrid).
 */

#include "index/ivfdisk.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "core/heap.h"
#include "core/sim_time.h"
#include "core/utils.h"
#include "index/hnsw.h"
#include "index/index_maintenance.h"
#include "schema/vector.h"
#include "search/distance.h"
#include "storage/disk_layout.h"
#include "storage/disk_page_cache.h"
#include "storage/posting_list.h"

typedef struct {
    float dist;
    size_t id;
} IVFDiskHeapItem;

typedef struct {
    size_t head_id;
    float dist;
} IVFDiskHeadDist;

typedef struct {
    uint64_t head_id;
    uint64_t secondary_head_id;
    uint8_t version;
    uint8_t deleted;
} IVFDiskVectorLoc;

#define GV_IVFDISK_NO_SECONDARY_HEAD UINT64_MAX

#define GV_HEAD_WAL_MAGIC     0x47564857u /* "GVHW" */
#define GV_HEAD_CKPT_MAGIC    0x47564843u /* "GVHC" */
#define GV_HEAD_WAL_ADD       1u
#define GV_HEAD_WAL_UPDATE    2u
#define GV_HEAD_WAL_SPLIT     3u

#define GV_IVFDISK_HEAD_WAL_CKPT_BYTES_DEFAULT (10u * 1024u * 1024u)
#define GV_IVFDISK_HEAD_CKPT_INTERVAL_SEC_DEFAULT 300u

GV_HEAP_DEFINE(ivfdisk_heap, IVFDiskHeapItem)

struct GV_IVFDiskIndex {
    size_t dimension;
    GV_IVFDiskConfig config;
    char *data_dir;
    float *centroids;
    int trained;
    size_t total_count;
    GV_PostingCatalog *catalog;
    GV_DiskPageCache *page_cache;
    int owns_cache;
    IVFDiskVectorLoc *vector_locs;
    size_t vector_loc_cap;
    void *head_hnsw;
    size_t maint_splits;
    size_t maint_merges;
    size_t maint_defrags;
    size_t maint_reassigns;
    uint64_t head_last_checkpoint_sec;
};

static int ivfdisk_head_wal_file_size(GV_IVFDiskIndex *index, size_t *out_bytes);
int ivfdisk_head_checkpoint_if_needed(GV_IVFDiskIndex *index);

static float ivfdisk_dist_raw(const float *query, const float *vec, size_t dim,
                              GV_DistanceType metric)
{
    GV_Vector qv, vv;
    qv.data = (float *)query;
    qv.dimension = dim;
    qv.metadata = NULL;
    vv.data = (float *)vec;
    vv.dimension = dim;
    vv.metadata = NULL;
    return distance(&qv, &vv, metric);
}

static int ivfdisk_kmeans(const float *data, size_t count, size_t dim,
                          size_t k, size_t iters, float *out_centroids)
{
    if (count < k || !data || !out_centroids) return -1;
    memcpy(out_centroids, data, k * dim * sizeof(float));

    int *assign = (int *)malloc(count * sizeof(int));
    float *new_centroids = (float *)calloc(k * dim, sizeof(float));
    size_t *counts = (size_t *)calloc(k, sizeof(size_t));
    if (!assign || !new_centroids || !counts) {
        free(assign);
        free(new_centroids);
        free(counts);
        return -1;
    }

    for (size_t iter = 0; iter < iters; ++iter) {
        for (size_t i = 0; i < count; ++i) {
            const float *vec = data + i * dim;
            float best = FLT_MAX;
            int best_c = 0;
            for (size_t c = 0; c < k; ++c) {
                float dist = 0.f;
                const float *cent = out_centroids + c * dim;
                for (size_t d = 0; d < dim; ++d) {
                    float diff = vec[d] - cent[d];
                    dist += diff * diff;
                }
                if (dist < best) {
                    best = dist;
                    best_c = (int)c;
                }
            }
            assign[i] = best_c;
        }

        memset(new_centroids, 0, k * dim * sizeof(float));
        memset(counts, 0, k * sizeof(size_t));
        for (size_t i = 0; i < count; ++i) {
            int c = assign[i];
            if (c < 0) continue;
            const float *vec = data + i * dim;
            for (size_t d = 0; d < dim; ++d) {
                new_centroids[(size_t)c * dim + d] += vec[d];
            }
            counts[(size_t)c]++;
        }
        for (size_t c = 0; c < k; ++c) {
            if (counts[c] > 0) {
                for (size_t d = 0; d < dim; ++d) {
                    new_centroids[c * dim + d] /= (float)counts[c];
                }
            }
        }
        memcpy(out_centroids, new_centroids, k * dim * sizeof(float));
    }

    free(assign);
    free(new_centroids);
    free(counts);
    return 0;
}

static int ivfdisk_head_dist_cmp(const void *a, const void *b)
{
    const IVFDiskHeadDist *ha = (const IVFDiskHeadDist *)a;
    const IVFDiskHeadDist *hb = (const IVFDiskHeadDist *)b;
    if (ha->dist < hb->dist) return -1;
    if (ha->dist > hb->dist) return 1;
    return 0;
}

static void ivfdisk_rank_centroids(const GV_IVFDiskIndex *idx, const float *vec,
                                   IVFDiskHeadDist *ranked)
{
    for (size_t c = 0; c < idx->config.nlist; ++c) {
        ranked[c].head_id = c;
        ranked[c].dist = 0.f;
        const float *cent = idx->centroids + c * idx->dimension;
        for (size_t d = 0; d < idx->dimension; ++d) {
            float diff = vec[d] - cent[d];
            ranked[c].dist += diff * diff;
        }
    }
    qsort(ranked, idx->config.nlist, sizeof(IVFDiskHeadDist), ivfdisk_head_dist_cmp);
}

static int ivfdisk_head_over_cap(const GV_IVFDiskIndex *idx, uint64_t head_id)
{
    if (!idx || !idx->catalog || idx->config.max_list_bytes == 0) return 0;
    return posting_catalog_head_byte_total(idx->catalog, head_id) >= idx->config.max_list_bytes;
}

static size_t ivfdisk_pick_insert_heads(GV_IVFDiskIndex *idx, const float *vec,
                                        uint64_t *heads, size_t max_heads)
{
    if (!idx || !vec || !heads || max_heads == 0) return 0;

    IVFDiskHeadDist *ranked =
        (IVFDiskHeadDist *)malloc(idx->config.nlist * sizeof(IVFDiskHeadDist));
    if (!ranked) return 0;

    ivfdisk_rank_centroids(idx, vec, ranked);

    size_t count = 0;
    for (size_t i = 0; i < idx->config.nlist && count < max_heads; ++i) {
        uint64_t hid = ranked[i].head_id;
        if (ivfdisk_head_over_cap(idx, hid)) continue;
        heads[count++] = hid;
        if (count >= 1) break;
    }
    if (count == 0) {
        heads[0] = ranked[0].head_id;
        count = 1;
    }

    if (count < max_heads && idx->config.nlist >= 2 && idx->config.border_ratio > 0.f) {
        float d1 = 0.f;
        for (size_t j = 0; j < idx->config.nlist; ++j) {
            if (ranked[j].head_id == heads[0]) {
                d1 = ranked[j].dist;
                break;
            }
        }
        for (size_t i = 1; i < idx->config.nlist; ++i) {
            if (ranked[i].head_id == heads[0]) continue;
            if (ivfdisk_head_over_cap(idx, ranked[i].head_id)) continue;
            if (d1 <= 0.f || ranked[i].dist <= d1 * idx->config.border_ratio) {
                heads[count++] = ranked[i].head_id;
                break;
            }
        }
    }

    if (count < max_heads && ivfdisk_head_over_cap(idx, heads[0])) {
        for (size_t i = 0; i < idx->config.nlist && count < max_heads; ++i) {
            uint64_t alt = ranked[i].head_id;
            if (alt == heads[0]) continue;
            if (!ivfdisk_head_over_cap(idx, alt)) {
                heads[count++] = alt;
                break;
            }
        }
    }

    free(ranked);
    return count;
}

static size_t ivfdisk_nearest_centroid(const GV_IVFDiskIndex *idx, const float *vec)
{
    IVFDiskHeadDist *ranked =
        (IVFDiskHeadDist *)malloc(idx->config.nlist * sizeof(IVFDiskHeadDist));
    if (!ranked) return 0;
    ivfdisk_rank_centroids(idx, vec, ranked);
    size_t best = ranked[0].head_id;
    free(ranked);
    return best;
}

static int ivfdisk_ensure_loc(GV_IVFDiskIndex *idx, size_t vector_id)
{
    if (vector_id < idx->vector_loc_cap) return 0;
    size_t new_cap = idx->vector_loc_cap ? idx->vector_loc_cap * 2 : 256;
    while (new_cap <= vector_id) new_cap *= 2;
    IVFDiskVectorLoc *tmp =
        (IVFDiskVectorLoc *)realloc(idx->vector_locs, new_cap * sizeof(IVFDiskVectorLoc));
    if (!tmp) return -1;
    memset(tmp + idx->vector_loc_cap, 0,
           (new_cap - idx->vector_loc_cap) * sizeof(IVFDiskVectorLoc));
    idx->vector_locs = tmp;
    idx->vector_loc_cap = new_cap;
    return 0;
}

static void ivfdisk_record_loc(GV_IVFDiskIndex *idx, size_t vector_id,
                               uint64_t head_id, uint64_t secondary_head_id,
                               uint8_t version, int deleted)
{
    if (ivfdisk_ensure_loc(idx, vector_id) != 0) return;
    idx->vector_locs[vector_id].head_id = head_id;
    idx->vector_locs[vector_id].secondary_head_id = secondary_head_id;
    idx->vector_locs[vector_id].version = version;
    idx->vector_locs[vector_id].deleted = (uint8_t)(deleted ? 1 : 0);
}

static int ivfdisk_build_head_hnsw(GV_IVFDiskIndex *idx)
{
    if (!idx || !idx->trained) return -1;
    if (!idx->config.use_hnsw_head) return 0;

    if (idx->head_hnsw) {
        gv_hnsw_destroy(idx->head_hnsw);
        idx->head_hnsw = NULL;
    }

    GV_HNSWConfig hcfg;
    memset(&hcfg, 0, sizeof(hcfg));
    hcfg.M = 16;
    hcfg.efConstruction = 200;
    hcfg.efSearch = idx->config.nprobe * 8;
    if (hcfg.efSearch < 64) hcfg.efSearch = 64;
    hcfg.distance_type = GV_DISTANCE_EUCLIDEAN;

    idx->head_hnsw = gv_hnsw_create(idx->dimension, &hcfg, NULL);
    if (!idx->head_hnsw) return -1;

    for (size_t c = 0; c < idx->config.nlist; ++c) {
        if (gv_hnsw_insert_raw(idx->head_hnsw, idx->centroids + c * idx->dimension,
                               idx->dimension) != 0) {
            gv_hnsw_destroy(idx->head_hnsw);
            idx->head_hnsw = NULL;
            return -1;
        }
    }
    return 0;
}

static int ivfdisk_select_probe_heads(const GV_IVFDiskIndex *idx, const float *query,
                                      IVFDiskHeadDist *heads, size_t *out_probe)
{
    size_t probe = idx->config.nprobe;
    if (probe > idx->config.nlist) probe = idx->config.nlist;

    if (idx->head_hnsw) {
        GV_Vector qv;
        qv.data = (float *)query;
        qv.dimension = idx->dimension;
        qv.metadata = NULL;

        GV_SearchResult *hres =
            (GV_SearchResult *)calloc(probe, sizeof(GV_SearchResult));
        if (!hres) return -1;

        int found = gv_hnsw_search(idx->head_hnsw, &qv, probe, hres,
                                   GV_DISTANCE_EUCLIDEAN, NULL, NULL);
        if (found <= 0) {
            free(hres);
            return -1;
        }
        for (int i = 0; i < found; ++i) {
            heads[i].head_id = hres[i].id;
            heads[i].dist = hres[i].distance;
        }
        *out_probe = (size_t)found;
        free(hres);
        return 0;
    }

    for (size_t c = 0; c < idx->config.nlist; ++c) {
        heads[c].head_id = c;
        heads[c].dist = 0.f;
        const float *cent = idx->centroids + c * idx->dimension;
        for (size_t d = 0; d < idx->dimension; ++d) {
            float diff = query[d] - cent[d];
            heads[c].dist += diff * diff;
        }
    }
    qsort(heads, idx->config.nlist, sizeof(IVFDiskHeadDist), ivfdisk_head_dist_cmp);
    *out_probe = probe;
    return 0;
}

typedef struct {
    GV_IVFDiskIndex *index;
    uint64_t head_id;
} IVFDiskRebuildCtx;

static int ivfdisk_rebuild_visit(void *ctx, const GV_PostingEntry *entry)
{
    IVFDiskRebuildCtx *rc = (IVFDiskRebuildCtx *)ctx;
    if (!entry) return 0;
    size_t vid = (size_t)entry->vector_id;
    if (ivfdisk_ensure_loc(rc->index, vid) != 0) return -1;

    IVFDiskVectorLoc *loc = &rc->index->vector_locs[vid];
    if (entry->version >= loc->version) {
        loc->version = entry->version;
        loc->head_id = rc->head_id;
        loc->deleted = (entry->flags & GV_POSTING_FLAG_DELETED) ? 1 : 0;
    }
    return 0;
}

typedef struct {
    IVFDiskHeapItem *heap;
    size_t *heap_size;
    size_t heap_cap;
    const float *query;
    size_t dimension;
    GV_DistanceType metric;
} IVFDiskSearchCtx;

static int ivfdisk_collect_visit(void *ctx, const GV_PostingEntry *entry)
{
    IVFDiskSearchCtx *sc = (IVFDiskSearchCtx *)ctx;
    if (!entry || !entry->data || (entry->flags & GV_POSTING_FLAG_DELETED)) return 0;

    float dist = ivfdisk_dist_raw(sc->query, entry->data, sc->dimension, sc->metric);
    if (dist < 0.f) return 0;

    ivfdisk_heap_push(sc->heap, sc->heap_size, sc->heap_cap,
                      (IVFDiskHeapItem){ .dist = dist, .id = (size_t)entry->vector_id });
    return 0;
}

static int ivfdisk_append_entry(GV_IVFDiskIndex *index, uint64_t head_id,
                                const float *data, size_t dimension, size_t vector_id,
                                uint8_t version, uint8_t flags, int bump_total)
{
    GV_PostingWriteEntry entry = {
        .vector_id = (uint64_t)vector_id,
        .version = version,
        .flags = flags,
        .data = data,
        .codes = NULL
    };

    GV_PostingSegmentParams params = { .payload_type = GV_POSTING_PAYLOAD_FLOAT };
    if (index->config.use_sq8 && !(flags & GV_POSTING_FLAG_DELETED)) {
        params.payload_type = GV_POSTING_PAYLOAD_SQ8;
    }

    int rc;
    if (params.payload_type == GV_POSTING_PAYLOAD_FLOAT) {
        rc = posting_catalog_append_segment(index->catalog, head_id, &entry, 1, dimension);
    } else {
        rc = posting_catalog_append_segment_ex(index->catalog, head_id, &entry, 1,
                                               dimension, &params);
    }
    if (rc != 0) return -1;

    if (!(flags & GV_POSTING_FLAG_DELETED) && bump_total) {
        index->total_count++;
    }
    return 0;
}

void ivfdisk_config_init(GV_IVFDiskConfig *config)
{
    if (!config) return;
    memset(config, 0, sizeof(*config));
    config->nlist = 64;
    config->nprobe = 4;
    config->train_iters = 15;
    config->cache_size_mb = 64;
    config->sector_size = GV_DISK_SECTOR_SIZE_DEFAULT;
    config->max_list_bytes = 64u * 1024u * 1024u;
    config->head_wal_checkpoint_bytes = GV_IVFDISK_HEAD_WAL_CKPT_BYTES_DEFAULT;
    config->head_checkpoint_interval_sec = GV_IVFDISK_HEAD_CKPT_INTERVAL_SEC_DEFAULT;
    config->head_ratio = 0.2f;
    config->border_ratio = 1.15f;
    config->use_hnsw_head = 0;
    config->use_sq8 = 0;
}

GV_IVFDiskIndex *ivfdisk_create(size_t dimension, const GV_IVFDiskConfig *config)
{
    if (dimension == 0) return NULL;

    GV_IVFDiskConfig cfg;
    ivfdisk_config_init(&cfg);
    if (config) cfg = *config;
    if (!cfg.data_dir || !*cfg.data_dir) return NULL;
    if (cfg.nprobe > cfg.nlist) cfg.nprobe = cfg.nlist;
    if (cfg.nlist == 0) return NULL;
    if (cfg.head_wal_checkpoint_bytes == 0) {
        cfg.head_wal_checkpoint_bytes = GV_IVFDISK_HEAD_WAL_CKPT_BYTES_DEFAULT;
    }
    if (cfg.head_checkpoint_interval_sec == 0) {
        cfg.head_checkpoint_interval_sec = GV_IVFDISK_HEAD_CKPT_INTERVAL_SEC_DEFAULT;
    }

    GV_IVFDiskIndex *idx = (GV_IVFDiskIndex *)calloc(1, sizeof(*idx));
    if (!idx) return NULL;

    idx->dimension = dimension;
    idx->config = cfg;
    idx->head_last_checkpoint_sec = gv_time_now_sec();
    idx->data_dir = gv_dup_cstr(cfg.data_dir);
    idx->centroids = (float *)malloc(cfg.nlist * dimension * sizeof(float));
    if (!idx->data_dir || !idx->centroids) {
        free(idx->data_dir);
        free(idx->centroids);
        free(idx);
        return NULL;
    }

    idx->page_cache = gv_disk_page_cache_create(cfg.cache_size_mb * 1024u * 1024u);
    if (!idx->page_cache) {
        free(idx->data_dir);
        free(idx->centroids);
        free(idx);
        return NULL;
    }
    idx->owns_cache = 1;

    idx->catalog = posting_catalog_open(cfg.data_dir, cfg.sector_size);
    if (!idx->catalog) {
        gv_disk_page_cache_destroy(idx->page_cache);
        free(idx->data_dir);
        free(idx->centroids);
        free(idx);
        return NULL;
    }
    posting_catalog_attach_page_cache(idx->catalog, idx->page_cache);

    return idx;
}

void ivfdisk_destroy(GV_IVFDiskIndex *index)
{
    if (!index) return;
    if (index->head_hnsw) {
        gv_hnsw_destroy(index->head_hnsw);
    }
    posting_catalog_close(index->catalog);
    if (index->owns_cache) {
        gv_disk_page_cache_destroy(index->page_cache);
    }
    free(index->vector_locs);
    free(index->data_dir);
    free(index->centroids);
    free(index);
}

static int ivfdisk_check_head_ratio(const GV_IVFDiskIndex *index)
{
    if (!index || index->config.head_ratio <= 0.f) return 0;
    size_t centroid_bytes = index->config.nlist * index->dimension * sizeof(float);
    size_t budget = (size_t)(index->config.head_ratio *
                             (double)(index->config.cache_size_mb * 1024u * 1024u));
    if (budget > 0 && centroid_bytes > budget) return -1;
    return 0;
}

int ivfdisk_train(GV_IVFDiskIndex *index, const float *data, size_t count)
{
    if (!index || !data || count < index->config.nlist) return -1;
    if (ivfdisk_kmeans(data, count, index->dimension, index->config.nlist,
                       index->config.train_iters, index->centroids) != 0) {
        return -1;
    }
    if (ivfdisk_check_head_ratio(index) != 0) return -1;
    index->trained = 1;
    return ivfdisk_build_head_hnsw(index);
}

int ivfdisk_is_trained(const GV_IVFDiskIndex *index)
{
    return index && index->trained;
}

int ivfdisk_insert_routed(GV_IVFDiskIndex *index, const float *data, size_t dimension,
                          size_t vector_id, uint64_t *out_heads, size_t *out_head_count,
                          size_t out_head_cap)
{
    if (!index || !data || dimension != index->dimension || !index->trained) return -1;

    uint64_t heads[2];
    size_t nh = ivfdisk_pick_insert_heads(index, data, heads, 2);
    if (nh == 0) return -1;

    uint8_t version = 1;
    if (vector_id < index->vector_loc_cap && index->vector_locs[vector_id].version > 0) {
        version = (uint8_t)(index->vector_locs[vector_id].version + 1);
    }

    for (size_t i = 0; i < nh; ++i) {
        if (ivfdisk_append_entry(index, heads[i], data, dimension, vector_id,
                                 version, 0, 0) != 0) {
            return -1;
        }
    }

    uint64_t secondary = (nh > 1) ? heads[1] : GV_IVFDISK_NO_SECONDARY_HEAD;
    ivfdisk_record_loc(index, vector_id, heads[0], secondary, version, 0);
    index->total_count++;

    if (out_heads && out_head_cap > 0) {
        size_t ncopy = nh < out_head_cap ? nh : out_head_cap;
        memcpy(out_heads, heads, ncopy * sizeof(uint64_t));
        if (out_head_count) *out_head_count = ncopy;
    } else if (out_head_count) {
        *out_head_count = nh;
    }
    return 0;
}

int ivfdisk_insert(GV_IVFDiskIndex *index, const float *data, size_t dimension, size_t vector_id)
{
    return ivfdisk_insert_routed(index, data, dimension, vector_id, NULL, NULL, 0);
}

int ivfdisk_insert_to_head(GV_IVFDiskIndex *index, uint64_t head_id, const float *data,
                           size_t dimension, size_t vector_id)
{
    if (!index || !data || dimension != index->dimension || !index->trained) return -1;
    if (head_id >= index->config.nlist) return -1;

    uint8_t version = 1;
    if (vector_id < index->vector_loc_cap && index->vector_locs[vector_id].version > 0) {
        version = (uint8_t)(index->vector_locs[vector_id].version + 1);
    }

    if (ivfdisk_append_entry(index, head_id, data, dimension, vector_id, version, 0, 0) != 0) {
        return -1;
    }
    ivfdisk_record_loc(index, vector_id, head_id, GV_IVFDISK_NO_SECONDARY_HEAD, version, 0);
    return 0;
}

static int ivfdisk_tombstone_head(GV_IVFDiskIndex *index, uint64_t head_id, size_t vector_id,
                                  uint8_t version, const float *payload)
{
    return ivfdisk_append_entry(index, head_id, payload, index->dimension, vector_id,
                                version, GV_POSTING_FLAG_DELETED, 0);
}

int ivfdisk_delete(GV_IVFDiskIndex *index, size_t vector_id, const float *data)
{
    if (!index || !index->trained) return -1;
    if (vector_id >= index->vector_loc_cap ||
        index->vector_locs[vector_id].version == 0 ||
        index->vector_locs[vector_id].deleted) {
        return -1;
    }

    IVFDiskVectorLoc *loc = &index->vector_locs[vector_id];
    uint8_t new_ver = (uint8_t)(loc->version + 1);
    float zeros[1] = {0.f};
    const float *payload = data ? data : zeros;

    if (ivfdisk_tombstone_head(index, loc->head_id, vector_id, new_ver, payload) != 0) {
        return -1;
    }
    if (loc->secondary_head_id != GV_IVFDISK_NO_SECONDARY_HEAD &&
        ivfdisk_tombstone_head(index, loc->secondary_head_id, vector_id, new_ver, payload) != 0) {
        return -1;
    }
    loc->version = new_ver;
    loc->deleted = 1;
    return 0;
}

int ivfdisk_update(GV_IVFDiskIndex *index, size_t vector_id, const float *new_data,
                   size_t dimension)
{
    if (!index || !new_data || dimension != index->dimension || !index->trained) return -1;
    if (vector_id >= index->vector_loc_cap ||
        index->vector_locs[vector_id].version == 0 ||
        index->vector_locs[vector_id].deleted) {
        return -1;
    }

    IVFDiskVectorLoc *loc = &index->vector_locs[vector_id];
    uint8_t new_ver = (uint8_t)(loc->version + 1);
    float zeros[1] = {0.f};

    if (loc->secondary_head_id != GV_IVFDISK_NO_SECONDARY_HEAD) {
        if (ivfdisk_tombstone_head(index, loc->secondary_head_id, vector_id, new_ver, zeros) != 0) {
            return -1;
        }
    }

    uint64_t heads[2];
    size_t nh = ivfdisk_pick_insert_heads(index, new_data, heads, 2);
    if (nh == 0) return -1;

    for (size_t i = 0; i < nh; ++i) {
        if (ivfdisk_append_entry(index, heads[i], new_data, dimension, vector_id,
                                 new_ver, 0, 0) != 0) {
            return -1;
        }
    }

    loc->head_id = heads[0];
    loc->secondary_head_id = (nh > 1) ? heads[1] : GV_IVFDISK_NO_SECONDARY_HEAD;
    loc->version = new_ver;
    return 0;
}

int ivfdisk_rebuild_vector_map(GV_IVFDiskIndex *index)
{
    if (!index || !index->catalog) return -1;
    free(index->vector_locs);
    index->vector_locs = NULL;
    index->vector_loc_cap = 0;

    IVFDiskRebuildCtx ctx = { .index = index, .head_id = 0 };
    for (size_t h = 0; h < index->config.nlist; ++h) {
        ctx.head_id = h;
        if (posting_catalog_visit_head(index->catalog, h, ivfdisk_rebuild_visit, &ctx) != 0) {
            return -1;
        }
    }
    return 0;
}

uint64_t ivfdisk_nearest_head(const GV_IVFDiskIndex *index, const float *data)
{
    if (!index || !data || !index->trained) return 0;
    return (uint64_t)ivfdisk_nearest_centroid(index, data);
}

int ivfdisk_search(GV_IVFDiskIndex *index, const float *query, size_t k,
                   GV_SearchResult *results, GV_DistanceType distance_type)
{
    if (!index || !query || !results || k == 0 || !index->trained) return -1;

    ivfdisk_head_checkpoint_if_needed(index);

    IVFDiskHeadDist *heads = (IVFDiskHeadDist *)malloc(index->config.nlist * sizeof(IVFDiskHeadDist));
    if (!heads) return -1;

    size_t probe = 0;
    if (ivfdisk_select_probe_heads(index, query, heads, &probe) != 0) {
        free(heads);
        return -1;
    }

    IVFDiskHeapItem *heap = (IVFDiskHeapItem *)malloc(k * sizeof(IVFDiskHeapItem));
    if (!heap) {
        free(heads);
        return -1;
    }

    size_t heap_size = 0;
    IVFDiskSearchCtx ctx = {
        .heap = heap,
        .heap_size = &heap_size,
        .heap_cap = k,
        .query = query,
        .dimension = index->dimension,
        .metric = distance_type
    };

    for (size_t i = 0; i < probe; ++i) {
        (void)ivfdisk_maintenance_maybe_merge_head(index, heads[i].head_id, NULL);
        GV_PostingHeadView view;
        memset(&view, 0, sizeof(view));
        if (posting_catalog_materialize_head(index->catalog, heads[i].head_id, &view) != 0) {
            free(heap);
            free(heads);
            return -1;
        }
        for (size_t j = 0; j < view.count; ++j) {
            if (ivfdisk_collect_visit(&ctx, &view.entries[j]) != 0) {
                posting_head_view_free(&view);
                free(heap);
                free(heads);
                return -1;
            }
        }
        posting_head_view_free(&view);
    }
    free(heads);

    int n = (int)heap_size;
    for (int i = n - 1; i >= 0; --i) {
        results[i].distance = heap[0].dist;
        results[i].id = heap[0].id;
        results[i].is_sparse = 0;
        results[i].sparse_vector = NULL;
        results[i].vector = NULL;

        heap[0] = heap[heap_size - 1];
        heap_size--;
        if (heap_size > 0) {
            ivfdisk_heap_sift_down(heap, heap_size, 0);
        }
    }
    free(heap);
    return n;
}

void ivfdisk_set_nprobe(GV_IVFDiskIndex *index, size_t nprobe)
{
    if (!index) return;
    if (nprobe == 0) nprobe = 1;
    if (nprobe > index->config.nlist) nprobe = index->config.nlist;
    index->config.nprobe = nprobe;
}

size_t ivfdisk_get_nprobe(const GV_IVFDiskIndex *index)
{
    return index ? index->config.nprobe : 0;
}

size_t ivfdisk_count(const GV_IVFDiskIndex *index)
{
    return index ? index->total_count : 0;
}

size_t ivfdisk_head_live_count(const GV_IVFDiskIndex *index, uint64_t head_id)
{
    if (!index || !index->catalog) return 0;
    return posting_catalog_head_live_count(index->catalog, head_id);
}

void ivfdisk_get_stats(const GV_IVFDiskIndex *index, GV_IVFDiskStats *out)
{
    if (!index || !out) return;
    memset(out, 0, sizeof(*out));
    out->total_count = index->total_count;
    out->segment_count = posting_catalog_segment_count(index->catalog);
    out->splits = index->maint_splits;
    out->merges = index->maint_merges;
    out->defrags = index->maint_defrags;
    out->reassigns = index->maint_reassigns;
    if (index->page_cache) {
        GV_DiskPageCacheStats cs;
        gv_disk_page_cache_get_stats(index->page_cache, &cs);
        out->cache_hits = cs.cache_hits;
        out->cache_misses = cs.cache_misses;
        out->cached_bytes = cs.used_bytes;
        out->cache_capacity_bytes = cs.max_bytes;
    }
}

int ivfdisk_save(const GV_IVFDiskIndex *index, FILE *out, uint32_t version)
{
    if (!index || !out) return -1;
    (void)version;

    if (write_u32(out, GV_IVFDISK_SAVE_MAGIC) != 0) return -1;
    if (write_u32(out, (uint32_t)index->dimension) != 0) return -1;
    if (write_u32(out, (uint32_t)index->config.nlist) != 0) return -1;
    if (write_u32(out, (uint32_t)index->config.nprobe) != 0) return -1;
    if (write_u32(out, (uint32_t)index->config.train_iters) != 0) return -1;
    if (write_u32(out, (uint32_t)index->config.cache_size_mb) != 0) return -1;
    if (write_u32(out, (uint32_t)index->config.sector_size) != 0) return -1;
    if (write_u64(out, (uint64_t)index->config.max_list_bytes) != 0) return -1;
    if (write_u32(out, (uint32_t)index->trained) != 0) return -1;
    if (write_u32(out, (uint32_t)index->config.use_hnsw_head) != 0) return -1;
    if (write_u32(out, (uint32_t)index->config.use_sq8) != 0) return -1;
    if (fwrite(&index->config.head_ratio, sizeof(float), 1, out) != 1) return -1;
    if (fwrite(&index->config.border_ratio, sizeof(float), 1, out) != 1) return -1;
    if (write_u64(out, (uint64_t)index->total_count) != 0) return -1;

    if (index->trained) {
        size_t nfloats = index->config.nlist * index->dimension;
        if (fwrite(index->centroids, sizeof(float), nfloats, out) != nfloats) return -1;
    }
    return 0;
}

int ivfdisk_load(GV_IVFDiskIndex **index_ptr, FILE *in, size_t dimension,
                 const char *data_dir, uint32_t version)
{
    if (!index_ptr || !in || !data_dir || !*data_dir) return -1;
    (void)version;

    uint32_t magic = 0;
    if (read_u32(in, &magic) != 0 || magic != GV_IVFDISK_SAVE_MAGIC) return -1;

    uint32_t file_dim = 0, nlist = 0, nprobe = 0, train_iters = 0;
    uint32_t cache_mb = 0, sector_size = 0, trained = 0, use_hnsw = 0, use_sq8 = 0;
    uint64_t max_list_bytes = 0, total_count = 0;
    float head_ratio = 0.2f;
    float border_ratio = 1.15f;

    if (read_u32(in, &file_dim) != 0) return -1;
    if (read_u32(in, &nlist) != 0) return -1;
    if (read_u32(in, &nprobe) != 0) return -1;
    if (read_u32(in, &train_iters) != 0) return -1;
    if (read_u32(in, &cache_mb) != 0) return -1;
    if (read_u32(in, &sector_size) != 0) return -1;
    if (read_u64(in, &max_list_bytes) != 0) return -1;
    if (read_u32(in, &trained) != 0) return -1;
    if (read_u32(in, &use_hnsw) != 0) return -1;
    if (read_u32(in, &use_sq8) != 0) return -1;
    if (fread(&head_ratio, sizeof(float), 1, in) != 1) return -1;
    if (fread(&border_ratio, sizeof(float), 1, in) != 1) {
        border_ratio = 1.15f;
    }
    if (read_u64(in, &total_count) != 0) return -1;

    if (dimension != 0 && dimension != (size_t)file_dim) return -1;
    if (nlist == 0) return -1;

    GV_IVFDiskConfig cfg;
    ivfdisk_config_init(&cfg);
    cfg.nlist = nlist;
    cfg.nprobe = nprobe;
    cfg.train_iters = train_iters;
    cfg.cache_size_mb = cache_mb ? cache_mb : cfg.cache_size_mb;
    cfg.sector_size = sector_size ? sector_size : cfg.sector_size;
    cfg.max_list_bytes = (size_t)max_list_bytes;
    cfg.head_ratio = head_ratio;
    cfg.border_ratio = border_ratio;
    cfg.use_hnsw_head = (int)use_hnsw;
    cfg.use_sq8 = (int)use_sq8;
    cfg.data_dir = data_dir;

    GV_IVFDiskIndex *idx = ivfdisk_create((size_t)file_dim, &cfg);
    if (!idx) return -1;

    idx->trained = (int)trained;
    idx->total_count = (size_t)total_count;

    if (trained) {
        size_t nfloats = idx->config.nlist * idx->dimension;
        if (fread(idx->centroids, sizeof(float), nfloats, in) != nfloats) {
            ivfdisk_destroy(idx);
            return -1;
        }
    }

    if (trained && ivfdisk_rebuild_vector_map(idx) != 0) {
        ivfdisk_destroy(idx);
        return -1;
    }
    if (trained && ivfdisk_head_wal_replay(idx) != 0) {
        ivfdisk_destroy(idx);
        return -1;
    }
    if (trained && ivfdisk_build_head_hnsw(idx) != 0) {
        ivfdisk_destroy(idx);
        return -1;
    }

    *index_ptr = idx;
    return 0;
}

size_t ivfdisk_nlist(const GV_IVFDiskIndex *index)
{
    return index ? index->config.nlist : 0;
}

struct GV_PostingCatalog *ivfdisk_catalog(GV_IVFDiskIndex *index)
{
    return index ? index->catalog : NULL;
}

const GV_IVFDiskConfig *ivfdisk_get_config(const GV_IVFDiskIndex *index)
{
    return index ? &index->config : NULL;
}

static int ivfdisk_grow_centroids(GV_IVFDiskIndex *index, size_t new_nlist)
{
    if (!index || new_nlist <= index->config.nlist) return 0;
    float *tmp = (float *)realloc(index->centroids, new_nlist * index->dimension * sizeof(float));
    if (!tmp) return -1;
    memset(tmp + index->config.nlist * index->dimension, 0,
           (new_nlist - index->config.nlist) * index->dimension * sizeof(float));
    index->centroids = tmp;
    index->config.nlist = new_nlist;
    if (index->config.nprobe > index->config.nlist) {
        index->config.nprobe = index->config.nlist;
    }
    return 0;
}

static uint32_t ivfdisk_crc32(const void *data, size_t len)
{
    return gv_crc32_finish(gv_crc32_update(gv_crc32_init(), data, len));
}

static int ivfdisk_head_wal_file_size(GV_IVFDiskIndex *index, size_t *out_bytes)
{
    if (!index || !index->data_dir || !out_bytes) return -1;
    char path[1024];
    if (snprintf(path, sizeof(path), "%s/head_wal.bin", index->data_dir) >= (int)sizeof(path)) {
        return -1;
    }

    FILE *f = fopen(path, "rb");
    if (!f) {
        *out_bytes = 0;
        return 0;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return -1;
    }
    long sz = ftell(f);
    fclose(f);
    if (sz < 0) return -1;
    *out_bytes = (size_t)sz;
    return 0;
}

static int ivfdisk_head_wal_append(GV_IVFDiskIndex *index, uint8_t type,
                                   const void *payload, size_t payload_len)
{
    if (!index || !index->data_dir) return -1;
    char path[1024];
    if (snprintf(path, sizeof(path), "%s/head_wal.bin", index->data_dir) >= (int)sizeof(path)) {
        return -1;
    }

    FILE *f = fopen(path, "ab");
    if (!f) return -1;

    uint32_t magic = GV_HEAD_WAL_MAGIC;
    uint32_t ver = 1;
    uint32_t dim = (uint32_t)index->dimension;
    uint32_t plen = (uint32_t)payload_len;
    uint32_t crc = ivfdisk_crc32(payload, payload_len);

    int rc = 0;
    if (fwrite(&magic, 4, 1, f) != 1 ||
        fwrite(&ver, 4, 1, f) != 1 ||
        fwrite(&dim, 4, 1, f) != 1 ||
        fputc(type, f) == EOF ||
        fwrite(&plen, 4, 1, f) != 1 ||
        (payload_len > 0 && fwrite(payload, 1, payload_len, f) != payload_len) ||
        fwrite(&crc, 4, 1, f) != 1) {
        rc = -1;
    }
    if (fflush(f) != 0 || fclose(f) != 0) rc = -1;
    if (rc == 0) {
        (void)ivfdisk_head_checkpoint_if_needed(index);
    }
    return rc;
}

int ivfdisk_add_centroid(GV_IVFDiskIndex *index, const float *centroid, uint64_t *out_head_id)
{
    if (!index || !centroid || !index->trained) return -1;
    size_t new_id = index->config.nlist;
    if (ivfdisk_grow_centroids(index, new_id + 1) != 0) return -1;
    memcpy(index->centroids + new_id * index->dimension, centroid,
           index->dimension * sizeof(float));

    size_t payload_len = 8 + index->dimension * sizeof(float);
    uint8_t *payload = (uint8_t *)malloc(payload_len);
    if (!payload) return -1;
    uint64_t hid = (uint64_t)new_id;
    memcpy(payload, &hid, 8);
    memcpy(payload + 8, centroid, index->dimension * sizeof(float));
    int rc = ivfdisk_head_wal_append(index, GV_HEAD_WAL_ADD, payload, payload_len);
    free(payload);
    if (rc != 0) return -1;

    if (out_head_id) *out_head_id = hid;
    return ivfdisk_rebuild_head_graph(index);
}

int ivfdisk_set_centroid(GV_IVFDiskIndex *index, uint64_t head_id, const float *centroid)
{
    if (!index || !centroid || head_id >= index->config.nlist) return -1;
    memcpy(index->centroids + head_id * index->dimension, centroid,
           index->dimension * sizeof(float));

    size_t payload_len = 8 + index->dimension * sizeof(float);
    uint8_t *payload = (uint8_t *)malloc(payload_len);
    if (!payload) return -1;
    memcpy(payload, &head_id, 8);
    memcpy(payload + 8, centroid, index->dimension * sizeof(float));
    int rc = ivfdisk_head_wal_append(index, GV_HEAD_WAL_UPDATE, payload, payload_len);
    free(payload);
    if (rc != 0) return -1;
    return ivfdisk_rebuild_head_graph(index);
}

float ivfdisk_centroid_dist_sq(const GV_IVFDiskIndex *index, const float *vec, uint64_t head_id)
{
    if (!index || !vec || head_id >= index->config.nlist) return FLT_MAX;
    const float *cent = index->centroids + head_id * index->dimension;
    float dist = 0.f;
    for (size_t d = 0; d < index->dimension; ++d) {
        float diff = vec[d] - cent[d];
        dist += diff * diff;
    }
    return dist;
}

int ivfdisk_rebuild_head_graph(GV_IVFDiskIndex *index)
{
    return ivfdisk_build_head_hnsw(index);
}

int ivfdisk_head_checkpoint(GV_IVFDiskIndex *index)
{
    if (!index || !index->trained || !index->data_dir) return -1;

    char ckpt_path[1024], wal_path[1024], tmp_path[1040];
    if (snprintf(ckpt_path, sizeof(ckpt_path), "%s/head_checkpoint.bin", index->data_dir) >= (int)sizeof(ckpt_path) ||
        snprintf(wal_path, sizeof(wal_path), "%s/head_wal.bin", index->data_dir) >= (int)sizeof(wal_path) ||
        snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", ckpt_path) >= (int)sizeof(tmp_path)) {
        return -1;
    }

    FILE *f = fopen(tmp_path, "wb");
    if (!f) return -1;

    uint32_t magic = GV_HEAD_CKPT_MAGIC;
    uint32_t ver = 1;
    uint32_t dim = (uint32_t)index->dimension;
    uint64_t nlist = (uint64_t)index->config.nlist;
    size_t nfloats = index->config.nlist * index->dimension;

    int rc = 0;
    if (fwrite(&magic, 4, 1, f) != 1 ||
        fwrite(&ver, 4, 1, f) != 1 ||
        fwrite(&dim, 4, 1, f) != 1 ||
        fwrite(&nlist, 8, 1, f) != 1 ||
        fwrite(index->centroids, sizeof(float), nfloats, f) != nfloats) {
        rc = -1;
    }
    if (rc == 0) {
        uint32_t crc = ivfdisk_crc32((const uint8_t *)index->centroids, nfloats * sizeof(float));
        if (fwrite(&crc, 4, 1, f) != 1) rc = -1;
    }
    if (fflush(f) != 0 || fclose(f) != 0) rc = -1;
    if (rc != 0) {
        remove(tmp_path);
        return -1;
    }
#ifndef _WIN32
    if (rename(tmp_path, ckpt_path) != 0) { remove(tmp_path); return -1; }
#else
    if (MoveFileExA(tmp_path, ckpt_path, MOVEFILE_REPLACE_EXISTING) == 0) {
        remove(tmp_path);
        return -1;
    }
#endif
    remove(wal_path);
    index->head_last_checkpoint_sec = gv_time_now_sec();
    return 0;
}

int ivfdisk_head_checkpoint_if_needed(GV_IVFDiskIndex *index)
{
    if (!index || !index->trained || !index->data_dir) return 0;

    size_t wal_bytes = 0;
    if (ivfdisk_head_wal_file_size(index, &wal_bytes) != 0) return -1;
    if (wal_bytes == 0) return 0;

    if (wal_bytes >= index->config.head_wal_checkpoint_bytes) {
        return ivfdisk_head_checkpoint(index);
    }

    uint64_t now = gv_time_now_sec();
    if (now >= index->head_last_checkpoint_sec &&
        now - index->head_last_checkpoint_sec >= index->config.head_checkpoint_interval_sec) {
        return ivfdisk_head_checkpoint(index);
    }
    return 0;
}

int ivfdisk_head_wal_replay(GV_IVFDiskIndex *index)
{
    if (!index || !index->data_dir) return 0;

    char ckpt_path[1024], wal_path[1024];
    if (snprintf(ckpt_path, sizeof(ckpt_path), "%s/head_checkpoint.bin", index->data_dir) >= (int)sizeof(ckpt_path) ||
        snprintf(wal_path, sizeof(wal_path), "%s/head_wal.bin", index->data_dir) >= (int)sizeof(wal_path)) {
        return -1;
    }

    FILE *ck = fopen(ckpt_path, "rb");
    if (ck) {
        uint32_t magic = 0, ver = 0, dim = 0;
        uint64_t nlist = 0;
        if (fread(&magic, 4, 1, ck) != 1 || magic != GV_HEAD_CKPT_MAGIC ||
            fread(&ver, 4, 1, ck) != 1 ||
            fread(&dim, 4, 1, ck) != 1 || dim != index->dimension ||
            fread(&nlist, 8, 1, ck) != 1 || nlist == 0) {
            fclose(ck);
            return -1;
        }
        if (ivfdisk_grow_centroids(index, (size_t)nlist) != 0) {
            fclose(ck);
            return -1;
        }
        size_t nfloats = (size_t)nlist * index->dimension;
        if (fread(index->centroids, sizeof(float), nfloats, ck) != nfloats) {
            fclose(ck);
            return -1;
        }
        uint32_t stored_crc = 0;
        if (fread(&stored_crc, 4, 1, ck) != 1 ||
            stored_crc != ivfdisk_crc32((const uint8_t *)index->centroids, nfloats * sizeof(float))) {
            fclose(ck);
            return -1;
        }
        fclose(ck);
    }

    FILE *wal = fopen(wal_path, "rb");
    if (!wal) return 0;

    for (;;) {
        uint32_t magic = 0, ver = 0, dim = 0, plen = 0, crc = 0;
        if (fread(&magic, 4, 1, wal) != 1) break;
        if (magic != GV_HEAD_WAL_MAGIC) {
            fclose(wal);
            return -1;
        }
        int type = fgetc(wal);
        if (type == EOF) break;
        if (fread(&ver, 4, 1, wal) != 1 ||
            fread(&dim, 4, 1, wal) != 1 || dim != index->dimension ||
            fread(&plen, 4, 1, wal) != 1) {
            fclose(wal);
            return -1;
        }
        uint8_t *payload = plen > 0 ? (uint8_t *)malloc(plen) : NULL;
        if (plen > 0 && !payload) { fclose(wal); return -1; }
        if (plen > 0 && fread(payload, 1, plen, wal) != plen) {
            free(payload);
            fclose(wal);
            return -1;
        }
        if (fread(&crc, 4, 1, wal) != 1 ||
            (plen > 0 && crc != ivfdisk_crc32(payload, plen))) {
            free(payload);
            fclose(wal);
            return -1;
        }

        if (type == GV_HEAD_WAL_ADD || type == GV_HEAD_WAL_UPDATE) {
            if (plen < 8 + index->dimension * sizeof(float)) {
                free(payload);
                fclose(wal);
                return -1;
            }
            uint64_t hid = 0;
            memcpy(&hid, payload, 8);
            if (type == GV_HEAD_WAL_ADD) {
                if (hid >= index->config.nlist) {
                    if (ivfdisk_grow_centroids(index, (size_t)hid + 1) != 0) {
                        free(payload);
                        fclose(wal);
                        return -1;
                    }
                }
            } else if (hid >= index->config.nlist) {
                free(payload);
                fclose(wal);
                return -1;
            }
            memcpy(index->centroids + hid * index->dimension, payload + 8,
                   index->dimension * sizeof(float));
        } else if (type == GV_HEAD_WAL_SPLIT) {
            if (plen < 16) {
                free(payload);
                fclose(wal);
                return -1;
            }
            uint64_t src = 0, neu = 0;
            memcpy(&src, payload, 8);
            memcpy(&neu, payload + 8, 8);
            (void)src;
            if (neu >= index->config.nlist) {
                if (ivfdisk_grow_centroids(index, (size_t)neu + 1) != 0) {
                    free(payload);
                    fclose(wal);
                    return -1;
                }
            }
            if (plen >= 16 + index->dimension * sizeof(float)) {
                memcpy(index->centroids + neu * index->dimension, payload + 16,
                       index->dimension * sizeof(float));
            }
            if (plen >= 16 + 2 * index->dimension * sizeof(float)) {
                memcpy(index->centroids + src * index->dimension,
                       payload + 16 + index->dimension * sizeof(float),
                       index->dimension * sizeof(float));
            }
        }
        free(payload);
    }
    fclose(wal);
    return 0;
}

size_t ivfdisk_dimension(const GV_IVFDiskIndex *index)
{
    return index ? index->dimension : 0;
}

const char *ivfdisk_data_dir_path(const GV_IVFDiskIndex *index)
{
    return index ? index->data_dir : NULL;
}

int ivfdisk_maint_tombstone(GV_IVFDiskIndex *index, uint64_t head_id, size_t vector_id,
                            uint8_t version, const float *data)
{
    return ivfdisk_tombstone_head(index, head_id, vector_id, version, data);
}

int ivfdisk_maint_append(GV_IVFDiskIndex *index, uint64_t head_id, const float *data,
                         size_t vector_id, uint8_t version)
{
    return ivfdisk_append_entry(index, head_id, data, index->dimension, vector_id, version, 0, 0);
}

void ivfdisk_maint_inc_split(GV_IVFDiskIndex *index) { if (index) index->maint_splits++; }
void ivfdisk_maint_inc_merge(GV_IVFDiskIndex *index) { if (index) index->maint_merges++; }
void ivfdisk_maint_inc_defrag(GV_IVFDiskIndex *index) { if (index) index->maint_defrags++; }
void ivfdisk_maint_add_reassign(GV_IVFDiskIndex *index, size_t count)
{
    if (index) index->maint_reassigns += count;
}

int ivfdisk_maint_wal_split(GV_IVFDiskIndex *index, uint64_t src, uint64_t neu,
                            const float *src_centroid, const float *new_centroid)
{
    if (!index || !src_centroid || !new_centroid) return -1;
    size_t dim = index->dimension;
    size_t payload_len = 16 + 2 * dim * sizeof(float);
    uint8_t *payload = (uint8_t *)malloc(payload_len);
    if (!payload) return -1;
    memcpy(payload, &src, 8);
    memcpy(payload + 8, &neu, 8);
    memcpy(payload + 16, new_centroid, dim * sizeof(float));
    memcpy(payload + 16 + dim * sizeof(float), src_centroid, dim * sizeof(float));
    int rc = ivfdisk_head_wal_append(index, GV_HEAD_WAL_SPLIT, payload, payload_len);
    free(payload);
    return rc;
}
