/**
 * @file index_maintenance.c
 * @brief SPFresh-lite maintenance for IVFDisk: split, reassign, merge, defrag.
 */

#include "index/index_maintenance.h"
#include "core/memory.h"

#include <float.h>
#include <stdlib.h>
#include <string.h>

#include "core/scope.h"
#include "core/utils.h"
#include "storage/posting_list.h"

typedef struct {
    GV_MaintenanceJobType type;
    uint64_t head_id;
    int priority;
} MaintJob;

typedef struct {
    float *centroids;
    int *assign;
    size_t k;
    size_t dim;
    size_t iters;
} SplitKMeans;

static int split_kmeans_run(SplitKMeans *km, const float *data, size_t count)
{
    if (!km || !data || count < km->k || km->k == 0) return -1;

    for (size_t c = 0; c < km->k; ++c) {
        size_t pick = (count * c) / km->k;
        if (pick >= count) pick = count - 1;
        memcpy(km->centroids + c * km->dim, data + pick * km->dim,
               km->dim * sizeof(float));
    }

    float *new_centroids = (float *)gv_calloc(km->k * km->dim, sizeof(float));
    size_t *counts = (size_t *)gv_calloc(km->k, sizeof(size_t));
    if (!new_centroids || !counts) {
        gv_free(new_centroids);
        gv_free(counts);
        return -1;
    }

    for (size_t iter = 0; iter < km->iters; ++iter) {
        for (size_t i = 0; i < count; ++i) {
            const float *vec = data + i * km->dim;
            float best = FLT_MAX;
            int best_c = 0;
            for (size_t c = 0; c < km->k; ++c) {
                float dist = 0.f;
                const float *cent = km->centroids + c * km->dim;
                for (size_t d = 0; d < km->dim; ++d) {
                    float diff = vec[d] - cent[d];
                    dist += diff * diff;
                }
                if (dist < best) {
                    best = dist;
                    best_c = (int)c;
                }
            }
            km->assign[i] = best_c;
        }

        memset(new_centroids, 0, km->k * km->dim * sizeof(float));
        memset(counts, 0, km->k * sizeof(size_t));
        for (size_t i = 0; i < count; ++i) {
            int c = km->assign[i];
            if (c < 0) continue;
            const float *vec = data + i * km->dim;
            for (size_t d = 0; d < km->dim; ++d) {
                new_centroids[(size_t)c * km->dim + d] += vec[d];
            }
            counts[(size_t)c]++;
        }
        for (size_t c = 0; c < km->k; ++c) {
            if (counts[c] > 0) {
                for (size_t d = 0; d < km->dim; ++d) {
                    new_centroids[c * km->dim + d] /= (float)counts[c];
                }
            }
        }
        memcpy(km->centroids, new_centroids, km->k * km->dim * sizeof(float));
    }

    gv_free(new_centroids);
    gv_free(counts);
    return 0;
}

static int maint_job_cmp(const void *a, const void *b)
{
    const MaintJob *ja = (const MaintJob *)a;
    const MaintJob *jb = (const MaintJob *)b;
    if (ja->priority != jb->priority) return jb->priority - ja->priority;
    if (ja->head_id < jb->head_id) return -1;
    if (ja->head_id > jb->head_id) return 1;
    return 0;
}

static size_t maint_split_k(size_t byte_total, size_t live_count, size_t max_list_bytes)
{
    if (max_list_bytes == 0 || live_count < 2) return 2;
    size_t k = (byte_total + max_list_bytes - 1) / max_list_bytes;
    if (k < 2) k = 2;
    if (k > live_count) k = live_count;
    if (k > 64) k = 64;
    return k;
}

static int maint_split_head(GV_IVFDiskIndex *index, uint64_t head_id)
{
    const GV_IVFDiskConfig *cfg = ivfdisk_get_config(index);
    struct GV_PostingCatalog *cat = ivfdisk_catalog(index);
    size_t dim = ivfdisk_dimension(index);
    if (!cfg || !cat || cfg->max_list_bytes == 0 || dim == 0) return -1;

    GV_PostingHeadStats st;
    if (posting_catalog_head_stats(cat, head_id, &st) != 0) return -1;

    GV_PostingHeadView view;
    memset(&view, 0, sizeof(view));
    if (posting_catalog_materialize_head(cat, head_id, &view) != 0) return -1;
    if (view.count < 2) {
        posting_head_view_free(&view);
        return 0;
    }

    size_t k = maint_split_k(st.byte_total, view.count, cfg->max_list_bytes);

    SplitKMeans km = {
        .centroids = (float *)gv_alloc(k * dim * sizeof(float)),
        .assign = (int *)gv_alloc(view.count * sizeof(int)),
        .k = k,
        .dim = dim,
        .iters = cfg->train_iters ? cfg->train_iters : 8
    };
    if (!km.centroids || !km.assign) {
        gv_free(km.centroids);
        gv_free(km.assign);
        posting_head_view_free(&view);
        return -1;
    }
    if (split_kmeans_run(&km, view.data_pool, view.count) != 0) {
        gv_free(km.centroids);
        gv_free(km.assign);
        posting_head_view_free(&view);
        return -1;
    }

    size_t *counts = (size_t *)gv_calloc(k, sizeof(size_t));
    if (!counts) {
        gv_free(km.centroids);
        gv_free(km.assign);
        posting_head_view_free(&view);
        return -1;
    }
    for (size_t i = 0; i < view.count; ++i) {
        int c = km.assign[i];
        if (c >= 0 && (size_t)c < k) counts[(size_t)c]++;
    }

    size_t nonempty = 0;
    for (size_t c = 0; c < k; ++c) {
        if (counts[c] > 0) nonempty++;
    }
    if (nonempty < 2) {
        gv_free(counts);
        gv_free(km.centroids);
        gv_free(km.assign);
        posting_head_view_free(&view);
        return 0;
    }

    size_t primary = 0;
    for (size_t c = 1; c < k; ++c) {
        if (counts[c] > counts[primary]) primary = c;
    }

    size_t nlist_before = ivfdisk_nlist(index);
    uint64_t *cluster_head = (uint64_t *)gv_calloc(k, sizeof(uint64_t));
    GV_PostingWriteEntry **by_cluster =
        (GV_PostingWriteEntry **)gv_calloc(k, sizeof(GV_PostingWriteEntry *));
    if (!cluster_head || !by_cluster) {
        gv_free(cluster_head);
        gv_free(by_cluster);
        gv_free(counts);
        gv_free(km.centroids);
        gv_free(km.assign);
        posting_head_view_free(&view);
        return -1;
    }
    for (size_t c = 0; c < k; ++c) {
        if (counts[c] == 0) continue;
        by_cluster[c] = (GV_PostingWriteEntry *)gv_calloc(counts[c], sizeof(GV_PostingWriteEntry));
        if (!by_cluster[c]) {
            for (size_t j = 0; j < k; ++j) gv_free(by_cluster[j]);
            gv_free(by_cluster);
            gv_free(cluster_head);
            gv_free(counts);
            gv_free(km.centroids);
            gv_free(km.assign);
            posting_head_view_free(&view);
            return -1;
        }
    }

    size_t *fill = (size_t *)gv_calloc(k, sizeof(size_t));
    if (!fill) {
        for (size_t j = 0; j < k; ++j) gv_free(by_cluster[j]);
        gv_free(by_cluster);
        gv_free(cluster_head);
        gv_free(counts);
        gv_free(km.centroids);
        gv_free(km.assign);
        posting_head_view_free(&view);
        return -1;
    }

    cluster_head[primary] = head_id;
    if (ivfdisk_set_centroid(index, head_id, km.centroids + primary * dim) != 0) {
        goto split_fail;
    }

    for (size_t c = 0; c < k; ++c) {
        if (c == primary || counts[c] == 0) continue;
        uint64_t new_head = 0;
        if (ivfdisk_add_centroid(index, km.centroids + c * dim, &new_head) != 0) {
            goto split_fail;
        }
        cluster_head[c] = new_head;
    }

    for (size_t i = 0; i < view.count; ++i) {
        int c = km.assign[i];
        if (c < 0 || (size_t)c >= k || counts[(size_t)c] == 0) continue;
        size_t slot = fill[(size_t)c]++;
        by_cluster[(size_t)c][slot] = (GV_PostingWriteEntry){
            .vector_id = view.entries[i].vector_id,
            .version = view.entries[i].version,
            .flags = view.entries[i].flags,
            .data = view.entries[i].data
        };
    }

    {
        int rc = 0;
        GV_PostingSegmentParams params = { .payload_type = GV_POSTING_PAYLOAD_FLOAT };
        if (cfg->use_sq8) params.payload_type = GV_POSTING_PAYLOAD_SQ8;

        for (size_t c = 0; c < k; ++c) {
            if (counts[c] == 0) continue;
            if (c == primary) {
                rc = posting_catalog_rewrite_head(cat, head_id, by_cluster[c], counts[c],
                                                  dim, cfg->use_sq8);
            } else if (params.payload_type == GV_POSTING_PAYLOAD_FLOAT) {
                rc = posting_catalog_append_segment(cat, cluster_head[c], by_cluster[c],
                                                    counts[c], dim);
            } else {
                rc = posting_catalog_append_segment_ex(cat, cluster_head[c], by_cluster[c],
                                                       counts[c], dim, &params);
            }
            if (rc != 0) break;
        }
        if (rc != 0) goto split_fail;
    }

    ivfdisk_maint_inc_split(index);
    (void)nlist_before;

    for (size_t j = 0; j < k; ++j) gv_free(by_cluster[j]);
    gv_free(by_cluster);
    gv_free(cluster_head);
    gv_free(fill);
    gv_free(counts);
    gv_free(km.centroids);
    gv_free(km.assign);
    posting_head_view_free(&view);
    return 0;

split_fail:
    for (size_t j = 0; j < k; ++j) gv_free(by_cluster[j]);
    gv_free(by_cluster);
    gv_free(cluster_head);
    gv_free(fill);
    gv_free(counts);
    gv_free(km.centroids);
    gv_free(km.assign);
    posting_head_view_free(&view);
    return -1;
}

static size_t maint_reassign_head(GV_IVFDiskIndex *index, uint64_t head_id)
{
    struct GV_PostingCatalog *cat = ivfdisk_catalog(index);
    if (!cat) return 0;

    GV_PostingHeadView view;
    memset(&view, 0, sizeof(view));
    if (posting_catalog_materialize_head(cat, head_id, &view) != 0) return 0;

    size_t moved = 0;
    for (size_t i = 0; i < view.count; ++i) {
        const float *vec = view.entries[i].data;
        if (!vec) continue;

        float d_here = ivfdisk_centroid_dist_sq(index, vec, head_id);
        uint64_t best = head_id;
        float best_d = d_here;
        for (size_t h = 0; h < ivfdisk_nlist(index); ++h) {
            float d = ivfdisk_centroid_dist_sq(index, vec, h);
            if (d < best_d) {
                best_d = d;
                best = h;
            }
        }
        if (best == head_id) continue;

        size_t vid = (size_t)view.entries[i].vector_id;
        uint8_t new_ver = (uint8_t)(view.entries[i].version + 1);

        if (ivfdisk_maint_tombstone(index, head_id, vid, new_ver, vec) != 0 ||
            ivfdisk_maint_append(index, best, vec, vid, new_ver) != 0) {
            posting_head_view_free(&view);
            return moved;
        }
        moved++;
    }

    posting_head_view_free(&view);
    if (moved > 0) ivfdisk_maint_add_reassign(index, moved);
    return moved;
}

static int maint_merge_head(GV_IVFDiskIndex *index, uint64_t head_id, int defrag)
{
    const GV_IVFDiskConfig *cfg = ivfdisk_get_config(index);
    struct GV_PostingCatalog *cat = ivfdisk_catalog(index);
    if (!cfg || !cat) return -1;

    if (posting_catalog_compact_head(cat, head_id, ivfdisk_dimension(index),
                                     cfg->use_sq8) != 0) {
        return -1;
    }
    if (defrag) ivfdisk_maint_inc_defrag(index);
    else ivfdisk_maint_inc_merge(index);
    return 0;
}

static int maint_head_needs_merge(const GV_PostingHeadStats *st,
                                  const GV_IVFDiskMaintenanceConfig *local)
{
    if (!st || st->live_count == 0) return 0;
    if (st->segment_count > 1 && st->live_ratio < local->live_ratio_threshold) {
        return 1;
    }
    if (st->segment_count > local->segment_count_max) {
        return 1;
    }
    return 0;
}

int ivfdisk_maintenance_maybe_merge_head(GV_IVFDiskIndex *index, uint64_t head_id,
                                         const GV_IVFDiskMaintenanceConfig *config)
{
    if (!index || !ivfdisk_is_trained(index)) return 0;

    GV_IVFDiskMaintenanceConfig local;
    ivfdisk_maintenance_config_init(&local);
    if (config) local = *config;

    struct GV_PostingCatalog *cat = ivfdisk_catalog(index);
    if (!cat) return -1;

    GV_PostingHeadStats st;
    if (posting_catalog_head_stats(cat, head_id, &st) != 0) return -1;
    if (!maint_head_needs_merge(&st, &local)) return 0;

    if (maint_merge_head(index, head_id, st.segment_count > local.segment_count_max) != 0) {
        return -1;
    }
    return 1;
}

void ivfdisk_maintenance_config_init(GV_IVFDiskMaintenanceConfig *config)
{
    if (!config) return;
    memset(config, 0, sizeof(*config));
    config->live_ratio_threshold = 0.5f;
    config->segment_count_max = 4;
    config->max_jobs_per_run = 8;
}

int ivfdisk_maintenance_run(GV_IVFDiskIndex *index,
                            const GV_IVFDiskMaintenanceConfig *config,
                            GV_IVFDiskMaintenanceStats *stats)
{
    if (!index || !ivfdisk_is_trained(index)) return -1;

    GV_IVFDiskMaintenanceConfig local;
    ivfdisk_maintenance_config_init(&local);
    if (config) local = *config;
    if (local.max_jobs_per_run == 0) local.max_jobs_per_run = 8;

    struct GV_PostingCatalog *cat = ivfdisk_catalog(index);
    const GV_IVFDiskConfig *cfg = ivfdisk_get_config(index);
    if (!cat || !cfg) return -1;

    size_t nheads = ivfdisk_nlist(index);
    GV_WITH_ARENA(scratch, 256u * 1024u) {
        MaintJob *jobs =
            (MaintJob *)gv_arena_calloc(&scratch, nheads * 2, sizeof(MaintJob));
        if (!jobs) return -1;
        size_t job_n = 0;

        for (size_t h = 0; h < nheads; ++h) {
            GV_PostingHeadStats st;
            if (posting_catalog_head_stats(cat, h, &st) != 0) continue;
            if (st.live_count == 0) continue;

            if (cfg->max_list_bytes > 0 && st.byte_total >= cfg->max_list_bytes) {
                jobs[job_n++] = (MaintJob){
                    .type = GV_MAINT_JOB_SPLIT, .head_id = h, .priority = 4
                };
                continue;
            }
            if (st.segment_count > 1 &&
                st.live_ratio < local.live_ratio_threshold) {
                jobs[job_n++] = (MaintJob){
                    .type = GV_MAINT_JOB_MERGE, .head_id = h, .priority = 2
                };
                continue;
            }
            if (st.segment_count > local.segment_count_max) {
                jobs[job_n++] = (MaintJob){
                    .type = GV_MAINT_JOB_DEFRAG, .head_id = h, .priority = 1
                };
            }
        }

        qsort(jobs, job_n, sizeof(MaintJob), maint_job_cmp);

        GV_IVFDiskMaintenanceStats local_stats;
        memset(&local_stats, 0, sizeof(local_stats));

        size_t ran = 0;
        for (size_t i = 0; i < job_n && ran < local.max_jobs_per_run; ++i) {
            int rc = 0;
            switch (jobs[i].type) {
            case GV_MAINT_JOB_SPLIT: {
                size_t nlist_before = ivfdisk_nlist(index);
                rc = maint_split_head(index, jobs[i].head_id);
                if (rc == 0) {
                    local_stats.splits++;
                    if (maint_reassign_head(index, jobs[i].head_id) > 0) {
                        local_stats.reassigns++;
                    }
                    for (size_t h = nlist_before; h < ivfdisk_nlist(index); ++h) {
                        if (maint_reassign_head(index, h) > 0) {
                            local_stats.reassigns++;
                        }
                    }
                }
                break;
            }
            case GV_MAINT_JOB_MERGE:
                rc = maint_merge_head(index, jobs[i].head_id, 0);
                if (rc == 0) local_stats.merges++;
                break;
            case GV_MAINT_JOB_DEFRAG:
                rc = maint_merge_head(index, jobs[i].head_id, 1);
                if (rc == 0) local_stats.defrags++;
                break;
            default:
                break;
            }
            if (rc == 0) {
                ran++;
                local_stats.jobs_run++;
            }
        }

        if (local_stats.jobs_run > 0) {
            ivfdisk_rebuild_head_graph(index);
            ivfdisk_rebuild_vector_map(index);
        }
        ivfdisk_head_checkpoint_if_needed(index);

        if (stats) *stats = local_stats;
        return 0;
    }
}
