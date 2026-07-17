#ifndef GIGAVECTOR_GV_IVFDISK_H
#define GIGAVECTOR_GV_IVFDISK_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "core/types.h"
#include "search/distance.h"

struct GV_PostingCatalog;

#ifdef __cplusplus
extern "C" {
#endif

#define GV_IVFDISK_SAVE_MAGIC 0x49464644u /* "IFDD" little-endian */

typedef struct {
    size_t nlist;
    size_t nprobe;
    size_t train_iters;
    size_t cache_size_mb;
    size_t sector_size;
    size_t max_list_bytes;   /**< Soft cap per posting list (Phase 3 split trigger). */
    size_t head_wal_checkpoint_bytes; /**< Head WAL size before checkpoint (default 10MB). */
    size_t head_checkpoint_interval_sec; /**< Max head WAL age before checkpoint (default 300s). */
    float head_ratio;        /**< Target RAM fraction for centroids (advisory). */
    float border_ratio;      /**< Replicate to 2nd head when d2/d1 <= border_ratio (0=off). */
    int use_hnsw_head;       /**< Use HNSW over centroids when nlist is large. */
    int use_sq8;             /**< Store posting payloads as SQ8 (dequantized on search). */
    const char *data_dir;
} GV_IVFDiskConfig;

typedef struct {
    size_t total_count;
    size_t segment_count;
    size_t cache_hits;
    size_t cache_misses;
    size_t cached_bytes;
    size_t cache_capacity_bytes;
    size_t splits;
    size_t merges;
    size_t defrags;
    size_t reassigns;
} GV_IVFDiskStats;

typedef struct GV_IVFDiskIndex GV_IVFDiskIndex;

void ivfdisk_config_init(GV_IVFDiskConfig *config);

GV_IVFDiskIndex *ivfdisk_create(size_t dimension, const GV_IVFDiskConfig *config);
void ivfdisk_destroy(GV_IVFDiskIndex *index);

int ivfdisk_train(GV_IVFDiskIndex *index, const float *data, size_t count);
int ivfdisk_is_trained(const GV_IVFDiskIndex *index);

int ivfdisk_insert(GV_IVFDiskIndex *index, const float *data, size_t dimension, size_t vector_id);
int ivfdisk_insert_to_head(GV_IVFDiskIndex *index, uint64_t head_id, const float *data,
                           size_t dimension, size_t vector_id);

/** Insert using border replication / list-size routing; writes assigned head ids. */
int ivfdisk_insert_routed(GV_IVFDiskIndex *index, const float *data, size_t dimension,
                          size_t vector_id, uint64_t *out_heads, size_t *out_head_count,
                          size_t out_head_cap);

uint64_t ivfdisk_nearest_head(const GV_IVFDiskIndex *index, const float *data);

int ivfdisk_delete(GV_IVFDiskIndex *index, size_t vector_id, const float *data);
int ivfdisk_update(GV_IVFDiskIndex *index, size_t vector_id, const float *new_data, size_t dimension);

/** Rebuild vector_id → head location map by scanning on-disk posting lists. */
int ivfdisk_rebuild_vector_map(GV_IVFDiskIndex *index);

int ivfdisk_search(GV_IVFDiskIndex *index, const float *query, size_t k,
                   GV_SearchResult *results, GV_DistanceType distance_type);

void ivfdisk_set_nprobe(GV_IVFDiskIndex *index, size_t nprobe);
size_t ivfdisk_get_nprobe(const GV_IVFDiskIndex *index);

size_t ivfdisk_count(const GV_IVFDiskIndex *index);
size_t ivfdisk_head_live_count(const GV_IVFDiskIndex *index, uint64_t head_id);
void ivfdisk_get_stats(const GV_IVFDiskIndex *index, GV_IVFDiskStats *out);

int ivfdisk_save(const GV_IVFDiskIndex *index, FILE *out, uint32_t version);
int ivfdisk_load(GV_IVFDiskIndex **index_ptr, FILE *in, size_t dimension,
                 const char *data_dir, uint32_t version);

/** Maintenance helpers (Phase 3). */
size_t ivfdisk_nlist(const GV_IVFDiskIndex *index);
struct GV_PostingCatalog *ivfdisk_catalog(GV_IVFDiskIndex *index);
const GV_IVFDiskConfig *ivfdisk_get_config(const GV_IVFDiskIndex *index);

int ivfdisk_add_centroid(GV_IVFDiskIndex *index, const float *centroid, uint64_t *out_head_id);
int ivfdisk_set_centroid(GV_IVFDiskIndex *index, uint64_t head_id, const float *centroid);
float ivfdisk_centroid_dist_sq(const GV_IVFDiskIndex *index, const float *vec, uint64_t head_id);
int ivfdisk_rebuild_head_graph(GV_IVFDiskIndex *index);

int ivfdisk_head_checkpoint(GV_IVFDiskIndex *index);
/** Checkpoint when head WAL exceeds size or age thresholds (mirrors main WAL policy). */
int ivfdisk_head_checkpoint_if_needed(GV_IVFDiskIndex *index);
int ivfdisk_head_wal_replay(GV_IVFDiskIndex *index);

int ivfdisk_maint_tombstone(GV_IVFDiskIndex *index, uint64_t head_id, size_t vector_id,
                            uint32_t version, const float *data);
int ivfdisk_maint_append(GV_IVFDiskIndex *index, uint64_t head_id, const float *data,
                         size_t vector_id, uint32_t version);
void ivfdisk_maint_inc_split(GV_IVFDiskIndex *index);
void ivfdisk_maint_inc_merge(GV_IVFDiskIndex *index);
void ivfdisk_maint_inc_defrag(GV_IVFDiskIndex *index);
void ivfdisk_maint_add_reassign(GV_IVFDiskIndex *index, size_t count);
size_t ivfdisk_dimension(const GV_IVFDiskIndex *index);
const char *ivfdisk_data_dir_path(const GV_IVFDiskIndex *index);
int ivfdisk_maint_wal_split(GV_IVFDiskIndex *index, uint64_t src, uint64_t neu,
                            const float *src_centroid, const float *new_centroid);

#ifdef __cplusplus
}
#endif

#endif
