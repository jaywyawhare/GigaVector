#ifndef GIGAVECTOR_GV_POSTING_LIST_H
#define GIGAVECTOR_GV_POSTING_LIST_H

#include <stddef.h>
#include <stdint.h>

#include "storage/disk_layout.h"
#include "core/id_bitmap.h"

struct GV_DiskPageCache;

#ifdef __cplusplus
extern "C" {
#endif

/*
 * On-disk posting list format (RFC)
 * =================================
 *
 * Layout: append-only segments per (head_id, sequence) under @c segments/,
 * catalog file @c posting_catalog.bin tracks segment refs.
 *
 * Segment file (.seg), magic @c "GVPS":
 *   [0..3]   magic "GVPS"
 *   [4..7]   format_version (1 = float-only, 2 = typed payloads)
 *   [8..15]  head_id (uint64 LE)
 *   [16..23] sequence (uint64 LE)
 *   [24..27] entry_count (uint32 LE)
 *   [28..31] dimension (uint32 LE)
 *   [32..35] entries_crc32 (CRC of entry bytes only)
 *   [36..39] header_crc32 (CRC of bytes [0..35])
 *   v2 only:
 *   [40]     payload_type (0=float, 1=sq8, 2=pq)
 *   [44..47] pq_m (uint32 LE; PQ only)
 *   [64..]   extension (SQ8: min[d]+max[d] floats; PQ: codebook floats)
 *   entries  sector-aligned; stride 16+dimension (float/sq8) or 16+pq_m (pq)
 *
 * Entry record (per vector in segment):
 *   [0..7]   vector_id (uint64 LE)
 *   [8]      version (uint8)
 *   [9]      flags (bit0 = deleted tombstone)
 *   [10..15] reserved
 *   [16..]   payload (floats, sq8 codes, or pq codes)
 *
 * Catalog (@c "GVPC"), version 1:
 *   magic, version, segment_count, repeated (head_id, sequence, byte_len,
 *   live_count, path_len, rel_path), trailing file_crc32.
 *
 * Merge semantics: same vector_id across segments — highest version wins;
 * @c GV_POSTING_FLAG_DELETED suppresses the id. @c live_count in catalog is
 * reconciled automatically after visit/materialize when enabled (default on).
 */

#define GV_POSTING_CATALOG_FILENAME  "posting_catalog.bin"
#define GV_POSTING_SEGMENTS_SUBDIR   "segments"

#define GV_POSTING_FLAG_DELETED      0x01u

typedef enum {
    GV_POSTING_PAYLOAD_FLOAT = 0,
    GV_POSTING_PAYLOAD_SQ8     = 1,
    GV_POSTING_PAYLOAD_PQ      = 2
} GV_PostingPayloadType;

typedef struct {
    uint64_t vector_id;
    uint32_t version;        /**< Per-vector update counter (32-bit; 0 = unset). */
    uint64_t commit_ts;      /**< Monotonic commit timestamp for MVCC snapshot reads. */
    uint8_t flags;
    uint8_t payload_type;
    size_t dimension;
    const float *data;       /**< Float payload or dequantized view; not owned. */
    const uint8_t *codes;    /**< Raw SQ8 / PQ codes when applicable; not owned. */
    size_t code_len;         /**< Byte length of @p codes. */
} GV_PostingEntry;

typedef struct {
    uint64_t vector_id;
    uint32_t version;        /**< Per-vector update counter (32-bit; 0 = unset). */
    uint64_t commit_ts;      /**< Optional commit timestamp; 0 lets the catalog assign one. */
    uint8_t flags;
    const float *data;       /**< Required for FLOAT / SQ8 source vectors. */
    const uint8_t *codes;    /**< Required for PQ when pre-encoded. */
} GV_PostingWriteEntry;

typedef struct {
    GV_PostingPayloadType payload_type;
    uint32_t pq_m;                 /**< PQ subquantizers (required for PQ). */
    const float *pq_codebook;      /**< m * 256 * (dimension/m) floats for PQ segments. */
} GV_PostingSegmentParams;

typedef struct {
    size_t segment_count;
    size_t byte_total;
    size_t live_count;
    size_t record_count;
    float live_ratio;
} GV_PostingHeadStats;

typedef struct {
    GV_PostingEntry *entries;
    float *data_pool;
    size_t count;
    size_t dimension;
} GV_PostingHeadView;

typedef struct {
    size_t cache_hits;
    size_t cache_misses;
    size_t cached_segments;
    size_t cache_capacity;
} GV_PostingCacheStats;

typedef struct GV_PostingCatalog GV_PostingCatalog;

typedef int (*GV_PostingVisitFn)(void *ctx, const GV_PostingEntry *entry);

GV_PostingCatalog *posting_catalog_open(const char *base_dir, size_t sector_size);
void posting_catalog_close(GV_PostingCatalog *cat);

int posting_catalog_load(GV_PostingCatalog *cat);
int posting_catalog_save(GV_PostingCatalog *cat);

void posting_catalog_set_cache_mb(GV_PostingCatalog *cat, size_t cache_size_mb);
void posting_catalog_get_cache_stats(const GV_PostingCatalog *cat, GV_PostingCacheStats *out);

/** Attach a shared LRU byte cache (not owned; survives catalog close). */
void posting_catalog_attach_page_cache(GV_PostingCatalog *cat, struct GV_DiskPageCache *cache);

void posting_catalog_set_auto_live_count(GV_PostingCatalog *cat, int enabled);
int posting_catalog_get_auto_live_count(const GV_PostingCatalog *cat);

uint32_t posting_catalog_segment_live_count(const GV_PostingCatalog *cat,
                                            uint64_t head_id, uint64_t sequence);

int posting_catalog_append_segment(GV_PostingCatalog *cat, uint64_t head_id,
                                   const GV_PostingWriteEntry *entries,
                                   size_t entry_count, size_t dimension);

int posting_catalog_append_segment_ex(GV_PostingCatalog *cat, uint64_t head_id,
                                      const GV_PostingWriteEntry *entries,
                                      size_t entry_count, size_t dimension,
                                      const GV_PostingSegmentParams *params);

size_t posting_catalog_head_byte_total(const GV_PostingCatalog *cat, uint64_t head_id);
size_t posting_catalog_segment_count_for_head(const GV_PostingCatalog *cat, uint64_t head_id);

/** Per-head segment/byte/live stats (live_ratio = live_count / record_count). */
int posting_catalog_head_stats(GV_PostingCatalog *cat, uint64_t head_id,
                               GV_PostingHeadStats *out);

/**
 * Compact all segments for @p head_id into a single live-only segment.
 * Old segment files are removed after the new segment is persisted.
 */
int posting_catalog_compact_head(GV_PostingCatalog *cat, uint64_t head_id,
                                 size_t dimension, int use_sq8);

/**
 * Incremental rollup: compact @p head_id only when it has accumulated at least
 * @p min_segments delta segments (min 2).  A head already condensed to a single
 * segment is skipped in O(1), avoiding redundant full rewrites.
 */
int posting_catalog_maybe_rollup_head(GV_PostingCatalog *cat, uint64_t head_id,
                                      size_t dimension, int use_sq8, size_t min_segments);

/**
 * Build a compact bitmap of the head's live (non-tombstoned, latest-version)
 * vector ids without materializing payloads.  On success *out owns a new
 * GV_IdBitmap (free with gv_id_bitmap_free); on failure *out is NULL.
 * Intended for IVF multi-probe set intersection/union across probed heads.
 */
int posting_catalog_head_live_ids(GV_PostingCatalog *cat, uint64_t head_id, GV_IdBitmap **out);

/** Replace all segments for @p head_id with a single segment containing @p entries. */
int posting_catalog_rewrite_head(GV_PostingCatalog *cat, uint64_t head_id,
                                 const GV_PostingWriteEntry *entries, size_t entry_count,
                                 size_t dimension, int use_sq8);

size_t posting_catalog_segment_count(const GV_PostingCatalog *cat);

size_t posting_catalog_head_live_count(GV_PostingCatalog *cat, uint64_t head_id);
int posting_catalog_reconcile_live_counts(GV_PostingCatalog *cat);

int posting_catalog_visit_head(GV_PostingCatalog *cat, uint64_t head_id,
                               GV_PostingVisitFn fn, void *ctx);

int posting_catalog_materialize_head(GV_PostingCatalog *cat, uint64_t head_id,
                                     GV_PostingHeadView *out);

void posting_head_view_free(GV_PostingHeadView *view);

int posting_segment_read_file(const char *path, GV_PostingVisitFn fn, void *ctx);

int posting_segment_parse_buffer(const uint8_t *data, size_t len, size_t max_dimension,
                                 GV_PostingVisitFn fn, void *ctx);

int posting_segment_encode(uint64_t head_id, uint64_t sequence,
                           const GV_PostingWriteEntry *entries, size_t entry_count,
                           size_t dimension, size_t sector_size,
                           uint8_t **out_buf, size_t *out_len);

int posting_segment_encode_ex(uint64_t head_id, uint64_t sequence,
                              const GV_PostingWriteEntry *entries, size_t entry_count,
                              size_t dimension, size_t sector_size,
                              const GV_PostingSegmentParams *params,
                              uint64_t commit_ts,
                              uint8_t **out_buf, size_t *out_len);

#ifdef __cplusplus
}
#endif

#endif
