/**
 * @file posting_list.c
 * @brief Append-only on-disk posting lists for larger-than-RAM vector partitions.
 */

#include "storage/posting_list.h"

#include <errno.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "core/compat.h"
#include "core/utils.h"
#include "storage/disk_page_cache.h"
#include "storage/mmap.h"

#ifndef _WIN32
#include <unistd.h>
#else
#include <direct.h>
#include <io.h>
#define mkdir(path, mode) _mkdir(path)
#define fsync(fd) _commit(fd)
#endif

#define GV_POSTING_SEG_MAGIC       "GVPS"
#define GV_POSTING_CAT_MAGIC       "GVPC"
#define GV_POSTING_CAT_VERSION     1u
#define GV_POSTING_SEG_FMT_V1      1u
#define GV_POSTING_SEG_FMT_V2      2u
#define GV_POSTING_SEG_HDR_SIZE    64u
#define GV_POSTING_CAT_HDR_SIZE    16u
#define GV_POSTING_HDR_PAYLOAD_OFF 40u
#define GV_POSTING_HDR_PQ_M_OFF    44u
#define GV_POSTING_DEFAULT_CACHE_MB 64u

typedef struct {
    uint64_t head_id;
    uint64_t sequence;
    uint64_t byte_len;
    uint32_t live_count;
    char *rel_path;
} PostingSegmentRef;

struct GV_PostingCatalog {
    char *base_dir;
    size_t sector_size;
    PostingSegmentRef *segments;
    size_t segment_count;
    size_t segment_cap;
    GV_DiskPageCache *page_cache;
    GV_DiskPageCache *shared_cache;
    int auto_live_count;
    int reconcile_depth;
};

typedef struct {
    uint32_t format_version;
    uint32_t entry_count;
    uint32_t dimension;
    uint8_t payload_type;
    uint32_t pq_m;
    size_t extension_bytes;
    size_t entries_offset;
    size_t entry_stride;
} PostingSegInfo;

typedef struct {
    GV_PostingVisitFn fn;
    void *ctx;
    const uint8_t *seg_data;
    PostingSegInfo info;
    const float *sq8_min;
    const float *sq8_max;
    const float *pq_codebook;
    float *scratch;
} PostingParseCtx;

static int posting_catalog_reconcile_head_live_counts(GV_PostingCatalog *cat,
                                                      uint64_t head_id, int persist);

static const float *posting_bytes_as_floats(const uint8_t *bytes)
{
    const float *out;
    memcpy(&out, &bytes, sizeof(out));
    return out;
}

static int posting_file_sync(FILE *f)
{
    if (fflush(f) != 0) return -1;
    return fsync(fileno(f)) == 0 ? 0 : -1;
}

static int posting_mkdir_p(const char *path)
{
    if (mkdir(path, 0700) == 0 || errno == EEXIST) return 0;
    return -1;
}

static int posting_join_path(char *buf, size_t buflen, const char *a, const char *b)
{
    if (!buf || buflen == 0 || !a || !b) return -1;
    int n = snprintf(buf, buflen, "%s/%s", a, b);
    return (n < 0 || (size_t)n >= buflen) ? -1 : 0;
}

static uint32_t posting_crc32(const void *data, size_t len)
{
    uint32_t crc = gv_crc32_init();
    crc = gv_crc32_update(crc, data, len);
    return gv_crc32_finish(crc);
}

static size_t posting_align_up(size_t value, size_t align)
{
    if (align == 0) return value;
    return (value + align - 1) / align * align;
}

static GV_DiskPageCache *posting_active_cache(const GV_PostingCatalog *cat)
{
    if (!cat) return NULL;
    return cat->shared_cache ? cat->shared_cache : cat->page_cache;
}

static void posting_cache_make_key(char *key, size_t keylen, const char *path)
{
    snprintf(key, keylen, "post:%s", path);
}

static size_t posting_entry_stride(uint8_t payload_type, size_t dimension, uint32_t pq_m)
{
    switch (payload_type) {
    case GV_POSTING_PAYLOAD_SQ8: return 16u + dimension;
    case GV_POSTING_PAYLOAD_PQ:  return 16u + pq_m;
    default:                     return 16u + dimension * sizeof(float);
    }
}

static size_t posting_extension_bytes(uint8_t payload_type, size_t dimension, uint32_t pq_m)
{
    switch (payload_type) {
    case GV_POSTING_PAYLOAD_SQ8:
        return 2u * dimension * sizeof(float);
    case GV_POSTING_PAYLOAD_PQ:
        if (pq_m == 0 || dimension % pq_m != 0) return 0;
        return (size_t)pq_m * 256u * (dimension / pq_m) * sizeof(float);
    default:
        return 0;
    }
}

static void posting_sq8_quant_row(const float *data, size_t dim,
                                  const float *min_vals, const float *max_vals,
                                  uint8_t *codes)
{
    for (size_t d = 0; d < dim; ++d) {
        float range = max_vals[d] - min_vals[d];
        float norm = range > 1e-8f ? (data[d] - min_vals[d]) / range : 0.f;
        int code = (int)(norm * 255.f + 0.5f);
        if (code < 0) code = 0;
        if (code > 255) code = 255;
        codes[d] = (uint8_t)code;
    }
}

static void posting_sq8_dequant_row(const uint8_t *codes, size_t dim,
                                    const float *min_vals, const float *max_vals,
                                    float *out)
{
    for (size_t d = 0; d < dim; ++d) {
        float t = (float)codes[d] / 255.f;
        out[d] = min_vals[d] + t * (max_vals[d] - min_vals[d]);
    }
}

static void posting_pq_dequant_row(const uint8_t *codes, uint32_t pq_m, size_t dimension,
                                   const float *codebook, float *out)
{
    if (!codes || !codebook || pq_m == 0 || dimension % pq_m != 0) return;
    size_t dsub = dimension / pq_m;
    for (uint32_t mi = 0; mi < pq_m; ++mi) {
        uint8_t code = codes[mi];
        const float *centroid =
            codebook + (size_t)mi * 256u * dsub + (size_t)code * dsub;
        memcpy(out + mi * dsub, centroid, dsub * sizeof(float));
    }
}

static void posting_compute_sq8_minmax(const GV_PostingWriteEntry *entries, size_t entry_count,
                                       size_t dimension, float *min_vals, float *max_vals)
{
    for (size_t d = 0; d < dimension; ++d) {
        min_vals[d] = FLT_MAX;
        max_vals[d] = -FLT_MAX;
    }
    for (size_t i = 0; i < entry_count; ++i) {
        if (!entries[i].data) continue;
        for (size_t d = 0; d < dimension; ++d) {
            float v = entries[i].data[d];
            if (v < min_vals[d]) min_vals[d] = v;
            if (v > max_vals[d]) max_vals[d] = v;
        }
    }
}

static uint64_t posting_next_sequence(const GV_PostingCatalog *cat, uint64_t head_id)
{
    uint64_t next = 0;
    for (size_t i = 0; i < cat->segment_count; ++i) {
        if (cat->segments[i].head_id == head_id && cat->segments[i].sequence >= next) {
            next = cat->segments[i].sequence + 1;
        }
    }
    return next;
}

static int posting_segment_ref_cmp(const void *a, const void *b)
{
    const PostingSegmentRef *sa = (const PostingSegmentRef *)a;
    const PostingSegmentRef *sb = (const PostingSegmentRef *)b;
    if (sa->head_id != sb->head_id) return (sa->head_id < sb->head_id) ? -1 : 1;
    if (sa->sequence != sb->sequence) return (sa->sequence < sb->sequence) ? -1 : 1;
    return 0;
}

static void posting_free_segment_refs(PostingSegmentRef *refs, size_t count)
{
    if (!refs) return;
    for (size_t i = 0; i < count; ++i) free(refs[i].rel_path);
    free(refs);
}

static int posting_catalog_add_ref(GV_PostingCatalog *cat, const PostingSegmentRef *ref)
{
    if (cat->segment_count >= cat->segment_cap) {
        size_t new_cap = cat->segment_cap ? cat->segment_cap * 2 : 16;
        PostingSegmentRef *tmp =
            (PostingSegmentRef *)realloc(cat->segments, new_cap * sizeof(PostingSegmentRef));
        if (!tmp) return -1;
        cat->segments = tmp;
        cat->segment_cap = new_cap;
    }
    cat->segments[cat->segment_count] = *ref;
    cat->segments[cat->segment_count].rel_path = gv_dup_cstr(ref->rel_path);
    if (!cat->segments[cat->segment_count].rel_path) return -1;
    cat->segment_count++;
    return 0;
}

static void posting_write_segment_header(uint8_t *hdr, uint64_t head_id, uint64_t sequence,
                                         uint32_t entry_count, uint32_t dimension,
                                         uint32_t entries_crc, uint8_t payload_type,
                                         uint32_t pq_m)
{
    memset(hdr, 0, GV_POSTING_SEG_HDR_SIZE);
    memcpy(hdr, GV_POSTING_SEG_MAGIC, 4);
    uint32_t version = GV_POSTING_SEG_FMT_V2;
    memcpy(hdr + 4, &version, 4);
    memcpy(hdr + 8, &head_id, 8);
    memcpy(hdr + 16, &sequence, 8);
    memcpy(hdr + 24, &entry_count, 4);
    memcpy(hdr + 28, &dimension, 4);
    memcpy(hdr + 32, &entries_crc, 4);
    hdr[GV_POSTING_HDR_PAYLOAD_OFF] = payload_type;
    memcpy(hdr + GV_POSTING_HDR_PQ_M_OFF, &pq_m, 4);
    uint32_t header_crc = posting_crc32(hdr, 36);
    memcpy(hdr + 36, &header_crc, 4);
}

static int posting_parse_seg_info(const uint8_t *hdr, size_t file_len, size_t sector_size,
                                  size_t max_dimension, PostingSegInfo *info)
{
    if (!hdr || !info || file_len < GV_POSTING_SEG_HDR_SIZE) return -1;
    if (memcmp(hdr, GV_POSTING_SEG_MAGIC, 4) != 0) return -1;

    uint32_t stored_hdr_crc = 0;
    memcpy(&stored_hdr_crc, hdr + 36, 4);
    if (stored_hdr_crc != posting_crc32(hdr, 36)) return -1;

    memset(info, 0, sizeof(*info));
    memcpy(&info->format_version, hdr + 4, 4);
    if (info->format_version != GV_POSTING_SEG_FMT_V1 &&
        info->format_version != GV_POSTING_SEG_FMT_V2) {
        return -1;
    }

    memcpy(&info->entry_count, hdr + 24, 4);
    memcpy(&info->dimension, hdr + 28, 4);
    if (info->dimension == 0 ||
        (max_dimension > 0 && info->dimension > max_dimension) ||
        info->entry_count > 10000000u) {
        return -1;
    }

    if (info->format_version == GV_POSTING_SEG_FMT_V1) {
        info->payload_type = GV_POSTING_PAYLOAD_FLOAT;
        info->pq_m = 0;
    } else {
        info->payload_type = hdr[GV_POSTING_HDR_PAYLOAD_OFF];
        memcpy(&info->pq_m, hdr + GV_POSTING_HDR_PQ_M_OFF, 4);
        if (info->payload_type > GV_POSTING_PAYLOAD_PQ) return -1;
    }

    if (sector_size == 0) sector_size = GV_DISK_SECTOR_SIZE_DEFAULT;
    info->extension_bytes = posting_extension_bytes(info->payload_type, info->dimension, info->pq_m);
    if (info->payload_type == GV_POSTING_PAYLOAD_PQ && info->extension_bytes == 0) return -1;

    if (info->format_version == GV_POSTING_SEG_FMT_V1) {
        info->entries_offset = posting_align_up(GV_POSTING_SEG_HDR_SIZE, sector_size);
    } else {
        info->entries_offset = posting_align_up(GV_POSTING_SEG_HDR_SIZE + info->extension_bytes,
                                                sector_size);
    }
    info->entry_stride = posting_entry_stride(info->payload_type, info->dimension, info->pq_m);
    return 0;
}

static int posting_validate_segment_header(const uint8_t *hdr, size_t file_len,
                                           size_t sector_size, size_t max_dimension,
                                           uint64_t *head_id_out, uint64_t *sequence_out,
                                           uint32_t *entry_count_out, uint32_t *dimension_out,
                                           size_t *entries_offset_out)
{
    (void)head_id_out;
    (void)sequence_out;
    (void)entry_count_out;
    (void)dimension_out;
    (void)entries_offset_out;
    PostingSegInfo info;
    if (posting_parse_seg_info(hdr, file_len, sector_size, max_dimension, &info) != 0) return -1;

    size_t entries_size = (size_t)info.entry_count * info.entry_stride;
    if (info.entries_offset > file_len || entries_size > file_len - info.entries_offset) return -1;

    uint32_t entries_crc = 0;
    memcpy(&entries_crc, hdr + 32, 4);
    if (entries_crc != posting_crc32(hdr + info.entries_offset, entries_size)) return -1;

    if (head_id_out) memcpy(head_id_out, hdr + 8, 8);
    if (sequence_out) memcpy(sequence_out, hdr + 16, 8);
    if (entry_count_out) *entry_count_out = info.entry_count;
    if (dimension_out) *dimension_out = info.dimension;
    if (entries_offset_out) *entries_offset_out = info.entries_offset;
    return 0;
}

static int posting_visit_entry(PostingParseCtx *pc, size_t off)
{
    GV_PostingEntry entry;
    memset(&entry, 0, sizeof(entry));
    const uint8_t *data = pc->seg_data;
    memcpy(&entry.vector_id, data + off, 8);
    entry.version = data[off + 8];
    entry.flags = data[off + 9];
    entry.dimension = pc->info.dimension;
    entry.payload_type = pc->info.payload_type;

    switch (pc->info.payload_type) {
    case GV_POSTING_PAYLOAD_FLOAT:
        entry.data = posting_bytes_as_floats(data + off + 16);
        break;
    case GV_POSTING_PAYLOAD_SQ8:
        entry.codes = data + off + 16;
        entry.code_len = pc->info.dimension;
        if (pc->sq8_min && pc->sq8_max && pc->scratch) {
            posting_sq8_dequant_row(entry.codes, pc->info.dimension,
                                    pc->sq8_min, pc->sq8_max, pc->scratch);
            entry.data = pc->scratch;
        }
        break;
    case GV_POSTING_PAYLOAD_PQ:
        entry.codes = data + off + 16;
        entry.code_len = pc->info.pq_m;
        if (pc->pq_codebook && pc->scratch && entry.codes) {
            posting_pq_dequant_row(entry.codes, pc->info.pq_m, pc->info.dimension,
                                   pc->pq_codebook, pc->scratch);
            entry.data = pc->scratch;
        }
        break;
    default:
        return -1;
    }

    return pc->fn ? pc->fn(pc->ctx, &entry) : 0;
}

static int posting_segment_parse_buffer_impl(const uint8_t *data, size_t len,
                                             size_t sector_size, size_t max_dimension,
                                             GV_PostingVisitFn fn, void *ctx)
{
    if (!data || len == 0) return -1;

    PostingSegInfo info;
    if (posting_parse_seg_info(data, len, sector_size, max_dimension, &info) != 0) return -1;

    size_t entries_size = (size_t)info.entry_count * info.entry_stride;
    if (info.entries_offset > len || entries_size > len - info.entries_offset) return -1;

    uint32_t entries_crc = 0;
    memcpy(&entries_crc, data + 32, 4);
    if (entries_crc != posting_crc32(data + info.entries_offset, entries_size)) return -1;

    const float *sq8_min = NULL;
    const float *sq8_max = NULL;
    const float *pq_codebook = NULL;
    if (info.payload_type == GV_POSTING_PAYLOAD_SQ8) {
        sq8_min = posting_bytes_as_floats(data + GV_POSTING_SEG_HDR_SIZE);
        sq8_max = sq8_min + info.dimension;
    } else if (info.payload_type == GV_POSTING_PAYLOAD_PQ) {
        pq_codebook = posting_bytes_as_floats(data + GV_POSTING_SEG_HDR_SIZE);
    }

    float *scratch = NULL;
    if (fn && (info.payload_type == GV_POSTING_PAYLOAD_SQ8 ||
               info.payload_type == GV_POSTING_PAYLOAD_PQ)) {
        scratch = (float *)malloc(info.dimension * sizeof(float));
        if (!scratch) return -1;
    }

    PostingParseCtx pc = {
        .fn = fn, .ctx = ctx, .seg_data = data, .info = info,
        .sq8_min = sq8_min, .sq8_max = sq8_max, .pq_codebook = pq_codebook,
        .scratch = scratch
    };

    size_t off = info.entries_offset;
    int rc = 0;
    for (uint32_t i = 0; i < info.entry_count; ++i) {
        if (off + info.entry_stride > len) { rc = -1; break; }
        if (posting_visit_entry(&pc, off) != 0) { rc = -1; break; }
        off += info.entry_stride;
    }
    free(scratch);
    return rc;
}

int posting_segment_parse_buffer(const uint8_t *data, size_t len, size_t max_dimension,
                                 GV_PostingVisitFn fn, void *ctx)
{
    return posting_segment_parse_buffer_impl(data, len, 0, max_dimension, fn, ctx);
}

int posting_segment_encode(uint64_t head_id, uint64_t sequence,
                           const GV_PostingWriteEntry *entries, size_t entry_count,
                           size_t dimension, size_t sector_size,
                           uint8_t **out_buf, size_t *out_len)
{
    GV_PostingSegmentParams params = { .payload_type = GV_POSTING_PAYLOAD_FLOAT };
    return posting_segment_encode_ex(head_id, sequence, entries, entry_count, dimension,
                                     sector_size, &params, out_buf, out_len);
}

int posting_segment_encode_ex(uint64_t head_id, uint64_t sequence,
                              const GV_PostingWriteEntry *entries, size_t entry_count,
                              size_t dimension, size_t sector_size,
                              const GV_PostingSegmentParams *params,
                              uint8_t **out_buf, size_t *out_len)
{
    if (!out_buf || !out_len || dimension == 0) return -1;
    if (entry_count > 0 && !entries) return -1;

    GV_PostingSegmentParams local = { .payload_type = GV_POSTING_PAYLOAD_FLOAT };
    if (params) local = *params;
    if (sector_size == 0) sector_size = GV_DISK_SECTOR_SIZE_DEFAULT;

    uint8_t payload_type = (uint8_t)local.payload_type;
    uint32_t pq_m = local.pq_m;
    if (payload_type == GV_POSTING_PAYLOAD_PQ) {
        if (pq_m == 0 || dimension % pq_m != 0 || !local.pq_codebook) return -1;
    }

    size_t extension = posting_extension_bytes(payload_type, dimension, pq_m);
    size_t entries_offset = posting_align_up(GV_POSTING_SEG_HDR_SIZE + extension, sector_size);
    size_t entry_stride = posting_entry_stride(payload_type, dimension, pq_m);
    size_t entries_size = entry_count * entry_stride;
    size_t total = posting_align_up(entries_offset + entries_size, sector_size);

    uint8_t *buf = (uint8_t *)calloc(1, total);
    if (!buf) return -1;

    float *sq8_min = NULL;
    float *sq8_max = NULL;
    if (payload_type == GV_POSTING_PAYLOAD_SQ8 && entry_count > 0) {
        sq8_min = (float *)malloc(2u * dimension * sizeof(float));
        if (!sq8_min) { free(buf); return -1; }
        sq8_max = sq8_min + dimension;
        posting_compute_sq8_minmax(entries, entry_count, dimension, sq8_min, sq8_max);
        memcpy(buf + GV_POSTING_SEG_HDR_SIZE, sq8_min, 2u * dimension * sizeof(float));
    } else if (payload_type == GV_POSTING_PAYLOAD_PQ) {
        memcpy(buf + GV_POSTING_SEG_HDR_SIZE, local.pq_codebook, extension);
    }

    size_t off = entries_offset;
    for (size_t i = 0; i < entry_count; ++i) {
        memcpy(buf + off, &entries[i].vector_id, 8);
        buf[off + 8] = entries[i].version;
        buf[off + 9] = entries[i].flags;
        uint32_t dim_u32 = (uint32_t)dimension;
        memcpy(buf + off + 12, &dim_u32, 4);

        switch (payload_type) {
        case GV_POSTING_PAYLOAD_FLOAT:
            if (!entries[i].data) { free(buf); free(sq8_min); return -1; }
            memcpy(buf + off + 16, entries[i].data, dimension * sizeof(float));
            break;
        case GV_POSTING_PAYLOAD_SQ8:
            if (!entries[i].data) { free(buf); free(sq8_min); return -1; }
            posting_sq8_quant_row(entries[i].data, dimension, sq8_min, sq8_max, buf + off + 16);
            break;
        case GV_POSTING_PAYLOAD_PQ:
            if (!entries[i].codes) { free(buf); free(sq8_min); return -1; }
            memcpy(buf + off + 16, entries[i].codes, pq_m);
            break;
        }
        off += entry_stride;
    }

    uint32_t entries_crc = posting_crc32(buf + entries_offset, entries_size);
    posting_write_segment_header(buf, head_id, sequence, (uint32_t)entry_count,
                                 (uint32_t)dimension, entries_crc, payload_type, pq_m);
    free(sq8_min);
    *out_buf = buf;
    *out_len = total;
    return 0;
}

static int posting_read_file_bytes(const char *path, uint8_t **out_buf, size_t *out_len)
{
    GV_MMap *mm = mmap_open_readonly(path);
    if (mm) {
        const void *data = mmap_data(mm);
        size_t size = mmap_size(mm);
        if (!data || size == 0) { mmap_close(mm); return -1; }
        uint8_t *copy = (uint8_t *)malloc(size);
        if (!copy) { mmap_close(mm); return -1; }
        memcpy(copy, data, size);
        mmap_close(mm);
        *out_buf = copy;
        *out_len = size;
        return 0;
    }

    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long sz = ftell(f);
    if (sz <= 0) { fclose(f); return -1; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return -1; }
    uint8_t *buf = (uint8_t *)malloc((size_t)sz);
    if (!buf) { fclose(f); return -1; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return -1;
    }
    fclose(f);
    *out_buf = buf;
    *out_len = (size_t)sz;
    return 0;
}

static int posting_segment_read_file_cached(GV_PostingCatalog *cat, const char *path,
                                            GV_PostingVisitFn fn, void *ctx)
{
    if (!path) return -1;

    GV_DiskPageCache *cache = posting_active_cache(cat);
    if (cache) {
        GV_DiskPageCacheStats stats;
        gv_disk_page_cache_get_stats(cache, &stats);
        if (stats.max_bytes > 0) {
            char key[1024];
            posting_cache_make_key(key, sizeof(key), path);
            size_t len = 0;
            const uint8_t *cached = gv_disk_page_cache_lookup(cache, key, &len);
            if (cached && len > 0) {
                return posting_segment_parse_buffer_impl(cached, len, cat->sector_size,
                                                         0, fn, ctx);
            }

            uint8_t *buf = NULL;
            if (posting_read_file_bytes(path, &buf, &len) != 0) return -1;
            int rc = posting_segment_parse_buffer_impl(buf, len, cat->sector_size, 0, fn, ctx);
            if (gv_disk_page_cache_insert(cache, key, buf, len) != 0) {
                free(buf);
            }
            return rc;
        }
    }

    uint8_t *buf = NULL;
    size_t len = 0;
    if (posting_read_file_bytes(path, &buf, &len) != 0) return -1;
    int rc = posting_segment_parse_buffer_impl(buf, len, cat ? cat->sector_size : 0, 0, fn, ctx);
    free(buf);
    return rc;
}

int posting_segment_read_file(const char *path, GV_PostingVisitFn fn, void *ctx)
{
    return posting_segment_read_file_cached(NULL, path, fn, ctx);
}

static int posting_buf_append(uint8_t **buf, size_t *len, size_t *cap,
                              const void *src, size_t n)
{
    if (*len + n > *cap) {
        size_t new_cap = *cap ? *cap * 2 : 256;
        while (new_cap < *len + n) new_cap *= 2;
        uint8_t *tmp = (uint8_t *)realloc(*buf, new_cap);
        if (!tmp) return -1;
        *buf = tmp;
        *cap = new_cap;
    }
    memcpy(*buf + *len, src, n);
    *len += n;
    return 0;
}

static int posting_catalog_write_file(const GV_PostingCatalog *cat, const char *path)
{
    size_t cap = 256;
    size_t len = 0;
    uint8_t *buf = (uint8_t *)malloc(cap);
    if (!buf) return -1;

    if (posting_buf_append(&buf, &len, &cap, GV_POSTING_CAT_MAGIC, 4) != 0) goto fail;
    uint32_t version = GV_POSTING_CAT_VERSION;
    if (posting_buf_append(&buf, &len, &cap, &version, 4) != 0) goto fail;
    uint64_t seg_count = (uint64_t)cat->segment_count;
    if (posting_buf_append(&buf, &len, &cap, &seg_count, 8) != 0) goto fail;

    for (size_t i = 0; i < cat->segment_count; ++i) {
        const PostingSegmentRef *ref = &cat->segments[i];
        uint16_t plen = ref->rel_path ? (uint16_t)strlen(ref->rel_path) : 0;
        if (ref->rel_path && strlen(ref->rel_path) > 65535u) goto fail;
        if (posting_buf_append(&buf, &len, &cap, &ref->head_id, 8) != 0) goto fail;
        if (posting_buf_append(&buf, &len, &cap, &ref->sequence, 8) != 0) goto fail;
        if (posting_buf_append(&buf, &len, &cap, &ref->byte_len, 8) != 0) goto fail;
        if (posting_buf_append(&buf, &len, &cap, &ref->live_count, 4) != 0) goto fail;
        if (posting_buf_append(&buf, &len, &cap, &plen, 2) != 0) goto fail;
        if (plen > 0 && posting_buf_append(&buf, &len, &cap, ref->rel_path, plen) != 0) goto fail;
    }

    uint32_t file_crc = posting_crc32(buf, len);
    if (posting_buf_append(&buf, &len, &cap, &file_crc, 4) != 0) goto fail;

    FILE *f = fopen(path, "wb");
    if (!f) goto fail;
    int ok = (fwrite(buf, 1, len, f) == len &&
              posting_file_sync(f) == 0 &&
              fclose(f) == 0);
    free(buf);
    return ok ? 0 : -1;

fail:
    free(buf);
    return -1;
}

int posting_catalog_save(GV_PostingCatalog *cat)
{
    if (!cat || !cat->base_dir) return -1;
    char tmp_path[1024], final_path[1024];
    if (posting_join_path(final_path, sizeof(final_path), cat->base_dir,
                          GV_POSTING_CATALOG_FILENAME) != 0) {
        return -1;
    }
    if (snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", final_path) >= (int)sizeof(tmp_path)) return -1;
    if (posting_catalog_write_file(cat, tmp_path) != 0) {
        remove(tmp_path);
        return -1;
    }
#ifndef _WIN32
    if (rename(tmp_path, final_path) != 0) { remove(tmp_path); return -1; }
#else
    if (MoveFileExA(tmp_path, final_path, MOVEFILE_REPLACE_EXISTING) == 0) {
        remove(tmp_path);
        return -1;
    }
#endif
    return 0;
}

int posting_catalog_load(GV_PostingCatalog *cat)
{
    if (!cat || !cat->base_dir) return -1;

    char path[1024];
    if (posting_join_path(path, sizeof(path), cat->base_dir, GV_POSTING_CATALOG_FILENAME) != 0) {
        return -1;
    }

    FILE *f = fopen(path, "rb");
    if (!f) return (errno == ENOENT) ? 0 : -1;

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, GV_POSTING_CAT_MAGIC, 4) != 0) {
        fclose(f);
        return -1;
    }
    uint32_t version = 0;
    uint64_t segment_count = 0;
    if (read_u32(f, &version) != 0 || version != GV_POSTING_CAT_VERSION ||
        read_u64(f, &segment_count) != 0) {
        fclose(f);
        return -1;
    }

    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long file_size = ftell(f);
    if (file_size < (long)(GV_POSTING_CAT_HDR_SIZE + sizeof(uint32_t))) { fclose(f); return -1; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return -1; }

    size_t body_len = (size_t)file_size - sizeof(uint32_t);
    uint8_t *body = (uint8_t *)malloc(body_len);
    if (!body) { fclose(f); return -1; }
    if (fread(body, 1, body_len, f) != body_len) { free(body); fclose(f); return -1; }
    uint32_t stored_crc = 0;
    if (read_u32(f, &stored_crc) != 0) { free(body); fclose(f); return -1; }
    fclose(f);
    if (posting_crc32(body, body_len) != stored_crc) { free(body); return -1; }

    posting_free_segment_refs(cat->segments, cat->segment_count);
    cat->segments = NULL;
    cat->segment_count = 0;
    cat->segment_cap = 0;

    const uint8_t *p = body + GV_POSTING_CAT_HDR_SIZE;
    const uint8_t *end = body + body_len;
    for (uint64_t i = 0; i < segment_count; ++i) {
        if ((size_t)(end - p) < 30) { free(body); return -1; }
        PostingSegmentRef ref;
        memset(&ref, 0, sizeof(ref));
        memcpy(&ref.head_id, p, 8); p += 8;
        memcpy(&ref.sequence, p, 8); p += 8;
        memcpy(&ref.byte_len, p, 8); p += 8;
        memcpy(&ref.live_count, p, 4); p += 4;
        uint16_t plen = 0;
        memcpy(&plen, p, 2); p += 2;
        if ((size_t)(end - p) < plen) { free(body); return -1; }
        ref.rel_path = (char *)malloc((size_t)plen + 1);
        if (!ref.rel_path) { free(body); return -1; }
        if (plen > 0) { memcpy(ref.rel_path, p, plen); p += plen; }
        ref.rel_path[plen] = '\0';
        if (posting_catalog_add_ref(cat, &ref) != 0) {
            free(ref.rel_path);
            free(body);
            return -1;
        }
        free(ref.rel_path);
    }
    free(body);
    qsort(cat->segments, cat->segment_count, sizeof(PostingSegmentRef), posting_segment_ref_cmp);
    return 0;
}

GV_PostingCatalog *posting_catalog_open(const char *base_dir, size_t sector_size)
{
    if (!base_dir || !*base_dir) return NULL;
    GV_PostingCatalog *cat = (GV_PostingCatalog *)calloc(1, sizeof(GV_PostingCatalog));
    if (!cat) return NULL;
    cat->base_dir = gv_dup_cstr(base_dir);
    cat->sector_size = gv_disk_normalize_sector_size(sector_size);
    if (!cat->base_dir) { free(cat); return NULL; }

    cat->page_cache = gv_disk_page_cache_create(GV_POSTING_DEFAULT_CACHE_MB * 1024u * 1024u);
    if (!cat->page_cache) {
        free(cat->base_dir);
        free(cat);
        return NULL;
    }
    cat->auto_live_count = 1;

    char segments_dir[1024];
    if (posting_mkdir_p(cat->base_dir) != 0 ||
        posting_join_path(segments_dir, sizeof(segments_dir), cat->base_dir,
                          GV_POSTING_SEGMENTS_SUBDIR) != 0 ||
        posting_mkdir_p(segments_dir) != 0) {
        posting_catalog_close(cat);
        return NULL;
    }
    if (posting_catalog_load(cat) != 0) {
        posting_catalog_close(cat);
        return NULL;
    }
    return cat;
}

void posting_catalog_close(GV_PostingCatalog *cat)
{
    if (!cat) return;
    gv_disk_page_cache_destroy(cat->page_cache);
    posting_free_segment_refs(cat->segments, cat->segment_count);
    free(cat->base_dir);
    free(cat);
}

void posting_catalog_attach_page_cache(GV_PostingCatalog *cat, GV_DiskPageCache *cache)
{
    if (!cat) return;
    cat->shared_cache = cache;
}

void posting_catalog_set_cache_mb(GV_PostingCatalog *cat, size_t cache_size_mb)
{
    if (!cat) return;
    GV_DiskPageCache *cache = posting_active_cache(cat);
    if (cache) {
        gv_disk_page_cache_set_max_bytes(cache, cache_size_mb * 1024u * 1024u);
    }
}

void posting_catalog_get_cache_stats(const GV_PostingCatalog *cat, GV_PostingCacheStats *out)
{
    if (!cat || !out) return;
    memset(out, 0, sizeof(*out));
    GV_DiskPageCache *cache = posting_active_cache(cat);
    if (!cache) return;
    GV_DiskPageCacheStats stats;
    gv_disk_page_cache_get_stats(cache, &stats);
    out->cache_hits = stats.cache_hits;
    out->cache_misses = stats.cache_misses;
    out->cached_segments = stats.cached_entries;
    out->cache_capacity = stats.max_bytes;
}

size_t posting_catalog_segment_count(const GV_PostingCatalog *cat)
{
    return cat ? cat->segment_count : 0;
}

size_t posting_catalog_segment_count_for_head(const GV_PostingCatalog *cat, uint64_t head_id)
{
    if (!cat) return 0;
    size_t n = 0;
    for (size_t i = 0; i < cat->segment_count; ++i) {
        if (cat->segments[i].head_id == head_id) n++;
    }
    return n;
}

size_t posting_catalog_head_byte_total(const GV_PostingCatalog *cat, uint64_t head_id)
{
    if (!cat) return 0;
    size_t total = 0;
    for (size_t i = 0; i < cat->segment_count; ++i) {
        if (cat->segments[i].head_id == head_id) {
            total += (size_t)cat->segments[i].byte_len;
        }
    }
    return total;
}

static uint32_t posting_count_append_live(const GV_PostingWriteEntry *entries, size_t entry_count)
{
    uint32_t n = 0;
    for (size_t i = 0; i < entry_count; ++i) {
        if (!(entries[i].flags & GV_POSTING_FLAG_DELETED)) n++;
    }
    return n;
}

int posting_catalog_append_segment(GV_PostingCatalog *cat, uint64_t head_id,
                                   const GV_PostingWriteEntry *entries, size_t entry_count,
                                   size_t dimension)
{
    GV_PostingSegmentParams params = { .payload_type = GV_POSTING_PAYLOAD_FLOAT };
    return posting_catalog_append_segment_ex(cat, head_id, entries, entry_count, dimension, &params);
}

int posting_catalog_append_segment_ex(GV_PostingCatalog *cat, uint64_t head_id,
                                      const GV_PostingWriteEntry *entries, size_t entry_count,
                                      size_t dimension, const GV_PostingSegmentParams *params)
{
    if (!cat || dimension == 0 || (entry_count > 0 && !entries)) return -1;

    GV_PostingSegmentParams local = { .payload_type = GV_POSTING_PAYLOAD_FLOAT };
    if (params) local = *params;

    uint64_t sequence = posting_next_sequence(cat, head_id);
    uint8_t *seg_buf = NULL;
    size_t seg_len = 0;
    if (posting_segment_encode_ex(head_id, sequence, entries, entry_count, dimension,
                                  cat->sector_size, &local, &seg_buf, &seg_len) != 0) {
        return -1;
    }

    char rel_path[256];
    if (snprintf(rel_path, sizeof(rel_path), "%s/head_%llu_seq_%llu.seg",
                 GV_POSTING_SEGMENTS_SUBDIR, (unsigned long long)head_id,
                 (unsigned long long)sequence) >= (int)sizeof(rel_path)) {
        free(seg_buf);
        return -1;
    }

    char abs_path[1024], tmp_path[1040];
    if (posting_join_path(abs_path, sizeof(abs_path), cat->base_dir, rel_path) != 0 ||
        snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", abs_path) >= (int)sizeof(tmp_path)) {
        free(seg_buf);
        return -1;
    }

    FILE *f = fopen(tmp_path, "wb");
    if (!f) { free(seg_buf); return -1; }
    if (fwrite(seg_buf, 1, seg_len, f) != seg_len || posting_file_sync(f) != 0 || fclose(f) != 0) {
        free(seg_buf);
        remove(tmp_path);
        return -1;
    }
    free(seg_buf);

#ifndef _WIN32
    if (rename(tmp_path, abs_path) != 0) { remove(tmp_path); return -1; }
#else
    if (MoveFileExA(tmp_path, abs_path, MOVEFILE_REPLACE_EXISTING) == 0) {
        remove(tmp_path);
        return -1;
    }
#endif

    PostingSegmentRef ref;
    memset(&ref, 0, sizeof(ref));
    ref.head_id = head_id;
    ref.sequence = sequence;
    ref.byte_len = seg_len;
    ref.live_count = posting_count_append_live(entries, entry_count);
    ref.rel_path = rel_path;
    if (posting_catalog_add_ref(cat, &ref) != 0) {
        remove(abs_path);
        return -1;
    }
    qsort(cat->segments, cat->segment_count, sizeof(PostingSegmentRef), posting_segment_ref_cmp);
    return posting_catalog_save(cat);
}

void posting_catalog_set_auto_live_count(GV_PostingCatalog *cat, int enabled)
{
    if (!cat) return;
    cat->auto_live_count = enabled ? 1 : 0;
}

int posting_catalog_get_auto_live_count(const GV_PostingCatalog *cat)
{
    return cat ? cat->auto_live_count : 0;
}

uint32_t posting_catalog_segment_live_count(const GV_PostingCatalog *cat,
                                            uint64_t head_id, uint64_t sequence)
{
    if (!cat) return 0;
    for (size_t i = 0; i < cat->segment_count; ++i) {
        if (cat->segments[i].head_id == head_id && cat->segments[i].sequence == sequence) {
            return cat->segments[i].live_count;
        }
    }
    return 0;
}

static int posting_catalog_visit_head_impl(const GV_PostingCatalog *cat, uint64_t head_id,
                                           GV_PostingVisitFn fn, void *ctx)
{
    if (!cat) return -1;
    for (size_t i = 0; i < cat->segment_count; ++i) {
        if (cat->segments[i].head_id != head_id) continue;
        char abs_path[1024];
        if (posting_join_path(abs_path, sizeof(abs_path), cat->base_dir,
                              cat->segments[i].rel_path) != 0) {
            return -1;
        }
        if (posting_segment_read_file_cached((GV_PostingCatalog *)cat, abs_path, fn, ctx) != 0) {
            return -1;
        }
    }
    return 0;
}

static void posting_catalog_maybe_reconcile_head(GV_PostingCatalog *cat, uint64_t head_id)
{
    if (!cat || !cat->auto_live_count || cat->reconcile_depth > 0) return;
    cat->reconcile_depth++;
    posting_catalog_reconcile_head_live_counts(cat, head_id, 1);
    cat->reconcile_depth--;
}

int posting_catalog_visit_head(GV_PostingCatalog *cat, uint64_t head_id,
                               GV_PostingVisitFn fn, void *ctx)
{
    int rc = posting_catalog_visit_head_impl(cat, head_id, fn, ctx);
    if (rc == 0) posting_catalog_maybe_reconcile_head(cat, head_id);
    return rc;
}

typedef struct {
    uint64_t vector_id;
    uint8_t version;
    uint8_t flags;
    size_t dimension;
    float *data;
} CollectedEntry;

typedef struct {
    CollectedEntry *items;
    size_t count;
    size_t cap;
} CollectState;

static int posting_collect_visit(void *ctx, const GV_PostingEntry *entry)
{
    CollectState *st = (CollectState *)ctx;
    if (!entry || entry->dimension == 0) return -1;
    if (st->count >= st->cap) {
        size_t new_cap = st->cap ? st->cap * 2 : 64;
        CollectedEntry *tmp = (CollectedEntry *)realloc(st->items, new_cap * sizeof(CollectedEntry));
        if (!tmp) return -1;
        st->items = tmp;
        st->cap = new_cap;
    }
    CollectedEntry *ce = &st->items[st->count++];
    ce->vector_id = entry->vector_id;
    ce->version = entry->version;
    ce->flags = entry->flags;
    ce->dimension = entry->dimension;
    if (entry->data) {
        ce->data = (float *)malloc(entry->dimension * sizeof(float));
        if (!ce->data) return -1;
        memcpy(ce->data, entry->data, entry->dimension * sizeof(float));
    } else {
        ce->data = NULL;
    }
    return 0;
}

static void posting_collect_free(CollectState *st)
{
    if (!st) return;
    for (size_t i = 0; i < st->count; ++i) free(st->items[i].data);
    free(st->items);
    st->items = NULL;
    st->count = 0;
    st->cap = 0;
}

static int posting_collected_cmp(const void *a, const void *b)
{
    const CollectedEntry *ea = (const CollectedEntry *)a;
    const CollectedEntry *eb = (const CollectedEntry *)b;
    if (ea->vector_id != eb->vector_id) return (ea->vector_id < eb->vector_id) ? -1 : 1;
    if (ea->version != eb->version) return (ea->version < eb->version) ? -1 : 1;
    return 0;
}

static size_t posting_materialize_live_count(CollectState *collected)
{
    size_t live = 0;
    for (size_t i = 0; i < collected->count; ++i) {
        if (i + 1 < collected->count &&
            collected->items[i].vector_id == collected->items[i + 1].vector_id) {
            continue;
        }
        if (!(collected->items[i].flags & GV_POSTING_FLAG_DELETED)) live++;
    }
    return live;
}

int posting_catalog_materialize_head(GV_PostingCatalog *cat, uint64_t head_id,
                                     GV_PostingHeadView *out)
{
    if (!cat || !out) return -1;
    memset(out, 0, sizeof(*out));

    CollectState collected;
    memset(&collected, 0, sizeof(collected));
    if (posting_catalog_visit_head_impl(cat, head_id, posting_collect_visit, &collected) != 0) {
        posting_collect_free(&collected);
        return -1;
    }
    if (collected.count == 0) {
        posting_catalog_maybe_reconcile_head(cat, head_id);
        return 0;
    }

    qsort(collected.items, collected.count, sizeof(CollectedEntry), posting_collected_cmp);

    size_t dim = collected.items[0].dimension;
    size_t live = posting_materialize_live_count(&collected);
    if (live == 0) {
        posting_collect_free(&collected);
        posting_catalog_maybe_reconcile_head(cat, head_id);
        return 0;
    }

    out->entries = (GV_PostingEntry *)calloc(live, sizeof(GV_PostingEntry));
    out->data_pool = (float *)malloc(live * dim * sizeof(float));
    if (!out->entries || !out->data_pool) {
        posting_head_view_free(out);
        posting_collect_free(&collected);
        return -1;
    }
    out->dimension = dim;

    size_t out_i = 0;
    for (size_t i = 0; i < collected.count; ++i) {
        if (i + 1 < collected.count &&
            collected.items[i].vector_id == collected.items[i + 1].vector_id) {
            continue;
        }
        if (collected.items[i].flags & GV_POSTING_FLAG_DELETED) continue;
        if (!collected.items[i].data) continue;
        out->entries[out_i].vector_id = collected.items[i].vector_id;
        out->entries[out_i].version = collected.items[i].version;
        out->entries[out_i].flags = collected.items[i].flags;
        out->entries[out_i].dimension = dim;
        out->entries[out_i].payload_type = GV_POSTING_PAYLOAD_FLOAT;
        out->entries[out_i].data = out->data_pool + out_i * dim;
        memcpy(out->data_pool + out_i * dim, collected.items[i].data, dim * sizeof(float));
        out_i++;
    }
    out->count = out_i;
    posting_collect_free(&collected);
    posting_catalog_maybe_reconcile_head(cat, head_id);
    return 0;
}

size_t posting_catalog_head_live_count(GV_PostingCatalog *cat, uint64_t head_id)
{
    GV_PostingHeadView view;
    memset(&view, 0, sizeof(view));
    if (posting_catalog_materialize_head(cat, head_id, &view) != 0) return 0;
    size_t n = view.count;
    posting_head_view_free(&view);
    return n;
}

typedef struct {
    uint64_t vector_id;
    uint8_t version;
    size_t segment_index;
} TaggedEntry;

typedef struct {
    TaggedEntry *items;
    size_t count;
    size_t cap;
} TaggedCollect;

typedef struct {
    TaggedCollect *collect;
    size_t segment_index;
} TaggedVisitCtx;

static int posting_tagged_visit_wrapper(void *ctx, const GV_PostingEntry *entry)
{
    TaggedVisitCtx *tv = (TaggedVisitCtx *)ctx;
    if (tv->collect->count >= tv->collect->cap) {
        size_t new_cap = tv->collect->cap ? tv->collect->cap * 2 : 64;
        TaggedEntry *tmp = (TaggedEntry *)realloc(tv->collect->items, new_cap * sizeof(TaggedEntry));
        if (!tmp) return -1;
        tv->collect->items = tmp;
        tv->collect->cap = new_cap;
    }
    tv->collect->items[tv->collect->count++] = (TaggedEntry){
        .vector_id = entry->vector_id,
        .version = entry->version,
        .segment_index = tv->segment_index
    };
    return 0;
}

static int tagged_entry_cmp(const void *a, const void *b)
{
    const TaggedEntry *ea = (const TaggedEntry *)a;
    const TaggedEntry *eb = (const TaggedEntry *)b;
    if (ea->vector_id != eb->vector_id) return (ea->vector_id < eb->vector_id) ? -1 : 1;
    if (ea->version != eb->version) return (ea->version < eb->version) ? -1 : 1;
    return 0;
}

static int posting_catalog_reconcile_head_live_counts(GV_PostingCatalog *cat,
                                                      uint64_t head_id, int persist)
{
    if (!cat) return -1;

    size_t seg_start = cat->segment_count;
    size_t seg_end = 0;
    for (size_t i = 0; i < cat->segment_count; ++i) {
        if (cat->segments[i].head_id == head_id) {
            if (seg_start > i) seg_start = i;
            seg_end = i + 1;
        }
    }
    if (seg_start >= seg_end) return 0;

    TaggedCollect tagged;
    memset(&tagged, 0, sizeof(tagged));

    for (size_t i = seg_start; i < seg_end; ++i) {
        char abs_path[1024];
        if (posting_join_path(abs_path, sizeof(abs_path), cat->base_dir,
                              cat->segments[i].rel_path) != 0) {
            free(tagged.items);
            return -1;
        }
        TaggedVisitCtx tv = { .collect = &tagged, .segment_index = i };
        if (posting_segment_read_file_cached(cat, abs_path, posting_tagged_visit_wrapper, &tv) != 0) {
            free(tagged.items);
            return -1;
        }
    }

    if (tagged.count == 0) return 0;

    qsort(tagged.items, tagged.count, sizeof(TaggedEntry), tagged_entry_cmp);

    uint32_t *seg_counts = (uint32_t *)calloc(cat->segment_count, sizeof(uint32_t));
    if (!seg_counts) {
        free(tagged.items);
        return -1;
    }

    CollectState collected;
    memset(&collected, 0, sizeof(collected));
    if (posting_catalog_visit_head_impl(cat, head_id, posting_collect_visit, &collected) != 0) {
        posting_collect_free(&collected);
        free(seg_counts);
        free(tagged.items);
        return -1;
    }
    qsort(collected.items, collected.count, sizeof(CollectedEntry), posting_collected_cmp);

    for (size_t i = 0; i < collected.count; ++i) {
        if (i + 1 < collected.count &&
            collected.items[i].vector_id == collected.items[i + 1].vector_id) {
            continue;
        }
        if (collected.items[i].flags & GV_POSTING_FLAG_DELETED) continue;

        for (size_t t = 0; t < tagged.count; ++t) {
            if (tagged.items[t].vector_id == collected.items[i].vector_id &&
                tagged.items[t].version == collected.items[i].version) {
                seg_counts[tagged.items[t].segment_index]++;
                break;
            }
        }
    }
    posting_collect_free(&collected);

    int changed = 0;
    for (size_t si = seg_start; si < seg_end; ++si) {
        if (cat->segments[si].live_count != seg_counts[si]) {
            cat->segments[si].live_count = seg_counts[si];
            changed = 1;
        }
    }

    free(seg_counts);
    free(tagged.items);

    if (changed && persist) return posting_catalog_save(cat);
    return 0;
}

int posting_catalog_reconcile_live_counts(GV_PostingCatalog *cat)
{
    if (!cat) return -1;

    int rc = 0;
    uint64_t prev_head = UINT64_MAX;
    for (size_t i = 0; i < cat->segment_count; ++i) {
        if (cat->segments[i].head_id == prev_head) continue;
        prev_head = cat->segments[i].head_id;
        if (posting_catalog_reconcile_head_live_counts(cat, prev_head, 0) != 0) rc = -1;
    }
    if (rc != 0) return rc;
    return posting_catalog_save(cat);
}

typedef struct {
    size_t record_count;
} HeadStatsCtx;

static int posting_head_stats_visit(void *ctx, const GV_PostingEntry *entry)
{
    (void)entry;
    HeadStatsCtx *st = (HeadStatsCtx *)ctx;
    st->record_count++;
    return 0;
}

int posting_catalog_head_stats(GV_PostingCatalog *cat, uint64_t head_id,
                               GV_PostingHeadStats *out)
{
    if (!cat || !out) return -1;
    memset(out, 0, sizeof(*out));

    HeadStatsCtx ctx = {0};
    if (posting_catalog_visit_head_impl(cat, head_id, posting_head_stats_visit, &ctx) != 0) {
        return -1;
    }
    out->record_count = ctx.record_count;
    out->segment_count = posting_catalog_segment_count_for_head(cat, head_id);
    out->byte_total = posting_catalog_head_byte_total(cat, head_id);
    out->live_count = posting_catalog_head_live_count(cat, head_id);
    out->live_ratio = out->record_count > 0
        ? (float)out->live_count / (float)out->record_count
        : 1.f;
    return 0;
}

static int posting_catalog_drop_head_segments(GV_PostingCatalog *cat, uint64_t head_id,
                                              char ***out_paths, size_t *out_count)
{
    if (!cat) return -1;

    char **paths = NULL;
    size_t path_cap = 0;
    size_t path_n = 0;

    PostingSegmentRef *kept = NULL;
    size_t kept_cap = cat->segment_count;
    size_t kept_n = 0;
    if (kept_cap > 0) {
        kept = (PostingSegmentRef *)malloc(kept_cap * sizeof(PostingSegmentRef));
        if (!kept) return -1;
    }

    for (size_t i = 0; i < cat->segment_count; ++i) {
        if (cat->segments[i].head_id == head_id) {
            char abs_path[1024];
            if (posting_join_path(abs_path, sizeof(abs_path), cat->base_dir,
                                  cat->segments[i].rel_path) != 0) {
                for (size_t j = 0; j < path_n; ++j) free(paths[j]);
                free(paths);
                posting_free_segment_refs(kept, kept_n);
                free(kept);
                return -1;
            }
            if (path_n >= path_cap) {
                size_t new_cap = path_cap ? path_cap * 2 : 4;
                char **tmp = (char **)realloc(paths, new_cap * sizeof(char *));
                if (!tmp) {
                    for (size_t j = 0; j < path_n; ++j) free(paths[j]);
                    free(paths);
                    posting_free_segment_refs(kept, kept_n);
                    free(kept);
                    return -1;
                }
                paths = tmp;
                path_cap = new_cap;
            }
            paths[path_n] = gv_dup_cstr(abs_path);
            if (!paths[path_n]) {
                for (size_t j = 0; j < path_n; ++j) free(paths[j]);
                free(paths);
                posting_free_segment_refs(kept, kept_n);
                free(kept);
                return -1;
            }
            path_n++;
            free(cat->segments[i].rel_path);
            GV_DiskPageCache *cache = posting_active_cache(cat);
            if (cache) {
                char key[1024];
                posting_cache_make_key(key, sizeof(key), abs_path);
                gv_disk_page_cache_remove(cache, key);
            }
        } else {
            kept[kept_n++] = cat->segments[i];
        }
    }

    free(cat->segments);
    cat->segments = kept;
    cat->segment_count = kept_n;
    cat->segment_cap = kept_n;

    if (out_paths) {
        *out_paths = paths;
    } else {
        for (size_t i = 0; i < path_n; ++i) {
            remove(paths[i]);
            free(paths[i]);
        }
        free(paths);
        paths = NULL;
    }
    if (out_count) *out_count = path_n;
    return 0;
}

int posting_catalog_compact_head(GV_PostingCatalog *cat, uint64_t head_id,
                                 size_t dimension, int use_sq8)
{
    if (!cat || dimension == 0) return -1;

    GV_PostingHeadView view;
    memset(&view, 0, sizeof(view));
    if (posting_catalog_materialize_head(cat, head_id, &view) != 0) return -1;

    char **old_paths = NULL;
    size_t old_count = 0;
    if (posting_catalog_drop_head_segments(cat, head_id, &old_paths, &old_count) != 0) {
        posting_head_view_free(&view);
        return -1;
    }
    for (size_t i = 0; i < old_count; ++i) {
        remove(old_paths[i]);
        free(old_paths[i]);
    }
    free(old_paths);

    if (view.count == 0) {
        posting_head_view_free(&view);
        return posting_catalog_save(cat);
    }

    GV_PostingWriteEntry *writes =
        (GV_PostingWriteEntry *)calloc(view.count, sizeof(GV_PostingWriteEntry));
    if (!writes) {
        posting_head_view_free(&view);
        return -1;
    }
    for (size_t i = 0; i < view.count; ++i) {
        writes[i].vector_id = view.entries[i].vector_id;
        writes[i].version = view.entries[i].version;
        writes[i].flags = view.entries[i].flags;
        writes[i].data = view.entries[i].data;
    }

    GV_PostingSegmentParams params = { .payload_type = GV_POSTING_PAYLOAD_FLOAT };
    if (use_sq8) params.payload_type = GV_POSTING_PAYLOAD_SQ8;

    int rc;
    if (params.payload_type == GV_POSTING_PAYLOAD_FLOAT) {
        rc = posting_catalog_append_segment(cat, head_id, writes, view.count, dimension);
    } else {
        rc = posting_catalog_append_segment_ex(cat, head_id, writes, view.count,
                                               dimension, &params);
    }

    free(writes);
    posting_head_view_free(&view);
    return rc;
}

int posting_catalog_rewrite_head(GV_PostingCatalog *cat, uint64_t head_id,
                                 const GV_PostingWriteEntry *entries, size_t entry_count,
                                 size_t dimension, int use_sq8)
{
    if (!cat || dimension == 0) return -1;
    if (entry_count > 0 && !entries) return -1;

    char **old_paths = NULL;
    size_t old_count = 0;
    if (posting_catalog_drop_head_segments(cat, head_id, &old_paths, &old_count) != 0) {
        return -1;
    }
    for (size_t i = 0; i < old_count; ++i) {
        remove(old_paths[i]);
        free(old_paths[i]);
    }
    free(old_paths);

    int rc = 0;
    if (entry_count > 0) {
        GV_PostingSegmentParams params = { .payload_type = GV_POSTING_PAYLOAD_FLOAT };
        if (use_sq8) params.payload_type = GV_POSTING_PAYLOAD_SQ8;
        if (params.payload_type == GV_POSTING_PAYLOAD_FLOAT) {
            rc = posting_catalog_append_segment(cat, head_id, entries, entry_count, dimension);
        } else {
            rc = posting_catalog_append_segment_ex(cat, head_id, entries, entry_count,
                                                   dimension, &params);
        }
    } else if (posting_catalog_save(cat) != 0) {
        rc = -1;
    }

    return rc;
}

void posting_head_view_free(GV_PostingHeadView *view)
{
    if (!view) return;
    free(view->entries);
    free(view->data_pool);
    memset(view, 0, sizeof(*view));
}
