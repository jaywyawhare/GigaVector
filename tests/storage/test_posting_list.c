#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "storage/posting_list.h"
#include "core/utils.h"
#include "../test_tmp.h"

static uint32_t posting_test_crc32(const void *data, size_t len)
{
    uint32_t crc = gv_crc32_init();
    crc = gv_crc32_update(crc, data, len);
    return gv_crc32_finish(crc);
}

#define ASSERT(cond, msg)         \
    do {                          \
        if (!(cond)) {            \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1;            \
        }                         \
    } while (0)

static int posting_test_count_visit(void *ctx, const GV_PostingEntry *entry)
{
    (void)entry;
    (*(size_t *)ctx)++;
    return 0;
}

static int test_posting_segment_encode_parse(void)
{
    GV_PostingWriteEntry entries[2];
    float v0[4] = {1.f, 0.f, 0.f, 0.f};
    float v1[4] = {0.f, 1.f, 0.f, 0.f};
    entries[0].vector_id = 10;
    entries[0].version = 1;
    entries[0].flags = 0;
    entries[0].data = v0;
    entries[1].vector_id = 20;
    entries[1].version = 1;
    entries[1].flags = 0;
    entries[1].data = v1;

    uint8_t *buf = NULL;
    size_t len = 0;
    ASSERT(posting_segment_encode(7, 0, entries, 2, 4, 4096, &buf, &len) == 0, "encode");
    ASSERT(buf != NULL && len >= 4096, "encoded length sector-aligned");

    size_t visit_count = 0;
    ASSERT(posting_segment_parse_buffer(buf, len, 8, posting_test_count_visit, &visit_count) == 0, "parse");
    ASSERT(visit_count == 2, "two entries parsed");

    free(buf);
    return 0;
}

static int test_posting_catalog_append_and_load(void)
{
    char dir[512];
    if (gv_test_make_temp_path(dir, sizeof(dir), "gv_posting_cat", "") != 0) return 0;

    GV_PostingCatalog *cat = posting_catalog_open(dir, 4096);
    ASSERT(cat != NULL, "open catalog");

    float vec[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    GV_PostingWriteEntry e;
    e.vector_id = 42;
    e.version = 1;
    e.flags = 0;
    e.data = vec;

    ASSERT(posting_catalog_append_segment(cat, 0, &e, 1, 8) == 0, "append segment");
    ASSERT(posting_catalog_segment_count(cat) == 1, "one segment");
    posting_catalog_close(cat);

    cat = posting_catalog_open(dir, 4096);
    ASSERT(cat != NULL, "reopen catalog");
    ASSERT(posting_catalog_segment_count(cat) == 1, "segment persisted");

    GV_PostingHeadView view;
    memset(&view, 0, sizeof(view));
    ASSERT(posting_catalog_materialize_head(cat, 0, &view) == 0, "materialize");
    ASSERT(view.count == 1, "one live vector");
    ASSERT(view.entries[0].vector_id == 42, "vector id");
    ASSERT(fabsf(view.entries[0].data[0] - 0.1f) < 1e-5f, "vector data");

    posting_head_view_free(&view);
    posting_catalog_close(cat);
    return 0;
}

static int test_posting_version_skip(void)
{
    char dir[512];
    if (gv_test_make_temp_path(dir, sizeof(dir), "gv_posting_ver", "") != 0) return 0;

    GV_PostingCatalog *cat = posting_catalog_open(dir, 4096);
    ASSERT(cat != NULL, "open");

    float old_v[4] = {1.f, 0.f, 0.f, 0.f};
    float new_v[4] = {0.f, 1.f, 0.f, 0.f};

    GV_PostingWriteEntry e1 = { .vector_id = 100, .version = 1, .flags = 0, .data = old_v };
    GV_PostingWriteEntry e2 = { .vector_id = 100, .version = 2, .flags = 0, .data = new_v };

    ASSERT(posting_catalog_append_segment(cat, 1, &e1, 1, 4) == 0, "append v1");
    ASSERT(posting_catalog_append_segment(cat, 1, &e2, 1, 4) == 0, "append v2");

    GV_PostingHeadView view;
    memset(&view, 0, sizeof(view));
    ASSERT(posting_catalog_materialize_head(cat, 1, &view) == 0, "materialize");
    ASSERT(view.count == 1, "dedup to one vector");
    ASSERT(view.entries[0].version == 2, "max version kept");
    ASSERT(fabsf(view.entries[0].data[1] - 1.f) < 1e-5f, "new vector data");

    posting_head_view_free(&view);
    posting_catalog_close(cat);
    return 0;
}

static int test_posting_deleted_tombstone(void)
{
    char dir[512];
    if (gv_test_make_temp_path(dir, sizeof(dir), "gv_posting_del", "") != 0) return 0;

    GV_PostingCatalog *cat = posting_catalog_open(dir, 4096);
    ASSERT(cat != NULL, "open");

    float v[4] = {1.f, 2.f, 3.f, 4.f};
    GV_PostingWriteEntry live = { .vector_id = 5, .version = 1, .flags = 0, .data = v };
    GV_PostingWriteEntry del = { .vector_id = 5, .version = 2, .flags = GV_POSTING_FLAG_DELETED, .data = v };

    ASSERT(posting_catalog_append_segment(cat, 2, &live, 1, 4) == 0, "append live");
    ASSERT(posting_catalog_append_segment(cat, 2, &del, 1, 4) == 0, "append tombstone");

    GV_PostingHeadView view;
    memset(&view, 0, sizeof(view));
    ASSERT(posting_catalog_materialize_head(cat, 2, &view) == 0, "materialize");
    ASSERT(view.count == 0, "deleted vector excluded");

    posting_head_view_free(&view);
    posting_catalog_close(cat);
    return 0;
}

static int test_posting_corrupt_segment(void)
{
    uint8_t bad[128];
    memset(bad, 0, sizeof(bad));
    memcpy(bad, "GVPS", 4);
    ASSERT(posting_segment_parse_buffer(bad, sizeof(bad), 128, NULL, NULL) != 0, "reject corrupt");
    return 0;
}

static int test_posting_bulk_append(void)
{
    char dir[512];
    if (gv_test_make_temp_path(dir, sizeof(dir), "gv_posting_bulk", "") != 0) return 0;

    GV_PostingCatalog *cat = posting_catalog_open(dir, 4096);
    ASSERT(cat != NULL, "open");

    const size_t n = 10000;
    const size_t dim = 16;
    GV_PostingWriteEntry *batch = (GV_PostingWriteEntry *)calloc(n, sizeof(GV_PostingWriteEntry));
    float *data = (float *)malloc(n * dim * sizeof(float));
    ASSERT(batch && data, "alloc batch");

    for (size_t i = 0; i < n; ++i) {
        batch[i].vector_id = (uint64_t)(i + 1);
        batch[i].version = 1;
        batch[i].flags = 0;
        batch[i].data = data + i * dim;
        for (size_t d = 0; d < dim; ++d) {
            data[i * dim + d] = (float)(i + d) * 0.001f;
        }
    }

    ASSERT(posting_catalog_append_segment(cat, 99, batch, n, dim) == 0, "append 10k vectors");
    ASSERT(posting_catalog_segment_count_for_head(cat, 99) == 1, "single segment");

    GV_PostingHeadView view;
    memset(&view, 0, sizeof(view));
    ASSERT(posting_catalog_materialize_head(cat, 99, &view) == 0, "materialize 10k");
    ASSERT(view.count == n, "all vectors materialized");
    ASSERT(view.entries[0].vector_id == 1, "first id");
    ASSERT(view.entries[n - 1].vector_id == n, "last id");

    posting_head_view_free(&view);
    free(batch);
    free(data);
    posting_catalog_close(cat);
    return 0;
}

static int test_posting_multi_segment_merge(void)
{
    char dir[512];
    if (gv_test_make_temp_path(dir, sizeof(dir), "gv_posting_multi", "") != 0) return 0;

    GV_PostingCatalog *cat = posting_catalog_open(dir, 4096);
    ASSERT(cat != NULL, "open");

    float a[4] = {1, 0, 0, 0};
    float b[4] = {0, 1, 0, 0};
    GV_PostingWriteEntry e1 = { .vector_id = 1, .version = 1, .flags = 0, .data = a };
    GV_PostingWriteEntry e2 = { .vector_id = 2, .version = 1, .flags = 0, .data = b };

    ASSERT(posting_catalog_append_segment(cat, 3, &e1, 1, 4) == 0, "seg0");
    ASSERT(posting_catalog_append_segment(cat, 3, &e2, 1, 4) == 0, "seg1");
    ASSERT(posting_catalog_segment_count_for_head(cat, 3) == 2, "two segments");

    GV_PostingHeadView view;
    memset(&view, 0, sizeof(view));
    ASSERT(posting_catalog_materialize_head(cat, 3, &view) == 0, "merge segments");
    ASSERT(view.count == 2, "two vectors");

    posting_head_view_free(&view);
    posting_catalog_close(cat);
    return 0;
}

static int test_posting_sq8_payload(void)
{
    char dir[512];
    if (gv_test_make_temp_path(dir, sizeof(dir), "gv_posting_sq8", "") != 0) return 0;

    GV_PostingCatalog *cat = posting_catalog_open(dir, 4096);
    ASSERT(cat != NULL, "open");

    float v[4] = {0.f, 0.5f, 1.f, -1.f};
    GV_PostingWriteEntry e = { .vector_id = 7, .version = 1, .flags = 0, .data = v };
    GV_PostingSegmentParams params = { .payload_type = GV_POSTING_PAYLOAD_SQ8 };

    ASSERT(posting_catalog_append_segment_ex(cat, 4, &e, 1, 4, &params) == 0, "append sq8");

    GV_PostingHeadView view;
    memset(&view, 0, sizeof(view));
    ASSERT(posting_catalog_materialize_head(cat, 4, &view) == 0, "materialize sq8");
    ASSERT(view.count == 1, "one vector");
    ASSERT(fabsf(view.entries[0].data[1] - 0.5f) < 0.02f, "sq8 round-trip");

    posting_head_view_free(&view);
    posting_catalog_close(cat);
    return 0;
}

static int test_posting_corrupt_catalog(void)
{
    char dir[512];
    if (gv_test_make_temp_path(dir, sizeof(dir), "gv_posting_badcat", "") != 0) return 0;

    GV_PostingCatalog *cat = posting_catalog_open(dir, 4096);
    ASSERT(cat != NULL, "open");
    float v[4] = {1, 0, 0, 0};
    GV_PostingWriteEntry e = { .vector_id = 1, .version = 1, .flags = 0, .data = v };
    ASSERT(posting_catalog_append_segment(cat, 0, &e, 1, 4) == 0, "append");
    posting_catalog_close(cat);

    char cat_path[640];
    snprintf(cat_path, sizeof(cat_path), "%s/%s", dir, GV_POSTING_CATALOG_FILENAME);
    FILE *f = fopen(cat_path, "r+b");
    ASSERT(f != NULL, "open catalog for corrupt");
    fputc('X', f);
    fclose(f);

    cat = posting_catalog_open(dir, 4096);
    ASSERT(cat == NULL, "reject corrupt catalog");
    return 0;
}

static int test_posting_live_count_reconcile(void)
{
    char dir[512];
    if (gv_test_make_temp_path(dir, sizeof(dir), "gv_posting_live", "") != 0) return 0;

    GV_PostingCatalog *cat = posting_catalog_open(dir, 4096);
    ASSERT(cat != NULL, "open");

    float old_v[4] = {1, 0, 0, 0};
    float new_v[4] = {0, 1, 0, 0};
    GV_PostingWriteEntry e1 = { .vector_id = 50, .version = 1, .flags = 0, .data = old_v };
    GV_PostingWriteEntry e2 = { .vector_id = 50, .version = 2, .flags = 0, .data = new_v };

    ASSERT(posting_catalog_append_segment(cat, 5, &e1, 1, 4) == 0, "append v1");
    ASSERT(posting_catalog_append_segment(cat, 5, &e2, 1, 4) == 0, "append v2");
    ASSERT(posting_catalog_head_live_count(cat, 5) == 1, "one live vector");

    ASSERT(posting_catalog_reconcile_live_counts(cat) == 0, "reconcile");
    ASSERT(posting_catalog_segment_count_for_head(cat, 5) == 2, "still two segments");

    posting_catalog_close(cat);
    return 0;
}

static int test_posting_segment_cache(void)
{
    char dir[512];
    if (gv_test_make_temp_path(dir, sizeof(dir), "gv_posting_cache", "") != 0) return 0;

    GV_PostingCatalog *cat = posting_catalog_open(dir, 4096);
    ASSERT(cat != NULL, "open");
    posting_catalog_set_cache_mb(cat, 16);

    float v[4] = {1, 2, 3, 4};
    GV_PostingWriteEntry e = { .vector_id = 1, .version = 1, .flags = 0, .data = v };
    ASSERT(posting_catalog_append_segment(cat, 6, &e, 1, 4) == 0, "append");

    GV_PostingHeadView view;
    memset(&view, 0, sizeof(view));
    ASSERT(posting_catalog_materialize_head(cat, 6, &view) == 0, "materialize pass 1");
    posting_head_view_free(&view);

    GV_PostingCacheStats stats;
    posting_catalog_get_cache_stats(cat, &stats);
    ASSERT(stats.cache_misses >= 1, "first read misses cache");

    memset(&view, 0, sizeof(view));
    ASSERT(posting_catalog_materialize_head(cat, 6, &view) == 0, "materialize pass 2");
    posting_head_view_free(&view);

    posting_catalog_get_cache_stats(cat, &stats);
    ASSERT(stats.cache_hits >= 1, "second read hits cache");

    posting_catalog_close(cat);
    return 0;
}

static int test_posting_v1_segment_compat(void)
{
    float v[4] = {0.25f, 0.5f, 0.75f, 1.f};
    GV_PostingWriteEntry e = { .vector_id = 9, .version = 1, .flags = 0, .data = v };
    uint8_t *buf = NULL;
    size_t len = 0;
    ASSERT(posting_segment_encode(0, 0, &e, 1, 4, 4096, &buf, &len) == 0, "encode v2 float");

    uint32_t v1 = 1;
    memcpy(buf + 4, &v1, 4);
    uint32_t hdr_crc = posting_test_crc32(buf, 36);
    memcpy(buf + 36, &hdr_crc, 4);

    size_t count = 0;
    ASSERT(posting_segment_parse_buffer(buf, len, 8, posting_test_count_visit, &count) == 0,
           "parse downgraded v1 header");
    ASSERT(count == 1, "v1 compat entry");

    free(buf);
    return 0;
}

static int test_posting_pq_payload(void)
{
    const size_t dim = 4;
    const uint32_t pq_m = 2;
    float codebook[1024];
    memset(codebook, 0, sizeof(codebook));
    /* subquantizer 0, code 3 -> {1, 2}; subquantizer 1, code 7 -> {3, 4} */
    codebook[3 * 2 + 0] = 1.f;
    codebook[3 * 2 + 1] = 2.f;
    codebook[1u * 256u * 2u + 7u * 2u + 0] = 3.f;
    codebook[1u * 256u * 2u + 7u * 2u + 1] = 4.f;
    uint8_t codes[2] = {3, 7};

    GV_PostingWriteEntry e = {
        .vector_id = 11, .version = 1, .flags = 0, .codes = codes
    };
    GV_PostingSegmentParams params = {
        .payload_type = GV_POSTING_PAYLOAD_PQ,
        .pq_m = pq_m,
        .pq_codebook = codebook
    };

    uint8_t *buf = NULL;
    size_t len = 0;
    ASSERT(posting_segment_encode_ex(0, 0, &e, 1, dim, 4096, &params, &buf, &len) == 0, "encode pq");

    size_t count = 0;
    ASSERT(posting_segment_parse_buffer(buf, len, dim, posting_test_count_visit, &count) == 0,
           "parse pq");
    ASSERT(count == 1, "one pq entry");

    free(buf);
    return 0;
}

static int test_posting_pq_materialize(void)
{
    char dir[512];
    if (gv_test_make_temp_path(dir, sizeof(dir), "gv_posting_pq_mat", "") != 0) return 0;

    const size_t dim = 4;
    const uint32_t pq_m = 2;
    float codebook[1024];
    memset(codebook, 0, sizeof(codebook));
    codebook[3 * 2 + 0] = 1.f;
    codebook[3 * 2 + 1] = 2.f;
    codebook[1u * 256u * 2u + 7u * 2u + 0] = 3.f;
    codebook[1u * 256u * 2u + 7u * 2u + 1] = 4.f;
    uint8_t codes[2] = {3, 7};

    GV_PostingCatalog *cat = posting_catalog_open(dir, 4096);
    ASSERT(cat != NULL, "open");

    GV_PostingWriteEntry e = {
        .vector_id = 21, .version = 1, .flags = 0, .codes = codes
    };
    GV_PostingSegmentParams params = {
        .payload_type = GV_POSTING_PAYLOAD_PQ,
        .pq_m = pq_m,
        .pq_codebook = codebook
    };
    float dummy[4] = {0};
    e.data = dummy;

    ASSERT(posting_catalog_append_segment_ex(cat, 8, &e, 1, dim, &params) == 0, "append pq");

    GV_PostingHeadView view;
    memset(&view, 0, sizeof(view));
    ASSERT(posting_catalog_materialize_head(cat, 8, &view) == 0, "materialize pq");
    ASSERT(view.count == 1, "one pq vector");
    ASSERT(fabsf(view.entries[0].data[0] - 1.f) < 1e-5f, "pq dequant d0");
    ASSERT(fabsf(view.entries[0].data[3] - 4.f) < 1e-5f, "pq dequant d3");

    posting_head_view_free(&view);
    posting_catalog_close(cat);
    return 0;
}

static int test_posting_auto_live_count_on_read(void)
{
    char dir[512];
    if (gv_test_make_temp_path(dir, sizeof(dir), "gv_posting_autolive", "") != 0) return 0;

    GV_PostingCatalog *cat = posting_catalog_open(dir, 4096);
    ASSERT(cat != NULL, "open");
    ASSERT(posting_catalog_get_auto_live_count(cat) == 1, "auto live default on");

    float old_v[4] = {1, 0, 0, 0};
    float new_v[4] = {0, 1, 0, 0};
    GV_PostingWriteEntry e1 = { .vector_id = 50, .version = 1, .flags = 0, .data = old_v };
    GV_PostingWriteEntry e2 = { .vector_id = 50, .version = 2, .flags = 0, .data = new_v };

    ASSERT(posting_catalog_append_segment(cat, 5, &e1, 1, 4) == 0, "append v1");
    ASSERT(posting_catalog_segment_live_count(cat, 5, 0) == 1, "append sets live_count");

    ASSERT(posting_catalog_append_segment(cat, 5, &e2, 1, 4) == 0, "append v2");

    GV_PostingHeadView view;
    memset(&view, 0, sizeof(view));
    ASSERT(posting_catalog_materialize_head(cat, 5, &view) == 0, "materialize triggers reconcile");
    posting_head_view_free(&view);

    ASSERT(posting_catalog_segment_live_count(cat, 5, 0) == 0, "stale segment live_count");
    ASSERT(posting_catalog_segment_live_count(cat, 5, 1) == 1, "winning segment live_count");

    posting_catalog_close(cat);
    return 0;
}

int main(void)
{
    struct { const char *name; int (*fn)(void); } tests[] = {
        { "segment encode/parse", test_posting_segment_encode_parse },
        { "catalog append/load", test_posting_catalog_append_and_load },
        { "version skip", test_posting_version_skip },
        { "deleted tombstone", test_posting_deleted_tombstone },
        { "corrupt segment", test_posting_corrupt_segment },
        { "bulk 10k append", test_posting_bulk_append },
        { "multi segment merge", test_posting_multi_segment_merge },
        { "sq8 payload", test_posting_sq8_payload },
        { "corrupt catalog", test_posting_corrupt_catalog },
        { "live count reconcile", test_posting_live_count_reconcile },
        { "auto live count on read", test_posting_auto_live_count_on_read },
        { "segment cache", test_posting_segment_cache },
        { "v1 segment compat", test_posting_v1_segment_compat },
        { "pq payload", test_posting_pq_payload },
        { "pq materialize", test_posting_pq_materialize },
    };

    int rc = 0;
    for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); ++i) {
        fprintf(stderr, "Running %s...\n", tests[i].name);
        if (tests[i].fn() != 0) rc = 1;
    }
    return rc;
}
