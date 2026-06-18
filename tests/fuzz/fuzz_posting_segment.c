/**
 * @file fuzz_posting_segment.c
 * @brief libFuzzer harness for posting_segment_parse_buffer.
 */

#include <stddef.h>
#include "core/memory.h"
#include <stdint.h>
#include <stdlib.h>

#include "storage/posting_list.h"

#define FUZZ_MAX_INPUT (512 * 1024)
#define FUZZ_MAX_DIM 256

static int noop_visit(void *ctx, const GV_PostingEntry *entry)
{
    (void)ctx;
    (void)entry;
    return 0;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    if (size == 0 || size > FUZZ_MAX_INPUT) return 0;
    (void)posting_segment_parse_buffer(data, size, FUZZ_MAX_DIM, noop_visit, NULL);
    (void)posting_segment_parse_buffer(data, size, 0, noop_visit, NULL);
    return 0;
}

#ifdef GV_FUZZ_STANDALONE
#include <stdio.h>
int main(int argc, char **argv)
{
    if (argc < 2) return 0;
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz > (long)FUZZ_MAX_INPUT) { fclose(f); return 0; }
    uint8_t *buf = (uint8_t *)gv_alloc((size_t)sz);
    if (!buf) { fclose(f); return 0; }
    fread(buf, 1, (size_t)sz, f);
    fclose(f);
    LLVMFuzzerTestOneInput(buf, (size_t)sz);
    gv_free(buf);
    return 0;
}
#endif
