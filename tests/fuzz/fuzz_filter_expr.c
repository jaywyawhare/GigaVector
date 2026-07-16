/**
 * @file fuzz_filter_expr.c
 * @brief libFuzzer harness for the filter expression parser (src/search/filter.c).
 *
 * Build: make fuzz  (requires clang)
 * Run:   build/fuzz/fuzz_filter_expr tests/fuzz/corpus/filter_expr -runs=100000
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "core/memory.h"
#include "search/filter.h"

#define FUZZ_MAX_INPUT (64 * 1024)

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size > FUZZ_MAX_INPUT) return 0;

    /* filter_parse expects a NUL-terminated expression; copy and terminate. */
    char *expr = (char *)gv_alloc(size + 1);
    if (!expr) return 0;
    if (size) memcpy(expr, data, size);
    expr[size] = '\0';

    GV_Filter *filter = filter_parse(expr);
    if (filter) {
        filter_destroy(filter);
    }

    gv_free(expr);
    return 0;
}

#ifdef GV_FUZZ_STANDALONE
#include <stdio.h>
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <input-file>\n", argv[0]);
        return 1;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 1;
    uint8_t buf[FUZZ_MAX_INPUT];
    size_t n = fread(buf, 1, sizeof(buf), f);
    fclose(f);
    return LLVMFuzzerTestOneInput(buf, n);
}
#endif
