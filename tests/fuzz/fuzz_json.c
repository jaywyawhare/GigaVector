/**
 * @file fuzz_json.c
 * @brief libFuzzer harness for the JSON parser (src/features/json.c).
 *
 * Build: make fuzz  (requires clang)
 * Run:   build/fuzz/fuzz_json tests/fuzz/corpus/json -runs=100000
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "core/memory.h"
#include "features/json.h"

#define FUZZ_MAX_INPUT (64 * 1024)

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size > FUZZ_MAX_INPUT) return 0;

    /* json_parse expects a NUL-terminated string; copy and terminate. */
    char *str = (char *)gv_alloc(size + 1);
    if (!str) return 0;
    if (size) memcpy(str, data, size);
    str[size] = '\0';

    GV_JsonError err = GV_JSON_OK;
    GV_JsonValue *value = json_parse(str, &err);
    if (value) {
        json_free(value);
    }

    gv_free(str);
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
