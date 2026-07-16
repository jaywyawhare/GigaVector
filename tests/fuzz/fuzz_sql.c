/**
 * @file fuzz_sql.c
 * @brief libFuzzer harness for the SQL-like query parser/executor
 *        (src/features/sql.c).
 *
 * Build: make fuzz  (requires clang)
 * Run:   build/fuzz/fuzz_sql tests/fuzz/corpus/sql -runs=100000
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "core/memory.h"
#include "storage/database.h"
#include "features/sql.h"

#define FUZZ_MAX_INPUT (64 * 1024)
#define FUZZ_DIM 4

static GV_Database *g_db = NULL;
static GV_SQLEngine *g_eng = NULL;

static void fuzz_setup_once(void) {
    if (g_eng) return;

    g_db = db_open(NULL, FUZZ_DIM, GV_INDEX_TYPE_FLAT);
    if (!g_db) return;

    /* Seed a few rows so SELECT/UPDATE/DELETE paths have data to touch. */
    for (int i = 0; i < 8; i++) {
        float v[FUZZ_DIM] = {(float)i, 1.0f, 2.0f, 3.0f};
        db_add_vector_with_metadata(g_db, v, FUZZ_DIM, "color",
                                    (i % 2) ? "red" : "blue");
    }

    g_eng = sql_create(g_db);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size == 0 || size > FUZZ_MAX_INPUT) return 0;

    fuzz_setup_once();
    if (!g_eng) return 0;

    /* sql_execute expects a NUL-terminated query string; copy and terminate. */
    char *query = (char *)gv_alloc(size + 1);
    if (!query) return 0;
    memcpy(query, data, size);
    query[size] = '\0';

    GV_SQLResult result;
    memset(&result, 0, sizeof(result));
    if (sql_execute(g_eng, query, &result) == 0) {
        sql_free_result(&result);
    }

    gv_free(query);
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
