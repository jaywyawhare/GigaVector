/* Bulk import from CSV and JSONL, including malformed-row accounting. */
#include <stdio.h>
#include <stdlib.h>
#include "storage/database.h"
#include "storage/bulk_import.h"
#include "../test_tmp.h"

static int failures = 0;
#define ASSERT(c, m) do { if (!(c)) { printf("FAIL: %s\n", (m)); failures++; } \
                          else { printf("ok: %s\n", (m)); } } while (0)

int main(void) {
    const size_t D = 4;
    GV_Database *db = db_open(NULL, D, GV_INDEX_TYPE_HNSW);
    ASSERT(db != NULL, "open db");

    char csv_path[512], jsonl_path[512], nope_path[512];
    gv_test_make_temp_path(csv_path, sizeof(csv_path), "gv_imp", ".csv");
    gv_test_make_temp_path(jsonl_path, sizeof(jsonl_path), "gv_imp", ".jsonl");
    gv_test_make_temp_path(nope_path, sizeof(nope_path), "gv_nope", ".csv");

    FILE *f = fopen(csv_path, "w");
    fputs("id,x0,x1,x2,x3\n", f);
    fputs("a,1,0,0,0\n", f);
    fputs("b,0,1,0,0\n", f);
    fputs("c,0,0,1,0\n", f);
    fputs("# comment line\n", f);
    fputs("bad,1,2\n", f);            /* wrong dimension -> error */
    fclose(f);

    GV_ImportReport r;
    ASSERT(db_import_csv(db, csv_path, ',', 1, 0, &r) == 0, "csv import ok");
    ASSERT(r.imported == 3 && r.skipped == 1 && r.errors == 1, "csv counts (3/1/1)");
    size_t idx;
    ASSERT(db_get_index_by_id(db, "b", &idx) == 0, "csv id 'b' resolvable");

    FILE *j = fopen(jsonl_path, "w");
    fputs("{\"id\":\"d\",\"vector\":[0,0,0,1],\"metadata\":{\"cat\":\"x\"}}\n", j);
    fputs("{\"vector\":[1,1,0,0]}\n", j);
    fputs("\n", j);                    /* blank -> skipped */
    fputs("{\"vector\":[1,2]}\n", j);  /* wrong dim -> error */
    fputs("not json\n", j);            /* parse error */
    fclose(j);

    GV_ImportReport r2;
    ASSERT(db_import_jsonl(db, jsonl_path, &r2) == 0, "jsonl import ok");
    ASSERT(r2.imported == 2 && r2.skipped == 1 && r2.errors == 2, "jsonl counts (2/1/2)");
    ASSERT(db_get_index_by_id(db, "d", &idx) == 0, "jsonl id 'd' resolvable");

    /* bad args */
    ASSERT(db_import_csv(NULL, csv_path, ',', 1, 0, &r) == -1, "null db -> -1");
    ASSERT(db_import_csv(db, nope_path, ',', 1, 0, &r) == -1, "missing file -> -1");

    db_close(db);
    remove(csv_path); remove(jsonl_path);
    printf(failures ? "\nSOME TESTS FAILED (%d)\n" : "\nALL BULK-IMPORT TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
