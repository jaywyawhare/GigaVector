#define _GNU_SOURCE
#include "storage/bulk_import.h"
#include "features/json.h"
#include "core/memory.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int db_import_csv(GV_Database *db, const char *path, char delimiter,
                  int has_header, int id_column, GV_ImportReport *out) {
    GV_ImportReport rep = {0, 0, 0};
    if (!db || !path) return -1;
    if (delimiter == 0) delimiter = ',';
    size_t dim = db->dimension;

    FILE *fp = fopen(path, "r");
    if (!fp) return -1;

    float *vec = (float *)gv_alloc(dim * sizeof(float));
    if (!vec) { fclose(fp); return -1; }

    char *line = NULL; size_t cap = 0; ssize_t len;
    int first = 1;
    while ((len = getline(&line, &cap, fp)) != -1) {
        if (first && has_header) { first = 0; continue; }
        first = 0;
        /* strip trailing newline/cr */
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = '\0';
        if (len == 0 || line[0] == '#') { rep.skipped++; continue; }

        /* Split into fields; extract id, parse the rest as floats. */
        size_t vi = 0; int col = 0; int ok = 1;
        char *id = NULL;
        char *p = line;
        while (p && ok) {
            char *sep = strchr(p, delimiter);
            if (sep) *sep = '\0';
            if (col == id_column) {
                id = p;                                  /* string id column */
            } else {
                if (vi >= dim) { ok = 0; break; }        /* too many numeric fields */
                char *end = NULL;
                float f = strtof(p, &end);
                while (*end == ' ' || *end == '\t') end++;
                if (end == p || *end != '\0') { ok = 0; break; }
                vec[vi++] = f;
            }
            col++;
            p = sep ? sep + 1 : NULL;
        }
        if (!ok || vi != dim) { rep.errors++; continue; }

        int rc = (id_column >= 0 && id) ? db_add_vector_with_id(db, id, vec, dim)
                                        : db_add_vector(db, vec, dim);
        if (rc == 0) rep.imported++; else rep.errors++;
    }

    free(line);
    gv_free(vec);
    fclose(fp);
    if (out) *out = rep;
    return 0;
}

int db_import_jsonl(GV_Database *db, const char *path, GV_ImportReport *out) {
    GV_ImportReport rep = {0, 0, 0};
    if (!db || !path) return -1;
    size_t dim = db->dimension;

    FILE *fp = fopen(path, "r");
    if (!fp) return -1;

    float *vec = (float *)gv_alloc(dim * sizeof(float));
    if (!vec) { fclose(fp); return -1; }

    char *line = NULL; size_t cap = 0; ssize_t len;
    while ((len = getline(&line, &cap, fp)) != -1) {
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = '\0';
        if (len == 0) { rep.skipped++; continue; }

        GV_JsonValue *root = json_parse(line, NULL);
        if (!root) { rep.errors++; continue; }

        GV_JsonValue *arr = json_object_get(root, "vector");
        if (!arr || json_array_length(arr) != dim) { json_free(root); rep.errors++; continue; }
        int ok = 1;
        for (size_t i = 0; i < dim; i++) {
            double d = 0.0;
            if (json_get_number(json_array_get(arr, i), &d) != GV_JSON_OK) { ok = 0; break; }
            vec[i] = (float)d;
        }
        if (!ok) { json_free(root); rep.errors++; continue; }

        const char *id = NULL;
        GV_JsonValue *idv = json_object_get(root, "id");
        if (idv) id = json_get_string(idv);

        /* Optional flat string metadata. */
        const char **keys = NULL, **vals = NULL; size_t nmeta = 0;
        GV_JsonValue *meta = json_object_get(root, "metadata");
        if (meta) {
            size_t mn = json_object_length(meta);
            keys = (const char **)gv_alloc(mn * sizeof(char *));
            vals = (const char **)gv_alloc(mn * sizeof(char *));
            for (size_t i = 0; keys && vals && i < mn; i++) {
                const char *k = json_object_key_at(meta, i);
                const char *v = json_get_string(json_object_value_at(meta, i));
                if (k && v) { keys[nmeta] = k; vals[nmeta] = v; nmeta++; }
            }
        }

        int rc;
        if (id && nmeta > 0)      rc = db_add_vector_with_id_meta(db, id, vec, dim, keys, vals, nmeta);
        else if (id)             rc = db_add_vector_with_id(db, id, vec, dim);
        else                     rc = db_add_vector(db, vec, dim);
        gv_free((void *)keys); gv_free((void *)vals);
        json_free(root);
        if (rc == 0) rep.imported++; else rep.errors++;
    }

    free(line);
    gv_free(vec);
    fclose(fp);
    if (out) *out = rep;
    return 0;
}
