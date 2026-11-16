#include <ctype.h>
#include "core/memory.h"
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "features/sql.h"
#include "storage/database.h"
#include "search/distance.h"
#include "schema/metadata.h"
#include "storage/soa_storage.h"
#include "core/utils.h"
#include "features/sql_internal.h"

static int sql_eval_where(const GV_SQLWhere *w, const GV_Vector *vec)
{
    if (!w) return 1; /* no WHERE => match all */

    switch (w->type) {
    case GV_SQL_WHERE_AND: {
        int l = sql_eval_where(w->left, vec);
        if (l <= 0) return l;
        return sql_eval_where(w->right, vec);
    }
    case GV_SQL_WHERE_OR: {
        int l = sql_eval_where(w->left, vec);
        if (l < 0) return l;
        if (l == 1) return 1;
        return sql_eval_where(w->right, vec);
    }
    case GV_SQL_WHERE_NOT: {
        int v = sql_eval_where(w->child, vec);
        if (v < 0) return v;
        return v ? 0 : 1;
    }
    case GV_SQL_WHERE_CMP: {
        const char *meta_val = vector_get_metadata(vec, w->field);

        /* Presence predicates define their own behaviour when the field is absent. */
        if (w->op == GV_SQL_CMP_IS_NULL)     return meta_val ? 0 : 1;
        if (w->op == GV_SQL_CMP_IS_NOT_NULL) return meta_val ? 1 : 0;

        if (!meta_val) return 0; /* every other predicate fails on an absent field */

        if (w->op == GV_SQL_CMP_IN) {
            for (size_t i = 0; i < w->list_count; i++) {
                if (w->list_is_numeric[i]) {
                    char *endptr = NULL;
                    double v = strtod(meta_val, &endptr);
                    if (endptr != meta_val && v == w->list_nums[i]) return 1;
                } else if (strcmp(meta_val, w->list_values[i]) == 0) {
                    return 1;
                }
            }
            return 0;
        }

        if (w->op == GV_SQL_CMP_BETWEEN) {
            if (w->is_numeric && w->is_numeric2) {
                char *endptr = NULL;
                double v = strtod(meta_val, &endptr);
                if (endptr == meta_val) return 0;
                return (v >= w->num_value && v <= w->num_value2) ? 1 : 0;
            }
            return (strcmp(meta_val, w->value) >= 0 &&
                    strcmp(meta_val, w->value2) <= 0) ? 1 : 0;
        }

        if (w->op == GV_SQL_CMP_LIKE) {
            /* Simple LIKE: treat '%' as wildcard prefix/suffix */
            if (!w->value) return 0;
            size_t vlen = strlen(w->value);
            if (vlen == 0) return strlen(meta_val) == 0 ? 1 : 0;
            int prefix_wild = (w->value[0] == '%');
            int suffix_wild = (vlen > 1 && w->value[vlen - 1] == '%');
            const char *pattern = w->value + (prefix_wild ? 1 : 0);
            size_t plen = strlen(pattern);
            if (suffix_wild && plen > 0) plen--;
            if (prefix_wild && suffix_wild) {
                /* contains */
                char *tmp = sql_strndup(pattern, plen);
                if (!tmp) return -1;
                int found = (strstr(meta_val, tmp) != NULL);
                gv_free(tmp);
                return found;
            } else if (prefix_wild) {
                /* ends with */
                size_t mlen = strlen(meta_val);
                if (plen > mlen) return 0;
                return memcmp(meta_val + mlen - plen, pattern, plen) == 0;
            } else if (suffix_wild) {
                /* starts with */
                return strncmp(meta_val, pattern, plen) == 0;
            } else {
                return strcmp(meta_val, w->value) == 0;
            }
        }

        if (w->is_numeric) {
            char *endptr = NULL;
            double v = strtod(meta_val, &endptr);
            if (endptr == meta_val) return 0;
            switch (w->op) {
            case GV_SQL_CMP_EQ: return v == w->num_value;
            case GV_SQL_CMP_NE: return v != w->num_value;
            case GV_SQL_CMP_LT: return v <  w->num_value;
            case GV_SQL_CMP_LE: return v <= w->num_value;
            case GV_SQL_CMP_GT: return v >  w->num_value;
            case GV_SQL_CMP_GE: return v >= w->num_value;
            default: return 0;
            }
        } else {
            int cmp = strcmp(meta_val, w->value);
            switch (w->op) {
            case GV_SQL_CMP_EQ: return cmp == 0;
            case GV_SQL_CMP_NE: return cmp != 0;
            case GV_SQL_CMP_LT: return cmp <  0;
            case GV_SQL_CMP_LE: return cmp <= 0;
            case GV_SQL_CMP_GT: return cmp >  0;
            case GV_SQL_CMP_GE: return cmp >= 0;
            default: return 0;
            }
        }
    }
    }
    return 0;
}

/* Metadata-to-JSON serialiser (lightweight, no dependency on json.h) */

/* Ensure *buf has room for `extra` more bytes; grows via realloc.
 * Returns 1 on success, 0 on OOM (buf is freed on failure by caller). */
static int sql_json_reserve(char **buf, size_t *cap, size_t len, size_t extra)
{
    size_t needed = len + extra;
    if (needed <= *cap) return 1;
    size_t nc = *cap ? *cap : 256;
    while (nc < needed) nc *= 2;
    char *t = (char *)gv_realloc(*buf, nc);
    if (!t) return 0;
    *buf = t;
    *cap = nc;
    return 1;
}

/* Append `s` to the JSON buffer, escaping ", \, and control chars per RFC 8259.
 * Returns 1 on success, 0 on OOM. */
static int sql_json_append_escaped(char **buf, size_t *cap, size_t *len, const char *s)
{
    if (!s) return 1;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        unsigned char c = *p;
        /* Worst case one input byte expands to 6 output bytes (\u00XX). */
        if (!sql_json_reserve(buf, cap, *len, 6)) return 0;
        switch (c) {
        case '"':  (*buf)[(*len)++] = '\\'; (*buf)[(*len)++] = '"';  break;
        case '\\': (*buf)[(*len)++] = '\\'; (*buf)[(*len)++] = '\\'; break;
        case '\b': (*buf)[(*len)++] = '\\'; (*buf)[(*len)++] = 'b';  break;
        case '\f': (*buf)[(*len)++] = '\\'; (*buf)[(*len)++] = 'f';  break;
        case '\n': (*buf)[(*len)++] = '\\'; (*buf)[(*len)++] = 'n';  break;
        case '\r': (*buf)[(*len)++] = '\\'; (*buf)[(*len)++] = 'r';  break;
        case '\t': (*buf)[(*len)++] = '\\'; (*buf)[(*len)++] = 't';  break;
        default:
            if (c < 0x20) {
                static const char hex[] = "0123456789abcdef";
                (*buf)[(*len)++] = '\\'; (*buf)[(*len)++] = 'u';
                (*buf)[(*len)++] = '0';  (*buf)[(*len)++] = '0';
                (*buf)[(*len)++] = hex[(c >> 4) & 0xF];
                (*buf)[(*len)++] = hex[c & 0xF];
            } else {
                (*buf)[(*len)++] = (char)c;
            }
            break;
        }
    }
    return 1;
}

static char *sql_metadata_to_json(const GV_Metadata *meta)
{
    if (!meta) {
        char *s = (char *)gv_alloc(3);
        if (s) { s[0] = '{'; s[1] = '}'; s[2] = '\0'; }
        return s;
    }

    /* Build JSON string incrementally */
    size_t cap = 256;
    size_t len = 0;
    char *buf = (char *)gv_alloc(cap);
    if (!buf) return NULL;
    buf[len++] = '{';

    int first = 1;
    for (const GV_Metadata *m = meta; m; m = m->next) {
        if (!first) {
            if (!sql_json_reserve(&buf, &cap, len, 1)) { gv_free(buf); return NULL; }
            buf[len++] = ',';
        }
        first = 0;

        /* "key":"value" - key and value escaped for valid JSON. */
        if (!sql_json_reserve(&buf, &cap, len, 1)) { gv_free(buf); return NULL; }
        buf[len++] = '"';
        if (!sql_json_append_escaped(&buf, &cap, &len, m->key)) { gv_free(buf); return NULL; }
        if (!sql_json_reserve(&buf, &cap, len, 3)) { gv_free(buf); return NULL; }
        buf[len++] = '"';
        buf[len++] = ':';
        buf[len++] = '"';
        if (!sql_json_append_escaped(&buf, &cap, &len, m->value)) { gv_free(buf); return NULL; }
        if (!sql_json_reserve(&buf, &cap, len, 1)) { gv_free(buf); return NULL; }
        buf[len++] = '"';
    }

    if (!sql_json_reserve(&buf, &cap, len, 2)) { gv_free(buf); return NULL; }
    buf[len++] = '}';
    buf[len] = '\0';
    return buf;
}

/* Engine internals */

struct GV_SQLEngine {
    GV_Database *db;
    pthread_mutex_t mutex;
    char last_error[GV_SQL_ERROR_SIZE];
};

static void sql_set_error(GV_SQLEngine *eng, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(eng->last_error, GV_SQL_ERROR_SIZE, fmt, ap);
    va_end(ap);
}

typedef struct {
    size_t index;
    float distance;
    int has_distance;
} GV_SQLRow;

static const GV_SQLAnn *sql_order_distance_ann(const GV_SQLStmt *stmt)
{
    if (stmt->has_order_ann) return &stmt->order_ann;
    if (stmt->order_field &&
        sql_kw_match(stmt->order_field, strlen(stmt->order_field),
                     "vector_distance") &&
        stmt->has_ann)
        return &stmt->ann;
    return NULL;
}

static float sql_row_vector_distance(GV_Database *db, const GV_SQLStmt *stmt,
                                     const GV_SQLRow *row)
{
    const GV_SQLAnn *ann = sql_order_distance_ann(stmt);
    if (!ann || !ann->query_vector || ann->query_dim == 0) {
        return row->has_distance ? row->distance : 0.0f;
    }

    GV_SoAStorage *soa = db->soa_storage;
    if (!soa) return row->has_distance ? row->distance : 0.0f;

    GV_Vector view;
    if (soa_storage_get_vector_view(soa, row->index, &view) != 0)
        return row->has_distance ? row->distance : 0.0f;

    GV_Vector query;
    query.data = (float *)ann->query_vector;
    query.dimension = ann->query_dim;
    query.metadata = NULL;
    return distance(&view, &query, ann->metric);
}

static int sql_get_metadata_value(GV_Database *db, size_t index,
                                  const char *field, char *out, size_t out_size)
{
    if (!out || out_size == 0) return -1;
    out[0] = '\0';

    GV_SoAStorage *soa = db->soa_storage;
    if (!soa || !field) return -1;

    if (sql_kw_match(field, strlen(field), "index")) {
        snprintf(out, out_size, "%zu", index);
        return 0;
    }

    GV_Vector view;
    if (soa_storage_get_vector_view(soa, index, &view) != 0) return -1;

    if (sql_kw_match(field, strlen(field), "metadata")) {
        char *json = sql_metadata_to_json(view.metadata);
        if (!json) return -1;
        snprintf(out, out_size, "%s", json);
        gv_free(json);
        return 0;
    }

    const char *val = vector_get_metadata(&view, field);
    if (!val) return -1;
    snprintf(out, out_size, "%s", val);
    return 0;
}

static int sql_compare_rows(GV_Database *db, const GV_SQLStmt *stmt,
                            const GV_SQLRow *a, const GV_SQLRow *b)
{
    if (!stmt->order_field) return 0;

    if (sql_order_distance_ann(stmt) ||
        sql_kw_match(stmt->order_field, strlen(stmt->order_field), "distance")) {
        float da = sql_row_vector_distance(db, stmt, a);
        float dbv = sql_row_vector_distance(db, stmt, b);
        if (da < dbv) return stmt->order_desc ? 1 : -1;
        if (da > dbv) return stmt->order_desc ? -1 : 1;
        return 0;
    }

    if (sql_kw_match(stmt->order_field, strlen(stmt->order_field), "index")) {
        if (a->index < b->index) return stmt->order_desc ? 1 : -1;
        if (a->index > b->index) return stmt->order_desc ? -1 : 1;
        return 0;
    }

    char va[512], vb[512];
    if (sql_get_metadata_value(db, a->index, stmt->order_field, va, sizeof(va)) != 0)
        va[0] = '\0';
    if (sql_get_metadata_value(db, b->index, stmt->order_field, vb, sizeof(vb)) != 0)
        vb[0] = '\0';

    char *ea = NULL, *eb = NULL;
    double na = strtod(va, &ea);
    double nb = strtod(vb, &eb);
    if (ea > va && eb > vb) {
        if (na < nb) return stmt->order_desc ? 1 : -1;
        if (na > nb) return stmt->order_desc ? -1 : 1;
        return 0;
    }

    int cmp = strcmp(va, vb);
    if (cmp < 0) return stmt->order_desc ? 1 : -1;
    if (cmp > 0) return stmt->order_desc ? -1 : 1;
    return 0;
}

static void sql_sort_rows(GV_Database *db, const GV_SQLStmt *stmt,
                          GV_SQLRow *rows, size_t count)
{
    if (!stmt->order_field || count < 2) return;
    for (size_t i = 0; i + 1 < count; i++) {
        for (size_t j = i + 1; j < count; j++) {
            if (sql_compare_rows(db, stmt, &rows[i], &rows[j]) > 0) {
                GV_SQLRow tmp = rows[i];
                rows[i] = rows[j];
                rows[j] = tmp;
            }
        }
    }
}

static void sql_apply_row_limit(GV_SQLRow *rows, size_t *count,
                                const GV_SQLStmt *stmt)
{
    /* OFFSET: drop the first `offset` rows (GV_SQLRow is POD, so shifting is safe). */
    if (stmt->has_offset && stmt->offset > 0) {
        if (stmt->offset >= *count) { *count = 0; return; }
        size_t remaining = *count - stmt->offset;
        memmove(rows, rows + stmt->offset, remaining * sizeof(GV_SQLRow));
        *count = remaining;
    }
    /* LIMIT: cap the row count. */
    if (stmt->has_limit && *count > stmt->limit) {
        *count = stmt->limit;
    }
}

static char *sql_dup_cell_value(GV_Database *db, const GV_SQLStmt *stmt,
                                const GV_SQLRow *row, const char *col,
                                int has_distances)
{
    char buf[4096];
    if (sql_kw_match(col, strlen(col), "index")) {
        snprintf(buf, sizeof(buf), "%zu", row->index);
        return gv_dup_cstr(buf);
    }
    if (sql_kw_match(col, strlen(col), "distance")) {
        float dist = row->has_distance ? row->distance
                                       : sql_row_vector_distance(db, stmt, row);
        (void)has_distances;
        snprintf(buf, sizeof(buf), "%g", (double)dist);
        return gv_dup_cstr(buf);
    }
    if (sql_kw_match(col, strlen(col), "metadata")) {
        GV_SoAStorage *soa = db->soa_storage;
        if (!soa) return gv_dup_cstr("{}");
        GV_Metadata *meta = soa_storage_get_metadata(soa, row->index);
        return sql_metadata_to_json(meta);
    }
    if (sql_get_metadata_value(db, row->index, col, buf, sizeof(buf)) != 0)
        return gv_dup_cstr("");
    return gv_dup_cstr(buf);
}

static size_t sql_default_column_count(const GV_SQLStmt *stmt, int has_distances)
{
    if (stmt->select_column_count > 0) return stmt->select_column_count;
    return has_distances ? 3 : 2;
}

static char *sql_default_column_name(const GV_SQLStmt *stmt, size_t idx,
                                     int has_distances)
{
    if (stmt->select_column_count > 0)
        return gv_dup_cstr(stmt->select_columns[idx]);
    if (has_distances) {
        if (idx == 0) return gv_dup_cstr("index");
        if (idx == 1) return gv_dup_cstr("distance");
        return gv_dup_cstr("metadata");
    }
    if (idx == 0) return gv_dup_cstr("index");
    return gv_dup_cstr("metadata");
}

/* SELECT DISTINCT: drop rows whose projected column tuple duplicates an earlier
 * kept row. Stable stream-compaction over the parallel result arrays. */
static void sql_apply_distinct(GV_SQLResult *r)
{
    if (r->row_count < 2 || r->column_count == 0) return;
    size_t cc = r->column_count, w = 0;
    for (size_t i = 0; i < r->row_count; i++) {
        int dup = 0;
        for (size_t j = 0; j < w && !dup; j++) {
            int same = 1;
            for (size_t c = 0; c < cc; c++)
                if (strcmp(r->column_values[i*cc+c], r->column_values[j*cc+c]) != 0) { same = 0; break; }
            dup = same;
        }
        if (dup) {
            for (size_t c = 0; c < cc; c++) gv_free(r->column_values[i*cc+c]);
            if (r->metadata_jsons) gv_free(r->metadata_jsons[i]);
        } else {
            if (w != i) {
                for (size_t c = 0; c < cc; c++) r->column_values[w*cc+c] = r->column_values[i*cc+c];
                if (r->metadata_jsons) r->metadata_jsons[w] = r->metadata_jsons[i];
                if (r->indices) r->indices[w] = r->indices[i];
                if (r->distances) r->distances[w] = r->distances[i];
            }
            w++;
        }
    }
    r->row_count = w;
}

static int sql_build_select_result(GV_SQLEngine *eng, const GV_SQLStmt *stmt,
                                   GV_SQLRow *rows, size_t row_count,
                                   int has_distances, GV_SQLResult *result)
{
    GV_Database *db = eng->db;
    size_t col_count = sql_default_column_count(stmt, has_distances);

    memset(result, 0, sizeof(*result));
    result->row_count = row_count;
    result->column_count = col_count;
    result->column_names = (char **)gv_calloc(col_count, sizeof(char *));
    result->indices = (size_t *)gv_calloc(row_count ? row_count : 1, sizeof(size_t));
    result->metadata_jsons =
        (char **)gv_calloc(row_count ? row_count : 1, sizeof(char *));
    result->column_values =
        (char **)gv_calloc((row_count ? row_count : 1) * col_count, sizeof(char *));
    if (has_distances)
        result->distances = (float *)gv_calloc(row_count ? row_count : 1, sizeof(float));

    if (!result->column_names || !result->indices || !result->metadata_jsons ||
        !result->column_values ||
        (has_distances && !result->distances)) {
        sql_free_result(result);
        sql_set_error(eng, "Out of memory building result set");
        return -1;
    }

    for (size_t c = 0; c < col_count; c++) {
        result->column_names[c] = sql_default_column_name(stmt, c, has_distances);
        if (!result->column_names[c]) {
            sql_free_result(result);
            sql_set_error(eng, "Out of memory building result set");
            return -1;
        }
    }

    for (size_t i = 0; i < row_count; i++) {
        result->indices[i] = rows[i].index;
        if (has_distances)
            result->distances[i] = rows[i].has_distance
                                       ? rows[i].distance
                                       : sql_row_vector_distance(db, stmt, &rows[i]);

        GV_SoAStorage *soa = db->soa_storage;
        GV_Metadata *meta = soa ? soa_storage_get_metadata(soa, rows[i].index) : NULL;
        result->metadata_jsons[i] = sql_metadata_to_json(meta);
        if (!result->metadata_jsons[i]) {
            sql_free_result(result);
            sql_set_error(eng, "Out of memory building result set");
            return -1;
        }

        for (size_t c = 0; c < col_count; c++) {
            const char *col = result->column_names[c];
            result->column_values[i * col_count + c] =
                sql_dup_cell_value(db, stmt, &rows[i], col, has_distances);
            if (!result->column_values[i * col_count + c]) {
                sql_free_result(result);
                sql_set_error(eng, "Out of memory building result set");
                return -1;
            }
        }
    }
    if (stmt->distinct) sql_apply_distinct(result);
    return 0;
}

/* Executor: SELECT with ANN */

static int sql_exec_ann(GV_SQLEngine *eng, const GV_SQLStmt *stmt, GV_SQLResult *result)
{
    GV_Database *db = eng->db;
    const GV_SQLAnn *ann = &stmt->ann;

    if (ann->query_dim != db->dimension) {
        sql_set_error(eng, "ANN query dimension %zu does not match database dimension %zu",
                      ann->query_dim, db->dimension);
        return -1;
    }

    size_t k = ann->k;
    GV_SearchResult *sr = (GV_SearchResult *)gv_calloc(k, sizeof(GV_SearchResult));
    if (!sr) { sql_set_error(eng, "Out of memory"); return -1; }

    int found;
    if (stmt->where) {
        size_t oversample = k * 4;
        if (oversample > db->count && db->count > 0) oversample = db->count;
        if (oversample < k) oversample = k;
        GV_SearchResult *all_sr =
            (GV_SearchResult *)gv_calloc(oversample, sizeof(GV_SearchResult));
        if (!all_sr) { gv_free(sr); sql_set_error(eng, "Out of memory"); return -1; }

        found = db_search(db, ann->query_vector, oversample, all_sr, ann->metric);
        if (found < 0) {
            gv_free(all_sr);
            gv_free(sr);
            sql_set_error(eng, "ANN search failed");
            return -1;
        }

        int matched = 0;
        for (int i = 0; i < found && (size_t)matched < k; i++) {
            if (!all_sr[i].vector) continue;
            if (sql_eval_where(stmt->where, all_sr[i].vector) == 1) {
                sr[matched] = all_sr[i];
                all_sr[i].vector = NULL; /* ownership transferred to sr */
                matched++;
            }
        }
        gv_search_results_free(all_sr, (size_t)found); /* free the unmatched result vectors */
        gv_free(all_sr);
        found = matched;
    } else {
        found = db_search(db, ann->query_vector, k, sr, ann->metric);
        if (found < 0) {
            gv_free(sr);
            sql_set_error(eng, "ANN search failed");
            return -1;
        }
    }

    if (found < 0) found = 0;

    GV_SQLRow *rows = (GV_SQLRow *)gv_calloc((size_t)found, sizeof(GV_SQLRow));
    if (!rows && found > 0) {
        gv_search_results_free(sr, (size_t)found);
        gv_free(sr);
        sql_set_error(eng, "Out of memory");
        return -1;
    }

    size_t row_count = (size_t)found;
    for (size_t i = 0; i < row_count; i++) {
        /* db_search stores the SoA storage index in .id (the vector payload is a
         * fresh heap copy, so pointer arithmetic against SoA would be invalid). */
        rows[i].index = sr[i].id;
        rows[i].distance = sr[i].distance;
        rows[i].has_distance = 1;
    }
    gv_search_results_free(sr, row_count); /* free owned result vectors */
    gv_free(sr);

    sql_sort_rows(db, stmt, rows, row_count);
    sql_apply_row_limit(rows, &row_count, stmt);

    int rc = sql_build_select_result(eng, stmt, rows, row_count, 1, result);
    gv_free(rows);
    return rc;
}

/* Executor: SELECT with WHERE (metadata scan, no ANN) */

static int sql_exec_where_scan(GV_SQLEngine *eng, const GV_SQLStmt *stmt,
                               GV_SQLResult *result)
{
    GV_Database *db = eng->db;
    GV_SoAStorage *soa = db->soa_storage;
    if (!soa) { sql_set_error(eng, "Database has no storage"); return -1; }

    size_t total = soa->count;
    size_t cap = 256;
    GV_SQLRow *rows = (GV_SQLRow *)gv_alloc(cap * sizeof(GV_SQLRow));
    if (!rows) { sql_set_error(eng, "Out of memory"); return -1; }
    size_t row_count = 0;

    for (size_t i = 0; i < total; i++) {
        if (soa_storage_is_deleted(soa, i)) continue;

        GV_Vector view;
        if (soa_storage_get_vector_view(soa, i, &view) != 0) continue;
        if (sql_eval_where(stmt->where, &view) != 1) continue;

        if (row_count >= cap) {
            cap *= 2;
            GV_SQLRow *tmp = (GV_SQLRow *)gv_realloc(rows, cap * sizeof(GV_SQLRow));
            if (!tmp) { gv_free(rows); sql_set_error(eng, "Out of memory"); return -1; }
            rows = tmp;
        }
        rows[row_count].index = i;
        rows[row_count].distance = 0.0f;
        rows[row_count].has_distance = 0;
        row_count++;
    }

    sql_sort_rows(db, stmt, rows, row_count);
    sql_apply_row_limit(rows, &row_count, stmt);

    int rc = sql_build_select_result(eng, stmt, rows, row_count, 0, result);
    gv_free(rows);
    return rc;
}

/* Executor: SELECT COUNT(*) */

/* Executor: aggregate projection - COUNT(*) / SUM / MIN / MAX / AVG(column).
 * Produces a single-row, single-column result.  COUNT keeps its historical
 * contract (result->indices[0] holds the count); every aggregate also exposes
 * a formatted value in result->column_values[0]. */
static int sql_exec_aggregate(GV_SQLEngine *eng, const GV_SQLStmt *stmt, GV_SQLResult *result)
{
    GV_Database *db = eng->db;
    GV_SoAStorage *soa = db->soa_storage;
    if (!soa) { sql_set_error(eng, "Database has no storage"); return -1; }

    size_t total = soa->count;
    size_t matched = 0;     /* rows passing WHERE */
    size_t numeric_n = 0;   /* rows with a numeric agg_column value */
    double acc = 0.0;       /* SUM / AVG accumulator */
    double best = 0.0;      /* MIN / MAX running value */
    int have_best = 0;
    int numeric_agg = (stmt->agg_kind != GV_SQL_AGG_COUNT && stmt->agg_kind != GV_SQL_AGG_NONE);

    for (size_t i = 0; i < total; i++) {
        if (soa_storage_is_deleted(soa, i)) continue;

        if (stmt->where) {
            GV_Vector view;
            if (soa_storage_get_vector_view(soa, i, &view) != 0) continue;
            if (sql_eval_where(stmt->where, &view) != 1) continue;
        }
        matched++;

        if (numeric_agg && stmt->agg_column) {
            char vbuf[4096];
            if (sql_get_metadata_value(db, i, stmt->agg_column, vbuf, sizeof(vbuf)) != 0) continue;
            char *endp = NULL;
            double v = strtod(vbuf, &endp);
            if (endp == vbuf) continue; /* non-numeric value: skipped like SQL NULLs */
            numeric_n++;
            acc += v;
            if (!have_best || (stmt->agg_kind == GV_SQL_AGG_MIN && v < best) ||
                              (stmt->agg_kind == GV_SQL_AGG_MAX && v > best)) {
                best = v;
                have_best = 1;
            }
        }
    }

    const char *col_name = "count";
    double dval = 0.0;
    int has_value = 1;
    switch (stmt->agg_kind) {
    case GV_SQL_AGG_SUM: col_name = "sum"; dval = acc; break;
    case GV_SQL_AGG_AVG: col_name = "avg"; if (numeric_n) dval = acc / (double)numeric_n; else has_value = 0; break;
    case GV_SQL_AGG_MIN: col_name = "min"; if (have_best) dval = best; else has_value = 0; break;
    case GV_SQL_AGG_MAX: col_name = "max"; if (have_best) dval = best; else has_value = 0; break;
    case GV_SQL_AGG_COUNT:
    default: col_name = "count"; dval = (double)matched; break;
    }

    memset(result, 0, sizeof(*result));
    result->row_count = 1;
    result->column_count = 1;
    result->column_names = (char **)gv_calloc(1, sizeof(char *));
    result->column_values = (char **)gv_calloc(1, sizeof(char *));
    result->indices = (size_t *)gv_calloc(1, sizeof(size_t));
    if (!result->column_names || !result->column_values || !result->indices) {
        sql_free_result(result);
        sql_set_error(eng, "Out of memory");
        return -1;
    }
    result->column_names[0] = gv_dup_cstr(col_name);
    char valbuf[64];
    if (!has_value) {
        result->column_values[0] = gv_dup_cstr("NULL");
    } else if (stmt->agg_kind == GV_SQL_AGG_COUNT || stmt->agg_kind == GV_SQL_AGG_NONE) {
        snprintf(valbuf, sizeof(valbuf), "%zu", matched);
        result->column_values[0] = gv_dup_cstr(valbuf);
    } else {
        snprintf(valbuf, sizeof(valbuf), "%g", dval);
        result->column_values[0] = gv_dup_cstr(valbuf);
    }
    result->indices[0] = matched; /* COUNT value; row count for other aggregates */
    if (!result->column_names[0] || !result->column_values[0]) {
        sql_free_result(result);
        sql_set_error(eng, "Out of memory");
        return -1;
    }
    return 0;
}

/* Executor: SELECT ... GROUP BY [HAVING].  Scans rows (respecting WHERE),
 * buckets by the concatenated group-key, accumulates each projected aggregate
 * per bucket, then emits one row per surviving group. Plain projected columns
 * take their group's representative-row value. */
typedef struct {
    char *key;           /* concatenated group-key */
    size_t rep;          /* representative (first) row index */
    size_t count;        /* rows in group (COUNT(*)) */
    double *acc;         /* per-proj SUM/AVG accumulator */
    double *best;        /* per-proj MIN/MAX */
    size_t *numn;        /* per-proj numeric-sample count */
    int    *have;        /* per-proj MIN/MAX-seen flag */
    /* HAVING accumulators (independent of projection) */
    double h_acc, h_best; size_t h_numn; int h_have;
} SqlGroup;

static const char *sql_agg_label(GV_SQLAggKind a)
{
    switch (a) {
    case GV_SQL_AGG_COUNT: return "count";
    case GV_SQL_AGG_SUM:   return "sum";
    case GV_SQL_AGG_MIN:   return "min";
    case GV_SQL_AGG_MAX:   return "max";
    case GV_SQL_AGG_AVG:   return "avg";
    default:               return "value";
    }
}

/* Fold one numeric sample into an aggregate accumulator group. */
static void sql_agg_fold(GV_SQLAggKind a, double v, double *acc, double *best,
                         size_t *numn, int *have)
{
    (*numn)++;
    *acc += v;
    if (!*have || (a == GV_SQL_AGG_MIN && v < *best) || (a == GV_SQL_AGG_MAX && v > *best)) {
        *best = v; *have = 1;
    }
}

/* Final scalar value of an aggregate over a group; has_value=0 => SQL NULL. */
static double sql_agg_finalize(GV_SQLAggKind a, size_t count, double acc, double best,
                               size_t numn, int have, int *has_value)
{
    *has_value = 1;
    switch (a) {
    case GV_SQL_AGG_COUNT: return (double)count;
    case GV_SQL_AGG_SUM:   return acc;
    case GV_SQL_AGG_AVG:   if (numn) return acc / (double)numn; *has_value = 0; return 0;
    case GV_SQL_AGG_MIN:
    case GV_SQL_AGG_MAX:   if (have) return best; *has_value = 0; return 0;
    default:               return 0;
    }
}

static int sql_having_pass(const GV_SQLStmt *stmt, const SqlGroup *g)
{
    if (!stmt->has_having) return 1;
    int hv = 1;
    double v = sql_agg_finalize(stmt->having_agg, g->count, g->h_acc, g->h_best,
                                g->h_numn, g->h_have, &hv);
    if (!hv) return 0;   /* NULL never satisfies a comparison */
    double t = stmt->having_value;
    switch (stmt->having_op) {
    case GV_SQL_TOK_EQ: return v == t;
    case GV_SQL_TOK_NE: return v != t;
    case GV_SQL_TOK_LT: return v <  t;
    case GV_SQL_TOK_LE: return v <= t;
    case GV_SQL_TOK_GT: return v >  t;
    case GV_SQL_TOK_GE: return v >= t;
    default: return 1;
    }
}

/* Executor: SELECT agg1(...), agg2(...), ... with no GROUP BY. Produces a
 * single row holding one column per projected aggregate, computed over all
 * rows passing WHERE. Mixed aggregate/plain projections are rejected by the
 * dispatcher before reaching here, so every projection has agg != NONE. */
static int sql_exec_multi_aggregate(GV_SQLEngine *eng, const GV_SQLStmt *stmt,
                                    GV_SQLResult *result)
{
    GV_Database *db = eng->db;
    GV_SoAStorage *soa = db->soa_storage;
    if (!soa) { sql_set_error(eng, "Database has no storage"); return -1; }

    size_t pc = stmt->proj_count;
    double *acc  = (double *)gv_calloc(pc, sizeof(double));
    double *best = (double *)gv_calloc(pc, sizeof(double));
    size_t *numn = (size_t *)gv_calloc(pc, sizeof(size_t));
    int    *have = (int *)gv_calloc(pc, sizeof(int));
    if (!acc || !best || !numn || !have) {
        gv_free(acc); gv_free(best); gv_free(numn); gv_free(have);
        sql_set_error(eng, "Out of memory");
        return -1;
    }

    size_t total = soa->count;
    size_t matched = 0;
    for (size_t i = 0; i < total; i++) {
        if (soa_storage_is_deleted(soa, i)) continue;
        if (stmt->where) {
            GV_Vector view;
            if (soa_storage_get_vector_view(soa, i, &view) != 0) continue;
            if (sql_eval_where(stmt->where, &view) != 1) continue;
        }
        matched++;

        for (size_t p = 0; p < pc; p++) {
            GV_SQLAggKind a = stmt->proj[p].agg;
            if (a == GV_SQL_AGG_COUNT || !stmt->proj[p].column) continue;
            char vbuf[4096];
            if (sql_get_metadata_value(db, i, stmt->proj[p].column, vbuf, sizeof(vbuf)) != 0)
                continue;
            char *endp = NULL;
            double v = strtod(vbuf, &endp);
            if (endp == vbuf) continue; /* non-numeric: treated as SQL NULL */
            sql_agg_fold(a, v, &acc[p], &best[p], &numn[p], &have[p]);
        }
    }

    memset(result, 0, sizeof(*result));
    result->row_count = 1;
    result->column_count = pc;
    result->column_names  = (char **)gv_calloc(pc, sizeof(char *));
    result->column_values = (char **)gv_calloc(pc, sizeof(char *));
    result->indices       = (size_t *)gv_calloc(1, sizeof(size_t));
    if (!result->column_names || !result->column_values || !result->indices) {
        gv_free(acc); gv_free(best); gv_free(numn); gv_free(have);
        sql_free_result(result);
        sql_set_error(eng, "Out of memory");
        return -1;
    }
    result->indices[0] = matched;

    int oom = 0;
    for (size_t p = 0; p < pc && !oom; p++) {
        GV_SQLAggKind a = stmt->proj[p].agg;
        const char *name = stmt->proj[p].alias ? stmt->proj[p].alias : sql_agg_label(a);
        result->column_names[p] = gv_dup_cstr(name);

        int hv = 1;
        double dv = sql_agg_finalize(a, matched, acc[p], best[p], numn[p], have[p], &hv);
        char valbuf[64];
        if (!hv) {
            result->column_values[p] = gv_dup_cstr("NULL");
        } else if (a == GV_SQL_AGG_COUNT) {
            snprintf(valbuf, sizeof(valbuf), "%zu", matched);
            result->column_values[p] = gv_dup_cstr(valbuf);
        } else {
            snprintf(valbuf, sizeof(valbuf), "%g", dv);
            result->column_values[p] = gv_dup_cstr(valbuf);
        }
        if (!result->column_names[p] || !result->column_values[p]) oom = 1;
    }

    gv_free(acc); gv_free(best); gv_free(numn); gv_free(have);
    if (oom) {
        sql_free_result(result);
        sql_set_error(eng, "Out of memory");
        return -1;
    }
    return 0;
}

static void sql_group_free(SqlGroup *groups, size_t gn)
{
    for (size_t i = 0; i < gn; i++) {
        gv_free(groups[i].key);
        gv_free(groups[i].acc); gv_free(groups[i].best);
        gv_free(groups[i].numn); gv_free(groups[i].have);
    }
    gv_free(groups);
}

static int sql_exec_group_by(GV_SQLEngine *eng, const GV_SQLStmt *stmt, GV_SQLResult *result)
{
    GV_Database *db = eng->db;
    GV_SoAStorage *soa = db->soa_storage;
    if (!soa) { sql_set_error(eng, "Database has no storage"); return -1; }

    size_t pc = stmt->proj_count;
    size_t gcap = 16, gn = 0;
    SqlGroup *groups = (SqlGroup *)gv_calloc(gcap, sizeof(SqlGroup));
    if (!groups) { sql_set_error(eng, "Out of memory"); return -1; }

    size_t total = soa->count;
    for (size_t i = 0; i < total; i++) {
        if (soa_storage_is_deleted(soa, i)) continue;
        GV_Vector view;
        if (soa_storage_get_vector_view(soa, i, &view) != 0) continue;
        if (stmt->where && sql_eval_where(stmt->where, &view) != 1) continue;

        /* Build the group key from the GROUP BY columns. */
        char key[4096]; size_t kl = 0; key[0] = '\0';
        for (size_t gcx = 0; gcx < stmt->group_by_count; gcx++) {
            char vb[512];
            if (sql_get_metadata_value(db, i, stmt->group_by[gcx], vb, sizeof(vb)) != 0) vb[0] = '\0';
            int wn = snprintf(key + kl, sizeof(key) - kl, "%s\x1f", vb);
            if (wn > 0) kl += (size_t)wn;
            if (kl >= sizeof(key)) { kl = sizeof(key) - 1; break; }
        }

        SqlGroup *g = NULL;
        for (size_t j = 0; j < gn; j++) if (strcmp(groups[j].key, key) == 0) { g = &groups[j]; break; }
        if (!g) {
            if (gn >= gcap) {
                gcap *= 2;
                SqlGroup *tmp = (SqlGroup *)gv_realloc(groups, gcap * sizeof(SqlGroup));
                if (!tmp) { sql_group_free(groups, gn); sql_set_error(eng, "Out of memory"); return -1; }
                groups = tmp;
            }
            g = &groups[gn];
            memset(g, 0, sizeof(*g));
            g->key = gv_dup_cstr(key);
            g->acc  = (double *)gv_calloc(pc ? pc : 1, sizeof(double));
            g->best = (double *)gv_calloc(pc ? pc : 1, sizeof(double));
            g->numn = (size_t *)gv_calloc(pc ? pc : 1, sizeof(size_t));
            g->have = (int *)gv_calloc(pc ? pc : 1, sizeof(int));
            if (!g->key || !g->acc || !g->best || !g->numn || !g->have) {
                gv_free(g->key); gv_free(g->acc); gv_free(g->best); gv_free(g->numn); gv_free(g->have);
                sql_group_free(groups, gn); sql_set_error(eng, "Out of memory"); return -1;
            }
            g->rep = i;
            gn++;
        }
        g->count++;

        /* Fold this row into each projected aggregate. */
        for (size_t p = 0; p < pc; p++) {
            if (stmt->proj[p].agg == GV_SQL_AGG_NONE || stmt->proj[p].agg == GV_SQL_AGG_COUNT) continue;
            char vb[512];
            if (sql_get_metadata_value(db, i, stmt->proj[p].column, vb, sizeof(vb)) != 0) continue;
            char *ep = NULL; double v = strtod(vb, &ep);
            if (ep == vb) continue;
            sql_agg_fold(stmt->proj[p].agg, v, &g->acc[p], &g->best[p], &g->numn[p], &g->have[p]);
        }
        /* Fold into the HAVING aggregate (if it needs a column). */
        if (stmt->has_having && stmt->having_agg != GV_SQL_AGG_COUNT && stmt->having_column) {
            char vb[512];
            if (sql_get_metadata_value(db, i, stmt->having_column, vb, sizeof(vb)) == 0) {
                char *ep = NULL; double v = strtod(vb, &ep);
                if (ep != vb) sql_agg_fold(stmt->having_agg, v, &g->h_acc, &g->h_best, &g->h_numn, &g->h_have);
            }
        }
    }

    /* Apply HAVING, then ORDER BY (by a group column via the representative row),
     * then OFFSET/LIMIT - all over the group list. */
    size_t *ord = (size_t *)gv_calloc(gn ? gn : 1, sizeof(size_t));
    if (!ord) { sql_group_free(groups, gn); sql_set_error(eng, "Out of memory"); return -1; }
    size_t on = 0;
    for (size_t j = 0; j < gn; j++) if (sql_having_pass(stmt, &groups[j])) ord[on++] = j;

    if (stmt->order_field) {
        for (size_t a = 0; a + 1 < on; a++)
            for (size_t b = a + 1; b < on; b++) {
                GV_SQLRow ra = { groups[ord[a]].rep, 0.0f, 0 };
                GV_SQLRow rb = { groups[ord[b]].rep, 0.0f, 0 };
                if (sql_compare_rows(db, stmt, &ra, &rb) > 0) { size_t t = ord[a]; ord[a] = ord[b]; ord[b] = t; }
            }
    }
    /* OFFSET/LIMIT on groups */
    size_t start = 0;
    if (stmt->has_offset && stmt->offset > 0) start = stmt->offset > on ? on : stmt->offset;
    size_t emit = on - start;
    if (stmt->has_limit && emit > stmt->limit) emit = stmt->limit;

    /* Build the result set. */
    memset(result, 0, sizeof(*result));
    result->row_count = emit;
    result->column_count = pc;
    result->column_names = (char **)gv_calloc(pc ? pc : 1, sizeof(char *));
    result->column_values = (char **)gv_calloc((emit ? emit : 1) * (pc ? pc : 1), sizeof(char *));
    result->indices = (size_t *)gv_calloc(emit ? emit : 1, sizeof(size_t));
    if (!result->column_names || !result->column_values || !result->indices) {
        gv_free(ord); sql_group_free(groups, gn); sql_free_result(result);
        sql_set_error(eng, "Out of memory"); return -1;
    }
    for (size_t p = 0; p < pc; p++) {
        const char *nm = stmt->proj[p].alias ? stmt->proj[p].alias :
                         stmt->proj[p].agg != GV_SQL_AGG_NONE ? sql_agg_label(stmt->proj[p].agg) :
                         stmt->proj[p].column;
        result->column_names[p] = gv_dup_cstr(nm ? nm : "");
        if (!result->column_names[p]) { gv_free(ord); sql_group_free(groups, gn); sql_free_result(result); sql_set_error(eng, "Out of memory"); return -1; }
    }
    for (size_t r = 0; r < emit; r++) {
        SqlGroup *g = &groups[ord[start + r]];
        result->indices[r] = g->rep;
        for (size_t p = 0; p < pc; p++) {
            char *cell;
            if (stmt->proj[p].agg == GV_SQL_AGG_NONE) {
                char vb[4096];
                if (sql_get_metadata_value(db, g->rep, stmt->proj[p].column, vb, sizeof(vb)) != 0) vb[0] = '\0';
                cell = gv_dup_cstr(vb);
            } else {
                int hv = 1;
                double dv = sql_agg_finalize(stmt->proj[p].agg, g->count, g->acc[p], g->best[p],
                                             g->numn[p], g->have[p], &hv);
                char vb[64];
                if (!hv) snprintf(vb, sizeof(vb), "NULL");
                else if (stmt->proj[p].agg == GV_SQL_AGG_COUNT) snprintf(vb, sizeof(vb), "%zu", g->count);
                else snprintf(vb, sizeof(vb), "%g", dv);
                cell = gv_dup_cstr(vb);
            }
            result->column_values[r * pc + p] = cell;
            if (!cell) { gv_free(ord); sql_group_free(groups, gn); sql_free_result(result); sql_set_error(eng, "Out of memory"); return -1; }
        }
    }

    gv_free(ord);
    sql_group_free(groups, gn);
    return 0;
}

/* Executor: DELETE */

static int sql_exec_delete(GV_SQLEngine *eng, const GV_SQLStmt *stmt, GV_SQLResult *result)
{
    GV_Database *db = eng->db;
    GV_SoAStorage *soa = db->soa_storage;
    if (!soa) { sql_set_error(eng, "Database has no storage"); return -1; }

    size_t total = soa->count;
    size_t deleted = 0;

    /* Collect matching indices first (iterate, then delete) */
    size_t cap = 256;
    size_t *del_idx = (size_t *)gv_alloc(cap * sizeof(size_t));
    if (!del_idx) { sql_set_error(eng, "Out of memory"); return -1; }

    for (size_t i = 0; i < total; i++) {
        if (soa_storage_is_deleted(soa, i)) continue;

        GV_Vector view;
        if (soa_storage_get_vector_view(soa, i, &view) != 0) continue;

        if (sql_eval_where(stmt->where, &view) == 1) {
            if (deleted >= cap) {
                cap *= 2;
                size_t *tmp = (size_t *)gv_realloc(del_idx, cap * sizeof(size_t));
                if (!tmp) { gv_free(del_idx); sql_set_error(eng, "Out of memory"); return -1; }
                del_idx = tmp;
            }
            del_idx[deleted++] = i;
        }
    }

    /* Perform deletions */
    for (size_t i = 0; i < deleted; i++) {
        db_delete_vector_by_index(db, del_idx[i]);
    }
    gv_free(del_idx);

    /* Build result: single row with delete count */
    memset(result, 0, sizeof(*result));
    result->row_count = 1;
    result->column_count = 1;
    result->column_names = (char **)gv_calloc(1, sizeof(char *));
    if (result->column_names) {
        result->column_names[0] = gv_dup_cstr("deleted_count");
    }
    result->indices = (size_t *)gv_calloc(1, sizeof(size_t));
    if (!result->indices || !result->column_names) {
        sql_free_result(result);
        sql_set_error(eng, "Out of memory");
        return -1;
    }
    result->indices[0] = deleted;
    return 0;
}

/* Executor: UPDATE */

static int sql_exec_update(GV_SQLEngine *eng, const GV_SQLStmt *stmt, GV_SQLResult *result)
{
    GV_Database *db = eng->db;
    GV_SoAStorage *soa = db->soa_storage;
    if (!soa) { sql_set_error(eng, "Database has no storage"); return -1; }

    size_t total = soa->count;
    size_t updated = 0;

    for (size_t i = 0; i < total; i++) {
        if (soa_storage_is_deleted(soa, i)) continue;

        GV_Vector view;
        if (soa_storage_get_vector_view(soa, i, &view) != 0) continue;

        if (sql_eval_where(stmt->where, &view) != 1) continue;

        /* Apply SET clauses as metadata updates */
        const char **keys = (const char **)gv_alloc(stmt->set_count * sizeof(char *));
        const char **vals = (const char **)gv_alloc(stmt->set_count * sizeof(char *));
        if (!keys || !vals) {
            gv_free(keys);
            gv_free(vals);
            sql_set_error(eng, "Out of memory");
            return -1;
        }
        for (size_t j = 0; j < stmt->set_count; j++) {
            keys[j] = stmt->set_clauses[j].field;
            vals[j] = stmt->set_clauses[j].value;
        }
        db_update_vector_metadata(db, i, keys, vals, stmt->set_count);
        gv_free(keys);
        gv_free(vals);
        updated++;
    }

    /* Build result: single row with update count */
    memset(result, 0, sizeof(*result));
    result->row_count = 1;
    result->column_count = 1;
    result->column_names = (char **)gv_calloc(1, sizeof(char *));
    if (result->column_names) {
        result->column_names[0] = gv_dup_cstr("updated_count");
    }
    result->indices = (size_t *)gv_calloc(1, sizeof(size_t));
    if (!result->indices || !result->column_names) {
        sql_free_result(result);
        sql_set_error(eng, "Out of memory");
        return -1;
    }
    result->indices[0] = updated;
    return 0;
}

/* Executor: INSERT */

static int sql_exec_insert(GV_SQLEngine *eng, const GV_SQLStmt *stmt, GV_SQLResult *result)
{
    GV_Database *db = eng->db;
    if (!stmt->insert_vector || stmt->insert_dim == 0) {
        sql_set_error(eng, "INSERT requires a vector value");
        return -1;
    }
    size_t dbdim = database_dimension(db);
    if (dbdim != 0 && stmt->insert_dim != dbdim) {
        sql_set_error(eng, "INSERT vector dimension %zu does not match database dimension %zu",
                      stmt->insert_dim, dbdim);
        return -1;
    }

    int rc;
    if (stmt->set_count > 0) {
        const char **keys = (const char **)gv_alloc(stmt->set_count * sizeof(char *));
        const char **vals = (const char **)gv_alloc(stmt->set_count * sizeof(char *));
        if (!keys || !vals) {
            gv_free(keys); gv_free(vals);
            sql_set_error(eng, "Out of memory");
            return -1;
        }
        for (size_t i = 0; i < stmt->set_count; i++) {
            keys[i] = stmt->set_clauses[i].field;
            vals[i] = stmt->set_clauses[i].value;
        }
        rc = db_add_vector_with_rich_metadata(db, stmt->insert_vector, stmt->insert_dim,
                                              keys, vals, stmt->set_count);
        gv_free(keys);
        gv_free(vals);
    } else {
        rc = db_add_vector(db, stmt->insert_vector, stmt->insert_dim);
    }
    if (rc < 0) {
        sql_set_error(eng, "INSERT failed");
        return -1;
    }

    memset(result, 0, sizeof(*result));
    result->row_count = 1;
    result->column_count = 1;
    result->column_names = (char **)gv_calloc(1, sizeof(char *));
    result->indices = (size_t *)gv_calloc(1, sizeof(size_t));
    if (!result->column_names || !result->indices) {
        sql_free_result(result);
        sql_set_error(eng, "Out of memory");
        return -1;
    }
    result->column_names[0] = gv_dup_cstr("inserted");
    result->indices[0] = 1;
    return 0;
}

/* Public API */

GV_SQLEngine *sql_create(void *db)
{
    if (!db) return NULL;

    GV_SQLEngine *eng = (GV_SQLEngine *)gv_calloc(1, sizeof(GV_SQLEngine));
    if (!eng) return NULL;

    eng->db = (GV_Database *)db;
    if (pthread_mutex_init(&eng->mutex, NULL) != 0) {
        gv_free(eng);
        return NULL;
    }
    eng->last_error[0] = '\0';
    return eng;
}

void sql_destroy(GV_SQLEngine *eng)
{
    if (!eng) return;
    pthread_mutex_destroy(&eng->mutex);
    gv_free(eng);
}

int sql_execute(GV_SQLEngine *eng, const char *query, GV_SQLResult *result)
{
    if (!eng || !query || !result) return -1;

    pthread_mutex_lock(&eng->mutex);
    eng->last_error[0] = '\0';

    /* Tokenize */
    GV_SQLTokenBuf tbuf;
    memset(&tbuf, 0, sizeof(tbuf));
    if (sql_tokenize(&tbuf, query) != 0) {
        sql_set_error(eng, "%s", tbuf.error);
        sql_tokenbuf_free(&tbuf);
        pthread_mutex_unlock(&eng->mutex);
        return -1;
    }

    /* Parse */
    GV_SQLStmt *stmt = sql_parse(&tbuf);
    if (!stmt) {
        if (tbuf.error[0])
            sql_set_error(eng, "Parse error: %s", tbuf.error);
        else
            sql_set_error(eng, "Parse error: invalid SQL syntax");
        sql_tokenbuf_free(&tbuf);
        pthread_mutex_unlock(&eng->mutex);
        return -1;
    }

    /* Check for trailing tokens */
    if (sql_consume_trailing_semicolons(&tbuf) != 0) {
        sql_set_error(eng, "Parse error: unexpected tokens after statement");
        sql_stmt_free(stmt);
        sql_tokenbuf_free(&tbuf);
        pthread_mutex_unlock(&eng->mutex);
        return -1;
    }

    /* Execute */
    int rc = -1;
    switch (stmt->type) {
    case GV_SQL_STMT_SELECT: {
        /* A projection mixing aggregate and plain columns is only meaningful with
         * GROUP BY. Without it, the parser populated neither select_columns nor
         * agg_kind, so the where-scan path below would silently return raw rows.
         * Reject it instead of producing a wrong result. */
        int sel_has_agg = 0, sel_has_plain = 0;
        for (size_t i = 0; i < stmt->proj_count; i++) {
            if (stmt->proj[i].agg != GV_SQL_AGG_NONE) sel_has_agg = 1;
            else sel_has_plain = 1;
        }
        /* Without GROUP BY a multi-column projection sets neither select_columns
         * nor agg_kind. Mixing aggregate and plain columns is ill-defined and
         * rejected; multiple aggregates over the whole table are computed by the
         * dedicated multi-aggregate executor (one row, one column per aggregate). */
        if (stmt->group_by_count == 0 && sel_has_agg && sel_has_plain && stmt->proj_count > 1) {
            sql_set_error(eng, "aggregate and non-aggregate columns cannot be mixed without GROUP BY");
            rc = -1;
        } else if (stmt->group_by_count == 0 && sel_has_agg && stmt->proj_count > 1) {
            rc = sql_exec_multi_aggregate(eng, stmt, result);
        } else if (stmt->group_by_count > 0) {
            rc = sql_exec_group_by(eng, stmt, result);
        } else if (stmt->agg_kind != GV_SQL_AGG_NONE) {
            rc = sql_exec_aggregate(eng, stmt, result);
        } else if (stmt->has_ann) {
            rc = sql_exec_ann(eng, stmt, result);
        } else {
            rc = sql_exec_where_scan(eng, stmt, result);
        }
        break;
    }
    case GV_SQL_STMT_DELETE:
        rc = sql_exec_delete(eng, stmt, result);
        break;
    case GV_SQL_STMT_UPDATE:
        rc = sql_exec_update(eng, stmt, result);
        break;
    case GV_SQL_STMT_INSERT:
        rc = sql_exec_insert(eng, stmt, result);
        break;
    }

    sql_stmt_free(stmt);
    sql_tokenbuf_free(&tbuf);
    pthread_mutex_unlock(&eng->mutex);
    return rc;
}

void sql_free_result(GV_SQLResult *result)
{
    if (!result) return;

    gv_free(result->indices);
    gv_free(result->distances);

    if (result->metadata_jsons) {
        for (size_t i = 0; i < result->row_count; i++)
            gv_free(result->metadata_jsons[i]);
        gv_free(result->metadata_jsons);
    }

    if (result->column_values) {
        size_t cells = result->row_count * result->column_count;
        for (size_t i = 0; i < cells; i++)
            gv_free(result->column_values[i]);
        gv_free(result->column_values);
    }

    if (result->column_names) {
        for (size_t i = 0; i < result->column_count; i++)
            gv_free(result->column_names[i]);
        gv_free(result->column_names);
    }

    memset(result, 0, sizeof(*result));
}

const char *sql_last_error(const GV_SQLEngine *eng)
{
    if (!eng) return "";
    return eng->last_error;
}

int sql_explain(GV_SQLEngine *eng, const char *query, char *plan, size_t plan_size)
{
    if (!eng || !query || !plan || plan_size == 0) return -1;

    pthread_mutex_lock(&eng->mutex);
    eng->last_error[0] = '\0';
    plan[0] = '\0';

    /* Tokenize */
    GV_SQLTokenBuf tbuf;
    memset(&tbuf, 0, sizeof(tbuf));
    if (sql_tokenize(&tbuf, query) != 0) {
        sql_set_error(eng, "%s", tbuf.error);
        sql_tokenbuf_free(&tbuf);
        pthread_mutex_unlock(&eng->mutex);
        return -1;
    }

    /* Parse */
    GV_SQLStmt *stmt = sql_parse(&tbuf);
    if (!stmt) {
        if (tbuf.error[0])
            sql_set_error(eng, "Parse error: %s", tbuf.error);
        else
            sql_set_error(eng, "Parse error: invalid SQL syntax");
        sql_tokenbuf_free(&tbuf);
        pthread_mutex_unlock(&eng->mutex);
        return -1;
    }

    if (sql_consume_trailing_semicolons(&tbuf) != 0) {
        sql_set_error(eng, "Parse error: unexpected tokens after statement");
        sql_stmt_free(stmt);
        sql_tokenbuf_free(&tbuf);
        pthread_mutex_unlock(&eng->mutex);
        return -1;
    }

    GV_Database *db = eng->db;
    size_t total_vectors = db->count;

    /* Determine index type name */
    const char *index_name;
    switch (db->index_type) {
    case GV_INDEX_TYPE_KDTREE:  index_name = "KDTREE";  break;
    case GV_INDEX_TYPE_HNSW:    index_name = "HNSW";    break;
    case GV_INDEX_TYPE_IVFPQ:   index_name = "IVFPQ";   break;
    case GV_INDEX_TYPE_SPARSE:  index_name = "SPARSE";   break;
    case GV_INDEX_TYPE_FLAT:    index_name = "FLAT";     break;
    case GV_INDEX_TYPE_IVFFLAT: index_name = "IVFFLAT"; break;
    case GV_INDEX_TYPE_IVFSQ8:  index_name = "IVFSQ8";  break;
    case GV_INDEX_TYPE_IVFTURBOQUANT: index_name = "IVFTURBOQUANT"; break;
    case GV_INDEX_TYPE_IVFDISK: index_name = "IVFDISK"; break;
    case GV_INDEX_TYPE_PQ:      index_name = "PQ";       break;
    case GV_INDEX_TYPE_LSH:     index_name = "LSH";      break;
    default:                    index_name = "UNKNOWN";  break;
    }

    size_t off = 0;

    switch (stmt->type) {
    case GV_SQL_STMT_SELECT:
        if (stmt->agg_kind != GV_SQL_AGG_NONE) {
            const char *aggname =
                stmt->agg_kind == GV_SQL_AGG_SUM ? "SUM" :
                stmt->agg_kind == GV_SQL_AGG_MIN ? "MIN" :
                stmt->agg_kind == GV_SQL_AGG_MAX ? "MAX" :
                stmt->agg_kind == GV_SQL_AGG_AVG ? "AVG" : "COUNT";
            off += (size_t)snprintf(plan + off, plan_size - off,
                "EXPLAIN: SELECT %s(%s)\n"
                "  Strategy: FULL_SCAN\n"
                "  Index: %s\n"
                "  Estimated rows: %zu\n"
                "  Filter: %s\n",
                aggname, stmt->agg_column ? stmt->agg_column : "*",
                index_name, total_vectors,
                stmt->where ? "WHERE predicate (post-filter)" : "NONE");
        } else if (stmt->has_ann) {
            off += (size_t)snprintf(plan + off, plan_size - off,
                "EXPLAIN: SELECT with ANN\n"
                "  Strategy: INDEX_ANN_SEARCH\n"
                "  Index: %s\n"
                "  Metric: %s\n"
                "  k: %zu\n"
                "  Query dimension: %zu\n"
                "  Total vectors: %zu\n"
                "  Filter: %s\n",
                index_name,
                stmt->ann.metric == GV_DISTANCE_COSINE ? "COSINE" :
                stmt->ann.metric == GV_DISTANCE_EUCLIDEAN ? "EUCLIDEAN" : "DOT_PRODUCT",
                stmt->ann.k,
                stmt->ann.query_dim,
                total_vectors,
                stmt->where ? "WHERE predicate (post-filter on ANN results)" : "NONE");
            if (stmt->where) {
                off += (size_t)snprintf(plan + off, plan_size - off,
                    "  Oversample factor: 4x\n");
            }
        } else {
            off += (size_t)snprintf(plan + off, plan_size - off,
                "EXPLAIN: SELECT with WHERE\n"
                "  Strategy: FULL_SCAN\n"
                "  Index: %s (not used for metadata-only query)\n"
                "  Estimated rows: %zu\n"
                "  Filter: WHERE predicate\n",
                index_name, total_vectors);
        }
        if (stmt->has_limit) {
            off += (size_t)snprintf(plan + off, plan_size - off,
                "  Limit: %zu\n", stmt->limit);
        }
        if (stmt->order_field) {
            (void)snprintf(plan + off, plan_size - off,
                "  Order by: %s %s\n", stmt->order_field,
                stmt->order_desc ? "DESC" : "ASC");
        }
        break;

    case GV_SQL_STMT_DELETE:
        (void)snprintf(plan + off, plan_size - off,
            "EXPLAIN: DELETE\n"
            "  Strategy: FULL_SCAN + DELETE\n"
            "  Index: %s\n"
            "  Estimated rows scanned: %zu\n"
            "  Filter: WHERE predicate\n",
            index_name, total_vectors);
        break;

    case GV_SQL_STMT_UPDATE:
        (void)snprintf(plan + off, plan_size - off,
            "EXPLAIN: UPDATE\n"
            "  Strategy: FULL_SCAN + METADATA_UPDATE\n"
            "  Index: %s\n"
            "  Estimated rows scanned: %zu\n"
            "  Filter: WHERE predicate\n"
            "  SET clauses: %zu\n",
            index_name, total_vectors, stmt->set_count);
        break;

    case GV_SQL_STMT_INSERT:
        (void)snprintf(plan + off, plan_size - off,
            "EXPLAIN: INSERT\n"
            "  Strategy: SINGLE_ROW_INSERT\n"
            "  Index: %s\n"
            "  Vector dimension: %zu\n"
            "  Metadata columns: %zu\n",
            index_name, stmt->insert_dim, stmt->set_count);
        break;
    }

    sql_stmt_free(stmt);
    sql_tokenbuf_free(&tbuf);
    pthread_mutex_unlock(&eng->mutex);
    return 0;
}
