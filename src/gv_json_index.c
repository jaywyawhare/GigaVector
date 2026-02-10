/**
 * @file gv_json_index.c
 * @brief JSON path indexing implementation for fast filtered vector search.
 *
 * Each registered path maintains a sorted array of (value, vector_index)
 * pairs.  Lookups use binary search; range queries find lower and upper
 * bounds.  Thread-safety is provided by a single pthread_rwlock_t.
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>

#include "gigavector/gv_json_index.h"
#include "gigavector/gv_json.h"

/* ============================================================================
 * Internal Constants
 * ============================================================================ */

#define GV_JPI_INITIAL_CAP  16
#define GV_JPI_PATH_MAXLEN 256

/* ============================================================================
 * Internal Entry Types
 * ============================================================================ */

typedef struct {
    char  *value;
    size_t vector_index;
} GV_JPStringEntry;

typedef struct {
    int64_t value;
    size_t  vector_index;
} GV_JPIntEntry;

typedef struct {
    double value;
    size_t vector_index;
} GV_JPFloatEntry;

typedef struct {
    bool   value;
    size_t vector_index;
} GV_JPBoolEntry;

/* ============================================================================
 * Per-Path Index
 * ============================================================================ */

typedef struct {
    char             path[GV_JPI_PATH_MAXLEN];
    GV_JSONPathType  type;

    union {
        struct { GV_JPStringEntry *entries; size_t count; size_t capacity; } str;
        struct { GV_JPIntEntry    *entries; size_t count; size_t capacity; } int_data;
        struct { GV_JPFloatEntry  *entries; size_t count; size_t capacity; } float_data;
        struct { GV_JPBoolEntry   *entries; size_t count; size_t capacity; } bool_data;
    } data;
} GV_JPPathIndex;

/* ============================================================================
 * Top-Level Index Structure
 * ============================================================================ */

struct GV_JSONPathIndex {
    GV_JPPathIndex  paths[GV_JSON_INDEX_MAX_PATHS];
    size_t          path_count;
    pthread_rwlock_t rwlock;
};

/* ============================================================================
 * Helpers: find a per-path index by name
 * ============================================================================ */

static GV_JPPathIndex *jpi_find_path(const GV_JSONPathIndex *idx, const char *path) {
    for (size_t i = 0; i < idx->path_count; i++) {
        if (strcmp(idx->paths[i].path, path) == 0) {
            return (GV_JPPathIndex *)&idx->paths[i];
        }
    }
    return NULL;
}

/* ============================================================================
 * Helpers: JSON path resolution
 *
 * Supports dot-notation ("a.b.c") and bracket array access ("a[0].b").
 * The path is split into segments; each segment is either an object key
 * or an array index.
 * ============================================================================ */

/**
 * @brief Resolve a dot-notation / bracket path against a parsed JSON tree.
 *
 * Examples:
 *   "address.city"     -> root["address"]["city"]
 *   "tags[0]"          -> root["tags"][0]
 *   "a[1].b"           -> root["a"][1]["b"]
 */
static GV_JsonValue *jpi_resolve_path(const GV_JsonValue *root, const char *path) {
    if (!root || !path || !*path) return NULL;

    char buf[GV_JPI_PATH_MAXLEN];
    strncpy(buf, path, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    const GV_JsonValue *current = root;
    char *saveptr = NULL;
    char *token = strtok_r(buf, ".", &saveptr);

    while (token && current) {
        /* Check for bracket array access: "key[N]" or just "[N]" */
        char *bracket = strchr(token, '[');
        if (bracket) {
            /* There may be a key part before the bracket */
            if (bracket != token) {
                /* Extract the key portion */
                *bracket = '\0';
                current = gv_json_object_get(current, token);
                if (!current) return NULL;
            }

            /* Process one or more bracket indices: [0][1]... */
            char *p = bracket + 1;  /* skip past '[' after we NUL'd it,
                                       but we need the original — use offset */
            /* Re-parse from bracket position in original segment */
            /* Since we NUL'd bracket, rebuild pointer arithmetic */
            p = bracket + 1; /* points into buf, after the '[' */
            while (*p) {
                char *end_bracket = strchr(p, ']');
                if (!end_bracket) return NULL;

                *end_bracket = '\0';
                char *endptr;
                long idx_val = strtol(p, &endptr, 10);
                if (*endptr != '\0' || idx_val < 0) return NULL;

                current = gv_json_array_get(current, (size_t)idx_val);
                if (!current) return NULL;

                p = end_bracket + 1;
                if (*p == '[') {
                    p++; /* skip next '[' */
                } else {
                    break;
                }
            }
        } else {
            /* Plain object key */
            if (current->type == GV_JSON_OBJECT) {
                current = gv_json_object_get(current, token);
            } else if (current->type == GV_JSON_ARRAY) {
                /* Allow numeric token as array index (compat with gv_json_get_path) */
                char *endptr;
                long idx_val = strtol(token, &endptr, 10);
                if (*endptr != '\0' || idx_val < 0) return NULL;
                current = gv_json_array_get(current, (size_t)idx_val);
            } else {
                return NULL;
            }
        }

        token = strtok_r(NULL, ".", &saveptr);
    }

    return (GV_JsonValue *)current;
}

/* ============================================================================
 * Helpers: binary search — strings (sorted by strcmp)
 * ============================================================================ */

static size_t jpi_str_lower_bound(const GV_JPStringEntry *entries, size_t count,
                                  const char *value) {
    size_t lo = 0, hi = count;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (strcmp(entries[mid].value, value) < 0) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

/* ============================================================================
 * Helpers: binary search — int64_t
 * ============================================================================ */

static size_t jpi_int_lower_bound(const GV_JPIntEntry *entries, size_t count,
                                  int64_t value) {
    size_t lo = 0, hi = count;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (entries[mid].value < value) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

static size_t jpi_int_upper_bound(const GV_JPIntEntry *entries, size_t count,
                                  int64_t value) {
    size_t lo = 0, hi = count;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (entries[mid].value <= value) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

/* ============================================================================
 * Helpers: binary search — double
 * ============================================================================ */

static size_t jpi_float_lower_bound(const GV_JPFloatEntry *entries, size_t count,
                                    double value) {
    size_t lo = 0, hi = count;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (entries[mid].value < value) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

static size_t jpi_float_upper_bound(const GV_JPFloatEntry *entries, size_t count,
                                    double value) {
    size_t lo = 0, hi = count;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (entries[mid].value <= value) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

/* ============================================================================
 * Helpers: per-path sorted insert
 * ============================================================================ */

static int jpi_insert_string(GV_JPPathIndex *pi, size_t vector_index, const char *value) {
    if (pi->data.str.count >= pi->data.str.capacity) {
        size_t new_cap = pi->data.str.capacity * 2;
        GV_JPStringEntry *tmp = (GV_JPStringEntry *)realloc(
            pi->data.str.entries, new_cap * sizeof(GV_JPStringEntry));
        if (!tmp) return -1;
        pi->data.str.entries = tmp;
        pi->data.str.capacity = new_cap;
    }

    char *dup = strdup(value);
    if (!dup) return -1;

    size_t pos = jpi_str_lower_bound(pi->data.str.entries, pi->data.str.count, value);
    if (pos < pi->data.str.count) {
        memmove(&pi->data.str.entries[pos + 1],
                &pi->data.str.entries[pos],
                (pi->data.str.count - pos) * sizeof(GV_JPStringEntry));
    }
    pi->data.str.entries[pos].value = dup;
    pi->data.str.entries[pos].vector_index = vector_index;
    pi->data.str.count++;
    return 0;
}

static int jpi_insert_int(GV_JPPathIndex *pi, size_t vector_index, int64_t value) {
    if (pi->data.int_data.count >= pi->data.int_data.capacity) {
        size_t new_cap = pi->data.int_data.capacity * 2;
        GV_JPIntEntry *tmp = (GV_JPIntEntry *)realloc(
            pi->data.int_data.entries, new_cap * sizeof(GV_JPIntEntry));
        if (!tmp) return -1;
        pi->data.int_data.entries = tmp;
        pi->data.int_data.capacity = new_cap;
    }

    size_t pos = jpi_int_lower_bound(pi->data.int_data.entries, pi->data.int_data.count, value);
    if (pos < pi->data.int_data.count) {
        memmove(&pi->data.int_data.entries[pos + 1],
                &pi->data.int_data.entries[pos],
                (pi->data.int_data.count - pos) * sizeof(GV_JPIntEntry));
    }
    pi->data.int_data.entries[pos].value = value;
    pi->data.int_data.entries[pos].vector_index = vector_index;
    pi->data.int_data.count++;
    return 0;
}

static int jpi_insert_float(GV_JPPathIndex *pi, size_t vector_index, double value) {
    if (pi->data.float_data.count >= pi->data.float_data.capacity) {
        size_t new_cap = pi->data.float_data.capacity * 2;
        GV_JPFloatEntry *tmp = (GV_JPFloatEntry *)realloc(
            pi->data.float_data.entries, new_cap * sizeof(GV_JPFloatEntry));
        if (!tmp) return -1;
        pi->data.float_data.entries = tmp;
        pi->data.float_data.capacity = new_cap;
    }

    size_t pos = jpi_float_lower_bound(pi->data.float_data.entries,
                                       pi->data.float_data.count, value);
    if (pos < pi->data.float_data.count) {
        memmove(&pi->data.float_data.entries[pos + 1],
                &pi->data.float_data.entries[pos],
                (pi->data.float_data.count - pos) * sizeof(GV_JPFloatEntry));
    }
    pi->data.float_data.entries[pos].value = value;
    pi->data.float_data.entries[pos].vector_index = vector_index;
    pi->data.float_data.count++;
    return 0;
}

static int jpi_insert_bool(GV_JPPathIndex *pi, size_t vector_index, bool value) {
    if (pi->data.bool_data.count >= pi->data.bool_data.capacity) {
        size_t new_cap = pi->data.bool_data.capacity * 2;
        GV_JPBoolEntry *tmp = (GV_JPBoolEntry *)realloc(
            pi->data.bool_data.entries, new_cap * sizeof(GV_JPBoolEntry));
        if (!tmp) return -1;
        pi->data.bool_data.entries = tmp;
        pi->data.bool_data.capacity = new_cap;
    }

    /* Keep false entries before true entries for consistent ordering */
    size_t insert_pos;
    if (!value) {
        insert_pos = 0;
        for (size_t i = 0; i < pi->data.bool_data.count; i++) {
            if (pi->data.bool_data.entries[i].value) {
                insert_pos = i;
                goto do_bool_insert;
            }
        }
        insert_pos = pi->data.bool_data.count;
    } else {
        insert_pos = pi->data.bool_data.count;
    }

do_bool_insert:
    if (insert_pos < pi->data.bool_data.count) {
        memmove(&pi->data.bool_data.entries[insert_pos + 1],
                &pi->data.bool_data.entries[insert_pos],
                (pi->data.bool_data.count - insert_pos) * sizeof(GV_JPBoolEntry));
    }
    pi->data.bool_data.entries[insert_pos].value = value;
    pi->data.bool_data.entries[insert_pos].vector_index = vector_index;
    pi->data.bool_data.count++;
    return 0;
}

/* ============================================================================
 * Helpers: per-path entry removal by vector_index
 * ============================================================================ */

static void jpi_remove_from_path(GV_JPPathIndex *pi, size_t vector_index) {
    switch (pi->type) {
        case GV_JP_STRING: {
            size_t w = 0;
            for (size_t i = 0; i < pi->data.str.count; i++) {
                if (pi->data.str.entries[i].vector_index != vector_index) {
                    if (w != i) {
                        pi->data.str.entries[w] = pi->data.str.entries[i];
                    }
                    w++;
                } else {
                    free(pi->data.str.entries[i].value);
                }
            }
            pi->data.str.count = w;
            break;
        }
        case GV_JP_INT: {
            size_t w = 0;
            for (size_t i = 0; i < pi->data.int_data.count; i++) {
                if (pi->data.int_data.entries[i].vector_index != vector_index) {
                    if (w != i) {
                        pi->data.int_data.entries[w] = pi->data.int_data.entries[i];
                    }
                    w++;
                }
            }
            pi->data.int_data.count = w;
            break;
        }
        case GV_JP_FLOAT: {
            size_t w = 0;
            for (size_t i = 0; i < pi->data.float_data.count; i++) {
                if (pi->data.float_data.entries[i].vector_index != vector_index) {
                    if (w != i) {
                        pi->data.float_data.entries[w] = pi->data.float_data.entries[i];
                    }
                    w++;
                }
            }
            pi->data.float_data.count = w;
            break;
        }
        case GV_JP_BOOL: {
            size_t w = 0;
            for (size_t i = 0; i < pi->data.bool_data.count; i++) {
                if (pi->data.bool_data.entries[i].vector_index != vector_index) {
                    if (w != i) {
                        pi->data.bool_data.entries[w] = pi->data.bool_data.entries[i];
                    }
                    w++;
                }
            }
            pi->data.bool_data.count = w;
            break;
        }
    }
}

/* ============================================================================
 * Helpers: free a single per-path index's entries
 * ============================================================================ */

static void jpi_free_path_entries(GV_JPPathIndex *pi) {
    switch (pi->type) {
        case GV_JP_STRING:
            for (size_t i = 0; i < pi->data.str.count; i++) {
                free(pi->data.str.entries[i].value);
            }
            free(pi->data.str.entries);
            pi->data.str.entries = NULL;
            pi->data.str.count = 0;
            pi->data.str.capacity = 0;
            break;
        case GV_JP_INT:
            free(pi->data.int_data.entries);
            pi->data.int_data.entries = NULL;
            pi->data.int_data.count = 0;
            pi->data.int_data.capacity = 0;
            break;
        case GV_JP_FLOAT:
            free(pi->data.float_data.entries);
            pi->data.float_data.entries = NULL;
            pi->data.float_data.count = 0;
            pi->data.float_data.capacity = 0;
            break;
        case GV_JP_BOOL:
            free(pi->data.bool_data.entries);
            pi->data.bool_data.entries = NULL;
            pi->data.bool_data.count = 0;
            pi->data.bool_data.capacity = 0;
            break;
    }
}

/* ============================================================================
 * Helpers: initialise per-path entry storage
 * ============================================================================ */

static int jpi_init_path_entries(GV_JPPathIndex *pi) {
    switch (pi->type) {
        case GV_JP_STRING:
            pi->data.str.entries = (GV_JPStringEntry *)malloc(
                GV_JPI_INITIAL_CAP * sizeof(GV_JPStringEntry));
            if (!pi->data.str.entries) return -1;
            pi->data.str.count = 0;
            pi->data.str.capacity = GV_JPI_INITIAL_CAP;
            break;
        case GV_JP_INT:
            pi->data.int_data.entries = (GV_JPIntEntry *)malloc(
                GV_JPI_INITIAL_CAP * sizeof(GV_JPIntEntry));
            if (!pi->data.int_data.entries) return -1;
            pi->data.int_data.count = 0;
            pi->data.int_data.capacity = GV_JPI_INITIAL_CAP;
            break;
        case GV_JP_FLOAT:
            pi->data.float_data.entries = (GV_JPFloatEntry *)malloc(
                GV_JPI_INITIAL_CAP * sizeof(GV_JPFloatEntry));
            if (!pi->data.float_data.entries) return -1;
            pi->data.float_data.count = 0;
            pi->data.float_data.capacity = GV_JPI_INITIAL_CAP;
            break;
        case GV_JP_BOOL:
            pi->data.bool_data.entries = (GV_JPBoolEntry *)malloc(
                GV_JPI_INITIAL_CAP * sizeof(GV_JPBoolEntry));
            if (!pi->data.bool_data.entries) return -1;
            pi->data.bool_data.count = 0;
            pi->data.bool_data.capacity = GV_JPI_INITIAL_CAP;
            break;
    }
    return 0;
}

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

GV_JSONPathIndex *gv_json_index_create(void) {
    GV_JSONPathIndex *idx = (GV_JSONPathIndex *)calloc(1, sizeof(GV_JSONPathIndex));
    if (!idx) return NULL;

    idx->path_count = 0;

    if (pthread_rwlock_init(&idx->rwlock, NULL) != 0) {
        free(idx);
        return NULL;
    }

    return idx;
}

void gv_json_index_destroy(GV_JSONPathIndex *idx) {
    if (!idx) return;

    for (size_t i = 0; i < idx->path_count; i++) {
        jpi_free_path_entries(&idx->paths[i]);
    }

    pthread_rwlock_destroy(&idx->rwlock);
    free(idx);
}

/* ============================================================================
 * Path Registration
 * ============================================================================ */

int gv_json_index_add_path(GV_JSONPathIndex *idx, const GV_JSONPathConfig *config) {
    if (!idx || !config || !config->path) return -1;

    pthread_rwlock_wrlock(&idx->rwlock);

    /* Check limits */
    if (idx->path_count >= GV_JSON_INDEX_MAX_PATHS) {
        pthread_rwlock_unlock(&idx->rwlock);
        return -1;
    }

    /* Reject duplicates */
    if (jpi_find_path(idx, config->path) != NULL) {
        pthread_rwlock_unlock(&idx->rwlock);
        return -1;
    }

    GV_JPPathIndex *pi = &idx->paths[idx->path_count];
    memset(pi, 0, sizeof(GV_JPPathIndex));
    strncpy(pi->path, config->path, sizeof(pi->path) - 1);
    pi->path[sizeof(pi->path) - 1] = '\0';
    pi->type = config->type;

    if (jpi_init_path_entries(pi) != 0) {
        pthread_rwlock_unlock(&idx->rwlock);
        return -1;
    }

    idx->path_count++;
    pthread_rwlock_unlock(&idx->rwlock);
    return 0;
}

int gv_json_index_remove_path(GV_JSONPathIndex *idx, const char *path) {
    if (!idx || !path) return -1;

    pthread_rwlock_wrlock(&idx->rwlock);

    for (size_t i = 0; i < idx->path_count; i++) {
        if (strcmp(idx->paths[i].path, path) == 0) {
            jpi_free_path_entries(&idx->paths[i]);

            /* Shift remaining paths down */
            if (i < idx->path_count - 1) {
                memmove(&idx->paths[i], &idx->paths[i + 1],
                        (idx->path_count - i - 1) * sizeof(GV_JPPathIndex));
            }
            idx->path_count--;
            pthread_rwlock_unlock(&idx->rwlock);
            return 0;
        }
    }

    pthread_rwlock_unlock(&idx->rwlock);
    return -1; /* Path not found */
}

/* ============================================================================
 * Data Manipulation
 * ============================================================================ */

int gv_json_index_insert(GV_JSONPathIndex *idx, size_t vector_index, const char *json_str) {
    if (!idx || !json_str) return -1;

    /* Parse JSON outside the lock to minimise hold time */
    GV_JsonError err;
    GV_JsonValue *root = gv_json_parse(json_str, &err);
    if (!root) return -1;

    pthread_rwlock_wrlock(&idx->rwlock);

    for (size_t i = 0; i < idx->path_count; i++) {
        GV_JPPathIndex *pi = &idx->paths[i];
        GV_JsonValue *val = jpi_resolve_path(root, pi->path);
        if (!val) continue;

        switch (pi->type) {
            case GV_JP_STRING: {
                const char *s = gv_json_get_string(val);
                if (s) jpi_insert_string(pi, vector_index, s);
                break;
            }
            case GV_JP_INT: {
                double num;
                if (gv_json_get_number(val, &num) == GV_JSON_OK) {
                    jpi_insert_int(pi, vector_index, (int64_t)num);
                }
                break;
            }
            case GV_JP_FLOAT: {
                double num;
                if (gv_json_get_number(val, &num) == GV_JSON_OK) {
                    jpi_insert_float(pi, vector_index, num);
                }
                break;
            }
            case GV_JP_BOOL: {
                bool b;
                if (gv_json_get_bool(val, &b) == GV_JSON_OK) {
                    jpi_insert_bool(pi, vector_index, b);
                }
                break;
            }
        }
    }

    pthread_rwlock_unlock(&idx->rwlock);
    gv_json_free(root);
    return 0;
}

int gv_json_index_remove(GV_JSONPathIndex *idx, size_t vector_index) {
    if (!idx) return -1;

    pthread_rwlock_wrlock(&idx->rwlock);

    for (size_t i = 0; i < idx->path_count; i++) {
        jpi_remove_from_path(&idx->paths[i], vector_index);
    }

    pthread_rwlock_unlock(&idx->rwlock);
    return 0;
}

/* ============================================================================
 * Lookup
 * ============================================================================ */

int gv_json_index_lookup_string(const GV_JSONPathIndex *idx, const char *path,
                                const char *value, size_t *out_indices, size_t max_count) {
    if (!idx || !path || !value || !out_indices || max_count == 0) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&idx->rwlock);

    GV_JPPathIndex *pi = jpi_find_path(idx, path);
    if (!pi || pi->type != GV_JP_STRING) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);
        return -1;
    }

    const GV_JPStringEntry *entries = pi->data.str.entries;
    size_t count = pi->data.str.count;
    size_t lo = jpi_str_lower_bound(entries, count, value);
    size_t n = 0;

    for (size_t i = lo; i < count && n < max_count; i++) {
        if (strcmp(entries[i].value, value) != 0) break;
        out_indices[n++] = entries[i].vector_index;
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);
    return (int)n;
}

int gv_json_index_lookup_int_range(const GV_JSONPathIndex *idx, const char *path,
                                   int64_t min_val, int64_t max_val,
                                   size_t *out_indices, size_t max_count) {
    if (!idx || !path || !out_indices || max_count == 0) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&idx->rwlock);

    GV_JPPathIndex *pi = jpi_find_path(idx, path);
    if (!pi || pi->type != GV_JP_INT) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);
        return -1;
    }

    const GV_JPIntEntry *entries = pi->data.int_data.entries;
    size_t count = pi->data.int_data.count;

    size_t lo = jpi_int_lower_bound(entries, count, min_val);
    size_t hi = jpi_int_upper_bound(entries, count, max_val);
    size_t n = 0;

    for (size_t i = lo; i < hi && n < max_count; i++) {
        out_indices[n++] = entries[i].vector_index;
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);
    return (int)n;
}

int gv_json_index_lookup_float_range(const GV_JSONPathIndex *idx, const char *path,
                                     double min_val, double max_val,
                                     size_t *out_indices, size_t max_count) {
    if (!idx || !path || !out_indices || max_count == 0) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&idx->rwlock);

    GV_JPPathIndex *pi = jpi_find_path(idx, path);
    if (!pi || pi->type != GV_JP_FLOAT) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);
        return -1;
    }

    const GV_JPFloatEntry *entries = pi->data.float_data.entries;
    size_t count = pi->data.float_data.count;

    size_t lo = jpi_float_lower_bound(entries, count, min_val);
    size_t hi = jpi_float_upper_bound(entries, count, max_val);
    size_t n = 0;

    for (size_t i = lo; i < hi && n < max_count; i++) {
        out_indices[n++] = entries[i].vector_index;
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);
    return (int)n;
}

/* ============================================================================
 * Statistics
 * ============================================================================ */

size_t gv_json_index_count(const GV_JSONPathIndex *idx, const char *path) {
    if (!idx || !path) return 0;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&idx->rwlock);

    GV_JPPathIndex *pi = jpi_find_path(idx, path);
    if (!pi) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);
        return 0;
    }

    size_t result = 0;
    switch (pi->type) {
        case GV_JP_STRING: result = pi->data.str.count;        break;
        case GV_JP_INT:    result = pi->data.int_data.count;    break;
        case GV_JP_FLOAT:  result = pi->data.float_data.count;  break;
        case GV_JP_BOOL:   result = pi->data.bool_data.count;   break;
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);
    return result;
}

/* ============================================================================
 * Persistence
 * ============================================================================ */

static const char GV_JPI_MAGIC[] = "GV_JPI";
#define GV_JPI_MAGIC_LEN 6
#define GV_JPI_VERSION    1

int gv_json_index_save(const GV_JSONPathIndex *idx, const char *path_file) {
    if (!idx || !path_file) return -1;

    FILE *fp = fopen(path_file, "wb");
    if (!fp) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&idx->rwlock);

    /* Header */
    fwrite(GV_JPI_MAGIC, 1, GV_JPI_MAGIC_LEN, fp);
    uint32_t version = GV_JPI_VERSION;
    fwrite(&version, sizeof(version), 1, fp);

    /* Path count */
    uint32_t pc = (uint32_t)idx->path_count;
    fwrite(&pc, sizeof(pc), 1, fp);

    /* Each path */
    for (size_t i = 0; i < idx->path_count; i++) {
        const GV_JPPathIndex *pi = &idx->paths[i];

        /* Path string (length-prefixed) */
        uint32_t path_len = (uint32_t)strlen(pi->path);
        fwrite(&path_len, sizeof(path_len), 1, fp);
        fwrite(pi->path, 1, path_len, fp);

        /* Type */
        uint32_t type = (uint32_t)pi->type;
        fwrite(&type, sizeof(type), 1, fp);

        /* Entry count and entries */
        switch (pi->type) {
            case GV_JP_STRING: {
                uint64_t cnt = (uint64_t)pi->data.str.count;
                fwrite(&cnt, sizeof(cnt), 1, fp);
                for (size_t j = 0; j < pi->data.str.count; j++) {
                    uint32_t slen = (uint32_t)strlen(pi->data.str.entries[j].value);
                    fwrite(&slen, sizeof(slen), 1, fp);
                    fwrite(pi->data.str.entries[j].value, 1, slen, fp);
                    uint64_t vi = (uint64_t)pi->data.str.entries[j].vector_index;
                    fwrite(&vi, sizeof(vi), 1, fp);
                }
                break;
            }
            case GV_JP_INT: {
                uint64_t cnt = (uint64_t)pi->data.int_data.count;
                fwrite(&cnt, sizeof(cnt), 1, fp);
                for (size_t j = 0; j < pi->data.int_data.count; j++) {
                    fwrite(&pi->data.int_data.entries[j].value, sizeof(int64_t), 1, fp);
                    uint64_t vi = (uint64_t)pi->data.int_data.entries[j].vector_index;
                    fwrite(&vi, sizeof(vi), 1, fp);
                }
                break;
            }
            case GV_JP_FLOAT: {
                uint64_t cnt = (uint64_t)pi->data.float_data.count;
                fwrite(&cnt, sizeof(cnt), 1, fp);
                for (size_t j = 0; j < pi->data.float_data.count; j++) {
                    fwrite(&pi->data.float_data.entries[j].value, sizeof(double), 1, fp);
                    uint64_t vi = (uint64_t)pi->data.float_data.entries[j].vector_index;
                    fwrite(&vi, sizeof(vi), 1, fp);
                }
                break;
            }
            case GV_JP_BOOL: {
                uint64_t cnt = (uint64_t)pi->data.bool_data.count;
                fwrite(&cnt, sizeof(cnt), 1, fp);
                for (size_t j = 0; j < pi->data.bool_data.count; j++) {
                    uint8_t bval = pi->data.bool_data.entries[j].value ? 1 : 0;
                    fwrite(&bval, sizeof(bval), 1, fp);
                    uint64_t vi = (uint64_t)pi->data.bool_data.entries[j].vector_index;
                    fwrite(&vi, sizeof(vi), 1, fp);
                }
                break;
            }
        }
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);
    fclose(fp);
    return 0;
}

GV_JSONPathIndex *gv_json_index_load(const char *path_file) {
    if (!path_file) return NULL;

    FILE *fp = fopen(path_file, "rb");
    if (!fp) return NULL;

    /* Verify magic */
    char magic[GV_JPI_MAGIC_LEN];
    if (fread(magic, 1, GV_JPI_MAGIC_LEN, fp) != GV_JPI_MAGIC_LEN ||
        memcmp(magic, GV_JPI_MAGIC, GV_JPI_MAGIC_LEN) != 0) {
        fclose(fp);
        return NULL;
    }

    /* Verify version */
    uint32_t version;
    if (fread(&version, sizeof(version), 1, fp) != 1 || version != GV_JPI_VERSION) {
        fclose(fp);
        return NULL;
    }

    /* Path count */
    uint32_t pc;
    if (fread(&pc, sizeof(pc), 1, fp) != 1 || pc > GV_JSON_INDEX_MAX_PATHS) {
        fclose(fp);
        return NULL;
    }

    GV_JSONPathIndex *idx = gv_json_index_create();
    if (!idx) {
        fclose(fp);
        return NULL;
    }

    for (uint32_t i = 0; i < pc; i++) {
        /* Path string */
        uint32_t path_len;
        if (fread(&path_len, sizeof(path_len), 1, fp) != 1 ||
            path_len >= GV_JPI_PATH_MAXLEN) {
            gv_json_index_destroy(idx);
            fclose(fp);
            return NULL;
        }

        GV_JPPathIndex *pi = &idx->paths[idx->path_count];
        memset(pi, 0, sizeof(GV_JPPathIndex));

        if (fread(pi->path, 1, path_len, fp) != path_len) {
            gv_json_index_destroy(idx);
            fclose(fp);
            return NULL;
        }
        pi->path[path_len] = '\0';

        /* Type */
        uint32_t type;
        if (fread(&type, sizeof(type), 1, fp) != 1) {
            gv_json_index_destroy(idx);
            fclose(fp);
            return NULL;
        }
        pi->type = (GV_JSONPathType)type;

        /* Entry count */
        uint64_t cnt;
        if (fread(&cnt, sizeof(cnt), 1, fp) != 1) {
            gv_json_index_destroy(idx);
            fclose(fp);
            return NULL;
        }

        /* Allocate entries */
        size_t cap = (cnt > GV_JPI_INITIAL_CAP) ? cnt : GV_JPI_INITIAL_CAP;

        switch (pi->type) {
            case GV_JP_STRING: {
                pi->data.str.entries = (GV_JPStringEntry *)malloc(
                    cap * sizeof(GV_JPStringEntry));
                if (!pi->data.str.entries) {
                    gv_json_index_destroy(idx);
                    fclose(fp);
                    return NULL;
                }
                pi->data.str.capacity = cap;
                pi->data.str.count = 0;

                for (uint64_t j = 0; j < cnt; j++) {
                    uint32_t slen;
                    if (fread(&slen, sizeof(slen), 1, fp) != 1) {
                        gv_json_index_destroy(idx);
                        fclose(fp);
                        return NULL;
                    }
                    char *s = (char *)malloc(slen + 1);
                    if (!s) {
                        gv_json_index_destroy(idx);
                        fclose(fp);
                        return NULL;
                    }
                    if (fread(s, 1, slen, fp) != slen) {
                        free(s);
                        gv_json_index_destroy(idx);
                        fclose(fp);
                        return NULL;
                    }
                    s[slen] = '\0';

                    uint64_t vi;
                    if (fread(&vi, sizeof(vi), 1, fp) != 1) {
                        free(s);
                        gv_json_index_destroy(idx);
                        fclose(fp);
                        return NULL;
                    }

                    pi->data.str.entries[pi->data.str.count].value = s;
                    pi->data.str.entries[pi->data.str.count].vector_index = (size_t)vi;
                    pi->data.str.count++;
                }
                break;
            }
            case GV_JP_INT: {
                pi->data.int_data.entries = (GV_JPIntEntry *)malloc(
                    cap * sizeof(GV_JPIntEntry));
                if (!pi->data.int_data.entries) {
                    gv_json_index_destroy(idx);
                    fclose(fp);
                    return NULL;
                }
                pi->data.int_data.capacity = cap;
                pi->data.int_data.count = 0;

                for (uint64_t j = 0; j < cnt; j++) {
                    int64_t val;
                    uint64_t vi;
                    if (fread(&val, sizeof(val), 1, fp) != 1 ||
                        fread(&vi, sizeof(vi), 1, fp) != 1) {
                        gv_json_index_destroy(idx);
                        fclose(fp);
                        return NULL;
                    }
                    pi->data.int_data.entries[pi->data.int_data.count].value = val;
                    pi->data.int_data.entries[pi->data.int_data.count].vector_index = (size_t)vi;
                    pi->data.int_data.count++;
                }
                break;
            }
            case GV_JP_FLOAT: {
                pi->data.float_data.entries = (GV_JPFloatEntry *)malloc(
                    cap * sizeof(GV_JPFloatEntry));
                if (!pi->data.float_data.entries) {
                    gv_json_index_destroy(idx);
                    fclose(fp);
                    return NULL;
                }
                pi->data.float_data.capacity = cap;
                pi->data.float_data.count = 0;

                for (uint64_t j = 0; j < cnt; j++) {
                    double val;
                    uint64_t vi;
                    if (fread(&val, sizeof(val), 1, fp) != 1 ||
                        fread(&vi, sizeof(vi), 1, fp) != 1) {
                        gv_json_index_destroy(idx);
                        fclose(fp);
                        return NULL;
                    }
                    pi->data.float_data.entries[pi->data.float_data.count].value = val;
                    pi->data.float_data.entries[pi->data.float_data.count].vector_index = (size_t)vi;
                    pi->data.float_data.count++;
                }
                break;
            }
            case GV_JP_BOOL: {
                pi->data.bool_data.entries = (GV_JPBoolEntry *)malloc(
                    cap * sizeof(GV_JPBoolEntry));
                if (!pi->data.bool_data.entries) {
                    gv_json_index_destroy(idx);
                    fclose(fp);
                    return NULL;
                }
                pi->data.bool_data.capacity = cap;
                pi->data.bool_data.count = 0;

                for (uint64_t j = 0; j < cnt; j++) {
                    uint8_t bval;
                    uint64_t vi;
                    if (fread(&bval, sizeof(bval), 1, fp) != 1 ||
                        fread(&vi, sizeof(vi), 1, fp) != 1) {
                        gv_json_index_destroy(idx);
                        fclose(fp);
                        return NULL;
                    }
                    pi->data.bool_data.entries[pi->data.bool_data.count].value = bval ? true : false;
                    pi->data.bool_data.entries[pi->data.bool_data.count].vector_index = (size_t)vi;
                    pi->data.bool_data.count++;
                }
                break;
            }
        }

        idx->path_count++;
    }

    fclose(fp);
    return idx;
}
