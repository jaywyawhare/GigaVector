#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "gigavector/gv_payload_index.h"

/*  Internal data structures                                          */

#define GV_PAYLOAD_INITIAL_CAP 16

typedef struct {
    char name[64];
    GV_FieldType type;
} GV_FieldSchema;

typedef struct {
    size_t vector_id;
    int64_t value;
} GV_IntEntry;

typedef struct {
    size_t vector_id;
    double value;
} GV_FloatEntry;

typedef struct {
    size_t vector_id;
    char *value;
} GV_StringEntry;

typedef struct {
    size_t vector_id;
    int value;
} GV_BoolEntry;

typedef struct {
    GV_FieldSchema schema;
    union {
        struct { GV_IntEntry    *entries; size_t count; size_t capacity; } int_data;
        struct { GV_FloatEntry  *entries; size_t count; size_t capacity; } float_data;
        struct { GV_StringEntry *entries; size_t count; size_t capacity; } string_data;
        struct { GV_BoolEntry   *entries; size_t count; size_t capacity; } bool_data;
    } data;
} GV_FieldIndex;

struct GV_PayloadIndex {
    GV_FieldIndex *fields;
    size_t field_count;
    size_t field_capacity;
};

/*  Helpers: find field by name                                       */

static GV_FieldIndex *gv_find_field(const GV_PayloadIndex *idx, const char *name) {
    for (size_t i = 0; i < idx->field_count; i++) {
        if (strcmp(idx->fields[i].schema.name, name) == 0) {
            return &idx->fields[i];
        }
    }
    return NULL;
}

/*  Helpers: sorted-insert utilities for int entries                   */

/**
 * @brief Binary search for the insertion position in a sorted int array.
 *        Returns the index where `value` should be inserted to keep order.
 */
static size_t gv_int_lower_bound(const GV_IntEntry *entries, size_t count, int64_t value) {
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

/**
 * @brief Upper bound: first position where entry.value > value.
 */
static size_t gv_int_upper_bound(const GV_IntEntry *entries, size_t count, int64_t value) {
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

/*  Helpers: sorted-insert utilities for float entries                 */

static size_t gv_float_lower_bound(const GV_FloatEntry *entries, size_t count, double value) {
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

static size_t gv_float_upper_bound(const GV_FloatEntry *entries, size_t count, double value) {
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

/*  Helpers: sorted-insert utilities for string entries                */

static size_t gv_string_lower_bound(const GV_StringEntry *entries, size_t count, const char *value) {
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

/*  Helpers: size_t array sorting and intersection                    */

static int gv_compare_size_t(const void *a, const void *b) {
    size_t va = *(const size_t *)a;
    size_t vb = *(const size_t *)b;
    if (va < vb) return -1;
    if (va > vb) return 1;
    return 0;
}

/**
 * @brief Intersect two sorted arrays of size_t, writing results to out.
 *        Returns the number of elements in the intersection.
 */
static size_t gv_intersect_sorted(const size_t *a, size_t a_len,
                                   const size_t *b, size_t b_len,
                                   size_t *out, size_t max_out) {
    size_t i = 0, j = 0, k = 0;
    while (i < a_len && j < b_len && k < max_out) {
        if (a[i] < b[j]) {
            i++;
        } else if (a[i] > b[j]) {
            j++;
        } else {
            out[k++] = a[i];
            i++;
            j++;
        }
    }
    return k;
}

/*  Create / Destroy                                                  */

GV_PayloadIndex *gv_payload_index_create(void) {
    GV_PayloadIndex *idx = (GV_PayloadIndex *)malloc(sizeof(GV_PayloadIndex));
    if (idx == NULL) {
        return NULL;
    }
    idx->field_count = 0;
    idx->field_capacity = GV_PAYLOAD_INITIAL_CAP;
    idx->fields = (GV_FieldIndex *)calloc(idx->field_capacity, sizeof(GV_FieldIndex));
    if (idx->fields == NULL) {
        free(idx);
        return NULL;
    }
    return idx;
}

void gv_payload_index_destroy(GV_PayloadIndex *idx) {
    if (idx == NULL) {
        return;
    }
    for (size_t i = 0; i < idx->field_count; i++) {
        GV_FieldIndex *fi = &idx->fields[i];
        switch (fi->schema.type) {
            case GV_FIELD_INT:
                free(fi->data.int_data.entries);
                break;
            case GV_FIELD_FLOAT:
                free(fi->data.float_data.entries);
                break;
            case GV_FIELD_STRING:
                for (size_t j = 0; j < fi->data.string_data.count; j++) {
                    free(fi->data.string_data.entries[j].value);
                }
                free(fi->data.string_data.entries);
                break;
            case GV_FIELD_BOOL:
                free(fi->data.bool_data.entries);
                break;
        }
    }
    free(idx->fields);
    free(idx);
}

/*  Schema management                                                 */

int gv_payload_index_add_field(GV_PayloadIndex *idx, const char *name, GV_FieldType type) {
    if (idx == NULL || name == NULL) {
        return -1;
    }
    /* Reject duplicate field names */
    if (gv_find_field(idx, name) != NULL) {
        return -1;
    }
    /* Grow fields array if needed */
    if (idx->field_count >= idx->field_capacity) {
        size_t new_cap = idx->field_capacity * 2;
        GV_FieldIndex *tmp = (GV_FieldIndex *)realloc(idx->fields, new_cap * sizeof(GV_FieldIndex));
        if (tmp == NULL) {
            return -1;
        }
        memset(tmp + idx->field_capacity, 0, (new_cap - idx->field_capacity) * sizeof(GV_FieldIndex));
        idx->fields = tmp;
        idx->field_capacity = new_cap;
    }

    GV_FieldIndex *fi = &idx->fields[idx->field_count];
    memset(fi, 0, sizeof(GV_FieldIndex));
    strncpy(fi->schema.name, name, sizeof(fi->schema.name) - 1);
    fi->schema.name[sizeof(fi->schema.name) - 1] = '\0';
    fi->schema.type = type;

    /* Pre-allocate initial entry storage */
    switch (type) {
        case GV_FIELD_INT:
            fi->data.int_data.entries = (GV_IntEntry *)malloc(GV_PAYLOAD_INITIAL_CAP * sizeof(GV_IntEntry));
            if (fi->data.int_data.entries == NULL) return -1;
            fi->data.int_data.count = 0;
            fi->data.int_data.capacity = GV_PAYLOAD_INITIAL_CAP;
            break;
        case GV_FIELD_FLOAT:
            fi->data.float_data.entries = (GV_FloatEntry *)malloc(GV_PAYLOAD_INITIAL_CAP * sizeof(GV_FloatEntry));
            if (fi->data.float_data.entries == NULL) return -1;
            fi->data.float_data.count = 0;
            fi->data.float_data.capacity = GV_PAYLOAD_INITIAL_CAP;
            break;
        case GV_FIELD_STRING:
            fi->data.string_data.entries = (GV_StringEntry *)malloc(GV_PAYLOAD_INITIAL_CAP * sizeof(GV_StringEntry));
            if (fi->data.string_data.entries == NULL) return -1;
            fi->data.string_data.count = 0;
            fi->data.string_data.capacity = GV_PAYLOAD_INITIAL_CAP;
            break;
        case GV_FIELD_BOOL:
            fi->data.bool_data.entries = (GV_BoolEntry *)malloc(GV_PAYLOAD_INITIAL_CAP * sizeof(GV_BoolEntry));
            if (fi->data.bool_data.entries == NULL) return -1;
            fi->data.bool_data.count = 0;
            fi->data.bool_data.capacity = GV_PAYLOAD_INITIAL_CAP;
            break;
    }

    idx->field_count++;
    return 0;
}

int gv_payload_index_remove_field(GV_PayloadIndex *idx, const char *name) {
    if (idx == NULL || name == NULL) {
        return -1;
    }
    for (size_t i = 0; i < idx->field_count; i++) {
        if (strcmp(idx->fields[i].schema.name, name) == 0) {
            GV_FieldIndex *fi = &idx->fields[i];
            /* Free entry storage */
            switch (fi->schema.type) {
                case GV_FIELD_INT:
                    free(fi->data.int_data.entries);
                    break;
                case GV_FIELD_FLOAT:
                    free(fi->data.float_data.entries);
                    break;
                case GV_FIELD_STRING:
                    for (size_t j = 0; j < fi->data.string_data.count; j++) {
                        free(fi->data.string_data.entries[j].value);
                    }
                    free(fi->data.string_data.entries);
                    break;
                case GV_FIELD_BOOL:
                    free(fi->data.bool_data.entries);
                    break;
            }
            /* Shift remaining fields down */
            if (i < idx->field_count - 1) {
                memmove(&idx->fields[i], &idx->fields[i + 1],
                        (idx->field_count - i - 1) * sizeof(GV_FieldIndex));
            }
            idx->field_count--;
            return 0;
        }
    }
    return -1; /* Field not found */
}

int gv_payload_index_field_count(const GV_PayloadIndex *idx) {
    if (idx == NULL) {
        return 0;
    }
    return (int)idx->field_count;
}

/*  Insert operations                                                 */

int gv_payload_index_insert_int(GV_PayloadIndex *idx, size_t vector_id,
                                 const char *field, int64_t value) {
    if (idx == NULL || field == NULL) {
        return -1;
    }
    GV_FieldIndex *fi = gv_find_field(idx, field);
    if (fi == NULL || fi->schema.type != GV_FIELD_INT) {
        return -1;
    }
    /* Grow if needed */
    if (fi->data.int_data.count >= fi->data.int_data.capacity) {
        size_t new_cap = fi->data.int_data.capacity * 2;
        GV_IntEntry *tmp = (GV_IntEntry *)realloc(fi->data.int_data.entries,
                                                    new_cap * sizeof(GV_IntEntry));
        if (tmp == NULL) return -1;
        fi->data.int_data.entries = tmp;
        fi->data.int_data.capacity = new_cap;
    }
    /* Find sorted insertion position by value */
    size_t pos = gv_int_lower_bound(fi->data.int_data.entries, fi->data.int_data.count, value);
    /* Shift entries to make room */
    if (pos < fi->data.int_data.count) {
        memmove(&fi->data.int_data.entries[pos + 1],
                &fi->data.int_data.entries[pos],
                (fi->data.int_data.count - pos) * sizeof(GV_IntEntry));
    }
    fi->data.int_data.entries[pos].vector_id = vector_id;
    fi->data.int_data.entries[pos].value = value;
    fi->data.int_data.count++;
    return 0;
}

int gv_payload_index_insert_float(GV_PayloadIndex *idx, size_t vector_id,
                                   const char *field, double value) {
    if (idx == NULL || field == NULL) {
        return -1;
    }
    GV_FieldIndex *fi = gv_find_field(idx, field);
    if (fi == NULL || fi->schema.type != GV_FIELD_FLOAT) {
        return -1;
    }
    if (fi->data.float_data.count >= fi->data.float_data.capacity) {
        size_t new_cap = fi->data.float_data.capacity * 2;
        GV_FloatEntry *tmp = (GV_FloatEntry *)realloc(fi->data.float_data.entries,
                                                        new_cap * sizeof(GV_FloatEntry));
        if (tmp == NULL) return -1;
        fi->data.float_data.entries = tmp;
        fi->data.float_data.capacity = new_cap;
    }
    size_t pos = gv_float_lower_bound(fi->data.float_data.entries, fi->data.float_data.count, value);
    if (pos < fi->data.float_data.count) {
        memmove(&fi->data.float_data.entries[pos + 1],
                &fi->data.float_data.entries[pos],
                (fi->data.float_data.count - pos) * sizeof(GV_FloatEntry));
    }
    fi->data.float_data.entries[pos].vector_id = vector_id;
    fi->data.float_data.entries[pos].value = value;
    fi->data.float_data.count++;
    return 0;
}

int gv_payload_index_insert_string(GV_PayloadIndex *idx, size_t vector_id,
                                    const char *field, const char *value) {
    if (idx == NULL || field == NULL || value == NULL) {
        return -1;
    }
    GV_FieldIndex *fi = gv_find_field(idx, field);
    if (fi == NULL || fi->schema.type != GV_FIELD_STRING) {
        return -1;
    }
    if (fi->data.string_data.count >= fi->data.string_data.capacity) {
        size_t new_cap = fi->data.string_data.capacity * 2;
        GV_StringEntry *tmp = (GV_StringEntry *)realloc(fi->data.string_data.entries,
                                                          new_cap * sizeof(GV_StringEntry));
        if (tmp == NULL) return -1;
        fi->data.string_data.entries = tmp;
        fi->data.string_data.capacity = new_cap;
    }
    char *dup = strdup(value);
    if (dup == NULL) return -1;

    size_t pos = gv_string_lower_bound(fi->data.string_data.entries, fi->data.string_data.count, value);
    if (pos < fi->data.string_data.count) {
        memmove(&fi->data.string_data.entries[pos + 1],
                &fi->data.string_data.entries[pos],
                (fi->data.string_data.count - pos) * sizeof(GV_StringEntry));
    }
    fi->data.string_data.entries[pos].vector_id = vector_id;
    fi->data.string_data.entries[pos].value = dup;
    fi->data.string_data.count++;
    return 0;
}

int gv_payload_index_insert_bool(GV_PayloadIndex *idx, size_t vector_id,
                                  const char *field, int value) {
    if (idx == NULL || field == NULL) {
        return -1;
    }
    GV_FieldIndex *fi = gv_find_field(idx, field);
    if (fi == NULL || fi->schema.type != GV_FIELD_BOOL) {
        return -1;
    }
    if (fi->data.bool_data.count >= fi->data.bool_data.capacity) {
        size_t new_cap = fi->data.bool_data.capacity * 2;
        GV_BoolEntry *tmp = (GV_BoolEntry *)realloc(fi->data.bool_data.entries,
                                                      new_cap * sizeof(GV_BoolEntry));
        if (tmp == NULL) return -1;
        fi->data.bool_data.entries = tmp;
        fi->data.bool_data.capacity = new_cap;
    }
    /* Two-bucket approach: false entries first (value==0), then true entries.
       Insert at the boundary to keep the two groups contiguous. */
    size_t insert_pos;
    int normalized = value ? 1 : 0;
    if (normalized == 0) {
        /* Find position of first true entry and insert before it */
        insert_pos = 0;
        for (size_t i = 0; i < fi->data.bool_data.count; i++) {
            if (fi->data.bool_data.entries[i].value != 0) {
                insert_pos = i;
                goto bool_do_insert;
            }
        }
        insert_pos = fi->data.bool_data.count;
    } else {
        /* Append at the end (true bucket) */
        insert_pos = fi->data.bool_data.count;
    }
bool_do_insert:
    if (insert_pos < fi->data.bool_data.count) {
        memmove(&fi->data.bool_data.entries[insert_pos + 1],
                &fi->data.bool_data.entries[insert_pos],
                (fi->data.bool_data.count - insert_pos) * sizeof(GV_BoolEntry));
    }
    fi->data.bool_data.entries[insert_pos].vector_id = vector_id;
    fi->data.bool_data.entries[insert_pos].value = normalized;
    fi->data.bool_data.count++;
    return 0;
}

/*  Remove: delete all entries for a given vector_id across all fields */

int gv_payload_index_remove(GV_PayloadIndex *idx, size_t vector_id) {
    if (idx == NULL) {
        return -1;
    }
    for (size_t f = 0; f < idx->field_count; f++) {
        GV_FieldIndex *fi = &idx->fields[f];
        switch (fi->schema.type) {
            case GV_FIELD_INT: {
                size_t write = 0;
                for (size_t i = 0; i < fi->data.int_data.count; i++) {
                    if (fi->data.int_data.entries[i].vector_id != vector_id) {
                        if (write != i) {
                            fi->data.int_data.entries[write] = fi->data.int_data.entries[i];
                        }
                        write++;
                    }
                }
                fi->data.int_data.count = write;
                break;
            }
            case GV_FIELD_FLOAT: {
                size_t write = 0;
                for (size_t i = 0; i < fi->data.float_data.count; i++) {
                    if (fi->data.float_data.entries[i].vector_id != vector_id) {
                        if (write != i) {
                            fi->data.float_data.entries[write] = fi->data.float_data.entries[i];
                        }
                        write++;
                    }
                }
                fi->data.float_data.count = write;
                break;
            }
            case GV_FIELD_STRING: {
                size_t write = 0;
                for (size_t i = 0; i < fi->data.string_data.count; i++) {
                    if (fi->data.string_data.entries[i].vector_id != vector_id) {
                        if (write != i) {
                            fi->data.string_data.entries[write] = fi->data.string_data.entries[i];
                        }
                        write++;
                    } else {
                        free(fi->data.string_data.entries[i].value);
                    }
                }
                fi->data.string_data.count = write;
                break;
            }
            case GV_FIELD_BOOL: {
                size_t write = 0;
                for (size_t i = 0; i < fi->data.bool_data.count; i++) {
                    if (fi->data.bool_data.entries[i].vector_id != vector_id) {
                        if (write != i) {
                            fi->data.bool_data.entries[write] = fi->data.bool_data.entries[i];
                        }
                        write++;
                    }
                }
                fi->data.bool_data.count = write;
                break;
            }
        }
    }
    return 0;
}

/*  Single-condition query helpers                                    */

static int gv_query_int(const GV_FieldIndex *fi, const GV_PayloadQuery *q,
                         size_t *result_ids, size_t max_results) {
    const GV_IntEntry *entries = fi->data.int_data.entries;
    size_t count = fi->data.int_data.count;
    int64_t val = q->value.int_val;
    size_t n = 0;

    switch (q->op) {
        case GV_PAYLOAD_OP_EQ: {
            size_t lo = gv_int_lower_bound(entries, count, val);
            for (size_t i = lo; i < count && entries[i].value == val && n < max_results; i++) {
                result_ids[n++] = entries[i].vector_id;
            }
            break;
        }
        case GV_PAYLOAD_OP_NE: {
            for (size_t i = 0; i < count && n < max_results; i++) {
                if (entries[i].value != val) {
                    result_ids[n++] = entries[i].vector_id;
                }
            }
            break;
        }
        case GV_PAYLOAD_OP_GT: {
            size_t lo = gv_int_upper_bound(entries, count, val);
            for (size_t i = lo; i < count && n < max_results; i++) {
                result_ids[n++] = entries[i].vector_id;
            }
            break;
        }
        case GV_PAYLOAD_OP_GE: {
            size_t lo = gv_int_lower_bound(entries, count, val);
            for (size_t i = lo; i < count && n < max_results; i++) {
                result_ids[n++] = entries[i].vector_id;
            }
            break;
        }
        case GV_PAYLOAD_OP_LT: {
            size_t hi = gv_int_lower_bound(entries, count, val);
            for (size_t i = 0; i < hi && n < max_results; i++) {
                result_ids[n++] = entries[i].vector_id;
            }
            break;
        }
        case GV_PAYLOAD_OP_LE: {
            size_t hi = gv_int_upper_bound(entries, count, val);
            for (size_t i = 0; i < hi && n < max_results; i++) {
                result_ids[n++] = entries[i].vector_id;
            }
            break;
        }
        case GV_PAYLOAD_OP_CONTAINS:
        case GV_PAYLOAD_OP_PREFIX:
            /* Not applicable for int fields */
            return -1;
    }
    return (int)n;
}

static int gv_query_float(const GV_FieldIndex *fi, const GV_PayloadQuery *q,
                           size_t *result_ids, size_t max_results) {
    const GV_FloatEntry *entries = fi->data.float_data.entries;
    size_t count = fi->data.float_data.count;
    double val = q->value.float_val;
    size_t n = 0;

    switch (q->op) {
        case GV_PAYLOAD_OP_EQ: {
            /* Exact float comparison: scan the range around the target */
            size_t lo = gv_float_lower_bound(entries, count, val);
            for (size_t i = lo; i < count && entries[i].value == val && n < max_results; i++) {
                result_ids[n++] = entries[i].vector_id;
            }
            break;
        }
        case GV_PAYLOAD_OP_NE: {
            for (size_t i = 0; i < count && n < max_results; i++) {
                if (entries[i].value != val) {
                    result_ids[n++] = entries[i].vector_id;
                }
            }
            break;
        }
        case GV_PAYLOAD_OP_GT: {
            size_t lo = gv_float_upper_bound(entries, count, val);
            for (size_t i = lo; i < count && n < max_results; i++) {
                result_ids[n++] = entries[i].vector_id;
            }
            break;
        }
        case GV_PAYLOAD_OP_GE: {
            size_t lo = gv_float_lower_bound(entries, count, val);
            for (size_t i = lo; i < count && n < max_results; i++) {
                result_ids[n++] = entries[i].vector_id;
            }
            break;
        }
        case GV_PAYLOAD_OP_LT: {
            size_t hi = gv_float_lower_bound(entries, count, val);
            for (size_t i = 0; i < hi && n < max_results; i++) {
                result_ids[n++] = entries[i].vector_id;
            }
            break;
        }
        case GV_PAYLOAD_OP_LE: {
            size_t hi = gv_float_upper_bound(entries, count, val);
            for (size_t i = 0; i < hi && n < max_results; i++) {
                result_ids[n++] = entries[i].vector_id;
            }
            break;
        }
        case GV_PAYLOAD_OP_CONTAINS:
        case GV_PAYLOAD_OP_PREFIX:
            return -1;
    }
    return (int)n;
}

static int gv_query_string(const GV_FieldIndex *fi, const GV_PayloadQuery *q,
                            size_t *result_ids, size_t max_results) {
    const GV_StringEntry *entries = fi->data.string_data.entries;
    size_t count = fi->data.string_data.count;
    const char *val = q->value.string_val;
    size_t n = 0;

    if (val == NULL) return -1;

    switch (q->op) {
        case GV_PAYLOAD_OP_EQ: {
            size_t lo = gv_string_lower_bound(entries, count, val);
            for (size_t i = lo; i < count && strcmp(entries[i].value, val) == 0 && n < max_results; i++) {
                result_ids[n++] = entries[i].vector_id;
            }
            break;
        }
        case GV_PAYLOAD_OP_NE: {
            for (size_t i = 0; i < count && n < max_results; i++) {
                if (strcmp(entries[i].value, val) != 0) {
                    result_ids[n++] = entries[i].vector_id;
                }
            }
            break;
        }
        case GV_PAYLOAD_OP_GT: {
            /* Find first entry strictly greater than val */
            size_t lo = gv_string_lower_bound(entries, count, val);
            /* Skip entries equal to val */
            while (lo < count && strcmp(entries[lo].value, val) == 0) {
                lo++;
            }
            for (size_t i = lo; i < count && n < max_results; i++) {
                result_ids[n++] = entries[i].vector_id;
            }
            break;
        }
        case GV_PAYLOAD_OP_GE: {
            size_t lo = gv_string_lower_bound(entries, count, val);
            for (size_t i = lo; i < count && n < max_results; i++) {
                result_ids[n++] = entries[i].vector_id;
            }
            break;
        }
        case GV_PAYLOAD_OP_LT: {
            size_t hi = gv_string_lower_bound(entries, count, val);
            for (size_t i = 0; i < hi && n < max_results; i++) {
                result_ids[n++] = entries[i].vector_id;
            }
            break;
        }
        case GV_PAYLOAD_OP_LE: {
            size_t hi = gv_string_lower_bound(entries, count, val);
            /* Include entries equal to val */
            while (hi < count && strcmp(entries[hi].value, val) == 0) {
                hi++;
            }
            for (size_t i = 0; i < hi && n < max_results; i++) {
                result_ids[n++] = entries[i].vector_id;
            }
            break;
        }
        case GV_PAYLOAD_OP_CONTAINS: {
            /* Linear scan for substring match */
            for (size_t i = 0; i < count && n < max_results; i++) {
                if (strstr(entries[i].value, val) != NULL) {
                    result_ids[n++] = entries[i].vector_id;
                }
            }
            break;
        }
        case GV_PAYLOAD_OP_PREFIX: {
            /* Binary search to lower bound of prefix, then scan forward */
            size_t prefix_len = strlen(val);
            size_t lo = gv_string_lower_bound(entries, count, val);
            for (size_t i = lo; i < count && n < max_results; i++) {
                if (strncmp(entries[i].value, val, prefix_len) == 0) {
                    result_ids[n++] = entries[i].vector_id;
                } else {
                    /* Since sorted, once prefix no longer matches we are done */
                    break;
                }
            }
            break;
        }
    }
    return (int)n;
}

static int gv_query_bool(const GV_FieldIndex *fi, const GV_PayloadQuery *q,
                          size_t *result_ids, size_t max_results) {
    const GV_BoolEntry *entries = fi->data.bool_data.entries;
    size_t count = fi->data.bool_data.count;
    int val = q->value.bool_val ? 1 : 0;
    size_t n = 0;

    switch (q->op) {
        case GV_PAYLOAD_OP_EQ: {
            for (size_t i = 0; i < count && n < max_results; i++) {
                if (entries[i].value == val) {
                    result_ids[n++] = entries[i].vector_id;
                }
            }
            break;
        }
        case GV_PAYLOAD_OP_NE: {
            for (size_t i = 0; i < count && n < max_results; i++) {
                if (entries[i].value != val) {
                    result_ids[n++] = entries[i].vector_id;
                }
            }
            break;
        }
        default:
            /* GT, GE, LT, LE, CONTAINS, PREFIX not meaningful for bool */
            return -1;
    }
    return (int)n;
}

/*  Query: single condition                                           */

int gv_payload_index_query(const GV_PayloadIndex *idx, const GV_PayloadQuery *query,
                            size_t *result_ids, size_t max_results) {
    if (idx == NULL || query == NULL || result_ids == NULL || max_results == 0) {
        return -1;
    }
    if (query->field_name == NULL) {
        return -1;
    }
    const GV_FieldIndex *fi = gv_find_field(idx, query->field_name);
    if (fi == NULL) {
        return -1;
    }
    if (fi->schema.type != query->field_type) {
        return -1;
    }

    switch (fi->schema.type) {
        case GV_FIELD_INT:
            return gv_query_int(fi, query, result_ids, max_results);
        case GV_FIELD_FLOAT:
            return gv_query_float(fi, query, result_ids, max_results);
        case GV_FIELD_STRING:
            return gv_query_string(fi, query, result_ids, max_results);
        case GV_FIELD_BOOL:
            return gv_query_bool(fi, query, result_ids, max_results);
    }
    return -1;
}

/*  Query: multi-condition (AND of all conditions)                    */

int gv_payload_index_query_multi(const GV_PayloadIndex *idx, const GV_PayloadQuery *queries,
                                  size_t query_count, size_t *result_ids, size_t max_results) {
    if (idx == NULL || queries == NULL || result_ids == NULL || max_results == 0 || query_count == 0) {
        return -1;
    }

    /* Allocate temporary buffers for intermediate results */
    size_t buf_size = max_results;
    size_t *buf_a = (size_t *)malloc(buf_size * sizeof(size_t));
    size_t *buf_b = (size_t *)malloc(buf_size * sizeof(size_t));
    if (buf_a == NULL || buf_b == NULL) {
        free(buf_a);
        free(buf_b);
        return -1;
    }

    /* Execute the first query */
    int first_count = gv_payload_index_query(idx, &queries[0], buf_a, buf_size);
    if (first_count < 0) {
        free(buf_a);
        free(buf_b);
        return -1;
    }

    size_t current_count = (size_t)first_count;
    /* Sort the first result set for intersection */
    qsort(buf_a, current_count, sizeof(size_t), gv_compare_size_t);

    /* For each subsequent query, execute and intersect */
    for (size_t q = 1; q < query_count; q++) {
        int next_count = gv_payload_index_query(idx, &queries[q], buf_b, buf_size);
        if (next_count < 0) {
            free(buf_a);
            free(buf_b);
            return -1;
        }
        size_t nc = (size_t)next_count;
        qsort(buf_b, nc, sizeof(size_t), gv_compare_size_t);

        /* Intersect buf_a[0..current_count) with buf_b[0..nc) into result_ids */
        size_t intersected = gv_intersect_sorted(buf_a, current_count, buf_b, nc,
                                                  result_ids, max_results);

        /* Copy intersection back to buf_a for next round */
        memcpy(buf_a, result_ids, intersected * sizeof(size_t));
        current_count = intersected;

        if (current_count == 0) {
            break; /* No point continuing; intersection is empty */
        }
    }

    /* Final results are in buf_a */
    if (current_count > max_results) {
        current_count = max_results;
    }
    memcpy(result_ids, buf_a, current_count * sizeof(size_t));

    free(buf_a);
    free(buf_b);
    return (int)current_count;
}

/*  Stats                                                             */

size_t gv_payload_index_total_entries(const GV_PayloadIndex *idx) {
    if (idx == NULL) {
        return 0;
    }
    size_t total = 0;
    for (size_t i = 0; i < idx->field_count; i++) {
        const GV_FieldIndex *fi = &idx->fields[i];
        switch (fi->schema.type) {
            case GV_FIELD_INT:
                total += fi->data.int_data.count;
                break;
            case GV_FIELD_FLOAT:
                total += fi->data.float_data.count;
                break;
            case GV_FIELD_STRING:
                total += fi->data.string_data.count;
                break;
            case GV_FIELD_BOOL:
                total += fi->data.bool_data.count;
                break;
        }
    }
    return total;
}
