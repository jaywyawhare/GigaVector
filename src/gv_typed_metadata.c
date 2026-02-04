/**
 * @file gv_typed_metadata.c
 * @brief Typed metadata implementation.
 */

#include "gigavector/gv_typed_metadata.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ============================================================================
 * Value Creation Functions
 * ============================================================================ */

GV_TypedValue gv_typed_null(void) {
    GV_TypedValue val;
    memset(&val, 0, sizeof(val));
    val.type = GV_META_TYPE_NULL;
    return val;
}

GV_TypedValue gv_typed_string(const char *str) {
    GV_TypedValue val;
    memset(&val, 0, sizeof(val));
    val.type = GV_META_TYPE_STRING;
    if (str) {
        val.data.string_val = strdup(str);
    }
    return val;
}

GV_TypedValue gv_typed_int(int64_t i) {
    GV_TypedValue val;
    memset(&val, 0, sizeof(val));
    val.type = GV_META_TYPE_INT64;
    val.data.int_val = i;
    return val;
}

GV_TypedValue gv_typed_float(double f) {
    GV_TypedValue val;
    memset(&val, 0, sizeof(val));
    val.type = GV_META_TYPE_FLOAT64;
    val.data.float_val = f;
    return val;
}

GV_TypedValue gv_typed_bool(bool b) {
    GV_TypedValue val;
    memset(&val, 0, sizeof(val));
    val.type = GV_META_TYPE_BOOL;
    val.data.bool_val = b;
    return val;
}

GV_TypedValue gv_typed_array(GV_MetaType element_type) {
    GV_TypedValue val;
    memset(&val, 0, sizeof(val));
    val.type = GV_META_TYPE_ARRAY;
    val.data.array_val.element_type = element_type;
    val.data.array_val.items = NULL;
    val.data.array_val.count = 0;
    val.data.array_val.capacity = 0;
    return val;
}

GV_TypedValue gv_typed_object(void) {
    GV_TypedValue val;
    memset(&val, 0, sizeof(val));
    val.type = GV_META_TYPE_OBJECT;
    val.data.object_val.keys = NULL;
    val.data.object_val.values = NULL;
    val.data.object_val.count = 0;
    val.data.object_val.capacity = 0;
    return val;
}

/* ============================================================================
 * Array Operations
 * ============================================================================ */

int gv_typed_array_push(GV_TypedValue *array, const GV_TypedValue *item) {
    if (!array || array->type != GV_META_TYPE_ARRAY || !item) {
        return -1;
    }

    /* Grow capacity if needed */
    if (array->data.array_val.count >= array->data.array_val.capacity) {
        size_t new_cap = array->data.array_val.capacity == 0 ? 4 : array->data.array_val.capacity * 2;
        GV_TypedValue *new_items = realloc(array->data.array_val.items, new_cap * sizeof(GV_TypedValue));
        if (!new_items) {
            return -1;
        }
        array->data.array_val.items = new_items;
        array->data.array_val.capacity = new_cap;
    }

    /* Copy item */
    array->data.array_val.items[array->data.array_val.count] = gv_typed_value_copy(item);
    array->data.array_val.count++;

    return 0;
}

GV_TypedValue *gv_typed_array_get(const GV_TypedValue *array, size_t index) {
    if (!array || array->type != GV_META_TYPE_ARRAY) {
        return NULL;
    }
    if (index >= array->data.array_val.count) {
        return NULL;
    }
    return &array->data.array_val.items[index];
}

size_t gv_typed_array_length(const GV_TypedValue *array) {
    if (!array || array->type != GV_META_TYPE_ARRAY) {
        return 0;
    }
    return array->data.array_val.count;
}

/* ============================================================================
 * Object Operations
 * ============================================================================ */

int gv_typed_object_set(GV_TypedValue *object, const char *key, const GV_TypedValue *value) {
    if (!object || object->type != GV_META_TYPE_OBJECT || !key || !value) {
        return -1;
    }

    /* Check if key already exists */
    for (size_t i = 0; i < object->data.object_val.count; i++) {
        if (strcmp(object->data.object_val.keys[i], key) == 0) {
            /* Update existing value */
            gv_typed_value_free(&object->data.object_val.values[i]);
            object->data.object_val.values[i] = gv_typed_value_copy(value);
            return 0;
        }
    }

    /* Grow capacity if needed */
    if (object->data.object_val.count >= object->data.object_val.capacity) {
        size_t new_cap = object->data.object_val.capacity == 0 ? 4 : object->data.object_val.capacity * 2;
        char **new_keys = realloc(object->data.object_val.keys, new_cap * sizeof(char *));
        GV_TypedValue *new_values = realloc(object->data.object_val.values, new_cap * sizeof(GV_TypedValue));
        if (!new_keys || !new_values) {
            free(new_keys);
            free(new_values);
            return -1;
        }
        object->data.object_val.keys = new_keys;
        object->data.object_val.values = new_values;
        object->data.object_val.capacity = new_cap;
    }

    /* Add new key-value pair */
    object->data.object_val.keys[object->data.object_val.count] = strdup(key);
    object->data.object_val.values[object->data.object_val.count] = gv_typed_value_copy(value);
    object->data.object_val.count++;

    return 0;
}

GV_TypedValue *gv_typed_object_get(const GV_TypedValue *object, const char *key) {
    if (!object || object->type != GV_META_TYPE_OBJECT || !key) {
        return NULL;
    }
    for (size_t i = 0; i < object->data.object_val.count; i++) {
        if (strcmp(object->data.object_val.keys[i], key) == 0) {
            return &object->data.object_val.values[i];
        }
    }
    return NULL;
}

bool gv_typed_object_has(const GV_TypedValue *object, const char *key) {
    return gv_typed_object_get(object, key) != NULL;
}

size_t gv_typed_object_length(const GV_TypedValue *object) {
    if (!object || object->type != GV_META_TYPE_OBJECT) {
        return 0;
    }
    return object->data.object_val.count;
}

/* ============================================================================
 * Value Extraction
 * ============================================================================ */

const char *gv_typed_get_string(const GV_TypedValue *value) {
    if (!value || value->type != GV_META_TYPE_STRING) {
        return NULL;
    }
    return value->data.string_val;
}

int gv_typed_get_int(const GV_TypedValue *value, int64_t *out) {
    if (!value || !out) return -1;
    if (value->type == GV_META_TYPE_INT64) {
        *out = value->data.int_val;
        return 0;
    }
    if (value->type == GV_META_TYPE_FLOAT64) {
        *out = (int64_t)value->data.float_val;
        return 0;
    }
    return -1;
}

int gv_typed_get_float(const GV_TypedValue *value, double *out) {
    if (!value || !out) return -1;
    if (value->type == GV_META_TYPE_FLOAT64) {
        *out = value->data.float_val;
        return 0;
    }
    if (value->type == GV_META_TYPE_INT64) {
        *out = (double)value->data.int_val;
        return 0;
    }
    return -1;
}

int gv_typed_get_bool(const GV_TypedValue *value, bool *out) {
    if (!value || !out) return -1;
    if (value->type != GV_META_TYPE_BOOL) return -1;
    *out = value->data.bool_val;
    return 0;
}

/* ============================================================================
 * Comparison Operations
 * ============================================================================ */

static double get_numeric_value(const GV_TypedValue *value) {
    if (value->type == GV_META_TYPE_INT64) {
        return (double)value->data.int_val;
    }
    if (value->type == GV_META_TYPE_FLOAT64) {
        return value->data.float_val;
    }
    return 0.0;
}

static bool is_numeric(const GV_TypedValue *value) {
    return value && (value->type == GV_META_TYPE_INT64 || value->type == GV_META_TYPE_FLOAT64);
}

int gv_typed_compare(const GV_TypedValue *a, const GV_TypedValue *b) {
    if (!a || !b) return 0;

    /* Numeric comparison */
    if (is_numeric(a) && is_numeric(b)) {
        double va = get_numeric_value(a);
        double vb = get_numeric_value(b);
        if (va < vb) return -1;
        if (va > vb) return 1;
        return 0;
    }

    /* Same type comparison */
    if (a->type != b->type) return 0;

    switch (a->type) {
        case GV_META_TYPE_NULL:
            return 0;
        case GV_META_TYPE_STRING:
            if (!a->data.string_val && !b->data.string_val) return 0;
            if (!a->data.string_val) return -1;
            if (!b->data.string_val) return 1;
            return strcmp(a->data.string_val, b->data.string_val);
        case GV_META_TYPE_BOOL:
            return (int)a->data.bool_val - (int)b->data.bool_val;
        default:
            return 0;
    }
}

bool gv_typed_equals(const GV_TypedValue *a, const GV_TypedValue *b) {
    if (!a || !b) return false;

    /* Numeric comparison */
    if (is_numeric(a) && is_numeric(b)) {
        double va = get_numeric_value(a);
        double vb = get_numeric_value(b);
        return fabs(va - vb) < 1e-10;
    }

    if (a->type != b->type) return false;

    switch (a->type) {
        case GV_META_TYPE_NULL:
            return true;
        case GV_META_TYPE_STRING:
            if (!a->data.string_val && !b->data.string_val) return true;
            if (!a->data.string_val || !b->data.string_val) return false;
            return strcmp(a->data.string_val, b->data.string_val) == 0;
        case GV_META_TYPE_BOOL:
            return a->data.bool_val == b->data.bool_val;
        case GV_META_TYPE_ARRAY: {
            if (a->data.array_val.count != b->data.array_val.count) return false;
            for (size_t i = 0; i < a->data.array_val.count; i++) {
                if (!gv_typed_equals(&a->data.array_val.items[i], &b->data.array_val.items[i])) {
                    return false;
                }
            }
            return true;
        }
        case GV_META_TYPE_OBJECT: {
            if (a->data.object_val.count != b->data.object_val.count) return false;
            for (size_t i = 0; i < a->data.object_val.count; i++) {
                GV_TypedValue *bval = gv_typed_object_get(b, a->data.object_val.keys[i]);
                if (!bval || !gv_typed_equals(&a->data.object_val.values[i], bval)) {
                    return false;
                }
            }
            return true;
        }
        default:
            return false;
    }
}

bool gv_typed_in_range(const GV_TypedValue *value, double min, double max) {
    if (!is_numeric(value)) return false;
    double v = get_numeric_value(value);
    return v >= min && v <= max;
}

bool gv_typed_string_contains(const GV_TypedValue *value, const char *substr) {
    if (!value || value->type != GV_META_TYPE_STRING || !value->data.string_val || !substr) {
        return false;
    }
    return strstr(value->data.string_val, substr) != NULL;
}

bool gv_typed_string_starts_with(const GV_TypedValue *value, const char *prefix) {
    if (!value || value->type != GV_META_TYPE_STRING || !value->data.string_val || !prefix) {
        return false;
    }
    size_t prefix_len = strlen(prefix);
    return strncmp(value->data.string_val, prefix, prefix_len) == 0;
}

bool gv_typed_array_contains(const GV_TypedValue *array, const GV_TypedValue *item) {
    if (!array || array->type != GV_META_TYPE_ARRAY || !item) {
        return false;
    }
    for (size_t i = 0; i < array->data.array_val.count; i++) {
        if (gv_typed_equals(&array->data.array_val.items[i], item)) {
            return true;
        }
    }
    return false;
}

/* ============================================================================
 * Memory Management
 * ============================================================================ */

void gv_typed_value_free(GV_TypedValue *value) {
    if (!value) return;

    switch (value->type) {
        case GV_META_TYPE_STRING:
            free(value->data.string_val);
            value->data.string_val = NULL;
            break;
        case GV_META_TYPE_ARRAY:
            for (size_t i = 0; i < value->data.array_val.count; i++) {
                gv_typed_value_free(&value->data.array_val.items[i]);
            }
            free(value->data.array_val.items);
            value->data.array_val.items = NULL;
            value->data.array_val.count = 0;
            value->data.array_val.capacity = 0;
            break;
        case GV_META_TYPE_OBJECT:
            for (size_t i = 0; i < value->data.object_val.count; i++) {
                free(value->data.object_val.keys[i]);
                gv_typed_value_free(&value->data.object_val.values[i]);
            }
            free(value->data.object_val.keys);
            free(value->data.object_val.values);
            value->data.object_val.keys = NULL;
            value->data.object_val.values = NULL;
            value->data.object_val.count = 0;
            value->data.object_val.capacity = 0;
            break;
        default:
            break;
    }
    value->type = GV_META_TYPE_NULL;
}

GV_TypedValue gv_typed_value_copy(const GV_TypedValue *src) {
    GV_TypedValue dst;
    memset(&dst, 0, sizeof(dst));

    if (!src) {
        dst.type = GV_META_TYPE_NULL;
        return dst;
    }

    dst.type = src->type;

    switch (src->type) {
        case GV_META_TYPE_NULL:
            break;
        case GV_META_TYPE_STRING:
            dst.data.string_val = src->data.string_val ? strdup(src->data.string_val) : NULL;
            break;
        case GV_META_TYPE_INT64:
            dst.data.int_val = src->data.int_val;
            break;
        case GV_META_TYPE_FLOAT64:
            dst.data.float_val = src->data.float_val;
            break;
        case GV_META_TYPE_BOOL:
            dst.data.bool_val = src->data.bool_val;
            break;
        case GV_META_TYPE_ARRAY:
            dst.data.array_val.element_type = src->data.array_val.element_type;
            dst.data.array_val.count = src->data.array_val.count;
            dst.data.array_val.capacity = src->data.array_val.count;
            if (src->data.array_val.count > 0) {
                dst.data.array_val.items = malloc(dst.data.array_val.capacity * sizeof(GV_TypedValue));
                for (size_t i = 0; i < src->data.array_val.count; i++) {
                    dst.data.array_val.items[i] = gv_typed_value_copy(&src->data.array_val.items[i]);
                }
            }
            break;
        case GV_META_TYPE_OBJECT:
            dst.data.object_val.count = src->data.object_val.count;
            dst.data.object_val.capacity = src->data.object_val.count;
            if (src->data.object_val.count > 0) {
                dst.data.object_val.keys = malloc(dst.data.object_val.capacity * sizeof(char *));
                dst.data.object_val.values = malloc(dst.data.object_val.capacity * sizeof(GV_TypedValue));
                for (size_t i = 0; i < src->data.object_val.count; i++) {
                    dst.data.object_val.keys[i] = strdup(src->data.object_val.keys[i]);
                    dst.data.object_val.values[i] = gv_typed_value_copy(&src->data.object_val.values[i]);
                }
            }
            break;
    }

    return dst;
}

void gv_typed_metadata_free(GV_TypedMetadata *meta) {
    if (!meta) return;
    free(meta->key);
    gv_typed_value_free(&meta->value);
    free(meta);
}

void gv_typed_metadata_free_all(GV_TypedMetadata *head) {
    while (head) {
        GV_TypedMetadata *next = head->next;
        gv_typed_metadata_free(head);
        head = next;
    }
}

/* ============================================================================
 * Serialization (Binary Format)
 *
 * Format:
 *   [type: 1 byte]
 *   [data: variable]
 *
 * String: [length: 4 bytes][chars: length bytes]
 * Int64: [value: 8 bytes]
 * Float64: [value: 8 bytes]
 * Bool: [value: 1 byte]
 * Array: [count: 4 bytes][element_type: 1 byte][items...]
 * Object: [count: 4 bytes][{key_length: 4, key: key_length, value}...]
 * ============================================================================ */

static int serialize_value(const GV_TypedValue *value, uint8_t **buf, size_t *len, size_t *cap);

static int ensure_capacity(uint8_t **buf, size_t *cap, size_t needed, size_t current) {
    if (current + needed <= *cap) return 0;
    size_t new_cap = *cap == 0 ? 256 : *cap * 2;
    while (new_cap < current + needed) new_cap *= 2;
    uint8_t *new_buf = realloc(*buf, new_cap);
    if (!new_buf) return -1;
    *buf = new_buf;
    *cap = new_cap;
    return 0;
}

static int serialize_value(const GV_TypedValue *value, uint8_t **buf, size_t *len, size_t *cap) {
    if (!value) return -1;

    /* Write type */
    if (ensure_capacity(buf, cap, 1, *len) < 0) return -1;
    (*buf)[(*len)++] = (uint8_t)value->type;

    switch (value->type) {
        case GV_META_TYPE_NULL:
            break;

        case GV_META_TYPE_STRING: {
            uint32_t str_len = value->data.string_val ? (uint32_t)strlen(value->data.string_val) : 0;
            if (ensure_capacity(buf, cap, 4 + str_len, *len) < 0) return -1;
            memcpy(*buf + *len, &str_len, 4); *len += 4;
            if (str_len > 0) {
                memcpy(*buf + *len, value->data.string_val, str_len);
                *len += str_len;
            }
            break;
        }

        case GV_META_TYPE_INT64:
            if (ensure_capacity(buf, cap, 8, *len) < 0) return -1;
            memcpy(*buf + *len, &value->data.int_val, 8); *len += 8;
            break;

        case GV_META_TYPE_FLOAT64:
            if (ensure_capacity(buf, cap, 8, *len) < 0) return -1;
            memcpy(*buf + *len, &value->data.float_val, 8); *len += 8;
            break;

        case GV_META_TYPE_BOOL:
            if (ensure_capacity(buf, cap, 1, *len) < 0) return -1;
            (*buf)[(*len)++] = value->data.bool_val ? 1 : 0;
            break;

        case GV_META_TYPE_ARRAY: {
            uint32_t count = (uint32_t)value->data.array_val.count;
            if (ensure_capacity(buf, cap, 5, *len) < 0) return -1;
            memcpy(*buf + *len, &count, 4); *len += 4;
            (*buf)[(*len)++] = (uint8_t)value->data.array_val.element_type;
            for (size_t i = 0; i < count; i++) {
                if (serialize_value(&value->data.array_val.items[i], buf, len, cap) < 0) {
                    return -1;
                }
            }
            break;
        }

        case GV_META_TYPE_OBJECT: {
            uint32_t count = (uint32_t)value->data.object_val.count;
            if (ensure_capacity(buf, cap, 4, *len) < 0) return -1;
            memcpy(*buf + *len, &count, 4); *len += 4;
            for (size_t i = 0; i < count; i++) {
                uint32_t key_len = (uint32_t)strlen(value->data.object_val.keys[i]);
                if (ensure_capacity(buf, cap, 4 + key_len, *len) < 0) return -1;
                memcpy(*buf + *len, &key_len, 4); *len += 4;
                memcpy(*buf + *len, value->data.object_val.keys[i], key_len);
                *len += key_len;
                if (serialize_value(&value->data.object_val.values[i], buf, len, cap) < 0) {
                    return -1;
                }
            }
            break;
        }
    }

    return 0;
}

int gv_typed_value_serialize(const GV_TypedValue *value, uint8_t **buf, size_t *len) {
    if (!value || !buf || !len) return -1;

    *buf = NULL;
    *len = 0;
    size_t cap = 0;

    if (serialize_value(value, buf, len, &cap) < 0) {
        free(*buf);
        *buf = NULL;
        *len = 0;
        return -1;
    }

    return 0;
}

static int deserialize_value(const uint8_t *buf, size_t len, size_t *pos, GV_TypedValue *out);

static int deserialize_value(const uint8_t *buf, size_t len, size_t *pos, GV_TypedValue *out) {
    if (*pos >= len) return -1;

    memset(out, 0, sizeof(*out));
    out->type = (GV_MetaType)buf[(*pos)++];

    switch (out->type) {
        case GV_META_TYPE_NULL:
            break;

        case GV_META_TYPE_STRING: {
            if (*pos + 4 > len) return -1;
            uint32_t str_len;
            memcpy(&str_len, buf + *pos, 4); *pos += 4;
            if (*pos + str_len > len) return -1;
            out->data.string_val = malloc(str_len + 1);
            if (!out->data.string_val) return -1;
            memcpy(out->data.string_val, buf + *pos, str_len);
            out->data.string_val[str_len] = '\0';
            *pos += str_len;
            break;
        }

        case GV_META_TYPE_INT64:
            if (*pos + 8 > len) return -1;
            memcpy(&out->data.int_val, buf + *pos, 8); *pos += 8;
            break;

        case GV_META_TYPE_FLOAT64:
            if (*pos + 8 > len) return -1;
            memcpy(&out->data.float_val, buf + *pos, 8); *pos += 8;
            break;

        case GV_META_TYPE_BOOL:
            if (*pos >= len) return -1;
            out->data.bool_val = buf[(*pos)++] != 0;
            break;

        case GV_META_TYPE_ARRAY: {
            if (*pos + 5 > len) return -1;
            uint32_t count;
            memcpy(&count, buf + *pos, 4); *pos += 4;
            out->data.array_val.element_type = (GV_MetaType)buf[(*pos)++];
            out->data.array_val.count = count;
            out->data.array_val.capacity = count;
            if (count > 0) {
                out->data.array_val.items = malloc(count * sizeof(GV_TypedValue));
                if (!out->data.array_val.items) return -1;
                for (uint32_t i = 0; i < count; i++) {
                    if (deserialize_value(buf, len, pos, &out->data.array_val.items[i]) < 0) {
                        return -1;
                    }
                }
            }
            break;
        }

        case GV_META_TYPE_OBJECT: {
            if (*pos + 4 > len) return -1;
            uint32_t count;
            memcpy(&count, buf + *pos, 4); *pos += 4;
            out->data.object_val.count = count;
            out->data.object_val.capacity = count;
            if (count > 0) {
                out->data.object_val.keys = malloc(count * sizeof(char *));
                out->data.object_val.values = malloc(count * sizeof(GV_TypedValue));
                if (!out->data.object_val.keys || !out->data.object_val.values) return -1;
                for (uint32_t i = 0; i < count; i++) {
                    if (*pos + 4 > len) return -1;
                    uint32_t key_len;
                    memcpy(&key_len, buf + *pos, 4); *pos += 4;
                    if (*pos + key_len > len) return -1;
                    out->data.object_val.keys[i] = malloc(key_len + 1);
                    if (!out->data.object_val.keys[i]) return -1;
                    memcpy(out->data.object_val.keys[i], buf + *pos, key_len);
                    out->data.object_val.keys[i][key_len] = '\0';
                    *pos += key_len;
                    if (deserialize_value(buf, len, pos, &out->data.object_val.values[i]) < 0) {
                        return -1;
                    }
                }
            }
            break;
        }

        default:
            return -1;
    }

    return 0;
}

int gv_typed_value_deserialize(const uint8_t *buf, size_t len, GV_TypedValue *out) {
    if (!buf || !out) return -1;
    size_t pos = 0;
    if (deserialize_value(buf, len, &pos, out) < 0) {
        return -1;
    }
    return (int)pos;
}

int gv_typed_metadata_serialize(const GV_TypedMetadata *meta, uint8_t **buf, size_t *len) {
    if (!meta || !buf || !len) return -1;

    *buf = NULL;
    *len = 0;
    size_t cap = 0;

    /* Serialize key */
    uint32_t key_len = (uint32_t)strlen(meta->key);
    if (ensure_capacity(buf, &cap, 4 + key_len, *len) < 0) return -1;
    memcpy(*buf + *len, &key_len, 4); *len += 4;
    memcpy(*buf + *len, meta->key, key_len); *len += key_len;

    /* Serialize value */
    if (serialize_value(&meta->value, buf, len, &cap) < 0) {
        free(*buf);
        *buf = NULL;
        *len = 0;
        return -1;
    }

    return 0;
}

GV_TypedMetadata *gv_typed_metadata_deserialize(const uint8_t *buf, size_t len) {
    if (!buf || len < 4) return NULL;

    size_t pos = 0;

    /* Deserialize key */
    uint32_t key_len;
    memcpy(&key_len, buf + pos, 4); pos += 4;
    if (pos + key_len > len) return NULL;

    GV_TypedMetadata *meta = calloc(1, sizeof(GV_TypedMetadata));
    if (!meta) return NULL;

    meta->key = malloc(key_len + 1);
    if (!meta->key) {
        free(meta);
        return NULL;
    }
    memcpy(meta->key, buf + pos, key_len);
    meta->key[key_len] = '\0';
    pos += key_len;

    /* Deserialize value */
    if (deserialize_value(buf, len, &pos, &meta->value) < 0) {
        free(meta->key);
        free(meta);
        return NULL;
    }

    meta->next = NULL;
    return meta;
}

/* ============================================================================
 * Conversion Functions
 * ============================================================================ */

char *gv_typed_to_string(const GV_TypedValue *value) {
    if (!value) return strdup("null");

    char buf[256];

    switch (value->type) {
        case GV_META_TYPE_NULL:
            return strdup("null");
        case GV_META_TYPE_STRING:
            return value->data.string_val ? strdup(value->data.string_val) : strdup("");
        case GV_META_TYPE_INT64:
            snprintf(buf, sizeof(buf), "%ld", (long)value->data.int_val);
            return strdup(buf);
        case GV_META_TYPE_FLOAT64:
            snprintf(buf, sizeof(buf), "%g", value->data.float_val);
            return strdup(buf);
        case GV_META_TYPE_BOOL:
            return strdup(value->data.bool_val ? "true" : "false");
        case GV_META_TYPE_ARRAY:
            snprintf(buf, sizeof(buf), "[array:%zu]", value->data.array_val.count);
            return strdup(buf);
        case GV_META_TYPE_OBJECT:
            snprintf(buf, sizeof(buf), "{object:%zu}", value->data.object_val.count);
            return strdup(buf);
        default:
            return strdup("unknown");
    }
}

GV_TypedMetadata *gv_typed_from_string_metadata(const char *key, const char *value) {
    if (!key) return NULL;

    GV_TypedMetadata *meta = calloc(1, sizeof(GV_TypedMetadata));
    if (!meta) return NULL;

    meta->key = strdup(key);
    meta->value = gv_typed_string(value);
    meta->next = NULL;

    return meta;
}

const char *gv_typed_type_name(GV_MetaType type) {
    switch (type) {
        case GV_META_TYPE_NULL:    return "null";
        case GV_META_TYPE_STRING:  return "string";
        case GV_META_TYPE_INT64:   return "int64";
        case GV_META_TYPE_FLOAT64: return "float64";
        case GV_META_TYPE_BOOL:    return "bool";
        case GV_META_TYPE_ARRAY:   return "array";
        case GV_META_TYPE_OBJECT:  return "object";
        default:                   return "unknown";
    }
}
