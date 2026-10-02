#include "core/prop_value.h"
#include "core/memory.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

GV_PropValue gv_prop_string(const char *s)
{
    GV_PropValue v;
    v.type = GV_PROP_STRING;
    v.as.s = s ? gv_strdup(s) : NULL;
    return v;
}

GV_PropValue gv_prop_int64(int64_t v)
{
    GV_PropValue pv;
    pv.type = GV_PROP_INT64;
    pv.as.i = v;
    return pv;
}

GV_PropValue gv_prop_float64(double v)
{
    GV_PropValue pv;
    pv.type = GV_PROP_FLOAT64;
    pv.as.f = v;
    return pv;
}

GV_PropValue gv_prop_bool(int v)
{
    GV_PropValue pv;
    pv.type = GV_PROP_BOOL;
    pv.as.b = v ? 1 : 0;
    return pv;
}

GV_PropValue gv_prop_blob(const uint8_t *data, size_t len)
{
    GV_PropValue pv;
    pv.type = GV_PROP_BLOB;
    if (data && len > 0) {
        pv.as.blob.data = (uint8_t *)gv_alloc(len);
        if (!pv.as.blob.data) {
            pv.type = GV_PROP_NULL;
            pv.as.blob.len = 0;
            return pv;
        }
        memcpy(pv.as.blob.data, data, len);
        pv.as.blob.len = len;
    } else {
        pv.as.blob.data = NULL;
        pv.as.blob.len = 0;
    }
    return pv;
}

GV_PropValue gv_prop_null(void)
{
    GV_PropValue v;
    memset(&v, 0, sizeof(v));
    v.type = GV_PROP_NULL;
    return v;
}

GV_PropValue gv_prop_clone(GV_PropValue v)
{
    switch (v.type) {
    case GV_PROP_STRING:
        return gv_prop_string(v.as.s);
    case GV_PROP_INT64:
        return gv_prop_int64(v.as.i);
    case GV_PROP_FLOAT64:
        return gv_prop_float64(v.as.f);
    case GV_PROP_BOOL:
        return gv_prop_bool(v.as.b);
    case GV_PROP_BLOB:
        return gv_prop_blob(v.as.blob.data, v.as.blob.len);
    case GV_PROP_NULL:
    default:
        return gv_prop_null();
    }
}

void gv_prop_free(GV_PropValue *v)
{
    if (!v) return;
    if (v->type == GV_PROP_STRING) {
        gv_free(v->as.s);
        v->as.s = NULL;
    } else if (v->type == GV_PROP_BLOB) {
        gv_free(v->as.blob.data);
        v->as.blob.data = NULL;
        v->as.blob.len = 0;
    }
    v->type = GV_PROP_NULL;
}

int gv_prop_compare(GV_PropValue a, GV_PropValue b)
{
    if (a.type != b.type) {
        return (int)a.type - (int)b.type;
    }
    switch (a.type) {
    case GV_PROP_NULL:
        return 0;
    case GV_PROP_STRING: {
        const char *sa = a.as.s ? a.as.s : "";
        const char *sb = b.as.s ? b.as.s : "";
        return strcmp(sa, sb);
    }
    case GV_PROP_INT64:
        if (a.as.i < b.as.i) return -1;
        if (a.as.i > b.as.i) return 1;
        return 0;
    case GV_PROP_FLOAT64: {
        double diff = a.as.f - b.as.f;
        if (diff < 0.0) return -1;
        if (diff > 0.0) return 1;
        return 0;
    }
    case GV_PROP_BOOL:
        return a.as.b - b.as.b;
    case GV_PROP_BLOB: {
        size_t min_len = a.as.blob.len < b.as.blob.len ? a.as.blob.len : b.as.blob.len;
        int cmp = memcmp(a.as.blob.data, b.as.blob.data, min_len);
        if (cmp != 0) return cmp;
        if (a.as.blob.len < b.as.blob.len) return -1;
        if (a.as.blob.len > b.as.blob.len) return 1;
        return 0;
    }
    default:
        return 0;
    }
}

char *gv_prop_to_string(GV_PropValue v)
{
    char buf[128];
    switch (v.type) {
    case GV_PROP_NULL:
        return gv_strdup("(null)");
    case GV_PROP_STRING:
        return gv_strdup(v.as.s ? v.as.s : "");
    case GV_PROP_INT64:
        snprintf(buf, sizeof(buf), "%" PRId64, v.as.i);
        return gv_strdup(buf);
    case GV_PROP_FLOAT64:
        snprintf(buf, sizeof(buf), "%g", v.as.f);
        return gv_strdup(buf);
    case GV_PROP_BOOL:
        return gv_strdup(v.as.b ? "true" : "false");
    case GV_PROP_BLOB: {
        /* Return hex-encoded representation */
        size_t hex_len = v.as.blob.len * 2 + 1;
        char *hex = (char *)gv_alloc(hex_len);
        if (!hex) return NULL;
        for (size_t i = 0; i < v.as.blob.len; i++) {
            snprintf(hex + i * 2, 3, "%02x", v.as.blob.data[i]);
        }
        return hex;
    }
    default:
        return gv_strdup("(unknown)");
    }
}

GV_PropValue gv_prop_parse(const char *raw, GV_PropType hint)
{
    if (!raw) return gv_prop_null();

    switch (hint) {
    case GV_PROP_STRING:
        return gv_prop_string(raw);
    case GV_PROP_INT64: {
        char *end = NULL;
        int64_t val = (int64_t)strtoll(raw, &end, 10);
        if (end == raw) return gv_prop_string(raw); /* parse failed, store as string */
        return gv_prop_int64(val);
    }
    case GV_PROP_FLOAT64: {
        char *end = NULL;
        double val = strtod(raw, &end);
        if (end == raw) return gv_prop_string(raw);
        return gv_prop_float64(val);
    }
    case GV_PROP_BOOL: {
        if (strcmp(raw, "true") == 0 || strcmp(raw, "1") == 0 ||
            strcmp(raw, "yes") == 0 || strcmp(raw, "on") == 0) {
            return gv_prop_bool(1);
        }
        if (strcmp(raw, "false") == 0 || strcmp(raw, "0") == 0 ||
            strcmp(raw, "no") == 0 || strcmp(raw, "off") == 0) {
            return gv_prop_bool(0);
        }
        return gv_prop_string(raw);
    }
    case GV_PROP_BLOB: {
        /* Hex-encoded input */
        size_t len = strlen(raw);
        if (len % 2 != 0) return gv_prop_null();
        size_t byte_count = len / 2;
        uint8_t *data = (uint8_t *)gv_alloc(byte_count);
        if (!data) return gv_prop_null();
        for (size_t i = 0; i < byte_count; i++) {
            unsigned int byte_val = 0;
            /* NOLINTNEXTLINE(cert-err34-c): return value is validated (!= 1). */
            if (sscanf(raw + i * 2, "%2x", &byte_val) != 1) {
                gv_free(data);
                return gv_prop_null();
            }
            data[i] = (uint8_t)byte_val;
        }
        GV_PropValue pv = gv_prop_blob(data, byte_count);
        gv_free(data);
        return pv;
    }
    case GV_PROP_NULL:
    default:
        return gv_prop_string(raw);
    }
}
