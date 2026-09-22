#include <stdio.h>
#include "core/memory.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>

#include "api/schema.h"
#include "core/utils.h"

#define GV_SCHEMA_MAGIC "GVSC"
#define GV_SCHEMA_MAGIC_LEN 4
#define GV_SCHEMA_INITIAL_CAPACITY 8

static const char *schema_type_to_string(GV_SchemaFieldType type) {
    switch (type) {
        case GV_SCHEMA_STRING: return "string";
        case GV_SCHEMA_INT:    return "int";
        case GV_SCHEMA_FLOAT:  return "float";
        case GV_SCHEMA_BOOL:   return "bool";
        default:               return "unknown";
    }
}

__attribute__((unused))
static GV_SchemaFieldType schema_type_from_string(const char *s) {
    if (strcmp(s, "int") == 0)    return GV_SCHEMA_INT;
    if (strcmp(s, "float") == 0)  return GV_SCHEMA_FLOAT;
    if (strcmp(s, "bool") == 0)   return GV_SCHEMA_BOOL;
    return GV_SCHEMA_STRING;
}

static int schema_find_field_index(const GV_Schema *schema, const char *name) {
    if (!schema || !name) return -1;
    for (size_t i = 0; i < schema->field_count; i++) {
        if (strcmp(schema->fields[i].name, name) == 0) {
            return (int)i;
        }
    }
    return -1;
}

/* Append src to a dynamically growing buffer; returns -1 on allocation failure. */
static int buf_append(char **buf, size_t *len, size_t *cap, const char *src) {
    size_t src_len = strlen(src);
    while (*len + src_len + 1 > *cap) {
        size_t new_cap = (*cap == 0) ? 256 : (*cap) * 2;
        char *tmp = (char *)gv_realloc(*buf, new_cap);
        if (!tmp) return -1;
        *buf = tmp;
        *cap = new_cap;
    }
    memcpy(*buf + *len, src, src_len);
    *len += src_len;
    (*buf)[*len] = '\0';
    return 0;
}

/* Escape a string for JSON output.  Handles \, ", and control characters. */
static int buf_append_json_string(char **buf, size_t *len, size_t *cap, const char *s) {
    if (buf_append(buf, len, cap, "\"") != 0) return -1;
    for (const char *p = s; *p; p++) {
        char esc[8];
        switch (*p) {
            case '"':  if (buf_append(buf, len, cap, "\\\"") != 0) return -1; break;
            case '\\': if (buf_append(buf, len, cap, "\\\\") != 0) return -1; break;
            case '\b': if (buf_append(buf, len, cap, "\\b") != 0)  return -1; break;
            case '\f': if (buf_append(buf, len, cap, "\\f") != 0)  return -1; break;
            case '\n': if (buf_append(buf, len, cap, "\\n") != 0)  return -1; break;
            case '\r': if (buf_append(buf, len, cap, "\\r") != 0)  return -1; break;
            case '\t': if (buf_append(buf, len, cap, "\\t") != 0)  return -1; break;
            default:
                if ((unsigned char)*p < 0x20) {
                    snprintf(esc, sizeof(esc), "\\u%04x", (unsigned char)*p);
                    if (buf_append(buf, len, cap, esc) != 0) return -1;
                } else {
                    char c[2] = { *p, '\0' };
                    if (buf_append(buf, len, cap, c) != 0) return -1;
                }
                break;
        }
    }
    if (buf_append(buf, len, cap, "\"") != 0) return -1;
    return 0;
}

GV_Schema *schema_create(uint32_t version) {
    GV_Schema *schema = (GV_Schema *)gv_calloc(1, sizeof(GV_Schema));
    if (!schema) return NULL;

    schema->version = version;
    schema->field_count = 0;
    schema->field_capacity = GV_SCHEMA_INITIAL_CAPACITY;
    schema->fields = (GV_SchemaField *)gv_calloc(schema->field_capacity, sizeof(GV_SchemaField));
    if (!schema->fields) {
        gv_free(schema);
        return NULL;
    }
    return schema;
}

void schema_destroy(GV_Schema *schema) {
    if (!schema) return;
    gv_free(schema->fields);
    gv_free(schema);
}

GV_Schema *schema_copy(const GV_Schema *schema) {
    if (!schema) return NULL;

    GV_Schema *copy = (GV_Schema *)gv_calloc(1, sizeof(GV_Schema));
    if (!copy) return NULL;

    copy->version = schema->version;
    copy->field_count = schema->field_count;
    copy->field_capacity = schema->field_capacity;
    copy->fields = (GV_SchemaField *)gv_calloc(copy->field_capacity, sizeof(GV_SchemaField));
    if (!copy->fields) {
        gv_free(copy);
        return NULL;
    }
    memcpy(copy->fields, schema->fields, schema->field_count * sizeof(GV_SchemaField));
    return copy;
}

int schema_add_field(GV_Schema *schema, const char *name, GV_SchemaFieldType type,
                         int required, const char *default_value) {
    if (!schema || !name) return -1;

    if (schema_find_field_index(schema, name) >= 0) return -1;

    if (schema->field_count >= schema->field_capacity) {
        size_t new_cap = schema->field_capacity * 2;
        GV_SchemaField *tmp = (GV_SchemaField *)gv_realloc(
            schema->fields, new_cap * sizeof(GV_SchemaField));
        if (!tmp) return -1;
        schema->fields = tmp;
        schema->field_capacity = new_cap;
    }

    GV_SchemaField *f = &schema->fields[schema->field_count];
    memset(f, 0, sizeof(*f));
    strncpy(f->name, name, sizeof(f->name) - 1);
    f->name[sizeof(f->name) - 1] = '\0';
    f->type = type;
    f->required = required;
    if (default_value) {
        strncpy(f->default_value, default_value, sizeof(f->default_value) - 1);
        f->default_value[sizeof(f->default_value) - 1] = '\0';
    }

    schema->field_count++;
    return 0;
}

int schema_remove_field(GV_Schema *schema, const char *name) {
    if (!schema || !name) return -1;

    int idx = schema_find_field_index(schema, name);
    if (idx < 0) return -1;

    size_t remaining = schema->field_count - (size_t)idx - 1;
    if (remaining > 0) {
        memmove(&schema->fields[idx], &schema->fields[idx + 1],
                remaining * sizeof(GV_SchemaField));
    }
    schema->field_count--;
    return 0;
}

int schema_has_field(const GV_Schema *schema, const char *name) {
    return schema_find_field_index(schema, name) >= 0 ? 1 : 0;
}

const GV_SchemaField *schema_get_field(const GV_Schema *schema, const char *name) {
    int idx = schema_find_field_index(schema, name);
    if (idx < 0) return NULL;
    return &schema->fields[idx];
}

size_t schema_field_count(const GV_Schema *schema) {
    if (!schema) return 0;
    return schema->field_count;
}

static int schema_validate_int(const char *value) {
    if (!value || *value == '\0') return 0;
    char *end = NULL;
    errno = 0;
    (void)strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0') return 0;
    return 1;
}

static int schema_validate_float(const char *value) {
    if (!value || *value == '\0') return 0;
    char *end = NULL;
    errno = 0;
    (void)strtod(value, &end);
    if (errno != 0 || end == value || *end != '\0') return 0;
    return 1;
}

static int schema_validate_bool(const char *value) {
    if (!value) return 0;
    return (strcmp(value, "true") == 0 ||
            strcmp(value, "false") == 0 ||
            strcmp(value, "0") == 0 ||
            strcmp(value, "1") == 0);
}

static int schema_validate_value(GV_SchemaFieldType type, const char *value) {
    switch (type) {
        case GV_SCHEMA_STRING:
            return 1;
        case GV_SCHEMA_INT:
            return schema_validate_int(value);
        case GV_SCHEMA_FLOAT:
            return schema_validate_float(value);
        case GV_SCHEMA_BOOL:
            return schema_validate_bool(value);
        default:
            return 0;
    }
}

int schema_validate(const GV_Schema *schema, const char *const *keys,
                        const char *const *values, size_t count) {
    if (!schema) return -1;

    /* Every required field must be present in keys[]. */
    for (size_t i = 0; i < schema->field_count; i++) {
        if (!schema->fields[i].required) continue;

        int found = 0;
        for (size_t k = 0; k < count; k++) {
            if (keys[k] && strcmp(keys[k], schema->fields[i].name) == 0) {
                found = 1;
                break;
            }
        }
        if (!found) {
            return -1;
        }
    }

    /* For each provided key that is in the schema, validate its type. */
    for (size_t k = 0; k < count; k++) {
        if (!keys[k]) continue;

        const GV_SchemaField *field = schema_get_field(schema, keys[k]);
        if (!field) {
            continue;
        }

        const char *val = values ? values[k] : NULL;
        if (val && !schema_validate_value(field->type, val)) {
            return -1;
        }
    }

    return 0;
}

int schema_diff(const GV_Schema *old_schema, const GV_Schema *new_schema,
                    GV_SchemaDiff *diffs, size_t max_diffs) {
    if (!old_schema || !new_schema || !diffs || max_diffs == 0) return -1;

    size_t diff_count = 0;

    /* new-only fields => added; fields in both with differing types => type_changed. */
    for (size_t i = 0; i < new_schema->field_count && diff_count < max_diffs; i++) {
        const char *name = new_schema->fields[i].name;
        int old_idx = schema_find_field_index(old_schema, name);

        if (old_idx < 0) {
            GV_SchemaDiff *d = &diffs[diff_count++];
            memset(d, 0, sizeof(*d));
            snprintf(d->name, sizeof(d->name), "%s", name);
            d->added = 1;
            d->new_type = new_schema->fields[i].type;
        } else {
            GV_SchemaFieldType ot = old_schema->fields[old_idx].type;
            GV_SchemaFieldType nt = new_schema->fields[i].type;
            if (ot != nt) {
                GV_SchemaDiff *d = &diffs[diff_count++];
                memset(d, 0, sizeof(*d));
                snprintf(d->name, sizeof(d->name), "%s", name);
                d->type_changed = 1;
                d->old_type = ot;
                d->new_type = nt;
            }
        }
    }

    /* old-only fields => removed. */
    for (size_t i = 0; i < old_schema->field_count && diff_count < max_diffs; i++) {
        const char *name = old_schema->fields[i].name;
        if (schema_find_field_index(new_schema, name) < 0) {
            GV_SchemaDiff *d = &diffs[diff_count++];
            memset(d, 0, sizeof(*d));
            snprintf(d->name, sizeof(d->name), "%s", name);
            d->removed = 1;
            d->old_type = old_schema->fields[i].type;
        }
    }

    return (int)diff_count;
}

int schema_is_compatible(const GV_Schema *old_schema, const GV_Schema *new_schema) {
    if (!old_schema || !new_schema) return -1;

    /* Rule 1: No required field in old_schema may be removed in new_schema. */
    for (size_t i = 0; i < old_schema->field_count; i++) {
        if (!old_schema->fields[i].required) continue;
        if (schema_find_field_index(new_schema, old_schema->fields[i].name) < 0) {
            return -1;
        }
    }

    /* Rule 2: No type change on existing fields. */
    for (size_t i = 0; i < old_schema->field_count; i++) {
        int new_idx = schema_find_field_index(new_schema, old_schema->fields[i].name);
        if (new_idx < 0) continue;  /* Removal handled by Rule 1 for required fields. */
        if (old_schema->fields[i].type != new_schema->fields[new_idx].type) {
            return -1;
        }
    }

    return 0;
}

int schema_save(const GV_Schema *schema, FILE *out) {
    if (!schema || !out) return -1;

    if (fwrite(GV_SCHEMA_MAGIC, 1, GV_SCHEMA_MAGIC_LEN, out) != GV_SCHEMA_MAGIC_LEN)
        return -1;

    if (write_u32(out, schema->version) != 0) return -1;
    if (write_u32(out, (uint32_t)schema->field_count) != 0) return -1;

    for (size_t i = 0; i < schema->field_count; i++) {
        const GV_SchemaField *f = &schema->fields[i];

        if (write_bytes(out, f->name, sizeof(f->name)) != 0) return -1;
        if (write_u32(out, (uint32_t)f->type) != 0) return -1;
        if (write_u32(out, (uint32_t)f->required) != 0) return -1;
        if (write_bytes(out, f->default_value, sizeof(f->default_value)) != 0)
            return -1;
    }

    return 0;
}

GV_Schema *schema_load(FILE *in) {
    if (!in) return NULL;

    char magic[GV_SCHEMA_MAGIC_LEN];
    if (fread(magic, 1, GV_SCHEMA_MAGIC_LEN, in) != GV_SCHEMA_MAGIC_LEN) return NULL;
    if (memcmp(magic, GV_SCHEMA_MAGIC, GV_SCHEMA_MAGIC_LEN) != 0) return NULL;

    uint32_t version;
    if (read_u32(in, &version) != 0) return NULL;

    uint32_t fc;
    if (read_u32(in, &fc) != 0) return NULL;

    GV_Schema *schema = schema_create(version);
    if (!schema) return NULL;

    for (uint32_t i = 0; i < fc; i++) {
        char name[64];
        uint32_t type;
        uint32_t req;
        char default_value[256];

        if (read_bytes(in, name, sizeof(name)) != 0) goto fail;
        name[sizeof(name) - 1] = '\0';

        if (read_u32(in, &type) != 0) goto fail;
        if (read_u32(in, &req) != 0) goto fail;

        if (read_bytes(in, default_value, sizeof(default_value)) != 0) goto fail;
        default_value[sizeof(default_value) - 1] = '\0';

        if (type > GV_SCHEMA_BOOL) goto fail;

        if (schema_add_field(schema, name, (GV_SchemaFieldType)type,
                                 (int)req, default_value) != 0) {
            goto fail;
        }
    }

    return schema;

fail:
    schema_destroy(schema);
    return NULL;
}

char *schema_to_json(const GV_Schema *schema) {
    if (!schema) return NULL;

    char *buf = NULL;
    size_t len = 0;
    size_t cap = 0;
    char tmp[64];

    if (buf_append(&buf, &len, &cap, "{\"version\":") != 0) goto fail;
    snprintf(tmp, sizeof(tmp), "%u", (unsigned)schema->version);
    if (buf_append(&buf, &len, &cap, tmp) != 0) goto fail;

    if (buf_append(&buf, &len, &cap, ",\"fields\":[") != 0) goto fail;

    for (size_t i = 0; i < schema->field_count; i++) {
        const GV_SchemaField *f = &schema->fields[i];

        if (i > 0) {
            if (buf_append(&buf, &len, &cap, ",") != 0) goto fail;
        }

        if (buf_append(&buf, &len, &cap, "{\"name\":") != 0) goto fail;
        if (buf_append_json_string(&buf, &len, &cap, f->name) != 0) goto fail;

        if (buf_append(&buf, &len, &cap, ",\"type\":") != 0) goto fail;
        if (buf_append_json_string(&buf, &len, &cap, schema_type_to_string(f->type)) != 0)
            goto fail;

        if (buf_append(&buf, &len, &cap, ",\"required\":") != 0) goto fail;
        if (buf_append(&buf, &len, &cap, f->required ? "true" : "false") != 0) goto fail;

        if (buf_append(&buf, &len, &cap, ",\"default\":") != 0) goto fail;
        if (buf_append_json_string(&buf, &len, &cap, f->default_value) != 0) goto fail;

        if (buf_append(&buf, &len, &cap, "}") != 0) goto fail;
    }

    if (buf_append(&buf, &len, &cap, "]}") != 0) goto fail;

    return buf;

fail:
    gv_free(buf);
    return NULL;
}
