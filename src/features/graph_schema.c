#include "features/graph_schema.h"
#include "core/memory.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define INITIAL_CONSTRAINT_CAP 16
#define INITIAL_UNIQUE_CAP     64
#define UNIQUE_LOAD_NUMER      4
#define UNIQUE_LOAD_DENOM      3

#define FNV_OFFSET_BASIS UINT64_C(14695981039346656037)
#define FNV_PRIME        UINT64_C(1099511628211)

typedef struct UniqueEntry {
    char *label;
    char *prop_key;
    GV_PropValue value;
    uint64_t entity_id;
    uint64_t hash;
    int is_node;
    int occupied;
} UniqueEntry;

struct GV_GraphSchema {
    GV_SchemaConstraint *constraints;
    size_t constraint_count;
    size_t constraint_cap;
    UniqueEntry *unique_set;
    size_t unique_cap;
    size_t unique_count;
};

static uint64_t fnv1a_hash(const void *data, size_t len) {
    const uint8_t *bytes = (const uint8_t *)data;
    uint64_t h = FNV_OFFSET_BASIS;
    for (size_t i = 0; i < len; i++) {
        h ^= bytes[i];
        h *= FNV_PRIME;
    }
    return h;
}

static uint64_t hash_prop_value(GV_PropValue v) {
    switch (v.type) {
    case GV_PROP_STRING:
        return fnv1a_hash(v.as.s, strlen(v.as.s));
    case GV_PROP_INT64:
        return fnv1a_hash(&v.as.i, sizeof(v.as.i));
    case GV_PROP_FLOAT64:
        return fnv1a_hash(&v.as.f, sizeof(v.as.f));
    case GV_PROP_BOOL:
        return fnv1a_hash(&v.as.b, sizeof(v.as.b));
    case GV_PROP_BLOB:
        return fnv1a_hash(v.as.blob.data, v.as.blob.len);
    default: {
        uint64_t seed = 0;
        return fnv1a_hash(&seed, sizeof(seed));
    }
    }
}

static uint64_t combine_hash(uint64_t a, uint64_t b) {
    a ^= b + UINT64_C(0x9e3779b97f4a7c15) + (a << 6) + (a >> 2);
    return a;
}

static uint64_t compute_unique_hash(const char *label, const char *prop_key,
                                    int is_node, GV_PropValue value) {
    uint64_t h = fnv1a_hash(label, strlen(label));
    h = combine_hash(h, fnv1a_hash(prop_key, strlen(prop_key)));
    h = combine_hash(h, fnv1a_hash(&is_node, sizeof(is_node)));
    h = combine_hash(h, hash_prop_value(value));
    return h;
}

static int values_equal(GV_PropValue a, GV_PropValue b) {
    if (a.type != b.type) return 0;
    switch (a.type) {
    case GV_PROP_STRING:
        return strcmp(a.as.s, b.as.s) == 0;
    case GV_PROP_INT64:
        return a.as.i == b.as.i;
    case GV_PROP_FLOAT64:
        return a.as.f == b.as.f;
    case GV_PROP_BOOL:
        return a.as.b == b.as.b;
    case GV_PROP_BLOB:
        return a.as.blob.len == b.as.blob.len &&
               memcmp(a.as.blob.data, b.as.blob.data, a.as.blob.len) == 0;
    default:
        return 1;
    }
}

GV_GraphSchema *graph_schema_create(void) {
    GV_GraphSchema *s = gv_calloc(1, sizeof(GV_GraphSchema));
    if (!s) return NULL;
    s->constraints = gv_calloc(INITIAL_CONSTRAINT_CAP, sizeof(GV_SchemaConstraint));
    if (!s->constraints) { gv_free(s); return NULL; }
    s->constraint_cap = INITIAL_CONSTRAINT_CAP;
    s->constraint_count = 0;
    s->unique_set = gv_calloc(INITIAL_UNIQUE_CAP, sizeof(UniqueEntry));
    if (!s->unique_set) { gv_free(s->constraints); gv_free(s); return NULL; }
    s->unique_cap = INITIAL_UNIQUE_CAP;
    s->unique_count = 0;
    return s;
}

void graph_schema_destroy(GV_GraphSchema *schema) {
    if (!schema) return;
    for (size_t i = 0; i < schema->constraint_count; i++) {
        gv_free(schema->constraints[i].label);
        gv_free(schema->constraints[i].property_key);
    }
    gv_free(schema->constraints);
    for (size_t i = 0; i < schema->unique_cap; i++) {
        if (schema->unique_set[i].occupied) {
            gv_free(schema->unique_set[i].label);
            gv_free(schema->unique_set[i].prop_key);
            gv_prop_free(&schema->unique_set[i].value);
        }
    }
    gv_free(schema->unique_set);
    gv_free(schema);
}

static int constraint_matches(const GV_SchemaConstraint *c, const char *label,
                              const char *prop_key, int is_node) {
    return c->is_node == is_node && strcmp(c->label, label) == 0 &&
           strcmp(c->property_key, prop_key) == 0;
}

static int add_constraint(GV_GraphSchema *schema, GV_SchemaConstraintType type,
                          const char *label, const char *prop_key, int is_node) {
    if (schema->constraint_count == schema->constraint_cap) {
        size_t new_cap = schema->constraint_cap * 2;
        GV_SchemaConstraint *p = gv_realloc(schema->constraints,
                                            new_cap * sizeof(GV_SchemaConstraint));
        if (!p) return -1;
        memset(p + schema->constraint_cap, 0,
               (new_cap - schema->constraint_cap) * sizeof(GV_SchemaConstraint));
        schema->constraints = p;
        schema->constraint_cap = new_cap;
    }
    GV_SchemaConstraint *c = &schema->constraints[schema->constraint_count];
    c->type = type;
    c->label = gv_strdup(label);
    c->property_key = gv_strdup(prop_key);
    c->is_node = is_node;
    if (!c->label || !c->property_key) {
        gv_free(c->label);
        gv_free(c->property_key);
        return -1;
    }
    schema->constraint_count++;
    return 0;
}

int graph_schema_add_required(GV_GraphSchema *schema, const char *label,
                              const char *prop_key, int is_node) {
    if (!schema || !label || !prop_key) return -1;
    return add_constraint(schema, GV_SCHEMA_CONSTRAINT_REQUIRED, label, prop_key, is_node);
}

int graph_schema_add_unique(GV_GraphSchema *schema, const char *label,
                            const char *prop_key, int is_node) {
    if (!schema || !label || !prop_key) return -1;
    return add_constraint(schema, GV_SCHEMA_CONSTRAINT_UNIQUE, label, prop_key, is_node);
}

int graph_schema_add_type(GV_GraphSchema *schema, const char *label,
                          const char *prop_key, int is_node, GV_PropType type) {
    if (!schema || !label || !prop_key) return -1;
    int rc = add_constraint(schema, GV_SCHEMA_CONSTRAINT_TYPE, label, prop_key, is_node);
    if (rc == 0) {
        schema->constraints[schema->constraint_count - 1].expected_type = type;
    }
    return rc;
}

int graph_schema_add_range_int(GV_GraphSchema *schema, const char *label,
                               const char *prop_key, int is_node,
                               int64_t min_val, int64_t max_val) {
    if (!schema || !label || !prop_key) return -1;
    int rc = add_constraint(schema, GV_SCHEMA_CONSTRAINT_RANGE_INT, label, prop_key, is_node);
    if (rc == 0) {
        GV_SchemaConstraint *c = &schema->constraints[schema->constraint_count - 1];
        c->int_min = min_val;
        c->int_max = max_val;
    }
    return rc;
}

int graph_schema_add_range_float(GV_GraphSchema *schema, const char *label,
                                  const char *prop_key, int is_node,
                                  double min_val, double max_val) {
    if (!schema || !label || !prop_key) return -1;
    int rc = add_constraint(schema, GV_SCHEMA_CONSTRAINT_RANGE_FLOAT, label, prop_key, is_node);
    if (rc == 0) {
        GV_SchemaConstraint *c = &schema->constraints[schema->constraint_count - 1];
        c->float_min = min_val;
        c->float_max = max_val;
    }
    return rc;
}

int graph_schema_remove_constraint(GV_GraphSchema *schema, const char *label,
                                   const char *prop_key, int is_node) {
    if (!schema || !label || !prop_key) return -1;
    size_t wrote = 0;
    for (size_t i = 0; i < schema->constraint_count; i++) {
        if (constraint_matches(&schema->constraints[i], label, prop_key, is_node)) {
            gv_free(schema->constraints[i].label);
            gv_free(schema->constraints[i].property_key);
        } else {
            if (wrote != i) {
                schema->constraints[wrote] = schema->constraints[i];
            }
            wrote++;
        }
    }
    size_t removed = schema->constraint_count - wrote;
    schema->constraint_count = wrote;
    return removed > 0 ? 0 : -1;
}

static const GV_PropValue *find_prop(const char **keys, const GV_PropValue *vals,
                                     size_t count, const char *key) {
    for (size_t i = 0; i < count; i++) {
        if (strcmp(keys[i], key) == 0) return &vals[i];
    }
    return NULL;
}

static int validate_constraints(const GV_GraphSchema *schema, const char *label,
                                int is_node, const char **prop_keys,
                                const GV_PropValue *prop_values, size_t prop_count,
                                char *error_buf, size_t error_buf_len) {
    int ok = 1;
    for (size_t i = 0; i < schema->constraint_count; i++) {
        const GV_SchemaConstraint *c = &schema->constraints[i];
        if (c->is_node != is_node || strcmp(c->label, label) != 0) continue;

        const GV_PropValue *val = find_prop(prop_keys, prop_values, prop_count,
                                            c->property_key);
        switch (c->type) {
        case GV_SCHEMA_CONSTRAINT_REQUIRED:
            if (!val) {
                if (error_buf && error_buf_len > 0) {
                    snprintf(error_buf, error_buf_len,
                             "property '%s' is required on %s label '%s'",
                             c->property_key, is_node ? "node" : "edge", label);
                }
                ok = 0;
            }
            break;
        case GV_SCHEMA_CONSTRAINT_TYPE:
            if (val && val->type != c->expected_type) {
                if (error_buf && error_buf_len > 0) {
                    snprintf(error_buf, error_buf_len,
                             "property '%s' on %s label '%s' has wrong type",
                             c->property_key, is_node ? "node" : "edge", label);
                }
                ok = 0;
            }
            break;
        case GV_SCHEMA_CONSTRAINT_RANGE_INT:
            if (val && val->type == GV_PROP_INT64) {
                if (val->as.i < c->int_min || val->as.i > c->int_max) {
                    if (error_buf && error_buf_len > 0) {
                        snprintf(error_buf, error_buf_len,
                                 "property '%s' on %s label '%s': value %ld out of range [%ld, %ld]",
                                 c->property_key, is_node ? "node" : "edge", label,
                                 (long)val->as.i, (long)c->int_min, (long)c->int_max);
                    }
                    ok = 0;
                }
            }
            break;
        case GV_SCHEMA_CONSTRAINT_RANGE_FLOAT:
            if (val && val->type == GV_PROP_FLOAT64) {
                if (val->as.f < c->float_min || val->as.f > c->float_max) {
                    if (error_buf && error_buf_len > 0) {
                        snprintf(error_buf, error_buf_len,
                                 "property '%s' on %s label '%s': value %g out of range [%g, %g]",
                                 c->property_key, is_node ? "node" : "edge", label,
                                 val->as.f, c->float_min, c->float_max);
                    }
                    ok = 0;
                }
            }
            break;
        case GV_SCHEMA_CONSTRAINT_UNIQUE:
            break;
        }
    }
    return ok ? 0 : -1;
}

int graph_schema_validate_node(const GV_GraphSchema *schema,
                               const char *label,
                               const char **prop_keys,
                               const GV_PropValue *prop_values,
                               size_t prop_count,
                               char *error_buf, size_t error_buf_len) {
    if (!schema || !label) return -1;
    return validate_constraints(schema, label, 1, prop_keys, prop_values,
                                prop_count, error_buf, error_buf_len);
}

int graph_schema_validate_edge(const GV_GraphSchema *schema,
                               const char *label,
                               const char **prop_keys,
                               const GV_PropValue *prop_values,
                               size_t prop_count,
                               char *error_buf, size_t error_buf_len) {
    if (!schema || !label) return -1;
    return validate_constraints(schema, label, 0, prop_keys, prop_values,
                                prop_count, error_buf, error_buf_len);
}

static int unique_resize(GV_GraphSchema *schema) {
    size_t new_cap = schema->unique_cap * 2;
    UniqueEntry *new_set = gv_calloc(new_cap, sizeof(UniqueEntry));
    if (!new_set) return -1;
    for (size_t i = 0; i < schema->unique_cap; i++) {
        if (!schema->unique_set[i].occupied) continue;
        uint64_t idx = schema->unique_set[i].hash & (new_cap - 1);
        while (new_set[idx].occupied) {
            idx = (idx + 1) & (new_cap - 1);
        }
        new_set[idx] = schema->unique_set[i];
    }
    gv_free(schema->unique_set);
    schema->unique_set = new_set;
    schema->unique_cap = new_cap;
    return 0;
}

static int unique_find_slot(const GV_GraphSchema *schema, uint64_t hash,
                            const char *label, const char *prop_key,
                            int is_node, GV_PropValue value) {
    uint64_t idx = hash & (schema->unique_cap - 1);
    for (size_t probe = 0; probe < schema->unique_cap; probe++) {
        const UniqueEntry *e = &schema->unique_set[idx];
        if (!e->occupied) return (int)idx;
        if (e->hash == hash && e->is_node == is_node &&
            strcmp(e->label, label) == 0 &&
            strcmp(e->prop_key, prop_key) == 0 &&
            values_equal(e->value, value)) {
            return (int)idx;
        }
        idx = (idx + 1) & (schema->unique_cap - 1);
    }
    return -1;
}

static int unique_find_for_value(const GV_GraphSchema *schema, uint64_t hash,
                                 const char *label, const char *prop_key,
                                 int is_node, GV_PropValue value,
                                 size_t *out_idx) {
    uint64_t idx = hash & (schema->unique_cap - 1);
    for (size_t probe = 0; probe < schema->unique_cap; probe++) {
        const UniqueEntry *e = &schema->unique_set[idx];
        if (!e->occupied) return -1;
        if (e->hash == hash && e->is_node == is_node &&
            strcmp(e->label, label) == 0 &&
            strcmp(e->prop_key, prop_key) == 0 &&
            values_equal(e->value, value)) {
            *out_idx = idx;
            return 0;
        }
        idx = (idx + 1) & (schema->unique_cap - 1);
    }
    return -1;
}

int graph_schema_register_unique(GV_GraphSchema *schema, const char *label,
                                 const char *prop_key, int is_node,
                                 GV_PropValue value, uint64_t entity_id) {
    if (!schema || !label || !prop_key) return -1;

    if (schema->unique_count * UNIQUE_LOAD_NUMER >= schema->unique_cap * UNIQUE_LOAD_DENOM) {
        if (unique_resize(schema) != 0) return -1;
    }

    uint64_t h = compute_unique_hash(label, prop_key, is_node, value);
    int slot = unique_find_slot(schema, h, label, prop_key, is_node, value);
    if (slot < 0) return -1;

    UniqueEntry *e = &schema->unique_set[slot];
    if (e->occupied) {
        if (e->entity_id != entity_id) return -1;
        return 0;
    }

    e->label = gv_strdup(label);
    e->prop_key = gv_strdup(prop_key);
    if (!e->label || !e->prop_key) {
        gv_free(e->label);
        gv_free(e->prop_key);
        e->label = NULL;
        e->prop_key = NULL;
        return -1;
    }
    e->value = gv_prop_clone(value);
    e->entity_id = entity_id;
    e->hash = h;
    e->occupied = 1;
    e->is_node = is_node;
    schema->unique_count++;
    return 0;
}

int graph_schema_unregister_unique(GV_GraphSchema *schema, const char *label,
                                   const char *prop_key, int is_node,
                                   GV_PropValue value) {
    if (!schema || !label || !prop_key) return -1;
    uint64_t h = compute_unique_hash(label, prop_key, is_node, value);
    size_t idx;
    if (unique_find_for_value(schema, h, label, prop_key, is_node, value, &idx) != 0) {
        return -1;
    }
    UniqueEntry *e = &schema->unique_set[idx];
    gv_free(e->label);
    gv_free(e->prop_key);
    gv_prop_free(&e->value);
    memset(e, 0, sizeof(*e));
    schema->unique_count--;
    return 0;
}

int graph_schema_check_unique(const GV_GraphSchema *schema, const char *label,
                              const char *prop_key, int is_node,
                              GV_PropValue value) {
    if (!schema || !label || !prop_key) return 1;
    uint64_t h = compute_unique_hash(label, prop_key, is_node, value);
    size_t idx;
    return unique_find_for_value(schema, h, label, prop_key, is_node, value, &idx) == 0 ? 0 : 1;
}

int graph_schema_get_constraints(const GV_GraphSchema *schema,
                                 const char *label, int is_node,
                                 GV_SchemaConstraint *out, size_t max_count) {
    if (!schema || !out || max_count == 0) return -1;
    size_t count = 0;
    for (size_t i = 0; i < schema->constraint_count; i++) {
        const GV_SchemaConstraint *c = &schema->constraints[i];
        if (c->is_node == is_node && strcmp(c->label, label) == 0) {
            if (count < max_count) {
                out[count] = *c;
                out[count].label = gv_strdup(c->label);
                out[count].property_key = gv_strdup(c->property_key);
            }
            count++;
        }
    }
    return (int)count;
}

static int write_string(FILE *f, const char *s) {
    size_t len = s ? strlen(s) : 0;
    if (fwrite(&len, sizeof(len), 1, f) != 1) return -1;
    if (len > 0 && fwrite(s, 1, len, f) != len) return -1;
    return 0;
}

static char *read_string(FILE *f) {
    size_t len;
    if (fread(&len, sizeof(len), 1, f) != 1) return NULL;
    if (len == 0) return gv_strdup("");
    char *buf = gv_alloc(len + 1);
    if (!buf) return NULL;
    if (fread(buf, 1, len, f) != len) { gv_free(buf); return NULL; }
    buf[len] = '\0';
    return buf;
}

int graph_schema_save(const GV_GraphSchema *schema, FILE *out) {
    if (!schema || !out) return -1;

    uint32_t magic = 0x47534348; /* GSCH */
    uint32_t version = 1;
    if (fwrite(&magic, sizeof(magic), 1, out) != 1) return -1;
    if (fwrite(&version, sizeof(version), 1, out) != 1) return -1;
    if (fwrite(&schema->constraint_count, sizeof(schema->constraint_count), 1, out) != 1)
        return -1;

    for (size_t i = 0; i < schema->constraint_count; i++) {
        const GV_SchemaConstraint *c = &schema->constraints[i];
        uint32_t type_u = (uint32_t)c->type;
        if (fwrite(&type_u, sizeof(type_u), 1, out) != 1) return -1;
        if (write_string(out, c->label) != 0) return -1;
        if (write_string(out, c->property_key) != 0) return -1;
        int32_t is_node_i = c->is_node;
        if (fwrite(&is_node_i, sizeof(is_node_i), 1, out) != 1) return -1;
        uint32_t etype = (uint32_t)c->expected_type;
        if (fwrite(&etype, sizeof(etype), 1, out) != 1) return -1;
        if (fwrite(&c->int_min, sizeof(c->int_min), 1, out) != 1) return -1;
        if (fwrite(&c->int_max, sizeof(c->int_max), 1, out) != 1) return -1;
        if (fwrite(&c->float_min, sizeof(c->float_min), 1, out) != 1) return -1;
        if (fwrite(&c->float_max, sizeof(c->float_max), 1, out) != 1) return -1;
    }
    return 0;
}

int graph_schema_load(GV_GraphSchema **schema, FILE *in) {
    if (!schema || !in) return -1;

    uint32_t magic, version;
    if (fread(&magic, sizeof(magic), 1, in) != 1 || magic != 0x47534348) return -1;
    if (fread(&version, sizeof(version), 1, in) != 1 || version != 1) return -1;

    GV_GraphSchema *s = graph_schema_create();
    if (!s) return -1;

    size_t count;
    if (fread(&count, sizeof(count), 1, in) != 1) { graph_schema_destroy(s); return -1; }

    for (size_t i = 0; i < count; i++) {
        uint32_t type_u;
        if (fread(&type_u, sizeof(type_u), 1, in) != 1) { graph_schema_destroy(s); return -1; }
        char *label = read_string(in);
        char *pkey = read_string(in);
        if (!label || !pkey) { gv_free(label); gv_free(pkey); graph_schema_destroy(s); return -1; }

        int32_t is_node_i;
        if (fread(&is_node_i, sizeof(is_node_i), 1, in) != 1) { gv_free(label); gv_free(pkey); graph_schema_destroy(s); return -1; }

        uint32_t etype;
        if (fread(&etype, sizeof(etype), 1, in) != 1) { gv_free(label); gv_free(pkey); graph_schema_destroy(s); return -1; }

        int64_t int_min, int_max;
        double float_min, float_max;
        if (fread(&int_min, sizeof(int_min), 1, in) != 1) { gv_free(label); gv_free(pkey); graph_schema_destroy(s); return -1; }
        if (fread(&int_max, sizeof(int_max), 1, in) != 1) { gv_free(label); gv_free(pkey); graph_schema_destroy(s); return -1; }
        if (fread(&float_min, sizeof(float_min), 1, in) != 1) { gv_free(label); gv_free(pkey); graph_schema_destroy(s); return -1; }
        if (fread(&float_max, sizeof(float_max), 1, in) != 1) { gv_free(label); gv_free(pkey); graph_schema_destroy(s); return -1; }

        if (s->constraint_count == s->constraint_cap) {
            size_t new_cap = s->constraint_cap * 2;
            GV_SchemaConstraint *p = gv_realloc(s->constraints, new_cap * sizeof(GV_SchemaConstraint));
            if (!p) { gv_free(label); gv_free(pkey); graph_schema_destroy(s); return -1; }
            memset(p + s->constraint_cap, 0, (new_cap - s->constraint_cap) * sizeof(GV_SchemaConstraint));
            s->constraints = p;
            s->constraint_cap = new_cap;
        }
        GV_SchemaConstraint *c = &s->constraints[s->constraint_count];
        c->type = (GV_SchemaConstraintType)type_u;
        c->label = label;
        c->property_key = pkey;
        c->is_node = is_node_i;
        c->expected_type = (GV_PropType)etype;
        c->int_min = int_min;
        c->int_max = int_max;
        c->float_min = float_min;
        c->float_max = float_max;
        s->constraint_count++;
    }
    *schema = s;
    return 0;
}

size_t graph_schema_constraint_count(const GV_GraphSchema *schema) {
    return schema ? schema->constraint_count : 0;
}
