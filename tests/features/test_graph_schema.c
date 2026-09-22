#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "core/memory.h"
#include "features/graph_schema.h"
#include "../test_tmp.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_create_destroy(void) {
    GV_GraphSchema *s = graph_schema_create();
    ASSERT(s != NULL, "schema creation");
    ASSERT(graph_schema_constraint_count(s) == 0, "empty schema has 0 constraints");
    graph_schema_destroy(s);
    graph_schema_destroy(NULL);
    return 0;
}

static int test_add_required(void) {
    GV_GraphSchema *s = graph_schema_create();
    ASSERT(graph_schema_add_required(s, "Person", "name", 1) == 0, "add required");
    ASSERT(graph_schema_constraint_count(s) == 1, "1 constraint");

    const char *keys[] = {"age"};
    GV_PropValue vals[] = {gv_prop_int64(30)};
    char err[256] = {0};
    ASSERT(graph_schema_validate_node(s, "Person", keys, vals, 1, err, sizeof(err)) != 0, "fail: missing required 'name' with only 'age'");
    ASSERT(strlen(err) > 0, "error message set");

    const char *keys2[] = {"name"};
    GV_PropValue name_val = gv_prop_string("Alice");
    GV_PropValue vals2[] = {name_val};
    memset(err, 0, sizeof(err));
    ASSERT(graph_schema_validate_node(s, "Person", keys2, vals2, 1, err, sizeof(err)) == 0, "pass: required 'name' present");
    gv_prop_free(&name_val);

    ASSERT(graph_schema_validate_node(s, "Person", NULL, NULL, 0, err, sizeof(err)) != 0, "fail: missing required prop");
    ASSERT(strlen(err) > 0, "error message set");

    graph_schema_destroy(s);
    return 0;
}

static int test_add_unique(void) {
    GV_GraphSchema *s = graph_schema_create();
    ASSERT(graph_schema_add_unique(s, "Person", "email", 1) == 0, "add unique");

    GV_PropValue v1 = gv_prop_string("alice@test.com");
    ASSERT(graph_schema_register_unique(s, "Person", "email", 1, v1, 1) == 0, "register first");
    ASSERT(graph_schema_check_unique(s, "Person", "email", 1, v1) == 0, "unique: no duplicate");
    ASSERT(graph_schema_register_unique(s, "Person", "email", 1, v1, 2) != 0, "duplicate rejected");
    ASSERT(graph_schema_register_unique(s, "Person", "email", 1, v1, 1) == 0, "same entity re-register ok");

    GV_PropValue v2 = gv_prop_string("bob@test.com");
    ASSERT(graph_schema_register_unique(s, "Person", "email", 1, v2, 2) == 0, "different value ok");

    ASSERT(graph_schema_unregister_unique(s, "Person", "email", 1, v1) == 0, "unregister");
    ASSERT(graph_schema_check_unique(s, "Person", "email", 1, v1) == 1, "value available after unregister");
    ASSERT(graph_schema_register_unique(s, "Person", "email", 1, v1, 3) == 0, "re-register after unregister");

    gv_prop_free(&v1);
    gv_prop_free(&v2);
    graph_schema_destroy(s);
    return 0;
}

static int test_add_type(void) {
    GV_GraphSchema *s = graph_schema_create();
    ASSERT(graph_schema_add_type(s, "Person", "age", 1, GV_PROP_INT64) == 0, "add type constraint");

    const char *keys[] = {"age"};
    GV_PropValue vals[] = {gv_prop_int64(25)};
    char err[256] = {0};
    ASSERT(graph_schema_validate_node(s, "Person", keys, vals, 1, err, sizeof(err)) == 0, "int64 matches");

    const char *keys2[] = {"age"};
    GV_PropValue vals2[] = {gv_prop_float64(25.0)};
    ASSERT(graph_schema_validate_node(s, "Person", keys2, vals2, 1, err, sizeof(err)) != 0, "float64 fails");
    ASSERT(strlen(err) > 0, "error on type mismatch");

    graph_schema_destroy(s);
    return 0;
}

static int test_add_range_int(void) {
    GV_GraphSchema *s = graph_schema_create();
    ASSERT(graph_schema_add_range_int(s, "Person", "age", 1, 0, 150) == 0, "add range int");

    const char *keys[] = {"age"};
    GV_PropValue vals_ok[] = {gv_prop_int64(50)};
    char err[256] = {0};
    ASSERT(graph_schema_validate_node(s, "Person", keys, vals_ok, 1, err, sizeof(err)) == 0, "in range");

    GV_PropValue vals_bad[] = {gv_prop_int64(200)};
    ASSERT(graph_schema_validate_node(s, "Person", keys, vals_bad, 1, err, sizeof(err)) != 0, "out of range");
    ASSERT(strlen(err) > 0, "error on range violation");

    GV_PropValue vals_neg[] = {gv_prop_int64(-5)};
    ASSERT(graph_schema_validate_node(s, "Person", keys, vals_neg, 1, err, sizeof(err)) != 0, "below min");

    graph_schema_destroy(s);
    return 0;
}

static int test_add_range_float(void) {
    GV_GraphSchema *s = graph_schema_create();
    ASSERT(graph_schema_add_range_float(s, "Person", "score", 1, 0.0, 100.0) == 0, "add range float");

    const char *keys[] = {"score"};
    GV_PropValue vals_ok[] = {gv_prop_float64(75.5)};
    char err[256] = {0};
    ASSERT(graph_schema_validate_node(s, "Person", keys, vals_ok, 1, err, sizeof(err)) == 0, "in range");

    GV_PropValue vals_bad[] = {gv_prop_float64(150.0)};
    ASSERT(graph_schema_validate_node(s, "Person", keys, vals_bad, 1, err, sizeof(err)) != 0, "out of range");
    ASSERT(strlen(err) > 0, "error on range violation");

    graph_schema_destroy(s);
    return 0;
}

static int test_edge_constraints(void) {
    GV_GraphSchema *s = graph_schema_create();
    ASSERT(graph_schema_add_required(s, "KNOWS", "since", 0) == 0, "add required on edge");

    const char *keys[] = {"weight"};
    GV_PropValue vals[] = {gv_prop_float64(1.0)};
    char err[256] = {0};
    ASSERT(graph_schema_validate_edge(s, "KNOWS", keys, vals, 1, err, sizeof(err)) != 0, "edge missing required");
    ASSERT(strlen(err) > 0, "error on edge validation");

    const char *keys2[] = {"since"};
    GV_PropValue since_val = gv_prop_string("2020");
    GV_PropValue vals2[] = {since_val};
    ASSERT(graph_schema_validate_edge(s, "KNOWS", keys2, vals2, 1, err, sizeof(err)) == 0, "edge has required");
    gv_prop_free(&since_val);

    graph_schema_destroy(s);
    return 0;
}

static int test_remove_constraint(void) {
    GV_GraphSchema *s = graph_schema_create();
    graph_schema_add_required(s, "Person", "name", 1);
    graph_schema_add_type(s, "Person", "age", 1, GV_PROP_INT64);
    ASSERT(graph_schema_constraint_count(s) == 2, "2 constraints");

    ASSERT(graph_schema_remove_constraint(s, "Person", "name", 1) == 0, "remove required");
    ASSERT(graph_schema_constraint_count(s) == 1, "1 constraint left");

    ASSERT(graph_schema_remove_constraint(s, "Person", "age", 1) == 0, "remove type");
    ASSERT(graph_schema_constraint_count(s) == 0, "0 constraints");

    ASSERT(graph_schema_remove_constraint(s, "Person", "name", 1) != 0, "remove non-existent fails");

    graph_schema_destroy(s);
    return 0;
}

static int test_get_constraints(void) {
    GV_GraphSchema *s = graph_schema_create();
    graph_schema_add_required(s, "Person", "name", 1);
    graph_schema_add_type(s, "Person", "age", 1, GV_PROP_INT64);
    graph_schema_add_required(s, "KNOWS", "since", 0);

    GV_SchemaConstraint out[10];
    int n = graph_schema_get_constraints(s, "Person", 1, out, 10);
    ASSERT(n == 2, "Person node has 2 constraints");
    for (int i = 0; i < n; i++) {
        gv_free(out[i].label);
        gv_free(out[i].property_key);
    }

    n = graph_schema_get_constraints(s, "KNOWS", 0, out, 10);
    ASSERT(n == 1, "KNOWS edge has 1 constraint");
    for (int i = 0; i < n; i++) {
        gv_free(out[i].label);
        gv_free(out[i].property_key);
    }

    n = graph_schema_get_constraints(s, "Person", 0, out, 10);
    ASSERT(n == 0, "Person edge has 0 constraints");

    graph_schema_destroy(s);
    return 0;
}

static int test_persistence(void) {
    char path[512];
    ASSERT(gv_test_make_temp_path(path, sizeof(path), "test_graph_schema", ".bin") == 0, "temp path");

    GV_GraphSchema *s = graph_schema_create();
    graph_schema_add_required(s, "Person", "name", 1);
    graph_schema_add_unique(s, "Person", "email", 1);
    graph_schema_add_type(s, "Person", "age", 1, GV_PROP_INT64);
    graph_schema_add_range_int(s, "Person", "age", 1, 0, 150);
    graph_schema_add_range_float(s, "Person", "score", 1, 0.0, 100.0);
    graph_schema_add_required(s, "KNOWS", "since", 0);

    FILE *fout = fopen(path, "wb");
    ASSERT(fout != NULL, "open for write");
    ASSERT(graph_schema_save(s, fout) == 0, "save");
    fclose(fout);

    FILE *fin = fopen(path, "rb");
    ASSERT(fin != NULL, "open for read");
    GV_GraphSchema *loaded = NULL;
    ASSERT(graph_schema_load(&loaded, fin) == 0, "load");
    fclose(fin);

    ASSERT(loaded != NULL, "loaded not NULL");
    ASSERT(graph_schema_constraint_count(loaded) == graph_schema_constraint_count(s), "same count");

    GV_SchemaConstraint out_s[10], out_l[10];
    int ns = graph_schema_get_constraints(s, "Person", 1, out_s, 10);
    int nl = graph_schema_get_constraints(loaded, "Person", 1, out_l, 10);
    ASSERT(ns == nl, "same Person constraint count");
    for (int i = 0; i < ns; i++) {
        ASSERT(out_s[i].type == out_l[i].type, "same type");
        ASSERT(strcmp(out_s[i].property_key, out_l[i].property_key) == 0, "same key");
        ASSERT(out_s[i].is_node == out_l[i].is_node, "same is_node");
        gv_free(out_s[i].label); gv_free(out_s[i].property_key);
        gv_free(out_l[i].label); gv_free(out_l[i].property_key);
    }

    graph_schema_destroy(s);
    graph_schema_destroy(loaded);
    remove(path);
    return 0;
}

static int test_validate_no_constraints(void) {
    GV_GraphSchema *s = graph_schema_create();
    const char *keys[] = {"anything"};
    GV_PropValue v = gv_prop_string("value");
    GV_PropValue vals[] = {v};
    char err[256] = {0};
    ASSERT(graph_schema_validate_node(s, "Person", keys, vals, 1, err, sizeof(err)) == 0, "no constraints passes");
    gv_prop_free(&v);
    graph_schema_destroy(s);
    return 0;
}

static int test_multiple_constraints_combined(void) {
    GV_GraphSchema *s = graph_schema_create();
    graph_schema_add_required(s, "Person", "name", 1);
    graph_schema_add_type(s, "Person", "age", 1, GV_PROP_INT64);
    graph_schema_add_range_int(s, "Person", "age", 1, 0, 150);

    GV_PropValue name_a = gv_prop_string("Alice");
    const char *keys[] = {"name", "age"};
    GV_PropValue vals[] = {name_a, gv_prop_int64(30)};
    char err[256] = {0};
    ASSERT(graph_schema_validate_node(s, "Person", keys, vals, 2, err, sizeof(err)) == 0, "all pass");
    gv_prop_free(&name_a);

    const char *keys2[] = {"age"};
    GV_PropValue vals2[] = {gv_prop_int64(30)};
    ASSERT(graph_schema_validate_node(s, "Person", keys2, vals2, 1, err, sizeof(err)) != 0, "missing name fails");

    GV_PropValue name_b = gv_prop_string("Bob");
    const char *keys3[] = {"name", "age"};
    GV_PropValue vals3[] = {name_b, gv_prop_float64(30.0)};
    ASSERT(graph_schema_validate_node(s, "Person", keys3, vals3, 2, err, sizeof(err)) != 0, "wrong type fails");
    gv_prop_free(&name_b);

    GV_PropValue name_b2 = gv_prop_string("Bob");
    const char *keys4[] = {"name", "age"};
    GV_PropValue vals4[] = {name_b2, gv_prop_int64(200)};
    ASSERT(graph_schema_validate_node(s, "Person", keys4, vals4, 2, err, sizeof(err)) != 0, "out of range fails");
    gv_prop_free(&name_b2);

    graph_schema_destroy(s);
    return 0;
}

static int test_uniqueness_growth(void) {
    GV_GraphSchema *s = graph_schema_create();
    graph_schema_add_unique(s, "Person", "email", 1);

    GV_PropValue *email_vals = gv_alloc(200 * sizeof(GV_PropValue));

    for (int i = 0; i < 200; i++) {
        char email[64];
        snprintf(email, sizeof(email), "user%d@example.com", i);
        email_vals[i] = gv_prop_string(email);
        ASSERT(graph_schema_register_unique(s, "Person", "email", 1, email_vals[i], (uint64_t)(i + 1)) == 0,
               "register unique value");
    }

    for (int i = 0; i < 200; i++) {
        ASSERT(graph_schema_check_unique(s, "Person", "email", 1, email_vals[i]) == 0,
               "all registered values present");
    }

    GV_PropValue dup = gv_prop_string("user0@example.com");
    ASSERT(graph_schema_register_unique(s, "Person", "email", 1, dup, 999) != 0, "duplicate rejected after growth");
    gv_prop_free(&dup);

    for (int i = 0; i < 200; i++) {
        gv_prop_free(&email_vals[i]);
    }
    gv_free(email_vals);
    graph_schema_destroy(s);
    return 0;
}

static int test_mixed_entity_types(void) {
    GV_GraphSchema *s = graph_schema_create();
    graph_schema_add_unique(s, "Person", "email", 1);
    graph_schema_add_unique(s, "KNOWS", "since", 0);

    GV_PropValue v = gv_prop_string("alice@test.com");
    ASSERT(graph_schema_register_unique(s, "Person", "email", 1, v, 1) == 0, "node unique ok");
    ASSERT(graph_schema_register_unique(s, "KNOWS", "since", 1, v, 100) == 0, "different label/type ok");
    ASSERT(graph_schema_check_unique(s, "Person", "email", 1, v) == 0, "check node unique");
    ASSERT(graph_schema_check_unique(s, "KNOWS", "since", 0, v) == 1, "edge prop not found");

    GV_PropValue v2 = gv_prop_string("2020");
    ASSERT(graph_schema_register_unique(s, "KNOWS", "since", 0, v2, 100) == 0, "edge unique ok");

    gv_prop_free(&v);
    gv_prop_free(&v2);
    graph_schema_destroy(s);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing create/destroy...", test_create_destroy},
        {"Testing required constraint...", test_add_required},
        {"Testing unique constraint...", test_add_unique},
        {"Testing type constraint...", test_add_type},
        {"Testing range int constraint...", test_add_range_int},
        {"Testing range float constraint...", test_add_range_float},
        {"Testing edge constraints...", test_edge_constraints},
        {"Testing remove constraint...", test_remove_constraint},
        {"Testing get constraints...", test_get_constraints},
        {"Testing persistence round-trip...", test_persistence},
        {"Testing no constraints passes...", test_validate_no_constraints},
        {"Testing combined constraints...", test_multiple_constraints_combined},
        {"Testing uniqueness growth...", test_uniqueness_growth},
        {"Testing mixed entity types...", test_mixed_entity_types},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
