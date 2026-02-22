#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_schema.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_schema_create_destroy(void) {
    GV_Schema *schema = gv_schema_create(1);
    ASSERT(schema != NULL, "schema creation with version=1");
    ASSERT(schema->version == 1, "version is 1");
    ASSERT(schema->field_count == 0, "initial field_count is 0");

    gv_schema_destroy(schema);

    /* Destroy NULL should be safe */
    gv_schema_destroy(NULL);
    return 0;
}

static int test_schema_add_field(void) {
    GV_Schema *schema = gv_schema_create(1);
    ASSERT(schema != NULL, "schema creation");

    ASSERT(gv_schema_add_field(schema, "name", GV_SCHEMA_STRING, 1, "") == 0,
           "add required string field 'name'");
    ASSERT(gv_schema_add_field(schema, "age", GV_SCHEMA_INT, 0, "0") == 0,
           "add optional int field 'age' with default '0'");
    ASSERT(gv_schema_add_field(schema, "score", GV_SCHEMA_FLOAT, 0, "0.0") == 0,
           "add optional float field 'score'");
    ASSERT(gv_schema_add_field(schema, "active", GV_SCHEMA_BOOL, 0, "true") == 0,
           "add optional bool field 'active'");

    ASSERT(gv_schema_field_count(schema) == 4, "field count is 4");

    gv_schema_destroy(schema);
    return 0;
}

static int test_schema_get_and_has_field(void) {
    GV_Schema *schema = gv_schema_create(1);
    ASSERT(schema != NULL, "schema creation");

    gv_schema_add_field(schema, "title", GV_SCHEMA_STRING, 1, "");
    gv_schema_add_field(schema, "count", GV_SCHEMA_INT, 0, "10");

    ASSERT(gv_schema_has_field(schema, "title") == 1, "has 'title'");
    ASSERT(gv_schema_has_field(schema, "count") == 1, "has 'count'");
    ASSERT(gv_schema_has_field(schema, "nonexistent") == 0, "does not have 'nonexistent'");

    const GV_SchemaField *field = gv_schema_get_field(schema, "title");
    ASSERT(field != NULL, "get field 'title'");
    ASSERT(strcmp(field->name, "title") == 0, "field name is 'title'");
    ASSERT(field->type == GV_SCHEMA_STRING, "field type is STRING");
    ASSERT(field->required == 1, "field is required");

    field = gv_schema_get_field(schema, "count");
    ASSERT(field != NULL, "get field 'count'");
    ASSERT(field->type == GV_SCHEMA_INT, "count field type is INT");
    ASSERT(strcmp(field->default_value, "10") == 0, "count default is '10'");

    gv_schema_destroy(schema);
    return 0;
}

static int test_schema_remove_field(void) {
    GV_Schema *schema = gv_schema_create(1);
    ASSERT(schema != NULL, "schema creation");

    gv_schema_add_field(schema, "a", GV_SCHEMA_STRING, 0, "");
    gv_schema_add_field(schema, "b", GV_SCHEMA_INT, 0, "0");
    gv_schema_add_field(schema, "c", GV_SCHEMA_FLOAT, 0, "0.0");
    ASSERT(gv_schema_field_count(schema) == 3, "field count is 3");

    ASSERT(gv_schema_remove_field(schema, "b") == 0, "remove field 'b'");
    ASSERT(gv_schema_field_count(schema) == 2, "field count is 2 after removal");
    ASSERT(gv_schema_has_field(schema, "b") == 0, "'b' is absent after removal");
    ASSERT(gv_schema_has_field(schema, "a") == 1, "'a' still present");
    ASSERT(gv_schema_has_field(schema, "c") == 1, "'c' still present");

    /* Removing nonexistent field */
    ASSERT(gv_schema_remove_field(schema, "nonexistent") == -1,
           "removing nonexistent returns -1");

    gv_schema_destroy(schema);
    return 0;
}

static int test_schema_validate(void) {
    GV_Schema *schema = gv_schema_create(1);
    ASSERT(schema != NULL, "schema creation");

    gv_schema_add_field(schema, "name", GV_SCHEMA_STRING, 1, "");
    gv_schema_add_field(schema, "age", GV_SCHEMA_INT, 0, "25");

    /* Valid: required field 'name' provided */
    const char *keys1[] = {"name", "age"};
    const char *vals1[] = {"Alice", "30"};
    ASSERT(gv_schema_validate(schema, keys1, vals1, 2) == 0,
           "validation passes with all fields");

    /* Valid: optional field 'age' omitted */
    const char *keys2[] = {"name"};
    const char *vals2[] = {"Bob"};
    ASSERT(gv_schema_validate(schema, keys2, vals2, 1) == 0,
           "validation passes with only required fields");

    /* Invalid: required field 'name' missing */
    const char *keys3[] = {"age"};
    const char *vals3[] = {"42"};
    int result = gv_schema_validate(schema, keys3, vals3, 1);
    ASSERT(result != 0, "validation fails when required field missing");

    gv_schema_destroy(schema);
    return 0;
}

static int test_schema_copy(void) {
    GV_Schema *schema = gv_schema_create(2);
    ASSERT(schema != NULL, "schema creation");

    gv_schema_add_field(schema, "x", GV_SCHEMA_FLOAT, 1, "");
    gv_schema_add_field(schema, "y", GV_SCHEMA_FLOAT, 1, "");

    GV_Schema *copy = gv_schema_copy(schema);
    ASSERT(copy != NULL, "schema copy");
    ASSERT(copy->version == 2, "copy version matches");
    ASSERT(gv_schema_field_count(copy) == 2, "copy field count matches");
    ASSERT(gv_schema_has_field(copy, "x") == 1, "copy has field 'x'");
    ASSERT(gv_schema_has_field(copy, "y") == 1, "copy has field 'y'");

    /* Modifying original should not affect copy */
    gv_schema_add_field(schema, "z", GV_SCHEMA_FLOAT, 0, "0");
    ASSERT(gv_schema_field_count(schema) == 3, "original now has 3 fields");
    ASSERT(gv_schema_field_count(copy) == 2, "copy still has 2 fields");

    gv_schema_destroy(schema);
    gv_schema_destroy(copy);
    return 0;
}

static int test_schema_diff(void) {
    GV_Schema *old_s = gv_schema_create(1);
    GV_Schema *new_s = gv_schema_create(2);
    ASSERT(old_s != NULL && new_s != NULL, "schema creation");

    /* old: {a: STRING, b: INT} */
    gv_schema_add_field(old_s, "a", GV_SCHEMA_STRING, 1, "");
    gv_schema_add_field(old_s, "b", GV_SCHEMA_INT, 0, "0");

    /* new: {a: STRING, c: FLOAT} -- b removed, c added */
    gv_schema_add_field(new_s, "a", GV_SCHEMA_STRING, 1, "");
    gv_schema_add_field(new_s, "c", GV_SCHEMA_FLOAT, 0, "0.0");

    GV_SchemaDiff diffs[10];
    int ndiff = gv_schema_diff(old_s, new_s, diffs, 10);
    ASSERT(ndiff >= 0, "schema diff did not fail");

    /* We expect at least 'b' removed and 'c' added */
    int found_b_removed = 0, found_c_added = 0;
    for (int i = 0; i < ndiff; i++) {
        if (strcmp(diffs[i].name, "b") == 0 && diffs[i].removed) found_b_removed = 1;
        if (strcmp(diffs[i].name, "c") == 0 && diffs[i].added) found_c_added = 1;
    }
    ASSERT(found_b_removed, "diff detected 'b' was removed");
    ASSERT(found_c_added, "diff detected 'c' was added");

    gv_schema_destroy(old_s);
    gv_schema_destroy(new_s);
    return 0;
}

static int test_schema_save_load_and_json(void) {
    const char *path = "/tmp/test_schema_save_load.bin";
    GV_Schema *schema = gv_schema_create(1);
    ASSERT(schema != NULL, "schema creation");

    gv_schema_add_field(schema, "name", GV_SCHEMA_STRING, 1, "");
    gv_schema_add_field(schema, "rating", GV_SCHEMA_FLOAT, 0, "0.0");

    /* Save */
    FILE *fout = fopen(path, "wb");
    ASSERT(fout != NULL, "open file for writing");
    ASSERT(gv_schema_save(schema, fout) == 0, "save schema");
    fclose(fout);

    /* Load */
    FILE *fin = fopen(path, "rb");
    ASSERT(fin != NULL, "open file for reading");
    GV_Schema *loaded = gv_schema_load(fin);
    fclose(fin);
    ASSERT(loaded != NULL, "load schema");
    ASSERT(loaded->version == 1, "loaded version is 1");
    ASSERT(gv_schema_field_count(loaded) == 2, "loaded field count is 2");
    ASSERT(gv_schema_has_field(loaded, "name") == 1, "loaded has 'name'");
    ASSERT(gv_schema_has_field(loaded, "rating") == 1, "loaded has 'rating'");

    /* JSON export */
    char *json = gv_schema_to_json(schema);
    ASSERT(json != NULL, "schema to JSON not NULL");
    ASSERT(strlen(json) > 0, "JSON string is non-empty");
    /* Basic sanity: should contain field names */
    ASSERT(strstr(json, "name") != NULL, "JSON contains 'name'");
    ASSERT(strstr(json, "rating") != NULL, "JSON contains 'rating'");
    free(json);

    gv_schema_destroy(schema);
    gv_schema_destroy(loaded);
    remove(path);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing schema create/destroy...", test_schema_create_destroy},
        {"Testing schema add field...", test_schema_add_field},
        {"Testing schema get and has field...", test_schema_get_and_has_field},
        {"Testing schema remove field...", test_schema_remove_field},
        {"Testing schema validate...", test_schema_validate},
        {"Testing schema copy...", test_schema_copy},
        {"Testing schema diff...", test_schema_diff},
        {"Testing schema save/load and JSON...", test_schema_save_load_and_json},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
