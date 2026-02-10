#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "gigavector/gv_typed_metadata.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_typed_null(void) {
    GV_TypedValue val = gv_typed_null();
    ASSERT(val.type == GV_META_TYPE_NULL, "null type should be GV_META_TYPE_NULL");

    gv_typed_value_free(&val);
    return 0;
}

static int test_typed_string(void) {
    GV_TypedValue val = gv_typed_string("hello world");
    ASSERT(val.type == GV_META_TYPE_STRING, "type should be string");

    const char *s = gv_typed_get_string(&val);
    ASSERT(s != NULL, "get_string should return non-NULL");
    ASSERT(strcmp(s, "hello world") == 0, "string content should match");

    /* Test string contains */
    ASSERT(gv_typed_string_contains(&val, "world") == true, "should contain 'world'");
    ASSERT(gv_typed_string_contains(&val, "xyz") == false, "should not contain 'xyz'");

    /* Test string starts_with */
    ASSERT(gv_typed_string_starts_with(&val, "hello") == true, "should start with 'hello'");
    ASSERT(gv_typed_string_starts_with(&val, "world") == false, "should not start with 'world'");

    gv_typed_value_free(&val);
    return 0;
}

static int test_typed_int(void) {
    GV_TypedValue val = gv_typed_int(42);
    ASSERT(val.type == GV_META_TYPE_INT64, "type should be int64");

    int64_t out = 0;
    int rc = gv_typed_get_int(&val, &out);
    ASSERT(rc == 0, "get_int should succeed");
    ASSERT(out == 42, "int value should be 42");

    /* get_float on int should fail */
    double fout = 0.0;
    rc = gv_typed_get_float(&val, &fout);
    ASSERT(rc == -1, "get_float on int type should fail");

    gv_typed_value_free(&val);
    return 0;
}

static int test_typed_float_and_bool(void) {
    GV_TypedValue fval = gv_typed_float(3.14);
    ASSERT(fval.type == GV_META_TYPE_FLOAT64, "type should be float64");

    double fout = 0.0;
    int rc = gv_typed_get_float(&fval, &fout);
    ASSERT(rc == 0, "get_float should succeed");
    ASSERT(fout > 3.13 && fout < 3.15, "float value should be approximately 3.14");

    GV_TypedValue bval = gv_typed_bool(true);
    ASSERT(bval.type == GV_META_TYPE_BOOL, "type should be bool");

    bool bout = false;
    rc = gv_typed_get_bool(&bval, &bout);
    ASSERT(rc == 0, "get_bool should succeed");
    ASSERT(bout == true, "bool value should be true");

    gv_typed_value_free(&fval);
    gv_typed_value_free(&bval);
    return 0;
}

static int test_typed_array(void) {
    GV_TypedValue arr = gv_typed_array(GV_META_TYPE_INT64);
    ASSERT(arr.type == GV_META_TYPE_ARRAY, "type should be array");
    ASSERT(gv_typed_array_length(&arr) == 0, "new array should be empty");

    GV_TypedValue item1 = gv_typed_int(10);
    GV_TypedValue item2 = gv_typed_int(20);
    GV_TypedValue item3 = gv_typed_int(30);

    int rc = gv_typed_array_push(&arr, &item1);
    ASSERT(rc == 0, "push item1 should succeed");
    rc = gv_typed_array_push(&arr, &item2);
    ASSERT(rc == 0, "push item2 should succeed");
    rc = gv_typed_array_push(&arr, &item3);
    ASSERT(rc == 0, "push item3 should succeed");

    ASSERT(gv_typed_array_length(&arr) == 3, "array should have 3 elements");

    GV_TypedValue *got = gv_typed_array_get(&arr, 1);
    ASSERT(got != NULL, "get index 1 should succeed");
    int64_t val = 0;
    gv_typed_get_int(got, &val);
    ASSERT(val == 20, "element at index 1 should be 20");

    /* Out of bounds */
    ASSERT(gv_typed_array_get(&arr, 99) == NULL, "out-of-bounds get should return NULL");

    /* Array contains */
    GV_TypedValue search = gv_typed_int(20);
    ASSERT(gv_typed_array_contains(&arr, &search) == true, "array should contain 20");
    GV_TypedValue missing = gv_typed_int(999);
    ASSERT(gv_typed_array_contains(&arr, &missing) == false, "array should not contain 999");

    gv_typed_value_free(&item1);
    gv_typed_value_free(&item2);
    gv_typed_value_free(&item3);
    gv_typed_value_free(&search);
    gv_typed_value_free(&missing);
    gv_typed_value_free(&arr);
    return 0;
}

static int test_typed_object(void) {
    GV_TypedValue obj = gv_typed_object();
    ASSERT(obj.type == GV_META_TYPE_OBJECT, "type should be object");
    ASSERT(gv_typed_object_length(&obj) == 0, "new object should be empty");

    GV_TypedValue name_val = gv_typed_string("Alice");
    GV_TypedValue age_val = gv_typed_int(30);

    int rc = gv_typed_object_set(&obj, "name", &name_val);
    ASSERT(rc == 0, "set 'name' should succeed");
    rc = gv_typed_object_set(&obj, "age", &age_val);
    ASSERT(rc == 0, "set 'age' should succeed");

    ASSERT(gv_typed_object_length(&obj) == 2, "object should have 2 keys");
    ASSERT(gv_typed_object_has(&obj, "name") == true, "object should have 'name'");
    ASSERT(gv_typed_object_has(&obj, "missing") == false, "object should not have 'missing'");

    GV_TypedValue *got_name = gv_typed_object_get(&obj, "name");
    ASSERT(got_name != NULL, "get 'name' should succeed");
    const char *s = gv_typed_get_string(got_name);
    ASSERT(s != NULL && strcmp(s, "Alice") == 0, "name value should be 'Alice'");

    gv_typed_value_free(&name_val);
    gv_typed_value_free(&age_val);
    gv_typed_value_free(&obj);
    return 0;
}

static int test_typed_compare_equals(void) {
    GV_TypedValue a = gv_typed_int(100);
    GV_TypedValue b = gv_typed_int(100);
    GV_TypedValue c = gv_typed_int(200);

    ASSERT(gv_typed_equals(&a, &b) == true, "equal ints should be equal");
    ASSERT(gv_typed_equals(&a, &c) == false, "different ints should not be equal");
    ASSERT(gv_typed_compare(&a, &c) < 0, "100 should be less than 200");
    ASSERT(gv_typed_compare(&c, &a) > 0, "200 should be greater than 100");

    /* in_range */
    ASSERT(gv_typed_in_range(&a, 50.0, 150.0) == true, "100 should be in range [50,150]");
    ASSERT(gv_typed_in_range(&a, 200.0, 300.0) == false, "100 should not be in range [200,300]");

    gv_typed_value_free(&a);
    gv_typed_value_free(&b);
    gv_typed_value_free(&c);
    return 0;
}

static int test_typed_copy_and_type_name(void) {
    GV_TypedValue original = gv_typed_string("test_copy");
    GV_TypedValue copy = gv_typed_value_copy(&original);

    ASSERT(copy.type == GV_META_TYPE_STRING, "copy type should be string");
    const char *s = gv_typed_get_string(&copy);
    ASSERT(s != NULL && strcmp(s, "test_copy") == 0, "copy value should match");

    /* Verify type name */
    const char *tn = gv_typed_type_name(GV_META_TYPE_STRING);
    ASSERT(tn != NULL && strlen(tn) > 0, "type name for string should be non-empty");

    const char *tn_null = gv_typed_type_name(GV_META_TYPE_NULL);
    ASSERT(tn_null != NULL, "type name for null should be non-NULL");

    gv_typed_value_free(&original);
    gv_typed_value_free(&copy);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing typed null...", test_typed_null},
        {"Testing typed string...", test_typed_string},
        {"Testing typed int...", test_typed_int},
        {"Testing typed float and bool...", test_typed_float_and_bool},
        {"Testing typed array...", test_typed_array},
        {"Testing typed object...", test_typed_object},
        {"Testing typed compare/equals...", test_typed_compare_equals},
        {"Testing typed copy and type name...", test_typed_copy_and_type_name},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        printf("%s", tests[i].name);
        if (tests[i].fn() == 0) { printf(" [OK]\n"); passed++; }
        else { printf(" [FAIL]\n"); }
    }
    printf("\n%d/%d tests passed\n", passed, n);
    return passed == n ? 0 : 1;
}
