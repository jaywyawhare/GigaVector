/**
 * @file test_json.c
 * @brief Tests for the JSON parser/serializer module.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gigavector/gv_json.h"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    printf("Testing %s... ", name); \
    tests_run++; \
} while(0)

#define PASS() do { \
    printf("[OK]\n"); \
    tests_passed++; \
} while(0)

#define FAIL(msg) do { \
    printf("[FAIL] %s\n", msg); \
} while(0)

void test_parse_null(void) {
    TEST("parse null");

    GV_JsonError err;
    GV_JsonValue *val = gv_json_parse("null", &err);
    if (val == NULL || err != GV_JSON_OK || !gv_json_is_null(val)) {
        FAIL("Failed to parse null");
        gv_json_free(val);
        return;
    }
    gv_json_free(val);
    PASS();
}

void test_parse_bool(void) {
    TEST("parse booleans");

    GV_JsonError err;
    GV_JsonValue *val_true = gv_json_parse("true", &err);
    if (val_true == NULL || !gv_json_is_bool(val_true)) {
        FAIL("Failed to parse true");
        gv_json_free(val_true);
        return;
    }
    bool b;
    if (gv_json_get_bool(val_true, &b) != GV_JSON_OK || b != true) {
        FAIL("Wrong bool value for true");
        gv_json_free(val_true);
        return;
    }
    gv_json_free(val_true);

    GV_JsonValue *val_false = gv_json_parse("false", &err);
    if (val_false == NULL || !gv_json_is_bool(val_false)) {
        FAIL("Failed to parse false");
        gv_json_free(val_false);
        return;
    }
    if (gv_json_get_bool(val_false, &b) != GV_JSON_OK || b != false) {
        FAIL("Wrong bool value for false");
        gv_json_free(val_false);
        return;
    }
    gv_json_free(val_false);
    PASS();
}

void test_parse_numbers(void) {
    TEST("parse numbers");

    GV_JsonError err;
    double d;

    // Integer
    GV_JsonValue *val = gv_json_parse("42", &err);
    if (val == NULL || !gv_json_is_number(val)) {
        FAIL("Failed to parse 42");
        gv_json_free(val);
        return;
    }
    gv_json_get_number(val, &d);
    if (d != 42.0) {
        FAIL("Wrong value for 42");
        gv_json_free(val);
        return;
    }
    gv_json_free(val);

    // Negative
    val = gv_json_parse("-123", &err);
    gv_json_get_number(val, &d);
    if (d != -123.0) {
        FAIL("Wrong value for -123");
        gv_json_free(val);
        return;
    }
    gv_json_free(val);

    // Float
    val = gv_json_parse("3.14159", &err);
    gv_json_get_number(val, &d);
    if (fabs(d - 3.14159) > 0.00001) {
        FAIL("Wrong value for 3.14159");
        gv_json_free(val);
        return;
    }
    gv_json_free(val);

    // Scientific
    val = gv_json_parse("1.5e10", &err);
    gv_json_get_number(val, &d);
    if (fabs(d - 1.5e10) > 1.0) {
        FAIL("Wrong value for 1.5e10");
        gv_json_free(val);
        return;
    }
    gv_json_free(val);

    PASS();
}

void test_parse_strings(void) {
    TEST("parse strings");

    GV_JsonError err;

    // Simple string
    GV_JsonValue *val = gv_json_parse("\"hello\"", &err);
    if (val == NULL || !gv_json_is_string(val)) {
        FAIL("Failed to parse simple string");
        gv_json_free(val);
        return;
    }
    const char *s = gv_json_get_string(val);
    if (strcmp(s, "hello") != 0) {
        FAIL("Wrong value for simple string");
        gv_json_free(val);
        return;
    }
    gv_json_free(val);

    // Escaped characters
    val = gv_json_parse("\"hello\\nworld\\t\\\"quoted\\\"\"", &err);
    if (val == NULL) {
        FAIL("Failed to parse escaped string");
        return;
    }
    s = gv_json_get_string(val);
    if (strcmp(s, "hello\nworld\t\"quoted\"") != 0) {
        printf("Got: %s\n", s);
        FAIL("Wrong value for escaped string");
        gv_json_free(val);
        return;
    }
    gv_json_free(val);

    // Unicode escape
    val = gv_json_parse("\"hello\\u0041\"", &err);
    if (val == NULL) {
        FAIL("Failed to parse unicode escape");
        return;
    }
    s = gv_json_get_string(val);
    if (strcmp(s, "helloA") != 0) {
        FAIL("Wrong value for unicode escape");
        gv_json_free(val);
        return;
    }
    gv_json_free(val);

    PASS();
}

void test_parse_arrays(void) {
    TEST("parse arrays");

    GV_JsonError err;

    // Empty array
    GV_JsonValue *val = gv_json_parse("[]", &err);
    if (val == NULL || !gv_json_is_array(val) || gv_json_array_length(val) != 0) {
        FAIL("Failed to parse empty array");
        gv_json_free(val);
        return;
    }
    gv_json_free(val);

    // Array with values
    val = gv_json_parse("[1, 2, 3]", &err);
    if (val == NULL || !gv_json_is_array(val) || gv_json_array_length(val) != 3) {
        FAIL("Failed to parse [1,2,3]");
        gv_json_free(val);
        return;
    }

    double d;
    gv_json_get_number(gv_json_array_get(val, 0), &d);
    if (d != 1.0) {
        FAIL("Wrong first element");
        gv_json_free(val);
        return;
    }
    gv_json_get_number(gv_json_array_get(val, 2), &d);
    if (d != 3.0) {
        FAIL("Wrong third element");
        gv_json_free(val);
        return;
    }
    gv_json_free(val);

    // Mixed types
    val = gv_json_parse("[1, \"hello\", true, null]", &err);
    if (val == NULL || gv_json_array_length(val) != 4) {
        FAIL("Failed to parse mixed array");
        gv_json_free(val);
        return;
    }
    if (!gv_json_is_number(gv_json_array_get(val, 0)) ||
        !gv_json_is_string(gv_json_array_get(val, 1)) ||
        !gv_json_is_bool(gv_json_array_get(val, 2)) ||
        !gv_json_is_null(gv_json_array_get(val, 3))) {
        FAIL("Wrong types in mixed array");
        gv_json_free(val);
        return;
    }
    gv_json_free(val);

    PASS();
}

void test_parse_objects(void) {
    TEST("parse objects");

    GV_JsonError err;

    // Empty object
    GV_JsonValue *val = gv_json_parse("{}", &err);
    if (val == NULL || !gv_json_is_object(val) || gv_json_object_length(val) != 0) {
        FAIL("Failed to parse empty object");
        gv_json_free(val);
        return;
    }
    gv_json_free(val);

    // Simple object
    val = gv_json_parse("{\"name\": \"John\", \"age\": 30}", &err);
    if (val == NULL || !gv_json_is_object(val) || gv_json_object_length(val) != 2) {
        FAIL("Failed to parse simple object");
        gv_json_free(val);
        return;
    }

    const char *name = gv_json_get_string(gv_json_object_get(val, "name"));
    if (name == NULL || strcmp(name, "John") != 0) {
        FAIL("Wrong name value");
        gv_json_free(val);
        return;
    }

    double age;
    gv_json_get_number(gv_json_object_get(val, "age"), &age);
    if (age != 30.0) {
        FAIL("Wrong age value");
        gv_json_free(val);
        return;
    }
    gv_json_free(val);

    PASS();
}

void test_parse_nested(void) {
    TEST("parse nested structures");

    GV_JsonError err;
    const char *json = "{"
        "\"choices\": ["
            "{"
                "\"message\": {"
                    "\"content\": \"Hello, world!\""
                "}"
            "}"
        "]"
    "}";

    GV_JsonValue *val = gv_json_parse(json, &err);
    if (val == NULL) {
        FAIL("Failed to parse nested JSON");
        return;
    }

    // Test path-based access
    const char *content = gv_json_get_string_path(val, "choices.0.message.content");
    if (content == NULL || strcmp(content, "Hello, world!") != 0) {
        FAIL("Wrong path access result");
        gv_json_free(val);
        return;
    }

    gv_json_free(val);
    PASS();
}

void test_parse_openai_response(void) {
    TEST("parse OpenAI-style response");

    GV_JsonError err;
    const char *json = "{"
        "\"id\": \"chatcmpl-123\","
        "\"choices\": ["
            "{"
                "\"index\": 0,"
                "\"message\": {"
                    "\"role\": \"assistant\","
                    "\"content\": \"This is a test response with \\\"quotes\\\" and\\nnewlines.\""
                "},"
                "\"finish_reason\": \"stop\""
            "}"
        "],"
        "\"usage\": {"
            "\"prompt_tokens\": 10,"
            "\"completion_tokens\": 20,"
            "\"total_tokens\": 30"
        "}"
    "}";

    GV_JsonValue *val = gv_json_parse(json, &err);
    if (val == NULL) {
        FAIL("Failed to parse OpenAI response");
        return;
    }

    const char *content = gv_json_get_string_path(val, "choices.0.message.content");
    if (content == NULL) {
        FAIL("Failed to get content");
        gv_json_free(val);
        return;
    }

    // Verify escaped characters are unescaped
    if (strstr(content, "\"quotes\"") == NULL || strstr(content, "\n") == NULL) {
        printf("Content: %s\n", content);
        FAIL("Escaped characters not properly unescaped");
        gv_json_free(val);
        return;
    }

    // Verify usage
    GV_JsonValue *usage = gv_json_object_get(val, "usage");
    if (usage == NULL || !gv_json_is_object(usage)) {
        FAIL("Failed to get usage object");
        gv_json_free(val);
        return;
    }

    double total;
    GV_JsonValue *total_tokens = gv_json_object_get(usage, "total_tokens");
    gv_json_get_number(total_tokens, &total);
    if (total != 30.0) {
        FAIL("Wrong total_tokens value");
        gv_json_free(val);
        return;
    }

    gv_json_free(val);
    PASS();
}

void test_parse_facts_response(void) {
    TEST("parse facts JSON response");

    GV_JsonError err;
    const char *json = "{\"facts\": [\"Name is John\", \"Is a Software engineer\", \"Lives in San Francisco\"]}";

    GV_JsonValue *val = gv_json_parse(json, &err);
    if (val == NULL) {
        FAIL("Failed to parse facts JSON");
        return;
    }

    GV_JsonValue *facts = gv_json_object_get(val, "facts");
    if (facts == NULL || !gv_json_is_array(facts)) {
        FAIL("Failed to get facts array");
        gv_json_free(val);
        return;
    }

    size_t len = gv_json_array_length(facts);
    if (len != 3) {
        FAIL("Wrong number of facts");
        gv_json_free(val);
        return;
    }

    const char *fact1 = gv_json_get_string(gv_json_array_get(facts, 0));
    if (fact1 == NULL || strcmp(fact1, "Name is John") != 0) {
        FAIL("Wrong first fact");
        gv_json_free(val);
        return;
    }

    gv_json_free(val);
    PASS();
}

void test_stringify(void) {
    TEST("stringify JSON");

    // Create object
    GV_JsonValue *obj = gv_json_object();
    gv_json_object_set(obj, "name", gv_json_string("John"));
    gv_json_object_set(obj, "age", gv_json_number(30));
    gv_json_object_set(obj, "active", gv_json_bool(true));

    GV_JsonValue *arr = gv_json_array();
    gv_json_array_push(arr, gv_json_number(1));
    gv_json_array_push(arr, gv_json_number(2));
    gv_json_array_push(arr, gv_json_number(3));
    gv_json_object_set(obj, "numbers", arr);

    char *str = gv_json_stringify(obj, false);
    if (str == NULL) {
        FAIL("Failed to stringify");
        gv_json_free(obj);
        return;
    }

    // Parse it back
    GV_JsonError err;
    GV_JsonValue *parsed = gv_json_parse(str, &err);
    if (parsed == NULL) {
        printf("Stringify result: %s\n", str);
        FAIL("Failed to parse stringified JSON");
        free(str);
        gv_json_free(obj);
        return;
    }

    // Verify
    const char *name = gv_json_get_string(gv_json_object_get(parsed, "name"));
    if (name == NULL || strcmp(name, "John") != 0) {
        FAIL("Name not preserved");
        free(str);
        gv_json_free(obj);
        gv_json_free(parsed);
        return;
    }

    free(str);
    gv_json_free(obj);
    gv_json_free(parsed);
    PASS();
}

void test_copy(void) {
    TEST("deep copy JSON");

    GV_JsonError err;
    const char *json = "{\"nested\": {\"array\": [1, 2, 3]}}";
    GV_JsonValue *original = gv_json_parse(json, &err);
    if (original == NULL) {
        FAIL("Failed to parse");
        return;
    }

    GV_JsonValue *copy = gv_json_copy(original);
    if (copy == NULL) {
        FAIL("Failed to copy");
        gv_json_free(original);
        return;
    }

    // Free original and verify copy still works
    gv_json_free(original);

    GV_JsonValue *arr = gv_json_get_path(copy, "nested.array");
    if (arr == NULL || gv_json_array_length(arr) != 3) {
        FAIL("Copy not independent");
        gv_json_free(copy);
        return;
    }

    gv_json_free(copy);
    PASS();
}

void test_error_handling(void) {
    TEST("error handling");

    GV_JsonError err;

    // Invalid JSON
    GV_JsonValue *val = gv_json_parse("{invalid}", &err);
    if (val != NULL || err == GV_JSON_OK) {
        FAIL("Should fail on invalid JSON");
        gv_json_free(val);
        return;
    }

    // Unclosed string
    val = gv_json_parse("\"unclosed", &err);
    if (val != NULL || err == GV_JSON_OK) {
        FAIL("Should fail on unclosed string");
        gv_json_free(val);
        return;
    }

    // Trailing content
    val = gv_json_parse("123 extra", &err);
    if (val != NULL || err == GV_JSON_OK) {
        FAIL("Should fail on trailing content");
        gv_json_free(val);
        return;
    }

    PASS();
}

int main(void) {
    printf("Running JSON parser tests...\n\n");

    test_parse_null();
    test_parse_bool();
    test_parse_numbers();
    test_parse_strings();
    test_parse_arrays();
    test_parse_objects();
    test_parse_nested();
    test_parse_openai_response();
    test_parse_facts_response();
    test_stringify();
    test_copy();
    test_error_handling();

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);

    return tests_passed == tests_run ? 0 : 1;
}
