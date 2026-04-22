/**
 * @file test_json.c
 * @brief Tests for the JSON parser/serializer module.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "features/json.h"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
} while(0)

#define PASS() do { \
    tests_passed++; \
} while(0)

#define FAIL(msg) do { \
    (void)(msg); \
} while(0)

void test_parse_null(void) {
    TEST("parse null");

    GV_JsonError err;
    GV_JsonValue *val = json_parse("null", &err);
    if (val == NULL || err != GV_JSON_OK || !json_is_null(val)) {
        FAIL("Failed to parse null");
        json_free(val);
        return;
    }
    json_free(val);
    PASS();
}

void test_parse_bool(void) {
    TEST("parse booleans");

    GV_JsonError err;
    GV_JsonValue *val_true = json_parse("true", &err);
    if (val_true == NULL || !json_is_bool(val_true)) {
        FAIL("Failed to parse true");
        json_free(val_true);
        return;
    }
    bool b;
    if (json_get_bool(val_true, &b) != GV_JSON_OK || b != true) {
        FAIL("Wrong bool value for true");
        json_free(val_true);
        return;
    }
    json_free(val_true);

    GV_JsonValue *val_false = json_parse("false", &err);
    if (val_false == NULL || !json_is_bool(val_false)) {
        FAIL("Failed to parse false");
        json_free(val_false);
        return;
    }
    if (json_get_bool(val_false, &b) != GV_JSON_OK || b != false) {
        FAIL("Wrong bool value for false");
        json_free(val_false);
        return;
    }
    json_free(val_false);
    PASS();
}

void test_parse_numbers(void) {
    TEST("parse numbers");

    GV_JsonError err;
    double d;

    GV_JsonValue *val = json_parse("42", &err);
    if (val == NULL || !json_is_number(val)) {
        FAIL("Failed to parse 42");
        json_free(val);
        return;
    }
    json_get_number(val, &d);
    if (d != 42.0) {
        FAIL("Wrong value for 42");
        json_free(val);
        return;
    }
    json_free(val);

    val = json_parse("-123", &err);
    json_get_number(val, &d);
    if (d != -123.0) {
        FAIL("Wrong value for -123");
        json_free(val);
        return;
    }
    json_free(val);

    val = json_parse("3.14159", &err);
    json_get_number(val, &d);
    if (fabs(d - 3.14159) > 0.00001) {
        FAIL("Wrong value for 3.14159");
        json_free(val);
        return;
    }
    json_free(val);

    val = json_parse("1.5e10", &err);
    json_get_number(val, &d);
    if (fabs(d - 1.5e10) > 1.0) {
        FAIL("Wrong value for 1.5e10");
        json_free(val);
        return;
    }
    json_free(val);

    PASS();
}

void test_parse_strings(void) {
    TEST("parse strings");

    GV_JsonError err;

    GV_JsonValue *val = json_parse("\"hello\"", &err);
    if (val == NULL || !json_is_string(val)) {
        FAIL("Failed to parse simple string");
        json_free(val);
        return;
    }
    const char *s = json_get_string(val);
    if (strcmp(s, "hello") != 0) {
        FAIL("Wrong value for simple string");
        json_free(val);
        return;
    }
    json_free(val);

    val = json_parse("\"hello\\nworld\\t\\\"quoted\\\"\"", &err);
    if (val == NULL) {
        FAIL("Failed to parse escaped string");
        return;
    }
    s = json_get_string(val);
    if (strcmp(s, "hello\nworld\t\"quoted\"") != 0) {
        FAIL("Wrong value for escaped string");
        json_free(val);
        return;
    }
    json_free(val);

    val = json_parse("\"hello\\u0041\"", &err);
    if (val == NULL) {
        FAIL("Failed to parse unicode escape");
        return;
    }
    s = json_get_string(val);
    if (strcmp(s, "helloA") != 0) {
        FAIL("Wrong value for unicode escape");
        json_free(val);
        return;
    }
    json_free(val);

    PASS();
}

void test_parse_arrays(void) {
    TEST("parse arrays");

    GV_JsonError err;

    GV_JsonValue *val = json_parse("[]", &err);
    if (val == NULL || !json_is_array(val) || json_array_length(val) != 0) {
        FAIL("Failed to parse empty array");
        json_free(val);
        return;
    }
    json_free(val);

    val = json_parse("[1, 2, 3]", &err);
    if (val == NULL || !json_is_array(val) || json_array_length(val) != 3) {
        FAIL("Failed to parse [1,2,3]");
        json_free(val);
        return;
    }

    double d;
    json_get_number(json_array_get(val, 0), &d);
    if (d != 1.0) {
        FAIL("Wrong first element");
        json_free(val);
        return;
    }
    json_get_number(json_array_get(val, 2), &d);
    if (d != 3.0) {
        FAIL("Wrong third element");
        json_free(val);
        return;
    }
    json_free(val);

    val = json_parse("[1, \"hello\", true, null]", &err);
    if (val == NULL || json_array_length(val) != 4) {
        FAIL("Failed to parse mixed array");
        json_free(val);
        return;
    }
    if (!json_is_number(json_array_get(val, 0)) ||
        !json_is_string(json_array_get(val, 1)) ||
        !json_is_bool(json_array_get(val, 2)) ||
        !json_is_null(json_array_get(val, 3))) {
        FAIL("Wrong types in mixed array");
        json_free(val);
        return;
    }
    json_free(val);

    PASS();
}

void test_parse_objects(void) {
    TEST("parse objects");

    GV_JsonError err;

    GV_JsonValue *val = json_parse("{}", &err);
    if (val == NULL || !json_is_object(val) || json_object_length(val) != 0) {
        FAIL("Failed to parse empty object");
        json_free(val);
        return;
    }
    json_free(val);

    val = json_parse("{\"name\": \"John\", \"age\": 30}", &err);
    if (val == NULL || !json_is_object(val) || json_object_length(val) != 2) {
        FAIL("Failed to parse simple object");
        json_free(val);
        return;
    }

    const char *name = json_get_string(json_object_get(val, "name"));
    if (name == NULL || strcmp(name, "John") != 0) {
        FAIL("Wrong name value");
        json_free(val);
        return;
    }

    double age;
    json_get_number(json_object_get(val, "age"), &age);
    if (age != 30.0) {
        FAIL("Wrong age value");
        json_free(val);
        return;
    }
    json_free(val);

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

    GV_JsonValue *val = json_parse(json, &err);
    if (val == NULL) {
        FAIL("Failed to parse nested JSON");
        return;
    }

    const char *content = json_get_string_path(val, "choices.0.message.content");
    if (content == NULL || strcmp(content, "Hello, world!") != 0) {
        FAIL("Wrong path access result");
        json_free(val);
        return;
    }

    json_free(val);
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

    GV_JsonValue *val = json_parse(json, &err);
    if (val == NULL) {
        FAIL("Failed to parse OpenAI response");
        return;
    }

    const char *content = json_get_string_path(val, "choices.0.message.content");
    if (content == NULL) {
        FAIL("Failed to get content");
        json_free(val);
        return;
    }

    if (strstr(content, "\"quotes\"") == NULL || strstr(content, "\n") == NULL) {
        FAIL("Escaped characters not properly unescaped");
        json_free(val);
        return;
    }

    GV_JsonValue *usage = json_object_get(val, "usage");
    if (usage == NULL || !json_is_object(usage)) {
        FAIL("Failed to get usage object");
        json_free(val);
        return;
    }

    double total;
    GV_JsonValue *total_tokens = json_object_get(usage, "total_tokens");
    json_get_number(total_tokens, &total);
    if (total != 30.0) {
        FAIL("Wrong total_tokens value");
        json_free(val);
        return;
    }

    json_free(val);
    PASS();
}

void test_parse_facts_response(void) {
    TEST("parse facts JSON response");

    GV_JsonError err;
    const char *json = "{\"facts\": [\"Name is John\", \"Is a Software engineer\", \"Lives in San Francisco\"]}";

    GV_JsonValue *val = json_parse(json, &err);
    if (val == NULL) {
        FAIL("Failed to parse facts JSON");
        return;
    }

    GV_JsonValue *facts = json_object_get(val, "facts");
    if (facts == NULL || !json_is_array(facts)) {
        FAIL("Failed to get facts array");
        json_free(val);
        return;
    }

    size_t len = json_array_length(facts);
    if (len != 3) {
        FAIL("Wrong number of facts");
        json_free(val);
        return;
    }

    const char *fact1 = json_get_string(json_array_get(facts, 0));
    if (fact1 == NULL || strcmp(fact1, "Name is John") != 0) {
        FAIL("Wrong first fact");
        json_free(val);
        return;
    }

    json_free(val);
    PASS();
}

void test_stringify(void) {
    TEST("stringify JSON");

    GV_JsonValue *obj = json_object();
    json_object_set(obj, "name", json_string("John"));
    json_object_set(obj, "age", json_number(30));
    json_object_set(obj, "active", json_bool(true));

    GV_JsonValue *arr = json_array();
    json_array_push(arr, json_number(1));
    json_array_push(arr, json_number(2));
    json_array_push(arr, json_number(3));
    json_object_set(obj, "numbers", arr);

    char *str = json_stringify(obj, false);
    if (str == NULL) {
        FAIL("Failed to stringify");
        json_free(obj);
        return;
    }

    GV_JsonError err;
    GV_JsonValue *parsed = json_parse(str, &err);
    if (parsed == NULL) {
        FAIL("Failed to parse stringified JSON");
        free(str);
        json_free(obj);
        return;
    }

    const char *name = json_get_string(json_object_get(parsed, "name"));
    if (name == NULL || strcmp(name, "John") != 0) {
        FAIL("Name not preserved");
        free(str);
        json_free(obj);
        json_free(parsed);
        return;
    }

    free(str);
    json_free(obj);
    json_free(parsed);
    PASS();
}

void test_copy(void) {
    TEST("deep copy JSON");

    GV_JsonError err;
    const char *json = "{\"nested\": {\"array\": [1, 2, 3]}}";
    GV_JsonValue *original = json_parse(json, &err);
    if (original == NULL) {
        FAIL("Failed to parse");
        return;
    }

    GV_JsonValue *copy = json_copy(original);
    if (copy == NULL) {
        FAIL("Failed to copy");
        json_free(original);
        return;
    }

    json_free(original);

    GV_JsonValue *arr = json_get_path(copy, "nested.array");
    if (arr == NULL || json_array_length(arr) != 3) {
        FAIL("Copy not independent");
        json_free(copy);
        return;
    }

    json_free(copy);
    PASS();
}

void test_error_handling(void) {
    TEST("error handling");

    GV_JsonError err;

    GV_JsonValue *val = json_parse("{invalid}", &err);
    if (val != NULL || err == GV_JSON_OK) {
        FAIL("Should fail on invalid JSON");
        json_free(val);
        return;
    }

    val = json_parse("\"unclosed", &err);
    if (val != NULL || err == GV_JSON_OK) {
        FAIL("Should fail on unclosed string");
        json_free(val);
        return;
    }

    val = json_parse("123 extra", &err);
    if (val != NULL || err == GV_JSON_OK) {
        FAIL("Should fail on trailing content");
        json_free(val);
        return;
    }

    PASS();
}

int main(void) {
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

    return tests_passed == tests_run ? 0 : 1;
}
