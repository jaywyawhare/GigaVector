/**
 * @file gv_json.c
 * @brief Lightweight JSON parser and serializer implementation.
 *
 * A proper recursive descent JSON parser with tokenization,
 * supporting the full JSON specification (RFC 8259).
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <float.h>

#include "gigavector/gv_json.h"

/*  Internal Constants  */

#define MAX_NESTING_DEPTH 128
#define INITIAL_ARRAY_CAPACITY 8
#define INITIAL_OBJECT_CAPACITY 8
#define STRING_BUFFER_INITIAL 256

/*  Parser State  */

typedef struct {
    const char *input;
    const char *pos;
    int depth;
    GV_JsonError error;
} ParserState;

/*  Forward Declarations  */

static GV_JsonValue *parse_value(ParserState *state);
static void skip_whitespace(ParserState *state);

/*  Error Handling  */

const char *gv_json_error_string(GV_JsonError error) {
    switch (error) {
        case GV_JSON_OK:
            return "Success";
        case GV_JSON_ERROR_NULL_INPUT:
            return "Null input";
        case GV_JSON_ERROR_MEMORY:
            return "Memory allocation failed";
        case GV_JSON_ERROR_UNEXPECTED_TOKEN:
            return "Unexpected token";
        case GV_JSON_ERROR_UNEXPECTED_END:
            return "Unexpected end of input";
        case GV_JSON_ERROR_INVALID_STRING:
            return "Invalid string";
        case GV_JSON_ERROR_INVALID_NUMBER:
            return "Invalid number";
        case GV_JSON_ERROR_INVALID_VALUE:
            return "Invalid value";
        case GV_JSON_ERROR_NESTING_TOO_DEEP:
            return "Nesting too deep";
        case GV_JSON_ERROR_KEY_NOT_FOUND:
            return "Key not found";
        case GV_JSON_ERROR_TYPE_MISMATCH:
            return "Type mismatch";
        case GV_JSON_ERROR_INDEX_OUT_OF_BOUNDS:
            return "Index out of bounds";
        default:
            return "Unknown error";
    }
}

/*  Value Creation  */

GV_JsonValue *gv_json_null(void) {
    GV_JsonValue *val = (GV_JsonValue *)calloc(1, sizeof(GV_JsonValue));
    if (val) {
        val->type = GV_JSON_NULL;
    }
    return val;
}

GV_JsonValue *gv_json_bool(bool value) {
    GV_JsonValue *val = (GV_JsonValue *)calloc(1, sizeof(GV_JsonValue));
    if (val) {
        val->type = GV_JSON_BOOL;
        val->data.boolean = value;
    }
    return val;
}

GV_JsonValue *gv_json_number(double value) {
    GV_JsonValue *val = (GV_JsonValue *)calloc(1, sizeof(GV_JsonValue));
    if (val) {
        val->type = GV_JSON_NUMBER;
        val->data.number = value;
    }
    return val;
}

GV_JsonValue *gv_json_string(const char *value) {
    if (value == NULL) {
        return NULL;
    }
    GV_JsonValue *val = (GV_JsonValue *)calloc(1, sizeof(GV_JsonValue));
    if (val) {
        val->type = GV_JSON_STRING;
        val->data.string = strdup(value);
        if (val->data.string == NULL) {
            free(val);
            return NULL;
        }
    }
    return val;
}

GV_JsonValue *gv_json_array(void) {
    GV_JsonValue *val = (GV_JsonValue *)calloc(1, sizeof(GV_JsonValue));
    if (val) {
        val->type = GV_JSON_ARRAY;
        val->data.array.items = NULL;
        val->data.array.count = 0;
        val->data.array.capacity = 0;
    }
    return val;
}

GV_JsonValue *gv_json_object(void) {
    GV_JsonValue *val = (GV_JsonValue *)calloc(1, sizeof(GV_JsonValue));
    if (val) {
        val->type = GV_JSON_OBJECT;
        val->data.object.entries = NULL;
        val->data.object.count = 0;
        val->data.object.capacity = 0;
    }
    return val;
}

/*  Memory Management  */

void gv_json_free(GV_JsonValue *value) {
    if (value == NULL) {
        return;
    }

    switch (value->type) {
        case GV_JSON_STRING:
            free(value->data.string);
            break;
        case GV_JSON_ARRAY:
            for (size_t i = 0; i < value->data.array.count; i++) {
                gv_json_free(value->data.array.items[i]);
            }
            free(value->data.array.items);
            break;
        case GV_JSON_OBJECT:
            for (size_t i = 0; i < value->data.object.count; i++) {
                free(value->data.object.entries[i].key);
                gv_json_free(value->data.object.entries[i].value);
            }
            free(value->data.object.entries);
            break;
        default:
            break;
    }

    free(value);
}

GV_JsonValue *gv_json_copy(const GV_JsonValue *value) {
    if (value == NULL) {
        return NULL;
    }

    GV_JsonValue *copy = NULL;

    switch (value->type) {
        case GV_JSON_NULL:
            copy = gv_json_null();
            break;
        case GV_JSON_BOOL:
            copy = gv_json_bool(value->data.boolean);
            break;
        case GV_JSON_NUMBER:
            copy = gv_json_number(value->data.number);
            break;
        case GV_JSON_STRING:
            copy = gv_json_string(value->data.string);
            break;
        case GV_JSON_ARRAY:
            copy = gv_json_array();
            if (copy) {
                for (size_t i = 0; i < value->data.array.count; i++) {
                    GV_JsonValue *item_copy = gv_json_copy(value->data.array.items[i]);
                    if (item_copy == NULL || gv_json_array_push(copy, item_copy) != GV_JSON_OK) {
                        gv_json_free(item_copy);
                        gv_json_free(copy);
                        return NULL;
                    }
                }
            }
            break;
        case GV_JSON_OBJECT:
            copy = gv_json_object();
            if (copy) {
                for (size_t i = 0; i < value->data.object.count; i++) {
                    GV_JsonValue *val_copy = gv_json_copy(value->data.object.entries[i].value);
                    if (val_copy == NULL ||
                        gv_json_object_set(copy, value->data.object.entries[i].key, val_copy) != GV_JSON_OK) {
                        gv_json_free(val_copy);
                        gv_json_free(copy);
                        return NULL;
                    }
                }
            }
            break;
    }

    return copy;
}

/*  Type Checking  */

bool gv_json_is_null(const GV_JsonValue *value) {
    return value != NULL && value->type == GV_JSON_NULL;
}

bool gv_json_is_bool(const GV_JsonValue *value) {
    return value != NULL && value->type == GV_JSON_BOOL;
}

bool gv_json_is_number(const GV_JsonValue *value) {
    return value != NULL && value->type == GV_JSON_NUMBER;
}

bool gv_json_is_string(const GV_JsonValue *value) {
    return value != NULL && value->type == GV_JSON_STRING;
}

bool gv_json_is_array(const GV_JsonValue *value) {
    return value != NULL && value->type == GV_JSON_ARRAY;
}

bool gv_json_is_object(const GV_JsonValue *value) {
    return value != NULL && value->type == GV_JSON_OBJECT;
}

/*  Value Extraction  */

GV_JsonError gv_json_get_bool(const GV_JsonValue *value, bool *out) {
    if (value == NULL || out == NULL) {
        return GV_JSON_ERROR_NULL_INPUT;
    }
    if (value->type != GV_JSON_BOOL) {
        return GV_JSON_ERROR_TYPE_MISMATCH;
    }
    *out = value->data.boolean;
    return GV_JSON_OK;
}

GV_JsonError gv_json_get_number(const GV_JsonValue *value, double *out) {
    if (value == NULL || out == NULL) {
        return GV_JSON_ERROR_NULL_INPUT;
    }
    if (value->type != GV_JSON_NUMBER) {
        return GV_JSON_ERROR_TYPE_MISMATCH;
    }
    *out = value->data.number;
    return GV_JSON_OK;
}

const char *gv_json_get_string(const GV_JsonValue *value) {
    if (value == NULL || value->type != GV_JSON_STRING) {
        return NULL;
    }
    return value->data.string;
}

/*  Array Operations  */

GV_JsonError gv_json_array_push(GV_JsonValue *array, GV_JsonValue *value) {
    if (array == NULL || value == NULL) {
        return GV_JSON_ERROR_NULL_INPUT;
    }
    if (array->type != GV_JSON_ARRAY) {
        return GV_JSON_ERROR_TYPE_MISMATCH;
    }

    if (array->data.array.count >= array->data.array.capacity) {
        size_t new_capacity = array->data.array.capacity == 0
            ? INITIAL_ARRAY_CAPACITY
            : array->data.array.capacity * 2;
        GV_JsonValue **new_items = (GV_JsonValue **)realloc(
            array->data.array.items,
            new_capacity * sizeof(GV_JsonValue *)
        );
        if (new_items == NULL) {
            return GV_JSON_ERROR_MEMORY;
        }
        array->data.array.items = new_items;
        array->data.array.capacity = new_capacity;
    }

    array->data.array.items[array->data.array.count++] = value;
    return GV_JSON_OK;
}

GV_JsonValue *gv_json_array_get(const GV_JsonValue *array, size_t index) {
    if (array == NULL || array->type != GV_JSON_ARRAY) {
        return NULL;
    }
    if (index >= array->data.array.count) {
        return NULL;
    }
    return array->data.array.items[index];
}

size_t gv_json_array_length(const GV_JsonValue *array) {
    if (array == NULL || array->type != GV_JSON_ARRAY) {
        return 0;
    }
    return array->data.array.count;
}

/*  Object Operations  */

GV_JsonError gv_json_object_set(GV_JsonValue *object, const char *key, GV_JsonValue *value) {
    if (object == NULL || key == NULL || value == NULL) {
        return GV_JSON_ERROR_NULL_INPUT;
    }
    if (object->type != GV_JSON_OBJECT) {
        return GV_JSON_ERROR_TYPE_MISMATCH;
    }

    // Check if key already exists
    for (size_t i = 0; i < object->data.object.count; i++) {
        if (strcmp(object->data.object.entries[i].key, key) == 0) {
            gv_json_free(object->data.object.entries[i].value);
            object->data.object.entries[i].value = value;
            return GV_JSON_OK;
        }
    }

    // Add new entry
    if (object->data.object.count >= object->data.object.capacity) {
        size_t new_capacity = object->data.object.capacity == 0
            ? INITIAL_OBJECT_CAPACITY
            : object->data.object.capacity * 2;
        GV_JsonEntry *new_entries = (GV_JsonEntry *)realloc(
            object->data.object.entries,
            new_capacity * sizeof(GV_JsonEntry)
        );
        if (new_entries == NULL) {
            return GV_JSON_ERROR_MEMORY;
        }
        object->data.object.entries = new_entries;
        object->data.object.capacity = new_capacity;
    }

    char *key_copy = strdup(key);
    if (key_copy == NULL) {
        return GV_JSON_ERROR_MEMORY;
    }

    object->data.object.entries[object->data.object.count].key = key_copy;
    object->data.object.entries[object->data.object.count].value = value;
    object->data.object.count++;

    return GV_JSON_OK;
}

GV_JsonValue *gv_json_object_get(const GV_JsonValue *object, const char *key) {
    if (object == NULL || object->type != GV_JSON_OBJECT || key == NULL) {
        return NULL;
    }

    for (size_t i = 0; i < object->data.object.count; i++) {
        if (strcmp(object->data.object.entries[i].key, key) == 0) {
            return object->data.object.entries[i].value;
        }
    }

    return NULL;
}

bool gv_json_object_has(const GV_JsonValue *object, const char *key) {
    return gv_json_object_get(object, key) != NULL;
}

size_t gv_json_object_length(const GV_JsonValue *object) {
    if (object == NULL || object->type != GV_JSON_OBJECT) {
        return 0;
    }
    return object->data.object.count;
}

/*  Path-based Access  */

GV_JsonValue *gv_json_get_path(const GV_JsonValue *root, const char *path) {
    if (root == NULL || path == NULL) {
        return NULL;
    }

    const GV_JsonValue *current = root;
    char *path_copy = strdup(path);
    if (path_copy == NULL) {
        return NULL;
    }

    char *token = strtok(path_copy, ".");
    while (token != NULL && current != NULL) {
        if (current->type == GV_JSON_OBJECT) {
            current = gv_json_object_get(current, token);
        } else if (current->type == GV_JSON_ARRAY) {
            // Parse numeric index
            char *endptr;
            long index = strtol(token, &endptr, 10);
            if (*endptr != '\0' || index < 0) {
                current = NULL;
            } else {
                current = gv_json_array_get(current, (size_t)index);
            }
        } else {
            current = NULL;
        }
        token = strtok(NULL, ".");
    }

    free(path_copy);
    return (GV_JsonValue *)current;
}

const char *gv_json_get_string_path(const GV_JsonValue *root, const char *path) {
    GV_JsonValue *value = gv_json_get_path(root, path);
    return gv_json_get_string(value);
}

/*  Parser Implementation  */

static void skip_whitespace(ParserState *state) {
    while (*state->pos && isspace((unsigned char)*state->pos)) {
        state->pos++;
    }
}

static int parse_hex4(const char *str, unsigned int *out) {
    unsigned int val = 0;
    for (int i = 0; i < 4; i++) {
        char c = str[i];
        if (c >= '0' && c <= '9') {
            val = (val << 4) | (c - '0');
        } else if (c >= 'a' && c <= 'f') {
            val = (val << 4) | (c - 'a' + 10);
        } else if (c >= 'A' && c <= 'F') {
            val = (val << 4) | (c - 'A' + 10);
        } else {
            return -1;
        }
    }
    *out = val;
    return 0;
}

static char *parse_string_content(ParserState *state) {
    if (*state->pos != '"') {
        state->error = GV_JSON_ERROR_INVALID_STRING;
        return NULL;
    }
    state->pos++;  // Skip opening quote

    // Calculate required size (with room for escapes)
    size_t capacity = STRING_BUFFER_INITIAL;
    size_t length = 0;
    char *buffer = (char *)malloc(capacity);
    if (buffer == NULL) {
        state->error = GV_JSON_ERROR_MEMORY;
        return NULL;
    }

    while (*state->pos && *state->pos != '"') {
        if (length + 6 >= capacity) {  // Room for UTF-8 + null
            capacity *= 2;
            char *new_buffer = (char *)realloc(buffer, capacity);
            if (new_buffer == NULL) {
                free(buffer);
                state->error = GV_JSON_ERROR_MEMORY;
                return NULL;
            }
            buffer = new_buffer;
        }

        if (*state->pos == '\\') {
            state->pos++;
            switch (*state->pos) {
                case '"':  buffer[length++] = '"';  break;
                case '\\': buffer[length++] = '\\'; break;
                case '/':  buffer[length++] = '/';  break;
                case 'b':  buffer[length++] = '\b'; break;
                case 'f':  buffer[length++] = '\f'; break;
                case 'n':  buffer[length++] = '\n'; break;
                case 'r':  buffer[length++] = '\r'; break;
                case 't':  buffer[length++] = '\t'; break;
                case 'u': {
                    state->pos++;
                    unsigned int codepoint;
                    if (parse_hex4(state->pos, &codepoint) != 0) {
                        free(buffer);
                        state->error = GV_JSON_ERROR_INVALID_STRING;
                        return NULL;
                    }
                    state->pos += 3;  // Will be incremented by 1 at end of loop

                    // Handle surrogate pairs
                    if (codepoint >= 0xD800 && codepoint <= 0xDBFF) {
                        state->pos++;
                        if (state->pos[0] == '\\' && state->pos[1] == 'u') {
                            state->pos += 2;
                            unsigned int low;
                            if (parse_hex4(state->pos, &low) != 0 ||
                                low < 0xDC00 || low > 0xDFFF) {
                                free(buffer);
                                state->error = GV_JSON_ERROR_INVALID_STRING;
                                return NULL;
                            }
                            state->pos += 3;
                            codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00);
                        }
                    }

                    // Encode as UTF-8
                    if (codepoint < 0x80) {
                        buffer[length++] = (char)codepoint;
                    } else if (codepoint < 0x800) {
                        buffer[length++] = (char)(0xC0 | (codepoint >> 6));
                        buffer[length++] = (char)(0x80 | (codepoint & 0x3F));
                    } else if (codepoint < 0x10000) {
                        buffer[length++] = (char)(0xE0 | (codepoint >> 12));
                        buffer[length++] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
                        buffer[length++] = (char)(0x80 | (codepoint & 0x3F));
                    } else {
                        buffer[length++] = (char)(0xF0 | (codepoint >> 18));
                        buffer[length++] = (char)(0x80 | ((codepoint >> 12) & 0x3F));
                        buffer[length++] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
                        buffer[length++] = (char)(0x80 | (codepoint & 0x3F));
                    }
                    break;
                }
                default:
                    free(buffer);
                    state->error = GV_JSON_ERROR_INVALID_STRING;
                    return NULL;
            }
        } else if ((unsigned char)*state->pos < 0x20) {
            // Control characters not allowed in strings
            free(buffer);
            state->error = GV_JSON_ERROR_INVALID_STRING;
            return NULL;
        } else {
            buffer[length++] = *state->pos;
        }
        state->pos++;
    }

    if (*state->pos != '"') {
        free(buffer);
        state->error = GV_JSON_ERROR_UNEXPECTED_END;
        return NULL;
    }
    state->pos++;  // Skip closing quote

    buffer[length] = '\0';
    return buffer;
}

static GV_JsonValue *parse_string(ParserState *state) {
    char *str = parse_string_content(state);
    if (str == NULL) {
        return NULL;
    }

    GV_JsonValue *value = (GV_JsonValue *)calloc(1, sizeof(GV_JsonValue));
    if (value == NULL) {
        free(str);
        state->error = GV_JSON_ERROR_MEMORY;
        return NULL;
    }

    value->type = GV_JSON_STRING;
    value->data.string = str;
    return value;
}

static GV_JsonValue *parse_number(ParserState *state) {
    const char *start = state->pos;

    // Optional minus
    if (*state->pos == '-') {
        state->pos++;
    }

    // Integer part
    if (*state->pos == '0') {
        state->pos++;
    } else if (*state->pos >= '1' && *state->pos <= '9') {
        while (*state->pos >= '0' && *state->pos <= '9') {
            state->pos++;
        }
    } else {
        state->error = GV_JSON_ERROR_INVALID_NUMBER;
        return NULL;
    }

    // Fractional part
    if (*state->pos == '.') {
        state->pos++;
        if (*state->pos < '0' || *state->pos > '9') {
            state->error = GV_JSON_ERROR_INVALID_NUMBER;
            return NULL;
        }
        while (*state->pos >= '0' && *state->pos <= '9') {
            state->pos++;
        }
    }

    // Exponent part
    if (*state->pos == 'e' || *state->pos == 'E') {
        state->pos++;
        if (*state->pos == '+' || *state->pos == '-') {
            state->pos++;
        }
        if (*state->pos < '0' || *state->pos > '9') {
            state->error = GV_JSON_ERROR_INVALID_NUMBER;
            return NULL;
        }
        while (*state->pos >= '0' && *state->pos <= '9') {
            state->pos++;
        }
    }

    // Parse the number
    char *endptr;
    double num = strtod(start, &endptr);
    if (endptr != state->pos) {
        state->error = GV_JSON_ERROR_INVALID_NUMBER;
        return NULL;
    }

    GV_JsonValue *value = (GV_JsonValue *)calloc(1, sizeof(GV_JsonValue));
    if (value == NULL) {
        state->error = GV_JSON_ERROR_MEMORY;
        return NULL;
    }

    value->type = GV_JSON_NUMBER;
    value->data.number = num;
    return value;
}

static GV_JsonValue *parse_array(ParserState *state) {
    if (*state->pos != '[') {
        state->error = GV_JSON_ERROR_UNEXPECTED_TOKEN;
        return NULL;
    }
    state->pos++;

    state->depth++;
    if (state->depth > MAX_NESTING_DEPTH) {
        state->error = GV_JSON_ERROR_NESTING_TOO_DEEP;
        state->depth--;
        return NULL;
    }

    GV_JsonValue *array = gv_json_array();
    if (array == NULL) {
        state->error = GV_JSON_ERROR_MEMORY;
        state->depth--;
        return NULL;
    }

    skip_whitespace(state);

    if (*state->pos == ']') {
        state->pos++;
        state->depth--;
        return array;
    }

    while (1) {
        skip_whitespace(state);

        GV_JsonValue *element = parse_value(state);
        if (element == NULL) {
            gv_json_free(array);
            state->depth--;
            return NULL;
        }

        if (gv_json_array_push(array, element) != GV_JSON_OK) {
            gv_json_free(element);
            gv_json_free(array);
            state->error = GV_JSON_ERROR_MEMORY;
            state->depth--;
            return NULL;
        }

        skip_whitespace(state);

        if (*state->pos == ']') {
            state->pos++;
            state->depth--;
            return array;
        }

        if (*state->pos != ',') {
            gv_json_free(array);
            state->error = GV_JSON_ERROR_UNEXPECTED_TOKEN;
            state->depth--;
            return NULL;
        }
        state->pos++;
    }
}

static GV_JsonValue *parse_object(ParserState *state) {
    if (*state->pos != '{') {
        state->error = GV_JSON_ERROR_UNEXPECTED_TOKEN;
        return NULL;
    }
    state->pos++;

    state->depth++;
    if (state->depth > MAX_NESTING_DEPTH) {
        state->error = GV_JSON_ERROR_NESTING_TOO_DEEP;
        state->depth--;
        return NULL;
    }

    GV_JsonValue *object = gv_json_object();
    if (object == NULL) {
        state->error = GV_JSON_ERROR_MEMORY;
        state->depth--;
        return NULL;
    }

    skip_whitespace(state);

    if (*state->pos == '}') {
        state->pos++;
        state->depth--;
        return object;
    }

    while (1) {
        skip_whitespace(state);

        if (*state->pos != '"') {
            gv_json_free(object);
            state->error = GV_JSON_ERROR_UNEXPECTED_TOKEN;
            state->depth--;
            return NULL;
        }

        char *key = parse_string_content(state);
        if (key == NULL) {
            gv_json_free(object);
            state->depth--;
            return NULL;
        }

        skip_whitespace(state);

        if (*state->pos != ':') {
            free(key);
            gv_json_free(object);
            state->error = GV_JSON_ERROR_UNEXPECTED_TOKEN;
            state->depth--;
            return NULL;
        }
        state->pos++;

        skip_whitespace(state);

        GV_JsonValue *value = parse_value(state);
        if (value == NULL) {
            free(key);
            gv_json_free(object);
            state->depth--;
            return NULL;
        }

        GV_JsonError err = gv_json_object_set(object, key, value);
        free(key);
        if (err != GV_JSON_OK) {
            gv_json_free(value);
            gv_json_free(object);
            state->error = err;
            state->depth--;
            return NULL;
        }

        skip_whitespace(state);

        if (*state->pos == '}') {
            state->pos++;
            state->depth--;
            return object;
        }

        if (*state->pos != ',') {
            gv_json_free(object);
            state->error = GV_JSON_ERROR_UNEXPECTED_TOKEN;
            state->depth--;
            return NULL;
        }
        state->pos++;
    }
}

static GV_JsonValue *parse_value(ParserState *state) {
    skip_whitespace(state);

    if (*state->pos == '\0') {
        state->error = GV_JSON_ERROR_UNEXPECTED_END;
        return NULL;
    }

    switch (*state->pos) {
        case '"':
            return parse_string(state);

        case '{':
            return parse_object(state);

        case '[':
            return parse_array(state);

        case 't':
            if (strncmp(state->pos, "true", 4) == 0) {
                state->pos += 4;
                return gv_json_bool(true);
            }
            state->error = GV_JSON_ERROR_INVALID_VALUE;
            return NULL;

        case 'f':
            if (strncmp(state->pos, "false", 5) == 0) {
                state->pos += 5;
                return gv_json_bool(false);
            }
            state->error = GV_JSON_ERROR_INVALID_VALUE;
            return NULL;

        case 'n':
            if (strncmp(state->pos, "null", 4) == 0) {
                state->pos += 4;
                return gv_json_null();
            }
            state->error = GV_JSON_ERROR_INVALID_VALUE;
            return NULL;

        case '-':
        case '0': case '1': case '2': case '3': case '4':
        case '5': case '6': case '7': case '8': case '9':
            return parse_number(state);

        default:
            state->error = GV_JSON_ERROR_UNEXPECTED_TOKEN;
            return NULL;
    }
}

GV_JsonValue *gv_json_parse(const char *json_str, GV_JsonError *error) {
    if (json_str == NULL) {
        if (error) *error = GV_JSON_ERROR_NULL_INPUT;
        return NULL;
    }

    ParserState state = {
        .input = json_str,
        .pos = json_str,
        .depth = 0,
        .error = GV_JSON_OK
    };

    GV_JsonValue *result = parse_value(&state);

    if (result != NULL) {
        skip_whitespace(&state);
        if (*state.pos != '\0') {
            // Extra content after valid JSON
            gv_json_free(result);
            result = NULL;
            state.error = GV_JSON_ERROR_UNEXPECTED_TOKEN;
        }
    }

    if (error) {
        *error = state.error;
    }

    return result;
}

/*  Serialization Implementation  */

typedef struct {
    char *buffer;
    size_t length;
    size_t capacity;
    bool pretty;
    int indent;
} StringifyState;

static int stringify_grow(StringifyState *state, size_t needed) {
    if (state->length + needed >= state->capacity) {
        size_t new_capacity = state->capacity * 2;
        if (new_capacity < state->length + needed + 1) {
            new_capacity = state->length + needed + 1;
        }
        char *new_buffer = (char *)realloc(state->buffer, new_capacity);
        if (new_buffer == NULL) {
            return -1;
        }
        state->buffer = new_buffer;
        state->capacity = new_capacity;
    }
    return 0;
}

static int stringify_append(StringifyState *state, const char *str) {
    size_t len = strlen(str);
    if (stringify_grow(state, len) != 0) {
        return -1;
    }
    memcpy(state->buffer + state->length, str, len);
    state->length += len;
    state->buffer[state->length] = '\0';
    return 0;
}

static int stringify_append_char(StringifyState *state, char c) {
    if (stringify_grow(state, 1) != 0) {
        return -1;
    }
    state->buffer[state->length++] = c;
    state->buffer[state->length] = '\0';
    return 0;
}

static int stringify_indent(StringifyState *state) {
    if (!state->pretty) {
        return 0;
    }
    for (int i = 0; i < state->indent; i++) {
        if (stringify_append(state, "  ") != 0) {
            return -1;
        }
    }
    return 0;
}

static int stringify_string(StringifyState *state, const char *str) {
    if (stringify_append_char(state, '"') != 0) {
        return -1;
    }

    for (const char *p = str; *p; p++) {
        char escape_buf[8];
        const char *to_append = NULL;

        switch (*p) {
            case '"':  to_append = "\\\""; break;
            case '\\': to_append = "\\\\"; break;
            case '\b': to_append = "\\b";  break;
            case '\f': to_append = "\\f";  break;
            case '\n': to_append = "\\n";  break;
            case '\r': to_append = "\\r";  break;
            case '\t': to_append = "\\t";  break;
            default:
                if ((unsigned char)*p < 0x20) {
                    snprintf(escape_buf, sizeof(escape_buf), "\\u%04x", (unsigned char)*p);
                    to_append = escape_buf;
                } else {
                    if (stringify_append_char(state, *p) != 0) {
                        return -1;
                    }
                    continue;
                }
        }

        if (to_append && stringify_append(state, to_append) != 0) {
            return -1;
        }
    }

    return stringify_append_char(state, '"');
}

static int stringify_value(StringifyState *state, const GV_JsonValue *value);

static int stringify_array(StringifyState *state, const GV_JsonValue *value) {
    if (stringify_append_char(state, '[') != 0) {
        return -1;
    }

    if (value->data.array.count > 0) {
        if (state->pretty) {
            if (stringify_append_char(state, '\n') != 0) {
                return -1;
            }
            state->indent++;
        }

        for (size_t i = 0; i < value->data.array.count; i++) {
            if (i > 0) {
                if (stringify_append_char(state, ',') != 0) {
                    return -1;
                }
                if (state->pretty && stringify_append_char(state, '\n') != 0) {
                    return -1;
                }
            }
            if (stringify_indent(state) != 0) {
                return -1;
            }
            if (stringify_value(state, value->data.array.items[i]) != 0) {
                return -1;
            }
        }

        if (state->pretty) {
            state->indent--;
            if (stringify_append_char(state, '\n') != 0) {
                return -1;
            }
            if (stringify_indent(state) != 0) {
                return -1;
            }
        }
    }

    return stringify_append_char(state, ']');
}

static int stringify_object(StringifyState *state, const GV_JsonValue *value) {
    if (stringify_append_char(state, '{') != 0) {
        return -1;
    }

    if (value->data.object.count > 0) {
        if (state->pretty) {
            if (stringify_append_char(state, '\n') != 0) {
                return -1;
            }
            state->indent++;
        }

        for (size_t i = 0; i < value->data.object.count; i++) {
            if (i > 0) {
                if (stringify_append_char(state, ',') != 0) {
                    return -1;
                }
                if (state->pretty && stringify_append_char(state, '\n') != 0) {
                    return -1;
                }
            }
            if (stringify_indent(state) != 0) {
                return -1;
            }
            if (stringify_string(state, value->data.object.entries[i].key) != 0) {
                return -1;
            }
            if (stringify_append_char(state, ':') != 0) {
                return -1;
            }
            if (state->pretty && stringify_append_char(state, ' ') != 0) {
                return -1;
            }
            if (stringify_value(state, value->data.object.entries[i].value) != 0) {
                return -1;
            }
        }

        if (state->pretty) {
            state->indent--;
            if (stringify_append_char(state, '\n') != 0) {
                return -1;
            }
            if (stringify_indent(state) != 0) {
                return -1;
            }
        }
    }

    return stringify_append_char(state, '}');
}

static int stringify_value(StringifyState *state, const GV_JsonValue *value) {
    if (value == NULL) {
        return stringify_append(state, "null");
    }

    switch (value->type) {
        case GV_JSON_NULL:
            return stringify_append(state, "null");

        case GV_JSON_BOOL:
            return stringify_append(state, value->data.boolean ? "true" : "false");

        case GV_JSON_NUMBER: {
            char num_buf[64];
            double num = value->data.number;

            // Handle special cases
            if (isnan(num) || isinf(num)) {
                return stringify_append(state, "null");
            }

            // Check if it's an integer
            if (num == floor(num) && fabs(num) < 9007199254740992.0) {
                snprintf(num_buf, sizeof(num_buf), "%.0f", num);
            } else {
                snprintf(num_buf, sizeof(num_buf), "%.17g", num);
            }
            return stringify_append(state, num_buf);
        }

        case GV_JSON_STRING:
            return stringify_string(state, value->data.string);

        case GV_JSON_ARRAY:
            return stringify_array(state, value);

        case GV_JSON_OBJECT:
            return stringify_object(state, value);

        default:
            return -1;
    }
}

char *gv_json_stringify(const GV_JsonValue *value, bool pretty) {
    StringifyState state = {
        .buffer = (char *)malloc(256),
        .length = 0,
        .capacity = 256,
        .pretty = pretty,
        .indent = 0
    };

    if (state.buffer == NULL) {
        return NULL;
    }
    state.buffer[0] = '\0';

    if (stringify_value(&state, value) != 0) {
        free(state.buffer);
        return NULL;
    }

    return state.buffer;
}
