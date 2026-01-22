#ifndef GIGAVECTOR_GV_JSON_H
#define GIGAVECTOR_GV_JSON_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_json.h
 * @brief Lightweight JSON parser and serializer for GigaVector.
 *
 * This module provides a proper JSON parser with tokenization,
 * supporting all JSON types: objects, arrays, strings, numbers, booleans, and null.
 */

/**
 * @brief JSON value types.
 */
typedef enum {
    GV_JSON_NULL = 0,
    GV_JSON_BOOL,
    GV_JSON_NUMBER,
    GV_JSON_STRING,
    GV_JSON_ARRAY,
    GV_JSON_OBJECT
} GV_JsonType;

/**
 * @brief JSON error codes.
 */
typedef enum {
    GV_JSON_OK = 0,
    GV_JSON_ERROR_NULL_INPUT = -1,
    GV_JSON_ERROR_MEMORY = -2,
    GV_JSON_ERROR_UNEXPECTED_TOKEN = -3,
    GV_JSON_ERROR_UNEXPECTED_END = -4,
    GV_JSON_ERROR_INVALID_STRING = -5,
    GV_JSON_ERROR_INVALID_NUMBER = -6,
    GV_JSON_ERROR_INVALID_VALUE = -7,
    GV_JSON_ERROR_NESTING_TOO_DEEP = -8,
    GV_JSON_ERROR_KEY_NOT_FOUND = -9,
    GV_JSON_ERROR_TYPE_MISMATCH = -10,
    GV_JSON_ERROR_INDEX_OUT_OF_BOUNDS = -11
} GV_JsonError;

/**
 * @brief Forward declaration for JSON value.
 */
typedef struct GV_JsonValue GV_JsonValue;

/**
 * @brief JSON object entry (key-value pair).
 */
typedef struct {
    char *key;
    GV_JsonValue *value;
} GV_JsonEntry;

/**
 * @brief JSON value structure.
 */
struct GV_JsonValue {
    GV_JsonType type;
    union {
        bool boolean;
        double number;
        char *string;
        struct {
            GV_JsonValue **items;
            size_t count;
            size_t capacity;
        } array;
        struct {
            GV_JsonEntry *entries;
            size_t count;
            size_t capacity;
        } object;
    } data;
};

/* ============================================================================
 * Parsing Functions
 * ============================================================================ */

/**
 * @brief Parse a JSON string into a JSON value tree.
 *
 * @param json_str The JSON string to parse.
 * @param error Output error code (can be NULL).
 * @return Parsed JSON value, or NULL on error.
 */
GV_JsonValue *gv_json_parse(const char *json_str, GV_JsonError *error);

/**
 * @brief Get error description string.
 *
 * @param error Error code.
 * @return Human-readable error description.
 */
const char *gv_json_error_string(GV_JsonError error);

/* ============================================================================
 * Serialization Functions
 * ============================================================================ */

/**
 * @brief Serialize a JSON value to a string.
 *
 * @param value JSON value to serialize.
 * @param pretty If true, format with indentation.
 * @return Serialized JSON string (caller must free), or NULL on error.
 */
char *gv_json_stringify(const GV_JsonValue *value, bool pretty);

/* ============================================================================
 * Value Creation Functions
 * ============================================================================ */

/**
 * @brief Create a null JSON value.
 * @return New JSON null value.
 */
GV_JsonValue *gv_json_null(void);

/**
 * @brief Create a boolean JSON value.
 * @param value Boolean value.
 * @return New JSON boolean value.
 */
GV_JsonValue *gv_json_bool(bool value);

/**
 * @brief Create a number JSON value.
 * @param value Numeric value.
 * @return New JSON number value.
 */
GV_JsonValue *gv_json_number(double value);

/**
 * @brief Create a string JSON value.
 * @param value String value (will be copied).
 * @return New JSON string value.
 */
GV_JsonValue *gv_json_string(const char *value);

/**
 * @brief Create an empty JSON array.
 * @return New JSON array value.
 */
GV_JsonValue *gv_json_array(void);

/**
 * @brief Create an empty JSON object.
 * @return New JSON object value.
 */
GV_JsonValue *gv_json_object(void);

/* ============================================================================
 * Array Operations
 * ============================================================================ */

/**
 * @brief Append a value to a JSON array.
 *
 * @param array JSON array value.
 * @param value Value to append (takes ownership).
 * @return GV_JSON_OK on success, error code on failure.
 */
GV_JsonError gv_json_array_push(GV_JsonValue *array, GV_JsonValue *value);

/**
 * @brief Get array element at index.
 *
 * @param array JSON array value.
 * @param index Array index.
 * @return Pointer to element, or NULL if out of bounds.
 */
GV_JsonValue *gv_json_array_get(const GV_JsonValue *array, size_t index);

/**
 * @brief Get array length.
 *
 * @param array JSON array value.
 * @return Number of elements in array.
 */
size_t gv_json_array_length(const GV_JsonValue *array);

/* ============================================================================
 * Object Operations
 * ============================================================================ */

/**
 * @brief Set a key-value pair in a JSON object.
 *
 * @param object JSON object value.
 * @param key Key string (will be copied).
 * @param value Value to set (takes ownership).
 * @return GV_JSON_OK on success, error code on failure.
 */
GV_JsonError gv_json_object_set(GV_JsonValue *object, const char *key, GV_JsonValue *value);

/**
 * @brief Get value by key from a JSON object.
 *
 * @param object JSON object value.
 * @param key Key to look up.
 * @return Pointer to value, or NULL if not found.
 */
GV_JsonValue *gv_json_object_get(const GV_JsonValue *object, const char *key);

/**
 * @brief Check if a key exists in a JSON object.
 *
 * @param object JSON object value.
 * @param key Key to check.
 * @return true if key exists, false otherwise.
 */
bool gv_json_object_has(const GV_JsonValue *object, const char *key);

/**
 * @brief Get number of keys in a JSON object.
 *
 * @param object JSON object value.
 * @return Number of key-value pairs.
 */
size_t gv_json_object_length(const GV_JsonValue *object);

/* ============================================================================
 * Type Checking and Value Extraction
 * ============================================================================ */

/**
 * @brief Check if value is null.
 */
bool gv_json_is_null(const GV_JsonValue *value);

/**
 * @brief Check if value is a boolean.
 */
bool gv_json_is_bool(const GV_JsonValue *value);

/**
 * @brief Check if value is a number.
 */
bool gv_json_is_number(const GV_JsonValue *value);

/**
 * @brief Check if value is a string.
 */
bool gv_json_is_string(const GV_JsonValue *value);

/**
 * @brief Check if value is an array.
 */
bool gv_json_is_array(const GV_JsonValue *value);

/**
 * @brief Check if value is an object.
 */
bool gv_json_is_object(const GV_JsonValue *value);

/**
 * @brief Get boolean value.
 *
 * @param value JSON value.
 * @param out Output boolean.
 * @return GV_JSON_OK on success, GV_JSON_ERROR_TYPE_MISMATCH if not a boolean.
 */
GV_JsonError gv_json_get_bool(const GV_JsonValue *value, bool *out);

/**
 * @brief Get number value.
 *
 * @param value JSON value.
 * @param out Output number.
 * @return GV_JSON_OK on success, GV_JSON_ERROR_TYPE_MISMATCH if not a number.
 */
GV_JsonError gv_json_get_number(const GV_JsonValue *value, double *out);

/**
 * @brief Get string value (pointer to internal string).
 *
 * @param value JSON value.
 * @return String pointer, or NULL if not a string.
 */
const char *gv_json_get_string(const GV_JsonValue *value);

/* ============================================================================
 * Path-based Access (convenience)
 * ============================================================================ */

/**
 * @brief Get value at a path (e.g., "choices.0.message.content").
 *
 * Supports both object keys and array indices (numeric).
 *
 * @param root Root JSON value.
 * @param path Dot-separated path.
 * @return Pointer to value at path, or NULL if not found.
 */
GV_JsonValue *gv_json_get_path(const GV_JsonValue *root, const char *path);

/**
 * @brief Get string at a path (convenience function).
 *
 * @param root Root JSON value.
 * @param path Dot-separated path.
 * @return String value, or NULL if not found or not a string.
 */
const char *gv_json_get_string_path(const GV_JsonValue *root, const char *path);

/* ============================================================================
 * Memory Management
 * ============================================================================ */

/**
 * @brief Free a JSON value and all its children.
 *
 * @param value JSON value to free.
 */
void gv_json_free(GV_JsonValue *value);

/**
 * @brief Deep copy a JSON value.
 *
 * @param value JSON value to copy.
 * @return New copy, or NULL on error.
 */
GV_JsonValue *gv_json_copy(const GV_JsonValue *value);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_JSON_H */
