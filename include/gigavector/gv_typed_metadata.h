#ifndef GIGAVECTOR_GV_TYPED_METADATA_H
#define GIGAVECTOR_GV_TYPED_METADATA_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_typed_metadata.h
 * @brief Typed metadata support for GigaVector.
 *
 * This module provides typed metadata values beyond simple string key-value pairs.
 * Supported types include: null, string, int64, float64, bool, arrays, and objects.
 */

/**
 * @brief Metadata value types.
 */
typedef enum {
    GV_META_TYPE_NULL = 0,       /**< Null value. */
    GV_META_TYPE_STRING = 1,     /**< String value. */
    GV_META_TYPE_INT64 = 2,      /**< 64-bit signed integer. */
    GV_META_TYPE_FLOAT64 = 3,    /**< 64-bit floating point. */
    GV_META_TYPE_BOOL = 4,       /**< Boolean value. */
    GV_META_TYPE_ARRAY = 5,      /**< Homogeneous array. */
    GV_META_TYPE_OBJECT = 6      /**< Nested object (map). */
} GV_MetaType;

/**
 * @brief Forward declaration for typed value.
 */
typedef struct GV_TypedValue GV_TypedValue;

/**
 * @brief Typed metadata value.
 */
struct GV_TypedValue {
    GV_MetaType type;            /**< Value type. */
    union {
        char *string_val;        /**< String value (owned, must be freed). */
        int64_t int_val;         /**< Integer value. */
        double float_val;        /**< Float value. */
        bool bool_val;           /**< Boolean value. */
        struct {
            GV_TypedValue *items;     /**< Array items. */
            size_t count;             /**< Number of items. */
            size_t capacity;          /**< Allocated capacity. */
            GV_MetaType element_type; /**< Type of elements (for type checking). */
        } array_val;             /**< Array value. */
        struct {
            char **keys;              /**< Object keys (owned). */
            GV_TypedValue *values;    /**< Object values. */
            size_t count;             /**< Number of key-value pairs. */
            size_t capacity;          /**< Allocated capacity. */
        } object_val;            /**< Object value. */
    } data;
};

/**
 * @brief Typed metadata entry (key-value pair).
 */
typedef struct GV_TypedMetadata {
    char *key;                        /**< Metadata key (owned). */
    GV_TypedValue value;              /**< Metadata value. */
    struct GV_TypedMetadata *next;    /**< Next entry in linked list. */
} GV_TypedMetadata;

/* Value Creation Functions */

/**
 * @brief Create a null value.
 * @return Typed null value.
 */
GV_TypedValue gv_typed_null(void);

/**
 * @brief Create a string value.
 * @param val String to copy.
 * @return Typed string value.
 */
GV_TypedValue gv_typed_string(const char *val);

/**
 * @brief Create an integer value.
 * @param val Integer value.
 * @return Typed integer value.
 */
GV_TypedValue gv_typed_int(int64_t val);

/**
 * @brief Create a float value.
 * @param val Float value.
 * @return Typed float value.
 */
GV_TypedValue gv_typed_float(double val);

/**
 * @brief Create a boolean value.
 * @param val Boolean value.
 * @return Typed boolean value.
 */
GV_TypedValue gv_typed_bool(bool val);

/**
 * @brief Create an empty array.
 * @param element_type Expected type of elements (for type checking).
 * @return Typed array value.
 */
GV_TypedValue gv_typed_array(GV_MetaType element_type);

/**
 * @brief Create an empty object.
 * @return Typed object value.
 */
GV_TypedValue gv_typed_object(void);

/* Array Operations */

/**
 * @brief Append a value to an array.
 * @param array Array value (must be GV_META_TYPE_ARRAY).
 * @param item Value to append (will be copied).
 * @return 0 on success, -1 on error.
 */
int gv_typed_array_push(GV_TypedValue *array, const GV_TypedValue *item);

/**
 * @brief Get array element at index.
 * @param array Array value.
 * @param index Element index.
 * @return Pointer to element, or NULL if out of bounds.
 */
GV_TypedValue *gv_typed_array_get(const GV_TypedValue *array, size_t index);

/**
 * @brief Get array length.
 * @param array Array value.
 * @return Number of elements.
 */
size_t gv_typed_array_length(const GV_TypedValue *array);

/* Object Operations */

/**
 * @brief Set a key-value pair in an object.
 * @param object Object value (must be GV_META_TYPE_OBJECT).
 * @param key Key string (will be copied).
 * @param value Value to set (will be copied).
 * @return 0 on success, -1 on error.
 */
int gv_typed_object_set(GV_TypedValue *object, const char *key, const GV_TypedValue *value);

/**
 * @brief Get value by key from an object.
 * @param object Object value.
 * @param key Key to look up.
 * @return Pointer to value, or NULL if not found.
 */
GV_TypedValue *gv_typed_object_get(const GV_TypedValue *object, const char *key);

/**
 * @brief Check if a key exists in an object.
 * @param object Object value.
 * @param key Key to check.
 * @return true if key exists, false otherwise.
 */
bool gv_typed_object_has(const GV_TypedValue *object, const char *key);

/**
 * @brief Get number of keys in an object.
 * @param object Object value.
 * @return Number of key-value pairs.
 */
size_t gv_typed_object_length(const GV_TypedValue *object);

/* Value Extraction */

/**
 * @brief Get string value.
 * @param value Typed value.
 * @return String pointer (internal), or NULL if not a string.
 */
const char *gv_typed_get_string(const GV_TypedValue *value);

/**
 * @brief Get integer value.
 * @param value Typed value.
 * @param out Output integer.
 * @return 0 on success, -1 if not an integer.
 */
int gv_typed_get_int(const GV_TypedValue *value, int64_t *out);

/**
 * @brief Get float value.
 * @param value Typed value.
 * @param out Output float.
 * @return 0 on success, -1 if not a float.
 */
int gv_typed_get_float(const GV_TypedValue *value, double *out);

/**
 * @brief Get boolean value.
 * @param value Typed value.
 * @param out Output boolean.
 * @return 0 on success, -1 if not a boolean.
 */
int gv_typed_get_bool(const GV_TypedValue *value, bool *out);

/* Comparison Operations */

/**
 * @brief Compare two typed values.
 * @param a First value.
 * @param b Second value.
 * @return Negative if a < b, 0 if a == b, positive if a > b.
 *         Returns 0 if types are incompatible.
 */
int gv_typed_compare(const GV_TypedValue *a, const GV_TypedValue *b);

/**
 * @brief Check if two typed values are equal.
 * @param a First value.
 * @param b Second value.
 * @return true if equal, false otherwise.
 */
bool gv_typed_equals(const GV_TypedValue *a, const GV_TypedValue *b);

/**
 * @brief Check if a numeric value is in a range.
 * @param value Value to check (must be numeric).
 * @param min Minimum value (inclusive).
 * @param max Maximum value (inclusive).
 * @return true if in range, false otherwise.
 */
bool gv_typed_in_range(const GV_TypedValue *value, double min, double max);

/**
 * @brief Check if a string contains a substring.
 * @param value String value.
 * @param substr Substring to search for.
 * @return true if contains, false otherwise.
 */
bool gv_typed_string_contains(const GV_TypedValue *value, const char *substr);

/**
 * @brief Check if a string starts with a prefix.
 * @param value String value.
 * @param prefix Prefix to check.
 * @return true if starts with prefix, false otherwise.
 */
bool gv_typed_string_starts_with(const GV_TypedValue *value, const char *prefix);

/**
 * @brief Check if an array contains a value.
 * @param array Array value.
 * @param item Value to search for.
 * @return true if array contains item, false otherwise.
 */
bool gv_typed_array_contains(const GV_TypedValue *array, const GV_TypedValue *item);

/* Memory Management */

/**
 * @brief Free a typed value and its contents.
 * @param value Value to free (safe to call with stack-allocated values).
 */
void gv_typed_value_free(GV_TypedValue *value);

/**
 * @brief Deep copy a typed value.
 * @param src Source value.
 * @return Copied value.
 */
GV_TypedValue gv_typed_value_copy(const GV_TypedValue *src);

/**
 * @brief Free a typed metadata entry and its contents.
 * @param meta Metadata entry to free.
 */
void gv_typed_metadata_free(GV_TypedMetadata *meta);

/**
 * @brief Free a linked list of typed metadata entries.
 * @param head Head of the linked list.
 */
void gv_typed_metadata_free_all(GV_TypedMetadata *head);

/* Serialization */

/**
 * @brief Serialize a typed value to binary format.
 * @param value Value to serialize.
 * @param buf Output buffer (caller must free).
 * @param len Output length.
 * @return 0 on success, -1 on error.
 */
int gv_typed_value_serialize(const GV_TypedValue *value, uint8_t **buf, size_t *len);

/**
 * @brief Deserialize a typed value from binary format.
 * @param buf Input buffer.
 * @param len Input length.
 * @param out Output value.
 * @return Number of bytes consumed, or -1 on error.
 */
int gv_typed_value_deserialize(const uint8_t *buf, size_t len, GV_TypedValue *out);

/**
 * @brief Serialize typed metadata to binary format.
 * @param meta Metadata entry to serialize.
 * @param buf Output buffer (caller must free).
 * @param len Output length.
 * @return 0 on success, -1 on error.
 */
int gv_typed_metadata_serialize(const GV_TypedMetadata *meta, uint8_t **buf, size_t *len);

/**
 * @brief Deserialize typed metadata from binary format.
 * @param buf Input buffer.
 * @param len Input length.
 * @return Deserialized metadata entry, or NULL on error.
 */
GV_TypedMetadata *gv_typed_metadata_deserialize(const uint8_t *buf, size_t len);

/* Conversion Functions */

/**
 * @brief Convert a typed value to string representation.
 * @param value Value to convert.
 * @return String representation (caller must free), or NULL on error.
 */
char *gv_typed_to_string(const GV_TypedValue *value);

/**
 * @brief Convert legacy string metadata to typed metadata.
 * @param key Metadata key.
 * @param value Metadata string value.
 * @return Typed metadata entry, or NULL on error.
 */
GV_TypedMetadata *gv_typed_from_string_metadata(const char *key, const char *value);

/**
 * @brief Get type name as string.
 * @param type Metadata type.
 * @return Type name string.
 */
const char *gv_typed_type_name(GV_MetaType type);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_TYPED_METADATA_H */
