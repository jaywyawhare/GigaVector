#ifndef GIGAVECTOR_GV_JSON_INDEX_H
#define GIGAVECTOR_GV_JSON_INDEX_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_json_index.h
 * @brief JSON path indexing for fast filtered vector search.
 *
 * Pre-index specific JSON paths within nested metadata fields so that
 * filtered searches can look up a sorted index instead of scanning all
 * metadata.  Supports dot-notation paths ("address.city") and array
 * access ("tags[0]").  Each registered path maintains a sorted array of
 * (value, vector_index) pairs enabling O(log n) lookups and range queries.
 *
 * Thread-safe: all operations are guarded by a pthread read-write lock.
 * A maximum of 64 indexed paths is supported per index instance.
 */

/* Constants */

#define GV_JSON_INDEX_MAX_PATHS 64

/* Types */

/**
 * @brief Supported value types for an indexed JSON path.
 */
typedef enum {
    GV_JP_STRING = 0,
    GV_JP_INT,
    GV_JP_FLOAT,
    GV_JP_BOOL
} GV_JSONPathType;

/**
 * @brief Configuration for a single indexed JSON path.
 */
typedef struct {
    const char     *path;   /**< Dot-notation path, e.g. "address.city" or "tags[0]" */
    GV_JSONPathType type;   /**< Expected value type at this path */
} GV_JSONPathConfig;

/**
 * @brief Opaque handle for a JSON path index.
 */
typedef struct GV_JSONPathIndex GV_JSONPathIndex;

/**
 * @brief Result entry returned by lookup helpers (informational).
 */
typedef struct {
    size_t vector_index;
    union {
        const char *str_val;
        int64_t     int_val;
        double      float_val;
        bool        bool_val;
    } value;
} GV_JSONPathResult;

/* Lifecycle */

/**
 * @brief Create an empty JSON path index.
 *
 * @return Allocated index or NULL on failure.
 */
GV_JSONPathIndex *gv_json_index_create(void);

/**
 * @brief Destroy a JSON path index and free all resources.
 *
 * @param idx Index to destroy; safe to call with NULL.
 */
void gv_json_index_destroy(GV_JSONPathIndex *idx);

/* Path Registration */

/**
 * @brief Register a new JSON path for indexing.
 *
 * @param idx    Index handle; must be non-NULL.
 * @param config Path configuration describing the path string and its type.
 * @return 0 on success, -1 on error (NULL args, duplicate path, limit reached).
 */
int gv_json_index_add_path(GV_JSONPathIndex *idx, const GV_JSONPathConfig *config);

/**
 * @brief Remove a previously registered JSON path and all its entries.
 *
 * @param idx  Index handle; must be non-NULL.
 * @param path The dot-notation path to remove.
 * @return 0 on success, -1 on error (NULL args, path not found).
 */
int gv_json_index_remove_path(GV_JSONPathIndex *idx, const char *path);

/* Data Manipulation */

/**
 * @brief Parse a JSON string and insert index entries for all registered paths.
 *
 * For each registered path, the JSON tree is walked to extract the value at
 * that path.  If the value exists and matches the expected type, an entry is
 * inserted into the corresponding sorted array.
 *
 * @param idx          Index handle; must be non-NULL.
 * @param vector_index The vector index to associate with extracted values.
 * @param json_str     The JSON string to parse and index.
 * @return 0 on success, -1 on error (NULL args, parse failure).
 */
int gv_json_index_insert(GV_JSONPathIndex *idx, size_t vector_index, const char *json_str);

/**
 * @brief Remove all entries for a given vector index across every registered path.
 *
 * @param idx          Index handle; must be non-NULL.
 * @param vector_index The vector index whose entries should be removed.
 * @return 0 on success, -1 on error.
 */
int gv_json_index_remove(GV_JSONPathIndex *idx, size_t vector_index);

/* Lookup */

/**
 * @brief Look up vector indices whose string value at @p path equals @p value.
 *
 * @param idx         Index handle; must be non-NULL.
 * @param path        Dot-notation JSON path to search.
 * @param value       String value to match.
 * @param out_indices Output array for matching vector indices.
 * @param max_count   Maximum number of results to write.
 * @return Number of results written (>= 0), or -1 on error.
 */
int gv_json_index_lookup_string(const GV_JSONPathIndex *idx, const char *path,
                                const char *value, size_t *out_indices, size_t max_count);

/**
 * @brief Look up vector indices whose int value at @p path falls in [min_val, max_val].
 *
 * @param idx         Index handle; must be non-NULL.
 * @param path        Dot-notation JSON path to search.
 * @param min_val     Inclusive lower bound.
 * @param max_val     Inclusive upper bound.
 * @param out_indices Output array for matching vector indices.
 * @param max_count   Maximum number of results to write.
 * @return Number of results written (>= 0), or -1 on error.
 */
int gv_json_index_lookup_int_range(const GV_JSONPathIndex *idx, const char *path,
                                   int64_t min_val, int64_t max_val,
                                   size_t *out_indices, size_t max_count);

/**
 * @brief Look up vector indices whose float value at @p path falls in [min_val, max_val].
 *
 * @param idx         Index handle; must be non-NULL.
 * @param path        Dot-notation JSON path to search.
 * @param min_val     Inclusive lower bound.
 * @param max_val     Inclusive upper bound.
 * @param out_indices Output array for matching vector indices.
 * @param max_count   Maximum number of results to write.
 * @return Number of results written (>= 0), or -1 on error.
 */
int gv_json_index_lookup_float_range(const GV_JSONPathIndex *idx, const char *path,
                                     double min_val, double max_val,
                                     size_t *out_indices, size_t max_count);

/* Statistics */

/**
 * @brief Return the number of indexed entries for a given path.
 *
 * @param idx  Index handle; must be non-NULL.
 * @param path Dot-notation JSON path.
 * @return Entry count, or 0 if the path is not registered or on error.
 */
size_t gv_json_index_count(const GV_JSONPathIndex *idx, const char *path);

/* Persistence */

/**
 * @brief Save the entire JSON path index to a binary file.
 *
 * @param idx       Index handle; must be non-NULL.
 * @param path_file File path to write.
 * @return 0 on success, -1 on error.
 */
int gv_json_index_save(const GV_JSONPathIndex *idx, const char *path_file);

/**
 * @brief Load a JSON path index from a binary file previously written by
 *        gv_json_index_save().
 *
 * @param path_file File path to read.
 * @return Loaded index, or NULL on error.
 */
GV_JSONPathIndex *gv_json_index_load(const char *path_file);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_JSON_INDEX_H */
