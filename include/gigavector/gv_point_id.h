#ifndef GIGAVECTOR_GV_POINT_ID_H
#define GIGAVECTOR_GV_POINT_ID_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_point_id.h
 * @brief User-defined string/UUID point ID mapping for GigaVector.
 *
 * Provides a bidirectional mapping between user-supplied string identifiers
 * (e.g. UUIDs, slugs, arbitrary strings) and internal integer indices used
 * by the vector storage engine.  The map is thread-safe and supports
 * persistence to disk.
 */

/**
 * @brief Opaque point-ID map handle.
 */
typedef struct GV_PointIDMap GV_PointIDMap;

/* Create / Destroy */

/**
 * @brief Create a new point-ID map.
 *
 * @param initial_capacity  Hint for the initial hash-table capacity.
 *                          The actual capacity is rounded up to a power of two.
 *                          Pass 0 for a reasonable default (64).
 * @return A new GV_PointIDMap, or NULL on allocation failure.
 */
GV_PointIDMap *gv_point_id_create(size_t initial_capacity);

/**
 * @brief Destroy a point-ID map and free all associated memory.
 *
 * @param map  Map to destroy.  NULL is safely ignored.
 */
void gv_point_id_destroy(GV_PointIDMap *map);

/* Map Operations */

/**
 * @brief Insert or update a mapping from a string ID to an internal index.
 *
 * The string is copied internally; the caller retains ownership of
 * @p string_id.  If the string ID already exists its index is updated.
 *
 * @param map           Point-ID map.
 * @param string_id     NUL-terminated string identifier (must not be NULL).
 * @param internal_index Internal integer index to associate.
 * @return 0 on success, -1 on error.
 */
int gv_point_id_set(GV_PointIDMap *map, const char *string_id, size_t internal_index);

/**
 * @brief Look up the internal index for a given string ID.
 *
 * @param map        Point-ID map (const).
 * @param string_id  NUL-terminated string identifier.
 * @param out_index  Output: the associated internal index.
 * @return 0 if found, -1 if not found or on error.
 */
int gv_point_id_get(const GV_PointIDMap *map, const char *string_id, size_t *out_index);

/**
 * @brief Remove a mapping by string ID.
 *
 * @param map        Point-ID map.
 * @param string_id  NUL-terminated string identifier.
 * @return 0 on success, -1 if the ID was not found or on error.
 */
int gv_point_id_remove(GV_PointIDMap *map, const char *string_id);

/**
 * @brief Test whether a string ID exists in the map.
 *
 * @param map        Point-ID map (const).
 * @param string_id  NUL-terminated string identifier.
 * @return 1 if present, 0 if absent, -1 on error.
 */
int gv_point_id_has(const GV_PointIDMap *map, const char *string_id);

/* Reverse Lookup */

/**
 * @brief Reverse lookup: retrieve the string ID for an internal index.
 *
 * The returned pointer is owned by the map and remains valid until the
 * entry is removed or the map is destroyed.
 *
 * @param map             Point-ID map (const).
 * @param internal_index  Internal integer index.
 * @return The string ID, or NULL if no mapping exists for this index.
 */
const char *gv_point_id_reverse_lookup(const GV_PointIDMap *map, size_t internal_index);

/* UUID Generation */

/**
 * @brief Generate a random UUID v4 string (RFC 4122).
 *
 * The output format is "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx" where y is
 * one of {8, 9, a, b}.  Uses /dev/urandom when available, falling back to
 * time-based seeding.
 *
 * @param buf       Output buffer (must hold at least 37 bytes including NUL).
 * @param buf_size  Size of @p buf in bytes.
 * @return 0 on success, -1 on error (NULL buffer or insufficient size).
 */
int gv_point_id_generate_uuid(char *buf, size_t buf_size);

/* Iteration */

/**
 * @brief Return the number of entries in the map.
 *
 * @param map  Point-ID map (const).
 * @return Entry count, or 0 if @p map is NULL.
 */
size_t gv_point_id_count(const GV_PointIDMap *map);

/**
 * @brief Iterate over all entries in the map.
 *
 * The callback is invoked once per entry.  Iteration stops early if the
 * callback returns a non-zero value; in that case the same value is
 * returned from this function.
 *
 * @param map       Point-ID map (const).
 * @param callback  Function called for each entry.
 * @param ctx       Opaque user context passed through to @p callback.
 * @return 0 on success, -1 on error, or the non-zero value returned by
 *         @p callback if it stopped iteration early.
 */
int gv_point_id_iterate(const GV_PointIDMap *map,
                         int (*callback)(const char *id, size_t index, void *ctx),
                         void *ctx);

/* Persistence */

/**
 * @brief Save the map to a binary file.
 *
 * Format: entry_count, then for each entry: string_len (size_t),
 * string_id (string_len bytes, no NUL), internal_index (size_t).
 *
 * @param map       Point-ID map (const).
 * @param filepath  Output file path.
 * @return 0 on success, -1 on error.
 */
int gv_point_id_save(const GV_PointIDMap *map, const char *filepath);

/**
 * @brief Load a map from a binary file previously written by gv_point_id_save().
 *
 * @param filepath  Input file path.
 * @return A new GV_PointIDMap, or NULL on error.
 */
GV_PointIDMap *gv_point_id_load(const char *filepath);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_POINT_ID_H */
