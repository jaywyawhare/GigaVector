#ifndef GIGAVECTOR_GV_WAL_H
#define GIGAVECTOR_GV_WAL_H

#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Write-ahead log handle used by the database.
 */
typedef struct GV_WAL GV_WAL;

/**
 * @brief Open (or create) a WAL file for a database.
 *
 * The WAL records the database dimension inside its header; opening will
 * validate that the on-disk WAL matches the expected dimension.
 *
 * @param path Filesystem path to the WAL file.
 * @param dimension Expected vector dimension.
 * @param index_type Index type identifier (as uint32_t); validated when nonzero.
 * @return Allocated WAL handle or NULL on failure.
 */
GV_WAL *gv_wal_open(const char *path, size_t dimension, uint32_t index_type);

/**
 * @brief Append an insert operation to the WAL.
 *
 * The vector payload and optional single metadata key/value are persisted.
 *
 * @param wal WAL handle; must be non-NULL.
 * @param data Vector data array.
 * @param dimension Number of elements in @p data.
 * @param metadata_key Optional metadata key; NULL to skip.
 * @param metadata_value Optional metadata value; NULL if key is NULL.
 * @return 0 on success, -1 on I/O or validation failure.
 */
int gv_wal_append_insert(GV_WAL *wal, const float *data, size_t dimension,
                         const char *metadata_key, const char *metadata_value);

/**
 * @brief Append an insert operation to the WAL with multiple metadata entries.
 *
 * The vector payload and optional metadata key/value pairs are persisted.
 * This function supports rich metadata (multiple key-value pairs per vector).
 *
 * @param wal WAL handle; must be non-NULL.
 * @param data Vector data array.
 * @param dimension Number of elements in @p data.
 * @param metadata_keys Array of metadata keys; NULL if count is 0.
 * @param metadata_values Array of metadata values; NULL if count is 0.
 * @param metadata_count Number of metadata entries (must match array lengths).
 * @return 0 on success, -1 on I/O or validation failure.
 */
int gv_wal_append_insert_rich(GV_WAL *wal, const float *data, size_t dimension,
                               const char *const *metadata_keys, const char *const *metadata_values,
                               size_t metadata_count);

/**
 * @brief Append a delete operation to the WAL.
 *
 * Records the deletion of a vector by its index (insertion order).
 *
 * @param wal WAL handle; must be non-NULL.
 * @param vector_index Index of the vector to delete (0-based insertion order).
 * @return 0 on success, -1 on I/O or validation failure.
 */
int gv_wal_append_delete(GV_WAL *wal, size_t vector_index);

/**
 * @brief Append an update operation to the WAL.
 *
 * Records the update of a vector by its index (insertion order).
 * The vector payload and optional metadata are persisted.
 *
 * @param wal WAL handle; must be non-NULL.
 * @param vector_index Index of the vector to update (0-based insertion order).
 * @param data Vector data array.
 * @param dimension Number of elements in @p data.
 * @param metadata_keys Array of metadata keys; NULL if count is 0.
 * @param metadata_values Array of metadata values; NULL if count is 0.
 * @param metadata_count Number of metadata entries (must match array lengths).
 * @return 0 on success, -1 on I/O or validation failure.
 */
int gv_wal_append_update(GV_WAL *wal, size_t vector_index, const float *data, size_t dimension,
                         const char *const *metadata_keys, const char *const *metadata_values,
                         size_t metadata_count);

/**
 * @brief Replay a WAL file by invoking a callback for every insert record.
 *
 * The callback is responsible for applying the operation to the in-memory
 * database. Replay stops on the first error.
 *
 * @param path WAL file path.
 * @param expected_dimension Dimension the WAL must match.
 * @param expected_index_type Index type; validated when nonzero (skipped for old WALs).
 * @param on_insert Callback invoked per insert record; must return 0 on success.
 *                   For vectors with multiple metadata entries, this callback may be
 *                   called multiple times (once per metadata entry) or with a vector
 *                   that has all metadata. The implementation handles both cases.
 * @return 0 on success, -1 on I/O or validation failure, or if the callback fails.
 */
int gv_wal_replay(const char *path, size_t expected_dimension,
                  int (*on_insert)(void *ctx, const float *data, size_t dimension,
                                   const char *metadata_key, const char *metadata_value),
                  void *ctx, uint32_t expected_index_type);

/**
 * @brief Replay a WAL file and deliver all metadata entries for each record.
 *
 * Newer API that passes arrays of metadata key/value pairs to the callback.
 * Existing WAL files with single metadata entries remain compatible.
 *
 * @param path WAL file path.
 * @param expected_dimension Dimension the WAL must match.
 * @param expected_index_type Index type; validated when nonzero (skipped for old WALs).
 * @param on_insert Callback invoked per insert record; must return 0 on success.
 * @return 0 on success, -1 on I/O or validation failure, or if the callback fails.
 */
int gv_wal_replay_rich(const char *path, size_t expected_dimension,
                       int (*on_insert)(void *ctx, const float *data, size_t dimension,
                                        const char *const *metadata_keys, const char *const *metadata_values,
                                        size_t metadata_count),
                       void *ctx, uint32_t expected_index_type);

/**
 * @brief Human-readable dump of WAL contents for debugging.
 *
 * Prints one line per insert record to @p out. Header validation must succeed
 * (magic, version, dimension) or -1 is returned.
 *
 * @param path WAL file path.
 * @param expected_dimension Expected vector dimension (must match the WAL).
 * @param expected_index_type Expected index type; 0 to skip (for old WALs).
 * @param out Output stream; must be non-NULL (e.g., stdout).
 * @return 0 on success, -1 on error.
 */
int gv_wal_dump(const char *path, size_t expected_dimension, uint32_t expected_index_type, FILE *out);

/**
 * @brief Close a WAL handle.
 *
 * Safe to call with NULL; flushes pending data.
 *
 * @param wal WAL handle.
 */
void gv_wal_close(GV_WAL *wal);

/**
 * @brief Truncate the WAL file (used after successful checkpoint/save).
 *
 * @param path WAL file path.
 * @return 0 on success, -1 on error.
 */
int gv_wal_reset(const char *path);

/**
 * @brief Truncate an open WAL file, resetting it to a fresh state.
 *
 * This function closes the current file handle, truncates the file,
 * and reopens it for future writes. The WAL header is rewritten.
 *
 * @param wal WAL handle; must be non-NULL.
 * @return 0 on success, -1 on error.
 */
int gv_wal_truncate(GV_WAL *wal);

#ifdef __cplusplus
}
#endif

#endif

