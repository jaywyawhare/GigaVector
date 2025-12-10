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
 * @brief Replay a WAL file by invoking a callback for every insert record.
 *
 * The callback is responsible for applying the operation to the in-memory
 * database. Replay stops on the first error.
 *
 * @param path WAL file path.
 * @param expected_dimension Dimension the WAL must match.
 * @param expected_index_type Index type; validated when nonzero (skipped for old WALs).
 * @param on_insert Callback invoked per insert record; must return 0 on success.
 * @return 0 on success, -1 on I/O or validation failure, or if the callback fails.
 */
int gv_wal_replay(const char *path, size_t expected_dimension,
                  int (*on_insert)(void *ctx, const float *data, size_t dimension,
                                   const char *metadata_key, const char *metadata_value),
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

#ifdef __cplusplus
}
#endif

#endif

