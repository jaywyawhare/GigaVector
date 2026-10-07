#ifndef GIGAVECTOR_GV_WAL_H
#define GIGAVECTOR_GV_WAL_H

#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

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
GV_WAL *wal_open(const char *path, size_t dimension, uint32_t index_type);

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
int wal_append_insert(GV_WAL *wal, const float *data, size_t dimension,
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
int wal_append_insert_rich(GV_WAL *wal, const float *data, size_t dimension,
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
int wal_append_delete(GV_WAL *wal, size_t vector_index);

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
int wal_append_update(GV_WAL *wal, size_t vector_index, const float *data, size_t dimension,
                         const char *const *metadata_keys, const char *const *metadata_values,
                         size_t metadata_count);

/**
 * @brief Append an IVFDisk posting-list assignment (audit / replication hook).
 */
int wal_append_ivfdisk_append(GV_WAL *wal, uint64_t head_id, uint64_t vector_id,
                              const float *data, size_t dimension);

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
int wal_replay(const char *path, size_t expected_dimension,
                  int (*on_insert)(void *ctx, const float *data, size_t dimension,
                                   const char *metadata_key, const char *metadata_value),
                  void *ctx, uint32_t expected_index_type);

/**
 * @brief Replay a WAL file and deliver all metadata entries for each record.
 *
 * Newer API that passes arrays of metadata key/value pairs to the callback.
 * All operation types (INSERT, DELETE, UPDATE) are delivered to the caller.
 * Pass NULL for on_delete or on_update to skip those record types.
 *
 * @param path WAL file path.
 * @param expected_dimension Dimension the WAL must match.
 * @param expected_index_type Index type; validated when nonzero (skipped for old WALs).
 * @param on_insert Callback invoked per insert record; must return 0 on success.
 * @param on_delete Optional callback invoked per delete record; NULL to skip.
 * @param on_update Optional callback invoked per update record; NULL to skip.
 * @param ctx Caller context passed to all callbacks.
 * @return 0 on success, -1 on I/O or validation failure, or if a callback fails.
 */
int wal_replay_rich(const char *path, size_t expected_dimension,
                       int (*on_insert)(void *ctx, const float *data, size_t dimension,
                                        const char *const *metadata_keys, const char *const *metadata_values,
                                        size_t metadata_count),
                       int (*on_delete)(void *ctx, size_t vector_index),
                       int (*on_update)(void *ctx, size_t vector_index, const float *data,
                                        size_t dimension, const char *const *metadata_keys,
                                        const char *const *metadata_values, size_t metadata_count),
                       int (*on_ivfdisk_append)(void *ctx, uint64_t head_id, uint64_t vector_id,
                                                const float *data, size_t dimension),
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
int wal_dump(const char *path, size_t expected_dimension, uint32_t expected_index_type, FILE *out);

/**
 * @brief Close a WAL handle.
 *
 * Safe to call with NULL; flushes pending data.
 *
 * @param wal WAL handle.
 */
void wal_close(GV_WAL *wal);

/**
 * @brief Set the fsync interval in records (group commit).
 * 1 (default) fsyncs every record (crash-durable). A larger interval fsyncs only
 * every N records — much faster for bulk loads at the cost of power-loss durability
 * for the last <N records. Forces a durable flush of anything currently pending.
 */
void wal_set_sync_interval(GV_WAL *wal, size_t interval);

/**
 * @brief Truncate the WAL file (used after successful checkpoint/save).
 *
 * @param path WAL file path.
 * @return 0 on success, -1 on error.
 */
int wal_reset(const char *path);

/**
 * @brief Truncate an open WAL file, resetting it to a fresh state.
 *
 * This function closes the current file handle, truncates the file,
 * and reopens it for future writes. The WAL header is rewritten.
 *
 * @param wal WAL handle; must be non-NULL.
 * @return 0 on success, -1 on error.
 */
int wal_truncate(GV_WAL *wal);

/**
 * @brief Count WAL records in a file (excluding header).
 */
uint64_t wal_count_entries(const char *path);

/**
 * @brief Current on-disk size of the WAL in bytes (0 if closed/NULL).
 *
 * Used to drive size-based checkpointing.
 */
uint64_t wal_size(const GV_WAL *wal);

/**
 * @brief Read one WAL record by zero-based entry index.
 *
 * @return 0 on success, -1 on error or if index out of range.
 */
int wal_read_entry_at(const char *path, uint64_t entry_index, uint8_t *out_type,
                      uint8_t **out_buf, size_t *out_len);

/**
 * @brief Append a pre-serialized WAL record (type byte + payload + optional CRC).
 */
int wal_append_raw(GV_WAL *wal, const uint8_t *record, size_t len);

/**
 * @brief Parse and apply one WAL record buffer via replay callbacks.
 */
int wal_apply_record_buffer(const uint8_t *record, size_t len, int has_crc,
                            size_t expected_dimension,
                            int (*on_insert)(void *ctx, const float *data, size_t dimension,
                                             const char *const *metadata_keys,
                                             const char *const *metadata_values,
                                             size_t metadata_count),
                            int (*on_delete)(void *ctx, size_t vector_index),
                            int (*on_update)(void *ctx, size_t vector_index,
                                             const float *data, size_t dimension,
                                             const char *const *metadata_keys,
                                             const char *const *metadata_values,
                                             size_t metadata_count),
                            int (*on_ivfdisk_append)(void *ctx, uint64_t head_id, uint64_t vector_id,
                                                     const float *data, size_t dimension),
                            void *ctx);

#ifdef __cplusplus
}
#endif

#endif

