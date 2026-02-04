#ifndef GIGAVECTOR_GV_BACKUP_H
#define GIGAVECTOR_GV_BACKUP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_backup.h
 * @brief Backup and restore utilities for GigaVector.
 *
 * Provides functionality for creating, verifying, and restoring database backups.
 */

/* Forward declarations */
struct GV_Database;
typedef struct GV_Database GV_Database;

/**
 * @brief Backup format version.
 */
#define GV_BACKUP_VERSION 1

/**
 * @brief Backup compression type.
 */
typedef enum {
    GV_BACKUP_COMPRESS_NONE = 0,    /**< No compression. */
    GV_BACKUP_COMPRESS_ZLIB = 1,    /**< zlib compression. */
    GV_BACKUP_COMPRESS_LZ4 = 2      /**< LZ4 compression (fast). */
} GV_BackupCompression;

/**
 * @brief Backup options.
 */
typedef struct {
    GV_BackupCompression compression; /**< Compression type (default: NONE). */
    int include_wal;                /**< Include WAL in backup (default: 1). */
    int include_metadata;           /**< Include metadata index (default: 1). */
    int verify_after;               /**< Verify backup after creation (default: 1). */
    const char *encryption_key;     /**< Optional encryption password (NULL = no encryption). */
} GV_BackupOptions;

/**
 * @brief Backup header information.
 */
typedef struct {
    uint32_t version;               /**< Backup format version. */
    uint32_t flags;                 /**< Backup flags. */
    uint64_t created_at;            /**< Creation timestamp. */
    uint64_t vector_count;          /**< Number of vectors. */
    uint32_t dimension;             /**< Vector dimension. */
    uint32_t index_type;            /**< Index type. */
    uint64_t original_size;         /**< Original data size. */
    uint64_t compressed_size;       /**< Compressed size (0 if uncompressed). */
    char checksum[65];              /**< SHA-256 checksum (hex). */
} GV_BackupHeader;

/**
 * @brief Backup progress callback.
 */
typedef void (*GV_BackupProgressCallback)(size_t current, size_t total, void *user_data);

/**
 * @brief Restore options.
 */
typedef struct {
    int overwrite;                  /**< Overwrite existing database (default: 0). */
    int verify_checksum;            /**< Verify checksum before restore (default: 1). */
    const char *decryption_key;     /**< Decryption password (NULL if not encrypted). */
} GV_RestoreOptions;

/**
 * @brief Backup result/status.
 */
typedef struct {
    int success;                    /**< 1 if successful, 0 if failed. */
    char *error_message;            /**< Error message if failed (NULL if success). */
    uint64_t bytes_processed;       /**< Bytes processed. */
    uint64_t vectors_processed;     /**< Vectors processed. */
    double elapsed_seconds;         /**< Time elapsed. */
} GV_BackupResult;

/* ============================================================================
 * Configuration
 * ============================================================================ */

/**
 * @brief Initialize backup options with defaults.
 *
 * @param options Options to initialize.
 */
void gv_backup_options_init(GV_BackupOptions *options);

/**
 * @brief Initialize restore options with defaults.
 *
 * @param options Options to initialize.
 */
void gv_restore_options_init(GV_RestoreOptions *options);

/* ============================================================================
 * Backup Operations
 * ============================================================================ */

/**
 * @brief Create a backup of a database.
 *
 * @param db Database to backup.
 * @param backup_path Output backup file path.
 * @param options Backup options (NULL for defaults).
 * @param progress Progress callback (NULL to disable).
 * @param user_data User data for progress callback.
 * @return Backup result (caller owns, use gv_backup_result_free).
 */
GV_BackupResult *gv_backup_create(GV_Database *db, const char *backup_path,
                                   const GV_BackupOptions *options,
                                   GV_BackupProgressCallback progress,
                                   void *user_data);

/**
 * @brief Create a backup from a database file.
 *
 * @param db_path Database file path.
 * @param backup_path Output backup file path.
 * @param options Backup options (NULL for defaults).
 * @param progress Progress callback (NULL to disable).
 * @param user_data User data for progress callback.
 * @return Backup result.
 */
GV_BackupResult *gv_backup_create_from_file(const char *db_path, const char *backup_path,
                                             const GV_BackupOptions *options,
                                             GV_BackupProgressCallback progress,
                                             void *user_data);

/**
 * @brief Free backup result.
 *
 * @param result Result to free.
 */
void gv_backup_result_free(GV_BackupResult *result);

/* ============================================================================
 * Restore Operations
 * ============================================================================ */

/**
 * @brief Restore a database from backup.
 *
 * @param backup_path Backup file path.
 * @param db_path Output database path.
 * @param options Restore options (NULL for defaults).
 * @param progress Progress callback (NULL to disable).
 * @param user_data User data for progress callback.
 * @return Backup result.
 */
GV_BackupResult *gv_backup_restore(const char *backup_path, const char *db_path,
                                    const GV_RestoreOptions *options,
                                    GV_BackupProgressCallback progress,
                                    void *user_data);

/**
 * @brief Restore to an in-memory database.
 *
 * @param backup_path Backup file path.
 * @param options Restore options (NULL for defaults).
 * @param db Output database pointer.
 * @return Backup result.
 */
GV_BackupResult *gv_backup_restore_to_db(const char *backup_path,
                                          const GV_RestoreOptions *options,
                                          GV_Database **db);

/* ============================================================================
 * Inspection Operations
 * ============================================================================ */

/**
 * @brief Read backup header without full restore.
 *
 * @param backup_path Backup file path.
 * @param header Output header.
 * @return 0 on success, -1 on error.
 */
int gv_backup_read_header(const char *backup_path, GV_BackupHeader *header);

/**
 * @brief Verify backup integrity.
 *
 * @param backup_path Backup file path.
 * @param decryption_key Decryption key (NULL if not encrypted).
 * @return Backup result (success=1 if valid).
 */
GV_BackupResult *gv_backup_verify(const char *backup_path, const char *decryption_key);

/**
 * @brief Get human-readable backup info.
 *
 * @param backup_path Backup file path.
 * @param info_buf Output buffer.
 * @param buf_size Buffer size.
 * @return 0 on success, -1 on error.
 */
int gv_backup_get_info(const char *backup_path, char *info_buf, size_t buf_size);

/* ============================================================================
 * Incremental Backup
 * ============================================================================ */

/**
 * @brief Create an incremental backup.
 *
 * Only backs up changes since the last full backup.
 *
 * @param db Database to backup.
 * @param backup_path Output backup file path.
 * @param base_backup_path Path to the base full backup.
 * @param options Backup options.
 * @return Backup result.
 */
GV_BackupResult *gv_backup_create_incremental(GV_Database *db, const char *backup_path,
                                               const char *base_backup_path,
                                               const GV_BackupOptions *options);

/**
 * @brief Merge incremental backups into a full backup.
 *
 * @param base_backup_path Base full backup.
 * @param incremental_paths Array of incremental backup paths.
 * @param incremental_count Number of incremental backups.
 * @param output_path Output merged backup path.
 * @return Backup result.
 */
GV_BackupResult *gv_backup_merge(const char *base_backup_path,
                                  const char **incremental_paths, size_t incremental_count,
                                  const char *output_path);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * @brief Compute checksum of a backup file.
 *
 * @param backup_path Backup file path.
 * @param checksum_out Output buffer (65 bytes for hex + null).
 * @return 0 on success, -1 on error.
 */
int gv_backup_compute_checksum(const char *backup_path, char *checksum_out);

/**
 * @brief Get compression name string.
 *
 * @param compression Compression type.
 * @return Compression name.
 */
const char *gv_backup_compression_string(GV_BackupCompression compression);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_BACKUP_H */
