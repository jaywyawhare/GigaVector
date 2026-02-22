#ifndef GIGAVECTOR_GV_MULTIMODAL_H
#define GIGAVECTOR_GV_MULTIMODAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_multimodal.h
 * @brief Multimodal media storage for GigaVector.
 *
 * Provides native storage for raw media (images, audio, video, documents)
 * alongside vector embeddings. Stores blobs on disk with metadata and links
 * them to vector indices via content-addressable SHA-256 hashing.
 */

/**
 * @brief Media type classification.
 */
typedef enum {
    GV_MEDIA_IMAGE    = 0,          /**< Image (JPEG, PNG, WebP, etc.). */
    GV_MEDIA_AUDIO    = 1,          /**< Audio (WAV, MP3, FLAC, etc.). */
    GV_MEDIA_VIDEO    = 2,          /**< Video (MP4, WebM, etc.). */
    GV_MEDIA_DOCUMENT = 3,          /**< Document (PDF, DOCX, etc.). */
    GV_MEDIA_BLOB     = 4           /**< Generic binary blob. */
} GV_MediaType;

/**
 * @brief Media store configuration.
 */
typedef struct {
    const char *storage_dir;        /**< Directory for blob storage. */
    size_t max_blob_size_mb;        /**< Maximum blob size in MB (default: 100). */
    int deduplicate;                /**< Skip store if SHA-256 hash matches (default: 1). */
    int compress_blobs;             /**< Compress blobs before writing (default: 0). */
} GV_MediaConfig;

/**
 * @brief Opaque media store handle.
 */
typedef struct GV_MediaStore GV_MediaStore;

/**
 * @brief Metadata entry for a stored media blob.
 */
typedef struct {
    size_t vector_index;            /**< Associated vector index. */
    GV_MediaType type;              /**< Media type. */
    char *filename;                 /**< Original filename. */
    size_t file_size;               /**< Size of stored blob in bytes. */
    char hash[65];                  /**< SHA-256 hex digest (64 chars + null). */
    uint64_t created_at;            /**< Creation timestamp (Unix epoch). */
    char *mime_type;                /**< MIME type string. */
} GV_MediaEntry;

/* Configuration */

/**
 * @brief Initialize media configuration with defaults.
 *
 * Default values:
 * - storage_dir: NULL (must be set by caller)
 * - max_blob_size_mb: 100
 * - deduplicate: 1
 * - compress_blobs: 0
 *
 * @param config Configuration to initialize.
 */
void gv_media_config_init(GV_MediaConfig *config);

/* Store Lifecycle */

/**
 * @brief Create a media store.
 *
 * Creates the storage directory if it does not exist.
 *
 * @param config Media configuration (storage_dir must be set).
 * @return Media store handle, or NULL on error.
 */
GV_MediaStore *gv_media_create(const GV_MediaConfig *config);

/**
 * @brief Destroy a media store and free all resources.
 *
 * Does not delete stored blob files on disk.
 *
 * @param store Media store (safe to call with NULL).
 */
void gv_media_destroy(GV_MediaStore *store);

/* Store Operations */

/**
 * @brief Store a media blob from memory.
 *
 * Computes the SHA-256 hash, writes the blob to storage_dir/{hash}.blob,
 * and records the metadata entry linked to vector_index.
 *
 * @param store     Media store.
 * @param vector_index Vector index to associate with this blob.
 * @param type      Media type classification.
 * @param data      Pointer to blob data.
 * @param data_size Size of blob data in bytes.
 * @param filename  Original filename (may be NULL).
 * @param mime_type MIME type string (may be NULL).
 * @return 0 on success, -1 on error.
 */
int gv_media_store_blob(GV_MediaStore *store, size_t vector_index,
                         GV_MediaType type, const void *data, size_t data_size,
                         const char *filename, const char *mime_type);

/**
 * @brief Store a media blob from a file on disk.
 *
 * Reads the file, computes its SHA-256 hash, and copies it into the
 * content-addressable store. The original filename and MIME type are
 * inferred from the file path.
 *
 * @param store        Media store.
 * @param vector_index Vector index to associate with this blob.
 * @param type         Media type classification.
 * @param file_path    Path to the source file.
 * @return 0 on success, -1 on error.
 */
int gv_media_store_file(GV_MediaStore *store, size_t vector_index,
                         GV_MediaType type, const char *file_path);

/**
 * @brief Retrieve a stored blob into a buffer.
 *
 * @param store       Media store.
 * @param vector_index Vector index of the blob to retrieve.
 * @param buffer      Output buffer.
 * @param buffer_size Size of output buffer.
 * @param actual_size Actual size of the blob (set even if buffer is too small).
 * @return 0 on success, -1 on error.
 */
int gv_media_retrieve(const GV_MediaStore *store, size_t vector_index,
                       void *buffer, size_t buffer_size, size_t *actual_size);

/**
 * @brief Get the on-disk file path for a stored blob.
 *
 * @param store        Media store.
 * @param vector_index Vector index of the blob.
 * @param path         Output buffer for the file path.
 * @param path_size    Size of the path buffer.
 * @return 0 on success, -1 on error.
 */
int gv_media_get_path(const GV_MediaStore *store, size_t vector_index,
                       char *path, size_t path_size);

/**
 * @brief Get metadata for a stored blob.
 *
 * The returned entry's filename and mime_type are freshly allocated copies;
 * the caller must free them.
 *
 * @param store        Media store.
 * @param vector_index Vector index of the blob.
 * @param entry        Output metadata entry.
 * @return 0 on success, -1 on error.
 */
int gv_media_get_info(const GV_MediaStore *store, size_t vector_index,
                       GV_MediaEntry *entry);

/**
 * @brief Delete a blob entry (removes metadata and on-disk file).
 *
 * If deduplication is active, the file is only removed when no other
 * entries reference the same hash.
 *
 * @param store        Media store.
 * @param vector_index Vector index of the blob to delete.
 * @return 0 on success, -1 on error.
 */
int gv_media_delete(GV_MediaStore *store, size_t vector_index);

/* Query Operations */

/**
 * @brief Check whether a blob exists for a vector index.
 *
 * @param store        Media store.
 * @param vector_index Vector index to check.
 * @return 1 if exists, 0 if not, -1 on error.
 */
int gv_media_exists(const GV_MediaStore *store, size_t vector_index);

/**
 * @brief Return the number of stored media entries.
 *
 * @param store Media store.
 * @return Number of entries.
 */
size_t gv_media_count(const GV_MediaStore *store);

/**
 * @brief Return the total size of all stored blobs on disk in bytes.
 *
 * @param store Media store.
 * @return Total bytes on disk.
 */
size_t gv_media_total_size(const GV_MediaStore *store);

/* Index Persistence */

/**
 * @brief Save the media index to a binary file.
 *
 * Format: count + entries (vector_index, type, hash, filename_len, filename,
 * size, mime_type_len, mime_type, created_at).
 *
 * @param store Media store.
 * @param path  Output file path.
 * @return 0 on success, -1 on error.
 */
int gv_media_save_index(const GV_MediaStore *store, const char *path);

/**
 * @brief Load a media index from a binary file.
 *
 * Reconstructs the in-memory index and associates it with the given
 * storage directory.
 *
 * @param index_path  Path to the saved index file.
 * @param storage_dir Path to the blob storage directory.
 * @return Media store handle, or NULL on error.
 */
GV_MediaStore *gv_media_load_index(const char *index_path,
                                    const char *storage_dir);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_MULTIMODAL_H */
