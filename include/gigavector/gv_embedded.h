#ifndef GIGAVECTOR_GV_EMBEDDED_H
#define GIGAVECTOR_GV_EMBEDDED_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Index types supported in embedded mode.
 */
typedef enum {
    GV_EMBEDDED_INDEX_FLAT = 0, /**< Brute-force linear scan. */
    GV_EMBEDDED_INDEX_HNSW = 1, /**< Simplified single-level HNSW graph. */
    GV_EMBEDDED_INDEX_LSH  = 2  /**< Random hyperplane LSH with multiple tables. */
} GV_EmbeddedIndexType;

/**
 * @brief Configuration for embedded / edge mode database.
 */
typedef struct {
    size_t dimension;        /**< Vector dimensionality; required, must be > 0. */
    int index_type;          /**< One of GV_EMBEDDED_INDEX_*; default FLAT. */
    size_t max_vectors;      /**< Hard limit on vector count; 0 = unlimited. */
    size_t memory_limit_mb;  /**< Soft memory budget in MiB; default 64. */
    int mmap_storage;        /**< Use mmap for file-backed storage; default 0. */
    const char *storage_path; /**< File path for persistence; NULL = in-memory only. */
    int quantize;            /**< Quantization bits: 0=none, 4=4bit, 8=8bit; default 0. */
} GV_EmbeddedConfig;

/**
 * @brief Opaque handle for an embedded vector database instance.
 */
typedef struct GV_EmbeddedDB GV_EmbeddedDB;

/**
 * @brief Single search result from embedded search.
 */
typedef struct {
    size_t index;   /**< Index (ID) of the matching vector. */
    float distance; /**< Distance from the query vector. */
} GV_EmbeddedResult;

/**
 * @brief Initialize an embedded config with sensible defaults.
 *
 * Sets dimension=0, index_type=FLAT, max_vectors=0 (unlimited),
 * memory_limit_mb=64, mmap_storage=0, storage_path=NULL, quantize=0.
 * Caller must set dimension before calling gv_embedded_open().
 *
 * @param config Config to initialize; must be non-NULL.
 */
void gv_embedded_config_init(GV_EmbeddedConfig *config);

/**
 * @brief Open an embedded vector database.
 *
 * Creates an in-process, single-threaded vector store suitable for edge
 * devices, mobile, IoT, and embedded applications. No server, no threads,
 * no WAL -- minimal memory footprint by design.
 *
 * @param config Configuration; must be non-NULL with dimension > 0.
 * @return New database handle, or NULL on error.
 */
GV_EmbeddedDB *gv_embedded_open(const GV_EmbeddedConfig *config);

/**
 * @brief Close an embedded database and release all resources.
 *
 * Safe to call with NULL; no action is taken.
 *
 * @param db Database handle.
 */
void gv_embedded_close(GV_EmbeddedDB *db);

/**
 * @brief Add a vector, assigning the next sequential index.
 *
 * @param db Database handle; must be non-NULL.
 * @param vector Float array of length dimension.
 * @return Assigned index (>= 0) on success, -1 on error or capacity exceeded.
 */
int gv_embedded_add(GV_EmbeddedDB *db, const float *vector);

/**
 * @brief Add a vector with a caller-chosen ID.
 *
 * If @p id exceeds the current capacity the storage is extended. Using an
 * ID that is already occupied overwrites the previous vector.
 *
 * @param db Database handle; must be non-NULL.
 * @param id Desired vector index.
 * @param vector Float array of length dimension.
 * @return 0 on success, -1 on error.
 */
int gv_embedded_add_with_id(GV_EmbeddedDB *db, size_t id, const float *vector);

/**
 * @brief Search for the k nearest neighbors of a query vector.
 *
 * @param db Database handle; must be non-NULL.
 * @param query Float array of length dimension.
 * @param k Maximum number of neighbors to return.
 * @param distance_type Distance metric (GV_DistanceType value).
 * @param results Output array of at least k elements.
 * @return Number of results found (0 to k), or -1 on error.
 */
int gv_embedded_search(const GV_EmbeddedDB *db, const float *query, size_t k,
                       int distance_type, GV_EmbeddedResult *results);

/**
 * @brief Mark a vector as deleted.
 *
 * The slot is marked in the deleted bitmap; actual storage reclamation
 * happens during gv_embedded_compact().
 *
 * @param db Database handle; must be non-NULL.
 * @param index Index of the vector to delete.
 * @return 0 on success, -1 on error.
 */
int gv_embedded_delete(GV_EmbeddedDB *db, size_t index);

/**
 * @brief Retrieve a vector by index.
 *
 * @param db Database handle; must be non-NULL.
 * @param index Index of the vector.
 * @param output Output buffer of at least dimension floats.
 * @return 0 on success, -1 if index is out of range or deleted.
 */
int gv_embedded_get(const GV_EmbeddedDB *db, size_t index, float *output);

/**
 * @brief Return the number of active (non-deleted) vectors.
 *
 * @param db Database handle; must be non-NULL.
 * @return Active vector count, or 0 if db is NULL.
 */
size_t gv_embedded_count(const GV_EmbeddedDB *db);

/**
 * @brief Return current memory usage in bytes.
 *
 * Includes vector data, index structures, quantization tables, and
 * internal bookkeeping.
 *
 * @param db Database handle; must be non-NULL.
 * @return Bytes used, or 0 if db is NULL.
 */
size_t gv_embedded_memory_usage(const GV_EmbeddedDB *db);

/**
 * @brief Save the database to a binary file.
 *
 * File format: header (magic "GVEM", dimension, count, index_type, quant)
 * followed by vector data and index-specific data.
 *
 * @param db Database handle; must be non-NULL.
 * @param path Output file path.
 * @return 0 on success, -1 on error.
 */
int gv_embedded_save(const GV_EmbeddedDB *db, const char *path);

/**
 * @brief Load a database from a binary file previously saved with
 *        gv_embedded_save().
 *
 * @param path File path to load.
 * @return New database handle, or NULL on error.
 */
GV_EmbeddedDB *gv_embedded_load(const char *path);

/**
 * @brief Compact the database by removing deleted entries.
 *
 * Shifts remaining vectors into a contiguous block and rebuilds the
 * index. After compaction, vector indices may change.
 *
 * @param db Database handle; must be non-NULL.
 * @return 0 on success, -1 on error.
 */
int gv_embedded_compact(GV_EmbeddedDB *db);

#ifdef __cplusplus
}
#endif

#endif
