#ifndef GIGAVECTOR_GV_DISKANN_H
#define GIGAVECTOR_GV_DISKANN_H
#include <stddef.h>
#include <stdint.h>
#include "search/distance.h"
#include "core/id_bitmap.h"

struct GV_DiskPageCache;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    GV_DistanceType metric;   /* Distance metric (default: GV_DISTANCE_EUCLIDEAN).
                               * Full-vector comparisons and the exact rerank use
                               * it via the shared SIMD kernels; PQ navigation is
                               * an L2 approximation refined by the exact rerank. */
    size_t max_degree;        /* Max graph degree R (default: 64) */
    float alpha;              /* Pruning parameter alpha (default: 1.2) */
    size_t build_beam_width;  /* Beam width during build (default: 128) */
    size_t search_beam_width; /* Beam width during search (default: 64) */
    size_t pq_dim;            /* PQ compressed dimension for in-memory nav (default: 0 = auto) */
    const char *data_path;    /* Path for on-disk vector storage */
    size_t cache_size_mb;     /* Memory cache size in MB (default: 256) */
    size_t sector_size;       /* Disk sector alignment (default: 4096) */
    int graph_on_disk;        /* Out-of-core graph (default 0): flush adjacency to
                               * "{data_path}.graph" after build/load and free it
                               * from RAM. Insert/delete/search all work after the
                               * flush; inserts patch adjacency slots on disk.
                               * The flush happens in build/load, so an index only
                               * ever grown by insert (never built/loaded) keeps
                               * its adjacency in RAM until the first flush. */
    int force_in_memory;      /* On load (default 0): keep the reconstructed graph
                               * in RAM even if the saved index used graph_on_disk.
                               * Overrides the stored flag and graph_on_disk. Lets
                               * an out-of-core-saved index be served from RAM.
                               * Ignored by create/build. */
} GV_DiskANNConfig;

/*
 * Concurrency: each GV_DiskANNIndex serializes its public API (build, insert,
 * search, delete, get_stats, save, count, attach_page_cache) with an internal
 * mutex, so concurrent calls are safe. Search is not read-only internally (it
 * mutates the LRU page cache and latency counters), so all calls take the same
 * exclusive lock rather than a shared read lock. diskann_create/diskann_load are
 * constructors and diskann_destroy must not race with other calls on the index.
 *
 * Deletion is lazy (tombstone): diskann_delete flags the node and records its
 * slot; search skips tombstones. A later diskann_insert reuses a freed slot
 * before appending, so space is reclaimed without renumbering live positional
 * ids (result.index / diskann_delete(vector_index) stay stable). A reused slot
 * may keep stale incoming edges, which only cost a few extra candidates since
 * search re-ranks by exact distance. diskann_build fully rebuilds and reclaims.
 */

typedef struct GV_DiskANNIndex GV_DiskANNIndex;

typedef struct {
    size_t index;
    float distance;
} GV_DiskANNResult;

typedef struct {
    size_t total_vectors;
    size_t graph_edges;
    size_t cache_hits;
    size_t cache_misses;
    size_t disk_reads;
    double avg_search_latency_us;
    size_t memory_usage_bytes;
    size_t disk_usage_bytes;
} GV_DiskANNStats;

/**
 * @brief Initialize a configuration structure with default values.
 *
 * @param config Configuration to apply/output.
 */
void diskann_config_init(GV_DiskANNConfig *config);
GV_DiskANNIndex *diskann_create(size_t dimension, const GV_DiskANNConfig *config);
/**
 * @brief Destroy an instance and free associated resources.
 *
 * @param index Index instance.
 */
void diskann_destroy(GV_DiskANNIndex *index);

/**
 * @brief Perform the operation.
 *
 * @param index Index instance.
 * @param data Input data buffer.
 * @param count Number of items.
 * @param dimension Vector dimensionality.
 * @return 0 on success, -1 on error.
 */
int diskann_build(GV_DiskANNIndex *index, const float *data, size_t count, size_t dimension);
/**
 * @brief Build the graph using multiple threads (parallel search, serial link).
 *
 * Same result shape as diskann_build; the two Vamana passes run their greedy
 * searches across @p num_threads workers. Falls back to serial when PQ is
 * untrained (parallel search requires PQ for race-free RAM-only distances).
 *
 * @param index Index instance.
 * @param data Input data buffer (count * dimension floats).
 * @param count Number of vectors.
 * @param dimension Vector dimensionality.
 * @param num_threads Worker thread count (0 or 1 = serial).
 * @return 0 on success, -1 on error.
 */
int diskann_build_parallel(GV_DiskANNIndex *index, const float *data, size_t count,
                           size_t dimension, size_t num_threads);
/**
 * @brief Perform the operation.
 *
 * @param index Index instance.
 * @param data Input data buffer.
 * @param dimension Vector dimensionality.
 * @return 0 on success, -1 on error.
 */
int diskann_insert(GV_DiskANNIndex *index, const float *data, size_t dimension);
int diskann_search(const GV_DiskANNIndex *index, const float *query, size_t dimension,
                       size_t k, GV_DiskANNResult *results);
/**
 * @brief Search with a per-query beam width (recall/latency override).
 *
 * @param index Index instance.
 * @param query Query vector.
 * @param dimension Vector dimensionality.
 * @param k Number of neighbours requested.
 * @param results Output array of at least k results.
 * @param beam_width Beam width for this query (0 = configured default).
 * @return Number of results written, or -1 on error.
 */
int diskann_search_ex(const GV_DiskANNIndex *index, const float *query, size_t dimension,
                      size_t k, GV_DiskANNResult *results, size_t beam_width);
/**
 * @brief Pre-filtered search: only ids in @p allowed are returned.
 *
 * The graph walk still traverses non-matching nodes for connectivity but
 * collects only allowed ids, so it keeps searching until k matches are found.
 *
 * @param index Index instance.
 * @param query Query vector.
 * @param dimension Vector dimensionality.
 * @param k Number of neighbours requested.
 * @param results Output array of at least k results.
 * @param allowed Bitmap of permitted ids (NULL = no filter).
 * @param beam_width Beam width for this query (0 = configured default).
 * @return Number of results written, or -1 on error.
 */
int diskann_search_filtered(const GV_DiskANNIndex *index, const float *query, size_t dimension,
                            size_t k, GV_DiskANNResult *results,
                            const GV_IdBitmap *allowed, size_t beam_width);
/**
 * @brief Delete an item.
 *
 * @param index Index instance.
 * @param vector_index Index value.
 * @return 0 on success, -1 on error.
 */
int diskann_delete(GV_DiskANNIndex *index, size_t vector_index);
/**
 * @brief Retrieve statistics.
 *
 * @param index Index instance.
 * @param stats Output statistics structure.
 * @return 0 on success, -1 on error.
 */
int diskann_get_stats(const GV_DiskANNIndex *index, GV_DiskANNStats *stats);
/**
 * @brief Save state to a file.
 *
 * @param index Index instance.
 * @param filepath Filesystem path.
 * @return 0 on success, -1 on error.
 */
int diskann_save(const GV_DiskANNIndex *index, const char *filepath);
GV_DiskANNIndex *diskann_load(const char *filepath, const GV_DiskANNConfig *config);

/** Attach a shared LRU byte cache (not owned by the index). */
void diskann_attach_page_cache(GV_DiskANNIndex *index, struct GV_DiskPageCache *cache);

/**
 * @brief Return the number of stored items.
 *
 * @param index Index instance.
 * @return Count value.
 */
size_t diskann_count(const GV_DiskANNIndex *index);

/**
 * @brief Remove tombstoned ids from all adjacency lists (cheap compaction).
 *
 * Keeps recall high under churn without a full rebuild; does not renumber ids
 * or reclaim vector slots (those stay available via the insert free list).
 *
 * @param index Index instance.
 * @return 0 on success, -1 on error.
 */
int diskann_consolidate(GV_DiskANNIndex *index);

#ifdef __cplusplus
}
#endif
#endif
