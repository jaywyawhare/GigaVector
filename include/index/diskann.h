#ifndef GIGAVECTOR_GV_DISKANN_H
#define GIGAVECTOR_GV_DISKANN_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t max_degree;        /* Max graph degree R (default: 64) */
    float alpha;              /* Pruning parameter alpha (default: 1.2) */
    size_t build_beam_width;  /* Beam width during build (default: 128) */
    size_t search_beam_width; /* Beam width during search (default: 64) */
    size_t pq_dim;            /* PQ compressed dimension for in-memory nav (default: 0 = auto) */
    const char *data_path;    /* Path for on-disk vector storage */
    size_t cache_size_mb;     /* Memory cache size in MB (default: 256) */
    size_t sector_size;       /* Disk sector alignment (default: 4096) */
} GV_DiskANNConfig;

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

/**
 * @brief Return the number of stored items.
 *
 * @param index Index instance.
 * @return Count value.
 */
size_t diskann_count(const GV_DiskANNIndex *index);

#ifdef __cplusplus
}
#endif
#endif
