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

void gv_diskann_config_init(GV_DiskANNConfig *config);
GV_DiskANNIndex *gv_diskann_create(size_t dimension, const GV_DiskANNConfig *config);
void gv_diskann_destroy(GV_DiskANNIndex *index);

/* Build index from batch of vectors */
int gv_diskann_build(GV_DiskANNIndex *index, const float *data, size_t count, size_t dimension);

/* Incremental insert */
int gv_diskann_insert(GV_DiskANNIndex *index, const float *data, size_t dimension);

/* Search */
int gv_diskann_search(const GV_DiskANNIndex *index, const float *query, size_t dimension,
                       size_t k, GV_DiskANNResult *results);

/* Delete (mark as deleted) */
int gv_diskann_delete(GV_DiskANNIndex *index, size_t vector_index);

/* Stats */
int gv_diskann_get_stats(const GV_DiskANNIndex *index, GV_DiskANNStats *stats);

/* Persistence */
int gv_diskann_save(const GV_DiskANNIndex *index, const char *filepath);
GV_DiskANNIndex *gv_diskann_load(const char *filepath, const GV_DiskANNConfig *config);

size_t gv_diskann_count(const GV_DiskANNIndex *index);

#ifdef __cplusplus
}
#endif
#endif
