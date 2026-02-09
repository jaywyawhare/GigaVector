#ifndef GIGAVECTOR_GV_COMPRESSION_H
#define GIGAVECTOR_GV_COMPRESSION_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GV_COMPRESS_NONE = 0,
    GV_COMPRESS_LZ4 = 1,     /* Fast compression */
    GV_COMPRESS_ZSTD = 2,    /* High ratio compression */
    GV_COMPRESS_SNAPPY = 3   /* Very fast, moderate ratio */
} GV_CompressionType;

typedef struct {
    GV_CompressionType type;
    int level;                /* Compression level (1-9, default: 1) */
    size_t min_size;          /* Min payload size to compress (default: 64 bytes) */
} GV_CompressionConfig;

typedef struct {
    uint64_t total_compressed;
    uint64_t total_decompressed;
    uint64_t bytes_in;
    uint64_t bytes_out;
    double avg_ratio;          /* Average compression ratio */
} GV_CompressionStats;

typedef struct GV_Compressor GV_Compressor;

void gv_compression_config_init(GV_CompressionConfig *config);
GV_Compressor *gv_compression_create(const GV_CompressionConfig *config);
void gv_compression_destroy(GV_Compressor *comp);

/* Compress data. Returns compressed size, or 0 on error. Caller allocates output buffer. */
size_t gv_compress(GV_Compressor *comp, const void *input, size_t input_len,
                    void *output, size_t output_capacity);

/* Decompress data. Returns decompressed size, or 0 on error. */
size_t gv_decompress(GV_Compressor *comp, const void *input, size_t input_len,
                      void *output, size_t output_capacity);

/* Get max compressed size for a given input size (for buffer allocation) */
size_t gv_compress_bound(const GV_Compressor *comp, size_t input_len);

/* Stats */
int gv_compression_get_stats(const GV_Compressor *comp, GV_CompressionStats *stats);

#ifdef __cplusplus
}
#endif
#endif
