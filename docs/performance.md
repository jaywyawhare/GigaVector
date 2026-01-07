# GigaVector Performance Tuning Guide

This guide provides recommendations for optimizing GigaVector performance based on your use case, data characteristics, and hardware capabilities.

## Table of Contents

1. [Index Selection Guidelines](#index-selection-guidelines)
2. [Parameter Tuning Recommendations](#parameter-tuning-recommendations)
3. [SIMD Optimization Notes](#simd-optimization-notes)
4. [Memory Usage Optimization](#memory-usage-optimization)
5. [Benchmark Results and Comparisons](#benchmark-results-and-comparisons)

## Index Selection Guidelines

GigaVector supports multiple index types, each optimized for different scenarios:

### KD-Tree (`GV_INDEX_TYPE_KDTREE`)

**Best for:**
- Small to medium datasets (< 1M vectors)
- Exact nearest neighbor search
- Low-dimensional vectors (< 100 dimensions)
- Static or rarely updated datasets
- Memory-constrained environments

**Characteristics:**
- O(log n) search time for balanced trees
- Exact results (no approximation)
- Memory efficient (Structure-of-Arrays layout)
- Fast insertion for small datasets
- Tree rebalancing may be needed for large datasets

**When to avoid:**
- High-dimensional vectors (> 100 dimensions) - curse of dimensionality
- Very large datasets (> 1M vectors) - tree depth becomes problematic
- Frequent updates - tree rebalancing overhead

### HNSW (`GV_INDEX_TYPE_HNSW`)

**Best for:**
- Large datasets (millions to billions of vectors)
- High-dimensional vectors (100+ dimensions)
- Approximate nearest neighbor search with high recall
- Dynamic datasets with frequent insertions
- Production systems requiring sub-millisecond search latency

**Characteristics:**
- O(log n) search time with high recall
- Hierarchical graph structure
- Excellent scalability
- Supports binary quantization for faster candidate selection
- ACORN-style exploration for filtered search

**When to avoid:**
- Very small datasets (< 10K vectors) - overhead not justified
- Exact search requirements - HNSW is approximate
- Memory-constrained environments - higher memory overhead

### IVFPQ (`GV_INDEX_TYPE_IVFPQ`)

**Best for:**
- Very large datasets (millions+ vectors)
- High-dimensional vectors (64+ dimensions)
- Memory-constrained environments
- Batch search scenarios
- Production systems with trained models

**Characteristics:**
- Product Quantization for memory efficiency
- Inverted File index for fast candidate retrieval
- Requires training phase before use
- Supports scalar quantization for additional compression
- Configurable trade-off between accuracy and speed

**When to avoid:**
- Small datasets (< 100K vectors) - training overhead not justified
- Low-dimensional vectors (< 32 dimensions) - PQ overhead
- Exact search requirements - IVFPQ is approximate
- Frequently changing data - requires retraining

### Sparse Index (`GV_INDEX_TYPE_SPARSE`)

**Best for:**
- Sparse vectors (most dimensions are zero)
- Text embeddings and bag-of-words representations
- High-dimensional sparse data

**Characteristics:**
- Optimized for sparse vector operations
- Memory efficient for sparse data
- Fast search for sparse queries

## Parameter Tuning Recommendations

### HNSW Parameters

#### `M` (Number of connections per node)
- **Default:** 16
- **Range:** 4-64
- **Higher values:**
  - Better recall and search quality
  - Higher memory usage
  - Slower insertion time
- **Lower values:**
  - Faster insertion
  - Lower memory usage
  - Potentially lower recall
- **Recommendation:** Start with 16, increase to 32-48 for higher quality, decrease to 8-12 for faster insertion

#### `efConstruction` (Candidate list size during construction)
- **Default:** 200
- **Range:** 50-500
- **Higher values:**
  - Better index quality and recall
  - Slower insertion time
  - Higher memory during construction
- **Lower values:**
  - Faster insertion
  - Potentially lower recall
- **Recommendation:** Use 200-400 for high quality, 100-200 for faster insertion

#### `efSearch` (Candidate list size during search)
- **Default:** 50
- **Range:** k to 500
- **Higher values:**
  - Better recall
  - Slower search time
- **Lower values:**
  - Faster search
  - Potentially lower recall
- **Recommendation:** Set to 2-4x your typical `k` value. For k=10, use efSearch=20-40

#### Binary Quantization
- **`use_binary_quant`:** Enable for 2-3x faster search with minimal recall loss
- **`quant_rerank`:** Number of candidates to rerank with exact distance
  - Set to 2-3x your `k` value
  - 0 disables reranking (fastest, lower accuracy)
  - Higher values improve accuracy at cost of speed

#### ACORN Exploration (for filtered search)
- **`use_acorn`:** Enable for better recall in filtered searches
- **`acorn_hops`:** Exploration depth (1-2 recommended)
  - 1 hop: Fast, good recall
  - 2 hops: Better recall, slower

### IVFPQ Parameters

#### `nlist` (Number of coarse centroids)
- **Default:** 64
- **Range:** 16-4096
- **Higher values:**
  - Better accuracy
  - More memory for centroids
  - Slower training
- **Lower values:**
  - Faster training and search
  - Lower memory
  - Potentially lower accuracy
- **Recommendation:** 
  - Small datasets (< 1M): 64-256
  - Medium datasets (1M-10M): 256-1024
  - Large datasets (> 10M): 1024-4096
  - Rule of thumb: sqrt(N) where N is dataset size

#### `m` (Number of subquantizers)
- **Default:** 8
- **Range:** Must divide dimension evenly
- **Higher values:**
  - Better accuracy
  - More memory for codebooks
  - Slightly slower search
- **Lower values:**
  - Faster search
  - Lower memory
  - Potentially lower accuracy
- **Recommendation:** 
  - Dimension 64: m=8 or m=16
  - Dimension 128: m=8, m=16, or m=32
  - Dimension 256: m=16, m=32, or m=64
  - Must evenly divide dimension

#### `nbits` (Bits per subquantizer code)
- **Default:** 8
- **Range:** 4-8
- **8 bits:** Standard, good accuracy
- **4 bits:** 2x memory reduction, lower accuracy
- **Recommendation:** Use 8 bits unless memory is critical

#### `nprobe` (Lists to probe during search)
- **Default:** 4
- **Range:** 1 to nlist
- **Higher values:**
  - Better recall
  - Slower search
- **Lower values:**
  - Faster search
  - Potentially lower recall
- **Recommendation:** 
  - Start with nlist/16 to nlist/8
  - Increase for better recall
  - Decrease for faster search

#### `default_rerank` (Rerank pool size)
- **Default:** 32
- **Range:** 0 (disabled) to several hundred
- **Higher values:**
  - Better accuracy
  - Slower search
- **0 (disabled):**
  - Fastest search
  - Lower accuracy
- **Recommendation:** Set to 2-4x your typical `k` value

#### Scalar Quantization
- **`use_scalar_quant`:** Enable for additional 2-4x memory reduction
- **`scalar_quant_config.bits`:** 4-8 bits per component
  - 8 bits: Minimal accuracy loss
  - 4 bits: Significant memory savings, noticeable accuracy loss
- **Recommendation:** Enable for very large datasets where memory is critical

#### Oversampling Factor
- **`oversampling_factor`:** Multiplier for candidate selection
- **Default:** 1.0
- **Range:** 1.0-3.0
- **Higher values:** Better recall, slower search
- **Recommendation:** Use 1.5-2.0 for high-recall scenarios

## SIMD Optimization Notes

GigaVector automatically detects and uses available CPU SIMD features for optimized distance calculations.

### Supported SIMD Features

- **SSE4.2:** Basic vector operations
- **AVX2:** 256-bit vector operations (2x speedup)
- **AVX-512F:** 512-bit vector operations (4x speedup on supported CPUs)
- **FMA:** Fused multiply-add operations

### Compilation Flags

The build system automatically enables SIMD optimizations:

**Makefile:**
```bash
make lib  # Automatically uses -march=native -msse4.2 -mavx2 -mavx512f -mfma
```

**CMake:**
```bash
cmake -B build  # Automatically detects and enables available SIMD features
```

### Runtime Detection

GigaVector detects CPU features at runtime using `gv_cpu_detect_features()`. The library automatically uses the best available SIMD implementation.

### Performance Impact

- **SSE4.2:** 1.5-2x speedup over scalar code
- **AVX2:** 2-3x speedup over scalar code
- **AVX-512F:** 3-4x speedup over scalar code (when available)

### Verification

Check which SIMD features are available:
```c
unsigned int features = gv_cpu_detect_features();
if (gv_cpu_has_feature(GV_CPU_FEATURE_AVX2)) {
    // AVX2 is available
}
```

## Memory Usage Optimization

### Structure-of-Arrays (SoA) Layout

GigaVector uses SoA storage for efficient memory access patterns:
- Better cache locality
- SIMD-friendly memory layout
- Reduced memory fragmentation

### Memory Usage Estimates

**KD-Tree:**
- Vectors: `N * dimension * sizeof(float)` bytes
- Tree nodes: `N * (2 * sizeof(void*) + sizeof(size_t) + sizeof(float))` bytes
- Total: ~`N * (dimension * 4 + 32)` bytes

**HNSW:**
- Vectors: `N * dimension * sizeof(float)` bytes
- Graph structure: `N * M * sizeof(size_t)` bytes (M = connections per node)
- Total: ~`N * (dimension * 4 + M * 8)` bytes

**IVFPQ:**
- Compressed vectors: `N * m * sizeof(uint8_t)` bytes (m = subquantizers)
- Centroids: `nlist * dimension * sizeof(float)` bytes
- Codebooks: `m * 256 * (dimension/m) * sizeof(float)` bytes
- Total: ~`N * m + nlist * dimension * 4 + m * 256 * (dimension/m) * 4` bytes

### Memory Optimization Tips

1. **Use IVFPQ for large datasets:** 10-100x memory reduction vs. exact storage
2. **Enable scalar quantization:** Additional 2-4x memory reduction
3. **Reduce HNSW M parameter:** Lower memory usage at cost of recall
4. **Use binary quantization:** Minimal memory overhead, significant speedup
5. **Batch operations:** Process vectors in batches to control peak memory

### Memory Profiling

Monitor memory usage:
```c
GV_DBStats stats;
gv_db_get_stats(db, &stats);
// Check stats.memory_usage_bytes
```

## Benchmark Results and Comparisons

### Typical Performance Characteristics

**KD-Tree:**
- Search latency: 0.1-1ms (1K-100K vectors)
- Insertion: 0.01-0.1ms per vector
- Memory: Low overhead
- Recall: 100% (exact)

**HNSW:**
- Search latency: 0.1-5ms (1M-1B vectors)
- Insertion: 0.1-1ms per vector
- Memory: Moderate overhead (M * 8 bytes per vector)
- Recall: 95-99% (with proper efSearch tuning)

**IVFPQ:**
- Search latency: 0.5-10ms (1M-1B vectors)
- Insertion: 0.01-0.1ms per vector
- Memory: Very low overhead (m bytes per vector)
- Recall: 85-95% (with proper nprobe tuning)

### Performance Tuning Workflow

1. **Start with defaults:** Use default parameters for your index type
2. **Measure baseline:** Run benchmarks with your data
3. **Tune for accuracy:** Increase quality parameters (efSearch, nprobe, rerank)
4. **Tune for speed:** Decrease quality parameters if latency is critical
5. **Enable optimizations:** Enable binary quantization, scalar quantization if applicable
6. **Iterate:** Fine-tune based on your accuracy/speed requirements

### Benchmarking Tools

Use the provided benchmark executables:
```bash
# Build benchmarks
make bench

# Run SIMD benchmarks
./build/bench/benchmark_simd

# Run IVFPQ benchmarks
./build/bench/benchmark_ivfpq

# Run recall benchmarks
./build/bench/benchmark_ivfpq_recall
```

### Expected Performance Improvements

With proper tuning and SIMD optimizations:
- **2-4x speedup** from SIMD (AVX2/AVX-512)
- **2-3x speedup** from binary quantization (HNSW)
- **10-100x memory reduction** from IVFPQ + scalar quantization
- **10-50% recall improvement** from proper parameter tuning

## Best Practices Summary

1. **Choose the right index:** Match index type to your dataset size and requirements
2. **Tune incrementally:** Start with defaults, adjust one parameter at a time
3. **Enable SIMD:** Always build with SIMD optimizations enabled
4. **Use quantization:** Enable binary/scalar quantization when appropriate
5. **Monitor metrics:** Track recall, latency, and memory usage
6. **Profile your workload:** Benchmark with your actual data and queries
7. **Balance trade-offs:** Optimize for your specific accuracy/speed/memory requirements

For specific tuning questions, refer to the API documentation or run benchmarks with your data to find optimal parameters.

