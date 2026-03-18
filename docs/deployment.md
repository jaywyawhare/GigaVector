# Deployment

## System Requirements

### Hardware

**Minimum:** 2 cores / 4 GB RAM / 10 GB SSD
**Recommended:** 8+ cores (AVX2/AVX-512) / 16+ GB RAM / NVMe SSD

### Software

- **OS**: Linux (Ubuntu 20.04+, RHEL 8+, Debian 11+)
- **Compiler**: GCC 9+ or Clang 10+ (C99)
- **Libraries**: `libcurl`, `pthread`, `m`

### SIMD Support

GigaVector auto-detects SIMD instructions (SSE4.2 minimum, AVX2 recommended, AVX-512F optimal).

```bash
gcc -march=native -Q --help=target | grep -E "mavx|msse"
```

---

## Installation

### Using Make

```bash
git clone https://github.com/jaywyawhare/GigaVector.git
cd GigaVector
make clean && make lib

# Install system-wide (optional)
sudo cp build/lib/libGigaVector.so /usr/local/lib/
sudo cp build/lib/libGigaVector.a /usr/local/lib/
sudo ldconfig
```

### Using CMake

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON \
      -DBUILD_TESTS=OFF \
      -DBUILD_BENCHMARKS=OFF \
      ..
make -j$(nproc)
sudo make install
```

### Python Bindings

```bash
pip install gigavector      # from PyPI
cd python && pip install .  # from source
```

### Docker

```dockerfile
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    build-essential libcurl4-openssl-dev python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN make lib && \
    cp build/lib/libGigaVector.so /usr/local/lib/ && \
    ldconfig

RUN cd python && pip install .
CMD ["python3", "app.py"]
```

---

## Configuration

### WAL Configuration

```c
// WAL is automatically enabled when filepath is provided
GV_Database *db = gv_db_open("/data/vectors.gvdb", 128, GV_INDEX_TYPE_HNSW);

// Custom WAL directory
setenv("GV_WAL_DIR", "/var/lib/gigavector/wal", 1);
```

### Index Configuration

```c
GV_HNSWConfig hnsw_config = {
    .M = 32,                // Higher for better recall, more memory
    .ef_construction = 400, // Higher for better quality
    .ef_search = 100,       // Higher for better recall
    .level_mult = 1.0
};
```

### Resource Limits

```c
GV_ResourceLimits limits = {
    .max_memory_bytes = 16ULL * 1024 * 1024 * 1024,  // 16 GB
    .max_vectors = 10000000,                          // 10M vectors
    .max_concurrent_operations = 1000
};
```

---

## Performance Tuning

### HNSW Parameters

| Parameter | Low Memory | Balanced | High Quality |
|-----------|-----------|----------|--------------|
| M | 16 | 32 | 64 |
| ef_construction | 100 | 200 | 400 |
| ef_search | 50 | 100 | 200 |

Higher values = better recall, more memory, slower inserts.

### IVFPQ Parameters

```c
GV_IVFPQConfig config = {
    .n_clusters = 256,      // More clusters = better quality, more memory
    .n_subvectors = 8,      // More subvectors = better quality
    .n_bits = 8             // 8 bits = good balance
};
```

### Memory Optimization

- **Cosine normalization**: `gv_db_set_cosine_normalized(db, 1);`
- **Scalar quantization**: 4 bytes to 1 byte (75% reduction)
- **Binary quantization**: 4 bytes to 1 bit (96% reduction)
- **SoA storage**: Automatically enabled for cache locality

---

## Monitoring

```c
GV_DetailedStats stats;
if (gv_db_get_detailed_stats(db, &stats) == 0) {
    printf("Vector count: %zu\n", stats.vector_count);
    printf("Memory usage: %.2f GB\n", stats.memory.total_bytes / 1e9);
    printf("QPS: %.2f\n", stats.qps);
    printf("Avg latency: %.2f ms\n", stats.avg_latency_ms);
    printf("Recall: %.2f%%\n", stats.recall.avg_recall * 100);
    printf("Health: %s\n", stats.health_status == 0 ? "Healthy" : "Degraded");
}
```

---

## Backup and Recovery

### Snapshots

```c
int save_snapshot(GV_Database *db, const char *backup_path) {
    char timestamp[64];
    time_t now = time(NULL);
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", localtime(&now));

    char snapshot_path[256];
    snprintf(snapshot_path, sizeof(snapshot_path), "%s/snapshot_%s.gvdb",
             backup_path, timestamp);

    return gv_db_save(db, snapshot_path);
}
```

### Recovery

```c
// Restore from snapshot; WAL is automatically replayed on open
GV_Database *db = gv_db_open("/backup/snapshot_20240115_120000.gvdb",
                              dimension, index_type);

GV_DBStats stats;
gv_db_get_stats(db, &stats);
printf("Recovered %zu vectors\n", stats.vector_count);
```

---

## Troubleshooting

| Problem | Solutions |
|---------|----------|
| **High memory / OOM** | Reduce `M` and `ef_construction`; enable quantization; lower `max_memory_bytes` |
| **Slow queries** | Tune `ef_search`; enable cosine normalization; verify SIMD support |
| **WAL growing large** | Take more frequent snapshots; compact database |
| **Low recall** | Increase `ef_search`, `ef_construction`, `M`; try a different index type |

### Debugging

```c
setenv("GV_LOG_LEVEL", "DEBUG", 1);

const char *error = gv_llm_get_last_error(llm);
if (error) {
    fprintf(stderr, "LLM error: %s\n", error);
}
```

```bash
perf record -g ./your_application && perf report
valgrind --tool=callgrind ./your_application
```

---

For additional support, see [Troubleshooting Guide](troubleshooting.md) or open an issue on GitHub.
