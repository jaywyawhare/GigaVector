# GigaVector Production Deployment Guide

Comprehensive guide for deploying GigaVector in production environments.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Performance Tuning](#performance-tuning)
5. [Monitoring and Observability](#monitoring-and-observability)
6. [Backup and Recovery](#backup-and-recovery)
7. [Scaling Strategies](#scaling-strategies)
8. [Security Hardening](#security-hardening)
9. [Disaster Recovery](#disaster-recovery)
10. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware Requirements

**Minimum:**
- CPU: 2 cores, 2.0 GHz
- RAM: 4 GB
- Storage: 10 GB SSD
- Network: 100 Mbps

**Recommended (Production):**
- CPU: 8+ cores, 3.0+ GHz (with AVX2/AVX-512 support)
- RAM: 16+ GB
- Storage: 100+ GB NVMe SSD
- Network: 1 Gbps+

### Software Requirements

- **Operating System**: Linux (Ubuntu 20.04+, RHEL 8+, Debian 11+)
- **Compiler**: GCC 9+ or Clang 10+ with C99 support
- **Libraries**:
  - `libcurl` (for LLM/embedding features)
  - `pthread` (for threading)
  - `m` (math library)

### SIMD Support

GigaVector automatically detects and uses SIMD instructions:
- **SSE4.2**: Minimum requirement
- **AVX2**: Recommended for better performance
- **AVX-512F**: Optimal for high-dimensional vectors

Check SIMD support:
```bash
gcc -march=native -Q --help=target | grep -E "mavx|msse"
```

---

## Installation

### Building from Source

#### Using Make (Recommended)

```bash
# Clone repository
git clone https://github.com/jaywyawhare/GigaVector.git
cd GigaVector

# Build optimized release
make clean
make lib

# Install system-wide (optional)
sudo cp build/lib/libGigaVector.so /usr/local/lib/
sudo cp build/lib/libGigaVector.a /usr/local/lib/
sudo ldconfig
```

#### Using CMake

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
# From PyPI
pip install gigavector

# From source
cd python
pip install .
```

### Docker Deployment

```dockerfile
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    libcurl4-openssl-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN make lib && \
    cp build/lib/libGigaVector.so /usr/local/lib/ && \
    ldconfig

# Python bindings
RUN cd python && pip install .

CMD ["python3", "app.py"]
```

---

## Configuration

### Environment Variables

Create `.env` file:

```bash
# Required for LLM features
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Optional
GOOGLE_API_KEY=...
GV_WAL_DIR=/var/lib/gigavector/wal
GV_MAX_MEMORY_GB=16
GV_LOG_LEVEL=INFO
```

### Database Configuration

#### Index Selection

```c
// Auto-select based on workload
GV_IndexType index = gv_index_suggest(dimension, expected_count);

// Or manually configure
GV_HNSWConfig hnsw_config = {
    .M = 32,                // Higher for better recall, more memory
    .ef_construction = 400, // Higher for better quality
    .ef_search = 100,       // Higher for better recall
    .level_mult = 1.0
};
```

#### Resource Limits

```c
GV_ResourceLimits limits = {
    .max_memory_bytes = 16ULL * 1024 * 1024 * 1024,  // 16 GB
    .max_vectors = 10000000,                          // 10M vectors
    .max_concurrent_operations = 1000
};
```

### WAL Configuration

Write-Ahead Logging ensures durability:

```c
// WAL is automatically enabled when filepath is provided
GV_Database *db = gv_db_open("/data/vectors.gvdb", 128, GV_INDEX_TYPE_HNSW);

// Custom WAL directory
setenv("GV_WAL_DIR", "/var/lib/gigavector/wal", 1);
```

---

## Performance Tuning

### Index-Specific Tuning

#### HNSW Parameters

| Parameter | Low Memory | Balanced | High Quality |
|-----------|-----------|----------|--------------|
| M | 16 | 32 | 64 |
| ef_construction | 100 | 200 | 400 |
| ef_search | 50 | 100 | 200 |

**Trade-offs:**
- Higher values = Better recall, more memory, slower inserts
- Lower values = Faster, less memory, lower recall

#### IVFPQ Parameters

```c
GV_IVFPQConfig config = {
    .n_clusters = 256,      // More clusters = better quality, more memory
    .n_subvectors = 8,      // More subvectors = better quality
    .n_bits = 8            // 8 bits = good balance
};
```

### Memory Optimization

1. **Enable cosine normalization** for cosine distance:
   ```c
   gv_db_set_cosine_normalized(db, 1);
   ```

2. **Use quantization** for large datasets:
   - Scalar quantization: 4 bytes → 1 byte (75% reduction)
   - Binary quantization: 4 bytes → 1 bit (96% reduction)

3. **Structure-of-Arrays storage**: Automatically enabled for better cache locality

### Query Optimization

1. **Batch operations**: Use batch embedding APIs
2. **Pre-filter**: Apply metadata filters before vector search
3. **Tune ef_search**: Balance between recall and latency

---

## Monitoring and Observability

### Statistics Collection

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

### Metrics to Monitor

**Critical:**
- Vector count
- Memory usage
- Query latency (p50, p95, p99)
- Error rate
- WAL size

**Important:**
- Recall metrics
- Insert rate
- Query throughput (QPS)
- Deleted vector ratio

### Logging

Implement custom logging:

```c
void log_operation(const char *op, int result, double latency_ms) {
    if (result != 0) {
        fprintf(stderr, "[ERROR] %s failed: %d (%.2f ms)\n", op, result, latency_ms);
    } else {
        fprintf(stdout, "[INFO] %s succeeded (%.2f ms)\n", op, latency_ms);
    }
}
```

### Health Checks

```c
int check_health(GV_Database *db) {
    GV_DetailedStats stats;
    if (gv_db_get_detailed_stats(db, &stats) != 0) {
        return -1;  // Unhealthy
    }
    
    // Check memory usage
    if (stats.memory.total_bytes > MAX_MEMORY_BYTES) {
        return -2;  // Degraded
    }
    
    // Check deleted ratio
    if (stats.deleted_ratio > 0.1) {
        return -2;  // Degraded - needs compaction
    }
    
    return 0;  // Healthy
}
```

---

## Backup and Recovery

### Snapshot Strategy

```c
// Regular snapshots
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

### Backup Schedule

**Recommended:**
- Full snapshot: Daily
- Incremental (WAL): Every 15 minutes
- Retention: 7 days daily, 30 days weekly

### Recovery Procedure

```c
// 1. Restore from latest snapshot
GV_Database *db = gv_db_open("/backup/snapshot_20240115_120000.gvdb", 
                              dimension, index_type);

// 2. WAL is automatically replayed on open
// 3. Verify integrity
GV_DBStats stats;
gv_db_get_stats(db, &stats);
printf("Recovered %zu vectors\n", stats.vector_count);
```

### Disaster Recovery Plan

1. **Identify failure**: Check logs, health endpoints
2. **Stop writes**: Disable write operations
3. **Restore snapshot**: From latest backup
4. **Replay WAL**: Automatic on open
5. **Verify data**: Check vector counts, run queries
6. **Resume operations**: Re-enable writes

---

## Scaling Strategies

### Vertical Scaling

**When to scale up:**
- Memory usage > 80%
- CPU usage consistently > 70%
- Query latency increasing

**Actions:**
- Increase RAM
- Add CPU cores
- Upgrade to faster storage (NVMe)

### Horizontal Scaling

#### Sharding Strategy

```c
// Shard by hash of vector ID
int shard_id = hash(vector_id) % num_shards;
GV_Database *shard = shards[shard_id];
gv_db_add_vector(shard, vector, dimension);
```

#### Replication

1. **Primary-Replica**: One write, multiple reads
2. **Multi-Master**: All nodes accept writes (requires conflict resolution)

### Load Balancing

**Query Distribution:**
- Round-robin for read-only workloads
- Consistent hashing for sharded data
- Health-aware routing

---

## Security Hardening

### API Key Management

**Never:**
- Commit API keys to version control
- Log API keys
- Transmit keys over unencrypted connections

**Do:**
- Use environment variables or secret management (Vault, AWS Secrets Manager)
- Rotate keys regularly
- Use least-privilege keys

### Network Security

1. **TLS/SSL**: Encrypt all API communications
2. **Firewall**: Restrict access to database ports
3. **VPN**: Use VPN for internal communications

### Access Control

```c
// Implement application-level authentication
int authenticate_request(const char *api_key) {
    // Validate against database or service
    return validate_api_key(api_key);
}
```

### Data Encryption

- **At rest**: Use encrypted filesystems (LUKS, BitLocker)
- **In transit**: TLS 1.2+ for all network communications
- **In memory**: Consider encrypted memory for sensitive data

---

## Disaster Recovery

### RTO/RPO Targets

- **RTO (Recovery Time Objective)**: < 1 hour
- **RPO (Recovery Point Objective)**: < 15 minutes (with WAL)

### Backup Testing

**Monthly:**
1. Restore from backup
2. Verify data integrity
3. Measure recovery time
4. Document issues

### Runbook

1. **Detection**: Automated alerts on failures
2. **Assessment**: Determine scope of failure
3. **Containment**: Stop writes, isolate affected systems
4. **Recovery**: Restore from backup, replay WAL
5. **Verification**: Test queries, check metrics
6. **Communication**: Notify stakeholders
7. **Post-mortem**: Document and improve

---

## Troubleshooting

### Common Issues

#### High Memory Usage

**Symptoms:** OOM kills, slow performance

**Solutions:**
1. Reduce `ef_construction` and `M` for HNSW
2. Enable quantization
3. Reduce `max_memory_bytes` limit
4. Scale up memory

#### Slow Queries

**Symptoms:** High p95 latency

**Solutions:**
1. Increase `ef_search` (trades latency for recall)
2. Enable cosine normalization
3. Check CPU usage (may need more cores)
4. Verify SIMD support

#### WAL Growing Large

**Symptoms:** Disk space issues, slow opens

**Solutions:**
1. Take more frequent snapshots
2. Reduce WAL retention
3. Compact database

#### Low Recall

**Symptoms:** Missing relevant results

**Solutions:**
1. Increase `ef_search`
2. Increase `ef_construction` and `M`
3. Consider different index type
4. Check vector quality

### Debugging

Enable verbose logging:

```c
// Set log level
setenv("GV_LOG_LEVEL", "DEBUG", 1);

// Check error messages
const char *error = gv_llm_get_last_error(llm);
if (error) {
    fprintf(stderr, "LLM error: %s\n", error);
}
```

### Performance Profiling

```bash
# Profile with perf
perf record -g ./your_application
perf report

# Profile with valgrind
valgrind --tool=callgrind ./your_application
kcachegrind callgrind.out.*
```

---

## Production Checklist

Before deploying to production:

- [ ] System requirements met
- [ ] Index parameters tuned for workload
- [ ] Resource limits configured
- [ ] Monitoring and alerting set up
- [ ] Backup strategy implemented
- [ ] Disaster recovery plan documented
- [ ] Security hardening applied
- [ ] Load testing completed
- [ ] Documentation reviewed
- [ ] Team trained on operations

---

For additional support, see [Troubleshooting Guide](troubleshooting.md) or open an issue on GitHub.

