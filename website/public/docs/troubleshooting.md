# GigaVector Troubleshooting Guide

Comprehensive troubleshooting guide for common issues and their solutions.

## Table of Contents

1. [Quick Diagnosis](#quick-diagnosis)
2. [Build and Installation Issues](#build-and-installation-issues)
3. [Runtime Errors](#runtime-errors)
4. [Performance Issues](#performance-issues)
5. [Memory Issues](#memory-issues)
6. [Data Integrity Issues](#data-integrity-issues)
7. [LLM Integration Issues](#llm-integration-issues)
8. [Network Issues](#network-issues)
9. [Debugging Tools](#debugging-tools)
10. [Getting Help](#getting-help)

---

## Quick Diagnosis

### Health Check

```c
GV_DetailedStats stats;
if (gv_db_get_detailed_stats(db, &stats) == 0) {
    printf("Health Status: %s\n", 
           stats.health_status == 0 ? "Healthy" : 
           stats.health_status == -1 ? "Degraded" : "Unhealthy");
    printf("Memory Usage: %.2f GB\n", stats.memory.total_bytes / 1e9);
    printf("Deleted Ratio: %.2f%%\n", stats.deleted_ratio * 100);
}
```

### Common Symptoms and Solutions

| Symptom | Likely Cause | Quick Fix |
|---------|--------------|-----------|
| High memory usage | Large dataset, high M/ef | Reduce HNSW parameters |
| Slow queries | Low ef_search, no SIMD | Increase ef_search, check SIMD |
| Low recall | Index not tuned | Increase ef_construction, M |
| WAL growing large | Infrequent snapshots | Take more snapshots |
| Crashes on insert | Memory limit exceeded | Increase limits or reduce dataset |

---

## Build and Installation Issues

### Compilation Errors

#### "undefined reference to `pthread_*`"

**Cause:** Missing pthread library linkage

**Solution:**
```bash
# Add -pthread flag
gcc -o app app.c -Lbuild/lib -lGigaVector -lm -pthread
```

#### "curl/curl.h: No such file or directory"

**Cause:** libcurl development headers not installed

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install libcurl4-openssl-dev

# RHEL/Fedora
sudo dnf install libcurl-devel

# macOS
brew install curl
```

#### SIMD compilation errors

**Cause:** CPU doesn't support requested SIMD instructions

**Solution:**
```bash
# Remove specific SIMD flags, use native
make clean
make CFLAGS="-march=native" lib
```

### Library Not Found at Runtime

**Error:** `error while loading shared libraries: libGigaVector.so: cannot open shared object file`

**Solution:**
```bash
# Add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/gigavector/build/lib:$LD_LIBRARY_PATH

# Or install system-wide
sudo cp build/lib/libGigaVector.so /usr/local/lib/
sudo ldconfig
```

### Python Bindings Issues

#### Import Error

**Error:** `ModuleNotFoundError: No module named 'gigavector'`

**Solution:**
```bash
# Install from source
cd python
pip install .

# Or use virtual environment
python -m venv venv
source venv/bin/activate
pip install .
```

---

## Runtime Errors

### Database Open Failures

#### "Failed to open database"

**Diagnosis:**
```c
GV_Database *db = gv_db_open("db.gvdb", 128, GV_INDEX_TYPE_HNSW);
if (db == NULL) {
    // Check file permissions
    if (access("db.gvdb", R_OK | W_OK) != 0) {
        fprintf(stderr, "Permission denied\n");
    }
    // Check disk space
    struct statvfs vfs;
    if (statvfs(".", &vfs) == 0) {
        printf("Free space: %lu MB\n", 
               (vfs.f_bavail * vfs.f_frsize) / (1024 * 1024));
    }
}
```

**Solutions:**
- Check file permissions: `chmod 644 db.gvdb`
- Check disk space: `df -h`
- Verify dimension matches: Check existing database dimension

#### Dimension Mismatch

**Error:** Database dimension doesn't match

**Solution:**
```c
// Always verify dimension before opening
size_t saved_dimension = get_database_dimension("db.gvdb");
if (saved_dimension != expected_dimension) {
    fprintf(stderr, "Dimension mismatch: saved=%zu, expected=%zu\n",
            saved_dimension, expected_dimension);
    return -1;
}
```

### Vector Insert Failures

#### "Failed to add vector"

**Diagnosis:**
```c
int result = gv_db_add_vector(db, vector, dimension);
if (result != 0) {
    // Check memory
    GV_DBStats stats;
    gv_db_get_stats(db, &stats);
    printf("Memory usage: %.2f GB\n", stats.memory_bytes / 1e9);
    
    // Check resource limits
    if (stats.vector_count >= MAX_VECTORS) {
        fprintf(stderr, "Maximum vector count reached\n");
    }
}
```

**Solutions:**
- Check memory limits: Increase `max_memory_bytes`
- Check vector count: Increase `max_vectors`
- Verify vector data: Check for NaN/Inf values

#### Invalid Vector Data

**Diagnosis:**
```c
// Validate vector before insertion
int is_valid_vector(const float *data, size_t dimension) {
    for (size_t i = 0; i < dimension; i++) {
        if (!isfinite(data[i])) {
            fprintf(stderr, "Invalid value at index %zu: %f\n", i, data[i]);
            return 0;
        }
    }
    return 1;
}
```

---

## Performance Issues

### Slow Queries

#### Diagnosis

```c
// Measure query latency
struct timespec start, end;
clock_gettime(CLOCK_MONOTONIC, &start);
int count = gv_db_search(db, query, k, distance_type, results);
clock_gettime(CLOCK_MONOTONIC, &end);

double latency_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                    (end.tv_nsec - start.tv_nsec) / 1e6;
printf("Query latency: %.2f ms\n", latency_ms);
```

#### Solutions

**1. Increase ef_search:**
```c
// For HNSW, higher ef_search = better recall but slower
GV_HNSWConfig config = {
    .ef_search = 200  // Increase from default 50
};
```

**2. Check SIMD support:**
```bash
# Verify SIMD is enabled
gcc -march=native -Q --help=target | grep mavx

# Rebuild with SIMD
make clean
make lib
```

**3. Enable cosine normalization:**
```c
// For cosine distance queries
gv_db_set_cosine_normalized(db, 1);
```

**4. Use appropriate index:**
```c
// For large datasets, consider IVFPQ
GV_IndexType index = gv_index_suggest(dimension, expected_count);
```

### High Memory Usage

#### Diagnosis

```c
GV_DetailedStats stats;
gv_db_get_detailed_stats(db, &stats);
printf("Memory breakdown:\n");
printf("  SoA storage: %.2f MB\n", stats.memory.soa_storage_bytes / 1e6);
printf("  Index: %.2f MB\n", stats.memory.index_bytes / 1e6);
printf("  Metadata: %.2f MB\n", stats.memory.metadata_index_bytes / 1e6);
printf("  Total: %.2f MB\n", stats.memory.total_bytes / 1e6);
```

#### Solutions

**1. Reduce HNSW parameters:**
```c
GV_HNSWConfig config = {
    .M = 16,              // Reduce from 32
    .ef_construction = 100 // Reduce from 200
};
```

**2. Enable quantization:**
```c
// Use scalar quantization (4 bytes -> 1 byte)
// Reduces memory by 75%
```

**3. Set memory limits:**
```c
GV_ResourceLimits limits = {
    .max_memory_bytes = 8ULL * 1024 * 1024 * 1024  // 8 GB
};
```

### Low Recall

#### Diagnosis

```c
GV_RecallMetrics recall;
// ... run recall test ...
printf("Average recall: %.2f%%\n", recall.avg_recall * 100);
printf("Min recall: %.2f%%\n", recall.min_recall * 100);
```

#### Solutions

**1. Increase ef_search:**
```c
// Higher ef_search improves recall
GV_HNSWConfig config = {
    .ef_search = 200  // Increase from default
};
```

**2. Increase construction parameters:**
```c
GV_HNSWConfig config = {
    .M = 64,                // Increase from 32
    .ef_construction = 400   // Increase from 200
};
```

**3. Retrain IVFPQ:**
```c
// Retrain with more clusters
GV_IVFPQConfig config = {
    .n_clusters = 512  // Increase from 256
};
```

---

## Memory Issues

### Memory Leaks

#### Detection

```bash
# Use valgrind
valgrind --leak-check=full --show-leak-kinds=all ./your_app

# Use AddressSanitizer
make CFLAGS="-fsanitize=address" test
```

#### Common Causes

1. **Not freeing LLM responses:**
```c
GV_LLMResponse response;
gv_llm_generate_response(llm, messages, 1, NULL, &response);
// ... use response ...
gv_llm_response_free(&response);  // Must free!
```

2. **Not closing database:**
```c
GV_Database *db = gv_db_open(...);
// ... use db ...
gv_db_close(db);  // Must close!
```

### Out of Memory

#### Diagnosis

```c
// Check available memory
struct sysinfo info;
sysinfo(&info);
printf("Available RAM: %.2f GB\n", 
       (info.freeram * info.mem_unit) / 1e9);
```

#### Solutions

1. **Reduce dataset size**
2. **Enable quantization**
3. **Use memory-mapped files:**
```c
GV_Database *db = gv_db_open_mmap("db.gvdb", dimension, index_type);
```

---

## Data Integrity Issues

### Corrupted Database

#### Detection

```c
// Try to open and verify
GV_Database *db = gv_db_open("db.gvdb", dimension, index_type);
if (db == NULL) {
    fprintf(stderr, "Database may be corrupted\n");
}
```

#### Recovery

```bash
# Restore from backup
cp backup/snapshot_20240115.gvdb db.gvdb

# WAL will be replayed automatically
```

### WAL Corruption

#### Symptoms
- Database fails to open
- WAL file exists but is invalid

#### Recovery

```bash
# Remove corrupted WAL (data loss risk)
rm db.gvdb.wal

# Or restore from snapshot
cp backup/snapshot.gvdb db.gvdb
```

### Missing Vectors

#### Diagnosis

```c
GV_DBStats stats;
gv_db_get_stats(db, &stats);
printf("Expected: %zu, Actual: %zu\n", expected_count, stats.vector_count);
```

#### Solutions

1. **Check WAL replay:**
   - WAL should replay automatically on open
   - Check WAL file exists and is readable

2. **Verify inserts succeeded:**
```c
int result = gv_db_add_vector(db, vector, dimension);
if (result != 0) {
    fprintf(stderr, "Insert failed\n");
    // Handle error
}
```

---

## LLM Integration Issues

### API Key Errors

#### "Invalid API key"

**Diagnosis:**
```c
GV_LLMConfig config = {
    .provider = GV_LLM_PROVIDER_OPENAI,
    .api_key = getenv("OPENAI_API_KEY")
};

if (config.api_key == NULL) {
    fprintf(stderr, "OPENAI_API_KEY not set\n");
    return -1;
}

GV_LLM *llm = gv_llm_create(&config);
if (llm == NULL) {
    const char *error = gv_llm_get_last_error(llm);
    fprintf(stderr, "Error: %s\n", error);
}
```

**Solutions:**
- Verify API key format: OpenAI keys start with `sk-`
- Check environment variable is set: `echo $OPENAI_API_KEY`
- Verify key is valid: Test with curl

### Network Timeouts

#### "Request timeout"

**Solutions:**

1. **Increase timeout:**
```c
GV_LLMConfig config = {
    .timeout_seconds = 60  // Increase from default 30
};
```

2. **Check network connectivity:**
```bash
curl -I https://api.openai.com/v1/models
```

3. **Retry with exponential backoff:**
```c
int retry_count = 0;
int max_retries = 3;
while (retry_count < max_retries) {
    int result = gv_llm_generate_response(llm, messages, count, NULL, &response);
    if (result == GV_LLM_SUCCESS) break;
    
    if (result == GV_LLM_ERROR_TIMEOUT) {
        sleep(1 << retry_count);  // Exponential backoff
        retry_count++;
    } else {
        break;  // Other error, don't retry
    }
}
```

### Response Parsing Errors

#### "Failed to parse response"

**Diagnosis:**
```c
int result = gv_llm_generate_response(llm, messages, 1, "json_object", &response);
if (result == GV_LLM_ERROR_PARSE_FAILED) {
    const char *error = gv_llm_get_last_error(llm);
    fprintf(stderr, "Parse error: %s\n", error);
    // Log raw response for debugging
}
```

**Solutions:**
- Use `"json_object"` response format for structured data
- Validate LLM response format
- Check LLM model supports JSON mode

---

## Network Issues

### Connection Failures

#### Diagnosis

```c
// Test connectivity
CURL *curl = curl_easy_init();
curl_easy_setopt(curl, CURLOPT_URL, "https://api.openai.com/v1/models");
CURLcode res = curl_easy_perform(curl);
if (res != CURLE_OK) {
    fprintf(stderr, "Connection failed: %s\n", curl_easy_strerror(res));
}
```

#### Solutions

1. **Check firewall rules**
2. **Verify DNS resolution**
3. **Check proxy settings:**
```c
curl_easy_setopt(curl, CURLOPT_PROXY, "http://proxy:port");
```

### SSL/TLS Errors

#### "SSL certificate problem"

**Solutions:**

1. **Update CA certificates:**
```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install ca-certificates

# Update curl's CA bundle
```

2. **Verify certificate:**
```c
curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
curl_easy_setopt(curl, CURLOPT_CAINFO, "/etc/ssl/certs/ca-certificates.crt");
```

---

## Debugging Tools

### GDB Debugging

```bash
# Compile with debug symbols
make CFLAGS="-g -O0" lib

# Run with GDB
gdb ./your_app
(gdb) break gv_db_search
(gdb) run
(gdb) print db->vector_count
```

### Valgrind

```bash
# Check for memory leaks
valgrind --leak-check=full ./your_app

# Check for errors
valgrind --tool=memcheck ./your_app
```

### Performance Profiling

```bash
# Use perf
perf record -g ./your_app
perf report

# Use callgrind
valgrind --tool=callgrind ./your_app
kcachegrind callgrind.out.*
```

### Logging

```c
// Enable debug logging
#define DEBUG 1

#ifdef DEBUG
#define DEBUG_LOG(fmt, ...) fprintf(stderr, "[DEBUG] " fmt "\n", ##__VA_ARGS__)
#else
#define DEBUG_LOG(fmt, ...)
#endif

DEBUG_LOG("Searching with k=%zu", k);
```

---

## Getting Help

### Before Asking for Help

1. **Check this guide** for common issues
2. **Review logs** for error messages
3. **Run diagnostics** (health check, stats)
4. **Test with minimal example** to isolate issue

### Information to Provide

When reporting issues, include:

1. **System information:**
   - OS and version
   - GigaVector version
   - Compiler and version

2. **Error details:**
   - Exact error message
   - Stack trace (if available)
   - Logs

3. **Reproduction:**
   - Minimal code to reproduce
   - Steps to reproduce
   - Expected vs actual behavior

4. **Configuration:**
   - Index type and parameters
   - Dataset size and dimension
   - Resource limits

### Resources

- **GitHub Issues**: https://github.com/jaywyawhare/GigaVector/issues
- **Documentation**: See `docs/` directory
- **Examples**: See `docs/examples/`

---

**Remember:** Most issues can be resolved by checking logs, verifying configuration, and reviewing this guide. When in doubt, start with a minimal test case to isolate the problem.

