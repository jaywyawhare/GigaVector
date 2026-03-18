# Troubleshooting

## Quick Diagnosis

```c
GV_DetailedStats stats;
if (gv_db_get_detailed_stats(db, &stats) == 0) {
    printf("Health: %s\n",
           stats.health_status == 0 ? "Healthy" :
           stats.health_status == -1 ? "Degraded" : "Unhealthy");
    printf("Memory: %.2f GB\n", stats.memory.total_bytes / 1e9);
    printf("Deleted Ratio: %.2f%%\n", stats.deleted_ratio * 100);
}
```

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

**"undefined reference to `pthread_*`"** -- Add `-pthread` flag:
```bash
gcc -o app app.c -Lbuild/lib -lGigaVector -lm -pthread
```

**"curl/curl.h: No such file or directory"** -- Install libcurl headers:
```bash
sudo apt-get install libcurl4-openssl-dev   # Debian/Ubuntu
sudo dnf install libcurl-devel              # RHEL/Fedora
brew install curl                           # macOS
```

**SIMD compilation errors** -- Rebuild with native arch detection:
```bash
make clean && make CFLAGS="-march=native" lib
```

### Library Not Found at Runtime

```bash
export LD_LIBRARY_PATH=/path/to/gigavector/build/lib:$LD_LIBRARY_PATH
# Or install system-wide:
sudo cp build/lib/libGigaVector.so /usr/local/lib/ && sudo ldconfig
```

### Python Import Error

```bash
cd python && pip install .
```

---

## Runtime Errors

### Database Open Failures

- Check file permissions: `chmod 644 db.gvdb`
- Check disk space: `df -h`
- Verify dimension matches the existing database -- a mismatch silently fails

### Vector Insert Failures

- Check memory limits: increase `max_memory_bytes`
- Check vector count: increase `max_vectors`
- Validate vector data for NaN/Inf before insertion

---

## Performance Tuning

### Slow Queries / Low Recall

These share the same knobs. Turning them up improves recall at the cost of speed; turning them down does the reverse.

```c
GV_HNSWConfig config = {
    .M = 64,                // default 32; higher = better recall, more memory
    .ef_construction = 400, // default 200; higher = better graph quality
    .ef_search = 200        // default 50; higher = better recall, slower search
};
```

Other levers:
- **Enable SIMD:** `make clean && make lib` (uses `-march=native` by default)
- **Cosine normalization:** `gv_db_set_cosine_normalized(db, 1);` avoids redundant normalization per query
- **Use IVFPQ for large datasets:** `gv_index_suggest(dimension, expected_count)` picks an appropriate index
- **Retrain IVFPQ with more clusters** if recall is still low (e.g. `n_clusters = 512`)

### High Memory Usage

1. **Reduce M / ef_construction** (see above)
2. **Enable scalar quantization** (4 bytes -> 1 byte per component, ~75% reduction)
3. **Set a hard limit:**
   ```c
   GV_ResourceLimits limits = {
       .max_memory_bytes = 8ULL * 1024 * 1024 * 1024  // 8 GB
   };
   ```
4. **Use memory-mapped files:**
   ```c
   GV_Database *db = gv_db_open_mmap("db.gvdb", dimension, index_type);
   ```

---

## Data Integrity

### Corrupted Database / WAL

Restore from the most recent snapshot:
```bash
cp backup/snapshot_20240115.gvdb db.gvdb
# WAL replays automatically on next open
```

If only the WAL is corrupt and no snapshot exists, removing it is the last resort (data since last snapshot is lost):
```bash
rm db.gvdb.wal
```

### Missing Vectors

- WAL should replay automatically on open -- verify the WAL file exists and is readable
- Always check return values from `gv_db_add_vector`; a non-zero return means the insert failed

---

## LLM and Network Issues

| Problem | Fix |
|---------|-----|
| `OPENAI_API_KEY` not set / invalid | Set env var; OpenAI keys start with `sk-` |
| Request timeout | Increase `config.timeout_seconds` (default 30) |
| Retry on transient failure | Use exponential backoff: `sleep(1 << retry_count)` |
| Response parse error | Request `"json_object"` format; verify model supports JSON mode |
| SSL certificate error | `sudo apt-get install ca-certificates` |
| Connection failure behind proxy | `curl_easy_setopt(curl, CURLOPT_PROXY, "http://proxy:port")` |

**Memory leak note:** Always call `gv_llm_response_free(&response)` after using an LLM response, and `gv_db_close(db)` when done with a database.
