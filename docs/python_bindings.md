# Python Bindings Guide

This guide explains how GigaVector's Python bindings work and how to use them effectively.

## Table of Contents

1. [Overview](#overview)
2. [How the FFI Works](#how-the-ffi-works)
3. [Using Python Bindings](#using-python-bindings)
4. [Best Practices](#best-practices)
5. [Extending the Bindings](#extending-the-bindings)
6. [Troubleshooting](#troubleshooting)

## Overview

GigaVector's Python bindings use **CFFI (C Foreign Function Interface)** to call the underlying C library. This provides:

- **High Performance:** Direct calls to optimized C code
- **Type Safety:** Automatic type conversion between Python and C
- **Memory Safety:** Automatic memory management for Python objects
- **Easy Integration:** Pythonic API that hides C complexity

## How the FFI Works

### Architecture

```
Python Code
    ↓
gigavector._core.py (Python wrapper classes)
    ↓
gigavector._ffi.py (CFFI interface)
    ↓
libGigaVector.so (C shared library)
    ↓
C Implementation
```

### CFFI Interface Definition

The FFI interface is defined in `_ffi.py` using C declarations:

```python
from cffi import FFI

ffi = FFI()

# Define C types and functions
ffi.cdef("""
    typedef struct {
        size_t dimension;
        float *data;
        GV_Metadata *metadata;
    } GV_Vector;
    
    GV_Database *gv_db_open(const char *filepath, size_t dimension, GV_IndexType index_type);
    int gv_db_add_vector(GV_Database *db, const float *data, size_t dimension);
    // ... more declarations
""")

# Load the shared library
lib = ffi.dlopen("libGigaVector.so")
```

### Type Conversions

CFFI automatically handles conversions:

| C Type | Python Type | Conversion |
|--------|-------------|------------|
| `float[]` | `list[float]` | `ffi.new("float[]", [1.0, 2.0, 3.0])` |
| `const char*` | `str` | `"text".encode()` |
| `GV_Database*` | Opaque pointer | Managed by wrapper |
| `int` | `int` | Automatic |
| `size_t` | `int` | Automatic |

### Memory Management

**Important:** The C library manages vector data. Python objects are wrappers:

```python
# Python creates a buffer
buf = ffi.new("float[]", [0.1, 0.2, 0.3])

# C library takes ownership of the data
lib.gv_db_add_vector(db, buf, 3)

# Python buffer is automatically freed when out of scope
# But the C library keeps a copy of the data
```

## Using Python Bindings

### Basic Usage

```python
from gigavector import Database, IndexType, DistanceType

# Open database
db = Database.open("example.db", dimension=128, index=IndexType.HNSW)

# Add vector
db.add_vector([0.1] * 128, metadata={"id": "1"})

# Search
results = db.search([0.1] * 128, k=10, distance=DistanceType.EUCLIDEAN)

# Close
db.close()
```

### Context Manager (Recommended)

```python
# Automatically closes database
with Database.open("example.db", dimension=128) as db:
    db.add_vector([0.1] * 128)
    results = db.search([0.1] * 128, k=10)
    # Database closed automatically on exit
```

### Working with Metadata

```python
# Add vector with metadata
db.add_vector(
    [0.1] * 128,
    metadata={
        "id": "12345",
        "category": "electronics",
        "price": "99.99"
    }
)

# Search with filter
results = db.search(
    query, k=10,
    filter_metadata=("category", "electronics")
)
```

### Batch Operations

```python
# Batch insertion (more efficient)
vectors = [[random.random() for _ in range(128)] for _ in range(1000)]
db.add_vectors(vectors)

# Batch search
queries = [[random.random() for _ in range(128)] for _ in range(10)]
all_results = db.search_batch(queries, k=5)
```

### Configuration

```python
from gigavector import HNSWConfig, IVFPQConfig

# HNSW with custom parameters
hnsw_config = HNSWConfig(
    M=32,                    # More connections
    ef_construction=200,     # Construction quality
    ef_search=50,           # Search quality
    use_binary_quant=True,   # Enable quantization
    quant_rerank=20          # Rerank candidates
)

db = Database.open(
    "db.db", dimension=128, index=IndexType.HNSW,
    hnsw_config=hnsw_config
)

# IVFPQ configuration
ivfpq_config = IVFPQConfig(
    nlist=256,
    m=16,
    nprobe=16,
    default_rerank=32
)

db = Database.open(
    None, dimension=128, index=IndexType.IVFPQ,
    ivfpq_config=ivfpq_config
)

# Must train before use
training_data = [[random.random() for _ in range(128)] for _ in range(1000)]
db.train_ivfpq(training_data)
```

## Best Practices

### 1. Always Use Context Managers

```python
# Good
with Database.open("db.db", 128) as db:
    # Use database
    pass

# Bad - may leak resources
db = Database.open("db.db", 128)
# Forgot to close!
```

### 2. Check Vector Dimensions

```python
# The library checks dimensions automatically
try:
    db.add_vector([0.1] * 64)  # Wrong dimension
except ValueError as e:
    print(f"Error: {e}")  # "expected vector of dim 128, got 64"
```

### 3. Handle Errors Properly

```python
try:
    db = Database.open("db.db", 128)
    db.add_vector(vector)
except RuntimeError as e:
    print(f"Database error: {e}")
    # Handle error appropriately
except ValueError as e:
    print(f"Invalid input: {e}")
    # Handle validation error
```

### 4. Use Batch Operations

```python
# Good - efficient
db.add_vectors(vectors)
results = db.search_batch(queries, k=10)

# Less efficient - many FFI calls
for vec in vectors:
    db.add_vector(vec)
```

### 5. Pre-allocate Lists for Results

```python
# Results are returned as Python lists
# No need to pre-allocate, but be aware of memory usage
results = db.search(query, k=1000)  # Large k uses more memory
```

### 6. Monitor Resource Usage

```python
# Check memory usage
memory_bytes = db.get_memory_usage()
print(f"Memory: {memory_bytes / 1024 / 1024:.2f} MB")

# Check resource limits
limits = db.get_resource_limits()
print(f"Max memory: {limits['max_memory_bytes']}")
print(f"Max vectors: {limits['max_vectors']}")
```

### 7. Use Appropriate Data Types

```python
# Good - list of floats
vector = [0.1, 0.2, 0.3, ...]

# Also works - numpy array
import numpy as np
vector = np.array([0.1, 0.2, 0.3, ...], dtype=np.float32)
db.add_vector(vector.tolist())  # Convert to list

# Good - Python dict for metadata
metadata = {"key": "value"}

# Bad - complex types in metadata
metadata = {"key": [1, 2, 3]}  # Only strings supported
```

## Extending the Bindings

### Understanding the FFI Layer

The FFI layer (`_ffi.py`) defines the C interface:

```python
# C type definitions
ffi.cdef("""
    typedef struct {
        size_t dimension;
        float *data;
    } GV_Vector;
    
    int gv_db_add_vector(GV_Database *db, const float *data, size_t dimension);
""")

# Load library
lib = ffi.dlopen("libGigaVector.so")
```

### Adding New Functions

To add a new function:

1. **Add C declaration to `_ffi.py`:**
```python
ffi.cdef("""
    int gv_db_new_function(GV_Database *db, const float *data);
""")
```

2. **Add Python wrapper in `_core.py`:**
```python
class Database:
    def new_function(self, data: Sequence[float]) -> int:
        """New function wrapper."""
        buf = ffi.new("float[]", list(data))
        rc = lib.gv_db_new_function(self._db, buf)
        if rc != 0:
            raise RuntimeError("gv_db_new_function failed")
        return rc
```

### Type Conversion Helpers

Common conversion patterns:

```python
# Python list to C float array
buf = ffi.new("float[]", list(vector))

# Python string to C string
c_str = key.encode("utf-8")

# Python dict to C metadata arrays
keys = [ffi.new("char[]", k.encode()) for k in metadata.keys()]
values = [ffi.new("char[]", v.encode()) for v in metadata.values()]
keys_c = ffi.new("const char * []", keys)
values_c = ffi.new("const char * []", values)

# C struct to Python dict
def _metadata_to_dict(meta_ptr):
    result = {}
    cur = meta_ptr
    while cur != ffi.NULL:
        key = ffi.string(cur.key).decode("utf-8")
        value = ffi.string(cur.value).decode("utf-8")
        result[key] = value
        cur = cur.next
    return result
```

### Error Handling Pattern

```python
def safe_call(func, *args):
    """Wrapper for C function calls with error handling."""
    rc = func(*args)
    if rc != 0:
        raise RuntimeError(f"{func.__name__} failed with code {rc}")
    return rc

# Usage
safe_call(lib.gv_db_add_vector, db, buf, dimension)
```

## Troubleshooting

### Common Issues

**1. "libGigaVector.so not found"**

**Cause:** Shared library not found

**Solutions:**
```python
# Check library path
from pathlib import Path
import gigavector._ffi as ffi_module
print(ffi_module._load_lib.__doc__)  # Shows search paths

# Build library first
# cd /path/to/GigaVector
# make lib
```

**2. "FFI type mismatch" errors**

**Cause:** C declaration doesn't match actual C function

**Solution:** Ensure `_ffi.py` declarations match C headers exactly

**3. Memory leaks**

**Cause:** Not closing databases or holding references

**Solution:**
```python
# Always use context managers
with Database.open("db.db", 128) as db:
    # Use database
    pass

# Or explicitly close
db = Database.open("db.db", 128)
try:
    # Use database
    pass
finally:
    db.close()
```

**4. Slow performance**

**Cause:** Too many individual FFI calls

**Solution:**
```python
# Use batch operations
db.add_vectors(vectors)  # One FFI call
# Instead of:
for vec in vectors:
    db.add_vector(vec)  # Many FFI calls
```

**5. Type errors**

**Cause:** Wrong Python types passed to C functions

**Solution:**
```python
# Ensure vectors are lists of floats
vector = [float(x) for x in vector]  # Convert to floats

# Ensure metadata values are strings
metadata = {k: str(v) for k, v in metadata.items()}
```

### Debugging Tips

**Enable FFI debugging:**
```python
import cffi
cffi.verifier.set_verbose(True)
```

**Check library loading:**
```python
from gigavector import _ffi
print(f"Library loaded: {_ffi.lib}")
```

**Inspect C objects:**
```python
# Database handle is an opaque pointer
db = Database.open("db.db", 128)
print(f"Database handle: {db._db}")  # C pointer (opaque)
```

**Monitor FFI calls:**
```python
import time

# Measure FFI overhead
start = time.time()
for _ in range(1000):
    db.add_vector(vector)
ffi_time = time.time() - start
print(f"FFI overhead: {ffi_time:.4f}s")
```

### Performance Optimization

**1. Minimize FFI Calls**
```python
# Good - single FFI call
db.add_vectors(vectors)

# Bad - many FFI calls
for vec in vectors:
    db.add_vector(vec)
```

**2. Pre-allocate Buffers (Advanced)**
```python
# For very high-performance scenarios
import cffi
ffi = cffi.FFI()

# Pre-allocate buffer
buf = ffi.new("float[]", dimension)
# Reuse buffer for multiple operations
```

**3. Use Appropriate Batch Sizes**
```python
# Optimal batch size depends on your use case
# Typically 100-1000 vectors per batch
batch_size = 500
for i in range(0, len(vectors), batch_size):
    batch = vectors[i:i+batch_size]
    db.add_vectors(batch)
```

## Summary

- GigaVector Python bindings use CFFI for high-performance C integration
- Always use context managers for resource management
- Use batch operations to minimize FFI overhead
- Handle errors appropriately with try/except
- Monitor resource usage for large datasets
- The C library manages vector data; Python provides convenient wrappers

For more information, see:
- [Usage Guide](usage.md) for general usage patterns
- [C API Guide](c_api_guide.md) for understanding the underlying C API
- [Performance Tuning Guide](performance.md) for optimization tips

