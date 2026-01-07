# GigaVector

<p align="center">
  <img src="https://raw.githubusercontent.com/jaywyawhare/GigaVector/master/gigavector-logo.png" alt="GigaVector Logo" width="200" />
</p>

<p align="center">
  <a href="https://pepy.tech/projects/gigavector">
    <img src="https://static.pepy.tech/personalized-badge/gigavector?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads" alt="PyPI Downloads" />
  </a>
</p>

A fast, modular vector database in C with optional Python bindings.

## Features
- KD-Tree, HNSW, and IVFPQ indexes
- Euclidean and cosine distances
- Rich per-vector metadata (multiple key-value pairs)
- Persistence with snapshot plus WAL replay
- Python bindings via `cffi` (published on PyPI as `gigavector`)

## Build (C library)

### Using Make (default)
```bash
make lib        # builds static and shared libraries into build/lib/
make c-test     # runs C tests (needs LD_LIBRARY_PATH=build/lib)
```

### Using CMake
```bash
# Configure build
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build library and executables
cmake --build build

# Run tests
cd build && ctest

# Install (optional)
cmake --install build --prefix /usr/local
```

**CMake Options:**
- `-DBUILD_SHARED_LIBS=ON/OFF` - Build shared library (default: ON)
- `-DBUILD_TESTS=ON/OFF` - Build test executables (default: ON)
- `-DBUILD_BENCHMARKS=ON/OFF` - Build benchmark executables (default: ON)
- `-DENABLE_SANITIZERS=ON/OFF` - Enable sanitizers (ASAN, TSAN, UBSAN) (default: OFF)
- `-DENABLE_COVERAGE=ON/OFF` - Enable code coverage (default: OFF)

**Example with options:**
```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=ON \
    -DBUILD_BENCHMARKS=ON \
    -DENABLE_SANITIZERS=OFF
cmake --build build
```

The CMake build system automatically detects and enables available SIMD optimizations (SSE4.2, AVX2, AVX-512F, FMA) for optimal performance.

## Python bindings
From PyPI:
```bash
pip install gigavector
```

From source:
```bash
cd python
python -m pip install .
```

## Quick start (Python)
```python
from gigavector import Database, DistanceType, IndexType

with Database.open("example.db", dimension=4, index=IndexType.KDTREE) as db:
    db.add_vector([1, 2, 3, 4], metadata={"tag": "sample", "owner": "user"})
    hits = db.search([1, 2, 3, 4], k=1, distance=DistanceType.EUCLIDEAN)
    print(hits[0].distance, hits[0].vector.metadata)
```

Persistence:
```python
db.save("example.db")          # snapshot
# On reopen, WAL is replayed automatically
with Database.open("example.db", dimension=4, index=IndexType.KDTREE) as db:
    ...
```

IVFPQ training (dimension must match):
```python
train = [[(i % 10) / 10.0 for _ in range(8)] for i in range(256)]
with Database.open(None, dimension=8, index=IndexType.IVFPQ) as db:
    db.train_ivfpq(train)
    db.add_vector([0.5] * 8)
```

## Documentation

### Getting Started
- [Usage Guide](docs/usage.md) - Comprehensive guide for using GigaVector
- [Python Bindings Guide](docs/python_bindings.md) - How Python bindings work and best practices
- [C API Guide](docs/c_api_guide.md) - Complete C API reference and usage patterns

### Examples
- [Basic Usage Examples](docs/examples/basic_usage.md) - Getting started with C and Python APIs
- [Advanced Features](docs/examples/advanced_features.md) - Advanced patterns and optimization techniques

### Performance
- [Performance Tuning Guide](docs/performance.md) - Index selection, parameter tuning, and optimization

## License
This project is licensed under the DBaJ-NC-CFL [License](./LICENCE).