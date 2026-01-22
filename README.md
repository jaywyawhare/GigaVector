# GigaVector

<p align="center">
  <img src="https://raw.githubusercontent.com/jaywyawhare/GigaVector/master/gigavector-logo.png" alt="GigaVector Logo" width="200" />
</p>

<p align="center">
  <a href="https://pepy.tech/projects/gigavector">
    <img src="https://static.pepy.tech/personalized-badge/gigavector?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads" alt="PyPI Downloads" />
  </a>
</p>

**GigaVector** is a high-performance, production-ready vector database library written in C with optional Python bindings. Designed for applications requiring fast approximate nearest neighbor search, semantic memory management, and LLM integration.

## Key Features

### Core Capabilities
- **Multiple Index Algorithms**: KD-Tree, HNSW (Hierarchical Navigable Small Worlds), IVFPQ (Inverted File with Product Quantization), and Sparse Index
- **Distance Metrics**: Euclidean and cosine distance with optimized implementations
- **Rich Metadata**: Support for multiple key-value metadata pairs per vector with efficient filtering
- **Persistence**: Snapshot-based persistence with Write-Ahead Logging (WAL) for durability
- **Memory Management**: Structure-of-Arrays storage, quantization options, and configurable resource limits

### Advanced Features
- **Semantic Memory Layer**: Extract, store, and consolidate memories from conversations with importance scoring
- **LLM Integration**: Support for OpenAI, Anthropic, and Google LLMs for memory extraction and generation
- **Embedding Services**: Integration with OpenAI, Google, and HuggingFace embedding APIs
- **Context Graphs**: Build entity-relationship graphs for context-aware retrieval
- **Production Ready**: Monitoring, statistics, health checks, and comprehensive error handling

### Performance
- **SIMD Optimizations**: Automatic detection and use of SSE4.2, AVX2, and AVX-512F
- **Thread-Safe**: Concurrent read operations with external write synchronization
- **Memory Efficient**: Optimized data structures and optional quantization
- **Scalable**: Supports millions of vectors with configurable resource limits

## Build (C library)

### Using Make (default)
```bash
make            # builds everything (library + main executable)
make lib        # builds static and shared libraries into build/lib/
make c-test     # runs C tests (needs LD_LIBRARY_PATH=build/lib)
```

**For complete build and test instructions including LLM tests, see [Build and Test Guide](docs/build_and_test.md)**

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

## Environment Variables

GigaVector supports various environment variables for API keys and configuration. 

**Quick Setup:**
```bash
# Copy the example file and fill in your API keys
cp .env.example .env
# Edit .env with your actual API keys
```

**Required for Tests:**
- `OPENAI_API_KEY` - For LLM and embedding tests
- `ANTHROPIC_API_KEY` - For Anthropic/Claude LLM tests

**Optional:**
- `GOOGLE_API_KEY` - For Google embeddings
- `GV_WAL_DIR` - Override WAL directory location

See [`.env.example`](.env.example) for a complete list with descriptions, or [API Keys Documentation](docs/api_keys_required.md) for detailed information.

## Documentation

### Getting Started
- [Usage Guide](docs/usage.md) - Comprehensive guide for using GigaVector
- [Build and Test Guide](docs/build_and_test.md) - Complete build instructions and testing
- [Python Bindings Guide](docs/python_bindings.md) - Python API documentation and best practices
- [C API Guide](docs/c_api_guide.md) - C API usage patterns and examples

### Reference Documentation
- [API Reference](docs/api_reference.md) - Complete API reference with detailed function documentation
- [Architecture Documentation](docs/architecture.md) - Deep dive into system architecture and design
- [API Keys Guide](docs/api_keys_required.md) - Environment variables and API key configuration

### Production Guides
- [Deployment Guide](docs/deployment.md) - Production deployment, scaling, and operations
- [Security Guide](docs/security.md) - Security best practices and hardening
- [Troubleshooting Guide](docs/troubleshooting.md) - Common issues and solutions

### Examples and Tutorials
- [Basic Usage Examples](docs/examples/basic_usage.md) - Getting started with C and Python APIs
- [Advanced Features](docs/examples/advanced_features.md) - Advanced patterns and optimization techniques

### Performance and Optimization
- [Performance Tuning Guide](docs/performance.md) - Index selection, parameter tuning, and optimization

### Contributing
- [Contributing Guidelines](CONTRIBUTING.md) - How to contribute to GigaVector

## License
This project is licensed under the DBaJ-NC-CFL [License](./LICENCE).