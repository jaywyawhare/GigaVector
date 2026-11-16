# Build and Test

## Prerequisites

1. **Build Tools:**
   - Linux: `gcc`, `make`, `cmake`, `libcurl-dev`
   - macOS: Xcode CLT + `brew install cmake curl`
   - Windows (source builds only): [MSYS2](https://www.msys2.org/) with `mingw-w64-x86_64-gcc`, `mingw-w64-x86_64-cmake`, `mingw-w64-x86_64-make` and `C:\msys64\mingw64\bin` on `PATH`. PyPI wheel users need nothing extra.

2. **API Keys (for LLM tests):**
   - `OPENAI_API_KEY` - Required for OpenAI LLM and embedding tests
   - `ANTHROPIC_API_KEY` - Required for Anthropic LLM tests
   - `GOOGLE_API_KEY` - Optional, for Google embedding tests

   See [`.env.example`](../.env.example) for setup instructions.

## Quick Start

### 1. Set Up Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API keys
# OPENAI_API_KEY=sk-your-key-here
# ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Or export them directly:
```bash
export OPENAI_API_KEY=sk-your-key-here
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 2. Build Everything

```bash
# Build the library and main executable
make

# Or build just the library
make lib
```

This will:
- Compile all source files
- Create static library: `build/lib/libGigaVector.a`
- Create shared library: `build/lib/libGigaVector.so`
- Build main executable: `build/main`

### 3. Run Standard C Tests

```bash
# Run all standard C tests
make c-test
```

This runs tests for:
- Database operations (`test_db`)
- Distance calculations (`test_distance`)
- Metadata (`test_metadata`)
- HNSW index (`test_hnsw`)
- IVFPQ index (`test_ivfpq`)
- Sparse vectors (`test_sparse`)
- WAL (Write-Ahead Log) (`test_wal`)
- Filters (`test_filter`)
- Advanced features (`test_advanced`)

### 4. Build and Run LLM Tests

The LLM tests (`test_llm.c`, `test_memory_llm.c`) are not included in the standard test suite. Build and run them manually:

```bash
# Ensure the library is built
make lib

# Build test_llm
gcc -O3 -g -Wall -Wextra -Iinclude -march=native -msse4.2 -mavx2 -mavx512f -mfma -pthread -fPIC -DHAVE_CURL \
    tests/test_llm.c -Lbuild/lib -lGigaVector -lm -pthread -lcurl \
    -Wl,-rpath,$(pwd)/build/lib -o build/test_llm

# Run test_llm (requires OPENAI_API_KEY and ANTHROPIC_API_KEY)
LD_LIBRARY_PATH=build/lib:${LD_LIBRARY_PATH} ./build/test_llm
```

Or use the Makefile helper:
```bash
# Build a single test
make c-test-single TEST=test_llm

# Run it
LD_LIBRARY_PATH=build/lib:${LD_LIBRARY_PATH} ./build/test_llm
```

### 5. Build and Run Memory LLM Tests

```bash
# Build test_memory_llm
gcc -O3 -g -Wall -Wextra -Iinclude -march=native -msse4.2 -mavx2 -mavx512f -mfma -pthread -fPIC -DHAVE_CURL \
    tests/test_memory_llm.c -Lbuild/lib -lGigaVector -lm -pthread -lcurl \
    -Wl,-rpath,$(pwd)/build/lib -o build/test_memory_llm

# Run test_memory_llm (requires OPENAI_API_KEY)
LD_LIBRARY_PATH=build/lib:${LD_LIBRARY_PATH} ./build/test_memory_llm
```

### 6. Build and Run Embedding Tests

```bash
# Build test_embedding
gcc -O3 -g -Wall -Wextra -Iinclude -march=native -msse4.2 -mavx2 -mavx512f -mfma -pthread -fPIC -DHAVE_CURL \
    tests/test_embedding.c -Lbuild/lib -lGigaVector -lm -pthread -lcurl \
    -Wl,-rpath,$(pwd)/build/lib -o build/test_embedding

# Run test_embedding (requires OPENAI_API_KEY, optional: GOOGLE_API_KEY)
LD_LIBRARY_PATH=build/lib:${LD_LIBRARY_PATH} ./build/test_embedding
```

## Complete Build and Test Script

Here's a complete script to build everything and run all tests:

```bash
#!/bin/bash
set -e

echo "=== Building GigaVector ==="
make clean
make lib

echo ""
echo "=== Running Standard C Tests ==="
make c-test

echo ""
echo "=== Building LLM Tests ==="
gcc -O3 -g -Wall -Wextra -Iinclude -march=native -msse4.2 -mavx2 -mavx512f -mfma -pthread -fPIC -DHAVE_CURL \
    tests/test_llm.c -Lbuild/lib -lGigaVector -lm -pthread -lcurl \
    -Wl,-rpath,$(pwd)/build/lib -o build/test_llm

gcc -O3 -g -Wall -Wextra -Iinclude -march=native -msse4.2 -mavx2 -mavx512f -mfma -pthread -fPIC -DHAVE_CURL \
    tests/test_memory_llm.c -Lbuild/lib -lGigaVector -lm -pthread -lcurl \
    -Wl,-rpath,$(pwd)/build/lib -o build/test_memory_llm

gcc -O3 -g -Wall -Wextra -Iinclude -march=native -msse4.2 -mavx2 -mavx512f -mfma -pthread -fPIC -DHAVE_CURL \
    tests/test_embedding.c -Lbuild/lib -lGigaVector -lm -pthread -lcurl \
    -Wl,-rpath,$(pwd)/build/lib -o build/test_embedding

echo ""
echo "=== Running LLM Tests ==="
if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY and/or ANTHROPIC_API_KEY not set. Some tests will be skipped."
fi
LD_LIBRARY_PATH=build/lib:${LD_LIBRARY_PATH} ./build/test_llm || echo "LLM tests completed with some skipped tests"

echo ""
echo "=== Running Memory LLM Tests ==="
LD_LIBRARY_PATH=build/lib:${LD_LIBRARY_PATH} ./build/test_memory_llm || echo "Memory LLM tests completed"

echo ""
echo "=== Running Embedding Tests ==="
LD_LIBRARY_PATH=build/lib:${LD_LIBRARY_PATH} ./build/test_embedding || echo "Embedding tests completed"

echo ""
echo "=== All Tests Complete ==="
```

Save this as `build_and_test.sh`, make it executable, and run:
```bash
chmod +x build_and_test.sh
./build_and_test.sh
```

## CMake Build

```bash
cmake -B build-cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build-cmake
cd build-cmake && ctest
```

Options (pass as `-D<OPTION>=ON|OFF`):

| Option | Default | Effect |
|--------|---------|--------|
| `BUILD_SHARED_LIBS` | ON | Build the shared library |
| `BUILD_TESTS` | ON | Build the test executables |
| `BUILD_BENCHMARKS` | ON | Build the benchmark executables |
| `ENABLE_SANITIZERS` | OFF | AddressSanitizer and UBSan |
| `ENABLE_TSAN` | OFF | ThreadSanitizer (not combinable with ASAN) |
| `ENABLE_COVERAGE` | OFF | Code coverage instrumentation |
| `ENABLE_NATIVE_OPTIMIZATIONS` | OFF | `-march=native -mtune=native` |
| `ENABLE_TCALLOC` | ON | Make the built-in slab allocator the default (`GV_ALLOC=libc` opts out at runtime) |

The HTTP REST server is compiled in only when CMake finds libmicrohttpd
(`libmicrohttpd-dev` on Debian/Ubuntu); the configure step prints whether it is enabled.
The plain `make lib` build does not enable it.

## Using a Local Build from Python

The Python package looks for the shared library in this order:

1. The path in `GIGAVECTOR_LIB`
2. `libGigaVector.so` next to the package (`python/src/gigavector/`), left there by a
   wheel or editable build
3. Repository build outputs: `build/lib/`, `build/`, `build-cmake/Release/`, `build-cmake/`

After changing C code, run `make lib` again. If an older `libGigaVector.so` sits next to
the package, it wins over `build/lib/`, so either delete it or point at the fresh build:

```bash
GIGAVECTOR_LIB=$PWD/build/lib/libGigaVector.so PYTHONPATH=python/src python3 your_script.py
```

## Environment Variables

```bash
cp .env.example .env   # then fill in your keys
```

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | For LLM and embedding tests | OpenAI API key |
| `ANTHROPIC_API_KEY` | For Anthropic tests | Anthropic API key |
| `GOOGLE_API_KEY` | Optional | Google Gemini (`GEMINI_API_KEY` is accepted as an alias) |
| `GV_WAL_DIR` | Optional | Override the WAL directory |
| `GIGAVECTOR_LIB` | Optional | Path to the shared library the Python package loads |
| `GV_ALLOC` | Optional | Set to `libc` to use the system allocator instead of the slab allocator |

## Python Tests

To run Python bindings tests:
```bash
# Run Python tests
make python-test

# Run with verbose output
make python-test-comprehensive
```

## Advanced Testing

### With Sanitizers

```bash
# AddressSanitizer (memory errors)
make test-asan

# ThreadSanitizer (threading issues)
make test-tsan

# UndefinedBehaviorSanitizer
make test-ubsan

# All sanitizers
make test-all
```

### With Valgrind

```bash
make test-valgrind
```

### Code Coverage

```bash
# Generate coverage data
make test-coverage

# Generate HTML coverage report
make test-coverage-html
# View at: build/coverage_html/index.html
```

## Troubleshooting

### Library Not Found

If you get "library not found" errors:
```bash
export LD_LIBRARY_PATH=$(pwd)/build/lib:${LD_LIBRARY_PATH}
```

### API Key Not Found

The LLM tests will skip real API calls if keys are not set. They'll still run validation tests. To run full tests:
1. Create `.env` file from `.env.example`
2. Add your API keys
3. Or export them: `export OPENAI_API_KEY=sk-...`

### Build Errors

If you encounter build errors:
```bash
# Clean and rebuild
make clean
make lib
```

### Missing Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install build-essential cmake libcurl4-openssl-dev

# Fedora/RHEL
sudo dnf install gcc make cmake libcurl-devel

# macOS
brew install cmake curl

# Windows (source builds only - PyPI wheel needs nothing extra)
pacman -S --needed mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-make
```

## Test Output

### Successful Test Run

```
Testing LLM creation with valid config...
  [PASS] LLM created successfully
  [PASS] LLM destroyed successfully

--- Real API Call Tests ---

Testing OpenAI API call...
  [PASS] API call successful
  Response: Hello! How can I help you today?
```

### Skipped Tests (No API Key)

```
Testing LLM creation with valid config...
  [SKIP] Skipping: OPENAI_API_KEY not set (using test key for validation only)
  [PASS] LLM created successfully
```

## Next Steps

- Read the [Usage Guide](usage.md) for how to use GigaVector
- Check [API Keys Documentation](api_keys_required.md) for detailed API key information
- See [Examples](examples/basic_usage.md) for code samples
