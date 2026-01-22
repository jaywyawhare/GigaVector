# Build and Test Guide

This guide explains how to build GigaVector and run all tests, including LLM tests.

## Prerequisites

1. **Build Tools:**
   - GCC compiler (with C99 support)
   - Make
   - libcurl development libraries
   - Python 3.x (for Python bindings tests)

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

Install required libraries:
```bash
# Ubuntu/Debian
sudo apt-get install build-essential libcurl4-openssl-dev

# Fedora/RHEL
sudo dnf install gcc make libcurl-devel

# macOS
brew install curl
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
- Check [API Keys Documentation](API_KEYS_REQUIRED.md) for detailed API key information
- See [Examples](examples/basic_usage.md) for code samples






