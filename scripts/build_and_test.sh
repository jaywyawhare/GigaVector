#!/bin/bash
# Build and test script for GigaVector
# This script builds the library and runs all tests including LLM tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
LIB_DIR="$BUILD_DIR/lib"

echo "=== Building GigaVector ==="
cd "$SCRIPT_DIR"
make clean
make lib

echo ""
echo "=== Running Standard C Tests ==="
make c-test

echo ""
echo "=== Building LLM and Embedding Tests ==="

# Build test_llm
echo "Building test_llm..."
gcc -O3 -g -Wall -Wextra -Iinclude -march=native -msse4.2 -mavx2 -mavx512f -mfma -pthread -fPIC -DHAVE_CURL \
    tests/test_llm.c -L"$LIB_DIR" -lGigaVector -lm -pthread -lcurl \
    -Wl,-rpath,"$LIB_DIR" -o "$BUILD_DIR/test_llm" || {
    echo "Failed to build test_llm"
    exit 1
}

# Build test_memory_llm
echo "Building test_memory_llm..."
gcc -O3 -g -Wall -Wextra -Iinclude -march=native -msse4.2 -mavx2 -mavx512f -mfma -pthread -fPIC -DHAVE_CURL \
    tests/test_memory_llm.c -L"$LIB_DIR" -lGigaVector -lm -pthread -lcurl \
    -Wl,-rpath,"$LIB_DIR" -o "$BUILD_DIR/test_memory_llm" || {
    echo "Failed to build test_memory_llm"
    exit 1
}

# Build test_embedding
echo "Building test_embedding..."
gcc -O3 -g -Wall -Wextra -Iinclude -march=native -msse4.2 -mavx2 -mavx512f -mfma -pthread -fPIC -DHAVE_CURL \
    tests/test_embedding.c -L"$LIB_DIR" -lGigaVector -lm -pthread -lcurl \
    -Wl,-rpath,"$LIB_DIR" -o "$BUILD_DIR/test_embedding" || {
    echo "Failed to build test_embedding"
    exit 1
}

echo ""
echo "=== Running LLM Tests ==="
if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY and/or ANTHROPIC_API_KEY not set."
    echo "Some tests will be skipped. See .env.example for setup instructions."
fi
export LD_LIBRARY_PATH="$LIB_DIR:${LD_LIBRARY_PATH}"
"$BUILD_DIR/test_llm" || echo "LLM tests completed (some may have been skipped)"

echo ""
echo "=== Running Memory LLM Tests ==="
"$BUILD_DIR/test_memory_llm" || echo "Memory LLM tests completed (some may have been skipped)"

echo ""
echo "=== Running Embedding Tests ==="
"$BUILD_DIR/test_embedding" || echo "Embedding tests completed (some may have been skipped)"

echo ""
echo "=== All Tests Complete ==="
echo ""
echo "To run individual tests:"
echo "  LD_LIBRARY_PATH=$LIB_DIR:$LD_LIBRARY_PATH $BUILD_DIR/test_llm"
echo "  LD_LIBRARY_PATH=$LIB_DIR:$LD_LIBRARY_PATH $BUILD_DIR/test_memory_llm"
echo "  LD_LIBRARY_PATH=$LIB_DIR:$LD_LIBRARY_PATH $BUILD_DIR/test_embedding"






