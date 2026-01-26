cmake_minimum_required(VERSION 3.15)
project(GigaVector VERSION 0.7.0 LANGUAGES C)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build options
option(BUILD_SHARED_LIBS "Build shared library" ON)
option(BUILD_TESTS "Build test executables" ON)
option(BUILD_BENCHMARKS "Build benchmark executables" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(ENABLE_SANITIZERS "Enable sanitizers (ASAN, TSAN, UBSAN)" OFF)
option(ENABLE_COVERAGE "Enable code coverage" OFF)

# Compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -g")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g")

# Architecture-specific optimizations
include(CheckCCompilerFlag)

# Check and enable SIMD optimizations
check_c_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
check_c_compiler_flag("-msse4.2" COMPILER_SUPPORTS_SSE42)
check_c_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_c_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
check_c_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)

if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
endif()

if(COMPILER_SUPPORTS_SSE42)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse4.2")
endif()

if(COMPILER_SUPPORTS_AVX2)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2")
endif()

if(COMPILER_SUPPORTS_AVX512F)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx512f")
endif()

if(COMPILER_SUPPORTS_FMA)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfma")
endif()

# Position Independent Code for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/gv_binary_quant.c
    src/gv_context_graph.c
    src/gv_embedding.c
    src/gv_database.c
    src/gv_distance.c
    src/gv_exact_search.c
    src/gv_filter.c
    src/gv_hnsw.c
    src/gv_importance.c
    src/gv_ivfpq.c
    src/gv_json.c
    src/gv_kdtree.c
    src/gv_llm.c
    src/gv_metadata_index.c
    src/gv_metadata.c
    src/gv_mmap.c
    src/gv_memory_consolidation.c
    src/gv_memory_extraction.c
    src/gv_memory_layer.c
    src/gv_scalar_quant.c
    src/gv_soa_storage.c
    src/gv_sparse_index.c
    src/gv_sparse_vector.c
    src/gv_utils.c
    src/gv_vector.c
    src/gv_wal.c
)

# Create library
add_library(GigaVector ${SOURCES})

# Check for libcurl (required for LLM support)
find_package(CURL)
if(CURL_FOUND)
    target_compile_definitions(GigaVector PRIVATE HAVE_CURL)
    target_link_libraries(GigaVector PRIVATE ${CURL_LIBRARIES})
    target_include_directories(GigaVector PRIVATE ${CURL_INCLUDE_DIRS})
    message(STATUS "  LLM support: Enabled (libcurl found)")
else()
    message(STATUS "  LLM support: Disabled (libcurl not found)")
    message(WARNING "Install libcurl-dev/libcurl-devel to enable LLM-based memory extraction")
endif()

# Link libraries
target_link_libraries(GigaVector PRIVATE m pthread)

# Set library properties
set_target_properties(GigaVector PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "include/gigavector/gigavector.h"
)

# Install library
install(TARGETS GigaVector
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include/gigavector
)

# Install headers
install(DIRECTORY include/gigavector/
    DESTINATION include/gigavector
    FILES_MATCHING PATTERN "*.h"
)

# Main executable
add_executable(gigavector_main main.c)
target_link_libraries(gigavector_main PRIVATE GigaVector)

# Install main executable
install(TARGETS gigavector_main
    RUNTIME DESTINATION bin
)

# Sanitizers
if(ENABLE_SANITIZERS)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -fsanitize=thread -fsanitize=undefined -fno-omit-frame-pointer")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -fsanitize=thread -fsanitize=undefined")
endif()

# Coverage
if(ENABLE_COVERAGE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage -O0")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    
    set(TEST_SOURCES
        tests/test_db.c
        tests/test_distance.c
        tests/test_metadata.c
        tests/test_hnsw.c
        tests/test_importance.c
        tests/test_ivfpq.c
        tests/test_sparse.c
        tests/test_wal.c
        tests/test_filter.c
        tests/test_advanced.c
        tests/test_llm.c
        tests/test_memory_llm.c
        tests/test_context_graph.c
        tests/test_json.c
    )
    
    foreach(TEST_SRC ${TEST_SOURCES})
        get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
        string(REPLACE "test_" "" TEST_NAME ${TEST_NAME})
        add_executable(test_${TEST_NAME} ${TEST_SRC})
        target_link_libraries(test_${TEST_NAME} PRIVATE GigaVector)
        add_test(NAME ${TEST_NAME} COMMAND test_${TEST_NAME})
    endforeach()
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    set(BENCHMARK_SOURCES
        benchmark_simd.c
        benchmark_compare.c
        benchmark_ivfpq.c
        benchmark_ivfpq_recall.c
    )
    
    foreach(BENCH_SRC ${BENCHMARK_SOURCES})
        get_filename_component(BENCH_NAME ${BENCH_SRC} NAME_WE)
        add_executable(${BENCH_NAME} ${BENCH_SRC})
        target_link_libraries(${BENCH_NAME} PRIVATE GigaVector m)
    endforeach()
endif()

# Python bindings (optional, requires Python development headers)
if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    
    # Note: Python bindings are typically built via setup.py
    # This section can be extended to build Python extensions directly
    message(STATUS "Python bindings should be built using: cd python && pip install .")
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "GigaVector Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Shared library: ${BUILD_SHARED_LIBS}")
message(STATUS "  Tests: ${BUILD_TESTS}")
message(STATUS "  Benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "  Sanitizers: ${ENABLE_SANITIZERS}")
message(STATUS "  Coverage: ${ENABLE_COVERAGE}")
message(STATUS "")

