cmake_minimum_required(VERSION 3.15)
project(GigaVector VERSION 0.8.21 LANGUAGES C)

# Install directory conventions
include(GNUInstallDirs)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build options
option(BUILD_SHARED_LIBS "Build shared library" ON)
option(BUILD_TESTS "Build test executables" ON)
option(BUILD_BENCHMARKS "Build benchmark executables" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(ENABLE_SANITIZERS "Enable AddressSanitizer + UBSan (use ENABLE_TSAN for thread sanitizer)" OFF)
option(ENABLE_COVERAGE "Enable code coverage" OFF)
option(ENABLE_NATIVE_OPTIMIZATIONS "Enable CPU-specific optimizations (-march=native -mtune=native)" OFF)

# Compiler flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -g")
elseif(MSVC)
    # Keep warnings high on MSVC, but do not inject GCC/Clang flags.
    add_compile_options(/W4)
endif()
# MSVC already has sensible per-config defaults (/O2 /Od /Zi); only set
# GCC/Clang style flags for those compilers.
if(NOT MSVC)
    set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
endif()

# Ensure objects do not request an executable stack on ELF targets only.
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang" AND NOT WIN32 AND NOT APPLE)
    add_compile_options(-Wa,--noexecstack)
    add_link_options(-Wl,-z,noexecstack)
endif()

# CPU tuning is opt-in to avoid shipping binaries with unsupported instruction sets.
if(ENABLE_NATIVE_OPTIMIZATIONS)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -mtune=native")
endif()

# Position Independent Code for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Export all symbols from the DLL on MSVC (mirrors MinGW --export-all-symbols
# default so callers don't need per-symbol __declspec(dllexport) annotations).
if(WIN32 AND BUILD_SHARED_LIBS)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

if(MINGW)
    # Static-link libgcc (no transitive deps) and pthread.
    # curl, ssl, crypto etc. remain dynamic so delvewheel can bundle them all
    # (including their transitive deps: zlib, brotli, etc.) into the wheel.
    set(CMAKE_SHARED_LINKER_FLAGS
        "${CMAKE_SHARED_LINKER_FLAGS} -static-libgcc -Wl,-Bstatic -lpthread -Wl,-Bdynamic")
    set(CMAKE_EXE_LINKER_FLAGS
        "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc")
endif()

# Source files
file(GLOB_RECURSE GV_SOURCE_FILES CONFIGURE_DEPENDS 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)
list(SORT GV_SOURCE_FILES)
set(SOURCES ${GV_SOURCE_FILES})

# Create library
add_library(GigaVector ${SOURCES})

# Public include root (do not flatten subdirectories)
target_include_directories(GigaVector
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Check for libcurl (required for LLM support, webhook delivery, SSO, embeddings)
find_package(CURL)
if(CURL_FOUND)
    target_compile_definitions(GigaVector PRIVATE HAVE_CURL)
    target_link_libraries(GigaVector PRIVATE ${CURL_LIBRARIES})
    message(STATUS "  LLM / Webhook / SSO / Embeddings: Enabled (libcurl found)")
else()
    message(STATUS "  LLM / Webhook / SSO / Embeddings: Disabled (libcurl not found)")
    message(WARNING "Install libcurl-dev/libcurl-devel to enable LLM, webhook delivery, SSO and embedding providers")
endif()

# Check for libmicrohttpd (required for HTTP REST server)
find_library(MICROHTTPD_LIBRARY NAMES microhttpd)
find_path(MICROHTTPD_INCLUDE_DIR NAMES microhttpd.h)
if(MICROHTTPD_LIBRARY AND MICROHTTPD_INCLUDE_DIR)
    target_compile_definitions(GigaVector PRIVATE HAVE_MICROHTTPD)
    target_include_directories(GigaVector PRIVATE ${MICROHTTPD_INCLUDE_DIR})
    target_link_libraries(GigaVector PRIVATE ${MICROHTTPD_LIBRARY})
    message(STATUS "  HTTP REST server: Enabled (libmicrohttpd found)")
else()
    message(STATUS "  HTTP REST server: Disabled (libmicrohttpd not found)")
    message(WARNING "Install libmicrohttpd-dev to enable the HTTP REST server")
endif()

# Check for OpenSSL (required for TLS support)
find_package(OpenSSL)
if(OPENSSL_FOUND)
    target_compile_definitions(GigaVector PRIVATE GV_HAVE_OPENSSL)
    target_include_directories(GigaVector PRIVATE ${OPENSSL_INCLUDE_DIR})
    target_link_libraries(GigaVector PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    message(STATUS "  TLS support: Enabled (OpenSSL found)")
else()
    message(STATUS "  TLS support: Disabled (OpenSSL not found)")
    message(WARNING "Install libssl-dev/openssl-devel to enable TLS support")
endif()

# Check for ONNX Runtime (optional, for model serving)
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime)
find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_c_api.h
    PATH_SUFFIXES onnxruntime)
if(ONNXRUNTIME_LIBRARY AND ONNXRUNTIME_INCLUDE_DIR)
    target_compile_definitions(GigaVector PRIVATE GV_HAVE_ONNX)
    target_include_directories(GigaVector PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(GigaVector PRIVATE ${ONNXRUNTIME_LIBRARY})
    message(STATUS "  ONNX model serving: Enabled (onnxruntime found)")
else()
    message(STATUS "  ONNX model serving: Disabled (onnxruntime not found)")
    message(WARNING "Install onnxruntime to enable ONNX model serving")
endif()

# Check for librdkafka (optional, for real Kafka streaming ingestion)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(RDKAFKA QUIET rdkafka)
endif()
if(NOT RDKAFKA_FOUND)
    find_library(RDKAFKA_LIBRARIES NAMES rdkafka)
    find_path(RDKAFKA_INCLUDE_DIRS NAMES librdkafka/rdkafka.h)
    if(RDKAFKA_LIBRARIES AND RDKAFKA_INCLUDE_DIRS)
        set(RDKAFKA_FOUND TRUE)
    endif()
endif()
if(RDKAFKA_FOUND)
    target_compile_definitions(GigaVector PRIVATE HAVE_RDKAFKA)
    target_include_directories(GigaVector PRIVATE ${RDKAFKA_INCLUDE_DIRS})
    target_link_libraries(GigaVector PRIVATE ${RDKAFKA_LIBRARIES})
    message(STATUS "  Kafka streaming: Enabled (librdkafka found)")
else()
    message(STATUS "  Kafka streaming: Disabled (librdkafka not found)")
    message(WARNING "Install librdkafka-dev to enable real Kafka streaming ingestion")
endif()

# SIMD runtime dispatch — compile distance.c with AVX2/AVX512 flags when the
# compiler supports them, without forcing -march=native across the whole build.
include(CheckCCompilerFlag)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang" AND NOT WIN32)
    check_c_compiler_flag(-mavx2 COMPILER_SUPPORTS_AVX2)
    check_c_compiler_flag(-mavx512f COMPILER_SUPPORTS_AVX512F)

    set(GV_DISTANCE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/search/distance.c")

    if(COMPILER_SUPPORTS_AVX512F)
        set_source_files_properties(${GV_DISTANCE_SRC} PROPERTIES
            COMPILE_FLAGS "-mavx2 -mfma -mavx512f")
        message(STATUS "  SIMD distance: AVX512 enabled")
    elseif(COMPILER_SUPPORTS_AVX2)
        set_source_files_properties(${GV_DISTANCE_SRC} PROPERTIES
            COMPILE_FLAGS "-mavx2 -mfma")
        message(STATUS "  SIMD distance: AVX2 enabled")
    else()
        message(STATUS "  SIMD distance: Scalar fallback (no AVX2/AVX512)")
    endif()
endif()

# Check for CUDA (optional, for GPU acceleration)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    option(ENABLE_CUDA "Enable CUDA GPU acceleration" ON)
    if(ENABLE_CUDA)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 11)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)

        target_sources(GigaVector PRIVATE src/specialized/gpu_kernels.cu)
        target_compile_definitions(GigaVector PRIVATE HAVE_CUDA)
        set_target_properties(GigaVector PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "75;80;86;89;90"
        )
        message(STATUS "  GPU Acceleration: Enabled (CUDA found)")
    else()
        message(STATUS "  GPU Acceleration: Disabled (CUDA disabled)")
    endif()
else()
    message(STATUS "  GPU Acceleration: Disabled (CUDA not found)")
endif()

# Link libraries — m and pthread are separate on POSIX; on Windows they are
# in the CRT / provided by MinGW's built-in runtime.
if(NOT WIN32)
    target_link_libraries(GigaVector PRIVATE m pthread)
else()
    target_link_libraries(GigaVector PRIVATE bcrypt)
endif()

# Set library properties
set_target_properties(GigaVector PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "include/gigavector.h"
)

# Install library
install(TARGETS GigaVector
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install headers
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)

# Main executable
add_executable(gigavector_main main.c)
target_link_libraries(gigavector_main PRIVATE GigaVector)

# Install main executable
install(TARGETS gigavector_main
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# CLI Tools
add_executable(gvbackup tools/backup_cli.c)
target_link_libraries(gvbackup PRIVATE GigaVector)

add_executable(gvrestore tools/restore_cli.c)
target_link_libraries(gvrestore PRIVATE GigaVector)

add_executable(gvinspect tools/inspect_cli.c)
target_link_libraries(gvinspect PRIVATE GigaVector)

# Install CLI tools
install(TARGETS gvbackup gvrestore gvinspect
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Sanitizers — ASAN and TSAN are mutually exclusive; ENABLE_SANITIZERS uses ASAN+UBSAN.
# Use ENABLE_TSAN for a separate thread-sanitizer build.
option(ENABLE_TSAN "Enable ThreadSanitizer (incompatible with ASAN)" OFF)

# Collect sanitizer/coverage flags into an INTERFACE target so they are applied
# per-target via transitive linkage rather than polluting global flag variables.
add_library(gv_instrumentation_flags INTERFACE)

if(ENABLE_TSAN)
    target_compile_options(gv_instrumentation_flags INTERFACE -fsanitize=thread -fno-omit-frame-pointer)
    target_link_options(gv_instrumentation_flags INTERFACE -fsanitize=thread)
elseif(ENABLE_SANITIZERS)
    target_compile_options(gv_instrumentation_flags INTERFACE -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer)
    target_link_options(gv_instrumentation_flags INTERFACE -fsanitize=address -fsanitize=undefined)
endif()

if(ENABLE_COVERAGE)
    target_compile_options(gv_instrumentation_flags INTERFACE --coverage -O0)
    target_link_options(gv_instrumentation_flags INTERFACE --coverage)
endif()

# Propagate instrumentation flags to GigaVector and all targets that link it.
target_link_libraries(GigaVector PUBLIC gv_instrumentation_flags)

# Tests
if(BUILD_TESTS)
    enable_testing()

    file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.c")
    list(SORT TEST_SOURCES)

    foreach(TEST_SRC ${TEST_SOURCES})
        get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
        string(REPLACE "test_" "" TEST_NAME ${TEST_NAME})
        add_executable(test_${TEST_NAME} ${TEST_SRC})
        if(WIN32)
            target_link_libraries(test_${TEST_NAME} PRIVATE GigaVector)
        else()
            target_link_libraries(test_${TEST_NAME} PRIVATE GigaVector m pthread)
        endif()
        add_test(NAME ${TEST_NAME} COMMAND test_${TEST_NAME})
    endforeach()

    # Shell-based tests (skip gracefully when prerequisites are absent)
    find_program(BASH_EXECUTABLE bash)
    if(BASH_EXECUTABLE AND NOT WIN32)
        add_test(NAME corrupt_wal
            COMMAND ${BASH_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/corrupt_wal.sh)
        set_tests_properties(corrupt_wal PROPERTIES SKIP_RETURN_CODE 77)

        add_test(NAME corrupt_snapshot
            COMMAND ${BASH_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/corrupt_snapshot.sh)
        set_tests_properties(corrupt_snapshot PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    set(BENCHMARK_SOURCES
        benchmarks/benchmark_simd.c
        benchmarks/benchmark_compare.c
        benchmarks/benchmark_ivfpq.c
        benchmarks/benchmark_ivfpq_recall.c
    )
    
    foreach(BENCH_SRC ${BENCHMARK_SOURCES})
        get_filename_component(BENCH_NAME ${BENCH_SRC} NAME_WE)
        add_executable(${BENCH_NAME} ${BENCH_SRC})
        if(WIN32)
            target_link_libraries(${BENCH_NAME} PRIVATE GigaVector)
        else()
            target_link_libraries(${BENCH_NAME} PRIVATE GigaVector m)
        endif()
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
message(STATUS "  Native optimizations: ${ENABLE_NATIVE_OPTIMIZATIONS}")
message(STATUS "")
