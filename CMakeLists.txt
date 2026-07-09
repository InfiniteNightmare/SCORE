cmake_minimum_required(VERSION 3.16)
project(score VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Architecture flags
include(CheckCXXCompilerFlag)
option(SCORE_SANITIZER_FRIENDLY "Disable LTO/aggressive optimization/allocator choices that interfere with sanitizers" OFF)
option(SCORE_ENABLE_NATIVE "Build with -march=native" ON)
option(SCORE_ENABLE_AVX2 "Build with AVX2 code generation when supported" ON)
option(SCORE_ENABLE_AVX512 "Build with AVX-512 code generation when supported" ON)
option(SCORE_ENABLE_OFAST "Build with -Ofast in addition to the selected CMake build type" ON)
option(SCORE_ENABLE_LTO "Enable interprocedural optimization/LTO" ON)
option(SCORE_ENABLE_TCMALLOC "Link tcmalloc when available" ON)
if(SCORE_SANITIZER_FRIENDLY)
    set(SCORE_ENABLE_OFAST OFF CACHE BOOL "Build with -Ofast in addition to the selected CMake build type" FORCE)
    set(SCORE_ENABLE_LTO OFF CACHE BOOL "Enable interprocedural optimization/LTO" FORCE)
    set(SCORE_ENABLE_TCMALLOC OFF CACHE BOOL "Link tcmalloc when available" FORCE)
endif()

check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
if(SCORE_ENABLE_NATIVE AND HAS_MARCH_NATIVE)
    add_compile_options(-march=native)
endif()
check_cxx_compiler_flag("-mavx2" HAS_AVX2)
if(SCORE_ENABLE_AVX2 AND HAS_AVX2)
    add_compile_options(-mavx2)
    check_cxx_compiler_flag("-mfma" HAS_FMA)
    if(HAS_FMA)
        add_compile_options(-mfma)
    endif()
endif()
check_cxx_compiler_flag("-mavx512f" HAS_AVX512F)
if(SCORE_ENABLE_AVX512 AND HAS_AVX512F)
    add_compile_options(-mavx512f -mavx512bw -mavx512vl)
    check_cxx_compiler_flag("-mavx512dq" HAS_AVX512DQ)
    if(HAS_AVX512DQ)
        add_compile_options(-mavx512dq)
    endif()
    check_cxx_compiler_flag("-mbmi2" HAS_BMI2)
    if(HAS_BMI2)
        add_compile_options(-mbmi2)
    endif()
    add_definitions(-DHAS_AVX512=1)
else()
    add_definitions(-DHAS_AVX512=0)
endif()

# Optimization: -Ofast + LTO (+2% QPS vs plain -O3; PGO tested but hurts IO-bound workloads)
if(SCORE_ENABLE_OFAST)
    add_compile_options(-Ofast)
endif()
add_compile_options(-Wall -Wextra)
if(SCORE_ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)  # LTO
endif()

# Dependencies
include(FetchContent)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

option(SCORE_USE_LIBURING "Enable experimental io_uring finalizer backend" OFF)
if(SCORE_USE_LIBURING)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LIBURING REQUIRED IMPORTED_TARGET liburing)
    message(STATUS "Found liburing: ${LIBURING_VERSION}")
endif()

find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    find_path(EIGEN3_INCLUDE_DIR
        NAMES Eigen/Dense
        HINTS
            /usr/include/eigen3
            ${CMAKE_CURRENT_SOURCE_DIR}/external/RaBitQ-Library/include/rabitqlib/third
            ${CMAKE_CURRENT_SOURCE_DIR}/baselines/SAQ/saqlib/third)
    if(NOT EIGEN3_INCLUDE_DIR)
        message(FATAL_ERROR "Eigen3 not found. Install Eigen3 or provide vendored Eigen headers.")
    endif()
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
    message(STATUS "Using vendored/header-only Eigen: ${EIGEN3_INCLUDE_DIR}")
endif()

find_path(RAPIDJSON_INCLUDE_DIR
    NAMES rapidjson/document.h
    HINTS
        /usr/include
        /usr/local/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/rapidjson/include)
if(NOT RAPIDJSON_INCLUDE_DIR)
    message(STATUS "RapidJSON not found, fetching v1.1.0")
    FetchContent_Declare(
        rapidjson
        GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
        GIT_TAG v1.1.0
        GIT_SHALLOW TRUE)
    FetchContent_Populate(rapidjson)
    set(RAPIDJSON_INCLUDE_DIR "${rapidjson_SOURCE_DIR}/include")
else()
    message(STATUS "Found RapidJSON: ${RAPIDJSON_INCLUDE_DIR}")
endif()

execute_process(
    COMMAND git rev-parse --git-common-dir
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMON_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
get_filename_component(COMMON_REPO_ROOT "${GIT_COMMON_DIR}" DIRECTORY)

# AOCL 5.2 — AMD-optimized BLAS (BLIS) + LAPACK (libFLAME) for Zen 4
set(AOCL_ROOT "/opt/AOCL" CACHE PATH "AOCL install prefix")
set(AOCL_ROOT_HINTS
    ${AOCL_ROOT}
    /opt/AMD/aocl/aocl-linux-gcc-5.2.0/gcc)

if(AOCL_INCLUDE_DIR AND NOT EXISTS "${AOCL_INCLUDE_DIR}/cblas.h")
    unset(AOCL_INCLUDE_DIR CACHE)
endif()
if(AOCL_BLIS_LIBRARY AND NOT EXISTS "${AOCL_BLIS_LIBRARY}")
    unset(AOCL_BLIS_LIBRARY CACHE)
endif()
if(AOCL_FLAME_LIBRARY AND NOT EXISTS "${AOCL_FLAME_LIBRARY}")
    unset(AOCL_FLAME_LIBRARY CACHE)
endif()
if(AOCL_UTILS_LIBRARY AND NOT EXISTS "${AOCL_UTILS_LIBRARY}")
    unset(AOCL_UTILS_LIBRARY CACHE)
endif()

find_path(AOCL_INCLUDE_DIR NAMES cblas.h HINTS ${AOCL_ROOT_HINTS} PATH_SUFFIXES include include_LP64 include_ILP64)
find_library(AOCL_BLIS_LIBRARY NAMES blis-mt blis HINTS ${AOCL_ROOT_HINTS} PATH_SUFFIXES lib lib_LP64 lib_ILP64)
find_library(AOCL_FLAME_LIBRARY NAMES flame HINTS ${AOCL_ROOT_HINTS} PATH_SUFFIXES lib lib_LP64 lib_ILP64)
find_library(AOCL_UTILS_LIBRARY NAMES aoclutils HINTS ${AOCL_ROOT_HINTS} PATH_SUFFIXES lib lib_LP64 lib_ILP64)
if(AOCL_INCLUDE_DIR AND AOCL_BLIS_LIBRARY)
    message(STATUS "Found AOCL-BLIS 5.2: ${AOCL_BLIS_LIBRARY}")
    add_definitions(-DHAS_AOCL_BLAS=1)
    set(AOCL_BLAS_INCLUDE_DIR ${AOCL_INCLUDE_DIR})
    set(AOCL_BLAS_LIBRARY ${AOCL_BLIS_LIBRARY})
    if(AOCL_FLAME_LIBRARY)
        message(STATUS "Found AOCL-libFLAME 5.2: ${AOCL_FLAME_LIBRARY}")
        add_definitions(-DHAS_AOCL_LAPACK=1)
    else()
        add_definitions(-DHAS_AOCL_LAPACK=0)
    endif()
else()
    message(STATUS "AOCL not found at ${AOCL_ROOT}, trying system BLAS fallback")
    add_definitions(-DHAS_AOCL_BLAS=0 -DHAS_AOCL_LAPACK=0)
    find_package(BLAS)
    if(BLAS_FOUND)
        message(STATUS "Found system BLAS: ${BLAS_LIBRARIES}")
        set(SYSTEM_BLAS_LIBRARIES ${BLAS_LIBRARIES})
    endif()
endif()

# SuperKMeans — ultra-fast k-means clustering
set(USE_SUPERKMEANS ON CACHE BOOL "Use SuperKMeans for IMI k-means training")
set(SUPERKMEANS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/SuperKMeans/include")
if(NOT EXISTS "${SUPERKMEANS_INCLUDE_DIR}/superkmeans/superkmeans.h" AND EXISTS "${COMMON_REPO_ROOT}/external/SuperKMeans/include/superkmeans/superkmeans.h")
    set(SUPERKMEANS_INCLUDE_DIR "${COMMON_REPO_ROOT}/external/SuperKMeans/include")
endif()
if(USE_SUPERKMEANS AND EXISTS "${SUPERKMEANS_INCLUDE_DIR}/superkmeans/superkmeans.h")
    message(STATUS "Found SuperKMeans: external/SuperKMeans")
    add_definitions(-DHAS_SUPERKMEANS=1)
else()
    message(STATUS "SuperKMeans not found, using built-in k-means")
    add_definitions(-DHAS_SUPERKMEANS=0)
endif()

set(RABITQ_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/RaBitQ-Library/include")
if(NOT EXISTS "${RABITQ_INCLUDE_DIR}/rabitqlib/utils/rotator.hpp" AND EXISTS "${COMMON_REPO_ROOT}/external/RaBitQ-Library/include/rabitqlib/utils/rotator.hpp")
    set(RABITQ_INCLUDE_DIR "${COMMON_REPO_ROOT}/external/RaBitQ-Library/include")
endif()

# tcmalloc — faster thread-caching allocator (small gains over glibc after pre-alloc)
if(SCORE_ENABLE_TCMALLOC)
    if(TCMALLOC_LIBRARY AND NOT EXISTS "${TCMALLOC_LIBRARY}")
        unset(TCMALLOC_LIBRARY CACHE)
    endif()
    find_library(TCMALLOC_LIBRARY NAMES tcmalloc tcmalloc_minimal
        HINTS /usr/lib /usr/lib/x86_64-linux-gnu)
    if(TCMALLOC_LIBRARY)
        message(STATUS "Found tcmalloc: ${TCMALLOC_LIBRARY}")
    else()
        message(STATUS "tcmalloc not found, using system malloc")
    endif()
else()
    unset(TCMALLOC_LIBRARY CACHE)
    message(STATUS "tcmalloc disabled, using system malloc")
endif()

# Main library (header-only implementation; keeps shared include/link settings)
add_library(score_lib INTERFACE)
target_include_directories(score_lib INTERFACE include)
target_include_directories(score_lib SYSTEM INTERFACE ${RAPIDJSON_INCLUDE_DIR})
if(USE_SUPERKMEANS AND EXISTS "${SUPERKMEANS_INCLUDE_DIR}")
    target_include_directories(score_lib INTERFACE
        ${SUPERKMEANS_INCLUDE_DIR})
endif()
# RaBitQ-Library headers for split_single_estdist integration
target_include_directories(score_lib INTERFACE
    ${RABITQ_INCLUDE_DIR})
target_link_libraries(score_lib INTERFACE
    Eigen3::Eigen
    Threads::Threads
    OpenMP::OpenMP_CXX
)
if(SCORE_USE_LIBURING)
    target_compile_definitions(score_lib INTERFACE SCORE_USE_LIBURING=1)
    target_link_libraries(score_lib INTERFACE PkgConfig::LIBURING)
endif()
if(AOCL_BLIS_LIBRARY)
    target_include_directories(score_lib INTERFACE ${AOCL_INCLUDE_DIR})
    target_link_libraries(score_lib INTERFACE ${AOCL_BLIS_LIBRARY})
    if(AOCL_FLAME_LIBRARY)
        target_link_libraries(score_lib INTERFACE ${AOCL_FLAME_LIBRARY})
    endif()
    if(AOCL_UTILS_LIBRARY)
        target_link_libraries(score_lib INTERFACE ${AOCL_UTILS_LIBRARY})
    endif()
elseif(SYSTEM_BLAS_LIBRARIES)
    target_link_libraries(score_lib INTERFACE ${SYSTEM_BLAS_LIBRARIES})
endif()
if(TCMALLOC_LIBRARY)
    target_link_libraries(score_lib INTERFACE ${TCMALLOC_LIBRARY})
endif()

# Main executable: build index
add_executable(score_build src/build_main.cpp)
target_link_libraries(score_build PRIVATE score_lib)

# Main executable: search
add_executable(score_search src/search_main.cpp)
target_link_libraries(score_search PRIVATE score_lib)

# Main executable: benchmark
add_executable(score_bench src/bench_main.cpp)
target_link_libraries(score_bench PRIVATE score_lib)

# Streaming build primitive: blockwise rotated verify writer
add_executable(score_stream_verify src/stream_verify_main.cpp)
target_link_libraries(score_stream_verify PRIVATE score_lib)

# Streaming RaBitQ artifact primitive: no PID reorder, no IVF/postings
add_executable(score_stream_rabitq src/stream_rabitq_main.cpp)
target_link_libraries(score_stream_rabitq PRIVATE score_lib)

# Streaming IVF/posting primitive: no PID reorder, bounded assignment passes
add_executable(score_stream_ivf_posting src/stream_ivf_posting_main.cpp)
target_link_libraries(score_stream_ivf_posting PRIVATE score_lib)

# End-to-end streaming no-PID-reorder builder
add_executable(score_stream_build_no_reorder src/stream_build_no_reorder_main.cpp)
target_link_libraries(score_stream_build_no_reorder PRIVATE score_lib)

# Post-hoc streaming index PID reorder converter
add_executable(score_stream_reorder_index src/stream_reorder_index_main.cpp)
target_link_libraries(score_stream_reorder_index PRIVATE score_lib)

# End-to-end streaming builder that directly emits a PID-reordered index
add_executable(score_stream_build_reorder src/stream_build_reorder_main.cpp)
target_link_libraries(score_stream_build_reorder PRIVATE score_lib)

# L0 update writer
add_executable(score_update_index src/update_index_main.cpp)
target_link_libraries(score_update_index PRIVATE score_lib)

# In-process small-batch update benchmark
add_executable(score_update_stream_bench src/update_stream_bench_main.cpp)
target_link_libraries(score_update_stream_bench PRIVATE score_lib)

# Mixed query/update contention benchmark
add_executable(score_update_query_mix_bench src/update_query_mix_bench_main.cpp)
target_link_libraries(score_update_query_mix_bench PRIVATE score_lib)

add_executable(score_update_search src/update_search_main.cpp)
target_link_libraries(score_update_search PRIVATE score_lib)

# Delta compaction tool
add_executable(score_compact_index src/compact_index_main.cpp)
target_link_libraries(score_compact_index PRIVATE score_lib)

# Test executable
add_executable(score_test
    tests/test_main.cpp
    tests/test_rotation_vector_io.cpp
    tests/test_streaming_build.cpp
    tests/test_fused_no_reorder.cpp
    tests/test_reorder.cpp
    tests/test_rabitq_sc_ivf.cpp
    tests/test_pid_reorder.cpp
    tests/test_update_compaction.cpp)
target_link_libraries(score_test PRIVATE score_lib)

# GT computation tool (uses BLAS for fast brute-force)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/gt_compute.cpp")
    add_executable(gt_compute src/gt_compute.cpp)
    target_link_libraries(gt_compute PRIVATE score_lib)
endif()
