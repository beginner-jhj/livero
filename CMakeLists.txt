cmake_minimum_required(VERSION 3.20)

project(livero C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler warnings
add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wno-unused-parameter
    -Wno-unused-function 
)

# Build type defaults to Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Release flags
set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_C_FLAGS_DEBUG "-g -O1 -fsanitize=address -fno-omit-frame-pointer") #use ASan
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-fsanitize=address")


set(LV_SOURCES
    src/livero.c
    src/file/file.c
    src/helper/crc32c.c
    src/helper/helper.c
    src/vector/ternary.c
    src/struct/row.c
    src/struct/commit.c
    src/struct/config.c
)

option(LV_USE_SIMD "Enable SIMD (NEON) code paths when the target supports them" ON)

set(LV_SIMD_SUPPORTED OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(LV_SIMD_SUPPORTED ON)
endif()

if(LV_USE_SIMD AND LV_SIMD_SUPPORTED)
    message(STATUS "livero: NEON SIMD enabled")
    list(APPEND LV_SOURCES src/vector/neon.c)
    set(LV_SIMD_ACTIVE ON)
else()
    message(STATUS "livero: using scalar fallback")
    set(LV_SIMD_ACTIVE OFF)
endif()

add_library(livero STATIC ${LV_SOURCES})
target_include_directories(livero PUBLIC include PRIVATE src)

if(LV_SIMD_ACTIVE)
    target_compile_definitions(livero PUBLIC LV_USE_SIMD)
endif()

target_compile_definitions(livero PRIVATE _POSIX_C_SOURCE=200809L _DARWIN_C_SOURCE)

find_library(LV_MATH_LIB m)
if(LV_MATH_LIB)
    target_link_libraries(livero PUBLIC ${LV_MATH_LIB})
endif()

option(LV_ENABLE_POPCNT "Use the hardware POPCNT instruction on x86-64" ON)

include(CheckCCompilerFlag)
check_c_compiler_flag(-mpopcnt LV_COMPILER_HAS_MPOPCNT)


if(LV_ENABLE_POPCNT
    AND LV_COMPILER_HAS_MPOPCNT
    AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    target_compile_options(livero PRIVATE -mpopcnt)
    message(STATUS "livero: hardware POPCNT enabled")
endif()


enable_testing()

function(lv_add_test name file)
    add_executable(${name} ${file})
    target_link_libraries(${name} PRIVATE livero)
    target_include_directories(${name} PRIVATE src include tests)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

if(LV_SIMD_ACTIVE)
    lv_add_test(test_simd tests/test_simd.c)
    message(STATUS "SIMD is active. test_simd is on.")
else()
    message(STATUS "SIMD is not active. test_simd is off.")
endif()
lv_add_test(test_pending tests/test_pending.c)
lv_add_test(test_recover tests/test_recover.c)
lv_add_test(test_disk_search tests/test_disk_search.c)
lv_add_test(test_livero tests/test_livero.c)


add_executable(benchmark tests/benchmark.c)
target_link_libraries(benchmark PRIVATE livero)
target_include_directories(benchmark PRIVATE src include tests)
target_compile_definitions(benchmark PRIVATE _POSIX_C_SOURCE=200809L)