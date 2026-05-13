cmake_minimum_required(VERSION 3.15)
project(lightvec C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Compiler warnings
add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wno-unused-parameter
)

# Build type defaults to Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Release flags
set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_C_FLAGS_DEBUG   "-g -fsanitize=address,undefined")

# ── Library ──────────────────────────────────────────────────────────
set(LV_SOURCES
    src/arena.c
    src/crc.c
    src/wal.c
    src/schema.c
    src/storage.c
    src/vector.c
    src/lightvec.c
    src/util.c
    src/helper.c
    src/node.c
    src/hash.c
    src/query.c
)

add_library(lightvec STATIC ${LV_SOURCES})

target_include_directories(lightvec
    PUBLIC  include
    PRIVATE src
)

# ── Tests ─────────────────────────────────────────────────────────────
enable_testing()

# Helper to add a test executable
function(lv_add_test name file)
    add_executable(${name} ${file})
    target_link_libraries(${name} PRIVATE lightvec)
    target_include_directories(${name} PRIVATE src include)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

lv_add_test(test_arena   tests/test_arena.c)
lv_add_test(test_storage tests/test_storage.c)
lv_add_test(test_vector  tests/test_vector.c)
lv_add_test(test_hnsw    tests/test_hnsw.c)
lv_add_test(test_main    tests/test_main.c)