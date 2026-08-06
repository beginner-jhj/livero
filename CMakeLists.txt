cmake_minimum_required(VERSION 3.15)
project(livero C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

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
set(CMAKE_C_FLAGS_DEBUG   "-g -O1 -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-fsanitize=address")

# ── Library ──────────────────────────────────────────────────────────
set(LV_SOURCES
    src/arena.c
    src/wal.c
    src/schema.c
    src/storage.c
    src/vector.c
    src/livero.c
    src/util.c
    src/helper.c
    src/node.c
    src/query.c
    src/sst.c
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch64)")
    message(STATUS "ARM architecture detected. Adding NEON simd source file.")
    list(APPEND LV_SOURCES src/simd_neon.c)

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|i386|i686)")
    message(STATUS "x86 architecture detected. Adding SSE2 simd source file.")
    list(APPEND LV_SOURCES src/simd_sse2.c)

else()
    message(FATAL_ERROR "Not supported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

add_library(livero STATIC ${LV_SOURCES})

target_include_directories(livero
    PUBLIC  include
    PRIVATE src
)

# ── Tests ─────────────────────────────────────────────────────────────
enable_testing()

# Helper to add a test executable
function(lv_add_test name file)
    add_executable(${name} ${file})
    target_link_libraries(${name} PRIVATE livero)
    target_include_directories(${name} PRIVATE src include)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

lv_add_test(test_simd tests/ctests/test_simd.c)
lv_add_test(test_arena tests/ctests/test_arena.c)