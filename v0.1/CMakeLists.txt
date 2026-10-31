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

add_library(livero STATIC ${LV_SOURCES})

target_include_directories(livero PUBLIC include PRIVATE src)
