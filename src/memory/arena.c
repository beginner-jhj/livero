#include "arena.h"

#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>

#include "internal/helper.h"
#include "livero_types.h"
#include "internal/internal.h"

static void* arena_align_ptr(const void* ptr, const uint32_t offset, const int alignment) {
    assert(alignment > 0 && alignment <= LV_ARENA_BLOCK_ALIGNMENT);
    assert((alignment & (alignment - 1)) == 0);
    const uintptr_t aligned = (((uintptr_t)((unsigned char*)ptr + offset) + (alignment - 1)) & ~(alignment - 1));
    return (void*)aligned;
}

LVArena* arena_create(const LVSize_t total_size, const LVSize_t (*block_sizes)[LV_ARENA_BLOCK_COUNT]) {
    assert((total_size & (PAGE_SIZE - 1)) == 0);
    assert(arena_reserved_size() + (*block_sizes)[0] + (*block_sizes)[1] + (*block_sizes)[2] + (*block_sizes)[3] <=
           total_size);

    void* p = mmap(nullptr, total_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if (!p) {
        LIVERO_ERROR("Failed to create an Arena (ERRNO: %d)", errno);
        return nullptr;
    };

    LVArena* arena = (LVArena*)p;
    arena->max_size = total_size;

    uint64_t offset = arena_reserved_size();
    for (int i = 0; i < LV_ARENA_BLOCK_COUNT; ++i) {
        const LVSize_t block_size = (*block_sizes)[i];
        const unsigned char* start = arena_align_ptr(arena, offset, LV_ARENA_BLOCK_ALIGNMENT);
        arena->blocks[i].start = start;
        arena->blocks[i].capacity = block_size;
        arena->blocks[i].current_size = 0;
        offset += (uint64_t)(start - (unsigned char*)arena) + block_size;
    }

    if(offset > total_size){
        LIVERO_ERROR("Arena blocks (%zu bytes incl. padding) exceed total size %zu", offset, (size_t)total_size);
        munmap(p, total_size);
        return nullptr;
    }

    return arena;
}

LVSize_t arena_reserved_size(void) { return sizeof(LVArena); }
LVSize_t arena_current_usage(const LVArena* arena) {
    LVSize_t current_usage = arena_reserved_size();
    for (int i = 0; i < LV_ARENA_BLOCK_COUNT; ++i) {
        current_usage += arena->blocks[i].current_size;
    }
    return current_usage;
}

LVArenaStatus arena_write(LVArena* arena, const int block_number, const void* data, const LVSize_t data_size,
                          const int alignment, void** out_ptr) {
    assert(block_number < LV_ARENA_BLOCK_COUNT);
    assert(alignment > 0 && alignment <= LV_ARENA_BLOCK_ALIGNMENT && (alignment & (alignment - 1)) == 0);

    LVArenaBlock* block = &arena->blocks[block_number];

    if (block->current_size >= block->capacity) {
        return LV_ARENA_BLOCK_FULL;
    }

    unsigned char* ptr_to_write = (unsigned char*)block->start + block->current_size;
    uint32_t offset_to_align =
        (((uintptr_t)(ptr_to_write) + (alignment - 1)) & ~(alignment - 1)) - (uintptr_t)ptr_to_write;

    if (offset_to_align + data_size > block->capacity - block->current_size) {
        return LV_ARENA_DATA_NOT_FIT;
    }

    ptr_to_write += offset_to_align;

    block->current_size += data_size + offset_to_align;

    if (data) {
        memcpy(ptr_to_write, data, data_size);
    }
    if (out_ptr) {
        *out_ptr = ptr_to_write;
    }

    return LV_ARENA_OK;
}

void arena_reset_block(LVArena* arena, const int block_number) {
    assert(block_number < LV_ARENA_BLOCK_COUNT);
    arena->blocks[block_number].current_size = 0;
}
