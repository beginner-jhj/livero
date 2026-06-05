#include "arena.h"
#include <stdlib.h>
#include <stdalign.h>
#include "helper.h"

LVArena *arena_create(const LVSize32_t block_size)
{
    int error_flag = 0;
    LVArenaBlock *initial_block = NULL;
    void *block_buffer = NULL;
    LVArena *arena = NULL;

    if(block_size < 0){
        goto cleanup;
    }

    initial_block = malloc(sizeof(LVArenaBlock));
    if (!initial_block) {
        error_flag = 1;
        goto cleanup;
    }

    initial_block->buffer = NULL;
    initial_block->prev = NULL;
    initial_block->capacity = 0;

    block_buffer = malloc(block_size);
    if (!block_buffer) {
        error_flag = 1;
        goto cleanup;
    }

    initial_block->buffer = block_buffer;

    arena = malloc(sizeof(LVArena));
    if (!arena) {
        error_flag = 1;
        goto cleanup;
    }

    arena->current_block = initial_block;
    arena->current_offset = 0;
    arena->block_size = block_size;

cleanup:
    if (error_flag) {
        safe_free(&block_buffer);
        safe_free(&initial_block);
        safe_free(&arena);
    }

    return arena;
}

void arena_destroy(LVArena *arena)
{
    if (arena)
    {
        LVArenaBlock *current = arena->current_block;
        while (current)
        {
            LVArenaBlock *prev = current->prev;
            safe_free(&current->buffer);
            safe_free(&current);
            current = prev;
        }
        safe_free(&arena);
    }
}

void *arena_allocate(LVArena *arena, const LVSize32_t total, int32_t align)
{
    if (align <= 0)
    {
        align = alignof(max_align_t);
    }
    LVSize32_t aligned_offset = (arena->current_offset + align - 1) & ~(align - 1);

    void *result = NULL;

    if (total > arena->block_size)
    {
        int error_flag = 0;
        LVArenaBlock *large_block = NULL;
        void *large_buffer = NULL;
        LVArenaBlock *new_block = NULL;
        void *new_buffer = NULL;

        large_block = malloc(sizeof(LVArenaBlock));
        if (!large_block) {
            error_flag = 1;
            goto cleanup;
        }

        large_buffer = malloc(total);
        if (!large_buffer) {
            error_flag = 1;
            goto cleanup;
        }

        large_block->buffer = large_buffer;
        large_block->prev = arena->current_block;
        large_block->capacity = total;

        new_block = malloc(sizeof(LVArenaBlock));
        if (!new_block) {
            error_flag = 1;
            goto cleanup;
        }

        new_buffer = malloc(arena->block_size);
        if (!new_buffer) {
            error_flag = 1;
            goto cleanup;
        }

        new_block->buffer = new_buffer;
        new_block->prev = large_block;
        new_block->capacity = arena->block_size;

        arena->current_block = new_block;
        arena->current_offset = 0;
        result = large_block->buffer;

    cleanup:
        if (error_flag) {
            safe_free(&large_buffer);
            safe_free(&large_block);
            safe_free(&new_buffer);
            safe_free(&new_block);
        }

        goto _return;
    }

    if (aligned_offset + total > arena->block_size)
    {
        LVArenaBlock *new_block = NULL;
        void *new_buffer = NULL;

        new_block = malloc(sizeof(LVArenaBlock));
        if (!new_block) {
            goto _return;
        }

        new_buffer = malloc(arena->block_size);
        if (!new_buffer) {
            goto _return;
        }

        new_block->buffer = new_buffer;
        new_block->prev = arena->current_block;
        new_block->capacity = arena->block_size;

        arena->current_block = new_block;
        arena->current_offset = total;
        result = arena->current_block->buffer;

        goto _return;
    }

    result = (char *)arena->current_block->buffer + aligned_offset;
    arena->current_offset = aligned_offset + total;
_return:
    return result;
}
