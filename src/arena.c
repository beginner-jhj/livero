#include "arena.h"
#include <stdlib.h>
#include <stdalign.h>
#include "helper.h"

/*
 * Create an arena with one initial block of `block_capacity` bytes.
 * Returns NULL on OOM or if block_capacity == 0. On any failure the partial
 * arena is cleaned up before returning (arena_destroy tolerates NULLs).
 */

LVArena* arena_create(const LVSize32_t block_capacity)
{
    LVArenaBlock* initial_block = NULL;
    void* block_buffer = NULL;
    LVArena* arena = NULL;

    if (block_capacity == 0) {
        goto cleanup;
    }

    arena = malloc(sizeof(LVArena));
    if (!arena) {
        goto cleanup;
    }

    arena->current_block = NULL;
    arena->current_offset = 0;
    arena->block_capacity = block_capacity;

    initial_block = malloc(sizeof(LVArenaBlock));
    if (!initial_block) goto cleanup;
    initial_block->buffer = NULL;
    initial_block->prev = NULL;
    arena->current_block = initial_block;

    block_buffer = malloc(block_capacity);
    if (!block_buffer) goto cleanup;
    initial_block->buffer = block_buffer;

    return arena;

cleanup:
    arena_destroy(arena);
    return NULL;
}

/*
 * Free every block in the chain (walking `prev`), then the arena itself.
 * Safe to call on NULL and on a partially-constructed arena.
 */
void arena_destroy(LVArena* arena)
{
    if (arena)
    {
        LVArenaBlock* current = arena->current_block;
        while (current)
        {
            LVArenaBlock* prev = current->prev;
            if (current->buffer) {
                free(current->buffer);
            }
            free(current);
            current = prev;
        }
        free(arena);
    }
}

/*
 * Allocate `total` bytes aligned to `align`, bumping the current block's
 * offset. Falls back to alignof(max_align_t) if align <= 0.
 *
 * Three cases:
 *   1. total > block_capacity : allocate a dedicated oversize block.
 *   2. doesn't fit in current block : start a fresh block.
 *   3. fits : bump the offset in the current block.
 * Returns NULL on OOM.
 */

void* arena_allocate(LVArena* arena, const LVSize32_t total, int32_t align)
{
    if (align <= 0)
    {
        align = alignof(max_align_t);
    }
    LVSize32_t aligned_offset = (arena->current_offset + align - 1) & ~(align - 1);

    LVArenaBlock* new_block = NULL;
    void* new_block_buffer = NULL;
    void* result = NULL;

    if (total > arena->block_capacity)
    {
        new_block = malloc(sizeof(LVArenaBlock));
        if (!new_block) goto cleanup;
        new_block->buffer = NULL;

        new_block_buffer = malloc(total);
        if (!new_block_buffer) goto cleanup;
        new_block->buffer = new_block_buffer;

        new_block->buffer = new_block_buffer;
        new_block->prev = arena->current_block;
        arena->current_block = new_block;
        arena->current_offset = total;

        result = new_block->buffer;
        return result;
    }

    if (aligned_offset + total > arena->block_capacity)
    {
        new_block = malloc(sizeof(LVArenaBlock));
        if (!new_block) goto cleanup;
        new_block->buffer = NULL;

        new_block_buffer = malloc(arena->block_capacity);
        if (!new_block_buffer) goto cleanup;

        new_block->buffer = new_block_buffer;
        new_block->prev = arena->current_block;
        arena->current_block = new_block;
        arena->current_offset = total;

        return new_block->buffer;
    }

    result = (char*)arena->current_block->buffer + aligned_offset;
    arena->current_offset = aligned_offset + total;
    return result;

cleanup:
    free(new_block);
    free(new_block_buffer);
    return NULL;
}
