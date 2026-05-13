#include "arena.h"
#include <stdlib.h>
#include <stdalign.h>
#include "helper.h"

Arena *create_arena(const LVSize32_t block_size)
{
    int flag = 0;
    Arena *arena = NULL;
    Block *block = NULL;
    void *block_data = NULL;

    Block *block_temp = malloc(sizeof(Block));
    if (!block_temp)
    {
        flag = 1;
        goto cleanup;
    }

    block = block_temp;

    block->data = NULL;
    block->prev = NULL;
    block->size = 0;

    block_data = malloc(block_size);
    if (!block_data)
    {
        flag = 1;
        goto cleanup;
    }

    block->data = block_data;

    Arena *arena_temp = malloc(sizeof(Arena));
    if (!arena_temp)
    {
        flag = 1;
        goto cleanup;
    }

    arena = arena_temp;

    arena->current_block = block;
    arena->current_offset = 0;
    arena->block_size = block_size;

cleanup:
    if (flag)
    {
        safe_free(&block_data);
        safe_free(&block);
        safe_free(&arena);
    }

    return arena;
}

void destroy_arena(Arena *arena)
{
    if (arena)
    {
        Block *current = arena->current_block;
        while (current)
        {
            Block *prev = current->prev;
            safe_free(&current->data);
            safe_free(&current);
            current = prev;
        }
        safe_free(&arena);
    }
}

void *arena_allocate(Arena *arena, const LVSize32_t total, LVSize32_t align)
{
    if (align == -1)
    {
        align = alignof(max_align_t);
    }
    LVSize32_t aligned_offset = (arena->current_offset + align - 1) & ~(align - 1);

    void *result = NULL;

    if (total > arena->block_size)
    {
        int flag = 0;

        Block *dedicated_block = NULL;
        void *dedicated_block_data = NULL;
        Block *normal_block = NULL;
        void *normal_block_data = NULL;

        Block *dedicated_block_temp = malloc(sizeof(Block));
        if (!dedicated_block_temp)
        {
            flag = 1;
            goto cleanup;
        }

        dedicated_block = dedicated_block_temp;

        dedicated_block->data = NULL;
        dedicated_block->prev = arena->current_block;
        dedicated_block->size = total;

        dedicated_block_data = malloc(total);
        if (!dedicated_block_data)
        {
            flag = 1;
            goto cleanup;
        }

        dedicated_block->data = dedicated_block_data;

        Block *normal_block_temp = malloc(sizeof(Block));
        if (!normal_block_temp)
        {
            flag = 1;
            goto cleanup;
        }

        normal_block = normal_block_temp;
        normal_block->data = NULL;
        normal_block->prev = NULL;
        normal_block->size = arena->block_size;

        normal_block_data = malloc(arena->block_size);
        if (!normal_block_data)
        {
            flag = 1;
            goto cleanup;
        }

        normal_block->data = normal_block_data;
        normal_block->prev = dedicated_block;

        arena->current_block = normal_block;
        arena->current_offset = 0;

        result = dedicated_block->data;

    cleanup:
        if (flag)
        {
            safe_free(&dedicated_block_data);
            safe_free(&dedicated_block);
            safe_free(&normal_block_data);
            safe_free(&normal_block);
        }

        goto _return;
    }

    if (aligned_offset + total > arena->block_size)
    {
        Block *new_block = malloc(sizeof(Block));

        if (!new_block)
        {
            goto _return;
        }

        void *temp_data = malloc(arena->block_size);

        if (!temp_data)
        {
            safe_free(&new_block);
            goto _return;
        }

        new_block->data = temp_data;
        new_block->prev = arena->current_block;
        new_block->size = arena->block_size;

        arena->current_block = new_block;
        arena->current_offset = total;

        result = arena->current_block->data;

        goto _return;
    }

    result = (char *)arena->current_block->data + aligned_offset;
    arena->current_offset = aligned_offset + total;
_return:
    return result;
}
