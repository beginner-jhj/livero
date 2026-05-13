#ifndef ARENA
#define ARENA

#include "lv_internal.h"

typedef struct Block{
    void* data;
    LVSize32_t size;
    struct Block* prev;
} Block;

typedef struct Arena {
    Block* current_block;
    LVSize32_t current_offset;
    LVSize32_t block_size;
} Arena;

Arena* create_arena(const LVSize32_t block_size);

void destroy_arena(Arena* arena);

void* arena_allocate(Arena* arena,const LVSize32_t total,LVSize32_t align);

#endif
