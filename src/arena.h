#ifndef ARENA
#define ARENA

#include "lv_internal.h"

typedef struct LVArenaBlock{
    void* data;
    LVSize32_t size;
    struct LVArenaBlock* prev;
} LVArenaBlock;

typedef struct LVArena {
    LVArenaBlock* current_block;
    LVSize32_t current_offset;
    LVSize32_t block_size;
} LVArena;

LVArena* create_arena(const LVSize32_t block_size);

void destroy_arena(LVArena* arena);

void* arena_allocate(LVArena* arena,const LVSize32_t total,int32_t align);

#endif
