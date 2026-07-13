#ifndef ARENA
#define ARENA

#include "lv_internal.h"

typedef struct LVArenaBlock{
    void* buffer;
    struct LVArenaBlock* prev;
} LVArenaBlock;

typedef struct LVArena {
    LVArenaBlock* current_block;
    LVSize32_t current_offset;
    LVSize32_t block_capacity;
} LVArena;

LVArena* arena_create(const LVSize32_t block_capacity);

void arena_destroy(LVArena* arena);

void* arena_allocate(LVArena* arena,const LVSize32_t total,int32_t align);

#endif
