#ifndef ARENA
#define ARENA

#include "lv_internal.h"


#define BLOCK_DEFAULT_SIZE 4096 //4kb

typedef struct Block{
    void* data;
    LVSize32_t size;
    struct Block* prev;
} Block;

typedef struct Arena {
    Block* current_block;
    LVSize32_t current_offset;
} Arena;

Arena* create_arena(void);

void destroy_arena(Arena* arena);

void* arena_allocate(Arena* arena,const LVSize32_t total,LVSize32_t align);

#endif
