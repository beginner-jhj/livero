#ifndef LV_ARENA
#define LV_ARENA

#include <stdint.h>
#include "livero_types.h"

#define LV_ARENA_BLOCK_COUNT 4
#define LV_ARENA_BLOCK_ALIGNMENT 64

typedef struct LVArenaBlock{
    void* start;  // block start address must be 64 aligned
    LVSize_t current_size;
    LVSize_t capacity;
} LVArenaBlock; //16bytes

typedef struct LVArena {
    LVSize_t max_size;  // max ram bytes capacity
    LVArenaBlock blocks[LV_ARENA_BLOCK_COUNT]; //48bytes
    uint8_t __pad[8];
} __attribute__((aligned(64))) LVArena;

typedef enum LVArenaStatus:uint8_t {
    LV_ARENA_OK = 0,
    LV_ARENA_OOM = 1,
    LV_ARENA_BLOCK_FULL = 2, //There is no available space in a block. 
    LV_ARENA_DATA_NOT_FIT = 3 //There is available space in the block, but requested data exceeds block's capacity.
} LVArenaStatus;

LVArena* arena_create(const LVSize_t total_size, const LVSize_t (*block_sizes)[LV_ARENA_BLOCK_COUNT]);

LVSize_t arena_reserved_size(void);
LVSize_t arena_current_usage(const LVArena* arena);
/*
    Only appends data to the end of the block's current offset.
    If data is not NULL, it directly writes data on the memory else just returns pointer.
*/
LVArenaStatus arena_write(LVArena* arena, const int block_number, const void* data, const LVSize_t data_size, const int alignment, void** out_ptr);

/*
    Resets the block's size
*/
void arena_reset_block(LVArena* arena, const int block_number);

#endif
