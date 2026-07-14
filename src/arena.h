#ifndef ARENA
#define ARENA

/*
 * arena.h — block-list bump allocator
 *
 * WHAT
 *   A region allocator that hands out memory by bumping an offset within a
 *   fixed-size block. When the current block can't fit a request, a new block
 *   is malloc'd and linked to the previous one (singly-linked via `prev`).
 *   There is no per-object free: the whole arena — every block — is released
 *   at once by arena_destroy.
 *
 * WHY
 *   livero allocates many small objects that share one lifetime and die
 *   together — e.g. every node in a memtable is freed in one shot when the
 *   memtable is flushed. A bump allocator makes each allocation O(1) (just an
 *   offset advance) with no fragmentation and no per-object bookkeeping, and
 *   tears the whole batch down in a single pass. That's far cheaper than a
 *   malloc/free per node, and it removes a whole class of use-after-free /
 *   leak bugs, since ownership is the arena's, not each caller's.
 *
 * LIFETIME  (the one rule that matters)
 *   Every pointer arena_allocate returns is valid until arena_destroy. Never
 *   free an individual allocation, and never keep a pointer past the arena's
 *   lifetime. Allocations do not move, so pointers stay stable until destroy.
 *
 * ALIGNMENT
 *   arena_allocate takes an explicit `align`. This matters because vectors
 *   stored in the arena are later read with SIMD loads, and ARM faults on
 *   under-aligned access — so callers pass the alignment the data needs
 *   (align <= 0 falls back to alignof(max_align_t)).
 *
 * OVERSIZE REQUESTS
 *   A request larger than block_capacity gets its own dedicated block sized
 *   exactly to the request, linked into the same chain and freed the same way.
 */

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
