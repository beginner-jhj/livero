#include "arena.h"
#include "livero_types.h"
#include "lv_internal.h"
#include "test_helper.h"
#include <stdalign.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const static LVSize32_t DEFAULT_BLOCK_SIZE = LV_DEFAULT_BLOCK_SIZE; // 4kb
const static LVSize32_t MID_BLOCK_SIZE = DEFAULT_BLOCK_SIZE * 16;   // 64kb
const static LVSize32_t LARGE_BLOCK_SIZE = DEFAULT_BLOCK_SIZE * 64; // 256kb
const static LVSize32_t MINIMUM_ALLOCATION_LENGTH = 128;
const static LVSize32_t MAXIMUM_ALLOCATION_LENGTH = DEFAULT_BLOCK_SIZE * 2;

const static LVSize32_t SIZES[] = {DEFAULT_BLOCK_SIZE, MID_BLOCK_SIZE,
                                   LARGE_BLOCK_SIZE};
const static LVSize32_t SIZES_COUNT = sizeof(SIZES) / sizeof(LVSize32_t);
const static LVCount32_t ALLOCATION_COUNT = 500;

typedef struct {
  unsigned char *ptr;
  LVSize32_t size;
  unsigned char pattern; // the byte this region was filled with
} Record;

int test_arena(int32_t alignment) {

  if (alignment > 0 && ((alignment & (alignment - 1)) != 0)) {
    fprintf(stderr, "Custom alignment must be powers of 2. \n");
    return -1;
  }

  if (alignment <= 0) {
    alignment = alignof(max_align_t);
  }

  Record records[ALLOCATION_COUNT];

  for (LVSize32_t i = 0; i < SIZES_COUNT; ++i) {
    LVSize32_t block_size = SIZES[i];
    LVArena *arena = arena_create(block_size);
    if (!arena) {
      fprintf(stderr, "Arena creation failed at block size: %u \n", block_size);
      return -1;
    }
    for (LVCount32_t j = 0; j < ALLOCATION_COUNT; ++j) {
      const LVSize32_t allocation_size =
          rand_i32_range(MINIMUM_ALLOCATION_LENGTH, MAXIMUM_ALLOCATION_LENGTH);
      void *allocated = arena_allocate(arena, allocation_size, alignment);
      if (!allocated) {
        fprintf(stderr,
                "Arena allocation failed at (j:%u, allocation size: "
                "%u, block size: %u, arena offset: %u) \n",
                j, allocation_size, arena->block_capacity,
                arena->current_offset);

        return -1;
      }

      if ((uintptr_t)allocated % alignment != 0) {
        fprintf(stderr,
                "Arena align failed at (j:%u, alignment: %d, allocation size: "
                "%u, block size: %u, arena offset: %u) \n",
                j, alignment, allocation_size, arena->block_capacity,
                arena->current_offset);
        return -1;
      }

      unsigned char pattern = (unsigned char)((j + 1) & 0xFF);
      memset(allocated, pattern, allocation_size);

      records[j].ptr = allocated;
      records[j].size = allocation_size;
      records[j].pattern = pattern;
    }
    for (LVCount32_t j = 0; j < ALLOCATION_COUNT; ++j) {
      for (LVSize32_t k = 0; k < records[j].size; ++k) {
        if (records[j].ptr[k] != records[j].pattern) {
          fprintf(stderr,
                  "Overlap/corruption at allocation j:%u, byte %u: "
                  "expected 0x%02X, got 0x%02X\n",
                  j, k, records[j].pattern, records[j].ptr[k]);
          return -1;
        }
      }
    }
    arena_destroy(arena);
  }
  return 0;
}

int main(void) {
  if (test_arena(-1) != 0) // align -1 means using system max align
    return -1;

  if (test_arena(32) != 0)
    return -1;
  return 0;
}
