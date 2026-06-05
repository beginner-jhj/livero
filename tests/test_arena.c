/*
 * test_arena.c — Unit tests for the LVArena allocator
 *
 * Tests are grouped by behavior, not by function name,
 * because a single function can have multiple distinct code paths.
 *
 * Run with valgrind to catch leaks and invalid frees:
 *   valgrind --leak-check=full ./test_arena
 */

#include "arena.h"
#include "test_util.h"
#include <stdalign.h>

/* ============================================================
 * Helpers
 * ============================================================ */

/*
 * Returns the alignment guarantee of the arena.
 * Mirrors the logic inside arena_allocate() so tests stay in sync
 * if the alignment policy ever changes.
 */
static LVSize32_t arena_alignment(void)
{
    return (LVSize32_t)alignof(max_align_t);
}

/*
 * Returns true if `ptr` is aligned to the arena's natural alignment.
 * We cast to uintptr_t so bitwise ops are well-defined on pointer values.
 */
static int is_arena_aligned(void *ptr)
{
    LVSize32_t align = arena_alignment();
    return IS_ALIGNED(ptr, align); /* macro from test_util.h */
}

/* ============================================================
 * Group 1 — arena_create
 * ============================================================ */

static void test_create_arena(void)
{
    int n = 1;
    printf("\n=== arena_create ===\n");

    /* 1-1. arena_create() must return a non-NULL pointer. */
    TEST_START(n++, "create returns non-null");
    LVArena *arena = arena_create(LV_DEFAULT_BLOCK_SIZE);
    expect_ptr_not_null(arena, "create returns non-null");

    if (!arena) {
        printf("    (skipping remaining create tests — arena is NULL)\n");
        return;
    }

    /* 1-2. After creation, the offset must be 0.
     *      Any other value means the arena thinks space is already used. */
    TEST_START(n++, "initial offset is zero");
    uint32_t expected_zero = 0;
    uint32_t actual_offset = arena->current_offset;
    expect(&actual_offset, sizeof(actual_offset),
           &expected_zero, sizeof(expected_zero),
           "initial offset is zero");

    /* 1-3. The first block must be allocated and non-NULL. */
    TEST_START(n++, "first block exists");
    expect_ptr_not_null(arena->current_block, "first block exists");

    if (arena->current_block) {
        /* 1-4. The block's data buffer must also be non-NULL. */
        TEST_START(n++, "first block data buffer exists");
        expect_ptr_not_null(arena->current_block->buffer, "first block data buffer exists");

        /* 1-5. A fresh arena has no previous block — it's a single-block chain. */
        TEST_START(n++, "no previous block on fresh arena");
        expect_ptr_null(arena->current_block->prev, "no previous block on fresh arena");
    }

    arena_destroy(arena);
}

/* ============================================================
 * Group 2 — basic allocation
 * ============================================================ */

static void test_basic_allocation(void)
{
    int n = 1;
    printf("\n=== basic allocation ===\n");

    LVArena *arena = arena_create(LV_DEFAULT_BLOCK_SIZE);
    if (!arena) {
        printf("    (skipping — create_arena failed)\n");
        return;
    }

    /* 2-1. A normal small allocation must succeed (non-NULL). */
    TEST_START(n++, "small alloc returns non-null");
    void *p = arena_allocate(arena, 16,-1);
    expect_ptr_not_null(p, "small alloc returns non-null");

    /* 2-2. The returned pointer must satisfy the alignment requirement.
     *      Misaligned pointers can cause SIGBUS on some architectures,
     *      and undefined behavior when casting to typed structs. */
    TEST_START(n++, "small alloc is aligned");
    expect_aligned(p, arena_alignment(), "small alloc is aligned");

    /* 2-3. We must be able to write to the allocated region without crashing.
     *      This also serves as a basic sanity check that the pointer is valid. */
    TEST_START(n++, "small alloc is writable");
    if (p) {
        memset(p, 0xAB, 16);
        uint8_t *bytes = (uint8_t *)p;
        int all_match = 1;
        for (int i = 0; i < 16; i++) {
            if (bytes[i] != 0xAB) { all_match = 0; break; }
        }
        expect_true(all_match, "small alloc is writable");
    }

    arena_destroy(arena);
}

/* ============================================================
 * Group 3 — consecutive allocations don't overlap
 *
 * The most important invariant of any allocator: two distinct
 * allocations must not share any byte of memory.
 * ============================================================ */

static void test_no_overlap(void)
{
    int n = 1;
    printf("\n=== consecutive allocations don't overlap ===\n");

    LVArena *arena = arena_create(LV_DEFAULT_BLOCK_SIZE);
    if (!arena) {
        printf("    (skipping — create_arena failed)\n");
        return;
    }

    /* Allocate several regions and write a unique byte pattern to each. */
    const int   COUNT = 8;
    const size_t EACH  = 32;
    void *ptrs[8];

    for (int i = 0; i < COUNT; i++) {
        ptrs[i] = arena_allocate(arena, (LVSize32_t)EACH,-1);
        if (ptrs[i]) {
            /* Write a distinct pattern so we can detect overwrites */
            memset(ptrs[i], (uint8_t)(0x10 + i), EACH);
        }
    }

    /* 3-1. After writing, each region must still contain its own pattern.
     *      If any byte has been overwritten, allocations overlap. */
    TEST_START(n++, "no overlap between 8 consecutive 32-byte allocations");
    int overlap_detected = 0;
    for (int i = 0; i < COUNT; i++) {
        if (!ptrs[i]) { overlap_detected = 1; break; }
        uint8_t *bytes = (uint8_t *)ptrs[i];
        for (size_t j = 0; j < EACH; j++) {
            if (bytes[j] != (uint8_t)(0x10 + i)) {
                overlap_detected = 1;
                printf("    overlap at alloc[%d] byte[%zu]: got 0x%02x, expected 0x%02x\n",
                       i, j, bytes[j], (uint8_t)(0x10 + i));
                break;
            }
        }
    }
    expect_false(overlap_detected, "no overlap between 8 consecutive 32-byte allocations");

    /* 3-2. All returned pointers must be distinct addresses. */
    TEST_START(n++, "all allocation addresses are distinct");
    int duplicate_found = 0;
    for (int i = 0; i < COUNT && !duplicate_found; i++) {
        for (int j = i + 1; j < COUNT && !duplicate_found; j++) {
            if (ptrs[i] && ptrs[j] && ptrs[i] == ptrs[j]) {
                duplicate_found = 1;
                printf("    ptrs[%d] == ptrs[%d] == %p\n", i, j, ptrs[i]);
            }
        }
    }
    expect_false(duplicate_found, "all allocation addresses are distinct");

    arena_destroy(arena);
}

/* ============================================================
 * Group 4 — alignment across multiple allocations
 *
 * Every allocation must be aligned, not just the first one.
 * Odd-sized allocations can misalign the internal offset, and
 * arena_allocate() must round up before serving the next request.
 * ============================================================ */

static void test_alignment(void)
{
    int n = 1;
    printf("\n=== alignment ===\n");

    LVArena *arena = arena_create(LV_DEFAULT_BLOCK_SIZE);
    if (!arena) {
        printf("    (skipping — create_arena failed)\n");
        return;
    }

    LVSize32_t align = arena_alignment();

    /* 4-1. Allocation of size 1 — the next alloc must still be aligned. */
    TEST_START(n++, "alloc(1) result is aligned");
    void *p1 = arena_allocate(arena, 1,-1);
    expect_aligned(p1, align, "alloc(1) result is aligned");

    /* 4-2. After a 1-byte alloc, offset is not a multiple of align.
     *      The arena must pad before the next alloc. */
    TEST_START(n++, "alloc after alloc(1) is aligned");
    void *p2 = arena_allocate(arena, 16,-1);
    expect_aligned(p2, align, "alloc after alloc(1) is aligned");

    /* 4-3. Odd-size chain: 3, 5, 7 bytes — each result must be aligned. */
    const LVSize32_t odd_sizes[] = {3, 5, 7};
    for (int i = 0; i < 3; i++) {
        char label[64];
        snprintf(label, sizeof(label), "alloc(%u) is aligned", odd_sizes[i]);
        TEST_START(n++, label);
        void *p = arena_allocate(arena, odd_sizes[i],-1);
        expect_aligned(p, align, label);
    }

    arena_destroy(arena);
}

/* ============================================================
 * Group 5 — block overflow (new block allocation)
 *
 * When the remaining space in the current block is not enough,
 * arena_allocate() must allocate a new block and serve from it.
 * ============================================================ */

static void test_block_overflow(void)
{
    int n = 1;
    printf("\n=== block overflow ===\n");

    LVArena *arena = arena_create(LV_DEFAULT_BLOCK_SIZE);
    if (!arena) {
        printf("    (skipping — arena_create failed)\n");
        return;
    }

    LVArenaBlock *first_block = arena->current_block;

    /* Fill the first block almost completely.
     * We allocate (LV_DEFAULT_BLOCK_SIZE - align) bytes so the next
     * allocation of `align` bytes will just barely not fit. */
    LVSize32_t align    = arena_alignment();
    LVSize32_t fill     = LV_DEFAULT_BLOCK_SIZE - align;
    void *fill_ptr = arena_allocate(arena, fill,-1);

    TEST_START(n++, "fill allocation succeeds");
    expect_ptr_not_null(fill_ptr, "fill allocation succeeds");

    /* 5-1. Now allocate something that won't fit in the remaining space.
     *      The arena must create a new block. */
    TEST_START(n++, "overflow triggers new block");
    void *overflow_ptr = arena_allocate(arena, align * 2,-1);
    expect_ptr_not_null(overflow_ptr, "overflow triggers new block");

    /* If a new block was created, current_block must have changed. */
    TEST_START(n++, "current_block pointer changed after overflow");
    expect_true(arena->current_block != first_block,
                "current_block pointer changed after overflow");

    /* 5-2. The new block must link back to the old one via ->prev. */
    TEST_START(n++, "new block's prev points to previous block");
    if (arena->current_block) {
        expect_true(arena->current_block->prev == first_block,
                    "new block's prev points to previous block");
    }

    /* 5-3. The overflow pointer must still be aligned. */
    TEST_START(n++, "overflow allocation is aligned");
    expect_aligned(overflow_ptr, align, "overflow allocation is aligned");

    arena_destroy(arena);
}

/* ============================================================
 * Group 6 — dedicated block (total > LV_DEFAULT_BLOCK_SIZE)
 *
 * Large allocations that exceed one block's size get their own
 * dedicated block. After serving the large alloc, the arena
 * creates a fresh normal block so subsequent small allocs work.
 * ============================================================ */

static void test_dedicated_block(void)
{
    int n = 1;
    printf("\n=== dedicated block (total > LV_DEFAULT_BLOCK_SIZE) ===\n");

    LVArena *arena = arena_create(LV_DEFAULT_BLOCK_SIZE);
    if (!arena) {
        printf("    (skipping — arena_create failed)\n");
        return;
    }

    /* 6-1. Request more than LV_DEFAULT_BLOCK_SIZE bytes. */
    LVSize32_t large = LV_DEFAULT_BLOCK_SIZE + 1;
    TEST_START(n++, "large alloc (> block size) returns non-null");
    void *large_ptr = arena_allocate(arena, large,-1);
    expect_ptr_not_null(large_ptr, "large alloc (> block size) returns non-null");

    /* 6-2. After the large alloc, the arena must have reset to a fresh
     *      normal block (offset == 0), so small allocs still work. */
    TEST_START(n++, "offset reset to 0 after dedicated block");
    uint32_t expected_zero = 0;
    expect(&arena->current_offset, sizeof(arena->current_offset),
           &expected_zero, sizeof(expected_zero),
           "offset reset to 0 after dedicated block");

    /* 6-3. Small alloc after large alloc must succeed. */
    TEST_START(n++, "small alloc after large alloc succeeds");
    void *small_ptr = arena_allocate(arena, 16,-1);
    expect_ptr_not_null(small_ptr, "small alloc after large alloc succeeds");

    /* 6-4. The two pointers must be different (no overlap). */
    TEST_START(n++, "large and small alloc addresses differ");
    expect_true(large_ptr != small_ptr,
                "large and small alloc addresses differ");

    /* 6-5. Write to the large region to confirm it's actually usable. */
    if (large_ptr) {
        TEST_START(n++, "large alloc region is writable");
        memset(large_ptr, 0xFF, large);
        uint8_t *bytes = (uint8_t *)large_ptr;
        int ok = 1;
        for (LVSize32_t i = 0; i < large; i++) {
            if (bytes[i] != 0xFF) { ok = 0; break; }
        }
        expect_true(ok, "large alloc region is writable");
    }

    arena_destroy(arena);
}

/* ============================================================
 * Group 7 — stress test
 *
 * Random-sized allocations across many blocks, seeded for
 * reproducibility. Checks alignment and non-null on every alloc.
 * ============================================================ */

static void test_stress(void)
{
    int n = 1;
    printf("\n=== stress test ===\n");

    /* Fixed seed so this test is reproducible. Change to time(NULL)
     * for fuzzing, but then failures won't be reproducible. */
    lv_rand_seed(42);

    LVArena *arena = arena_create(LV_DEFAULT_BLOCK_SIZE);
    if (!arena) {
        printf("    (skipping — arena_create failed)\n");
        return;
    }

    const int     ITERS     = 500;
    LVSize32_t    align      = arena_alignment();
    int           null_count = 0;
    int           misaligned = 0;

    lv_timer_t timer = lv_timer_start();

    for (int i = 0; i < ITERS; i++) {
        /* Mix of small (1–256) and occasionally large (> block size) allocs */
        LVSize32_t size;
        if (lv_rand_u32() % 20 == 0) {
            /* ~5% chance of a dedicated-block allocation */
            size = LV_DEFAULT_BLOCK_SIZE + (LVSize32_t)lv_rand_int_range(1, 512);
        } else {
            size = (LVSize32_t)lv_rand_int_range(1, 256);
        }

        void *p = arena_allocate(arena, size,-1);

        if (!p) {
            null_count++;
        } else if (!IS_ALIGNED(p, align)) {
            misaligned++;
            printf("    misaligned pointer at iter %d: %p (size=%u)\n", i, p, size);
        } else {
            /* Write to every byte to catch any buffer overrun */
            memset(p, (uint8_t)(i & 0xFF), size);
        }
    }

    double elapsed = lv_timer_elapsed_ms(timer);

    TEST_START(n++, "no null returns in stress test");
    expect_true(null_count == 0, "no null returns in stress test");
    if (null_count) printf("    null count: %d / %d\n", null_count, ITERS);

    TEST_START(n++, "no misaligned pointers in stress test");
    expect_true(misaligned == 0, "no misaligned pointers in stress test");

    printf("    %d allocations completed in %.3f ms\n", ITERS, elapsed);

    arena_destroy(arena);
}

/* ============================================================
 * Group 8 — destroy_arena
 *
 * There's no direct way to assert "no leak" from C code —
 * run under valgrind for that. What we CAN check is that
 * arena_destroy(NULL) doesn't crash (defensive null check).
 * ============================================================ */

static void test_destroy(void)
{
    int n = 1;
    printf("\n=== arena_destroy ===\n");

    /* 8-1. destroy_arena(NULL) must not crash. */
    TEST_START(n++, "arena_destroy(NULL) does not crash");
    arena_destroy(NULL); /* would segfault if there's no null guard */
    TEST_SUCCESS("arena_destroy(NULL) does not crash");

    /* 8-2. Normal destroy after allocations (valgrind will catch leaks). */
    TEST_START(n++, "destroy after multi-block usage does not crash");
    LVArena *arena = arena_create(LV_DEFAULT_BLOCK_SIZE);
    if (arena) {
        /* Force at least 2 blocks */
        arena_allocate(arena, LV_DEFAULT_BLOCK_SIZE - 1,-1);
        arena_allocate(arena, LV_DEFAULT_BLOCK_SIZE - 1,-1);
        arena_allocate(arena, LV_DEFAULT_BLOCK_SIZE + 1,-1); /* dedicated block */
        arena_destroy(arena);
        TEST_SUCCESS("destroy after multi-block usage does not crash");
    }
}

/* ============================================================
 * main
 * ============================================================ */

int main(void)
{
    printf("========================================\n");
    printf("  LightVec — LVArena Allocator Unit Tests \n");
    printf("========================================\n");

    test_create_arena();
    test_basic_allocation();
    test_no_overlap();
    test_alignment();
    test_block_overflow();
    test_dedicated_block();
    test_stress();
    test_destroy();

    printf("\n========================================\n");
    printf("  Done. Run with valgrind for leak check.\n");
    printf("========================================\n");

    return 0;
}
