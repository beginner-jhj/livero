#ifndef TEST_UTIL_H
#define TEST_UTIL_H

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>   /* clock_gettime, struct timespec */
#include <stdlib.h> /* rand, srand */
#include <float.h>  /* FLT_MAX */

/* ============================================================
 * Basic test macros
 * ============================================================ */

#define TEST_START(n, name) printf("[%d] %s ...\n", n, name)
#define TEST_FAIL(name)     printf("    FAIL  — %s\n", name)
#define TEST_SUCCESS(name)  printf("    OK    — %s\n", name)

/* ============================================================
 * expect — byte-level equality assertion
 *
 * Compares `input` and `target` byte-for-byte.
 * Fails if sizes differ or bytes don't match.
 * ============================================================ */
static void expect(const void    *input,
                   const uint32_t input_size,
                   const void    *target,
                   const uint32_t target_size,
                   const char    *test_name)
{
    if (input_size != target_size) {
        TEST_FAIL(test_name);
        printf("    size mismatch: got %u, expected %u\n",
               input_size, target_size);
        return;
    }

    if (memcmp(input, target, input_size) == 0) {
        TEST_SUCCESS(test_name);
    } else {
        TEST_FAIL(test_name);
    }
}

/* ============================================================
 * expect_true / expect_false — boolean assertions
 *
 * Lightweight wrappers for conditions that don't need
 * byte-level comparison (e.g. pointer != NULL, alignment check).
 * ============================================================ */
static void expect_true(int condition, const char *test_name)
{
    if (condition) {
        TEST_SUCCESS(test_name);
    } else {
        TEST_FAIL(test_name);
        printf("    expected: true, got: false\n");
    }
}

static void expect_false(int condition, const char *test_name)
{
    if (!condition) {
        TEST_SUCCESS(test_name);
    } else {
        TEST_FAIL(test_name);
        printf("    expected: false, got: true\n");
    }
}

/* ============================================================
 * expect_ptr — pointer equality assertion
 *
 * Useful for checking that returned pointers are NULL or non-NULL.
 * ============================================================ */
static void expect_ptr_not_null(const void *ptr, const char *test_name)
{
    if (ptr != NULL) {
        TEST_SUCCESS(test_name);
    } else {
        TEST_FAIL(test_name);
        printf("    expected non-NULL pointer\n");
    }
}

static void expect_ptr_null(const void *ptr, const char *test_name)
{
    if (ptr == NULL) {
        TEST_SUCCESS(test_name);
    } else {
        TEST_FAIL(test_name);
        printf("    expected NULL pointer, got: %p\n", ptr);
    }
}

/* ============================================================
 * Alignment check
 *
 * LVArena allocations often need to be aligned to a power of 2
 * (e.g. 8 bytes on 64-bit systems) to avoid unaligned access
 * which can cause crashes or slowdowns depending on the CPU.
 *
 * IS_ALIGNED(ptr, align):
 *   - align must be a power of 2 (e.g. 4, 8, 16)
 *   - uses bitwise AND: if the low bits of the address are
 *     all zero, the address is a multiple of `align`
 *   - e.g. 0x...FF8 & 7 == 0  → 8-byte aligned ✓
 *          0x...FF9 & 7 == 1  → not aligned    ✗
 * ============================================================ */
#define IS_ALIGNED(ptr, align) \
    (((uintptr_t)(ptr) & ((align) - 1)) == 0)

static void expect_aligned(const void *ptr, size_t align, const char *test_name)
{
    if (IS_ALIGNED(ptr, align)) {
        TEST_SUCCESS(test_name);
    } else {
        TEST_FAIL(test_name);
        printf("    pointer %p is not aligned to %zu bytes\n", ptr, align);
    }
}

/* ============================================================
 * Timer — nanosecond resolution wall clock
 *
 * Uses CLOCK_MONOTONIC which is guaranteed to be monotonically
 * increasing and not affected by system time changes (unlike
 * CLOCK_REALTIME). Good for benchmarking.
 *
 * Usage:
 *   lv_timer_t t = lv_timer_start();
 *   // ... work ...
 *   double ms = lv_timer_elapsed_ms(t);
 * ============================================================ */
typedef struct timespec lv_timer_t;

static lv_timer_t lv_timer_start(void)
{
    lv_timer_t t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t;
}

/* Returns elapsed time in milliseconds (double for sub-ms precision). */
static double lv_timer_elapsed_ms(lv_timer_t start)
{
    lv_timer_t now;
    clock_gettime(CLOCK_MONOTONIC, &now);

    /* timespec stores seconds and nanoseconds separately,
     * so we combine them: total_ns = Δsec * 1e9 + Δnsec */
    long delta_sec  = now.tv_sec  - start.tv_sec;
    long delta_nsec = now.tv_nsec - start.tv_nsec;

    return (double)delta_sec * 1e3 + (double)delta_nsec * 1e-6;
}

/* Returns elapsed time in nanoseconds (uint64_t). */
static uint64_t lv_timer_elapsed_ns(lv_timer_t start)
{
    lv_timer_t now;
    clock_gettime(CLOCK_MONOTONIC, &now);

    long delta_sec  = now.tv_sec  - start.tv_sec;
    long delta_nsec = now.tv_nsec - start.tv_nsec;

    return (uint64_t)delta_sec * 1000000000ULL + (uint64_t)delta_nsec;
}

/* ============================================================
 * RNG — xorshift64 pseudo-random number generator
 *
 * xorshift is a fast, lightweight PRNG that works by XORing a
 * value with shifted versions of itself. NOT cryptographically
 * secure, but perfectly fine for test data generation.
 *
 * The global seed is initialized once via lv_rand_seed().
 * If you don't call it, the seed defaults to 1 (still works,
 * but every run will produce the same sequence).
 * ============================================================ */
static uint64_t _lv_rand_state = 1;

/* Seed the RNG. Pass time(NULL) for different results each run,
 * or a fixed value for reproducible test sequences. */
static void lv_rand_seed(uint64_t seed)
{
    /* xorshift64 must never have state == 0, it would get stuck */
    _lv_rand_state = seed ? seed : 1;
}

/* Core xorshift64 step — returns next pseudo-random uint64_t. */
static uint64_t lv_rand_u64(void)
{
    uint64_t x = _lv_rand_state;
    x ^= x << 13;  /* shift left, XOR — mixes the bits */
    x ^= x >> 7;   /* shift right, XOR — different direction */
    x ^= x << 17;  /* shift left again — final mixing */
    _lv_rand_state = x;
    return x;
}

/* Random uint32_t — upper half of the 64-bit output */
static uint32_t lv_rand_u32(void)
{
    return (uint32_t)(lv_rand_u64() >> 32);
}

/* Random int in [min, max] (inclusive).
 * Uses modulo — slight bias for large ranges, fine for testing. */
static int lv_rand_int_range(int min, int max)
{
    uint32_t range = (uint32_t)(max - min + 1);
    return min + (int)(lv_rand_u32() % range);
}

/* Random float in [0.0, 1.0) */
static float lv_rand_f32(void)
{
    /* Divide by 2^32 to map uint32 → [0, 1).
     * The cast to double prevents precision loss mid-division. */
    return (float)((double)lv_rand_u32() / 4294967296.0);
}

/* Random float in [min, max) */
static float lv_rand_f32_range(float min, float max)
{
    return min + lv_rand_f32() * (max - min);
}

/* ============================================================
 * Random data generation
 *
 * Fills a buffer with random bytes. Useful for generating
 * arbitrary keys or values to stress-test allocations.
 * ============================================================ */
static void lv_rand_bytes(void *buf, size_t len)
{
    uint8_t *p = (uint8_t *)buf;
    size_t   i;

    for (i = 0; i + 8 <= len; i += 8) {
        /* Write 8 bytes at a time using the 64-bit output */
        uint64_t r = lv_rand_u64();
        memcpy(p + i, &r, 8);
    }

    /* Handle remaining bytes (len not a multiple of 8) */
    if (i < len) {
        uint64_t r = lv_rand_u64();
        memcpy(p + i, &r, len - i);
    }
}

/* Fill buf with random printable ASCII characters (0x20–0x7E).
 * Handy for generating human-readable test keys. */
static void lv_rand_string(char *buf, size_t len)
{
    size_t i;
    for (i = 0; i < len; i++) {
        /* Map to printable ASCII range [32, 126] */
        buf[i] = (char)lv_rand_int_range(0x20, 0x7E);
    }
    buf[len] = '\0'; /* null-terminate so it's a valid C string */
}

/* ============================================================
 * Print helpers
 * ============================================================ */

/* Dump `len` bytes from `buf` as hex. Useful for inspecting
 * raw memory contents when a test fails. */
static void lv_print_hex(const char *label, const void *buf, size_t len)
{
    const uint8_t *p = (const uint8_t *)buf;
    size_t         i;

    printf("%s: ", label);
    for (i = 0; i < len; i++) {
        printf("%02x ", p[i]);
    }
    printf("\n");
}

#endif /* TEST_UTIL_H */
