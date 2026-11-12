#include <stdint.h>

#include "livero_types.h"
#include "tests_helper.h"
#include "vector/simd.h"
#include "vector/ternary.h"

#define TEST_MAX_TERNARY_CHUNK_COUNT TEST_TERNARY_CHUNK_COUNT
#define TEST_TERNARY_CHUNK_COUNT_N 6

static LVCount_t TEST_TERNARY_CHUNKS[TEST_TERNARY_CHUNK_COUNT_N] = {
    1, 2, 3, 4, TEST_TERNARY_CHUNK_COUNT - 1, TEST_TERNARY_CHUNK_COUNT};

static LVTernaryChunk TEST_TERNARY_A[TEST_MAX_TERNARY_CHUNK_COUNT] = {0};
static LVTernaryChunk TEST_TERNARY_B[TEST_MAX_TERNARY_CHUNK_COUNT] = {0};

static void fill_test_ternary_pair(const LVCount_t count) {
    memset(TEST_TERNARY_A, 0, sizeof(TEST_TERNARY_A));
    memset(TEST_TERNARY_B, 0, sizeof(TEST_TERNARY_B));
    for (LVCount_t i = 0; i < count; ++i) {
        TEST_TERNARY_A[i].mask = rand_uint64();
        TEST_TERNARY_A[i].sign = rand_uint64();
        TEST_TERNARY_B[i].mask = rand_uint64();
        TEST_TERNARY_B[i].sign = rand_uint64();
    }
}

int test_ternary_dot(void) {
    for (int i = 0; i < TEST_TERNARY_CHUNK_COUNT_N; ++i) {
        const LVCount_t chunk_count = TEST_TERNARY_CHUNKS[i];
        if (chunk_count > TEST_MAX_TERNARY_CHUNK_COUNT) {
            fprintf(stderr, "Invalid chunk count. Max is %u, but got %u . \n", TEST_MAX_TERNARY_CHUNK_COUNT,
                    chunk_count);
            return -1;
        }
        fill_test_ternary_pair(chunk_count);
        const int64_t scalar = ternary_dot_scalar(TEST_TERNARY_A, TEST_TERNARY_B, chunk_count);
        const int64_t simd = ternary_dot_simd(TEST_TERNARY_A, TEST_TERNARY_B, chunk_count);

        if (scalar != simd) {
            fprintf(stderr, "Simd ternary dot is incorrect at chunk count %u. \n", chunk_count);
            PRINT_ERR_INVALID("Ternary dot", "%" PRId64, scalar, simd);
            return -1;
        }
    }
    fprintf(stdout, "Simd Ternary dot product passed. \n");
    return 0;
}

int main(void) {
#if defined(LV_USE_SIMD)
    fprintf(stdout, "LV_USE_SIMD is on. Start simd test... \n");
    return test_ternary_dot();
#else
    fprintf(stdout, "LV_USE_SIMD is off. Skip simd test... \n");
    return 0;
#endif
}
