#ifndef LV_SIMD
#define LV_SIMD

#include "livero_types.h"
#include "ternary.h"

int64_t ternary_dot_simd(const LVTernaryChunk* ternary_a, const LVTernaryChunk* ternary_b, const LVCount_t ternary_chunk_count);

#endif
