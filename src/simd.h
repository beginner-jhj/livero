#ifndef LV_SIMD
#define LV_SIMD

#include "livero_types.h"
#include <assert.h>

#define SIMD_ALIGN64 64

static void simd_align_guard(const void *a, const void *b, const LVDim32_t dim) {
  // Precondition: dim must be a multiple of 64 AND a, b must be 64-byte aligned.
  assert(dim % SIMD_ALIGN64 == 0);
  assert(((uintptr_t)a % SIMD_ALIGN64) == 0);
  assert(((uintptr_t)b % SIMD_ALIGN64) == 0);
}

float simd_f32_l2_sq(const float *a, const float *b, const LVDim32_t dim);
float simd_f32_dot(const float *a, const float *b, const LVDim32_t dim);
int32_t simd_i8_l2_sq(const int8_t *a, const int8_t *b, const LVDim32_t dim);
int32_t simd_i8_dot(const int8_t *a, const int8_t *b, const LVDim32_t dim);

#endif
