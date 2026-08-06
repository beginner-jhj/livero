#include "livero_types.h"
#include "simd.h"
#include "test_helper.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

const static LVDim32_t DIMS[] = {
    384, 768, 1536, 3072}; // no padding needed, they are multiples of 64
const static LVCount32_t DIMS_COUNT = sizeof(DIMS) / sizeof(LVDim32_t);

static float ref_f32_l2_sq(const float *a, const float *b,
                           const LVDim32_t dim) {
  float acc = 0.0f;
  for (LVDim32_t i = 0; i < dim; ++i) {
    const float diff = a[i] - b[i];
    acc += diff * diff;
  }
  return acc;
}

static float ref_f32_dot(const float *a, const float *b, const LVDim32_t dim) {
  float acc = 0.0f;
  for (LVDim32_t i = 0; i < dim; ++i) {
    acc += a[i] * b[i];
  }
  return -acc; // smaller is closer
}

static int32_t ref_i8_l2_sq(const int8_t *a, const int8_t *b,
                            const LVDim32_t dim) {
  int32_t acc = 0;
  for (LVDim32_t i = 0; i < dim; ++i) {
    const int32_t diff = a[i] - b[i];
    acc += diff * diff;
  }
  return acc;
}

static int32_t ref_i8_dot(const int8_t *a, const int8_t *b,
                          const LVDim32_t dim) {
  int32_t acc = 0;
  for (LVDim32_t i = 0; i < dim; ++i) {
    acc += a[i] * b[i];
  }
  return -acc; // smaller is closer
}

int test_f32_l2_sq(void) {
  for (LVCount32_t i = 0; i < DIMS_COUNT; ++i) {
    const LVDim32_t dim = DIMS[i];
    float vector_a[dim];
    float vector_b[dim];
    fill_f32_vector(vector_a, dim);
    fill_f32_vector(vector_b, dim);

    const float got = simd_f32_l2_sq(vector_a, vector_b, dim);
    const float want = ref_f32_l2_sq(vector_a, vector_b, dim);
    if (!f32_close(got, want,dim)) {
      fprintf(stderr, "F32 l2_sq failed at dim=%u: got %f, want %f\n", dim, got,
              want);
      return -1;
    }
  }

  return 0;
}

int test_f32_dot(void) {
  for (LVCount32_t i = 0; i < DIMS_COUNT; ++i) {
    const LVDim32_t dim = DIMS[i];
    float vector_a[dim];
    float vector_b[dim];
    fill_f32_vector(vector_a, dim);
    fill_f32_vector(vector_b, dim);

    const float got = simd_f32_dot(vector_a, vector_b, dim);
    const float want = ref_f32_dot(vector_a, vector_b, dim);
    if (!f32_close(got, want, dim)) {
      fprintf(stderr, "F32 dot failed at dim=%u: got %f, want %f\n", dim, got,
              want);
      return -1;
    }
  }

  return 0;
}

int test_i8_l2_sq(void) {
  for (LVCount32_t i = 0; i < DIMS_COUNT; ++i) {
    const LVDim32_t dim = DIMS[i];
    int8_t vector_a[dim];
    int8_t vector_b[dim];
    fill_i8_vector(vector_a, dim);
    fill_i8_vector(vector_b, dim);

    const int32_t got = simd_i8_l2_sq(vector_a, vector_b, dim);
    const int32_t want = ref_i8_l2_sq(vector_a, vector_b, dim);
    if (got != want) {
      fprintf(stderr, "I8 l2_sq failed at dim=%u: got %d, want %d\n", dim, got,
              want);
      return -1;
    }
  }

  return 0;
}

int test_i8_dot(void) {
  for (LVCount32_t i = 0; i < DIMS_COUNT; ++i) {
    const LVDim32_t dim = DIMS[i];
    int8_t vector_a[dim];
    int8_t vector_b[dim];
    fill_i8_vector(vector_a, dim);
    fill_i8_vector(vector_b, dim);

    const int32_t got = simd_i8_dot(vector_a, vector_b, dim);
    const int32_t want = ref_i8_dot(vector_a, vector_b, dim);
    if (got != want) {
      fprintf(stderr, "I8 dot failed at dim=%u: got %d, want %d\n", dim, got,
              want);
      return -1;
    }
  }

  return 0;
}

int main(void) {
  if (test_f32_l2_sq() != 0)
    return -1;
  if (test_f32_dot() != 0)
    return -1;
  if (test_i8_l2_sq() != 0)
    return -1;
  if (test_i8_dot() != 0)
    return -1;
  return 0;
}
