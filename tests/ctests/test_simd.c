#include "livero_types.h"
#include "simd.h"
#include "test_helper.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>

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
  return acc;
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
  return acc;
}

int test_f32_l2_sq(void) {
  for (LVCount32_t i = 0; i < DIMS_COUNT; ++i) {
    const LVDim32_t dim = DIMS[i];
    float vector_a[dim];
    float vector_b[dim];
    fill_f32_vector(vector_a, dim);
    fill_f32_vector(vector_b, dim);

    const double diff = fabs(simd_f32_l2_sq(vector_a, vector_b, dim) -
                             ref_f32_l2_sq(vector_a, vector_b, dim));
    if (diff > F32_EPS)
      return -1;
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

    const double diff = fabs(simd_f32_dot(vector_a, vector_b, dim) -
                             ref_f32_dot(vector_a, vector_b, dim));
    if (diff > F32_EPS)
      return -1;
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

    if (simd_i8_l2_sq(vector_a, vector_b, dim) !=
        ref_i8_l2_sq(vector_a, vector_b, dim))
      return -1;
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

    if (simd_i8_dot(vector_a, vector_b, dim) !=
        ref_i8_dot(vector_a, vector_b, dim))
      return -1;
  }

  return 0;
}

int main(void) {
  assert(test_f32_l2_sq() == 0);
  assert(test_f32_dot() == 0);
  assert(test_i8_l2_sq() == 0);
  assert(test_i8_dot() == 0);
  return 0;
}
