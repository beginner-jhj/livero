#include <stdint.h>
#if defined(__x86_64__) || defined(_M_X64)

#include "livero_types.h"
#include "simd.h"
#include <assert.h>
#include <emmintrin.h>
#include <xmmintrin.h>

// Reduce a 4-lane vector to one float. SSE2 has no horizontal add.
static inline float hsum_ps(__m128 v) {
  __m128 hi = _mm_movehl_ps(v, v); // upper 2 -> lower 2
  v = _mm_add_ps(v, hi);
  __m128 sh = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
  v = _mm_add_ss(v, sh); // lane 0 only
  return _mm_cvtss_f32(v);
}

// Horizontal sum of 4 int32 lanes. Same fold-in-half idea as hsum_ps, but
// with integer shuffles. SSE2 has no horizontal-add for integers either.
static inline int32_t hsum_epi32(__m128i v) {
  // _mm_shuffle_epi32 reorders 32-bit lanes. (2,3,2,3) brings the upper two
  // lanes down to the lower two positions.
  __m128i hi = _mm_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 2, 3));
  v = _mm_add_epi32(v, hi); // lane0 += lane2, lane1 += lane3
  // Now bring lane 1 down to lane 0 and add once more.
  hi = _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 1, 1, 1));
  v = _mm_add_epi32(v, hi);
  return _mm_cvtsi128_si32(v); // extract lane 0
}

float simd_f32_l2_sq(const float *a, const float *b, const LVDim32_t dim) {
  simd_align_guard(a, b, dim);
  __m128 acc0 = _mm_setzero_ps();
  __m128 acc1 = _mm_setzero_ps();
  __m128 acc2 = _mm_setzero_ps();
  __m128 acc3 = _mm_setzero_ps();

  for (LVDim32_t i = 0; i < dim; i += 16) {
    __m128 d0 = _mm_sub_ps(_mm_load_ps(a + i), _mm_load_ps(b + i));
    __m128 d1 = _mm_sub_ps(_mm_load_ps(a + i + 4), _mm_load_ps(b + i + 4));
    __m128 d2 = _mm_sub_ps(_mm_load_ps(a + i + 8), _mm_load_ps(b + i + 8));
    __m128 d3 = _mm_sub_ps(_mm_load_ps(a + i + 12), _mm_load_ps(b + i + 12));

    acc0 = _mm_add_ps(acc0, _mm_mul_ps(d0, d0));
    acc1 = _mm_add_ps(acc1, _mm_mul_ps(d1, d1));
    acc2 = _mm_add_ps(acc2, _mm_mul_ps(d2, d2));
    acc3 = _mm_add_ps(acc3, _mm_mul_ps(d3, d3));
  }

  acc0 = _mm_add_ps(acc0, acc1);
  acc2 = _mm_add_ps(acc2, acc3);
  acc0 = _mm_add_ps(acc0, acc2);

  return hsum_ps(acc0);
}

float simd_f32_dot(const float *a, const float *b, const LVDim32_t dim) {
  simd_align_guard(a, b, dim);
  __m128 acc0 = _mm_setzero_ps();
  __m128 acc1 = _mm_setzero_ps();
  __m128 acc2 = _mm_setzero_ps();
  __m128 acc3 = _mm_setzero_ps();

  for (LVDim32_t i = 0; i < dim; i += 16) {
    acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_load_ps(a + i), _mm_load_ps(b + i)));
    acc1 = _mm_add_ps(
        acc1, _mm_mul_ps(_mm_load_ps(a + i + 4), _mm_load_ps(b + i + 4)));
    acc2 = _mm_add_ps(
        acc2, _mm_mul_ps(_mm_load_ps(a + i + 8), _mm_load_ps(b + i + 8)));
    acc3 = _mm_add_ps(
        acc3, _mm_mul_ps(_mm_load_ps(a + i + 12), _mm_load_ps(b + i + 12)));
  }

  acc0 = _mm_add_ps(acc0, acc1);
  acc2 = _mm_add_ps(acc2, acc3);
  acc0 = _mm_add_ps(acc0, acc2);

  return -hsum_ps(acc0); // negated so smaller = closer, consistent with L2
}

int32_t simd_i8_l2_sq(const int8_t *a, const int8_t *b, const LVDim32_t dim) {
  simd_align_guard(a, b, dim);

  __m128i acc = _mm_setzero_si128();
  __m128i zero = _mm_setzero_si128();

  for (LVDim32_t i = 0; i < dim; i += 16) {
    __m128i a8 = _mm_load_si128((const __m128i *)(a + i)); //
    __m128i b8 = _mm_load_si128((const __m128i *)(b + i));

    // Widen both to int16 (identical to the dot kernel).
    __m128i a_sign = _mm_cmpgt_epi8(zero, a8);
    __m128i b_sign = _mm_cmpgt_epi8(zero, b8);
    __m128i a_lo = _mm_unpacklo_epi8(a8, a_sign);
    __m128i a_hi = _mm_unpackhi_epi8(a8, a_sign);
    __m128i b_lo = _mm_unpacklo_epi8(b8, b_sign);
    __m128i b_hi = _mm_unpackhi_epi8(b8, b_sign);

    // Difference in int16. Worst case 127-(-128)=255, which fits in int16.
    __m128i d_lo = _mm_sub_epi16(a_lo, b_lo);
    __m128i d_hi = _mm_sub_epi16(a_hi, b_hi);

    // Square + widen in one step: madd(d,d) computes d*d (up to 255^2=65025,
    // which needs int32) and sums adjacent pairs into 4 int32 lanes. This is
    // why we reach for madd here rather than a plain int16 multiply -- the
    // square would overflow int16.
    __m128i sq_lo =
        _mm_madd_epi16(d_lo, d_lo); // [int32_0, int32_1, int32_2, int32_3]
    __m128i sq_hi =
        _mm_madd_epi16(d_hi, d_hi); // [int32_0, int32_1, int32_2, int32_3]

    acc = _mm_add_epi32(acc, sq_lo);
    acc = _mm_add_epi32(acc, sq_hi);
  }

  return hsum_epi32(acc); 
}

int32_t simd_i8_dot(const int8_t *a, const int8_t *b, const LVDim32_t dim) {
  simd_align_guard(a, b, dim);

  __m128i acc = _mm_setzero_si128(); // 4x int32 accumulator
  __m128i zero = _mm_setzero_si128();

  for (LVDim32_t i = 0; i < dim; i += 16) {
    // Load 16 int8 from each vector.
    __m128i a8 = _mm_load_si128((const __m128i *)(a + i));
    __m128i b8 = _mm_load_si128((const __m128i *)(b + i));

    // Sign-extend int8 -> int16. SSE2 has no cvtepi8_epi16, so we build the
    // sign byte by hand (0xFF for negatives, 0x00 for non-negatives) and
    // interleave it above each value byte via unpack. See the l2 kernel
    // comment for the bit-level walkthrough.
    __m128i a_sign = _mm_cmpgt_epi8(zero, a8);
    __m128i b_sign = _mm_cmpgt_epi8(zero, b8);
    __m128i a_lo = _mm_unpacklo_epi8(a8, a_sign); // lower 8 -> int16 x8
    __m128i a_hi = _mm_unpackhi_epi8(a8, a_sign); // upper 8 -> int16 x8
    __m128i b_lo = _mm_unpacklo_epi8(b8, b_sign);
    __m128i b_hi = _mm_unpackhi_epi8(b8, b_sign);

    // madd multiplies 8 int16 pairs and adds adjacent pairs into 4 int32.
    // int8*int8 maxes at 127*127=16129, safely inside int16 before madd,
    // and madd's output is int32 so the pairwise sum can't overflow either.
    __m128i p_lo =
        _mm_madd_epi16(a_lo, b_lo); // [int32_0, int32_1, int32_2, int32_3]
    __m128i p_hi =
        _mm_madd_epi16(a_hi, b_hi); // [int32_0, int32_1, int32_2, int32_3]

    acc = _mm_add_epi32(acc, p_lo);
    acc = _mm_add_epi32(acc, p_hi);
  }

  return -hsum_epi32(acc); // negated so smaller = closer, consistent with L2
}

#else
typedef int simd_sse2_unused_;
#endif
