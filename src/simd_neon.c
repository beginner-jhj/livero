#include "livero_types.h"
#include <stdint.h>
#if defined(__aarch64__) || defined(_M_ARM64)

#include "simd.h"
#include <arm_neon.h>

float simd_f32_l2_sq(const float *a, const float *b, const LVDim32_t dim) {
  simd_align_guard(a, b, dim);

  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  float32x4_t acc2 = vdupq_n_f32(0.0f);
  float32x4_t acc3 = vdupq_n_f32(0.0f);

  for (LVDim32_t i = 0; i < dim; i += 16) {
    float32x4_t diff0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
    float32x4_t diff1 = vsubq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    float32x4_t diff2 = vsubq_f32(vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    float32x4_t diff3 = vsubq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));

    acc0 =
        vfmaq_f32(acc0, diff0,
                  diff0); // fma: Fused Multiply-Add, 'multiply-add' one shot.
    acc1 = vfmaq_f32(acc1, diff1, diff1);
    acc2 = vfmaq_f32(acc2, diff2, diff2);
    acc3 = vfmaq_f32(acc3, diff3, diff3);
  }

  acc0 = vaddq_f32(acc0, acc1);
  acc2 = vaddq_f32(acc2, acc3);
  acc0 = vaddq_f32(acc0, acc2);

  return vaddvq_f32(acc0);
}

float simd_f32_dot(const float *a, const float *b, const LVDim32_t dim) {
  simd_align_guard(a, b, dim);

  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  float32x4_t acc2 = vdupq_n_f32(0.0f);
  float32x4_t acc3 = vdupq_n_f32(0.0f);

  for (LVDim32_t i = 0; i < dim; i += 16) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
    acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    acc2 = vfmaq_f32(acc2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
  }

  acc0 = vaddq_f32(acc0, acc1);
  acc2 = vaddq_f32(acc2, acc3);
  acc0 = vaddq_f32(acc0, acc2);

  return -vaddvq_f32(acc0);
}

int32_t simd_i8_l2_sq(const int8_t *a, const int8_t *b, const LVDim32_t dim) {
  simd_align_guard(a, b, dim);

  int32x4_t acc0 = vdupq_n_s32(0);
  int32x4_t acc1 = vdupq_n_s32(0);
  int32x4_t acc2 = vdupq_n_s32(0);
  int32x4_t acc3 = vdupq_n_s32(0);

  for (LVDim32_t i = 0; i < dim; i += 64) {
    int8x16_t a0 = vld1q_s8(a + i);
    int8x16_t b0 = vld1q_s8(b + i);

    int8x16_t a1 = vld1q_s8(a + i + 16);
    int8x16_t b1 = vld1q_s8(b + i + 16);

    int8x16_t a2 = vld1q_s8(a + i + 32);
    int8x16_t b2 = vld1q_s8(b + i + 32);

    int8x16_t a3 = vld1q_s8(a + i + 48);
    int8x16_t b3 = vld1q_s8(b + i + 48);

    int16x8_t a0_low = vmovl_s8(vget_low_s8(a0));
    int16x8_t a0_high = vmovl_s8(vget_high_s8(a0));
    int16x8_t b0_low = vmovl_s8(vget_low_s8(b0));
    int16x8_t b0_high = vmovl_s8(vget_high_s8(b0));

    int16x8_t a1_low = vmovl_s8(vget_low_s8(a1));
    int16x8_t a1_high = vmovl_s8(vget_high_s8(a1));
    int16x8_t b1_low = vmovl_s8(vget_low_s8(b1));
    int16x8_t b1_high = vmovl_s8(vget_high_s8(b1));

    int16x8_t a2_low = vmovl_s8(vget_low_s8(a2));
    int16x8_t a2_high = vmovl_s8(vget_high_s8(a2));
    int16x8_t b2_low = vmovl_s8(vget_low_s8(b2));
    int16x8_t b2_high = vmovl_s8(vget_high_s8(b2));

    int16x8_t a3_low = vmovl_s8(vget_low_s8(a3));
    int16x8_t a3_high = vmovl_s8(vget_high_s8(a3));
    int16x8_t b3_low = vmovl_s8(vget_low_s8(b3));
    int16x8_t b3_high = vmovl_s8(vget_high_s8(b3));

    int16x8_t diff0_low = vsubq_s16(a0_low, b0_low);
    int16x8_t diff0_high = vsubq_s16(a0_high, b0_high);

    int16x8_t diff1_low = vsubq_s16(a1_low, b1_low);
    int16x8_t diff1_high = vsubq_s16(a1_high, b1_high);

    int16x8_t diff2_low = vsubq_s16(a2_low, b2_low);
    int16x8_t diff2_high = vsubq_s16(a2_high, b2_high);

    int16x8_t diff3_low = vsubq_s16(a3_low, b3_low);
    int16x8_t diff3_high = vsubq_s16(a3_high, b3_high);

    acc0 = vmlal_s16(acc0, vget_low_s16(diff0_low), vget_low_s16(diff0_low));
    acc0 = vmlal_s16(acc0, vget_high_s16(diff0_low), vget_high_s16(diff0_low));
    acc0 = vmlal_s16(acc0, vget_low_s16(diff0_high), vget_low_s16(diff0_high));
    acc0 =
        vmlal_s16(acc0, vget_high_s16(diff0_high), vget_high_s16(diff0_high));

    acc1 = vmlal_s16(acc1, vget_low_s16(diff1_low), vget_low_s16(diff1_low));
    acc1 = vmlal_s16(acc1, vget_high_s16(diff1_low), vget_high_s16(diff1_low));
    acc1 = vmlal_s16(acc1, vget_low_s16(diff1_high), vget_low_s16(diff1_high));
    acc1 =
        vmlal_s16(acc1, vget_high_s16(diff1_high), vget_high_s16(diff1_high));

    acc2 = vmlal_s16(acc2, vget_low_s16(diff2_low), vget_low_s16(diff2_low));
    acc2 = vmlal_s16(acc2, vget_high_s16(diff2_low), vget_high_s16(diff2_low));
    acc2 = vmlal_s16(acc2, vget_low_s16(diff2_high), vget_low_s16(diff2_high));
    acc2 =
        vmlal_s16(acc2, vget_high_s16(diff2_high), vget_high_s16(diff2_high));

    acc3 = vmlal_s16(acc3, vget_low_s16(diff3_low), vget_low_s16(diff3_low));
    acc3 = vmlal_s16(acc3, vget_high_s16(diff3_low), vget_high_s16(diff3_low));
    acc3 = vmlal_s16(acc3, vget_low_s16(diff3_high), vget_low_s16(diff3_high));
    acc3 =
        vmlal_s16(acc3, vget_high_s16(diff3_high), vget_high_s16(diff3_high));
  }

  acc0 = vaddq_s32(acc0, acc1);
  acc2 = vaddq_s32(acc2, acc3);
  acc0 = vaddq_s32(acc0, acc2);

  return (int32_t)vaddvq_s32(acc0);
}

int32_t simd_i8_dot(const int8_t *a, const int8_t *b, const LVDim32_t dim) {
  simd_align_guard(a, b, dim);

  int32x4_t acc0 = vdupq_n_s32(0);
  int32x4_t acc1 = vdupq_n_s32(0);
  int32x4_t acc2 = vdupq_n_s32(0);
  int32x4_t acc3 = vdupq_n_s32(0);

  for (LVDim32_t i = 0; i < dim; i += 64) {
    int8x16_t a0 = vld1q_s8(a + i);
    int8x16_t b0 = vld1q_s8(b + i);

    int8x16_t a1 = vld1q_s8(a + i + 16);
    int8x16_t b1 = vld1q_s8(b + i + 16);

    int8x16_t a2 = vld1q_s8(a + i + 32);
    int8x16_t b2 = vld1q_s8(b + i + 32);

    int8x16_t a3 = vld1q_s8(a + i + 48);
    int8x16_t b3 = vld1q_s8(b + i + 48);

    int16x8_t a0_low = vmovl_s8(vget_low_s8(a0));
    int16x8_t a0_high = vmovl_s8(vget_high_s8(a0));
    int16x8_t b0_low = vmovl_s8(vget_low_s8(b0));
    int16x8_t b0_high = vmovl_s8(vget_high_s8(b0));

    int16x8_t a1_low = vmovl_s8(vget_low_s8(a1));
    int16x8_t a1_high = vmovl_s8(vget_high_s8(a1));
    int16x8_t b1_low = vmovl_s8(vget_low_s8(b1));
    int16x8_t b1_high = vmovl_s8(vget_high_s8(b1));

    int16x8_t a2_low = vmovl_s8(vget_low_s8(a2));
    int16x8_t a2_high = vmovl_s8(vget_high_s8(a2));
    int16x8_t b2_low = vmovl_s8(vget_low_s8(b2));
    int16x8_t b2_high = vmovl_s8(vget_high_s8(b2));

    int16x8_t a3_low = vmovl_s8(vget_low_s8(a3));
    int16x8_t a3_high = vmovl_s8(vget_high_s8(a3));
    int16x8_t b3_low = vmovl_s8(vget_low_s8(b3));
    int16x8_t b3_high = vmovl_s8(vget_high_s8(b3));

    acc0 = vmlal_s16(acc0, vget_low_s16(a0_low), vget_low_s16(b0_low));
    acc0 = vmlal_s16(acc0, vget_high_s16(a0_low), vget_high_s16(b0_low));
    acc0 = vmlal_s16(acc0, vget_low_s16(a0_high), vget_low_s16(b0_high));
    acc0 = vmlal_s16(acc0, vget_high_s16(a0_high), vget_high_s16(b0_high));

    acc1 = vmlal_s16(acc1, vget_low_s16(a1_low), vget_low_s16(b1_low));
    acc1 = vmlal_s16(acc1, vget_high_s16(a1_low), vget_high_s16(b1_low));
    acc1 = vmlal_s16(acc1, vget_low_s16(a1_high), vget_low_s16(b1_high));
    acc1 = vmlal_s16(acc1, vget_high_s16(a1_high), vget_high_s16(b1_high));

    acc2 = vmlal_s16(acc2, vget_low_s16(a2_low), vget_low_s16(b2_low));
    acc2 = vmlal_s16(acc2, vget_high_s16(a2_low), vget_high_s16(b2_low));
    acc2 = vmlal_s16(acc2, vget_low_s16(a2_high), vget_low_s16(b2_high));
    acc2 = vmlal_s16(acc2, vget_high_s16(a2_high), vget_high_s16(b2_high));

    acc3 = vmlal_s16(acc3, vget_low_s16(a3_low), vget_low_s16(b3_low));
    acc3 = vmlal_s16(acc3, vget_high_s16(a3_low), vget_high_s16(b3_low));
    acc3 = vmlal_s16(acc3, vget_low_s16(a3_high), vget_low_s16(b3_high));
    acc3 = vmlal_s16(acc3, vget_high_s16(a3_high), vget_high_s16(b3_high));
  }

  acc0 = vaddq_s32(acc0, acc1);
  acc2 = vaddq_s32(acc2, acc3);
  acc0 = vaddq_s32(acc0, acc2);

  return -(int32_t)vaddvq_s32(acc0);
}

#else
typedef int simd_neon_unused_;
#endif
