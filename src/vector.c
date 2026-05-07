#include "vector.h"
#include "helper.h"
#include "arena.h"
#include <math.h>
#include <arm_neon.h>

/**
 * ARM NEON SIMD Naming Convention Reference
 * -----------------------------------------
 * Structure: v[instruction][shaping][type][size]q_[datatype][bits]
 * 1. [instruction]: The core operation (e.g., add, sub, mul, mla, fma, ldr).
 * 2. [shaping]: How the data width changes during the operation.
 * - (None): Normal. Input and output widths are identical.
 * - l (Long): Output width is twice the input width (e.g., 8-bit * 8-bit -> 16-bit).
 * - w (Wide): One input is twice the width of the other (e.g., 16-bit + 8-bit -> 16-bit).
 * - n (Narrow): Output width is half the input width (e.g., 16-bit -> 8-bit).
 * 3. [type]: Specific behavior of the operation.
 * - p (Pairwise): Operates on adjacent elements in the same register.
 * - v (Across): Operates across all elements in a single register (e.g., vaddvq).
 * - h (Halving): Right-shifts the result by 1 (effective average).
 * - r (Rounding): Adds 0.5 before truncating for better precision.
 * - q (Saturating): Clamps the result to max/min range instead of overflowing.
 * 4. [size]: The register bit-width.
 * - (None): 64-bit "Double-word" register (D0-D31).
 * - q (Quadword): 128-bit "Quad-word" register (V0-V31). Handles 4x f32 or 16x i8.
 * 5. [datatype][bits]: The element format.
 * - s8, s16, s32, s64: Signed integers.
 * - u8, u16, u32, u64: Unsigned integers.
 * - f16, f32: Floating-point (Half and Single precision).
 * Example: vmlal_u16
 * [v]ector [m]ultiply [l]ong [a]ccumulate [l]ong on [u]nsigned [16]-bit.
 * (Multiplies two 16-bit values and adds the result to a 32-bit accumulator).
 */

LVHnsw *create_hnsw(const LVVectorType vector_type, const LVDim32_t dim)
{
    int flag = 1;
    LVHnsw *hnsw = NULL;
    Arena *node_arena = NULL;
    Arena *vector_arena = NULL;

    LVHnsw *hnsw_tmp = malloc(sizeof(LVHnsw));
    if (!hnsw_tmp)
    {
        goto cleanup;
    }

    hnsw = hnsw_tmp;

    hnsw->current_max_level = 0;
    hnsw->dim = dim;
    hnsw->entry_node = NULL;
    hnsw->m_l = 1 / log(HNSW_M);
    hnsw->node_count = 0;
    hnsw->vector_type = vector_type;

    node_arena = create_arena();
    if (!node_arena)
    {
        flag = 1;
        goto cleanup;
    }

    hnsw->node_arena = node_arena;

    vector_arena = create_arena();
    if (!vector_arena)
    {
        flag = 1;
        goto cleanup;
    }

    hnsw->vector_arena = vector_arena;

cleanup:
    if (flag)
    {
        safe_free(&node_arena);
        safe_free(&vector_arena);
        safe_free(&hnsw);
    }
    return hnsw;
}

uint32_t vector_i8_l2_sq(const int8_t *a, const int8_t *b, const LVDim32_t dim)
{
    uint32x4_t sum_v1 = vdupq_n_u32(0); // vector to store sum
    uint32x4_t sum_v2 = vdupq_n_u32(0);
    for (int i = 0; i < dim; i += 32)
    {
        int8x16_t diff1 = vabdq_s8(vld1q_s8(a + i), vld1q_s8(b + i));
        int8x16_t diff2 = vabdq_s8(vld1q_s8(a + i + 16), vld1q_s8(b + i + 16));

        uint16x8_t diff_low1 = vmovl_u8(vget_low_u8(diff1)); // square will be bigger than 8 bit, so we have to move bits to the bigger container
        /*
            The suffix 'l' in vmla'l'_u16 means that it will make size of data doubled, so
            diff_low will be 32*8=256 bits.
            But NEON register can only store 128bits,
            so we have to split diff_low into two parts (low, high) to store data in a 32*4 container

        */
        sum_v1 = vmlal_u16(sum_v1, vget_low_u16(diff_low1), vget_low_u16(diff_low1));
        sum_v1 = vmlal_u16(sum_v1, vget_high_u16(diff_low1), vget_high_u16(diff_low1));

        uint16x8_t diff_low2 = vmovl_u8(vget_low_u8(diff2));
        sum_v2 = vmlal_u16(sum_v2, vget_low_u16(diff_low2), vget_low_u16(diff_low2));
        sum_v2 = vmlal_u16(sum_v2, vget_high_u16(diff_low2), vget_high_u16(diff_low2));

        uint16x8_t diff_high1 = vmovl_u8(vget_high_u8(diff1));
        sum_v1 = vmlal_u16(sum_v1, vget_low_u16(diff_high1), vget_low_u16(diff_high1));
        sum_v1 = vmlal_u16(sum_v1, vget_high_u16(diff_high1), vget_high_u16(diff_high1));

        uint16x8_t diff_high2 = vmovl_u8(vget_high_u8(diff2));
        sum_v2 = vmlal_u16(sum_v2, vget_low_u16(diff_high2), vget_low_u16(diff_high2));
        sum_v2 = vmlal_u16(sum_v2, vget_high_u16(diff_high2), vget_high_u16(diff_high2));
    }

    return vaddvq_u32(vaddq_u32(sum_v1, sum_v2));
}

float vector_f32_l2_sq(const float *a, const float *b, const LVDim32_t dim)
{
    float32x4_t sum_v1 = vdupq_n_f32(0.0f);
    float32x4_t sum_v2 = vdupq_n_f32(0.0f);

    for (int i = 0; i < dim; i += 8)
    {
        float32x4_t diff1 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        float32x4_t diff2 = vsubq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));

        sum_v1 = vfmaq_f32(sum_v1, diff1, diff1); // fma: Fused Multiply-Add, 'multiply-add' one shot.
        sum_v2 = vfmaq_f32(sum_v2, diff2, diff2);
    }

    return vaddvq_f32(vaddq_f32(sum_v1, sum_v2));
}

float vector_i8_cos(const int8_t *a, const int8_t *b, const LVDim32_t dim)
{
    int32x4_t sum_sq_a = vdupq_n_s32(0);
    int32x4_t sum_sq_b = vdupq_n_s32(0);
    int32x4_t sum_dot = vdupq_n_s32(0);

    for (int i = 0; i < dim; i += 16)
    {
        int8x16_t va = vld1q_s8(a + i);
        int8x16_t vb = vld1q_s8(b + i);

        int16x8_t va_l = vmovl_s8(vget_low_s8(va));
        int16x8_t vb_l = vmovl_s8(vget_low_s8(vb));

        sum_sq_a = vmlal_s16(sum_sq_a, vget_low_s16(va_l), vget_low_s16(va_l));
        sum_sq_a = vmlal_s16(sum_sq_a, vget_high_s16(va_l), vget_high_s16(va_l));

        sum_sq_b = vmlal_s16(sum_sq_b, vget_low_s16(vb_l), vget_low_s16(vb_l));
        sum_sq_b = vmlal_s16(sum_sq_b, vget_high_s16(vb_l), vget_high_s16(vb_l));

        sum_dot = vmlal_s16(sum_dot, vget_low_s16(va_l), vget_low_s16(vb_l));
        sum_dot = vmlal_s16(sum_dot, vget_high_s16(va_l), vget_high_s16(vb_l));

        int16x8_t va_h = vmovl_s8(vget_high_s8(va));
        int16x8_t vb_h = vmovl_s8(vget_high_s8(vb));

        sum_sq_a = vmlal_s16(sum_sq_a, vget_low_s16(va_h), vget_low_s16(va_h));
        sum_sq_a = vmlal_s16(sum_sq_a, vget_high_s16(va_h), vget_high_s16(va_h));

        sum_sq_b = vmlal_s16(sum_sq_b, vget_low_s16(vb_h), vget_low_s16(vb_h));
        sum_sq_b = vmlal_s16(sum_sq_b, vget_high_s16(vb_h), vget_high_s16(vb_h));

        sum_dot = vmlal_s16(sum_dot, vget_low_s16(va_h), vget_low_s16(vb_h));
        sum_dot = vmlal_s16(sum_dot, vget_high_s16(va_h), vget_high_s16(vb_h));
    }

    int32_t final_dot = vaddvq_s32(sum_dot);
    int32_t final_sq_a = vaddvq_s32(sum_sq_a);
    int32_t final_sq_b = vaddvq_s32(sum_sq_b);

    if (final_sq_a <= 0 || final_sq_b <= 0)
    {
        return 0.0f;
    }

    float norm_a = sqrtf(final_sq_a);
    float norm_b = sqrtf(final_sq_b);

    float denominator = norm_a * norm_b;
    float result = (float)final_dot / denominator;

    return fmaxf(-1.0f, fminf(1.0f, result));
}

float vector_f32_cos(const float *a, const float *b, const LVDim32_t dim)
{
    float32x4_t sum_sq_a1 = vdupq_n_f32(0.0f);
    float32x4_t sum_sq_a2 = vdupq_n_f32(0.0f);
    float32x4_t sum_sq_b1 = vdupq_n_s32(0.0f);
    float32x4_t sum_sq_b2 = vdupq_n_f32(0.0f);
    float32x4_t sum_dot1 = vdupq_n_f32(0.0f);
    float32x4_t sum_dot2 = vdupq_n_f32(0.0f);

    for (int i = 0; i < dim; i += 8)
    {
        float32x4_t va1 = vld1q_f32(a + i);
        float32x4_t vb1 = vld1q_f32(b + i);
        float32x4_t va2 = vld1q_f32(a + i + 4);
        float32x4_t vb2 = vld1q_f32(b + i + 4);

        sum_sq_a1 = vfmaq_f32(sum_sq_a1, va1, va1);
        sum_sq_a2 = vfmaq_f32(sum_sq_a2, va2, va2);

        sum_sq_b1 = vfmaq_f32(sum_sq_b1, vb1, vb1);
        sum_sq_b2 = vfmaq_f32(sum_sq_b2, vb2, vb2);

        sum_dot1 = vfmaq_f32(sum_dot1, va1, vb1);
        sum_dot2 = vfmaq_f32(sum_dot2, va2, vb2);
    }

    float32_t final_sq_a = vaddvq_f32(vaddq_f32(sum_sq_a1, sum_sq_a2));
    float32_t final_sq_b = vaddvq_f32(vaddq_f32(sum_sq_b1, sum_sq_b2));
    float32_t final_dot = vaddvq_f32(vaddq_f32(sum_dot1, sum_dot2));

    if (final_sq_a <= 0.0f || final_sq_b <= 0.0f)
        return 0.0f;

    float result = final_dot / (sqrtf(final_sq_a) * sqrtf(final_sq_b));

    return fmaxf(-1.0f, fminf(1.0f, result));
}

int32_t vector_i8_dot(const int8_t *a, const int8_t *b, const LVDim32_t dim)
{
    int32x4_t sum_dot1 = vdupq_n_s32(0);
    int32x4_t sum_dot2 = vdupq_n_s32(0);
    for (int i = 0; i < dim; i += 32)
    {
        int8x16_t va_1 = vld1q_s8(a + i);
        int8x16_t va_2 = vld1q_s8(a + i + 16);

        int8x16_t vb_1 = vld1q_s8(b + i);
        int8x16_t vb_2 = vld1q_s8(b + i + 16);

        // get low of each vector
        int16x8_t va_1_low = vmovl_s8(vget_low_s8(va_1));
        int16x8_t va_2_low = vmovl_s8(vget_low_s8(va_2));

        int16x8_t vb_1_low = vmovl_s8(vget_low_s8(vb_1));
        int16x8_t vb_2_low = vmovl_s8(vget_low_s8(vb_2));

        // calc each dot of each (va vb) pair
        sum_dot1 = vmlal_s16(sum_dot1, vget_low_s16(va_1_low), vget_low_s16(vb_1_low));
        sum_dot1 = vmlal_s16(sum_dot1, vget_high_s16(va_1_low), vget_high_s16(vb_1_low));

        sum_dot2 = vmlal_s16(sum_dot2, vget_low_s16(va_2_low), vget_low_s16(vb_2_low));
        sum_dot2 = vmlal_s16(sum_dot2, vget_high_s16(va_2_low), vget_high_s16(vb_2_low));

        // get high of each vector
        int16x8_t va_1_high = vmovl_s8(vget_high_s8(va_1));
        int16x8_t va_2_high = vmovl_s8(vget_high_s8(va_2));

        int16x8_t vb_1_high = vmovl_s8(vget_high_s8(vb_1));
        int16x8_t vb_2_high = vmovl_s8(vget_high_s8(vb_2));

        // calc each dot of each (va vb) pair
        sum_dot1 = vmlal_s16(sum_dot1, vget_low_s16(va_1_high), vget_low_s16(vb_1_high));
        sum_dot1 = vmlal_s16(sum_dot1, vget_high_s16(va_1_high), vget_high_s16(vb_1_high));

        sum_dot2 = vmlal_s16(sum_dot2, vget_low_s16(va_2_high), vget_low_s16(vb_2_high));
        sum_dot2 = vmlal_s16(sum_dot2, vget_high_s16(va_2_high), vget_high_s16(vb_2_high));
    }

    return vaddvq_s32(vaddq_s32(sum_dot1,sum_dot2));
}


float vector_f32_dot(const float *a, const float *b, const LVDim32_t dim)
{
    float32x4_t sum_dot1 = vdupq_n_f32(0.0f);
    float32x4_t sum_dot2 = vdupq_n_f32(0.0f);

    for (int i = 0; i < dim; i += 8)
    {
        float32x4_t va1 = vld1q_f32(a + i);
        float32x4_t vb1 = vld1q_f32(b + i);
        float32x4_t va2 = vld1q_f32(a + i + 4);
        float32x4_t vb2 = vld1q_f32(b + i + 4);

  
        sum_dot1 = vfmaq_f32(sum_dot1, va1, vb1);
        sum_dot2 = vfmaq_f32(sum_dot2, va2, vb2);
    }

    return vaddvq_f32(vaddq_f32(sum_dot1, sum_dot2));
}
