#include <arm_neon.h>
#include <stdint.h>

#include "livero_types.h"
#include "simd.h"

int64_t ternary_dot_simd(const LVTernaryChunk *ternary_a, const LVTernaryChunk *ternary_b,
                         const LVCount_t ternary_chunk_count) {
    /*
        ternary_a: [mask_a0][sign_a0][mask_a1][sign_a1]...
        ternary_b: [mask_b0][sign_b0][mask_b1][sign_b1]...
    */

    uint16x8_t acc_mask = vdupq_n_u16(0);
    uint16x8_t acc_sign = vdupq_n_u16(0);
    for (LVCount_t i = 0; i + 1 < ternary_chunk_count; i += 2) {
        uint64x2x2_t ta = vld2q_u64((const uint64_t *)&ternary_a[i]);  //[mask_a0][mask_a1][sign_a0][sign_a1]
        uint64x2x2_t tb = vld2q_u64((const uint64_t *)&ternary_b[i]);  //[mask_b0][mask_b1][sign_b0][sign_b1]

        uint64x2_t ta_mask = ta.val[0];
        uint64x2_t ta_sign = ta.val[1];
        uint64x2_t tb_mask = tb.val[0];
        uint64x2_t tb_sign = tb.val[1];

        uint64x2_t mask = vandq_u64(ta_mask, tb_mask);
        uint64x2_t sign = vandq_u64(veorq_u64(ta_sign, tb_sign), mask);

        uint8x16_t mask_cnt = vcntq_u8(vreinterpretq_u8_u64(mask));
        uint8x16_t sign_cnt = vcntq_u8(vreinterpretq_u8_u64(sign));

        acc_mask = vpadalq_u8(acc_mask, mask_cnt);
        acc_sign = vpadalq_u8(acc_sign, sign_cnt);
    }

    int64_t sum = 0;
    const int64_t acc_mask_scalar = (int64_t)vaddlvq_u16(acc_mask);
    const int64_t acc_sign_scalar = (int64_t)vaddlvq_u16(acc_sign);
    sum += acc_mask_scalar - 2 * acc_sign_scalar;

    if (ternary_chunk_count % 2 != 0) {
        const int tail = ternary_chunk_count - 1;
        sum += ternary_dot_scalar(&ternary_a[tail], &ternary_b[tail], 1);
    }

    return sum;
}
