#include "ternary.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <memory.h>
#include <stdint.h>
#include <stdlib.h>

#include "livero_types.h"
#include "simd.h"  // simd.h itself does not contain any simd intrinsics, it only provids simd interface

static void ternary_make(LVTernaryChunk* ternary_chunks, const int ternary_value, const LVDim_t vector_index) {
    const LVDim_t chunk_index = vector_index / LV_TERNARY_CHUNK_CAPACITY;
    const LVDim_t bit = vector_index % LV_TERNARY_CHUNK_CAPACITY;
    const uint64_t mask_bit = 1ull << (LV_TERNARY_CHUNK_CAPACITY - 1 - bit);
    if (ternary_value != 0) {
        ternary_chunks[chunk_index].mask |= mask_bit;
    }
    if (ternary_value < 0) {
        ternary_chunks[chunk_index].sign |= mask_bit;
    }
}

static void ternary_convert_fp32(const float* vector, const LVDim_t vector_dimension, LVTernaryChunk* ternary_chunks) {
    assert(vector && vector_dimension > 0);

    double sum = 0.0;
    for (LVDim_t vector_index = 0; vector_index < vector_dimension; ++vector_index) {
        sum += fabs((double)vector[vector_index]);
    }
    // gamma >= 0 now, so adding epsilon only guards the all-zero vector.
    const float gamma = (float)(sum / (double)vector_dimension) + FLT_EPSILON;

    for (LVDim_t vector_index = 0; vector_index < vector_dimension; ++vector_index) {
        float r = roundf(vector[vector_index] / gamma);

        if (r > 1.0f) {
            r = 1.0f;
        }
        if (r < -1.0f) {
            r = -1.0f;
        }
        const int ternary_value = (int)r;  // exactly -1, 0, or 1 at this point
        ternary_make(ternary_chunks, ternary_value, vector_index);
    }
}
static void ternary_convert_int8(const int8_t* vector, const LVDim_t vector_dimension, LVTernaryChunk* ternary_chunks) {
    assert(vector && vector_dimension > 0);
    uint64_t sum = 0;
    for (LVDim_t vector_index = 0; vector_index < vector_dimension; ++vector_index) {
        sum += abs(vector[vector_index]);
    }
    const float gamma = (float)(sum / (double)vector_dimension) + FLT_EPSILON;
    for (LVDim_t vector_index = 0; vector_index < vector_dimension; ++vector_index) {
        float r = roundf(vector[vector_index] / gamma);
        if (r > 1.0f) {
            r = 1.0f;
        }
        if (r < -1.0f) {
            r = -1.0f;
        }
        const int ternary_value = (int)r;
        ternary_make(ternary_chunks, ternary_value, vector_index);
    }
}

void ternary_convert(const void* vector, const LVVectorType vector_type, const LVDim_t vector_dimension,
                     LVTernaryChunk* ternary_chunks) {
    assert(vector && ternary_chunks);
    memset(ternary_chunks, 0, ternary_chunks_bytes(vector_dimension));
    switch (vector_type) {
        case LV_VEC_TYPE_FP32:
            ternary_convert_fp32((const float*)vector, vector_dimension, ternary_chunks);
            break;

        case LV_VEC_TYPE_INT8:
            ternary_convert_int8((const int8_t*)vector, vector_dimension, ternary_chunks);
            break;
    }
}

float ternary_dot_scalar(const LVTernaryChunk* ternary_a, const LVTernaryChunk* ternary_b,
                           const LVCount_t ternary_chunk_count) {
    /*
        Ternary Dot Product's result is (a number of +1s) - (a number of -1s).

        Let's say M is a number of nozeros and S is a number of sign differs.

        A number of +1s is M - S.

        So the formula is (M - S) - S = M - 2S.
    */
    int32_t both_nonzero = 0;
    int32_t sign_differs = 0;

    for (LVCount_t chunk_index = 0; chunk_index < ternary_chunk_count; ++chunk_index) {
        /*
            Find both non-zero positions which actually join in the calculation.
        */
        int64_t m = ternary_a[chunk_index].mask & ternary_b[chunk_index].mask;

        /*
            Find diffrent sign positions where a_i * b_i is -1, and remove signs of zeros by & operation with m.
        */
        int64_t s = (ternary_a[chunk_index].sign ^ ternary_b[chunk_index].sign) & m;

        both_nonzero += __builtin_popcountll(m);
        sign_differs += __builtin_popcountll(s);
    }

    return (float)(both_nonzero - 2 * sign_differs);
}

float ternary_dot(const LVTernaryChunk* ternary_a, const LVTernaryChunk* ternary_b,
                  const LVCount_t ternary_chunk_count) {
    assert(ternary_a && ternary_b);
#if defined(LV_USE_SIMD) && defined(__aarch64__)
    return (float)ternary_dot_simd(ternary_a, ternary_b, ternary_chunk_count);
#else
    return (float)ternary_dot_scalar(ternary_a, ternary_b, ternary_chunk_count);
#endif
}

// Asymmetric score: fp32 query x ternary doc, cosine up to the constant ||q||.
// float accumulator: ESP32-S3's FPU is single precision only; double is emulated.
static float ternary_fp32_asymmetric_score(const float* q, const LVTernaryChunk* t, const LVCount_t chunk_count) {
    assert(q && t);
    float acc = 0.0f;
    uint32_t nnz = 0;

    for (LVCount_t w = 0; w < chunk_count; ++w) {
        uint64_t m = t[w].mask;
        uint64_t s = t[w].sign;
        nnz += (uint32_t)__builtin_popcountll(m);
        const float* qw = &q[LV_TERNARY_CHUNK_CAPACITY * w];

        for (uint32_t b = 0; b < LV_TERNARY_CHUNK_CAPACITY; ++b) {
            uint32_t bits;
            memcpy(&bits, &qw[b], sizeof bits);
            uint32_t active = -(uint32_t)(m >> 63);     // all ones if element is nonzero
            uint32_t flip = (uint32_t)(s >> 63) << 31;  // IEEE-754: bit 31 is the sign
            bits = (bits & active) ^ flip;             // 0, +q, or -q (sign-magnitude!)
            float f;
            memcpy(&f, &bits, sizeof f);
            acc += f;
            m <<= 1;
            s <<= 1;
        }
    }
    // cos(q, t) * ||q||: dividing by ||t|| = sqrt(nnz) removes the bias toward
    return nnz ? acc / sqrtf((float)nnz) : 0.0f;
}

static float ternary_int8_asymmetric_score(const int8_t* q, const LVTernaryChunk* t, const LVCount_t chunk_count) {
    assert(q && t);
    int32_t acc = 0;
    uint32_t nnz = 0;

    for (LVCount_t w = 0; w < chunk_count; ++w) {
        uint64_t m = t[w].mask;
        uint64_t s = t[w].sign;
        nnz += (uint32_t)__builtin_popcountll(m);
        const int8_t* qw = &q[LV_TERNARY_CHUNK_CAPACITY * w];

        for (uint32_t b = 0; b < LV_TERNARY_CHUNK_CAPACITY; ++b) {
            int32_t active = -(int32_t)(m >> 63);
            int32_t neg = -(int32_t)(s >> 63);
            int32_t x = (int32_t)qw[b] & active;
            acc += (x ^ neg) - neg;
            m <<= 1;
            s <<= 1;
        }
    }
    return nnz ? (float)acc / sqrtf((float)nnz) : 0.0f;
}

float ternary_asymmetric_score(const void* vector, const LVTernaryChunk* ternary, const LVVectorType vector_type,
                               const LVCount_t ternary_chunk_count) {
    assert(vector && ternary);

    switch (vector_type) {
        case LV_VEC_TYPE_INT8:
            return ternary_int8_asymmetric_score(vector, ternary, ternary_chunk_count);

        case LV_VEC_TYPE_FP32:
            return ternary_fp32_asymmetric_score(vector, ternary, ternary_chunk_count);

        default:
            return 0.0;
    }
}
