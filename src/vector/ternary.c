#include "ternary.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "internal/internal.h"
#include "livero_types.h"

LVSize_t ternary_chunks_count(const LVDim_t dim) {
    assert(dim % LV_DIM_MULTIPLE == 0);
    assert(dim >= LV_TERNARY_CHUNK_CAPACITY && dim < LV_MAX_VECTOR_DIMENSION);
    return dim / LV_TERNARY_CHUNK_CAPACITY;
}

static void ternary_convert_fp32(const float* vector, const LVDim_t dim, LVTernaryChunk* chunks) {
    assert(vector && dim > 0);
    double sum = 0.0;
    for (LVDim_t i = 0; i < dim; ++i) {
        sum += vector[i];
    }
    float mean = (float)(sum / (double)dim);
    const LVSize_t chunks_count = ternary_chunks_count(dim);
    for (LVDim_t i = 0; i < dim; ++i) {
        float rounded = roundf(vector[i] / (mean + FLT_EPSILON));
        int ternary = rounded < 1.0f ? rounded : 1.0f;
        ternary = ternary > -1.0f ? ternary : -1.0f;
        chunks[i / LV_TERNARY_CHUNK_CAPACITY].mask |= ternary != 0 ? (1ul << (63 - (i % 64))) : 0ull;
        chunks[i / LV_TERNARY_CHUNK_CAPACITY].sign |= ternary < 0 ? (1ul << (63 - (i % 64))) : 0ull;
    }
}

static void ternary_convert_int8(const int8_t* vector, const LVDim_t dim, LVTernaryChunk* chunks) {
    assert(vector && dim > 0);
    int sum = 0;
    for (LVDim_t i = 0; i < dim; ++i) {
        sum += vector[i];
    }
    double mean = (double)(sum) / dim;
    const LVSize_t chunks_count = ternary_chunks_count(dim);
    for (LVDim_t i = 0; i < dim; ++i) {
        double rounded = round((double)vector[i] / mean);
        int ternary = rounded < 1.0 ? rounded : 1.0;
        ternary = ternary > -1.0 ? ternary : -1.0;
        chunks[i / LV_TERNARY_CHUNK_CAPACITY].mask |= ternary != 0 ? (1ul << (63 - (i % 64))) : 0ull;
        chunks[i / LV_TERNARY_CHUNK_CAPACITY].sign |= ternary < 0 ? (1ul << (63 - (i % 64))) : 0ull;
    }
}

LVSize_t ternary_chunks_size(const LVDim_t vector_dim) {
    assert(vector_dim % LV_DIM_MULTIPLE == 0);
    return sizeof(LVTernaryChunk) * ternary_chunks_count(vector_dim);
}

void ternary_convert(const void* vector, const LVVectorType type, const LVDim_t dim, LVTernaryChunk* chunks) {
    assert(vector && chunks);
    memset(chunks, 0, ternary_chunks_size(dim));
    switch (type) {
        case LV_VEC_TYPE_FP32:
            ternary_convert_fp32((const float*)vector, dim, chunks);
            break;

        case LV_VEC_TYPE_INT8:
            ternary_convert_int8((const int8_t*)vector, dim, chunks);
            break;
    }
}

int64_t ternary_dot(const LVTernaryChunk* a, const LVTernaryChunk* b, const LVSize_t words) {
    /*
        Ternary Dot Product's result is (a number of +1s) - (a number of -1s).

        Let's say M is a number of nozeros and S is a number of sign differs.

        A number of +1s is M - S.

        So the formula is (M - S) - S = M - 2S.
    */
    int64_t both_nonzero = 0;
    int64_t sign_differs = 0;

    for (LVSize_t i = 0; i < words; ++i) {
        /*
            Find both non-zero positions which actually join in the calculation.
        */
        int64_t m = a[i].mask & b[i].mask;

        /*
            Find diffrent sign positions where a_i * b_i is -1, and remove signs of zeros by & operation with m.
        */
        int64_t s = (a[i].sign ^ b[i].sign) & m;

        both_nonzero += __builtin_popcountll(m);
        sign_differs += __builtin_popcountll(s);
    }

    return both_nonzero - 2 * sign_differs;
}
