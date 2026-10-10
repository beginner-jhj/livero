#include "ternary.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "internal/internal.h"
#include "livero_types.h"

LVSize_t ternary_chunks_count(const LVDim_t dim) {
    assert(dim % LV_DIM_MULTIPLE == 0);
    assert(dim >= LV_TERNARY_CHUNK_CAPACITY && dim < LV_MAX_VECTOR_DIMENSION);
    return dim / LV_TERNARY_CHUNK_CAPACITY;
}

static void ternary_make(LVTernaryChunk* chunks, const int ternary, const LVDim_t i) {
    const LVDim_t chunk = i / LV_TERNARY_CHUNK_CAPACITY;
    const LVDim_t bit = i % LV_TERNARY_CHUNK_CAPACITY;
    const uint64_t mask_bit = 1ull << (LV_TERNARY_CHUNK_CAPACITY - 1 - bit);
    if (ternary != 0) {
        chunks[chunk].mask |= mask_bit;
    }
    if (ternary < 0) {
        chunks[chunk].sign |= mask_bit;
    }
}

static void ternary_convert_fp32(const float* vector, const LVDim_t dim, LVTernaryChunk* chunks) {
    assert(vector && dim > 0);

    double sum = 0.0;
    for (LVDim_t i = 0; i < dim; ++i) {
        sum += fabs((double)vector[i]);
    }
    // gamma >= 0 now, so adding epsilon only guards the all-zero vector.
    const float gamma = (float)(sum / (double)dim) + FLT_EPSILON;

    for (LVDim_t i = 0; i < dim; ++i) {
        float r = roundf(vector[i] / gamma);

        if (r > 1.0f) {
            r = 1.0f;
        }
        if (r < -1.0f) {
            r = -1.0f;
        }
        const int ternary = (int)r;  // exactly -1, 0, or 1 at this point
        ternary_make(chunks, ternary, i);
    }
}
static void ternary_convert_int8(const int8_t* vector, const LVDim_t dim, LVTernaryChunk* chunks) {
    assert(vector && dim > 0);
    uint64_t sum = 0;
    for (LVDim_t i = 0; i < dim; ++i) {
        sum += abs(vector[i]);
    }
    const float gamma = (float)(sum / (double)dim) + FLT_EPSILON;
    for (LVDim_t i = 0; i < dim; ++i) {
        float r = roundf(vector[i] / gamma);
        if (r > 1.0f) {
            r = 1.0f;
        }
        if (r < -1.0f) {
            r = -1.0f;
        }
        const int ternary = (int)r;
        ternary_make(chunks, ternary, i);
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
