#include "ternary.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <memory.h>
#include <stdlib.h>

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

void ternary_convert(const void* vector, const LVVectorType type, const LVDim_t dim, LVTernaryChunk* chunks) {
    assert(vector && chunks);
    memset(chunks, 0, ternary_chunks_bytes(dim));
    switch (type) {
        case LV_VEC_TYPE_FP32:
            ternary_convert_fp32((const float*)vector, dim, chunks);
            break;

        case LV_VEC_TYPE_INT8:
            ternary_convert_int8((const int8_t*)vector, dim, chunks);
            break;
    }
}
