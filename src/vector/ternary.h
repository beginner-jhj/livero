#ifndef LV_TERNARY_VECTOR
#define LV_TERNARY_VECTOR

#include <stdint.h>
#include <assert.h>
#include "livero_types.h"

typedef struct LVTernaryChunk {
    uint64_t mask;
    uint64_t sign;
} LVTernaryChunk;

#define LV_TERNARY_CHUNK_CAPACITY 64
#define LV_TERNARY_CHUNK_BYTES 16

static_assert(sizeof(LVTernaryChunk) == LV_TERNARY_CHUNK_BYTES, "LVTernaryChunk size must be 16bytes.");

#define LV_DIMENSION_MULTIPLE 64



static inline LVCount_t ternary_chunks_count(const LVDim_t vector_dimension) {
    assert(vector_dimension != 0 && vector_dimension % LV_DIMENSION_MULTIPLE == 0);
    return vector_dimension / LV_TERNARY_CHUNK_CAPACITY;
}

static inline LVSize32_t ternary_chunks_bytes(const LVDim_t vector_dimension) {
    return ternary_chunks_count(vector_dimension) * LV_TERNARY_CHUNK_BYTES;
}

void ternary_convert(const void* vector, const LVVectorType vector_type, const LVDim_t vector_dimension,
                     LVTernaryChunk* ternary_chunks);

int64_t ternary_dot_scalar(const LVTernaryChunk* ternary_a, const LVTernaryChunk* ternary_b,
                                  const LVCount_t ternary_chunk_count);

int64_t ternary_dot(const LVTernaryChunk* ternary_a, const LVTernaryChunk* ternary_b,
                    const LVCount_t ternary_chunk_count);


#endif
