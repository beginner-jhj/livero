#ifndef LV_TERNARY_VECTOR
#define LV_TERNARY_VECTOR

#include <stdint.h>

#include "livero_types.h"

typedef struct LVTernaryChunk {
    uint64_t mask;
    uint64_t sign;
} LVTernaryChunk;

#define LV_TERNARY_CHUNK_CAPACITY 64

LVSize_t ternary_chunks_count(const LVDim_t vector_dim);

LVSize_t ternary_chunks_size(const LVDim_t vector_dim);

void ternary_convert(const void* vector, const LVVectorType type, const LVDim_t dim, LVTernaryChunk* chunks);

int64_t ternary_dot(const LVTernaryChunk* a, const LVTernaryChunk* b, const LVSize_t words);

#endif
