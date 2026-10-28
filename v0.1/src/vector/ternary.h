#ifndef LV_TERNARY_VECTOR
#define LV_TERNARY_VECTOR

#include <stdint.h>
#include <assert.h>
#include "livero_types.h"
#include "livero_internal.h"

typedef struct LVTernaryChunk {
    uint64_t mask;
    uint64_t sign;
} LVTernaryChunk;

#define LV_TERNARY_CHUNK_CAPACITY 64
#define LV_TERNARY_CHUNK_BYTES 16

static_assert(sizeof(LVTernaryChunk) == LV_TERNARY_CHUNK_BYTES, "LVTernaryChunk size must be 16bytes.");


static inline LVCount_t ternary_chunks_count(const LVDim_t dim) {
    assert(dim != 0 && dim % LV_DIMENSION_MULTIPLE); 
    return dim / LV_TERNARY_CHUNK_CAPACITY;
}

static inline LVSize32_t ternary_chunks_bytes(const LVDim_t dim){
    return ternary_chunks_count(dim)*LV_TERNARY_CHUNK_BYTES;
}

#endif
