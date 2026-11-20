#ifndef LV_QUERY_HEAP
#define LV_QUERY_HEAP

#include <stdint.h>

#include "livero_types.h"
#include "vector/ternary.h"

typedef struct {
    LVQueryResult q_result;
    float score;
    int is_on_pending;
    LVCount_t ternary_idx;
    LVCount_t ternary_heap_idx;
} LVQueryHeapEntry;

typedef struct {
    const uint64_t min_epoch_date;
    const uint64_t max_epoch_date;
    const LVCount_t capacity;
    const LVCount_t ternary_chunk_count;
    const LVSize16_t ternary_bytes;
    LVTernaryChunk* entries_ternaries;
    LVQueryHeapEntry* entries;
    LVSize32_t size;
} LVQueryHeap;

void lv_query_heap_build(LVQueryHeap* heap);

void lv_query_heap_sort(LVQueryHeap* heap);

int lv_query_heap_admit(const LVQueryHeap *heap, const float new_score);

void lv_query_heap_update(LVQueryHeap* heap, const LVTernaryChunk* ternary, const uint8_t* row,const float score, const int is_on_pending, const LVCount_t ternary_idx);

#endif
