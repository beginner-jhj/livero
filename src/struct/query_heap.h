#ifndef LV_QUERY_HEAP
#define LV_QUERY_HEAP

#include <stdint.h>
#include <assert.h>

#include "livero_types.h"
#include "struct/row.h"
#include "vector/ternary.h"

typedef struct {
    LVQueryResult q_result;
    float score;
    int is_on_pending;
    LVCount_t idx;
    LVCount_t ternary_heap_idx;
    int filter_passed; // When posfilter is off, this is always 1 because only valid entries are accepted when prefiltering.
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

void lv_query_heap_update(LVQueryHeap* heap,const int use_postfilter, const LVTernaryChunk* ternary, const uint8_t* row, const float score, const int is_on_pending, const LVCount_t idx);

static inline int query_heap_row_filter(const LVQueryHeap* heap, const uint8_t* row){
    assert(heap && row);
    return (row_is_alive(row)) && (heap->min_epoch_date <= row_get_epoch_date(row) && row_get_epoch_date(row) <= heap->max_epoch_date);
}

#endif
