#include "query_heap.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "livero_types.h"
#include "struct/row.h"
#include "vector/ternary.h"

static void lv_query_heap_sift_down(LVQueryHeap* heap, uint32_t i) {
    assert(heap);
    while (1) {
        uint32_t smallest = i;
        uint32_t left = 2 * i + 1;
        uint32_t right = 2 * i + 2;

        if (left < heap->size && heap->entries[left].score < heap->entries[smallest].score) {
            smallest = left;
        }

        if (right < heap->size && heap->entries[right].score < heap->entries[smallest].score) {
            smallest = right;
        }

        if (smallest != i) {
            LVQueryHeapEntry tmp = heap->entries[i];
            heap->entries[i] = heap->entries[smallest];
            heap->entries[smallest] = tmp;
            i = smallest;
        } else {
            break;
        }
    }
}

static void lv_query_heap_ternary_insert(LVQueryHeap* heap, const LVTernaryChunk* ternary, const LVCount_t idx) {
    assert(heap && ternary);
    memcpy(&heap->entries_ternaries[idx * heap->ternary_chunk_count], ternary, heap->ternary_bytes);
}

static void lv_query_heap_insert(LVQueryHeap* heap, const LVQueryHeapEntry entry) {
    assert(heap);
    heap->entries[heap->size] = entry;
    int i = heap->size;
    (heap->size)++;

    while (i > 0 && heap->entries[(i - 1) / 2].score > heap->entries[i].score) {
        LVQueryHeapEntry parent = heap->entries[(i - 1) / 2];
        heap->entries[(i - 1) / 2] = heap->entries[i];
        heap->entries[i] = parent;
        i = (i - 1) / 2;
    }
}

static void lv_query_heap_pop(LVQueryHeap* heap, LVQueryHeapEntry* pop_out) {
    assert(heap);
    assert(heap->size > 0);
    LVQueryHeapEntry root = heap->entries[0];
    if (pop_out) {
        *pop_out = root;
    }
    (heap->size)--;
    heap->entries[0] = heap->entries[heap->size];
    heap->entries[heap->size] = root;

    lv_query_heap_sift_down(heap, 0);
}

void lv_query_heap_build(LVQueryHeap* heap) {
    assert(heap);
    for (uint32_t i = heap->size / 2; i-- > 0;) {  // size/2 - 1, ..., 1, 0
        lv_query_heap_sift_down(heap, i);
    }
}

void lv_query_heap_sort(LVQueryHeap* heap) {
    assert(heap);
    while (heap->size > 0) {
        lv_query_heap_pop(heap, NULL);
    }
}

int lv_query_heap_admit(const LVQueryHeap* heap, const float new_score) {
    assert(heap);
    return heap->size < heap->capacity || new_score > heap->entries[0].score;
}

void lv_query_heap_update(LVQueryHeap* heap, const int use_postfilter, const LVTernaryChunk* ternary,
                          const uint8_t* row, const float score, const int is_on_pending, const LVCount_t idx) {
    assert(heap);

    if (!use_postfilter) {
        assert(row);
    }

    const int full = heap->size >= heap->capacity;
    if (!lv_query_heap_admit(heap, score)) {
        return;
    }

    if (!use_postfilter) {
        const int filter_passed = query_heap_row_filter(heap, row);

        if (filter_passed) {
            LVQueryHeapEntry entry = {.q_result = row_make_query_result(row),
                                      .is_on_pending = is_on_pending,
                                      .idx = idx,
                                      .score = score,
                                      .ternary_heap_idx = heap->size,
                                      .filter_passed = 1};

            if (full) {
                // Replace the weakest kept candidate (the root) and restore the heap:
                // one sift-down instead of a pop (sift-down) followed by an insert (sift-up).
                entry.ternary_heap_idx = heap->entries[0].ternary_heap_idx;
                heap->entries[0] = entry;
                lv_query_heap_sift_down(heap, 0);
            } else {
                lv_query_heap_insert(heap, entry);
            }

            lv_query_heap_ternary_insert(heap, ternary, entry.ternary_heap_idx);
        }
    } else {
        /*
        0 is a default value for filter_passed when the use_postfilter is
        on. This will change to real value after
        */
        LVQueryResult q_result = (LVQueryResult){0};
        LVQueryHeapEntry entry = {.q_result = q_result,
                                  .is_on_pending = is_on_pending,
                                  .idx = idx,
                                  .score = score,
                                  .ternary_heap_idx = heap->size,
                                  .filter_passed = 0};

        if (full) {
            entry.ternary_heap_idx = heap->entries[0].ternary_heap_idx;
            heap->entries[0] = entry;
            lv_query_heap_sift_down(heap, 0);
        } else {
            lv_query_heap_insert(heap, entry);
        }

        lv_query_heap_ternary_insert(heap, ternary, entry.ternary_heap_idx);
    }
}
