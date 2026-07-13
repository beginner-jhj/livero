#ifndef STORAGE
#define STORAGE

#include "lv_internal.h"

/* ── LVMemTable skip list parameters ───────────────────────────────────────────*/
#define LV_SKIPLIST_MAX_LEVEL 20
#define LV_SKIPLIST_P 50 /* probability expressed as integer 0-100 */

typedef struct LVMemTable
{
    LVNode* head;
    LVNode* tail;
    LVArena* arena;
    LVSize32_t node_count;
    LVLevel8_t current_level;
} LVMemTable;

LVMemTable* table_create(void);
void table_destroy(LVMemTable* table);

LVNode* table_insert(LVMemTable* table, const LVNodeOp op, const LVSeq64_t seq, const LVLevel8_t level, const LVSize32_t key_len, const void* key, const LVSize32_t value_len, const void* value, const LVVectorId64_t vector_id, const LVFieldMask32_t field_mask, const LVCount32_t field_count, const LVSize32_t field_size, const void* field_buffer);

void table_direct_insert(LVMemTable* table, LVNode* node);

LVNode* table_search(const LVMemTable* table, const void* key, const LVKeyLen32_t key_len);

LVStatus table_query_filter_scan(const LVMemTable* table, const LVSchema* schema, const LVAstNode* query, const LVFieldMask32_t query_field_mask, const LVOrdbyType ordbytype, const LVFieldMask32_t ordby_field_mask, const LVQVSetAppendFn qv_append_fn, LVQVSet* qv_set);

LVNode* table_get_next_node(const LVNode* current);



#endif
