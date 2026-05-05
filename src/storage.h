#ifndef STORAGE
#define STORAGE

#include "node.h"
#include "helper.h"

/* ── MemTable skip list parameters ───────────────────────────────────────────*/
#define LV_SKIPLIST_MAX_LEVEL 20
#define LV_SKIPLIST_P 50 /* probability expressed as integer 0-100 */

/* ── Flush threshold ────────────────────────────────────────────────────────
 * MemTable is flushed to an SSTable when WAL size exceeds this value.
 */
#define LV_FLUSH_THRESHOLD (1 * 1024 * 1024) /* 1 MB */

typedef struct
{
    Node *head;
    Node* tail;
    Arena *arena;
    LVLevel8_t current_level;
} MemTable;

MemTable *create_table(const LVSeq64_t seq);

LVStatus table_insert(MemTable* table,const LVWalOp op, const LVSeq64_t seq,const LVLevel8_t level, const LVSize32_t key_len, const void *key, const LVSize32_t value_len, const void *value,const uint64_t vector_id ,const uint32_t field_mask, const uint32_t field_count,const LVSize32_t field_size,const LVMetaField* field_list);


void table_direct_insert(MemTable* table, Node* node);

#endif
