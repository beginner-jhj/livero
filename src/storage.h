#ifndef STORAGE
#define STORAGE

/*
 * storage.h — LVMemTable: the in-memory write buffer of livero's LSM engine
 *
 * WHAT
 *   An ordered, in-memory table of records backed by a skip list. All writes
 *   (put / delete) land here first; when it fills past a threshold it is
 *   flushed to an immutable SST on disk. This is the "MemTable" of the LSM-tree.
 *
 * WHY A SKIP LIST
 *   Records must stay sorted by key so that (a) flush produces a sorted SST in
 *   one linear pass, and (b) lookups and range/filter scans are efficient. A
 *   skip list gives O(log n) search/insert with simple code and no rebalancing
 *   (unlike a balanced tree), which keeps the write path fast and the
 *   implementation small — fitting livero's zero-dependency, embeddable goal.
 *
 * VERSIONING — the invariant that everything else depends on
 *   livero never mutates a record in place. An update or delete inserts a NEW
 *   node with the same key and a higher sequence number (seq). So one key can
 *   have several nodes coexisting, and they are ordered key-ascending, then
 *   seq-DESCENDING (latest version first among equal keys). Readers must return
 *   the FIRST match for a key — that is the newest version. (Delete is just a
 *   node with op = delete, a "tombstone".)
 *
 * MEMORY
 *   Every node is allocated from the table's arena. Nothing is freed
 *   individually; table_destroy frees the whole arena at once (see arena.h).
 *
 * NOT THREAD-SAFE
 *   Single-writer model; no internal locking.
 */

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
