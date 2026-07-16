#ifndef NODE
#define NODE

/*
 * node.h — LVNode: the record cell shared by the memtable, WAL, and SST paths
 *
 * WHAT
 *   One LVNode holds a single record: its op (put/delete), seq, key, value,
 *   vector_id, and metadata fields, plus the skip-list level pointers. It's the
 *   in-memory unit the memtable stores, the WAL rebuilds on recovery, and the
 *   SST reads/writes.
 *
 * FLEXIBLE LAYOUT (one allocation)
 *   A node is a single arena block laid out as:
 *     [ LVNode header ][ levels[]: level ptrs ][ key ][ value ][ field bytes ]
 *   levels[] is a flexible array member; key/value/fields follow it, located by
 *   node_key_offset / node_value_offset / node_field_offset. One allocation per
 *   node (no separate malloc per part) keeps nodes compact and arena-friendly.
 *
 * ORDERING (node_cmp) — the rule the whole engine leans on
 *   Nodes sort by key ascending, and among EQUAL keys by seq DESCENDING (newest
 *   first). Sentinels (HEAD/TAIL) always sort to the ends. This single rule is
 *   why table_search returns the newest version (first match at level 0), and
 *   why SST merge lets the memtable (newer seq) win over the old SST for the
 *   same key. See node_cmp.
 *
 * MEMORY: nodes are arena-allocated; never freed individually.
 */

#include "lv_internal.h"

typedef struct LVNode
{
    LVNodeType type;
    LVSeq64_t seq;
    LVNodeOp op;
    LVLevel8_t level;
    LVKeyLen32_t key_len;
    LVValueLen32_t value_len;
    LVFieldMask32_t field_mask;
    LVCount32_t field_count;
    LVVectorId64_t vector_id;
    LVHnswNode* hnsw_node;
    struct LVNode* levels[];
} LVNode;

LVNode* node_create(LVArena* arena, const LVNodeType type, const LVSeq64_t seq, const LVNodeOp op, const LVLevel8_t level, const LVKeyLen32_t key_len, const void* key, const LVValueLen32_t value_len, const void* value, const LVVectorId64_t vector_id, const LVFieldMask32_t field_mask, const LVCount32_t field_count, const LVSize32_t field_size, const void* field_buffer);
LVNode* node_reserve(LVArena* arena, const LVLevel8_t level, const LVKeyLen32_t key_len, const LVValueLen32_t value_len, const LVSize32_t field_size);

LVSize32_t node_key_offset(const LVLevel8_t level);

LVSize32_t node_value_offset(const LVLevel8_t level, const LVKeyLen32_t klen);

LVSize32_t node_field_offset(const LVLevel8_t level, const LVKeyLen32_t klen, const LVValueLen32_t vlen);

int node_cmp(const LVNodeType type_a, const void* key_a, const LVKeyLen32_t klen_a, const LVSeq64_t seq_a, const LVNodeType type_b, const void* key_b, const LVKeyLen32_t klen_b, const LVSeq64_t seq_b);
int node_key_equal(const void* key_a, const LVKeyLen32_t klen_a, const void* key_b, const LVKeyLen32_t klen_b);

void* node_access_key(const LVNode* node);
void* node_access_value(const LVNode* node);
void* node_access_field(const LVNode* node, const LVCount32_t number);
void* node_field_buffer_access(const void* field_buffer, const LVCount32_t field_number);

double node_get_f64_field(const LVNode* node, const LVFieldMask32_t mask);
int64_t node_get_int64_field(const LVNode* node, const LVFieldMask32_t mask);


int node_field_number_of_mask(const LVFieldMask32_t node_field_mask, const LVFieldMask32_t target_mask);

LVFieldMask32_t node_field_number_to_mask(const LVFieldMask32_t field_mask, const LVCount32_t number);


int node_eval_query(const LVNode* node, const LVAstNode* query, const LVSchema* schema);

LVSize32_t node_field_size(const LVNode* node);

void node_link_hnsw_node(LVNode* node, const LVHnswNode* hnsw_node);

#endif
