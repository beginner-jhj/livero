#ifndef STORAGE
#define STORAGE

#include "lv_internal.h"

/* ── LVMemTable skip list parameters ───────────────────────────────────────────*/
#define LV_SKIPLIST_MAX_LEVEL 20
#define LV_SKIPLIST_P 50 /* probability expressed as integer 0-100 */

/* ── Flush threshold ────────────────────────────────────────────────────────
 * LVMemTable is flushed to an SSTable when WAL size exceeds this value.
 */
#define LV_FLUSH_THRESHOLD (1 * 1024 * 1024) /* 1 MB */

typedef struct LVMemTable
{
    LVNode* head;
    LVNode* tail;
    LVArena* arena;
    LVSize32_t node_count;
    LVLevel8_t current_level;
} LVMemTable;

typedef struct LVTableQueryValue
{
    LVNode* node;
    LVVectorDisValue dis;
    uint32_t ordby_field_mask;
} LVTableQueryValue;

typedef struct LVTableQVList
{
    LVTableQueryValue* values;
    LVSize32_t size;
    LVSize32_t capacity;
} LVTableQVList;

typedef struct LVTableQueryResult {
    LVSeq64_t node_seq;
    LVVectorId64_t vector_id;
    LVNode* node;
    void* key;
    LVKeyLen32_t key_len;
    void* value;
    LVValueLen32_t value_len;
    void* vector;
} LVTableQueryResult;

typedef struct LVTableQueryResultSet {
    LVSize32_t size;
    LVTableQueryResult* results;
} LVTableQueryResultSet;

LVMemTable* create_table(const LVSeq64_t seq);

LVStatus table_insert(LVMemTable* table, const LVNodeOp op, const LVSeq64_t seq, const LVLevel8_t level, const LVSize32_t key_len, const void* key, const LVSize32_t value_len, const void* value, const uint64_t vector_id, const uint32_t field_mask, const uint32_t field_count, const LVSize32_t field_size, const LVMetaField* field_list);

void table_direct_insert(LVMemTable* table, LVNode* node);

LVNode* table_search(const LVMemTable* table, const void* key, const LVKeyLen32_t key_len);

LVTableQueryResultSet* table_query(const LVMemTable* table, const LVSchema* schema, const LVAstNode* query, const void* query_vector, const LVDim32_t dim, const LVHnswIDMap* id_vector_map, const LVSize32_t field_mask, const LVQueryOption* option);

void table_query_apply_range(LVTableQVList* qv_list, const LVVectorType vector_type, const LVQueryOption* option);
void table_query_apply_ordby(LVTableQVList* qv_list, const LVQueryOption* option, const LVSchema* schema);

int ordvec_f32_desc(const void* a, const void* b);

int ordvec_f32_asc(const void* a, const void* b);

int ordvec_i8_desc(const void* a, const void* b);

int ordvec_i8_asc(const void* a, const void* b);

#define ordvec_f32_dot_nearest ordvec_f32_desc
#define ordvec_f32_dot_farthest ordvec_f32_asc
#define ordvec_f32_l2_nearest ordvec_f32_asc
#define ordvec_f32_l2_farthest ordvec_f32_desc

#define ordvec_i8_dot_nearest ordvec_i8_desc
#define ordvec_i8_dot_farthest ordvec_i8_asc
#define ordvec_i8_l2_nearest ordvec_i8_asc
#define ordvec_i8_l2_farthest ordvec_i8_desc

void table_query_apply_limit(LVTableQVList* qv_list, const LVSize32_t limit);

int ordby_f64_asc(const void* a, const void* b);
int ordby_f64_desc(const void* a, const void* b);

int ordby_i64_asc(const void* a, const void* b);
int ordby_i64_desc(const void* a, const void* b);

LVTableQVList* create_qv_list(void);

LVStatus table_qv_list_append(LVTableQVList* qv_list, const LVNode* node, const LVVectorDisValue dis, const uint32_t ordby_field_mask);

void destroy_qv_list(LVTableQVList* qv_list);

#endif
