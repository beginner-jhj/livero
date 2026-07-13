#ifndef NODE
#define NODE

#include "lv_internal.h"

typedef struct LVNode
{
    LVNodeType type;
    LVSeq64_t seq;
    LVNodeOp op;
    LVLevel8_t level;
    LVKeyLen32_t key_len;
    LVValueLen32_t value_len;
    LVSize32_t field_mask;
    LVCount32_t field_count;
    LVVectorId64_t vector_id;
    LVHnswNode* hnsw_node;
    struct LVNode* levels[];
} LVNode;

LVNode* node_create(LVArena* arena, const LVNodeType type, const LVSeq64_t seq, const LVNodeOp op, const LVLevel8_t level, const LVKeyLen32_t key_len, const void* key, const LVValueLen32_t value_len, const void* value, const LVVectorId64_t vector_id, const LVSize32_t field_mask, const LVCount32_t field_count, const LVSize32_t field_size, const void* field_buffer);
LVNode* node_reserve(LVArena* arena, const LVLevel8_t level, const LVKeyLen32_t key_len, const LVValueLen32_t value_len, const LVSize32_t field_size);

uint32_t node_key_offset(const LVLevel8_t level);

uint32_t node_value_offset(const LVLevel8_t level, const LVKeyLen32_t klen);

uint32_t node_field_offset(const LVLevel8_t level, const LVKeyLen32_t klen, const LVValueLen32_t vlen);

int node_cmp(const LVNodeType type_a, const void* key_a, const LVKeyLen32_t klen_a, const LVSeq64_t seq_a, const LVNodeType type_b, const void* key_b, const LVKeyLen32_t klen_b, const LVSeq64_t seq_b);
int node_key_equal(const void* key_a, const LVKeyLen32_t klen_a, const void* key_b, const LVKeyLen32_t klen_b);

void* node_access_key(const LVNode* node);
void* node_access_value(const LVNode* node);
void* node_access_field(const LVNode* node, const int number);
void* node_field_buffer_access(const void* field_buffer, const int field_number);

double node_get_double_field(const LVNode* node, const LVCount32_t mask);
int64_t node_get_int64_field(const LVNode* node, const LVCount32_t mask);


int node_field_number_of_mask(const LVCount32_t node_field_mask, const LVCount32_t target_mask);

uint32_t node_field_number_to_mask(const LVCount32_t field_mask, int number);


int node_eval_query(const LVNode* node, const LVAstNode* query, const LVSchema* schema);

LVSize32_t node_field_size(const LVNode* node);

void node_link_hnsw_node(LVNode* node, const LVHnswNode* hnsw_node);

#endif
