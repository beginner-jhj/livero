#ifndef NODE
#define NODE

#include "lv_internal.h"

typedef struct Node
{
    LVNodeType type;
    LVSeq64_t seq;
    LVInsertOp op;
    LVLevel8_t level;
    LVKeyLen32_t key_len;
    LVValueLen32_t value_len;
    LVSize32_t field_mask;
    LVCount32_t field_count;
    LVVectorId64_t vector_id;
    struct Node *levels[];
} Node;

Node *create_node(const Arena *arena, const LVNodeType type, const LVSeq64_t seq, const LVInsertOp op, const LVLevel8_t level, const LVKeyLen32_t key_len, const void *key, const LVValueLen32_t value_len, const void *value, const LVVectorId64_t vector_id, const LVSize32_t field_mask, const LVCount32_t field_count, const LVSize32_t field_size, const LVMetaField *field_list);
void *node_reserve(const Arena *arena, const LVSize32_t node_size);

uint32_t node_key_offset(const LVLevel8_t level)
{
    return sizeof(Node) + sizeof(Node *) * level;
}

uint32_t node_value_offset(const LVLevel8_t level, const LVKeyLen32_t klen)
{
    return node_key_offset(level) + klen;
}

uint32_t node_field_offset(const LVLevel8_t level, const LVKeyLen32_t klen, const LVValueLen32_t vlen)
{
    return node_value_offset(level, klen) + vlen;
}

int node_cmp(const LVNodeType type_a, const void *key_a, const LVKeyLen32_t klen_a, const LVSeq64_t seq_a, const LVNodeType type_b, const void *key_b, const LVKeyLen32_t klen_b, const LVSeq64_t seq_b);
int node_key_equal(const void *key_a, const LVKeyLen32_t klen_a, const void *key_b, const LVKeyLen32_t klen_b);

void *node_access_key(const Node *node);
void *node_access_value(const Node *node);
void *node_access_field(const Node *node, const int number);

int node_field_number(const Node* node, const LVSize32_t mask);

#endif
