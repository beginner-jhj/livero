#ifndef NODE
#define NODE

#include "lv_internal.h"
#include "wal.h"
#include "arena.h"

typedef enum NodeType
{
    HEAD = 0,
    TAIL = 1,
    DATA = 2
} NodeType;

typedef struct Node
{
    NodeType type;
    LVSeq64_t seq;
    LVWalOp op;
    LVLevel8_t level;
    LVKeyLen32_t key_len;
    LVValueLen32_t value_len;
    LVSize32_t field_mask;
    LVCount32_t field_count;
    LVVectorId64_t vector_id;
    struct Node *levels[];
} Node;

Node *create_node(const Arena *arena, const NodeType type, const LVSeq64_t seq, const LVWalOp op, const LVLevel8_t level, const LVKeyLen32_t key_len, const void *key, const LVValueLen32_t value_len, const void *value, const LVVectorId64_t vector_id, const LVSize32_t field_mask, const LVCount32_t field_count,const LVSize32_t field_size, const LVMetaField *field_list);

uint32_t node_key_offset(const LVLevel8_t level)
{
    return sizeof(Node)+ sizeof(Node *) * level;
}

uint32_t node_value_offset(const LVLevel8_t level, const LVKeyLen32_t klen)
{
    return node_key_offset(level) + klen;
}

uint32_t node_field_offset(const LVLevel8_t level, const LVKeyLen32_t klen, const LVValueLen32_t vlen)
{
    return node_value_offset(level, klen) + vlen;
}

int node_cmp(const NodeType type_a, const void *key_a, const LVKeyLen32_t klen_a, const LVSeq64_t seq_a, const NodeType type_b, const void *key_b, const LVKeyLen32_t klen_b, const LVSeq64_t seq_b);

void* node_access_key(const Node* node);
void* node_access_value(const Node* node);

void* node_reserve(const Arena* arena,const LVSize32_t node_size);

#endif
