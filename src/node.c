#include "node.h"

Node *create_node(const Arena *arena, const NodeType type, const LVSeq64_t seq, const LVWalOp op, const LVLevel8_t level, const LVKeyLen32_t key_len, const void *key, const LVValueLen32_t value_len, const void *value, const LVVectorId64_t vector_id, const LVSize32_t field_mask, const LVCount32_t field_count,const LVSize32_t field_size, const LVMetaField *field_list)
{
    int flag = 0;
    Node *node = NULL;

    // Node + level*(Node ptr) + key + value + field
    uint32_t total_node_size = sizeof(Node) + level * sizeof(Node *) + key_len + value_len + field_size;


    if ((node = (Node *)arena_allocate(arena, total_node_size)) == NULL)
    {
        goto _return;
    }

    // init node
    node->type = type;

    node->seq = seq;

    node->op = op;

    node->level = level;

    node->key_len = key_len;

    node->value_len = value_len;

    node->field_count = field_count;

    node->field_mask = field_mask;

    node->vector_id = vector_id;

    memset(node->levels, 0, node->level * sizeof(Node *));

    // copy key
    if (node->key_len > 0)
    {
        memcpy((char *)node + node_key_offset(node->level), key, node->key_len);
    }

    // copy value

    if (node->value_len > 0)
    {
        memcpy((char *)node + node_value_offset(node->level, node->key_len), value, node->value_len);
    }

    if (node->field_count > 0)
    {
        char *current_ptr = (char *)node + node_field_offset(node->level, node->key_len, node->value_len);

        for (int i = 0; i < field_count; ++i)
        {
            LVMetaField *current_field = field_list + i;

            memcpy(current_ptr, &current_field->type, sizeof(LVMetaType));
            current_ptr += sizeof(LVMetaType);

            if (current_field->type == LV_META_STRING)
            {
                memcpy(current_ptr, &current_field->value.str.len, sizeof(uint32_t));
                current_ptr += sizeof(uint32_t);

                if (current_field->value.str.len > 0)
                {
                    memcpy(current_ptr, current_field->value.str.string, current_field->value.str.len);
                    current_ptr += current_field->value.str.len;
                }
            }
            else if (current_field->type == LV_META_INT)
            {
                memcpy(current_ptr, &current_field->value.i64, sizeof(int64_t));
                current_ptr += sizeof(int64_t);
            }
            else
            { // FLOAT
                memcpy(current_ptr, &current_field->value.f64, sizeof(double));
                current_ptr += sizeof(double);
            }
        }
    }

_return:
    return node;
}

void *node_reserve(const Arena *arena, const LVSize32_t node_size)
{
    return (Node*)arena_allocate(arena, node_size);
}

//-1: a is smaller than b
// 1: a is bigger than b

int node_cmp(const NodeType type_a, const void *key_a, const LVKeyLen32_t klen_a, const LVSeq64_t seq_a, const NodeType type_b, const void *key_b, const LVKeyLen32_t klen_b, const LVSeq64_t seq_b)
{
    if (type_a == DATA && type_b == DATA)
    {
        const LVKeyLen32_t min_len = klen_a < klen_b ? klen_a : klen_b;

        const int mcmp_result = memcmp(key_a, key_b, min_len); // compare their headers

        if (mcmp_result != 0)
        {
            return mcmp_result;
        }

        // if headers are same, compare their lengthes

        if (klen_a < klen_b)
        {
            return -1;
        }

        else if (klen_a > klen_b)
        {
            return 1;
        }

        else
        { // if lengthes are same, comapre their sequences
            // bigger seq must be prior
            if (seq_a > seq_b)
            {
                return -1;
            }

            else // seqs of data nodes can not be equal
            {
                return 1;
            }
        }
    }

    else
    {
        if (type_a == HEAD)
        {
            return -1;
        }

        else if (type_a == TAIL)
        {
            return 1;
        }

        else
        {
            if (type_b == HEAD)
            {
                return 1;
            }

            else if (type_b == TAIL)
            {
                return -1;
            }
        }
    }
}

void *node_access_key(const Node *node)
{
    return (char *)node + node_key_offset(node->level);
}

void*node_access_value(const Node* node){
    return (char*)node + node_value_offset(node->level, node->key_len);
}
