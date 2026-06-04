#include "node.h"
#include "schema.h"
#include "arena.h"
#include <string.h>
#include "query.h"

LVNode* create_node(const LVArena* arena, const LVNodeType type, const LVSeq64_t seq, const LVNodeOp op, const LVLevel8_t level, const LVKeyLen32_t key_len, const void* key, const LVValueLen32_t value_len, const void* value, const LVVectorId64_t vector_id, const LVSize32_t field_mask, const LVCount32_t field_count, const LVSize32_t field_size, const LVMetaField* field_list)
{
    LVNode* node = NULL;

    // LVNode + level*(LVNode ptr) + key + value + field
    uint32_t total_node_size = sizeof(LVNode) + level * sizeof(LVNode*) + key_len + value_len + field_size;

    if ((node = (LVNode*)arena_allocate(arena, total_node_size, -1)) == NULL)
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

    memset(node->levels, 0, node->level * sizeof(LVNode*));

    // copy key
    if (node->key_len > 0)
    {
        memcpy((char*)node + node_key_offset(node->level), key, node->key_len);
    }

    // copy value

    if (node->value_len > 0 && op == LV_PUT)
    {
        memcpy((char*)node + node_value_offset(node->level, node->key_len), value, node->value_len);
    }

    if (node->field_count > 0 && field_list && op == LV_PUT)
    {
        char* current_ptr = (char*)node + node_field_offset(node->level, node->key_len, node->value_len);

        for (int i = 0; i < field_count; ++i)
        {
            LVMetaField* current_field = field_list + i;

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

    else if (node->field_count > 0 && op == LV_UPDATE) {
        char* field_cpy_ptr = (char*)node + node_field_offset(node->level, node->key_len, node->value_len);
        memcpy(field_cpy_ptr, field_list, field_size);
    }

_return:
    return node;
}

LVNode* node_reserve(const LVArena* arena, const LVLevel8_t level, const LVKeyLen32_t key_len, const LVValueLen32_t value_len, const LVSize32_t field_size) {
    const LVSize32_t size_to_reserve = sizeof(LVNode) + level * sizeof(LVNode*) + key_len + value_len + field_size;
    return (LVNode*)arena_allocate(arena, size_to_reserve, -1);
}

uint32_t node_key_offset(const LVLevel8_t level)
{
    return sizeof(LVNode) + sizeof(LVNode*) * level;
}

uint32_t node_value_offset(const LVLevel8_t level, const LVKeyLen32_t klen)
{
    return node_key_offset(level) + klen;
}

uint32_t node_field_offset(const LVLevel8_t level, const LVKeyLen32_t klen, const LVValueLen32_t vlen)
{
    return node_value_offset(level, klen) + vlen;
}

//-1: a < b
// 1: a > b

int node_cmp(const LVNodeType type_a, const void* key_a, const LVKeyLen32_t klen_a, const LVSeq64_t seq_a, const LVNodeType type_b, const void* key_b, const LVKeyLen32_t klen_b, const LVSeq64_t seq_b)
{
    if (type_a == LV_NODE_DATA && type_b == LV_NODE_DATA)
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
        if (type_a == LV_NODE_HEAD)
        {
            return -1;
        }

        else if (type_a == LV_NODE_TAIL)
        {
            return 1;
        }

        else
        {
            if (type_b == LV_NODE_HEAD)
            {
                return 1;
            }

            else if (type_b == LV_NODE_TAIL)
            {
                return -1;
            }
        }
    }
}

int node_key_equal(const void* key_a, const LVKeyLen32_t klen_a, const void* key_b, const LVKeyLen32_t klen_b)
{
    if (klen_a != klen_b)
        return 0;
    return memcmp(key_a, key_b, klen_a) == 0;
}

void* node_access_key(const LVNode* node)
{
    return (char*)node + node_key_offset(node->level);
}

void* node_access_value(const LVNode* node)
{
    return (char*)node + node_value_offset(node->level, node->key_len);
}

void* node_access_field(const LVNode* node, const int number)
{
    char* field = (char*)node + node_field_offset(node->level, node->key_len, node->value_len);
    LVMetaType type;
    for (int i = 0; i < number; ++i)
    {
        memcpy(&type, field, sizeof(LVMetaType));
        field += sizeof(LVMetaType);

        switch (type)
        {
        case LV_META_FLOAT:
            field += sizeof(double);
            break;
        case LV_META_INT:
            field += sizeof(int64_t);
            break;

        case LV_META_STRING:
        {
            uint32_t len = 0;
            memcpy(&len, field, sizeof(uint32_t));
            field += sizeof(uint32_t) + len;
            break;
        }
        default:
            break;
        }
    }

    return field;
}


double node_get_f64_field(const LVNode* node, const LVSize32_t mask) {
    int number = node_field_number(node, mask);
    char* f64_field = (char*)node_access_field(node, number);
    f64_field += sizeof(LVMetaType);
    double value = 0.0;
    memcpy(&value, f64_field, sizeof(double));
    return value;
}
int64_t node_get_i64_field(const LVNode* node, const LVSize32_t mask) {
    int number = node_field_number(node, mask);
    char* i64_field = (char*)node_access_field(node, number);
    i64_field += sizeof(LVMetaType);
    int64_t value = 0.0;
    memcpy(&value, i64_field, sizeof(int64_t));
    return value;
}

int node_field_number(const LVNode* node, const LVSize32_t target_mask)
{
    if (!(node->field_mask & target_mask) || node->field_count == 0) return -1;

    int number = 0;
    for (int i = 0; i < LV_MAX_META_FIELDS - 1; ++i) {
        uint32_t bit = (1u << i);
        if (bit == target_mask) {
            return number;
        }

        if (node->field_mask & bit) {
            ++number;
        }
    }

    return -1;
}

uint32_t node_field_number_to_mask(const LVNode* node, const int number) {
    if (number < 0 || number > node->field_count - 1) return 0;
    uint32_t first_mask = 1;
    int i = 0;
    while (!((first_mask = first_mask << i) & node->field_mask) && i < LV_MAX_META_FIELDS) i++;
    for(int j=0; j<LV_MAX_META_FIELDS - i; ++j){
        if(node->field_mask & (first_mask << j) && j==number){
            return first_mask << j;
        }
    }
    return 0;
}

int node_eval_query(const LVNode* node, const LVAstNode* query, const LVSchema* schema) {
    return query_eval_ast(query, node, schema);
}

LVSize32_t node_field_size(const LVNode* node, const int is_serialized) {
    if (node->field_count <= 0) return 0;
    char* field_ptr = (char*)node_access_field(node, 0);
    LVSize32_t size = 0;
    for (int i = 0; i < node->field_count; ++i) {
        LVMetaType type;
        memcpy(&type, field_ptr, sizeof(LVMetaType));

        size += is_serialized == 1 ? 1 : sizeof(LVMetaType);

        field_ptr += sizeof(LVMetaType);

        if (type == LV_META_STRING) {
            uint32_t len = 0;
            memcpy(&len, field_ptr, sizeof(uint32_t));
            size += sizeof(uint32_t) + len;
            field_ptr += sizeof(uint32_t) + len;
        }
        else {
            size += 8;
            field_ptr += 8;
        }
    }
    return size;
}
