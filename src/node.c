#include "node.h"
#include "schema.h"
#include "arena.h"
#include <string.h>
#include "query.h"


/*
 * Allocate one node sized for its level + key + value + fields, and copy the
 * record data in. Single arena allocation; key/value/fields are placed after
 * the levels[] array via the node_*_offset helpers. align = -1 lets the arena
 * pick max_align_t (the vector isn't stored here, so no SIMD alignment needed).
 */

LVNode* node_create(LVArena* arena, const LVNodeType type, const LVSeq64_t seq, const LVNodeOp op, const LVLevel8_t level, const LVKeyLen32_t key_len, const void* key, const LVValueLen32_t value_len, const void* value, const LVVectorId64_t vector_id, const LVFieldMask32_t field_mask, const LVCount32_t field_count, const LVSize32_t field_size, const void* field_buffer)
{
    LVNode* node = NULL;

    // LVNode + level*(LVNode ptr) + key + value + field
    LVSize32_t total_node_size = sizeof(LVNode) + level * sizeof(LVNode*) + key_len + value_len + field_size;

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

    node->hnsw_node = NULL;

    memset(node->levels, 0, node->level * sizeof(LVNode*));

    // copy key
    if (node->key_len > 0)
    {
        memcpy((char*)node + node_key_offset(node->level), key, node->key_len);
    }

    // copy value

    if (node->value_len > 0)
    {
        memcpy((char*)node + node_value_offset(node->level, node->key_len), value, node->value_len);
    }

    //copy field
    if (node->field_count > 0) {
        char* field_copy_ptr = (char*)node + node_field_offset(node->level, node->key_len, node->value_len);
        memcpy(field_copy_ptr, field_buffer, field_size);
    }

_return:
    return node;
}

/*
 * Like node_create but allocates WITHOUT copying data in — reserves a
 * correctly-sized node for the caller to fill. Used by WAL recovery, which
 * reads each field directly into the reserved node's key/value/field regions.
 */

LVNode* node_reserve(LVArena* arena, const LVLevel8_t level, const LVKeyLen32_t key_len, const LVValueLen32_t value_len, const LVSize32_t field_size) {
    const LVSize32_t size_to_reserve = sizeof(LVNode) + level * sizeof(LVNode*) + key_len + value_len + field_size;
    return (LVNode*)arena_allocate(arena, size_to_reserve, -1);
}

LVSize32_t node_key_offset(const LVLevel8_t level)
{
    return sizeof(LVNode) + sizeof(LVNode*) * level;
}

LVSize32_t node_value_offset(const LVLevel8_t level, const LVKeyLen32_t klen)
{
    return node_key_offset(level) + klen;
}

LVSize32_t node_field_offset(const LVLevel8_t level, const LVKeyLen32_t klen, const LVValueLen32_t vlen)
{
    return node_value_offset(level, klen) + vlen;
}

/*
 * Total order over nodes. Returns <0 if a<b, >0 if a>b (never 0 for two DATA
 * nodes — see below).
 *
 * For two DATA nodes:
 *   1. Compare keys lexicographically (memcmp over the shorter length).
 *   2. If one key is a prefix of the other, the shorter key is smaller.
 *   3. If keys are fully EQUAL, order by seq DESCENDING — the HIGHER seq
 *      (newer version) sorts FIRST. This is the core versioning rule: newest
 *      wins. table_search relies on it (first match = latest), and SST merge
 *      relies on it (memtable's newer seq beats the old SST copy).
 *
 * Two DATA nodes never compare equal: keys can tie, but seqs are unique per
 * key, so there's always a tiebreak. (That's why the equal-key branch returns
 * only -1/1, never 0.)
 *
 * Sentinels: HEAD sorts before everything, TAIL after everything — so skip-list
 * boundaries need no special-casing at call sites.
 */

int node_cmp(const LVNodeType type_a, const void* key_a, const LVKeyLen32_t klen_a, const LVSeq64_t seq_a, const LVNodeType type_b, const void* key_b, const LVKeyLen32_t klen_b, const LVSeq64_t seq_b)
{
    if (type_a == LV_NODE_DATA && type_b == LV_NODE_DATA)
    {
        const LVKeyLen32_t min_len = klen_a < klen_b ? klen_a : klen_b;

        const int mcmp_result = memcmp(key_a, key_b, min_len);

        if (mcmp_result != 0)
        {
            return mcmp_result;
        }

        if (klen_a < klen_b)
        {
            return -1;
        }

        else if (klen_a > klen_b)
        {
            return 1;
        }

        else
        { // keys equal -> newer (bigger) seq comes FIRST. This ordering is the
          // basis of the whole versioning scheme (see header/table_search).
            if (seq_a > seq_b)
            {
                return -1;
            }

            else 
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

            else //type_b == LV_NODE_TAIL
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
    if (node->value_len <= 0) return NULL;
    return (char*)node + node_value_offset(node->level, node->key_len);
}

void* node_access_field(const LVNode* node, const LVCount32_t number)
{
    if (node->field_count <= 0 || number < 0 || number > node->field_count - 1) return NULL;
    char* field = (char*)node + node_field_offset(node->level, node->key_len, node->value_len);
    return node_field_buffer_access(field, number);
}

void* node_field_buffer_access(const void* field_buffer, const LVCount32_t field_number) {
    char* field = (char*)field_buffer;
    for (LVCount32_t i = 0; i < field_number; ++i)
    {
        uint8_t saved_type;
        memcpy(&saved_type, field, sizeof(uint8_t));
        field += sizeof(uint8_t);

        LVMetaType type = (LVMetaType)saved_type;

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


double node_get_f64_field(const LVNode* node, const LVFieldMask32_t mask) {
    int number = node_field_number_of_mask(node->field_mask, mask);
    char* double_field = (char*)node_access_field(node, number);
    double_field += sizeof(uint8_t);
    double value = 0.0;
    memcpy(&value, double_field, sizeof(double));
    return value;
}
int64_t node_get_int64_field(const LVNode* node, const LVFieldMask32_t mask) {
    int number = node_field_number_of_mask(node->field_mask, mask);
    char* int64_field = (char*)node_access_field(node, number);
    int64_field += sizeof(uint8_t);
    int64_t value = 0;
    memcpy(&value, int64_field, sizeof(int64_t));
    return value;
}

int node_field_number_of_mask(const LVFieldMask32_t node_field_mask, const LVFieldMask32_t target_mask){
    if (!(node_field_mask & target_mask)) return -1;

    int number = 0;
    for(LVCount32_t i=0; i<LV_MAX_META_FIELDS - 1; ++i){
        LVFieldMask32_t mask  = (1u << i);
        if(mask == target_mask){
            return number;
        }

        if(node_field_mask & mask){
            ++number;
        }
    }

    return -1;
}

LVFieldMask32_t node_field_number_to_mask(const LVFieldMask32_t field_mask, const LVCount32_t number) {
    if (number < 0 ) return 0;
    LVCount32_t count = 0;
    for (LVCount32_t bit = 0; bit < LV_MAX_META_FIELDS; ++bit) {
        LVFieldMask32_t mask = 1u << bit;
        if (field_mask & mask) {
            if (count == number) return mask;
            ++count;
        }
    }
    return 0;
}



int node_eval_query(const LVNode* node, const LVAstNode* query, const LVSchema* schema) {
    return query_eval_ast(query, node, schema);
}

LVSize32_t node_field_size(const LVNode* node) {
    if (node->field_count <= 0) return 0;
    char* field_ptr = (char*)node_access_field(node, 0);
    LVSize32_t size = 0;
    for (LVCount32_t i = 0; i < node->field_count; ++i) {
        LVMetaType saved_type;
        memcpy(&saved_type, field_ptr, sizeof(uint8_t));

        size += 1;

        field_ptr += sizeof(uint8_t);

        LVMetaType type = (LVMetaType)saved_type;

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

void node_link_hnsw_node(LVNode* node, const LVHnswNode* hnsw_node){
    node->hnsw_node = hnsw_node;
}
