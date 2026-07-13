#include <stdlib.h>
#include <string.h>
#include "storage.h"
#include "arena.h"
#include "node.h"
#include "helper.h"
#include "query.h"
#include "schema.h"
#include "vector.h"

LVMemTable* table_create()
{
    LVMemTable* table = malloc(sizeof(LVMemTable));
    if (!table) goto cleanup;

    table->arena = NULL;
    table->head = NULL;
    table->tail = NULL;
    table->current_level = 1;
    table->node_count = 0;

    table->arena = arena_create(LV_DEFAULT_BLOCK_SIZE);
    if (!table->arena) goto cleanup;

    table->head = node_create(table->arena, LV_NODE_HEAD, 0, LV_PUT,
        LV_SKIPLIST_MAX_LEVEL, 0, NULL, 0, NULL, 0, 0, 0, 0, NULL);
    if (!table->head) goto cleanup;

    table->tail = node_create(table->arena, LV_NODE_TAIL, 0, LV_PUT,
        LV_SKIPLIST_MAX_LEVEL, 0, NULL, 0, NULL, 0, 0, 0, 0, NULL);
    if (!table->tail) goto cleanup;

    for (int i = 0; i < LV_SKIPLIST_MAX_LEVEL; ++i)
    {
        table->head->levels[i] = table->tail;
        table->tail->levels[i] = NULL;
    }

    return table;

cleanup:
    table_destroy(table);
    return NULL;
}

void table_destroy(LVMemTable* table) {
    if (table) {
        arena_destroy(table->arena);
        free(table);
    }
}

LVNode* table_insert(LVMemTable* table, const LVNodeOp op, const LVSeq64_t seq, const LVLevel8_t level, const LVSize32_t key_len, const void* key, const LVSize32_t value_len, const void* value, const LVVectorId64_t vector_id, const LVFieldMask32_t field_mask, const LVCount32_t field_count, const LVSize32_t field_size, const void* field_buffer)
{
    LVNode* result = NULL;
    LVNode* update[LV_SKIPLIST_MAX_LEVEL];
    memset(update, 0, sizeof(LVNode*) * LV_SKIPLIST_MAX_LEVEL);

    LVLevel8_t current_update_level_index = table->current_level - 1;
    LVNode* current_head = table->head;
    LVNode* current_cmp_node = current_head->levels[current_update_level_index];

    while (current_update_level_index >= 0)
    {
        while (node_cmp(current_cmp_node->type, node_access_key(current_cmp_node), current_cmp_node->key_len, current_cmp_node->seq, LV_NODE_DATA, key, key_len, seq) < 0)
        {
            current_head = current_cmp_node;
            current_cmp_node = current_head->levels[current_update_level_index];
        }

        update[current_update_level_index] = current_head;
        if (current_update_level_index == 0)
            break;
        --current_update_level_index;
        current_cmp_node = current_head->levels[current_update_level_index];
    }

    LVNode* new_node = node_create(table->arena, LV_NODE_DATA, seq, op, level, key_len, key, value_len, value, vector_id, field_mask, field_count, field_size, field_buffer);
    if (!new_node)
    {
        result = NULL;
        goto _return;
    }

    if (level > table->current_level)
    {
        for (int i = table->current_level; i < level; ++i)
        {
            update[i] = table->head;
        }
        table->current_level = level;
    }

    for (int i = 0; i < level; ++i)
    {
        new_node->levels[i] = update[i]->levels[i];
        update[i]->levels[i] = new_node;
    }

    table->node_count += 1;

    result = new_node;
_return:
    return result;
}

void table_direct_insert(LVMemTable* table, LVNode* node)
{
    LVNode* update[LV_SKIPLIST_MAX_LEVEL];
    memset(update, 0, sizeof(LVNode*) * LV_SKIPLIST_MAX_LEVEL);

    LVLevel8_t current_update_level_index = table->current_level - 1;
    LVNode* current_head = table->head;
    LVNode* current_cmp_node = current_head->levels[current_update_level_index];

    while (current_update_level_index >= 0)
    {
        while (node_cmp(current_cmp_node->type, node_access_key(current_cmp_node), current_cmp_node->key_len, current_cmp_node->seq, node->type, node_access_key(node), node->key_len, node->seq) < 0)
        {
            current_head = current_cmp_node;
            current_cmp_node = current_head->levels[current_update_level_index];
        }

        update[current_update_level_index] = current_head;
        if (current_update_level_index == 0)
            break;
        --current_update_level_index;
        current_cmp_node = current_head->levels[current_update_level_index];
    }

    if (node->level > table->current_level)
    {
        for (int i = table->current_level; i < node->level; ++i)
        {
            update[i] = table->head;
        }
        table->current_level = node->level;
    }

    for (int i = 0; i < node->level; ++i)
    {
        node->levels[i] = update[i]->levels[i];
        update[i]->levels[i] = node;
    }

    table->node_count += 1;
}

LVNode* table_search(const LVMemTable* table, const void* key, const LVKeyLen32_t key_len)
{
    LVNode* result = NULL;
    const LVSeq64_t seq_for_search = UINT64_MAX;
    LVLevel8_t current_level = table->current_level - 1;
    LVNode* current_candidate = table->head;
    LVNode* current_cmp_node = current_candidate->levels[current_level];


    while (current_level >= 0)
    {
        while (node_cmp(current_cmp_node->type, node_access_key(current_cmp_node), current_cmp_node->key_len, current_cmp_node->seq, LV_NODE_DATA, key, key_len, seq_for_search) < 0)
        {

            current_candidate = current_cmp_node;
            current_cmp_node = current_candidate->levels[current_level];
        }
        if (current_level == 0) break;
        --current_level;
        current_cmp_node = current_candidate->levels[current_level];
    }

    if (node_key_equal(node_access_key(current_cmp_node), current_cmp_node->key_len, key, key_len))
    {
        result = current_cmp_node;
    }

    return result;
}

LVStatus table_query_filter_scan(const LVMemTable* table, const LVSchema* schema,
    const LVAstNode* query, const LVFieldMask32_t query_field_mask,
    LVOrdbyType ordbytype, const LVFieldMask32_t ordby_field_mask,
    const LVQVSetAppendFn qv_append_fn, LVQVSet* qv_set)
{
    LVStatus result = LV_OK;
    LVNode* current_node = table->head->levels[0];

    while (current_node->type != LV_NODE_TAIL)
    {
        if (current_node->op == LV_DELETE) {
            LVOrdbyValue ordbyvalue;
            ordbyvalue.i64 = 0;
            if ((result = qv_append_fn(qv_set, current_node->seq,
                LV_NO_VECTOR_ID,
                node_access_key(current_node), current_node->key_len,
                NULL, 0,
                0.0f, ordbyvalue,
                1)) != LV_OK) {
                return result;
            }
            current_node = table_get_next_node(current_node);
            continue;
        }
        if (query_field_mask & current_node->field_mask) {
            if (query_eval_ast(query, current_node, schema)) {
                float vector_score = 0.0f;
                LVOrdbyValue ordbyvalue;
                ordbyvalue.i64 = 0;

                switch (ordbytype) {
                case LV_ORDBY_FLOAT:
                    ordbyvalue.f64 = node_get_f64_field(current_node, ordby_field_mask);
                    break;
                case LV_ORDBY_INT:
                    ordbyvalue.i64 = node_get_int64_field(current_node, ordby_field_mask);
                    break;
                case LV_ORDBY_VEC:
                    ordbyvalue.score = 0.0f;
                    break;
                default:
                    break;
                }

                if ((result = qv_append_fn(qv_set, current_node->seq,
                    current_node->vector_id, node_access_key(current_node), current_node->key_len,
                    node_access_value(current_node), current_node->value_len,
                    vector_score, ordbyvalue, 0)) != LV_OK) {
                    return result;
                }
            }
        }

        current_node = table_get_next_node(current_node);
    }

    return result;
}

LVNode* table_get_next_node(const LVNode* current) {
    if (current->type == LV_NODE_TAIL) return current;
    else if (current->type == LV_NODE_HEAD) return current->levels[0];
    const void* key = node_access_key(current);
    const LVKeyLen32_t key_len = current->key_len;

    LVNode* next = current->levels[0];
    while (next->type != LV_NODE_TAIL &&
        node_key_equal(node_access_key(next), next->key_len, key, key_len)) {
        next = next->levels[0];
    }
    return next;
}


