#include <stdlib.h>
#include <string.h>
#include "storage.h"
#include "arena.h"
#include "node.h"
#include "helper.h"
#include "query.h"
#include "schema.h"
#include "vector.h"

LVMemTable* create_table(const LVSeq64_t seq)
{
    int flag = 0;
    LVNode* head = NULL;
    LVNode* tail = NULL;
    LVArena* arena = NULL;
    LVMemTable* table = NULL;

    LVMemTable* table_temp = malloc(sizeof(LVMemTable));

    if (!table_temp)
    {
        flag = 1;
        goto cleanup;
    }

    table = table_temp;

    LVArena* arena_temp = arena_create(LV_DEFAULT_BLOCK_SIZE);

    if (!arena_temp)
    {
        flag = 1;
        goto cleanup;
    }

    arena = arena_temp;

    table->arena = arena;

    LVNode* head_temp = node_create(table->arena, LV_NODE_HEAD, seq, LV_PUT, LV_SKIPLIST_MAX_LEVEL, 0, NULL, 0, NULL, 0, 0, 0, 0, NULL);

    if (!head_temp)
    {
        flag = 1;
        goto cleanup;
    }

    head = head_temp;

    LVNode* tail_temp = node_create(table->arena, LV_NODE_TAIL, seq, LV_PUT, LV_SKIPLIST_MAX_LEVEL, 0, NULL, 0, NULL, 0, 0, 0, 0, NULL);

    if (!tail_temp)
    {
        flag = 1;
        goto cleanup;
    }

    tail = tail_temp;

    table->head = head;
    table->tail = tail;

    for (int i = 0; i < LV_SKIPLIST_MAX_LEVEL; ++i)
    {
        table->head->levels[i] = table->tail;
        table->tail->levels[i] = NULL;
    }

    table->current_level = 1;
    table->node_count = 0;

cleanup:
    if (flag)
    {
        safe_free(&arena);
        safe_free(&table);
    }

    return table;
}

void destroy_table(LVMemTable* table) {
    if (table) {
        arena_destroy(table->arena);
        free(table);
    }
}

LVNode* table_insert(LVMemTable* table, const LVNodeOp op, const LVSeq64_t seq, const LVLevel8_t level, const LVSize32_t key_len, const void* key, const LVSize32_t value_len, const void* value, const uint64_t vector_id, const uint32_t field_mask, const uint32_t field_count, const LVSize32_t field_size, const LVMetaField* field_list)
{
    LVNode* result = NULL;
    LVNode* update[LV_SKIPLIST_MAX_LEVEL];
    memset(update, 0, sizeof(LVNode*) * LV_SKIPLIST_MAX_LEVEL);

    LVLevel8_t current_update_level = table->current_level - 1;
    LVNode* current_head = table->head;
    LVNode* current_cmp_node = current_head->levels[current_update_level];

    while (current_update_level >= 0)
    {
        while (node_cmp(current_cmp_node->type, node_access_key(current_cmp_node), current_cmp_node->key_len, current_cmp_node->seq, LV_NODE_DATA, key, key_len, seq) < 0)
        {
            current_head = current_cmp_node;
            current_cmp_node = current_head->levels[current_update_level];
        }

        update[current_update_level] = current_head;
        if (current_update_level == 0)
            break;
        --current_update_level;
        current_cmp_node = current_head->levels[current_update_level];
    }

    LVNode* new_node = node_create(table->arena, LV_NODE_DATA, seq, op, level, key_len, key, value_len, value, vector_id, field_mask, field_count, field_size, field_list);
    if (!new_node)
    {
        result = NULL;
        goto _return;
    }

    if (level > table->current_level)
    {
        for (int i = table->current_level - 1; i < level; ++i)
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

    LVLevel8_t current_update_level = table->current_level - 1;
    LVNode* current_head = table->head;
    LVNode* current_cmp_node = current_head->levels[current_update_level];

    while (current_update_level >= 0)
    {
        while (node_cmp(current_cmp_node->type, node_access_key(current_cmp_node), current_cmp_node->key_len, current_cmp_node->seq, node->type, node_access_key(node), node->key_len, node->seq) < 0)
        {
            current_head = current_cmp_node;
            current_cmp_node = current_head->levels[current_update_level];
        }

        update[current_update_level] = current_head;
        if (current_update_level == 0)
            break;
        --current_update_level;
        current_cmp_node = current_head->levels[current_update_level];
    }

    if (node->level > table->current_level)
    {
        for (int i = table->current_level - 1; i < node->level; ++i)
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
        if (node_key_equal(node_access_key(current_cmp_node), current_cmp_node->key_len, key, key_len))
        {
            if (current_cmp_node->op != LV_DELETE)
            {
                result = current_cmp_node;
                goto _return;
            }
            else
            {
                break;
            }
        }

        if (current_level == 0)
            break;
        --current_level;
        current_cmp_node = current_candidate->levels[current_level];
    }

_return:
    return result;
}

LVStatus table_query_filter_scan(const LVMemTable* table, const LVSchema* schema, const LVAstNode* query, const LVSize32_t query_field_mask,  LVOrdbyType ordbytype, const LVSize32_t ordby_field_mask, const LVQVSetAppendFn qv_append_fn, LVQVSet* qv_set)
{
    LVStatus result = LV_OK;
    LVNode* current_node = table->head->levels[0];

    while (current_node->type != LV_NODE_TAIL)
    {
        if ((query_field_mask & current_node->field_mask) && current_node->op != LV_DELETE)
        {
            if (query_eval_ast(query, current_node, schema))
            {
                float vector_score = 0.0f;
                LVOrdbyValue ordbyvalue;
                ordbyvalue.i64 = 0;

                switch (ordbytype)
                {
                case LV_ORDBY_FLOAT: {
                    double value = node_get_f64_field(current_node, ordby_field_mask);
                    ordbyvalue.f64 = value;
                    break;
                }
                case LV_ORDBY_INT:{
                    int64_t value = node_get_i64_field(current_node, ordby_field_mask);
                    ordbyvalue.i64 = value;
                    break;
                }
                case LV_ORDBY_VEC:{
                    ordbyvalue.score = 0.0f;
                    break;
                }

                default:
                    break;
                }

                //append to qv_set
                if ((result = qv_append_fn(qv_set, current_node->seq, current_node->vector_id, node_access_key(current_node), current_node->key_len, node_access_value(current_node), current_node->value_len, vector_score, ordbyvalue)) != LV_OK) return result;

            }
        }
        current_node = current_node->levels[0];
    }

    return result;
}


