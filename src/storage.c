#include <stdlib.h>
#include <string.h>
#include "storage.h"
#include "arena.h"
#include "node.h"
#include "helper.h"
#include "query.h"

LVMemTable *create_table(const LVSeq64_t seq)
{
    int flag = 0;
    Node *head = NULL;
    Node *tail = NULL;
    Arena *arena = NULL;
    LVMemTable *table = NULL;

    LVMemTable *table_temp = malloc(sizeof(LVMemTable));

    if (!table_temp)
    {
        flag = 1;
        goto cleanup;
    }

    table = table_temp;

    Arena *arena_temp = create_arena(LV_DEFAULT_BLOCK_SIZE);

    if (!arena_temp)
    {
        flag = 1;
        goto cleanup;
    }

    arena = arena_temp;

    table->arena = arena;

    Node *head_temp = create_node(table->arena, LV_NODE_HEAD, seq, LV_PUT, LV_SKIPLIST_MAX_LEVEL, 0, NULL, 0, NULL, 0, 0, 0, 0, NULL);

    if (!head_temp)
    {
        flag = 1;
        goto cleanup;
    }

    head = head_temp;

    Node *tail_temp = create_node(table->arena, LV_NODE_TAIL, seq, LV_PUT, LV_SKIPLIST_MAX_LEVEL, 0, NULL, 0, NULL, 0, 0, 0, 0, NULL);

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

LVStatus table_insert(LVMemTable *table, const LVInsertOp op, const LVSeq64_t seq, const LVLevel8_t level, const LVSize32_t key_len, const void *key, const LVSize32_t value_len, const void *value, const uint64_t vector_id, const uint32_t field_mask, const uint32_t field_count, const LVSize32_t field_size, const LVMetaField *field_list)
{
    LVStatus result = LV_OK;
    Node *update[LV_SKIPLIST_MAX_LEVEL];
    memset(update, 0, sizeof(Node *) * LV_SKIPLIST_MAX_LEVEL);

    LVLevel8_t current_update_level = table->current_level - 1;
    Node *current_head = table->head;
    Node *current_cmp_node = current_head->levels[current_update_level];

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

    Node *new_node = create_node(table->arena, LV_NODE_DATA, seq, op, level, key_len, key, value_len, value, vector_id, field_mask, field_count, field_size, field_list);
    if (!new_node)
    {
        result = LV_ERR_FULL;
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

_return:
    return result;
}

void table_direct_insert(LVMemTable *table, Node *node)
{
    Node *update[LV_SKIPLIST_MAX_LEVEL];
    memset(update, 0, sizeof(Node *) * LV_SKIPLIST_MAX_LEVEL);

    LVLevel8_t current_update_level = table->current_level - 1;
    Node *current_head = table->head;
    Node *current_cmp_node = current_head->levels[current_update_level];

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

Node *table_search(const LVMemTable *table, const void *key, const LVKeyLen32_t key_len)
{
    Node *result = NULL;
    const LVSeq64_t seq_for_search = UINT64_MAX;
    LVLevel8_t current_level = table->current_level - 1;
    Node *current_candidate = table->head;
    Node *current_cmp_node = current_candidate->levels[current_level];

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

LVResultSet *table_query(const LVMemTable *table, const LVSchema *schema, const LVAstNode *query, const LVSize32_t field_mask)
{

    LVResultSet *result_set = create_result_set();
    if (!result_set)
    {
        return NULL;
    }

    if (table->node_count == 0)
    {
        return result_set;
    }

    Node *current_node = table->head->levels[0];
    while (current_node->type != LV_NODE_TAIL)
    {
        if ((field_mask & current_node->field_mask))
        {
            if (query_eval_ast(query, current_node, schema))
            {
                if (table_result_set_append(result_set, current_node) != LV_OK)
                {
                    destroy_result_set(result_set);
                    return NULL;
                }
            }
        }
        current_node = current_node->levels[0];
    }

    return result_set;
}

LVResultSet *create_result_set(void)
{
    int flag = 0;
    LVResultSet *result_set = NULL;

    LVResultSet *result_set_tmp = malloc(sizeof(LVResultSet));

    if (!result_set_tmp)
    {
        return NULL;
    }

    Node **nodes = malloc(sizeof(Node *) * LV_DEFAULT_CAPACITY);
    if (!nodes)
    {
        flag = 1;
        goto cleanup;
    }

    result_set = result_set_tmp;
    result_set->capacity = LV_DEFAULT_CAPACITY;
    result_set->size = 0;
    result_set->nodes = nodes;

cleanup:
    if (flag)
    {
        safe_free(&result_set);
    }

    return result_set;
}

LVStatus table_result_set_append(LVResultSet *result_set, const Node *node)
{
    if (result_set->size >= result_set->capacity)
    {
        LVSize32_t new_capacity = result_set->capacity * 2;
        Node **nodes_tmp = realloc(result_set->nodes, new_capacity * sizeof(Node *));
        if (!nodes_tmp)
        {
            return LV_ERR_FULL;
        }
        result_set->capacity = new_capacity;
        result_set->nodes = nodes_tmp;
    }

    result_set->nodes[result_set->size] = node;
    result_set->size += 1;
    return LV_OK;
}

void destroy_result_set(LVResultSet *result_set)
{
    if (result_set)
    {
        free(result_set->nodes);
        free(result_set);
    }
}
