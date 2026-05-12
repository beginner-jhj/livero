#include <stdlib.h>
#include <string.h>
#include "storage.h"
#include "arena.h"
#include "node.h"
#include "helper.h"

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

    Arena *arena_temp = create_arena(BLOCK_DEFAULT_SIZE);

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
    }

    table->current_level = 1;

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
