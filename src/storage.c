#include <stdlib.h>
#include <string.h>
#include "storage.h"
#include "arena.h"
#include "node.h"
#include "helper.h"
#include "query.h"
#include "schema.h"
#include "vector.h"

/*
 * Create an empty memtable: an arena, plus head and tail sentinel nodes linked
 * at every level (head.levels[i] = tail for all i). Sentinels remove edge cases
 * from insert/search — there is always a node to the left (head) and right
 * (tail), so no NULL checks at the boundaries. Returns NULL on OOM (partial
 * state is cleaned up via table_destroy, which tolerates NULLs).
 */

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

/*
 * Destroy the whole memtable in one shot: freeing the arena frees every node
 * at once (nodes are arena-allocated, never individually freed — see arena.h).
 */

void table_destroy(LVMemTable* table) {
    if (table) {
        arena_destroy(table->arena);
        free(table);
    }
}


/*
 * Insert a new version node, preserving the skip list's order
 * (key-ascending, then seq-descending among equal keys).
 *
 * `level` (the node's tower height) is chosen by the CALLER, in
 * lv_put_internal, via randomized coin-flips (P = LV_SKIPLIST_P). It is passed
 * in rather than generated here so that insert stays a pure structural op.
 *
 * Standard skip-list splice:
 *   1. Walk top-down; at each level record in update[] the rightmost node whose
 *      key < the new key (node_cmp < 0). These are the new node's predecessors.
 *   2. If the new node is taller than the table's current height, point the
 *      extra levels' predecessors at head and raise current_level.
 *   3. Link the new node in at levels 1..level.
 *
 * Note we never overwrite an existing key: an update or delete is just another
 * node with a higher seq, which sorts BEFORE the old versions of that key.
 */

LVNode* table_insert(LVMemTable* table, const LVNodeOp op, const LVSeq64_t seq, const LVLevel8_t level, const LVSize32_t key_len, const void* key, const LVSize32_t value_len, const void* value, const LVVectorId64_t vector_id, const LVFieldMask32_t field_mask, const LVCount32_t field_count, const LVSize32_t field_size, const void* field_buffer)
{
    LVNode* result = NULL;
    LVNode* update[LV_SKIPLIST_MAX_LEVEL];
    memset(update, 0, sizeof(LVNode*) * LV_SKIPLIST_MAX_LEVEL);

    LVLevel8_t current_update_level_index = table->current_level - 1; //levels start at 1, so index must be level - 1.
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


/*
 * Same skip-list splice as table_insert, but takes an already-built node
 * instead of constructing one from fields.
 *
 * WHY it exists: WAL recovery. On reopen we read serialized records from the
 * WAL, rebuild each into an LVNode, and re-insert it here to reconstruct the
 * exact memtable state that existed before the crash/close. The node already
 * carries its seq and level, so we just splice it in.
 */

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



/*
 * Return the LATEST version of `key`, or NULL if the key isn't present.
 *
 * The probe uses seq = UINT64_MAX. Since equal keys are ordered seq-DESCENDING,
 * a max-seq probe compares as "newer than every stored version", so the search
 * settles immediately to the left of the newest node for `key`.
 * The caller inspects the returned node's op: a DELETE tombstone means the key
 * is logically absent.
 * */

LVNode* table_search(const LVMemTable* table, const void* key, const LVKeyLen32_t key_len)
{
    LVNode* result = NULL;
    const LVSeq64_t seq_for_search = UINT64_MAX;
    LVLevel8_t current_level_index = table->current_level - 1;
    LVNode* current_candidate = table->head;
    LVNode* current_cmp_node = current_candidate->levels[current_level_index];


    while (current_level_index >= 0)
    {
        while (node_cmp(current_cmp_node->type, node_access_key(current_cmp_node), current_cmp_node->key_len, current_cmp_node->seq, LV_NODE_DATA, key, key_len, seq_for_search) < 0)
        {

            current_candidate = current_cmp_node;
            current_cmp_node = current_candidate->levels[current_level_index];
        }
        if (current_level_index == 0) break;
        --current_level_index;
        current_cmp_node = current_candidate->levels[current_level_index];
    }


    /*
     * WHY we must descend all the way to level-index 0 before testing the key:
 *   A key may have several versions (e.g. a PUT then a DELETE) sitting at
 *   different tower heights. Tower heights are RANDOM, so if we returned at the
 *   first level where a key matched, we could hand back an OLDER version that
 *   happens to reach a tall level, while the newest version lives lower down.
 *   Only the bottom level 1(index = 0) links every node in full order. So we descend to the bottom level,
 *   stop, and take the first node there whose key matches — guaranteed newest.
 *
 *   This was a real, non-deterministic bug: returning mid-descent sometimes
 *   surfaced a stale pre-delete version, seemingly at random (it depended on
 *   the random tower heights of that run). Do NOT "optimize" this by returning
 *   early on a key match above level-index 0.
    */

    if (node_key_equal(node_access_key(current_cmp_node), current_cmp_node->key_len, key, key_len))
    {
        result = current_cmp_node;
    }

    return result;
}


/*
 * Linear scan over bottom level (every record, in order), emitting one row per key
 * at its newest version into qv_set via qv_append_fn. table_get_next_node
 * handles the "newest version per key" part by skipping older duplicates.
 *
 * Two cases per key:
 *   - Newest version is a DELETE tombstone: we STILL emit it (is_tombstone=1),
 *     even though it has no value. WHY: the same key may have an older,
 *     still-live record in an SST on disk. The query layer merges memtable +
 *     SST results; emitting the tombstone here tells the merge to suppress that
 *     on-disk version. Without it, a key deleted in memory would "resurrect"
 *     from the SST. (Do not drop this branch as "why emit a deleted row".)
 *   - Newest version is live: if it passes the field-mask prefilter and the
 *     query AST, emit it with its order-by value extracted per ordbytype.
 */

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


/*
 * Advance to the newest version of the NEXT distinct key.
 *
 * Level 1 holds every version of every key; equal keys are seq-descending, so
 * the first node of a key is its newest. A per-key scan must therefore skip the
 * current key's OLDER versions. From `current` we walk level 1 until the key
 * changes, landing on the newest node of the next key.
 *
 * head -> first data node; tail -> tail (terminal). This skipping is what lets
 * table_query_filter_scan see each key exactly once, at its latest version.
 */

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


