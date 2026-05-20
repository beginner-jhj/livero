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

    LVArena* arena_temp = create_arena(LV_DEFAULT_BLOCK_SIZE);

    if (!arena_temp)
    {
        flag = 1;
        goto cleanup;
    }

    arena = arena_temp;

    table->arena = arena;

    LVNode* head_temp = create_node(table->arena, LV_NODE_HEAD, seq, LV_PUT, LV_SKIPLIST_MAX_LEVEL, 0, NULL, 0, NULL, 0, 0, 0, 0, NULL);

    if (!head_temp)
    {
        flag = 1;
        goto cleanup;
    }

    head = head_temp;

    LVNode* tail_temp = create_node(table->arena, LV_NODE_TAIL, seq, LV_PUT, LV_SKIPLIST_MAX_LEVEL, 0, NULL, 0, NULL, 0, 0, 0, 0, NULL);

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

LVStatus table_insert(LVMemTable* table, const LVNodeOp op, const LVSeq64_t seq, const LVLevel8_t level, const LVSize32_t key_len, const void* key, const LVSize32_t value_len, const void* value, const uint64_t vector_id, const uint32_t field_mask, const uint32_t field_count, const LVSize32_t field_size, const LVMetaField* field_list)
{
    LVStatus result = LV_OK;
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

    LVNode* new_node = create_node(table->arena, LV_NODE_DATA, seq, op, level, key_len, key, value_len, value, vector_id, field_mask, field_count, field_size, field_list);
    if (!new_node)
    {
        result = LV_ERR_OOM;
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

LVTableQueryResultSet* table_query(const LVMemTable* table, const LVSchema* schema, const LVAstNode* query, const void* query_vector, const LVDim32_t dim, const LVHnswIDMap* id_vector_map, const LVSize32_t field_mask, const LVQueryOption* option)
{

    int flag = 0;
    LVTableQueryResultSet* result = NULL;
    result = malloc(sizeof(LVTableQueryResultSet));
    if (!result) {
        goto cleanup;
    }
    result->size = 0;
    result->results = NULL;
    LVTableQVList* qv_list = create_qv_list();
    if (!qv_list)
    {
        flag = 1;
        goto cleanup;
    }

    LVNode* current_node = table->head->levels[0];

    const int has_option = option && !(option->flags & LV_QOPT_NONE);

    const int is_limit_on = has_option && option->flags & LV_QOPT_LIMIT;
    const int is_range_on = has_option && query_vector && option->flags & LV_QOPT_VECTOR_RANGE;
    const int is_ordby_on = has_option && option->flags & LV_QOPT_ORDER_BY;
    const int is_ordby_vec = is_ordby_on && query_vector && (strncasecmp(option->order.by, "vector", strlen("vector")) == 0);
    const int needs_calc_dis = is_range_on || is_ordby_vec;

    uint32_t ordby_field_mask = 0;

    if (is_ordby_on) {
        const LVMetaFieldHash* hash = schema_search_field_hash(schema->field_hashes, option->order.by, strlen(option->order.by));
        if (hash->type != LV_META_STRING) {
            ordby_field_mask = hash->mask;
        }
    }

    while (current_node->type != LV_NODE_TAIL)
    {
        if ((field_mask & current_node->field_mask) && current_node->op != LV_DELETE)
        {
            if (query_eval_ast(query, current_node, schema))
            {
                LVVectorDisValue dis;
                if (current_node->vector_id != LV_NO_VECTOR_ID && needs_calc_dis)
                {
                    if (schema->vector_type == LV_VEC_FLOAT32)
                    {
                        dis.f32 = option->vector_metric == LV_METRIC_L2 ? vector_f32_l2_sq((float*)query_vector, (float*)(id_vector_map->map[current_node->vector_id]), dim) : vector_f32_dot((float*)query_vector, (float*)(id_vector_map->map[current_node->vector_id]), dim);
                    }
                    else
                    {
                        dis.i32 = option->vector_metric == LV_METRIC_L2 ? vector_i8_l2_sq((int8_t*)query_vector, (int8_t*)(id_vector_map->map[current_node->vector_id]), dim) : vector_i8_dot((int8_t*)query_vector, (int8_t*)(id_vector_map->map[current_node->vector_id]), dim);
                    }
                }
                else
                {
                    dis.i32 = -1;
                }

                if (table_qv_list_append(qv_list, current_node, dis, ordby_field_mask) != LV_OK)
                {
                    flag = 1;
                    goto cleanup;
                }
            }
        }
        current_node = current_node->levels[0];
    }

    if (is_range_on)
    {
        table_query_apply_range(qv_list, schema->vector_type, option);
    }

    if (is_ordby_on && ordby_field_mask != 0)
    {
        table_query_apply_ordby(qv_list, option, schema);
    }

    if (is_limit_on) {
        table_query_apply_limit(qv_list, option->limit);
    }

    LVTableQueryResult* results_tmp = malloc(sizeof(LVTableQueryResult) * qv_list->size);
    if (!results_tmp) {
        flag = 1;
        goto cleanup;
    }

    for (int i = 0; i < qv_list->size; ++i) {
        const LVNode* current_node = qv_list->values[i].node;
        results_tmp[i].node_seq = current_node->seq;
        results_tmp[i].vector_id = current_node->vector_id;
        results_tmp[i].node = current_node;
        results_tmp[i].key = node_access_key(current_node);
        results_tmp[i].key_len = current_node->key_len;
        results_tmp[i].value = node_access_value(current_node);
        results_tmp[i].value_len = current_node->value_len;
        results_tmp[i].vector = current_node->vector_id == LV_NO_VECTOR_ID ? NULL : id_vector_map->map[current_node->vector_id];
    }

    result->size = qv_list->size;
    result->results = results_tmp;

cleanup:
    if (flag) {
        safe_free(&result);
    }
    destroy_qv_list(qv_list);
    return result;
}

void table_query_apply_range(LVTableQVList* qv_list, const LVVectorType vector_type, const LVQueryOption* option)
{
    for (int i = 0; i < qv_list->size; ++i)
    {

        if (qv_list->values[i].dis.f32 < option->vector_range.left_endpoint || qv_list->values[i].dis.f32 > option->vector_range.right_endpoint)
        {
            qv_list->values[i].node = NULL;
        }
    }

    int left = 0;
    int right = qv_list->size - 1;

    while (left < right)
    {
        while (left < right && qv_list->values[left].node != NULL) left++;
        while (left < right && qv_list->values[right].node != NULL) right--;

        if (left < right)
        {
            LVTableQueryValue temp = qv_list->values[left];
            qv_list->values[left] = qv_list->values[right];
            qv_list->values[right] = temp;
        }
    }

    qv_list->size -= left;
}

void table_query_apply_ordby(LVTableQVList* qv_list, const LVQueryOption* option, const LVSchema* schema)
{
    const int is_range_on = option->flags & LV_QOPT_VECTOR_RANGE;
    const int is_ordby_vec = (strncasecmp(option->order.by, "vector", strlen("vector")) == 0);

    LVSize32_t set_size = qv_list->size;

    if (is_ordby_vec) {
        const int is_f32 = schema->vector_type == LV_VEC_FLOAT32;
        if (option->order.dir == LV_ORDER_ASC) {
            if (option->vector_metric == LV_METRIC_DOT) {
                qsort(qv_list->values, set_size, sizeof(LVTableQueryValue), is_f32 ? ordvec_f32_dot_nearest : ordvec_i8_dot_nearest);
            }
            else {
                qsort(qv_list->values, set_size, sizeof(LVTableQueryValue), is_f32 ? ordvec_f32_l2_nearest : ordvec_i8_l2_nearest);
            }
        }
        else {
            if (option->vector_metric == LV_METRIC_DOT) {
                qsort(qv_list->values, set_size, sizeof(LVTableQueryValue), is_f32 ? ordvec_f32_dot_farthest : ordvec_i8_dot_farthest);
            }
            else {
                qsort(qv_list->values, set_size, sizeof(LVTableQueryValue), is_f32 ? ordvec_f32_l2_farthest : ordvec_i8_l2_farthest);
            }
        }
    }
    else {
        const char* key = option->order.by;
        LVMetaFieldHash* hash = schema_search_field_hash(schema->field_hashes, key, strlen(key));

        if (hash->type == LV_META_STRING) return; //string can not be ordered

        else {
            const int is_float = hash->type == LV_META_FLOAT;

            if (option->order.dir == LV_ORDER_ASC) {
                qsort(qv_list->values, set_size, sizeof(LVTableQueryValue), is_float ? ordby_f64_asc : ordby_i64_asc);
            }
            else {
                qsort(qv_list->values, set_size, sizeof(LVTableQueryValue), is_float ? ordby_f64_desc : ordby_i64_desc);

            }

        }

    }
}

void table_query_apply_limit(LVTableQVList* qv_list, const LVSize32_t limit) {
    if (qv_list->size > limit) {
        qv_list->size = limit;
    }
}

int ordvec_f32_desc(const void* a, const void* b) {
    LVTableQueryValue* qva = a;
    LVTableQueryValue* qvb = b;
    return (qva->dis.f32 < qvb->dis.f32) - (qva->dis.f32 > qvb->dis.f32);
}

int ordvec_f32_asc(const void* a, const void* b) {
    LVTableQueryValue* qva = a;
    LVTableQueryValue* qvb = b;
    return (qva->dis.f32 > qvb->dis.f32) - (qva->dis.f32 < qvb->dis.f32);
}

int ordvec_i8_desc(const void* a, const void* b) {
    LVTableQueryValue* qva = a;
    LVTableQueryValue* qvb = b;
    return (qva->dis.i32 < qvb->dis.i32) - (qva->dis.i32 > qvb->dis.i32);
}

int ordvec_i8_asc(const void* a, const void* b) {
    LVTableQueryValue* qva = a;
    LVTableQueryValue* qvb = b;
    return (qva->dis.i32 > qvb->dis.i32) - (qva->dis.i32 < qvb->dis.i32);
}

int ordby_f64_asc(const void* a, const void* b) {
    LVTableQueryValue* qva = a;
    LVTableQueryValue* qvb = b;

    const int a_field_number = node_field_number(qva->node, qva->ordby_field_mask);
    const int b_field_number = node_field_number(qvb->node, qvb->ordby_field_mask);

    char* a_field = (char*)node_access_field(qva->node, a_field_number);
    char* b_field = (char*)node_access_field(qvb->node, b_field_number);

    a_field += sizeof(LVMetaType);
    b_field += sizeof(LVMetaType);

    double a_value = 0;
    double b_value = 0;

    memcpy(&a_value, a_field, sizeof(double));
    memcpy(&b_value, b_field, sizeof(double));

    return (a_value > b_value) - (a_value < b_value);
}

int ordby_f64_desc(const void* a, const void* b) {
    LVTableQueryValue* qva = a;
    LVTableQueryValue* qvb = b;

    const int a_field_number = node_field_number(qva->node, qva->ordby_field_mask);
    const int b_field_number = node_field_number(qvb->node, qvb->ordby_field_mask);

    char* a_field = (char*)node_access_field(qva->node, a_field_number);
    char* b_field = (char*)node_access_field(qvb->node, b_field_number);

    a_field += sizeof(LVMetaType);
    b_field += sizeof(LVMetaType);

    double a_value = 0;
    double b_value = 0;

    memcpy(&a_value, a_field, sizeof(double));
    memcpy(&b_value, b_field, sizeof(double));

    return (a_value < b_value) - (a_value > b_value);
}

int ordby_i64_asc(const void* a, const void* b) {
    LVTableQueryValue* qva = a;
    LVTableQueryValue* qvb = b;

    const int a_field_number = node_field_number(qva->node, qva->ordby_field_mask);
    const int b_field_number = node_field_number(qvb->node, qvb->ordby_field_mask);

    char* a_field = (char*)node_access_field(qva->node, a_field_number);
    char* b_field = (char*)node_access_field(qvb->node, b_field_number);

    a_field += sizeof(LVMetaType);
    b_field += sizeof(LVMetaType);

    int64_t a_value = 0;
    int64_t b_value = 0;

    memcpy(&a_value, a_field, sizeof(int64_t));
    memcpy(&b_value, b_field, sizeof(int64_t));

    return (a_value > b_value) - (a_value < b_value);
}

int ordby_i64_desc(const void* a, const void* b) {
    LVTableQueryValue* qva = a;
    LVTableQueryValue* qvb = b;

    const int a_field_number = node_field_number(qva->node, qva->ordby_field_mask);
    const int b_field_number = node_field_number(qvb->node, qvb->ordby_field_mask);

    char* a_field = (char*)node_access_field(qva->node, a_field_number);
    char* b_field = (char*)node_access_field(qvb->node, b_field_number);

    a_field += sizeof(LVMetaType);
    b_field += sizeof(LVMetaType);

    int64_t a_value = 0;
    int64_t b_value = 0;

    memcpy(&a_value, a_field, sizeof(int64_t));
    memcpy(&b_value, b_field, sizeof(int64_t));

    return (a_value < b_value) - (a_value > b_value);
}

LVTableQVList* create_qv_list(void)
{
    int flag = 0;
    LVTableQVList* qv_list = NULL;

    LVTableQVList* qv_list_tmp = malloc(sizeof(LVTableQVList));

    if (!qv_list_tmp)
    {
        return NULL;
    }

    LVTableQueryValue* values = malloc(sizeof(LVTableQueryValue) * LV_DEFAULT_CAPACITY);
    if (!values)
    {
        flag = 1;
        goto cleanup;
    }

    qv_list = qv_list_tmp;
    qv_list->capacity = LV_DEFAULT_CAPACITY;
    qv_list->size = 0;
    qv_list->values = values;

cleanup:
    if (flag)
    {
        safe_free(&qv_list);
    }

    return qv_list;
}

LVStatus table_qv_list_append(LVTableQVList* qv_list, const LVNode* node, const LVVectorDisValue dis, const uint32_t ordby_field_mask)
{
    if (qv_list->size >= qv_list->capacity)
    {
        LVSize32_t new_capacity = qv_list->capacity * 2;
        LVTableQueryValue* tmp = realloc(qv_list->values, new_capacity * sizeof(LVTableQueryValue));
        if (!tmp)
        {
            return LV_ERR_OOM;
        }
        qv_list->capacity = new_capacity;
        qv_list->values = tmp;
    }

    qv_list->values[qv_list->size].node = node;
    qv_list->values[qv_list->size].dis = dis;
    qv_list->values[qv_list->size].ordby_field_mask = ordby_field_mask;
    qv_list->size += 1;
    return LV_OK;
}

void destroy_qv_list(LVTableQVList* qv_list)
{
    if (qv_list)
    {
        free(qv_list->values);
        free(qv_list);
    }
}
