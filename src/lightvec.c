#include "lightvec.h"
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include "util.h"
#include "schema.h"
#include "wal.h"
#include "storage.h"
#include "vector.h"
#include "helper.h"
#include "query.h"
#include "node.h"
#include "sst.h"

struct LightVec
{
    char path[LV_PATH_MAX];
    LVSize32_t flush_threshold;
    int schema_fd;
    LVSchema* schema;


    // LSM-Tree memtable
    int wal_fd;
    LVMemTable* memtable;
    LVSeq64_t next_seq;
    int sst_fd;

    // Vector
    int vector_fd;                 // vectors.lv (O(1) access)
    int vector_index_fd; //vector_id to sst record offset
    LVBigCount64_t next_vector_id; //
    int hnsw_index_fd;             // hnsw_index.lv
    int hnsw_graph_fd;             // hnsw_graph.lv
    LVHnsw* hnsw;

    int32_t magic;
};

static LVStatus lv_check_db_corruption_internal(const LightVec* db);
static LVQVSet* lv_create_qvset_internal(void);
static LVStatus lv_flush_internal(LightVec* db);
static LVStatus lv_qvset_append_internal(LVQVSet* qvset, const LVSeq64_t node_seq, const LVVectorId64_t vector_id, const void* key, const LVKeyLen32_t key_len, const void* value, const LVValueLen32_t value_len, const float vector_score, const LVOrdbyValue ordbyvalue);
static LVStatus lv_qvset_light_append_internal(LVQVSet* qvset, const LVSeq64_t node_seq, const LVVectorId64_t vector_id, const void* key, const LVKeyLen32_t key_len, const void* value, const LVValueLen32_t value_len, const float vector_score, const LVOrdbyValue ordbyvalue);
static void lv_destory_qvset_internal(LVQVSet* qvset);
static void lv_light_destory_qvset_internal(LVQVSet* qvset);
static LVStatus lv_merge_qvsets_internal(LVQVSet* result_qvset, const LVQVSet* memtable_qvset, const LVQVSet* sst_qvset);
static void lv_apply_score_filter_internal(LVQVSet* qvset, const float threshold, const LVScoreBound bound);
static void lv_apply_ordby_internal(LVQVSet* qvset, const LVOrdbyType type, const LVQueryOrderDir dir);
static void lv_apply_limit_internal(LVQVSet* qvset, const LVSize32_t limit);
static LVQueryResultSet* lv_create_query_result_set_internal(const LVQVSet* result_qvset);

LVStatus lv_open(LightVec** db, const LVSchema* schema, const char* path, const LVSize32_t flush_threshold)
{
    int flag = 0;
    LVStatus result = LV_OK;
    LightVec* LV_DB = NULL;
    LVSchema* LV_SCHEMA = NULL;
    LVMemTable* LV_MTABLE = NULL;
    LVQueryResultSet* LV_QUERY_RESULT_SET = NULL;
    LVHnsw* LV_HNSW = NULL;

    LV_DB = malloc(sizeof(LightVec));

    if (!LV_DB)
    {
        flag = 1;
        result = LV_ERR_OOM;
        goto cleanup;
    }
    LV_DB->flush_threshold = flush_threshold > 0 ? flush_threshold : 1024;
    LV_DB->magic = LV_MAGIC;

    LV_DB->next_seq = 0;
    LV_DB->next_vector_id = 0;

    // set DB default save path here
    if ((result = path_join(LV_DB->path, LV_PATH_MAX, path, "LV")) != LV_OK)
    {
        flag = 1;
        goto cleanup;
    }

    // SCHEMA
    // read when it already exists
    // else write

    char schema_path[LV_PATH_MAX];

    if ((result = path_join(schema_path, LV_PATH_MAX, LV_DB->path, "schema.lv")) != LV_OK)
    {
        flag = 1;
        goto cleanup;
    }

    int schema_exists = access(schema_path, F_OK) == 0;

    int schema_fd;

    if (schema_exists)
    {
        LV_SCHEMA = malloc(sizeof(LVSchema));

        if (!LV_SCHEMA)
        {
            flag = 1;
            goto cleanup;
        }
        // read saved schema
        schema_fd = open(schema_path, O_RDONLY);
        if ((result = schema_read(schema_fd, LV_SCHEMA)) != LV_OK)
        {
            flag = 1;
            goto cleanup;
        }
    }

    else
    {
        schema_fd = open(schema_path, O_RDWR | O_CREAT, 0644); // 0644 (S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH):  owner can read/write, else only can read

        if ((result = schema_write(schema_fd, schema)) != LV_OK)
        {
            flag = 1;
            goto cleanup;
        }

        LV_SCHEMA = create_schema(schema->vector_dim, schema->vector_type, schema->field_count, schema->field_defs);
        if (!LV_SCHEMA)
        {
            flag = 1;
            goto cleanup;
        }
    }

    LV_DB->schema_fd = schema_fd;
    LV_DB->schema = LV_SCHEMA;

    // create a Memory Table
    LV_MTABLE = create_table(LV_DB->next_seq);
    if (!LV_MTABLE)
    {
        flag = 1;
        goto cleanup;
    }

    LV_DB->memtable = LV_MTABLE;

    // WAL
    // if it exists, then recover.
    // else create a new wal file

    char wal_path[LV_PATH_MAX];
    if ((result = path_join(wal_path, LV_PATH_MAX, LV_DB->path, "wal.lv")) != LV_OK)
    {
        flag = 1;
        goto cleanup;
    }

    int wal_exists = access(wal_path, F_OK) == 0;
    int wal_fd;

    if (wal_exists)
    {
        wal_fd = open(wal_path, O_RDWR | O_CREAT);

        if ((result = wal_recover(wal_fd, LV_MTABLE)) != LV_OK)
        {
            flag = 1;
            goto cleanup;
        }
    }
    else
    {
        wal_fd = open(wal_path, O_RDWR | O_CREAT, 0644);
    }

    LV_DB->wal_fd = wal_fd;

    //SST
    char sst_path[LV_PATH_MAX];
    if ((result = path_join(sst_path, LV_PATH_MAX, LV_DB->path, "sst.lv")) != LV_OK) {
        flag = 1;
        goto cleanup;
    }

    int sst_fd = -1;

    const int sst_exist = access(sst_path, F_OK) == 0;
    if (sst_exist) {
        sst_fd = open(sst_path, O_RDONLY);
    }

    LV_DB->sst_fd = sst_fd;


    // vector
    // create a LVHnsw, and write vectors.lv header

    LV_HNSW = create_hnsw(schema->vector_type, schema->vector_dim);
    if (!LV_HNSW)
    {
        flag = 1;
        goto cleanup;
    }

    LV_DB->hnsw = LV_HNSW;

    char vectors_path[LV_PATH_MAX];
    char vector_index_path[LV_PATH_MAX];
    char hnsw_index_path[LV_PATH_MAX];
    char hnsw_graph_path[LV_PATH_MAX];

    // set vectors.lv path
    if ((result = path_join(vectors_path, LV_PATH_MAX, LV_DB->path, "vectors.lv")) != LV_OK)
    {
        flag = 1;
        goto cleanup;
    }

    //set vector_index.lv path
    if ((result = path_join(vector_index_path, LV_PATH_MAX, LV_DB->path, "vector_index.lv")) != LV_OK)
    {
        flag = 1;
        goto cleanup;
    }

    // set hnsw_index.lv path
    if ((result = path_join(hnsw_index_path, LV_PATH_MAX, LV_DB->path, "hnsw_index.lv")) != LV_OK)
    {
        flag = 1;
        goto cleanup;
    }

    // set hnsw_graph.lv path
    if ((result = path_join(hnsw_graph_path, LV_PATH_MAX, LV_DB->path, "hnsw_graph.lv")) != LV_OK)
    {
        flag = 1;
        goto cleanup;
    }

    int vectors_exists = access(vectors_path, F_OK) == 0;

    int vector_fd = open(vectors_path, O_RDWR | O_CREAT, 0644);
    int vector_index_fd = open(vector_index_path, O_RDWR | O_CREAT, 0644);
    int hnsw_index_fd = open(hnsw_index_path, O_RDWR | O_CREAT, 0644);
    int hnsw_graph_fd = open(hnsw_graph_path, O_RDWR | O_CREAT, 0644);

    if (vectors_exists)
    {
        // todo recover hnsw
    }

    else
    {
        vector_fd = open(vectors_path, O_RDWR | O_CREAT, 0644);
        if ((result = vector_write_header(vector_fd, schema->vector_type, schema->vector_dim, 1)) != LV_OK)
        {
            flag = 1;
            goto cleanup;
        }
    }

    LV_DB->vector_fd = vector_fd;
    LV_DB->vector_index_fd = vector_index_fd;
    LV_DB->hnsw_index_fd = hnsw_index_fd;
    LV_DB->hnsw_graph_fd = hnsw_graph_fd;

    *db = LV_DB;

cleanup:
    if (flag)
    {
        destroy_table(LV_MTABLE);
        destroy_schema(LV_SCHEMA);
        destroy_hnsw(LV_HNSW);
        safe_free(&LV_DB);
    }

    return result;
}

LVStatus lv_put(LightVec* db, const void* key, const LVKeyLen32_t key_len, const void* value, const LVValueLen32_t value_len, const void* vector, const LVCount32_t field_count, const LVMetaField* fields, const LVVectorMetric vecotr_metric)
{
    LVStatus result = LV_OK;

    // check db is properly initialized
    if ((result = lv_check_db_corruption_internal(db)) != LV_OK) goto _return;

    // check field count is valid

    if (field_count > db->schema->field_count)
    {
        result = LV_ERR_INVALID;
        goto _return;
    }

    // check key length and value length are valid

    if (key_len < 0 || value_len < 0)
    {
        result = LV_ERR_INVALID;
        goto _return;
    }

    uint32_t field_mask = 0;
    LVSize32_t field_size = 0;

    for (int i = 0; i < field_count; ++i)
    {
        LVMetaField* current_field = fields + i;
        LVMetaFieldHash* search_result = schema_search_field_hash(db->schema->field_hashes, current_field->name, strlen(current_field->name));
        if (!search_result)
        { // check field name exists
            result = LV_ERR_INVALID;
            goto _return;
        }
        else
        {
            field_mask |= search_result->mask;
        }

        // field type size
        field_size += 1;

        switch (current_field->type)
        {
        case LV_META_STRING:
            // string value size + value size
            field_size += sizeof(uint32_t) + current_field->value.str.len;
            break;

        case LV_META_FLOAT:
            field_size += sizeof(double);
            break;

        case LV_META_INT:
            field_size += sizeof(int64_t);
            break;
        default:
            break;
        }
    }

    LVLevel8_t new_level = 1;

    while ((xorshift() % 100 <= LV_SKIPLIST_P) && new_level < LV_SKIPLIST_MAX_LEVEL)
    {
        ++new_level;
    }

    const LVSeq64_t current_seq = db->next_seq;
    const LVSeq64_t current_vector_id = db->next_vector_id;

    // append to vectors.lv and hnsw
    if (vector)
    {
        if (db->schema->vector_type == LV_VEC_FLOAT32)
        {
            const float* f32_vector = (float*)vector;
            if ((result = vector_write_f32_vector(db->vector_fd, db->schema->vector_dim, f32_vector)) != LV_OK)
            {
                goto _return;
            }
            if ((result = vector_hnsw_f32_insert(db->hnsw, current_vector_id, f32_vector, vecotr_metric == LV_METRIC_L2 ? vector_f32_l2_sq : vector_f32_dot)) != LV_OK)
            {
                goto _return;
            }
        }
        else
        {
            const int8_t* i8_vector = (int8_t*)vector;
            if ((result = vector_write_i8_vector(db->vector_fd, db->schema->vector_dim, i8_vector)) != LV_OK)
            {
                goto _return;
            }
            if ((result = vector_hnsw_i8_insert(db->hnsw, current_vector_id, i8_vector, vecotr_metric == LV_METRIC_L2 ? vector_i8_l2_sq : vector_i8_dot)) != LV_OK)
            {
                goto _return;
            }
        }

        db->next_vector_id += 1; // increase vector id for next
    }

    void* KEY = key;
    LVKeyLen32_t KEY_LEN = key_len;
    void* VALUE = value;
    LVValueLen32_t VALUE_LEN = value_len;

    if (KEY == NULL)
    {
        KEY = &current_seq;
        KEY_LEN = sizeof(LVSeq64_t);
    }

    if (VALUE == NULL)
    {
        VALUE_LEN = 0;
    }

    // append to WAL

    if ((result = wal_append(db->wal_fd, LV_PUT, current_seq, new_level, KEY_LEN, KEY, VALUE_LEN, VALUE, vector ? current_vector_id : LV_NO_VECTOR_ID, field_mask, field_count, field_size, fields)) != LV_OK)
    {
        goto _return;
    }

    // append to MemTable

    const LVNode* inserted_memtable_node = table_insert(db->memtable, LV_PUT, current_seq, new_level, KEY_LEN, KEY, VALUE_LEN, VALUE, vector ? current_vector_id : LV_NO_VECTOR_ID, field_mask, field_count, field_size, fields);

    if (!inserted_memtable_node)
    {
        result = LV_ERR_OOM;
        goto _return;
    }

    if (vector) {
        LVHnswNode* inserted_hnsw_node = db->hnsw->id_node_map->map[current_vector_id];
        inserted_hnsw_node->memtable_node = inserted_memtable_node;
    }

    db->next_seq += 1;

    if (db->memtable->node_count >= db->flush_threshold) {
        result = lv_flush_internal(db);
    }

_return:
    return result;
}

LVStatus lv_query(const LightVec* db, const char* query, const void* query_vector, const LVQueryOption* option, LVQueryResultSet** outputs)
{
    LVStatus result = LV_OK;

    LVSQLParser* parser = NULL;
    LVAstNode* query_tree = NULL;
    LVQVSet* memtable_qvset = NULL;
    LVQVSet* sst_qvset = NULL;
    LVQVSet* merged_qvset = NULL;

    // check db is properly initialized
    if ((result = lv_check_db_corruption_internal(db)) != LV_OK) goto _return;

    // check query
    if (query == NULL || strlen(query) <= 0)
    {
        result = LV_ERR_INVALID_QUERY;
        goto _return;
    }

    const int has_option = option != NULL && !(option->flags & LV_QOPT_NONE);


    const int is_limit_on = has_option && option->flags & LV_QOPT_LIMIT && option->limit > 0;
    const int is_top_k_on = has_option && option->top_k > 0;
    const int is_score_filter_on = has_option && query_vector && option->flags & LV_QOPT_SCORE_FILTER;
    const int is_ordby_on = (has_option && (option->flags & LV_QOPT_ORDER_BY)) || is_top_k_on;
    const int is_ordby_vec = is_ordby_on && query_vector && ((strncasecmp(option->order.by, "vector", strlen("vector")) == 0) || is_top_k_on);
    const int needs_calc_dis = is_score_filter_on || is_ordby_vec;

    uint32_t ordby_field_mask = 0;
    LVOrdbyType ordbytype = LV_ORDBY_NONE;

    if (is_ordby_on) {
        const LVMetaFieldHash* hash = schema_search_field_hash(db->schema->field_hashes, option->order.by, strlen(option->order.by));
        if (hash->type != LV_META_STRING) {
            ordby_field_mask = hash->mask;
        }
        if (is_ordby_vec) {
            ordbytype = LV_ORDBY_VEC;
        }
        else {
            if (hash->type == LV_META_FLOAT) {
                ordbytype = LV_ORDBY_FLOAT;
            }
            else {
                ordbytype = LV_ORDBY_INT;
            }
        }
    }

    parser = create_parser();
    if (!parser) {
        result = LV_ERR_OOM;
        goto _return;
    }
    if ((result = query_tokenize(query, parser)) != LV_OK)
    {
        goto _return;
    }

    query_tree = query_parse(parser, db->schema);

    if (!query_tree)
    {
        result = LV_ERR_INVALID_QUERY;
        goto _return;
    }

    const uint32_t query_field_mask = query_get_field_mask(query_tree, db->schema);

    //create qvsets
    memtable_qvset = lv_create_qvset_internal();
    if (!memtable_qvset) {
        result = LV_ERR_OOM;
        goto _return;
    }


    sst_qvset = lv_create_qvset_internal();
    if (!sst_qvset) {
        result = LV_ERR_OOM;
        goto _return;
    }


    merged_qvset = lv_create_qvset_internal();
    if (!merged_qvset) {
        result = LV_ERR_OOM;
        goto _return;
    }

    if (needs_calc_dis) {
        const int is_f32 = db->schema->vector_type == LV_VEC_FLOAT32;
        LVSize32_t search_ef = HNSW_EF_DEFAULT + parser->complexity_score * 10;
        if (option->top_k > 0) {
            search_ef = option->top_k > search_ef ? option->top_k : search_ef;
        }

        LVF32DistFn f32_dist_fn = NULL;
        LVI8DistFn i8_dist_fn = NULL;

        if (is_f32) {
            f32_dist_fn = option->vector_metric == LV_METRIC_L2 ? vector_f32_l2_sq : vector_f32_dot;
        }
        else {
            i8_dist_fn = option->vector_metric == LV_METRIC_L2 ? vector_i8_l2_sq : vector_i8_dot;
        }

        LVHnswQueryCtx hnsw_qctx = {
            .search_ef = search_ef,
            .is_f32 = is_f32,
            .f32_dist_fn = f32_dist_fn,
            .i8_dist_fn = i8_dist_fn,
            .vector_metric = option->vector_metric,
            .ordby_field_mask = ordby_field_mask,
            .ordbytype = ordbytype,
            .query_field_mask = query_field_mask,
            .memtable_qvset = memtable_qvset,
            .memtable_qvset_append_fn = lv_qvset_light_append_internal,
            .sst_qvset = sst_qvset,
            .sst_qvset_append_fn = lv_qvset_append_internal,
            .sst_fd = db->sst_fd,
            .vector_index_fd = db->vector_index_fd,
        };

        if((result = vector_hnsw_query(db->hnsw,db->schema, query_tree,query_vector,&hnsw_qctx)) != LV_OK) goto _return;
        if((result = lv_merge_qvsets_internal(merged_qvset, memtable_qvset, sst_qvset)) != LV_OK) goto _return;
    }
    else {
        if ((result = table_query_filter_scan(db->memtable, db->schema, query_tree, query_field_mask, ordbytype, ordby_field_mask, lv_qvset_light_append_internal, memtable_qvset)) != LV_OK) goto _return;
        if ((result = sst_query_filter_scan(db->sst_fd, db->schema, query_tree, query_field_mask, ordbytype, ordby_field_mask, lv_qvset_append_internal, sst_qvset)) != LV_OK) goto _return;

        if ((result = lv_merge_qvsets_internal(merged_qvset, memtable_qvset, sst_qvset)) != LV_OK) goto _return;
    }


    if (is_score_filter_on) {
        lv_apply_score_filter_internal(merged_qvset, option->vector_score_filter.score, option->vector_score_filter.bound);
    }

    if (is_ordby_on) {
        lv_apply_ordby_internal(merged_qvset, ordbytype, option->order.dir);
    }

    if (is_limit_on || is_top_k_on) {
        lv_apply_limit_internal(merged_qvset, is_top_k_on ? option->top_k : option->limit);
    }

    *outputs = lv_create_query_result_set_internal(merged_qvset);

_return:
    lv_light_destory_qvset_internal(memtable_qvset);
    lv_destory_qvset_internal(sst_qvset);
    lv_light_destory_qvset_internal(merged_qvset);
    destory_parser(parser);
    destroy_ast(query_tree);
    return result;
}

void lv_destroy_query_result_set(LVQueryResultSet* qrset) {
    if (qrset) {
        for (int i = 0; i < qrset->size; ++i) {
            free(qrset->results[i].key);
            free(qrset->results[i].value);
        }
        free(qrset->results);
        free(qrset);
    }
}

static LVStatus lv_check_db_corruption_internal(const LightVec* db) {
    return db->magic == LV_MAGIC ? LV_OK : LV_ERR_CORRUPT;
}


static LVStatus lv_flush_internal(LightVec* db) {
    LVStatus result = LV_OK;

    char sst_path[LV_PATH_MAX];
    char sst_tmp_path[LV_PATH_MAX];

    if ((result = path_join(sst_path, LV_PATH_MAX, db->path, "sst.lv")) != LV_OK) goto _return;
    if ((result = path_join(sst_tmp_path, LV_PATH_MAX, db->path, "sst.tmp")) != LV_OK) goto _return;

    int old_fd = db->sst_fd;
    int new_fd = open(sst_tmp_path, O_RDWR | O_CREAT | O_TRUNC, 0644);

    if (new_fd < 0) {
        result = LV_ERR_IO;
        goto _return;
    };

    if ((result = sst_flush(new_fd, old_fd, db->vector_index_fd, db->memtable->head->levels[0])) != LV_OK) goto _return;

    write_helper_flush(new_fd, 1);
    if (old_fd >= 0) close(old_fd);
    db->sst_fd = new_fd;

    // atomic rename
    if (rename(sst_tmp_path, sst_path) != 0) {
        result = LV_ERR_IO;
        goto _return;
    };

    LVNode* current = db->memtable->head->levels[0];
    while (current->type != LV_NODE_TAIL) {
        if (current->op != LV_DELETE && current->vector_id != LV_NO_VECTOR_ID) {
            vector_hnsw_mark_flushed(db->hnsw, current->vector_id);
        }
        current = current->levels[0];
    }

    // memtable reset
    destroy_table(db->memtable);

    db->memtable = create_table(db->next_seq);
    if (!db->memtable) {
        result = LV_ERR_OOM;
        goto _return;
    };

    // wal truncate
    ftruncate(db->wal_fd, 0);
    lseek(db->wal_fd, 0, SEEK_SET);


_return:
    return result;
}

static LVQVSet* lv_create_qvset_internal(void) {
    int flag = 0;
    LVQVSet* result = NULL;

    result = malloc(sizeof(LVQVSet));
    if (!result) goto cleanup;
    result->capacity = LV_DEFAULT_CAPACITY;
    result->size = 0;
    result->values = NULL;

    LVQueryValue* values = malloc(sizeof(LVQueryValue) * result->capacity);
    if (!values) {
        flag = 1;
        goto cleanup;
    }

    result->values = values;
cleanup:
    if (flag) {
        lv_destory_qvset_internal(result);
    }
    return result;
}

static LVStatus lv_qvset_append_internal(LVQVSet* qvset, const LVSeq64_t node_seq, const LVVectorId64_t vector_id, const void* key, const LVKeyLen32_t key_len, const void* value, const LVValueLen32_t value_len, const float vector_score, const LVOrdbyValue ordbyvalue) {
    LVStatus result = LV_OK;

    if (qvset->size >= qvset->capacity)
    {
        LVSize32_t new_capacity = qvset->capacity * 2;
        LVQueryValue* tmp = realloc(qvset->values, new_capacity * sizeof(LVQueryValue));
        if (!tmp)
        {
            result = LV_ERR_OOM;
            goto _return;
        }
        qvset->capacity = new_capacity;
        qvset->values = tmp;
    }

    const void* key_to_save = malloc(key_len);
    if (!key_to_save) {
        result = LV_ERR_OOM;
        goto _return;
    }
    memcpy(key_to_save, key, key_len);

    const void* value_to_save = malloc(value_len);
    if (!value_to_save) {
        free(key_to_save);
        result = LV_ERR_OOM;
        goto _return;
    }
    memcpy(value_to_save, value, value_len);


    const int index = qvset->size;

    qvset->values[index].node_seq = node_seq;
    qvset->values[index].vector_id = vector_id;
    qvset->values[index].key = key_to_save;
    qvset->values[index].key_len = key_len;
    qvset->values[index].value = value_to_save;
    qvset->values[index].value_len = value_len;
    qvset->values[index].vector_score = vector_score;
    qvset->values[index].ordbyvalue = ordbyvalue;

    qvset->size += 1;
_return:
    return result;
}

static LVStatus lv_qvset_light_append_internal(LVQVSet* qvset, const LVSeq64_t node_seq, const LVVectorId64_t vector_id, const void* key, const LVKeyLen32_t key_len, const void* value, const LVValueLen32_t value_len, const float vector_score, const LVOrdbyValue ordbyvalue) {
    LVStatus result = LV_OK;

    if (qvset->size >= qvset->capacity)
    {
        LVSize32_t new_capacity = qvset->capacity * 2;
        LVQueryValue* tmp = realloc(qvset->values, new_capacity * sizeof(LVQueryValue));
        if (!tmp)
        {
            result = LV_ERR_OOM;
            goto _return;
        }
        qvset->capacity = new_capacity;
        qvset->values = tmp;
    }

    const int index = qvset->size;

    qvset->values[index].node_seq = node_seq;
    qvset->values[index].vector_id = vector_id;
    qvset->values[index].key = key; //must be pre-heap allocated
    qvset->values[index].key_len = key_len;
    qvset->values[index].value = value; //must be pre-heap allocated
    qvset->values[index].value_len = value_len;
    qvset->values[index].vector_score = vector_score;
    qvset->values[index].ordbyvalue = ordbyvalue;

    qvset->size += 1;
_return:
    return result;
}

static void lv_destory_qvset_internal(LVQVSet* qvset) {
    if (qvset) {
        for (int i = 0; i < qvset->size; ++i) {
            free(qvset->values[i].key);
            free(qvset->values[i].value);
        }
        free(qvset->values);
        free(qvset);
    }
}

static void lv_light_destory_qvset_internal(LVQVSet* qvset) {
    if (qvset) {
        free(qvset->values);
        free(qvset);
    }
}

static LVStatus lv_merge_qvsets_internal(LVQVSet* result_qvset, const LVQVSet* memtable_qvset, const LVQVSet* sst_qvset) {
    LVStatus result = LV_OK;

    LVSize32_t i = 0;
    LVSize32_t j = 0;

    while (i < memtable_qvset->size && j < sst_qvset->size)
    {
        const void* current_mtable_key = memtable_qvset->values[i].key;
        const LVKeyLen32_t current_mtable_klen = memtable_qvset->values[i].key_len;
        const void* current_sst_key = sst_qvset->values[j].key;
        const LVKeyLen32_t current_sst_klen = sst_qvset->values[j].key_len;

        if (node_key_equal(current_mtable_key, current_mtable_klen, current_sst_key, current_sst_klen)) {
            if ((result = lv_qvset_light_append_internal(result_qvset,
                memtable_qvset->values[i].node_seq,
                memtable_qvset->values[i].vector_id,
                current_mtable_key,
                current_mtable_klen,
                memtable_qvset->values[i].value,
                memtable_qvset->values[i].value_len,
                memtable_qvset->values[i].vector_score,
                memtable_qvset->values[i].ordbyvalue
            )) != LV_OK) goto _return;
            ++i;
            ++j;
        }
        else {
            if (node_cmp(LV_NODE_DATA, current_mtable_key, current_mtable_klen, memtable_qvset->values[i].node_seq,
                LV_NODE_DATA, current_sst_key, current_sst_klen, sst_qvset->values[j].node_seq
            ) < 0) {
                if ((result = lv_qvset_light_append_internal(result_qvset,
                    memtable_qvset->values[i].node_seq,
                    memtable_qvset->values[i].vector_id,
                    current_mtable_key,
                    current_mtable_klen,
                    memtable_qvset->values[i].value,
                    memtable_qvset->values[i].value_len,
                    memtable_qvset->values[i].vector_score,
                    memtable_qvset->values[i].ordbyvalue
                )) != LV_OK) goto _return;
                ++i;
            }
            else {
                if ((result = lv_qvset_light_append_internal(result_qvset,
                    sst_qvset->values[j].node_seq,
                    sst_qvset->values[j].vector_id,
                    current_sst_key,
                    current_sst_klen,
                    sst_qvset->values[j].value,
                    sst_qvset->values[j].value_len,
                    sst_qvset->values[j].vector_score,
                    sst_qvset->values[j].ordbyvalue
                )) != LV_OK) goto _return;
                ++j;
            }
        }
    }


    while (i < memtable_qvset->size) {
        if ((result = lv_qvset_light_append_internal(result_qvset,
            memtable_qvset->values[i].node_seq,
            memtable_qvset->values[i].vector_id,
            memtable_qvset->values[i].key,
            memtable_qvset->values[i].key_len,
            memtable_qvset->values[i].value,
            memtable_qvset->values[i].value_len,
            memtable_qvset->values[i].vector_score,
            memtable_qvset->values[i].ordbyvalue
        )) != LV_OK) goto _return;
        ++i;
    }

    while (j < sst_qvset->size) {
        if ((result = lv_qvset_light_append_internal(result_qvset,
            sst_qvset->values[j].node_seq,
            sst_qvset->values[j].vector_id,
            sst_qvset->values[j].key,
            sst_qvset->values[j].key_len,
            sst_qvset->values[j].value,
            sst_qvset->values[j].value_len,
            sst_qvset->values[j].vector_score,
            sst_qvset->values[j].ordbyvalue
        )) != LV_OK) goto _return;
        ++j;
    }

_return:
    return result;
}

static void lv_apply_score_filter_internal(LVQVSet* qvset, const float threshold, const LVScoreBound bound) {
    LVSize32_t write_idx = 0;

    for (LVSize32_t i = 0; i < qvset->size; i++) {
        int keep = 0;
        if (bound == LV_SCORE_ABOVE) {
            keep = (qvset->values[i].vector_score >= threshold);
        } else {
            keep = (qvset->values[i].vector_score <= threshold);
        }

        if (keep) {
            if (write_idx != i) {
                qvset->values[write_idx] = qvset->values[i];
            }
            write_idx++;
        }
    }
    qvset->size = write_idx;
}


static int cmp_f64_asc(const void* a, const void* b) {
    const LVQueryValue* qva = a;
    const LVQueryValue* qvb = b;

    return (qva->ordbyvalue.f64 > qvb->ordbyvalue.f64) - (qva->ordbyvalue.f64 < qvb->ordbyvalue.f64);
}

static int cmp_f64_desc(const void* a, const void* b) {
    const LVQueryValue* qva = a;
    const LVQueryValue* qvb = b;

    return (qva->ordbyvalue.f64 < qvb->ordbyvalue.f64) - (qva->ordbyvalue.f64 > qvb->ordbyvalue.f64);
}

static int cmp_i64_asc(const void* a, const void* b) {
    const LVQueryValue* qva = a;
    const LVQueryValue* qvb = b;

    return (qva->ordbyvalue.i64 > qvb->ordbyvalue.i64) - (qva->ordbyvalue.i64 < qvb->ordbyvalue.i64);
}

static int cmp_i64_desc(const void* a, const void* b) {
    const LVQueryValue* qva = a;
    const LVQueryValue* qvb = b;

    return (qva->ordbyvalue.i64 < qvb->ordbyvalue.i64) - (qva->ordbyvalue.i64 > qvb->ordbyvalue.i64);
}

static int cmp_vecscore_asc(const void* a, const void* b) {
    const LVQueryValue* qva = a;
    const LVQueryValue* qvb = b;

    return (qva->ordbyvalue.score > qvb->ordbyvalue.score) - (qva->ordbyvalue.score < qvb->ordbyvalue.score);
}

static int cmp_vecscore_desc(const void* a, const void* b) {
    const LVQueryValue* qva = a;
    const LVQueryValue* qvb = b;

    return (qva->ordbyvalue.score < qvb->ordbyvalue.score) - (qva->ordbyvalue.score > qvb->ordbyvalue.score);
}

static void lv_apply_ordby_internal(LVQVSet* qvset, const LVOrdbyType type, const LVQueryOrderDir dir) {
    switch (type)
    {
    case LV_ORDBY_FLOAT: {
        if (dir == LV_ORDER_ASC) {
            qsort(qvset->values, qvset->size, sizeof(LVQueryValue), cmp_f64_asc);
        }
        else {
            qsort(qvset->values, qvset->size, sizeof(LVQueryValue), cmp_f64_desc);
        }
        break;
    }

    case LV_ORDBY_INT: {
        if (dir == LV_ORDER_ASC) {
            qsort(qvset->values, qvset->size, sizeof(LVQueryValue), cmp_i64_asc);
        }
        else {
            qsort(qvset->values, qvset->size, sizeof(LVQueryValue), cmp_i64_desc);
        }
        break;
    }

    case LV_ORDBY_VEC: {
        if (dir == LV_ORDER_ASC) {
            qsort(qvset->values, qvset->size, sizeof(LVQueryValue), cmp_vecscore_asc);
        }
        else {
            qsort(qvset->values, qvset->size, sizeof(LVQueryValue), cmp_vecscore_desc);
        }
        break;
    }

    default:
        break;
    }
}

static void lv_apply_limit_internal(LVQVSet* qvset, const LVSize32_t limit) {
    if (qvset->size > limit) {
        qvset->size = limit;
    }
}

static LVQueryResultSet* lv_create_query_result_set_internal(const LVQVSet* result_qvset) {
    int flag = 0;
    LVQueryResultSet* result = NULL;

    result = malloc(sizeof(LVQueryResultSet));
    if (!result) goto cleanup;

    result->size = result_qvset->size;
    result->results = malloc(sizeof(LVQueryResult) * result->size);

    if (!result->results) {
        flag = 1;
        goto cleanup;
    }

    for (LVSize32_t i = 0; i < result_qvset->size; ++i) {
        result->results[i].key = NULL;
        result->results[i].value = NULL;
        void* key_copy = malloc(result_qvset->values[i].key_len);
        if (!key_copy) {
            flag = 1;
            goto cleanup;
        }

        void* value_copy = malloc(result_qvset->values[i].value_len);
        if (!value_copy) {
            flag = 1;
            goto cleanup;
        }

        memcpy(key_copy, result_qvset->values[i].key, result_qvset->values[i].key_len);
        memcpy(value_copy, result_qvset->values[i].value, result_qvset->values[i].value_len);

        result->results[i].node_seq = result_qvset->values[i].node_seq;
        result->results[i].vector_id = result_qvset->values[i].vector_id;
        result->results[i].key = key_copy;
        result->results[i].key_len = result_qvset->values[i].key_len;
        result->results[i].value = value_copy;
        result->results[i].value_len = result_qvset->values[i].value_len;
        result->results[i].vector_score = result_qvset->values[i].vector_score;
    }
cleanup:
    if (flag) {
        lv_destroy_query_result_set(result);
        result = NULL;
    }
    return result;
}
