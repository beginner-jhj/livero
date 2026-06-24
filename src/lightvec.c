#include "lightvec.h"
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <errno.h>
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
    int vectors_fd;                 // vectors.lv (O(1) access)
    int vector_index_fd; //vector_id to sst record offset
    LVBigCount64_t next_vector_id;

    LVHnsw* hnsw;

    int32_t magic;
};

static LVStatus lv_recover_internal(LightVec* db);
static LVStatus lv_check_db_corruption_internal(const LightVec* db);
static LVStatus lv_put_internal(LightVec* db, const LVNodeOp op, const LVSeq64_t current_seq, const LVVectorId64_t current_vector_id, const void* key, const LVKeyLen32_t key_len, const void* value, const LVValueLen32_t value_len, const void* vector, const LVSize32_t field_mask, const LVSize32_t field_count, const LVSize32_t field_size, const void* memory_field_buffer);
static LVQVSet* lv_create_qvset_internal(void);
static LVStatus lv_flush_internal(LightVec* db);
static LVStatus lv_qvset_append_internal(LVQVSet* qvset, const LVSeq64_t node_seq, const LVVectorId64_t vector_id, const void* key, const LVKeyLen32_t key_len, const void* value, const LVValueLen32_t value_len, const float vector_score, const LVOrdbyValue ordbyvalue, const int is_tombstone);
static LVStatus lv_qvset_light_append_internal(LVQVSet* qvset, const LVSeq64_t node_seq, const LVVectorId64_t vector_id, const void* key, const LVKeyLen32_t key_len, const void* value, const LVValueLen32_t value_len, const float vector_score, const LVOrdbyValue ordbyvalue, const int is_tombstone);
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
    LVHnsw* LV_HNSW = NULL;

    LV_DB = malloc(sizeof(LightVec));

    if (!LV_DB)
    {
        flag = 1;
        result = LV_ERR_OOM;
        goto cleanup;
    }
    memset(LV_DB, 0, sizeof(LightVec));

    LV_DB->flush_threshold = flush_threshold > 0 ? flush_threshold : 1024;
    LV_DB->magic = LV_MAGIC;
    LV_DB->schema_fd = -1;
    LV_DB->wal_fd = -1;
    LV_DB->sst_fd = -1;
    LV_DB->vectors_fd = -1;
    LV_DB->vector_index_fd = -1;

    // set DB default save path here
    if ((result = path_join(LV_DB->path, LV_PATH_MAX, path, "LV")) != LV_OK)
    {
        flag = 1;
        goto cleanup;
    }

    if (mkdir(LV_DB->path, 0755) == -1 && errno != EEXIST) {
        flag = 1;
        result = LV_ERR_IO;
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
            result = LV_ERR_OOM;
            goto cleanup;
        }
        // read saved schema
        schema_fd = open(schema_path, O_RDONLY);
        result = schema_read(schema_fd, LV_SCHEMA);
        if (result != LV_OK)
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

        LV_SCHEMA = create_schema(schema->vector_dim, schema->vector_type, schema->vector_metric, schema->field_count, schema->field_defs);
        if (!LV_SCHEMA)
        {
            flag = 1;
            result = LV_ERR_OOM;
            goto cleanup;
        }
    }

    LV_DB->schema_fd = schema_fd;
    LV_DB->schema = LV_SCHEMA;

    // create a Memory Table
    LV_MTABLE = create_table();
    if (!LV_MTABLE)
    {
        flag = 1;
        result = LV_ERR_OOM;
        goto cleanup;
    }

    LV_DB->memtable = LV_MTABLE;

    // WAL
    char wal_path[LV_PATH_MAX];
    if ((result = path_join(wal_path, LV_PATH_MAX, LV_DB->path, "wal.lv")) != LV_OK)
    {
        flag = 1;
        goto cleanup;
    }

    int wal_fd = open(wal_path, O_RDWR | O_CREAT, 0644);
    LV_DB->wal_fd = wal_fd;

    //SST
    char sst_path[LV_PATH_MAX];
    if ((result = path_join(sst_path, LV_PATH_MAX, LV_DB->path, "sst.lv")) != LV_OK) {
        flag = 1;
        goto cleanup;
    }

    //if sst exists then open returns proper fd.
    //else open returns -1.
    int sst_fd = open(sst_path, O_RDONLY); //this sst is old_sst, O_RDONLY is ok for sst_flush.

    LV_DB->sst_fd = sst_fd;


    // vector
    // create a LVHnsw

    LV_HNSW = create_hnsw(schema->vector_type, schema->vector_dim);
    if (!LV_HNSW)
    {
        flag = 1;
        goto cleanup;
    }

    LV_DB->hnsw = LV_HNSW;

    char vectors_path[LV_PATH_MAX];
    char vector_index_path[LV_PATH_MAX];

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

    int vectors_fd = open(vectors_path, O_RDWR | O_CREAT, 0644);
    int vector_index_fd = open(vector_index_path, O_RDWR | O_CREAT, 0644);


    LV_DB->vectors_fd = vectors_fd;
    LV_DB->vector_index_fd = vector_index_fd;

    if ((result = lv_recover_internal(LV_DB)) != LV_OK) {
        flag = 1;
        goto cleanup;
    }

    *db = LV_DB;

cleanup:
    if (flag)
    {
        lv_close(LV_DB);
    }

    return result;
}

LVStatus lv_put(LightVec* db, const void* key, const LVKeyLen32_t key_len, const void* value, const LVValueLen32_t value_len, const void* vector, const LVCount32_t field_count, const LVMetaField* fields)
{
    LVStatus result = LV_OK;


    // check db is properly initialized
    if ((result = lv_check_db_corruption_internal(db)) != LV_OK) return result;

    // check field count is valid

    if (field_count > db->schema->field_count)
    {
        result = LV_ERR_INVALID;
        return result;
    }

    // check key length and value length are valid

    if (key_len == 0 || key_len > LV_MAX_KEY_LEN || value_len > LV_MAX_VALUE_LEN) {
        result = LV_ERR_INVALID;
        return result;
    }

    uint32_t field_mask = 0;

    for (int i = 0; i < field_count; ++i)
    {
        LVMetaField* current_field = fields + i;
        LVMetaFieldHash* search_result = schema_search_field_hash(db->schema->field_hashes, current_field->name, strlen(current_field->name));
        if (!search_result)
        { // check field name exists
            result = LV_ERR_INVALID;
            return result;
        }
        else
        {
            field_mask |= search_result->mask;
        }
    }

    const LVSeq64_t current_seq = db->next_seq;
    const LVSeq64_t current_vector_id = db->next_vector_id;

    LVSize32_t field_size = schema_field_serialized_size(fields, field_count);
    if (field_size > 0) {
        char memory_field_buffer[field_size];

        schema_serialize_field(db->schema, memory_field_buffer, fields, field_count, 0);

        result = lv_put_internal(db, LV_PUT, current_seq, current_vector_id, key, key_len, value, value_len, vector, field_mask, field_count, field_size, memory_field_buffer);
    }
    else {
        result = lv_put_internal(db, LV_PUT, current_seq, current_vector_id, key, key_len, value, value_len, vector, 0, 0, 0, NULL);
    }

    return result;
}

LVStatus lv_update_value(LightVec* db, const void* key, const LVKeyLen32_t key_len, const void* value, const LVValueLen32_t value_len) {
    LVStatus result = LV_OK;

    if ((result = lv_check_db_corruption_internal(db)) != LV_OK) return result;

    // check key length and value length are valid
    if (key_len == 0 || key_len > LV_MAX_KEY_LEN || value_len > LV_MAX_VALUE_LEN) {
        result = LV_ERR_INVALID;
        return result;
    }

    LVVectorId64_t found_vector_id = LV_NO_VECTOR_ID;
    LVCount32_t found_field_mask = 0;
    LVCount32_t found_field_count = 0;
    LVSize32_t found_field_size = 0;

    LVNode* mt_node = table_search(db->memtable, key, key_len);

    if (mt_node) {
        if (mt_node->op == LV_DELETE) {
            result = LV_ERR_NOT_FOUND;
            return result;
        }
        found_vector_id = mt_node->vector_id;
        found_field_mask = mt_node->field_mask;
        found_field_count = mt_node->field_count;
        found_field_size = node_field_size(mt_node);
        result = lv_put_internal(db, LV_UPDATE, db->next_seq, found_vector_id, key, key_len, value, value_len, NULL, found_field_mask, found_field_count, found_field_size, node_access_field(mt_node, 0));
        return result;
    }

    else if (db->sst_fd < 0) {
        result = LV_ERR_NOT_FOUND;
        return result;
    }

    else {
        LVSSTIndexBlockEntry entry;
        const LVStatus sst_search_result = sst_search_index_block(db->sst_fd, &entry, key, key_len);

        if (sst_search_result == LV_OK) {
            uint64_t record_head_size = 0;
            if ((result = sst_read_record_head(db->sst_fd, entry.offset, NULL, NULL, NULL, NULL, NULL, &found_vector_id, &found_field_mask, &found_field_count, &found_field_size, &record_head_size)) != LV_OK) {
                free(entry.key);
                return result;
            };
            if (found_field_size > 0) {
                char memory_field_buffer[found_field_size];
                if ((result = sst_read_record_tail(db->sst_fd, entry.offset + record_head_size, NULL, 0, NULL, 0, memory_field_buffer, found_field_size, NULL)) != LV_OK) {
                    free(entry.key);
                    return result;
                };

                result = lv_put_internal(db, LV_UPDATE, db->next_seq, found_vector_id, key, key_len, value, value_len, NULL, found_field_mask, found_field_count, found_field_size, memory_field_buffer);
            }
            else {
                result = lv_put_internal(db, LV_UPDATE, db->next_seq, found_vector_id, key, key_len, value, value_len, NULL, 0, 0, 0, NULL);
            }
            free(entry.key);
            return result;

        }
        else {
            result = sst_search_result;
            //no need to free entry key
            return result;
        }
    }

    return LV_ERR_NOT_FOUND;
}

LVStatus lv_update_vector(LightVec* db, const void* key, const LVKeyLen32_t key_len, const void* vector) {
    LVStatus result = LV_OK;

    if ((result = lv_check_db_corruption_internal(db)) != LV_OK) return result;

    if (key_len == 0 || key_len > LV_MAX_KEY_LEN || !vector) {
        result = LV_ERR_INVALID;
        return result;
    }

    LVVectorId64_t found_vector_id = LV_NO_VECTOR_ID;
    LVCount32_t found_field_mask = 0;
    LVCount32_t found_field_count = 0;
    LVSize32_t found_field_size = 0;
    LVValueLen32_t found_value_len = 0;

    LVNode* mt_node = table_search(db->memtable, key, key_len);

    if (mt_node) {
        if (mt_node->op == LV_DELETE) {
            result = LV_ERR_NOT_FOUND;
            return result;
        }

        found_vector_id = mt_node->vector_id;
        found_field_mask = mt_node->field_mask;
        found_field_count = mt_node->field_count;
        found_field_size = node_field_size(mt_node);
        found_value_len = mt_node->value_len;

        const LVSeq64_t current_seq = db->next_seq;
        const LVSeq64_t current_vector_id = db->next_vector_id;

        if ((result = lv_put_internal(db, LV_UPDATE, current_seq, current_vector_id, key, key_len,
            node_access_value(mt_node), found_value_len, vector,
            found_field_mask, found_field_count, found_field_size, node_access_field(mt_node, 0))) != LV_OK) return result;
    }
    else if (db->sst_fd < 0) {
        result = LV_ERR_NOT_FOUND;
        return result;
    }
    else {
        LVSSTIndexBlockEntry entry;
        const LVStatus sst_search_result = sst_search_index_block(db->sst_fd, &entry, key, key_len);

        if (sst_search_result != LV_OK) {
            result = sst_search_result;
            return result;
        }

        uint64_t record_head_size = 0;
        if ((result = sst_read_record_head(db->sst_fd, entry.offset, NULL, NULL, NULL, NULL, &found_value_len, &found_vector_id, &found_field_mask, &found_field_count, &found_field_size, &record_head_size)) != LV_OK) {
            free(entry.key);
            return result;
        }

        char found_value[found_value_len];
        if ((result = sst_read_record_tail(db->sst_fd, entry.offset + record_head_size, NULL, 0, found_value, found_value_len, NULL, found_field_size, NULL)) != LV_OK) {
            free(entry.key);
            return result;
        }

        if ((result = lv_put_internal(db, LV_UPDATE, db->next_seq, db->next_vector_id, key, key_len,
            found_value, found_value_len, vector,
            found_field_mask, found_field_count, found_field_size, NULL)) != LV_OK) {
            free(entry.key);
            return result;
        }

        found_vector_id = entry.vector_id;
        free(entry.key);
    }

    if (found_vector_id != LV_NO_VECTOR_ID) {
        vector_hnsw_mark_updated(db->hnsw, found_vector_id);
    }

    return result;
}

LVStatus lv_update_field(LightVec* db, const void* key, const LVKeyLen32_t key_len, const LVSize32_t field_count, const LVMetaField* fields) {
    LVStatus result = LV_OK;

    LVSize32_t field_count_to_add = 0;
    int new_fields_offsets[field_count];
    uint32_t new_field_masks[field_count];
    uint32_t field_mask_to_add = 0;

    LVSize32_t field_count_to_update = 0;
    int update_fields_offsets[field_count];
    uint32_t field_mask_to_update = 0;

    LVValueLen32_t found_value_len = 0;
    LVCount32_t found_field_mask = 0;
    LVCount32_t found_field_count = 0;
    LVSize32_t found_field_size = 0;
    LVVectorId64_t found_vector_id = LV_NO_VECTOR_ID;

    void* found_value = NULL;
    void* found_field = NULL;
    void* new_field = NULL;
    LVSize32_t* new_field_accrued_offsets = NULL;
    LVSize32_t new_field_size = 0;
    LVCount32_t new_field_mask = 0;
    LVCount32_t new_field_count = 0;

    if ((result = lv_check_db_corruption_internal(db)) != LV_OK) goto cleanup;

    if (key_len == 0 || key_len > LV_MAX_KEY_LEN || field_count > db->schema->field_count || (field_count > 0 && fields == NULL)) {
        result = LV_ERR_INVALID;
        goto cleanup;
    }

    LVNode* mt_node = table_search(db->memtable, key, key_len);

    if (mt_node) {
        if (mt_node->op == LV_DELETE) {
            result = LV_ERR_NOT_FOUND;
            goto cleanup;
        }

        found_value_len = mt_node->value_len;
        found_field_mask = mt_node->field_mask;
        found_field_count = mt_node->field_count;
        found_field_size = node_field_size(mt_node);
        found_vector_id = mt_node->vector_id;

        found_value = malloc(found_value_len);
        if (!found_value) {
            result = LV_ERR_OOM;
            goto cleanup;
        }
        memcpy(found_value, node_access_value(mt_node), found_value_len);

        if (found_field_count > 0) {
            found_field = malloc(found_field_size);
            if (!found_field) {
                result = LV_ERR_OOM;
                goto cleanup;
            }
            memcpy(found_field, node_access_field(mt_node, 0), found_field_size);
        }
    }

    else if (db->sst_fd < 0) {
        result = LV_ERR_NOT_FOUND;
        goto cleanup;
    }

    else {
        LVSSTIndexBlockEntry entry;
        LVStatus sst_search_status = sst_search_index_block(db->sst_fd, &entry, key, key_len);

        if (sst_search_status != LV_OK) {
            result = sst_search_status;
            goto cleanup;
        }

        uint64_t record_head_size = 0;
        if ((result = sst_read_record_head(db->sst_fd, entry.offset, NULL, NULL, NULL, NULL, &found_value_len, &found_vector_id, &found_field_mask, &found_field_count, &found_field_size, &record_head_size)) != LV_OK) goto cleanup;

        found_value = malloc(found_value_len);
        if (!found_value) {
            result = LV_ERR_OOM;
            goto cleanup;
        }

        if (found_field_count > 0) {
            found_field = malloc(found_field_size);
            if (!found_field) {
                result = LV_ERR_OOM;
                goto cleanup;
            }
        }

        if ((result = sst_read_record_tail(db->sst_fd, entry.offset + record_head_size, NULL, 0, found_value, found_value_len, found_field, found_field_size, NULL)) != LV_OK) goto cleanup;
    }

    new_field_size = found_field_size;

    for (int offset = 0; offset < field_count; ++offset) {
        const LVMetaField* current_field = fields + offset;

        const LVMetaFieldHash* searched_hash = schema_search_field_hash(db->schema->field_hashes, current_field->name, strlen(current_field->name));
        if (!searched_hash) {
            result = LV_ERR_INVALID;
            goto cleanup;
        }

        if (searched_hash->type != current_field->type) {
            result = LV_ERR_INVALID;
            goto cleanup;
        }

        //new added field
        if (!(found_field_mask & searched_hash->mask)) {
            new_fields_offsets[field_count_to_add] = offset;
            new_field_masks[field_count_to_add] = searched_hash->mask;
            field_count_to_add += 1;
            field_mask_to_add |= searched_hash->mask;

            new_field_size += sizeof(uint8_t);

            if (current_field->type == LV_META_STRING) {
                new_field_size += sizeof(uint32_t) + current_field->value.str.len;
            }
            else if (current_field->type == LV_META_FLOAT) {
                new_field_size += sizeof(double);
            }
            else {
                new_field_size += sizeof(int64_t);
            }
        }
        else {
            update_fields_offsets[field_count_to_update] = offset;
            field_mask_to_update |= searched_hash->mask;
            field_count_to_update += 1;

            // string update may change length; int/float are fixed 8 bytes so no adjustment.
            if (current_field->type == LV_META_STRING) {
                int old_number = node_field_number_of_mask(found_field_mask, searched_hash->mask);
                char* old_field_ptr = (char*)node_field_buffer_access(found_field, old_number);
                old_field_ptr += sizeof(uint8_t);
                uint32_t old_len = 0;
                memcpy(&old_len, old_field_ptr, sizeof(uint32_t));

                new_field_size -= old_len;
                new_field_size += current_field->value.str.len;

            }

        }
    }

    new_field = malloc(new_field_size);
    if (!new_field) {
        result = LV_ERR_OOM;
        goto cleanup;
    }

    new_field_mask = found_field_mask | field_mask_to_add;
    new_field_count = found_field_count + field_count_to_add;

    new_field_accrued_offsets = (LVSize32_t*)malloc(new_field_count * sizeof(LVSize32_t));
    if (!new_field_accrued_offsets) {
        result = LV_ERR_OOM;
        goto cleanup;
    }

    LVCount32_t upd_cnt = 0;
    LVCount32_t new_cnt = 0;
    LVSize32_t acc_offset = 0;
    for (int new_field_number = 0; new_field_number < new_field_count; ++new_field_number) {
        uint32_t current_field_mask = node_field_number_to_mask(new_field_mask, new_field_number); // field mask is always same.
        LVMetaType current_type = LV_META_FLOAT;
        uint32_t current_str_len = 0;

        if (current_field_mask & field_mask_to_update) {
            LVMetaField* upd_field = fields + update_fields_offsets[upd_cnt];

            current_type = upd_field->type;
            if (current_type == LV_META_STRING) {
                current_str_len = upd_field->value.str.len;
            }
            ++upd_cnt;
        }

        else if (current_field_mask & field_mask_to_add) {
            LVMetaField* new_field = fields + new_fields_offsets[new_cnt];

            current_type = new_field->type;
            if (current_type == LV_META_STRING) {
                current_str_len = new_field->value.str.len;
            }
            ++new_cnt;
        }

        else {
            int prev_field_number = node_field_number_of_mask(found_field_mask, current_field_mask);
            char* prev_field_ptr = node_field_buffer_access(found_field, prev_field_number);

            uint8_t saved_type;
            memcpy(&saved_type, prev_field_ptr, sizeof(uint8_t));
            current_type = (LVMetaType)saved_type;
            prev_field_ptr += sizeof(uint8_t);
            if (current_type == LV_META_STRING) {
                memcpy(&current_str_len, prev_field_ptr, sizeof(uint32_t));
            }
        }

        new_field_accrued_offsets[new_field_number] = acc_offset;

        if (current_type == LV_META_STRING) {
            acc_offset += sizeof(uint8_t) + sizeof(uint32_t) + current_str_len;
        }
        else if (current_type == LV_META_FLOAT) {
            acc_offset += sizeof(uint8_t) + sizeof(double);
        }
        else {
            acc_offset += sizeof(uint8_t) + sizeof(int64_t);
        }

    }

    LVCount32_t update_done_count = 0;
    for (int i = 0; i < found_field_count; ++i) {
        uint32_t current_field_mask = node_field_number_to_mask(new_field_mask, i);
        int prev_field_number = node_field_number_of_mask(found_field_mask, current_field_mask);
        char* prev_field_ptr = (char*)node_field_buffer_access(found_field, prev_field_number);

        int new_field_number = node_field_number_of_mask(new_field_mask, current_field_mask);
        char* new_field_ptr = (char*)new_field + new_field_accrued_offsets[new_field_number];

        if (current_field_mask & field_mask_to_update && update_done_count < field_count_to_update) {
            LVMetaField* update_field_data = fields + update_fields_offsets[update_done_count];

            uint8_t type_to_save = (uint8_t)update_field_data->type;
            memcpy(new_field_ptr, &type_to_save, sizeof(uint8_t));
            new_field_ptr += sizeof(uint8_t);


            if (update_field_data->type == LV_META_STRING) {
                memcpy(new_field_ptr, &update_field_data->value.str.len, sizeof(uint32_t));
                new_field_ptr += sizeof(uint32_t);
                memcpy(new_field_ptr, update_field_data->value.str.string, update_field_data->value.str.len);
            }
            else if (update_field_data->type == LV_META_FLOAT) {
                memcpy(new_field_ptr, &update_field_data->value.f64, sizeof(double));
            }
            else {
                memcpy(new_field_ptr, &update_field_data->value.i64, sizeof(int64_t));
            }

            update_done_count += 1;
        }
        else {
            uint8_t saved_type;

            memcpy(&saved_type, prev_field_ptr, sizeof(uint8_t));
            memcpy(new_field_ptr, &saved_type, sizeof(uint8_t));

            prev_field_ptr += sizeof(uint8_t);
            new_field_ptr += sizeof(uint8_t);

            LVMetaType type = (LVMetaType)saved_type;

            if (type == LV_META_STRING) {
                uint32_t len = 0;
                memcpy(&len, prev_field_ptr, sizeof(uint32_t));
                memcpy(new_field_ptr, &len, sizeof(uint32_t)); //length

                prev_field_ptr += sizeof(uint32_t);
                new_field_ptr += sizeof(uint32_t);

                memcpy(new_field_ptr, prev_field_ptr, len);
            }
            else if (type == LV_META_FLOAT) {
                memcpy(new_field_ptr, prev_field_ptr, sizeof(double));
            }
            else {
                memcpy(new_field_ptr, prev_field_ptr, sizeof(int64_t));
            }
        }
    }

    for (int i = 0; i < field_count_to_add; ++i) {
        int field_number = node_field_number_of_mask(new_field_mask, new_field_masks[i]);
        char* field_ptr = (char*)new_field + new_field_accrued_offsets[field_number];
        LVMetaField* new_field_data = fields + new_fields_offsets[i];

        uint8_t type_to_save = (uint8_t)new_field_data->type;
        memcpy(field_ptr, &type_to_save, sizeof(uint8_t));
        field_ptr += sizeof(uint8_t);

        if (new_field_data->type == LV_META_STRING) {
            memcpy(field_ptr, &new_field_data->value.str.len, sizeof(uint32_t));
            field_ptr += sizeof(uint32_t);
            memcpy(field_ptr, new_field_data->value.str.string, new_field_data->value.str.len);
        }
        else if (new_field_data->type == LV_META_FLOAT) {
            memcpy(field_ptr, &new_field_data->value.f64, sizeof(double));
        }
        else {
            memcpy(field_ptr, &new_field_data->value.i64, sizeof(int64_t));
        }
    }

    result = lv_put_internal(db, LV_UPDATE, db->next_seq, found_vector_id, key, key_len, found_value, found_value_len, NULL, new_field_mask, new_field_count, new_field_size, new_field);

cleanup:
    free(found_value);
    free(found_field);
    free(new_field);
    free(new_field_accrued_offsets);

    return result;
}

LVStatus lv_delete(LightVec* db, const void* key, const LVKeyLen32_t key_len) {
    LVStatus result = LV_OK;

    if ((result = lv_check_db_corruption_internal(db)) != LV_OK) goto _return;

    LVVectorId64_t found_vector_id = LV_NO_VECTOR_ID;

    LVNode* mt_node = table_search(db->memtable, key, key_len);

    if (mt_node) {
        found_vector_id = mt_node->vector_id;
    }

    else if (db->sst_fd < 0) {
        result = LV_ERR_NOT_FOUND;
        goto _return;
    }

    else if (db->sst_fd >= 0) {
        LVSSTIndexBlockEntry entry;
        LVStatus sst_search_result = sst_search_index_block(db->sst_fd, &entry, key, key_len);

        if (sst_search_result == LV_OK) {
            found_vector_id = entry.vector_id;
            free(entry.key);
        }
        else {
            result = sst_search_result; //LV_ERR_NOT_FOUND or something else
            goto _return;
        }
    }

    if ((result = lv_put_internal(db, LV_DELETE, db->next_seq, LV_NO_VECTOR_ID, key, key_len, NULL, 0, NULL, 0, 0, 0, NULL)) != LV_OK) goto _return;

    if (found_vector_id != LV_NO_VECTOR_ID) {
        vector_hnsw_mark_deleted(db->hnsw, found_vector_id);
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

    const int is_limit_on = option && (option->flags & LV_QOPT_LIMIT) && option->limit > 0;
    const int is_top_k_on = option && option->top_k > 0;
    const int is_score_filter_on = option && query_vector && (option->flags & LV_QOPT_SCORE_FILTER);
    const int is_ordby_on = (option && (option->flags & LV_QOPT_ORDER_BY));
    const int is_ordby_vec = is_ordby_on && query_vector && ((strncasecmp(option->order.by, "vector", strlen("vector")) == 0) || is_top_k_on);
    const int needs_hnsw = is_score_filter_on || is_ordby_vec || is_top_k_on;

    uint32_t ordby_field_mask = 0;
    LVOrdbyType ordbytype = LV_ORDBY_NONE;

    if (is_top_k_on) {
        ordbytype = LV_ORDBY_VEC;
    }

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

    if (needs_hnsw) {
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

        if ((result = vector_hnsw_query(db->hnsw, db->schema, query_tree, query_vector, &hnsw_qctx)) != LV_OK) goto _return;
        if ((result = lv_merge_qvsets_internal(merged_qvset, memtable_qvset, sst_qvset)) != LV_OK) goto _return;
    }
    else {
        if ((result = table_query_filter_scan(db->memtable, db->schema, query_tree, query_field_mask, ordbytype, ordby_field_mask, lv_qvset_light_append_internal, memtable_qvset)) != LV_OK) goto _return;
        if (db->sst_fd >= 0) {
            if ((result = sst_query_filter_scan(db->sst_fd, db->schema, query_tree, query_field_mask, ordbytype, ordby_field_mask, lv_qvset_append_internal, sst_qvset)) != LV_OK) goto _return;
        }

        if ((result = lv_merge_qvsets_internal(merged_qvset, memtable_qvset, sst_qvset)) != LV_OK) goto _return;
    }


    if (is_score_filter_on) {
        lv_apply_score_filter_internal(merged_qvset, option->vector_score_filter.score, option->vector_score_filter.bound);
    }

    if (is_ordby_on || is_top_k_on) {
        lv_apply_ordby_internal(merged_qvset, ordbytype, is_top_k_on ? LV_ORDER_DESC : option->order.dir);
    }

    if (is_limit_on || is_top_k_on) {
        LVSize32_t limit = option->limit;
        if (is_top_k_on && is_limit_on) {
            limit = option->top_k < option->limit ? option->top_k : option->limit;
        }
        else if (!is_limit_on && is_top_k_on) {
            limit = option->top_k;
        }

        lv_apply_limit_internal(merged_qvset, limit);
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

//must be called after open wal, sst, vectors
static LVStatus lv_recover_internal(LightVec* db) {
    LVStatus result = LV_OK;

    LVSeq64_t next_seq_from_wal = 0;
    LVVectorId64_t next_vector_id_from_wal = 0;
    LVSeq64_t next_seq_from_sst = 0;
    LVVectorId64_t next_vector_id_from_sst = 0;

    struct stat wal_st;
    wal_st.st_size = 0;
    struct stat sst_st;
    sst_st.st_size = 0;
    struct stat vectors_st;
    vectors_st.st_size = 0;

    if (fstat(db->wal_fd, &wal_st) != 0) {
        result = LV_ERR_IO;
        goto _return;
    }

    if (db->sst_fd >= 0) {
        if (fstat(db->sst_fd, &sst_st) != 0) {
            result = LV_ERR_IO;
            goto _return;
        }
    }

    if (fstat(db->vectors_fd, &vectors_st) != 0) {
        result = LV_ERR_IO;
        goto _return;
    }

    //no need to recover
    if (wal_st.st_size <= 0 && sst_st.st_size <= 0) goto _return;

    if (wal_st.st_size > 0) {
        if ((result = wal_recover(db->wal_fd, db->memtable, &next_seq_from_wal, &next_vector_id_from_wal)) != LV_OK) goto _return;

        //recover hnsw from wal
        LVNode* current_node = db->memtable->head->levels[0];
        while (current_node->type != LV_NODE_TAIL) {
            //tombstone node's vector id is LV_NO_VECTOR_ID
            //it will filter tombstone nodes
            if (current_node->vector_id != LV_NO_VECTOR_ID) {
                if (db->schema->vector_type == LV_VEC_FLOAT32) {
                    float vector[db->schema->vector_dim];
                    if ((result = vector_read_f32_vector(db->vectors_fd, current_node->vector_id, db->schema->vector_dim, vector)) != LV_OK) goto _return;
                    if ((result = vector_hnsw_f32_insert(db->hnsw, current_node->vector_id, vector, db->schema->vector_metric == LV_METRIC_L2 ? vector_f32_l2_sq : vector_f32_dot)) != LV_OK) goto _return;
                }
                else {
                    int8_t vector[db->schema->vector_dim];
                    if ((result = vector_read_i8_vector(db->vectors_fd, current_node->vector_id, db->schema->vector_dim, vector)) != LV_OK) goto _return;
                    if ((result = vector_hnsw_i8_insert(db->hnsw, current_node->vector_id, vector, db->schema->vector_metric == LV_METRIC_L2 ? vector_i8_l2_sq : vector_i8_dot)) != LV_OK) goto _return;
                }
                vector_hnsw_link_memtable_node(db->hnsw, current_node->vector_id, current_node);
            }


            current_node = table_get_next_node(current_node);
        }
    }

    if (db->sst_fd >= 0 && sst_st.st_size > 0) {
        uint64_t saved_index_block_offset = 0;
        uint64_t saved_record_count = 0;

        if ((result = sst_read_footer(db->sst_fd, &saved_index_block_offset, &saved_record_count, &next_seq_from_sst, &next_vector_id_from_sst)) != LV_OK) goto _return;

        //recover hnsw from sst
        LVSSTIndexBlockEntry entry;
        uint64_t read_offset = saved_index_block_offset;
        uint64_t record_read = 0;
        while (record_read < saved_record_count) {
            uint64_t next_offset = 0;
            if ((result = sst_read_index_entry_at_offset(db->sst_fd, read_offset, &entry, &next_offset)) != LV_OK) goto _return;
            if (entry.vector_id != LV_NO_VECTOR_ID) {
                if (table_search(db->memtable, entry.key, entry.key_len) != NULL) {
                    // skip (latest node is already proccessed)
                }
                else {
                    if (db->schema->vector_type == LV_VEC_FLOAT32) {
                        float vector[db->schema->vector_dim];
                        if ((result = vector_read_f32_vector(db->vectors_fd, entry.vector_id, db->schema->vector_dim, vector)) != LV_OK) {
                            free(entry.key);
                            goto _return;
                        };
                        if ((result = vector_hnsw_f32_insert(db->hnsw, entry.vector_id, vector, db->schema->vector_metric == LV_METRIC_L2 ? vector_f32_l2_sq : vector_f32_dot)) != LV_OK) {
                            free(entry.key);
                            goto _return;
                        };
                    }
                    else {
                        int8_t vector[db->schema->vector_dim];
                        if ((result = vector_read_i8_vector(db->vectors_fd, entry.vector_id, db->schema->vector_dim, vector)) != LV_OK) {
                            free(entry.key);
                            goto _return;
                        };
                        if ((result = vector_hnsw_i8_insert(db->hnsw, entry.vector_id, vector, db->schema->vector_metric == LV_METRIC_L2 ? vector_i8_l2_sq : vector_i8_dot)) != LV_OK) {
                            free(entry.key);
                            goto _return;
                        };
                    }

                    vector_hnsw_mark_flushed(db->hnsw, entry.vector_id);
                }
            }

            free(entry.key);
            ++record_read;
            read_offset = next_offset;
        }
    }

    db->next_seq = next_seq_from_wal > next_seq_from_sst ? next_seq_from_wal : next_seq_from_sst;
    db->next_vector_id = next_vector_id_from_wal > next_vector_id_from_sst ? next_vector_id_from_wal : next_vector_id_from_sst;

_return:
    return result;
}

static LVStatus lv_check_db_corruption_internal(const LightVec* db) {
    return db->magic == LV_MAGIC ? LV_OK : LV_ERR_CORRUPT;
}

static LVStatus lv_put_internal(LightVec* db, const LVNodeOp op, const LVSeq64_t current_seq, const LVVectorId64_t current_vector_id, const void* key, const LVKeyLen32_t key_len, const void* value, const LVValueLen32_t value_len, const void* vector, const LVSize32_t field_mask, const LVSize32_t field_count, const LVSize32_t field_size, const void* memory_field_buffer) {
    LVStatus result = LV_OK;

    // append to vectors.lv and hnsw
    if (vector)
    {
        if (db->schema->vector_type == LV_VEC_FLOAT32)
        {
            const float* f32_vector = (float*)vector;
            if ((result = vector_write_f32_vector(db->vectors_fd, current_vector_id, db->schema->vector_dim, f32_vector)) != LV_OK)
            {
                return result;
            }
            if ((result = vector_hnsw_f32_insert(db->hnsw, current_vector_id, f32_vector, db->schema->vector_metric == LV_METRIC_L2 ? vector_f32_l2_sq : vector_f32_dot)) != LV_OK)
            {
                return result;
            }
        }
        else
        {
            const int8_t* i8_vector = (int8_t*)vector;
            if ((result = vector_write_i8_vector(db->vectors_fd, current_vector_id, db->schema->vector_dim, i8_vector)) != LV_OK)
            {
                return result;
            }
            if ((result = vector_hnsw_i8_insert(db->hnsw, current_vector_id, i8_vector, db->schema->vector_metric == LV_METRIC_L2 ? vector_i8_l2_sq : vector_i8_dot)) != LV_OK)
            {
                return result;
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

    LVLevel8_t new_level = 1;

    while ((xorshift() % 100 <= LV_SKIPLIST_P) && new_level < LV_SKIPLIST_MAX_LEVEL)
    {
        ++new_level;
    }

    // append to WAL

    if (field_size > 0) {
        char wal_field_buffer[field_size];
        schema_field_memmory_to_disk(memory_field_buffer, field_size, wal_field_buffer);

        if ((result = wal_append(db->wal_fd, op, current_seq, new_level, KEY_LEN, KEY, VALUE_LEN, VALUE, vector ? current_vector_id : LV_NO_VECTOR_ID, field_mask, field_count, field_size, wal_field_buffer)) != LV_OK)
        {
            return result;
        }
    }
    else {
        if ((result = wal_append(db->wal_fd, op, current_seq, new_level, KEY_LEN, KEY, VALUE_LEN, VALUE, vector ? current_vector_id : LV_NO_VECTOR_ID, 0, 0, 0, NULL)) != LV_OK)
        {
            return result;
        }
    }

    // append to MemTable

    const LVNode* inserted_memtable_node = table_insert(db->memtable, op, current_seq, new_level, KEY_LEN, KEY, VALUE_LEN, VALUE, vector ? current_vector_id : LV_NO_VECTOR_ID, field_mask, field_count, field_size, memory_field_buffer);

    if (!inserted_memtable_node)
    {
        result = LV_ERR_OOM;
        return result;
    }

    if (vector) {
        vector_hnsw_link_memtable_node(db->hnsw, current_vector_id, inserted_memtable_node);
    }

    db->next_seq += 1;

    if (db->memtable->node_count >= db->flush_threshold) {
        result = lv_flush_internal(db);
    }

    return result;
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
        if (current->vector_id != LV_NO_VECTOR_ID) {
            vector_hnsw_mark_flushed(db->hnsw, current->vector_id);
        }
        current = current->levels[0];
    }

    // memtable reset
    destroy_table(db->memtable);

    db->memtable = create_table();
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

static LVStatus lv_qvset_append_internal(LVQVSet* qvset, const LVSeq64_t node_seq, const LVVectorId64_t vector_id, const void* key, const LVKeyLen32_t key_len, const void* value, const LVValueLen32_t value_len, const float vector_score, const LVOrdbyValue ordbyvalue, const int is_tombstone) {
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
    qvset->values[index].is_tombstone = is_tombstone;

    qvset->size += 1;
_return:
    return result;
}

static LVStatus lv_qvset_light_append_internal(LVQVSet* qvset, const LVSeq64_t node_seq, const LVVectorId64_t vector_id, const void* key, const LVKeyLen32_t key_len, const void* value, const LVValueLen32_t value_len, const float vector_score, const LVOrdbyValue ordbyvalue, const int is_tombstone) {
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
    qvset->values[index].is_tombstone = is_tombstone;

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

    LVSize32_t memtable_pointer = 0;
    LVSize32_t sst_pointer = 0;

    while (memtable_pointer < memtable_qvset->size && sst_pointer < sst_qvset->size)
    {
        const void* current_mtable_key = memtable_qvset->values[memtable_pointer].key;
        const LVKeyLen32_t current_mtable_klen = memtable_qvset->values[memtable_pointer].key_len;
        const void* current_sst_key = sst_qvset->values[sst_pointer].key;
        const LVKeyLen32_t current_sst_klen = sst_qvset->values[sst_pointer].key_len;

        if (node_key_equal(current_mtable_key, current_mtable_klen, current_sst_key, current_sst_klen)) { //memtable has the latest node
            if (memtable_qvset->values[memtable_pointer].is_tombstone == 1) {
                //drop both
            }
            else {
                if ((result = lv_qvset_light_append_internal(result_qvset,
                    memtable_qvset->values[memtable_pointer].node_seq,
                    memtable_qvset->values[memtable_pointer].vector_id,
                    current_mtable_key,
                    current_mtable_klen,
                    memtable_qvset->values[memtable_pointer].value,
                    memtable_qvset->values[memtable_pointer].value_len,
                    memtable_qvset->values[memtable_pointer].vector_score,
                    memtable_qvset->values[memtable_pointer].ordbyvalue,
                    0
                )) != LV_OK) goto _return;
            }
            ++memtable_pointer;
            ++sst_pointer;
        }
        else {
            if (node_cmp(LV_NODE_DATA, current_mtable_key, current_mtable_klen, memtable_qvset->values[memtable_pointer].node_seq,
                LV_NODE_DATA, current_sst_key, current_sst_klen, sst_qvset->values[sst_pointer].node_seq
            ) < 0) {
                if (memtable_qvset->values[memtable_pointer].is_tombstone == 1) {
                    //drop
                }
                else {
                    if ((result = lv_qvset_light_append_internal(result_qvset,
                        memtable_qvset->values[memtable_pointer].node_seq,
                        memtable_qvset->values[memtable_pointer].vector_id,
                        current_mtable_key,
                        current_mtable_klen,
                        memtable_qvset->values[memtable_pointer].value,
                        memtable_qvset->values[memtable_pointer].value_len,
                        memtable_qvset->values[memtable_pointer].vector_score,
                        memtable_qvset->values[memtable_pointer].ordbyvalue,
                        0
                    )) != LV_OK) goto _return;
                }
                ++memtable_pointer;
            }
            else {
                if ((result = lv_qvset_light_append_internal(result_qvset,
                    sst_qvset->values[sst_pointer].node_seq,
                    sst_qvset->values[sst_pointer].vector_id,
                    current_sst_key,
                    current_sst_klen,
                    sst_qvset->values[sst_pointer].value,
                    sst_qvset->values[sst_pointer].value_len,
                    sst_qvset->values[sst_pointer].vector_score,
                    sst_qvset->values[sst_pointer].ordbyvalue,
                    0
                )) != LV_OK) goto _return;
                ++sst_pointer;
            }
        }


    }


    while (memtable_pointer < memtable_qvset->size) {
        if (memtable_qvset->values[memtable_pointer].is_tombstone != 1) {
            if ((result = lv_qvset_light_append_internal(result_qvset,
                memtable_qvset->values[memtable_pointer].node_seq,
                memtable_qvset->values[memtable_pointer].vector_id,
                memtable_qvset->values[memtable_pointer].key,
                memtable_qvset->values[memtable_pointer].key_len,
                memtable_qvset->values[memtable_pointer].value,
                memtable_qvset->values[memtable_pointer].value_len,
                memtable_qvset->values[memtable_pointer].vector_score,
                memtable_qvset->values[memtable_pointer].ordbyvalue,
                0
            )) != LV_OK) goto _return;
        }
        ++memtable_pointer;
    }

    while (sst_pointer < sst_qvset->size) {
        if ((result = lv_qvset_light_append_internal(result_qvset,
            sst_qvset->values[sst_pointer].node_seq,
            sst_qvset->values[sst_pointer].vector_id,
            sst_qvset->values[sst_pointer].key,
            sst_qvset->values[sst_pointer].key_len,
            sst_qvset->values[sst_pointer].value,
            sst_qvset->values[sst_pointer].value_len,
            sst_qvset->values[sst_pointer].vector_score,
            sst_qvset->values[sst_pointer].ordbyvalue,
            0
        )) != LV_OK) goto _return;
        ++sst_pointer;
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
        }
        else {
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

LVStatus lv_close(LightVec* db) {
    if (!db) return LV_ERR_INVALID;
    if (lv_check_db_corruption_internal(db) != LV_OK) return LV_ERR_CORRUPT;

    write_helper_reset();
    if (db->schema_fd >= 0)       close(db->schema_fd);
    if (db->wal_fd >= 0)          close(db->wal_fd);
    if (db->sst_fd >= 0)          close(db->sst_fd);
    if (db->vectors_fd >= 0)       close(db->vectors_fd);
    if (db->vector_index_fd >= 0) close(db->vector_index_fd);

    destroy_table(db->memtable);
    destroy_schema(db->schema);
    destroy_hnsw(db->hnsw);
    free(db);

    return LV_OK;

}
