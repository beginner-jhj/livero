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

struct LightVec
{
    char path[LV_PATH_MAX];
    int schema_fd;
    LVSchema *schema;

    // LSM-Tree memtable
    int wal_fd;
    LVMemTable *memtable;
    LVSeq64_t next_seq;

    // Vector
    int vector_fd;                 // vectors.lv (O(1) access)[cite: 1, 2]
    LVBigCount64_t next_vector_id; //
    int hnsw_index_fd;             // hnsw_index.lv
    int hnsw_graph_fd;             // hnsw_graph.lv
    LVHnsw *hnsw;

    int32_t magic;
};

LVStatus lv_open(LightVec **db, const LVSchema *schema, const char *path)
{
    int flag = 0;
    LVStatus result = LV_OK;
    LightVec *LV_DB = NULL;
    LVSchema *LV_SCHEMA = NULL;
    LVMemTable *LV_MTABLE = NULL;
    LVHnsw *LV_HNSW = NULL;

    LV_DB = malloc(sizeof(LightVec));

    if (!LV_DB)
    {
        flag = 1;
        result = LV_ERR_OOM;
        goto cleanup;
    }

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
        schema_fd = open(schema_path, O_RDONLY | O_CREAT, 0644); // 0644 (S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH):  owner can read/write, else only can read

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
        wal_fd = open(wal_path, O_RDONLY);

        if ((result = wal_recover(wal_fd, LV_MTABLE)) != LV_OK)
        {
            flag = 1;
            goto cleanup;
        }
    }
    else
    {
        wal_fd = open(wal_path, O_RDONLY | O_CREAT, 0644);
    }

    LV_DB->wal_fd = wal_fd;

    // vector
    // create a LVHnsw, and write vectors.lv header

    LV_HNSW = create_hnsw(schema->vector_type, schema->vector_type);
    if (!LV_HNSW)
    {
        flag = 1;
        goto cleanup;
    }

    LV_DB->hnsw = LV_HNSW;

    char vectors_path[LV_PATH_MAX];
    char hnsw_index_path[LV_PATH_MAX];
    char hnsw_graph_path[LV_PATH_MAX];

    // set vectors.lv path
    if ((result = path_join(vectors_path, LV_PATH_MAX, LV_DB->path, "vectors.lv")) != LV_OK)
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

    int vector_fd;
    int hnsw_index_fd = open(hnsw_index_path, O_RDONLY | O_CREAT, 0644);
    int hnsw_graph_fd = open(hnsw_graph_path, O_RDONLY | O_CREAT, 0644);

    if (vectors_exists)
    {
        // todo recover hnsw
        vector_fd = open(vectors_path, O_RDONLY);
    }

    else
    {
        vector_fd = open(vectors_path, O_RDONLY | O_CREAT, 0644);
        if ((result = vector_write_header(vector_fd, schema->vector_type, schema->vector_dim, 1)) != LV_OK)
        {
            flag = 1;
            goto cleanup;
        }
    }

    LV_DB->vector_fd = vector_fd;
    LV_DB->hnsw_index_fd = hnsw_index_fd;
    LV_DB->hnsw_graph_fd = hnsw_graph_fd;

cleanup:
    if (flag)
    {
        safe_free(&LV_MTABLE);
        safe_free(&LV_SCHEMA);
        safe_free(&LV_DB);
    }

    return result;
}

LVStatus lv_put(const LightVec *db, const void *key, const LVKeyLen32_t key_len, const void *value, const LVValueLen32_t value_len, const void *vector, const LVCount32_t field_count, const LVMetaField *fields)
{
    LVStatus result = LV_OK;

    // check db is properly initialized
    if (db->magic != LV_MAGIC)
    {
        result = LV_ERR_INVALID_DB;
        goto _return;
    }

    // check field count is valid

    if (field_count > db->schema->field_count)
    {
        result = LV_ERR_INVALID;
        goto _return;
    }

    uint32_t field_mask = 0;
    LVSize32_t field_size = 0;

    for (int i = 0; i < field_count; ++i)
    {
        LVMetaField *current_field = fields + i;
        LVMetaFieldHash *search_result = schema_search_field_hash(db->schema->field_hashes, current_field->name);
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

    // append to vectors.lv and hnsw
    if (vector)
    {
        if (db->schema->vector_type == LV_VEC_FLOAT32)
        {
            const float *f32_vector = (float *)vector;
            if ((result = vector_write_f32_vector(db->vector_fd, db->schema->vector_dim, f32_vector)) != LV_OK)
            {
                goto _return;
            }
            if ((result = vector_hnsw_f32_insert(db->hnsw, db->next_vector_id, f32_vector)) != LV_OK)
            {
                goto _return;
            }
        }
        else
        {
            const int8_t *i8_vector = (int8_t *)vector;
            if ((result = vector_write_i8_vector(db->vector_fd, db->schema->vector_dim, i8_vector)) != LV_OK)
            {
                goto _return;
            }
            if ((result = vector_hnsw_i8_insert(db->hnsw, db->next_vector_id, i8_vector)) != LV_OK)
            {
                goto _return;
            }
        }
    }

    else
    {
        const int8_t *i8_vector = (int8_t *)vector;
        if ((result = vector_write_i8_vector(db->vector_fd, db->schema->vector_dim, i8_vector)) != LV_OK)
        {
            goto _return;
        }
    }

    // append to WAL

    if ((result = wal_append(db->wal_fd, LV_WAL_PUT, db->next_seq, new_level, key_len, key, value_len, value, db->next_vector_id, field_mask, field_count, field_size, fields)) != LV_OK)
    {
        goto _return;
    }

    // append to MemTable

    if ((result = table_insert(db->memtable, LV_WAL_PUT, db->next_seq, new_level, key_len, key, value_len, value, db->next_vector_id, field_mask, field_count, field_size, fields)) != LV_OK)
    {
        goto _return;
    }

_return:
    return result;
}
