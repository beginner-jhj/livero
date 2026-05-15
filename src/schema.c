#include "schema.h"
#include <stdlib.h>
#include <string.h>
#include "util.h"
#include "helper.h"
#include "hash.h"
#include "crc.h"

LVSchema *create_schema(const LVDim32_t vector_dim, const LVVectorType vector_type, const LVCount32_t field_count, const LVMetaFieldDef *field_defs)
{
    int flag = 0;
    LVSchema *schema = NULL;

    if (vector_dim > LV_MAX_DIMENSION || field_count > LV_MAX_META_FIELDS)
    {
        goto cleanup;
    }

    LVSchema *schema_temp = malloc(sizeof(LVSchema));
    if (!schema_temp)
    {
        flag = 1;
        goto cleanup;
    }

    schema = schema_temp;

    schema->vector_dim = vector_dim;
    schema->vector_type = vector_type;

    schema->field_count = field_count;

    schema->next_field_mask = 1;
    schema->total_field_mask = 0;

    memset(schema->field_hashes, 0, sizeof(schema->field_hashes));

    static const char *LV_RESERVED_NAMES[] = {
        "VECTOR",
        "AND",
        "OR",
        "==",
        "!=",
        "<=",
        ">=",
        NULL // sentinel
    };

    for (int i = 0; i < schema->field_count; ++i)
    {
        const LVMetaFieldDef *current_def = field_defs + i;

        if (strlen(current_def->name) > LV_META_NAME_MAX)
        {
            flag = 1;
            goto cleanup;
        }

        for (int k = 0; LV_RESERVED_NAMES[i] != NULL; ++k)
        {
            if (strncasecmp(current_def->name, LV_RESERVED_NAMES[k],
                            strlen(LV_RESERVED_NAMES[k]) + 1) == 0)
            {
                flag = 1;
                goto cleanup;
            }
        }

        for (int j = 0; j < strlen(current_def->name); ++j) // check name is valid
        {
            if (!isalnum(current_def->name[j]) && current_def->name[j] != '_')
            {
                flag = 1;
                goto cleanup;
            }
        }

        schema->field_defs[i] = *current_def;

        if (schema_insert_field_hash(schema->field_hashes, current_def->name, current_def->type, schema->next_field_mask) != LV_OK)
        {
            flag = 1;
            goto cleanup;
        }

        schema->total_field_mask |= schema->next_field_mask;
        schema->next_field_mask <<= 1;
    }

cleanup:
    if (flag)
    {
        destroy_schema(schema);
    }
    return schema;
}

void schema_destroy_field_hashes(LVMetaFieldHash **hashes)
{
    for (int i = 0; i < LV_MAX_META_FIELDS; ++i)
    {
        LVMetaFieldHash *current = hashes[i];
        if (!current)
        {
            continue;
        }
        else
        {
            LVMetaFieldHash *current_next = current->next;
            while (current_next)
            {
                LVMetaFieldHash *tmp = current_next->next;
                free(current_next);
                current_next = tmp;
            }
            free(current);
        }
    }
}

void destroy_schema(LVSchema *schema)
{
    if (schema)
    {
        schema_destroy_field_hashes(schema->field_hashes);
        safe_free(&schema);
    }
}

LVStatus schema_write(const int fd, const LVSchema *schema)
{
    LVStatus result = LV_OK;

    uint8_t BUF_32[4];
    memset(BUF_32, 0, 4);

    if (schema->vector_dim > 4096)
    {
        result = LV_ERR_INVALID;
        goto _return;
    }

    if (schema->field_count > LV_MAX_META_FIELDS)
    {
        result = LV_ERR_FULL;
        goto _return;
    }

    uint32_t checksum = 0;

    // write magic
    if ((result = write_helper(fd, LV_MAGIC_SCHEMA, LV_MAGIC_SIZE)) != LV_OK)
    {
        goto _return;
    }

    checksum = crc_calc(LV_MAGIC_SCHEMA, LV_MAGIC_SIZE, CRC32_SEED);

    // write version
    uint32_t version_to_save = (uint32_t)LV_FORMAT_VERSION;
    put_fixed_32(BUF_32, version_to_save);
    checksum = crc_calc(BUF_32, sizeof(uint32_t), checksum);

    if ((result = write_helper(fd, BUF_32, sizeof(uint32_t))) != LV_OK)
    {
        goto _return;
    }

    // write vector dim
    uint32_t dim_to_save = schema->vector_dim;
    put_fixed_32(BUF_32, dim_to_save);
    checksum = crc_calc(BUF_32, sizeof(uint32_t), checksum);

    if ((result = write_helper(fd, BUF_32, sizeof(uint32_t))) != LV_OK)
    {
        goto _return;
    }

    // write vector type
    uint8_t vtype_to_save = (uint8_t)schema->vector_type;
    checksum = crc_calc(&vtype_to_save, 1, checksum);

    if ((result = write_helper(fd, &vtype_to_save, 1)) != LV_OK)
    {
        goto _return;
    }

    // write field count
    uint32_t fcount_to_save = schema->field_count;
    put_fixed_32(BUF_32, fcount_to_save);
    checksum = crc_calc(BUF_32, sizeof(uint32_t), checksum);

    if ((result = write_helper(fd, BUF_32, sizeof(uint32_t))) != LV_OK)
    {
        goto _return;
    }

    // write field defs
    int count = 0;
    while (count < schema->field_count)
    {
        const LVMetaFieldDef *current_field = schema->field_defs + count;
        if ((result = write_helper(fd, current_field->name, LV_META_NAME_MAX)) != LV_OK)
        {
            goto _return;
        }
        checksum = crc_calc(schema->field_defs + count, LV_META_NAME_MAX, checksum);

        const uint8_t current_filed_type_to_save = (uint8_t)(current_field->type);
        if ((result = write_helper(fd, &current_filed_type_to_save, 1)) != LV_OK)
        {
            goto _return;
        }

        checksum = crc_calc(&current_filed_type_to_save, 1, checksum);

        const uint32_t hash = fnv1a_hash(current_field->name, strlen(current_field->name));
        put_fixed_32(BUF_32, hash);
        if ((result = write_helper(fd, BUF_32, sizeof(uint32_t))) != LV_OK)
        {
            goto _return;
        }

        checksum = crc_calc(BUF_32, sizeof(uint32_t), checksum);

        ++count;
    }

    // write checksum
    put_fixed_32(BUF_32, checksum);
    if ((result = write_helper(fd, BUF_32, sizeof(uint32_t))) != LV_OK)
    {
        goto _return;
    }

    result = write_helper_flush(fd, 1);
_return:
    return result;
}

LVStatus schema_read(const int fd, LVSchema *schema)
{
    LVStatus result = LV_OK;

    uint8_t BUF_32[4];
    memset(BUF_32, 0, 4);

    uint32_t checksum = 0;

    char schema_magic[LV_MAGIC_SIZE];

    if ((result = read_helper(fd, schema_magic, LV_MAGIC_SIZE)) != LV_OK)
    {
        goto _return;
    }

    checksum = crc_calc(schema_magic, LV_MAGIC_SIZE, CRC32_SEED);

    if (memcmp(schema_magic, LV_MAGIC_SCHEMA, LV_MAGIC_SIZE) != 0)
    {
        result = LV_ERR_CORRUPT;
        goto _return;
    }

    // read version
    if ((result = read_helper(fd, BUF_32, sizeof(uint32_t))) != LV_OK)
    {
        goto _return;
    }

    checksum = crc_calc(BUF_32, sizeof(uint32_t), checksum);

    uint32_t saved_version = get_fixed_32(BUF_32);
    if (saved_version != LV_FORMAT_VERSION)
    {
        result = LV_ERR_CORRUPT;
        goto _return;
    }

    // read vector dimension
    if ((result = read_helper(fd, BUF_32, sizeof(uint32_t))) != LV_OK)
    {
        goto _return;
    }

    checksum = crc_calc(BUF_32, sizeof(uint32_t), checksum);

    uint32_t saved_vdim = get_fixed_32(BUF_32);

    schema->vector_dim = saved_vdim;

    // read vector type
    uint8_t saved_vector_type = 0;
    if ((result = read_helper(fd, &saved_vector_type, 1)) != LV_OK)
    {
        goto _return;
    }

    checksum = crc_calc(&saved_vector_type, sizeof(uint8_t), checksum);

    schema->vector_type = (LVVectorType)saved_vector_type;

    // read field count
    if ((result = read_helper(fd, BUF_32, sizeof(uint32_t))) != LV_OK)
    {
        goto _return;
    }

    checksum = crc_calc(BUF_32, sizeof(uint32_t), checksum);

    uint32_t saved_fcount = get_fixed_32(BUF_32);

    if (saved_fcount > LV_MAX_META_FIELDS)
    {
        result = LV_ERR_CORRUPT;
        goto _return;
    }

    schema->field_count = saved_fcount;

    int count = 0;

    uint32_t next_filed_mask = 1;
    uint32_t total_filed_mask = 0;

    char saved_field_name[LV_META_NAME_MAX];
    memset(saved_field_name, 0, LV_META_NAME_MAX);
    while (count < saved_fcount)
    {
        if ((result = read_helper(fd, saved_field_name, LV_META_NAME_MAX)) != LV_OK)
        {
            goto _return;
        }

        checksum = crc_calc(saved_field_name, LV_META_NAME_MAX, checksum);

        memcpy(schema->field_defs[count].name, saved_field_name, LV_META_NAME_MAX);

        uint8_t saved_field_type = 0;
        if ((result = read_helper(fd, &saved_field_type, 1)) != LV_OK)
        {
            goto _return;
        }

        checksum = crc_calc(&saved_field_type, sizeof(uint8_t), checksum);

        schema->field_defs[count].type = (LVMetaType)saved_field_type;

        // read field name hash

        if ((result = read_helper(fd, BUF_32, sizeof(uint32_t))) != LV_OK)
        {
            goto _return;
        }

        const uint32_t saved_hash = get_fixed_32(BUF_32);
        const uint32_t expected_hash = fnv1a_hash(saved_field_name, strlen(saved_field_name));

        if (saved_hash != expected_hash)
        {
            result = LV_ERR_CORRUPT;
            goto _return;
        }

        checksum = crc_calc(BUF_32, sizeof(uint32_t), checksum);

        if ((result = schema_insert_field_hash(schema->field_hashes, saved_field_name, saved_field_type, next_filed_mask)) != LV_OK)
        {
            goto _return;
        }

        total_filed_mask |= next_filed_mask;
        next_filed_mask <<= 1;

        ++count;
    }

    schema->total_field_mask = total_filed_mask;
    schema->next_field_mask = next_filed_mask;

    // read checksum
    if ((result = read_helper(fd, BUF_32, sizeof(uint32_t))) != LV_OK)
    {
        goto _return;
    }

    uint32_t disk_checksum = get_fixed_32(BUF_32);

    if (checksum != disk_checksum)
    {
        result = LV_ERR_CORRUPT;
    }

_return:
    return result;
}

LVStatus schema_insert_field_hash(LVMetaFieldHash **hashes, const char *field_name, const LVMetaType type, const uint32_t mask)
{
    LVStatus result = LV_OK;
    const LVHash32_t hash = fnv1a_hash(field_name, strlen(field_name));

    const int index = hash % LV_MAX_META_FIELDS;

    if (hashes[index] == NULL)
    {
        LVMetaFieldHash *meta_hash = malloc(sizeof(LVMetaFieldHash));
        if (!meta_hash)
        {
            result = LV_ERR_FULL;
            goto _return;
        }
        meta_hash->hash = hash;
        meta_hash->next = NULL;

        memset(meta_hash->field_name, 0, LV_META_NAME_MAX);
        memcpy(meta_hash->field_name, field_name, strlen(field_name));
        meta_hash->field_name[strlen(field_name)] = '\0';
        meta_hash->type = type;
        meta_hash->mask = mask;

        hashes[index] = meta_hash;
    }
    else
    {
        LVMetaFieldHash *current = hashes[index];
        while (current->next)
        {
            current = current->next;
        }
        LVMetaFieldHash *meta_hash = malloc(sizeof(LVMetaFieldHash));
        if (!meta_hash)
        {
            result = LV_ERR_FULL;
            goto _return;
        }
        meta_hash->hash = hash;
        meta_hash->next = NULL;

        memset(meta_hash->field_name, 0, LV_META_NAME_MAX);
        memcpy(meta_hash->field_name, field_name, strlen(field_name));
        meta_hash->field_name[strlen(field_name)] = '\0';
        meta_hash->type = type;
        meta_hash->mask = mask;

        current->next = meta_hash;
    }

_return:
    return result;
}

LVMetaFieldHash *schema_search_field_hash(LVMetaFieldHash **hashes, const char *field_name, const LVSize32_t field_len)
{
    const LVHash32_t hash = fnv1a_hash(field_name, field_len);

    const int index = hash % LV_MAX_META_FIELDS;

    if (hashes[index] == NULL)
    {
        return NULL;
    }

    else
    {
        LVMetaFieldHash *current_field_hash = hashes[index];
        while (current_field_hash)
        {
            if (strcmp(current_field_hash->field_name, field_name) == 0)
            {
                return current_field_hash;
            }
            current_field_hash = current_field_hash->next;
        }
    }

    return NULL;
}
