#include "wal.h"
#include "storage.h"
#include "node.h"
#include "schema.h"
#include "util.h"
#include "crc.h"
#include <unistd.h>
#include "helper.h"

LVStatus wal_append(const int fd, const LVNodeOp op, const LVSeq64_t seq, const LVLevel8_t level, const LVSize32_t key_len, const void *key, const LVSize32_t value_len, const void *value, const uint64_t vector_id, const uint32_t field_mask, const uint32_t field_count, const LVSize32_t field_size, const LVMetaField *field_list)
{
    LVStatus result = LV_OK;
    uint8_t BUF_32[4];
    uint8_t BUF_64[8];

    uint32_t checksum = 0;

    // write op
    uint8_t op_to_save = (uint8_t)op;

    if ((result = write_helper(fd, &op_to_save, 1)) != LV_OK)
    {
        goto _return;
    }

    checksum = crc_calc(&op_to_save, 1, checksum);

    // write seq
    put_fixed_64(BUF_64, seq);

    if ((result = write_helper(fd, BUF_64, sizeof(uint64_t))) != LV_OK)
    {
        goto _return;
    }

    checksum = crc_calc(BUF_64, sizeof(uint64_t), checksum);

    // write level

    if ((result = write_helper(fd, &level, 1)) != LV_OK)
    {
        goto _return;
    }

    checksum = crc_calc(&level, 1, checksum);

    // write key_len

    put_fixed_32(BUF_32, key_len);

    if ((result = write_helper(fd, BUF_32, sizeof(uint32_t))) != LV_OK)
    {
        goto _return;
    }

    checksum = crc_calc(BUF_32, sizeof(uint32_t), checksum);

    // DELETE op
    if (op == LV_DELETE)
    {
        if ((result = write_helper(fd, key, key_len)) != LV_OK)
        {
            goto _return;
        }

        checksum = crc_calc(key, key_len, checksum);
        put_fixed_32(BUF_32, checksum);
        result = write_helper(fd, BUF_32, sizeof(uint32_t));
        goto _return;
    }

    // write value_len

    put_fixed_32(BUF_32, value_len);

    if ((result = write_helper(fd, BUF_32, sizeof(uint32_t))) != LV_OK)
    {
        goto _return;
    }

    checksum = crc_calc(BUF_32, sizeof(uint32_t), checksum);

    // write vector_id

    put_fixed_64(BUF_64, vector_id);

    if ((result = write_helper(fd, BUF_64, sizeof(uint64_t))) != LV_OK)
    {
        goto _return;
    }

    checksum = crc_calc(BUF_64, sizeof(uint64_t), checksum);

    // write field mask

    put_fixed_32(BUF_32, field_mask);

    if ((result = write_helper(fd, BUF_32, sizeof(uint32_t))) != LV_OK)
    {
        goto _return;
    }

    checksum = crc_calc(BUF_32, sizeof(uint32_t), checksum);

    // write field count

    put_fixed_32(BUF_32, field_count);

    if ((result = write_helper(fd, BUF_32, sizeof(uint32_t))) != LV_OK)
    {
        goto _return;
    }

    checksum = crc_calc(BUF_32, sizeof(uint32_t), checksum);

    // write field total size

    put_fixed_32(BUF_32, field_size);
    if ((result = write_helper(fd, BUF_32, sizeof(uint32_t))) != LV_OK)
    {
        goto _return;
    }

    checksum = crc_calc(BUF_32, sizeof(uint32_t), checksum);

    // write key

    if ((result = write_helper(fd, key, key_len)) != LV_OK)
    {
        goto _return;
    }

    checksum = crc_calc(key, key_len, checksum);

    // write value

    if ((result = write_helper(fd, value, value_len)) != LV_OK)
    {
        goto _return;
    }

    checksum = crc_calc(value, value_len, checksum);

    // write field values
    int count = 0;

    while (count < field_count)
    {
        LVMetaField *current_field = field_list + count;
        uint8_t type_to_save = (uint8_t)(current_field->type);
        if ((result = write_helper(fd, &type_to_save, 1)) != LV_OK)
        {
            goto _return;
        }

        checksum = crc_calc(&type_to_save, 1, checksum);

        switch (current_field->type)
        {
        case LV_META_STRING:
            put_fixed_32(BUF_32, current_field->value.str.len);

            if ((result = write_helper(fd, BUF_32, sizeof(uint32_t))) != LV_OK)
            {
                goto _return;
            }

            checksum = crc_calc(BUF_32, sizeof(uint32_t), checksum);

            if ((result = write_helper(fd, current_field->value.str.string, current_field->value.str.len)) != LV_OK)
            {
                goto _return;
            }

            checksum = crc_calc(current_field->value.str.string, current_field->value.str.len, checksum);

            break;

        case LV_META_FLOAT:
        {

            uint64_t value;
            memcpy(&value, &current_field->value.f64, 8);

            put_fixed_64(BUF_64, value);

            if ((result = write_helper(fd, BUF_64, 8)) != LV_OK)
            {
                goto _return;
            }

            checksum = crc_calc(BUF_64, 8, checksum);

            break;
        }

        case LV_META_INT:
            put_fixed_64(BUF_64, current_field->value.i64);

            if ((result = write_helper(fd, BUF_64, 8)) != LV_OK)
            {
                goto _return;
            }

            checksum = crc_calc(BUF_64, 8, checksum);

            break;
        default:
            break;
        }

        ++count;
    }

    // write checksum

    put_fixed_32(BUF_32, checksum);

    if((result = write_helper(fd, BUF_32, sizeof(uint32_t))) != LV_OK){
        goto _return;
    }

    result = write_helper_flush(fd, 1);

_return:
    return result;
}

LVStatus wal_recover(const int fd, const LVMemTable *table)
{
    LVStatus result = LV_OK;
    off_t wal_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    off_t current_offset = 0;

    uint8_t saved_op;
    LVSeq64_t saved_seq;
    uint8_t saved_level;
    LVKeyLen32_t saved_key_len;
    LVValueLen32_t saved_value_len;
    LVVectorId64_t saved_vector_id;
    LVSize32_t saved_field_mask;
    LVSize32_t saved_field_count;
    LVSize32_t saved_field_size;

    LVSize32_t node_size_to_reserve = 0;

    while (current_offset < wal_size)
    {
        uint32_t checksum = CRC32_SEED;

        if ((result = wal_read_head(fd, &checksum, &saved_op, &saved_seq, &saved_level, &saved_key_len, &saved_value_len, &saved_vector_id, &saved_field_mask, &saved_field_count, &saved_field_size)) != LV_OK)
        {
            goto _return;
        }

        node_size_to_reserve = sizeof(LVNode) + saved_level * sizeof(LVNode *) + saved_key_len + saved_value_len + saved_field_size;

        LVNode *reserved_node = node_reserve(table->arena, node_size_to_reserve);

        reserved_node->type = LV_NODE_DATA;
        reserved_node->op = (LVNodeOp)saved_op;
        reserved_node->seq = saved_seq;
        reserved_node->level = saved_level;
        reserved_node->key_len = saved_key_len;
        reserved_node->value_len = saved_value_len;
        reserved_node->vector_id = saved_vector_id;
        reserved_node->field_count = saved_field_count;
        reserved_node->field_mask = saved_field_mask;

        memset(reserved_node->levels, 0, saved_level * sizeof(LVNode *));

        if (reserved_node->op == LV_DELETE)
        {
            if ((result = read_helper(fd, node_access_key(reserved_node), saved_key_len)) != LV_OK)
            {
                goto _return;
            }

            checksum = crc_calc(node_access_key(reserved_node), saved_key_len, checksum);

            uint8_t CHECKSUM_BUF[4];
            if ((result = read_helper(fd, CHECKSUM_BUF, 4)) != LV_OK)
            {
                goto _return;
            }

            uint32_t saved_checksum = get_fixed_32(CHECKSUM_BUF);

            if (saved_checksum != checksum)
            {
                result = LV_ERR_CORRUPT;
                goto _return;
            }
        }
        else
        {
            if ((result = wal_read_tail(fd, &checksum, node_access_key(reserved_node), saved_key_len, saved_value_len, saved_field_size)) != LV_OK)
            {
                goto _return;
            }
        }

        current_offset = lseek(fd, 0, SEEK_CUR);

        table_direct_insert(table, reserved_node);
    }

_return:
    return result;
}

LVStatus wal_read_head(const int fd, uint32_t *checksum, uint8_t *op, LVSeq64_t *seq, LVLevel8_t *level, LVKeyLen32_t *key_len, LVValueLen32_t *value_len, LVVectorId64_t *vector_id, LVSize32_t *field_mask, LVSize32_t *field_count, LVSize32_t *field_size)
{
    LVStatus result = LV_OK;

    uint8_t BUF_32[4];
    uint8_t BUF_64[8];

    // read op
    uint8_t op_tmp;
    if ((result = read_helper(fd, &op_tmp, 1)) != LV_OK)
    {
        goto _return;
    }

    *op = op_tmp;
    *checksum = crc_calc(op, 1, *checksum);

    // read seq
    if ((result = read_helper(fd, BUF_64, 8)) != LV_OK)
    {
        goto _return;
    }

    *checksum = crc_calc(BUF_64, 8, *checksum);
    *seq = get_fixed_64(BUF_64);

    // read level
    LVLevel8_t level_tmp;
    if ((result = read_helper(fd, &level_tmp, 1)) != LV_OK)
    {
        goto _return;
    }
    *level = level_tmp;
    *checksum = crc_calc(level, 1, *checksum);

    // read key len
    if ((result = read_helper(fd, BUF_32, 4)) != LV_OK)
    {
        goto _return;
    }
    *checksum = crc_calc(BUF_32, 4, *checksum);
    *key_len = get_fixed_32(BUF_32);

    // if it is a to be deleted node then skip else
    if (*op == LV_DELETE)
    {
        *value_len = 0;
        *vector_id = (uint64_t)-1;
        *field_mask = 0;
        *field_count = 0;
        *field_size = 0;
        goto _return;
    }

    // read value len
    if ((result = read_helper(fd, BUF_32, 4)) != LV_OK)
    {
        goto _return;
    }

    *checksum = crc_calc(BUF_32, 4, *checksum);
    *value_len = get_fixed_32(BUF_32);

    // read vector id
    if ((result = read_helper(fd, BUF_64, 8)) != LV_OK)
    {
        goto _return;
    }

    *checksum = crc_calc(BUF_64, 8, *checksum);
    *vector_id = get_fixed_64(BUF_64);

    // read field mask
    if ((result = read_helper(fd, BUF_32, 4)) != LV_OK)
    {
        goto _return;
    }

    *checksum = crc_calc(BUF_32, 4, *checksum);
    *field_mask = get_fixed_32(BUF_32);

    // read field count

    if ((result = read_helper(fd, BUF_32, 4)) != LV_OK)
    {
        goto _return;
    }

    *checksum = crc_calc(BUF_32, 4, *checksum);
    *field_count = get_fixed_32(BUF_32);

    // read field total size

    if ((result = read_helper(fd, BUF_32, 4)) != LV_OK)
    {
        goto _return;
    }

    *checksum = crc_calc(BUF_32, 4, *checksum);
    *field_size = get_fixed_32(BUF_32);

_return:
    return result;
}

LVStatus wal_read_tail(const int fd, uint32_t *checksum, void *ptr, LVKeyLen32_t key_len, LVValueLen32_t value_len, LVSize32_t field_total_size)
{
    LVStatus result = LV_OK;

    uint8_t BUF_32[4];
    uint8_t BUF_64[8];

    if ((result = read_helper(fd, ptr, key_len)) != LV_OK)
    {
        goto _return;
    }

    *checksum = crc_calc(ptr, key_len, *checksum);

    ptr = (char *)ptr + key_len;

    if ((result = read_helper(fd, ptr, value_len)) != LV_OK)
    {
        goto _return;
    }

    *checksum = crc_calc(ptr, value_len, *checksum);

    ptr = (char *)ptr + value_len;

    LVSize32_t current_field_read_size = 0;

    while (current_field_read_size < field_total_size)
    {
        uint8_t saved_type;
        if ((result = read_helper(fd, &saved_type, 1)) != LV_OK)
        {
            goto _return;
        }

        LVMetaType type = (LVMetaType)saved_type;
        memcpy(ptr, &type, sizeof(LVMetaType));

        *checksum = crc_calc(&saved_type, 1, *checksum);
        current_field_read_size += 1;
        ptr = (char *)ptr + sizeof(LVMetaType);

        switch (type)
        {
        case LV_META_STRING:
        {
            if ((result = read_helper(fd, BUF_32, 4)) != LV_OK)
            {
                goto _return;
            }
            *checksum = crc_calc(BUF_32, 4, *checksum);

            uint32_t saved_string_len = get_fixed_32(BUF_32);
            memcpy(ptr, &saved_string_len, sizeof(uint32_t));

            current_field_read_size += 4;
            ptr = (char *)ptr + sizeof(uint32_t);

            if (saved_string_len > 0)
            {
                if ((result = read_helper(fd, ptr, saved_string_len)) != LV_OK)
                {
                    goto _return;
                }
                *checksum = crc_calc(ptr, saved_string_len, *checksum);
                current_field_read_size += saved_string_len;
                ptr = (char *)ptr + saved_string_len;
            }

            break;
        }

        case LV_META_FLOAT:
        {
            if ((result = read_helper(fd, BUF_64, 8)) != LV_OK)
            {
                goto _return;
            }
            *checksum = crc_calc(BUF_64, 8, *checksum);
            uint64_t tmp = get_fixed_64(BUF_64);
            memcpy(ptr, &tmp, sizeof(double));

            current_field_read_size += 8;
            ptr = (char *)ptr + 8;
            break;
        }
        case LV_META_INT:
        {
            if ((result = read_helper(fd, BUF_64, 8)) != LV_OK)
            {
                goto _return;
            }
            *checksum = crc_calc(BUF_64, 8, *checksum);
            uint64_t tmp = get_fixed_64(BUF_64);
            memcpy(ptr, &tmp, sizeof(int64_t));

            current_field_read_size += 8;
            ptr = (char *)ptr + 8;
            break;
        }
        default:
            break;
        }
    }

    if ((result = read_helper(fd, BUF_32, 4)) != LV_OK)
    {
        goto _return;
    }

    uint32_t saved_checksum = get_fixed_32(BUF_32);

    if (*checksum != saved_checksum)
    {
        result = LV_ERR_CORRUPT;
        goto _return;
    }

_return:
    return result;
}
