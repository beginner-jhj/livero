#include "sst.h"
#include "helper.h"
#include "node.h"
#include "util.h"
#include "hash.h"

LVStatus sst_flush(const int new_fd, const int old_fd, const LVNode* node) {
    LVStatus result = LV_OK;
    uint8_t BUF_32[4];
    uint8_t BUF_64[8];

    LVSSTIndexBlockSet* index_set = malloc(sizeof(LVSSTIndexBlockSet));
    if (!index_set) {
        result = LV_ERR_OOM;
        goto _return;
    }

    index_set->capacity = LV_DEFAULT_CAPACITY;
    index_set->size = 0;
    index_set->entries = NULL;

    LVSSTIndexBlockEntry* entries = malloc(sizeof(LVSSTIndexBlockEntry) * index_set->capacity);
    if (!entries) {
        result = LV_ERR_OOM;
        goto _return;
    }

    index_set->entries = entries;

    //check old sst corruption
    if (old_fd >= 0) {
        char sst_magic[LV_MAGIC_SIZE];
        if ((result = read_helper(old_fd, sst_magic, LV_MAGIC_SIZE)) != LV_OK) goto _return;
        if (memcmp(sst_magic, LV_MAGIC_SST, LV_MAGIC_SIZE) != 0) {
            result = LV_ERR_CORRUPT;
            goto _return;
        }
    }

    //write magic on new sst
    if ((result = write_helper(new_fd, LV_MAGIC_SST, LV_MAGIC_SIZE)) != LV_OK) goto _return;

    //write data
    uint64_t record_count = 0;
    if (old_fd < 0) {
        LVNode* current_node = node;
        while (current_node->type != LV_NODE_TAIL) {
            if (current_node->op != LV_DELETE) {
                const uint64_t record_start_offset = write_helper_get_offset(new_fd);
                if ((result = sst_write_record_with_node(new_fd, current_node)) != LV_OK) goto _return;
                if ((result = sst_indexblockset_append(index_set, current_node->key_len, node_access_key(current_node), current_node->seq, record_start_offset)) != LV_OK) goto _return;
                record_count += 1;
            }
            current_node = current_node->levels[0];
        }
    }
    else {
        lseek(old_fd, -16, SEEK_END);
        //read index_block_offset
        if ((result = read_helper(old_fd, BUF_64, 8)) != LV_OK) goto _return;
        const uint64_t index_block_offset = get_fixed_64(BUF_64);

        //read record_count
        if ((result = read_helper(old_fd, BUF_64, 8)) != LV_OK) goto _return;
        const uint64_t saved_record_count = get_fixed_64(BUF_64);

        //go to index_block
        lseek(old_fd, index_block_offset, SEEK_SET);

        LVSSTIndexBlockEntry old_entry;
        if ((result = sst_read_next_index_entry(old_fd, &old_entry)) != LV_OK) goto _return;
        int has_old_entry = 1;

        uint64_t record_read = 0;
        LVNode* current_node = node;
        while (has_old_entry && current_node->type != LV_NODE_TAIL) {
            if (current_node->op == LV_DELETE) {
                current_node = current_node->levels[0];
                continue;
            }

            const int cmp_result = node_cmp(LV_NODE_DATA, old_entry.key, old_entry.key_len, old_entry.seq, LV_NODE_DATA, node_access_key(current_node), current_node->key_len, current_node->seq);

            const uint64_t record_start_offset = write_helper_get_offset(new_fd);

            if (cmp_result < 0) {
                if ((result = sst_write_record_with_old_sst(new_fd, old_fd, old_entry.offset)) != LV_OK) goto _return;
                if ((result = sst_indexblockset_append(index_set, old_entry.key_len, old_entry.key, old_entry.seq, record_start_offset)) != LV_OK) goto _return;
                record_read += 1;
                free(old_entry.key);
                if (record_read < saved_record_count) {
                    if ((result = sst_read_next_index_entry(old_fd, &old_entry)) != LV_OK) goto _return;
                    has_old_entry = 1;
                }
                else {
                    has_old_entry = 0;
                }
            }
            else {
                if ((result = sst_write_record_with_node(new_fd, current_node)) != LV_OK) goto _return;
                if ((result = sst_indexblockset_append(index_set, current_node->key_len, node_access_key(current_node), current_node->seq, record_start_offset)) != LV_OK) goto _return;
                current_node = current_node->levels[0];
            }
            record_count += 1;
        }

        while (current_node->type != LV_NODE_TAIL) {
            const uint64_t record_start_offset = write_helper_get_offset(new_fd);

            if ((result = sst_write_record_with_node(new_fd, current_node)) != LV_OK) goto _return;
            if ((result = sst_indexblockset_append(index_set, current_node->key_len, node_access_key(current_node), current_node->seq, record_start_offset)) != LV_OK) goto _return;

            current_node = current_node->levels[0];
        }

        while (has_old_entry) {
            const uint64_t record_start_offset = write_helper_get_offset(new_fd);

            if ((result = sst_write_record_with_old_sst(new_fd, old_fd, old_entry.offset)) != LV_OK) goto _return;
            if ((result = sst_indexblockset_append(index_set, old_entry.key_len, old_entry.key, old_entry.seq, record_start_offset)) != LV_OK) goto _return;

            record_read += 1;
            free(old_entry.key);

            if (record_read < saved_record_count) {
                if ((result = sst_read_next_index_entry(old_fd, &old_entry)) != LV_OK) goto _return;
                has_old_entry = 1;
            }
            else {
                has_old_entry = 0;
            }
        }
    }

    const uint64_t index_block_offset = write_helper_get_offset(new_fd);


    //write index_block
    for (int i = 0; i < index_set->size; ++i) {
        put_fixed_32(BUF_32, index_set->entries[i].key_len);
        if ((result = write_helper(new_fd, BUF_32, 4)) != LV_OK) goto _return;

        if ((result = write_helper(new_fd, index_set->entries[i].key, index_set->entries[i].key_len)) != LV_OK) goto _return;

        put_fixed_64(BUF_64, index_set->entries[i].seq);
        if ((result = write_helper(new_fd, BUF_64, 8)) != LV_OK) goto _return;

        put_fixed_64(BUF_64, index_set->entries[i].offset);
        if ((result = write_helper(new_fd, BUF_64, 8)) != LV_OK) goto _return;
    }


    //write footer
    put_fixed_64(BUF_64, index_block_offset);
    if ((result = write_helper(new_fd, BUF_64, 8)) != LV_OK) goto _return;

    put_fixed_64(BUF_64, record_count);
    if ((result = write_helper(new_fd, BUF_64, 8)) != LV_OK) goto _return;

_return:
    destroy_indexblockset(index_set);
    return result;
}

LVStatus sst_read_next_index_entry(const int fd, LVSSTIndexBlockEntry* entry) {
    LVStatus result = LV_OK;
    uint8_t BUF_32[4];
    uint8_t BUF_64[8];
    //read key_len
    if ((result = read_helper(fd, BUF_32, 4)) != LV_OK) return result;
    const LVKeyLen32_t saved_key_len = get_fixed_32(BUF_32);
    entry->key_len = saved_key_len;
    //read key
    char* saved_key = malloc(saved_key_len);
    if (!saved_key) {
        result = LV_ERR_OOM;
        return result;
    }
    if ((result = read_helper(fd, saved_key, saved_key_len)) != LV_OK) {
        free(saved_key);
        return result;
    };
    entry->key = saved_key;

    //read seq
    if ((result = read_helper(fd, BUF_64, 8)) != LV_OK) {
        free(saved_key);
        return result;
    };
    const LVSeq64_t saved_seq = get_fixed_64(BUF_64);
    entry->seq = saved_seq;

    //read offset
    if ((result = read_helper(fd, BUF_64, 8)) != LV_OK) {
        free(saved_key);
        return result;
    };
    const LVSeq64_t saved_offset = get_fixed_64(BUF_64);
    entry->offset = saved_offset;

    return result;
}

LVStatus sst_write_record_with_node(const int fd, const LVNode* node) {
    LVStatus result = LV_OK;

    uint8_t BUF_32[4];
    uint8_t BUF_64[8];

    //write seq
    put_fixed_64(BUF_64, node->seq);
    if ((result = write_helper(fd, BUF_64, 8)) != LV_OK) goto _return;

    //write op;
    uint8_t op_to_save = (uint8_t)node->op;
    if ((result = write_helper(fd, &op_to_save, sizeof(uint8_t))) != LV_OK) goto _return;

    //write key_len
    put_fixed_32(BUF_32, node->key_len);
    if ((result = write_helper(fd, BUF_32, sizeof(LVKeyLen32_t))) != LV_OK) goto _return;

    //write key
    if ((result = write_helper(fd, node_access_key(node), node->key_len)) != LV_OK) goto _return;

    //write value_len
    put_fixed_32(BUF_32, node->value_len);
    if ((result = write_helper(fd, BUF_32, sizeof(LVValueLen32_t))) != LV_OK) goto _return;

    //write value
    if ((result = write_helper(fd, node_access_value(node), node->value_len)) != LV_OK) goto _return;

    //write vector_id
    put_fixed_64(BUF_64, node->vector_id);
    if ((result = write_helper(fd, BUF_64, 8)) != LV_OK) goto _return;

    //write field mask
    put_fixed_32(BUF_32, node->field_mask);
    if ((result = write_helper(fd, BUF_32, 4)) != LV_OK) goto _return;

    //write field count
    put_fixed_32(BUF_32, node->field_count);
    if ((result = write_helper(fd, BUF_32, 4)) != LV_OK) goto _return;


    //write meta fields
    if (node->field_count > 0) {
        for (int i = 0; i < node->field_count; ++i) {
            void* field_ptr = node_access_field(node, i);
            LVMetaType type;
            memcpy(&type, field_ptr, sizeof(LVMetaType));

            uint8_t type_to_save = (uint8_t)type;
            if ((result = write_helper(fd, &type_to_save, sizeof(uint8_t))) != LV_OK) goto _return;

            field_ptr = (char*)field_ptr + sizeof(LVMetaType);

            switch (type)
            {
            case LV_META_FLOAT: {
                int64_t value = 0;
                memcpy(&value, field_ptr, 8);
                put_fixed_64(BUF_64, value);
                if ((result = write_helper(fd, BUF_64, 8)) != LV_OK) goto _return;
                break;
            }
            case LV_META_INT: {
                int64_t value = 0;
                memcpy(&value, field_ptr, 8);
                put_fixed_64(BUF_64, value);
                if ((result = write_helper(fd, BUF_64, 8)) != LV_OK) goto _return;

                break;
            }
            case LV_META_STRING: {
                uint32_t len = 0;
                memcpy(&len, field_ptr, sizeof(uint32_t));
                put_fixed_32(BUF_32, len);
                if ((result = write_helper(fd, BUF_32, 4)) != LV_OK) goto _return;

                field_ptr = (char*)field_ptr + sizeof(uint32_t);

                if ((result = write_helper(fd, field_ptr, len)) != LV_OK) goto _return;
                break;
            }
            default:
                break;
            }
        }
    }

_return:
    return result;
}

LVStatus sst_write_record_with_old_sst(const int new_fd, const int old_fd, const uint64_t read_offset) {
    LVStatus result = LV_OK;
    uint8_t BUF_32[4];
    uint8_t BUF_64[8];

    lseek(old_fd, read_offset, SEEK_SET);

    //seq
    if ((result = read_helper(old_fd, BUF_64, 8)) != LV_OK) return result;
    if ((result = write_helper(new_fd, BUF_64, 8)) != LV_OK) return result;


    //op
    uint8_t saved_op;
    if ((result = read_helper(old_fd, &saved_op, sizeof(uint8_t))) != LV_OK) return result;
    if ((result = write_helper(new_fd, &saved_op, sizeof(uint8_t))) != LV_OK) return result;

    //key_len
    if ((result = read_helper(old_fd, BUF_32, 4)) != LV_OK) return result;
    if ((result = write_helper(new_fd, BUF_32, 4)) != LV_OK) return result;
    const LVKeyLen32_t saved_key_len = get_fixed_32(BUF_32);


    //key
    char saved_key[saved_key_len];
    if ((result = read_helper(old_fd, saved_key, saved_key_len)) != LV_OK) return result;
    if ((result = write_helper(new_fd, saved_key, saved_key_len)) != LV_OK) return result;


    //value_len
    if ((result = read_helper(old_fd, BUF_32, 4)) != LV_OK) return result;
    if ((result = write_helper(new_fd, BUF_32, 4)) != LV_OK) return result;
    const LVKeyLen32_t saved_value_len = get_fixed_32(BUF_32);

    //value
    char saved_value[saved_value_len];
    if ((result = read_helper(old_fd, saved_value, saved_value_len)) != LV_OK) return result;
    if ((result = write_helper(new_fd, saved_value, saved_value_len)) != LV_OK) return result;


    //vector_id
    if ((result = read_helper(old_fd, BUF_64, 8)) != LV_OK) return result;
    if ((result = write_helper(new_fd, BUF_64, 8)) != LV_OK) return result;

    //field_mask
    if ((result = read_helper(old_fd, BUF_32, 4)) != LV_OK) return result;
    if ((result = write_helper(new_fd, BUF_32, 4)) != LV_OK) return result;

    //field_count
    if ((result = read_helper(old_fd, BUF_32, 4)) != LV_OK) return result;
    if ((result = write_helper(new_fd, BUF_32, 4)) != LV_OK) return result;
    const LVSize32_t saved_field_count = get_fixed_32(BUF_32);

    for (int i = 0; i < saved_field_count; ++i) {
        uint8_t saved_type;
        if ((result = read_helper(old_fd, &saved_type, sizeof(uint8_t))) != LV_OK) return result;
        const LVMetaType type = (LVMetaType)saved_type;

        if (type == LV_META_STRING) {
            if ((result = read_helper(old_fd, BUF_32, 4)) != LV_OK) return result;
            if ((result = write_helper(new_fd, BUF_32, 4)) != LV_OK) return result;
            const LVSize32_t saved_str_len = get_fixed_32(BUF_32);

            char string[saved_str_len];
            if ((result = read_helper(old_fd, string, saved_str_len)) != LV_OK) return result;
            if ((result = write_helper(new_fd, string, saved_str_len)) != LV_OK) return result;
        }
        else { //int64 or double
            if ((result = read_helper(old_fd, BUF_64, 8)) != LV_OK) return result;
            if ((result = write_helper(new_fd, BUF_64, 8)) != LV_OK) return result;
        }
    }

    return result;
}

LVStatus sst_indexblockset_append(LVSSTIndexBlockSet* index_buffer, const LVKeyLen32_t key_len, const void* key, const LVSeq64_t seq, const uint64_t offset) {
    if (index_buffer->size >= index_buffer->capacity) {
        const LVSize32_t new_capacity = index_buffer->capacity * 2;
        const uint8_t* tmp = realloc(index_buffer->entries, new_capacity * sizeof(LVSSTIndexBlockEntry));
        if (!tmp) {
            return LV_ERR_OOM;
        }
        index_buffer->capacity = new_capacity;
        index_buffer->entries = tmp;
    }

    index_buffer->entries[index_buffer->size].key_len = key_len;
    void* KEY = malloc(key_len);
    if (!KEY) {
        return LV_ERR_OOM;
    }
    memcpy(KEY, key, key_len);
    index_buffer->entries[index_buffer->size].key = KEY;
    index_buffer->entries[index_buffer->size].seq = seq;
    index_buffer->entries[index_buffer->size].offset = offset;
    index_buffer->size += 1;
    return LV_OK;
}

void destroy_indexblockset(LVSSTIndexBlockSet* index_block) {
    if (index_block) {
        for (int i = 0; i < index_block->size; ++i) {
            free(index_block->entries[i].key);
        }
        free(index_block->entries);
        free(index_block);
    }
}
