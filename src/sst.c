#include "sst.h"
#include "helper.h"
#include "node.h"
#include "util.h"
#include "hash.h"
#include "vector.h"
#include "query.h"

LVStatus sst_flush(const int new_fd, const int old_fd, const int vector_index_fd, const LVNode* node) {
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
                if ((result = sst_indexblockset_append(index_set, current_node->key_len, node_access_key(current_node), current_node->seq, current_node->vector_id, record_start_offset)) != LV_OK) goto _return;

                if (current_node->vector_id != LV_NO_VECTOR_ID) {
                    uint64_t offset = record_start_offset;
                    pwrite(vector_index_fd, &offset, 8, current_node->vector_id * 8);
                }

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
                if ((result = sst_indexblockset_append(index_set, old_entry.key_len, old_entry.key, old_entry.seq, old_entry.vector_id, record_start_offset)) != LV_OK) goto _return;

                if (old_entry.vector_id != LV_NO_VECTOR_ID) {
                    uint64_t offset = record_start_offset;
                    pwrite(vector_index_fd, &offset, 8, old_entry.vector_id * 8);
                }

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
                if ((result = sst_indexblockset_append(index_set, current_node->key_len, node_access_key(current_node), current_node->seq, current_node->vector_id, record_start_offset)) != LV_OK) goto _return;
                if (current_node->vector_id != LV_NO_VECTOR_ID) {
                    uint64_t offset = record_start_offset;
                    pwrite(vector_index_fd, &offset, 8, current_node->vector_id * 8);
                }
                current_node = current_node->levels[0];
            }
            record_count += 1;
        }

        while (current_node->type != LV_NODE_TAIL) {
            const uint64_t record_start_offset = write_helper_get_offset(new_fd);

            if ((result = sst_write_record_with_node(new_fd, current_node)) != LV_OK) goto _return;
            if ((result = sst_indexblockset_append(index_set, current_node->key_len, node_access_key(current_node), current_node->seq, current_node->vector_id, record_start_offset)) != LV_OK) goto _return;

            if (current_node->vector_id != LV_NO_VECTOR_ID) {
                uint64_t offset = record_start_offset;
                pwrite(vector_index_fd, &offset, 8, current_node->vector_id * 8);
            }
            current_node = current_node->levels[0];
        }

        while (has_old_entry) {
            const uint64_t record_start_offset = write_helper_get_offset(new_fd);

            if ((result = sst_write_record_with_old_sst(new_fd, old_fd, old_entry.offset)) != LV_OK) goto _return;
            if ((result = sst_indexblockset_append(index_set, old_entry.key_len, old_entry.key, old_entry.seq, old_entry.vector_id, record_start_offset)) != LV_OK) goto _return;

            if (old_entry.vector_id != LV_NO_VECTOR_ID) {
                uint64_t offset = record_start_offset;
                pwrite(vector_index_fd, &offset, 8, old_entry.vector_id * 8);
            }

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

        put_fixed_64(BUF_64, index_set->entries[i].vector_id);
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

    //read vector_id
    if ((result = read_helper(fd, BUF_64, 8)) != LV_OK) {
        free(saved_key);
        return result;
    };
    const LVSeq64_t saved_vectod_id = get_fixed_64(BUF_64);
    entry->vector_id = saved_vectod_id;

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

    //write level
    if ((result = write_helper(fd, &node->level, sizeof(LVLevel8_t))) != LV_OK) goto _return;

    //write key_len
    put_fixed_32(BUF_32, node->key_len);
    if ((result = write_helper(fd, BUF_32, sizeof(LVKeyLen32_t))) != LV_OK) goto _return;

    //write value_len
    put_fixed_32(BUF_32, node->value_len);
    if ((result = write_helper(fd, BUF_32, sizeof(LVValueLen32_t))) != LV_OK) goto _return;

    //write vector_id
    put_fixed_64(BUF_64, node->vector_id);
    if ((result = write_helper(fd, BUF_64, 8)) != LV_OK) goto _return;

    //write field mask
    put_fixed_32(BUF_32, node->field_mask);
    if ((result = write_helper(fd, BUF_32, 4)) != LV_OK) goto _return;

    //write field count
    put_fixed_32(BUF_32, node->field_count);
    if ((result = write_helper(fd, BUF_32, 4)) != LV_OK) goto _return;

    //write field size (nonserialized)
    LVSize32_t field_nonserialized_size = sst_node_field_size(node, sizeof(LVMetaType));
    put_fixed_32(BUF_32, field_nonserialized_size);
    if ((result = write_helper(fd, BUF_32, 4)) != LV_OK) goto _return;

    //write field size (serialized)
    LVSize32_t field_serialized_size = sst_node_field_size(node, sizeof(uint8_t));
    put_fixed_32(BUF_32, field_serialized_size);
    if ((result = write_helper(fd, BUF_32, 4)) != LV_OK) goto _return;

    //write key
    if ((result = write_helper(fd, node_access_key(node), node->key_len)) != LV_OK) goto _return;

    //write value
    if ((result = write_helper(fd, node_access_value(node), node->value_len)) != LV_OK) goto _return;

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

    //level
    LVLevel8_t saved_level;
    if ((result = read_helper(old_fd, &saved_level, sizeof(LVLevel8_t))) != LV_OK) return result;
    if ((result = write_helper(new_fd, &saved_level, sizeof(LVLevel8_t))) != LV_OK) return result;

    //key_len
    if ((result = read_helper(old_fd, BUF_32, 4)) != LV_OK) return result;
    if ((result = write_helper(new_fd, BUF_32, 4)) != LV_OK) return result;
    const LVKeyLen32_t saved_key_len = get_fixed_32(BUF_32);


    //value_len
    if ((result = read_helper(old_fd, BUF_32, 4)) != LV_OK) return result;
    if ((result = write_helper(new_fd, BUF_32, 4)) != LV_OK) return result;
    const LVKeyLen32_t saved_value_len = get_fixed_32(BUF_32);


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

    //field_size (nonserialized)
    if ((result = read_helper(old_fd, BUF_32, 4)) != LV_OK) return result;
    if ((result = write_helper(new_fd, BUF_32, 4)) != LV_OK) return result;

    //field_size (serialized)
    if ((result = read_helper(old_fd, BUF_32, 4)) != LV_OK) return result;
    if ((result = write_helper(new_fd, BUF_32, 4)) != LV_OK) return result;

    //key
    char saved_key[saved_key_len];
    if ((result = read_helper(old_fd, saved_key, saved_key_len)) != LV_OK) return result;
    if ((result = write_helper(new_fd, saved_key, saved_key_len)) != LV_OK) return result;

    //value
    char saved_value[saved_value_len];
    if ((result = read_helper(old_fd, saved_value, saved_value_len)) != LV_OK) return result;
    if ((result = write_helper(new_fd, saved_value, saved_value_len)) != LV_OK) return result;

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

LVStatus sst_indexblockset_append(LVSSTIndexBlockSet* index_buffer, const LVKeyLen32_t key_len, const void* key, const LVSeq64_t seq, const LVVectorId64_t vector_id, const uint64_t offset) {
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
    index_buffer->entries[index_buffer->size].vector_id = vector_id;
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

LVSize32_t sst_node_field_size(const LVNode* node, const uint8_t type_size) {
    if (node->field_count <= 0) return 0;
    char* field_ptr = (char*)node_access_field(node, 0);
    LVSize32_t size = 0;
    for (int i = 0; i < node->field_count; ++i) {
        LVMetaType type;
        memcpy(&type, field_ptr, sizeof(LVMetaType));

        size += type_size;

        field_ptr += sizeof(LVMetaType);

        if (type == LV_META_STRING) {
            uint32_t len = 0;
            memcpy(&len, field_ptr, sizeof(uint32_t));
            size += sizeof(uint32_t) + len;
            field_ptr += sizeof(uint32_t) + len;
        }
        else {
            size += 8;
            field_ptr += 8;
        }
    }
}

LVStatus sst_query_filter_scan(const int fd, const LVSchema* schema, const LVAstNode* query, const LVSize32_t query_field_mask, const LVOrdbyType ordbytype, const LVSize32_t ordby_field_mask, const LVQVListAppend qv_append_func, LVQVSet* qv_set) {
    LVStatus result = LV_OK;
    uint8_t BUF_32[4];
    uint8_t BUF_64[8];

    lseek(fd, -16, SEEK_END);

    //read index block offset
    if ((result = read_helper(fd, BUF_64, 8)) != LV_OK) goto _return;
    const uint64_t saved_index_offset = get_fixed_64(BUF_64);

    //read total record count
    if ((result = read_helper(fd, BUF_64, 8)) != LV_OK) goto _return;
    const uint64_t saved_record_count = get_fixed_64(BUF_64);

    //goto index block
    lseek(fd, saved_index_offset, SEEK_SET);

    //read first index
    LVSSTIndexBlockEntry index_entry;
    if ((result = sst_read_next_index_entry(fd, &index_entry)) != LV_OK) goto _return;

    //goto first record
    lseek(fd, index_entry.offset, SEEK_SET);

    uint64_t record_read = 0;
    while (record_read < saved_record_count) {
        LVSeq64_t seq;
        LVNodeOp op;
        LVLevel8_t level;
        LVKeyLen32_t key_len;
        LVValueLen32_t value_len;
        LVVectorId64_t vector_id;
        LVSize32_t field_mask;
        LVSize32_t field_count;
        LVSize32_t field_nonserialized_size;
        LVSize32_t field_serialized_size;

        if ((result = sst_read_record_head(fd, &seq, &op, &level, &key_len, &value_len, &vector_id, &field_mask, &field_count, &field_nonserialized_size, &field_serialized_size)) != LV_OK) goto _return;

        if (field_count > 0 && (query_field_mask & field_mask)) {
            char node_buf[sizeof(LVNode) + field_nonserialized_size];
            LVNode* dummy_node = (LVNode*)node_buf;
            dummy_node->level = 0;
            dummy_node->key_len = key_len;
            dummy_node->value_len = value_len;
            dummy_node->field_mask = field_mask;
            dummy_node->field_count = field_count;

            char key[key_len];
            char value[value_len];

            if ((result = sst_read_record_tail(fd, key, key_len, value, value_len, (char*)(node_access_field(dummy_node, 0)), field_count)) != LV_OK) goto _return;

            if (query_eval_ast(query, dummy_node, schema)) {
                float vector_score = 0.0f;
                LVOrdbyValue ordbyvalue;
                ordbyvalue.i64 = 0;

                switch (ordbytype)
                {
                case LV_ORDBY_FLOAT: {
                    double value = node_get_f64_field(dummy_node, ordby_field_mask);
                    ordbyvalue.f64 = value;
                    break;
                }
                case LV_ORDBY_INT: {
                    int64_t value = node_get_i64_field(dummy_node, ordby_field_mask);
                    ordbyvalue.i64 = value;
                    break;
                }
                case LV_ORDBY_VEC: {
                    ordbyvalue.score = 0.0f;
                    break;
                }

                default:
                    break;
                }


                if ((result = qv_append_func(qv_set, dummy_node->seq, vector_id, key, key_len, value, value_len, vector_score, ordbyvalue)) != LV_OK) return result;

            }

        }
        else {
            lseek(fd, key_len + value_len + field_serialized_size, SEEK_CUR);
        }

        record_read += 1;

    }

_return:
    return result;
}

LVStatus sst_read_record_head(const int fd, LVSeq64_t* seq, LVNodeOp* op, LVLevel8_t* level, LVKeyLen32_t* key_len, LVValueLen32_t* value_len, LVVectorId64_t* vector_id, LVSize32_t* field_mask, LVSize32_t* field_count, LVSize32_t* field_nonserialized_size, LVSize32_t* field_serialized_size) {
    LVStatus result = LV_OK;

    uint8_t BUF_32[4];
    uint8_t BUF_64[8];

    if ((result = read_helper(fd, BUF_64, 8)) != LV_OK) goto _return;
    *seq = get_fixed_64(BUF_64);

    uint8_t saved_op;
    if ((result = read_helper(fd, &saved_op, sizeof(uint8_t))) != LV_OK) goto _return;
    *op = (LVNodeOp)saved_op;

    if ((result = read_helper(fd, BUF_32, 4)) != LV_OK)goto _return;
    *key_len = get_fixed_32(BUF_32);

    if ((result = read_helper(fd, BUF_32, 4)) != LV_OK)goto _return;
    *value_len = get_fixed_32(BUF_32);


    if ((result = read_helper(fd, BUF_64, 8)) != LV_OK) goto _return;
    *vector_id = get_fixed_64(BUF_64);

    if ((result = read_helper(fd, BUF_32, 4)) != LV_OK)goto _return;
    *field_mask = get_fixed_32(BUF_32);

    if ((result = read_helper(fd, BUF_32, 4)) != LV_OK)goto _return;
    *field_count = get_fixed_32(BUF_32);

    if ((result = read_helper(fd, BUF_32, 4)) != LV_OK)goto _return;
    *field_nonserialized_size = get_fixed_32(BUF_32);

    if ((result = read_helper(fd, BUF_32, 4)) != LV_OK)goto _return;
    *field_serialized_size = get_fixed_32(BUF_32);

_return:
    return result;
}

LVStatus sst_read_record_tail(const int fd, char* key, const LVKeyLen32_t key_len, char* value, const LVValueLen32_t value_len, char* field, const LVSize32_t field_count) {
    LVStatus result = LV_OK;
    uint8_t BUF_32[4];
    uint8_t BUF_64[8];

    if ((result = read_helper(fd, key, key_len)) != LV_OK) goto _return;

    if ((result = read_helper(fd, value, value_len)) != LV_OK) goto _return;


    char* field_ptr = field;
    for (int i = 0; i < field_count; ++i) {
        uint8_t saved_type;
        if ((result = read_helper(fd, &saved_type, sizeof(uint8_t))) != LV_OK) goto _return;
        const LVMetaType type = (LVMetaType)saved_type;
        memcpy(&type, field_ptr, sizeof(LVMetaType));

        field_ptr += sizeof(LVMetaType);

        if (type == LV_META_STRING) {
            if ((result = read_helper(fd, BUF_32, 4)) != LV_OK) goto _return;
            LVSize32_t saved_len = get_fixed_32(BUF_32);
            memcpy(field_ptr, &saved_len, sizeof(uint32_t));

            field_ptr += sizeof(uint32_t);

            char string[saved_len];
            if ((result = read_helper(fd, string, saved_len)) != LV_OK) goto _return;
            memcpy(field_ptr, string, saved_len);

            field_ptr += saved_len;
        }
        else {
            if ((result = read_helper(fd, BUF_64, 8)) != LV_OK) goto _return;
            uint64_t saved_value = get_fixed_64(BUF_64);
            memcpy(field_ptr, &saved_value, sizeof(uint64_t));
            field_ptr += sizeof(uint64_t);
        }
    }

_return:
    return result;
}
