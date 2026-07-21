#include "sst.h"
#include "helper.h"
#include "node.h"
#include "util.h"
#include "vector.h"
#include "query.h"
#include "storage.h"
#include "schema.h"

/*
 * Flush the memtable to a new SST, optionally MERGING with an existing SST.
 * This is the LSM compaction step.
 *
 * Two modes:
 *   old_fd < 0  : fresh flush. Walk the memtable in key order, write each live
 *                 record to the new SST. (Deletes are dropped — see below.)
 *   old_fd >= 0 : merge flush. The memtable AND an existing sorted SST are both
 *                 sorted by key; merge them like the merge step of merge-sort,
 *                 producing one sorted new SST.
 *
 * Both inputs are sorted by key, so we advance whichever side has the smaller
 * key (node_cmp), and when both sides have the SAME key the memtable wins (it's
 * newer) — that's how updates take effect.
 *
 * DELETES (tombstones):
 *   A memtable delete means "this key is gone". On a fresh flush we simply skip
 *   it. On a merge, a delete must also SUPPRESS the old SST's copy of that key:
 *   when the memtable delete matches the old entry's key, we drop BOTH and
 *   emit nothing. This is where a deleted key finally disappears from disk.
 *   (Deletes never carry a vector, so they never touch vector_index.lv.)
 *
 * As each record is written we record its offset in the key index (for by-key
 * lookup) AND, if it has a vector, write that offset into vector_index.lv at
 * vector_id*8 — keeping the O(1) vector_id -> offset map in sync (see header).
 *
 * Finally we append the index block and a footer (index offset, record count,
 * next seq/vector_id) so the SST is self-describing on reopen.
 *
 * MEMORY: index_set and old_entry.key are heap-allocated; the _return label
 * frees both. old_entry.key is set to NULL after every free so the cleanup
 * can't double-free (the pattern you'll see repeated at each advance).
 */

LVStatus sst_flush(const int new_fd, const int old_fd, const int vector_index_fd, const LVNode* node) {
    LVStatus result = LV_OK;
    uint8_t BUF_32[4];
    uint8_t BUF_64[8];

    LVSSTIndexBlockEntry old_entry;
    old_entry.key = NULL;

    LVSSTIndexBlockSet* index_set = malloc(sizeof(LVSSTIndexBlockSet));
    if (!index_set) {
        result = LV_ERR_OOM;
        goto _return;
    }

    index_set->capacity = LV_DEFAULT_CAPACITY;
    index_set->size = 0;
    index_set->entries = NULL;

    index_set->entries = malloc(sizeof(LVSSTIndexBlockEntry) * index_set->capacity);
    if (!index_set->entries) {
        result = LV_ERR_OOM;
        goto _return;
    }
    
    //check old sst corruption
    if (old_fd >= 0) {
        lseek(old_fd, 0, SEEK_SET);
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
    LVSeq64_t next_seq = 0;
    LVVectorId64_t next_vector_id = 0;
    if (old_fd < 0) {
        LVNode* current_node = node;
        LVSeq64_t last_seq = 0;
        LVVectorId64_t last_vector_id = 0;
        while (current_node->type != LV_NODE_TAIL) {
            const void* cur_key = node_access_key(current_node);
            const LVKeyLen32_t cur_key_len = current_node->key_len;

            if (current_node->op == LV_DELETE) {
                last_seq = current_node->seq;
                // deleted node's vector id is always LV_NO_VECTOR_ID
                current_node = table_get_next_node(current_node);
                continue;
            }

            const uint64_t record_start_offset = lseek(new_fd,0, SEEK_CUR);
            if ((result = sst_write_record_with_node(new_fd, current_node)) != LV_OK) goto _return;
            if ((result = sst_indexblockset_append(index_set, cur_key_len, cur_key,
                current_node->seq, current_node->vector_id, record_start_offset)) != LV_OK) goto _return;

            if (current_node->vector_id != LV_NO_VECTOR_ID) {
                uint64_t offset = record_start_offset;
                put_fixed_64(BUF_64, offset);
                if ((result = pwrite_helper(vector_index_fd, BUF_64, 8, current_node->vector_id * 8)) != LV_OK) goto _return;
            }

            record_count += 1;
            last_seq = current_node->seq;
            if (current_node->vector_id != LV_NO_VECTOR_ID) {
                last_vector_id = current_node->vector_id;
            }
            current_node = table_get_next_node(current_node);
        }

        next_seq = last_seq + 1;
        next_vector_id = last_vector_id + 1;
    }
    else {
        uint64_t index_block_offset = 0;
        uint64_t saved_record_count = 0;

        if ((result = sst_read_footer(old_fd, &index_block_offset, &saved_record_count, NULL, NULL)) != LV_OK) goto _return;

        uint64_t old_entry_offset = index_block_offset;
        /* old_entry declared at function top; first read populates it. On
         * failure sst_read leaves old_entry.key == NULL, so _return is safe. */
        if ((result = sst_read_index_entry_at_offset(old_fd, old_entry_offset, &old_entry, &old_entry_offset)) != LV_OK) {
            goto _return;
        }
        int has_old_entry = 1;

        uint64_t record_read = 0;
        LVNode* current_node = node;
        LVSeq64_t last_seq = 0;
        LVVectorId64_t last_vector_id = 0;

        // ── merge loop: both memtable and old SST still have entries ──
        // Compare current memtable key vs current old-SST key and advance the
        // smaller. Equal keys -> memtable wins (newer); delete -> suppress old.

        while (has_old_entry && current_node->type != LV_NODE_TAIL) {
            const void* current_key = node_access_key(current_node);
            const LVKeyLen32_t current_key_len = current_node->key_len;

            if (current_node->op == LV_DELETE) { // memtable says this key is deleted.
                if (node_key_equal(current_key, current_key_len, old_entry.key, old_entry.key_len)) {
                    // same key on both sides -> drop BOTH: the delete cancels the
                    // old SST record. Nothing written; key is now truly gone.
                    free(old_entry.key);
                    old_entry.key = NULL;  /* freed; null so the _return cleanup can't double-free */
                    record_read += 1;
                    if (record_read < saved_record_count) {
                        if ((result = sst_read_index_entry_at_offset(old_fd, old_entry_offset, &old_entry, &old_entry_offset)) != LV_OK) goto _return;
                        has_old_entry = 1;
                    }
                    else {
                        has_old_entry = 0;
                    }
                    last_seq = current_node->seq;
                    current_node = table_get_next_node(current_node);

                    continue;
                }

                if (node_cmp(LV_NODE_DATA, old_entry.key, old_entry.key_len, old_entry.seq,
                    LV_NODE_DATA, current_key, current_key_len, current_node->seq) < 0) {

                    const uint64_t record_start_offset = lseek(new_fd, 0, SEEK_CUR);
                    if ((result = sst_write_record_with_old_sst(new_fd, old_fd, old_entry.offset)) != LV_OK) goto _return;
                    if ((result = sst_indexblockset_append(index_set, old_entry.key_len, old_entry.key,
                        old_entry.seq, old_entry.vector_id, record_start_offset)) != LV_OK) goto _return;
                    if (old_entry.vector_id != LV_NO_VECTOR_ID) {
                        uint64_t offset = record_start_offset;
                        put_fixed_64(BUF_64, offset);
                        if ((result = pwrite_helper(vector_index_fd, BUF_64, 8, old_entry.vector_id * 8)) != LV_OK) goto _return;
                    }
                    record_read += 1;
                    record_count += 1;
                    free(old_entry.key);
                    old_entry.key = NULL;  /* freed; null so the _return cleanup can't double-free */
                    if (record_read < saved_record_count) {
                        if ((result = sst_read_index_entry_at_offset(old_fd, old_entry_offset, &old_entry, &old_entry_offset)) != LV_OK) goto _return;
                        has_old_entry = 1;
                    }
                    else {
                        has_old_entry = 0;
                    }

                    continue;
                }
                else {
                    //drop
                    last_seq = current_node->seq;
                    current_node = table_get_next_node(current_node);

                    continue;
                }
            }
            const uint64_t record_start_offset = lseek(new_fd, 0, SEEK_CUR);


            
            // same key, memtable side is live -> memtable version wins (newer),
            // write it and skip the old SST copy. This is how an update replaces
            // the on-disk record.
            if (node_key_equal(old_entry.key, old_entry.key_len, current_key, current_key_len)) {
                const uint64_t record_start_offset = lseek(new_fd,0, SEEK_CUR);
                if ((result = sst_write_record_with_node(new_fd, current_node)) != LV_OK) goto _return;
                if ((result = sst_indexblockset_append(index_set, current_key_len, current_key,
                    current_node->seq, current_node->vector_id, record_start_offset)) != LV_OK) goto _return;
                if (current_node->vector_id != LV_NO_VECTOR_ID) {
                    uint64_t offset = record_start_offset;
                    put_fixed_64(BUF_64, offset);
                    if ((result = pwrite_helper(vector_index_fd, BUF_64, 8, current_node->vector_id * 8)) != LV_OK) goto _return;
                    if (current_node->vector_id > last_vector_id) {
                        last_vector_id = current_node->vector_id;
                    }
                }
                last_seq = current_node->seq;

                free(old_entry.key);
                old_entry.key = NULL;  /* freed; null so the _return cleanup can't double-free */
                record_read += 1;
                if (record_read < saved_record_count) {
                    if ((result = sst_read_index_entry_at_offset(old_fd, old_entry_offset, &old_entry, &old_entry_offset)) != LV_OK) goto _return;
                    has_old_entry = 1;
                }
                else {
                    has_old_entry = 0;
                }
                current_node = table_get_next_node(current_node);
                record_count += 1;
                continue;
            }
            /* different keys: existing cmp_result < 0 logic stays */
            const int cmp_result = node_cmp(LV_NODE_DATA, old_entry.key, old_entry.key_len, old_entry.seq, LV_NODE_DATA, current_key, current_key_len, current_node->seq);

            if (cmp_result < 0) {
                // old SST key smaller -> emit old record, advance old side.
                if ((result = sst_write_record_with_old_sst(new_fd, old_fd, old_entry.offset)) != LV_OK) goto _return;
                if ((result = sst_indexblockset_append(index_set, old_entry.key_len, old_entry.key, old_entry.seq, old_entry.vector_id, record_start_offset)) != LV_OK) goto _return;

                if (old_entry.vector_id != LV_NO_VECTOR_ID) {
                    uint64_t offset = record_start_offset;
                    put_fixed_64(BUF_64, offset);
                    if ((result = pwrite_helper(vector_index_fd, BUF_64, 8, old_entry.vector_id * 8)) != LV_OK) goto _return;
                }

                record_read += 1;
                free(old_entry.key);
                old_entry.key = NULL;  /* freed; null so the _return cleanup can't double-free */

                if (record_read < saved_record_count) {
                    if ((result = sst_read_index_entry_at_offset(old_fd, old_entry_offset, &old_entry, &old_entry_offset)) != LV_OK) goto _return;
                    has_old_entry = 1;
                }
                else {
                    has_old_entry = 0;
                }
            }
            else {
                // memtable key smaller -> emit memtable record, advance memtable.
                if ((result = sst_write_record_with_node(new_fd, current_node)) != LV_OK) goto _return;
                if ((result = sst_indexblockset_append(index_set, current_key_len, current_key, current_node->seq, current_node->vector_id, record_start_offset)) != LV_OK) goto _return;
                if (current_node->vector_id != LV_NO_VECTOR_ID) {
                    uint64_t offset = record_start_offset;
                    put_fixed_64(BUF_64, offset);
                    if ((result = pwrite_helper(vector_index_fd, BUF_64, 8, current_node->vector_id * 8)) != LV_OK) goto _return;
                }
                last_seq = current_node->seq;
                if (current_node->vector_id != LV_NO_VECTOR_ID && current_node->vector_id > last_vector_id) {
                    last_vector_id = current_node->vector_id;
                }
                current_node = table_get_next_node(current_node);
            }
            record_count += 1;
        }

        // memtable exhausted the merge but old SST may remain (and vice versa):
        // drain whichever still has entries, in order.
        while (current_node->type != LV_NODE_TAIL) {
            const void* current_key = node_access_key(current_node);
            const LVKeyLen32_t current_key_len = current_node->key_len;

            if (current_node->op == LV_DELETE) {
                last_seq = current_node->seq;
                current_node = table_get_next_node(current_node);
                continue;
            }

            const uint64_t record_start_offset = lseek(new_fd, 0, SEEK_CUR);

            if ((result = sst_write_record_with_node(new_fd, current_node)) != LV_OK) goto _return;
            if ((result = sst_indexblockset_append(index_set, current_key_len, current_key, current_node->seq, current_node->vector_id, record_start_offset)) != LV_OK) goto _return;

            if (current_node->vector_id != LV_NO_VECTOR_ID) {
                uint64_t offset = record_start_offset;
                put_fixed_64(BUF_64, offset);
                if ((result = pwrite_helper(vector_index_fd, BUF_64, 8, current_node->vector_id * 8)) != LV_OK) goto _return;
            }
            record_count += 1;
            last_seq = current_node->seq;
            if (current_node->vector_id != LV_NO_VECTOR_ID && current_node->vector_id > last_vector_id) {
                last_vector_id = current_node->vector_id;
            }
            current_node = table_get_next_node(current_node);
        }

        while (has_old_entry) {
            const uint64_t record_start_offset = lseek(new_fd, 0, SEEK_CUR);

            if ((result = sst_write_record_with_old_sst(new_fd, old_fd, old_entry.offset)) != LV_OK) goto _return;
            if ((result = sst_indexblockset_append(index_set, old_entry.key_len, old_entry.key, old_entry.seq, old_entry.vector_id, record_start_offset)) != LV_OK) goto _return;

            if (old_entry.vector_id != LV_NO_VECTOR_ID) {
                uint64_t offset = record_start_offset;
                put_fixed_64(BUF_64, offset);
                if ((result = pwrite_helper(vector_index_fd, BUF_64, 8, old_entry.vector_id * 8)) != LV_OK) goto _return;
            }

            record_read += 1;
            record_count += 1;
            free(old_entry.key);
            old_entry.key = NULL;  /* freed; null so the _return cleanup can't double-free */

            if (record_read < saved_record_count) {
                if ((result = sst_read_index_entry_at_offset(old_fd, old_entry_offset, &old_entry, &old_entry_offset)) != LV_OK) goto _return;
                has_old_entry = 1;
            }
            else {
                has_old_entry = 0;
            }
        }

        next_seq = last_seq + 1;
        next_vector_id = last_vector_id + 1;

    }


    const uint64_t index_block_offset = lseek(new_fd, 0, SEEK_CUR);

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

    put_fixed_64(BUF_64, next_seq);
    if ((result = write_helper(new_fd, BUF_64, 8)) != LV_OK) goto _return;

    put_fixed_64(BUF_64, next_vector_id);
    if ((result = write_helper(new_fd, BUF_64, 8)) != LV_OK) goto _return;

_return:
    free(old_entry.key);
    sst_destroy_indexblockset(index_set);
    return result;
}

LVStatus sst_read_footer(const int fd, LVOffset64_t* indexblock_offset, LVBigCount64_t* record_count, LVSeq64_t* next_seq, LVVectorId64_t* next_vector_id) {
    LVStatus result = LV_OK;

    uint8_t BUF_64[8];
    off_t current_offset = lseek(fd, 0, SEEK_CUR);
    if (current_offset == -1) {
        result = LV_ERR_IO;
        goto _return;
    }

    off_t file_size = lseek(fd, 0, SEEK_END);
    if (file_size == -1 || file_size < 32) {
        result = LV_ERR_CORRUPT;
        goto _return;
    }

    const uint64_t footer_offset = (uint64_t)(file_size - 32);

    if ((result = pread_helper(fd, BUF_64, 8, footer_offset)) != LV_OK) goto _return;
    if (indexblock_offset) {
        *indexblock_offset = get_fixed_64(BUF_64);
    }

    if ((result = pread_helper(fd, BUF_64, 8, footer_offset + 8)) != LV_OK) goto _return;
    if (record_count) {
        *record_count = get_fixed_64(BUF_64);
    }

    if ((result = pread_helper(fd, BUF_64, 8, footer_offset + 16)) != LV_OK) goto _return;
    if (next_seq) {
        *next_seq = get_fixed_64(BUF_64);
    }

    if ((result = pread_helper(fd, BUF_64, 8, footer_offset + 24)) != LV_OK) goto _return;
    if (next_vector_id) {
        *next_vector_id = get_fixed_64(BUF_64);
    }

_return:
    if (current_offset != -1) {
        lseek(fd, current_offset, SEEK_SET);
    }
    return result;
}

LVStatus sst_read_index_entry_at_offset(const int fd, const uint64_t offset, LVSSTIndexBlockEntry* entry, LVOffset64_t* next_offset) {
    LVStatus result = LV_OK;
    uint8_t BUF_32[4];
    uint8_t BUF_64[8];
    uint64_t off = offset;

    entry->key = NULL;

    if ((result = pread_helper(fd, BUF_32, 4, off)) != LV_OK) return result;
    const LVKeyLen32_t saved_key_len = get_fixed_32(BUF_32);
    entry->key_len = saved_key_len;
    off += 4;

    char* saved_key = malloc(saved_key_len);
    if (!saved_key) {
        return LV_ERR_OOM;
    }
    if ((result = pread_helper(fd, saved_key, saved_key_len, off)) != LV_OK) {
        free(saved_key);
        return result;   
    }
    off += saved_key_len;

    if ((result = pread_helper(fd, BUF_64, 8, off)) != LV_OK) {
        free(saved_key);
        return result;   
    }
    entry->seq = get_fixed_64(BUF_64);
    off += 8;

    if ((result = pread_helper(fd, BUF_64, 8, off)) != LV_OK) {
        free(saved_key);
        return result;  
    }
    entry->vector_id = get_fixed_64(BUF_64);
    off += 8;

    if ((result = pread_helper(fd, BUF_64, 8, off)) != LV_OK) {
        free(saved_key);
        return result;   
    }
    entry->offset = get_fixed_64(BUF_64);
    off += 8;

    entry->key = saved_key;

    if (next_offset) {
        *next_offset = off;
    }

    return LV_OK;
}

LVStatus sst_write_record_with_node(const int fd, const LVNode* node) {
    LVStatus result = LV_OK;

    uint8_t BUF_32[4];
    uint8_t BUF_64[8];

    const LVSize32_t field_size = node->field_count > 0 ? node_field_size(node) : 0;

    char disk_field_buffer[field_size];

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

    //write field size
    put_fixed_32(BUF_32, field_size);
    if ((result = write_helper(fd, BUF_32, 4)) != LV_OK) goto _return;

    //write key
    if ((result = write_helper(fd, node_access_key(node), node->key_len)) != LV_OK) goto _return;

    //write value
    if ((result = write_helper(fd, node_access_value(node), node->value_len)) != LV_OK) goto _return;

    //write meta fields
    if (node->field_count > 0) {
        schema_field_memmory_to_disk(node_access_field(node, 0), field_size, disk_field_buffer);
        if ((result = write_helper(fd, disk_field_buffer, field_size)) != LV_OK) goto _return;
    }

_return:
    return result;
}

LVStatus sst_write_record_with_old_sst(const int new_fd, const int old_fd, const LVOffset64_t read_offset) {
    LVStatus result = LV_OK;
    uint8_t BUF_32[4];
    uint8_t BUF_64[8];
    uint64_t off = read_offset;
    
    //seq
    if ((result = pread_helper(old_fd, BUF_64, 8, off)) != LV_OK) return result;
    off += 8;
    if ((result = write_helper(new_fd, BUF_64, 8)) != LV_OK) return result;


    //op
    uint8_t saved_op;
    if ((result = pread_helper(old_fd, &saved_op, sizeof(uint8_t), off)) != LV_OK) return result;
    off += sizeof(uint8_t);
    if ((result = write_helper(new_fd, &saved_op, sizeof(uint8_t))) != LV_OK) return result;

    //level
    LVLevel8_t saved_level;
    if ((result = pread_helper(old_fd, &saved_level, sizeof(LVLevel8_t), off)) != LV_OK) return result;
    off += sizeof(LVLevel8_t);
    if ((result = write_helper(new_fd, &saved_level, sizeof(LVLevel8_t))) != LV_OK) return result;

    //key_len
    if ((result = pread_helper(old_fd, BUF_32, 4, off)) != LV_OK) return result;
    off += 4;
    if ((result = write_helper(new_fd, BUF_32, 4)) != LV_OK) return result;
    const LVKeyLen32_t saved_key_len = get_fixed_32(BUF_32);


    //value_len
    if ((result = pread_helper(old_fd, BUF_32, 4, off)) != LV_OK) return result;
    off += 4;
    if ((result = write_helper(new_fd, BUF_32, 4)) != LV_OK) return result;
    const LVKeyLen32_t saved_value_len = get_fixed_32(BUF_32);


    //vector_id
    if ((result = pread_helper(old_fd, BUF_64, 8, off)) != LV_OK) return result;
    off += 8;
    if ((result = write_helper(new_fd, BUF_64, 8)) != LV_OK) return result;

    //field_mask
    if ((result = pread_helper(old_fd, BUF_32, 4, off)) != LV_OK) return result;
    off += 4;
    if ((result = write_helper(new_fd, BUF_32, 4)) != LV_OK) return result;

    //field_count
    if ((result = pread_helper(old_fd, BUF_32, 4, off)) != LV_OK) return result;
    off += 4;
    if ((result = write_helper(new_fd, BUF_32, 4)) != LV_OK) return result;

    //field_size
    if ((result = pread_helper(old_fd, BUF_32, 4, off)) != LV_OK) return result;
    off += 4;
    if ((result = write_helper(new_fd, BUF_32, 4)) != LV_OK) return result;
    const LVSize32_t saved_field_size = get_fixed_32(BUF_32);

    //key
    char saved_key[saved_key_len];
    if ((result = pread_helper(old_fd, saved_key, saved_key_len, off)) != LV_OK) return result;
    off += saved_key_len;
    if ((result = write_helper(new_fd, saved_key, saved_key_len)) != LV_OK) return result;

    //value
    char saved_value[saved_value_len];
    if ((result = pread_helper(old_fd, saved_value, saved_value_len, off)) != LV_OK) return result;
    off += saved_value_len;
    if ((result = write_helper(new_fd, saved_value, saved_value_len)) != LV_OK) return result;

    //fields
    char saved_field_buffer[saved_field_size];
    if ((result = pread_helper(old_fd, saved_field_buffer, saved_field_size, off)) != LV_OK) return result;
    off += saved_field_size;
    if ((result = write_helper(new_fd, saved_field_buffer, saved_field_size)) != LV_OK) return result;

    return result;
}

LVStatus sst_indexblockset_append(LVSSTIndexBlockSet* index_buffer, const LVKeyLen32_t key_len, const void* key, const LVSeq64_t seq, const LVVectorId64_t vector_id, const LVOffset64_t offset) {
    if (index_buffer->size >= index_buffer->capacity) {
        const LVSize32_t new_capacity = index_buffer->capacity * 2;
        LVSSTIndexBlockEntry* tmp = realloc(index_buffer->entries, new_capacity * sizeof(LVSSTIndexBlockEntry));
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

void sst_destroy_indexblockset(LVSSTIndexBlockSet* index_block) {
    if (index_block) {
        if (index_block->entries) {
            for (int i = 0; i < index_block->size; ++i) {
                free(index_block->entries[i].key);
            }
            free(index_block->entries);
        }
        free(index_block);
    }

}


LVStatus sst_query_filter_scan(const int fd, const LVSchema* schema, const LVAstNode* query, const LVFieldMask32_t query_field_mask, const LVOrdbyType ordbytype, const LVFieldMask32_t ordby_field_mask, const LVQVSetAppendFn qv_append_fn, LVQVSet* qv_set) {
    LVStatus result = LV_OK;
    uint8_t BUF_32[4];
    uint8_t BUF_64[8];

    if (fd < 0) goto _return;

    uint64_t saved_index_offset = 0;
    uint64_t saved_record_count = 0;

    if ((result = sst_read_footer(fd, &saved_index_offset, &saved_record_count, NULL, NULL)) != LV_OK) goto _return;

    uint64_t index_entry_offset = 0;
    LVSSTIndexBlockEntry index_entry;
    if ((result = sst_read_index_entry_at_offset(fd, saved_index_offset, &index_entry, &index_entry_offset)) != LV_OK) goto _return;

    uint64_t record_offset = index_entry.offset;
    free(index_entry.key);

    uint64_t record_read = 0;
    while (record_read < saved_record_count) {
        LVSeq64_t seq;
        LVNodeOp op;
        LVLevel8_t level;
        LVKeyLen32_t key_len;
        LVValueLen32_t value_len;
        LVVectorId64_t vector_id;
        LVFieldMask32_t field_mask;
        LVSize32_t field_count;
        LVSize32_t field_size;

        uint64_t record_head_size = 0;
        if ((result = sst_read_record_head(fd, record_offset, &seq, &op, &level, &key_len, &value_len, &vector_id, &field_mask, &field_count, &field_size, &record_head_size)) != LV_OK) goto _return;

        const uint64_t tail_offset = record_offset + record_head_size;

        if (field_count > 0 && (query_field_mask & field_mask)) {
            char node_buf[sizeof(LVNode) + key_len + value_len + field_size];
            LVNode* dummy_node = (LVNode*)node_buf;
            dummy_node->level = 0;
            dummy_node->key_len = key_len;
            dummy_node->value_len = value_len;
            dummy_node->field_mask = field_mask;
            dummy_node->field_count = field_count;

            char key[key_len];
            char value[value_len];

            if ((result = sst_read_record_tail(fd, tail_offset, key, key_len, value, value_len, (char*)(node_access_field(dummy_node, 0)), field_size, NULL)) != LV_OK) goto _return;

            if (node_eval_query(dummy_node, query, schema)) {
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
                    int64_t value = node_get_int64_field(dummy_node, ordby_field_mask);
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

                if ((result = qv_append_fn(qv_set, dummy_node->seq, vector_id, key, key_len, value, value_len, vector_score, ordbyvalue, 0)) != LV_OK) goto _return;
            }
        }

        record_offset += record_head_size + key_len + value_len + field_size;
        record_read += 1;
    }

_return:
    return result;
}

LVStatus sst_read_record_head(const int fd, const LVOffset64_t read_offset, LVSeq64_t* seq, LVNodeOp* op, LVLevel8_t* level, LVKeyLen32_t* key_len, LVValueLen32_t* value_len, LVVectorId64_t* vector_id, LVFieldMask32_t* field_mask, LVCount32_t* field_count, LVSize32_t* field_size, LVOffset64_t* read_bytes_out) {
    LVStatus result = LV_OK;

    uint8_t BUF_32[4];
    uint8_t BUF_64[8];
    uint64_t off = read_offset;

    if ((result = pread_helper(fd, BUF_64, 8, off)) != LV_OK) goto _return;
    if (seq) {
        *seq = get_fixed_64(BUF_64);
    }
    off += 8;

    uint8_t saved_op;
    if ((result = pread_helper(fd, &saved_op, sizeof(uint8_t), off)) != LV_OK) goto _return;
    if (op) {
        *op = (LVNodeOp)saved_op;
    }
    off += sizeof(uint8_t);

    uint8_t saved_level;
    if ((result = pread_helper(fd, &saved_level, sizeof(uint8_t), off)) != LV_OK) goto _return;
    if (level) {
        *level = saved_level;
    }
    off += sizeof(uint8_t);

    if ((result = pread_helper(fd, BUF_32, 4, off)) != LV_OK) goto _return;
    if (key_len) {
        *key_len = get_fixed_32(BUF_32);
    }
    off += 4;

    if ((result = pread_helper(fd, BUF_32, 4, off)) != LV_OK) goto _return;
    if (value_len) {
        *value_len = get_fixed_32(BUF_32);
    }
    off += 4;

    if ((result = pread_helper(fd, BUF_64, 8, off)) != LV_OK) goto _return;
    if (vector_id) {
        *vector_id = get_fixed_64(BUF_64);
    }
    off += 8;

    if ((result = pread_helper(fd, BUF_32, 4, off)) != LV_OK) goto _return;
    if (field_mask) {
        *field_mask = get_fixed_32(BUF_32);
    }
    off += 4;

    if ((result = pread_helper(fd, BUF_32, 4, off)) != LV_OK) goto _return;
    if (field_count) {
        *field_count = get_fixed_32(BUF_32);
    }
    off += 4;

    if ((result = pread_helper(fd, BUF_32, 4, off)) != LV_OK) goto _return;
    if (field_size) {
        *field_size = get_fixed_32(BUF_32);
    }
    off += 4;

    if (read_bytes_out) {
        *read_bytes_out = off - read_offset;
    }

_return:
    return result;
}

LVStatus sst_read_record_tail(const int fd, const LVOffset64_t read_offset, char* key, const LVKeyLen32_t key_len, char* value, const LVValueLen32_t value_len, char* field, const LVSize32_t field_size, LVOffset64_t* read_bytes_out) {
    LVStatus result = LV_OK;
    uint64_t off = read_offset;

    char saved_key_buffer[key_len];
    char saved_value_buffer[value_len];
    char saved_field_buffer[field_size];

    if ((result = pread_helper(fd, saved_key_buffer, key_len, off)) != LV_OK) goto _return;
    if (key) {
        memcpy(key, saved_key_buffer, key_len);
    }
    off += key_len;

    if ((result = pread_helper(fd, saved_value_buffer, value_len, off)) != LV_OK) goto _return;
    if (value) {
        memcpy(value, saved_value_buffer, value_len);
    }
    off += value_len;

    if ((result = pread_helper(fd, saved_field_buffer, field_size, off)) != LV_OK) goto _return;
    if (field) {
        schema_field_disk_to_memory(saved_field_buffer, field_size, field);
    }
    off += field_size;

    if (read_bytes_out) {
        *read_bytes_out = off - read_offset;
    }

_return:
    return result;
}

LVStatus sst_query_with_hnsw(const int fd, const int vector_index_fd, const LVVectorId64_t vector_id, const LVSchema* schema, const LVAstNode* query, const LVSSTQueryCtx* query_ctx) {
    LVStatus result = LV_OK;
    uint8_t BUF_32[4];
    uint8_t BUF_64[8];

    if (fd < 0) goto _return;

    if ((result = pread_helper(vector_index_fd, BUF_64, 8, vector_id * 8)) != LV_OK) goto _return;

    const uint64_t record_offset = get_fixed_64(BUF_64);
    uint64_t record_head_size = 0;

    LVSeq64_t seq;
    LVNodeOp op;
    LVLevel8_t level;
    LVKeyLen32_t key_len;
    LVValueLen32_t value_len;
    LVVectorId64_t saved_vector_id;
    LVFieldMask32_t field_mask;
    LVSize32_t field_count;
    LVSize32_t field_size;

    if ((result = sst_read_record_head(fd, record_offset, &seq, &op, &level, &key_len, &value_len, &vector_id, &field_mask, &field_count, &field_size, &record_head_size)) != LV_OK) goto _return;
    const uint64_t tail_offset = record_offset + record_head_size;

    if (field_count > 0 && (query_ctx->query_field_mask & field_mask)) {
        char node_buf[sizeof(LVNode) + key_len + value_len + field_size];
        LVNode* dummy_node = (LVNode*)node_buf;
        dummy_node->level = 0;
        dummy_node->key_len = key_len;
        dummy_node->value_len = value_len;
        dummy_node->field_mask = field_mask;
        dummy_node->field_count = field_count;

        char key[key_len];
        char value[value_len];

        if ((result = sst_read_record_tail(fd, tail_offset, key, key_len, value, value_len, (char*)(node_access_field(dummy_node, 0)), field_size, NULL)) != LV_OK) goto _return;

        if (node_eval_query(dummy_node, query, schema)) {
            LVOrdbyValue ordbyvalue;
            ordbyvalue.i64 = 0;

            switch (query_ctx->ordbytype)
            {
            case LV_ORDBY_FLOAT: {
                double value = node_get_f64_field(dummy_node, query_ctx->ordby_field_mask);
                ordbyvalue.f64 = value;
                break;
            }
            case LV_ORDBY_INT: {
                int64_t value = node_get_int64_field(dummy_node, query_ctx->ordby_field_mask);
                ordbyvalue.i64 = value;
                break;
            }
            case LV_ORDBY_VEC: {
                ordbyvalue.score = query_ctx->vector_score;
                break;
            }

            default:
                break;
            }

            if ((result = query_ctx->qvset_append_fn(query_ctx->qvset, dummy_node->seq, vector_id, key, key_len, value, value_len, query_ctx->vector_score, ordbyvalue, 0)) != LV_OK) goto _return;

            result = LV_QFILTER_T;
        }
        else {
            result = LV_QFILTER_F;
        }

    }

_return:
    return result;
}

// On LV_OK, *entry is filled and the caller OWNS entry->key (must free it).
// On any non-OK return, entry->key is already freed internally.
LVStatus sst_search_index_block(const int fd, LVSSTIndexBlockEntry* entry, const void* key, const LVKeyLen32_t key_len) {
    LVStatus result = LV_OK;

    uint64_t index_block_offset = 0;
    uint64_t saved_record_count = 0;

    if ((result = sst_read_footer(fd, &index_block_offset, &saved_record_count, NULL, NULL)) != LV_OK) goto _return;

    uint64_t index_entry_offset = index_block_offset;
    uint64_t index_read = 0;
    int is_found = 0;

    while (index_read < saved_record_count) {
        if ((result = sst_read_index_entry_at_offset(fd, index_entry_offset, entry, &index_entry_offset)) != LV_OK) goto _return;

        if (node_key_equal(entry->key, entry->key_len, key, key_len)) {
            is_found = 1;
            break;
        }
        else {
            free(entry->key);
        }

        index_read += 1;
    }

    if (!is_found) {
        result = LV_ERR_NOT_FOUND;
    }

_return:
    return result;
}
