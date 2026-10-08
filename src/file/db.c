#include "db.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>

#include "internal/helper.h"
#include "internal/internal.h"
#include "livero_types.h"
#include "vector/ternary.h"

typedef struct LVDBHeaderRecord {
    uint8_t next_seq_key[8];
    uint8_t record_count[4];
    uint8_t max_disk_bytes[4];
    uint8_t current_disk_bytes[4];
    uint8_t next_val_log_offset[4];
    uint8_t vector_dim[2];
    LVVectorType vector_type;
} __attribute__((packed)) LVDBHeaderRecord;

static_assert(sizeof(LVDBHeaderRecord) == LV_DB_FILE_HEADER_SIZE);

typedef struct LVDBRecord {
    uint8_t tombstone;
    uint8_t seq_key[8];
    uint8_t date[8];
    uint8_t value_len[2];
    uint8_t val_log_offset[4];
} __attribute__((packed)) LVDBRecord;

static_assert(sizeof(LVDBRecord) == LV_DB_FILE_RECORD_SIZE);

LVStatus db_file_init(const int fd, const LVConfig* config) {
    assert(fd != -1);
    LVDBHeaderRecord header;
    put_fixed_64(header.next_seq_key, 0ull);
    put_fixed_32(header.record_count, 0u);
    put_fixed_32(header.current_disk_bytes, LV_DB_FILE_HEADER_SIZE);
    put_fixed_32(header.next_val_log_offset, 0u);
    put_fixed_32(header.max_disk_bytes, config->max_disk_bytes);
    put_fixed_16(header.vector_dim, config->vector_dimension);
    header.vector_type = config->vector_type;

    return pwrite_helper(fd, &header, sizeof(header), 0);
}

LVStatus db_file_read_header(const int fd, LVDBHeader* header_out) {
    assert(fd != -1);
    LVDBHeaderRecord saved_header;
    LVStatus status = pread_helper(fd, &saved_header, sizeof(saved_header), 0);
    if (status != LV_OK) {
        return status;
    }

    header_out->next_seq = get_fixed_64(saved_header.next_seq_key);
    header_out->record_count = get_fixed_32(saved_header.record_count);
    header_out->max_disk_bytes = get_fixed_32(saved_header.max_disk_bytes);
    header_out->current_disk_bytes = get_fixed_32(saved_header.current_disk_bytes);
    header_out->next_val_log_offset = get_fixed_32(saved_header.next_val_log_offset);
    header_out->vector_dim = get_fixed_16(saved_header.vector_dim);
    header_out->vector_type = saved_header.vector_type;

    return LV_OK;
}

LVStatus db_file_write(const LVDBWriteCtx* ctx, LVSeq_t* used_key_out) {
    LVDBHeader saved_header;
    LVStatus status = db_file_read_header(ctx->db_file_fd, &saved_header);
    if (status != LV_OK) {
        return status;
    }

    LVSize_t prev_disk_bytes = saved_header.current_disk_bytes;
    LVSize_t prev_record_count = saved_header.record_count;
    LVDim_t vector_dim = saved_header.vector_dim;
    LVSize_t max_disk_bytes = saved_header.max_disk_bytes;

    const LVSize_t vector_chucnks_bytes = ternary_chunks_size(vector_dim);
    const LVSize_t write_size = LV_DB_FILE_RECORD_SIZE + vector_chucnks_bytes + ctx->value_len;

    if (write_size > max_disk_bytes - prev_disk_bytes) {
        return LV_ERR_DISK_FULL;
    }

    const LVSeq_t seq_key = saved_header.next_seq;
    const LVOffset_t vec_log_record_offset = vector_chucnks_bytes * prev_record_count;
    const LVOffset_t val_log_record_offset = saved_header.next_val_log_offset;
    const LVOffset_t db_record_offset = LV_DB_FILE_HEADER_SIZE + LV_DB_FILE_RECORD_SIZE * prev_record_count;

    /*
        Write ternary vector chunks on the vec_log.lv.
    */

    if ((status = pwrite_helper(ctx->vec_log_fd, ctx->ternary_chunks, vector_chucnks_bytes, vec_log_record_offset)) !=
        LV_OK) {
        return status;
    }

    /*
        Write a value on the val_log.lv
    */
    if ((status = pwrite_helper(ctx->val_log_fd, ctx->value, ctx->value_len, val_log_record_offset)) != LV_OK) {
        return status;
    }

    LVDBRecord db_record;
    db_record.tombstone = 0x00;
    put_fixed_64(db_record.seq_key, seq_key);
    put_fixed_64(db_record.date, ctx->epoch_date);
    put_fixed_16(db_record.value_len, ctx->value_len);
    put_fixed_32(db_record.val_log_offset, val_log_record_offset);

    /*
        Write a record on the db.lv.
    */

    if ((status = pwrite_helper(ctx->db_file_fd, &db_record, sizeof(db_record), db_record_offset)) != LV_OK) {
        return status;
    }

    LVDBHeaderRecord updated_header_record;
    const LVSeq_t next_seq_key = seq_key + 1;
    const LVSize_t next_record_count = prev_record_count + 1;
    const LVSize_t next_current_disk_bytes = prev_disk_bytes + write_size;
    const LVOffset_t next_val_log_offset = val_log_record_offset + ctx->value_len;

    put_fixed_64(updated_header_record.next_seq_key, next_seq_key);
    put_fixed_32(updated_header_record.record_count, next_record_count);
    put_fixed_32(updated_header_record.current_disk_bytes, next_current_disk_bytes);
    put_fixed_32(updated_header_record.next_val_log_offset, next_val_log_offset);

    put_fixed_32(updated_header_record.max_disk_bytes, max_disk_bytes);
    put_fixed_16(updated_header_record.vector_dim, vector_dim);
    updated_header_record.vector_type = saved_header.vector_type;

    if ((status = pwrite_helper(ctx->db_file_fd, &updated_header_record, sizeof(updated_header_record), 0)) != LV_OK) {
        return status;
    }

    if (used_key_out) {
        *used_key_out = seq_key;
    };

    return LV_OK;
}

LVStatus db_file_search(const int fd, const LVSize_t record_count, const LVSeq_t targe_key, uint64_t* time_out,
                        LVOffset_t* value_offset_out, LVSize_t* value_len_out, LVOffset_t* offset_out) {
    assert(fd != -1);

    int left = 0;
    int right = record_count - 1;

    LVOffset_t offset_to_read = 0;
    LVDBRecord record;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        offset_to_read = sizeof(LVDBHeaderRecord) + mid * sizeof(LVDBRecord);

        LVStatus status = pread_helper(fd, &record, sizeof(record), offset_to_read);
        if (status != LV_OK) {
            return status;
        }

        const LVSeq_t found_seq = get_fixed_64(record.seq_key);
        if (found_seq == targe_key) {
            if (record.tombstone == 0xff) {
                return LV_ERR_NOT_FOUND;
            }
            if (value_offset_out) {
                *value_offset_out = get_fixed_32(record.val_log_offset);
            }
            if (value_len_out) {
                *value_len_out = get_fixed_16(record.value_len);
            }
            if (time_out) {
                *time_out = get_fixed_64(record.date);
            }
            if (offset_out) {
                *offset_out = offset_to_read;
            }
            return LV_OK;
        }

        if (found_seq > targe_key) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    return LV_ERR_NOT_FOUND;
}

LVStatus db_file_get_value(const int val_log_fd, const LVOffset_t offset, const LVSize_t len, void* out_buf) {
    assert(val_log_fd != -1 && out_buf);
    return pread_helper(val_log_fd, out_buf, len, offset);
}

LVStatus db_file_date_filter(const int fd, const uint32_t internal_seq, const uint64_t date1, const uint64_t date2,
                             bool* result_out) {
    assert(fd != -1 && result_out);
    LVDBRecord record;
    LVOffset_t offset_to_read = LV_DB_FILE_HEADER_SIZE + internal_seq * LV_DB_FILE_RECORD_SIZE;
    LVStatus status = pread_helper(fd, &record, sizeof(record), offset_to_read);
    if (status != LV_OK) {
        return status;
    }
    uint64_t date = get_fixed_64(record.date);
    *result_out = (record.tombstone == 0x00) & (date1 <= date) & (date <= date2);
    return LV_OK;
}

LVStatus db_file_get_vector(const int vec_log_fd, const uint32_t internal_seq, const LVDim_t dim, void* vector_out) {
    assert(vec_log_fd != -1 && vector_out);
    LVSize_t ternary_size = ternary_chunks_size(dim);
    LVOffset_t offset_to_read = ternary_size * internal_seq;
    return pread_helper(vec_log_fd, vector_out, ternary_size, offset_to_read);
}

LVStatus db_file_query_result(const int db_file_fd, const int val_log_fd, const uint32_t internal_seq,
                              LVQueryResult* query_result_out) {
    assert(db_file_fd != -1 && val_log_fd != -1 && query_result_out);

    LVDBRecord record;
    LVOffset_t offset_to_read = LV_DB_FILE_HEADER_SIZE + internal_seq * LV_DB_FILE_RECORD_SIZE;

    LVStatus status = pread_helper(db_file_fd, &record, sizeof(record), offset_to_read);
    if (status != LV_OK) {
        return status;
    }

    uint8_t value[LV_MAX_VALUE_LEN] = {0};
    const uint16_t value_len = get_fixed_16(record.value_len);
    const LVOffset_t value_offset = get_fixed_32(record.val_log_offset);
    if ((status = pread_helper(val_log_fd, value, value_len, value_offset)) != LV_OK) {
        return status;
    }

    uint64_t epoch_date = get_fixed_64(record.date);
    LVDate converted_date;
    convert_epoch_to_date(epoch_date, &converted_date);

    query_result_out->key = get_fixed_64(record.seq_key);
    query_result_out->value_len = value_len;
    query_result_out->year = converted_date.year;
    query_result_out->month = converted_date.month;
    query_result_out->day = converted_date.day;
    query_result_out->hour = converted_date.hour;
    query_result_out->min = converted_date.min;
    query_result_out->sec = converted_date.sec;
    memcpy(query_result_out->value, value, sizeof(query_result_out->value));

    return LV_OK;
}

LVStatus db_file_delete(const int fd, const LVSize_t record_count, const LVSeq_t key) {
    assert(fd != -1);
    LVOffset_t offset = 0;
    LVStatus status = db_file_search(fd, record_count, key, nullptr, nullptr, nullptr, &offset);

    if (status != LV_OK) {
        return status;
    }
    const uint8_t tombstone = 0xff;
    if ((status = pwrite_helper(fd, &tombstone, 1, offset)) != LV_OK) {
        return status;
    }
    return LV_OK;
}
