#include "row.h"

#include <string.h>

#include "file/file.h"
#include "helper/crc32c.h"
#include "vector/ternary.h"

LVRowDisk lv_row_m2d(const uint8_t* row) {
    assert(row);
    LVRowDisk row_disk;
    LVOffset_t off = 0;

    LVSeq_t seq;
    memcpy(&seq, row, sizeof(seq));
    lv_put_fixed_64(row_disk.seq, seq);
    off += sizeof(seq);

    uint64_t date;
    memcpy(&date, row + off, sizeof(date));
    lv_put_fixed_64(row_disk.date, date);
    off += sizeof(date);

    LVOffset_t val_offset;
    memcpy(&val_offset, row + off, sizeof(val_offset));
    lv_put_fixed_32(row_disk.value_offset, val_offset);
    off += sizeof(val_offset);

    LVSize16_t val_len;
    memcpy(&val_len, row + off, sizeof(val_len));
    lv_put_fixed_16(row_disk.value_len, val_len);
    off += sizeof(val_len);

    row_disk.tombstone_flag = *(row + off);
    off += 1;

    uint32_t checksum;
    memcpy(&checksum, row + off, sizeof(checksum));
    lv_put_fixed_32(row_disk.checksum, checksum);

    return row_disk;
}

LVStatus lv_row_d2m(const int fd, const LVOffset_t offset, uint8_t* row_out) {
    assert(row_out);
    LVRowDisk row_disk;
    LVStatus s = lv_read_file(fd, &row_disk, LV_ROW_DISK_BYTES, offset);
    if (s != LV_OK) {
        return s;
    }

    LVOffset_t off = 0;
    LVSeq_t seq = lv_get_fixed_64(row_disk.seq);
    memcpy(row_out, &seq, sizeof(seq));
    off += sizeof(seq);

    uint64_t date = lv_get_fixed_64(row_disk.date);
    memcpy(row_out + off, &date, sizeof(date));
    off += sizeof(date);

    LVOffset_t value_offset = lv_get_fixed_32(row_disk.value_offset);
    memcpy(row_out + off, &value_offset, sizeof(value_offset));
    off += sizeof(value_offset);

    LVSize16_t value_len = lv_get_fixed_16(row_disk.value_len);
    memcpy(row_out + off, &value_len, sizeof(value_len));
    off += sizeof(value_len);

    memcpy(row_out + off, &row_disk.tombstone_flag, sizeof(row_disk.tombstone_flag));
    off += sizeof(row_disk.tombstone_flag);

    uint32_t checksum = lv_get_fixed_32(row_disk.checksum);
    memcpy(row_out + off, &checksum, sizeof(checksum));

    return LV_OK;
}

uint32_t lv_calc_row_checksum(const LVRowCalcChecksumCtx* ctx) {
    assert(ctx);
    assert(ctx->vector && ctx->value);

    uint8_t BUF64[8];
    uint8_t BUF32[4];
    uint8_t BUF16[2];

    uint32_t checksum = lv_crc32c_begin();

    lv_put_fixed_64(BUF64, ctx->seq);
    checksum = lv_crc32c_update(checksum, BUF64, 8);

    lv_put_fixed_64(BUF64, ctx->date);
    checksum = lv_crc32c_update(checksum, BUF64, 8);

    lv_put_fixed_32(BUF32, ctx->value_offset);
    checksum = lv_crc32c_update(checksum, BUF32, 4);

    lv_put_fixed_16(BUF16, ctx->value_len);
    checksum = lv_crc32c_update(checksum, BUF16, 2);

    checksum = lv_crc32c_update(checksum, ctx->vector, ternary_chunks_bytes(ctx->vector_dim));
    checksum = lv_crc32c_update(checksum, ctx->value, ctx->value_len);

    return lv_crc32c_end(checksum);
}

LVStatus lv_row_checksum_check_from_disk(LVRowCCFDCtx* ctx) {
    assert(ctx);
    uint8_t row[LV_ROW_DISK_BYTES] = {0};
    uint8_t BUF64[8];
    uint8_t BUF32[4];
    uint8_t BUF16[2];

    LVStatus s = lv_row_d2m(ctx->row_fd, ctx->row_offset, row);

    if (s != LV_OK) {
        return s;
    }

    uint32_t checksum = lv_crc32c_begin();

    lv_put_fixed_64(BUF64, lv_get_row_seq(row));
    checksum = lv_crc32c_update(checksum, BUF64, 8);

    lv_put_fixed_64(BUF64, lv_get_row_date(row));
    checksum = lv_crc32c_update(checksum, BUF64, 8);

    const LVOffset_t value_offset = lv_get_row_value_offset(row);
    lv_put_fixed_32(BUF32, value_offset);
    checksum = lv_crc32c_update(checksum, BUF32, 4);

    const LVSize16_t value_len = lv_get_row_value_len(row);
    lv_put_fixed_16(BUF16, value_len);
    checksum = lv_crc32c_update(checksum, BUF16, 2);

    const LVSize32_t vector_bytes = ternary_chunks_bytes(ctx->vector_dim);
    const LVSize32_t vec_vla_len = vector_bytes < LV_MAX_VLA_LEN ? vector_bytes : LV_MAX_VLA_LEN;
    uint8_t vector[vec_vla_len];
    LVSize32_t vec_total_read = 0;
    while (vec_total_read < vector_bytes) {
        LVSize32_t remaining = vector_bytes - vec_total_read;
        LVSize32_t read_bytes = remaining > vec_vla_len ? vec_vla_len : remaining;

        if ((s = lv_read_file(ctx->vec_fd, vector, read_bytes, ctx->vec_offset + vec_total_read)) != LV_OK) {
            return s;
        }

        vec_total_read += read_bytes;
        checksum = lv_crc32c_update(checksum, vector, read_bytes);
    }

    const LVSize32_t val_vla_len = value_len < LV_MAX_VLA_LEN ? value_len : LV_MAX_VLA_LEN;
    uint8_t value[val_vla_len];
    LVSize32_t val_total_read = 0;
    while (val_total_read < value_len) {
        LVSize32_t remaining = value_len - val_total_read;
        LVSize32_t read_bytes = remaining > val_vla_len ? val_vla_len : remaining;

        if ((s = lv_read_file(ctx->val_fd, value, read_bytes, value_offset + val_total_read)) != LV_OK) {
            return s;
        }

        val_total_read += read_bytes;
        checksum = lv_crc32c_update(checksum, value, read_bytes);
    }

    checksum = lv_crc32c_end(checksum);

    if (checksum != lv_get_row_checksum(row)) {
        return LV_ERR_CORRUPT;
    }

    return LV_OK;
}
