#include "row.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "file/file.h"
#include "helper/crc32c.h"
#include "livero_types.h"
#include "vector/ternary.h"

void row_write_on_memory(uint8_t* buf, const LVRow* data) {
    assert(buf && data);
    lv_put_fixed_64(buf, data->seq);
    lv_put_fixed_64(buf + LV_ROW_DISK_DATE_OFFSET, data->date);
    lv_put_fixed_32(buf + LV_ROW_DISK_VALOFFSET_OFFSET, data->value_offset);
    lv_put_fixed_16(buf + LV_ROW_DISK_VALLEN_OFFSET, data->value_len);
    memcpy(buf + LV_ROW_DISK_TOBSTONEFLAG_OFFSET, &data->tombstone_flag, sizeof(data->tombstone_flag));
    lv_put_fixed_32(buf + LV_ROW_DISK_CEHCKSUM_OFFSET, data->checksum);
}

uint32_t row_calc_checksum(const LVRowCalcChecksumCtx* ctx) {
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

LVStatus row_checksum_check_from_disk(LVRowCCFDCtx* ctx) {
    assert(ctx);
    uint8_t row[LV_ROW_DISK_BYTES] = {0};
    uint8_t BUF64[8];
    uint8_t BUF32[4];
    uint8_t BUF16[2];

    LVStatus s = lv_read_file(ctx->row_fd, row, LV_ROW_DISK_BYTES, ctx->row_offset);

    if (s != LV_OK) {
        return s;
    }

    uint32_t checksum = lv_crc32c_begin();

    lv_put_fixed_64(BUF64, row_get_seq(row));
    checksum = lv_crc32c_update(checksum, BUF64, 8);

    lv_put_fixed_64(BUF64, row_get_date(row));
    checksum = lv_crc32c_update(checksum, BUF64, 8);

    const LVOffset_t value_offset = row_get_value_offset(row);
    lv_put_fixed_32(BUF32, value_offset);
    checksum = lv_crc32c_update(checksum, BUF32, 4);

    const LVSize16_t value_len = row_get_value_len(row);
    lv_put_fixed_16(BUF16, value_len);
    checksum = lv_crc32c_update(checksum, BUF16, 2);

    const LVSize32_t vector_bytes = ctx->vector_bytes;
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

    if (checksum != row_get_checksum(row)) {
        return LV_ERR_DATA_CORRUPT;
    }

    return LV_OK;
}
