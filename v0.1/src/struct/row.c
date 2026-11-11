#include "row.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "file/file.h"
#include "helper/crc32c.h"
#include "livero_internal.h"
#include "livero_types.h"
#include "vector/ternary.h"

void row_write_on_memory(uint8_t* buf, const LVRow* data) {
    assert(buf && data);
    lv_put_fixed_64(buf, data->seq);
    lv_put_fixed_64(buf + LV_ROW_DATE_OFFSET, data->date);
    lv_put_fixed_32(buf + LV_ROW_VALOFFSET_OFFSET, data->value_offset);
    lv_put_fixed_16(buf + LV_ROW_VALLEN_OFFSET, data->value_len);
    memcpy(buf + LV_ROW_TOMBSTONEFLAG_OFFSET, &data->tombstone_flag, sizeof(data->tombstone_flag));
    lv_put_fixed_32(buf + LV_ROW_CEHCKSUM_OFFSET, data->checksum);
}

uint32_t row_calc_checksum(const LVRowCalcChecksumCtx* ctx) {
    assert(ctx);
    assert(ctx->ternary_chunks && ctx->value);

    uint8_t BUF64[8];
    uint8_t BUF32[4];
    uint8_t BUF16[2];

    uint32_t checksum = lv_crc32c_begin();

    lv_put_fixed_64(BUF64, ctx->seq);
    checksum = lv_crc32c_update(checksum, BUF64, 8);

    lv_put_fixed_64(BUF64, ctx->epoch_date);
    checksum = lv_crc32c_update(checksum, BUF64, 8);

    lv_put_fixed_32(BUF32, ctx->value_offset);
    checksum = lv_crc32c_update(checksum, BUF32, 4);

    lv_put_fixed_16(BUF16, ctx->value_len);
    checksum = lv_crc32c_update(checksum, BUF16, 2);

    checksum = lv_crc32c_update(checksum, ctx->ternary_chunks, ternary_chunks_bytes(ctx->vector_dimension));
    checksum = lv_crc32c_update(checksum, ctx->value, ctx->value_len);

    return lv_crc32c_end(checksum);
}

LVStatus row_checksum_check_from_disk(LVRowCCFDCtx* ctx) {
    assert(ctx);
    uint8_t row[LV_ROW_BYTES] = {0};

    LVStatus s = lv_read_file(ctx->row_fd, row, LV_ROW_BYTES, ctx->row_offset);

    if (s != LV_OK) {
        return s;
    }

    uint32_t checksum = lv_crc32c_begin();
    checksum = lv_crc32c_update(checksum, row, LV_ROW_TOMBSTONEFLAG_OFFSET);

    const LVOffset_t value_offset = row_get_value_offset(row);
    const LVSize16_t value_len = row_get_value_len(row);

    uint8_t checksum_buffer[LV_MAX_VLA_LEN] = {0};

    if ((s = lv_crc32c_update_from_file(ctx->ternary_fd, ctx->ternary_offset, ctx->ternary_bytes, checksum_buffer,
                                        sizeof(checksum_buffer), &checksum)) != LV_OK) {
        return s;
    }

    if ((s = lv_crc32c_update_from_file(ctx->val_fd, value_offset, value_len, checksum_buffer,
                                        sizeof(checksum_buffer), &checksum)) != LV_OK) {
        return s;
    }

    checksum = lv_crc32c_end(checksum);

    if (checksum != row_get_checksum(row)) {
        return LV_ERR_DATA_CORRUPT;
    }

    return LV_OK;
}
