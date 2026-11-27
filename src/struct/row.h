#ifndef LV_ROW
#define LV_ROW

#include <assert.h>
#include <stdint.h>

#include "file/file.h"
#include "livero_types.h"
#include "vector/ternary.h"

typedef struct LVRow {
    LVSeq_t seq;
    uint64_t epoch_date;
    LVOffset_t value_offset;
    LVSize16_t value_len;
    uint8_t tombstone_flag;
    uint32_t checksum;
} LVRow;

/*
    row follows little endian bytes order on both memory and disk.
    You should access the row data by using row_get_* functions.
*/

#define LV_ROW_BYTES 27
#define LV_ROW_EPOCH_DATE_OFFSET 8
#define LV_ROW_VALOFFSET_OFFSET 16
#define LV_ROW_VALLEN_OFFSET 20
#define LV_ROW_TOMBSTONEFLAG_OFFSET 22
#define LV_ROW_CEHCKSUM_OFFSET 23

static inline LVSeq_t lv_row_get_seq(const uint8_t* row) {
    assert(row);
    return lv_get_fixed_64(row);
}
static inline uint64_t lv_row_get_epoch_date(const uint8_t* row) {
    assert(row);
    return lv_get_fixed_64(row + LV_ROW_EPOCH_DATE_OFFSET);
}
static inline LVOffset_t lv_row_get_value_offset(const uint8_t* row) {
    assert(row);
    return lv_get_fixed_32(row + LV_ROW_VALOFFSET_OFFSET);
}
static inline LVSize16_t lv_row_get_value_len(const uint8_t* row) {
    assert(row);
    return lv_get_fixed_16(row + LV_ROW_VALLEN_OFFSET);
}
static inline uint8_t lv_row_get_tombstone_flag(const uint8_t* row) {
    assert(row);
    return *(row + LV_ROW_TOMBSTONEFLAG_OFFSET);
}

static inline int lv_row_is_alive(const uint8_t* row) {
    assert(row);
    return lv_row_get_tombstone_flag(row) == 0;
}

static inline uint32_t lv_row_get_checksum(const uint8_t* row) {
    assert(row);
    return lv_get_fixed_32(row + LV_ROW_CEHCKSUM_OFFSET);
}

void lv_row_write_on_memory(uint8_t* buf, const LVRow* data);

typedef struct LVRowCalcChecksumCtx {
    LVSeq_t seq;
    uint64_t epoch_date;
    LVOffset_t value_offset;
    LVSize16_t value_len;
    LVTernaryChunk* ternary_chunks;
    void* value;
    LVDim_t vector_dimension;
} LVRowCalcChecksumCtx;

uint32_t lv_row_calc_checksum(const LVRowCalcChecksumCtx* ctx);

typedef struct LVRowCCFDCtx {
    LVFILEDef* row_file;
    LVFILEDef* ter_file;
    LVFILEDef* val_file;
    lv_file_read read;
    LVOffset_t row_offset;
    LVOffset_t ternary_offset;
    LVSize32_t ternary_bytes;
} LVRowCCFDCtx;

LVStatus lv_row_checksum_check_from_disk(LVRowCCFDCtx* ctx);

LVQueryResult lv_row_make_query_result(const uint8_t* row);

#endif
