#ifndef LV_ROW
#define LV_ROW

#include <stdint.h>
#include <assert.h>

#include "file/file.h"
#include "livero_types.h"

/*
    row follows little endian bytes order on both memory and disk.
    You should access the row data by using row_get_* functions.
*/

#define LV_ROW_DISK_BYTES 27
#define LV_ROW_DISK_DATE_OFFSET 8
#define LV_ROW_DISK_VALOFFSET_OFFSET 16
#define LV_ROW_DISK_VALLEN_OFFSET 20
#define LV_ROW_DISK_TOBSTONEFLAG_OFFSET 22
#define LV_ROW_DISK_CEHCKSUM_OFFSET 23

static inline LVSeq_t row_get_seq(const uint8_t *row) {
    assert(row);
    return lv_get_fixed_64(row);
}
static inline uint64_t row_get_date(const uint8_t *row) {
    assert(row);
    return lv_get_fixed_64(row + LV_ROW_DISK_DATE_OFFSET);
}
static inline LVOffset_t row_get_value_offset(const uint8_t *row) {
    assert(row);
    return lv_get_fixed_32(row + LV_ROW_DISK_VALOFFSET_OFFSET);
}
static inline LVSize16_t row_get_value_len(const uint8_t *row) {
    assert(row);
    return lv_get_fixed_16(row + LV_ROW_DISK_VALLEN_OFFSET);
}
static inline int row_is_alive(const uint8_t *row) {
    assert(row);
    return *(row + LV_ROW_DISK_TOBSTONEFLAG_OFFSET) == 0;
}

static inline uint32_t row_get_checksum(const uint8_t *row) {
    assert(row);
    return lv_get_fixed_32(row + LV_ROW_DISK_CEHCKSUM_OFFSET);
}

typedef struct LVRow{
    LVSeq_t seq;
    uint64_t date;
    LVOffset_t value_offset;
    LVSize16_t value_len;
    uint8_t tombstone_flag;
    uint32_t checksum;
} LVRow;
void row_write_on_memory(uint8_t* buf, const LVRow* data);


typedef struct LVRowCalcChecksumCtx {
    LVSeq_t seq;
    uint64_t date;
    LVOffset_t value_offset;
    LVSize16_t value_len;
    void* vector;
    void* value;
    LVDim_t vector_dim;
} LVRowCalcChecksumCtx;

uint32_t row_calc_checksum(const LVRowCalcChecksumCtx* ctx);

typedef struct LVRowCCFDCtx {
    int row_fd;
    int vec_fd;
    int val_fd;
    LVOffset_t row_offset;
    LVOffset_t vec_offset;
    LVSize32_t vector_bytes;
} LVRowCCFDCtx;

LVStatus row_checksum_check_from_disk(LVRowCCFDCtx* ctx);


#endif
