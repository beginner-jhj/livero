#ifndef LV_ROW
#define LV_ROW

#include <stdint.h>
#include <assert.h>

#include "livero_types.h"

typedef struct LVRowDisk {
    uint8_t seq[8];
    uint8_t date[8];
    uint8_t value_offset[4];
    uint8_t value_len[2];
    uint8_t tombstone_flag;
    /* LVRowDisk checksum covers its own row+vector+value (row->vector->value order).
    They share the same lifecycle.
    The tombstone_flag is excluded from checksum because it can vary after checksum calculation.
    */
    uint8_t checksum[4];
} __attribute__((packed)) LVRowDisk;

#define LV_ROW_DISK_BYTES 27
#define LV_ROW_DISK_DATE_OFFSET 8
#define LV_ROW_DISK_VALOFFSET_OFFSET 16
#define LV_ROW_DISK_VALLEN_OFFSET 20
#define LV_ROW_DISK_TOBSTONEFLAG_OFFSET 22
#define LV_ROW_DISK_CEHCKSUM_OFFSET 23

static_assert(sizeof(LVRowDisk) == LV_ROW_DISK_BYTES, "LVRowDisk size must be 27 bytes.");

static inline LVSeq_t lv_get_row_seq(const uint8_t *row) {
    assert(row);
    return *(LVSeq_t *)row;
}
static inline uint64_t lv_get_row_date(const uint8_t *row) {
    assert(row);
    return *(uint64_t *)(row + LV_ROW_DISK_DATE_OFFSET);
}
static inline LVOffset_t lv_get_row_value_offset(const uint8_t *row) {
    assert(row);
    return *(LVOffset_t *)(row + LV_ROW_DISK_VALOFFSET_OFFSET);
}
static inline LVSize16_t lv_get_row_value_len(const uint8_t *row) {
    assert(row);
    return *(LVSize16_t *)(row + LV_ROW_DISK_VALLEN_OFFSET);
}
static inline int lv_row_is_alive(const uint8_t *row) {
    assert(row);
    return *(row + LV_ROW_DISK_TOBSTONEFLAG_OFFSET) == 0;
}

static inline uint32_t lv_get_row_checksum(const uint8_t *row) {
    assert(row);
    return *(uint32_t *)(row + LV_ROW_DISK_CEHCKSUM_OFFSET);
}

/*
    Convert a row buffer to a LVRowDisk.
    -Every byte is set to little-endian
*/
LVRowDisk lv_row_m2d(const uint8_t* row);

/*
    Convert a LVRowDisk to a LV_ROW_DISK_BYTES buffer
    -does not check checksum, only move data to the buffer.
*/
LVStatus lv_row_d2m(const int fd,const LVOffset_t offset, uint8_t* row_out);


typedef struct LVRowCalcChecksumCtx {
    LVSeq_t seq;
    uint64_t date;
    LVOffset_t value_offset;
    LVSize16_t value_len;
    void* vector;
    void* value;
    LVDim_t vector_dim;
} LVRowCalcChecksumCtx;

uint32_t lv_calc_row_checksum(const LVRowCalcChecksumCtx* ctx);

typedef struct LVRowCCFDCtx {
    int row_fd;
    int vec_fd;
    int val_fd;
    LVOffset_t row_offset;
    LVOffset_t vec_offset;
    LVDim_t vector_dim;
} LVRowCCFDCtx;

LVStatus lv_row_checksum_check_from_disk(LVRowCCFDCtx* ctx);

#endif
