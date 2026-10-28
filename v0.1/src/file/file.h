#ifndef LV_FILE_H
#define LV_FILE_H

#include <assert.h>
#include <stdint.h>

#include "livero_types.h"


typedef struct LVConfigDisk {
    uint8_t max_disk_bytes[4];
    uint8_t disk_sector_bytes[4];
    uint8_t max_value_len[2];
    uint8_t vector_dimension[2];
    LVVectorType vector_type;  // 1B
    uint8_t checksum[4];
} __attribute__((packed)) LVConfigDisk;

#define LV_CONFIG_DISK_BYTES 17

static_assert(sizeof(LVConfigDisk) == LV_CONFIG_DISK_BYTES, "LVConfigDisk size must be 17 bytes.");

typedef struct LVRowDisk {
    uint8_t seq[8];              
    uint8_t date[8];           
    uint8_t value_offset[4];  
    uint8_t value_len[2];
    uint8_t tombstone_flag;
} __attribute__((packed)) LVRowDisk;

#define LV_ROW_DISK_BYTES 23

static_assert(sizeof(LVRowDisk) == LV_ROW_DISK_BYTES,  "LVRowDisk size must be 23 bytes.");

LVStatus lv_open_file(const char* db_path, const char* file_name, const int flags, int* fd_out);
LVStatus lv_close_file(int fd);
LVStatus lv_write_file(int fd, const void* data, const LVSize32_t size, const LVOffset_t offset);

void lv_put_fixed_16(uint8_t* buf, uint16_t value);
uint16_t lv_get_fixed_16(const uint8_t* buf);
void lv_put_fixed_32(uint8_t* buf, uint32_t value);
uint32_t lv_get_fixed_32(const uint8_t* buf);
void lv_put_fixed_64(uint8_t* buf, uint64_t value);
uint64_t lv_get_fixed_64(const uint8_t* buf);

/*
    Convert a LVConfig(memory) to a LVConfigDisk.
    -Every bytes are set to little-endian
    -Checksum is calculated based on littel-endian bytes.
*/
LVConfigDisk lv_config_m2d(const LVConfig* config);



#endif
