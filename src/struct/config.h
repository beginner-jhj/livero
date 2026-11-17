#ifndef LV_CONFIG
#define LV_CONFIG

#include <assert.h>
#include <stdint.h>

#include "livero_types.h"

typedef struct LVConfigDisk {
    uint8_t max_disk_bytes[4];
    uint8_t max_value_len[2];
    uint8_t vector_dimension[2];
    LVVectorType vector_type;  // 1B
    uint8_t checksum[4];
} __attribute__((packed)) LVConfigDisk;

#define LV_CONFIG_DISK_BYTES 13

static_assert(sizeof(LVConfigDisk) == LV_CONFIG_DISK_BYTES, "LVConfigDisk size must be 17 bytes.");

/*
    Convert a LVConfig(memory) to a LVConfigDisk.
    -Every byte is set to little-endian
    -Checksum is calculated based on little-endian bytes.
*/
LVConfigDisk lv_config_m2d(const LVConfig* config);

/*
    Convert a LVConfigDisk to a LVConfig
*/
LVStatus lv_config_d2m(LVFILEDef* config_file, lv_file_read read, LVConfig* config_out);

/*
    Write LVConfigDisk on the config.lv
*/
LVStatus lv_save_config(LVFILEDef* config_file, const LVConfig* config, lv_file_write write, lv_file_sync sync);

#endif
