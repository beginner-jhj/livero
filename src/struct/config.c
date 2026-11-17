#include "config.h"

#include <stdint.h>

#include "file/file.h"
#include "helper/crc32c.h"
#include "livero_types.h"

LVConfigDisk lv_config_m2d(const LVConfig* config) {
    assert(config);
    LVConfigDisk cfg_disk;
    uint32_t checksum = lv_crc32c_begin();

    lv_put_fixed_32(cfg_disk.max_disk_bytes, config->max_disk_bytes);
    checksum = lv_crc32c_update(checksum, cfg_disk.max_disk_bytes, sizeof(cfg_disk.max_disk_bytes));

    lv_put_fixed_16(cfg_disk.max_value_len, config->max_value_len);
    checksum = lv_crc32c_update(checksum, cfg_disk.max_value_len, sizeof(cfg_disk.max_value_len));

    lv_put_fixed_16(cfg_disk.vector_dimension, config->vector_dimension);
    checksum = lv_crc32c_update(checksum, cfg_disk.vector_dimension, sizeof(cfg_disk.vector_dimension));

    cfg_disk.vector_type = config->vector_type;
    checksum = lv_crc32c_update(checksum, &cfg_disk.vector_type,
                                sizeof(cfg_disk.vector_type));  // fix bug, didn't update checksum with vector_type

    checksum = lv_crc32c_end(checksum);
    lv_put_fixed_32(cfg_disk.checksum, checksum);

    return cfg_disk;
}

LVStatus lv_config_d2m(LVFILEDef* config_file, lv_file_read read, LVConfig* config_out) {
    assert(config_out);

    LVConfigDisk cfg_disk;
    LVStatus s = read(config_file, &cfg_disk, sizeof(cfg_disk), 0);
    if (s != LV_OK) {
        return s;
    }

    uint32_t checksum = lv_crc32c(&cfg_disk, LV_CONFIG_DISK_BYTES - sizeof(cfg_disk.checksum));
    if (checksum != lv_get_fixed_32(cfg_disk.checksum)) {
        return LV_ERR_DATA_CORRUPT;
    }

    config_out->max_disk_bytes = lv_get_fixed_32(cfg_disk.max_disk_bytes);
    config_out->max_value_len = lv_get_fixed_16(cfg_disk.max_value_len);
    config_out->vector_dimension = lv_get_fixed_16(cfg_disk.vector_dimension);
    config_out->vector_type = cfg_disk.vector_type;

    return LV_OK;
}

LVStatus lv_save_config(LVFILEDef* config_file, const LVConfig* config, lv_file_write write, lv_file_sync sync) {
    assert(config_file && config && write && sync);
    LVConfigDisk cfg_disk = lv_config_m2d(config);
    LVStatus s = write(config_file, &cfg_disk, LV_CONFIG_DISK_BYTES, 0);
    if (s != LV_OK) {
        return s;
    }
    return sync(config_file);
}
