#include "file.h"

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include "helper/crc32c.h"
#include "livero_internal.h"

static LVStatus lv_concat_path(const char* path, const char* file_name, char* outbuf, const LVSize32_t outbuf_size) {
    assert(outbuf_size <= LV_MAX_PATH_LEN);
    assert(file_name[0] != '/');

    LVSize32_t path_len = strnlen(path, outbuf_size);
    LVSize32_t file_name_len = strnlen(file_name, outbuf_size);

    int path_ends_with_slash = path[path_len - 1] == '/';

    if (path_ends_with_slash) {
        if (path_len + file_name_len + 1 > outbuf_size) {  // path len + file name len + null terminator
            LIVERO_ERROR("Can not concatenate path, the result path is too long.");
            return LV_ERR_INVALID_PARAM;
        }
        memcpy(outbuf, path, path_len);
        memcpy(outbuf + path_len, file_name, file_name_len);
        outbuf[path_len + file_name_len] = '\0';
        return LV_OK;
    } else {
        if (path_len + file_name_len + 2 > outbuf_size) {  // path len + file name len + slash + null terminator
            LIVERO_ERROR("Can not concatenate path, the result path is too long.");
            return LV_ERR_INVALID_PARAM;
        }
        memcpy(outbuf, path, path_len);
        outbuf[path_len] = '/';
        memcpy(outbuf + path_len + 1, file_name, file_name_len);
        outbuf[path_len + file_name_len + 1] = '\0';
        return LV_OK;
    }
}

LVStatus lv_open_file(const char* db_path, const char* file_name, const int flags, int* fd_out) {
    assert(db_path && file_name && fd_out);

    char PATH[LV_MAX_PATH_LEN] = {0};
    LVStatus status = lv_concat_path(db_path, file_name, PATH, LV_MAX_PATH_LEN);
    if (status != LV_OK) {
        return status;
    }

    int fd = open(PATH, flags, 0644);

    if (fd < 0) {
        if (errno == EEXIST) {
            return LV_ERR_EXSIST;
        } else if (errno == ENOENT) {
            return LV_ERR_DNEXSIST;
        }
        return LV_ERR_IO;
    }

    *fd_out = fd;

    return LV_OK;
}

LVStatus lv_close_file(int fd) {
    if (fd == -1) {
        return LV_OK;
    }
    return close(fd) < 0 ? LV_ERR_IO : LV_OK;
}

LVStatus lv_write_file(int fd, const void* data, const LVSize32_t size, const LVOffset_t offset) {
    LVSize32_t _written = pwrite(fd, data, size, offset);
    if (_written < 0) {
        return LV_ERR_IO;
    }

    return LV_OK;
}

void lv_put_fixed_16(uint8_t* buf, uint16_t value) {
    buf[0] = (uint8_t)(value & 0xff);
    buf[1] = (uint8_t)((value >> 8) & 0xff);
}
uint16_t lv_get_fixed_16(const uint8_t* buf) { return ((uint16_t)buf[0]) | ((uint16_t)buf[1] << 8); }

void lv_put_fixed_32(uint8_t* buf, uint32_t value) {
    buf[0] = (uint8_t)(value & 0xff);
    buf[1] = (uint8_t)((value >> 8) & 0xff);
    buf[2] = (uint8_t)((value >> 16) & 0xff);
    buf[3] = (uint8_t)((value >> 24) & 0xff);
}

uint32_t lv_get_fixed_32(const uint8_t* buf) {
    return (((uint32_t)buf[0]) | (((uint32_t)buf[1] << 8)) | (((uint32_t)buf[2] << 16)) | (((uint32_t)buf[3] << 24)));
}

void lv_put_fixed_64(uint8_t* buf, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        int shift = 8 * i;
        buf[i] = (uint8_t)((value >> shift) & 0xff);
    }
}

uint64_t lv_get_fixed_64(const uint8_t* buf) {
    uint64_t result = 0x00;
    for (int i = 0; i < 8; ++i) {
        int shift = 8 * i;
        result |= (((uint64_t)buf[i]) << shift);
    }

    return result;
}

LVConfigDisk lv_config_m2d(const LVConfig* config) {
    assert(config);
    LVConfigDisk cfg_disk;
    uint32_t checksum = lv_crc32c_begin();

    lv_put_fixed_32(cfg_disk.max_disk_bytes, config->max_disk_bytes);
    checksum = lv_crc32c_update(checksum, cfg_disk.max_disk_bytes, 4);

    lv_put_fixed_32(cfg_disk.disk_sector_bytes, config->disk_sector_bytes);
    checksum = lv_crc32c_update(checksum, cfg_disk.disk_sector_bytes, 4);

    lv_put_fixed_16(cfg_disk.max_value_len, config->max_value_len);
    checksum = lv_crc32c_update(checksum, cfg_disk.max_value_len, 2);

    lv_put_fixed_16(cfg_disk.vector_dimension, config->vector_dimension);
    checksum = lv_crc32c_update(checksum, cfg_disk.vector_dimension, 2);

    cfg_disk.vector_type = config->vector_type;

    lv_put_fixed_32(cfg_disk.checksum, checksum);

    return cfg_disk;
}
