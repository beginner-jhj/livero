#include "file.h"

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include "livero_internal.h"
#include "livero_types.h"

LVStatus lv_concat_path(const char* path, const char* file_name, char* outbuf, const LVSize32_t outbuf_size) {
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

LVStatus lv_file_open_single(const char* db_path, const int is_create, const char* file_name, LVFILEDef* file,
                             lv_file_open open) {
    char PATH[LV_MAX_PATH_LEN] = {0};
    LVStatus s = lv_concat_path(db_path, file_name, PATH, sizeof(PATH));
    if (s != LV_OK) {
        return s;
    }
    if ((s = open(PATH, is_create, file)) != LV_OK) {
        LIVERO_ERROR("Failed to open %s.", file_name);
    }
    return s;
}

LVStatus lv_file_open_all(const char* db_path, const int is_create, LVFileDefList* file_list, lv_file_open open) {
    assert(db_path && file_list && open);

    LVStatus s = lv_file_open_single(db_path, is_create, LV_ROW_FILE_NAME, file_list->row_file, open);
    if (s != LV_OK) {
        return s;
    }

    if ((s = lv_file_open_single(db_path, is_create, LV_TERNARY_FILE_NAME, file_list->ternary_file, open)) != LV_OK) {
        return s;
    }

    if ((s = lv_file_open_single(db_path, is_create, LV_VAL_FILE_NAME, file_list->value_file, open)) != LV_OK) {
        return s;
    }

    return lv_file_open_single(db_path, is_create, LV_COMMIT_FILE_NAME, file_list->commit_file, open);
}
