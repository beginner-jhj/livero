#include "helper.h"

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include "internal.h"
#include "livero_types.h"

LVSize_t PAGE_SIZE = 0;

__attribute__((constructor)) void page_size_init() { PAGE_SIZE = sysconf(_SC_PAGE_SIZE); }

LVStatus concat_file_with_path(const char* path, const char* file_name, char* outbuf, const LVSize_t outbuf_size) {
    assert(outbuf_size <= LV_MAX_PATH_LEN);
    assert(file_name[0] != '/');

    LVSize_t path_len = strnlen(path, outbuf_size);
    LVSize_t file_name_len = strnlen(file_name, outbuf_size);

    bool path_ends_with_slash = path[path_len - 1] == '/';

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

LVStatus open_file(const char* db_path, const char* file_name, const int flags, int* fd_out) {
    assert(db_path && file_name && fd_out);

    char PATH[LV_MAX_PATH_LEN] = {0};
    LVStatus status = concat_file_with_path(db_path, file_name, PATH, LV_MAX_PATH_LEN);
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

LVStatus close_file(int fd) {
    if (fd == -1) {
        return LV_OK;
    }
    return close(fd) < 0 ? LV_ERR_IO : LV_OK;
}

void put_fixed_16(uint8_t* buf, uint16_t value) {
    buf[0] = (uint8_t)(value & 0xff);
    buf[1] = (uint8_t)((value >> 8) & 0xff);
}
uint16_t get_fixed_16(const uint8_t* buf) { return ((uint16_t)buf[0]) | ((uint16_t)buf[1] << 8); }

void put_fixed_32(uint8_t* buf, uint32_t value) {
    buf[0] = (uint8_t)(value & 0xff);
    buf[1] = (uint8_t)((value >> 8) & 0xff);
    buf[2] = (uint8_t)((value >> 16) & 0xff);
    buf[3] = (uint8_t)((value >> 24) & 0xff);
}

uint32_t get_fixed_32(const uint8_t* buf) {
    return (((uint32_t)buf[0]) | (((uint32_t)buf[1] << 8)) | (((uint32_t)buf[2] << 16)) | (((uint32_t)buf[3] << 24)));
}

void put_fixed_64(uint8_t* buf, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        int shift = 8 * i;
        buf[i] = (uint8_t)((value >> shift) & 0xff);
    }
}

uint64_t get_fixed_64(const uint8_t* buf) {
    uint64_t result = 0x00;
    for (int i = 0; i < 8; ++i) {
        int shift = 8 * i;
        result |= (((uint64_t)buf[i]) << shift);
    }

    return result;
}

LVStatus pread_helper(const int fd, void* buf, const LVSize_t len, const LVSize_t offset) {
    LVSize_t _read = pread(fd, buf, len, offset);
    if (_read < 0) {
        return LV_ERR_IO;
    }

    return LV_OK;
}

LVStatus pwrite_helper(const int fd, const void* buf, const LVSize_t len, const LVSize_t offset) {
    LVSize_t _written = pwrite(fd, buf, len, offset);
    if (_written < 0) {
        return LV_ERR_IO;
    }

    return LV_OK;
}

uint64_t convert_date_to_epoch(const LVDate* date) {
    assert(date_is_valid(date));

    uint32_t year  = date->year;
    uint32_t month = date->month;
    uint32_t day   = date->day;

    // Treat the year as starting on March 1st, so Feb 29 falls at the very end.
    if (month <= 2) {
        year -= 1;
    }

    // Month index counted from March: Mar=0, ..., Jan=10, Feb=11.
    uint32_t mp = (month > 2) ? month - 3 : month + 9;

    // 0-based day of the March-based year: [0, 365].
    uint32_t doy = (153 * mp + 2) / 5 + (day - 1);

    // Days from 0000-03-01 to this date.
    uint64_t days = 365ULL * year + year / 4 - year / 100 + year / 400 + doy;

    // 719468 = days from 0000-03-01 to 1970-01-01.
    days -= 719468;

    return days * 86400ULL
         + (uint64_t)date->hour * 3600
         + (uint64_t)date->min * 60
         + date->sec;
}

void convert_epoch_to_date(const uint64_t epoch_time, LVDate* date) {
    date->sec = epoch_time % 60;
    uint64_t remaining_minutes = epoch_time / 60;

    date->min = remaining_minutes % 60;
    uint64_t remaining_hours = remaining_minutes / 60;

    date->hour = remaining_hours % 24;
    uint64_t days = remaining_hours / 24;

    days += 719468;

    // 3. 400years = 146097days
    uint64_t era = days / 146097;
    uint32_t doe = days % 146097; // era [0, 146096]

    uint32_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // era [0, 399]
    uint32_t y = yoe + era * 400;
    uint32_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // 0, 365]

    uint32_t mp = (5 * doy + 2) / 153; // [0, 11]
    uint8_t day = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    uint8_t month = mp < 10 ? mp + 3 : mp - 9;  // [1, 12]

    date->year = y + (month <= 2);
    date->month = month;
    date->day = day;
}

bool date_is_valid(const LVDate* date){
    assert(date);

    if(date->month < 1 || date->month > 12){
        return false;
    }

    if(date->day < 1 || date->day > 31){
        return false;
    }

    const uint8_t time_values[] = {
        date->hour,
        date->min,
        date->sec
    };

    for(LVSize_t i=0; i<sizeof(time_values)/sizeof(time_values[0]); ++i){
        const uint8_t t_value = time_values[i];
        if(t_value>59){
            return false;
        }
    }

    return true;
}
