#ifndef LV_HELPER
#define LV_HELPER

#include <fcntl.h>
#include <stdint.h>

#include "livero_types.h"

extern LVSize_t PAGE_SIZE;
void page_size_init(void);

/*
Concatenate the file name with the path and store a result path in the out-buffer.
The outbuf_size must be smaller than LV_MAX_PATH_LEN.
*/
LVStatus concat_file_with_path(const char* path, const char* file_name, char* outbuf, const LVSize_t outbuf_size);

LVStatus open_file(const char* db_path, const char* file_name, const int flags, int* fd_out);

LVStatus close_file(int fd);

LVStatus pread_helper(const int fd, void* buf, const LVSize_t len, const LVSize_t offset);

LVStatus pwrite_helper(const int fd, const void* buf, const LVSize_t len, const LVSize_t offset);

void put_fixed_16(uint8_t* buf, uint16_t value);
uint16_t get_fixed_16(const uint8_t* buf);
void put_fixed_32(uint8_t* buf, uint32_t value);
uint32_t get_fixed_32(const uint8_t* buf);
void put_fixed_64(uint8_t* buf, uint64_t value);
uint64_t get_fixed_64(const uint8_t* buf);

uint64_t convert_date_to_epoch(const LVDate* date);
void convert_epoch_to_date(const uint64_t epoch_time, LVDate* date);
bool date_is_valid(const LVDate* date);

#endif
