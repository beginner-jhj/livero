#ifndef LV_FILE_H
#define LV_FILE_H

#include "livero_types.h"

LVStatus lv_concat_path(const char* path, const char* file_name, char* outbuf, const LVSize32_t outbuf_size);
LVStatus lv_open_file(const char* db_path, const char* file_name, const int flags, int* fd_out);
LVStatus lv_close_file(int fd);
LVStatus lv_write_file(int fd, const void* data, const LVSize32_t size, const LVOffset_t offset);
LVStatus lv_read_file(const int fd, void* buf, const LVSize32_t size, const LVOffset_t offset);
LVStatus lv_sync_file(int fd);

void lv_put_fixed_16(uint8_t* buf, uint16_t value);
uint16_t lv_get_fixed_16(const uint8_t* buf);
void lv_put_fixed_32(uint8_t* buf, uint32_t value);
uint32_t lv_get_fixed_32(const uint8_t* buf);
void lv_put_fixed_64(uint8_t* buf, uint64_t value);
uint64_t lv_get_fixed_64(const uint8_t* buf);

#endif
