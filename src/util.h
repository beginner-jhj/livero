#ifndef UTIL
#define UTIL

#include "lv_internal.h"
#include <string.h>

LVStatus path_join(char* buf,uint32_t buf_size, const char* path, const char* dir);

void put_fixed_32(uint8_t* buf, uint32_t value);
uint32_t get_fixed_32(const uint8_t* buf);

void put_fixed_64(uint8_t* buf, uint64_t value);
uint64_t get_fixed_64(const uint8_t *buf);

uint32_t xorshift(void);

#endif
