#ifndef UTIL
#define UTIL

#include "lv_internal.h"
#include <string.h>

LVStatus path_join(char* buf, LVSize32_t buf_size, const char* path, const char* dir);

void put_fixed_32(uint8_t* buf, LVSize32_t value);
LVSize32_t get_fixed_32(const uint8_t* buf);

void put_fixed_64(uint8_t* buf, LVOffset64_t value);
LVOffset64_t get_fixed_64(const uint8_t *buf);

LVCount32_t xorshift(void);

#endif
