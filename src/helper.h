#ifndef HELPER
#define HELPER

#include <unistd.h>
#include "lv_internal.h"
#include <time.h>
#include <stdlib.h>

LVStatus write_helper(const int fd, const void* buf, const uint32_t len);
LVStatus read_helper(const int fd, const void* buf, const uint32_t len);

uint32_t xorshift(void);

void safe_free(void** ptr);

#endif
