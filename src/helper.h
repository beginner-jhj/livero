#ifndef HELPER
#define HELPER

#include <unistd.h>
#include "lv_internal.h"
#include <time.h>
#include <stdlib.h>

#define WRITE_BUFFER_SIZE 4096

LVStatus write_helper(const int fd, const void* buf, const uint32_t len);
LVStatus write_helper_flush(const int fd, const int sync);
uint64_t write_helper_get_offset(const int fd);
LVStatus read_helper(const int fd, void* buf, const uint32_t len);
LVStatus pread_helper(const int fd, void* buf, const uint32_t len, const uint64_t offset );

uint32_t xorshift(void);

void safe_free(void** ptr);

#endif
