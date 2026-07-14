#ifndef HELPER
#define HELPER

#include <unistd.h>
#include "lv_internal.h"
#include <stdlib.h>

#define WRITE_BUFFER_SIZE 4096

LVStatus write_helper(const int fd, const void* buf, const LVSize32_t len);
LVStatus read_helper(const int fd, void* buf, const LVSize32_t len);

LVStatus pwrite_helper(const int fd, const void* buf, const LVSize32_t len, const LVOffset64_t offset);
LVStatus pread_helper(const int fd, void* buf, const LVSize32_t len, const LVOffset64_t offset );

#endif
