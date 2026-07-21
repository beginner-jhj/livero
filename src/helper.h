#ifndef HELPER
#define HELPER

#include <unistd.h>
#include "lv_internal.h"
#include <stdlib.h>

#define WRITE_BUFFER_SIZE 4096

//CRC32: IEEE 802.3 polynomial in reflected (LSB-first) form.
#define LV_CRC32_SEED       0xFFFFFFFFu
#define LV_CRC32_POLY       0xEDB88320u

#define LV_FNV_OFFSET_BASIS 0x811C9DC5u
#define LV_FNV_PRIME        0x01000193u
#define LV_FNV_SEED         0xDEADBEEFu

LVHash32_t fnv1a_hash(const void* value, const LVSize32_t size);

uint32_t crc_calc(const void* data, const LVSize32_t size, const uint32_t seed);

LVStatus write_helper(const int fd, const void* buf, const LVSize32_t len);
LVStatus read_helper(const int fd, void* buf, const LVSize32_t len);

LVStatus pwrite_helper(const int fd, const void* buf, const LVSize32_t len, const LVOffset64_t offset);
LVStatus pread_helper(const int fd, void* buf, const LVSize32_t len, const LVOffset64_t offset );

#endif
