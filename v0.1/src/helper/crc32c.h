#ifndef LV_CRC32C_H
#define LV_CRC32C_H

#include <stddef.h>
#include <stdint.h>

#include "livero_types.h"

// CRC-32C (Castagnoli), the same variant computed by the x86 SSE4.2 `crc32`
// instruction and the ARMv8 CRC32C instructions, so a hardware path can be
// added later without changing any stored checksum.
//
// Usage, one shot:
//     uint32_t c = lv_crc32c(buf, len);
//
// Usage, streaming (e.g. rows + vectors + values of one batch):
//     uint32_t c = lv_crc32c_begin();
//     c = lv_crc32c_update(c, rows, rows_len);
//     c = lv_crc32c_update(c, vecs, vecs_len);
//     c = lv_crc32c_end(c);
// Streaming gives exactly the same result as hashing the concatenated bytes.

uint32_t lv_crc32c_begin(void);
uint32_t lv_crc32c_update(uint32_t crc, const void* data, size_t len);
uint32_t lv_crc32c_end(uint32_t crc);
uint32_t lv_crc32c(const void* data, size_t len);
LVStatus lv_crc32c_update_from_file(const int fd, const LVOffset_t offset, const LVSize32_t size, uint8_t* buf,
                                 const LVSize32_t buf_len, uint32_t* checksum);

#endif  // LV_CRC32C_H
