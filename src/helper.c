#include "helper.h"
#include <stdint.h>
#include <string.h>

LVHash32_t fnv1a_hash(const void* value, const LVSize32_t size){
    const uint8_t* ptr = (const uint8_t*)value;

    LVHash32_t hash = LV_FNV_OFFSET_BASIS;

    for (LVSize32_t i = 0; i < size; ++i) {
        hash ^= ptr[i];
        hash *= LV_FNV_PRIME;
    }

    return hash;
}

/*
 * Build the 256-entry lookup table on first use (lazy init).
 * Each entry is the CRC of a single byte value, precomputed so crc_calc can
 * process one byte per table lookup instead of 8 bit-shifts.
 *
 * NOTE: not thread-safe. The `is_crc_table_initialized` guard has a race if
 * two threads first-call concurrently. Fine for livero's single-writer model;
 * revisit if CRC is ever called from multiple threads before the table exists.
 */

static uint32_t CRC_TABLE[256];
static int is_crc_table_initialized = 0;

static void create_crc_table(void)
{
    if (is_crc_table_initialized)
        return;

    for (int i = 0; i < 256; ++i)
    {
        uint32_t crc = i;
        for (int j = 0; j < 8; ++j)
        {
            if (crc & 1)
            {
                crc = (crc >> 1) ^ LV_CRC32_POLY;
            }
            else
            {
                crc >>= 1;
            }
        }
        CRC_TABLE[i] = crc;
    }
    is_crc_table_initialized = 1;
}


/*
 * Compute CRC32 over `size` bytes, starting from `seed`.
 * Pass LV_CRC32_SEED for a fresh checksum, or a previous result to chain.
 */
uint32_t crc_calc(const void* data, const LVSize32_t size, const uint32_t seed){
    create_crc_table();
    const uint8_t *p = (const uint8_t*)data;
    uint32_t crc = seed;

    for(LVSize32_t i=0; i<size; ++i){
        const uint8_t index = (crc ^ p[i])&(0xFF);
        crc = (crc>>8)^(CRC_TABLE[index]);
    }

    return crc ^ 0xFFFFFFFF;;
}

LVStatus write_helper(const int fd, const void* buf, const LVSize32_t len)
{
    ssize_t written = write(fd, buf, len);
    if (written < 0) return LV_ERR_IO;
    if ((uint32_t)written < len) return LV_ERR_FULL;
    return LV_OK;
}


LVStatus read_helper(const int fd, void* buf, const LVSize32_t len)
{
    ssize_t _read = read(fd, buf, len);
    if (_read < 0)
    {
        return LV_ERR_IO;
    }

    if ((uint32_t)_read < len)
    {
        // requested a fixed-width / length-prefixed field but the file ended
        // early -> the record is truncated, i.e. the file is corrupt.
        return LV_ERR_TRUNCATED;     // distinct from LV_ERR_IO: not a disk error,
        // the bytes simply aren't there
    }

    return LV_OK;
}

LVStatus pread_helper(const int fd, void* buf, const LVSize32_t len, const LVOffset64_t offset) {
    ssize_t _read = pread(fd, buf, len, offset);
    if (_read < 0) {
        return LV_ERR_IO;
    }

    if ((uint32_t)_read < len)
    {
        // requested a fixed-width / length-prefixed field but the file ended
        // early -> the record is truncated, i.e. the file is corrupt.
        return LV_ERR_TRUNCATED;     // distinct from LV_ERR_IO: not a disk error,
        // the bytes simply aren't there
    }

    return LV_OK;
}

LVStatus pwrite_helper(const int fd, const void* buf, const LVSize32_t len, const LVOffset64_t offset) {
    ssize_t _written = pwrite(fd, buf, len, offset);
    if (_written < 0) {
        return LV_ERR_IO;
    }

    if ((uint32_t)_written < len)
    {
        return LV_ERR_FULL;
    }

    return LV_OK;
}
