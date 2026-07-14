#include "crc.h"


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
