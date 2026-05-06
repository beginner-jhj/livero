#include "crc.h"

static uint32_t CRC_TABLE[256];
static int TABLE_INITIALIZED = 0;

void create_crc_table(void)
{
    if (TABLE_INITIALIZED)
        return;

    for (int i = 0; i < 256; ++i)
    {
        uint32_t crc = i;
        for (int j = 0; j < 8; ++j)
        {
            if (crc & 1)
            {
                crc = (crc >> 1) ^ CRC32_POLY;
            }
            else
            {
                crc >>= 1;
            }
        }
        CRC_TABLE[i] = crc;
    }
}

uint32_t crc_calc(const void* data, const uint32_t size, const uint32_t seed){
    const uint8_t *p = (const uint8_t*)data;
    uint32_t crc = seed;

    for(int i=0; i<size; ++i){
        const uint8_t index = (crc ^ p[i])&(0xFF);
        crc = (crc>>8)^(CRC_TABLE[index]);
    }

    return crc;
}
