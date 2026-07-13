#ifndef CRC
#define CRC

#include "lv_internal.h"

//CRC32: IEEE 802.3 polynomial in reflected (LSB-first) form.
#define LV_CRC32_SEED       0xFFFFFFFFu
#define LV_CRC32_POLY       0xEDB88320u

uint32_t crc_calc(const void* data, const uint32_t size, const uint32_t seed);


#endif
