#include "util.h"
#include <time.h>

LVStatus path_join(char* buf, LVSize32_t buf_size, const char* path, const char* dir)
{
    LVStatus result = LV_OK;
    size_t path_len = 0;
    size_t dir_len = 0;

    if (!buf || !path || !dir || buf_size == 0)
    {
        result = LV_ERR_INVALID;
        goto _return;
    }

    path_len = strlen(path);
    dir_len = strlen(dir);

    const int need_sep = (path_len > 0 && path[path_len - 1] != '/') ? 1 : 0;
    if (path_len + need_sep + dir_len + 1 > buf_size ||
        path_len + need_sep + dir_len + 1 > LV_PATH_MAX)
    {
        result = LV_ERR_FULL;
        goto _return;
    }

    memset(buf, 0, buf_size);
    memcpy(buf, path, path_len);
    if (need_sep)
    {
        buf[path_len] = '/';
    }
    memcpy(buf + path_len + need_sep, dir, dir_len);
    buf[path_len + need_sep + dir_len] = '\0';

_return:
    return result;
}

void put_fixed_32(uint8_t* buf, LVSize32_t value)
{
    buf[0] = (uint8_t)(value & 0xff);
    buf[1] = (uint8_t)((value >> 8) & 0xff);
    buf[2] = (uint8_t)((value >> 16) & 0xff);
    buf[3] = (uint8_t)((value >> 24) & 0xff);
}

LVSize32_t get_fixed_32(const uint8_t* buf)
{
    return (
        ((uint32_t)buf[0]) |
        ((uint32_t)(buf[1] << 8)) |
        ((uint32_t)(buf[2] << 16)) |
        ((uint32_t)(buf[3] << 24)));
}

void put_fixed_64(uint8_t* buf, LVOffset64_t value)
{
    for (int i = 0; i < 8; ++i)
    {
        int shift = 8 * i;
        buf[i] = (uint8_t)((value >> shift) & 0xff);
    }
}

LVOffset64_t get_fixed_64(const uint8_t* buf)
{
    uint64_t result = 0x00;
    for (int i = 0; i < 8; ++i)
    {
        int shift = 8 * i;
        result |= (((uint64_t)buf[i]) << shift);
    }

    return result;
}

LVCount32_t xorshift(void)
{
    static uint32_t state = 0;
    if (state == 0) state = (uint32_t)time(NULL);

    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}
