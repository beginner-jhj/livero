#include "helper.h"
#include <stdint.h>
#include <string.h>

static char WRITE_BUFFER[WRITE_BUFFER_SIZE];
static uint32_t write_helper_pos = 0;
static int write_helper_last_fd = -1;

LVStatus write_helper(const int fd, const void *buf, const uint32_t len)
{
    LVStatus result = LV_OK;

    if (len > WRITE_BUFFER_SIZE)
    {
        if (write(fd, buf, len) < (ssize_t)len)
            return LV_ERR_IO;
        return LV_OK;
    }

    if ((write_helper_last_fd != -1 && write_helper_last_fd != fd) || (write_helper_pos + len > WRITE_BUFFER_SIZE))
    {
        if (write_helper_pos > 0)
        {
            ssize_t written = write(write_helper_last_fd, WRITE_BUFFER, write_helper_pos);
            if (written < (ssize_t)write_helper_pos)
            {
                return LV_ERR_IO;
            }
            write_helper_pos = 0;
        }
    }

    write_helper_last_fd = fd;

    memcpy(WRITE_BUFFER + write_helper_pos, buf, len);
    write_helper_pos += len;

    return LV_OK;
}

LVStatus write_helper_flush(const int fd, const int sync)
{

    if (write_helper_last_fd != fd)
    {
        return LV_ERR_INVALID;
    }

    if (write_helper_pos > 0 && write_helper_last_fd != -1)
    {
        ssize_t written = write(fd, WRITE_BUFFER, write_helper_pos);
        if (written < 0)
        {
            return LV_ERR_IO;
        }

        else if (written < (ssize_t)write_helper_pos)
        {
            return LV_ERR_FULL;
        }

        if (sync == 1)
        {
            fsync(fd);
        }

        write_helper_pos = 0;
    }

    return LV_OK;
}

LVStatus read_helper(const int fd, const void *buf, const uint32_t len)
{
    ssize_t _read = read(fd, buf, len);
    if (read < 0)
    {
        return LV_ERR_IO;
    }

    else if (read < len)
    {
        return LV_ERR_FULL;
    }

    return LV_OK;
}

uint32_t xorshift(void)
{
    uint32_t x = time(NULL);
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

void safe_free(void **ptr)
{
    if (ptr && *ptr)
    {
        free(*ptr);
        *ptr = NULL;
    }
}