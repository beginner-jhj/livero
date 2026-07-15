#include "helper.h"
#include <stdint.h>
#include <string.h>

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
