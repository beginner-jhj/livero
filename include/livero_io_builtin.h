#ifndef LV_IO_BUILTIN
#define LV_IO_BUILTIN

#include "livero_types.h"

#if defined(__unix__) || defined(__APPLE__) || defined (ESP_PLATFORM)

#if defined(__APPLE__)
#ifndef _DARWIN_C_SOURCE
#define _DARWIN_C_SOURCE
#endif
#endif

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <unistd.h>

struct LVFILEDef {
    int fd;
};

static LVStatus lv_posix_file_open(const char* path, const int is_create, LVFILEDef* file_out) {
    assert(path && file_out);

    int flags = O_CLOEXEC;
    if (is_create) {
        flags |= O_CREAT | O_EXCL | O_RDWR;
    } else {
        flags |= O_RDWR;
    }

    file_out->fd = -1;

    int fd = open(path, flags, 0644);

    if (fd < 0) {
        if (errno == EEXIST) {
            return LV_ERR_EXIST;
        } else if (errno == ENOENT) {
            return LV_ERR_NOT_EXIST;
        }
        return LV_ERR_IO;
    }

    file_out->fd = fd;

    return LV_OK;
}

static LVStatus lv_posix_file_close(LVFILEDef* file) {
    assert(file);
    if (file->fd == -1) {
        return LV_ERR_INVALID_PARAM;
    }
    LVStatus s = LV_OK;
    if (close(file->fd) < 0) {
        s = LV_ERR_IO;
    }
    file->fd = -1;
    return s;
}

static LVStatus lv_posix_file_write(LVFILEDef* file, const void* data, const LVSize32_t size, const LVOffset_t offset) {
    assert(file && data);
    uint8_t* p = (uint8_t*)data;
    size_t left = size;
    off_t off = (off_t)offset;

    while (left > 0) {
        ssize_t n = pwrite(file->fd, p, left, off);
        if (n < 0) {
            if (errno == EINTR) {
                continue;  // interrupted by a signal: retry
            }
            return LV_ERR_IO;
        }
        if (n == 0) {
            return LV_ERR_IO;  // EOF before `size` bytes: file is truncated
        }
        p += n;
        left -= (size_t)n;
        off += n;
    }
    return LV_OK;
}

static LVStatus lv_posix_file_read(LVFILEDef* file, void* buf, const LVSize32_t size, const LVOffset_t offset) {
    assert(file && buf);
    uint8_t* p = buf;
    size_t left = size;
    off_t off = (off_t)offset;

    while (left > 0) {
        ssize_t n = pread(file->fd, p, left, off);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return LV_ERR_IO;
        }
        if (n == 0) {
            return LV_ERR_IO;
        }
        p += n;
        left -= (size_t)n;
        off += n;
    }
    return LV_OK;
}

static LVStatus lv_posix_file_sync(LVFILEDef* file) {
    assert(file);
#if defined(__APPLE__)
    // fsync on macOS does not flush the drive's write cache.
    if (fcntl(file->fd, F_FULLFSYNC) == 0) {
        return LV_OK;
    } else {
        return LV_ERR_IO;
    }
#endif
    return fsync(file->fd) < 0 ? LV_ERR_IO : LV_OK;
}

static const LVIOOps POSIX_IO_OPS = {.open = lv_posix_file_open,
                                     .close = lv_posix_file_close,
                                     .write = lv_posix_file_write,
                                     .read = lv_posix_file_read,
                                     .sync = lv_posix_file_sync};

static const LVSize16_t POSIX_FILE_DEF_SIZE = sizeof(LVFILEDef);

#endif

#endif
