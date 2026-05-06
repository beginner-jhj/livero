#include "helper.h"

LVStatus write_helper(const int fd, const void* buf, const uint32_t len){
    ssize_t written = write(fd, buf, len);
    if(written < 0){
        return LV_ERR_IO;
    }

    else if(written < len){
        return LV_ERR_FULL;
    }

    return LV_OK;
}

LVStatus read_helper(const int fd, const void* buf, const uint32_t len){
    ssize_t _read = read(fd, buf, len);
    if(read < 0){
        return LV_ERR_IO;
    }

    else if(read < len){
        return LV_ERR_FULL;
    }

    return LV_OK;
}

uint32_t xorshift(void){
    uint32_t x = time(NULL);
    x ^= x<<13;
    x ^= x>>17;
    x ^= x<<5;
    return x;
}

void safe_free(void** ptr){
    if(ptr && *ptr){
        free(*ptr);
        *ptr = NULL;
    }
}