#ifndef HELPER
#define HELPER

#include <unistd.h>
#include "lv_internal.h"
#include <time.h>

LVStatus write_helper(const int fd, const void* buf, const uint32_t len);
LVStatus read_helper(const int fd, const void* buf, const uint32_t len);

uint32_t xorshift(){
    uint32_t x = time(NULL);
    x ^= x<<13;
    x ^= x>>17;
    x ^= x<<5;
    return x;
}

void safe_free(void* ptr){
    if(ptr){
        free(ptr);
        ptr = NULL;
    }
}

#endif
