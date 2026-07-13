#include "hash.h"

LVHash32_t fnv1a_hash(const void* value, const uint32_t size){
    uint8_t* ptr = (const uint8_t*)value;

    uint32_t hash = LV_FNV_OFFSET_BASIS;

    for(int i=0; i<size; ++i){
        hash ^= ptr[i];
        hash *= LV_FNV_PRIME;
    }

    return hash;
}