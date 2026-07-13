#include "hash.h"

LVHash32_t fnv1a_hash(const void* value, const LVSize32_t size){
    const uint8_t* ptr = (const uint8_t*)value;

    LVHash32_t hash = LV_FNV_OFFSET_BASIS;

    for (LVSize32_t i = 0; i < size; ++i) {
        hash ^= ptr[i];
        hash *= LV_FNV_PRIME;
    }

    return hash;
}