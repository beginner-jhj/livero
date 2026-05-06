#ifndef HASH
#define HASH

#include "lv_internal.h"

//FNV1a: used for Bloom filter double hashing.
#define FNV_OFFSET_BASIS 0x811C9DC5u
#define FNV_PRIME        0x01000193u
#define FNV_SEED         0xDEADBEEFu

LVHash32_t fnv1a_hash(const void* value, const uint32_t size);

#endif
