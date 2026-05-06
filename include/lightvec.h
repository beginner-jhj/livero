#ifndef LV
#define LV

#include "lv_internal.h"

LVStatus lv_open(LightVec **db, const LVSchema* schema,const char *path);

LVStatus lv_put(const LightVec* db, const void* key, const LVKeyLen32_t key_len, const void* value, const LVValueLen32_t value_len,const LVSize32_t vector_dim, const float* vector,const LVCount32_t field_count, const LVMetaField* fields);

#endif
