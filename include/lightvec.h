#ifndef LV
#define LV

#include "lv_internal.h"

LVStatus lv_open(LightVec **db, const LVSchema* schema,const char *path);

LVStatus lv_put(LightVec* db, const void* key, const LVKeyLen32_t key_len, const void* value, const LVValueLen32_t value_len,const void* vector,const LVCount32_t field_count, const LVMetaField* fields, const LVVectorMetric vecotr_metric);

LVStatus lv_query(const LightVec* db, const char* query, const void* query_vector, const LVQueryOption* option, LVTableQueryResultSet**outputs);

#endif
