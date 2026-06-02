#ifndef LV
#define LV

#include "lv_internal.h"

LVStatus lv_open(LightVec **db, const LVSchema* schema,const char *path,const LVSize32_t flush_threshold);

LVStatus lv_put(LightVec* db, const void* key, const LVKeyLen32_t key_len, const void* value, const LVValueLen32_t value_len,const void* vector,const LVCount32_t field_count, const LVMetaField* fields, const LVVectorMetric vecotr_metric);

LVStatus lv_query(const LightVec* db, const char* query, const void* query_vector, const LVQueryOption* option, LVQueryResultSet**outputs);

LVStatus lv_close(LightVec* db);

void lv_destroy_query_result_set(LVQueryResultSet* qrset);

//todo lv_update, lv_delete
//LV_NODE_UPDATE flag should be added
#endif
