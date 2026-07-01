#ifndef LV
#define LV

#include "lv_internal.h"

LVStatus lv_create(LightVec** db, const char* path, const LVSize32_t flush_threshold,
    const LVDim32_t vector_dim, const LVVectorType vector_type,
    const LVVectorMetric vector_metric,
    const LVCount32_t field_count, const LVMetaFieldDef* field_defs);

LVStatus lv_open(LightVec** db, const char* path, const LVSize32_t flush_threshold);

LVStatus lv_put(LightVec* db, const void* key, const LVKeyLen32_t key_len, const void* value, const LVValueLen32_t value_len, const void* vector, LVCount32_t field_count, const LVMetaField* fields);

LVStatus lv_update_value(LightVec* db, const void* key, const LVKeyLen32_t key_len, const void* value, const LVValueLen32_t value_len);

LVStatus lv_update_vector(LightVec* db, const void* key, const LVKeyLen32_t key_len, const void* vector);

LVStatus lv_update_field(LightVec* db, const void* key, const LVKeyLen32_t key_len, const LVSize32_t field_count, const LVMetaField* fields);

LVStatus lv_delete(LightVec* db, const void* key, const LVKeyLen32_t key_len);

LVStatus lv_query(const LightVec* db, const char* query, const void* query_vector, const LVQueryOption* option, LVQueryResultSet** outputs);

LVStatus lv_close(LightVec* db);

void lv_destroy_query_result_set(LVQueryResultSet* qrset);

#endif
