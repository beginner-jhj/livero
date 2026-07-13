#ifndef LV
#define LV

#include "livero_types.h"

struct Livero;

LVStatus lv_create(Livero** db, const char* path, const LVSize32_t flush_threshold,
    const LVDim32_t vector_dim, const LVVectorType vector_type,
    const LVVectorMetric vector_metric,
    const LVCount32_t field_count, const LVMetaFieldDef* field_defs);

LVStatus lv_open(Livero** db, const char* path, const LVSize32_t flush_threshold);

LVStatus lv_put(Livero* db, const void* key, const LVKeyLen32_t key_len, const void* value, const LVValueLen32_t value_len, const void* vector, LVCount32_t field_count, const LVMetaField* fields);

LVStatus lv_get(const Livero* db, const void* key, const LVKeyLen32_t key_len, LVGetResult** output);
void lv_destroy_get_result(LVGetResult* result);

LVStatus lv_update_value(Livero* db, const void* key, const LVKeyLen32_t key_len, const void* value, const LVValueLen32_t value_len);

LVStatus lv_update_vector(Livero* db, const void* key, const LVKeyLen32_t key_len, const void* vector);

LVStatus lv_update_field(Livero* db, const void* key, const LVKeyLen32_t key_len, const LVSize32_t field_count, const LVMetaField* fields);

LVStatus lv_delete(Livero* db, const void* key, const LVKeyLen32_t key_len);

LVStatus lv_query(const Livero* db, const char* query, const void* query_vector, const LVQueryOption* option, LVQueryResultSet** outputs);

LVStatus lv_close(Livero* db);

LVDim32_t lv_get_vector_dim(const Livero* db);

LVVectorType lv_get_vector_type(const Livero* db);

void lv_destroy_query_result_set(LVQueryResultSet* qrset);

#endif
