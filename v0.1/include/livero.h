#ifndef LIVERO
#define LIVERO

#include "livero_types.h"

LVStatus lv_create(const char* db_path, const LVConfig* config, const LVOption* option, Livero** db_out);
LVStatus lv_open(const char* db_path, const LVOption* option, Livero** db_out);
LVStatus lv_put(Livero* db, void* vector,void* value,const LVSize16_t value_len, const LVDate* date, LVSeq_t* seq_out);
LVStatus lv_get(Livero* db, const LVSeq_t target_seq, LVGetResult* result);
LVStatus lv_delete(Livero* db, const LVSeq_t target_seq);
LVStatus lv_query(Livero* db,const LVTopK_t k, const void* query_vector, const LVDate* date1, const LVDate* date2, LVQueryResult* result, LVSize32_t* result_size);
void lv_close(Livero* db);
void lv_stat(const Livero* db, LVDBStat* stat);

#endif
