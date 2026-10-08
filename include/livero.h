#ifndef LIVERO
#define LIVERO

#include <stdint.h>

#include "livero_types.h"

typedef struct Livero Livero;

LVStatus lv_create(const char* db_path, const LVConfig* config, Livero** db_out);
LVStatus lv_open(const char* db_path, Livero** db_out);
LVStatus lv_put(Livero* db, const void* vector, const LVDate* date, const void* value, const uint16_t value_len,
                LVSeq_t* key_out);
LVStatus lv_get(const Livero* db, const LVSeq_t key, LVDate* date_out, void* value_out);
LVStatus lv_query(Livero* db, const LVTopK_t k, const void* qeury_vector, const LVDate* date1, const LVDate* date2,
                  LVQueryResult** result_out, LVSize_t* result_size_out);
LVStatus lv_delete(Livero* db, const LVSeq_t key);
void lv_db_info(const Livero* db, LVDBInfo* info_out);
void lv_close(Livero* db);

#endif
