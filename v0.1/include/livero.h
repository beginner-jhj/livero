#ifndef LIVERO
#define LIVERO

#include "livero_types.h"

LVStatus lv_create(const char* db_path, const LVConfig* config, const LVOption* option, Livero** db_out);
LVStatus lv_open(const char* db_path, const LVOption* option, Livero** db_out);
LVStatus lv_put(Livero* db, void* vector,void* value,const LVSize16_t value_len, const LVDate* date, LVSeq_t* seq_out);

#endif
