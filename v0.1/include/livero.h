#ifndef LIVERO
#define LIVERO

#include "livero_types.h"

typedef struct Livero Livero;

LVStatus lv_create(const char* db_path, const LVConfig* config, const LVOption* option, Livero** db_out);
LVStatus lv_open(const char* db_path, const LVOption* option, Livero** db_out);

#endif
