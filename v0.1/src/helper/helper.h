#ifndef LV_HELPER
#define LV_HELPER
#include "livero_types.h"
#include <stdint.h>
uint64_t lv_convert_date_to_epoch(const LVDate* date);
void lv_convert_epoch_to_date(const uint64_t epoch_time, LVDate* date);
int lv_date_is_valid(const LVDate* date);
int lv_check_option(const LVOption* option);
#endif

