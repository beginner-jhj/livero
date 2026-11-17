#ifndef LV_FILE_H
#define LV_FILE_H

#include "livero_types.h"

typedef struct {
    LVFILEDef* row_file;
    LVFILEDef* ternary_file;
    LVFILEDef* value_file;
    LVFILEDef* commit_file;
} LVFileDefList;

LVStatus lv_concat_path(const char* path, const char* file_name, char* outbuf, const LVSize32_t outbuf_size);

void lv_put_fixed_16(uint8_t* buf, uint16_t value);
uint16_t lv_get_fixed_16(const uint8_t* buf);
void lv_put_fixed_32(uint8_t* buf, uint32_t value);
uint32_t lv_get_fixed_32(const uint8_t* buf);
void lv_put_fixed_64(uint8_t* buf, uint64_t value);
uint64_t lv_get_fixed_64(const uint8_t* buf);

LVStatus lv_file_open_single(const char* db_path, const int is_create, const char* file_name, LVFILEDef* file,
                             lv_file_open open);
LVStatus lv_file_open_all(const char* db_path, const int is_create, LVFileDefList* file_list, lv_file_open open);

#endif
