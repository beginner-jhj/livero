#ifndef LV_TEST_HELPER
#define LV_TEST_HELPER

#include <stdint.h>
#include "livero_types.h"

static const uint32_t _1kb = 1u << 10;
static const uint32_t _1mb = _1kb*1024;
static const char TEST_CHAR_POOL[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789[]{}():;,.?!@#$^&*";
static const uint32_t TEST_CHAR_POOL_SIZE = sizeof(TEST_CHAR_POOL) - 1;

int clean_test_dir(const char* dir);

int8_t rand_int8(int8_t min, int8_t max);
int rand_int(int min, int max);
float rand_fp32(float min, float max);

void fill_int8_vector(const LVDim_t dim, int8_t* vector_out);

void fill_fp32_vector(const LVDim_t dim, float* vector_out);

void fill_random_chr_data(const LVSize_t len, char* data_out);

void fill_random_date(LVDate* date_out, const uint16_t min_year, const uint16_t max_year);

int compare_date(const LVDate* date_a, const LVDate* date_b);

void print_status(const LVStatus status);
void  make_first_date(const uint16_t year,LVDate* date);
void make_last_date(const uint16_t year, LVDate* date);

#endif
