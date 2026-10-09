#include "test_helper.h"

#include <ftw.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <float.h>
#include <math.h>

#include "internal/helper.h"
#include "livero_types.h"

static int remove_callback(const char* fpath, const struct stat* sb, int typeflag, struct FTW* ftwbuf) {
    int rv = remove(fpath);  // remove(): rmdir() for a directory, unlink() for a file
    if (rv) {
        perror(fpath);
    }
    return rv;
}

int clean_test_dir(const char* dir) { return nftw(dir, remove_callback, 64, FTW_DEPTH | FTW_PHYS); }

int8_t rand_int8(int8_t min, int8_t max) {
    if (min == max) {
        return min;
    }
    return min + rand() % (max - min + 1);
}

int rand_int(int min, int max) {
    if (min == max) {
        return min;
    }
    return min + rand() % (max - min + 1);
}

float rand_fp32(float min, float max) {
    if(fabsf(min - max) <= FLT_EPSILON){
        return min;
    }
    float scale = (float)rand() / ((float)RAND_MAX + 1.0f);
    return min + scale * (max - min);
}

void fill_int8_vector(const LVDim_t dim, int8_t* vector_out) {
    for (LVDim_t i = 0; i < dim; ++i) {
        vector_out[i] = rand_int8(INT8_MIN, INT8_MAX);
    }
}

void fill_fp32_vector(const LVDim_t dim, float* vector_out) {
    for (LVDim_t i = 0; i < dim; ++i) {
        vector_out[i] = rand_fp32(-1.0f, 1.0f);
    }
}

void fill_random_chr_data(const LVSize_t len, char* data_out) {
    for (LVSize_t k = 0; k < len; ++k) {
        data_out[k] = TEST_CHAR_POOL[rand() % TEST_CHAR_POOL_SIZE];
    }
}

void fill_random_date(LVDate* date, const uint16_t min_year, const uint16_t max_year) {
    uint8_t days_in_month[13] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    const uint16_t MIN = min_year < max_year ? min_year : max_year;
    const uint16_t MAX = max_year > min_year ? max_year : min_year;
    const uint16_t year = rand_int(MIN, MAX);
    const uint8_t month = rand_int8(1, 12);
    days_in_month[2] += ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0);
    const uint8_t day = rand_int8(1, days_in_month[month]);
    const uint8_t hour = rand_int8(0, 23);
    const uint8_t min = rand_int8(0, 59);
    const uint8_t sec = rand_int8(0, 59);

    date->year = year;
    date->month = month;
    date->day = day;
    date->hour = hour;
    date->min = min;
    date->sec = sec;
}

int compare_date(const LVDate* date_a, const LVDate* date_b) {
    return convert_date_to_epoch(date_a) - convert_date_to_epoch(date_b);
}

void print_status(const LVStatus status) {
    switch (status) {
        case LV_OK:
            printf("LV_OK \n");
            break;

        case LV_ERR_INVALID_PARAM:
            fprintf(stderr, "LV_ERR_INVALID_PARAM \n");
            break;

        case LV_ERR_IO:
            fprintf(stderr, "LV_ERR_IO \n");
            break;

        case LV_ERR_DISK_FULL:
            fprintf(stderr, "LV_ERR_DISK_FULL \n");
            break;

        case LV_ERR_DNEXSIST:
            fprintf(stderr, "LV_ERR_DNEXSIST \n");
            break;

        case LV_ERR_EXSIST:
            fprintf(stderr, "LV_ERR_EXSIST \n");
            break;

        case LV_ERR_NOT_FOUND:
            fprintf(stderr, "LV_ERR_NOT_FOUND \n");
            break;

        case LV_ERR_INIT:
            fprintf(stderr, "LV_ERR_INIT \n");
            break;

        default:
            fprintf(stderr, "Invalid status. \n");
            break;
    }
}

void make_first_date(const uint16_t year, LVDate* date) {
    assert(date);
    date->year = year;
    date->month = 1;
    date->day = 1;
    date->hour = 0;
    date->min = 0;
    date->sec = 0;
}
void make_last_date(const uint16_t year, LVDate* date) {
    assert(date);
    date->year = year;
    date->month = 12;
    date->day = 31;
    date->hour = 23;
    date->min = 59;
    date->sec = 59;
}
