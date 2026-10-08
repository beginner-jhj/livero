#ifndef LV_INTERNAL
#define LV_INTERNAL

#define LV_MAX_PATH_LEN 512
#define DB_FILE_NAME "db.lv"
#define VAL_LOG_FILE_NAME "val_log.lv"
#define VEC_LOG_FILE_NAME "vec_log.lv"
#define LV_DIM_MULTIPLE 64

#ifndef NDEBUG
#include <stdio.h>
#define LIVERO_ERROR(fmt, ...) \
    do {\
        fprintf(stderr, "[ERROR] At %s %d (%s): " fmt "\n", __FILE__, __LINE__, __func__ __VA_OPT__(,) __VA_ARGS__); \
    } while(0)
#else
#define LIVERO_ERROR(fmt, ...) ((void)0)
#endif

#define LV_MAX_K 64
#define LV_MAX_VALUE_LEN 487
#define LV_MAX_VECTOR_DIMENSION 768
#define LV_RAM_USAGE (1u << 10 ) * 32

#endif
