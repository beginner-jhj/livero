#ifndef LIVERO_TYPES
#define LIVERO_TYPES

#include <stdint.h>
#include <assert.h>
#include "internal/internal.h"

typedef enum LVStatus {
    LV_OK = -1,
    LV_ERR_INVALID_PARAM = 0,
    LV_ERR_IO = 1,
    LV_ERR_EXSIST = 2,
    LV_ERR_DNEXSIST = 3,
    LV_ERR_DISK_FULL = 4,
    LV_ERR_NOT_FOUND = 5,
    LV_ERR_INIT = 6,
} LVStatus;

typedef uint64_t LVSeq_t;
typedef uint32_t LVSize_t;
typedef uint32_t LVOffset_t;
typedef uint16_t LVDim_t;
typedef uint8_t LVTopK_t;

typedef enum LVVectorType : uint8_t {
    LV_VEC_TYPE_INT8 = 0,
    LV_VEC_TYPE_FP32 = 1
} LVVectorType;

typedef struct LVConfig {
    LVSize_t max_disk_bytes;
    LVVectorType vector_type;
    LVDim_t vector_dimension;
} LVConfig;

typedef struct LVDBInfo{
    LVSize_t max_disk_bytes;
    LVVectorType vector_type;
    LVDim_t vector_dimension;
    
    LVSize_t record_count;
    LVSize_t db_file_size;
    LVSize_t val_log_size;
    LVSize_t vec_log_size;
    uint64_t total_disk_size;    
    
} LVDBInfo;

typedef struct LVDate{
    uint16_t year;
    uint8_t month;
    uint8_t day;
    uint8_t hour;
    uint8_t min;
    uint8_t sec;
} LVDate;

typedef struct LVQueryResult{
    LVSeq_t key;
    uint16_t value_len;
    uint16_t year;
    uint8_t month;
    uint8_t day;
    uint8_t hour;
    uint8_t min;
    uint8_t sec;
    uint8_t value[LV_MAX_VALUE_LEN];
} LVQueryResult;

static_assert(sizeof(LVQueryResult) == 504, "LVQueryResult size is incorrect.");

#endif
