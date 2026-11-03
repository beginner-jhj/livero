#ifndef LIVERO_TYPES
#define LIVERO_TYPES

#include <stdint.h>

typedef struct Livero Livero;

typedef enum LVStatus {
    LV_OK = -1,
    LV_ERR_INVALID_PARAM = 0,
    LV_ERR_IO = 1,
    LV_ERR_EXSIST = 2,
    LV_ERR_DNEXSIST = 3,
    LV_ERR_DISK_FULL = 4,
    LV_ERR_NOT_FOUND = 5,
    LV_ERR_ALLOCATE_MEMORY = 6,
    LV_ERR_DATA_CORRUPT = 7,
    LV_ERR_MEDIA_CRASHED = 8
} LVStatus;

typedef uint64_t LVSeq_t;
typedef uint32_t LVSize32_t;
typedef uint16_t LVSize16_t;
typedef uint32_t LVOffset_t;
typedef uint32_t LVCount_t;
typedef uint16_t LVDim_t;
typedef uint8_t LVTopK_t;

typedef uint8_t LVVectorType;
enum { LV_VEC_TYPE_INT8 = 0, LV_VEC_TYPE_FP32 = 1 };

typedef uint8_t LVSyncMode;
enum {
    LV_SYNC_BATCH = 0,
    LV_SYNC_EVERY_PUT = 1,
    LV_SYNC_NONE = 2,
};

typedef struct LVConfig {
    LVSize32_t max_disk_bytes;
    LVSize16_t max_value_len;
    LVDim_t vector_dimension;
    LVVectorType vector_type;
} LVConfig;

typedef struct LVOption {
    LVCount_t read_batch_m;
    LVCount_t write_batch_m;
    LVSyncMode sync_mode;
} LVOption;

typedef struct {
    uint16_t year;
    uint8_t month;
    uint8_t day;
    uint8_t hour;
    uint8_t min;
    uint8_t sec;
} LVDate;

typedef struct {
    LVDate date;
    void* value;
    LVSize16_t value_len;
} LVGetResult;

typedef struct {
    LVDate date;
    LVSeq_t seq;
    LVSize16_t value_len;
} LVQueryResult;

#endif
