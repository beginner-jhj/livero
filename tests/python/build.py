from _path import ROOT_DIR, INCLUDE_DIR, SRC_DIR
import os

CDEF = """

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
    LV_ERR_MEDIA_CRASHED = 8,
    LV_ERR_DELETED = 9,
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

typedef struct {
    LVConfig config;
    LVOption option;
    LVSize32_t row_file_bytes;
    LVSize32_t ternary_file_bytes;
    LVSize32_t val_file_bytes;
    LVSize32_t row_pending_bytes;
    LVSize32_t ternary_pending_bytes;
    LVSize32_t val_pending_bytes;
} LVDBStat;

LVStatus lv_create(const char* db_path, const LVConfig* config, const LVOption* option, Livero** db_out);
LVStatus lv_open(const char* db_path, const LVOption* option, Livero** db_out);
LVStatus lv_put(Livero* db, void* vector,void* value,const LVSize16_t value_len, const LVDate* date, LVSeq_t* seq_out);
LVStatus lv_get(Livero* db, const LVSeq_t target_seq, LVGetResult* result);
LVStatus lv_delete(Livero* db, const LVSeq_t target_seq);
LVStatus lv_query(Livero* db,const LVTopK_t k, const void* query_vector, const LVDate* date1, const LVDate* date2, LVQueryResult* result, LVSize32_t* result_size);
void lv_close(Livero* db);
void lv_stat(const Livero* db, LVDBStat* stat);

"""

LV_SOURCES = [
    os.path.join(SRC_DIR, name)
    for name in [
        "file/file.c",
        "helper/helper.c",
        "helper/crc32c.c",
        "struct/commit.c",
        "struct/config.c",
        "struct/row.c",
        "vector/ternary.c",
        "livero.c"
    ]
]

if __name__ == "__main__":
    build_dir = os.path.join(ROOT_DIR, "build", "cffi")
    os.makedirs(build_dir, exist_ok=True)
    from cffi import FFI

    builder = FFI()
    builder.cdef(CDEF)

    builder.set_source(
        "_livero",
        '#include "livero.h"',
        sources=LV_SOURCES,
        include_dirs=[INCLUDE_DIR, SRC_DIR],
        extra_compile_args=["-O3", "-DNDEBUG", "-std=c11"]
    )

    builder.compile(verbose=True, tmpdir=build_dir)

