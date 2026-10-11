from _path import *
CDEF = """
typedef struct Livero Livero;
#define LV_MAX_VALUE_LEN 487

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

typedef uint8_t LVVectorType;
enum{
    LV_VEC_TYPE_INT8 = 0,
    LV_VEC_TYPE_FP32 = 1
};

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

LVStatus lv_create(const char* db_path, const LVConfig* config, Livero** db_out);
LVStatus lv_open(const char* db_path, Livero** db_out);
LVStatus lv_put(Livero* db, const void* vector, const LVDate* date, const void* value, const uint16_t value_len,
                LVSeq_t* key_out);
LVStatus lv_get(const Livero* db, const LVSeq_t key, LVDate* date_out, void* value_out, uint16_t* value_len_out);
LVStatus lv_query(Livero* db, const LVTopK_t k, const void* qeury_vector, const LVDate* date1, const LVDate* date2,
                  LVQueryResult** result_out, LVSize_t* result_size_out);
LVStatus lv_delete(Livero* db, const LVSeq_t key);
void lv_db_info(const Livero* db, LVDBInfo* info_out);
void lv_close(Livero* db);


"""

LV_SOURCES = [
    os.path.join(SRC_DIR, name)
    for name in [
        "file/db.c",
        "internal/helper.c",
        "livero.c",
        "vector/ternary.c"
    ]
]

if __name__ == "__main__":
    build_dir = os.path.join(ROOT_DIR, "build", "cffi")
    os.makedirs(build_dir, exist_ok=True)
    from cffi import FFI

    debug_builder = FFI()
    debug_builder.cdef(CDEF)

    debug_builder.set_source(
        "_livero_debug",
        '#include "livero.h"',
        sources=LV_SOURCES,
        include_dirs=[INCLUDE_DIR, SRC_DIR],
        extra_compile_args=["-O0", "-DDEBUG", "-std=c23"]
    )

    debug_builder.compile(verbose=True, tmpdir=build_dir)
