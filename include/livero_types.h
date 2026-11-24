#ifndef LIVERO_TYPES
#define LIVERO_TYPES

#include <stdint.h>

typedef struct Livero Livero;

typedef enum LVStatus {
    LV_OK = -1,
    LV_ERR_INVALID_PARAM = 0,
    LV_ERR_IO = 1,
    LV_ERR_EXIST = 2,
    LV_ERR_NOT_EXIST = 3,
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

/*
    To make livero flexable at various IO environments,
    Livero doesn't use any pre-defined file handler and IO functions (For example: FD, read, write ... for Posix compliants).

    Instead, user must define its own file handler and fucntions for its environment.

    Livero only recieves the size of LVFILEDef and IO function pointers (open, close, write, read, sync).

    It doesn't restrict livero to run on a specific environment, and it enables MCU envrionments to handle disk data without a middle layer, which makes it faster.
*/
typedef struct LVFILEDef LVFILEDef;

/*
* lv_file_open: Open a signle file.

* Parameters:
* --path: A directory to open a file.
* --is_create: Opening a file may differ depending on the situations.
               if is_create is true, it means that the directory is empty, otherwise, the directory has already had files.
* --file: A file handler to open. 
        Livero will allocate LVFILEDefs on memory and provide them. 
        User doesn't need to allocate its own memory for file handler.

* Returns:
* --LV_OK: On success.
* --LV_ERR_EXIST: The given directory was expected to be empty, but already has a database.
* --LV_ERR_NOT_EXIST: The given was expected to be non-empty, but doesn't have a database.
* --LV_ERR_IO: Other IO erros happen during opening a file.

* It must explicitly handle LV_ERR_EXIST and LV_ERR_NOT_EXIST.
* Do not returns LV_ERR_IO for them.
*/
typedef LVStatus (*lv_file_open)(const char* path, const int is_create, LVFILEDef* file);

/*
* lv_file_close: Close a single file.

* Parameter:
* --file: Open file handler.
          Do not free file pointer, just close file IO buffers.

* Returns:
* --LV_OK: On success.
* --LV_ERR_IO: IO erros happen during closing a file.
*/
typedef LVStatus (*lv_file_close)(LVFILEDef* file);

/*
* lv_file_write: Write N bytes on a file.

* Parameters:
* --file: Open file handler.
* --data: A buffer which contains bytes to write.
* --len: How many bytes the buffer is.
* --offset: Data's absolute offset in the file.

* Returns:
* --LV_OK: On success.
* --LV_ERR_IO: IO erros happen during write.
*/
typedef LVStatus (*lv_file_write)(LVFILEDef* file, const void* data, const LVSize32_t len, const LVOffset_t offset);

/*
* lv_file_read: Read N bytes from a file.

* Parameters:
* --file: Open file handler.
* --buffer: A buffer to fill with data. 
            This buffer is pre-allocated, user doesn't need to allocate new memory.
* --len: Total bytes to read.
* --offset: Data's absolute offset in the file.

* Returns:
* --LV_OK: On success.
* --LV_ERR_IO: IO erros happen during read.
*/
typedef LVStatus (*lv_file_read)(LVFILEDef* file, void* buffer, const LVSize32_t len, const LVOffset_t offset);

/*
* lv_file_sync: Sync file data to the physical flash memory.

* Parameter:
* --file: Open file handler.

* Returns:
* --LV_OK: On success.
* --LV_ERR_IO: IO erros happen during sync.
*/
typedef LVStatus (*lv_file_sync)(LVFILEDef* file);

typedef struct {
    lv_file_open open;
    lv_file_close close;
    lv_file_write write;
    lv_file_read read;
    lv_file_sync sync;
} LVIOOps;

/*
    A config is only used at lv_create, once it saved, it never changes.
*/
typedef struct LVConfig {
    LVSize32_t max_disk_bytes;
    LVSize16_t max_value_len; // It affects size of value pending buffer (max_value_len * write_batch_m).
    LVDim_t vector_dimension; // It must be a multiple of 64.
    LVVectorType vector_type;
} LVConfig;

/*
    An option is used at lv_create and lv_open, you can use diffrent options every open.

    Also members (read_batch_m, write_batch_m, file_def_size, max_top_k, rerank_factor) 
    in LVOption are strongly related to memory consumtion.
    You should think carefully when you decide each member especially your environment is not rich in memory.
*/
typedef struct LVOption {
    LVCount_t read_batch_m; //It must be greater than 0 and it also decides window buffer size.
    LVCount_t write_batch_m; //It must be greater than 0 and it also decides pending buffer size.
    /* 
    If sync_mode is LV_SYNC_EVERY_PUT, livero only use pending buffer as a temporal scratch buffer 
    You should set write_batch_m to 1, if your sync_mode is LV_SYNC_EVERYY_PUT.
    */
    LVSyncMode sync_mode; 
    LVSize16_t file_def_size;
    const LVIOOps* io_ops;
    LVTopK_t max_top_k;
    uint8_t rerank_factor;
} LVOption;

typedef struct {
    uint16_t year;
    uint8_t month;
    uint8_t day;
    uint8_t hour;
    uint8_t min;
    uint8_t sec;
} LVDate;


/*
    User must pass its pointer to use lv_get.

    Also value buffer is not allocated by livero.
    User must reserve a buffer to get a value before call lv_get.
    If value buffer is NULL, lv_get will only provide date and value_len.
*/
typedef struct {
    LVDate date;
    void* value; //should be NULL or pre-allocated buffer
    LVSize16_t value_len;
} LVGetResult;

/*    
    If it's NULL lv_query will use default option which has no date filters and uses pre-filtering.

    Date Filtering:
    -- min(date1, date2) <= date <= max(date1, date2)
    -- Set both to NULL not to use date filter.
    -- Set only one to cover one side. (-inf <= date <= filter OR filter <= date <= +inf)

    Filtering Order:
    -- If use_postfilter is true, 
       livero will gather candidates first, 
       then filtering them after (out of date filter range OR deleted record).
       This may returns fewer results than K, but it is faster than pre-filtering.
    -- Default filtering order is pre-filtering (use_postfilter is false) which only 
       gathers candidates that are alive and fit in date filter.
    
*/
typedef struct {
    const LVDate* date1;
    const LVDate* date2;
    int use_postfilter;
} LVQueryOption;

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

#endif
