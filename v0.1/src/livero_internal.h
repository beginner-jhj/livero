#ifndef LIVERO_INTERNAL
#define LIVERO_INTERNAL

#include <assert.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>

#include "livero_types.h"
#include "struct/commit.h"
#include "struct/row.h"

static void lv_log_error(const char* file, int line, const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[ERROR] At %s %d (%s): ", file, line, func);
    vfprintf(stderr, fmt, args);
    fputc('\n', stderr);
    va_end(args);
}

#ifndef NDEBUG
#define LIVERO_ERROR(...) lv_log_error(__FILE__, __LINE__, __func__, __VA_ARGS__)
#else
#define LIVERO_ERROR(...) ((void)0)
#endif

#define LV_DEFAULT_SECTOR_BYTES 1024 * 4
#define LV_MEMORY_ALIGNMENT 64
#define LV_DIMENSION_MULTIPLE 64
#define LV_MAX_BATCH_M 64

#define LV_ROW_FILE_NAME "row.lv"
#define LV_VEC_FILE_NAME "vec.lv"
#define LV_VAL_FILE_NAME "val.lv"
#define LV_COMMIT_FILE_NAME "commit.lv"
#define LV_CONFIG_FILE_NAME "config.lv"

#define LV_MAX_PATH_LEN 512
#define LV_MAX_VLA_LEN 256

int lv_check_option(const LVOption* option);
LVStatus lv_allocate_memory(const LVSize32_t size, void** ptr_out);

typedef struct LVInitCtx {
    LVSeq_t next_seq;
    LVSeq_t next_commit_seq;
    int row_fd;
    int vec_fd;
    int val_fd;
    int commit_fd;
    LVSize32_t row_bytes;
    LVSize32_t vec_bytes;
    LVSize32_t val_bytes;

    LVSize32_t max_disk_bytes;
    LVSize16_t max_value_len;
    LVDim_t vector_dimension;
    LVVectorType vector_type;

    LVSize32_t read_batch_m;
    LVSize32_t write_batch_m;
    LVSyncMode sync_mode;
} LVInitCtx;
void lv_init_memory(uint8_t* m_arena, const LVInitCtx* ctx);

typedef struct LVRecoverCtx {
    int row_fd;
    int vec_fd;
    int val_fd;
    int commit_fd;
    LVSize32_t max_disk_bytes;
    LVDim_t vector_dim;
} LVRecoverCtx;
/*
    Full scan commit.lv and find the last valid commit
*/
LVStatus lv_recover(const LVRecoverCtx* ctx, LVCommit* commit_out);

typedef struct {
    const uint8_t* rows;
    const uint8_t* vecs;
    const uint8_t* vals;
    const LVCommit* commit;
    int row_fd;
    int vec_fd;
    int val_fd;
    int commit_fd;
    LVSize32_t row_bytes_to_flush;
    LVSize32_t vector_bytes_to_flush;
    LVSize32_t value_bytes_to_flush;
    LVOffset_t row_offset;
    LVOffset_t vec_offset;
    LVOffset_t val_offset;
    uint8_t sync;
} LVFlushCtx;
LVStatus lv_flush(Livero* db,const LVFlushCtx* ctx);

typedef struct {
    uint8_t* row_batch_ptr;
    uint8_t* vector_batch_ptr;
    uint8_t* value_batch_ptr;
    const LVRow* row_data;
    const void* vector;
    const void* value;
    const LVSize32_t vector_bytes;
    const LVSize32_t value_len;
    const LVOffset_t row_batch_offset;
    const LVOffset_t vector_batch_offset;
    const LVOffset_t value_batch_offset;
} LVBatchCtx;
void lv_batch(Livero* db,const LVBatchCtx* ctx);

#endif
