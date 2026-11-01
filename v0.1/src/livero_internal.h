#ifndef LIVERO_INTERNAL
#define LIVERO_INTERNAL

#include <assert.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>

#include "livero_types.h"
#include "struct/commit.h"
#include "struct/row.h"
#include "vector/ternary.h"

struct Livero {
    LVSeq_t next_seq;
    LVSeq_t next_commit_seq;
    LVTernaryChunk* ternary_scratch;
    LVTernaryChunk* read_vector_batch;
    LVTernaryChunk* write_vector_batch;
    uint8_t* read_row_batch;
    uint8_t* write_row_batch;
    uint8_t* write_value_batch;
    int row_fd;
    int vec_fd;
    int val_fd;
    int commit_fd;
    LVSize32_t row_bytes;
    LVSize32_t vec_bytes;
    LVSize32_t val_bytes;
    LVSize32_t max_disk_bytes;
    LVCount_t batched_count;
    LVCount_t read_batch_m;
    LVCount_t write_batch_m;
    LVSize16_t batched_value_bytes;
    LVSize16_t max_value_len;
    LVDim_t vector_dimension;
    LVVectorType vector_type;
    LVSyncMode sync_mode;
};

static_assert(sizeof(Livero) <= 128, "Livero struct size must be under 128 bytes.");

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
LVStatus lv_flush(Livero* db, const LVFlushCtx* ctx);

LVStatus lv_interpolation_search_from_disk(Livero* db, const LVSeq_t target, LVGetResult* result);

LVStatus lv_binary_search_from_disk(Livero* db, const LVSeq_t target, LVGetResult* result);

typedef enum { LV_BATCH_CENTER, LV_BATCH_LEFT, LV_BATCH_RIGHT } LVBatchAlign;

typedef struct {
    int start_idx;
    int end_idx;
    LVBatchAlign align;
} LVBatchReadRange;

int lv_batch(Livero* db, const LVRow* row, const void* ternary_vector, const void* value, const LVSize16_t value_len);

/*
    Returns a good batch read range.
    1. [p-m/2, p+m/2] if start >= low and end <= high
    2. [p, min(p+m-1, high)] if p - m/2 < low
    3. [max(p-(m-1), low), p] if p + m/2 > high
*/
LVBatchReadRange lv_read_batch_range(const int p, const int low, const int high, const int m);

const uint8_t* lv_read_batch_expected_row(const Livero* db, const LVBatchAlign align, const LVCount_t batched_count);

LVStatus lv_read_batch_search(const Livero* db, const LVSeq_t target, const LVBatchAlign align, const LVCount_t count,
                              LVGetResult* result);

LVStatus lv_write_batch_search(const Livero* db, const LVSeq_t target, LVGetResult* result);

LVStatus lv_fill_get_result(const uint8_t* row, const void* value, const int val_fd, LVGetResult* result);

#endif
