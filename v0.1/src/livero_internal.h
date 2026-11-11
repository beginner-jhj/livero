#ifndef LIVERO_INTERNAL
#define LIVERO_INTERNAL

#include <assert.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "livero_types.h"
#include "struct/commit.h"
#include "struct/config.h"
#include "struct/row.h"
#include "vector/ternary.h"

struct Livero {
    LVSeq_t next_seq;
    LVSeq_t next_commit_seq;
    LVTernaryChunk* ternary_scratch;
    LVTernaryChunk* ternary_window;
    LVTernaryChunk* ternary_pending;
    uint8_t* row_window;
    uint8_t* row_pending;
    uint8_t* value_pending;
    int row_fd;
    int ternary_fd;
    int val_fd;
    int commit_fd;
    LVSize32_t row_bytes;
    LVSize32_t ternary_bytes;
    LVSize32_t val_bytes;
    LVSize32_t max_disk_bytes;
    LVCount_t pending_count;
    LVCount_t window_batch_m;
    LVCount_t pending_batch_m;
    LVSize16_t value_pending_bytes;
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

#define LV_MEMORY_ALIGNMENT 64
#define LV_MAX_BATCH_M 512

#define LV_ROW_FILE_NAME "row.lv"
#define LV_TERNARY_FILE_NAME "ter.lv"
#define LV_VAL_FILE_NAME "val.lv"
#define LV_COMMIT_FILE_NAME "commit.lv"
#define LV_CONFIG_FILE_NAME "config.lv"

#define LV_SEQ_START 0

#define LV_MAX_PATH_LEN 512
#define LV_MAX_VLA_LEN 512
#define LV_MAX_K 32
#define LV_TOMBSTONE_FLAG 0xff

#define LV_BASELINE_DISK_USAGE LV_CONFIG_DISK_BYTES + LV_COMMIT_SECTOR_BYTES* LV_COMMIT_SLOT_M

static_assert(LV_MAX_K * LV_TERNARY_CHUNK_BYTES <= LV_MAX_VLA_LEN, "top_k vla results can exceed LV_MAX_VLA_LEN.");

/*
    Count of flushed rows (on disk)
*/
static inline LVCount_t lv_disk_record_count(const Livero* db) {
    assert(db);
    return db->row_bytes / LV_ROW_BYTES;
}

/*
    Count of flushed rows (on diks) and pending rows (on memory)
*/
static inline LVCount_t lv_total_record_count(const Livero* db) {
    assert(db);
    return db->row_bytes / LV_ROW_BYTES + db->pending_count;
}

static inline const uint8_t* lv_first_pending_row(const Livero* db) {
    assert(db);
    return db->pending_count == 0 ? NULL : db->row_pending;
}

static inline const uint8_t* lv_last_pending_row(const Livero* db) {
    assert(db);
    return db->pending_count == 0 ? NULL : db->row_pending + (db->pending_count - 1) * LV_ROW_BYTES;
}

static inline const uint8_t* lv_n_pending_row(const Livero* db, const LVOffset_t idx) {
    assert(db);
    return db->pending_count == 0 ? NULL : db->row_pending + (idx)*LV_ROW_BYTES;
}

static inline const uint8_t* lv_n_window_row(const Livero* db, const LVOffset_t idx) {
    assert(db);
    return db->window_batch_m == 0 ? NULL : db->row_window + idx * LV_ROW_BYTES;
}

static inline const LVTernaryChunk* lv_n_pending_ternary(const Livero* db, const LVOffset_t idx) {
    assert(db);
    const LVCount_t ternary_chunk_count = ternary_chunks_count(db->vector_dimension);
    return db->pending_count == 0 ? NULL : &db->ternary_pending[idx * ternary_chunk_count];
}

static inline const LVTernaryChunk* lv_n_window_ternary(const Livero* db, const LVOffset_t idx) {
    assert(db);
    const LVCount_t ternary_chunk_count = ternary_chunks_count(db->vector_dimension);
    return db->window_batch_m == 0 ? NULL : &db->ternary_window[idx * ternary_chunk_count];
}

static inline const uint8_t* lv_n_pending_value(const Livero* db, const LVOffset_t idx) {
    assert(db);
    return db->value_pending_bytes == 0 ? NULL : db->value_pending + idx * db->max_value_len;
}

static inline LVSize32_t lv_pending_row_bytse(const Livero* db) {
    assert(db);
    return db->value_pending_bytes * LV_ROW_BYTES;
}

static inline LVSize32_t lv_pending_ternary_bytes(const Livero* db) {
    assert(db);
    return db->value_pending_bytes * ternary_chunks_bytes(db->vector_dimension);
}

static inline LVSize32_t lv_pending_val_bytes(const Livero* db) {
    assert(db);
    return db->value_pending_bytes;
}

LVStatus lv_allocate_memory(const LVSize32_t size, void** ptr_out);

typedef struct LVInitCtx {
    LVSeq_t next_seq;
    LVSeq_t next_commit_seq;
    int row_fd;
    int ternary_fd;
    int val_fd;
    int commit_fd;
    LVSize32_t row_bytes;
    LVSize32_t ternary_bytes;
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
    int ternary_fd;
    int val_fd;
    int commit_fd;
    LVSize32_t max_disk_bytes;
    LVDim_t vector_dimension;
} LVRecoverCtx;
/*
    Full scan commit.lv and find the last valid commit
*/
LVStatus lv_recover(const LVRecoverCtx* ctx, LVCommit* commit_out, int* first_commit_torn_out);

LVStatus lv_flush(Livero* db);

typedef struct {
    uint8_t* row_window;
    int row_fd;
    int val_fd;
    LVCount_t disk_record_count;
    LVCount_t window_batch_m;
} LVDiskSearchCtx;

static inline LVDiskSearchCtx lv_disk_search_ctx(const Livero* db) {
    assert(db);
    return (LVDiskSearchCtx){
        .disk_record_count = lv_disk_record_count(db),
        .row_fd = db->row_fd,
        .val_fd = db->val_fd,
        .row_window = db->row_window,
        .window_batch_m = db->window_batch_m,
    };
}

LVStatus lv_interpolation_search_from_disk(LVDiskSearchCtx* ctx, const LVSeq_t target, LVGetResult* result,
                                           LVOffset_t* row_offset_out);

LVStatus lv_binary_search_from_disk(LVDiskSearchCtx* ctx, const LVSeq_t target, LVGetResult* result,
                                    LVOffset_t* row_offset_out);

/* Append one record (row, ternary, value) to the pending (uncommitted)
  write buffers. */
int lv_pending_append(Livero* db, const LVRow* row, const LVTernaryChunk* ternary, const void* value,
                      const LVSize16_t value_len);

/* Find target among pending records, still in RAM, not yet on disk.
 */
LVStatus lv_pending_search(const Livero* db, const LVSeq_t target, LVGetResult* result, LVOffset_t* row_offset_out);

typedef struct {
    int start_idx;
    int end_idx;
} LVWindowRange;

LVWindowRange lv_window_range(const int p, const int low, const int high, const int m);

static inline LVOffset_t lv_window_target_offset(const int p, const int start_idx, const LVSize32_t item_size) {
    return (p - start_idx) * item_size;
}

/* Find target in the window; the value is read from disk through the
 * row's value_offset, since values have no window of their own.
 */
LVStatus lv_window_search(const uint8_t* row_window, const int val_fd, const LVSeq_t target_seq, const LVCount_t count,
                          const LVOffset_t target_offset, LVGetResult* result, LVOffset_t* row_offset_out);

LVStatus lv_fill_get_result(const uint8_t* row, const void* value_src, const int val_fd, LVGetResult* result);

typedef struct {
    int64_t dot_result;
    LVQueryResult q_result;
} LVQueryIndex;

typedef struct {
    uint8_t* row;
    uint64_t date_flt1;
    uint64_t date_flt2;
    int64_t dot_result;
    LVTopK_t top_k;
} LVQueryHeapCtx;

void lv_query_heap_insert(LVQueryIndex* heap, const LVQueryIndex index, LVSize32_t* heap_size);
void lv_query_heap_pop(LVQueryIndex* heap, LVSize32_t* heap_size, LVQueryIndex* pop_out);

int lv_query_heap_admit(const LVQueryIndex* heap, const LVSize32_t size, const LVTopK_t threshold,
                        const int64_t dot_result);

void lv_query_heap_update(LVQueryIndex* heap, LVSize32_t* size, const LVQueryHeapCtx* ctx);

#endif
