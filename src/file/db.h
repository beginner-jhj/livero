#ifndef LV_DB_FILE
#define LV_DB_FILE

#include <stdint.h>

#include "livero_types.h"
#include "vector/ternary.h"

#define LV_DB_FILE_HEADER_SIZE 27
#define LV_DB_FILE_RECORD_SIZE 23

typedef struct LVDBHeader {
    LVSeq_t next_seq;
    LVSize_t record_count;
    LVSize_t max_disk_bytes;
    LVSize_t current_disk_bytes;
    LVOffset_t next_val_log_offset;
    LVDim_t vector_dim;
    LVVectorType vector_type;
} LVDBHeader;

typedef struct LVDBWriteCtx {
    const int db_file_fd;
    const int val_log_fd;
    const int vec_log_fd;

    const LVTernaryChunk* ternary_chunks;
    const uint64_t epoch_date;
    const void* value;
    const uint16_t value_len;
} LVDBWriteCtx;

LVStatus db_file_init(const int fd, const LVConfig* config);
LVStatus db_file_read_header(const int fd, LVDBHeader* header_out);
LVStatus db_file_write(const LVDBWriteCtx* ctx, LVSeq_t* used_key_out);
LVStatus db_file_search(const int fd, const LVSize_t record_count, const LVSeq_t target_key, uint64_t* time_out,
                        LVOffset_t* value_offset_out, LVSize_t* value_len_out, LVOffset_t* offset_out);
LVStatus db_file_get_value(const int val_log_fd, const LVOffset_t offset, const LVSize_t len, void* out_buf);
LVStatus db_file_date_filter(const int fd, const uint32_t internal_seq, const uint64_t date1, const uint64_t date2,
                             bool* result_out);
LVStatus db_file_get_vector(const int vec_log_fd, const uint32_t internal_seq, const LVDim_t vector_dim,
                            void* vector_out);
LVStatus db_file_query_result(const int db_file_fd, const int val_log_fd, const uint32_t internal_seq,
                              LVQueryResult* query_result_out);
LVStatus db_file_delete(const int fd, const LVSize_t record_count, const LVSeq_t key);

#endif
