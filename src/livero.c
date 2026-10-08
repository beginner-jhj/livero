#include "livero.h"

#include <assert.h>
#include <errno.h>
#include <memory.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "file/db.h"
#include "internal/helper.h"
#include "internal/internal.h"
#include "livero_types.h"
#include "vector/ternary.h"

struct Livero {
    LVQueryResult* reserved_query_results;
    int db_file_fd;
    int val_log_fd;
    int vec_log_fd;
    LVSize_t record_count;
    LVSize_t db_file_size;
    LVSize_t val_log_size;
    LVSize_t vec_log_size;
    LVSize_t max_disk_bytes;
    LVDim_t vector_dim;
    LVVectorType vector_type;
    uint8_t __pad[21];
};

static_assert(sizeof(Livero) == 64, "Livero size is incorrect.");

typedef struct LVMemoryInitCtx {
    int db_fd;
    int val_log_fd;
    int vec_log_fd;

    LVSize_t record_count;
    LVSize_t db_file_size;
    LVSize_t val_log_size;
    LVSize_t vec_log_size;

    LVSize_t max_disk_bytes;
    LVDim_t vector_dim;
    LVVectorType vector_type;
} LVMemoryInitCtx;

typedef struct LVQueryIndex {
    int64_t dot_result;
    uint32_t internal_seq;
} LVQueryIndex;

static LVStatus lv_memory_init(const LVMemoryInitCtx* ctx, Livero** db_out);
static void lv_query_heap_insert(LVQueryIndex* heap, const LVQueryIndex index, LVSize_t* heap_size);
static void lv_query_heap_pop(LVQueryIndex* heap, LVSize_t* heap_size, LVQueryIndex* pop_out);

LVStatus lv_create(const char* db_path, const LVConfig* config, Livero** db_out) {
    if (!db_path) {
        LIVERO_ERROR("Path where db files will be stored is missing.");
        return LV_ERR_INVALID_PARAM;
    }

    if (config->vector_dimension > LV_MAX_VECTOR_DIMENSION) {
        LIVERO_ERROR("The vector dimension is too big. (MAX: %d)", LV_MAX_VECTOR_DIMENSION);
        return LV_ERR_INVALID_PARAM;
    }

    int db_fd = -1;
    int val_log_fd = -1;
    int vec_log_fd = -1;

    const int create_flag = O_CREAT | O_EXCL | O_RDWR | O_FSYNC;

    LVStatus status = open_file(db_path, DB_FILE_NAME, create_flag, &db_fd);

    if (status != LV_OK) {
        return status;
    }

    if ((status = open_file(db_path, VAL_LOG_FILE_NAME, create_flag, &val_log_fd)) != LV_OK) {
        goto cleanup;
    }

    if ((status = open_file(db_path, VEC_LOG_FILE_NAME, create_flag, &vec_log_fd)) != LV_OK) {
        goto cleanup;
    }

    if ((status = db_file_init(db_fd, config)) != LV_OK) {
        goto cleanup;
    }

    LVMemoryInitCtx mem_init = {.db_fd = db_fd,
                                .db_file_size = LV_DB_FILE_HEADER_SIZE,
                                .val_log_fd = val_log_fd,
                                .val_log_size = 0,
                                .vec_log_fd = vec_log_fd,
                                .vec_log_size = 0,
                                .record_count = 0,
                                .max_disk_bytes = config->max_disk_bytes,
                                .vector_dim = config->vector_dimension,
                                .vector_type = config->vector_type};

    if ((status = lv_memory_init(&mem_init, db_out)) != LV_OK) {
        goto cleanup;
    }

    return LV_OK;

cleanup:
    close_file(db_fd);
    close_file(val_log_fd);
    close_file(vec_log_fd);

    return status;
}

LVStatus lv_open(const char* db_path, Livero** db_out) {
    if (!db_path) {
        LIVERO_ERROR("Path where db files will be stored is missing.");
        return LV_ERR_INVALID_PARAM;
    }

    int db_fd = -1;
    int val_log_fd = -1;
    int vec_log_fd = -1;

    const int open_flag = O_RDWR | O_FSYNC;

    LVStatus status = open_file(db_path, DB_FILE_NAME, open_flag, &db_fd);

    if (status != LV_OK) {
        return status;
    }

    if ((status = open_file(db_path, VAL_LOG_FILE_NAME, open_flag, &val_log_fd)) != LV_OK) {
        goto cleanup;
    }

    if ((status = open_file(db_path, VEC_LOG_FILE_NAME, open_flag, &vec_log_fd)) != LV_OK) {
        goto cleanup;
    }

    LVDBHeader header;

    if ((status = db_file_read_header(db_fd, &header)) != LV_OK) {
        goto cleanup;
    }

    LVMemoryInitCtx mem_init = {.db_fd = db_fd,
                                .db_file_size = LV_DB_FILE_HEADER_SIZE + LV_DB_FILE_RECORD_SIZE * header.record_count,
                                .val_log_fd = val_log_fd,
                                .val_log_size = header.next_val_log_offset,
                                .vec_log_fd = vec_log_fd,
                                .vec_log_size = ternary_chunks_size(header.vector_dim) * header.record_count,
                                .record_count = header.record_count,
                                .max_disk_bytes = header.max_disk_bytes,
                                .vector_dim = header.vector_dim,
                                .vector_type = header.vector_type

    };

    if ((status = lv_memory_init(&mem_init, db_out)) != LV_OK) {
        goto cleanup;
    }

    return LV_OK;

cleanup:
    close_file(db_fd);
    close_file(val_log_fd);
    close_file(vec_log_fd);

    return status;
}

LVStatus lv_put(Livero* db, const void* vector, const LVDate* date, const void* value, const uint16_t value_len,
                LVSeq_t* key_out) {
    if (!db) {
        LIVERO_ERROR("You missed a db.");
        return LV_ERR_INVALID_PARAM;
    }

    if (!vector) {
        LIVERO_ERROR("You must pass a vector.");
        return LV_ERR_INVALID_PARAM;
    }

    if (!value || value_len == 0) {
        LIVERO_ERROR("You must pass a value.");
        return LV_ERR_INVALID_PARAM;
    }

    uint64_t epoch_time = 0;
    if (date) {
        if (date_is_valid(date)) {
            epoch_time = convert_date_to_epoch(date);
        } else {
            LIVERO_ERROR("You passed an invalid date object.");
            return LV_ERR_INVALID_PARAM;
        }
    } else {
        time_t current_time;
        time(&current_time);

        struct tm utc;
        gmtime_r(&current_time, &utc);

        LVDate now = {.year = utc.tm_year + 1900,
                      .month = utc.tm_mon + 1,
                      .day = utc.tm_mday,
                      .hour = utc.tm_hour,
                      .min = utc.tm_min,
                      .sec = utc.tm_sec};

        epoch_time = convert_date_to_epoch(&now);
    }

    /*
        db->vector_dim is always <= LV_MAX_VECTOR_DIMENSION = 768DIM. Ternary Chunks are under 16Words = 256B.
    */
    LVTernaryChunk converted_vector[ternary_chunks_count(db->vector_dim)];
    ternary_convert(vector, db->vector_type, db->vector_dim, converted_vector);

    LVDBWriteCtx write_ctx = {
        .db_file_fd = db->db_file_fd,
        .val_log_fd = db->val_log_fd,
        .vec_log_fd = db->vec_log_fd,
        .epoch_date = epoch_time,
        .ternary_chunks = converted_vector,
        .value = value,
        .value_len = value_len,
    };

    LVStatus status = db_file_write(&write_ctx, key_out);
    if (status != LV_OK) {
        LIVERO_ERROR("Failed to put a record on the disk. (Status: %d)", status);
        return status;
    }

    db->db_file_size += LV_DB_FILE_RECORD_SIZE;
    db->val_log_size += value_len;
    db->vec_log_size += ternary_chunks_size(db->vector_dim);
    db->record_count += 1;

    return LV_OK;
}

LVStatus lv_get(const Livero* db, const LVSeq_t key, LVDate* date_out, void* value_out) {
    if (!db) {
        LIVERO_ERROR("You missed a db.");
        return LV_ERR_INVALID_PARAM;
    }

    uint64_t epoch_time = 0;
    LVOffset_t value_offset = 0;
    LVSize_t value_len = 0;

    LVStatus status =
        db_file_search(db->db_file_fd, db->record_count, key, &epoch_time, &value_offset, &value_len, nullptr);

    if (status != LV_OK) {
        return status;
    }

    if (value_out) {
        if ((status = db_file_get_value(db->val_log_fd, value_offset, value_len, value_out)) != LV_OK) {
            return status;
        }
    }

    if (date_out) {
        convert_epoch_to_date(epoch_time, date_out);
    }

    return LV_OK;
}

LVStatus lv_query(Livero* db, const LVTopK_t k, const void* query_vector, const LVDate* date1, const LVDate* date2,
                  LVQueryResult** result_out, LVSize_t* result_size_out) {
    if (!db) {
        LIVERO_ERROR("You missed a db.");
        return LV_ERR_INVALID_PARAM;
    }

    if (!query_vector) {
        LIVERO_ERROR("You must pass a vector to query.");
        return LV_ERR_INVALID_PARAM;
    }

    if (k > LV_MAX_K || k == 0) {
        LIVERO_ERROR("The requested k invalid. (Max: %d)", LV_MAX_K);
        return LV_ERR_INVALID_PARAM;
    }

    const LVTopK_t TOP_K = k <= db->record_count ? k : db->record_count;

    /*
        It will filter target dates like this: date1 <= target <= date2.
        But date1 and date2 are both optional.
        To remove if branch, make the default range cover all.
    */
    uint64_t date1_epoch = 0;
    uint64_t date2_epoch = UINT64_MAX;

    if (date1) {
        date1_epoch = convert_date_to_epoch(date1);
    }

    if (date2) {
        date2_epoch = convert_date_to_epoch(date2);
    }

    const uint64_t min_date = date1_epoch < date2_epoch ? date1_epoch : date2_epoch;
    const uint64_t max_date = date2_epoch > date1_epoch ? date2_epoch : date1_epoch;

    LVSize_t words = ternary_chunks_count(db->vector_dim);
    LVSize_t vector_bytes = ternary_chunks_size(db->vector_dim);
    LVTernaryChunk vector[words];
    LVTernaryChunk converted_query_vector[words];
    memset(converted_query_vector, 0, vector_bytes);
    ternary_convert(query_vector, db->vector_type, db->vector_dim, converted_query_vector);

    LVQueryIndex query_index_heap[TOP_K];  // min heap.
    LVSize_t query_index_heap_size = 0;
    for (uint32_t internal_seq = 0; internal_seq < db->record_count; ++internal_seq) {
        bool needs_calc = false;
        LVStatus status = db_file_date_filter(db->db_file_fd, internal_seq, min_date, max_date, &needs_calc);
        if (status != LV_OK) {
            return status;
        } else {
            if (needs_calc) {
                memset(vector, 0, vector_bytes);
                if ((status = db_file_get_vector(db->vec_log_fd, internal_seq, db->vector_dim, vector)) != LV_OK) {
                    return status;
                }
                int64_t dot_result = ternary_dot(converted_query_vector, vector, words);
                LVQueryIndex index = {.dot_result = dot_result, .internal_seq = internal_seq};
                if (query_index_heap_size < TOP_K) {
                    lv_query_heap_insert(query_index_heap, index, &query_index_heap_size);
                } else {
                    if (index.dot_result > query_index_heap[0].dot_result) {
                        lv_query_heap_pop(query_index_heap, &query_index_heap_size, nullptr);
                        lv_query_heap_insert(query_index_heap, index, &query_index_heap_size);
                    }
                }
            }
        }
    }

    LVQueryIndex query_index_result[query_index_heap_size];
    const LVSize_t copied_query_index_heap_size = query_index_heap_size;
    for (int i = copied_query_index_heap_size - 1; i >= 0; --i) {
        lv_query_heap_pop(query_index_heap, &query_index_heap_size, &query_index_result[i]);
    }

    for (LVSize_t i = 0; i < copied_query_index_heap_size; ++i) {
        LVStatus status = db_file_query_result(db->db_file_fd, db->val_log_fd, query_index_result[i].internal_seq,
                                               &db->reserved_query_results[i]);
        if (status != LV_OK) {
            return status;
        }
    }
    if (result_out) {
        *result_out = db->reserved_query_results;
    }

    if (result_size_out) {
        *result_size_out = copied_query_index_heap_size;
    }

    return LV_OK;
}

LVStatus lv_delete(Livero* db, const LVSeq_t key) {
    if (!db) {
        LIVERO_ERROR("You missed a db.");
        return LV_ERR_INVALID_PARAM;
    }

    return db_file_delete(db->db_file_fd, db->record_count, key);
}

void lv_db_info(const Livero* db, LVDBInfo* info_out) {
    if (db && info_out) {
        info_out->max_disk_bytes = db->max_disk_bytes;
        info_out->vector_type = db->vector_type;
        info_out->vector_dimension = db->vector_dim;

        info_out->total_disk_size = db->db_file_size + db->val_log_size + db->vec_log_size;
        info_out->db_file_size = db->db_file_size;
        info_out->val_log_size = db->val_log_size;
        info_out->vec_log_size = db->vec_log_size;

        info_out->record_count = db->record_count;
    }
}

void lv_close(Livero* db) {
    if (db) {
        close_file(db->db_file_fd);
        close_file(db->val_log_fd);
        close_file(db->vec_log_fd);

        if (db != MAP_FAILED) {
            munmap(db, LV_RAM_USAGE);
        }
    }
}

static LVStatus lv_memory_init(const LVMemoryInitCtx* ctx, Livero** db_out) {
    /*
        Livero use fixed ram size 32KiB which is 32768B.
        [Livero (64B)]
        [Reserved Query Results (504B * 64 = 32256B)]

        Total exaxt ram size is 64B + 32256B = 32320B < 32KiB.
    */

    void* memory_arena = nullptr;

    memory_arena = mmap(NULL, LV_RAM_USAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if (memory_arena == MAP_FAILED) {
        LIVERO_ERROR("Failed to allocate a memory arena.");
        return LV_ERR_INIT;
    }

    *db_out = (Livero*)memory_arena;

    (*db_out)->reserved_query_results = (LVQueryResult*)((uint8_t*)memory_arena + sizeof(Livero));
    (*db_out)->db_file_fd = ctx->db_fd;
    (*db_out)->val_log_fd = ctx->val_log_fd;
    (*db_out)->vec_log_fd = ctx->vec_log_fd;
    (*db_out)->record_count = ctx->record_count;
    (*db_out)->db_file_size = ctx->db_file_size;
    (*db_out)->val_log_size = ctx->val_log_size;
    (*db_out)->vec_log_size = ctx->vec_log_size;
    (*db_out)->max_disk_bytes = ctx->max_disk_bytes;
    (*db_out)->vector_dim = ctx->vector_dim;
    (*db_out)->vector_type = ctx->vector_type;

    return LV_OK;
}

static void lv_query_heap_insert(LVQueryIndex* heap, const LVQueryIndex index, LVSize_t* heap_size) {
    assert(*heap_size < LV_MAX_K);
    heap[*heap_size] = index;
    int i = *heap_size;
    (*heap_size)++;

    while (i > 0 && heap[(i - 1) / 2].dot_result > heap[i].dot_result) {
        LVQueryIndex parent = heap[(i - 1) / 2];
        heap[(i - 1) / 2] = heap[i];
        heap[i] = parent;
        i = (i - 1) / 2;
    }
}

static void lv_query_heap_pop(LVQueryIndex* heap, LVSize_t* heap_size, LVQueryIndex* pop_out) {
    assert(*heap_size > 0);
    if (pop_out) {
        *pop_out = heap[0];
    }
    (*heap_size)--;
    heap[0] = heap[*heap_size];

    uint32_t i = 0;
    while (1) {
        uint32_t smallest = i;
        uint32_t left = 2 * i + 1;
        uint32_t right = 2 * i + 2;
        if (left < *heap_size && heap[left].dot_result < heap[smallest].dot_result) {
            smallest = left;
        }

        if (right < *heap_size && heap[right].dot_result < heap[smallest].dot_result) {
            smallest = right;
        }

        if (smallest != i) {
            LVQueryIndex tmp = heap[i];
            heap[i] = heap[smallest];
            heap[smallest] = tmp;
            i = smallest;
        } else {
            break;
        }
    }
}
