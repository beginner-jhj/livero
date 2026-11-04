#include "livero.h"

#include <assert.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "file/file.h"
#include "helper/helper.h"
#include "livero_internal.h"
#include "livero_types.h"
#include "struct/commit.h"
#include "struct/config.h"
#include "struct/row.h"
#include "vector/ternary.h"

static inline LVSize32_t lv_total_memory_bytes(const LVCount_t read_batch_m, const LVCount_t write_batch_m,
                                               const LVSize32_t max_value_len, const LVDim_t dim) {
    LVSize32_t total = sizeof(Livero);
    const LVSize32_t vector_bytes = ternary_chunks_bytes(dim);
    total += vector_bytes;
    total += vector_bytes * read_batch_m;
    total += vector_bytes * write_batch_m;
    total += LV_ROW_BYTES * read_batch_m;
    total += LV_ROW_BYTES * write_batch_m;
    total += max_value_len * write_batch_m;
    return total;
}

LVStatus lv_create(const char *db_path, const LVConfig *config, const LVOption *option, Livero **db_out) {
    if (!db_path) {
        LIVERO_ERROR("You missed a path to store data.");
        return LV_ERR_INVALID_PARAM;
    }

    if (!config) {
        LIVERO_ERROR("You missed the config.");
        return LV_ERR_INVALID_PARAM;
    }

    if (!option) {
        LIVERO_ERROR("You missed the option.");
        return LV_ERR_INVALID_PARAM;
    }

    if (!db_out) {
        LIVERO_ERROR("You missed the db_out pointer.");
        return LV_ERR_INVALID_PARAM;
    }

    if (config->max_disk_bytes == 0) {
        LIVERO_ERROR("max_disk_bytes must be greater than 0.");
        return LV_ERR_INVALID_PARAM;
    }

    if (config->vector_dimension == 0 || config->vector_dimension % LV_DIMENSION_MULTIPLE != 0) {
        LIVERO_ERROR("vector_dimension must be a multiple of %d.", LV_DIMENSION_MULTIPLE);
        return LV_ERR_INVALID_PARAM;
    }

    if (config->max_value_len == 0) {
        LIVERO_ERROR("max_value_len must be greater than 0.");
        return LV_ERR_INVALID_PARAM;
    }

    if (!lv_check_option(option)) {
        return LV_ERR_INVALID_PARAM;
    }

    int row_fd = -1;
    int vec_fd = -1;
    int val_fd = -1;
    int config_fd = -1;
    int commit_fd = -1;
    void *m_arena = NULL;

    const int create_flag = O_CREAT | O_EXCL | O_RDWR;

    LVStatus s = lv_open_file(db_path, LV_ROW_FILE_NAME, create_flag, &row_fd);

    if (s != LV_OK) {
        LIVERO_ERROR("Failed to open row.lv.");
        goto cleanup;
    }

    if ((s = lv_open_file(db_path, LV_VEC_FILE_NAME, create_flag, &vec_fd)) != LV_OK) {
        LIVERO_ERROR("Failed to open vec.lv.");
        goto cleanup;
    }

    if ((s = lv_open_file(db_path, LV_VAL_FILE_NAME, create_flag, &val_fd)) != LV_OK) {
        LIVERO_ERROR("Failed to open val.lv.");
        goto cleanup;
    }

    if ((s = lv_open_file(db_path, LV_CONFIG_FILE_NAME, create_flag, &config_fd)) != LV_OK) {
        LIVERO_ERROR("Faield to open config.lv");
        goto cleanup;
    }

    if ((s = lv_open_file(db_path, LV_COMMIT_FILE_NAME, create_flag, &commit_fd)) != LV_OK) {
        LIVERO_ERROR("Failed to open commit.lv.");
        goto cleanup;
    }

    if ((s = lv_save_config(config_fd, config)) != LV_OK) {
        LIVERO_ERROR("Faield to save the config.");
        goto cleanup;
    }

    if ((s = commit_init(commit_fd)) != LV_OK) {
        LIVERO_ERROR("Failed to initialize commit.lv.");
        goto cleanup;
    }

    const LVSize32_t total_meory_size = lv_total_memory_bytes(option->read_batch_m, option->write_batch_m,
                                                              config->max_value_len, config->vector_dimension);
    if ((s = lv_allocate_memory(total_meory_size, &m_arena)) != LV_OK) {
        LIVERO_ERROR("Failed to allocate %d bytes.", total_meory_size);
        goto cleanup;
    }

    LVInitCtx ctx = {.next_seq = LV_SEQ_START,
                     .next_commit_seq = LV_COMMIT_SEQ_START,
                     .row_fd = row_fd,
                     .vec_fd = vec_fd,
                     .val_fd = val_fd,
                     .row_bytes = 0,
                     .vec_bytes = 0,
                     .val_bytes = 0,
                     .max_disk_bytes = config->max_disk_bytes,
                     .max_value_len = config->max_value_len,
                     .vector_dimension = config->vector_dimension,
                     .vector_type = config->vector_type,
                     .read_batch_m = option->read_batch_m,
                     .write_batch_m = option->write_batch_m,
                     .sync_mode = option->sync_mode};

    lv_init_memory(m_arena, &ctx);
    *db_out = (Livero *)m_arena;

    lv_close_file(config_fd);
    return LV_OK;

cleanup:
    lv_close_file(row_fd);
    lv_close_file(vec_fd);
    lv_close_file(val_fd);
    lv_close_file(config_fd);
    lv_close_file(commit_fd);
    free(m_arena);
    return s;
}

LVStatus lv_open(const char *db_path, const LVOption *option, Livero **db_out) {
    if (!db_path) {
        LIVERO_ERROR("You missed a path to store data.");
        return LV_ERR_INVALID_PARAM;
    }

    if (!option) {
        LIVERO_ERROR("You missed the option.");
        return LV_ERR_INVALID_PARAM;
    }

    if (!db_out) {
        LIVERO_ERROR("You missed the db_out pointer.");
        return LV_ERR_INVALID_PARAM;
    }

    if (!lv_check_option(option)) {
        return LV_ERR_INVALID_PARAM;
    }

    int row_fd = -1;
    int vec_fd = -1;
    int val_fd = -1;
    int config_fd = -1;
    int commit_fd = -1;
    void *m_arena = NULL;

    const int open_flag = O_RDWR;

    LVStatus s = lv_open_file(db_path, LV_ROW_FILE_NAME, open_flag, &row_fd);

    if (s != LV_OK) {
        LIVERO_ERROR("Failed to open row.lv.");
        goto cleanup;
    }

    if ((s = lv_open_file(db_path, LV_VEC_FILE_NAME, open_flag, &vec_fd)) != LV_OK) {
        LIVERO_ERROR("Failed to open vec.lv.");
        goto cleanup;
    }

    if ((s = lv_open_file(db_path, LV_VAL_FILE_NAME, open_flag, &val_fd)) != LV_OK) {
        LIVERO_ERROR("Failed to open val.lv.");
        goto cleanup;
    }

    if ((s = lv_open_file(db_path, LV_COMMIT_FILE_NAME, open_flag, &commit_fd)) != LV_OK) {
        LIVERO_ERROR("Failed to open commit.lv.");
        goto cleanup;
    }

    if ((s = lv_open_file(db_path, LV_COMMIT_FILE_NAME, open_flag, &commit_fd)) != LV_OK) {
        LIVERO_ERROR("Failed to open commit.lv.");
        goto cleanup;
    }

    LVConfig config;
    if ((s = lv_config_d2m(config_fd, &config)) != LV_OK) {
        LIVERO_ERROR("Failed to read the config.");
        goto cleanup;
    }

    LVCommit commit;
    LVRecoverCtx recover_ctx = {
        .row_fd = row_fd,
        .vec_fd = vec_fd,
        .val_fd = val_fd,
        .commit_fd = commit_fd,
        .max_disk_bytes = config.max_disk_bytes,
        .vector_dim = config.vector_dimension,
    };
    if ((s = lv_recover(&recover_ctx, &commit)) != LV_OK) {
        LIVERO_ERROR("Failed to recover the db.");
        goto cleanup;
    }

    const LVSize32_t total_memory_size = lv_total_memory_bytes(option->read_batch_m, option->write_batch_m,
                                                               config.max_value_len, config.vector_dimension);
    if ((s = lv_allocate_memory(total_memory_size, &m_arena)) != LV_OK) {
        LIVERO_ERROR("Failed to allocate %d bytes.", total_memory_size);
        goto cleanup;
    }

    const int is_empty_db = commit.commit_seq == LV_COMMIT_SEQ_ZERO;
    LVSeq_t next_commit_seq;
    LVSeq_t next_seq;

    if (is_empty_db) {
        next_commit_seq = LV_COMMIT_SEQ_START;
        next_seq = LV_SEQ_START;
    } else {
        next_commit_seq = commit.commit_seq + 1;
        next_seq = commit.reserved_seq + 1;
    }

    LVInitCtx init_ctx = {
        .row_fd = row_fd,
        .vec_fd = vec_fd,
        .val_fd = val_fd,
        .commit_fd = commit_fd,
        .max_disk_bytes = config.max_disk_bytes,
        .max_value_len = config.max_value_len,
        .vector_dimension = config.vector_dimension,
        .vector_type = config.vector_type,
        .read_batch_m = option->read_batch_m,
        .write_batch_m = option->write_batch_m,
        .sync_mode = option->sync_mode,
        .next_commit_seq = next_commit_seq,
        .next_seq = next_seq,
        .row_bytes = LV_ROW_BYTES * commit.total_count,
        .vec_bytes = ternary_chunks_bytes(config.vector_dimension) * commit.total_count,
        .val_bytes = commit.total_value_bytes,
    };

    lv_init_memory(m_arena, &init_ctx);

    lv_close_file(config_fd);
    return LV_OK;

cleanup:
    lv_close_file(row_fd);
    lv_close_file(vec_fd);
    lv_close_file(val_fd);
    lv_close_file(config_fd);
    lv_close_file(commit_fd);
    free(m_arena);
    return s;
}

LVStatus lv_put(Livero *db, void *vector, void *value, const LVSize16_t value_len, const LVDate *date,
                LVSeq_t *seq_out) {
    if (!db) {
        LIVERO_ERROR("You miss a db.");
        return LV_ERR_INVALID_PARAM;
    }

    if (!vector) {
        LIVERO_ERROR("You must pass vector.");
        return LV_ERR_INVALID_PARAM;
    }

    if (!value || value_len == 0) {
        LIVERO_ERROR("You must pass value.");
        return LV_ERR_INVALID_PARAM;
    }

    if (value_len > db->max_value_len) {
        LIVERO_ERROR("value_len exceeds max_value_len(%d)", db->max_value_len);
        return LV_ERR_INVALID_PARAM;
    }

    if (date && !lv_date_is_valid(date)) {
        LIVERO_ERROR("You passed an invalid date object.");
        return LV_ERR_INVALID_PARAM;
    }

    const LVSeq_t current_seq = db->next_seq;
    const uint64_t epoch_time = lv_convert_date_to_epoch(date);
    const LVOffset_t value_disk_offset = db->val_bytes;

    ternary_convert(vector, db->vector_type, db->vector_dimension, db->ternary_scratch);
    LVRowCalcChecksumCtx row_checksum_ctx = {.seq = current_seq,
                                             .epoch_date = epoch_time,
                                             .value_offset = value_disk_offset,
                                             .value_len = value_len,
                                             .vector_dim = db->vector_dimension,
                                             .value = value,
                                             .ternary = db->ternary_scratch};
    const uint32_t row_checksum = row_calc_checksum(&row_checksum_ctx);
    LVRow row = {.seq = current_seq,
                 .date = epoch_time,
                 .value_offset = value_disk_offset,
                 .value_len = value_len,
                 .tombstone_flag = 0,
                 .checksum = row_checksum};

    const int pending_result = lv_pending_append(db, &row, db->ternary_scratch, value, value_len);

    if (pending_result < 0 || db->sync_mode == LV_SYNC_EVERY_PUT) {
        LVStatus s = lv_flush(db);
        if (s != LV_OK) {
            LIVERO_ERROR("Failed to flush data.");
            return s;
        }
    }

    if (pending_result < 0) {
        lv_pending_append(db, &row, db->ternary_scratch, value, value_len);
    }

    db->next_seq = current_seq + 1;

    return LV_OK;
}

LVStatus lv_get(Livero *db, const LVSeq_t target_seq, LVGetResult *result) {
    if (!db) {
        LIVERO_ERROR("You miss a db.");
        return LV_ERR_INVALID_PARAM;
    }

    if (!result) {
        LIVERO_ERROR("You must pass a LVGetResult buffer.");
        return LV_ERR_INVALID_PARAM;
    }

    if (target_seq >= db->next_seq) {
        return LV_ERR_NOT_FOUND;
    }

    LVStatus s = lv_pending_search(db, target_seq, result, NULL);

    if (s == LV_OK) {
        return LV_OK;
    }

    s = lv_interpolation_search_from_disk(db, target_seq, result, NULL);

    if (s == LV_ERR_NOT_FOUND) {
        return lv_binary_search_from_disk(db, target_seq, result, NULL);
    } else {
        return s;
    }
}

LVStatus lv_delete(Livero *db, const LVSeq_t target_seq) {
    if (!db) {
        LIVERO_ERROR("You miss a db.");
        return LV_ERR_INVALID_PARAM;
    }

    if (target_seq >= db->next_seq) {
        return LV_ERR_NOT_FOUND;
    }

    const uint8_t tombstone = LV_TOMBSTONE_FLAG;
    LVOffset_t row_offset = 0;

    LVStatus s = lv_pending_search(db, target_seq, NULL, &row_offset);

    if (s == LV_OK) {
        *(db->row_pending + row_offset + LV_ROW_TOMBSTONEFLAG_OFFSET) = tombstone;
        return LV_OK;
    }

    else if (s == LV_ERR_NOT_FOUND) {
        s = lv_binary_search_from_disk(db, target_seq, NULL, &row_offset);
        if (s == LV_OK) {
            return lv_write_file(db->row_fd, &tombstone, 1, row_offset + LV_ROW_TOMBSTONEFLAG_OFFSET);
        }
        return s;
    } else {
        return s;
    }
}

LVStatus lv_query(Livero *db, const LVTopK_t k, const void *query_vector, const LVDate *date1, const LVDate *date2,
                  LVQueryResult *result, LVSize32_t *result_size) {
    if (!db) {
        LIVERO_ERROR("You missed a db.");
        return LV_ERR_INVALID_PARAM;
    }
    if (!query_vector) {
        LIVERO_ERROR("You must pass a vector to query.");
        return LV_ERR_INVALID_PARAM;
    }
    if (k == 0) {
        LIVERO_ERROR("The k must be greater than 0.");
        return LV_ERR_INVALID_PARAM;
    }

    if (k > LV_MAX_K) {
        LIVERO_ERROR("The k exceeds LV_MAX_K(%d).", LV_MAX_K);
        return LV_ERR_INVALID_PARAM;
    }

    if (!result || !result_size) {
        LIVERO_ERROR("You must pass both result pointer and result_size pointer.");
        return LV_ERR_INVALID_PARAM;
    }

    const LVCount_t total_record_count = lv_total_record_count(db);
    const LVTopK_t TOP_K = k <= total_record_count ? k : (LVTopK_t)total_record_count;

    /*
        Filter rule: min_date <= target <= max_date. Both bounds are optional;
        a missing bound defaults to the full range so the scan has no branch for it.
    */
    uint64_t date1_epoch = 0;
    uint64_t date2_epoch = UINT64_MAX;
    if (date1) {
        date1_epoch = lv_convert_date_to_epoch(date1);
    }
    if (date2) {
        date2_epoch = lv_convert_date_to_epoch(date2);
    }
    const uint64_t min_date = date1_epoch < date2_epoch ? date1_epoch : date2_epoch;
    const uint64_t max_date = date2_epoch > date1_epoch ? date2_epoch : date1_epoch;

    const LVCount_t words = ternary_chunks_count(db->vector_dimension);
    const LVSize32_t vector_bytes = ternary_chunks_bytes(db->vector_dimension);

    memset(db->ternary_scratch, 0, vector_bytes);
    ternary_convert(query_vector, db->vector_type, db->vector_dimension, db->ternary_scratch);

    const LVTernaryChunk *q_vector = db->ternary_scratch;

    LVQueryIndex heap[TOP_K];  // min-heap: [0] is the current k-th best (the "threshold")
    LVSize32_t heap_size = 0;
    LVQueryHeapCtx h_ctx = {.date_flt1 = min_date, .date_flt2 = max_date, .top_k = TOP_K};

    for (LVCount_t i = 0; i < db->pending_count; i++) {
        const LVTernaryChunk *candidate = lv_n_pending_ternary(db, i);
        const int64_t dot_result = ternary_dot(q_vector, candidate, words);
        if (lv_query_heap_admit(heap, heap_size, TOP_K, dot_result)) {
            const uint8_t *row = lv_n_pending_row(db, i);
            h_ctx.row_seq = row_get_seq(row);
            h_ctx.row_date = row_get_date(row);
            h_ctx.valuen_len = row_get_value_len(row);
            h_ctx.dot_result = dot_result;
            lv_query_heap_update(heap, &heap_size, &h_ctx);
        }
    }

    const LVCount_t disk_record_count = lv_disk_record_count(db);
    const LVCount_t batch_m = db->window_batch_m;
    for (LVCount_t base = 0; base < disk_record_count; base += batch_m) {
        const LVCount_t remaining = disk_record_count - base;
        const LVCount_t n = remaining < batch_m ? remaining : batch_m;

        LVStatus s = lv_read_file(db->vec_fd, db->vector_window, n * vector_bytes, (LVOffset_t)base * vector_bytes);

        if (s != LV_OK) {
            return s;
        }

        int fetched = 0;
        LVOffset_t fetch_idx = 0;

        for (LVCount_t j = 0; j < n; ++j) {
            const LVTernaryChunk *candidate = lv_n_window_ternary(db, j);
            const int64_t dot_result = ternary_dot(q_vector, candidate, words);
            if (lv_query_heap_admit(heap, heap_size, TOP_K, dot_result)) {
                if (!fetched) {
                    fetch_idx = j;
                    const LVCount_t fetch_count = n - fetch_idx;
                    const LVOffset_t fetch_offset = (LVOffset_t)((base + fetch_count - 1) * LV_ROW_BYTES);
                    s = lv_read_file(db->row_fd, db->row_window, fetch_count * LV_ROW_BYTES, fetch_offset);
                    if (s != LV_OK) {
                        return s;
                    }
                    fetched = 1;
                }
                const uint8_t *row = lv_n_window_row(db, j - fetch_idx);
                h_ctx.row_seq = row_get_seq(row);
                h_ctx.row_date = row_get_date(row);
                h_ctx.valuen_len = row_get_value_len(row);
                h_ctx.dot_result = dot_result;
                lv_query_heap_update(heap, &heap_size, &h_ctx);
            }
        }
    }

    LVSize32_t heap_result_size = (int)heap_size;
    if (heap_result_size > 0) {
        LVQueryIndex heap_result[heap_result_size];
        for (int i = (int)heap_result_size - 1; i >= 0; i--) {
            lv_query_heap_pop(heap, &heap_result_size, &heap_result[i]);
        }

        for (LVSize32_t i = 0; i < heap_result_size; i++) {
            result[i].date = heap_result[i].q_result.date;
            result[i].seq = heap_result[i].q_result.seq;
            result[i].value_len = heap_result[i].q_result.value_len;
        }
    }

    *result_size = heap_result_size;

    return LV_OK;
}

LVStatus lv_close(Livero *db) {
    if (!db) {
        LIVERO_ERROR("You missed a db.");
        return LV_ERR_INVALID_PARAM;
    }

    if (db->pending_count > 0) {
        LVStatus s = lv_flush(db);
        if (s != LV_OK) {
            LIVERO_ERROR("Failed to flush data.");
            return s;
        }
    }

    lv_close_file(db->row_fd);
    lv_close_file(db->vec_fd);
    lv_close_file(db->val_fd);
    lv_close_file(db->commit_fd);
    free(db);
    return LV_OK;
}

void lv_stat(const Livero *db, LVDBStat *stat) {
    if (db && stat) {
        stat->config.max_disk_bytes = db->max_disk_bytes;
        stat->config.max_value_len = db->max_value_len;
        stat->config.vector_dimension = db->vector_dimension;
        stat->config.vector_type = db->vector_type;
        stat->option.write_batch_m = db->pending_batch_m;
        stat->option.read_batch_m = db->window_batch_m;
        stat->option.sync_mode = db->sync_mode;
        stat->row_file_bytes = db->row_bytes;
        stat->vec_file_bytes = db->vec_bytes;
        stat->val_file_bytes = db->val_bytes;
        stat->row_pending_bytes = lv_pending_row_bytse(db);
        stat->vec_pending_bytes = lv_pending_vec_bytes(db);
        stat->val_pending_bytes = lv_pending_val_bytes(db);
    }
}

/*
  Below functions are all internal apis.
*/

LVStatus lv_allocate_memory(const LVSize32_t size, void **ptr_out) {
    assert(ptr_out);

    const LVSize16_t rounded = (size + LV_MEMORY_ALIGNMENT - 1) & ~(LV_MEMORY_ALIGNMENT - 1);
    void *m = aligned_alloc(LV_MEMORY_ALIGNMENT, rounded);

    if (!m) {
        return LV_ERR_ALLOCATE_MEMORY;
    }

    *ptr_out = m;

    return LV_OK;
}

void lv_init_memory(uint8_t *m_arena, const LVInitCtx *ctx) {
    assert(m_arena && ctx);
    assert((uintptr_t)m_arena % LV_MEMORY_ALIGNMENT == 0);

    LVOffset_t offset = 0;
    const LVSize32_t vector_bytes = ternary_chunks_bytes(ctx->vector_dimension);

    Livero *db = (Livero *)m_arena;
    offset += sizeof(Livero);

    LVTernaryChunk *ternary_scratch = (LVTernaryChunk *)(m_arena + offset);
    offset += vector_bytes;

    LVTernaryChunk *vector_window = (LVTernaryChunk *)(m_arena + offset);
    offset += vector_bytes * ctx->read_batch_m;

    LVTernaryChunk *vector_pending = (LVTernaryChunk *)(m_arena + offset);
    offset += vector_bytes * ctx->write_batch_m;

    uint8_t *row_window = m_arena + offset;
    offset += LV_ROW_BYTES * ctx->read_batch_m;

    uint8_t *row_pending = (m_arena + offset);
    offset += LV_ROW_BYTES * ctx->write_batch_m;

    uint8_t *value_pending = m_arena + offset;

    db->next_seq = ctx->next_seq;
    db->next_commit_seq = ctx->next_commit_seq;
    db->ternary_scratch = ternary_scratch;
    db->vector_window = vector_window;
    db->vector_pending = vector_pending;
    db->row_window = row_window;
    db->row_pending = row_pending;
    db->value_pending = value_pending;
    db->row_fd = ctx->row_fd;
    db->vec_fd = ctx->vec_fd;
    db->val_fd = ctx->val_fd;
    db->commit_fd = ctx->commit_fd;
    db->row_bytes = ctx->row_bytes;
    db->vec_bytes = ctx->vec_bytes;
    db->val_bytes = ctx->val_bytes;
    db->max_disk_bytes = ctx->max_disk_bytes;
    db->pending_count = 0;
    db->window_batch_m = ctx->read_batch_m;
    db->pending_batch_m = ctx->write_batch_m;
    db->max_value_len = ctx->max_value_len;
    db->value_pending_bytes = 0;
    db->vector_dimension = ctx->vector_dimension;
    db->vector_type = ctx->vector_type;
    db->sync_mode = ctx->sync_mode;
}

LVStatus lv_recover(const LVRecoverCtx *ctx, LVCommit *commit_out) {
    assert(ctx && commit_out);

    const LVSize32_t vector_bytes = ternary_chunks_bytes(ctx->vector_dim);

    LVCommit cmt = {0};
    LVCount_t nonempty_count = 0;
    LVCommit best = {0};
    int best_idx = -1;
    LVCount_t corrupted_count = 0;
    int corrupted_idx = -1;

    for (int i = 0; i < LV_COMMIT_SLOT_M; ++i) {
        LVCommitStatus cmt_status = commit_d2m(ctx->commit_fd, i * LV_COMMIT_SECTOR_BYTES, &cmt);

        if (cmt_status.db == LV_OK) {
            if (cmt_status.commit == LV_COMMIT_EMPTY) {
                /*
                    This is a zero initialized commit. Neither an error and a corruption
                */
                continue;
            } else if (cmt_status.commit == LV_COMMIT_CORRUPT) {
                /*
                    This is a corrupted commit.
                */
                corrupted_count++;
                corrupted_idx = i;
            } else if (cmt_status.commit == LV_COMMIT_OK) {
                if (!commit_sanity(&cmt, i, vector_bytes, ctx->max_disk_bytes)) {
                    /*
                        This commit's checksum is correct. But it saved invalid values.
                    */
                    corrupted_count++;
                    corrupted_idx = i;
                } else {
                    /*
                        The latest valid commit wins
                    */
                    if (cmt.commit_seq > best.commit_seq) {
                        best = cmt;
                        best_idx = i;
                    }
                }
            }
            nonempty_count++;
        } else {
            return cmt_status.db;
        }
    }

    if (nonempty_count == 0) {
        /*
            This is a new db.
        */
        *commit_out = (LVCommit){0};
        return LV_OK;
    }

    /* Non-empty slots exist but none is trustworthy. Without this check,
     * best_idx == -1 would leak into the arithmetic below and make the
     * "expected slot" silently become slot 0. */
    if (best_idx < 0) {
        return LV_ERR_MEDIA_CRASHED;
    }

    /*
        Only torn-write that happened in the last write is the expected crash.
        If the corrupted commit is more than 1 or
        a corrupted commit is not the exact next one of the latest valid one,
        it means there was an unpredicted crash on the successfully written commit
        which is not recoverable.
    */
    if (corrupted_count > 1) {
        return LV_ERR_MEDIA_CRASHED;
    } else if (corrupted_count == 1 && corrupted_idx != (best_idx + 1) % LV_COMMIT_SLOT_M) {
        return LV_ERR_MEDIA_CRASHED;
    }

    /*
        Finally check batched data that the best commit saved.
        possible errors are LV_ERR_DATA_CORRUPT or LV_ERR_IO.
    */
    LVRowCCFDCtx rccfd = {.row_fd = ctx->row_fd,
                          .vec_fd = ctx->vec_fd,
                          .val_fd = ctx->val_fd,
                          .row_offset = 0,
                          .vec_offset = 0,
                          .vector_bytes = vector_bytes};

    for (uint64_t k = (uint64_t)best.total_count - best.batch_count; k < (uint64_t)best.total_count; ++k) {
        rccfd.row_offset = k * LV_ROW_BYTES;
        rccfd.vec_offset = k * vector_bytes;

        LVStatus s = row_checksum_check_from_disk(&rccfd);
        if (s != LV_OK) {
            return s;
        }
    }

    *commit_out = best;
    return LV_OK;
}

LVStatus lv_flush(Livero *db) {
    assert(db);

    const LVSize32_t teranry_bytes = ternary_chunks_bytes(db->vector_dimension);
    const LVOffset_t row_offset_to_write = db->row_bytes;
    const LVOffset_t vec_offset_to_write = db->vec_bytes;
    const LVOffset_t val_offset_to_write = db->val_bytes;
    const int sync = db->sync_mode != LV_SYNC_NONE;

    LVSize32_t packed = 0;
    for (LVCount_t i = 0; i < db->pending_count; ++i) {
        const LVSize16_t len = row_get_value_len(lv_n_pending_row(db, i));
        memmove(db->value_pending + packed, lv_n_pending_value(db, i), len);
        packed += len;
    }

    LVStatus s = lv_write_file(db->val_fd, db->value_pending, packed, val_offset_to_write);

    if (s != LV_OK) {
        return s;
    }

    if (sync) {
        if ((s = lv_sync_file(db->val_fd)) != LV_OK) {
            LIVERO_ERROR("Faield to sync val.lv.");
            return s;
        }
    }

    const LVSize32_t teranry_bytes_to_flush = db->pending_count * teranry_bytes;
    if ((s = lv_write_file(db->vec_fd, db->vector_pending, teranry_bytes_to_flush, vec_offset_to_write)) != LV_OK) {
        return s;
    }

    if (sync) {
        if ((s = lv_sync_file(db->vec_fd)) != LV_OK) {
            LIVERO_ERROR("Failed to sync vec.lv.");
            return s;
        }
    }

    const LVSize32_t row_bytes_to_flush = db->pending_count * LV_ROW_BYTES;

    if ((s = lv_write_file(db->row_fd, db->row_pending, row_bytes_to_flush, row_offset_to_write)) != LV_OK) {
        return s;
    }

    if (sync) {
        if ((s = lv_sync_file(db->row_fd)) != LV_OK) {
            LIVERO_ERROR("Failed to sync row.lv.");
            return s;
        }
    }

    const LVCommit cmt = {.commit_seq = db->next_commit_seq,
                          .first_seq = row_get_seq(lv_first_pending_row(db)),
                          .reserved_seq = row_get_seq(lv_last_pending_row(db)),
                          .total_count = lv_total_record_count(db),
                          .batch_count = db->pending_count,
                          .total_value_bytes = db->row_bytes + db->value_pending_bytes,
                          .batch_value_bytes = db->value_pending_bytes};

    if ((s = commit_write(db->commit_fd, &cmt)) != LV_OK) {
        return s;
    }

    if (sync) {
        if ((s = lv_sync_file(db->commit_fd)) != LV_OK) {
            LIVERO_ERROR("Failed to sync commit.lv.");
            return s;
        }
    }

    db->row_bytes += row_bytes_to_flush;
    db->vec_bytes += teranry_bytes_to_flush;
    db->val_bytes += db->value_pending_bytes;

    db->pending_count = 0;
    db->value_pending_bytes = 0;

    return LV_OK;
}

int lv_pending_append(Livero *db, const LVRow *row, const void *ternary_vector, const void *value,
                      const LVSize16_t value_len) {
    assert(db && row && ternary_vector && value);

    if (db->pending_count >= db->pending_batch_m) {
        return -1;
    }

    const LVSize32_t vector_bytes = ternary_chunks_bytes(db->vector_dimension);
    const LVOffset_t row_batch_offset = db->pending_count * LV_ROW_BYTES;
    const LVOffset_t vector_batch_offset = db->pending_count * vector_bytes;
    const LVOffset_t value_batch_offset = db->pending_count * db->max_value_len;

    row_write_on_memory(db->row_pending + row_batch_offset, row);
    memcpy((uint8_t *)db->vector_pending + vector_batch_offset, ternary_vector, vector_bytes);
    memcpy(db->value_pending + value_batch_offset, value, value_len);

    db->pending_count += 1;
    db->value_pending_bytes += value_len;

    return 0;
}

LVStatus lv_pending_search(const Livero *db, const LVSeq_t target_seq, LVGetResult *result,
                           LVOffset_t *row_offset_out) {
    assert(db);

    if (db->pending_count == 0) {
        return LV_ERR_NOT_FOUND;
    }
    const LVSeq_t first_pending_seq = row_get_seq(db->row_pending);

    if (target_seq < first_pending_seq || target_seq >= first_pending_seq + db->pending_count) {
        return LV_ERR_NOT_FOUND;
    }

    const LVOffset_t row_offset = (target_seq - first_pending_seq) * LV_ROW_BYTES;
    const LVOffset_t value_offset = (target_seq - first_pending_seq) * db->max_value_len;

    const uint8_t *row = db->row_pending + row_offset;
    const uint8_t *value = db->value_pending + value_offset;

    if (result) {
        lv_fill_get_result(row, value, -1, result);
    }

    if (row_offset_out) {
        *row_offset_out = row_offset;
    }

    return LV_OK;
}

LVStatus lv_interpolation_search_from_disk(Livero *db, const LVSeq_t target_seq, LVGetResult *result,
                                           LVOffset_t *row_offset_out) {
    assert(db);

    uint8_t low[LV_ROW_BYTES] = {0};
    uint8_t high[LV_ROW_BYTES] = {0};

    const LVCount_t disk_record_count = lv_disk_record_count(db);
    if (disk_record_count == 0) {
        return LV_ERR_NOT_FOUND;
    }

    const int low_idx = 0;
    const int high_idx = (int)(disk_record_count - 1);

    LVStatus s = lv_read_file(db->row_fd, low, LV_ROW_BYTES, (LVOffset_t)low_idx * LV_ROW_BYTES);
    if (s != LV_OK) {
        return s;
    }

    s = lv_read_file(db->row_fd, high, LV_ROW_BYTES, (LVOffset_t)high_idx * LV_ROW_BYTES);
    if (s != LV_OK) {
        return s;
    }

    const LVSeq_t low_seq = row_get_seq(low);
    const LVSeq_t high_seq = row_get_seq(high);

    if (low_seq == high_seq) {
        if (result && target_seq == low_seq) {
            if ((s = lv_fill_get_result(low, NULL, db->val_fd, result)) != LV_OK) {
                return s;
            }
        }
        return LV_ERR_NOT_FOUND;
    }

    if (target_seq < low_seq || target_seq > high_seq) {
        return LV_ERR_NOT_FOUND;
    }

    const int p_idx = low_idx + (int)(((target_seq - low_seq) * (high_idx - low_idx)) / (high_seq - low_seq));

    LVWindowRange r = lv_window_range(p_idx, low_idx, high_idx, (int)db->window_batch_m);

    int start_idx = r.start_idx;
    int end_idx = r.end_idx;

    LVOffset_t read_offset = (LVOffset_t)start_idx * LV_ROW_BYTES;
    LVCount_t read_count = (LVCount_t)(end_idx - start_idx + 1);

    if ((s = lv_read_file(db->row_fd, db->row_window, read_count * LV_ROW_BYTES, read_offset)) != LV_OK) {
        return s;
    }

    const LVOffset_t expected_offset = lv_window_target_offset(db, r.align, read_count, LV_ROW_BYTES);
    LVOffset_t row_offset = expected_offset;
    s = lv_window_search(db, target_seq, read_count, expected_offset, result, &row_offset);

    if (s == LV_OK) {
        if (row_offset_out) {
            *row_offset_out = read_offset + row_offset;
        }
    }

    return s;
}

LVStatus lv_binary_search_from_disk(Livero *db, const LVSeq_t target_seq, LVGetResult *result,
                                    LVOffset_t *row_offset_out) {
    assert(db);

    const LVCount_t disk_record_count = lv_disk_record_count(db);

    if (disk_record_count == 0) {
        return LV_ERR_NOT_FOUND;
    }

    int low = 0;
    int high = (int)(disk_record_count - 1);
    int mid = low + (high - low) / 2;

    while (low <= high) {
        LVWindowRange r = lv_window_range(mid, low, high, (int)db->window_batch_m);

        LVOffset_t read_offset = (LVOffset_t)r.start_idx * LV_ROW_BYTES;
        LVCount_t read_count = (LVCount_t)(r.end_idx - r.start_idx + 1);

        LVStatus s = lv_read_file(db->row_fd, db->row_window, read_count * LV_ROW_BYTES, read_offset);

        if (s != LV_OK) {
            return s;
        }

        const LVOffset_t expected_offset = lv_window_target_offset(db, r.align, read_count, LV_ROW_BYTES);
        const uint8_t *expected_row = db->row_window + expected_offset;
        const LVSeq_t expected_seq = row_get_seq(expected_row);
        LVOffset_t row_offset = expected_offset;

        s = lv_window_search(db, target_seq, read_count, expected_offset, result, &row_offset);

        if (s == LV_OK) {
            if (row_offset_out) {
                *row_offset_out = read_offset + row_offset;
            }

            return LV_OK;
        }

        else if (s == LV_ERR_NOT_FOUND) {
            if (expected_seq < target_seq) {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        } else {
            return s;
        }

        mid = low + (high - low) / 2;
    }
    return LV_ERR_NOT_FOUND;
}

LVWindowRange lv_window_range(const int p, const int low, const int high, const int m) {
    assert(low <= high);
    assert(p >= low && p <= high);
    const int read_m = m;
    const int half_m = read_m / 2;

    int start_idx = 0;
    int end_idx = 0;
    LVWindowAlign align = LV_BATCH_CENTER;

    //[pos - m/2, pos + m/2]
    if (p - half_m >= low && p + (read_m - half_m - 1) <= high) {
        start_idx = p - half_m;
        end_idx = start_idx + read_m - 1;
    } else if (p - half_m < low) {  // [pos, min(pos + m - 1, high)]
        start_idx = p;
        end_idx = p + read_m - 1;
        if (end_idx > high) {
            end_idx = high;
        }
        align = LV_BATCH_LEFT;
    } else {  // [max(p - ( m- 1), low), pos]
        end_idx = p;
        start_idx = p - read_m + 1;
        if (start_idx < low) {
            start_idx = low;
        }
        align = LV_BATCH_RIGHT;
    }

    return (LVWindowRange){.start_idx = start_idx, .end_idx = end_idx, .align = align};
}

LVOffset_t lv_window_target_offset(const Livero *db, const LVWindowAlign align, const LVCount_t count,
                                   const LVSize32_t item_size) {
    assert(db);
    LVOffset_t idx = 0;
    switch (align) {
        case LV_BATCH_CENTER: {
            idx = count / 2;
            break;
        }

        case LV_BATCH_LEFT: {
            idx = 0;
            break;
        }
        case LV_BATCH_RIGHT: {
            idx = count - 1;
            break;
        }
    }
    return idx * item_size;
}

LVStatus lv_window_search(const Livero *db, const LVSeq_t target_seq, const LVCount_t count,
                          const LVOffset_t target_offset, LVGetResult *result, LVOffset_t *row_offset_out) {
    assert(db);

    const uint8_t *expected_row = db->row_window + target_offset;
    const LVSeq_t expected_seq = row_get_seq(expected_row);
    if (expected_seq == target_seq) {
        if (result) {
            LVStatus s = lv_fill_get_result(expected_row, NULL, db->val_fd, result);
            if (s != LV_OK) {
                return s;
            }
        }

        if (row_offset_out) {
            *row_offset_out = target_offset;
        }
        return LV_OK;
    } else {
        for (LVCount_t i = 0; i < count; ++i) {
            const uint8_t *row = db->row_window + i * LV_ROW_BYTES;
            if (row_get_seq(row) == target_seq) {
                if (result) {
                    LVStatus s = lv_fill_get_result(row, NULL, db->val_fd, result);
                    if (s != LV_OK) {
                        return s;
                    }
                }

                if (row_offset_out) {
                    *row_offset_out = i * LV_ROW_BYTES;
                }
                return LV_OK;
            }
        }
        return LV_ERR_NOT_FOUND;
    }
}

LVStatus lv_fill_get_result(const uint8_t *row, const void *value, const int val_fd, LVGetResult *result) {
    assert(row && result);
    assert(!value && val_fd != -1);
    const LVOffset_t value_offset = row_get_value_offset(row);
    const LVSize16_t value_len = row_get_value_len(row);
    if (result->value) {
        if (value) {
            memcpy(result->value, value, value_len);
        } else {
            LVStatus s = lv_read_file(val_fd, result->value, value_len, value_offset);
            if (s != LV_OK) {
                return s;
            }
        }
    }
    lv_convert_epoch_to_date(row_get_date(row), &result->date);
    result->value_len = value_len;
    return LV_OK;
}

void lv_query_heap_insert(LVQueryIndex *heap, const LVQueryIndex index, LVSize32_t *heap_size) {
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

void lv_query_heap_pop(LVQueryIndex *heap, LVSize32_t *heap_size, LVQueryIndex *pop_out) {
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

int lv_query_heap_admit(const LVQueryIndex *heap, const LVSize32_t size, const LVTopK_t threshold,
                        const int64_t dot_result) {
    assert(heap);
    return size < threshold || dot_result > heap[0].dot_result;
}

void lv_query_heap_update(LVQueryIndex *heap, LVSize32_t *size, const LVQueryHeapCtx *ctx) {
    assert(heap && size && ctx);
    const int full = *size >= ctx->top_k;
    int passes_filter = 0;
    if (ctx->date_flt1 <= ctx->row_date && ctx->row_date <= ctx->date_flt2) {
        passes_filter = 1;
    }

    if (passes_filter) {
        LVDate d;
        lv_convert_epoch_to_date(ctx->row_date, &d);
        const LVQueryResult q_result = {.date = d, .seq = ctx->row_seq, .value_len = ctx->valuen_len};
        const LVQueryIndex index = {.dot_result = ctx->dot_result, .q_result = q_result};
        if (full) {
            lv_query_heap_pop(heap, size, NULL);
        }
        lv_query_heap_insert(heap, index, size);
    }
}
