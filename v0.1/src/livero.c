#include "livero.h"

#include <assert.h>
#include <fcntl.h>
#include <inttypes.h>
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
                                               const LVSize32_t max_value_len, const LVDim_t vector_dimension) {
    LVSize32_t total = sizeof(Livero);
    const LVSize32_t ternary_bytes = ternary_chunks_bytes(vector_dimension);
    total += ternary_bytes;
    total += ternary_bytes * read_batch_m;
    total += ternary_bytes * write_batch_m;
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

    if (config->max_disk_bytes <= LV_BASELINE_DISK_USAGE) {
        LIVERO_ERROR("max_disk_bytes is too few. Baseline usage is %" PRIu64 "bytes.\n", LV_BASELINE_DISK_USAGE);
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
    int ternary_fd = -1;
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

    if ((s = lv_open_file(db_path, LV_TERNARY_FILE_NAME, create_flag, &ternary_fd)) != LV_OK) {
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
                     .ternary_fd = ternary_fd,
                     .val_fd = val_fd,
                     .commit_fd = commit_fd,
                     .row_bytes = 0,
                     .ternary_bytes = 0,
                     .val_bytes = 0,
                     .max_disk_bytes = config->max_disk_bytes,
                     .max_value_len = config->max_value_len,
                     .vector_dimension = config->vector_dimension,
                     .vector_type = config->vector_type,
                     .read_batch_m = option->read_batch_m,
                     .write_batch_m = option->write_batch_m,
                     .sync_mode = option->sync_mode};

    lv_init_memory(m_arena, &ctx);
    lv_close_file(config_fd);
    *db_out = (Livero *)m_arena;
    return LV_OK;

cleanup:
    lv_close_file(row_fd);
    lv_close_file(ternary_fd);
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
    int ternary_fd = -1;
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

    if ((s = lv_open_file(db_path, LV_TERNARY_FILE_NAME, open_flag, &ternary_fd)) != LV_OK) {
        LIVERO_ERROR("Failed to open vec.lv.");
        goto cleanup;
    }

    if ((s = lv_open_file(db_path, LV_VAL_FILE_NAME, open_flag, &val_fd)) != LV_OK) {
        LIVERO_ERROR("Failed to open val.lv.");
        goto cleanup;
    }

    // fix bug. opened commit.lv twice, did not open config.lv
    if ((s = lv_open_file(db_path, LV_CONFIG_FILE_NAME, open_flag, &config_fd)) != LV_OK) {
        LIVERO_ERROR("Failed to open config.lv.");
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
        .ternary_fd = ternary_fd,
        .val_fd = val_fd,
        .commit_fd = commit_fd,
        .max_disk_bytes = config.max_disk_bytes,
        .vector_dimension = config.vector_dimension,
    };
    int first_commit_torn_out = 0;
    if ((s = lv_recover(&recover_ctx, &commit, &first_commit_torn_out)) != LV_OK) {
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
        if (first_commit_torn_out) {
            next_commit_seq = LV_COMMIT_SEQ_START;
            next_seq = LV_SEQ_START + LV_MAX_BATCH_M;
        } else {
            next_commit_seq = LV_COMMIT_SEQ_START;
            next_seq = LV_SEQ_START;
        }

    } else {
        next_commit_seq = commit.commit_seq + 1;
        next_seq = commit.reserved_seq + 1;
    }

    LVInitCtx init_ctx = {
        .row_fd = row_fd,
        .ternary_fd = ternary_fd,
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
        .ternary_bytes = ternary_chunks_bytes(config.vector_dimension) * commit.total_count,
        .val_bytes = commit.total_value_bytes,
    };

    lv_init_memory(m_arena, &init_ctx);
    lv_close_file(config_fd);
    *db_out = (Livero *)m_arena;  // fix bug
    return LV_OK;

cleanup:
    lv_close_file(row_fd);
    lv_close_file(ternary_fd);
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
    const uint64_t epoch_date = lv_convert_date_to_epoch(date);
    const LVOffset_t value_disk_offset = db->val_file_bytes;
    const LVOffset_t value_pending_offset = db->value_pending_bytes;
    const LVOffset_t value_offset =
        value_disk_offset + value_pending_offset;  // fix offset bug: value_offset should be disk + pending bytes

    ternary_convert(vector, db->vector_type, db->vector_dimension, db->ternary_scratch);
    LVRowCalcChecksumCtx row_checksum_ctx = {.seq = current_seq,
                                             .epoch_date = epoch_date,
                                             .value_offset = value_offset,
                                             .value_len = value_len,
                                             .vector_dimension = db->vector_dimension,
                                             .value = value,
                                             .ternary_chunks = db->ternary_scratch};
    const uint32_t row_checksum = row_calc_checksum(&row_checksum_ctx);
    LVRow row = {.seq = current_seq,
                 .epoch_date = epoch_date,
                 .value_offset = value_offset,
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
        int r = lv_pending_append(db, &row, db->ternary_scratch, value, value_len);
        assert(r == 0);
    }

    if (seq_out) {
        *seq_out = current_seq;
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

    if (s == LV_OK || s == LV_ERR_DELETED) {
        return s;
    }

    LVDiskSearchCtx c = lv_disk_search_ctx(db);
    s = lv_interpolation_search_from_disk(&c, target_seq, result, NULL);

    if (s == LV_ERR_NOT_FOUND) {
        return lv_binary_search_from_disk(&c, target_seq, result, NULL);
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
        LVDiskSearchCtx c = lv_disk_search_ctx(db);
        s = lv_binary_search_from_disk(&c, target_seq, NULL, &row_offset);
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
        Filter rule: min_epoch_date <= target <= max_epoch_date. Both bounds are optional;
        a missing bound defaults to the full range so the scan has no branch for it.
    */
    uint64_t epoch_date1 = 0;
    uint64_t epoch_date2 = UINT64_MAX;
    if (date1) {
        epoch_date1 = lv_convert_date_to_epoch(date1);
    }
    if (date2) {
        epoch_date2 = lv_convert_date_to_epoch(date2);
    }
    const uint64_t min_epoch_date = epoch_date1 < epoch_date2 ? epoch_date1 : epoch_date2;
    const uint64_t max_epoch_date = epoch_date2 > epoch_date1 ? epoch_date2 : epoch_date1;

    const LVCount_t ternary_chunk_count = ternary_chunks_count(db->vector_dimension);
    const LVSize32_t ternary_bytes = ternary_chunks_bytes(db->vector_dimension);

    memset(db->ternary_scratch, 0, ternary_bytes);
    ternary_convert(query_vector, db->vector_type, db->vector_dimension, db->ternary_scratch);

    const LVTernaryChunk *query_ternary = db->ternary_scratch;

    LVQueryIndex heap[TOP_K];  // min-heap: [0] is the current k-th best (the "threshold")
    LVSize32_t heap_size = 0;
    LVQueryHeapCtx h_ctx = {
        .min_epoch_date = min_epoch_date, .max_epoch_date = max_epoch_date, .top_k = TOP_K};

    for (LVCount_t i = 0; i < db->pending_count; i++) {
        const LVTernaryChunk *ternary_candidate = lv_n_pending_ternary(db, i);
        const int64_t dot_result = ternary_dot(query_ternary, ternary_candidate, ternary_chunk_count);
        if (lv_query_heap_admit(heap, heap_size, TOP_K, dot_result)) {
            const uint8_t *row = lv_n_pending_row(db, i);
            h_ctx.row = (uint8_t *)row;
            h_ctx.dot_result = dot_result;
            lv_query_heap_update(heap, &heap_size, &h_ctx);
        }
    }

    const LVCount_t disk_record_count = lv_disk_record_count(db);
    const LVCount_t batch_m = db->window_batch_m;
    for (LVCount_t base = 0; base < disk_record_count; base += batch_m) {
        const LVCount_t remaining = disk_record_count - base;
        const LVCount_t n = remaining < batch_m ? remaining : batch_m;

        LVStatus s = lv_read_file(db->ternary_fd, db->ternary_window, n * ternary_bytes,
                                  (LVOffset_t)base * ternary_bytes);

        if (s != LV_OK) {
            return s;
        }

        int row_fetched = 0;
        LVCount_t row_fetch_idx = 0;

        for (LVCount_t j = 0; j < n; ++j) {
            const LVTernaryChunk *ternary_candidate = lv_n_window_ternary(db, j);
            const int64_t dot_result = ternary_dot(query_ternary, ternary_candidate, ternary_chunk_count);
            if (lv_query_heap_admit(heap, heap_size, TOP_K, dot_result)) {
                if (!row_fetched) {
                    row_fetch_idx = j;
                    const LVCount_t fetch_count = n - row_fetch_idx;
                    const LVOffset_t fetch_offset = (LVOffset_t)((base + row_fetch_idx) * LV_ROW_BYTES);  // fix bug
                    s = lv_read_file(db->row_fd, db->row_window, fetch_count * LV_ROW_BYTES, fetch_offset);
                    if (s != LV_OK) {
                        return s;
                    }
                    row_fetched = 1;
                }
                const uint8_t *row = lv_n_window_row(db, j - row_fetch_idx);
                h_ctx.row = (uint8_t *)row;
                h_ctx.dot_result = dot_result;
                lv_query_heap_update(heap, &heap_size, &h_ctx);
            }
        }
    }

    const LVSize32_t heap_result_size = (int)heap_size;
    if (heap_result_size > 0) {
        LVQueryIndex heap_result[heap_result_size];
        for (int i = (int)heap_result_size - 1; i >= 0; i--) {
            lv_query_heap_pop(heap, &heap_size, &heap_result[i]);
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

void lv_close(Livero *db) {
    if (!db) {
        LIVERO_ERROR("You missed a db.");
        return;
    }

    if (db->pending_count > 0) {
        LVStatus s = lv_flush(db);
        if (s != LV_OK) {
            LIVERO_ERROR("Failed to flush data.");
        }
    }

    lv_close_file(db->row_fd);
    lv_close_file(db->ternary_fd);
    lv_close_file(db->val_fd);
    lv_close_file(db->commit_fd);
    free(db);
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
        stat->row_file_bytes = db->row_file_bytes;
        stat->ternary_file_bytes = db->ternary_file_bytes;
        stat->val_file_bytes = db->val_file_bytes;
        stat->row_pending_bytes = lv_pending_row_bytse(db);
        stat->ternary_pending_bytes = lv_pending_ternary_bytes(db);
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
    const LVSize32_t ternary_bytes = ternary_chunks_bytes(ctx->vector_dimension);

    Livero *db = (Livero *)m_arena;
    offset += sizeof(Livero);

    LVTernaryChunk *ternary_scratch = (LVTernaryChunk *)(m_arena + offset);
    offset += ternary_bytes;

    LVTernaryChunk *ternary_window = (LVTernaryChunk *)(m_arena + offset);
    offset += ternary_bytes * ctx->read_batch_m;

    LVTernaryChunk *ternary_pending = (LVTernaryChunk *)(m_arena + offset);
    offset += ternary_bytes * ctx->write_batch_m;

    uint8_t *row_window = m_arena + offset;
    offset += LV_ROW_BYTES * ctx->read_batch_m;

    uint8_t *row_pending = (m_arena + offset);
    offset += LV_ROW_BYTES * ctx->write_batch_m;

    uint8_t *value_pending = m_arena + offset;

    db->next_seq = ctx->next_seq;
    db->next_commit_seq = ctx->next_commit_seq;
    db->ternary_scratch = ternary_scratch;
    db->ternary_window = ternary_window;
    db->ternary_pending = ternary_pending;
    db->row_window = row_window;
    db->row_pending = row_pending;
    db->value_pending = value_pending;
    db->row_fd = ctx->row_fd;
    db->ternary_fd = ctx->ternary_fd;
    db->val_fd = ctx->val_fd;
    db->commit_fd = ctx->commit_fd;
    db->row_file_bytes = ctx->row_bytes;
    db->ternary_file_bytes = ctx->ternary_bytes;
    db->val_file_bytes = ctx->val_bytes;
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

LVStatus lv_recover(const LVRecoverCtx *ctx, LVCommit *commit_out, int *first_commit_torn_out) {
    assert(ctx && commit_out);

    const LVSize32_t ternary_bytes = ternary_chunks_bytes(ctx->vector_dimension);

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
                if (!commit_sanity(&cmt, i, ternary_bytes, ctx->max_disk_bytes)) {
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

    /* No slot passed validation. The only crash-consistent way to reach this
     * state is a torn first commit: commit_seq 1 always lands in slot 0, so a
     * power loss during the very first flush leaves slot 0 corrupt and every
     * other slot empty. Treat that as a fresh database. Any other layout means
     * committed history was damaged, so refuse to open. */
    if (best_idx < 0) {
        if (corrupted_count == 1 && corrupted_idx == 0) {
            // The first batch was never acknowledged, so start from an empty commit.
            // Its seqs may already have been returned by put, so the caller must skip them.
            memset(commit_out, 0, sizeof(*commit_out));
            *first_commit_torn_out = 1;
            return LV_OK;
        }
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
                          .ternary_fd = ctx->ternary_fd,
                          .val_fd = ctx->val_fd,
                          .row_offset = 0,
                          .ternary_offset = 0,
                          .ternary_bytes = ternary_bytes};

    for (uint64_t k = (uint64_t)best.total_count - best.batch_count; k < (uint64_t)best.total_count; ++k) {
        rccfd.row_offset = k * LV_ROW_BYTES;
        rccfd.ternary_offset = k * ternary_bytes;

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

    const LVSize32_t ternary_bytes = ternary_chunks_bytes(db->vector_dimension);
    const LVOffset_t row_offset_to_write = db->row_file_bytes;
    const LVOffset_t ternary_offset_to_write = db->ternary_file_bytes;
    const LVOffset_t val_offset_to_write = db->val_file_bytes;
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

    const LVSize32_t ternary_bytes_to_flush = db->pending_count * ternary_bytes;
    if ((s = lv_write_file(db->ternary_fd, db->ternary_pending, ternary_bytes_to_flush,
                           ternary_offset_to_write)) != LV_OK) {
        return s;
    }

    if (sync) {
        if ((s = lv_sync_file(db->ternary_fd)) != LV_OK) {
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

    const LVCommit cmt = {
        .commit_seq = db->next_commit_seq,
        .first_seq = row_get_seq(lv_first_pending_row(db)),
        .reserved_seq = row_get_seq(lv_last_pending_row(db)),
        .total_count = lv_total_record_count(db),
        .batch_count = db->pending_count,
        .total_value_bytes = db->val_file_bytes + db->value_pending_bytes,  // fix db->row_bytes to db->val_bytes
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

    db->row_file_bytes += row_bytes_to_flush;
    db->ternary_file_bytes += ternary_bytes_to_flush;
    db->val_file_bytes += db->value_pending_bytes;

    db->pending_count = 0;
    db->value_pending_bytes = 0;

    return LV_OK;
}

int lv_pending_append(Livero *db, const LVRow *row, const LVTernaryChunk *ternary, const void *value,
                      const LVSize16_t value_len) {
    assert(db && row && ternary && value);

    if (db->pending_count >= db->pending_batch_m) {
        return -1;
    }

    const LVSize32_t ternary_bytes = ternary_chunks_bytes(db->vector_dimension);
    const LVOffset_t row_batch_offset = db->pending_count * LV_ROW_BYTES;
    const LVOffset_t ternary_batch_offset = db->pending_count * ternary_bytes;
    const LVOffset_t value_batch_offset = db->pending_count * db->max_value_len;

    row_write_on_memory(db->row_pending + row_batch_offset, row);
    memcpy((uint8_t *)db->ternary_pending + ternary_batch_offset, ternary, ternary_bytes);
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
    const LVSeq_t first_pending_seq = row_get_seq(lv_first_pending_row(db));

    if (target_seq < first_pending_seq || target_seq >= first_pending_seq + db->pending_count) {
        return LV_ERR_NOT_FOUND;
    }

    // const LVOffset_t row_offset = (target_seq - first_pending_seq) * LV_ROW_BYTES;
    // const LVOffset_t value_offset = (target_seq - first_pending_seq) * db->max_value_len;
    const LVCount_t idx = (target_seq - first_pending_seq);

    const uint8_t *row = lv_n_pending_row(db, idx);
    const uint8_t *value = lv_n_pending_value(db, idx);
    const int is_dead = !row_is_alive(row);

    if (is_dead) {
        return LV_ERR_DELETED;
    }

    if (result) {
        lv_fill_get_result(row, value, -1, result);
    }

    if (row_offset_out) {
        *row_offset_out = idx * LV_ROW_BYTES;
    }

    return LV_OK;
}

LVStatus lv_interpolation_search_from_disk(LVDiskSearchCtx *ctx, const LVSeq_t target_seq, LVGetResult *result,
                                           LVOffset_t *row_offset_out) {
    assert(ctx);

    uint8_t low[LV_ROW_BYTES] = {0};
    uint8_t high[LV_ROW_BYTES] = {0};

    const LVCount_t disk_record_count = ctx->disk_record_count;
    if (disk_record_count == 0) {
        return LV_ERR_NOT_FOUND;
    }

    const int low_idx = 0;
    const int high_idx = (int)(disk_record_count - 1);

    LVStatus s = lv_read_file(ctx->row_fd, low, LV_ROW_BYTES, (LVOffset_t)low_idx * LV_ROW_BYTES);
    if (s != LV_OK) {
        return s;
    }

    s = lv_read_file(ctx->row_fd, high, LV_ROW_BYTES, (LVOffset_t)high_idx * LV_ROW_BYTES);
    if (s != LV_OK) {
        return s;
    }

    const LVSeq_t low_seq = row_get_seq(low);
    const LVSeq_t high_seq = row_get_seq(high);

    if (low_seq == high_seq) {
        if (result && target_seq == low_seq) {
            if (!row_is_alive(low)) {
                return LV_ERR_DELETED;
            }
            if ((s = lv_fill_get_result(low, NULL, ctx->val_fd, result)) != LV_OK) {
                return s;
            }
        }
        return LV_ERR_NOT_FOUND;
    }

    if (target_seq < low_seq || target_seq > high_seq) {
        return LV_ERR_NOT_FOUND;
    }

    const int p_idx = low_idx + (int)(((target_seq - low_seq) * (high_idx - low_idx)) / (high_seq - low_seq));

    LVWindowRange r = lv_window_range(p_idx, low_idx, high_idx, (int)ctx->window_batch_m);

    int start_idx = r.start_idx;
    int end_idx = r.end_idx;

    LVOffset_t read_offset = (LVOffset_t)start_idx * LV_ROW_BYTES;
    LVCount_t read_count = (LVCount_t)(end_idx - start_idx + 1);

    if ((s = lv_read_file(ctx->row_fd, ctx->row_window, read_count * LV_ROW_BYTES, read_offset)) != LV_OK) {
        return s;
    }

    const LVOffset_t expected_offset = lv_window_target_offset(p_idx, r.start_idx, LV_ROW_BYTES);
    LVOffset_t row_offset = expected_offset;
    s = lv_window_search(ctx->row_window, ctx->val_fd, target_seq, read_count, expected_offset, result, &row_offset);

    if (s == LV_OK) {
        if (row_offset_out) {
            *row_offset_out = read_offset + row_offset;
        }
    }

    return s;
}

LVStatus lv_binary_search_from_disk(LVDiskSearchCtx *ctx, const LVSeq_t target_seq, LVGetResult *result,
                                    LVOffset_t *row_offset_out) {
    assert(ctx);

    const LVCount_t disk_record_count = ctx->disk_record_count;

    if (disk_record_count == 0) {
        return LV_ERR_NOT_FOUND;
    }

    int low = 0;
    int high = (int)(disk_record_count - 1);
    int mid = low + (high - low) / 2;

    while (low <= high) {
        LVWindowRange r = lv_window_range(mid, low, high, (int)ctx->window_batch_m);

        LVOffset_t read_offset = (LVOffset_t)r.start_idx * LV_ROW_BYTES;
        LVCount_t read_count = (LVCount_t)(r.end_idx - r.start_idx + 1);

        LVStatus s = lv_read_file(ctx->row_fd, ctx->row_window, read_count * LV_ROW_BYTES, read_offset);

        if (s != LV_OK) {
            return s;
        }

        const LVOffset_t expected_offset = lv_window_target_offset(mid, r.start_idx, LV_ROW_BYTES);
        const uint8_t *expected_row = ctx->row_window + expected_offset;
        const LVSeq_t expected_seq = row_get_seq(expected_row);
        LVOffset_t row_offset = expected_offset;

        s = lv_window_search(ctx->row_window, ctx->val_fd, target_seq, read_count, expected_offset, result,
                             &row_offset);

        if (s == LV_OK) {
            if (row_offset_out) {
                *row_offset_out = read_offset + row_offset;
            }

            return LV_OK;
        }

        else if (s == LV_ERR_NOT_FOUND) {
            if (expected_seq > target_seq) {
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

    // Center the window on p first.
    int start_idx = p - m / 2;
    int end_idx = start_idx + m - 1;

    // If the window runs past the right edge, slide it left so it ends at high.
    if (end_idx > high) {
        end_idx = high;
        start_idx = high - m + 1;
    }

    // If it runs past the left edge, slide it right so it starts at low.
    if (start_idx < low) {
        start_idx = low;
        end_idx = start_idx + m - 1;
        // The range may be shorter than m, so the end must stay inside it.
        if (end_idx > high) {
            end_idx = high;
        }
    }

    return (LVWindowRange){.start_idx = start_idx, .end_idx = end_idx};
}

LVStatus lv_window_search(const uint8_t *row_window, const int val_fd, const LVSeq_t target_seq, const LVCount_t count,
                          const LVOffset_t target_offset, LVGetResult *result, LVOffset_t *row_offset_out) {
    assert(row_window);

    const uint8_t *expected_row = row_window + target_offset;
    const LVSeq_t expected_seq = row_get_seq(expected_row);
    if (expected_seq == target_seq) {
        if (!row_is_alive(expected_row)) {
            return LV_ERR_DELETED;
        }
        if (result) {
            LVStatus s = lv_fill_get_result(expected_row, NULL, val_fd, result);
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
            const uint8_t *row = row_window + i * LV_ROW_BYTES;
            if (row_get_seq(row) == target_seq) {
                if (!row_is_alive(row)) {
                    return LV_ERR_DELETED;
                }
                if (result) {
                    LVStatus s = lv_fill_get_result(row, NULL, val_fd, result);
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

LVStatus lv_fill_get_result(const uint8_t *row, const void *value_src, const int val_fd, LVGetResult *result) {
    assert(row && result);
    const LVOffset_t value_offset = row_get_value_offset(row);
    const LVSize16_t value_len = row_get_value_len(row);
    if (result->value) {
        if (value_src) {
            memcpy(result->value, value_src, value_len);
        } else {
            LVStatus s = lv_read_file(val_fd, result->value, value_len, value_offset);
            if (s != LV_OK) {
                return s;
            }
        }
    }
    lv_convert_epoch_to_date(row_get_epoch_date(row), &result->date);
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
    if (!row_is_alive(ctx->row)) {
        return;
    }
    const int full = *size >= ctx->top_k;
    int passes_filter = 0;
    const uint64_t epoch_date = row_get_epoch_date(ctx->row);
    if (ctx->min_epoch_date <= epoch_date && epoch_date <= ctx->max_epoch_date) {
        passes_filter = 1;
    }
    if (passes_filter) {
        LVDate date;
        lv_convert_epoch_to_date(epoch_date, &date);
        const LVQueryResult q_result = {
            .date = date, .seq = row_get_seq(ctx->row), .value_len = row_get_value_len(ctx->row)};
        const LVQueryIndex index = {.dot_result = ctx->dot_result, .q_result = q_result};
        if (full) {
            lv_query_heap_pop(heap, size, NULL);
        }
        lv_query_heap_insert(heap, index, size);
    }
}
