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
    total += LV_ROW_DISK_BYTES * read_batch_m;
    total += LV_ROW_DISK_BYTES * write_batch_m;
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

    LVInitCtx ctx = {.next_seq = 0,
                     .next_commit_seq = 0,
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

    const LVSeq_t next_commit_seq = commit.commit_seq + 1;                            // starts at 1
    const LVSeq_t next_seq = commit.reserved_seq == 0 ? 0 : commit.reserved_seq + 1;  // starts at 0

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
        .row_bytes = LV_ROW_DISK_BYTES * commit.total_count,
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

    const LVSeq_t current_commit_seq = db->next_commit_seq;
    const LVSeq_t current_seq = db->next_seq;
    const LVCount_t current_record_count = db->row_bytes / LV_ROW_DISK_BYTES;

    const uint64_t epoch_time = lv_convert_date_to_epoch(date);
    const LVSize32_t vector_bytes = ternary_chunks_bytes(db->vector_dimension);
    const LVOffset_t row_disk_offset = current_record_count * LV_ROW_DISK_BYTES;
    const LVOffset_t vector_disk_offset = current_record_count * vector_bytes;
    const LVOffset_t value_disk_offset = db->val_bytes + db->batched_value_bytes;

    ternary_convert(vector, db->vector_type, db->vector_dimension, db->ternary_scratch);
    LVRowCalcChecksumCtx row_checksum_ctx = {.seq = current_seq,
                                             .date = epoch_time,
                                             .value_offset = value_disk_offset,
                                             .value_len = value_len,
                                             .vector_dim = db->vector_dimension,
                                             .value = value,
                                             .vector = vector};
    const uint32_t row_checksum = row_calc_checksum(&row_checksum_ctx);
    LVRow row_data = {.seq = current_seq,
                      .date = epoch_time,
                      .value_offset = value_disk_offset,
                      .value_len = value_len,
                      .tombstone_flag = 0,
                      .checksum = row_checksum};

    int flush = 0;

    LVSeq_t first_seq_to_commit = 0;
    LVSeq_t reserved_seq_to_commit = 0;
    LVCount_t total_count_to_commit = 0;
    LVCount_t batch_count_to_commit = 0;
    LVSize32_t total_value_bytes_to_commit = 0;
    LVSize32_t batch_value_bytes_to_commit = 0;

    uint8_t *row_buf_to_flush = NULL;
    uint8_t *vector_buf_to_flush = NULL;
    uint8_t *value_buf_to_flush = NULL;
    LVSize32_t row_bytes_to_flush = 0;
    LVSize32_t vector_bytes_to_flush = 0;
    LVSize32_t value_bytes_to_flush = 0;

    if (db->sync_mode == LV_SYNC_EVERY_PUT) {
        uint8_t row_buffer[LV_ROW_DISK_BYTES] = {0};
        row_write_on_memory(row_buffer, &row_data);

        flush = 1;

        first_seq_to_commit = current_seq;
        reserved_seq_to_commit = current_seq + 1;
        total_count_to_commit = current_record_count + 1;
        batch_count_to_commit = 1;
        total_value_bytes_to_commit = db->val_bytes + value_len;
        batch_value_bytes_to_commit = value_len;

        row_buf_to_flush = row_buffer;
        vector_buf_to_flush = (uint8_t *)db->ternary_scratch;
        value_buf_to_flush = value;
        row_bytes_to_flush = LV_ROW_DISK_BYTES;
        vector_bytes_to_flush = vector_bytes;
        value_bytes_to_flush = value_len;
    } else {
        if (db->batched_count >= db->write_batch_m) {
            flush = 1;

            first_seq_to_commit = row_get_seq(db->write_row_batch);
            reserved_seq_to_commit = current_seq + db->batched_count;
            total_count_to_commit = current_record_count + db->batched_count;
            batch_count_to_commit = db->batched_count;
            total_value_bytes_to_commit = db->val_bytes + db->batched_value_bytes;
            batch_value_bytes_to_commit = db->batched_value_bytes;

            row_buf_to_flush = db->write_value_batch;
            vector_buf_to_flush = (uint8_t *)db->write_vector_batch;
            value_buf_to_flush = db->write_value_batch;
            row_bytes_to_flush = LV_ROW_DISK_BYTES * db->batched_count;
            vector_bytes_to_flush = vector_bytes * db->batched_count;
            value_bytes_to_flush = db->batched_value_bytes;
        }
    }

    if (flush) {
        LVCommit commit = {
            .commit_seq = current_commit_seq,
            .first_seq = first_seq_to_commit,
            .reserved_seq = reserved_seq_to_commit,
            .total_count = total_count_to_commit,
            .batch_count = batch_count_to_commit,
            .total_value_bytes = total_value_bytes_to_commit,
            .batch_value_bytes = batch_value_bytes_to_commit,
        };
        LVFlushCtx flush_ctx = {.rows = row_buf_to_flush,
                                .vecs = vector_buf_to_flush,
                                .vals = value_buf_to_flush,
                                .commit = &commit,
                                .row_fd = db->row_fd,
                                .vec_fd = db->vec_fd,
                                .val_fd = db->val_fd,
                                .commit_fd = db->commit_fd,
                                .row_bytes_to_flush = row_bytes_to_flush,
                                .value_bytes_to_flush = value_bytes_to_flush,
                                .vector_bytes_to_flush = vector_bytes_to_flush,
                                .row_offset = row_disk_offset,
                                .vec_offset = vector_disk_offset,
                                .val_offset = value_disk_offset,
                                .sync = db->sync_mode != LV_SYNC_NONE};

        LVStatus s = lv_flush(db, &flush_ctx);

        if (s != LV_OK) {
            LIVERO_ERROR("Failed to flush data.");
            return s;
        }
    }

    if (db->batched_count < db->write_batch_m) {
        lv_batch(db, &row_data, db->ternary_scratch, value, value_len);
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
        LIVERO_ERROR("You must pass a result buffer.");
        return LV_ERR_INVALID_PARAM;
    }

    if (target_seq >= db->next_seq) {
        return LV_ERR_NOT_FOUND;
    }

    LVStatus s = lv_write_batch_search(db, target_seq, result);

    if (s == LV_OK) {
        return LV_OK;
    }

    s = lv_interpolation_search_from_disk(db, target_seq, result);

    if (s == LV_ERR_NOT_FOUND) {
        return lv_binary_search_from_disk(db, target_seq, result);
    } else {
        return s;
    }
}
/*
  Below functions are all internal apis.
*/

int lv_check_option(const LVOption *option) {
    assert(option);
    if (option->read_batch_m == 0 || option->write_batch_m == 0) {
        LIVERO_ERROR("Both read_batch_m and write_batch_m must be greater than 0.");
        return 0;
    }

    if (option->read_batch_m > LV_MAX_BATCH_M || option->write_batch_m > LV_MAX_BATCH_M) {
        LIVERO_ERROR("Both read_batch_mand write_batch_m must be smaller than %d .", LV_MAX_BATCH_M);
        return 0;
    }
    return 1;
}

LVStatus lv_allocate_memory(const LVSize32_t size, void **ptr_out) {
    assert(ptr_out);

    void *m = aligned_alloc(LV_MEMORY_ALIGNMENT, size);

    if (!m) {
        return LV_ERR_ALLOCATE_MEMORY;
    }

    *ptr_out = m;

    return LV_OK;
}

void lv_init_memory(uint8_t *m_arena, const LVInitCtx *ctx) {
    assert(m_arena && ctx);

    LVOffset_t offset = 0;
    const LVSize32_t vector_bytes = ternary_chunks_bytes(ctx->vector_dimension);

    Livero *db = (Livero *)m_arena;
    offset += sizeof(Livero);

    LVTernaryChunk *ternary_scratch = (LVTernaryChunk *)(m_arena + offset);
    offset += vector_bytes;

    LVTernaryChunk *read_vector_batch = (LVTernaryChunk *)(m_arena + offset);
    offset += vector_bytes * ctx->read_batch_m;

    LVTernaryChunk *write_vector_batch = (LVTernaryChunk *)(m_arena + offset);
    offset += vector_bytes * ctx->write_batch_m;

    uint8_t *read_row_batch = m_arena + offset;
    offset += LV_ROW_DISK_BYTES * ctx->read_batch_m;

    uint8_t *write_row_batch = (m_arena + offset);
    offset += LV_ROW_DISK_BYTES * ctx->write_batch_m;

    uint8_t *write_value_batch = m_arena + offset;

    db->next_seq = ctx->next_seq;
    db->next_commit_seq = ctx->next_commit_seq;
    db->ternary_scratch = ternary_scratch;
    db->read_vector_batch = read_vector_batch;
    db->write_vector_batch = write_vector_batch;
    db->read_row_batch = read_row_batch;
    db->write_row_batch = write_row_batch;
    db->write_value_batch = write_value_batch;
    db->row_fd = ctx->row_fd;
    db->vec_fd = ctx->vec_fd;
    db->val_fd = ctx->val_fd;
    db->commit_fd = ctx->commit_fd;
    db->row_bytes = ctx->row_bytes;
    db->vec_bytes = ctx->vec_bytes;
    db->val_bytes = ctx->val_bytes;
    db->max_disk_bytes = ctx->max_disk_bytes;
    db->batched_count = 0;
    db->read_batch_m = ctx->read_batch_m;
    db->write_batch_m = ctx->write_batch_m;
    db->max_value_len = ctx->max_value_len;
    db->batched_value_bytes = 0;
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
        rccfd.row_offset = k * LV_ROW_DISK_BYTES;
        rccfd.vec_offset = k * vector_bytes;

        LVStatus s = row_checksum_check_from_disk(&rccfd);
        if (s != LV_OK) {
            return s;
        }
    }

    *commit_out = best;
    return LV_OK;
}

LVStatus lv_flush(Livero *db, const LVFlushCtx *ctx) {
    assert(db && ctx);

    LVStatus s = lv_write_file(ctx->vec_fd, ctx->vecs, ctx->vector_bytes_to_flush, ctx->vec_offset);

    if (s != LV_OK) {
        LIVERO_ERROR("Failed to flush vector (%d bytes).", ctx->vector_bytes_to_flush);
        return s;
    }

    if ((s = lv_write_file(ctx->val_fd, ctx->vals, ctx->value_bytes_to_flush, ctx->val_offset)) != LV_OK) {
        LIVERO_ERROR("Failed to flush value (%d bytes).", ctx->value_bytes_to_flush);
        return s;
    }

    if ((s = lv_write_file(ctx->row_fd, ctx->rows, ctx->row_bytes_to_flush, ctx->row_offset)) != LV_OK) {
        LIVERO_ERROR("Failed to flush row (%d bytes).", ctx->row_bytes_to_flush);
        return s;
    }

    if (ctx->sync) {
        if ((s = lv_sync_file(ctx->vec_fd)) != LV_OK) {
            LIVERO_ERROR("Failed to sync vec.lv.");
            return s;
        }

        if ((s = lv_sync_file(ctx->val_fd)) != LV_OK) {
            LIVERO_ERROR("Failed to sync val.lv.");
            return s;
        }

        if ((s = lv_sync_file(ctx->row_fd)) != LV_OK) {
            LIVERO_ERROR("Failed to sync row.lv.");
            return s;
        }
    }

    if ((s = commit_write(ctx->commit_fd, ctx->commit)) != LV_OK) {
        LIVERO_ERROR("Faield to write a commit on commit.lv");
        return s;
    }

    if (ctx->sync) {
        if ((s = lv_sync_file(ctx->commit_fd)) != LV_OK) {
            LIVERO_ERROR("Failed to sync commit.lv.");
            return s;
        }
    }

    db->next_commit_seq = ctx->commit->commit_seq + 1;
    db->row_bytes += ctx->row_bytes_to_flush;
    db->vec_bytes += ctx->vector_bytes_to_flush;
    db->val_bytes += ctx->value_bytes_to_flush;

    db->batched_count = 0;
    db->batched_value_bytes = 0;

    return LV_OK;
}

int lv_batch(Livero *db, const LVRow *row, const void *ternary_vector, const void *value, const LVSize16_t value_len) {
    assert(db && row && ternary_vector && value);

    if (db->batched_count >= db->write_batch_m) {
        return -1;
    }

    const LVSize32_t vector_bytes = ternary_chunks_bytes(db->vector_dimension);
    const LVOffset_t row_batch_offset = db->batched_count * LV_ROW_DISK_BYTES;
    const LVOffset_t vector_batch_offset = db->batched_count * vector_bytes;
    const LVOffset_t value_batch_offset = db->batched_count * db->max_value_len;

    row_write_on_memory(db->write_row_batch + row_batch_offset, row);
    memcpy((uint8_t *)db->write_vector_batch + vector_batch_offset, ternary_vector, vector_bytes);
    memcpy(db->write_value_batch + value_batch_offset, value, value_len);

    db->batched_count += 1;
    db->batched_value_bytes += value_len;

    return 0;
}

LVStatus lv_write_batch_search(const Livero *db, const LVSeq_t target_seq, LVGetResult *result) {
    assert(db && result);

    if (db->batched_count == 0) {
        return LV_ERR_NOT_FOUND;
    }
    const LVSeq_t first_batched_seq = row_get_seq(db->write_row_batch);

    if (target_seq < first_batched_seq || target_seq >= first_batched_seq + db->batched_count) {
        return LV_ERR_NOT_FOUND;
    }

    const LVOffset_t row_offset = (target_seq - first_batched_seq) * LV_ROW_DISK_BYTES;
    const LVOffset_t value_offset = (target_seq - first_batched_seq) * db->max_value_len;

    const uint8_t *row = db->write_row_batch + row_offset;
    const uint8_t *value = db->write_value_batch + value_offset;

    return lv_fill_get_result(row, value, -1, result);
}

LVStatus lv_interpolation_search_from_disk(Livero *db, const LVSeq_t target_seq, LVGetResult *result) {
    assert(db && result);

    uint8_t low[LV_ROW_DISK_BYTES] = {0};
    uint8_t high[LV_ROW_DISK_BYTES] = {0};

    const LVCount_t total_record_count = db->row_bytes / LV_ROW_DISK_BYTES;
    if (total_record_count == 0) {
        return LV_ERR_NOT_FOUND;
    }

    const int low_idx = 0;
    const int high_idx = (int)(total_record_count - 1);

    LVStatus s = lv_read_file(db->row_fd, low, LV_ROW_DISK_BYTES, (LVOffset_t)low_idx * LV_ROW_DISK_BYTES);
    if (s != LV_OK) {
        return s;
    }

    s = lv_read_file(db->row_fd, high, LV_ROW_DISK_BYTES, (LVOffset_t)high_idx * LV_ROW_DISK_BYTES);
    if (s != LV_OK) {
        return s;
    }

    const LVSeq_t low_seq = row_get_seq(low);
    const LVSeq_t high_seq = row_get_seq(high);

    if (low_seq == high_seq) {
        if (target_seq == low_seq) {
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

    LVBatchReadRange read_range = lv_read_batch_range(p_idx, low_idx, high_idx, (int)db->read_batch_m);

    int start_idx = read_range.start_idx;
    int end_idx = read_range.end_idx;

    LVOffset_t read_offset = (LVOffset_t)start_idx * LV_ROW_DISK_BYTES;
    LVCount_t read_count = (LVCount_t)(end_idx - start_idx + 1);

    if ((s = lv_read_file(db->row_fd, db->read_row_batch, read_count * LV_ROW_DISK_BYTES, read_offset)) != LV_OK) {
        return s;
    }

    return lv_read_batch_search(db, target_seq, read_range.align, read_count, result);
}

LVStatus lv_binary_search_from_disk(Livero *db, const LVSeq_t target_seq, LVGetResult *result) {
    assert(db && result);

    const LVCount_t total_record_count = db->row_bytes / LV_ROW_DISK_BYTES;

    if (total_record_count == 0) {
        return LV_ERR_NOT_FOUND;
    }

    int low = 0;
    int high = (int)(total_record_count - 1);
    int mid = low + (high - low) / 2;

    while (low <= high) {
        LVBatchReadRange r = lv_read_batch_range(mid, low, high, (int)db->read_batch_m);

        LVOffset_t read_offset = (LVOffset_t)r.start_idx * LV_ROW_DISK_BYTES;
        LVCount_t read_count = (LVCount_t)(r.end_idx - r.start_idx + 1);

        LVStatus s = lv_read_file(db->row_fd, db->read_row_batch, read_count * LV_ROW_DISK_BYTES, read_offset);

        if (s != LV_OK) {
            return s;
        }

        const uint8_t *expected_row = lv_read_batch_expected_row(db, r.align, read_count);
        const LVSeq_t expected_seq = row_get_seq(expected_row);

        s = lv_read_batch_search(db, target_seq, r.align, read_count, result);

        if (s != LV_ERR_NOT_FOUND) {
            /*
                The answer is found or An error occurred.
            */
            return s;
        }

        else {
            if (expected_seq < target_seq) {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }

        mid = low + (high - low) / 2;
    }
    return LV_ERR_NOT_FOUND;
}

LVBatchReadRange lv_read_batch_range(const int p, const int low, const int high, const int m) {
    const int read_m = m;
    const int half_m = read_m / 2;

    int start_idx = 0;
    int end_idx = 0;
    LVBatchAlign align = LV_BATCH_CENTER;

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

    return (LVBatchReadRange){.start_idx = start_idx, .end_idx = end_idx, .align = align};
}

const uint8_t *lv_read_batch_expected_row(const Livero *db, const LVBatchAlign align, const LVCount_t batched_count) {
    LVOffset_t offset = 0;
    switch (align) {
        case LV_BATCH_CENTER: {
            offset = batched_count / 2;
            break;
        }

        case LV_BATCH_LEFT: {
            offset = 0;
            break;
        }
        case LV_BATCH_RIGHT: {
            offset = batched_count - 1;
            break;
        }
    }
    return db->read_row_batch + offset * LV_ROW_DISK_BYTES;
}

LVStatus lv_read_batch_search(const Livero *db, const LVSeq_t target, const LVBatchAlign align, const LVCount_t count,
                              LVGetResult *result) {
    assert(db && result && count > 0);

    const uint8_t *expected_row = lv_read_batch_expected_row(db, align, count);
    const LVSeq_t expected_seq = row_get_seq(expected_row);
    if (expected_seq == target) {
        return lv_fill_get_result(expected_row, NULL, db->val_fd, result);
    } else {
        for (LVCount_t i = 0; i < count; ++i) {
            const uint8_t *row = db->read_row_batch + i * LV_ROW_DISK_BYTES;
            if (row_get_seq(row) == target) {
                return lv_fill_get_result(row, NULL, db->val_fd, result);
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
    if (result->value_out) {
        if (value) {
            memcpy(result->value_out, value, value_len);
        } else {
            LVStatus s = lv_read_file(val_fd, result->value_out, value_len, value_offset);
            if (s != LV_OK) {
                return s;
            }
        }
    }
    lv_convert_epoch_to_date(row_get_date(row), &result->date_out);
    result->value_len_out = value_len;
    return LV_OK;
}
