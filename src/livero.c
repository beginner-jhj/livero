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
#include "struct/query_heap.h"
#include "struct/row.h"
#include "vector/ternary.h"

/*************** lv_create, lv_open and relevant internal functions (memory allocation/initialization, flash data
 * recover) section. ******************/

LVStatus lv_create(const char *db_path, const LVConfig *config, const LVOption *option, Livero **db_out) {
    LIVERO_CHECK_MISSING(db_path, "db_path");

    LIVERO_CHECK_MISSING(db_out, "db_out");

    LIVERO_CHECK_MISSING(config, "config");

    LIVERO_CHECK_GT(config->max_disk_bytes, "max_disk_bytes", "%u", LV_BASELINE_DISK_USAGE);

    LIVERO_CHECK_GT(config->max_value_len, "max_value_len", "%d", 0);

    if (config->vector_dimension == 0 || config->vector_dimension % LV_DIMENSION_MULTIPLE != 0) {
        LIVERO_ERROR("vector_dimension must be a multiple of %d.", LV_DIMENSION_MULTIPLE);
        return LV_ERR_INVALID_PARAM;
    }

    LIVERO_CHECK_MISSING(option, "option");

    if (!lv_check_option(option)) {
        return LV_ERR_INVALID_PARAM;
    }

    void *m_arena = NULL;
    uint8_t config_file_def[option->file_def_size];
    LVStatus s =
        lv_file_open_single(db_path, 1, LV_CONFIG_FILE_NAME, (LVFILEDef *)config_file_def, option->io_ops->open);
    if (s != LV_OK) {
        goto cleanup;
    }

    if ((s = lv_save_config((LVFILEDef *)config_file_def, config, option->io_ops->write, option->io_ops->sync)) !=
        LV_OK) {
        LIVERO_ERROR("Failed to save the config.");
        goto cleanup;
    }

    if ((s = option->io_ops->close((LVFILEDef *)config_file_def)) != LV_OK) {
        LIVERO_ERROR("Failed to close config.lv.");
        goto cleanup;
    }

    const LVSize32_t total_meory_size = lv_total_memory_bytes(config, option);
    s = lv_allocate_memory(total_meory_size, &m_arena);
    if (s != LV_OK) {
        LIVERO_ERROR("Failed to allocate %d bytes.", total_meory_size);
        goto cleanup;
    }

    lv_init_memory(m_arena, db_out, config, option);

    LVFileDefList file_list = {
        .row_file = (*db_out)->row_file,
        .ternary_file = (*db_out)->ter_file,
        .value_file = (*db_out)->val_file,
        .commit_file = (*db_out)->commit_file,
    };

    if ((s = lv_file_open_all(db_path, 1, &file_list, option->io_ops->open)) != LV_OK) {
        goto cleanup;
    }

    if ((s = commit_init((*db_out)->commit_file, (*db_out)->commit_sector, option->io_ops->write,
                         option->io_ops->sync)) != LV_OK) {
        LIVERO_ERROR("Failed to initialize commit.lv.");
        goto cleanup;
    }

    LVCommit empty_commit = {0};
    empty_commit.commit_seq = LV_COMMIT_SEQ_ZERO;
    lv_init_db_seq_and_file_stat(*db_out, &empty_commit, 0);

    return LV_OK;

cleanup:
    lv_close(*db_out);
    return s;
}

LVStatus lv_open(const char *db_path, const LVOption *option, Livero **db_out) {
    LIVERO_CHECK_MISSING(db_path, "db_path");

    LIVERO_CHECK_MISSING(db_out, "db_out");

    LIVERO_CHECK_MISSING(option, "option");

    if (!lv_check_option(option)) {
        return LV_ERR_INVALID_PARAM;
    }

    void *m_arena = NULL;

    LVConfig config;

    uint8_t config_file_def[option->file_def_size];
    LVStatus s =
        lv_file_open_single(db_path, 0, LV_CONFIG_FILE_NAME, (LVFILEDef *)config_file_def, option->io_ops->open);
    if (s != LV_OK) {
        goto cleanup;
    }

    if ((s = lv_config_d2m((LVFILEDef *)config_file_def, option->io_ops->read, &config)) != LV_OK) {
        LIVERO_ERROR("Failed to read the config.");
        goto cleanup;
    }

    if ((s = option->io_ops->close((LVFILEDef *)config_file_def)) != LV_OK) {
        LIVERO_ERROR("Failed to close config.lv.");
        goto cleanup;
    }

    const LVSize32_t total_memory_size = lv_total_memory_bytes(&config, option);
    if ((s = lv_allocate_memory(total_memory_size, &m_arena)) != LV_OK) {
        LIVERO_ERROR("Failed to allocate %d bytes.", total_memory_size);
        goto cleanup;
    }

    lv_init_memory(m_arena, db_out, &config, option);

    LVFileDefList file_list = {
        .row_file = (*db_out)->row_file,
        .ternary_file = (*db_out)->ter_file,
        .value_file = (*db_out)->val_file,
        .commit_file = (*db_out)->commit_file,
    };

    if ((s = lv_file_open_all(db_path, 0, &file_list, option->io_ops->open)) != LV_OK) {
        goto cleanup;
    }

    LVCommit commit;
    LVRecoverCtx recover_ctx = {
        .row_file = (*db_out)->row_file,
        .ter_file = (*db_out)->ter_file,
        .val_file = (*db_out)->val_file,
        .commit_file = (*db_out)->commit_file,
        .read = option->io_ops->read,
        .file_def_size = option->file_def_size,
        .max_disk_bytes = config.max_disk_bytes,
        .vector_dimension = config.vector_dimension,
    };
    int first_commit_torn = 0;
    if ((s = lv_recover(&recover_ctx, &commit, &first_commit_torn)) != LV_OK) {
        LIVERO_ERROR("Failed to recover the db.");
        goto cleanup;
    }

    lv_init_db_seq_and_file_stat(*db_out, &commit, first_commit_torn);

    return LV_OK;

cleanup:
    lv_close(*db_out);
    return s;
}

LVStatus lv_allocate_memory(const LVSize32_t size, void **ptr_out) {
    assert(ptr_out);

    const LVSize32_t rounded =
        (size + LV_MEMORY_ALIGNMENT - 1) &
        ~(LV_MEMORY_ALIGNMENT -
          1);  // Critical! rounded was 16 bytes so it easily eceeds total size when the max value len is big
    void *m = aligned_alloc(LV_MEMORY_ALIGNMENT, rounded);

    if (!m) {
        return LV_ERR_ALLOCATE_MEMORY;
    }

    *ptr_out = m;

    return LV_OK;
}

void lv_init_memory(uint8_t *m_arena, Livero **db_out, const LVConfig *config, const LVOption *option) {
    assert(m_arena && config && option);
    assert((uintptr_t)m_arena % LV_MEMORY_ALIGNMENT == 0);

    LVOffset_t offset = 0;
    const LVSize32_t ternary_bytes = ternary_chunks_bytes(config->vector_dimension);

    Livero *db = (Livero *)m_arena;
    offset += sizeof(Livero);

    LVTernaryChunk *ternary_scratch = (LVTernaryChunk *)(m_arena + offset);
    offset += ternary_bytes;

    LVTernaryChunk *ternary_window = (LVTernaryChunk *)(m_arena + offset);
    offset += ternary_bytes * option->read_batch_m;

    LVTernaryChunk *ternary_pending = (LVTernaryChunk *)(m_arena + offset);
    offset += ternary_bytes * option->write_batch_m;

    LVQueryHeapEntry *heap_entries = (LVQueryHeapEntry *)(m_arena + offset);
    offset += sizeof(LVQueryHeapEntry) * option->max_top_k * option->rerank_factor;

    LVFILEDef *row_file = (LVFILEDef *)(m_arena + offset);
    offset += option->file_def_size;

    LVFILEDef *ter_file = (LVFILEDef *)(m_arena + offset);
    offset += option->file_def_size;

    LVFILEDef *val_file = (LVFILEDef *)(m_arena + offset);
    offset += option->file_def_size;

    LVFILEDef *commit_file = (LVFILEDef *)(m_arena + offset);
    offset += option->file_def_size;

    uint8_t *row_window = m_arena + offset;
    offset += LV_ROW_BYTES * option->read_batch_m;

    uint8_t *row_pending = (m_arena + offset);
    offset += LV_ROW_BYTES * option->write_batch_m;

    uint8_t *value_pending = m_arena + offset;
    offset += config->max_value_len * option->write_batch_m;

    uint8_t *commit_sector = m_arena + offset;

    db->next_seq = LV_SEQ_START;
    db->next_commit_seq = LV_COMMIT_SEQ_START;
    db->ternary_scratch = ternary_scratch;
    db->ternary_window = ternary_window;
    db->ternary_pending = ternary_pending;
    db->heap_entries = heap_entries;
    db->row_file = row_file;
    db->ter_file = ter_file;
    db->val_file = val_file;
    db->commit_file = commit_file;
    db->row_window = row_window;
    db->row_pending = row_pending;
    db->value_pending = value_pending;
    db->commit_sector = commit_sector;
    db->option = *option;
    db->config = *config;
    db->pending_count = 0;
    db->value_pending_bytes = 0;
    db->row_file_bytes = 0;
    db->ternary_file_bytes = 0;
    db->value_file_bytes = 0;

    *db_out = (Livero *)m_arena;
}

void lv_init_db_seq_and_file_stat(Livero *db, const LVCommit *commit, const int is_first_torn) {
    assert(db && commit);

    const int is_empty_db = commit->commit_seq == LV_COMMIT_SEQ_ZERO;
    LVSeq_t next_commit_seq;
    LVSeq_t next_seq;

    if (is_empty_db) {
        if (is_first_torn) {
            next_commit_seq = LV_COMMIT_SEQ_START;
            next_seq = LV_SEQ_START + LV_MAX_BATCH_M;
        } else {
            next_commit_seq = LV_COMMIT_SEQ_START;
            next_seq = LV_SEQ_START;
        }

    } else {
        next_commit_seq = commit->commit_seq + 1;
        next_seq = commit->reserved_seq + 1;
    }

    if (is_empty_db) {
        db->row_file_bytes = 0;
        db->ternary_file_bytes = 0;
        db->value_file_bytes = 0;
    } else {
        db->row_file_bytes = commit->total_count * LV_ROW_BYTES;
        db->ternary_file_bytes = commit->total_count * ternary_chunks_bytes(db->config.vector_dimension);
        db->value_file_bytes = commit->total_value_bytes;
    }

    db->next_commit_seq = next_commit_seq;
    db->next_seq = next_seq;
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
        LVCommitStatus cmt_status = commit_d2m(ctx->commit_file, ctx->read, i * LV_COMMIT_SECTOR_BYTES, &cmt);

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
    LVRowCCFDCtx rccfd = {.row_file = ctx->row_file,
                          .ter_file = ctx->ter_file,
                          .val_file = ctx->val_file,
                          .read = ctx->read,
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

/************************************************************************************************/

/*************** lv_put and relevant internal functions (pending append, flush) section. ******************/

LVStatus lv_put(Livero *db, void *vector, void *value, const LVSize16_t value_len, const LVDate *date,
                LVSeq_t *seq_out) {
    LIVERO_CHECK_MISSING(db, "Db");

    LIVERO_CHECK_MISSING(vector, "vector");

    LIVERO_CHECK_MISSING(value, "value");

    LIVERO_CHECK_GT(value_len, "value_len", "%d", 0);

    LIVERO_CHECK_LT(value_len, "value_len", "%u", db->config.max_value_len + 1);

    if (date && !lv_date_is_valid(date)) {
        LIVERO_ERROR("You passed an invalid date object.");
        return LV_ERR_INVALID_PARAM;
    }

    if (lv_is_disk_full(db, value_len)) {
        LIVERO_ERROR("Disk is full");
        return LV_ERR_DISK_FULL;
    }

    const LVSeq_t current_seq = db->next_seq;
    const uint64_t epoch_date = lv_convert_date_to_epoch(date);
    const LVOffset_t value_disk_offset = db->value_file_bytes;
    const LVOffset_t value_pending_offset = db->value_pending_bytes;
    const LVOffset_t value_offset =
        value_disk_offset + value_pending_offset;  // fix offset bug: value_offset should be disk + pending bytes

    ternary_convert(vector, db->config.vector_type, db->config.vector_dimension, db->ternary_scratch);
    LVRowCalcChecksumCtx row_checksum_ctx = {.seq = current_seq,
                                             .epoch_date = epoch_date,
                                             .value_offset = value_offset,
                                             .value_len = value_len,
                                             .vector_dimension = db->config.vector_dimension,
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

    if (pending_result < 0 || db->option.sync_mode == LV_SYNC_EVERY_PUT) {
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

int lv_pending_append(Livero *db, const LVRow *row, const LVTernaryChunk *ternary, const void *value,
                      const LVSize16_t value_len) {
    assert(db && row && ternary && value);

    if (db->pending_count >= db->option.write_batch_m) {
        return -1;
    }

    const LVSize32_t ternary_bytes = ternary_chunks_bytes(db->config.vector_dimension);
    const LVOffset_t row_batch_offset = db->pending_count * LV_ROW_BYTES;
    const LVOffset_t ternary_batch_offset = db->pending_count * ternary_bytes;
    const LVOffset_t value_batch_offset = db->pending_count * db->config.max_value_len;

    row_write_on_memory(db->row_pending + row_batch_offset, row);
    memcpy((uint8_t *)db->ternary_pending + ternary_batch_offset, ternary, ternary_bytes);
    memcpy(db->value_pending + value_batch_offset, value, value_len);

    db->pending_count += 1;
    db->value_pending_bytes += value_len;

    return 0;
}

LVStatus lv_flush(Livero *db) {
    assert(db);

    const LVSize32_t ternary_bytes = ternary_chunks_bytes(db->config.vector_dimension);
    const LVOffset_t row_offset_to_write = db->row_file_bytes;
    const LVOffset_t ternary_offset_to_write = db->ternary_file_bytes;
    const LVOffset_t val_offset_to_write = db->value_file_bytes;
    const int sync = db->option.sync_mode != LV_SYNC_NONE;

    LVSize32_t packed = 0;
    for (LVCount_t i = 0; i < db->pending_count; ++i) {
        const LVSize16_t len = row_get_value_len(lv_n_pending_row(db, i));
        memmove(db->value_pending + packed, lv_n_pending_value(db, i), len);
        packed += len;
    }

    LVStatus s = db->option.io_ops->write(db->val_file, db->value_pending, packed, val_offset_to_write);

    if (s != LV_OK) {
        return s;
    }

    if (sync) {
        if ((s = db->option.io_ops->sync(db->val_file)) != LV_OK) {
            LIVERO_ERROR("Faield to sync val.lv.");
            return s;
        }
    }

    const LVSize32_t ternary_bytes_to_flush = db->pending_count * ternary_bytes;
    if ((s = db->option.io_ops->write(db->ter_file, db->ternary_pending, ternary_bytes_to_flush,
                                      ternary_offset_to_write)) != LV_OK) {
        return s;
    }

    if (sync) {
        if ((s = db->option.io_ops->sync(db->ter_file)) != LV_OK) {
            LIVERO_ERROR("Failed to sync vec.lv.");
            return s;
        }
    }

    const LVSize32_t row_bytes_to_flush = db->pending_count * LV_ROW_BYTES;

    if ((s = db->option.io_ops->write(db->row_file, db->row_pending, row_bytes_to_flush, row_offset_to_write)) !=
        LV_OK) {
        return s;
    }

    if (sync) {
        if ((s = db->option.io_ops->sync(db->row_file)) != LV_OK) {
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
        .total_value_bytes = db->value_file_bytes + db->value_pending_bytes,  // fix db->row_bytes to db->val_bytes
        .batch_value_bytes = db->value_pending_bytes};

    if ((s = commit_write(db->commit_file, db->commit_sector, db->option.io_ops->write, &cmt)) != LV_OK) {
        return s;
    }

    if (sync) {
        if ((s = db->option.io_ops->sync(db->commit_file)) != LV_OK) {
            LIVERO_ERROR("Failed to sync commit.lv.");
            return s;
        }
    }

    db->row_file_bytes += row_bytes_to_flush;
    db->ternary_file_bytes += ternary_bytes_to_flush;
    db->value_file_bytes += db->value_pending_bytes;

    db->pending_count = 0;
    db->value_pending_bytes = 0;

    return LV_OK;
}

/************************************************************************************************/

/*************** lv_get, lv_delete and relevant internal functions (pending search, disk search, window search) section.
 * ******************/

LVStatus lv_get(Livero *db, const LVSeq_t target_seq, LVGetResult *result) {
    LIVERO_CHECK_MISSING(db, "Db");

    LIVERO_CHECK_MISSING(result, "Get result");

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
    LIVERO_CHECK_MISSING(db, "Db");

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
            return db->option.io_ops->write(db->row_file, &tombstone, 1, row_offset + LV_ROW_TOMBSTONEFLAG_OFFSET);
        }
        return s;
    } else {
        return s;
    }
}

LVStatus lv_fill_get_result(LVFillGetResultCtx *ctx, LVGetResult *result) {
    assert(ctx && result);
    const LVSize16_t value_len = row_get_value_len(ctx->row);
    const LVOffset_t value_offset = row_get_value_offset(ctx->row);
    if (result->value) {
        assert(ctx->value_memory_src || ctx->value_file);
        if (ctx->value_memory_src) {
            memcpy(result->value, ctx->value_memory_src, value_len);

        } else {
            assert(ctx->read);
            LVStatus s = ctx->read(ctx->value_file, result->value, value_len, value_offset);
            if (s != LV_OK) {
                return s;
            }
        }
    }
    lv_convert_epoch_to_date(row_get_epoch_date(ctx->row), &result->date);
    result->value_len = value_len;
    return LV_OK;
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
        LVFillGetResultCtx get_result_ctx = {
            .row = (uint8_t *)row,
            .value_memory_src = (uint8_t *)value,
            .value_file = NULL,
            .read = NULL,
        };
        lv_fill_get_result(&get_result_ctx, result);
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

    LVStatus s = ctx->read(ctx->row_file, low, LV_ROW_BYTES, (LVOffset_t)low_idx * LV_ROW_BYTES);
    if (s != LV_OK) {
        return s;
    }

    s = ctx->read(ctx->row_file, high, LV_ROW_BYTES, (LVOffset_t)high_idx * LV_ROW_BYTES);
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

            LVFillGetResultCtx get_result_ctx = {
                .row = low,
                .value_memory_src = NULL,
                .value_file = ctx->val_file,
                .read = ctx->read,
            };
            if ((s = lv_fill_get_result(&get_result_ctx, result)) != LV_OK) {
                return s;
            }
        }
        return LV_ERR_NOT_FOUND;
    }

    if (target_seq < low_seq || target_seq > high_seq) {
        return LV_ERR_NOT_FOUND;
    }

    const int p_idx = low_idx + (int)(((target_seq - low_seq) * (high_idx - low_idx)) / (high_seq - low_seq));

    LVWindowRange r = lv_window_range(p_idx, low_idx, high_idx, (int)ctx->read_batch_m);

    int start_idx = r.start_idx;
    int end_idx = r.end_idx;

    LVOffset_t read_offset = (LVOffset_t)start_idx * LV_ROW_BYTES;
    LVCount_t read_count = (LVCount_t)(end_idx - start_idx + 1);

    if ((s = ctx->read(ctx->row_file, ctx->row_window, read_count * LV_ROW_BYTES, read_offset)) != LV_OK) {
        return s;
    }

    const LVOffset_t expected_offset = lv_window_target_offset(p_idx, r.start_idx, LV_ROW_BYTES);
    LVOffset_t row_offset = expected_offset;
    LVWindowSearchCtx window_search_ctx = {
        .row_window = ctx->row_window,
        .value_file = ctx->val_file,
        .target_offset = expected_offset,
        .target_seq = target_seq,
        .count_on_window = read_count,
        .read = ctx->read,
    };
    s = lv_window_search(&window_search_ctx, result, &row_offset);

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
        LVWindowRange r = lv_window_range(mid, low, high, (int)ctx->read_batch_m);

        LVOffset_t read_offset = (LVOffset_t)r.start_idx * LV_ROW_BYTES;
        LVCount_t read_count = (LVCount_t)(r.end_idx - r.start_idx + 1);

        LVStatus s = ctx->read(ctx->row_file, ctx->row_window, read_count * LV_ROW_BYTES, read_offset);

        if (s != LV_OK) {
            return s;
        }

        const LVOffset_t expected_offset = lv_window_target_offset(mid, r.start_idx, LV_ROW_BYTES);
        const uint8_t *expected_row = ctx->row_window + expected_offset;
        const LVSeq_t expected_seq = row_get_seq(expected_row);
        LVOffset_t row_offset = expected_offset;

        LVWindowSearchCtx window_search_ctx = {.row_window = ctx->row_window,
                                               .value_file = ctx->val_file,
                                               .target_offset = expected_offset,
                                               .target_seq = target_seq,
                                               .count_on_window = read_count,
                                               .read = ctx->read};
        s = lv_window_search(&window_search_ctx, result, &row_offset);

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

LVStatus lv_window_search(LVWindowSearchCtx *ctx, LVGetResult *result, LVOffset_t *row_offset_out) {
    assert(ctx);

    const uint8_t *expected_row = ctx->row_window + ctx->target_offset;
    const LVSeq_t expected_seq = row_get_seq(expected_row);
    if (expected_seq == ctx->target_seq) {
        if (!row_is_alive(expected_row)) {
            return LV_ERR_DELETED;
        }
        if (result) {
            LVFillGetResultCtx get_result_ctx = {
                .row = expected_row,
                .value_memory_src = NULL,
                .value_file = ctx->value_file,
                .read = ctx->read,
            };
            LVStatus s = lv_fill_get_result(&get_result_ctx, result);
            if (s != LV_OK) {
                return s;
            }
        }

        if (row_offset_out) {
            *row_offset_out = ctx->target_offset;
        }
        return LV_OK;
    } else {
        for (LVCount_t i = 0; i < ctx->count_on_window; ++i) {
            const uint8_t *row = ctx->row_window + i * LV_ROW_BYTES;
            if (row_get_seq(row) == ctx->target_seq) {
                if (!row_is_alive(row)) {
                    return LV_ERR_DELETED;
                }
                if (result) {
                    LVFillGetResultCtx get_result_ctx = {
                        .row = row,
                        .value_memory_src = NULL,
                        .value_file = ctx->value_file,
                        .read = ctx->read,
                    };
                    LVStatus s = lv_fill_get_result(&get_result_ctx, result);
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

/************************************************************************************************/

/*************** lv_query and relevant internal functions (heap helpers) section. *******************/

LVStatus lv_query(Livero *db, const LVTopK_t k, const void *query_vector, const LVDate *date1, const LVDate *date2,
                  LVQueryResult *result, LVSize32_t *result_size) {
    LIVERO_CHECK_MISSING(db, "Db");

    LIVERO_CHECK_MISSING(query_vector, "query_vector");

    LIVERO_CHECK_GT(k, "k", "%d", 0);

    LIVERO_CHECK_LT(k, "k", "%u", (LVCount_t)db->option.max_top_k + 1);

    LIVERO_CHECK_MISSING(result, "Query result");

    LIVERO_CHECK_MISSING(result_size, "Query result_size");

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

    const LVSize32_t ternary_bytes = ternary_chunks_bytes(db->config.vector_dimension);

    memset(db->ternary_scratch, 0, ternary_bytes);
    ternary_convert(query_vector, db->config.vector_type, db->config.vector_dimension, db->ternary_scratch);

    const LVTernaryChunk *query_ternary = db->ternary_scratch;
    const void *query_original = query_vector;

    LVQueryHeap heap = {.min_epoch_date = min_epoch_date,
                        .max_epoch_date = max_epoch_date,
                        .capacity = k * db->option.rerank_factor,
                        .entries = db->heap_entries,
                        .size = 0};

    lv_update_query_heap_from_pending(db, &heap, query_ternary);

    LVStatus s = lv_update_query_heap_from_disk(db, &heap, query_ternary);

    if (s != LV_OK) {
        return s;
    }

    const LVSize32_t oversample_size = heap.size;

    if (db->option.rerank_factor > 1) {
        for (uint32_t i = 0; i < oversample_size; ++i) {
            LVQueryHeapEntry *entry = &heap.entries[i];
            const LVTernaryChunk *cand_ternary = NULL;

            if (entry->is_on_pending) {
                cand_ternary = lv_n_pending_ternary(db, entry->ternary_idx);
            } else {
                s = db->option.io_ops->read(db->ter_file, db->ternary_window, ternary_bytes,
                                            ternary_bytes * entry->ternary_idx);
                if (s != LV_OK) {
                    return s;
                }
                cand_ternary = db->ternary_window;
            }

            entry->score = ternary_asymmetric_score(query_original, cand_ternary, db->config.vector_type,
                                                    ternary_chunks_count(db->config.vector_dimension));
        }

        lv_query_heap_build(&heap);
    }

    const LVSize32_t heap_result_size = oversample_size > k ? k : oversample_size;

    if (heap_result_size > 0) {
        lv_query_heap_sort(&heap);

        for (uint32_t i = 0; i < heap_result_size; ++i) {
            result[i] = heap.entries[i].q_result;
        }
    }

    *result_size = heap_result_size;

    return LV_OK;
}

void lv_update_query_heap_from_pending(Livero *db, LVQueryHeap *heap, const LVTernaryChunk *query_ternary) {
    assert(db && heap);
    const LVCount_t ternary_chunk_count = ternary_chunks_count(db->config.vector_dimension);

    for (LVCount_t i = 0; i < db->pending_count; i++) {
        const LVTernaryChunk *ternary_candidate = lv_n_pending_ternary(db, i);
        const float score = ternary_dot(query_ternary, ternary_candidate, ternary_chunk_count);
        if (lv_query_heap_admit(heap, score)) {
            const uint8_t *row = lv_n_pending_row(db, i);
            lv_query_heap_update(heap, row, score, 1, i);
        }
    }
}

LVStatus lv_update_query_heap_from_disk(Livero *db, LVQueryHeap *heap, const LVTernaryChunk *query_ternary) {
    assert(db && heap && query_ternary);
    const LVSize32_t ternary_bytes = ternary_chunks_bytes(db->config.vector_dimension);
    const LVCount_t ternary_chunk_count = ternary_chunks_count(db->config.vector_dimension);
    const LVCount_t disk_record_count = lv_disk_record_count(db);
    const LVCount_t batch_m = db->option.read_batch_m;
    for (LVCount_t base = 0; base < disk_record_count; base += batch_m) {
        const LVCount_t remaining = disk_record_count - base;
        const LVCount_t n = remaining < batch_m ? remaining : batch_m;

        LVStatus s = db->option.io_ops->read(db->ter_file, db->ternary_window, n * ternary_bytes,
                                             (LVOffset_t)base * ternary_bytes);

        if (s != LV_OK) {
            return s;
        }

        LVCount_t cache_lo = 0;
        LVCount_t cache_hi = 0;

        for (LVCount_t j = 0; j < n; ++j) {
            const LVTernaryChunk *ternary_candidate = lv_n_window_ternary(db, j);
            const float score = ternary_dot(query_ternary, ternary_candidate, ternary_chunk_count);
            if (!lv_query_heap_admit(heap, score)) {
                continue;
            }

            if (j >= cache_hi) {
                const LVCount_t fetch_count = (heap->size * 10 <= heap->capacity * 3) ? n - j : 1;
                s = db->option.io_ops->read(db->row_file, db->row_window, fetch_count * LV_ROW_BYTES,
                                            (LVOffset_t)(base + j) * LV_ROW_BYTES);
                if (s != LV_OK) {
                    return s;
                }
                cache_lo = j;
                cache_hi = j + fetch_count;
            }
            const uint8_t *row = lv_n_window_row(db, j - cache_lo);
            lv_query_heap_update(heap, row, score, 0, base + j);
        }
    }

    return LV_OK;
}

/************************************************************************************************/

/*************** lv_close and lv_stat section. *******************/

void lv_close(Livero *db) {
    if (!db) {
        LIVERO_ERROR("Db is missing.");
        return;
    }

    if (db->pending_count > 0) {
        LVStatus s = lv_flush(db);
        if (s != LV_OK) {
            LIVERO_ERROR("Failed to flush data.");
        }
    }

    LVStatus s = db->option.io_ops->close(db->row_file);
    if (s != LV_OK) {
        LIVERO_ERROR("Failed to close row.lv.");
    }

    if ((s = db->option.io_ops->close(db->ter_file)) != LV_OK) {
        LIVERO_ERROR("Failed to close ter.lv.");
    }

    if ((s = db->option.io_ops->close(db->val_file)) != LV_OK) {
        LIVERO_ERROR("Failed to close val.lv.");
    }

    if ((s = db->option.io_ops->close(db->commit_file)) != LV_OK) {
        LIVERO_ERROR("Failed to close commit.lv.");
    }

    free(db);
}

void lv_stat(const Livero *db, LVDBStat *stat) {
    if (db && stat) {
        stat->config = db->config;
        stat->option = db->option;
        stat->row_file_bytes = db->row_file_bytes;
        stat->ternary_file_bytes = db->ternary_file_bytes;
        stat->val_file_bytes = db->value_file_bytes;
        stat->row_pending_bytes = lv_pending_row_bytes(db);
        stat->ternary_pending_bytes = lv_pending_ternary_bytes(db);
        stat->val_pending_bytes = lv_pending_val_bytes(db);
    }
}

/************************************************************************************************/
