#include "livero.h"

#include <assert.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "file/file.h"
#include "livero_internal.h"
#include "livero_types.h"
#include "struct/config.h"
#include "struct/row.h"
#include "vector/ternary.h"

struct Livero {
    LVSeq_t next_seq;
    LVSeq_t next_commit_seq;
    LVTernaryChunk *ternary_scratch;
    uint8_t *value_scratch;
    uint8_t *read_row_batch;
    uint8_t *read_vector_batch;
    uint8_t *read_value_batch;
    uint8_t *write_row_batch;
    uint8_t *write_vector_batch;
    uint8_t *write_value_batch;
    int row_fd;
    int vec_fd;
    int val_fd;
    int commit_fd;
    LVSize32_t row_bytes;
    LVSize32_t vec_bytes;
    LVSize32_t val_bytes;
    LVSize32_t max_disk_bytes;
    LVCount_t batched_count;
    LVCount_t batch_threshold_count;
    LVSize16_t batched_value_bytes;
    LVSize16_t max_value_len;
    LVDim_t vector_dimension;
    LVVectorType vector_type;
    LVSyncMode sync_mode;
};

static_assert(sizeof(Livero) <= 128, "Livero struct size must be under 128 bytes.");
static inline LVSize32_t lv_total_memory_bytes(const LVCount_t read_batch_m, const LVCount_t write_batch_m,
                                               const LVSize32_t max_value_len, const LVDim_t dim) {
    LVSize32_t total = sizeof(Livero);
    const LVSize32_t vector_bytes = ternary_chunks_bytes(dim);
    total += vector_bytes;
    total += max_value_len;
    total += LV_ROW_DISK_BYTES * read_batch_m;
    total += vector_bytes * read_batch_m;
    total += max_value_len * read_batch_m;
    total += LV_ROW_DISK_BYTES * write_batch_m;
    total += vector_bytes * write_batch_m;
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

    if ((s = lv_init_commit(commit_fd)) != LV_OK) {
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
        LIVERO_ERROR("Failed to allocate %d bytes.");
        goto cleanup;
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
        .next_commit_seq = commit.commit_seq + 1,
        .next_seq = commit.reserved_seq + 1,
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

    uint8_t *value_scratch = (m_arena + offset);
    offset += ctx->max_value_len;

    uint8_t *read_row_batch = m_arena + offset;
    offset += LV_ROW_DISK_BYTES * ctx->read_batch_m;

    uint8_t *read_vector_batch = (m_arena + offset);
    offset += vector_bytes * ctx->read_batch_m;

    uint8_t *read_value_batch = m_arena + offset;
    offset += ctx->max_value_len * ctx->read_batch_m;

    uint8_t *write_row_batch = (m_arena + offset);
    offset += LV_ROW_DISK_BYTES * ctx->write_batch_m;

    uint8_t *write_vector_batch = (m_arena + offset);
    offset += vector_bytes * ctx->write_batch_m;

    uint8_t *write_value_batch = m_arena + offset;

    db->next_seq = ctx->next_seq;
    db->next_commit_seq = ctx->next_commit_seq;
    db->ternary_scratch = ternary_scratch;
    db->value_scratch = value_scratch;
    db->read_row_batch = read_row_batch;
    db->read_vector_batch = read_vector_batch;
    db->read_value_batch = read_value_batch;
    db->write_row_batch = write_row_batch;
    db->write_vector_batch = write_vector_batch;
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
    db->batch_threshold_count = ctx->write_batch_m;
    db->max_value_len = ctx->max_value_len;
    db->batched_value_bytes = 0;
    db->vector_dimension = ctx->vector_dimension;
    db->vector_type = ctx->vector_type;
    db->sync_mode = ctx->sync_mode;
}

LVStatus lv_recover(const LVRecoverCtx *ctx, LVCommit *commit_out) {
    assert(ctx);
    const LVSize32_t vector_bytes = ternary_chunks_bytes(ctx->vector_dim);
    LVCommit cmt;
    LVOffset_t offset = 0;
    LVSeq_t max_commit_seq = LV_COMMIT_SEQ_ZERO;
    int confirmed_commit_idx = 0;
    int corrupted_commit_idx = 0;
    int has_corrupted = 0;
    for (int i = 0; i < LV_COMMIT_SLOT_M; ++i) {
        LVStatus s = lv_commit_d2m(ctx->commit_fd, offset, &cmt);

        if (s == LV_OK) {
            if (cmt.commit_seq > max_commit_seq) {
                const uint64_t saved_bytes =
                    LV_ROW_DISK_BYTES * cmt.total_count + vector_bytes * cmt.total_count + cmt.total_value_bytes;
                if (saved_bytes > (uint64_t)ctx->max_disk_bytes || cmt.batch_count > cmt.total_count) {
                    /*
                        This commit saved invalid values.
                    */
                    has_corrupted = 1;
                    corrupted_commit_idx = lv_commit_slot_idx(offset);
                } else {
                    LVRowCCFDCtx rccfd;
                    rccfd.row_fd = ctx->row_fd;
                    rccfd.vec_fd = ctx->vec_fd;
                    rccfd.val_fd = ctx->val_fd;
                    rccfd.vector_dim = ctx->vector_dim;
                    for (int j = cmt.batch_count; j >= 0; --j) {
                        rccfd.row_offset = (cmt.total_count - j) * LV_ROW_DISK_BYTES;
                        rccfd.vec_offset = (cmt.total_count - j) * vector_bytes;

                        LVStatus _s = lv_row_checksum_check_from_disk(&rccfd);
                        if (_s == LV_ERR_CORRUPT) {
                            /*
                                This commit has a corrupted row.
                            */
                            has_corrupted = 1;
                            corrupted_commit_idx = lv_commit_slot_idx(offset);
                            break;
                        } else if (_s == LV_ERR_IO) {
                            return s;
                        }
                    }

                    max_commit_seq = cmt.commit_seq;
                    confirmed_commit_idx = lv_commit_slot_idx(offset);
                }
            }
        }

        else if (s == LV_ERR_CORRUPT) {
            /*
              This commit is corrupted, skip and find the next valid commit.
                          */
            has_corrupted = 1;
            corrupted_commit_idx = lv_commit_slot_idx(offset);
        } else {
            /*
                This is an unexpected error.
            */
            return s;
        }
        offset += LV_COMMIT_SECTOR_BYTES;
    }

    if (has_corrupted) {
        /*
            Every commit slot doesn't touch other slots,
            so in the expected crash scenario, the corrupted commit must be the exact next slot of the confirmed commit.

            If the corrupted commit is not in the exact next slot, this means the hardware is broken and it's not
           recoverable
        */
        if (confirmed_commit_idx == LV_COMMIT_SLOT_M - 1 && corrupted_commit_idx != 0) {
            return LV_ERR_MEDIA_CRASHED;
        }
        if (confirmed_commit_idx + 1 != corrupted_commit_idx) {
            return LV_ERR_MEDIA_CRASHED;
        }
    }
    if (max_commit_seq == LV_COMMIT_SEQ_ZERO) {
        memset(commit_out, 0, sizeof(LVCommit));
    } else {
        memcpy(commit_out, &cmt, sizeof(cmt));
    }
    return LV_OK;
}
