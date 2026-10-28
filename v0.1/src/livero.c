
#include "livero.h"

#include <assert.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>

#include "file/file.h"
#include "livero_internal.h"
#include "livero_types.h"
#include "vector/ternary.h"

struct Livero {
    LVSeq_t next_seq;                   // 8B
    LVSeq_t next_commit_seq;            // 8B
    LVTernaryChunk *ternary_scratch;    // <= 8B
    uint8_t *read_vector_batch;  // <= 8B
    uint8_t *write_row_batch;           // <= 8B
    uint8_t *write_vector_batch;        // <= 8B
    uint8_t *write_value_batch;         // <= 8B
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
    LVSize16_t max_value_len;
    LVDim_t vector_dimension;
    LVVectorType vector_type;
    uint8_t flushed;
    LVSyncMode sync_mode;
};

static_assert(sizeof(Livero) <= 104, "Livero struct size must be under 128 bytes.");

LVStatus lv_create(const char *db_path, const LVConfig *config, const LVOption *option, Livero **db_out) {
    if (!db_path) {
        LIVERO_ERROR("You missed a path to store data.");
        return LV_ERR_INVALID_PARAM;
    }

    if (!config) {
        LIVERO_ERROR("You missed a config.");
        return LV_ERR_INVALID_PARAM;
    }

    if (!option) {
        LIVERO_ERROR("You missed an option.");
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
    int commit_fd = -1;

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

    if ((s = lv_open_file(db_path, LV_COMMIT_FILE_NAME, create_flag, &commit_fd)) != LV_OK) {
        LIVERO_ERROR("Failed to open commit.lv.");
        goto cleanup;
    }

    LVConfig cfg;

    cfg.max_disk_bytes = config->max_disk_bytes;
    cfg.disk_sector_bytes = config->disk_sector_bytes > 0 ? config->disk_sector_bytes : LV_DEFAULT_SECTOR_BYTES;
    cfg.max_value_len = config->max_value_len;
    cfg.vector_dimension = config->vector_dimension;
    cfg.vector_type = config->vector_type;

    if ((s = lv_save_config(commit_fd, &cfg)) != LV_OK) {
        LIVERO_ERROR("Failed to save the config in commit.lv.");
        goto cleanup;
    }

    void *m_arena = NULL;
    if ((s = lv_allocate_memory(option, cfg.max_value_len, cfg.vector_dimension, &m_arena)) != LV_OK) {
        LIVERO_ERROR("Failed to allocate a memory arena.");
        goto cleanup;
    }

    LVInitCtx ctx = {
      .next_seq = 0,
      .next_commit_seq = 0,
      .row_fd = row_fd,
      .vec_fd = vec_fd,
      .val_fd = val_fd,
      .row_bytes = 0,
      .vec_bytes = 0,
      .val_bytes = 0,
    };

    lv_init_livero(m_arena, &cfg, option, &ctx, db_out);

    return LV_OK;

cleanup:
    lv_close_file(row_fd);
    lv_close_file(vec_fd);
    lv_close_file(val_fd);
    lv_close_file(commit_fd);
    return s;
}

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

LVStatus lv_allocate_memory(const LVOption *option, const LVSize16_t max_value_len, const LVDim_t dim, void **ptr_out) {
    assert(ptr_out && lv_check_option(option));
    const LVSize32_t vector_bytes = ternary_chunks_bytes(dim);
    const LVSize32_t size = sizeof(Livero) + vector_bytes + vector_bytes * option->read_batch_m +
                            LV_ROW_DISK_BYTES * option->write_batch_m + vector_bytes * option->write_batch_m +
                            max_value_len * option->write_batch_m;
    void *m = aligned_alloc(LV_MEMORY_ALIGNMENT, size);

    if (!m) {
        return LV_ERR_ALLOCATE_MEMORY;
    }

    *ptr_out = m;

    return LV_OK;
}

LVStatus lv_save_config(int fd, const LVConfig *config) {
    assert(config);
    LVConfigDisk cfg_disk = lv_config_m2d(config);
    return lv_write_file(fd, &cfg_disk, LV_CONFIG_DISK_BYTES, 0);
}

void lv_init_livero(uint8_t *m_arena, const LVConfig *config, const LVOption *option, const LVInitCtx *ctx,
                    Livero **db_out) {
    assert(m_arena && config && option && ctx && db_out);
    assert(lv_check_option(option));

    LVOffset_t offset = 0;
    const LVSize32_t vector_bytes = ternary_chunks_bytes(config->vector_dimension);

    Livero *db = (Livero *)m_arena;
    offset += sizeof(Livero);

    LVTernaryChunk *ternary_scratch = (LVTernaryChunk *)(m_arena + offset);
    offset += vector_bytes;

    uint8_t *read_vector_batch = (m_arena + offset);
    offset += vector_bytes * option->read_batch_m;

    uint8_t *write_row_batch = (m_arena + offset);
    offset += LV_ROW_DISK_BYTES * option->write_batch_m;

    uint8_t *write_vector_batch = (m_arena + offset);
    offset += vector_bytes * option->write_batch_m;

    uint8_t *write_value_batch = m_arena + offset;

    db->next_seq = ctx->next_seq;
    db->next_commit_seq = ctx->next_commit_seq;
    db->ternary_scratch = ternary_scratch;
    db->read_vector_batch = read_vector_batch;
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
    db->max_disk_bytes = config->max_disk_bytes;
    db->batched_count = 0;
    db->batch_threshold_count = option->write_batch_m;
    db->max_value_len = config->max_value_len;
    db->vector_dimension = config->vector_dimension;
    db->vector_type = config->vector_type;
    db->flushed = 0;
    db->sync_mode = option->sync_mode;
}
