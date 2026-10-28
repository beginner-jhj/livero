#ifndef LIVERO_INTERNAL
#define LIVERO_INTERNAL

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>

#include "livero.h"
#include "livero_types.h"

static void lv_log_error(const char *file, int line, const char *func, const char *fmt, ...) {
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
#define LV_COMMIT_SLOT_M 64
#define LV_DIMENSION_MULTIPLE 64
#define LV_MAX_BATCH_M 64
#define LV_ROW_FILE_NAME "row.lv"
#define LV_VEC_FILE_NAME "vec.lv"
#define LV_VAL_FILE_NAME "val.lv"
#define LV_COMMIT_FILE_NAME "commit.lv"
#define LV_MAX_PATH_LEN 512

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
} LVInitCtx;

int lv_check_option(const LVOption *option);
LVStatus lv_allocate_memory(const LVOption *option,const LVSize16_t max_value_len, const LVDim_t dim, void **ptr_out);

/*
  Write the config on the commit.lv
*/
LVStatus lv_save_config(int fd, const LVConfig *config);

void lv_init_livero(uint8_t* m_arena,const LVConfig* config, const LVOption* option, const LVInitCtx* ctx, Livero** db_out);

#endif
