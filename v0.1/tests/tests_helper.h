#ifndef LV_TEST_HELPER
#define LV_TEST_HELPER

#include <assert.h>
#include <float.h>
#include <ftw.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "helper/helper.h"
#include "livero.h"
#include "livero_internal.h"
#include "livero_types.h"
#include "struct/row.h"
#include "vector/ternary.h"
#define _1KiB 1024
#define _1MiB (1u << 20)
static const char TEST_CHAR_POOL[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789[]{}():;,.?!@#$^&*";
static const uint32_t TEST_CHAR_POOL_SIZE = sizeof(TEST_CHAR_POOL) - 1;

#define TEST_MAX_DISK_BYTES _1MiB * 4
#define TEST_MAX_VALUE_LEN 200
#define TEST_VECTOR_DIMENSION 384
#define TEST_TERNARY_CHUNK_COUNT (TEST_VECTOR_DIMENSION / LV_TERNARY_CHUNK_CAPACITY)
#define TEST_VECTOR_TYPE LV_VEC_TYPE_INT8
#define TEST_WRITE_BATCH_M 100
#define TEST_READ_BATCH_M 100
#define TEST_SYNC_MODE LV_SYNC_BATCH

static LVConfig TEST_CONFIG = {
    .max_disk_bytes = TEST_MAX_DISK_BYTES,
    .max_value_len = TEST_MAX_VALUE_LEN,
    .vector_dimension = TEST_VECTOR_DIMENSION,
    .vector_type = TEST_VECTOR_TYPE,
};

static LVOption TEST_OPTION = {
    .write_batch_m = TEST_WRITE_BATCH_M,
    .read_batch_m = TEST_READ_BATCH_M,
    .sync_mode = TEST_SYNC_MODE,
};

static int8_t INT8_VEC_BUFFER[TEST_VECTOR_DIMENSION] = {0};

typedef struct {
    LVRow row;
    uint8_t value[TEST_MAX_VALUE_LEN];
    LVTernaryChunk ternary[TEST_TERNARY_CHUNK_COUNT];
} LVTestAnswer;

typedef struct {
    LVSeq_t seq;
    int value_len;  // fallback to random len (1, TEST_MAX_VALUE_LEN) if value_len is zero or negative or exceeds max
                    // len.
    uint16_t min_year;
    uint16_t max_year;
    LVOffset_t prev_value_offset;
    int alive;
} LVMakeTestAnswerCtx;

static int create_test_dir(const char* dir) { return mkdir(dir, 0755); }

static int remove_callback(const char* fpath, const struct stat* sb, int typeflag, struct FTW* ftwbuf) {
    int rv = remove(fpath);  // remove(): rmdir() for a directory, unlink() for a file
    if (rv) {
        perror(fpath);
    }
    return rv;
}

static int clean_test_dir(const char* dir) { return nftw(dir, remove_callback, 64, FTW_DEPTH | FTW_PHYS); }

static int8_t rand_int8(int8_t min, int8_t max) {
    if (min == max) {
        return min;
    }
    return min + rand() % (max - min + 1);
}

static int rand_int(int min, int max) {
    if (min == max) {
        return min;
    }
    return min + rand() % (max - min + 1);
}

static uint64_t rand_uint64(void) {
    uint64_t r = 0;

    for (int i = 0; i < 8; ++i) {
        r = (r << 8) | (uint64_t)(rand() & 0xFF);
    }

    return r;
}

static float rand_fp32(float min, float max) {
    if (fabsf(min - max) <= FLT_EPSILON) {
        return min;
    }
    float scale = (float)rand() / ((float)RAND_MAX + 1.0f);
    return min + scale * (max - min);
}

static void fill_random_chr_data(const LVSize32_t len, char* data_out) {
    for (LVSize32_t k = 0; k < len; ++k) {
        data_out[k] = TEST_CHAR_POOL[rand() % TEST_CHAR_POOL_SIZE];
    }
}

static LVDate make_random_date(const uint16_t min_year, const uint16_t max_year) {
    uint8_t days_in_month[13] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    const uint16_t MIN = min_year < max_year ? min_year : max_year;
    const uint16_t MAX = max_year > min_year ? max_year : min_year;
    const uint16_t year = rand_int(MIN, MAX);
    const uint8_t month = rand_int8(1, 12);
    days_in_month[2] += ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0);
    const uint8_t day = rand_int8(1, days_in_month[month]);
    const uint8_t hour = rand_int8(0, 23);
    const uint8_t min = rand_int8(0, 59);
    const uint8_t sec = rand_int8(0, 59);

    LVDate date;
    date.year = year;
    date.month = month;
    date.day = day;
    date.hour = hour;
    date.min = min;
    date.sec = sec;
    return date;
}

static uint64_t make_random_epoch_date(const uint16_t min_year, const uint16_t max_year) {
    LVDate date = make_random_date(min_year, max_year);
    return lv_convert_date_to_epoch(&date);
}

static void make_test_answer(const LVMakeTestAnswerCtx* ctx, LVTestAnswer* answer_out) {
    assert(ctx && answer_out);

    char value[TEST_MAX_VALUE_LEN] = {0};
    const LVSize16_t v_len =
        ctx->value_len <= 0 || ctx->value_len > TEST_MAX_VALUE_LEN ? rand_int(1, TEST_MAX_VALUE_LEN) : ctx->value_len;
    fill_random_chr_data(v_len, value);
    LVTernaryChunk ternary[TEST_TERNARY_CHUNK_COUNT] = {0};

    for (LVCount_t i = 0; i < TEST_TERNARY_CHUNK_COUNT; ++i) {
        ternary[i].mask = rand_uint64();
        ternary[i].sign = rand_uint64();
    }

    const uint64_t epoch_date = make_random_epoch_date(ctx->min_year, ctx->max_year);
    const LVOffset_t v_offset = ctx->prev_value_offset;
    LVRowCalcChecksumCtx checksum_ctx = {.epoch_date = epoch_date,
                                         .seq = ctx->seq,
                                         .value_len = v_len,
                                         .value_offset = v_offset,
                                         .value = value,
                                         .ternary = ternary,
                                         .vector_dim = TEST_VECTOR_DIMENSION};

    answer_out->row.seq = ctx->seq;
    answer_out->row.date = epoch_date;
    answer_out->row.value_len = v_len;
    answer_out->row.value_offset = v_offset;
    answer_out->row.tombstone_flag = ctx->alive ? 0 : LV_TOMBSTONE_FLAG;
    answer_out->row.checksum = row_calc_checksum(&checksum_ctx);
    memcpy(answer_out->value, value, v_len);
    memcpy(answer_out->ternary, ternary, ternary_chunks_bytes(TEST_VECTOR_DIMENSION));
}

static int compare_date(const LVDate* date_a, const LVDate* date_b) {
    return lv_convert_date_to_epoch(date_a) - lv_convert_date_to_epoch(date_b);
}

// Returns 0 if the row and value match the expected answer, -1 otherwise.
static int compare_answer(const LVTestAnswer* answer, const uint8_t* row, const uint8_t* value,
                          const LVTernaryChunk* ternary) {
    if (!answer || !row || !value || !ternary) {
        fprintf(stderr, "compare_answer received NULL.\n");
        return -1;
    }

    if (answer->row.seq != row_get_seq(row)) {
        fprintf(stderr, "seq is incorrect. expect: %" PRIu64 ", got: %" PRIu64 "\n", answer->row.seq, row_get_seq(row));
        return -1;
    }

    if (answer->row.date != row_get_date(row)) {
        fprintf(stderr, "date is incorrect. expect: %" PRIu64 ", got: %" PRIu64 "\n", answer->row.date,
                row_get_date(row));
        return -1;
    }

    if (answer->row.value_offset != row_get_value_offset(row)) {
        fprintf(stderr, "value_offset is incorrect. expect: %" PRIu32 ", got: %" PRIu32 "\n", answer->row.value_offset,
                row_get_value_offset(row));
        return -1;
    }

    if (answer->row.value_len != row_get_value_len(row)) {
        fprintf(stderr, "value_len is incorrect. expect: %" PRIu16 ", got: %" PRIu16 "\n", answer->row.value_len,
                row_get_value_len(row));
        return -1;
    }

    if (answer->row.tombstone_flag != row_get_tombstone_flag(row)) {
        fprintf(stderr, "tombstone_flag is incorrect. expect: %" PRIu8 ", got: %" PRIu8 "\n",
                answer->row.tombstone_flag, row_get_tombstone_flag(row));
        return -1;
    }

    if (answer->row.checksum != row_get_checksum(row)) {
        fprintf(stderr, "checksum is incorrect. expect: %" PRIu32 ", got: %" PRIu32 "\n", answer->row.checksum,
                row_get_checksum(row));
        return -1;
    }

    // Guard the read below: answer->value holds at most TEST_MAX_VALUE_LEN bytes.
    if (answer->row.value_len > TEST_MAX_VALUE_LEN) {
        fprintf(stderr, "value_len exceeds TEST_MAX_VALUE_LEN. value_len: %" PRIu16 "\n", answer->row.value_len);
        return -1;
    }

    if (memcmp(answer->value, value, answer->row.value_len) != 0) {
        fprintf(stderr, "value is incorrect. seq: %" PRIu64 "\n", answer->row.seq);
        return -1;
    }

    if (memcmp(answer->ternary, ternary, ternary_chunks_bytes(TEST_VECTOR_DIMENSION)) != 0) {
        fprintf(stderr, "ternary is incorrect. seq: %" PRIu64 "\n", answer->row.seq);
        return -1;
    }

    return 0;
}

static void test_end(const int result, const char* test_dir, const char* test_name) {
    clean_test_dir(test_dir);
    if (result < 0) {
        fprintf(stderr, "%s failed.\n", test_name);
    } else {
        fprintf(stdout, "%s passed.\n", test_name);
    }
}

static void make_first_date(const uint16_t year, LVDate* date) {
    assert(date);
    date->year = year;
    date->month = 1;
    date->day = 1;
    date->hour = 0;
    date->min = 0;
    date->sec = 0;
}

static void make_last_date(const uint16_t year, LVDate* date) {
    assert(date);
    date->year = year;
    date->month = 12;
    date->day = 31;
    date->hour = 23;
    date->min = 59;
    date->sec = 59;
}

static void print_status(const LVStatus status) {
    switch (status) {
        case LV_OK:
            fprintf(stdout, "LV_OK \n");
            break;

        case LV_ERR_INVALID_PARAM:
            fprintf(stderr, "LV_ERR_INVALID_PARAM \n");
            break;

        case LV_ERR_IO:
            fprintf(stderr, "LV_ERR_IO \n");
            break;

        case LV_ERR_DISK_FULL:
            fprintf(stderr, "LV_ERR_DISK_FULL \n");
            break;

        case LV_ERR_DNEXSIST:
            fprintf(stderr, "LV_ERR_DNEXSIST \n");
            break;

        case LV_ERR_EXSIST:
            fprintf(stderr, "LV_ERR_EXSIST \n");
            break;

        case LV_ERR_NOT_FOUND:
            fprintf(stderr, "LV_ERR_NOT_FOUND \n");
            break;

        case LV_ERR_ALLOCATE_MEMORY:
            fprintf(stderr, "LV_ERR_ALLOCATE_MEMORY \n");
            break;

        case LV_ERR_MEDIA_CRASHED:
            fprintf(stderr, "LV_ERR_MEDIA_CRASHED \n");
            break;

        default:
            fprintf(stderr, "Invalid status. \n");
            break;
    }
}

/*
    Clean recived directory first and
    Create a db based on given config and option (fallback to TEST_CONFIG, TSET_OPTION)
*/
static int create_test_db(const char* dir, const LVConfig* config, const LVOption* option, Livero** db_out) {
    assert(dir && db_out);
    clean_test_dir(dir);
    if (create_test_dir(dir) < 0) {
        fprintf(stderr, "Failed to create a test directory. \n");
        return -1;
    }
    LVConfig cfg;
    LVOption opt;
    if (!config) {
        cfg = TEST_CONFIG;
    } else {
        cfg = *config;
    }
    if (!option) {
        opt = TEST_OPTION;
    } else {
        opt = *option;
    }
    const LVStatus s = lv_create(dir, &cfg, &opt, db_out);
    if (s == LV_OK) {
        return 0;
    } else {
        fprintf(stderr, "Failed to craete a test db. Status: ");
        print_status(s);
        return -1;
    }
}

static void print_err_unexpected_status(const char* function_name, const LVSeq_t seq, const LVStatus expected,
                                        const LVStatus got) {
    if (!function_name) {
        return;
    }
    fprintf(stderr, "%s returned unexpected status. (Seq: %" PRIu64 ")\n", function_name, seq);
    fprintf(stderr, "Expected: ");
    print_status(expected);
    fprintf(stderr, " Got: ");
    print_status(got);
    fprintf(stderr, "\n");
}

// Returns 0 if the stored config matches the expected one, -1 otherwise.
static int check_config_match(const LVConfig* stored, const LVConfig* expected) {
    if (stored->max_disk_bytes != expected->max_disk_bytes) {
        fprintf(stderr, "DB config is invalid: max_disk_bytes (expect: %" PRIu32 ", got: %" PRIu32 ").\n",
                expected->max_disk_bytes, stored->max_disk_bytes);
        return -1;
    }

    if (stored->max_value_len != expected->max_value_len) {
        fprintf(stderr, "DB config is invalid: max_value_len (expect: %d, got: %d).\n", expected->max_value_len,
                stored->max_value_len);
        return -1;
    }

    if (stored->vector_dimension != expected->vector_dimension) {
        fprintf(stderr, "DB config is invalid: vector_dimension (expect: %d, got: %d).\n", expected->vector_dimension,
                stored->vector_dimension);
        return -1;
    }

    if (stored->vector_type != expected->vector_type) {
        fprintf(stderr, "DB config is invalid: vector_type (expect: %d, got: %d).\n", expected->vector_type,
                stored->vector_type);
        return -1;
    }

    return 0;
}

// Returns 0 if the stored option matches the expected one, -1 otherwise.
static int check_option_match(const LVOption* stored, const LVOption* expected) {
    if (stored->read_batch_m != expected->read_batch_m) {
        fprintf(stderr, "DB option is invalid: read_batch_m (expect: %" PRIu32 ", got: %" PRIu32 ").\n",
                expected->read_batch_m, stored->read_batch_m);
        return -1;
    }

    if (stored->write_batch_m != expected->write_batch_m) {
        fprintf(stderr, "DB option is invalid: write_batch_m (expect: %" PRIu32 ", got: %" PRIu32 ").\n",
                expected->write_batch_m, stored->write_batch_m);
        return -1;
    }

    if (stored->sync_mode != expected->sync_mode) {
        // Cast because an enum's underlying type is implementation-defined.
        fprintf(stderr, "DB option is invalid: sync_mode (expect: %d, got: %d).\n", (int)expected->sync_mode,
                (int)stored->sync_mode);
        return -1;
    }

    return 0;
}

// Returns 0 if both values match, -1 otherwise. `name` is used only for the error message.
static int check_size_match(const char* name, LVSize32_t stored, LVSize32_t expected) {
    if (stored != expected) {
        fprintf(stderr, "DB stat is invalid: %s (expect: %" PRIu32 ", got: %" PRIu32 ").\n", name, expected, stored);
        return -1;
    }

    return 0;
}

// Returns 0 if the stored stat matches the expected one, -1 otherwise.
// Stops at the first mismatch, so only that field is reported.
static int check_stat_match(const LVDBStat* stored, const LVDBStat* expected) {
    if (check_config_match(&stored->config, &expected->config) != 0) {
        return -1;
    }

    if (check_option_match(&stored->option, &expected->option) != 0) {
        return -1;
    }

    if (check_size_match("row_file_bytes", stored->row_file_bytes, expected->row_file_bytes) != 0 ||
        check_size_match("vec_file_bytes", stored->vec_file_bytes, expected->vec_file_bytes) != 0 ||
        check_size_match("val_file_bytes", stored->val_file_bytes, expected->val_file_bytes) != 0 ||
        check_size_match("row_pending_bytes", stored->row_pending_bytes, expected->row_pending_bytes) != 0 ||
        check_size_match("vec_pending_bytes", stored->vec_pending_bytes, expected->vec_pending_bytes) != 0 ||
        check_size_match("val_pending_bytes", stored->val_pending_bytes, expected->val_pending_bytes) != 0) {
        return -1;
    }

    return 0;
}

#endif
