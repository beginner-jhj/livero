#ifndef LV_TEST_HELPER
#define LV_TEST_HELPER

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "livero_types.h"
#include "struct/row.h"
#include "vector/ternary.h"
#define _1KiB 1024
#define _1MiB (1u << 20)
static const char TEST_CHAR_POOL[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789[]{}():;,.?!@#$^&*";
static const uint32_t TEST_CHAR_POOL_SIZE = sizeof(TEST_CHAR_POOL) - 1;

static const LVSize32_t TEST_MAX_DISK_BYTES = _1MiB * 4;
static const LVSize16_t TEST_MAX_VALUE_LEN = 200;
static const LVDim_t TEST_VECTOR_DIMENSION = 384;
static const LVCount_t TEST_TERNARY_CHUNK_COUNT = TEST_VECTOR_DIMENSION / LV_TERNARY_CHUNK_CAPACITY;
static const LVVectorType TEST_VECTOR_TYPE = LV_VEC_TYPE_INT8;
static const LVCount_t TEST_WRITE_BATCH_M = 100;
static const LVCount_t TEST_READ_BATCH_M = 100;
static const LVSyncMode TEST_SYNC_MODE = LV_SYNC_BATCH;

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

typedef struct {
    LVRow row;
    uint8_t value[TEST_MAX_VALUE_LEN];
    LVTernaryChunk ternary[TEST_TERNARY_CHUNK_COUNT];
} LVTestAnswer;

int create_test_dir(const char* dir);

int clean_test_dir(const char* dir);

void print_status(const LVStatus status);

int8_t rand_int8(int8_t min, int8_t max);
int rand_int(int min, int max);
uint64_t rand_uint64(void);
float rand_fp32(float min, float max);

void fill_random_chr_data(const LVSize32_t len, char* data_out);

LVDate make_random_date(const uint16_t min_year, const uint16_t max_year);
uint64_t make_random_epoch_date(const uint16_t min_year, const uint16_t max_year);

int compare_date(const LVDate* date_a, const LVDate* date_b);
int compare_answer(const LVTestAnswer* answer, const uint8_t* row, const uint8_t* value, const LVTernaryChunk* ternary);

void make_first_date(const uint16_t year, LVDate* date);
void make_last_date(const uint16_t year, LVDate* date);

typedef struct {
    LVSeq_t seq;
    int value_len; //fallback to random len (1, TEST_MAX_VALUE_LEN) if value_len is zero or negative or exceeds max len.
    uint16_t min_year;
    uint16_t max_year;
    LVOffset_t prev_value_offset;
    int alive;
} LVMakeTestAnswerCtx;
void make_test_answer(const LVMakeTestAnswerCtx* ctx, LVTestAnswer* answer_out);

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

void test_end(const int result, const char* test_dir, const char* test_name);

/*
    Clean recived directory first and
    Create a db based on given config and option (fallback to TEST_CONFIG, TSET_OPTION)
*/
int create_test_db(const char* dir, const LVConfig* config, const LVOption* option, Livero** db_out);

#endif
