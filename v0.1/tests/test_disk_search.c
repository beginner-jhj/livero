#include <stdint.h>
#include <string.h>

#include "livero_internal.h"
#include "livero_types.h"
#include "struct/commit.h"
#include "struct/row.h"
#include "tests_helper.h"

static const char* TEST_DIR = "./TEST_DISK_TMP/";
#define TEST_DATA_COUNT 1000
#define TEST_SEQ_LEAP_POINT 200
#define TEST_SEQ_LEAP_POINT_COUNT TEST_DATA_COUNT / TEST_SEQ_LEAP_POINT
static LVTestAnswer TEST_DATA[TEST_DATA_COUNT] = {0};
static const int TEST_SEQ_LEAPS[TEST_SEQ_LEAP_POINT_COUNT] = {0, 1, 10, 100, 1000};
static const int TEST_TOMBSTONE_MULTIPLE = 10;  // Mark a tombstone on every record whose seq is a multiplem of 10

// Each gap contributes its first, middle and last missing seq; the ends add up to 3 more.
#define TEST_MISSING_SEQS_CAP (3 * (TEST_SEQ_LEAP_POINT_COUNT))
static LVSeq_t TEST_MISSING_SEQS[TEST_MISSING_SEQS_CAP] = {0};
static LVCount_t TEST_MISSING_SEQ_COUNT = 0;

// Function pointer type for any disk search strategy.
typedef LVStatus (*LVDiskSearchFn)(LVDiskSearchCtx* ctx, const LVSeq_t target, LVGetResult* result,
                                   LVOffset_t* row_offset_out);

/*
Seqs:
    0->199
    201->400
    410->609
    700->899
    1800->1999
*/
static void init_test_data(void) {
    LVOffset_t value_offset = 0;
    LVMakeTestAnswerCtx make_tac = {
        .min_year = 2000,
        .max_year = 2016,
        .fixed_date = NULL,
        .value_len = -1,
        .use_test_vector = 0,
    };
    for (int i = 0; i < TEST_SEQ_LEAP_POINT_COUNT; ++i) {
        const LVCount_t leap = TEST_SEQ_LEAPS[i];
        for (int j = 0; j < TEST_SEQ_LEAP_POINT; ++j) {
            const int idx = i * TEST_SEQ_LEAP_POINT + j;
            LVTestAnswer* ans = &TEST_DATA[idx];
            const LVSeq_t seq = idx + leap;
            make_tac.seq = seq;
            make_tac.alive = !(seq % TEST_TOMBSTONE_MULTIPLE == 0);
            make_tac.prev_value_offset = value_offset;
            make_test_answer(&make_tac, ans);
            value_offset += ans->row.value_len;
        }
    }
}

// Append a seq, skipping a duplicate of the previous entry when a gap is tiny.
static void push_missing_seq(const LVSeq_t seq) {
    assert(TEST_MISSING_SEQ_COUNT < TEST_MISSING_SEQS_CAP);
    if (TEST_MISSING_SEQ_COUNT > 0 && TEST_MISSING_SEQS[TEST_MISSING_SEQ_COUNT - 1] == seq) {
        return;
    }
    TEST_MISSING_SEQS[TEST_MISSING_SEQ_COUNT++] = seq;
}

// Must run after TEST_DATA is filled, because gaps are derived from the stored seqs.
static void init_missing_seqs(void) {
    TEST_MISSING_SEQ_COUNT = 0;

    // A seq below the first record, when one exists.
    const LVSeq_t first = TEST_DATA[0].row.seq;
    if (first > 0) {
        push_missing_seq(first - 1);
    }

    for (LVCount_t i = 0; i + 1 < TEST_DATA_COUNT; ++i) {
        const LVSeq_t cur = TEST_DATA[i].row.seq;
        const LVSeq_t next = TEST_DATA[i + 1].row.seq;

        // Adjacent seqs leave no hole.
        if (next - cur <= 1) {
            continue;
        }

        // Both edges of the hole catch off-by-one errors; the middle catches bad narrowing.
        push_missing_seq(cur + 1);
        push_missing_seq(cur + (next - cur) / 2);
        push_missing_seq(next - 1);
    }

    // Seqs just past and far past the last record.
    const LVSeq_t last = TEST_DATA[TEST_DATA_COUNT - 1].row.seq;
    push_missing_seq(last + 1);
    push_missing_seq(last + 1000);
}

static int full_compare(int row_fd, int val_fd, uint8_t* row_window, LVCount_t window_batch_m, LVDiskSearchFn fn,
                        const char* search_fn_name, const int is_interp) {
    LVDiskSearchCtx search_ctx = {
        .disk_record_count = TEST_DATA_COUNT,
        .row_fd = row_fd,
        .val_fd = val_fd,
        .row_window = row_window,
        .window_batch_m = window_batch_m,
    };
    uint8_t value[TEST_MAX_VALUE_LEN] = {0xAA};
    LVGetResult got = {
        .value = value,
    };
    LVOffset_t row_offset = 0;

    for (LVCount_t i = 0; i < TEST_DATA_COUNT; ++i) {
        memset(value, 0xAA, sizeof(value));
        got.value_len = 0;
        memset(&got.date, 0, sizeof(got.date));
        row_offset = (LVOffset_t)-1;
        const LVTestAnswer* ans = &TEST_DATA[i];
        LVStatus expected_status = LV_OK;
        if (ans->row.tombstone_flag != 0) {
            expected_status = LV_ERR_DELETED;
        }
        if (is_interp) {
            int should_hit = interp_should_hit(TEST_DATA_COUNT, TEST_DATA[0].row.seq,
                                               TEST_DATA[TEST_DATA_COUNT - 1].row.seq, ans->row.seq, i, window_batch_m);
            if (!should_hit) {
                expected_status = LV_ERR_NOT_FOUND;
                fprintf(stdout, "Testing interpolation search: %" PRIu64 " can not be hit. \n", ans->row.seq);
            }
        }
        LVStatus s = fn(&search_ctx, ans->row.seq, &got, &row_offset);

        if (s != expected_status) {
            PRINT_ERR_UNEXPECTED_STATUS(search_fn_name, expected_status, s);
            return -1;
        }

        if (expected_status == LV_OK) {
            if (check_get_result(ans, &got) < 0) {
                return -1;
            }

            if (row_offset != i * LV_ROW_BYTES) {
                PRINT_ERR_INVALID("row disk offset", "%u", i * LV_ROW_BYTES, row_offset);
                return -1;
            }
        }
    }

    for (LVCount_t i = 0; i < TEST_MISSING_SEQ_COUNT; ++i) {
        LVStatus s = fn(&search_ctx, TEST_MISSING_SEQS[i], &got, &row_offset);
        if (s != LV_ERR_NOT_FOUND) {
            PRINT_ERR_UNEXPECTED_STATUS(search_fn_name, LV_ERR_NOT_FOUND, s);
            return -1;
        }
    }

    return 0;
}

int test_window_range(void) {
    typedef struct {
        int p, low, high, m;
        int start_idx, end_idx;  // Expected window, both ends inclusive.
    } Case;

    static const Case CASES[] = {
        // Centered: the window fits on both sides of p.
        {50, 0, 99, 10, 45, 54},
        // Near the left edge: slide right instead of starting at p.
        {2, 0, 99, 10, 0, 9},
        // Near the right edge: slide left instead of ending at p.
        {98, 0, 99, 10, 90, 99},
        // Exactly on each edge.
        {0, 0, 99, 10, 0, 9},
        {99, 0, 99, 10, 90, 99},
        // Odd m puts the extra row after p, since m / 2 rounds down.
        {50, 0, 99, 11, 45, 55},
        // Smallest windows.
        {50, 0, 99, 1, 50, 50},
        {50, 0, 99, 2, 49, 50},
        // Range shorter than m: the window shrinks to the whole range.
        {5, 0, 7, 10, 0, 7},
        {0, 0, 0, 10, 0, 0},
        // Range exactly m long.
        {0, 0, 9, 10, 0, 9},
        // A non-zero low, as in later binary search steps.
        {25, 20, 30, 4, 23, 26},
        {21, 20, 30, 4, 20, 23},
        {29, 20, 30, 4, 27, 30},
    };

    int result = 0;

    for (size_t i = 0; i < sizeof(CASES) / sizeof(CASES[0]); ++i) {
        const Case* c = &CASES[i];
        const LVWindowRange r = lv_window_range(c->p, c->low, c->high, c->m);

        if (r.start_idx != c->start_idx || r.end_idx != c->end_idx) {
            fprintf(stderr, "lv_window_range(p=%d, low=%d, high=%d, m=%d) (expect: [%d, %d], got: [%d, %d])\n", c->p,
                    c->low, c->high, c->m, c->start_idx, c->end_idx, r.start_idx, r.end_idx);
            result = -1;
        }
    }

    if (result < 0) {
        fprintf(stderr, "test_window_range failed.\n");
    } else {
        fprintf(stdout, "test_window_range passed.\n");
    }
    return result;
}

int test_interpolation_search(void) {
    /*
        1. Create a db
        2. Write test data on the disk
        3. Full compare with test data
    */

    int result = 0;
    Livero* db = NULL;

    if (create_test_db(TEST_DIR, NULL, NULL, &db) < 0) {
        result = -1;
        goto cleanup;
    }

    LVWriteTestDataCtx wtdc = {
        .data = TEST_DATA,
        .data_count = TEST_DATA_COUNT,
        .row_fd = db->row_fd,
        .vec_fd = db->vec_fd,
        .val_fd = db->val_fd,
        .commit_fd = db->commit_fd,
        .commit_seq = LV_COMMIT_SEQ_START,
        .row_start_offset = 0,
        .vec_start_offset = 0,
        .prev_count = 0,
        .prev_val_bytes = 0,
    };

    if ((result = write_test_data(&wtdc, NULL)) < 0) {
        goto cleanup;
    }

    const char* fn_name = "lv_interpolation_search_from_disk";

    if ((result = full_compare(db->row_fd, db->val_fd, db->row_window, db->window_batch_m,
                               lv_interpolation_search_from_disk, fn_name, 1)) < 0) {
        fprintf(stderr, "%s. \n", fn_name);
        result = -1;
        goto cleanup;
    }

cleanup:
    lv_close(db);
    test_end(result, TEST_DIR, "test_interpolation_search");
    return result;
}

int test_binary_search(void) {
    /*
        1. Create a db
        2. Write test data on the disk
        3. Full compare with test data
    */

    int result = 0;
    Livero* db = NULL;

    if (create_test_db(TEST_DIR, NULL, NULL, &db) < 0) {
        result = -1;
        goto cleanup;
    }

    LVWriteTestDataCtx wtdc = {
        .data = TEST_DATA,
        .data_count = TEST_DATA_COUNT,
        .row_fd = db->row_fd,
        .vec_fd = db->vec_fd,
        .val_fd = db->val_fd,
        .commit_fd = db->commit_fd,
        .commit_seq = LV_COMMIT_SEQ_START,
        .row_start_offset = 0,
        .vec_start_offset = 0,
        .prev_count = 0,
        .prev_val_bytes = 0,
    };

    if ((result = write_test_data(&wtdc, NULL)) < 0) {
        goto cleanup;
    }

    const char* fn_name = "lv_binary_search_from_disk";

    if ((result = full_compare(db->row_fd, db->val_fd, db->row_window, db->window_batch_m, lv_binary_search_from_disk,
                               fn_name, 0)) < 0) {
        fprintf(stderr, "%s. \n", fn_name);
        result = -1;
        goto cleanup;
    }

cleanup:
    lv_close(db);
    test_end(result, TEST_DIR, "test_binary_search");
    return result;
}

int main(void) {
    clean_test_dir(TEST_DIR);
    init_test_data();
    init_missing_seqs();
    if (test_window_range() < 0) {
        return -1;
    }
    if (test_interpolation_search() < 0) {
        return -1;
    }
    if (test_binary_search() < 0) {
        return -1;
    }
    return 0;
}
