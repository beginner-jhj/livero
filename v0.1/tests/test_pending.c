#include <stdint.h>
#include <string.h>

#include "helper/helper.h"
#include "livero.h"
#include "livero_internal.h"
#include "livero_types.h"
#include "struct/row.h"
#include "tests_helper.h"

static const char* TEST_DIR = "./TEST_CRUD_TMP";

int test_pending_append(void) {
    /*
        1. Create a db (pending m = 10)
        2. Append 10 items to pending
        3. Get all items
        4. Try to append 11th item and check if it returns -1
    */

    const int write_batch_m = 10;
    LVOption opt = TEST_OPTION;
    opt.write_batch_m = write_batch_m;

    LVTestAnswer answers[write_batch_m];

    int result = 0;

    Livero* db = NULL;
    if (create_test_db(TEST_DIR, NULL, &opt, &db) < 0) {
        return -1;
    }

    LVOffset_t value_offset = 0;

    for (int i = 0; i < write_batch_m; ++i) {
        int v_len = -1;
        if (i == 0) {
            v_len = 1;
        } else if (i == write_batch_m / 2) {
            v_len = TEST_MAX_VALUE_LEN;
        }
        LVMakeTestAnswerCtx c = {.seq = i,
                                 .value_len = v_len,
                                 .prev_value_offset = value_offset,
                                 .min_year = 2000,
                                 .max_year = 2026,
                                 .fixed_date = NULL,
                                 .alive = 1,
                                 .use_test_vector = 0};
        make_test_answer(&c, &answers[i]);

        int r = lv_pending_append(db, &answers[i].row, answers[i].ternary, answers[i].value, answers[i].row.value_len);

        if (r < 0) {
            fprintf(stderr, "Pending append failed. (Expect success) \n");
            result = -1;
            goto cleanup;
        }

        value_offset += answers[i].row.value_len;
    }
    // Compare every slot so that overflow into any neighbor is detected.
    for (LVCount_t i = 0; i < write_batch_m; ++i) {
        if (compare_answer(&answers[i], lv_n_pending_row(db, i), lv_n_pending_value(db, i),
                           lv_n_pending_ternary(db, i)) < 0) {
            result = -1;
            goto cleanup;
        }
    }

    LVTestAnswer extra;
    LVMakeTestAnswerCtx c = {
        .seq = write_batch_m,
        .prev_value_offset = value_offset,
        .min_year = 2000,
        .max_year = 2026,
        .fixed_date = NULL,
        .alive = 1,
        .use_test_vector = 0,
    };
    make_test_answer(&c, &extra);

    if (lv_pending_append(db, &extra.row, extra.ternary, extra.value, extra.row.value_len) != -1) {
        fprintf(stderr, "Pending append succeeded, (Expect fail)\n");
        result = -1;
        goto cleanup;
    }

    // A rejected append must not modify the existing pending items.
    if (compare_answer(&answers[write_batch_m - 1], lv_last_pending_row(db), lv_n_pending_value(db, write_batch_m - 1),
                       lv_n_pending_ternary(db, write_batch_m - 1)) < 0) {
        result = -1;
        goto cleanup;
    }

cleanup:
    lv_close(db);
    test_end(result, TEST_DIR, "test_pending_append");
    return result;
}

int test_pending_search(void) {
    /*
       1. Create a db
       2. Append 10 items to pending. (Start seq at 5 so that check if it returns the not found err when a non-exsiting
       key is requested)
       3. Get all exsiting items by using lv_pending_search. (it only gives you LVDate, value, value_len and the row's
       offset in the pending buffer)
       4. Try to get non-exsisting items
    */

    const int write_batch_m = 10;
    LVOption opt = TEST_OPTION;
    opt.write_batch_m = write_batch_m;

    LVTestAnswer answers[write_batch_m];
    const int seq_jump = 5;

    int result = 0;

    Livero* db = NULL;
    if (create_test_db(TEST_DIR, NULL, &opt, &db) < 0) {
        return -1;
    }

    LVOffset_t value_offset = 0;

    for (int i = 0; i < write_batch_m; ++i) {
        int v_len = -1;
        if (i == 0) {
            v_len = 1;
        } else if (i == write_batch_m / 2) {
            v_len = TEST_MAX_VALUE_LEN;
        }
        LVMakeTestAnswerCtx c = {
            .seq = i + seq_jump,
            .value_len = v_len,
            .prev_value_offset = value_offset,
            .min_year = 2000,
            .max_year = 2026,
            .fixed_date = NULL,
            .alive = 1,
            .use_test_vector = 0,
        };
        make_test_answer(&c, &answers[i]);

        int r = lv_pending_append(db, &answers[i].row, answers[i].ternary, answers[i].value, answers[i].row.value_len);

        if (r < 0) {
            fprintf(stderr, "Pending append failed. (Expect success) \n");
            result = -1;
            goto cleanup;
        }

        value_offset += answers[i].row.value_len;
    }

    uint8_t value[TEST_MAX_VALUE_LEN] = {0};
    LVGetResult got;
    got.value = value;
    for (int i = 0; i < write_batch_m; ++i) {
        memset(value, 0, TEST_MAX_VALUE_LEN);
        const LVTestAnswer* ans = &answers[i];
        const LVSeq_t seq = ans->row.seq;
        LVStatus s = lv_pending_search(db, seq, &got, NULL);
        if (s != LV_OK) {
            fprintf(stderr, "Failed to get an item. seq: %" PRIu64 ". \n", seq);
            print_status(s);
            result = -1;
            goto cleanup;
        }

        const uint64_t stored_date = lv_convert_date_to_epoch(&got.date);
        if (ans->row.date != stored_date) {
            fprintf(stderr, "Date is incorrect. (Expect: %" PRIu64 " Got: %" PRIu64 "\n", ans->row.date, stored_date);
            result = -1;
            goto cleanup;
        }

        const LVSize16_t stored_v_len = got.value_len;
        if (ans->row.value_len != stored_v_len) {
            fprintf(stderr, "value_len is incorrect. (Expect: %d, Got: %d) seq: %llu \n", ans->row.value_len,
                    stored_v_len, seq);
            result = -1;
            goto cleanup;
        }

        if (memcmp(ans->value, got.value, stored_v_len) != 0) {
            fprintf(stderr, "value is incorrect. seq: %" PRIu64 "\n", seq);
            result = -1;
            goto cleanup;
        }
    }

    for (int i = 0; i < seq_jump; ++i) {
        LVStatus s = lv_pending_search(db, i, NULL, NULL);
        if (s == LV_ERR_NOT_FOUND) {
            // OK
        } else {
            fprintf(stderr, "lv_pending_search failed at seq %" PRIu64 ".\n", (LVSeq_t)i);
            PRINT_ERR_UNEXPECTED_STATUS("lv_pending_search", LV_ERR_NOT_FOUND, s);
            result = -1;
            goto cleanup;
        }
    }

    const LVSeq_t range_overflow_seq = write_batch_m + seq_jump + 1;
    LVStatus s = lv_pending_search(db, range_overflow_seq, NULL, NULL);

    if (s == LV_ERR_NOT_FOUND) {
        // OK
    } else {
        fprintf(stderr,
                "Target seq is out of pending range, but lv_pending_search returned a result. (Seq: %" PRIu64 ")\n",
                range_overflow_seq);
        PRINT_ERR_UNEXPECTED_STATUS("lv_pending_search", LV_ERR_NOT_FOUND, s);
        result = -1;
        goto cleanup;
    }

cleanup:
    lv_close(db);
    test_end(result, TEST_DIR, "test_pending_search");
    return result;
}

int main(void) {
    clean_test_dir(TEST_DIR);
    if (test_pending_append() < 0) {
        return -1;
    }
    if (test_pending_search() < 0) {
        return -1;
    }
    return 0;
}
