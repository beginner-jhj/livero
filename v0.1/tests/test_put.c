#include <stdint.h>
#include <string.h>

#include "livero.h"
#include "livero_internal.h"
#include "livero_types.h"
#include "struct/row.h"
#include "tests_helper.h"

static const char* TEST_DIR = "./TEST_PUT_TMP";

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
        LVMakeTestAnswerCtx c = {
            .seq = i,
            .value_len = v_len,
            .prev_value_offset = value_offset,
            .min_year = 2000,
            .max_year = 2026,
            .alive = 1,
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
        .alive = 1,
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
    test_end(result, TEST_DIR, "test_pending_append_1");
    return result;
}

int main(void) {
    clean_test_dir(TEST_DIR);
    if (test_pending_append() < 0) {
        return -1;
    }
    return 0;
}
