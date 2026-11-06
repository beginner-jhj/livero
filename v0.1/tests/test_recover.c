#include <fcntl.h>
#include <stdint.h>
#include <string.h>

#include "file/file.h"
#include "livero_internal.h"
#include "livero_types.h"
#include "struct/commit.h"
#include "struct/row.h"
#include "tests_helper.h"

static const char* TEST_DIR = "./TEST_RECOVER_TMP/";

int test_recover_1(void) {
    /*
        1. create a initialized commit.lv
        2. check if lv_recover returns an empty commit
    */

    int result = 0;
    Livero* db = NULL;

    if (create_test_db(TEST_DIR, NULL, NULL, &db) < 0) {
        result = -1;
        goto cleanup;
    }

    LVRecoverCtx ctx = {
        .row_fd = db->row_fd,
        .val_fd = db->val_fd,
        .vec_fd = db->vec_fd,
        .commit_fd = db->commit_fd,
        .max_disk_bytes = db->max_disk_bytes,
        .vector_dim = db->vector_dimension,
    };

    LVCommit cmt;

    LVStatus s = lv_recover(&ctx, &cmt);

    if (s != LV_OK) {
        PRINT_ERR_UNEXPECTED_STATUS("test_recover_1", LV_OK, s);
        result = -1;
        goto cleanup;
    }

    LVCommit empty_commit = {0};

    if (memcmp(&empty_commit, &cmt, sizeof(cmt)) != 0) {
        fprintf(stderr, "Commit is not empty. \n");
        result = -1;
        goto cleanup;
    }

cleanup:
    lv_close(db);
    test_end(result, TEST_DIR, "test_recover_1");
    return result;
}

int test_recover_2(void) {
    /*
        1. Create a db.
        2. Write a commit (10 valid data).
        3. Recover the commit
    */

    int result = 0;
    const LVCount_t data_count = 10;
    Livero* db = NULL;

    LVTestAnswer answers[data_count];

    if (create_test_db(TEST_DIR, NULL, NULL, &db) < 0) {
        result = -1;
        goto cleanup;
    }

    LVOffset_t value_offset = 0;
    for (LVCount_t i = 0; i < data_count; ++i) {
        LVMakeTestAnswerCtx c = {
            .alive = i % 2,
            .max_year = 2026,
            .min_year = 2000,
            .prev_value_offset = value_offset,
            .value_len = -1,  // random
            .seq = i,
        };

        make_test_answer(&c, &answers[i]);
        value_offset += answers[i].row.value_len;
    }
    const LVSeq_t commit_seq = 1;

    LVWriteTestDataCtx wtdc = {.data = answers,
                               .data_count = data_count,
                               .prev_count = 0,
                               .prev_val_bytes = 0,
                               .commit_seq = commit_seq,
                               .row_start_offset = 0,
                               .vec_start_offset = 0,
                               .row_fd = db->row_fd,
                               .val_fd = db->val_fd,
                               .vec_fd = db->vec_fd,
                               .commit_fd = db->commit_fd};

    LVCommit expected = {0};

    if (write_test_data(&wtdc, &expected) < 0) {
        result = -1;
        goto cleanup;
    }

    LVRecoverCtx recover_ctx = {.commit_fd = db->commit_fd,
                                .row_fd = db->row_fd,
                                .val_fd = db->val_fd,
                                .vec_fd = db->vec_fd,
                                .max_disk_bytes = db->max_disk_bytes,
                                .vector_dim = db->vector_dimension};
    LVCommit recovered = {0};

    LVStatus s = lv_recover(&recover_ctx, &recovered);

    if (s != LV_OK) {
        PRINT_ERR_UNEXPECTED_STATUS("lv_recover", LV_OK, s);
        result = -1;
        goto cleanup;
    }

    if (check_commit(&recovered, &expected) != 0) {
        result = -1;
        goto cleanup;
    }

cleanup:
    lv_close(db);
    test_end(result, TEST_DIR, "test_recover_2");
    return result;
}

int test_recover_3(void) {
    /*
        1. Create a db
        2. Write M commits (record_count = 1)
        3. Write M+1 commit whose index is 0.
        4. Check if lv_recover returns M+1th commit
    */

    int result = 0;
    Livero* db = NULL;

    if (create_test_db(TEST_DIR, NULL, NULL, &db) < 0) {
        result = -1;
        goto cleanup;
    }

    LVTestAnswer answer = {0};
    LVOffset_t val_offset = 0;
    LVWriteTestDataCtx wtdc = {
        .row_fd = db->row_fd,
        .vec_fd = db->vec_fd,
        .val_fd = db->val_fd,
        .commit_fd = db->commit_fd,
    };
    for (int i = 0; i < LV_COMMIT_SLOT_M; ++i) {
        LVMakeTestAnswerCtx mtac = {
            .seq = i,
            .max_year = 2026,
            .min_year = 2000,
            .alive = 1,
            .prev_value_offset = val_offset,
            .value_len = -1,
        };
        make_test_answer(&mtac, &answer);

        wtdc.commit_seq = (LVSeq_t)(i + 1);
        wtdc.data = &answer;
        wtdc.data_count = 1;
        wtdc.prev_count = (LVCount_t)i;
        wtdc.prev_val_bytes = (LVSize32_t)val_offset;
        wtdc.row_start_offset = i * LV_ROW_BYTES;
        wtdc.vec_start_offset = i * TEST_TERNARY_BYTES;

        if (write_test_data(&wtdc, NULL) < 0) {
            result = -1;
            goto cleanup;
        }

        val_offset += answer.row.value_len;
    }

    LVTestAnswer last_ans;
    LVMakeTestAnswerCtx mtac = {
        .seq = LV_COMMIT_SLOT_M,
        .max_year = 2026,
        .min_year = 2000,
        .alive = 1,
        .prev_value_offset = val_offset,
        .value_len = -1,
    };
    make_test_answer(&mtac, &last_ans);

    wtdc.commit_seq = LV_COMMIT_SLOT_M + 1;
    wtdc.data = &last_ans;
    wtdc.data_count = 1;
    wtdc.prev_count = LV_COMMIT_SLOT_M;
    wtdc.prev_val_bytes = (LVSize32_t)val_offset;
    wtdc.row_start_offset = (LV_COMMIT_SLOT_M)*LV_ROW_BYTES;
    wtdc.vec_start_offset = (LV_COMMIT_SLOT_M)*TEST_TERNARY_BYTES;

    LVCommit expected = {0};

    if (write_test_data(&wtdc, &expected) < 0) {
        result = -1;
        goto cleanup;
    }

    LVRecoverCtx recover_ctx = {.commit_fd = db->commit_fd,
                                .row_fd = db->row_fd,
                                .val_fd = db->val_fd,
                                .vec_fd = db->vec_fd,
                                .max_disk_bytes = db->max_disk_bytes,
                                .vector_dim = db->vector_dimension};
    LVCommit recovered = {0};

    LVStatus s = lv_recover(&recover_ctx, &recovered);

    if (s != LV_OK) {
        PRINT_ERR_UNEXPECTED_STATUS("lv_recover", LV_OK, s);
        result = -1;
        goto cleanup;
    }

    if (check_commit(&recovered, &expected) != 0) {
        result = -1;
        goto cleanup;
    }

cleanup:
    lv_close(db);
    test_end(result, TEST_DIR, "test_recover_3");
    return result;
}

int test_recover_4(void) {
    /*
        1. Create a db
        2. Write M commits (record_count = 1)
        3. Corrupt Mth commit
        4. Check if lv_recover returns the M-1th commit
    */

    int result = 0;
    Livero* db = NULL;
    const LVSeq_t cmt_seq_to_corrupt = LV_COMMIT_SLOT_M;

    if (create_test_db(TEST_DIR, NULL, NULL, &db) < 0) {
        result = -1;
        goto cleanup;
    }

    LVTestAnswer answer = {0};
    LVOffset_t val_offset = 0;
    LVWriteTestDataCtx wtdc = {
        .row_fd = db->row_fd,
        .vec_fd = db->vec_fd,
        .val_fd = db->val_fd,
        .commit_fd = db->commit_fd,
    };

    LVCommit expected = {0};
    for (int i = 0; i < LV_COMMIT_SLOT_M; ++i) {
        LVMakeTestAnswerCtx mtac = {
            .seq = i,
            .max_year = 2026,
            .min_year = 2000,
            .alive = 1,
            .prev_value_offset = val_offset,
            .value_len = -1,
        };
        make_test_answer(&mtac, &answer);

        wtdc.commit_seq = (LVSeq_t)(i + 1);
        wtdc.data = &answer;
        wtdc.data_count = 1;
        wtdc.prev_count = (LVCount_t)i;
        wtdc.prev_val_bytes = (LVSize32_t)val_offset;
        wtdc.row_start_offset = i * LV_ROW_BYTES;
        wtdc.vec_start_offset = i * TEST_TERNARY_BYTES;

        if (i == LV_COMMIT_SLOT_M - 2) {  // save M-1th commit which is the answer
            if (write_test_data(&wtdc, &expected) < 0) {
                result = -1;
                goto cleanup;
            }
        } else {
            if (write_test_data(&wtdc, NULL) < 0) {
                result = -1;
                goto cleanup;
            }
        }

        val_offset += answer.row.value_len;
    }

    /*
        Corrupt the Mth commit
    */
    const uint64_t trash = 0xdeadbeef;
    LVStatus s = lv_write_file(db->commit_fd, &trash, sizeof(trash),
                               commit_slot_index(cmt_seq_to_corrupt) * LV_COMMIT_SECTOR_BYTES);

    if (s != LV_OK) {
        PRINT_ERR_UNEXPECTED_STATUS("lv_write_file", LV_OK, s);
        result = -1;
        goto cleanup;
    }

    LVCommit recovered = {0};
    LVRecoverCtx recover_ctx = {
        .commit_fd = db->commit_fd,
        .val_fd = db->val_fd,
        .vec_fd = db->vec_fd,
        .row_fd = db->row_fd,
        .max_disk_bytes = db->max_disk_bytes,
        .vector_dim = db->vector_dimension,
    };
    s = lv_recover(&recover_ctx, &recovered);

    if (check_commit(&recovered, &expected) < 0) {
        result = -1;
        goto cleanup;
    }

cleanup:
    lv_close(db);
    test_end(result, TEST_DIR, "test_recover_4");
    return result;
}

int test_recover_5(void) {
    /*
        1. Create a db.
        2. Write M commits
        3. Corrupt 2th commit which is an unexpected media crash. (An expected crash is a crash on the (best idx +1)th
       commit)
        4. Check if lv_recover returns LV_ERR_MEDIA_CRASHED
    */

    int result = 0;
    Livero* db = NULL;
    const LVSeq_t cmt_seq_to_corrupt = 2;

    if (create_test_db(TEST_DIR, NULL, NULL, &db) < 0) {
        result = -1;
        goto cleanup;
    }

    LVTestAnswer answer = {0};
    LVOffset_t val_offset = 0;
    LVWriteTestDataCtx wtdc = {
        .row_fd = db->row_fd,
        .vec_fd = db->vec_fd,
        .val_fd = db->val_fd,
        .commit_fd = db->commit_fd,
    };

    for (int i = 0; i < LV_COMMIT_SLOT_M; ++i) {
        LVMakeTestAnswerCtx mtac = {
            .seq = i,
            .max_year = 2026,
            .min_year = 2000,
            .alive = 1,
            .prev_value_offset = val_offset,
            .value_len = -1,
        };
        make_test_answer(&mtac, &answer);

        wtdc.commit_seq = (LVSeq_t)(i + 1);
        wtdc.data = &answer;
        wtdc.data_count = 1;
        wtdc.prev_count = (LVCount_t)i;
        wtdc.prev_val_bytes = (LVSize32_t)val_offset;
        wtdc.row_start_offset = i * LV_ROW_BYTES;
        wtdc.vec_start_offset = i * TEST_TERNARY_BYTES;

        if (write_test_data(&wtdc, NULL) < 0) {
            result = -1;
            goto cleanup;
        }

        val_offset += answer.row.value_len;
    }

    /*
    Corrupt the second commit
*/
    const uint64_t trash = 0xdeadbeef;
    LVStatus s = lv_write_file(db->commit_fd, &trash, sizeof(trash),
                               commit_slot_index(cmt_seq_to_corrupt) * LV_COMMIT_SECTOR_BYTES);

    if (s != LV_OK) {
        PRINT_ERR_UNEXPECTED_STATUS("lv_write_file", LV_OK, s);
        result = -1;
        goto cleanup;
    }

    LVCommit recovered = {0};
    LVRecoverCtx recover_ctx = {
        .commit_fd = db->commit_fd,
        .val_fd = db->val_fd,
        .vec_fd = db->vec_fd,
        .row_fd = db->row_fd,
        .max_disk_bytes = db->max_disk_bytes,
        .vector_dim = db->vector_dimension,
    };
    s = lv_recover(&recover_ctx, &recovered);

    if (s == LV_ERR_MEDIA_CRASHED) {
        // ok
    } else {
        PRINT_ERR_UNEXPECTED_STATUS("lv_flush", LV_ERR_MEDIA_CRASHED, s);
        result = -1;
        goto cleanup;
    }

cleanup:
    lv_close(db);
    test_end(result, TEST_DIR, "test_recover_5");
    return result;
}

int test_recover_6(void) {
    /*
        1. Create a db
        2. Write M commits
        3. Corrupt all commits
        4. Check if lv_recover returns LV_ERR_MEDIA_CRASHED
    */

    int result = 0;
    Livero* db = NULL;

    if (create_test_db(TEST_DIR, NULL, NULL, &db) < 0) {
        result = -1;
        goto cleanup;
    }

    LVTestAnswer answer = {0};
    LVOffset_t val_offset = 0;
    LVWriteTestDataCtx wtdc = {
        .row_fd = db->row_fd,
        .vec_fd = db->vec_fd,
        .val_fd = db->val_fd,
        .commit_fd = db->commit_fd,
    };

    for (int i = 0; i < LV_COMMIT_SLOT_M; ++i) {
        LVMakeTestAnswerCtx mtac = {
            .seq = i,
            .max_year = 2026,
            .min_year = 2000,
            .alive = 1,
            .prev_value_offset = val_offset,
            .value_len = -1,
        };
        make_test_answer(&mtac, &answer);

        wtdc.commit_seq = (LVSeq_t)(i + 1);
        wtdc.data = &answer;
        wtdc.data_count = 1;
        wtdc.prev_count = (LVCount_t)i;
        wtdc.prev_val_bytes = (LVSize32_t)val_offset;
        wtdc.row_start_offset = i * LV_ROW_BYTES;
        wtdc.vec_start_offset = i * TEST_TERNARY_BYTES;

        if (write_test_data(&wtdc, NULL) < 0) {
            result = -1;
            goto cleanup;
        }

        val_offset += answer.row.value_len;
    }

    const uint64_t trash = 0xCADFCADF;
    for (int i = 0; i < LV_COMMIT_SLOT_M; ++i) {
        LVStatus s = lv_write_file(db->commit_fd, &trash, sizeof(trash), i * LV_COMMIT_SECTOR_BYTES);
        if (s != LV_OK) {
            PRINT_ERR_UNEXPECTED_STATUS("lv_write_file", LV_OK, s);
            result = -1;
            goto cleanup;
        }
    }

    LVCommit recovered = {0};
    LVRecoverCtx recover_ctx = {
        .commit_fd = db->commit_fd,
        .val_fd = db->val_fd,
        .vec_fd = db->vec_fd,
        .row_fd = db->row_fd,
        .max_disk_bytes = db->max_disk_bytes,
        .vector_dim = db->vector_dimension,
    };
    LVStatus s = lv_recover(&recover_ctx, &recovered);

    if (s == LV_ERR_MEDIA_CRASHED) {
        // ok
    } else {
        PRINT_ERR_UNEXPECTED_STATUS("lv_flush", LV_ERR_MEDIA_CRASHED, s);
        result = -1;
        goto cleanup;
    }

cleanup:
    lv_close(db);
    test_end(result, TEST_DIR, "test_recover_6");
    return result;
}

int test_recover_7(void) {
    /*
        1. Create a db
        2. Write M commits
        3. Corrupt Mth commit's batch data
        4. Check if lv_recover returns LV_ERR_DATA_CORRUPT
    */
    int result = 0;
    Livero* db = NULL;

    if (create_test_db(TEST_DIR, NULL, NULL, &db) < 0) {
        result = -1;
        goto cleanup;
    }

    LVTestAnswer answer = {0};
    LVOffset_t val_offset = 0;
    LVWriteTestDataCtx wtdc = {
        .row_fd = db->row_fd,
        .vec_fd = db->vec_fd,
        .val_fd = db->val_fd,
        .commit_fd = db->commit_fd,
    };

    for (int i = 0; i < LV_COMMIT_SLOT_M; ++i) {
        LVMakeTestAnswerCtx mtac = {
            .seq = i,
            .max_year = 2026,
            .min_year = 2000,
            .alive = 1,
            .prev_value_offset = val_offset,
            .value_len = -1,
        };
        make_test_answer(&mtac, &answer);

        wtdc.commit_seq = (LVSeq_t)(i + 1);
        wtdc.data = &answer;
        wtdc.data_count = 1;
        wtdc.prev_count = (LVCount_t)i;
        wtdc.prev_val_bytes = (LVSize32_t)val_offset;
        wtdc.row_start_offset = i * LV_ROW_BYTES;
        wtdc.vec_start_offset = i * TEST_TERNARY_BYTES;

        if (write_test_data(&wtdc, NULL) < 0) {
            result = -1;
            goto cleanup;
        }

        val_offset += answer.row.value_len;
    }

    /*
        Corrupt last row which is covered by the last commit
    */
    const uint64_t trash = 0xbeefbeef;
    LVStatus s = lv_write_file(db->row_fd, &trash, sizeof(trash), (LV_COMMIT_SLOT_M-1)*LV_ROW_BYTES);
    if (s != LV_OK) {
        PRINT_ERR_UNEXPECTED_STATUS("lv_write_file", LV_OK, s);
        result = -1;
        goto cleanup;
    }

    LVCommit recovered = {0};
    LVRecoverCtx recover_ctx = {
        .commit_fd = db->commit_fd,
        .val_fd = db->val_fd,
        .vec_fd = db->vec_fd,
        .row_fd = db->row_fd,
        .max_disk_bytes = db->max_disk_bytes,
        .vector_dim = db->vector_dimension,
    };
    s = lv_recover(&recover_ctx, &recovered);

    if (s == LV_ERR_DATA_CORRUPT) {
        // ok
    } else {
        PRINT_ERR_UNEXPECTED_STATUS("lv_flush", LV_ERR_DATA_CORRUPT, s);
        result = -1;
        goto cleanup;
    }

cleanup:
    lv_close(db);
    test_end(result, TEST_DIR, "test_recover_7");
    return result;
}

int main(void) {
    if (test_recover_1() < 0) {
        return -1;
    }

    if (test_recover_2() < 0) {
        return -1;
    }

    if (test_recover_3() < 0) {
        return -1;
    }

    if (test_recover_4() < 0) {
        return -1;
    }

    if (test_recover_5() < 0) {
        return -1;
    }

    if (test_recover_6() < 0) {
        return -1;
    }

    if (test_recover_7() < 0) {
        return -1;
    }

    return 0;
}
