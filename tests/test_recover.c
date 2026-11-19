#include <fcntl.h>
#include <stdint.h>
#include <string.h>

#include <assert.h>

#include "livero_internal.h"
#include "livero_types.h"
#include "struct/commit.h"
#include "struct/row.h"
#include "tests_helper.h"

static const char* TEST_DIR = "./TEST_RECOVER_TMP/";

static LVRecoverCtx get_recover_ctx(const Livero* db) {
    assert(db);
    LVRecoverCtx ctx = {.row_file = db->row_file,
                        .ter_file = db->ter_file,
                        .val_file = db->val_file,
                        .commit_file = db->commit_file,
                        .max_disk_bytes = db->config.max_disk_bytes,
                        .vector_dimension = db->config.vector_dimension,
                        .file_def_size = db->option.file_def_size,
                        .read = db->option.io_ops->read};
    return ctx;
}

typedef struct {
    LVSeq_t last_commit_seq;
    LVSeq_t answer_commit_seq;
    LVCommit* answer_commit_out;
    LVCount_t data_count;
    int make_multiple_commits;
} LVTestWDACCtx;
static int write_data_and_commit(Livero* db, LVTestWDACCtx* ctx) {
    assert(db && ctx->data_count > 0);
    LVTestAnswer answers[ctx->data_count];
    LVOffset_t val_offset = 0;
    LVWriteTestDataCtx wtdc = {
        .row_file = db->row_file,
        .ter_file = db->ter_file,
        .val_file = db->val_file,
        .commit_file = db->commit_file,
        .commit_sector = db->commit_sector,
    };
    LVCommit answer_commit = {0};
    for (LVCount_t i = 0; i < ctx->data_count; ++i) {
        LVMakeTestAnswerCtx mtac = {.seq = i,
                                    .max_year = 2026,
                                    .min_year = 2000,
                                    .fixed_date = NULL,
                                    .alive = 1,
                                    .prev_value_offset = val_offset,
                                    .value_len = -1,
                                    .use_test_vector = 0

        };
        make_test_answer(&mtac, &answers[i]);

        if (ctx->make_multiple_commits) {
            const LVSeq_t commit_seq = ctx->last_commit_seq + (LVSeq_t)(i + 1);
            wtdc.commit_seq = commit_seq;
            wtdc.data = &answers[i];
            wtdc.data_count = 1;
            wtdc.prev_count = (LVCount_t)i;
            wtdc.prev_val_bytes = (LVSize32_t)val_offset;
            wtdc.row_start_offset = i * LV_ROW_BYTES;
            wtdc.ternary_start_offset = i * TEST_TERNARY_BYTES;

            const int is_answer = commit_seq == ctx->answer_commit_seq;
            if (write_test_data(&wtdc, is_answer ? &answer_commit : NULL) < 0) {
                fprintf(stderr, "Failed to write data at record seq: %" PRIu64 ", commit seq: %" PRIu64 ".\n",
                        (LVSeq_t)(i), commit_seq);
                return -1;
            }
        }

        val_offset += answers[i].row.value_len;
    }

    if (!ctx->make_multiple_commits) {
        wtdc.commit_seq = LV_COMMIT_SEQ_START + ctx->last_commit_seq;
        wtdc.data = answers;
        wtdc.data_count = ctx->data_count;
        wtdc.prev_count = 0;
        wtdc.prev_val_bytes = 0;
        wtdc.row_start_offset = 0;
        wtdc.ternary_start_offset = 0;

        const int is_answer = wtdc.commit_seq == ctx->answer_commit_seq;
        if (write_test_data(&wtdc, is_answer ?  &answer_commit:NULL) < 0) {
            fprintf(stderr, "Failed to write data at record seq: %" PRIu64 ", commit seq: %" PRIu64 ".\n", (LVSeq_t)0,
                    (LVSeq_t)(LV_COMMIT_SEQ_START));
            return -1;
        }
    }

    if (ctx->answer_commit_out) {
        *ctx->answer_commit_out = answer_commit;
    }

    return 0;
}

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

    LVRecoverCtx ctx = get_recover_ctx(db);

    LVCommit cmt;

    int first_commit_torn_out = 0;
    LVStatus s = lv_recover(&ctx, &cmt, &first_commit_torn_out);

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

    if (create_test_db(TEST_DIR, NULL, NULL, &db) < 0) {
        result = -1;
        goto cleanup;
    }

    LVCommit expected = {0};

    LVTestWDACCtx wdacc = {
        .answer_commit_out = &expected,
        .answer_commit_seq = LV_COMMIT_SEQ_START,
        .data_count = data_count,
        .last_commit_seq = LV_COMMIT_SEQ_ZERO,
        .make_multiple_commits = 0
    };

    if (write_data_and_commit(db, &wdacc) < 0) {
        result = -1;
        goto cleanup;
    }

    LVRecoverCtx recover_ctx = get_recover_ctx(db);
    LVCommit recovered = {0};
    int first_commit_torn_out = 0;

    LVStatus s = lv_recover(&recover_ctx, &recovered, &first_commit_torn_out);

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

    LVTestWDACCtx wdacc = {
        .answer_commit_seq = LV_COMMIT_SEQ_ZERO,
        .answer_commit_out = NULL,
        .make_multiple_commits = 1,
        .last_commit_seq = LV_COMMIT_SEQ_ZERO,
        .data_count = LV_COMMIT_SLOT_M
    };
    if (write_data_and_commit(db, &wdacc) < 0) {
        result = -1;
        goto cleanup;
    }

    LVCommit expected = {0};
    wdacc.answer_commit_seq = (LVSeq_t)LV_COMMIT_SLOT_M+1;
    wdacc.answer_commit_out = &expected;
    wdacc.last_commit_seq = (LVSeq_t)LV_COMMIT_SLOT_M;
    wdacc.make_multiple_commits = 0;
    wdacc.data_count = LV_COMMIT_SLOT_M;
    if (write_data_and_commit(db, &wdacc) < 0) {
        result = -1;
        goto cleanup;
    }
    LVRecoverCtx recover_ctx = get_recover_ctx(db);
    LVCommit recovered = {0};

    int first_commit_torn_out = 0;

    LVStatus s = lv_recover(&recover_ctx, &recovered, &first_commit_torn_out);

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

    LVCommit expected = {0};
    LVTestWDACCtx wdacc = {
        .answer_commit_out = &expected,
        .answer_commit_seq = LV_COMMIT_SLOT_M-1,
        .last_commit_seq = LV_COMMIT_SEQ_ZERO,
        .make_multiple_commits = 1,
        .data_count = LV_COMMIT_SLOT_M
    };
    if(write_data_and_commit(db, &wdacc) < 0){
        result = -1;
        goto cleanup;
    }

    /*
        Corrupt the Mth commit
    */
    const uint64_t trash = 0xdeadbeef;
    LVStatus s = TEST_IO_OPS.write(db->commit_file, &trash, sizeof(trash),
                               commit_slot_index(cmt_seq_to_corrupt) * LV_COMMIT_SECTOR_BYTES);

    if (s != LV_OK) {
        PRINT_ERR_UNEXPECTED_STATUS("lv_write_file", LV_OK, s);
        result = -1;
        goto cleanup;
    }

    LVCommit recovered = {0};
    LVRecoverCtx recover_ctx = get_recover_ctx(db);
    int first_commit_torn_out = 0;

    s = lv_recover(&recover_ctx, &recovered, &first_commit_torn_out);

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

    LVTestWDACCtx wdacc = {
        .answer_commit_out = NULL,
        .answer_commit_seq = LV_COMMIT_SEQ_ZERO,
        .last_commit_seq = LV_COMMIT_SEQ_ZERO,
        .make_multiple_commits = 1,
        .data_count = LV_COMMIT_SLOT_M
    };
    if(write_data_and_commit(db, &wdacc) < 0){
        result = -1;
        goto cleanup;
    }

    /*
    Corrupt the second commit
*/
    const uint64_t trash = 0xdeadbeef;
    LVStatus s = TEST_IO_OPS.write(db->commit_file, &trash, sizeof(trash),
                               commit_slot_index(cmt_seq_to_corrupt) * LV_COMMIT_SECTOR_BYTES);

    if (s != LV_OK) {
        PRINT_ERR_UNEXPECTED_STATUS("lv_write_file", LV_OK, s);
        result = -1;
        goto cleanup;
    }

    LVCommit recovered = {0};
    LVRecoverCtx recover_ctx = get_recover_ctx(db);
    int first_commit_torn_out = 0;

    s = lv_recover(&recover_ctx, &recovered, &first_commit_torn_out);

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

    LVTestWDACCtx wdacc = {
        .answer_commit_out = NULL,
        .answer_commit_seq = LV_COMMIT_SEQ_ZERO,
        .make_multiple_commits = 1,
        .last_commit_seq = LV_COMMIT_SEQ_ZERO,
        .data_count = LV_COMMIT_SLOT_M
    };
    if(write_data_and_commit(db, &wdacc) < 0){
        result = -1;
        goto cleanup;
    }


    const uint64_t trash = 0xCADFCADF;
    for (int i = 0; i < LV_COMMIT_SLOT_M; ++i) {
        LVStatus s = TEST_IO_OPS.write(db->commit_file, &trash, sizeof(trash), i * LV_COMMIT_SECTOR_BYTES);
        if (s != LV_OK) {
            PRINT_ERR_UNEXPECTED_STATUS("lv_write_file", LV_OK, s);
            result = -1;
            goto cleanup;
        }
    }

    LVCommit recovered = {0};
    LVRecoverCtx recover_ctx = get_recover_ctx(db);
    int first_commit_torn_out = 0;

    LVStatus s = lv_recover(&recover_ctx, &recovered, &first_commit_torn_out);

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

    LVTestWDACCtx wdacc = {
        .answer_commit_out = NULL,
        .answer_commit_seq = LV_COMMIT_SEQ_ZERO,
        .last_commit_seq = LV_COMMIT_SEQ_ZERO,
        .make_multiple_commits = 1,
        .data_count = LV_COMMIT_SLOT_M
    };
    
    if(write_data_and_commit(db, &wdacc) < 0){
        result = -1;
        goto cleanup;
    }

    /*
        Corrupt last row which is covered by the last commit
    */
    const uint64_t trash = 0xbeefbeef;
    LVStatus s = TEST_IO_OPS.write(db->row_file, &trash, sizeof(trash), (LV_COMMIT_SLOT_M - 1) * LV_ROW_BYTES);
    if (s != LV_OK) {
        PRINT_ERR_UNEXPECTED_STATUS("lv_write_file", LV_OK, s);
        result = -1;
        goto cleanup;
    }

    LVCommit recovered = {0};
    LVRecoverCtx recover_ctx = get_recover_ctx(db);
    int first_commit_torn_out = 0;
    s = lv_recover(&recover_ctx, &recovered, &first_commit_torn_out);

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

int test_recover_8(void) {
    /*
        1. Create a db
        2. Write a first commit
        3. Corrupt the first commit
        4. Check if lv_recover retruns an empty commit and set first_commit_torn_out to 1
    */

    int result = 0;
    Livero* db = NULL;

    if (create_test_db(TEST_DIR, NULL, NULL, &db) < 0) {
        result = -1;
        goto cleanup;
    }

    LVTestWDACCtx wdacc = {
        .answer_commit_out = NULL,
        .answer_commit_seq = LV_COMMIT_SEQ_ZERO,
        .last_commit_seq = LV_COMMIT_SEQ_ZERO,
        .make_multiple_commits = 0,
        .data_count = 1
    };

    if(write_data_and_commit(db, &wdacc) < 0){
        result = -1;
        goto cleanup;
    }

    const uint64_t trash = 0xbeefbeef;
    LVStatus s = TEST_IO_OPS.write(db->commit_file, &trash, sizeof(trash), 0);
    if (s != LV_OK) {
        PRINT_ERR_UNEXPECTED_STATUS("lv_write_file", LV_OK, s);
        result = -1;
        goto cleanup;
    }

    LVCommit recovered = {0};
    LVRecoverCtx recover_ctx = get_recover_ctx(db);
    int first_commit_torn_out = 0;
    s = lv_recover(&recover_ctx, &recovered, &first_commit_torn_out);

    const LVCommit empty_commit = {0};
    if (memcmp(&empty_commit, &recovered, sizeof(recovered)) != 0) {
        fprintf(stderr, "lv_recover returned a non-empty commit for the first torn write.\n");
        result = -1;
        goto cleanup;
    }

    if (!first_commit_torn_out) {
        fprintf(stderr, "first_commit_torn_out isn't turned on. \n");
        result = -1;
        goto cleanup;
    }
cleanup:
    lv_close(db);
    test_end(result, TEST_DIR, "test_recover_8s");
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

    if (test_recover_8() < 0) {
        return -1;
    }

    return 0;
}
