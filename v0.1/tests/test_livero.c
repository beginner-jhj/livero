#include <assert.h>
#include <string.h>

#include "helper/helper.h"
#include "livero_types.h"
#include "struct/row.h"
#include "tests_helper.h"
#include "vector/ternary.h"

static const char* TEST_DIR = "./TEST_LIVERO_TMP/";

/*
    Public-API tests ported from the v0 suite (test_livero.c).

    v0 test        -> this file
    test_create    -> test_create
    test_open      -> test_open
    test_1         -> test_put_get
    test_2         -> test_reopen
    test_3         -> test_delete        (a deleted seq now returns LV_ERR_DELETED, not LV_ERR_NOT_FOUND)
    test_4         -> test_query_topk
    test_5         -> test_query_date
    test_6         -> test_query_tombstone

    Every test keeps an answer table of what it put, so each check compares
    livero's output with what the test knows it wrote.
*/

/* ------------------------------------------------------------------------- */
/* Answer table                                                              */
/* ------------------------------------------------------------------------- */

// Every record put in the current test, in put order.
#define TEST_MAX_ANSWERS 1024
static LVTestAnswer ANSWERS[TEST_MAX_ANSWERS];
static LVCount_t ANSWER_COUNT = 0;
static LVSize32_t ANSWER_VALUE_BYTES = 0;

// The first put after a reopen may skip seqs reserved by the previous session,
// so it only has to be larger than the last seq. Every other put must be last + 1.
static int SEQ_GAP_ALLOWED = 0;

// Vector shared by every query ground truth. It lives apart from
// TEST_INT8_VEC_BUFFER because each dummy put overwrites that buffer.
static int8_t QUERY_VEC[TEST_VECTOR_DIMENSION];

static void answers_reset(void) {
    ANSWER_COUNT = 0;
    ANSWER_VALUE_BYTES = 0;
    SEQ_GAP_ALLOWED = 0;
}

// Returns the answer stored for `seq`, or NULL when that seq was never put.
// Linear search, because seqs may skip ahead after a reopen.
static LVTestAnswer* answer_of(const LVSeq_t seq) {
    for (LVCount_t i = 0; i < ANSWER_COUNT; ++i) {
        if (ANSWERS[i].row.seq == seq) {
            return &ANSWERS[i];
        }
    }
    return NULL;
}

static LVSeq_t last_seq(void) {
    assert(ANSWER_COUNT > 0);
    return ANSWERS[ANSWER_COUNT - 1].row.seq;
}

/* ------------------------------------------------------------------------- */
/* Steps                                                                     */
/* ------------------------------------------------------------------------- */

/*
    Put one record and record it in the answer table.

    vector:     NULL puts a fresh random vector.
    fixed_date: NULL picks a random date in [min_year, max_year].
*/
static int put_one(Livero* db, int8_t* vector, const uint16_t min_year, const uint16_t max_year,
                   const LVDate* fixed_date, LVSeq_t* seq_out) {
    if (ANSWER_COUNT >= TEST_MAX_ANSWERS) {
        fprintf(stderr, "[put] answer table is full (cap: %d).\n", TEST_MAX_ANSWERS);
        return -1;
    }

    LVTestAnswer* ans = &ANSWERS[ANSWER_COUNT];
    LVDate* FIXED_DATE = fixed_date ? (LVDate*)fixed_date : NULL;
    LVMakeTestAnswerCtx c = {
        .seq = LV_SEQ_START + ANSWER_COUNT,
        .value_len = -1,
        .min_year = min_year,
        .max_year = max_year,
        .fixed_date = FIXED_DATE,
        .prev_value_offset = ANSWER_VALUE_BYTES,
        .alive = 1,
        .use_test_vector = 0,
    };
    make_test_answer(&c, ans);

    LVDate date = {0};
    lv_convert_epoch_to_date(ans->row.date, &date);

    if (!vector) {
        fill_test_int8_vector();
        vector = TEST_INT8_VEC_BUFFER;
    }

    // Poison the output so a put that forgets to set it cannot pass by accident.
    LVSeq_t seq = (LVSeq_t)-1;
    const LVStatus s = lv_put(db, vector, ans->value, ans->row.value_len, &date, &seq);
    if (s != LV_OK) {
        fprintf(stderr, "[put] failed at answer index %" PRIu32 ".\n", ANSWER_COUNT);
        PRINT_ERR_UNEXPECTED_STATUS("lv_put", LV_OK, s);
        return -1;
    }

    // Seqs are monotonic and never reused.
    if (ANSWER_COUNT == 0) {
        if (seq != LV_SEQ_START) {
            fprintf(stderr, "[put] first seq of a new db is wrong. ");
            PRINT_ERR_INVALID("seq", "%" PRIu64, (LVSeq_t)LV_SEQ_START, seq);
            return -1;
        }
    } else if (SEQ_GAP_ALLOWED) {
        if (seq <= last_seq()) {
            fprintf(stderr, "[put] seq after reopen must be larger than %" PRIu64 ", got %" PRIu64 ".\n", last_seq(),
                    seq);
            return -1;
        }
    } else if (seq != last_seq() + 1) {
        fprintf(stderr, "[put] seq must follow the previous one. ");
        PRINT_ERR_INVALID("seq", "%" PRIu64, last_seq() + 1, seq);
        return -1;
    }

    ans->row.seq = seq;
    SEQ_GAP_ALLOWED = 0;
    ANSWER_VALUE_BYTES += ans->row.value_len;
    ANSWER_COUNT++;

    if (seq_out) {
        *seq_out = seq;
    }
    return 0;
}

// Put `count` records with random vectors, which almost never score close to QUERY_VEC.
static int put_dummies(Livero* db, const LVCount_t count, const uint16_t min_year, const uint16_t max_year) {
    for (LVCount_t i = 0; i < count; ++i) {
        if (put_one(db, NULL, min_year, max_year, NULL, NULL) < 0) {
            return -1;
        }
    }
    return 0;
}

// Put `count` records whose vector equals QUERY_VEC, so they score highest for that query.
static int put_ground_truths(Livero* db, const LVCount_t count, LVSeq_t* seqs_out) {
    for (LVCount_t i = 0; i < count; ++i) {
        if (put_one(db, QUERY_VEC, 2000, 2026, NULL, &seqs_out[i]) < 0) {
            return -1;
        }
    }
    return 0;
}

/*
    Put ground truths on both sides of the flush boundary:

        [first half GT][dummies up to 2 full batches][second half GT][tail dummies]

    The first half is flushed to disk; the second half stays in the pending buffer.
*/
static int put_query_layout(Livero* db, const LVCount_t write_batch_m, const LVCount_t gt_count,
                            const LVCount_t tail_dummies, LVSeq_t* gt_seqs_out) {
    const LVCount_t first_half = gt_count / 2;
    const LVCount_t second_half = gt_count - first_half;

    // Both assumptions are needed for the halves to land where the comment says.
    assert(first_half < 2 * write_batch_m);
    assert(second_half + tail_dummies < write_batch_m);

    if (put_ground_truths(db, first_half, gt_seqs_out) < 0) {
        return -1;
    }
    if (put_dummies(db, 2 * write_batch_m - first_half, 2000, 2026) < 0) {
        return -1;
    }
    if (put_ground_truths(db, second_half, gt_seqs_out + first_half) < 0) {
        return -1;
    }
    if (put_dummies(db, tail_dummies, 2000, 2026) < 0) {
        return -1;
    }
    return 0;
}

// Get every record in the answer table. Deleted ones must return LV_ERR_DELETED.
static int check_all_gets(Livero* db, const char* phase) {
    LVGetResult got = {.value = TEST_VALUE_BUFFER};

    for (LVCount_t i = 0; i < ANSWER_COUNT; ++i) {
        const LVTestAnswer* ans = &ANSWERS[i];
        const LVStatus expected = ans->row.tombstone_flag ? LV_ERR_DELETED : LV_OK;

        // Poison outputs so fields lv_get forgets to fill are caught by the compare.
        memset(TEST_VALUE_BUFFER, 0xAA, sizeof(TEST_VALUE_BUFFER));
        got.value_len = 0;
        memset(&got.date, 0, sizeof(got.date));

        const LVStatus s = lv_get(db, ans->row.seq, &got);
        if (s != expected) {
            fprintf(stderr, "[%s] lv_get at seq %" PRIu64 ".\n", phase, ans->row.seq);
            PRINT_ERR_UNEXPECTED_STATUS("lv_get", expected, s);
            return -1;
        }

        if (s == LV_OK && check_get_result(ans, &got) < 0) {
            fprintf(stderr, "[%s] lv_get returned a wrong record.\n", phase);
            return -1;
        }
    }
    return 0;
}

// A seq that was never put must return LV_ERR_NOT_FOUND.
static int check_get_missing(Livero* db, const LVSeq_t seq, const char* phase) {
    LVGetResult got = {.value = TEST_VALUE_BUFFER};
    const LVStatus s = lv_get(db, seq, &got);
    if (s != LV_ERR_NOT_FOUND) {
        fprintf(stderr, "[%s] lv_get at missing seq %" PRIu64 ".\n", phase, seq);
        PRINT_ERR_UNEXPECTED_STATUS("lv_get", LV_ERR_NOT_FOUND, s);
        return -1;
    }
    return 0;
}

// Delete a seq and mark it in the answer table so later checks expect a tombstone.
static int delete_one(Livero* db, const LVSeq_t seq) {
    LVTestAnswer* ans = answer_of(seq);
    if (!ans) {
        fprintf(stderr, "[delete] seq %" PRIu64 " was never put; fix the test input.\n", seq);
        return -1;
    }

    const LVStatus s = lv_delete(db, seq);
    if (s != LV_OK) {
        fprintf(stderr, "[delete] failed at seq %" PRIu64 ".\n", seq);
        PRINT_ERR_UNEXPECTED_STATUS("lv_delete", LV_OK, s);
        return -1;
    }

    ans->row.tombstone_flag = LV_TOMBSTONE_FLAG;
    return 0;
}

// Close and reopen. On failure *db is NULL, so the caller's cleanup stays safe.
static int reopen_db(Livero** db, const LVOption* option) {
    lv_close(*db);
    *db = NULL;

    const LVStatus s = lv_open(TEST_DIR, option, db);
    if (s != LV_OK) {
        PRINT_ERR_UNEXPECTED_STATUS("lv_open", LV_OK, s);
        return -1;
    }
    if (!*db) {
        fprintf(stderr, "[reopen] lv_open returned LV_OK but did not set the handle.\n");
        return -1;
    }

    SEQ_GAP_ALLOWED = 1;
    return 0;
}

// Stat right after create or open: every record on disk, nothing pending.
// Deleted records still occupy disk space, so they are counted too.
static int check_stat_all_flushed(const Livero* db, const LVConfig* config, const LVOption* option) {
    const LVDBStat expected = {
        .config = *config,
        .option = *option,
        .row_file_bytes = ANSWER_COUNT * LV_ROW_BYTES,
        .ternary_file_bytes = ANSWER_COUNT * ternary_chunks_bytes(config->vector_dimension),
        .val_file_bytes = ANSWER_VALUE_BYTES,
        .row_pending_bytes = 0,
        .ternary_pending_bytes = 0,
        .val_pending_bytes = 0,
    };

    LVDBStat got = {0};
    lv_stat(db, &got);
    return check_stat_match(&got, &expected);
}

/*
    Query with QUERY_VEC and check the result.

    Always checked, for every returned item:
      - it exists, is not deleted, and its date and value_len match the answer
      - its date is inside [date1, date2] (NULL means open on that side)
      - no seq is returned twice
    Plus:
      - the result size equals expected_size
      - every alive seq in expected_seqs is in the result
*/
static int check_query(Livero* db, const LVTopK_t k, const LVDate* date1, const LVDate* date2,
                       const LVSeq_t* expected_seqs, const LVCount_t expected_count, const LVSize32_t expected_size,
                       const char* phase) {
    // LVTopK_t is uint8_t, so 255 slots hold any k.
    LVQueryResult results[UINT8_MAX];
    LVSize32_t size = (LVSize32_t)-1;

    const LVStatus s = lv_query(db, k, QUERY_VEC, date1, date2, results, &size);
    if (s != LV_OK) {
        fprintf(stderr, "[%s] lv_query failed.\n", phase);
        PRINT_ERR_UNEXPECTED_STATUS("lv_query", LV_OK, s);
        return -1;
    }

    if (size > k || size != expected_size) {
        fprintf(stderr, "[%s] ", phase);
        PRINT_ERR_INVALID("query result size", "%" PRIu32, expected_size, size);
        return -1;
    }

    const uint64_t lo = date1 ? lv_convert_date_to_epoch(date1) : 0;
    const uint64_t hi = date2 ? lv_convert_date_to_epoch(date2) : UINT64_MAX;

    for (LVSize32_t i = 0; i < size; ++i) {
        const LVQueryResult* r = &results[i];
        const LVTestAnswer* ans = answer_of(r->seq);

        if (!ans) {
            fprintf(stderr, "[%s] query returned seq %" PRIu64 ", which was never put.\n", phase, r->seq);
            return -1;
        }
        if (ans->row.tombstone_flag) {
            fprintf(stderr, "[%s] query returned deleted seq %" PRIu64 ".\n", phase, r->seq);
            return -1;
        }

        const uint64_t epoch = lv_convert_date_to_epoch(&r->date);
        if (epoch != ans->row.date) {
            fprintf(stderr, "[%s] seq %" PRIu64 ": ", phase, r->seq);
            PRINT_ERR_INVALID("query date", "%" PRIu64, ans->row.date, epoch);
            return -1;
        }
        if (epoch < lo || epoch > hi) {
            fprintf(stderr, "[%s] seq %" PRIu64 " is outside the date filter.\n", phase, r->seq);
            return -1;
        }
        if (r->value_len != ans->row.value_len) {
            fprintf(stderr, "[%s] seq %" PRIu64 ": ", phase, r->seq);
            PRINT_ERR_INVALID("query value_len", "%" PRIu16, ans->row.value_len, r->value_len);
            return -1;
        }

        for (LVSize32_t j = 0; j < i; ++j) {
            if (results[j].seq == r->seq) {
                fprintf(stderr, "[%s] query returned seq %" PRIu64 " twice.\n", phase, r->seq);
                return -1;
            }
        }
    }

    for (LVCount_t e = 0; e < expected_count; ++e) {
        const LVTestAnswer* ans = answer_of(expected_seqs[e]);
        if (!ans) {
            fprintf(stderr, "[%s] expected seq %" PRIu64 " was never put; fix the test input.\n", phase,
                    expected_seqs[e]);
            return -1;
        }
        // Deleted ground truths are already rejected above if they appear.
        if (ans->row.tombstone_flag) {
            continue;
        }

        int found = 0;
        for (LVSize32_t i = 0; i < size; ++i) {
            if (results[i].seq == expected_seqs[e]) {
                found = 1;
                break;
            }
        }
        if (!found) {
            fprintf(stderr, "[%s] ground truth seq %" PRIu64 " is missing from the query result.\n", phase,
                    expected_seqs[e]);
            return -1;
        }
    }

    return 0;
}

/* ------------------------------------------------------------------------- */
/* Tests                                                                     */
/* ------------------------------------------------------------------------- */

int test_create(void) {
    /*
        1. Create a db
        2. Check db stat (config and option as given, every size zero)
        3. Close the db
        4. Create again on the same directory and check LV_ERR_EXSIST
    */
    Livero* db = NULL;
    Livero* second = NULL;
    int result = -1;
    answers_reset();

    if (create_test_db(TEST_DIR, NULL, NULL, &db) < 0) {
        goto cleanup;
    }
    if (check_stat_all_flushed(db, &TEST_CONFIG, &TEST_OPTION) < 0) {
        goto cleanup;
    }

    lv_close(db);
    db = NULL;

    // Creating over an existing database must not overwrite it.
    const LVStatus s = lv_create(TEST_DIR, &TEST_CONFIG, &TEST_OPTION, &second);
    if (s != LV_ERR_EXSIST) {
        PRINT_ERR_UNEXPECTED_STATUS("lv_create on an existing db", LV_ERR_EXSIST, s);
        goto cleanup;
    }

    result = 0;

cleanup:
    lv_close(second);
    lv_close(db);
    test_end(result, TEST_DIR, "test_create");
    return result;
}

int test_open(void) {
    /*
        1. Create a db and close it
        2. Open the db
        3. Check db stat (config as created, every size zero)
        4. Close the db, then make the directory empty
        5. Open the empty directory and check LV_ERR_DNEXSIST
    */
    Livero* db = NULL;
    int result = -1;
    answers_reset();

    if (create_test_db(TEST_DIR, NULL, NULL, &db) < 0) {
        goto cleanup;
    }
    if (reopen_db(&db, &TEST_OPTION) < 0) {
        goto cleanup;
    }
    if (check_stat_all_flushed(db, &TEST_CONFIG, &TEST_OPTION) < 0) {
        goto cleanup;
    }

    lv_close(db);
    db = NULL;

    clean_test_dir(TEST_DIR);
    if (create_test_dir(TEST_DIR) < 0) {
        goto cleanup;
    }

    const LVStatus s = lv_open(TEST_DIR, &TEST_OPTION, &db);
    if (s != LV_ERR_DNEXSIST) {
        PRINT_ERR_UNEXPECTED_STATUS("lv_open on an empty directory", LV_ERR_DNEXSIST, s);
        goto cleanup;
    }

    result = 0;

cleanup:
    lv_close(db);
    test_end(result, TEST_DIR, "test_open");
    return result;
}

int test_put_get(void) {
    /*
        1. Create a db (write_batch_m = 100)
        2. Put 250 (200 flushed, 50 pending)
        3. Get every record
        4. Get seqs that were never put and check LV_ERR_NOT_FOUND
    */
    Livero* db = NULL;
    int result = -1;
    const LVCount_t put_count = 2 * TEST_WRITE_BATCH_M + TEST_WRITE_BATCH_M / 2;
    answers_reset();

    if (create_test_db(TEST_DIR, NULL, NULL, &db) < 0) {
        goto cleanup;
    }
    if (put_dummies(db, put_count, 2000, 2026) < 0) {
        goto cleanup;
    }
    if (check_all_gets(db, "put_get") < 0) {
        goto cleanup;
    }

    // Just past the end and far past it.
    if (check_get_missing(db, last_seq() + 1, "put_get") < 0) {
        goto cleanup;
    }
    if (check_get_missing(db, last_seq() + 1000, "put_get") < 0) {
        goto cleanup;
    }

    result = 0;

cleanup:
    lv_close(db);
    test_end(result, TEST_DIR, "test_put_get");
    return result;
}

int test_reopen(void) {
    /*
        1. Create a db (write_batch_m = 100)
        2. Put 30 (all pending)
        3. Get every record
        4. Close and reopen (close must flush the pending batch)
        5. Check db stat
        6. Get every record
        7. Put 10 and check the seqs continue at 30
        8. Get every record
        9. Close and reopen again, check db stat and get every record
           (the second session appended to files that already had data)
    */
    Livero* db = NULL;
    int result = -1;
    const LVCount_t put1_count = 30;
    const LVCount_t put2_count = 10;
    answers_reset();

    if (create_test_db(TEST_DIR, NULL, NULL, &db) < 0) {
        goto cleanup;
    }
    if (put_dummies(db, put1_count, 2000, 2026) < 0) {
        goto cleanup;
    }
    if (check_all_gets(db, "before 1st reopen") < 0) {
        goto cleanup;
    }

    if (reopen_db(&db, &TEST_OPTION) < 0) {
        goto cleanup;
    }
    if (check_stat_all_flushed(db, &TEST_CONFIG, &TEST_OPTION) < 0) {
        goto cleanup;
    }
    if (check_all_gets(db, "after 1st reopen") < 0) {
        goto cleanup;
    }

    if (put_dummies(db, put2_count, 2000, 2026) < 0) {
        goto cleanup;
    }

    // put_one only requires a larger seq after reopen; v0 expected no gap after a clean close.
    // Change this check if clean close is meant to leave the reserved seqs unused.
    if (ANSWERS[put1_count].row.seq != LV_SEQ_START + put1_count) {
        fprintf(stderr, "[reopen] seqs must continue after a clean close. ");
        PRINT_ERR_INVALID("seq", "%" PRIu64, (LVSeq_t)(LV_SEQ_START + put1_count), ANSWERS[put1_count].row.seq);
        goto cleanup;
    }

    if (check_all_gets(db, "after 2nd put") < 0) {
        goto cleanup;
    }

    if (reopen_db(&db, &TEST_OPTION) < 0) {
        goto cleanup;
    }
    if (check_stat_all_flushed(db, &TEST_CONFIG, &TEST_OPTION) < 0) {
        goto cleanup;
    }
    if (check_all_gets(db, "after 2nd reopen") < 0) {
        goto cleanup;
    }

    result = 0;

cleanup:
    lv_close(db);
    test_end(result, TEST_DIR, "test_reopen");
    return result;
}

int test_delete(void) {
    /*
        1. Create a db (write_batch_m = 100, read_batch_m = 100)
        2. Put 150 (100 flushed, 50 pending)
        3. Delete one flushed and one pending
        4. Get every record (deleted ones must return LV_ERR_DELETED)
        5. Delete a seq that was never put and check LV_ERR_NOT_FOUND
        6. Close and reopen, check db stat
        7. Repeat step 4
    */
    Livero* db = NULL;
    int result = -1;
    const LVSeq_t flushed_to_delete = LV_SEQ_START + 47;
    const LVSeq_t pending_to_delete = LV_SEQ_START + 101;
    answers_reset();

    if (create_test_db(TEST_DIR, NULL, NULL, &db) < 0) {
        goto cleanup;
    }
    if (put_dummies(db, 150, 2000, 2026) < 0) {
        goto cleanup;
    }

    if (delete_one(db, flushed_to_delete) < 0) {
        goto cleanup;
    }
    if (delete_one(db, pending_to_delete) < 0) {
        goto cleanup;
    }
    if (check_all_gets(db, "before reopen") < 0) {
        goto cleanup;
    }

    const LVStatus s = lv_delete(db, last_seq() + 1);
    if (s != LV_ERR_NOT_FOUND) {
        PRINT_ERR_UNEXPECTED_STATUS("lv_delete on a missing seq", LV_ERR_NOT_FOUND, s);
        goto cleanup;
    }

    // lv_close must flush the pending batch, including its tombstone.
    if (reopen_db(&db, &TEST_OPTION) < 0) {
        goto cleanup;
    }
    if (check_stat_all_flushed(db, &TEST_CONFIG, &TEST_OPTION) < 0) {
        goto cleanup;
    }
    if (check_all_gets(db, "after reopen") < 0) {
        goto cleanup;
    }

    result = 0;

cleanup:
    lv_close(db);
    test_end(result, TEST_DIR, "test_delete");
    return result;
}

int test_query_topk(void) {
    /*
        1. Create a db (write_batch_m = 100, read_batch_m = 50)
        2. Put 10 ground truths sharing the query vector: 5 flushed, 5 pending
        3. Query top-10 without a date filter
        4. Check the result is exactly the 10 ground truths
        5. Close and reopen, repeat step 3-4
    */
    Livero* db = NULL;
    int result = -1;
    LVOption option = TEST_OPTION;
    option.write_batch_m = 100;
    option.read_batch_m = 50;
    const LVTopK_t k = 10;
    LVSeq_t gt[10];
    answers_reset();

    fill_test_int8_vector();
    memcpy(QUERY_VEC, TEST_INT8_VEC_BUFFER, sizeof(QUERY_VEC));

    if (create_test_db(TEST_DIR, NULL, &option, &db) < 0) {
        goto cleanup;
    }
    if (put_query_layout(db, option.write_batch_m, 10, 20, gt) < 0) {
        goto cleanup;
    }

    if (check_query(db, k, NULL, NULL, gt, 10, 10, "topk before reopen") < 0) {
        goto cleanup;
    }

    if (reopen_db(&db, &option) < 0) {
        goto cleanup;
    }
    if (check_query(db, k, NULL, NULL, gt, 10, 10, "topk after reopen") < 0) {
        goto cleanup;
    }

    result = 0;

cleanup:
    lv_close(db);
    test_end(result, TEST_DIR, "test_query_topk");
    return result;
}

// Queries shared by test_query_date before and after reopen.
// groups[y][i] is the i-th ground truth of year 2024 + y.
static int run_date_queries(Livero* db, const LVSeq_t groups[3][3], const char* phase) {
    const LVTopK_t k = 10;
    LVDate start[3], end[3];
    for (int y = 0; y < 3; ++y) {
        make_first_date((uint16_t)(2024 + y), &start[y]);
        make_last_date((uint16_t)(2024 + y), &end[y]);
    }

    // Each year alone: exactly its 3 ground truths, including both boundary dates.
    for (int y = 0; y < 3; ++y) {
        if (check_query(db, k, &start[y], &end[y], groups[y], 3, 3, phase) < 0) {
            return -1;
        }
    }

    // [2025-01-01 00:00:00, open): 2025 and 2026, but not 2024-12-31 23:59:59.
    const LVSeq_t from_2025[6] = {groups[1][0], groups[1][1], groups[1][2], groups[2][0], groups[2][1], groups[2][2]};
    if (check_query(db, k, &start[1], NULL, from_2025, 6, 6, phase) < 0) {
        return -1;
    }

    // Nothing exists from 2027 on.
    LVDate start_2027;
    make_first_date(2027, &start_2027);
    if (check_query(db, k, &start_2027, NULL, NULL, 0, 0, phase) < 0) {
        return -1;
    }

    // (open, 2023-12-31 23:59:59]: only dummies, still k of them.
    LVDate end_2023;
    make_last_date(2023, &end_2023);
    if (check_query(db, k, NULL, &end_2023, NULL, 0, k, phase) < 0) {
        return -1;
    }

    return 0;
}

int test_query_date(void) {
    /*
        1. Create a db (write_batch_m = 100, read_batch_m = 50)
        2. Put 500 dummies dated 2000-2023 (all flushed)
        3. Put 9 ground truths (pending), 3 per year in 2024, 2025, 2026:
           Jan 1 00:00:00, Dec 31 23:59:59 and a random date in that year
        4. Query each year's range and check only that year's ground truths come back
        5. Query from 2025 on, from 2027 on, and up to 2023
        6. Close and reopen (ground truths are now flushed), repeat step 4-5
    */
    Livero* db = NULL;
    int result = -1;
    LVOption option = TEST_OPTION;
    option.write_batch_m = 100;
    option.read_batch_m = 50;
    LVSeq_t groups[3][3];
    answers_reset();

    fill_test_int8_vector();
    memcpy(QUERY_VEC, TEST_INT8_VEC_BUFFER, sizeof(QUERY_VEC));

    if (create_test_db(TEST_DIR, NULL, &option, &db) < 0) {
        goto cleanup;
    }

    // Dummies sit strictly before 2024, so every 2024+ range holds only ground truths.
    if (put_dummies(db, 5 * option.write_batch_m, 2000, 2023) < 0) {
        goto cleanup;
    }

    for (int y = 0; y < 3; ++y) {
        const uint16_t year = (uint16_t)(2024 + y);
        LVDate first, last;
        make_first_date(year, &first);
        make_last_date(year, &last);

        if (put_one(db, QUERY_VEC, year, year, &first, &groups[y][0]) < 0) {
            goto cleanup;
        }
        if (put_one(db, QUERY_VEC, year, year, &last, &groups[y][1]) < 0) {
            goto cleanup;
        }
        if (put_one(db, QUERY_VEC, year, year, NULL, &groups[y][2]) < 0) {
            goto cleanup;
        }
    }

    if (run_date_queries(db, (const LVSeq_t(*)[3])groups, "date before reopen") < 0) {
        goto cleanup;
    }

    if (reopen_db(&db, &option) < 0) {
        goto cleanup;
    }
    if (run_date_queries(db, (const LVSeq_t(*)[3])groups, "date after reopen") < 0) {
        goto cleanup;
    }

    result = 0;

cleanup:
    lv_close(db);
    test_end(result, TEST_DIR, "test_query_date");
    return result;
}

int test_query_tombstone(void) {
    /*
        1. Create a db (write_batch_m = 100, read_batch_m = 50)
        2. Put 10 ground truths sharing the query vector: 5 flushed, 5 pending
        3. Delete one flushed and one pending ground truth
        4. Query top-10 and check the result has the 8 alive ground truths,
           no deleted one, and is filled up to 10 by dummies
        5. Close and reopen, repeat step 4
    */
    Livero* db = NULL;
    int result = -1;
    LVOption option = TEST_OPTION;
    option.write_batch_m = 100;
    option.read_batch_m = 50;
    const LVTopK_t k = 10;
    LVSeq_t gt[10];
    answers_reset();

    fill_test_int8_vector();
    memcpy(QUERY_VEC, TEST_INT8_VEC_BUFFER, sizeof(QUERY_VEC));

    if (create_test_db(TEST_DIR, NULL, &option, &db) < 0) {
        goto cleanup;
    }
    if (put_query_layout(db, option.write_batch_m, 10, 20, gt) < 0) {
        goto cleanup;
    }

    // gt[0..4] are flushed and gt[5..9] are pending.
    if (delete_one(db, gt[2]) < 0) {
        goto cleanup;
    }
    if (delete_one(db, gt[7]) < 0) {
        goto cleanup;
    }

    // Two slots freed by the deletes are filled by the best dummies, so the size stays k.
    if (check_query(db, k, NULL, NULL, gt, 10, k, "tombstone before reopen") < 0) {
        goto cleanup;
    }

    if (reopen_db(&db, &option) < 0) {
        goto cleanup;
    }
    if (check_query(db, k, NULL, NULL, gt, 10, k, "tombstone after reopen") < 0) {
        goto cleanup;
    }

    result = 0;

cleanup:
    lv_close(db);
    test_end(result, TEST_DIR, "test_query_tombstone");
    return result;
}

int main(void) {
    // Disable stdout buffering so pass lines and stderr failures appear in order.
    setvbuf(stdout, NULL, _IONBF, 0);
    clean_test_dir(TEST_DIR);

    if (test_create() < 0) {
        return -1;
    }
    if (test_open() < 0) {
        return -1;
    }
    if (test_put_get() < 0) {
        return -1;
    }
    if (test_reopen() < 0) {
        return -1;
    }
    if (test_delete() < 0) {
        return -1;
    }
    if (test_query_topk() < 0) {
        return -1;
    }
    if (test_query_date() < 0) {
        return -1;
    }
    if (test_query_tombstone() < 0) {
        return -1;
    }
    return 0;
}
