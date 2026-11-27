#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "helper/helper.h"
#include "livero_types.h"
#include "struct/commit.h"
#include "struct/row.h"
#include "tests_helper.h"
#include "vector/ternary.h"

static const char* TEST_DIR = "./TEST_LIVERO_TMP/";

/*
    Public-API tests ported from the v0 suite (test_livero.c), plus recovery tests
    that replace the old test_recover.c.

    v0 test        -> this file
    test_create    -> test_create
    test_open      -> test_open
    test_1         -> test_put_get
    test_2         -> test_reopen
    test_3         -> test_delete        (a deleted seq now returns LV_ERR_DELETED, not LV_ERR_NOT_FOUND)
    test_4         -> test_query_topk
    test_5         -> test_query_date
    test_6         -> test_query_tombstone

    Recovery tests (new):
    test_empty_reopen               create -> close -> open starts at LV_SEQ_START
    test_crash_pending_lost         crash drops the pending batch, its seqs are never reused
    test_crash_after_clean_reopen   clean close -> reopen -> put -> crash must not reuse seqs
    test_crash_without_put          a crashed session that issued nothing burns no seqs
    test_commit_one_slot_corrupt    either slot corrupt -> recover from the other one
    test_commit_first_slot_torn     first commit torn, other slot empty -> new db
    test_commit_second_slot_torn    slot 1 torn, slot 0 empty -> LV_ERR_MEDIA_CRASHED
    test_commit_both_corrupt        both slots corrupt -> LV_ERR_MEDIA_CRASHED
    test_commit_both_empty          no commit at all -> new db

    Every test keeps an answer table of what it put, so each check compares
    livero's output with what the test knows it wrote.

    Recovery tests only go through the public API. Crashes and torn commits are
    made from outside: a crash drops the handle without lv_close, and a torn commit
    is made by damaging commit.lv on disk.
*/

/* ------------------------------------------------------------------------- */
/* Answer table                                                              */
/* ------------------------------------------------------------------------- */

// Every record put in the current test, in put order.
#define TEST_MAX_ANSWERS 1024
static LVTestAnswer ANSWERS[TEST_MAX_ANSWERS];
static LVCount_t ANSWER_COUNT = 0;
static LVSize32_t ANSWER_VALUE_BYTES = 0;

// The first put after a crash may skip seqs reserved by the crashed session,
// so it only has to be past every seq ever issued. Every other put must be max + 1,
// including the first put after a clean close.
static int SEQ_GAP_ALLOWED = 0;

// Every seq livero has ever handed out in the current test, lost ones included.
// Seq checks compare against this, not the answer table, because a crash
// removes answers but never makes their seqs reusable.
static int HAS_ISSUED = 0;
static LVSeq_t MAX_ISSUED_SEQ = 0;

// Seqs that were handed out but lost (pending at a crash, or newer than the commit
// recovery fell back to). They must stay LV_ERR_NOT_FOUND forever.
#define TEST_MAX_LOST 1024
static LVSeq_t LOST[TEST_MAX_LOST];
static LVCount_t LOST_COUNT = 0;

// Vector shared by every query ground truth. It lives apart from
// TEST_INT8_VEC_BUFFER because each dummy put overwrites that buffer.
static int8_t QUERY_VEC[TEST_VECTOR_DIMENSION];

static void answers_reset(void) {
    ANSWER_COUNT = 0;
    ANSWER_VALUE_BYTES = 0;
    SEQ_GAP_ALLOWED = 0;
    HAS_ISSUED = 0;
    MAX_ISSUED_SEQ = 0;
    LOST_COUNT = 0;
}

/*
    Drop every answer from `index` on, because recovery is expected to have lost them.
    Their seqs move to LOST: they were handed out, so they must never come back.
*/
static void answers_drop_from(const LVCount_t index) {
    assert(index <= ANSWER_COUNT);

    for (LVCount_t i = index; i < ANSWER_COUNT; ++i) {
        assert(LOST_COUNT < TEST_MAX_LOST);
        LOST[LOST_COUNT++] = ANSWERS[i].row.seq;
    }
    ANSWER_COUNT = index;

    // Recovery restores file stats from the commit, so the next value lands right
    // after the surviving ones. Recompute instead of subtracting to keep it obvious.
    ANSWER_VALUE_BYTES = 0;
    for (LVCount_t i = 0; i < ANSWER_COUNT; ++i) {
        ANSWER_VALUE_BYTES += ANSWERS[i].row.value_len;
    }
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
    lv_convert_epoch_to_date(ans->row.epoch_date, &date);

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

    // Seqs are monotonic and never reused. Compare against every seq ever issued,
    // not the answer table, because lost answers still own their seqs.
    if (!HAS_ISSUED) {
        if (seq != LV_SEQ_START) {
            fprintf(stderr, "[put] first seq of a new db is wrong. ");
            PRINT_ERR_INVALID("seq", "%" PRIu64, (LVSeq_t)LV_SEQ_START, seq);
            return -1;
        }
    } else if (SEQ_GAP_ALLOWED) {
        if (seq <= MAX_ISSUED_SEQ) {
            fprintf(stderr, "[put] seq after a crash must be larger than %" PRIu64 ", got %" PRIu64 " (seq reused).\n",
                    MAX_ISSUED_SEQ, seq);
            return -1;
        }
    } else if (seq != MAX_ISSUED_SEQ + 1) {
        fprintf(stderr, "[put] seq must follow the previous one. ");
        PRINT_ERR_INVALID("seq", "%" PRIu64, MAX_ISSUED_SEQ + 1, seq);
        return -1;
    }

    ans->row.seq = seq;
    SEQ_GAP_ALLOWED = 0;
    HAS_ISSUED = 1;
    MAX_ISSUED_SEQ = seq;
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

// Open TEST_DIR and expect LV_OK. On failure *db is NULL, so the caller's cleanup stays safe.
static int open_db(Livero** db, const LVOption* option, const char* phase) {
    *db = NULL;
    const LVStatus s = lv_open(TEST_DIR, option, db);
    if (s != LV_OK) {
        fprintf(stderr, "[%s] ", phase);
        PRINT_ERR_UNEXPECTED_STATUS("lv_open", LV_OK, s);
        *db = NULL;
        return -1;
    }
    if (!*db) {
        fprintf(stderr, "[%s] lv_open returned LV_OK but did not set the handle.\n", phase);
        return -1;
    }
    return 0;
}

/*
    Close cleanly and reopen.

    lv_close leaves a clean-close commit, so the next session continues at the
    exact next seq. SEQ_GAP_ALLOWED stays 0: any gap after a clean close fails put_one.
*/
static int reopen_db(Livero** db, const LVOption* option) {
    lv_close(*db);
    *db = NULL;

    if (open_db(db, option, "reopen") < 0) {
        return -1;
    }

    SEQ_GAP_ALLOWED = 0;
    return 0;
}

/*
    Simulate a process crash (kill -9) and reopen.

    The handle is dropped without lv_close, so the pending batch in RAM is lost and
    no clean-close commit is written. Bytes livero already gave to the OS stay
    visible to the next open, because the page cache survives a process crash.

    This is NOT a power-loss test: unsynced bytes would also survive here.

    The handle is leaked on purpose; it can't be freed without lv_close.
    Run with ASAN_OPTIONS=detect_leaks=0 if LeakSanitizer complains.

    committed_count: how many answers (in put order) the test expects to survive.
*/
static int crash_and_reopen(Livero** db, const LVOption* option, const LVCount_t committed_count) {
    *db = NULL;  // leaked on purpose, see above

    answers_drop_from(committed_count);

    if (open_db(db, option, "crash reopen") < 0) {
        return -1;
    }

    // The crashed session has reserved seqs it never handed out, so the next
    // put only has to be past every seq ever issued.
    SEQ_GAP_ALLOWED = 1;
    return 0;
}

// Every lost seq must stay LV_ERR_NOT_FOUND: it was handed out once, so it can't belong to a new record.
static int check_lost_gets(Livero* db, const char* phase) {
    for (LVCount_t i = 0; i < LOST_COUNT; ++i) {
        if (check_get_missing(db, LOST[i], phase) < 0) {
            fprintf(stderr, "[%s] lost seq %" PRIu64 " came back.\n", phase, LOST[i]);
            return -1;
        }
    }
    return 0;
}

/*
    After recovery falls back to an older commit, the db must be exactly that commit's
    snapshot: a prefix of the answer table in put order. Records after the prefix must
    be gone completely, not half there.

    The missing tail is moved to LOST, so later checks can confirm those seqs never come back.
    Use this function instead of check_all_gets when the test can't know which commit recovery fell back to.
*/
static int check_gets_prefix(Livero* db, const char* phase) {
    LVGetResult got = {.value = TEST_VALUE_BUFFER};
    LVCount_t kept = ANSWER_COUNT;
    int missing_seen = 0;

    for (LVCount_t i = 0; i < ANSWER_COUNT; ++i) {
        const LVTestAnswer* ans = &ANSWERS[i];

        memset(TEST_VALUE_BUFFER, 0xAA, sizeof(TEST_VALUE_BUFFER));
        got.value_len = 0;
        memset(&got.date, 0, sizeof(got.date));

        const LVStatus s = lv_get(db, ans->row.seq, &got);
        if (s == LV_OK) {
            if (missing_seen) {
                fprintf(stderr, "[%s] seq %" PRIu64 " exists after a missing one: not a snapshot.\n", phase,
                        ans->row.seq);
                return -1;
            }
            if (check_get_result(ans, &got) < 0) {
                fprintf(stderr, "[%s] lv_get returned a wrong record.\n", phase);
                return -1;
            }
        } else if (s == LV_ERR_NOT_FOUND) {
            if (!missing_seen) {
                missing_seen = 1;
                kept = i;
            }
        } else {
            fprintf(stderr, "[%s] lv_get at seq %" PRIu64 ".\n", phase, ans->row.seq);
            PRINT_ERR_UNEXPECTED_STATUS("lv_get", LV_OK, s);
            return -1;
        }
    }

    answers_drop_from(kept);
    return 0;
}

/* ------------------------------------------------------------------------- */
/* commit.lv damage (media simulation)                                       */
/* ------------------------------------------------------------------------- */

typedef enum {
    SLOT_MAKE_EMPTY,    // all zero bytes: a slot that was never written
    SLOT_MAKE_CORRUPT,  // commit seq kept, every other byte flipped: a torn write
} LVSlotDamage;

#define TEST_COMMIT_SEQ_BYTES 8

/*
    Damage one commit slot on disk while no handle is open.

    This touches only the file, never engine state: the engine still has to find out
    by itself which slot to trust.
*/
static int damage_commit_slot(const int slot, const LVSlotDamage how) {
    char path[LV_MAX_PATH_LEN];
    if (lv_concat_path(TEST_DIR, LV_COMMIT_FILE_NAME, path, LV_MAX_PATH_LEN) != LV_OK) {
        fprintf(stderr, "[damage] path too long.\n");
        return -1;
    }

    const int fd = open(path, O_RDWR);
    if (fd < 0) {
        perror(path);
        return -1;
    }

    uint8_t buf[LV_COMMIT_DISK_BYTES];
    const LVOffset_t offset = lv_commit_slot_offset(slot);
    int result = -1;

    if (how == SLOT_MAKE_EMPTY) {
        memset(buf, 0, sizeof(buf));
    } else {
        // A slot that was never written can't be torn, so refuse instead of forging one.
        if (pread(fd, buf, sizeof(buf), offset) != (ssize_t)sizeof(buf)) {
            fprintf(stderr, "[damage] slot %d was never written, can't corrupt it.\n", slot);
            goto done;
        }
        for (size_t i = TEST_COMMIT_SEQ_BYTES; i < sizeof(buf); ++i) {
            buf[i] ^= 0xFF;
        }
    }

    if (pwrite(fd, buf, sizeof(buf), offset) != (ssize_t)sizeof(buf)) {
        perror("[damage] pwrite");
        goto done;
    }
    if (fsync(fd) < 0) {
        perror("[damage] fsync");
        goto done;
    }
    result = 0;

done:
    close(fd);
    return result;
}

// Open TEST_DIR and expect a specific failure status.
static int open_expect_fail(const LVStatus expected, const char* phase) {
    Livero* db = NULL;
    const LVStatus s = lv_open(TEST_DIR, &TEST_OPTION, &db);
    if (s != expected) {
        fprintf(stderr, "[%s] ", phase);
        PRINT_ERR_UNEXPECTED_STATUS("lv_open", expected, s);
        if (s == LV_OK) {
            lv_close(db);
        }
        return -1;
    }
    return 0;
}

/*
    Leave both commit slots written: two sessions, each with puts and a clean close.
    Each session writes at least a reservation commit and a close commit.
*/
static int make_two_sessions(Livero** db, const LVCount_t per_session) {
    if (create_test_db(TEST_DIR, NULL, NULL, db) < 0) {
        return -1;
    }
    if (put_dummies(*db, per_session, 2000, 2026) < 0) {
        return -1;
    }
    if (reopen_db(db, &TEST_OPTION) < 0) {
        return -1;
    }
    if (put_dummies(*db, per_session, 2000, 2026) < 0) {
        return -1;
    }
    lv_close(*db);
    *db = NULL;
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
      - the result size equals expected_result_size (it can be larger than the
        ground truths, because dummies fill the remaining top-k slots)
      - every alive seq in expected_seqs is in the result (deleted ones are skipped, but ground_truth_count may include
   them)
*/
static int check_query(Livero* db, const LVTopK_t k, const LVDate* date1, const LVDate* date2,
                       const LVSeq_t* expected_seqs, const LVCount_t ground_truth_count,
                       const LVSize32_t expected_result_size, const char* phase) {
    // LVTopK_t is uint8_t, so 255 slots hold any k.
    LVQueryResult results[UINT8_MAX];
    LVSize32_t size = (LVSize32_t)-1;

    const LVQueryOption query_option = {
        .date1 = date1, .date2 = date2, .use_postfilter = TEST_DEFAULT_POST_FILTERING_USE};

    const LVStatus s = lv_query(db, k, QUERY_VEC, &query_option, results, &size);
    if (s != LV_OK) {
        fprintf(stderr, "[%s] lv_query failed.\n", phase);
        PRINT_ERR_UNEXPECTED_STATUS("lv_query", LV_OK, s);
        return -1;
    }

    if (size > k || size != expected_result_size) {
        fprintf(stderr, "[%s] ", phase);
        PRINT_ERR_INVALID("query result size", "%" PRIu32, expected_result_size, size);
        return -1;
    }

    const uint64_t min_epoch_date = date1 ? lv_convert_date_to_epoch(date1) : 0;
    const uint64_t max_epoch_date = date2 ? lv_convert_date_to_epoch(date2) : UINT64_MAX;

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

        const uint64_t epoch_date = lv_convert_date_to_epoch(&r->date);
        if (epoch_date != ans->row.epoch_date) {
            fprintf(stderr, "[%s] seq %" PRIu64 ": ", phase, r->seq);
            PRINT_ERR_INVALID("query epoch_date", "%" PRIu64, ans->row.epoch_date, epoch_date);
            return -1;
        }
        if (epoch_date < min_epoch_date || epoch_date > max_epoch_date) {
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

    for (LVCount_t e = 0; e < ground_truth_count; ++e) {
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
    if (s != LV_ERR_EXIST) {
        PRINT_ERR_UNEXPECTED_STATUS("lv_create on an existing db", LV_ERR_EXIST, s);
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
    if (s != LV_ERR_NOT_EXIST) {
        PRINT_ERR_UNEXPECTED_STATUS("lv_open on an empty directory", LV_ERR_NOT_EXIST, s);
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

    // put_one fails on any gap here: a clean close must continue at the exact next seq.
    if (put_dummies(db, put2_count, 2000, 2026) < 0) {
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

/* ------------------------------------------------------------------------- */
/* Recovery tests                                                            */
/* ------------------------------------------------------------------------- */

int test_empty_reopen(void) {
    /*
        1. Create a db and close it right away (no records)
        2. Reopen and put one
        3. The seq must be LV_SEQ_START: a clean close with nothing issued burns no seqs
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
    // HAS_ISSUED is 0, so put_one requires LV_SEQ_START.
    if (put_one(db, NULL, 2000, 2026, NULL, NULL) < 0) {
        goto cleanup;
    }

    result = 0;

cleanup:
    lv_close(db);
    test_end(result, TEST_DIR, "test_empty_reopen");
    return result;
}

int test_crash_pending_lost(void) {
    /*
        1. Create a db (write_batch_m = 100)
        2. Put 150 (100 flushed and committed, 50 pending)
        3. Crash without lv_close and reopen
        4. The 100 committed records are intact, the 50 pending are LV_ERR_NOT_FOUND
        5. Put 10: past every seq ever issued, including the 50 lost ones
        6. The lost seqs are still LV_ERR_NOT_FOUND (never reused)
        7. Close and reopen, check stat, every get and the lost seqs again
    */
    Livero* db = NULL;
    int result = -1;
    const LVCount_t committed = TEST_WRITE_BATCH_M;
    const LVCount_t put_count = TEST_WRITE_BATCH_M + TEST_WRITE_BATCH_M / 2;
    answers_reset();

    if (create_test_db(TEST_DIR, NULL, NULL, &db) < 0) {
        goto cleanup;
    }
    if (put_dummies(db, put_count, 2000, 2026) < 0) {
        goto cleanup;
    }

    if (crash_and_reopen(&db, &TEST_OPTION, committed) < 0) {
        goto cleanup;
    }
    if (check_stat_all_flushed(db, &TEST_CONFIG, &TEST_OPTION) < 0) {
        goto cleanup;
    }
    if (check_all_gets(db, "after crash") < 0) {
        goto cleanup;
    }
    if (check_lost_gets(db, "after crash") < 0) {
        goto cleanup;
    }

    // put_one rejects any seq <= MAX_ISSUED_SEQ, which includes the lost ones.
    if (put_dummies(db, 10, 2000, 2026) < 0) {
        goto cleanup;
    }
    if (check_lost_gets(db, "after put past crash") < 0) {
        goto cleanup;
    }

    if (reopen_db(&db, &TEST_OPTION) < 0) {
        goto cleanup;
    }
    if (check_stat_all_flushed(db, &TEST_CONFIG, &TEST_OPTION) < 0) {
        goto cleanup;
    }
    if (check_all_gets(db, "after clean reopen") < 0) {
        goto cleanup;
    }
    if (check_lost_gets(db, "after clean reopen") < 0) {
        goto cleanup;
    }

    result = 0;

cleanup:
    lv_close(db);
    test_end(result, TEST_DIR, "test_crash_pending_lost");
    return result;
}

int test_crash_after_clean_reopen(void) {
    /*
        Regression: a clean-close commit must not survive into a session that issued seqs.
        If the first put of a session doesn't write a reservation commit, the latest commit
        is still the clean one, and recovery would reuse every seq that session handed out.

        1. Create, put 30, close and reopen (clean: no gap)
        2. Put 10 (pending), crash, reopen
        3. The 30 are intact, the 10 are gone
        4. Put 5 (pending): past the lost 10
        5. Crash again and reopen (crash right after a crash recovery)
        6. Put 5: past both lost groups
        7. Close, reopen, check everything
    */
    Livero* db = NULL;
    int result = -1;
    const LVCount_t first = 30;
    answers_reset();

    if (create_test_db(TEST_DIR, NULL, NULL, &db) < 0) {
        goto cleanup;
    }
    if (put_dummies(db, first, 2000, 2026) < 0) {
        goto cleanup;
    }
    if (reopen_db(&db, &TEST_OPTION) < 0) {
        goto cleanup;
    }

    if (put_dummies(db, 10, 2000, 2026) < 0) {
        goto cleanup;
    }
    if (crash_and_reopen(&db, &TEST_OPTION, first) < 0) {
        goto cleanup;
    }
    if (check_all_gets(db, "after 1st crash") < 0) {
        goto cleanup;
    }
    if (check_lost_gets(db, "after 1st crash") < 0) {
        goto cleanup;
    }

    if (put_dummies(db, 5, 2000, 2026) < 0) {
        goto cleanup;
    }
    if (crash_and_reopen(&db, &TEST_OPTION, first) < 0) {
        goto cleanup;
    }
    if (check_all_gets(db, "after 2nd crash") < 0) {
        goto cleanup;
    }
    if (check_lost_gets(db, "after 2nd crash") < 0) {
        goto cleanup;
    }

    if (put_dummies(db, 5, 2000, 2026) < 0) {
        goto cleanup;
    }

    if (reopen_db(&db, &TEST_OPTION) < 0) {
        goto cleanup;
    }
    if (check_stat_all_flushed(db, &TEST_CONFIG, &TEST_OPTION) < 0) {
        goto cleanup;
    }
    if (check_all_gets(db, "after clean reopen") < 0) {
        goto cleanup;
    }
    if (check_lost_gets(db, "after clean reopen") < 0) {
        goto cleanup;
    }

    result = 0;

cleanup:
    lv_close(db);
    test_end(result, TEST_DIR, "test_crash_after_clean_reopen");
    return result;
}

int test_crash_without_put(void) {
    /*
        1. Create, put 20, close and reopen
        2. Crash without any put, reopen
        3. Put 1: it must be exactly the next seq. A session that issued nothing
           wrote no reservation, so the clean commit is still the latest one.
    */
    Livero* db = NULL;
    int result = -1;
    const LVCount_t count = 20;
    answers_reset();

    if (create_test_db(TEST_DIR, NULL, NULL, &db) < 0) {
        goto cleanup;
    }
    if (put_dummies(db, count, 2000, 2026) < 0) {
        goto cleanup;
    }
    if (reopen_db(&db, &TEST_OPTION) < 0) {
        goto cleanup;
    }

    if (crash_and_reopen(&db, &TEST_OPTION, count) < 0) {
        goto cleanup;
    }
    // Override the crash default: no seq was issued in the crashed session, so no gap is allowed.
    SEQ_GAP_ALLOWED = 0;

    if (put_one(db, NULL, 2000, 2026, NULL, NULL) < 0) {
        goto cleanup;
    }
    if (check_all_gets(db, "after crash without put") < 0) {
        goto cleanup;
    }

    result = 0;

cleanup:
    lv_close(db);
    test_end(result, TEST_DIR, "test_crash_without_put");
    return result;
}

static int run_one_slot_corrupt(const int slot, const char* name) {
    /*
        1. Two sessions, 10 puts each, clean closes (both slots hold a commit)
        2. Corrupt `slot`
        3. Open must succeed with the other slot
        4. The db is a prefix of what was put (the test can't know which snapshot survived)
        5. Put 5: past every seq ever issued
        6. Close, reopen, check everything
    */
    Livero* db = NULL;
    int result = -1;
    answers_reset();

    if (make_two_sessions(&db, 10) < 0) {
        goto cleanup;
    }
    if (damage_commit_slot(slot, SLOT_MAKE_CORRUPT) < 0) {
        goto cleanup;
    }

    if (open_db(&db, &TEST_OPTION, name) < 0) {
        goto cleanup;
    }
    if (check_gets_prefix(db, name) < 0) {
        goto cleanup;
    }
    if (check_lost_gets(db, name) < 0) {
        goto cleanup;
    }

    // Whichever commit survived, the next seq must be past everything issued.
    SEQ_GAP_ALLOWED = 1;
    if (put_dummies(db, 5, 2000, 2026) < 0) {
        goto cleanup;
    }

    if (reopen_db(&db, &TEST_OPTION) < 0) {
        goto cleanup;
    }
    if (check_stat_all_flushed(db, &TEST_CONFIG, &TEST_OPTION) < 0) {
        goto cleanup;
    }
    if (check_all_gets(db, name) < 0) {
        goto cleanup;
    }
    if (check_lost_gets(db, name) < 0) {
        goto cleanup;
    }

    result = 0;

cleanup:
    lv_close(db);
    test_end(result, TEST_DIR, name);
    return result;
}

int test_commit_one_slot_corrupt(void) {
    if (run_one_slot_corrupt(0, "test_commit_one_slot_corrupt (slot 0)") < 0) {
        return -1;
    }
    return run_one_slot_corrupt(1, "test_commit_one_slot_corrupt (slot 1)");
}

int test_commit_first_slot_torn(void) {
    /*
        A crash while writing the very first commit: slot 0 torn, slot 1 never written.
        No reservation was ever durable, so no seq was issued.

        1. Create and close (slot 0 holds a commit)
        2. Make slot 1 empty, corrupt slot 0
        3. Open must succeed as a new db
        4. The first put gets LV_SEQ_START
    */
    Livero* db = NULL;
    int result = -1;
    answers_reset();

    if (create_test_db(TEST_DIR, NULL, NULL, &db) < 0) {
        goto cleanup;
    }
    lv_close(db);
    db = NULL;

    if (damage_commit_slot(1, SLOT_MAKE_EMPTY) < 0 || damage_commit_slot(0, SLOT_MAKE_CORRUPT) < 0) {
        goto cleanup;
    }

    if (open_db(&db, &TEST_OPTION, "first slot torn") < 0) {
        goto cleanup;
    }
    if (put_one(db, NULL, 2000, 2026, NULL, NULL) < 0) {
        goto cleanup;
    }

    result = 0;

cleanup:
    lv_close(db);
    test_end(result, TEST_DIR, "test_commit_first_slot_torn");
    return result;
}

int test_commit_second_slot_torn(void) {
    /*
        Slot 1 torn while slot 0 is empty can't happen: the first commit always goes to slot 0.
        Recovery must refuse instead of guessing.

        1. Create, close, open, close (both slots written)
        2. Make slot 0 empty, corrupt slot 1
        3. Open must return LV_ERR_MEDIA_CRASHED
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
    lv_close(db);
    db = NULL;

    if (damage_commit_slot(0, SLOT_MAKE_EMPTY) < 0 || damage_commit_slot(1, SLOT_MAKE_CORRUPT) < 0) {
        goto cleanup;
    }
    if (open_expect_fail(LV_ERR_MEDIA_CRASHED, "second slot torn") < 0) {
        goto cleanup;
    }

    result = 0;

cleanup:
    lv_close(db);
    test_end(result, TEST_DIR, "test_commit_second_slot_torn");
    return result;
}

int test_commit_both_corrupt(void) {
    /*
        1. Two sessions, 10 puts each, clean closes (both slots hold a commit)
        2. Corrupt both slots
        3. Open must return LV_ERR_MEDIA_CRASHED
    */
    Livero* db = NULL;
    int result = -1;
    answers_reset();

    if (make_two_sessions(&db, 10) < 0) {
        goto cleanup;
    }
    if (damage_commit_slot(0, SLOT_MAKE_CORRUPT) < 0 || damage_commit_slot(1, SLOT_MAKE_CORRUPT) < 0) {
        goto cleanup;
    }
    if (open_expect_fail(LV_ERR_MEDIA_CRASHED, "both corrupt") < 0) {
        goto cleanup;
    }

    result = 0;

cleanup:
    lv_close(db);
    test_end(result, TEST_DIR, "test_commit_both_corrupt");
    return result;
}

int test_commit_both_empty(void) {
    /*
        1. Create and close
        2. Make both slots empty (no commit was ever written)
        3. Open must succeed as a new db, the first put gets LV_SEQ_START
    */
    Livero* db = NULL;
    int result = -1;
    answers_reset();

    if (create_test_db(TEST_DIR, NULL, NULL, &db) < 0) {
        goto cleanup;
    }
    lv_close(db);
    db = NULL;

    if (damage_commit_slot(0, SLOT_MAKE_EMPTY) < 0 || damage_commit_slot(1, SLOT_MAKE_EMPTY) < 0) {
        goto cleanup;
    }
    if (open_db(&db, &TEST_OPTION, "both empty") < 0) {
        goto cleanup;
    }
    if (put_one(db, NULL, 2000, 2026, NULL, NULL) < 0) {
        goto cleanup;
    }

    result = 0;

cleanup:
    lv_close(db);
    test_end(result, TEST_DIR, "test_commit_both_empty");
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

    // Recovery
    if (test_empty_reopen() < 0) {
        return -1;
    }
    if (test_crash_pending_lost() < 0) {
        return -1;
    }
    if (test_crash_after_clean_reopen() < 0) {
        return -1;
    }
    if (test_crash_without_put() < 0) {
        return -1;
    }
    if (test_commit_one_slot_corrupt() < 0) {
        return -1;
    }
    if (test_commit_first_slot_torn() < 0) {
        return -1;
    }
    if (test_commit_second_slot_torn() < 0) {
        return -1;
    }
    if (test_commit_both_corrupt() < 0) {
        return -1;
    }
    if (test_commit_both_empty() < 0) {
        return -1;
    }
    return 0;
}
