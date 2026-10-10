#include <assert.h>
#include <memory.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h>

#include "internal/helper.h"
#include "livero.h"
#include "livero_types.h"
#include "test_helper.h"

static constexpr LVSize_t TEST_MIN_VALUE_LEN = 100;
static constexpr LVSize_t TEST_MAX_VALUE_LEN = 300;
static constexpr LVDim_t TEST_VECTOR_DIM = 384;
static constexpr LVVectorType TEST_VECTOR_TYPE = LV_VEC_TYPE_FP32;

const char* TEST_DIR = "./TEST_DB/";
const LVConfig TEST_CONFIG = {
    .max_disk_bytes = _1mb, .vector_dimension = TEST_VECTOR_DIM, .vector_type = TEST_VECTOR_TYPE};

typedef struct LVTestAnswer {
    LVSeq_t key;
    char value[TEST_MAX_VALUE_LEN];
    LVSize_t value_len;
    LVDate date;
    bool tombstone;
} LVTestAnswer;

int test_create(Livero** out) {
    clean_test_dir(TEST_DIR);
    if (mkdir(TEST_DIR, 0755) != 0) {
        fprintf(stderr, "Failed to create a db because it failed to make the test dirrectory. \n");
        return -1;
    }

    Livero* db = nullptr;
    LVStatus status = lv_create(TEST_DIR, &TEST_CONFIG, &db);
    if (status != LV_OK) {
        fprintf(stderr, "Failed to create a db. \n");
        print_status(status);
        return -1;
    }
    if (out) {
        *out = db;
    } else {
        lv_close(db);
        clean_test_dir(TEST_DIR);
    }
    return 0;
}

int test_open(Livero** out, const LVDBInfo* expected_db_info) {
    Livero* db = nullptr;

    struct stat st;
    if (stat(TEST_DIR, &st) != 0) {
        if (test_create(&db) < 0) {
            fprintf(stderr, "Failed to open a db because it failed to create a db first.\n");
            return -1;
        } else {
            lv_close(db);
        }
    }
    db = nullptr;
    int result = 0;
    LVStatus status = lv_open(TEST_DIR, &db);
    if (status != LV_OK) {
        fprintf(stderr, "Failed to open a db. \n");
        print_status(status);
        result = -1;
        goto cleanup;
    }

    if (expected_db_info) {
        LVDBInfo opened_info;
        lv_db_info(db, &opened_info);

        if (opened_info.max_disk_bytes != expected_db_info->max_disk_bytes) {
            fprintf(stderr, "Opened db info mismatched: max_disk_bytes is incorrect (Expect: %u, Got: %u) \n",
                    expected_db_info->max_disk_bytes, opened_info.max_disk_bytes);
            result = -1;
            goto cleanup;
        }

        if (opened_info.vector_type != expected_db_info->vector_type) {
            fprintf(stderr, "Opened db info mismatched: vector_type is incorrect (Expect: %d, Got: %d]) \n",
                    expected_db_info->vector_type, opened_info.vector_type);
            result = -1;
            goto cleanup;
        }

        if (opened_info.vector_dimension != expected_db_info->vector_dimension) {
            fprintf(stderr, "Opened db info mismatched: vector_dimension is incorrect (Expect: %d, Got: %d) \n",
                    expected_db_info->vector_dimension, opened_info.vector_dimension);
            result = -1;
            goto cleanup;
        }

        if (opened_info.total_disk_size != expected_db_info->total_disk_size) {
            fprintf(stderr, "Opebed db info mismatched: total_disk_size is incorrect (Expect: %llu, Got: %llu) \n",
                    expected_db_info->total_disk_size, opened_info.total_disk_size);
            result = -1;
            goto cleanup;
        }

        if (opened_info.db_file_size != expected_db_info->db_file_size) {
            fprintf(stderr, "Opened db info mismatched: db_file_size is incorrect (Expect: %u, Got: %u) \n",
                    expected_db_info->db_file_size, opened_info.db_file_size);
            result = -1;
            goto cleanup;
        }

        if (opened_info.val_log_size != expected_db_info->val_log_size) {
            fprintf(stderr, "Opened db info mismatched; val_log_size is incorrect (Expect:%u, Got:%u) \n",
                    expected_db_info->val_log_size, opened_info.val_log_size);

            result = -1;
            goto cleanup;
        }

        if (opened_info.vec_log_size != expected_db_info->vec_log_size) {
            fprintf(stderr, "Opened db info mismatched: vec_log_size is incorrect (Expect: %u, Got:%u) \n",
                    expected_db_info->vec_log_size, opened_info.vec_log_size);
            result = -1;
            goto cleanup;
        }

        if (opened_info.record_count != expected_db_info->record_count) {
            fprintf(stderr, "Opened db info mismatched: record_count is incorrect (Expect: %u, Got:%u)",
                    expected_db_info->record_count, opened_info.record_count);
            result = -1;
            goto cleanup;
        }
    }

    if (out) {
        *out = db;
        return 0;
    }

cleanup:
    lv_close(db);
    clean_test_dir(TEST_DIR);
    return result;
}

int test_put(Livero* db, const void* vector, const LVDate* date, const void* value, const uint16_t* value_len,
             LVTestAnswer* answer) {
    assert(db);

    float FP32_VECTOR[TEST_VECTOR_DIM] = {0.0f};
    int8_t INT8_VECTOR[TEST_VECTOR_DIM] = {0};
    int DUMMY_VALUE_LEN = rand_int(TEST_MIN_VALUE_LEN, TEST_MAX_VALUE_LEN);
    char DUMMY_VALUE[DUMMY_VALUE_LEN];
    if (!value) {
        fill_random_chr_data(DUMMY_VALUE_LEN, DUMMY_VALUE);
    }
    LVDate DUMMY_DATE;
    if (!date) {
        fill_random_date(&DUMMY_DATE, 2000, 2026);
    }

    const void* vector_to_put = nullptr;
    const LVDate* date_to_put = date ? date : &DUMMY_DATE;
    const void* value_to_put = value ? value : DUMMY_VALUE;
    const uint16_t value_len_to_put = value_len ? *value_len : DUMMY_VALUE_LEN;
    LVSeq_t key = 0;

    if (vector) {
        vector_to_put = vector;
    } else {
        if (TEST_CONFIG.vector_type == LV_VEC_TYPE_FP32) {
            fill_fp32_vector(TEST_CONFIG.vector_dimension, FP32_VECTOR);
            vector_to_put = FP32_VECTOR;
        } else if (TEST_CONFIG.vector_type == LV_VEC_TYPE_INT8) {
            fill_int8_vector(TEST_CONFIG.vector_dimension, INT8_VECTOR);
            vector_to_put = INT8_VECTOR;
        }
    }

    LVStatus status = lv_put(db, vector_to_put, date_to_put, value_to_put, value_len_to_put, &key);

    if (status != LV_OK) {
        fprintf(stderr, "Failed to put.\n");
        print_status(status);
        return -1;
    }

    if (answer) {
        answer->key = key;
        answer->date = *date_to_put;
        memcpy(answer->value, value_to_put, value_len_to_put);
        answer->value_len = value_len_to_put;
        answer->tombstone = false;
    }

    return 0;
}

int test_get(const Livero* db, const LVTestAnswer* answer) {
    assert(db && answer);

    char SAVED_VALUE[TEST_MAX_VALUE_LEN] = {0};
    LVDate SAVED_DATE;

    memset(SAVED_VALUE, 0, sizeof(SAVED_VALUE));
    memset(&SAVED_DATE, 0, sizeof(SAVED_DATE));

    LVStatus status = lv_get(db, answer->key, &SAVED_DATE, SAVED_VALUE);
    if (status != LV_OK) {
        fprintf(stderr, "Failed to get an item at (Key: %llu Value Len: %d, Epoch Date: %llu, Status: %d) \n",
                answer->key, answer->value_len, convert_date_to_epoch(&answer->date), status);
        print_status(status);
        return -1;
    }

    if (memcmp(answer->value, SAVED_VALUE, answer->value_len) != 0) {
        fprintf(stderr, "The value is incorrect. (Value Len: %d) \n", answer->value_len);
        return -1;
    }

    if (compare_date(&answer->date, &SAVED_DATE) != 0) {
        fprintf(stderr, "The date is incorrect. (Expect: %llu Got: %llu) \n", convert_date_to_epoch(&answer->date),
                convert_date_to_epoch(&SAVED_DATE));
        return -1;
    }

    return 0;
}

int test_not_found(const Livero* db, const LVSeq_t non_exsists) {
    assert(db);

    LVStatus status = lv_get(db, non_exsists, nullptr, nullptr);
    if (status == LV_ERR_NOT_FOUND) {
        return 0;
    } else {
        fprintf(stderr, "Failed to test_not_found. \n");
        print_status(status);
        return -1;
    }
}

int test_delete(Livero* db, const LVSeq_t key) {
    assert(db);
    LVStatus status = lv_delete(db, key);

    if (status != LV_OK) {
        fprintf(stderr, "Failed to delete the key(%llu) \n", key);
        print_status(status);
        return -1;
    }

    return 0;
}

int test_query(Livero* db, const LVTopK_t top_k, const void* query_vector, const LVDate* date1, const LVDate* date2,
               const LVTestAnswer* ground_truth_set, const LVSize_t ground_truth_set_size,
               const int expected_query_size) {
    LVQueryResult* query_result = nullptr;
    LVSize_t query_result_size = 0;

    LVStatus status = lv_query(db, top_k, query_vector, date1, date2, &query_result, &query_result_size);
    if (status != LV_OK) {
        fprintf(stderr, "Failed to query.\n");
        print_status(status);
        return -1;
    }

    if (expected_query_size >= 0) {
        if ((const LVSize_t)expected_query_size != query_result_size) {
            fprintf(stderr, "Failed to query. The result size is incorrect. (Expect:%u, Got:%u)\n", expected_query_size,
                    query_result_size);
            return -1;
        }
    }
    for (LVSize_t j = 0; j < ground_truth_set_size; ++j) {
        const bool is_tombstone = ground_truth_set[j].tombstone;
        bool found = false;
        for (LVSize_t i = 0; i < query_result_size; ++i) {
            if (query_result[i].key == ground_truth_set[j].key) {
                if (query_result[i].value_len != ground_truth_set[j].value_len) {
                    fprintf(stderr, "Failed to qeury. The key matched, but value_len is incorrect.\n");
                    return -1;
                }
                if (memcmp(query_result[i].value, ground_truth_set[j].value, query_result[i].value_len) != 0) {
                    fprintf(stderr, "Failed to query. The key matched, but value is incorrect.\n");
                    return -1;
                }
                LVDate query_result_date = {.year = query_result[i].year,
                                            .month = query_result[i].month,
                                            .day = query_result[i].day,
                                            .hour = query_result[i].hour,
                                            .min = query_result[i].min,
                                            .sec = query_result[i].sec};
                if (compare_date(&query_result_date, &ground_truth_set[j].date) != 0) {
                    fprintf(stderr, "Failed to query. The key matched, but date is incorrect.\n");
                    return -1;
                }
                found = true;
                break;
            }
        }

        if (is_tombstone) {
            if (found) {
                fprintf(stderr, "Failed to query. The tombstone was found.\n");
                return -1;
            }
        } else {
            if (!found) {
                fprintf(stderr, "Failed to query. The key does not match any ground truth keys.\n");
                return -1;
            }
        }
    }

    return 0;
}

int test_1(void) {
    /*
        1. Create a db.
        2. Put N
        3. Get M
        4. Get a non-exsists key
        4. Close the db.
    */
    Livero* db = nullptr;
    int result = 0;

    const uint32_t PUT_COUNT = 200;
    const uint32_t GET_COUNT = 30;

    LVTestAnswer ANSWERS[PUT_COUNT];

    if ((result = test_create(&db)) < 0) {
        goto cleanup;
    }

    for (uint32_t i = 0; i < PUT_COUNT; ++i) {
        if ((result = test_put(db, nullptr, nullptr, nullptr, nullptr, &ANSWERS[i])) < 0) {
            goto cleanup;
        }
    }

    if ((result = test_get(db, &ANSWERS[0])) < 0) {
        goto cleanup;
    }

    if ((result = test_get(db, &ANSWERS[PUT_COUNT - 1])) < 0) {
        goto cleanup;
    }

    for (uint32_t i = 0; i < GET_COUNT - 2; ++i) {
        const LVSeq_t index = rand_int(0, PUT_COUNT - 1);
        if ((result = test_get(db, &ANSWERS[index])) < 0) {
            goto cleanup;
        }
    }

    if ((result = test_not_found(db, PUT_COUNT + 1)) < 0) {
        goto cleanup;
    }

cleanup:
    lv_close(db);
    if (result < 0) {
        fprintf(stderr, "TEST1 FAILED.\n");
    } else {
        fprintf(stdout, "TEST1 SUCCEEDED.\n");
    }
    return result;
}

int test_2(void) {
    /*
        1. Create a DB
        2. Put 30
        3. Get 30
        4. Close the db
        5. Reopen the db
        6. Get first 30
        7. Put 10 and Check the seqs start at 30
        8. Get 10
        9. Close the db
    */

    Livero* db = nullptr;
    int result = 0;
    const uint32_t PUT1_COUNT = 30, GET1_COUNT = 30;
    const uint32_t PUT2_COUNT = 10, GET2_COUNT = 10;
    LVTestAnswer PUT1_ANSWERS[PUT1_COUNT];
    LVTestAnswer PUT2_ANSWERS[PUT2_COUNT];

    if ((result = test_create(&db)) < 0) {
        goto cleanup;
    }

    for (uint32_t i = 0; i < PUT1_COUNT; ++i) {
        if ((result = test_put(db, nullptr, nullptr, nullptr, nullptr, &PUT1_ANSWERS[i])) < 0) {
            goto cleanup;
        }
    }

    for (uint32_t i = 0; i < GET1_COUNT; ++i) {
        if ((result = test_get(db, &PUT1_ANSWERS[i])) < 0) {
            goto cleanup;
        }
    }

    LVDBInfo expected;
    lv_db_info(db, &expected);

    lv_close(db);
    db = nullptr;

    if ((result = test_open(&db, &expected)) < 0) {
        goto cleanup;
    }

    for (uint32_t i = 0; i < GET1_COUNT; ++i) {
        if ((result = test_get(db, &PUT1_ANSWERS[i])) < 0) {
            goto cleanup;
        }
    }

    for (uint32_t i = 0; i < PUT2_COUNT; ++i) {
        if ((result = test_put(db, nullptr, nullptr, nullptr, nullptr, &PUT2_ANSWERS[i])) < 0) {
            goto cleanup;
        }
    }

    if (PUT2_ANSWERS[0].key != PUT1_COUNT) {
        fprintf(stderr, "The seq key is incorrect, expect to start at %u, but starts at %llu\n", PUT1_COUNT,
                PUT2_ANSWERS[0].key);
        result = -1;
        goto cleanup;
    }

    for (uint32_t i = 0; i < GET2_COUNT; ++i) {
        if ((result = test_get(db, &PUT2_ANSWERS[i])) < 0) {
            goto cleanup;
        }
    }

cleanup:
    lv_close(db);
    if (result < 0) {
        fprintf(stderr, "TEST2 FAILED.\n");
    } else {
        fprintf(stdout, "TEST2 SUCCEEDED.\n");
    }
    return result;
}

int test_3(void) {
    /*
        1. Create a db
        2. Put 10
        3. Delete seq_key = 3
        4. Get the deleted key and check db returns LV_ERR_NOT_FOUND
        5. Close the db
        6/ Reopen the db
        8. Redo 4
        9. Get not deleted keys
    */

    Livero* db = nullptr;
    int result = 0;
    const LVSize_t put_count = 10;
    const LVSeq_t key_to_delete = 3;
    LVTestAnswer answers[put_count];

    if ((result = test_create(&db)) < 0) {
        goto cleanup;
    }

    for (LVSize_t i = 0; i < put_count; ++i) {
        if ((result = test_put(db, nullptr, nullptr, nullptr, nullptr, &answers[i])) < 0) {
            goto cleanup;
        }
    }

    if ((result = test_delete(db, key_to_delete)) < 0) {
        goto cleanup;
    }

    if ((result = test_not_found(db, key_to_delete)) < 0) {
        goto cleanup;
    }

    lv_close(db);
    db = nullptr;
    if ((result = test_open(&db, nullptr)) < 0) {
        goto cleanup;
    }

    if ((result = test_not_found(db, key_to_delete)) < 0) {
        goto cleanup;
    }

    for (LVSize_t key = 0; key < put_count; ++key) {
        if (key != key_to_delete) {
            if ((result = test_get(db, &answers[key])) < 0) {
                goto cleanup;
            }
        }
    }

cleanup:
    lv_close(db);
    if (result < 0) {
        fprintf(stderr, "TEST3 FAILED.\n");
    } else {
        fprintf(stdout, "TEST3 SUCCEEDED.\n");
    }
    return result;
}

int test_4(void) {
    /*
        1. Create a db
        2. Make a ground truth set (size: 10)
        3. Put N
        4. Query 10 (no date filter)
        5. Check the result query size is exact top_k
    */
    Livero* db = nullptr;
    const LVSize_t ground_truth_set_size = 10;
    const LVSize_t put_count = 500;
    const LVTopK_t top_k = ground_truth_set_size;

    int result = 0;

    LVTestAnswer QUERY_GROUND_TRUTH[ground_truth_set_size];
    float fp32_query_vector[TEST_VECTOR_DIM] = {0.0f};
    fill_fp32_vector(TEST_CONFIG.vector_dimension, fp32_query_vector);

    if ((result = test_create(&db)) < 0) {
        goto cleanup;
    }

    for (LVSize_t i = 0; i < ground_truth_set_size; ++i) {
        if ((result = test_put(db, fp32_query_vector, nullptr, nullptr, nullptr, &QUERY_GROUND_TRUTH[i])) < 0) {
            goto cleanup;
        }
    }

    for (LVSize_t i = 0; i < put_count; ++i) {
        if ((result = test_put(db, nullptr, nullptr, nullptr, nullptr, nullptr)) < 0) {
            goto cleanup;
        }
    }

    if ((result = test_query(db, top_k, fp32_query_vector, nullptr, nullptr, QUERY_GROUND_TRUTH, ground_truth_set_size,
                             top_k)) < 0) {
        goto cleanup;
    }

cleanup:
    lv_close(db);
    if (result < 0) {
        fprintf(stderr, "TEST4 FAILED.\n");
    } else {
        fprintf(stdout, "TEST4 SUCCEEDED. \n");
    }
    return result;
}

int test_5(void) {
    /*
        1. Create a db
        2. Make a ground truth set with specific dates (Size:9, (1-3): in 2024, (4-6): in 2025,(7-9): in 2026)
        3. Put N
        4. Query each group with the date range filters
        5. Query with a date filter which can cover the date range of group 2,3 (2025-2026)
        6. Query out of range date filter over 2027
    */

    Livero* db = nullptr;
    int result = 0;
    const LVSize_t ground_truth_size = 9;
    const LVSize_t ground_truth_group_size = 3;
    const LVTopK_t top_k = 10;
    const LVSize_t put_count = 500;
    LVTestAnswer ground_truth[ground_truth_size];
    LVTestAnswer ground_truth_group1[ground_truth_group_size];
    LVTestAnswer ground_truth_group2[ground_truth_group_size];
    LVTestAnswer ground_truth_group3[ground_truth_group_size];
    float query_vector[TEST_VECTOR_DIM] = {0.0f};
    fill_fp32_vector(TEST_VECTOR_DIM, query_vector);

    if ((result = test_create(&db)) < 0) {
        goto cleanup;
    }

    LVDate ground_truth_date;
    for (LVSize_t i = 0; i < ground_truth_group_size; ++i) {
        if (i == 0) {
            make_first_date(2024, &ground_truth_date);
        } else {
            fill_random_date(&ground_truth_date, 2024, 2024);
        }

        if ((result = test_put(db, query_vector, &ground_truth_date, nullptr, nullptr, &ground_truth_group1[i])) < 0) {
            goto cleanup;
        }
    }
    memcpy(&ground_truth[0], ground_truth_group1, sizeof(LVTestAnswer) * ground_truth_group_size);

    for (LVSize_t i = 0; i < ground_truth_group_size; ++i) {
        if (i == 0) {
            make_first_date(2025, &ground_truth_date);
        } else {
            fill_random_date(&ground_truth_date, 2025, 2025);
        }
        if ((result = test_put(db, query_vector, &ground_truth_date, nullptr, nullptr, &ground_truth_group2[i])) < 0) {
            goto cleanup;
        }
    }
    memcpy(&ground_truth[3], ground_truth_group2, sizeof(LVTestAnswer) * ground_truth_group_size);
    for (LVSize_t i = 0; i < ground_truth_group_size; ++i) {
        if (i == 0) {
            make_first_date(2026, &ground_truth_date);
        } else {
            fill_random_date(&ground_truth_date, 2026, 2026);
        }
        if ((result = test_put(db, query_vector, &ground_truth_date, nullptr, nullptr, &ground_truth_group3[i])) < 0) {
            goto cleanup;
        }
    }
    memcpy(&ground_truth[6], ground_truth_group3, sizeof(LVTestAnswer) * ground_truth_group_size);

    // Dummies live strictly before 2024, so every tested range
    // (2024, 2025, 2026) contains only planted ground truths.
    LVDate dummy_date;
    for (LVSize_t i = 0; i < put_count; ++i) {
        fill_random_date(&dummy_date, 2000, 2023);
        if ((result = test_put(db, nullptr, &dummy_date, nullptr, nullptr, nullptr)) < 0) {
            goto cleanup;
        }
    }
    LVDate _2024_start;
    make_first_date(2024, &_2024_start);
    LVDate _2024_end;
    make_last_date(2024, &_2024_end);

    LVDate _2025_start;
    make_first_date(2025, &_2025_start);
    LVDate _2025_end;
    make_last_date(2025, &_2025_end);

    LVDate _2026_start;
    make_first_date(2026, &_2026_start);
    LVDate _2026_end;
    make_last_date(2026, &_2026_end);

    if ((result = test_query(db, top_k, query_vector, &_2024_start, &_2024_end, ground_truth_group1,
                             ground_truth_group_size, 3)) < 0) {
        goto cleanup;
    }

    if ((result = test_query(db, top_k, query_vector, &_2025_start, &_2025_end, ground_truth_group2,
                             ground_truth_group_size, 3)) < 0) {
        goto cleanup;
    }

    if ((result = test_query(db, top_k, query_vector, &_2026_start, &_2026_end, ground_truth_group3,
                             ground_truth_group_size, 3)) < 0) {
        goto cleanup;
    }

    if ((result = test_query(db, top_k, query_vector, &_2025_start, nullptr, &ground_truth[3], 6, 6)) < 0) {
        goto cleanup;
    }

    LVDate _2027_start;
    make_first_date(2027, &_2027_start);
    if ((result = test_query(db, top_k, query_vector, &_2027_start, nullptr, nullptr, 0, 0)) < 0) {
        goto cleanup;
    }
cleanup:
    lv_close(db);

    if (result < 0) {
        fprintf(stderr, "TEST5 FAILED.\n");
    } else {
        fprintf(stdout, "TEST5 SUCCEEDED.\n");
    }
    return result;
}

int test_6(void) {
    /*
        1. Create a db.
        2. Make a ground truth set (Size:10)
        3. Put N
        4. Delete one ground truth in the set (key=2)
        5. Query ground trutes and check that Query result does not contain the tombstone
        6. Close the db
        7. Reopen the db
        8. Repeat 5.
    */

    Livero* db = nullptr;
    int result = 0;
    const LVSize_t ground_truth_size = 10;
    const LVSeq_t key_to_delete = 2;
    const LVTopK_t top_k = 10;
    const LVSize_t put_count = 300;
    LVTestAnswer ground_truth[ground_truth_size];

    float query_vector[TEST_VECTOR_DIM] = {0.0f};
    fill_fp32_vector(TEST_VECTOR_DIM, query_vector);

    if ((result = test_create(&db)) < 0) {
        goto cleanup;
    }

    for (LVSize_t i = 0; i < ground_truth_size; ++i) {
        if ((result = test_put(db, query_vector, nullptr, nullptr, nullptr, &ground_truth[i])) < 0) {
            goto cleanup;
        }
    }

    for (LVSize_t i = 0; i < put_count; ++i) {
        if ((result = test_put(db, nullptr, nullptr, nullptr, nullptr, nullptr)) < 0) {
            goto cleanup;
        }
    }

    if ((result = test_delete(db, key_to_delete)) < 0) {
        goto cleanup;
    }
    ground_truth[key_to_delete].tombstone = true;

    if ((result = test_query(db, top_k, query_vector, nullptr, nullptr, ground_truth, ground_truth_size,
                             ground_truth_size)) < 0) {
        goto cleanup;
    }

    lv_close(db);
    db = nullptr;
    if ((result = test_open(&db, nullptr)) < 0) {
        goto cleanup;
    }

    if ((result = test_query(db, top_k, query_vector, nullptr, nullptr, ground_truth, ground_truth_size,
                             ground_truth_size)) < 0) {
        goto cleanup;
    }

cleanup:
    lv_close(db);
    if (result < 0) {
        fprintf(stderr, "TEST6 FAILED.\n");
    } else {
        fprintf(stdout, "TEST6 SUCCEEDED.\n");
    }
    return result;
}

int main(void) {
    if (test_create(nullptr) < 0) {
        return -1;
    }
    if (test_open(nullptr, nullptr) < 0) {
        return -1;
    }

    if (test_1() < 0) {
        return -1;
    }

    if (test_2() < 0) {
        return -1;
    }

    if (test_3() < 0) {
        return -1;
    }

    if (test_4() < 0) {
        return -1;
    }

    if (test_5() < 0) {
        return -1;
    }
    if (test_6() < 0) {
        return -1;
    }
    return 0;
}
